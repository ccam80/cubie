"""Matrix-free preconditioned linear solver.

This module builds CUDA device functions that implement steepest-descent or
minimal-residual iterations without forming Jacobian matrices explicitly.
The helpers interact with the nonlinear solvers in :mod:`cubie.integrators`
and expect caller-supplied operator and preconditioner callbacks.
"""

from typing import Callable, Optional

from numba import cuda, int32, from_dtype
import numpy as np

from cubie._utils import PrecisionDType
from cubie.cuda_simsafe import activemask, all_sync, selp


def linear_solver_factory(
    operator_apply: Callable,
    n: int,
    preconditioner: Optional[Callable] = None,
    correction_type: str = "minimal_residual",
    tolerance: float = 1e-6,
    max_iters: int = 100,
    precision: PrecisionDType = np.float64,
) -> Callable:
    """Create a CUDA device function implementing steepest-descent or MR.

    Parameters
    ----------
    operator_apply
        Callback that overwrites its output vector with ``F @ v``.
    n
        Length of the one-dimensional residual and search-direction vectors.
    preconditioner
        Approximate inverse preconditioner invoked as ``(state, parameters,
        drivers, t, h, residual, z, scratch)``. ``scratch`` can be overwritten.
        If ``None`` the identity preconditioner is used.
    correction_type
        Line-search strategy. Must be ``"steepest_descent"`` or
        ``"minimal_residual"``.
    tolerance
        Target on the squared residual norm that signals convergence.
    max_iters
        Maximum number of iterations permitted.
    precision
        Floating-point precision used when building the device function.

    Returns
    -------
    Callable
        CUDA device function returning ``0`` on convergence and ``4`` when the
        iteration limit is reached.

    Notes
    -----
    The operator typically has the form ``F = β M - γ h J`` where ``M`` is the
    mass matrix (often the identity), ``J`` is the Jacobian, ``h`` is the step
    size, and ``β`` and ``γ`` are scalar parameters captured in the closure.
    The solver instantiates its own local scratch buffers so callers only need
    to provide the residual and correction vectors.
    """

    sd_flag = 1 if correction_type == "steepest_descent" else 0
    mr_flag = 1 if correction_type == "minimal_residual" else 0
    if correction_type not in ("steepest_descent", "minimal_residual"):
        raise ValueError(
            "Correction type must be 'steepest_descent' or 'minimal_residual'."
        )
    preconditioned = 1 if preconditioner is not None else 0

    precision_dtype = np.dtype(precision)
    precision_scalar = from_dtype(precision_dtype)
    typed_zero = precision_scalar(0.0)
    tol_squared = tolerance * tolerance

    # no cover: start
    @cuda.jit(device=True)
    def linear_solver(
        state,
        parameters,
        drivers,
        base_state,
        t,
        h,
        a_ij,
        rhs,
        x,
    ):
        """Run one preconditioned steepest-descent or minimal-residual solve.

        Parameters
        ----------
        state
            State vector forwarded to the operator and preconditioner.
        parameters
            Model parameters forwarded to the operator and preconditioner.
        drivers
            External drivers forwarded to the operator and preconditioner.
        base_state
            Base state for n-stage operators (unused for single-stage).
        t
            Stage time forwarded to the operator and preconditioner.
        h
            Step size used by the operator evaluation.
        a_ij
            Stage coefficient forwarded to the operator and preconditioner.
        rhs
            Right-hand side of the linear system. Overwritten with the current
            residual.
        x
            Iterand provided as the initial guess and overwritten with the
            final solution.

        Returns
        -------
        int
            ``0`` on convergence or ``4`` when the iteration limit is reached.

        Notes
        -----
        ``rhs`` is updated in place to hold the running residual, and ``temp``
        is reused as the scratch vector passed to the preconditioner. The
        iteration therefore keeps just two auxiliary vectors of length ``n``.
        The operator, preconditioner behaviour, and correction strategy are
        fixed by the factory closure, while ``state``, ``parameters``, and
        ``drivers`` are treated as read-only context values.
        """

        preconditioned_vec = cuda.local.array(n, precision_scalar)
        temp = cuda.local.array(n, precision_scalar)

        operator_apply(state, parameters, drivers, base_state, t, h, a_ij, x, temp)
        acc = typed_zero
        for i in range(n):
            residual_value = rhs[i] - temp[i]
            rhs[i] = residual_value
            acc += residual_value * residual_value
        mask = activemask()
        converged = acc <= tol_squared

        for _ in range(max_iters):
            if preconditioned:
                preconditioner(
                    state,
                    parameters,
                    drivers,
                    base_state,
                    t,
                    h,
                    a_ij,
                    rhs,
                    preconditioned_vec,
                    temp,
                )
            else:
                for i in range(n):
                    preconditioned_vec[i] = rhs[i]

            operator_apply(
                state,
                parameters,
                drivers,
                base_state,
                t,
                h,
                a_ij,
                preconditioned_vec,
                temp,
            )
            numerator = typed_zero
            denominator = typed_zero
            if sd_flag:
                for i in range(n):
                    zi = preconditioned_vec[i]
                    numerator += rhs[i] * zi
                    denominator += temp[i] * zi
            elif mr_flag:
                for i in range(n):
                    ti = temp[i]
                    numerator += ti * rhs[i]
                    denominator += ti * ti

            alpha = selp(
                denominator != typed_zero,
                numerator / denominator,
                typed_zero,
            )
            alpha_effective = selp(converged, 0.0, alpha)

            acc = typed_zero
            for i in range(n):
                x[i] += alpha_effective * preconditioned_vec[i]
                rhs[i] -= alpha_effective * temp[i]
                residual_value = rhs[i]
                acc += residual_value * residual_value
            converged = converged or (acc <= tol_squared)

            if all_sync(mask, converged):
                return int32(0)
        return int32(4)

    # no cover: end
    return linear_solver


def linear_solver_cached_factory(
    operator_apply: Callable,
    n: int,
    preconditioner: Optional[Callable] = None,
    correction_type: str = "minimal_residual",
    tolerance: float = 1e-6,
    max_iters: int = 100,
    precision: PrecisionDType = np.float64,
) -> Callable:
    """Create a CUDA linear solver that forwards cached auxiliaries."""

    sd_flag = 1 if correction_type == "steepest_descent" else 0
    mr_flag = 1 if correction_type == "minimal_residual" else 0
    if correction_type not in ("steepest_descent", "minimal_residual"):
        raise ValueError(
            "Correction type must be 'steepest_descent' or 'minimal_residual'."
        )
    preconditioned = 1 if preconditioner is not None else 0

    precision_dtype = np.dtype(precision)
    precision_scalar = from_dtype(precision_dtype)
    typed_zero = precision_scalar(0.0)
    tol_squared = tolerance * tolerance

    # no cover: start
    @cuda.jit(device=True)
    def linear_solver_cached(
        state,
        parameters,
        drivers,
        base_state,
        cached_aux,
        t,
        h,
        a_ij,
        rhs,
        x,
    ):
        """Run one cached preconditioned steepest-descent or MR solve."""

        # Short life vectors declared locally for l
        preconditioned_vec = cuda.local.array(n, precision_scalar)
        temp = cuda.local.array(n, precision_scalar)

        operator_apply(
            state, parameters, drivers, base_state, cached_aux, t, h, a_ij,
                x, temp
        )
        acc = typed_zero
        for i in range(n):
            residual_value = rhs[i] - temp[i]
            rhs[i] = residual_value
            acc += residual_value * residual_value
        mask = activemask()
        converged = acc <= tol_squared

        for _ in range(max_iters):
            if preconditioned:
                preconditioner(
                    state,
                    parameters,
                    drivers,
                    base_state,
                    cached_aux,
                    t,
                    h,
                    a_ij,
                    rhs,
                    preconditioned_vec,
                    temp,
                )
            else:
                for i in range(n):
                    preconditioned_vec[i] = rhs[i]

            operator_apply(
                state,
                parameters,
                drivers,
                base_state,
                cached_aux,
                t,
                h,
                a_ij,
                preconditioned_vec,
                temp,
            )
            numerator = typed_zero
            denominator = typed_zero
            if sd_flag:
                for i in range(n):
                    zi = preconditioned_vec[i]
                    numerator += rhs[i] * zi
                    denominator += temp[i] * zi
            elif mr_flag:
                for i in range(n):
                    ti = temp[i]
                    numerator += ti * rhs[i]
                    denominator += ti * ti

            alpha = selp(
                denominator != typed_zero,
                numerator / denominator,
                typed_zero,
            )
            alpha_effective = selp(converged, 0.0, alpha)

            acc = typed_zero
            for i in range(n):
                x[i] += alpha_effective * preconditioned_vec[i]
                rhs[i] -= alpha_effective * temp[i]
                residual_value = rhs[i]
                acc += residual_value * residual_value
            converged = converged or (acc <= tol_squared)

            if all_sync(mask, converged):
                return int32(0)
        return int32(4)

    # no cover: end
    return linear_solver_cached
