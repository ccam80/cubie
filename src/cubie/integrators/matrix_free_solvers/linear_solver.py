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
from cubie.buffer_registry import buffer_registry
from cubie.cuda_simsafe import activemask, all_sync, compile_kwargs, selp


def linear_solver_factory(
    operator_apply: Callable,
    n: int,
    factory: object,
    preconditioner: Optional[Callable] = None,
    correction_type: str = "minimal_residual",
    tolerance: float = 1e-6,
    max_iters: int = 100,
    precision: PrecisionDType = np.float64,
    preconditioned_vec_location: str = 'local',
    temp_location: str = 'local',
) -> Callable:
    """Create a CUDA device function implementing steepest-descent or MR.

    Parameters
    ----------
    operator_apply
        Callback that overwrites its output vector with ``F @ v``.
    n
        Length of the one-dimensional residual and search-direction vectors.
    factory
        Owning factory instance for buffer registration.
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
    preconditioned_vec_location
        Memory location for preconditioned_vec buffer: 'local' or 'shared'.
    temp_location
        Memory location for temp buffer: 'local' or 'shared'.

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
    n_val = int32(n)
    max_iters = int32(max_iters)
    precision_numba = from_dtype(precision)
    typed_zero = precision_numba(0.0)
    tol_squared = precision_numba(tolerance * tolerance)

    # Register buffers with central registry
    buffer_registry.register(
        'lin_preconditioned_vec', factory, n, preconditioned_vec_location,
        precision=precision
    )
    buffer_registry.register(
        'lin_temp', factory, n, temp_location, precision=precision
    )

    # Get allocators from registry
    alloc_precond = buffer_registry.get_allocator(
        'lin_preconditioned_vec', factory
    )
    alloc_temp = buffer_registry.get_allocator('lin_temp', factory)

    # no cover: start
    @cuda.jit(
        device=True,
        inline=True,
        **compile_kwargs,
    )
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
        shared,
        krylov_iters_out,
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
        shared
            Shared memory array for selective buffer allocation.
        krylov_iters_out
            Single-element int32 array to receive the iteration count.

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

        # Allocate buffers from registry
        preconditioned_vec = alloc_precond(shared, shared)
        temp = alloc_temp(shared, shared)

        operator_apply(state, parameters, drivers, base_state, t, h, a_ij, x, temp)
        acc = typed_zero
        for i in range(n_val):
            residual_value = rhs[i] - temp[i]
            rhs[i] = residual_value
            acc += residual_value * residual_value
        mask = activemask()
        converged = acc <= tol_squared

        iter_count = int32(0)
        for _ in range(max_iters):
            if all_sync(mask, converged):
                break

            iter_count += int32(1)
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
                for i in range(n_val):
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
                for i in range(n_val):
                    zi = preconditioned_vec[i]
                    numerator += rhs[i] * zi
                    denominator += temp[i] * zi
            elif mr_flag:
                for i in range(n_val):
                    ti = temp[i]
                    numerator += ti * rhs[i]
                    denominator += ti * ti

            alpha = selp(
                denominator != typed_zero,
                numerator / denominator,
                typed_zero,
            )
            alpha_effective = selp(converged, precision_numba(0.0), alpha)

            acc = typed_zero
            for i in range(n_val):
                x[i] += alpha_effective * preconditioned_vec[i]
                rhs[i] -= alpha_effective * temp[i]
                residual_value = rhs[i]
                acc += residual_value * residual_value
            converged = converged or (acc <= tol_squared)

        # Single exit point - status based on converged flag
        final_status = selp(converged, int32(0), int32(4))
        krylov_iters_out[0] = iter_count
        return final_status

    # no cover: end
    return linear_solver


def linear_solver_cached_factory(
    operator_apply: Callable,
    n: int,
    factory: object,
    preconditioner: Optional[Callable] = None,
    correction_type: str = "minimal_residual",
    tolerance: float = 1e-6,
    max_iters: int = 100,
    precision: PrecisionDType = np.float64,
    preconditioned_vec_location: str = 'local',
    temp_location: str = 'local',
) -> Callable:
    """Create a CUDA linear solver that forwards cached auxiliaries.

    Parameters
    ----------
    operator_apply
        Callback that overwrites its output vector with ``F @ v``.
    n
        Length of the one-dimensional residual and search-direction vectors.
    factory
        Owning factory instance for buffer registration.
    preconditioner
        Approximate inverse preconditioner. If ``None`` the identity
        preconditioner is used.
    correction_type
        Line-search strategy. Must be ``"steepest_descent"`` or
        ``"minimal_residual"``.
    tolerance
        Target on the squared residual norm that signals convergence.
    max_iters
        Maximum number of iterations permitted.
    precision
        Floating-point precision used when building the device function.
    preconditioned_vec_location
        Memory location for preconditioned_vec buffer: 'local' or 'shared'.
    temp_location
        Memory location for temp buffer: 'local' or 'shared'.

    Returns
    -------
    Callable
        CUDA device function that runs the linear solver with cached aux.
    """

    sd_flag = 1 if correction_type == "steepest_descent" else 0
    mr_flag = 1 if correction_type == "minimal_residual" else 0
    if correction_type not in ("steepest_descent", "minimal_residual"):
        raise ValueError(
            "Correction type must be 'steepest_descent' or 'minimal_residual'."
        )
    preconditioned = 1 if preconditioner is not None else 0
    n_val = int32(n)
    max_iters = int32(max_iters)

    precision_dtype = np.dtype(precision)
    precision_scalar = from_dtype(precision_dtype)
    typed_zero = precision_scalar(0.0)
    tol_squared = tolerance * tolerance

    # Register buffers with central registry
    buffer_registry.register(
        'lin_cached_preconditioned_vec', factory, n, preconditioned_vec_location,
        precision=precision
    )
    buffer_registry.register(
        'lin_cached_temp', factory, n, temp_location, precision=precision
    )

    # Get allocators from registry
    alloc_precond = buffer_registry.get_allocator(
        'lin_cached_preconditioned_vec', factory
    )
    alloc_temp = buffer_registry.get_allocator('lin_cached_temp', factory)

    # no cover: start
    @cuda.jit(
        device=True,
        **compile_kwargs,
    )
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
        shared,
        krylov_iters_out,
    ):
        """Run one cached preconditioned steepest-descent or MR solve."""

        # Allocate buffers from registry
        preconditioned_vec = alloc_precond(shared, shared)
        temp = alloc_temp(shared, shared)

        operator_apply(
            state, parameters, drivers, base_state, cached_aux, t, h, a_ij,
                x, temp
        )
        acc = typed_zero
        for i in range(n_val):
            residual_value = rhs[i] - temp[i]
            rhs[i] = residual_value
            acc += residual_value * residual_value
        mask = activemask()
        converged = acc <= tol_squared

        iter_count = int32(0)
        for _ in range(max_iters):
            if all_sync(mask, converged):
                break

            iter_count += int32(1)
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
                for i in range(n_val):
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
                for i in range(n_val):
                    zi = preconditioned_vec[i]
                    numerator += rhs[i] * zi
                    denominator += temp[i] * zi
            elif mr_flag:
                for i in range(n_val):
                    ti = temp[i]
                    numerator += ti * rhs[i]
                    denominator += ti * ti

            alpha = selp(
                denominator != typed_zero,
                numerator / denominator,
                typed_zero,
            )
            alpha_effective = selp(converged, precision_scalar(0.0), alpha)

            acc = typed_zero
            for i in range(n_val):
                x[i] += alpha_effective * preconditioned_vec[i]
                rhs[i] -= alpha_effective * temp[i]
                residual_value = rhs[i]
                acc += residual_value * residual_value
            converged = converged or (acc <= tol_squared)

        # Single exit point - status based on converged flag
        final_status = selp(converged, int32(0), int32(4))
        krylov_iters_out[0] = iter_count
        return final_status

    # no cover: end
    return linear_solver_cached
