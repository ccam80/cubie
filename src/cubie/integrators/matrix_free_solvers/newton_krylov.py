"""Newton--Krylov solver factories for matrix-free integrators.

The helpers in this module wrap the linear solver provided by
:mod:`cubie.integrators.matrix_free_solvers.linear_solver` to build damped
Newton iterations suitable for CUDA device execution.
"""

from typing import Callable

from numba import cuda, int32, from_dtype
import numpy as np

from cubie._utils import ALLOWED_PRECISIONS, PrecisionDType
from cubie.cuda_simsafe import activemask, all_sync, selp


def newton_krylov_solver_factory(
    residual_function: Callable,
    linear_solver: Callable,
    n: int,
    tolerance: float,
    max_iters: int,
    damping: float = 0.5,
    max_backtracks: int = 8,
    precision: PrecisionDType = np.float32,
) -> Callable:
    """Create a damped Newton--Krylov solver device function.

    Parameters
    ----------
    residual_function
        Matrix-free residual evaluator with signature
        ``(state, parameters, drivers, h, a_ij, base_state, residual)``.
    linear_solver
        Matrix-free linear solver created by :func:`linear_solver_factory`.
    n
        Size of the flattened residual and state vectors.
    tolerance
        Residual norm threshold for convergence.
    max_iters
        Maximum number of Newton iterations performed.
    damping
        Step shrink factor used during backtracking.
    max_backtracks
        Maximum number of damping attempts per Newton step.
    precision
        Floating-point precision used when compiling the device function.

    Returns
    -------
    Callable
        CUDA device function implementing the damped Newton--Krylov scheme.
        The return value encodes the iteration count in the upper 16 bits and
        a :class:`~cubie.integrators.matrix_free_solvers.SolverRetCodes`
        value in the lower 16 bits.

    Notes
    -----
    The lower 16 bits of the returned status report the convergence outcome:
    ``0`` for success, ``1`` when backtracking cannot find a suitable step,
    ``2`` when the Newton iteration limit is exceeded, and ``4`` when the
    inner linear solver signals failure. The upper 16 bits hold the number of
    Newton iterations performed.
    """

    precision_dtype = np.dtype(precision)
    if precision_dtype not in ALLOWED_PRECISIONS:
        raise ValueError("precision must be float16, float32, or float64.")

    dtype = from_dtype(precision_dtype)
    tol_squared = dtype(tolerance * tolerance)
    typed_zero = dtype(0.0)
    typed_one = dtype(1.0)
    typed_damping = dtype(damping)
    status_active = int32(-1)

    # no cover: start
    @cuda.jit(device=True, inline=True)
    def newton_krylov_solver(
        state,
        parameters,
        drivers,
        h,
        a_ij,
        base_state,
        delta,
        residual,
        preconditioned_vec,
        work_vec,
    ):
        """Solve a nonlinear system with a damped Newton--Krylov iteration.

        Parameters
        ----------
        state
            Current iterate; overwritten with the accepted solution.
        parameters
            Model parameters forwarded to the residual evaluation.
        drivers
            External drivers forwarded to the residual evaluation.
        h
            Timestep scaling factor supplied by the outer integrator.
        a_ij
            Stage weight used by multi-stage integrators.
        base_state
            Reference state used when evaluating the residual.
        delta
            Working vector storing the Newton direction.
        residual
            Working vector storing residual evaluations.
        preconditioned_vec
            Working vector passed to the linear solver.
        work_vec
            Additional working vector passed to the linear solver.

        Returns
        -------
        int
            Status word with convergence information and iteration count.

        Notes
        -----
        Scratch space requirements total four vectors of length ``n`` and the
        state, delta, residual, preconditioned, and work arrays are all updated
        in place during the iteration. ``delta`` is reset to zero before the
        first linear solve so it can be reused as the Newton direction buffer.
        The linear solver is invoked on the Jacobian system ``J * delta = rhs``
        with ``rhs`` stored in ``residual``. The tentative state updates are
        reverted if no acceptable backtracking step is found.
        """

        residual_function(
            state,
            parameters,
            drivers,
            h,
            a_ij,
            base_state,
            residual,
        )
        norm2_prev = typed_zero
        for i in range(n):
            residual_value = residual[i]
            residual[i] = -residual_value
            delta[i] = typed_zero
            norm2_prev += residual_value * residual_value

        status = status_active
        if norm2_prev <= tol_squared:
            status = int32(0)

        iters_count = int32(0)
        mask = activemask()
        for _ in range(max_iters):
            if all_sync(mask, status >= 0):
                break

            iters_count += int32(1)
            if status < 0:
                lin_return = linear_solver(
                    state,
                    parameters,
                    drivers,
                    h,
                    residual,
                    delta,
                    preconditioned_vec,
                    work_vec,
                )
                if lin_return != int32(0):
                    status = int32(lin_return)

            scale = typed_one
            scale_applied = typed_zero
            found_step = False

            for _ in range(max_backtracks + 1):
                if status < 0:
                    delta_scale = scale - scale_applied
                    for i in range(n):
                        state[i] += delta_scale * delta[i]
                    scale_applied = scale

                    residual_function(
                        state,
                        parameters,
                        drivers,
                        h,
                        a_ij,
                        base_state,
                        residual,
                    )

                    norm2_new = typed_zero
                    for i in range(n):
                        residual_value = residual[i]
                        norm2_new += residual_value * residual_value

                    if norm2_new <= tol_squared:
                        status = int32(0)

                    accept = (status < 0) and (norm2_new < norm2_prev)
                    found_step = found_step or accept

                    for i in range(n):
                        residual[i] = selp(
                            accept,
                            -residual[i],
                            residual[i],
                        )
                    norm2_prev = selp(accept, norm2_new, norm2_prev)

                if all_sync(mask, found_step or status >= 0):
                    break
                scale *= typed_damping

            if (status < 0) and (not found_step):
                for i in range(n):
                    state[i] -= scale_applied * delta[i]
                status = int32(1)

        if status < 0:
            status = int32(2)

        status |= (iters_count + 1) << 16
        return status



    # no cover: end
    return newton_krylov_solver
