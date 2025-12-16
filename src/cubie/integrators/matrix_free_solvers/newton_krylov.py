"""Newton--Krylov solver factories for matrix-free integrators.

The helpers in this module wrap the linear solver provided by
:mod:`cubie.integrators.matrix_free_solvers.linear_solver` to build damped
Newton iterations suitable for CUDA device execution.
"""

from typing import Callable

from numba import cuda, int32, from_dtype
import numpy as np
from cubie._utils import ALLOWED_PRECISIONS, PrecisionDType
from cubie.cuda_simsafe import activemask, all_sync, selp, any_sync


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
        ``(stage_increment, parameters, drivers, t, h, a_ij, base_state,
        residual)``.
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
        value in the lower 16 bits. Iteration counts are also returned via
        the counters parameter.

    Notes
    -----
    The lower 16 bits of the returned status report the convergence outcome:
    ``0`` for success, ``1`` when backtracking cannot find a suitable step,
    ``2`` when the Newton iteration limit is exceeded, and ``4`` when the
    inner linear solver signals failure. The upper 16 bits hold the number of
    Newton iterations performed. Iteration counts are also written to
    the counters array: counters[0] holds Newton iterations and counters[1]
    holds total Krylov iterations.
    """

    precision_dtype = np.dtype(precision)
    if precision_dtype not in ALLOWED_PRECISIONS:
        raise ValueError("precision must be float16, float32, or float64.")

    precision = from_dtype(precision_dtype)
    n_arraysize = int(n)
    tol_squared = precision(tolerance * tolerance)
    typed_zero = precision(0.0)
    typed_one = precision(1.0)
    typed_damping = precision(damping)
    n = int32(n)
    max_iters = int32(max_iters)
    max_backtracks = int32(max_backtracks)
    # no cover: start

    @cuda.jit(
            # [(precision[::1],
            #   precision[::1],
            #   precision[::1],
            #   precision,
            #   precision,
            #   precision,
            #   precision[::1],
            #   precision[::1],
            #   int32[::1])],
            device=True,
            inline=True)
    def newton_krylov_solver(
        stage_increment,
        parameters,
        drivers,
        t,
        h,
        a_ij,
        base_state,
        shared_scratch,
        counters,
    ):
        """Solve a nonlinear system with a damped Newton--Krylov iteration.

        Parameters
        ----------
        stage_increment
            Current Newton iterate representing the stage increment.
        parameters
            Model parameters forwarded to the residual evaluation.
        drivers
            External drivers forwarded to the residual evaluation.
        t
            Stage time forwarded to the residual and linear solver.
        h
            Timestep scaling factor supplied by the outer integrator.
        a_ij
            Stage weight used by multi-stage integrators.
        base_state
            Reference state used when evaluating the residual.
        shared_scratch
            Shared scratch buffer providing Newton direction, residual,
            and linear solver storage. The first ``n`` entries store the
            Newton direction, the next ``n`` entries store the residual,
            and remaining entries are available for the linear solver.
        counters
            Size (2,) int32 array for iteration counters. Index 0 receives
            Newton iteration count, index 1 receives cumulative Krylov
            iteration count.

        Returns
        -------
        int
            Status word with convergence information and iteration count.

        Notes
        -----
        Scratch space requirements total two vectors of length ``n`` drawn
        from ``shared_scratch`` plus any additional space needed by the
        linear solver. No need to zero scratch space before passing - it's
        write-first in this function.
        ``delta`` is reset to zero before the first linear solve so it can be
        reused as the Newton direction buffer. The linear solver is invoked
        on the Jacobian system ``J * delta = rhs`` with ``rhs`` stored in
        ``residual``. Operators and residuals compute the evaluation state
        ``base_state + a_ij * stage_increment`` inline. The tentative state
        updates are reverted if no acceptable backtracking step is found.
        """

        delta = shared_scratch[:n]
        residual = shared_scratch[n: 2 * n]

        residual_function(
            stage_increment,
            parameters,
            drivers,
            t,
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

        # Boolean control flags replace status-code-based loop control
        converged = norm2_prev <= tol_squared
        has_error = False
        final_status = int32(0)

        # Local array for linear solver iteration count output
        krylov_iters_local = cuda.local.array(1, int32)

        iters_count = int32(0)
        total_krylov_iters = int32(0)
        mask = activemask()
        for _ in range(max_iters):
            done = converged or has_error
            if all_sync(mask, done):
                break

            # Predicated iteration count update
            active = not done
            iters_count = selp(
                active, int32(iters_count + int32(1)), iters_count
            )

            # Linear solver shared memory starts after newton's scratch
            lin_shared = shared_scratch[2 * n:]
            krylov_iters_local[0] = int32(0)
            lin_status = linear_solver(
                stage_increment,
                parameters,
                drivers,
                base_state,
                t,
                h,
                a_ij,
                residual,
                delta,
                lin_shared,
                krylov_iters_local,
            )

            lin_failed = lin_status != int32(0)
            has_error = has_error or lin_failed
            final_status = selp(
                    lin_failed, int32(final_status | lin_status), final_status
            )
            total_krylov_iters += selp(active, krylov_iters_local[0], int32(0))

            # Backtracking loop
            scale = typed_one
            scale_applied = typed_zero
            found_step = False
            active_bt = True

            for _ in range(max_backtracks + 1):
                if not any_sync(mask, active_bt):
                    break

                active_bt = active and (not found_step) and (not converged)
                delta_scale = selp(
                    active_bt, scale - scale_applied, typed_zero
                )
                for i in range(n):
                    stage_increment[i] += delta_scale * delta[i]
                scale_applied = selp(active_bt, scale, scale_applied)
                residual_temp = cuda.local.array(n_arraysize, precision)

                residual_function(
                    stage_increment,
                    parameters,
                    drivers,
                    t,
                    h,
                    a_ij,
                    base_state,
                    residual_temp,
                )

                for i in range(n):
                    residual[i] = selp(
                        active_bt, residual_temp[i], residual[i]
                    )

                norm2_new = typed_zero
                for i in range(n):
                    residual_value = residual[i]
                    norm2_new += residual_value * residual_value

                # Check convergence
                just_converged = active_bt and (norm2_new <= tol_squared)
                converged = converged or just_converged

                accept = active_bt and (not converged) and (norm2_new < norm2_prev)
                found_step = found_step or accept

                for i in range(n):
                    residual[i] = selp(
                        accept,
                        -residual[i],
                        residual[i],
                    )
                norm2_prev = selp(accept, norm2_new, norm2_prev)
                scale *= typed_damping

            # Backtrack failure handling
            backtrack_failed = active and (not found_step) and (not converged)
            has_error = has_error or backtrack_failed
            final_status = selp(
                backtrack_failed, int32(final_status | int32(1)), final_status
            )

            # Revert state if backtrack failed using predicated pattern
            revert_scale = selp(backtrack_failed, -scale_applied, typed_zero)
            for i in range(n):
                stage_increment[i] += revert_scale * delta[i]

        # Max iterations exceeded without convergence
        max_iters_exceeded = (not converged) and (not has_error)
        final_status = selp(
            max_iters_exceeded, int32(final_status | int32(2)), final_status
        )

        # Write iteration counts to counters array
        counters[0] = +iters_count
        counters[1] = +total_krylov_iters

        # Return status without encoding iterations (breaking change per plan)
        return final_status

    # no cover: end
    return newton_krylov_solver
