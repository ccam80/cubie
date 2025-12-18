"""Newton--Krylov solver factories for matrix-free integrators.

The helpers in this module wrap the linear solver provided by
:mod:`cubie.integrators.matrix_free_solvers.linear_solver` to build damped
Newton iterations suitable for CUDA device execution.
"""

from typing import Callable, Optional

from numba import cuda, int32, from_dtype
import numpy as np
from cubie._utils import ALLOWED_PRECISIONS, PrecisionDType
from cubie.buffer_registry import buffer_registry
from cubie.cuda_simsafe import activemask, all_sync, selp, any_sync


def newton_krylov_solver_factory(
    residual_function: Callable,
    linear_solver: Callable,
    n: int,
    factory: object,
    tolerance: float,
    max_iters: int,
    damping: float = 0.5,
    max_backtracks: int = 8,
    precision: PrecisionDType = np.float32,
    delta_location: str = 'shared',
    residual_location: str = 'shared',
    residual_temp_location: str = 'local',
    stage_base_bt_location: str = 'local',
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
    factory
        Owning factory instance for buffer registration.
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
    delta_location
        Memory location for delta buffer: 'local' or 'shared'.
    residual_location
        Memory location for residual buffer: 'local' or 'shared'.
    residual_temp_location
        Memory location for residual_temp buffer: 'local' or 'shared'.
    stage_base_bt_location
        Memory location for stage_base_bt buffer: 'local' or 'shared'.

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

    # Register buffers with central registry
    buffer_registry.register(
        'newton_delta', factory, n, delta_location, precision=precision
    )
    buffer_registry.register(
        'newton_residual', factory, n, residual_location, precision=precision
    )
    buffer_registry.register(
        'newton_residual_temp', factory, n, residual_temp_location,
        precision=precision
    )
    buffer_registry.register(
        'newton_stage_base_bt', factory, n, stage_base_bt_location,
        precision=precision
    )

    # Get allocators from registry
    alloc_delta = buffer_registry.get_allocator('newton_delta', factory)
    alloc_residual = buffer_registry.get_allocator('newton_residual', factory)
    alloc_residual_temp = buffer_registry.get_allocator(
        'newton_residual_temp', factory
    )
    alloc_stage_base_bt = buffer_registry.get_allocator(
        'newton_stage_base_bt', factory
    )

    numba_precision = from_dtype(precision_dtype)
    tol_squared = numba_precision(tolerance * tolerance)
    typed_zero = numba_precision(0.0)
    typed_one = numba_precision(1.0)
    typed_damping = numba_precision(damping)
    n_val = int32(n)
    max_iters = int32(max_iters)
    max_backtracks = int32(max_backtracks)
    # no cover: start

    @cuda.jit(device=True, inline=True)
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
            and linear solver storage.
        counters
            Size (2,) int32 array for iteration counters.

        Returns
        -------
        int
            Status word with convergence information and iteration count.
        """

        # Allocate buffers from registry
        delta = alloc_delta(shared_scratch, shared_scratch)
        residual = alloc_residual(shared_scratch, shared_scratch)
        residual_temp = alloc_residual_temp(shared_scratch, shared_scratch)

        # Initialize local arrays
        for _i in range(n_val):
            delta[_i] = typed_zero
            residual[_i] = typed_zero

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
        for i in range(n_val):
            residual_value = residual[i]
            residual[i] = -residual_value
            delta[i] = typed_zero
            norm2_prev += residual_value * residual_value

        converged = norm2_prev <= tol_squared
        has_error = False
        final_status = int32(0)

        krylov_iters_local = cuda.local.array(1, int32)

        iters_count = int32(0)
        total_krylov_iters = int32(0)
        mask = activemask()
        for _ in range(max_iters):
            done = converged or has_error
            if all_sync(mask, done):
                break

            active = not done
            iters_count = selp(
                active, int32(iters_count + int32(1)), iters_count
            )

            # Linear solver uses remaining shared space after Newton buffers
            lin_start = buffer_registry.shared_buffer_size(factory)
            lin_shared = shared_scratch[lin_start:]
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

            stage_base_bt = alloc_stage_base_bt(shared_scratch, shared_scratch)
            for i in range(n_val):
                stage_base_bt[i] = stage_increment[i]
            found_step = False
            alpha = typed_one

            for _ in range(max_backtracks):
                active_bt = active and (not found_step) and (not converged)
                if not any_sync(mask, active_bt):
                    break

                if active_bt:
                    for i in range(n_val):
                        stage_increment[i] = stage_base_bt[i] + alpha * delta[i]

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

                    norm2_new = typed_zero
                    for i in range(n_val):
                        residual_value = residual_temp[i]
                        norm2_new += residual_value * residual_value

                    if norm2_new <= tol_squared:
                        converged = True
                        found_step = True

                    if norm2_new < norm2_prev:
                        for i in range(n_val):
                            residual[i] = -residual_temp[i]
                        norm2_prev = norm2_new
                        found_step = True

                alpha *= typed_damping

            backtrack_failed = active and (not found_step) and (not converged)
            has_error = has_error or backtrack_failed
            final_status = selp(
                backtrack_failed, int32(final_status | int32(1)), final_status
            )

            if backtrack_failed:
                for i in range(n_val):
                    stage_increment[i] = stage_base_bt[i]

        max_iters_exceeded = (not converged) and (not has_error)
        final_status = selp(
            max_iters_exceeded, int32(final_status | int32(2)), final_status
        )

        counters[0] = iters_count
        counters[1] = total_krylov_iters

        return final_status

    # no cover: end
    return newton_krylov_solver
