"""Instrumented matrix-free solver factories for CUDA device kernels."""

from typing import Callable, Optional

import numpy as np
from numba import cuda, int32, from_dtype
from cubie._utils import ALLOWED_PRECISIONS, PrecisionDType
from cubie.cuda_simsafe import activemask, all_sync, selp, any_sync


def inst_linear_solver_factory(
    operator_apply: Callable,
    n: int,
    precision: PrecisionDType,
    preconditioner: Optional[Callable] = None,
    correction_type: str = "minimal_residual",
    tolerance: float = 1e-6,
    max_iters: int = 100,
) -> Callable:
    """Create an instrumented steepest-descent or minimal-residual solver."""

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
    n_val = int32(n)
    max_iters_val = int32(max_iters)

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
        slot_index,
        linear_initial_guesses,
        linear_iteration_guesses,
        linear_residuals,
        linear_squared_norms,
        linear_preconditioned_vectors,
        krylov_iters_out,
    ):
        preconditioned_vec = cuda.local.array(n, precision_scalar)
        temp = cuda.local.array(n, precision_scalar)

        operator_apply(state, parameters, drivers, base_state, t, h, a_ij, x, temp)
        acc = typed_zero
        for i in range(n_val):
            residual_value = rhs[i] - temp[i]
            rhs[i] = residual_value
            acc += residual_value * residual_value
        mask = activemask()
        converged = acc <= tol_squared

        log_slot = int32(slot_index)
        for i in range(n_val):
            linear_initial_guesses[log_slot, i] = x[i]

        iteration = int32(0)
        for _ in range(max_iters_val):
            if all_sync(mask, converged):
                break

            iteration += int32(1)
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
            alpha_effective = selp(converged, precision_scalar(0.0), alpha)

            acc = typed_zero
            for i in range(n_val):
                x[i] += alpha_effective * preconditioned_vec[i]
                rhs[i] -= alpha_effective * temp[i]
                residual_value = rhs[i]
                acc += residual_value * residual_value
            converged = converged or (acc <= tol_squared)

            # Logging uses iteration - 1 as index (0-based)
            log_iter = iteration - int32(1)
            for i in range(n_val):
                linear_iteration_guesses[log_slot, log_iter, i] = x[i]
                linear_residuals[log_slot, log_iter, i] = rhs[i]
                linear_preconditioned_vectors[log_slot, log_iter, i] = (
                    preconditioned_vec[i]
                )
            linear_squared_norms[log_slot, log_iter] = acc

        # Single exit point - status based on converged flag
        final_status = selp(converged, int32(0), int32(4))
        krylov_iters_out[0] = iteration
        return final_status

    # no cover: end
    return linear_solver


def inst_linear_solver_cached_factory(
    operator_apply: Callable,
    n: int,
    precision: PrecisionDType,
    preconditioner: Optional[Callable] = None,
    correction_type: str = "minimal_residual",
    tolerance: float = 1e-6,
    max_iters: int = 100,
) -> Callable:
    """Create a cached instrumented steepest-descent or MR solver."""

    sd_flag = True if correction_type == "steepest_descent" else False
    mr_flag = True if correction_type == "minimal_residual" else False
    if correction_type not in ("steepest_descent", "minimal_residual"):
        raise ValueError(
            "Correction type must be 'steepest_descent' or 'minimal_residual'."
        )
    preconditioned = preconditioner is not None

    precision_dtype = np.dtype(precision)
    precision_scalar = from_dtype(precision_dtype)
    typed_zero = precision_scalar(0.0)
    tol_squared = tolerance * tolerance
    n_val = int32(n)
    max_iters_val = int32(max_iters)

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
        slot_index,
        linear_initial_guesses,
        linear_iteration_guesses,
        linear_residuals,
        linear_squared_norms,
        linear_preconditioned_vectors,
        krylov_iters_out,
    ):
        preconditioned_vec = cuda.local.array(n, precision_scalar)
        temp = cuda.local.array(n, precision_scalar)

        operator_apply(
            state,
            parameters,
            drivers,
            base_state,
            cached_aux,
            t,
            h,
            a_ij,
            x,
            temp
        )
        acc = typed_zero
        for i in range(n_val):
            residual_value = rhs[i] - temp[i]
            rhs[i] = residual_value
            acc += residual_value * residual_value
        mask = activemask()
        converged = acc <= tol_squared

        log_slot = int32(slot_index)
        for i in range(n_val):
            linear_initial_guesses[log_slot, i] = x[i]

        iteration = int32(0)
        for _ in range(max_iters_val):
            if all_sync(mask, converged):
                break

            iteration += int32(1)
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

            # Logging uses iteration - 1 as index (0-based)
            log_iter = iteration - int32(1)
            for i in range(n_val):
                linear_iteration_guesses[log_slot, log_iter, i] = x[i]
                linear_residuals[log_slot, log_iter, i] = rhs[i]
                linear_preconditioned_vectors[log_slot, log_iter, i] = (
                    preconditioned_vec[i]
                )
            linear_squared_norms[log_slot, log_iter] = acc

        # Single exit point - status based on converged flag
        final_status = selp(converged, int32(0), int32(4))
        krylov_iters_out[0] = iteration
        return final_status

    # no cover: end
    return linear_solver_cached


def inst_newton_krylov_solver_factory(
    residual_function: Callable,
    linear_solver: Callable,
    n: int,
    tolerance: float,
    max_iters: int,
    precision: PrecisionDType,
    damping: float = 0.5,
    max_backtracks: int = 8,
) -> Callable:
    """Create an instrumented damped Newton--Krylov solver."""
    n_arraysize = int(n)
    precision_dtype = np.dtype(precision)
    if precision_dtype not in ALLOWED_PRECISIONS:
        raise ValueError("precision must be float16, float32, or float64.")

    numba_precision = from_dtype(precision)
    tol_squared = numba_precision(tolerance * tolerance)
    typed_zero = numba_precision(0.0)
    typed_one = numba_precision(1.0)
    typed_damping = numba_precision(damping)
    n = int32(n)
    max_iters = int32(max_iters)
    max_backtracks = int32(max_backtracks)

    # no cover: start
    @cuda.jit(
            # [(numba_precision[::1],
            #  numba_precision[::1],
            #  numba_precision[::1],
            #  numba_precision,
            #  numba_precision,
            #  numba_precision,
            #  numba_precision[::1],
            #  numba_precision[::1],
            #  int32[::1],
            #  int32,
            #  numba_precision[:, ::1],
            #  numba_precision[:, ::1],
            #  numba_precision[:, ::1],
            #  numba_precision[::1],
            #  numba_precision[::1],
            #  numba_precision[::1],
            #  numba_precision[:, ::1],
            #  numba_precision[:, :, ::1],
            #  numba_precision[:, :, ::1],
            #  numba_precision[:, ::1],
            #  numba_precision[:, :, ::1],
            #  )],
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
        stage_index,
        newton_initial_guesses,
        newton_iteration_guesses,
        newton_residuals,
        newton_squared_norms,
        newton_iteration_scale,
        linear_initial_guesses,
        linear_iteration_guesses,
        linear_residuals,
        linear_squared_norms,
        linear_preconditioned_vectors,
    ):
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
        linear_slot_base = int32(stage_index * max_iters)
        log_index = int32(0)
        residual_copy = cuda.local.array(n, numba_precision)
        for i in range(n):
            residual_value = residual[i]
            norm2_prev += residual_value * residual_value
            delta[i] = typed_zero
            residual[i] = -residual_value
            residual_copy[i] = residual_value
            newton_initial_guesses[stage_index, i] = stage_increment[i]

        for i in range(n):
            newton_iteration_guesses[stage_index, log_index, i] = (
                stage_increment[i]
            )
            newton_residuals[stage_index, log_index, i] = residual_copy[i]
        newton_squared_norms[stage_index, log_index] = norm2_prev
        log_index += int32(1)

        converged = norm2_prev <= tol_squared
        has_error = False
        final_status = int32(0)

        krylov_iters_local = cuda.local.array(1, int32)

        iters_count = int32(0)
        total_krylov_iters = int32(0)
        mask = activemask()
        stage_increment_snapshot = cuda.local.array(n, numba_precision)
        residual_snapshot = cuda.local.array(n, numba_precision)

        for _ in range(max_iters):
            done = converged or has_error
            if all_sync(mask, done):
                break

            # Predicated iteration count update
            active = not done
            iters_count = selp(
                active, int32(iters_count + int32(1)), iters_count
            )

            iter_slot = int(iters_count) - 1
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
                linear_slot_base + iter_slot,
                linear_initial_guesses,
                linear_iteration_guesses,
                linear_residuals,
                linear_squared_norms,
                linear_preconditioned_vectors,
                krylov_iters_local,
            )
            lin_failed = lin_status != int32(0)
            has_error = has_error or lin_failed
            final_status = selp(
                lin_failed, int32(final_status | lin_status), final_status
            )
            total_krylov_iters += selp(active, krylov_iters_local[0], int32(0))

            stage_base_bt = cuda.local.array(n_arraysize, numba_precision)
            for i in range(n):
                stage_base_bt[i] = stage_increment[i]

            alpha = typed_one
            found_step = False
            snapshot_ready = False

            for _ in range(max_backtracks):
                active_bt = active and (not found_step) and (not converged)
                if not any_sync(mask, active_bt):
                    break

                if active_bt:
                    for i in range(n):
                        stage_increment[i] = (
                            stage_base_bt[i] + alpha * delta[i]
                        )

                    residual_temp = cuda.local.array(
                        n_arraysize, dtype=numba_precision
                    )

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
                    for i in range(n):
                        residual_value = residual_temp[i]
                        norm2_new += residual_value * residual_value
                        stage_increment_snapshot[i] = stage_increment[i]
                        residual_snapshot[i] = residual_value
                    snapshot_ready = True

                    # Check convergence
                    if norm2_new <= tol_squared:
                        converged = True
                        found_step = True

                    if norm2_new < norm2_prev:
                        for i in range(n):
                            residual[i] = -residual_temp[i]
                        norm2_prev = norm2_new
                        found_step = True

                alpha *= typed_damping

            # Backtrack failure handling
            backtrack_failed = active and (not found_step) and (not converged)
            has_error = has_error or backtrack_failed
            final_status = selp(
                backtrack_failed, int32(final_status | int32(1)), final_status
            )

            # Revert state if backtrack failed
            if backtrack_failed:
                for i in range(n):
                    stage_increment[i] = stage_base_bt[i]

            iter_slot = int(iters_count) - 1
            if iter_slot >= 0:
                if snapshot_ready:
                    for i in range(n):
                        newton_iteration_guesses[stage_index, log_index, i] = (
                            stage_increment_snapshot[i]
                        )
                        newton_residuals[stage_index, log_index, i] = (
                            residual_snapshot[i]
                        )
                    newton_squared_norms[stage_index, log_index] = norm2_new
                    log_index += int32(1)
                newton_iteration_scale[stage_index, iter_slot] = alpha

        # Max iterations exceeded without convergence
        max_iters_exceeded = (not converged) and (not has_error)
        final_status = selp(
            max_iters_exceeded, int32(final_status | int32(2)), final_status
        )

        # Write iteration counts to counters array
        counters[0] = iters_count
        counters[1] = total_krylov_iters

        # Return status without encoding iterations (breaking change per plan)
        return int32(final_status)

    # no cover: end
    return newton_krylov_solver


__all__ = ["inst_linear_solver_factory", "inst_linear_solver_cached_factory",
           "inst_newton_krylov_solver_factory",
]
