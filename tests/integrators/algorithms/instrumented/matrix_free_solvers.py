"""Instrumented matrix-free solver factories for CUDA device kernels."""

from typing import Callable, Optional

import numpy as np
from numba import cuda, int32, from_dtype

from cubie._utils import ALLOWED_PRECISIONS, PrecisionDType
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

    # no cover: start
    @cuda.jit(device=True)
    def linear_solver(
        state,
        parameters,
        drivers,
        t,
        h,
        rhs,
        x,
        slot_index,
        linear_initial_guesses,
        linear_iteration_guesses,
        linear_residuals,
        linear_squared_norms,
        linear_preconditioned_vectors,
    ):
        preconditioned_vec = cuda.local.array(n, precision_scalar)
        temp = cuda.local.array(n, precision_scalar)

        operator_apply(state, parameters, drivers, t, h, x, temp)
        acc = typed_zero
        for i in range(n):
            residual_value = rhs[i] - temp[i]
            rhs[i] = residual_value
            acc += residual_value * residual_value
        mask = activemask()
        converged = acc <= tol_squared

        log_slot = int32(slot_index)
        for i in range(n):
            linear_initial_guesses[log_slot, i] = x[i]

        status = int32(4)
        iteration = int32(0)
        while iteration < max_iters:
            if preconditioned:
                preconditioner(
                    state,
                    parameters,
                    drivers,
                    t,
                    h,
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
                t,
                h,
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

            for i in range(n):
                linear_iteration_guesses[log_slot, iteration, i] = x[i]
                linear_residuals[log_slot, iteration, i] = rhs[i]
                linear_preconditioned_vectors[log_slot, iteration, i] = (
                    preconditioned_vec[i]
                )
            linear_squared_norms[log_slot, iteration] = acc

            if all_sync(mask, converged):
                status = int32(0)
                break

            iteration += int32(1)

        return status

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
    """Create a cached instrumented steepest-descent or MR solver."""

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
        cached_aux,
        t,
        h,
        rhs,
        x,
        slot_index,
        linear_initial_guesses,
        linear_iteration_guesses,
        linear_residuals,
        linear_squared_norms,
        linear_preconditioned_vectors,
    ):
        preconditioned_vec = cuda.local.array(n, precision_scalar)
        temp = cuda.local.array(n, precision_scalar)

        operator_apply(state, parameters, drivers, cached_aux, t, h, x, temp)
        acc = typed_zero
        for i in range(n):
            residual_value = rhs[i] - temp[i]
            rhs[i] = residual_value
            acc += residual_value * residual_value
        mask = activemask()
        converged = acc <= tol_squared

        status = int32(4)
        log_slot = int32(slot_index)
        for i in range(n):
            linear_initial_guesses[log_slot, i] = x[i]

        iteration = int32(0)
        while iteration < max_iters:
            if preconditioned:
                preconditioner(
                    state,
                    parameters,
                    drivers,
                    cached_aux,
                    t,
                    h,
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
                cached_aux,
                t,
                h,
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

            for i in range(n):
                linear_iteration_guesses[log_slot, iteration, i] = x[i]
                linear_residuals[log_slot, iteration, i] = rhs[i]
                linear_preconditioned_vectors[log_slot, iteration, i] = (
                    preconditioned_vec[i]
                )
            linear_squared_norms[log_slot, iteration] = acc

            if all_sync(mask, converged):
                status = int32(0)
                break

            iteration += int32(1)

        return status

    # no cover: end
    return linear_solver_cached


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
    """Create an instrumented damped Newton--Krylov solver."""

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
        t,
        h,
        a_ij,
        base_state,
        shared_scratch,
        stage_index,
        solver_initial_guesses,
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
            state,
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
        residual_copy = cuda.local.array(n, dtype)
        for i in range(n):
            residual_value = residual[i]
            norm2_prev += residual_value * residual_value
            delta[i] = typed_zero
            residual[i] = -residual_value
            solver_initial_guesses[stage_index, i] = state[i]
            residual_copy[i] = residual_value
            newton_initial_guesses[stage_index, i] = state[i]

        for i in range(n):
            newton_iteration_guesses[stage_index, log_index, i] = state[i]
            newton_residuals[stage_index, log_index, i] = residual_copy[i]
        newton_squared_norms[stage_index, log_index] = norm2_prev
        log_index += int32(1)

        status = status_active
        if norm2_prev <= tol_squared:
            status = int32(0)

        iters_count = int32(0)
        mask = activemask()
        state_snapshot = cuda.local.array(n, dtype)
        residual_snapshot = cuda.local.array(n, dtype)
        snapshot_ready = False

        for _ in range(max_iters):
            if all_sync(mask, status >= 0):
                break

            iters_count += int32(1)
            if status < 0:
                iter_slot = int(iters_count) - 1
                lin_return = linear_solver(
                    state,
                    parameters,
                    drivers,
                    t,
                    h,
                    residual,
                    delta,
                    linear_slot_base + iter_slot,
                    linear_initial_guesses,
                    linear_iteration_guesses,
                    linear_residuals,
                    linear_squared_norms,
                    linear_preconditioned_vectors,
                )
                if lin_return != int32(0):
                    status = int32(lin_return)

            scale = typed_one
            scale_applied = typed_zero
            found_step = False
            snapshot_ready = False
            norm2_new = typed_zero

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
                        t,
                        h,
                        a_ij,
                        base_state,
                        residual,
                    )

                    norm2_new = typed_zero
                    for i in range(n):
                        residual_value = residual[i]
                        norm2_new += residual_value * residual_value
                        state_snapshot[i] = state[i]
                        residual_snapshot[i] = residual_value
                    snapshot_ready = True

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

            iter_slot = int(iters_count) - 1
            if iter_slot >= 0:
                if snapshot_ready:
                    for i in range(n):
                        newton_iteration_guesses[stage_index, log_index, i] = (
                            state_snapshot[i]
                        )
                        newton_residuals[stage_index, log_index, i] = (
                            residual_snapshot[i]
                        )
                    newton_squared_norms[stage_index, log_index] = norm2_new
                    log_index += int32(1)
                newton_iteration_scale[stage_index, iter_slot] = scale_applied

        if status < 0:
            status = int32(2)

        status |= (iters_count + 1) << 16
        return status

    # no cover: end
    return newton_krylov_solver


__all__ = [
    "linear_solver_factory",
    "linear_solver_cached_factory",
    "newton_krylov_solver_factory",
]
