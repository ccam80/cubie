"""All-in-one debug file for Numba lineinfo debugging.

All CUDA device functions are consolidated in this single file to enable
proper line-level debugging with Numba's lineinfo feature.
"""
# ruff: noqa: E402
from math import ceil

import numpy as np
from numba import cuda, int16, int32, float32
from numba import from_dtype as numba_from_dtype
from cubie.cuda_simsafe import activemask, all_sync, selp, compile_kwargs
from cubie.cuda_simsafe import from_dtype as simsafe_dtype
from cubie.integrators.algorithms.generic_dirk_tableaus import (
    SDIRK_2_2_TABLEAU,
)
from cubie.integrators.algorithms.generic_erk_tableaus import (
    DORMAND_PRINCE_54_TABLEAU,
)

# =========================================================================
# CONFIGURATION
# =========================================================================

precision = np.float32
numba_precision = numba_from_dtype(precision)
simsafe_precision = simsafe_dtype(precision)
simsafe_int32 = simsafe_dtype(np.int32)

# Typed constants for device code
typed_zero = numba_precision(0.0)

# Lorenz system constants
constants = {'sigma': 10.0, 'beta': 8.0 / 3.0}

# System dimensions
n_states = 3
n_parameters = 1
n_observables = 0
n_drivers = 0
n_counters = 4

# Time parameters
duration = precision(2e-7)
warmup = precision(0.0)
dt_save = precision(0.1e-7)
dt0 = precision(0.01)
dt_min = precision(1e-12)

# SDIRK 2,2 tableau stage count (for buffer sizing)
stage_count = SDIRK_2_2_TABLEAU.stage_count

# Solver helper parameters (beta, gamma, mass matrix scaling)
beta_solver = precision(1.0)
gamma_solver = precision(1.0)
preconditioner_order = 2

# Linear solver (Krylov) parameters
krylov_tolerance = precision(1e-6)
max_linear_iters = 200

# Newton-Krylov nonlinear solver parameters
newton_tolerance = precision(1e-6)
max_newton_iters = 100
newton_damping = precision(0.85)
max_backtracks = 8

# PID controller parameters
algorithm_order = 2
kp = precision(1.0 / 18.0)
ki = precision(1.0 / 9.0)
kd = precision(1.0 / 18.0)
min_gain = precision(0.2)
max_gain = precision(5.0)
dt_min_ctrl = precision(1e-6)
dt_max_ctrl = precision(1.0)
deadband_min = precision(1.0)
deadband_max = precision(1.2)
safety = precision(0.9)
atol = np.full(n_states, precision(1e-6), dtype=precision)
rtol = np.full(n_states, precision(1e-6), dtype=precision)

# Output dimensions
n_output_samples = int(ceil(float(duration) / float(dt_save))) + 1

# =========================================================================
# AUTO-GENERATED DEVICE FUNCTION FACTORIES
# =========================================================================

def dxdt_factory(constants, prec):
    """Auto-generated dxdt factory."""
    sigma = prec(constants['sigma'])
    beta = prec(constants['beta'])
    numba_prec = numba_from_dtype(prec)

    @cuda.jit((numba_prec[::1], numba_prec[::1], numba_prec[::1],
               numba_prec[::1], numba_prec[::1], numba_prec),
              device=True, inline=True, **compile_kwargs)
    def dxdt(state, parameters, drivers, observables, out, t):
        out[2] = -beta * state[2] + state[0] * state[1]
        _cse0 = -state[1]
        out[0] = sigma * (-_cse0 - state[0])
        out[1] = _cse0 + state[0] * (parameters[0] - state[2])
    return dxdt


def observables_factory(constants, prec):
    """Auto-generated observables factory."""
    numba_prec = numba_from_dtype(prec)

    @cuda.jit((numba_prec[::1], numba_prec[::1], numba_prec[::1],
               numba_prec[::1], numba_prec),
              device=True, inline=True, **compile_kwargs)
    def get_observables(state, parameters, drivers, observables, t):
        pass
    return get_observables


def neumann_preconditioner_factory(constants, prec, beta, gamma, order):
    """Auto-generated Neumann preconditioner."""
    n = 3
    beta_inv = 1.0 / beta
    h_eff_factor = gamma * beta_inv
    sigma = prec(constants['sigma'])
    beta_const = prec(constants['beta'])
    numba_prec = numba_from_dtype(prec)

    @cuda.jit((numba_prec[::1], numba_prec[::1], numba_prec[::1],
               numba_prec[::1], numba_prec, numba_prec, numba_prec,
               numba_prec[::1], numba_prec[::1], numba_prec[::1]),
              device=True, inline=True, **compile_kwargs)
    def preconditioner(state, parameters, drivers, base_state, t, h, a_ij,
                       v, out, jvp):
        for i in range(n):
            out[i] = v[i]
        h_eff = h * numba_prec(h_eff_factor) * a_ij
        for _ in range(order):
            j_00 = -sigma
            j_01 = sigma
            j_10 = -a_ij * state[2] + parameters[0] - base_state[2]
            j_11 = int32(-1)
            j_12 = -a_ij * state[0] - base_state[0]
            j_20 = a_ij * state[1] + base_state[1]
            j_21 = a_ij * state[0] + base_state[0]
            j_22 = -beta_const
            jvp[0] = j_00 * out[0] + j_01 * out[1]
            jvp[1] = j_10 * out[0] + j_11 * out[1] + j_12 * out[2]
            jvp[2] = j_20 * out[0] + j_21 * out[1] + j_22 * out[2]
            for i in range(n):
                out[i] = v[i] + h_eff * jvp[i]
        for i in range(n):
            out[i] = numba_prec(beta_inv) * out[i]
    return preconditioner


def stage_residual_factory(constants, prec, beta, gamma, order):
    """Auto-generated nonlinear residual for implicit updates."""
    sigma = prec(constants['sigma'])
    beta_const = prec(constants['beta'])
    numba_prec = numba_from_dtype(prec)
    beta_val = numba_prec(1.0) * numba_prec(beta)

    @cuda.jit((numba_prec[::1], numba_prec[::1], numba_prec[::1],
               numba_prec, numba_prec, numba_prec, numba_prec[::1],
               numba_prec[::1]),
              device=True, inline=True, **compile_kwargs)
    def residual(u, parameters, drivers, t, h, a_ij, base_state, out):
        _cse0 = a_ij * u[1]
        _cse1 = a_ij * u[0] + base_state[0]
        _cse2 = a_ij * u[2] + base_state[2]
        _cse5 = numba_prec(gamma) * h
        _cse3 = _cse0 + base_state[1]
        dx_0 = sigma * (_cse0 - _cse1 + base_state[1])
        dx_2 = _cse1 * _cse3 - _cse2 * beta_const
        dx_1 = _cse1 * (-_cse2 + parameters[0]) - _cse3
        out[0] = beta_val * u[0] - _cse5 * dx_0
        out[2] = beta_val * u[2] - _cse5 * dx_2
        out[1] = beta_val * u[1] - _cse5 * dx_1
    return residual


def linear_operator_factory(constants, prec, beta, gamma, order):
    """Auto-generated linear operator."""
    sigma = prec(constants['sigma'])
    beta_const = prec(constants['beta'])
    numba_prec = numba_from_dtype(prec)

    @cuda.jit((numba_prec[::1], numba_prec[::1], numba_prec[::1],
               numba_prec[::1], numba_prec, numba_prec, numba_prec,
               numba_prec[::1], numba_prec[::1]),
              device=True, inline=True, **compile_kwargs)
    def operator_apply(state, parameters, drivers, base_state, t, h, a_ij,
                       v, out):
        m_00 = numba_prec(1.0)
        m_11 = numba_prec(1.0)
        m_22 = numba_prec(1.0)
        j_00 = -sigma
        j_01 = sigma
        j_10 = -a_ij * state[2] + parameters[0] - base_state[2]
        j_11 = int32(-1)
        j_12 = -a_ij * state[0] - base_state[0]
        j_20 = a_ij * state[1] + base_state[1]
        j_21 = a_ij * state[0] + base_state[0]
        j_22 = -beta_const
        gamma_val = numba_prec(gamma)
        beta_val = numba_prec(beta)
        out[0] = (-a_ij * gamma_val * h * (j_00 * v[0] + j_01 * v[1])
                  + beta_val * m_00 * v[0])
        out[1] = (-a_ij * gamma_val * h * (j_10 * v[0] + j_11 * v[1]
                  + j_12 * v[2]) + beta_val * m_11 * v[1])
        out[2] = (-a_ij * gamma_val * h * (j_20 * v[0] + j_21 * v[1]
                  + j_22 * v[2]) + beta_val * m_22 * v[2])
    return operator_apply


# =========================================================================
# INLINE LINEAR SOLVER FACTORY
# =========================================================================

def linear_solver_inline_factory(operator_apply, n, preconditioner,
                                 tolerance, max_iters, prec):
    """Create inline linear solver device function."""
    numba_prec = numba_from_dtype(prec)
    tol_squared = tolerance * tolerance

    @cuda.jit(device=True, inline=True, **compile_kwargs)
    def linear_solver(state, parameters, drivers, base_state, t, h, a_ij,
                      rhs, x):
        preconditioned_vec = cuda.local.array(n, numba_prec)
        temp = cuda.local.array(n, numba_prec)

        operator_apply(state, parameters, drivers, base_state, t, h, a_ij,
                       x, temp)
        acc = typed_zero
        for i in range(n):
            residual_value = rhs[i] - temp[i]
            rhs[i] = residual_value
            acc += residual_value * residual_value
        mask = activemask()
        converged = acc <= tol_squared

        iter_count = int32(0)
        for _ in range(max_iters):
            iter_count += int32(1)
            preconditioner(state, parameters, drivers, base_state, t, h,
                           a_ij, rhs, preconditioned_vec, temp)

            operator_apply(state, parameters, drivers, base_state, t, h,
                           a_ij, preconditioned_vec, temp)
            numerator = typed_zero
            denominator = typed_zero
            for i in range(n):
                ti = temp[i]
                numerator += ti * rhs[i]
                denominator += ti * ti

            alpha = selp(denominator != typed_zero,
                         numerator / denominator, typed_zero)
            alpha_effective = selp(converged, numba_prec(0.0), alpha)

            acc = typed_zero
            for i in range(n):
                x[i] += alpha_effective * preconditioned_vec[i]
                rhs[i] -= alpha_effective * temp[i]
                residual_value = rhs[i]
                acc += residual_value * residual_value
            converged = converged or (acc <= tol_squared)

            if all_sync(mask, converged):
                return_status = int32(0)
                return_status |= (iter_count + int32(1)) << 16
                return return_status
        return_status = int32(4)
        return_status |= (iter_count + int32(1)) << 16
        return return_status
    return linear_solver


# =========================================================================
# INLINE NEWTON-KRYLOV SOLVER FACTORY
# =========================================================================

def newton_krylov_inline_factory(residual_fn, linear_solver, n, tolerance,
                                 max_iters, damping, max_backtracks, prec):
    """Create inline Newton-Krylov solver device function."""
    numba_prec = numba_from_dtype(prec)
    tol_squared = numba_prec(tolerance * tolerance)
    typed_zero = numba_prec(0.0)
    typed_one = numba_prec(1.0)
    typed_damping = numba_prec(damping)
    status_active = int32(-1)

    @cuda.jit(device=True, inline=True, **compile_kwargs)
    def newton_krylov_solver(stage_increment, parameters, drivers, t, h,
                             a_ij, base_state, shared_scratch, counters):
        delta = shared_scratch[:n]
        residual = shared_scratch[n:2 * n]

        residual_fn(stage_increment, parameters, drivers, t, h, a_ij,
                    base_state, residual)
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
        total_krylov_iters = int32(0)
        mask = activemask()
        for _ in range(max_iters):
            if all_sync(mask, status >= 0):
                break

            iters_count += int32(1)
            if status < 0:
                lin_return = linear_solver(stage_increment, parameters,
                                           drivers, base_state, t, h, a_ij,
                                           residual, delta)
                krylov_iters = (lin_return >> 16) & int32(0xFFFF)
                total_krylov_iters += krylov_iters
                lin_status = lin_return & int32(0xFFFF)
                if lin_status != int32(0):
                    status = int32(lin_status)

            scale = typed_one
            scale_applied = typed_zero
            found_step = False

            for _ in range(max_backtracks + 1):
                if status < 0:
                    delta_scale = scale - scale_applied
                    for i in range(n):
                        stage_increment[i] += delta_scale * delta[i]
                    scale_applied = scale

                    residual_fn(stage_increment, parameters, drivers, t, h,
                                a_ij, base_state, residual)

                    norm2_new = typed_zero
                    for i in range(n):
                        residual_value = residual[i]
                        norm2_new += residual_value * residual_value

                    if norm2_new <= tol_squared:
                        status = int32(0)

                    accept = (status < 0) and (norm2_new < norm2_prev)
                    found_step = found_step or accept

                    for i in range(n):
                        residual[i] = selp(accept, -residual[i], residual[i])
                    norm2_prev = selp(accept, norm2_new, norm2_prev)

                if all_sync(mask, found_step or status >= 0):
                    break
                scale *= typed_damping

            if (status < 0) and (not found_step):
                for i in range(n):
                    stage_increment[i] -= scale_applied * delta[i]
                status = int32(1)

        if status < 0:
            status = int32(2)

        counters[0] = iters_count
        counters[1] = total_krylov_iters

        status |= iters_count << 16
        return status
    return newton_krylov_solver


# =========================================================================
# INLINE DIRK STEP FACTORY (Generic DIRK with tableau)
# =========================================================================

def dirk_step_inline_factory(
    nonlinear_solver,
    dxdt_fn,
    observables_function,
    n,
    prec,
    tableau,
):
    """Create inline DIRK step device function matching generic_dirk.py."""
    numba_precision = numba_from_dtype(prec)
    typed_zero = numba_precision(0.0)

    # Extract tableau properties
    stage_count = tableau.stage_count

    # Compile-time toggles
    has_driver_function = False  # No driver function in this test
    has_error = tableau.has_error_estimate
    multistage = stage_count > 1
    first_same_as_last = False  # SDIRK does not share first/last stage
    can_reuse_accepted_start = False

    stage_rhs_coeffs = tableau.typed_rows(tableau.a, numba_precision)
    solution_weights = tableau.typed_vector(tableau.b, numba_precision)
    error_weights = tableau.error_weights(numba_precision)
    if error_weights is None or not has_error:
        error_weights = tuple(typed_zero for _ in range(stage_count))
    stage_time_fractions = tableau.typed_vector(tableau.c, numba_precision)
    diagonal_coeffs = tableau.diagonal(numba_precision)

    # Last-step caching optimization
    accumulates_output = tableau.accumulates_output
    accumulates_error = tableau.accumulates_error
    b_row = tableau.b_matches_a_row
    b_hat_row = tableau.b_hat_matches_a_row

    stage_implicit = tuple(coeff != numba_precision(0.0)
                           for coeff in diagonal_coeffs)
    accumulator_length = max(stage_count - 1, 0) * n
    solver_shared_elements = 2 * n  # delta + residual for Newton solver

    # Shared memory indices
    acc_start = 0
    acc_end = accumulator_length
    solver_start = acc_end
    solver_end = acc_end + solver_shared_elements

    # Compile-time fixed controller (always use dt_scalar for this test)
    is_controller_fixed = False
    dt_compile = None

    @cuda.jit(
        (
            numba_precision[::1],
            numba_precision[::1],
            numba_precision[::1],
            numba_precision[:, :, ::1],
            numba_precision[::1],
            numba_precision[::1],
            numba_precision[::1],
            numba_precision[::1],
            numba_precision[::1],
            numba_precision,
            numba_precision,
            int16,
            int16,
            numba_precision[::1],
            numba_precision[::1],
            int32[::1],
        ),
        device=True,
        inline=True,
    )
    def step(
        state,
        proposed_state,
        parameters,
        driver_coeffs,
        drivers_buffer,
        proposed_drivers,
        observables,
        proposed_observables,
        error,
        dt_scalar,
        time_scalar,
        first_step_flag,
        accepted_flag,
        shared,
        persistent_local,
        counters,
    ):
        # ----------------------------------------------------------- #
        # Shared and local buffer guide:
        # stage_accumulator: size (stage_count-1) * n, shared memory.
        #   Default behaviour:
        #       - Stores accumulated explicit contributions for successors.
        #       - Slice k feeds the base state for stage k+1.
        #   Reuse:
        #       - stage_base: first slice (size n)
        #           - Holds the working state during the current stage.
        #           - New data lands only after the prior stage has finished.
        # solver_scratch: size solver_shared_elements, shared memory.
        #   Default behaviour:
        #       - Provides workspace for the Newton iteration helpers.
        #   Reuse:
        #       - stage_rhs: first slice (size n)
        #           - Carries the Newton residual and then the stage rhs.
        #           - Once a stage closes we reuse it for the next residual,
        #             so no live data remains.
        #       - increment_cache: second slice (size n)
        #           - Receives the accepted increment at step end for FSAL.
        #           - Solver stops touching it once convergence is reached.
        #   Note:
        #       - Evaluation state is computed inline by operators and
        #         residuals; no dedicated buffer required.
        # stage_increment: size n, per-thread local memory.
        #   Default behaviour:
        #       - Starts as the Newton guess and finishes as the step.
        #       - Copied into increment_cache once the stage closes.
        # proposed_state: size n, global memory.
        #   Default behaviour:
        #       - Carries the running solution with each stage update.
        #       - Only updated after a stage converges, keeping data stable.
        # proposed_drivers / proposed_observables: size n each, global.
        #   Default behaviour:
        #       - Refresh to the stage time before rhs or residual work.
        #       - Later stages reuse only the newest values, so no clashes.
        # ----------------------------------------------------------- #
        stage_increment = cuda.local.array(n, numba_precision)

        # Use compile-time constant dt if fixed controller, else runtime dt
        if is_controller_fixed:
            dt_value = dt_compile
        else:
            dt_value = dt_scalar
        current_time = time_scalar
        end_time = current_time + dt_value

        stage_accumulator = shared[acc_start:acc_end]
        solver_scratch = shared[solver_start:solver_end]
        stage_rhs = solver_scratch[:n]
        increment_cache = solver_scratch[n:2*n]

        #Alias stage base onto first stage accumulator - lifetimes disjoint
        if multistage:
            stage_base = stage_accumulator[:n]
        else:
            stage_base = cuda.local.array(n, numba_precision)

        for idx in range(n):
            if has_error and accumulates_error:
                error[idx] = typed_zero
            stage_increment[idx] = increment_cache[idx] # cache spent

        status_code = int32(0)
        # --------------------------------------------------------------- #
        #            Stage 0: may reuse cached values                     #
        # --------------------------------------------------------------- #

        first_step = first_step_flag != int16(0)
        prev_state_accepted = accepted_flag != int16(0)

        # Only use cache if all threads in warp can - otherwise no gain
        use_cached_rhs = False
        if first_same_as_last and multistage:
            if not first_step_flag:
                mask = activemask()
                all_threads_accepted = all_sync(
                    mask, accepted_flag != int16(0)
                )
                use_cached_rhs = all_threads_accepted
        else:
            use_cached_rhs = False

        stage_time = current_time + dt_value * stage_time_fractions[0]
        diagonal_coeff = diagonal_coeffs[0]

        for idx in range(n):
            stage_base[idx] = state[idx]
            if accumulates_output:
                proposed_state[idx] = typed_zero

        if use_cached_rhs:
            # RHS is aliased onto solver scratch cache at step-end
            pass

        else:
            if can_reuse_accepted_start:
                for idx in range(drivers_buffer.shape[0]):
                    # Use step-start driver values
                    proposed_drivers[idx] = drivers_buffer[idx]

            else:
                if has_driver_function:
                    pass  # driver_function would be called here

            if stage_implicit[0]:
                status_code |= nonlinear_solver(
                    stage_increment,
                    parameters,
                    proposed_drivers,
                    stage_time,
                    dt_value,
                    diagonal_coeffs[0],
                    stage_base,
                    solver_scratch,
                    counters,
                )
                for idx in range(n):
                    stage_base[idx] += (
                        diagonal_coeff * stage_increment[idx]
                    )

            # Get obs->dxdt from stage_base
            observables_function(
                stage_base,
                parameters,
                proposed_drivers,
                proposed_observables,
                stage_time,
            )

            dxdt_fn(
                stage_base,
                parameters,
                proposed_drivers,
                proposed_observables,
                stage_rhs,
                stage_time,
            )

        solution_weight = solution_weights[0]
        error_weight = error_weights[0]
        for idx in range(n):
            rhs_value = stage_rhs[idx]
            # Accumulate if required; save directly if tableau allows
            if accumulates_output:
                # Standard accumulation
                proposed_state[idx] += solution_weight * rhs_value
            elif b_row == 0:
                # Direct assignment when stage 0 matches b_row
                proposed_state[idx] = stage_base[idx]
            if has_error:
                if accumulates_error:
                    # Standard accumulation
                    error[idx] += error_weight * rhs_value
                elif b_hat_row == 0:
                    # Direct assignment for error
                    error[idx] = stage_base[idx]

        for idx in range(accumulator_length):
            stage_accumulator[idx] = typed_zero

        # --------------------------------------------------------------- #
        #            Stages 1-s: must refresh all qtys                    #
        # --------------------------------------------------------------- #

        for stage_idx in range(1, stage_count):
            prev_idx = stage_idx - 1
            successor_range = stage_count - stage_idx
            stage_time = (
                current_time + dt_value * stage_time_fractions[stage_idx]
            )

            # Fill accumulators with previous step's contributions
            for successor_offset in range(successor_range):
                successor_idx = stage_idx + successor_offset
                base = (successor_idx - 1) * n
                for idx in range(n):
                    state_coeff = stage_rhs_coeffs[successor_idx][prev_idx]
                    contribution = state_coeff * stage_rhs[idx] * dt_value
                    stage_accumulator[base + idx] += contribution

            if has_driver_function:
                pass  # driver_function would be called here

            # Grab a view of the completed accumulator slice, add state
            stage_base = stage_accumulator[(stage_idx-1) * n:stage_idx * n]
            for idx in range(n):
                stage_base[idx] += state[idx]

            diagonal_coeff = diagonal_coeffs[stage_idx]

            if stage_implicit[stage_idx]:
                status_code |= nonlinear_solver(
                    stage_increment,
                    parameters,
                    proposed_drivers,
                    stage_time,
                    dt_value,
                    diagonal_coeffs[stage_idx],
                    stage_base,
                    solver_scratch,
                    counters,
                )

                for idx in range(n):
                    stage_base[idx] += diagonal_coeff * stage_increment[idx]

            observables_function(
                stage_base,
                parameters,
                proposed_drivers,
                proposed_observables,
                stage_time,
            )

            dxdt_fn(
                stage_base,
                parameters,
                proposed_drivers,
                proposed_observables,
                stage_rhs,
                stage_time,
            )

            solution_weight = solution_weights[stage_idx]
            error_weight = error_weights[stage_idx]
            for idx in range(n):
                increment = stage_rhs[idx]
                if accumulates_output:
                    proposed_state[idx] += solution_weight * increment
                elif b_row == stage_idx:
                    proposed_state[idx] = stage_base[idx]

                if has_error:
                    if accumulates_error:
                        error[idx] += error_weight * increment
                    elif b_hat_row == stage_idx:
                        # Direct assignment for error
                        error[idx] = stage_base[idx]

        # --------------------------------------------------------------- #

        for idx in range(n):
            if accumulates_output:
                proposed_state[idx] *= dt_value
                proposed_state[idx] += state[idx]
            if has_error:
                if accumulates_error:
                    error[idx] *= dt_value
                else:
                    error[idx] = proposed_state[idx] - error[idx]

        if has_driver_function:
            pass  # driver_function would be called here

        observables_function(
            proposed_state,
            parameters,
            proposed_drivers,
            proposed_observables,
            end_time,
        )

        # RHS auto-cached through aliasing to solver scratch
        for idx in range(n):
            increment_cache[idx] = stage_increment[idx]

        return status_code

    return step


# =========================================================================
# INLINE ERK STEP FACTORY (Generic ERK with tableau)
# =========================================================================

def erk_step_inline_factory(
    dxdt_fn,
    observables_function,
    n,
    prec,
    tableau,
):
    """Create inline ERK step device function matching generic_erk.py."""
    numba_precision = numba_from_dtype(prec)
    typed_zero = numba_precision(0.0)

    stage_count = tableau.stage_count
    accumulator_length = max(stage_count - 1, 0) * n

    # Compile-time toggles
    has_driver_function = False  # No driver function in this test
    first_same_as_last = tableau.first_same_as_last
    multistage = stage_count > 1
    has_error = tableau.has_error_estimate

    stage_rhs_coeffs = tableau.typed_rows(tableau.a, numba_precision)
    solution_weights = tableau.typed_vector(tableau.b, numba_precision)
    stage_nodes = tableau.typed_vector(tableau.c, numba_precision)

    if has_error:
        embedded_weights = tableau.typed_vector(tableau.b_hat, numba_precision)
        error_weights = tableau.error_weights(numba_precision)
    else:
        embedded_weights = tuple(typed_zero for _ in range(stage_count))
        error_weights = tuple(typed_zero for _ in range(stage_count))

    # Last-step caching optimization
    accumulates_output = tableau.accumulates_output
    accumulates_error = tableau.accumulates_error
    b_row = tableau.b_matches_a_row
    b_hat_row = tableau.b_hat_matches_a_row

    # Compile-time fixed controller (always use dt_scalar for this test)
    is_controller_fixed = False
    dt_compile = None

    @cuda.jit(
        (
            numba_precision[::1],
            numba_precision[::1],
            numba_precision[::1],
            numba_precision[:, :, ::1],
            numba_precision[::1],
            numba_precision[::1],
            numba_precision[::1],
            numba_precision[::1],
            numba_precision[::1],
            numba_precision,
            numba_precision,
            int16,
            int16,
            numba_precision[::1],
            numba_precision[::1],
            int32[::1],
        ),
        device=True,
        inline=True,
    )
    def step(
        state,
        proposed_state,
        parameters,
        driver_coeffs,
        drivers_buffer,
        proposed_drivers,
        observables,
        proposed_observables,
        error,
        dt_scalar,
        time_scalar,
        first_step_flag,
        accepted_flag,
        shared,
        persistent_local,
        counters,
    ):
        # ----------------------------------------------------------- #
        # Shared and local buffer guide:
        # stage_accumulator: size (stage_count-1) * n, shared memory.
        #   Default behaviour:
        #       - Holds finished stage rhs * dt for later stage sums.
        #       - Slice k stores contributions streamed into stage k+1.
        #   Reuse:
        #       - stage_cache: first slice (size n)
        #           - Saves the FSAL rhs when the tableau supports it.
        #           - Cache survives after the loop so no live slice is hit.
        # proposed_state: size n, global memory.
        #   Default behaviour:
        #       - Starts as the accepted state and gathers stage updates.
        #       - Each stage applies its weighted increment before moving on.
        # proposed_drivers / proposed_observables: size n each, global.
        #   Default behaviour:
        #       - Refresh to the current stage time before rhs evaluation.
        #       - Later stages only read the newest values, so nothing lingers.
        # stage_rhs: size n, per-thread local memory.
        #   Default behaviour:
        #       - Holds the current stage rhs before scaling by dt.
        #   Reuse:
        #       - When FSAL hits we copy cached rhs here before touching
        #         shared memory, keeping lifetimes separate.
        # error: size n, global memory (adaptive runs only).
        #   Default behaviour:
        #       - Accumulates error-weighted f(y_jn) during the loop.
        #       - Cleared at loop entry so prior steps cannot leak in.
        # ----------------------------------------------------------- #
        stage_rhs = cuda.local.array(n, numba_precision)

        # Use compile-time constant dt if fixed controller, else runtime dt
        if is_controller_fixed:
            dt_value = dt_compile
        else:
            dt_value = dt_scalar

        current_time = time_scalar
        end_time = current_time + dt_value

        stage_accumulator = shared[:accumulator_length]
        if multistage:
            stage_cache = stage_accumulator[:n]

        for idx in range(n):
            if accumulates_output:
                proposed_state[idx] = typed_zero
            if has_error and accumulates_error:
                error[idx] = typed_zero

        # ----------------------------------------------------------- #
        #            Stage 0: may use cached values                   #
        # ----------------------------------------------------------- #
        # Only use cache if all threads in warp can - otherwise no gain
        use_cached_rhs = False
        if first_same_as_last and multistage:
            if not first_step_flag:
                mask = activemask()
                all_threads_accepted = all_sync(
                    mask, accepted_flag != int16(0)
                )
                use_cached_rhs = all_threads_accepted
        else:
            use_cached_rhs = False

        if multistage:
            if use_cached_rhs:
                for idx in range(n):
                    stage_rhs[idx] = stage_cache[idx]
            else:
                dxdt_fn(
                    state,
                    parameters,
                    drivers_buffer,
                    observables,
                    stage_rhs,
                    current_time,
                )
        else:
            dxdt_fn(
                state,
                parameters,
                drivers_buffer,
                observables,
                stage_rhs,
                current_time,
            )

        # b weights can't match a rows for erk, as they would return 0
        # So we include ifs to skip accumulating but do no direct assign.
        for idx in range(n):
            increment = stage_rhs[idx]
            if accumulates_output:
                proposed_state[idx] += solution_weights[0] * increment
            if has_error:
                if accumulates_error:
                    error[idx] += error_weights[0] * increment

        for idx in range(accumulator_length):
            stage_accumulator[idx] = typed_zero

        # ----------------------------------------------------------- #
        #            Stages 1-s: refresh observables and drivers       #
        # ----------------------------------------------------------- #

        for stage_idx in range(1, stage_count):

            # Stream last result into the accumulators
            prev_idx = stage_idx - 1
            successor_range = stage_count - stage_idx

            for successor_offset in range(successor_range):
                successor_idx = stage_idx + successor_offset
                state_coeff = stage_rhs_coeffs[successor_idx][prev_idx]
                base = (successor_idx - 1) * n
                for idx in range(n):
                    increment = stage_rhs[idx]
                    contribution = state_coeff * increment
                    stage_accumulator[base + idx] += contribution

            stage_offset = (stage_idx - 1) * n
            dt_stage = dt_value * stage_nodes[stage_idx]
            stage_time = current_time + dt_stage

            # Convert accumulated gradients sum(f(y_nj) into a state y_j
            for idx in range(n):
                stage_accumulator[stage_offset + idx] *= dt_value
                stage_accumulator[stage_offset + idx] += state[idx]

            # Rename the slice for clarity
            stage_state = stage_accumulator[stage_offset:stage_offset + n]

            # get rhs for next stage
            stage_drivers = proposed_drivers
            if has_driver_function:
                pass  # driver_function would be called here

            observables_function(
                stage_state,
                parameters,
                stage_drivers,
                proposed_observables,
                stage_time,
            )

            dxdt_fn(
                stage_state,
                parameters,
                stage_drivers,
                proposed_observables,
                stage_rhs,
                stage_time,
            )

            # Accumulate f(y_jn) terms or capture direct stage state
            solution_weight = solution_weights[stage_idx]
            error_weight = error_weights[stage_idx]
            for idx in range(n):
                if accumulates_output:
                    increment = stage_rhs[idx]
                    proposed_state[idx] += solution_weight * increment
                elif b_row == stage_idx:
                    proposed_state[idx] = stage_state[idx]

                if has_error:
                    if accumulates_error:
                        increment = stage_rhs[idx]
                        error[idx] += error_weight * increment
                    elif b_hat_row == stage_idx:
                        error[idx] = stage_state[idx]

        # ----------------------------------------------------------- #
        for idx in range(n):

            # Scale and shift f(Y_n) value if accumulated
            if accumulates_output:
                proposed_state[idx] *= dt_value
                proposed_state[idx] += state[idx]

            if has_error:
                # Scale error if accumulated
                if accumulates_error:
                    error[idx] *= dt_value

                # Or form error from difference if captured from a-row
                else:
                    error[idx] = proposed_state[idx] - error[idx]

        if has_driver_function:
            pass  # driver_function would be called here

        observables_function(
            proposed_state,
            parameters,
            proposed_drivers,
            proposed_observables,
            end_time,
        )

        if first_same_as_last:
            for idx in range(n):
                stage_cache[idx] = stage_rhs[idx]

        return int32(0)

    return step


# =========================================================================
# OUTPUT FUNCTIONS
# =========================================================================

@cuda.jit(device=True, inline=True, **compile_kwargs)
def save_state_inline(current_state, current_observables, current_counters,
                      current_step, output_states_slice, output_obs_slice,
                      output_counters_slice):
    for k in range(n_states):
        output_states_slice[k] = current_state[k]
    output_states_slice[n_states] = current_step
    for i in range(n_counters):
        output_counters_slice[i] = current_counters[i]


# =========================================================================
# SUMMARY METRIC FUNCTIONS (Mean metric with chained pattern)
# =========================================================================

@cuda.jit(
    ["float32, float32[::1], int32, int32",
     "float64, float64[::1], int32, int32"],
    device=True,
    inline=True,
)
def update_mean(
    value,
    buffer,
    current_index,
    customisable_variable,
):
    """Update the running sum with a new value."""
    buffer[0] += value


@cuda.jit(
    ["float32[::1], float32[::1], int32, int32",
     "float64[::1], float64[::1], int32, int32"],
    device=True,
    inline=True,
)
def save_mean(
    buffer,
    output_array,
    summarise_every,
    customisable_variable,
):
    """Calculate the mean and reset the buffer."""
    output_array[0] = buffer[0] / summarise_every
    buffer[0] = precision(0.0)


@cuda.jit(device=True, inline=True, **compile_kwargs)
def chain_update_metrics(
    value,
    buffer,
    current_step,
):
    """Chain all metric update functions."""
    # For mean metric: buffer offset 0, size 1, param 0
    update_mean(value, buffer[0:1], current_step, 0)


@cuda.jit(device=True, inline=True, **compile_kwargs)
def chain_save_metrics(
    buffer,
    output,
    summarise_every,
):
    """Chain all metric save functions."""
    # For mean metric: buffer offset 0, size 1, output offset 0, size 1, param 0
    save_mean(buffer[0:1], output[0:1], summarise_every, 0)


@cuda.jit(device=True, inline=True, **compile_kwargs)
def update_summaries_inline(
    current_state,
    current_observables,
    state_summary_buffer,
    obs_summary_buffer,
    current_step,
):
    """Accumulate summary metrics from the current state sample."""
    total_buffer_size = 1  # 1 slot for mean metric per variable
    for idx in range(n_states):
        start = idx * total_buffer_size
        end = start + total_buffer_size
        chain_update_metrics(
            current_state[idx],
            state_summary_buffer[start:end],
            current_step,
        )


@cuda.jit(device=True, inline=True, **compile_kwargs)
def save_summaries_inline(
    buffer_state,
    buffer_obs,
    output_state,
    output_obs,
    summarise_every,
):
    """Export summary metrics from buffers to output windows."""
    total_buffer_size = 1  # 1 slot for mean metric per variable
    total_output_size = 1  # 1 output for mean metric per variable
    for state_index in range(n_states):
        buffer_start = state_index * total_buffer_size
        out_start = state_index * total_output_size
        chain_save_metrics(
            buffer_state[buffer_start:buffer_start + total_buffer_size],
            output_state[out_start:out_start + total_output_size],
            summarise_every,
        )


# =========================================================================
# STEP CONTROLLER (Fixed and adaptive options)
# =========================================================================

@cuda.jit(device=True, inline=True)
def clamp(value, min_val, max_val):
    """Clamp a value between min and max."""
    return max(min_val, min(value, max_val))


@cuda.jit(device=True, inline=True, **compile_kwargs)
def step_controller_fixed(dt, state, state_prev, error, niters, accept_out,
                          local_temp):
    accept_out[0] = int32(1)
    return int32(0)


def controller_PID_factory(
    precision,
    n,
    atol,
    rtol,
    algorithm_order,
    kp,
    ki,
    kd,
    min_gain,
    max_gain,
    dt_min_ctrl,
    dt_max_ctrl,
    deadband_min,
    deadband_max,
    safety,
):
    """Create PID controller device function matching adaptive_PID_controller.py."""
    numba_precision = numba_from_dtype(precision)

    expo1 = precision(kp / (2 * (algorithm_order + 1)))
    expo2 = precision(ki / (2 * (algorithm_order + 1)))
    expo3 = precision(kd / (2 * (algorithm_order + 1)))
    unity_gain = precision(1.0)
    deadband_min = precision(deadband_min)
    deadband_max = precision(deadband_max)
    deadband_disabled = (deadband_min == unity_gain) and (
        deadband_max == unity_gain
    )
    dt_min = precision(dt_min_ctrl)
    dt_max = precision(dt_max_ctrl)
    min_gain = precision(min_gain)
    max_gain = precision(max_gain)
    safety = precision(safety)

    @cuda.jit(
        [
            (
                numba_precision[::1],
                numba_precision[::1],
                numba_precision[::1],
                numba_precision[::1],
                int32,
                int32[::1],
                numba_precision[::1],
            )
        ],
        device=True,
        inline=True,
        fastmath=True,
        **compile_kwargs,
    )
    def controller_PID(
        dt,
        state,
        state_prev,
        error,
        niters,
        accept_out,
        local_temp,
    ):
        """Proportional–integral–derivative accept/step controller."""
        err_prev = local_temp[0]
        err_prev_prev = local_temp[1]
        nrm2 = precision(0.0)

        for i in range(n):
            error_i = max(abs(error[i]), precision(1e-12))
            tol = atol[i] + rtol[i] * max(
                abs(state[i]), abs(state_prev[i])
            )
            ratio = tol / error_i
            nrm2 += ratio * ratio

        nrm2 = precision(nrm2/n)
        accept = nrm2 >= precision(1.0)
        accept_out[0] = int32(1) if accept else int32(0)
        err_prev_safe = err_prev if err_prev > precision(0.0) else nrm2
        err_prev_prev_safe = (
            err_prev_prev if err_prev_prev > precision(0.0) else err_prev_safe
        )

        gain_new = precision(
            safety
            * (nrm2 ** (expo1))
            * (err_prev_safe ** (expo2))
            * (err_prev_prev_safe ** (expo3))
        )
        gain = precision(clamp(gain_new, min_gain, max_gain))
        if not deadband_disabled:
            within_deadband = (
                (gain >= deadband_min)
                and (gain <= deadband_max)
            )
            gain = selp(within_deadband, unity_gain, gain)

        dt_new_raw = dt[0] * gain
        dt[0] = clamp(dt_new_raw, dt_min, dt_max)
        local_temp[1] = err_prev
        local_temp[0] = nrm2

        ret = int32(0) if dt_new_raw > dt_min else int32(8)
        return ret

    return controller_PID


# =========================================================================
# BUILD DEVICE FUNCTIONS
# =========================================================================

# Build all device functions using factories with explicit arguments
dxdt_fn = dxdt_factory(constants, precision)
observables_function = observables_factory(constants, precision)
preconditioner_fn = neumann_preconditioner_factory(
    constants,
    precision,
    beta=float(beta_solver),
    gamma=float(gamma_solver),
    order=preconditioner_order,
)
residual_fn = stage_residual_factory(
    constants,
    precision,
    beta=float(beta_solver),
    gamma=float(gamma_solver),
    order=preconditioner_order,
)
operator_fn = linear_operator_factory(
    constants,
    precision,
    beta=float(beta_solver),
    gamma=float(gamma_solver),
    order=preconditioner_order,
)

linear_solver_fn = linear_solver_inline_factory(
    operator_fn,
    n_states,
    preconditioner_fn,
    float(krylov_tolerance),
    max_linear_iters,
    precision,
)

newton_solver_fn = newton_krylov_inline_factory(
    residual_fn,
    linear_solver_fn,
    n_states,
    float(newton_tolerance),
    max_newton_iters,
    float(newton_damping),
    max_backtracks,
    precision,
)

dirk_step_fn = dirk_step_inline_factory(
    newton_solver_fn,
    dxdt_fn,
    observables_function,
    n_states,
    precision,
    SDIRK_2_2_TABLEAU,
)

# Optional: ERK step for explicit integration
# erk_step_fn = erk_step_inline_factory(
#     dxdt_fn,
#     observables_function,
#     n_states,
#     precision,
#     DORMAND_PRINCE_54_TABLEAU,
# )

# Optional: PID controller for adaptive stepping
# controller_PID_fn = controller_PID_factory(
#     precision,
#     n_states,
#     atol,
#     rtol,
#     algorithm_order,
#     float(kp),
#     float(ki),
#     float(kd),
#     float(min_gain),
#     float(max_gain),
#     float(dt_min_ctrl),
#     float(dt_max_ctrl),
#     float(deadband_min),
#     float(deadband_max),
#     float(safety),
# )


# =========================================================================
# INTEGRATION LOOP
# =========================================================================

# Buffer layout
state_shared_start = 0
state_shared_end = n_states
proposed_state_start = state_shared_end
proposed_state_end = proposed_state_start + n_states
params_start = proposed_state_end
params_end = params_start + n_parameters
drivers_start = params_end
drivers_end = drivers_start + max(n_drivers, 1)
proposed_drivers_start = drivers_end
proposed_drivers_end = proposed_drivers_start + max(n_drivers, 1)
obs_start = proposed_drivers_end
obs_end = obs_start + max(n_observables, 1)
proposed_obs_start = obs_end
proposed_obs_end = proposed_obs_start + max(n_observables, 1)
error_start = proposed_obs_end
error_end = error_start + n_states
counters_start = error_end
counters_end = counters_start + n_counters
proposed_counters_start = counters_end
proposed_counters_end = proposed_counters_start + 2
# DIRK needs accumulator + solver scratch
accumulator_size = (stage_count - 1) * n_states
solver_scratch_size = 2 * n_states
dirk_scratch_start = proposed_counters_end
dirk_scratch_end = dirk_scratch_start + accumulator_size + solver_scratch_size
state_summ_start = dirk_scratch_end
state_summ_end = state_summ_start + n_states
obs_summ_start = state_summ_end
obs_summ_end = obs_summ_start + max(n_observables, 1)
scratch_start = obs_summ_end
shared_elements = scratch_start + 64

local_dt_slice = slice(0, 1)
local_accept_slice = slice(1, 2)
local_controller_slice = slice(2, 4)
local_algo_slice = slice(4, 8)
local_elements = 8

steps_per_save = int32(ceil(float(dt_save) / float(dt0)))
saves_per_summary = int32(2)
status_mask = int32(0xFFFF)


@cuda.jit(device=True, inline=True, **compile_kwargs)
def loop_fn(initial_states, parameters, driver_coefficients, shared_scratch,
            persistent_local, state_output, observables_output,
            state_summaries_output, observable_summaries_output,
            iteration_counters_output, duration_arg, settling_time, t0_arg):
    t = precision(t0_arg)
    t_end = precision(settling_time + duration_arg)
    max_steps = (int32(ceil(t_end / dt_min)) + int32(2))
    max_steps = max_steps << 2

    n_output_samples_local = state_output.shape[0]
    shared_scratch[:] = typed_zero

    state_buffer = shared_scratch[state_shared_start:state_shared_end]
    state_proposal = shared_scratch[proposed_state_start:proposed_state_end]
    params_buffer = shared_scratch[params_start:params_end]
    drivers_buffer = shared_scratch[drivers_start:drivers_end]
    drivers_proposal = shared_scratch[proposed_drivers_start:
                                      proposed_drivers_end]
    obs_buffer = shared_scratch[obs_start:obs_end]
    obs_proposal = shared_scratch[proposed_obs_start:proposed_obs_end]
    error_buffer = shared_scratch[error_start:error_end]
    counters_since_save = shared_scratch[counters_start:counters_end]
    proposed_counters = shared_scratch[proposed_counters_start:
                                       proposed_counters_end]
    dirk_scratch = shared_scratch[dirk_scratch_start:dirk_scratch_end]
    state_summ_buffer = shared_scratch[state_summ_start:state_summ_end]
    obs_summ_buffer = shared_scratch[obs_summ_start:obs_summ_end]

    dt_local = persistent_local[local_dt_slice]
    accept_step = persistent_local[local_accept_slice].view(simsafe_int32)
    _controller_temp = persistent_local[local_controller_slice]
    algo_local = persistent_local[local_algo_slice]

    for k in range(n_states):
        state_buffer[k] = initial_states[k]
    for k in range(n_parameters):
        params_buffer[k] = parameters[k]

    save_idx = int32(0)
    summary_idx = int32(0)
    _next_save = precision(dt_save)  # Available for adaptive stepping

    # Save initial state
    save_state_inline(state_buffer, obs_buffer, counters_since_save, t,
                      state_output[save_idx, :], observables_output[0, :],
                      iteration_counters_output[save_idx, :])
    update_summaries_inline(state_buffer, obs_buffer, state_summ_buffer,
                            obs_summ_buffer, save_idx)
    save_idx += int32(1)

    status = int32(0)
    dt_local[0] = dt0
    dt_eff = dt_local[0]
    accept_step[0] = int32(0)

    for i in range(n_counters):
        counters_since_save[i] = int32(0)

    step_counter = int32(0)
    mask = activemask()

    for _ in range(max_steps):
        finished = save_idx >= n_output_samples_local
        if all_sync(mask, finished):
            return status

        if not finished:
            step_counter += 1
            do_save = (step_counter % steps_per_save) == 0
            if do_save:
                step_counter = int32(0)

            step_status = dirk_step_fn(
                state_buffer, state_proposal, params_buffer,
                driver_coefficients, drivers_buffer, drivers_proposal,
                obs_buffer, obs_proposal, error_buffer, dt_eff, t,
                int16(0), int16(1), dirk_scratch, algo_local,
                proposed_counters.view(simsafe_int32))

            status |= step_status & status_mask

            t_proposal = t + dt_eff
            t = t_proposal

            for i in range(n_states):
                state_buffer[i] = state_proposal[i]

            for i in range(n_counters):
                if i < 2:
                    counters_since_save[i] += proposed_counters.view(
                        simsafe_int32)[i]
                elif i == 2:
                    counters_since_save[i] += int32(1)

            if do_save:
                save_state_inline(state_buffer, obs_buffer,
                                  counters_since_save, t,
                                  state_output[save_idx, :],
                                  observables_output[0, :],
                                  iteration_counters_output[save_idx, :])
                update_summaries_inline(state_buffer, obs_buffer,
                                        state_summ_buffer, obs_summ_buffer,
                                        save_idx)

                if (save_idx + 1) % saves_per_summary == 0:
                    save_summaries_inline(state_summ_buffer, obs_summ_buffer,
                                          state_summaries_output[summary_idx,
                                                                 :],
                                          observable_summaries_output[0, :],
                                          saves_per_summary)
                    summary_idx += 1

                save_idx += 1
                for i in range(n_counters):
                    counters_since_save[i] = int32(0)

    if status == int32(0):
        status = int32(32)
    return status


# =========================================================================
# KERNEL
# =========================================================================

local_elements_per_run = local_elements
shared_elems_per_run = shared_elements
f32_per_element = 2 if (precision == np.float64) else 1


@cuda.jit(**compile_kwargs)
def integration_kernel(inits, params, d_coefficients, state_output,
                       observables_output, state_summaries_output,
                       observables_summaries_output, iteration_counters_output,
                       status_codes_output, duration_k, warmup_k, t0_k,
                       n_runs_k):
    run_index = cuda.grid(1)
    if run_index >= n_runs_k:
        return

    shared_memory = cuda.shared.array(0, dtype=float32)
    local_scratch = cuda.local.array(local_elements_per_run, dtype=float32)
    c_coefficients = cuda.const.array_like(d_coefficients)

    run_idx_low = 0
    run_idx_high = f32_per_element * shared_elems_per_run
    rx_shared_memory = shared_memory[run_idx_low:run_idx_high].view(
        simsafe_precision)

    rx_inits = inits[run_index, :]
    rx_params = params[run_index, :]
    rx_state = state_output[:, run_index, :]
    rx_observables = observables_output[:, 0, :]
    rx_state_summaries = state_summaries_output[:, run_index, :]
    rx_observables_summaries = observables_summaries_output[:, 0, :]
    rx_iteration_counters = iteration_counters_output[run_index, :, :]

    status = loop_fn(rx_inits, rx_params, c_coefficients, rx_shared_memory,
                     local_scratch, rx_state, rx_observables,
                     rx_state_summaries, rx_observables_summaries,
                     rx_iteration_counters, duration_k, warmup_k, t0_k)

    status_codes_output[run_index] = status


# =========================================================================
# MAIN EXECUTION
# =========================================================================

def run_debug_integration(n_runs=2**23, rho_min=0.0, rho_max=21.0):
    """Execute debug integration with 2^23 runs."""
    print("=" * 70)
    print("Debug Integration - DIRK SDIRK_2_2 with Newton-Krylov")
    print("=" * 70)
    print(f"\nRunning {n_runs:,} integrations with rho in [{rho_min}, {rho_max}]")

    # Generate rho values
    rho_values = np.linspace(rho_min, rho_max, n_runs, dtype=precision)

    # Input arrays (NumPy)
    inits = np.zeros((n_runs, n_states), dtype=precision)
    inits[:, 0] = precision(1.0)
    inits[:, 1] = precision(0.0)
    inits[:, 2] = precision(0.0)

    params = np.zeros((n_runs, n_parameters), dtype=precision)
    params[:, 0] = rho_values

    d_coefficients = np.zeros((1, max(n_drivers, 1), 6), dtype=precision)

    # Create device arrays for inputs (BatchInputArrays pattern)
    d_inits = cuda.to_device(inits)
    d_params = cuda.to_device(params)
    d_driver_coefficients = cuda.to_device(d_coefficients)

    # Create mapped arrays for outputs (BatchOutputArrays pattern)
    n_summary_samples = int(ceil(n_output_samples / saves_per_summary))
    state_output = cuda.mapped_array(
        (n_output_samples, n_runs, n_states + 1), dtype=precision
    )
    observables_output = cuda.mapped_array(
        (n_output_samples, 1, 1), dtype=precision
    )
    state_summaries_output = cuda.mapped_array(
        (n_summary_samples, n_runs, n_states), dtype=precision
    )
    observable_summaries_output = cuda.mapped_array(
        (n_summary_samples, 1, 1), dtype=precision
    )
    iteration_counters_output = cuda.mapped_array(
        (n_runs, n_output_samples, n_counters), dtype=np.int32
    )
    status_codes_output = cuda.mapped_array(
        (n_runs,), dtype=np.int32
    )

    print(f"State output shape: {state_output.shape}")

    # Kernel configuration
    MAX_SHARED_MEMORY_PER_BLOCK = 32768
    blocksize = 256
    runs_per_block = blocksize
    dynamic_sharedmem = int(
        (f32_per_element * shared_elems_per_run) * 4 * runs_per_block)

    while dynamic_sharedmem > MAX_SHARED_MEMORY_PER_BLOCK:
        blocksize = blocksize // 2
        runs_per_block = blocksize
        dynamic_sharedmem = int(
            (f32_per_element * shared_elems_per_run) * 4 * runs_per_block)

    blocks_per_grid = int(ceil(n_runs / runs_per_block))

    print("\nKernel configuration:")
    print(f"  Block size: {blocksize}")
    print(f"  Blocks per grid: {blocks_per_grid}")
    print(f"  Shared memory per block: {dynamic_sharedmem} bytes")

    print("\nLaunching kernel...")

    integration_kernel[blocks_per_grid, blocksize, 0, dynamic_sharedmem](
        d_inits, d_params, d_driver_coefficients, state_output,
        observables_output, state_summaries_output,
        observable_summaries_output, iteration_counters_output,
        status_codes_output, duration, warmup, precision(0.0), n_runs)

    cuda.synchronize()
    # Mapped arrays provide direct host access after synchronization
    # No explicit copy_to_host required

    print("\n" + "=" * 70)
    print("Integration Complete")
    print("=" * 70)

    success_count = np.sum(status_codes_output == 0)
    dt_min_exceeded_runs = np.sum(status_codes_output == 8)
    max_steps_exceeded_runs = np.sum(status_codes_output == 32)
    max_newton_backtracks_runs = np.sum(status_codes_output == 1)
    max_newton_iters_runs = np.sum(status_codes_output == 2)
    linear_max_iters_runs = np.sum(status_codes_output == 4)
    print(f"\nSuccessful runs: {success_count:,} / {n_runs:,}")
    print(f"\ndt_min exceeded runs: {dt_min_exceeded_runs:,}")
    print(f"\nmax steps exceeded runs: {max_steps_exceeded_runs:,}")
    print(f"\nmax newton backtracks runs: {max_newton_backtracks_runs:,}")
    print(f"\nmax newton iters exceeded runs: {max_newton_iters_runs:,}")
    print(f"\nlinear solver max iters exceeded runs: {linear_max_iters_runs:,}")

    print(f"\nSuccessful runs: {success_count:,} / {n_runs:,}")


    print(f"Final state sample (run 0): {state_output[-1, 0, :n_states]}")
    print(f"Final state sample (run -1): {state_output[-1, -1, :n_states]}")

    return state_output, status_codes_output


if __name__ == "__main__":
    run_debug_integration()