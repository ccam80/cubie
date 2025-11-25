"""All-in-one debug file for Numba lineinfo debugging.

All CUDA device functions are consolidated in this single file to enable
proper line-level debugging with Numba's lineinfo feature.
"""
# ruff: noqa: E402
from math import ceil

import numpy as np
from numba import cuda, int16, int32, float32, float64
from numba import from_dtype as numba_from_dtype
from cubie.cuda_simsafe import activemask, all_sync, selp, compile_kwargs
from cubie.cuda_simsafe import from_dtype as simsafe_dtype

# =========================================================================
# CONFIGURATION
# =========================================================================

precision = np.float32
numba_precision = numba_from_dtype(precision)
simsafe_precision = simsafe_dtype(precision)
simsafe_int32 = simsafe_dtype(np.int32)

# Lorenz system constants
constants = {'sigma': 10.0, 'beta': 8.0 / 3.0}

# System dimensions
n_states = 3
n_parameters = 1
n_observables = 0
n_drivers = 0
n_counters = 4

# Time parameters
duration = precision(1.0)
warmup = precision(0.0)
dt_save = precision(0.1)
dt0 = precision(0.01)
dt_min = precision(1e-7)

# SDIRK 2,2 tableau (Alexander)
SQRT2 = 2.0 ** 0.5
gamma_tableau = (2.0 - SQRT2) / 2.0  # diagonal coefficient ~0.2929
a_00 = precision(gamma_tableau)
a_10 = precision(1.0 - gamma_tableau)
a_11 = precision(gamma_tableau)
b_0 = precision(0.5)
b_1 = precision(0.5)
b_hat_0 = precision(-0.5)  # for error estimate
b_hat_1 = precision(0.5)
c_0 = precision(gamma_tableau)
c_1 = precision(1.0)
stage_count = 2

# Newton-Krylov parameters
krylov_tolerance = precision(1e-6)
newton_tolerance = precision(1e-6)
max_linear_iters = 200
max_newton_iters = 100
newton_damping = precision(0.5)
max_backtracks = 8
preconditioner_order = 2
beta_solver = 1.0
gamma_solver = 1.0

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


def neumann_preconditioner_factory(constants, prec, beta=1.0, gamma=1.0,
                                   order=1):
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


def stage_residual_factory(constants, prec, beta=1.0, gamma=1.0, order=None):
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


def linear_operator_factory(constants, prec, beta=1.0, gamma=1.0, order=None):
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
    typed_zero = numba_prec(0.0)
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
# INLINE DIRK STEP FACTORY (SDIRK 2,2)
# =========================================================================

def dirk_step_inline_factory(nonlinear_solver, dxdt_fn, observables_fn,
                             n, prec):
    """Create inline DIRK step device function for SDIRK 2,2 tableau."""
    numba_prec = numba_from_dtype(prec)
    typed_zero = numba_prec(0.0)
    solver_shared_elements = 2 * n  # delta + residual for Newton solver
    accumulator_length = (stage_count - 1) * n
    acc_start = 0
    acc_end = accumulator_length
    solver_start = acc_end
    solver_end = solver_start + solver_shared_elements

    @cuda.jit(device=True, inline=True, **compile_kwargs)
    def step(state, proposed_state, parameters, driver_coeffs, drivers_buffer,
             proposed_drivers, observables, proposed_observables, error,
             dt_scalar, time_scalar, first_step_flag, accepted_flag, shared,
             persistent_local, counters):
        stage_increment = cuda.local.array(n, numba_prec)
        dt_value = dt_scalar
        current_time = time_scalar

        stage_accumulator = shared[acc_start:acc_end]
        solver_scratch = shared[solver_start:solver_end]
        stage_rhs = solver_scratch[:n]
        increment_cache = solver_scratch[n:2 * n]
        stage_base = stage_accumulator[:n]

        for idx in range(n):
            error[idx] = typed_zero
            stage_increment[idx] = increment_cache[idx]

        status_code = int32(0)

        # --- Stage 0 ---
        stage_time_0 = current_time + dt_value * c_0
        for idx in range(n):
            stage_base[idx] = state[idx]
            proposed_state[idx] = typed_zero

        # Implicit solve for stage 0
        status_code |= nonlinear_solver(stage_increment, parameters,
                                        proposed_drivers, stage_time_0,
                                        dt_value, a_00, stage_base,
                                        solver_scratch, counters)
        for idx in range(n):
            stage_base[idx] += a_00 * stage_increment[idx]

        # Evaluate observables and dxdt at stage 0
        observables_fn(stage_base, parameters, proposed_drivers,
                       proposed_observables, stage_time_0)
        dxdt_fn(stage_base, parameters, proposed_drivers, proposed_observables,
                stage_rhs, stage_time_0)

        # Accumulate stage 0 contributions
        for idx in range(n):
            rhs_value = stage_rhs[idx]
            proposed_state[idx] += b_0 * rhs_value
            error[idx] += b_hat_0 * rhs_value

        for idx in range(accumulator_length):
            stage_accumulator[idx] = typed_zero

        # --- Stage 1 ---
        stage_time_1 = current_time + dt_value * c_1

        # Fill accumulator with stage 0 contributions for stage 1
        for idx in range(n):
            stage_accumulator[idx] += a_10 * stage_rhs[idx] * dt_value

        # Build stage_base for stage 1
        for idx in range(n):
            stage_base[idx] = stage_accumulator[idx] + state[idx]

        # Implicit solve for stage 1
        status_code |= nonlinear_solver(stage_increment, parameters,
                                        proposed_drivers, stage_time_1,
                                        dt_value, a_11, stage_base,
                                        solver_scratch, counters)
        for idx in range(n):
            stage_base[idx] += a_11 * stage_increment[idx]

        # Evaluate observables and dxdt at stage 1
        observables_fn(stage_base, parameters, proposed_drivers,
                       proposed_observables, stage_time_1)
        dxdt_fn(stage_base, parameters, proposed_drivers, proposed_observables,
                stage_rhs, stage_time_1)

        # Accumulate stage 1 contributions
        for idx in range(n):
            rhs_value = stage_rhs[idx]
            proposed_state[idx] += b_1 * rhs_value
            error[idx] += b_hat_1 * rhs_value

        # Finalize proposed_state and error
        for idx in range(n):
            proposed_state[idx] *= dt_value
            proposed_state[idx] += state[idx]
            error[idx] *= dt_value

        # Cache increment for FSAL
        for idx in range(n):
            increment_cache[idx] = stage_increment[idx]

        return status_code
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


@cuda.jit(device=True, inline=True, **compile_kwargs)
def update_summaries_inline(current_state, current_observables,
                            state_summary_buffer, obs_summary_buffer,
                            current_step):
    for idx in range(n_states):
        state_summary_buffer[idx] += current_state[idx]


@cuda.jit(device=True, inline=True, **compile_kwargs)
def save_summaries_inline(buffer_state, buffer_obs, output_state,
                          output_obs, summarise_every):
    for idx in range(n_states):
        mean_val = buffer_state[idx] / numba_precision(summarise_every)
        output_state[idx] = mean_val
        buffer_state[idx] = numba_precision(0.0)


# =========================================================================
# STEP CONTROLLER (Fixed for now, adaptive can be added)
# =========================================================================

@cuda.jit(device=True, inline=True, **compile_kwargs)
def step_controller_fixed(dt, state, state_prev, error, niters, accept_out,
                          local_temp):
    accept_out[0] = int32(1)
    return int32(0)


# =========================================================================
# BUILD DEVICE FUNCTIONS
# =========================================================================

# Build all device functions using factories
dxdt_fn = dxdt_factory(constants, precision)
observables_fn = observables_factory(constants, precision)
preconditioner_fn = neumann_preconditioner_factory(
    constants, precision, beta=beta_solver, gamma=gamma_solver,
    order=preconditioner_order)
residual_fn = stage_residual_factory(
    constants, precision, beta=beta_solver, gamma=gamma_solver)
operator_fn = linear_operator_factory(
    constants, precision, beta=beta_solver, gamma=gamma_solver)

linear_solver_fn = linear_solver_inline_factory(
    operator_fn, n_states, preconditioner_fn,
    float(krylov_tolerance), max_linear_iters, precision)

newton_solver_fn = newton_krylov_inline_factory(
    residual_fn, linear_solver_fn, n_states,
    float(newton_tolerance), max_newton_iters,
    float(newton_damping), max_backtracks, precision)

dirk_step_fn = dirk_step_inline_factory(
    newton_solver_fn, dxdt_fn, observables_fn, n_states, precision)


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
    t = numba_precision(t0_arg)
    t_end = numba_precision(settling_time + duration_arg)
    max_steps = (int32(ceil(t_end / dt_min)) + int32(2))
    max_steps = max_steps << 2

    n_output_samples_local = state_output.shape[0]
    shared_scratch[:] = numba_precision(0.0)

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
    _next_save = numba_precision(dt_save)  # Available for adaptive stepping

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
f32_per_element = 2 if (numba_precision == float64) else 1


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

    # Input arrays
    inits = np.zeros((n_runs, n_states), dtype=precision)
    inits[:, 0] = precision(1.0)
    inits[:, 1] = precision(0.0)
    inits[:, 2] = precision(0.0)

    params = np.zeros((n_runs, n_parameters), dtype=precision)
    params[:, 0] = rho_values

    d_coefficients = np.zeros((1, max(n_drivers, 1), 6), dtype=precision)

    # Output arrays
    state_output = np.zeros((n_output_samples, n_runs, n_states + 1),
                            dtype=precision)
    observables_output = np.zeros((n_output_samples, 1, 1), dtype=precision)
    n_summary_samples = int(ceil(n_output_samples / saves_per_summary))
    state_summaries_output = np.zeros((n_summary_samples, n_runs, n_states),
                                      dtype=precision)
    observable_summaries_output = np.zeros((n_summary_samples, 1, 1),
                                           dtype=precision)
    iteration_counters_output = np.zeros((n_runs, n_output_samples, n_counters),
                                         dtype=np.int32)
    status_codes_output = np.zeros(n_runs, dtype=np.int32)

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
        inits, params, d_coefficients, state_output, observables_output,
        state_summaries_output, observable_summaries_output,
        iteration_counters_output, status_codes_output,
        duration, warmup, precision(0.0), n_runs)

    cuda.synchronize()

    print("\n" + "=" * 70)
    print("Integration Complete")
    print("=" * 70)

    success_count = np.sum(status_codes_output == 0)
    print(f"\nSuccessful runs: {success_count:,} / {n_runs:,}")
    print(f"Final state sample (run 0): {state_output[-1, 0, :n_states]}")
    print(f"Final state sample (run -1): {state_output[-1, -1, :n_states]}")

    return state_output, status_codes_output


if __name__ == "__main__":
    run_debug_integration()
