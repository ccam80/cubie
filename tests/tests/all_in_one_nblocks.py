"""All-in-one debug file for Numba lineinfo debugging.

All CUDA device functions are consolidated in this single file to enable
proper line-level debugging with Numba's lineinfo feature.
"""
# ruff: noqa: E402
from math import ceil, floor
from typing import Optional
import os

import numpy as np
from numba import cuda, int16, int32, int64, float32, float64
from numba import from_dtype as numba_from_dtype
from cubie.cuda_simsafe import activemask, all_sync, selp, compile_kwargs
from cubie.cuda_simsafe import from_dtype as simsafe_dtype
from cubie.integrators.algorithms.generic_dirk_tableaus import (
    DIRK_TABLEAU_REGISTRY,
)
from cubie.integrators.algorithms.generic_erk_tableaus import (
    ERK_TABLEAU_REGISTRY,
)


# Simulation-safe syncwarp wrapper
CUDA_SIMULATION = os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"

if CUDA_SIMULATION:
    def syncwarp():
        """No-op syncwarp for CUDA simulator."""
        pass
else:
    @cuda.jit(device=True, inline=True, **compile_kwargs)
    def syncwarp():
        """Warp synchronization on real GPU."""
        cuda.syncwarp()


# =========================================================================
# STRIDE ORDERING UTILITIES
# =========================================================================

# Default stride order matches memory manager: (time, run, variable)
# Represented as indices: 0=time, 1=run, 2=variable
DEFAULT_STRIDE_ORDER = (0, 2, 1)


def get_strides(
    shape: tuple,
    dtype,
    array_native_order: tuple = (0, 1, 2),
    desired_order: tuple = DEFAULT_STRIDE_ORDER,
) -> Optional[tuple]:
    """Calculate memory strides for a given access pattern (stride order).

    This function replicates the striding logic from MemoryManager.get_strides()
    to allow all_in_one.py to allocate arrays with the same stride patterns
    as the production batch solver.

    Parameters
    ----------
    shape
        Tuple describing the array shape.
    dtype
        NumPy dtype for the array elements.
    array_native_order
        Tuple of indices describing the logical dimension ordering for
        the array's shape. For 3D arrays, indices represent:
        0=time, 1=run, 2=variable. Defaults to (0, 1, 2).
    desired_order
        Tuple of indices describing the desired memory stride ordering.
        The last index changes fastest (contiguous). Defaults to (0, 1, 2)
        which matches cubie's default: time, run, variable.

    Returns
    -------
    tuple or None
        Stride tuple for the array, or None if no custom strides needed.

    Notes
    -----
    Only 3D arrays get custom stride optimization. Arrays with fewer
    dimensions use default strides.
    """
    if len(shape) != 3:
        return None

    if array_native_order == desired_order:
        return None

    itemsize = np.dtype(dtype).itemsize

    # Map index to size
    dims = {int(idx): int(size) for idx, size in zip(array_native_order,
                                                     shape)}

    strides = {}
    current_stride = int(itemsize)

    # Iterate over the desired order reversed; the last dimension
    # in the order changes fastest so it gets the smallest stride.
    for idx in reversed(desired_order):
        strides[idx] = current_stride
        current_stride *= int(dims[idx])

    return tuple(strides[dim] for dim in array_native_order)

# =========================================================================
# CONFIGURATION
# =========================================================================

# Algorithm configuration: 'erk' or 'dirk'
# Use ERK_TABLEAU_REGISTRY or DIRK_TABLEAU_REGISTRY keys
algorithm_type = 'erk'  # 'erk' or 'dirk'
algorithm_tableau_name = 'tsit5'  # Registry key for the tableau

# Controller configuration: 'fixed' or 'pid'
controller_type = 'fixed'  # 'fixed' or 'pid'

# Look up tableau from registry based on algorithm type
if algorithm_type == 'erk':
    if algorithm_tableau_name not in ERK_TABLEAU_REGISTRY:
        raise ValueError(
            f"Unknown ERK tableau: '{algorithm_tableau_name}'. "
            f"Available: {list(ERK_TABLEAU_REGISTRY.keys())}"
        )
    tableau = ERK_TABLEAU_REGISTRY[algorithm_tableau_name]
elif algorithm_type == 'dirk':
    if algorithm_tableau_name not in DIRK_TABLEAU_REGISTRY:
        raise ValueError(
            f"Unknown DIRK tableau: '{algorithm_tableau_name}'. "
            f"Available: {list(DIRK_TABLEAU_REGISTRY.keys())}"
        )
    tableau = DIRK_TABLEAU_REGISTRY[algorithm_tableau_name]
else:
    raise ValueError(f"Unknown algorithm type: '{algorithm_type}'. "
                     "Use 'erk' or 'dirk'.")

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
duration = precision(2e-5)
warmup = precision(0.0)
dt = precision(1e-3)
dt_save = precision(1e-5)
dt_max = precision(1e6)
dt_min = precision(1e-12) #TODO: when 1e-15, infinite loop


# Tableau stage count (for buffer sizing)
stage_count = tableau.stage_count

# Solver helper parameters (beta, gamma, mass matrix scaling)
beta_solver = precision(1.0)
gamma_solver = precision(1.0)
preconditioner_order = 2

# Linear solver (Krylov) parameters
krylov_tolerance = precision(1e-6)
max_linear_iters = 200
linear_correction_type = "minimal_residual"
# Newton-Krylov nonlinear solver parameters
newton_tolerance = precision(1e-6)
max_newton_iters = 100
newton_damping = precision(0.85)
max_backtracks = 15

# PID controller parameters
algorithm_order = 2
kp = precision(0.7)
ki = precision(0.0)
kd = precision(-0.4)
min_gain = precision(0.2)
max_gain = precision(5.0)
dt_min_ctrl = dt_min
dt_max_ctrl = dt_max
deadband_min = precision(1.0)
deadband_max = precision(1.0)
safety = precision(0.9)
atol = np.full(n_states, precision(1e-8), dtype=precision)
rtol = np.full(n_states, precision(1e-8), dtype=precision)

# Output dimensions
n_output_samples = int(floor(float(duration) / float(dt_save))) + 1


# Compile-time flags for loop behavior
save_obs_bool = False
save_state_bool = True
summarise_obs_bool = False
summarise_state_bool = False
summarise = summarise_obs_bool or summarise_state_bool
save_counters_bool = False
fixed_mode = (controller_type == 'fixed')
save_last = False
dt0 = precision(0.001) if fixed_mode else np.sqrt(dt_min*dt_max)

# =========================================================================
# AUTO-GENERATED DEVICE FUNCTION FACTORIES
# =========================================================================

def dxdt_factory(constants, prec):
    """Auto-generated dxdt factory."""
    sigma = prec(constants['sigma'])
    beta = prec(constants['beta'])
    numba_prec = numba_from_dtype(prec)

    @cuda.jit(
            # (numba_prec[::1], numba_prec[::1], numba_prec[::1],
            #    numba_prec[::1], numba_prec[::1], numba_prec),
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

    @cuda.jit(
            # (numba_prec[::1], numba_prec[::1], numba_prec[::1],
            #    numba_prec[::1], numba_prec),
              device=True, inline=True, **compile_kwargs)
    def get_observables(state, parameters, drivers, observables, t):
        pass
    return get_observables


def neumann_preconditioner_factory(constants, prec, beta, gamma, order):
    """Auto-generated Neumann preconditioner."""
    n = int32(3)
    order = int32(order)
    beta_inv = prec(1.0 / beta)
    gamma = prec(gamma)
    h_eff_factor = prec(gamma * beta_inv)
    sigma = prec(constants['sigma'])
    beta_const = prec(constants['beta'])
    numba_prec = numba_from_dtype(prec)

    @cuda.jit(
            # (numba_prec[::1], numba_prec[::1], numba_prec[::1],
            #    numba_prec[::1], numba_prec, numba_prec, numba_prec,
            #    numba_prec[::1], numba_prec[::1], numba_prec[::1]),
              device=True, inline=True, **compile_kwargs)
    def preconditioner(state, parameters, drivers, base_state, t, h, a_ij,
                       v, out, jvp):
        ty = cuda.threadIdx.y
        mask = activemask()
        out[ty] = v[ty]
        h_eff = h * numba_prec(h_eff_factor) * a_ij
        for _ in range(order):
            if ty == 0:
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
            syncwarp()
            out[ty] = v[ty] + h_eff * jvp[ty]
        out[ty] = numba_prec(beta_inv) * out[ty]
    return preconditioner


def stage_residual_factory(constants, prec, beta, gamma, order):
    """Auto-generated nonlinear residual for implicit updates."""
    sigma = prec(constants['sigma'])
    beta_const = prec(constants['beta'])
    numba_prec = numba_from_dtype(prec)
    beta_val = numba_prec(1.0) * numba_prec(beta)

    @cuda.jit(
            # (numba_prec[::1], numba_prec[::1], numba_prec[::1],
            #    numba_prec, numba_prec, numba_prec, numba_prec[::1],
            #    numba_prec[::1]),
              device=True, inline=True, **compile_kwargs)
    def residual(u, parameters, drivers, t, h, a_ij, base_state, out):
        ty = cuda.threadIdx.y
        if ty == 0:
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
    gamma = prec(gamma)
    beta = prec(beta)
    numba_prec = numba_from_dtype(prec)

    @cuda.jit(
        # (numba_prec[::1], numba_prec[::1], numba_prec[::1],
        #    numba_prec[::1], numba_prec, numba_prec, numba_prec,
        #    numba_prec[::1], numba_prec[::1]),
        device=True,
        inline=True,
        **compile_kwargs,
    )
    def operator_apply(
        state, parameters, drivers, base_state, t, h, a_ij, v, out
    ):
        ty = cuda.threadIdx.y
        m = cuda.local.array(n_states, numba_prec)
        m[ty] = numba_prec(1.0)
        j = cuda.local.array((n_states, n_states), numba_prec)
        # Initialize row entries for current thread
        j[ty, 0] = numba_prec(0.0)
        j[ty, 1] = numba_prec(0.0)
        j[ty, 2] = numba_prec(0.0)
        if ty == 0:
            j[0, 0] = -sigma
            j[0,1] = sigma
            j[1, 0] = -a_ij * state[2] + parameters[0] - base_state[2]
            j[1, 1] = int32(-1)
            j[1, 2] = -a_ij * state[0] - base_state[0]
            j[2, 0] = a_ij * state[1] + base_state[1]
            j[2, 1] = a_ij * state[0] + base_state[0]
            j[2, 2] = -beta_const
        # gamma_val = numba_prec(gamma) #TODO: Get rid of these from codegen
        # beta_val = numba_prec(beta)
        syncwarp()
        out[ty] = (-a_ij * gamma * h * (j[ty, 0] * v[0] + j[ty,1] * v[1] +
                                        j[ty,2] * v[2]) + beta * m[ty] * v[ty])
    return operator_apply


# =========================================================================
# INLINE LINEAR SOLVER FACTORY
# =========================================================================

def linear_solver_inline_factory(
        operator_apply, n, preconditioner, tolerance, max_iters, prec, correction_type
):
    """Create inline linear solver device function.

    Parameters
    ----------
    correction_type
    """
    numba_prec = numba_from_dtype(prec)
    tol_squared = precision(tolerance * tolerance)
    n = int32(n)
    max_iters = int32(max_iters)
    sd_flag = 1 if correction_type == "steepest_descent" else 0
    mr_flag = 1 if correction_type == "minimal_residual" else 0

    @cuda.jit(
        # [(numba_prec[::1],
        #   numba_prec[::1],
        #   numba_prec[::1],
        #   numba_prec[::1],
        #   numba_prec,
        #   numba_prec,
        #   numba_prec,
        #   numba_prec[::1],
        #   numba_prec[::1])],
        device=True,
        inline=True,
        **compile_kwargs,
    )
    def linear_solver(state, parameters, drivers, base_state, t, h, a_ij,
                      rhs, x):
        ty = cuda.threadIdx.y
        preconditioned_vec = cuda.local.array(n, numba_prec)
        temp = cuda.local.array(n, numba_prec)

        operator_apply(state, parameters, drivers, base_state, t, h, a_ij,
                       x, temp)
        acc = typed_zero
        # Replace loop over n with per-thread operation
        residual_value = rhs[ty] - temp[ty]
        rhs[ty] = residual_value
        acc += residual_value * residual_value
        mask = activemask()
        converged = acc <= tol_squared

        iter_count = int32(0)
        for _ in range(max_iters):
            iter_count += int32(1)
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
                # Per-thread contribution for steepest descent
                zi = preconditioned_vec[ty]
                numerator += rhs[ty] * zi
                denominator += temp[ty] * zi
            elif mr_flag:
                # Per-thread contribution for minimal residual
                ti = temp[ty]
                numerator += ti * rhs[ty]
                denominator += ti * ti

            alpha = selp(denominator != typed_zero,
                         numerator / denominator, typed_zero)
            alpha_effective = selp(converged, numba_prec(0.0), alpha)

            acc = typed_zero
            # Per-thread update and accumulation
            x[ty] += alpha_effective * preconditioned_vec[ty]
            rhs[ty] -= alpha_effective * temp[ty]
            residual_value = rhs[ty]
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
    n = int32(n)
    max_iters = int32(max_iters)
    max_backtracks = int32(max_backtracks)
    tol_squared = numba_prec(tolerance * tolerance)
    typed_zero = numba_prec(0.0)
    typed_one = numba_prec(1.0)
    typed_damping = numba_prec(damping)
    status_active = int32(-1)

    @cuda.jit(
            # [(numba_prec[::1],
            #     numba_prec[::1],
            #     numba_prec[::1],
            #     numba_prec,
            #     numba_prec,
            #     numba_prec,
            #     numba_prec[::1],
            #     numba_prec[::1],
            #     int32[::1])],
              device=True,
              inline=True,
              **compile_kwargs
    )
    def newton_krylov_solver(stage_increment, parameters, drivers, t, h,
                             a_ij, base_state, shared_scratch, counters):
        ty = cuda.threadIdx.y
        delta = shared_scratch[:n]
        residual = shared_scratch[n:2 * n]

        residual_fn(stage_increment, parameters, drivers, t, h, a_ij,
                    base_state, residual)
        norm2_prev = typed_zero
        # Initialize per-thread entries instead of looping
        residual_value = residual[ty]
        residual[ty] = -residual_value
        delta[ty] = typed_zero
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
                    # Per-thread update of stage_increment
                    stage_increment[ty] += delta_scale * delta[ty]
                    scale_applied = scale

                    residual_fn(stage_increment, parameters, drivers, t, h,
                                a_ij, base_state, residual)

                    norm2_new = typed_zero
                    residual_value = residual[ty]
                    norm2_new += residual_value * residual_value

                    if norm2_new <= tol_squared:
                        status = int32(0)

                    accept = (status < 0) and (norm2_new < norm2_prev)
                    found_step = found_step or accept

                    residual[ty] = selp(accept, -residual[ty], residual[ty])
                    norm2_prev = selp(accept, norm2_new, norm2_prev)

                if all_sync(mask, found_step or status >= 0):
                    break
                scale *= typed_damping

            if (status < 0) and (not found_step):
                # Per-thread rollback of stage_increment
                stage_increment[ty] -= scale_applied * delta[ty]
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
    n = int32(n)
    stage_count = int32(tableau.stage_count)

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
    if b_row is not None:
        b_row = int32(b_row)
    if b_hat_row is not None:
        b_hat_row = int32(b_hat_row)

    stage_implicit = tuple(coeff != numba_precision(0.0)
                           for coeff in diagonal_coeffs)
    accumulator_length = int32(max(stage_count - 1, 0) * n)
    solver_shared_elements = 2 * n  # delta + residual for Newton solver

    # Shared memory indices
    acc_start = 0
    acc_end = accumulator_length
    solver_start = acc_end
    solver_end = acc_end + solver_shared_elements

    @cuda.jit(
        # (
        #     numba_precision[::1],
        #     numba_precision[::1],
        #     numba_precision[::1],
        #     numba_precision[:, :, ::1],
        #     numba_precision[::1],
        #     numba_precision[::1],
        #     numba_precision[::1],
        #     numba_precision[::1],
        #     numba_precision[::1],
        #     numba_precision,
        #     numba_precision,
        #     int16,
        #     int16,
        #     numba_precision[::1],
        #     numba_precision[::1],
        #     int32[::1],
        # ),
        device=True,
        inline=True,
        **compile_kwargs,
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
        ty = cuda.threadIdx.y
        # Reintroduced core per-step variables
        current_time = time_scalar
        end_time = current_time + dt_scalar
        stage_accumulator = shared[acc_start:acc_end]
        solver_scratch = shared[solver_start:solver_end]
        stage_rhs = solver_scratch[:n]
        increment_cache = solver_scratch[n:2*n]
        status_code = int32(0)

        stage_increment = cuda.local.array(n, numba_precision)

        if multistage:
            stage_base = stage_accumulator[:n]
        else:
            stage_base = cuda.local.array(n, numba_precision)

        # Per-thread initialisation replacing loop
        if has_error and accumulates_error:
            error[ty] = typed_zero
        # FSAL increment cache reused if tableau supports it; otherwise stale zeros
        stage_increment[ty] = increment_cache[ty]
        syncwarp()

        first_step = first_step_flag != int16(0)
        prev_state_accepted = accepted_flag != int16(0)

        # Cache reuse (unchanged logic; only evaluation flag computed by thread 0)
        use_cached_rhs = False
        if first_same_as_last and multistage:
            if not first_step_flag:
                mask = activemask()
                all_threads_accepted = all_sync(mask, accepted_flag != int16(0))
                use_cached_rhs = all_threads_accepted

        # --------- Stage 0 ---------
        stage_time = current_time + dt_scalar * stage_time_fractions[0]
        diagonal_coeff = diagonal_coeffs[0]
        # Base state per-thread
        stage_base[ty] = state[ty]
        if accumulates_output:
            proposed_state[ty] = typed_zero
        syncwarp()

        if not use_cached_rhs:
            if can_reuse_accepted_start:
                # Drivers (n_drivers may be 0); broadcast by thread 0 only
                if ty == 0:
                    for d in range(drivers_buffer.shape[0]):
                        proposed_drivers[d] = drivers_buffer[d]
            else:
                if has_driver_function and ty == 0:
                    pass  # driver function call placeholder

            if stage_implicit[0]:
                status_code |= nonlinear_solver(
                    stage_increment,
                    parameters,
                    proposed_drivers,
                    stage_time,
                    dt_scalar,
                    diagonal_coeffs[0],
                    stage_base,
                    solver_scratch,
                    counters,
                )
                # Implicit increment application per-thread
                stage_base[ty] += diagonal_coeff * stage_increment[ty]
            syncwarp()

            # Observables and dxdt only need one thread to write full vector
            if ty == 0:
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
            syncwarp()
        # else: cached rhs already in stage_rhs slice

        solution_weight = solution_weights[0]
        error_weight = error_weights[0]
        # Per-thread accumulation
        rhs_value = stage_rhs[ty]
        if accumulates_output:
            proposed_state[ty] += solution_weight * rhs_value
        elif b_row == int32(0):
            proposed_state[ty] = stage_base[ty]
        if has_error:
            if accumulates_error:
                error[ty] += error_weight * rhs_value
            elif b_hat_row == int32(0):
                error[ty] = stage_base[ty]
        syncwarp()
        # Zero accumulator slices strided by variable index
        for acc_offset in range(0, accumulator_length, n):
            stage_accumulator[acc_offset + ty] = typed_zero
        syncwarp()

        # --------- Successor Stages (1..s) ---------
        for stage_idx in range(int32(1), stage_count):
            prev_idx = stage_idx - int32(1)
            successor_range = stage_count - stage_idx
            stage_time = current_time + dt_scalar * stage_time_fractions[stage_idx]

            # Accumulate explicit contributions into successor accumulators
            # Each thread contributes its own variable component
            for successor_offset in range(successor_range):
                successor_idx = stage_idx + successor_offset
                base = (successor_idx - int32(1)) * n
                state_coeff = stage_rhs_coeffs[successor_idx][prev_idx]
                contribution = state_coeff * stage_rhs[ty] * dt_scalar
                stage_accumulator[base + ty] += contribution
            syncwarp()

            # Stage base view and add original state
            stage_base = stage_accumulator[(stage_idx-int32(1))*n:stage_idx*n]
            stage_base[ty] += state[ty]
            syncwarp()

            diagonal_coeff = diagonal_coeffs[stage_idx]
            if stage_implicit[stage_idx]:
                status_code |= nonlinear_solver(
                    stage_increment,
                    parameters,
                    proposed_drivers,
                    stage_time,
                    dt_scalar,
                    diagonal_coeffs[stage_idx],
                    stage_base,
                    solver_scratch,
                    counters,
                )
                stage_base[ty] += diagonal_coeff * stage_increment[ty]
            syncwarp()

            if ty == 0:
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
            syncwarp()

            solution_weight = solution_weights[stage_idx]
            error_weight = error_weights[stage_idx]
            increment = stage_rhs[ty]
            if accumulates_output:
                proposed_state[ty] += solution_weight * increment
            elif b_row == stage_idx:
                proposed_state[ty] = stage_base[ty]
            if has_error:
                if accumulates_error:
                    error[ty] += error_weight * increment
                elif b_hat_row == stage_idx:
                    error[ty] = stage_base[ty]
            syncwarp()

        # --------- Final combination ---------
        if accumulates_output:
            proposed_state[ty] = proposed_state[ty] * dt_scalar + state[ty]
        if has_error:
            if accumulates_error:
                error[ty] *= dt_scalar
            else:
                error[ty] = proposed_state[ty] - error[ty]
        increment_cache[ty] = stage_increment[ty]
        syncwarp()
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

    n_arraysize = n
    n = int32(n)
    stage_count = int32(tableau.stage_count)
    stages_except_first = stage_count - int32(1)
    accumulator_length = (tableau.stage_count - 1) * n_arraysize

    # Compile-time toggles
    has_driver_function = False  # No driver function in this test
    first_same_as_last = tableau.first_same_as_last
    multistage = stage_count > int32(1)
    has_error = tableau.has_error_estimate

    stage_rhs_coeffs = tableau.typed_columns(tableau.a, numba_precision)
    solution_weights = tableau.typed_vector(tableau.b, numba_precision)
    stage_nodes = tableau.typed_vector(tableau.c, numba_precision)

    if has_error:
        error_weights = tableau.error_weights(numba_precision)
    else:
        error_weights = tuple(typed_zero for _ in range(stage_count))

    # Last-step caching optimization
    accumulates_output = tableau.accumulates_output
    accumulates_error = tableau.accumulates_error
    b_row = tableau.b_matches_a_row
    b_hat_row = tableau.b_hat_matches_a_row
    if b_row is not None:
        b_row = int32(b_row)
    if b_hat_row is not None:
        b_hat_row = int32(b_hat_row)

    @cuda.jit(
        # (
        #     numba_precision[::1],
        #     numba_precision[::1],
        #     numba_precision[::1],
        #     numba_precision[:, :, ::1],
        #     numba_precision[::1],
        #     numba_precision[::1],
        #     numba_precision[::1],
        #     numba_precision[::1],
        #     numba_precision[::1],
        #     numba_precision,
        #     numba_precision,
        #     int16,
        #     int16,
        #     numba_precision[::1],
        #     numba_precision[::1],
        #     int32[::1],
        # ),
        device=True,
        inline=True,
        **compile_kwargs
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
        ty = cuda.threadIdx.y
        current_time = time_scalar
        end_time = current_time + dt_scalar
        stage_accumulator = shared[:accumulator_length]
        stage_rhs = shared[accumulator_length: accumulator_length + n]
        if multistage:
            stage_cache = stage_rhs  # FSAL cache alias
        # Per-thread init
        if accumulates_output:
            proposed_state[ty] = typed_zero
        if has_error and accumulates_error:
            error[ty] = typed_zero
        syncwarp()

        # Stage 0 cache reuse check
        use_cached_rhs = False
        if first_same_as_last and multistage:
            if not first_step_flag:
                mask = activemask()
                all_threads_accepted = all_sync(mask, accepted_flag != int16(0))
                use_cached_rhs = all_threads_accepted
        # Evaluate rhs (only one thread writes full vector) unless cached
        if multistage and use_cached_rhs:
            pass
        elif ty==0:

            dxdt_fn(
                state,
                parameters,
                drivers_buffer,
                observables,
                stage_rhs,
                current_time,
            )
        syncwarp()

        # Accumulate stage 0
        increment0 = stage_rhs[ty]
        if accumulates_output:
            proposed_state[ty] += solution_weights[0] * increment0
        if has_error and accumulates_error:
            error[ty] += error_weights[0] * increment0
        syncwarp()

        # Zero accumulator (strided)
        for offset in range(ty, accumulator_length, n):
            stage_accumulator[offset] = typed_zero
        syncwarp()

        # Successor stages
        for prev_idx in range(stages_except_first):
            stage_offset = prev_idx * n
            stage_idx = prev_idx + int32(1)
            matrix_col = stage_rhs_coeffs[prev_idx]
            # Stream contributions
            for successor_idx in range(stages_except_first):
                coeff = matrix_col[successor_idx+int32(1)]
                row_offset = successor_idx * n
                stage_accumulator[row_offset + ty] += coeff * stage_rhs[ty]
            syncwarp()
            # Form stage state per-thread
            base = stage_offset + ty
            stage_accumulator[base] = stage_accumulator[base] * dt_scalar + state[ty]
            syncwarp()
            # Observables and rhs for next stage
            stage_state = stage_accumulator[stage_offset:stage_offset + n]
            stage_time = current_time + dt_scalar * stage_nodes[stage_idx]
            if ty == 0:
                observables_function(
                    stage_state,
                    parameters,
                    proposed_drivers,
                    proposed_observables,
                    stage_time,
                )
                dxdt_fn(
                    stage_state,
                    parameters,
                    proposed_drivers,
                    proposed_observables,
                    stage_rhs,
                    stage_time,
                )
            syncwarp()
            increment = stage_rhs[ty]
            if accumulates_output:
                proposed_state[ty] += solution_weights[stage_idx] * increment
            if has_error and accumulates_error:
                error[ty] += error_weights[stage_idx] * increment
            syncwarp()
        # Direct state/error capture for b_row / b_hat_row
        if b_row is not None:
            proposed_state[ty] = stage_accumulator[(b_row-1)*n + ty]
        if b_hat_row is not None:
            error[ty] = stage_accumulator[(b_hat_row-1)*n + ty]
        syncwarp()
        # Final scaling per-thread
        if accumulates_output:
            proposed_state[ty] = proposed_state[ty] * dt_scalar + state[ty]
        if has_error:
            if accumulates_error:
                error[ty] *= dt_scalar
            else:
                error[ty] = proposed_state[ty] - error[ty]
        if first_same_as_last and multistage:
            stage_cache[ty] = stage_rhs[ty]
        syncwarp()
        return int32(0)
    return step

# =========================================================================
# OUTPUT FUNCTIONS
# =========================================================================
n_states32 = int32(n_states)
n_counters32 = int32(n_counters)
@cuda.jit(device=True, inline=True, **compile_kwargs)
def save_state_inline(current_state, current_observables, current_counters,
                      current_step, output_states_slice, output_obs_slice,
                      output_counters_slice):
    ty = cuda.threadIdx.y
    output_states_slice[ty] = current_state[ty]
    if ty == 0:
        output_states_slice[n_states32] = current_step
    if save_counters_bool and ty < n_counters32:
        output_counters_slice[ty] = current_counters[ty]
    syncwarp()


# =========================================================================
# SUMMARY METRIC FUNCTIONS (Mean metric with chained pattern)
# =========================================================================

@cuda.jit(
    # ["float32, float32[::1], int32, int32",
    #  "float64, float64[::1], int32, int32"],
    device=True,
    inline=True,
    **compile_kwargs
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
    # ["float32[::1], float32[::1], int32, int32",
    #  "float64[::1], float64[::1], int32, int32"],
    device=True,
    inline=True,
    **compile_kwargs
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
    ty = cuda.threadIdx.y
    total_buffer_size = int32(1)
    start = ty * total_buffer_size
    end = start + total_buffer_size
    chain_update_metrics(
        current_state[ty],
        state_summary_buffer[start:end],
        current_step,
    )
    syncwarp()


@cuda.jit(device=True, inline=True, **compile_kwargs)
def save_summaries_inline(
    buffer_state,
    buffer_obs,
    output_state,
    output_obs,
    summarise_every,
):
    ty = cuda.threadIdx.y
    total_buffer_size = int32(1)
    total_output_size = int32(1)
    buffer_start = ty * total_buffer_size
    out_start = ty * total_output_size
    chain_save_metrics(
        buffer_state[buffer_start:buffer_start + total_buffer_size],
        output_state[out_start:out_start + total_output_size],
        summarise_every,
    )
    syncwarp()


# =========================================================================
# STEP CONTROLLER (Fixed and adaptive options)
# =========================================================================

@cuda.jit(device=True, inline=True, **compile_kwargs)
def clamp(value, min_val, max_val):
    """Clamp a value between min and max."""
    return max(min_val, min(value, max_val))


@cuda.jit(
    # [(
    #     numba_precision[::1],
    #     numba_precision[::1],
    #     numba_precision[::1],
    #     numba_precision[::1],
    #     int32,
    #     int32[::1],
    #     numba_precision[::1],
    # )],
    device=True,
    inline=True,
    **compile_kwargs,
)
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
    typed_one = precision(1.0)
    typed_zero = precision(0.0)
    deadband_min = precision(deadband_min)
    deadband_max = precision(deadband_max)
    deadband_disabled = (deadband_min == typed_one) and (
        deadband_max == typed_one
    )
    dt_min = precision(dt_min_ctrl)
    dt_max = precision(dt_max_ctrl)
    min_gain = precision(min_gain)
    max_gain = precision(max_gain)
    safety = precision(safety)
    inv_safety = precision(1.0) / safety
    n = int32(n)
    inv_n = precision(1.0 / n)
    @cuda.jit(
        # [
        #     (
        #         numba_precision[::1],
        #         numba_precision[::1],
        #         numba_precision[::1],
        #         numba_precision[::1],
        #         int32,
        #         int32[::1],
        #         numba_precision[::1],
        #     )
        # ],
        device=True,
        inline=True,
        # fastmath=True,
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
        ty = cuda.threadIdx.y
        err_prev = local_temp[0]
        err_prev_prev = local_temp[1]
        nrm2 = typed_zero

        for i in range(n):
            error_i = max(abs(error[i]), precision(1e-16))
            tol = atol[i] + rtol[i] * max(
                abs(state[i]), abs(state_prev[i])
            )
            ratio = error_i / tol
            nrm2 += ratio * ratio

        nrm2 = nrm2 * inv_n
        accept = nrm2 <= typed_one
        accept_out[0] = int32(1) if accept else int32(0)
        err_prev_safe = err_prev if err_prev > typed_zero else nrm2
        err_prev_prev_safe = (
            err_prev_prev if err_prev_prev > typed_zero else err_prev_safe
        )

        gain_new = precision(
            safety
            * (nrm2 ** (-expo1))
            * (err_prev_safe ** (-expo2))
            * (err_prev_prev_safe ** (-expo3))
        )
        gain = clamp(gain_new, min_gain, max_gain)
        if not deadband_disabled:
            within_deadband = (
                (gain >= deadband_min)
                and (gain <= deadband_max)
            )
            gain = selp(within_deadband, typed_one, gain)

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

# Build step function based on algorithm type
if algorithm_type == 'erk':
    # ERK step for explicit integration
    step_fn = erk_step_inline_factory(
        dxdt_fn,
        observables_function,
        n_states,
        precision,
        tableau,
    )
elif algorithm_type == 'dirk':
    # Build implicit solver components for DIRK
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

    linear_solver_fn = linear_solver_inline_factory(operator_fn, n_states,
                                                    preconditioner_fn,
                                                    krylov_tolerance,
                                                    max_linear_iters,
                                                    precision,
                                                    linear_correction_type)

    newton_solver_fn = newton_krylov_inline_factory(
        residual_fn,
        linear_solver_fn,
        n_states,
        newton_tolerance,
        max_newton_iters,
        newton_damping,
        max_backtracks,
        precision,
    )

    step_fn = dirk_step_inline_factory(
        newton_solver_fn,
        dxdt_fn,
        observables_function,
        n_states,
        precision,
        tableau,
    )
else:
    raise ValueError(f"Unknown algorithm type: '{algorithm_type}'. "
                     "Use 'erk' or 'dirk'.")

# Build controller function based on controller type
if controller_type == 'fixed':
    step_controller_fn = step_controller_fixed
elif controller_type == 'pid':
    step_controller_fn = controller_PID_factory(
        precision,
        n_states,
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
    )
else:
    raise ValueError(f"Unknown controller type: '{controller_type}'. "
                     "Use 'fixed' or 'pid'.")


# =========================================================================
# INTEGRATION LOOP
# =========================================================================

# Buffer layout
state_shared_start = int32(0)
state_shared_end = int32(n_states)
proposed_state_start = int32(state_shared_end)
proposed_state_end = proposed_state_start + int32(n_states)
params_start = proposed_state_end
params_end = params_start + int32(n_parameters)
drivers_start = params_end
drivers_end = drivers_start + int32(n_drivers)
proposed_drivers_start = drivers_end
proposed_drivers_end = proposed_drivers_start + int32(n_drivers)
obs_start = proposed_drivers_end
obs_end = obs_start + int32(n_observables)
proposed_obs_start = obs_end
proposed_obs_end = proposed_obs_start + int32(n_observables)
error_start = proposed_obs_end
error_end = error_start + int32(n_states)
counters_start = error_end
counters_end = counters_start + (int32(n_counters) if save_counters_bool
                                 else int32(0))
proposed_counters_start = counters_end
proposed_counters_end = proposed_counters_start + (
    int32(2) if save_counters_bool else int32(0)
)
scratch_start = proposed_counters_end

accumulator_size = int32((stage_count - 1) * n_states)
if algorithm_type == 'dirk':
    solver_scratch_size = int32(2 * n_states)
    scratch_end = scratch_start + accumulator_size + solver_scratch_size
else:
    scratch_end = scratch_start + accumulator_size
    scratch_end = scratch_end + int32(n_states) + 1 #This isn't good,
    # but a 1 is needed to aboid writing off the end of it. Let's say...
    # alignment.
state_summ_start = scratch_end
state_summ_end = state_summ_start + (
    int32(n_states) if summarise_state_bool else int32(0)
)
obs_summ_start = state_summ_end
obs_summ_end = obs_summ_start + (
    int32(n_observables if summarise_obs_bool else int32(0))
)
shared_elements = obs_summ_end


local_dt_slice = slice(0, 1)
local_accept_slice = slice(1, 2)
local_controller_slice = slice(2, 4)
local_algo_slice = slice(4, 8)
local_elements = 8

saves_per_summary = int32(2)
status_mask = int32(0xFFFF)



@cuda.jit(
    # [
    #     (
    #         numba_precision[:, ::1],
    #         numba_precision[:, ::1],
    #         numba_precision[:, :, ::1],
    #         numba_precision[:, :, :],
    #         numba_precision[:, :, :],
    #         numba_precision[:, :, :],
    #         numba_precision[:, :, :],
    #         int32[:, :, :],
    #         int32[::1],
    #         float64,
    #         float64,
    #         float64,
    #         int32,
    #     )],
    device=True,
    inline=True,
    **compile_kwargs,
)
def loop_fn(initial_states, parameters, driver_coefficients, shared_scratch,
            persistent_local, state_output, observables_output,
            state_summaries_output, observable_summaries_output,
            iteration_counters_output, duration, settling_time, t0):
    """Advance an integration using a compiled CUDA device loop.

    The loop terminates when the time of the next saved sample
    exceeds the end time (t0 + settling_time + duration), or when
    the maximum number of iterations is reached.
    """
    ty = cuda.threadIdx.y
    t = float64(t0)
    t_prec = numba_precision(t)
    t_end = float64(settling_time + t0 + duration)

    # Cap max iterations - all internal steps at dt_min, plus a bonus
    # end/start, plus one failure per successful step.
    # 64-bits required to get any reasonable duration with small step
    total_duration = duration + settling_time
    max_steps = min(
            int64(2**62), (int64(ceil(total_duration/dt_min)) + 2)
    )
    max_steps = max_steps << 1

    # Initialize shared memory - all threads initialize strided elements
    n_shared = len(shared_scratch)
    for i in range(ty, n_shared, n_states):
        shared_scratch[i] = numba_precision(0.0)
    syncwarp()

    state_buffer = shared_scratch[state_shared_start:state_shared_end]
    state_proposal_buffer = shared_scratch[proposed_state_start:
                                           proposed_state_end]
    observables_buffer = shared_scratch[obs_start:obs_end]
    observables_proposal_buffer = shared_scratch[proposed_obs_start:
                                                 proposed_obs_end]
    parameters_buffer = shared_scratch[params_start:params_end]
    drivers_buffer = shared_scratch[drivers_start:drivers_end]
    drivers_proposal_buffer = shared_scratch[proposed_drivers_start:
                                             proposed_drivers_end]
    state_summary_buffer = shared_scratch[state_summ_start:state_summ_end]
    observable_summary_buffer = shared_scratch[obs_summ_start:obs_summ_end]
    remaining_shared_scratch = shared_scratch[scratch_start:scratch_end]
    counters_since_save = shared_scratch[counters_start:counters_end]

    if save_counters_bool:
        # When enabled, use shared memory buffers
        proposed_counters = shared_scratch[proposed_counters_start:
                                           proposed_counters_end]
    else:
        # When disabled, use a dummy local "proposed_counters" buffer
        proposed_counters = cuda.local.array(2, dtype=simsafe_int32)

    dt = persistent_local[local_dt_slice]
    accept_step = persistent_local[local_accept_slice].view(simsafe_int32)
    error = shared_scratch[error_start:error_end]
    controller_temp = persistent_local[local_controller_slice]
    algo_local = persistent_local[local_algo_slice]

    first_step_flag = int16(1)
    prev_step_accepted_flag = int16(1)

    # ----------------------------------------------------------------------- #
    #                       Seed t=0 values                                   #
    # ----------------------------------------------------------------------- #
    state_buffer[ty] = initial_states[ty]
    if ty==0:
        for k in range(n_parameters):
            parameters_buffer[k] = parameters[k]
    syncwarp()
    # Seed initial observables from initial state.
    # driver_function not used in this test (n_drivers = 0)
    if n_observables > 0 and ty == 0:
        observables_function(
            state_buffer,
            parameters_buffer,
            drivers_buffer,
            observables_buffer,
            t_prec,
        )

    save_idx = int32(0)
    summary_idx = int32(0)

    # Set next save for settling time, or save first value if
    # starting at t0
    next_save = settling_time + t0
    syncwarp()
    if settling_time == 0.0:
        # Save initial state at t0, then advance to first interval save
        next_save += float64(dt_save)

        save_state_inline(
            state_buffer,
            observables_buffer,
            counters_since_save,
            t_prec,
            state_output[save_idx * save_state_bool, :],
            observables_output[save_idx * save_obs_bool, :],
            iteration_counters_output[save_idx * save_counters_bool, :],
        )
        if summarise:
            # Reset temp buffers to starting state - will be overwritten
            save_summaries_inline(state_summary_buffer,
                                  observable_summary_buffer,
                                  state_summaries_output[
                                      summary_idx * summarise_state_bool, :
                                  ],
                                  observable_summaries_output[
                                      summary_idx * summarise_obs_bool, :
                                  ],
                                  saves_per_summary)

            # Log first summary update
            update_summaries_inline(
                state_buffer,
                observables_buffer,
                state_summary_buffer,
                observable_summary_buffer,
                save_idx,
            )
        save_idx += int32(1)

    status = int32(0)
    # Initialize shared scalars and counters from thread 0 only
    if ty == 0:
        dt[0] = dt0
        accept_step[0] = int32(0)

        # Initialize iteration counters (only if saving counters)
        if save_counters_bool:
            for i in range(n_counters):
                counters_since_save[i] = int32(0)
                if i < 2:
                    proposed_counters[i] = int32(0)
    syncwarp()

    mask = activemask()

    # ----------------------------------------------------------------------- #
    #                        Main Loop                                        #
    # ----------------------------------------------------------------------- #
    for _ in range(max_steps):
        # Exit as soon as we've saved the final step
        finished = next_save > t_end
        if save_last:
            # If last save requested, predicated commit dt, finished,
            # do_save
            at_last_save = finished and t < t_end
            finished = selp(at_last_save, False, True)
            dt[0] = selp(at_last_save, numba_precision(t_end - t),
                         dt[0])

        # also exit loop if min step size limit hit - things are bad
        finished |= (status & 0x8)

        if all_sync(mask, finished):
            return status

        if not finished:
            do_save = (t + dt[0]) >= next_save
            dt_eff = selp(do_save, numba_precision(next_save - t), dt[0])

            # Fixed mode auto-accepts all steps; adaptive uses controller

            step_status = step_fn(
                state_buffer,
                state_proposal_buffer,
                parameters_buffer,
                driver_coefficients,
                drivers_buffer,
                drivers_proposal_buffer,
                observables_buffer,
                observables_proposal_buffer,
                error,
                dt_eff,
                t_prec,
                first_step_flag,
                prev_step_accepted_flag,
                remaining_shared_scratch,
                algo_local,
                proposed_counters,
            )

            first_step_flag = int16(0)

            niters = (step_status >> 16) & status_mask
            status |= step_status & status_mask

            # Adjust dt if step rejected - auto-accepts if fixed-step
            if not fixed_mode and ty==1:
                status |= step_controller_fn(
                    dt,
                    state_proposal_buffer,
                    state_buffer,
                    error,
                    niters,
                    accept_step,
                    controller_temp,
                )

                accept = accept_step[0] != int32(0)

            else:
                accept = True
            syncwarp()
            # Accumulate iteration counters if active
            if save_counters_bool and ty==0:
                for i in range(n_counters):
                    if i < 2:
                        # Write newton, krylov iterations from buffer
                        counters_since_save[i] += proposed_counters[i]
                    elif i == 2:
                        # Increment total steps counter
                        counters_since_save[i] += int32(1)
                    elif not accept:
                        # Increment rejected steps counter
                        counters_since_save[i] += int32(1)
            syncwarp()
            t_proposal = t + dt_eff
            t = selp(accept, t_proposal, t)
            t_prec = numba_precision(t)
            syncwarp()

            newv = state_proposal_buffer[ty]
            oldv = state_buffer[ty]
            state_buffer[ty] = selp(accept, newv, oldv)
            if ty==0:
                for i in range(n_drivers):
                    new_drv = drivers_proposal_buffer[i]
                    old_drv = drivers_buffer[i]
                    drivers_buffer[i] = selp(accept, new_drv, old_drv)

                for i in range(n_observables):
                    new_obs = observables_proposal_buffer[i]
                    old_obs = observables_buffer[i]
                    observables_buffer[i] = selp(accept, new_obs, old_obs)

            prev_step_accepted_flag = selp(
                accept,
                int16(1),
                int16(0),
            )

            # Predicated update of next_save; update if save is accepted.
            do_save = accept and do_save
            if do_save:
                next_save = selp(do_save, next_save + dt_save, next_save)

                save_state_inline(
                    state_buffer,
                    observables_buffer,
                    counters_since_save,
                    t_prec,
                    state_output[save_idx * save_state_bool, :],
                    observables_output[save_idx * save_obs_bool],
                    iteration_counters_output[save_idx * save_counters_bool,
                                              :],
                )
                if summarise:
                    update_summaries_inline(
                        state_buffer,
                        observables_buffer,
                        state_summary_buffer,
                        observable_summary_buffer,
                        save_idx,
                    )

                    if (save_idx + int32(1)) % saves_per_summary == int32(0):
                        save_summaries_inline(
                            state_summary_buffer,
                            observable_summary_buffer,
                            state_summaries_output[
                                summary_idx * summarise_state_bool, :
                            ],
                            observable_summaries_output[
                                summary_idx * summarise_obs_bool, :
                            ],
                            saves_per_summary,
                        )
                        summary_idx += int32(1)
                save_idx += int32(1)

                # Reset iteration counters after save
                if save_counters_bool and ty==0:
                    for i in range(n_counters):
                        counters_since_save[i] = int32(0)
                syncwarp()

    if status == int32(0):
        # Max iterations exhausted without other error
        status = int32(32)
    return status


# =========================================================================
# KERNEL
# =========================================================================

local_elements_per_run = local_elements
shared_elems_per_run = shared_elements
f32_per_element = 2 if (precision == np.float64) else 1
f32_pad_perrun = (
    1 if (shared_elems_per_run % 2 == 0 and f32_per_element == 1) else 0
)
run_stride_f32 = int32(
            (f32_per_element * shared_elems_per_run + f32_pad_perrun)
        )
numba_prec = numba_from_dtype(precision)

@cuda.jit(
        # [(
        #         numba_prec[:, ::1],
        #         numba_prec[:, ::1],
        #         numba_prec[:, :, ::1],
        #         numba_prec[:, :, :],
        #         numba_prec[:, :, :],
        #         numba_prec[:, :, :],
        #         numba_prec[:, :, :],
        #         int32[:, :, :],
        #         int32[::1],
        #         float64,
        #         float64,
        #         float64,
        #         int32,
        #     )],
        **compile_kwargs)
def integration_kernel(inits, params, d_coefficients, state_output,
                       observables_output, state_summaries_output,
                       observables_summaries_output, iteration_counters_output,
                       status_codes_output, duration_k, warmup_k, t0_k,
                       n_runs_k):

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    block_index = int32(cuda.blockIdx.x)
    runs_per_block = cuda.blockDim.x
    run_index = int32(runs_per_block * block_index + tx)
    if run_index >= n_runs_k or ty >= n_states:
        return None

    shared_memory = cuda.shared.array(0, dtype=float32)
    local_scratch = cuda.local.array(local_elements_per_run, dtype=float32)
    c_coefficients = cuda.const.array_like(d_coefficients)

    run_idx_low = tx * run_stride_f32
    run_idx_high = (run_idx_low + f32_per_element * shared_elems_per_run
    )
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

    if ty == 0:
        status_codes_output[run_index] = status


# =========================================================================
# MAIN EXECUTION
# =========================================================================

def run_debug_integration(n_runs=2**23, rho_min=0.0, rho_max=21.0):
    """Execute debug integration with 2^23 runs."""
    print("=" * 70)
    algo_name = algorithm_type.upper()
    ctrl_name = controller_type.upper()
    print(f"Debug Integration - {algo_name} ({algorithm_tableau_name}) "
          f"with {ctrl_name} controller")
    print("=" * 70)
    print(f"\nRunning {n_runs:,} integrations with rho in [{rho_min}, {rho_max}]")

    # Generate rho values
    rho_values = np.linspace(rho_min, rho_max, n_runs, dtype=precision)

    # Input arrays (NumPy)
    inits = np.zeros((n_runs, n_states), order='F', dtype=precision)
    inits[:, 0] = precision(1.0)
    inits[:, 1] = precision(0.0)
    inits[:, 2] = precision(0.0)

    params = np.zeros((n_runs, n_parameters), dtype=precision)
    params[:, 0] = rho_values

    d_coefficients = np.zeros((1, max(n_drivers, 1), 6), dtype=precision)

    # Create device arrays for inputs (BatchInputArrays pattern)
    itemsize = np.dtype(precision).itemsize

    d_inits = cuda.device_array((n_runs, n_states), dtype=precision,
                             order='F')
    d_inits.copy_to_device(inits)
    d_params = cuda.to_device(params)
    d_driver_coefficients = cuda.to_device(d_coefficients)

    # Create mapped arrays for outputs (BatchOutputArrays pattern)
    # Using stride ordering to match production batch solver memory layout.
    # Dimension indices: 0=time, 1=run, 2=variable
    n_summary_samples = int(ceil(n_output_samples / saves_per_summary))

    # state_output: shape=(samples, runs, states+1), native order=(time, run, var)
    state_shape = (n_output_samples, n_runs, n_states + 1)
    state_strides = get_strides(state_shape, precision, (0, 1, 2))
    state_output = cuda.mapped_array(
        state_shape, dtype=precision, strides=state_strides
    )

    # observables_output: shape=(samples, 1, 1), native order=(time, run, var)
    obs_shape = (n_output_samples, 1, 1)
    obs_strides = get_strides(obs_shape, precision, (0, 1, 2))
    observables_output = cuda.mapped_array(
        obs_shape, dtype=precision, strides=obs_strides
    )

    # state_summaries_output: shape=(summaries, runs, states)
    # native order=(time, run, var)
    state_summ_shape = (n_summary_samples, n_runs, n_states)
    state_summ_strides = get_strides(state_summ_shape, precision, (0, 1, 2))
    state_summaries_output = cuda.mapped_array(state_summ_shape, dtype=precision,
                                               strides=state_summ_strides)

    # observable_summaries_output: shape=(summaries, 1, 1)
    # native order=(time, run, var)
    obs_summ_shape = (n_summary_samples, 1, 1)
    obs_summ_strides = get_strides(obs_summ_shape, precision, (0, 1, 2))
    observable_summaries_output = cuda.mapped_array(obs_summ_shape,
                                                    dtype=precision,
                                                    strides=obs_summ_strides)

    # iteration_counters_output: shape=(runs, samples, counters)
    # native order=(run, time, var) - different from default!
    iter_shape = (n_runs, n_output_samples, n_counters)
    iter_strides = get_strides(iter_shape, np.int32, (1, 0, 2))
    iteration_counters_output = cuda.mapped_array(iter_shape, dtype=np.int32,
                                                  strides=iter_strides)

    # status_codes_output: 1D array, no custom strides needed
    status_codes_output = cuda.mapped_array((n_runs,), dtype=np.int32)

    print(f"State output shape: {state_output.shape}")

    # Kernel configuration
    MAX_SHARED_MEMORY_PER_BLOCK = 32768
    # Compute ysize as the next power-of-two >= n_states that also divides
    # a warp-aligned block size. Start at 32 and double until it is >= n_states.
    block_size = 128
    while int(n_states) > block_size:
        block_size <<= 1
    ysize_val = 1
    while ysize_val < int(n_states):
        ysize_val <<= 1
    # Cap ysize to the block size determined above
    if ysize_val > block_size:
        ysize_val = block_size
    ysize = int32(ysize_val)

    runs_per_block = block_size // ysize
    blocksize = (runs_per_block, ysize)
    dynamic_sharedmem = int32(
        (f32_per_element * run_stride_f32) * 4 * runs_per_block)


    # while dynamic_sharedmem > MAX_SHARED_MEMORY_PER_BLOCK:
    #     blocksize = blocksize // 2
    #     runs_per_block = blocksize
    #     dynamic_sharedmem = int(
    #         (f32_per_element * run_stride_f32) * 4 * runs_per_block)

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
    print(f"dt_min exceeded runs: {dt_min_exceeded_runs:,}")
    print(f"max steps exceeded runs: {max_steps_exceeded_runs:,}")
    print(f"max newton backtracks runs: {max_newton_backtracks_runs:,}")
    print(f"max newton iters exceeded runs: {max_newton_iters_runs:,}")
    print(f"linear solver max iters exceeded runs: {linear_max_iters_runs:,}")

    print(f"\nSuccessful runs: {success_count:,} / {n_runs:,}")


    print(f"Final state sample (run 0): {state_output[-1, 0, :n_states]}")
    print(f"Final state sample (run -1): {state_output[-1, -1, :n_states]}")

    return state_output, status_codes_output


if __name__ == "__main__":
    run_debug_integration(n_runs=int32(2**23))
