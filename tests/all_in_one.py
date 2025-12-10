"""All-in-one debug file for Numba lineinfo debugging.

All CUDA device functions are consolidated in this single file to enable
proper line-level debugging with Numba's lineinfo.
"""
# ruff: noqa: E402
from math import ceil, floor
from time import perf_counter
from typing import Optional

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
from cubie.integrators.algorithms.generic_firk_tableaus import (
    FIRK_TABLEAU_REGISTRY,
)
from cubie.integrators.algorithms.generic_rosenbrockw_tableaus import (
    ROSENBROCK_TABLEAUS,
)
from cubie.integrators.array_interpolator import ArrayInterpolator

# =========================================================================
# CONFIGURATION - ALL CONFIGURABLE PARAMETERS AT TOP OF FILE
# =========================================================================

# -------------------------------------------------------------------------
# Algorithm Configuration
# -------------------------------------------------------------------------
# Algorithm type options:
#   'erk'         - Explicit Runge-Kutta (ERK_TABLEAU_REGISTRY)
#   'dirk'        - Diagonally Implicit Runge-Kutta (DIRK_TABLEAU_REGISTRY)
#   'firk'        - Fully Implicit Runge-Kutta (FIRK_TABLEAU_REGISTRY)
#   'rosenbrock'  - Rosenbrock-W methods (ROSENBROCK_TABLEAUS)
algorithm_type = 'dirk'  # 'erk', 'dirk', 'firk', or 'rosenbrock'
algorithm_tableau_name = 'l_stable_sdirk_4'  # Registry key for the tableau

# Controller type: 'fixed' (fixed step) or 'pid' (adaptive PID)
controller_type = 'pid'  # 'fixed' or 'pid'

# -------------------------------------------------------------------------
# Precision Configuration
# -------------------------------------------------------------------------
precision = np.float32

# -------------------------------------------------------------------------
# System Definition (Lorenz system)
# -------------------------------------------------------------------------
# Lorenz system constants
constants = {'sigma': 10.0, 'beta': 8.0 / 3.0}

# System dimensions
n_states = 3
n_parameters = 1
n_observables = 0
n_drivers = 0
n_counters = 4

# -------------------------------------------------------------------------
# Driver Interpolation Configuration (when n_drivers > 0)
# -------------------------------------------------------------------------
# Driver input dictionary for ArrayInterpolator
# When n_drivers > 0, provide a dictionary with:
#   - Driver signal names as keys with 1D numpy arrays as values
#   - "dt" or "time": time information for samples
#   - Optional: "order", "wrap", "boundary_condition"
# 
# Example usage with drivers:
#   driver_input_dict = {
#       "driver_0": np.sin(np.linspace(0, 2*np.pi, 10)),
#       "driver_1": np.cos(np.linspace(0, 2*np.pi, 10)),
#       "dt": 0.1,
#       "t0": 0.0,
#       "order": 3,
#       "wrap": True,
#       "boundary_condition": "periodic"
#   }
#
driver_input_dict = None

# -------------------------------------------------------------------------
# Time Parameters
# -------------------------------------------------------------------------
duration = precision(0.1)
warmup = precision(0.0)
dt = precision(1e-3) # TODO: should be able to set starting dt for adaptive
# runs
dt_save = precision(0.1)
dt_max = precision(1e3)
dt_min = precision(1e-12)  # TODO: when 1e-15, infinite loop

# -------------------------------------------------------------------------
# Implicit Solver Parameters (DIRK only)
# -------------------------------------------------------------------------
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

# -------------------------------------------------------------------------
# PID Controller Parameters (adaptive mode only)
# -------------------------------------------------------------------------
algorithm_order = 2
kp = precision(0.7)
ki = precision(-0.4)
kd = precision(0.0)
min_gain = precision(0.2)
max_gain = precision(5.0)
deadband_min = precision(1.0)
deadband_max = precision(1.0)
safety = precision(0.9)

# Tolerances for adaptive step control
atol_value = precision(1e-8)
rtol_value = precision(1e-8)

# -------------------------------------------------------------------------
# Output Configuration
# -------------------------------------------------------------------------
# Compile-time flags for loop behavior
save_obs_bool = False
save_state_bool = True
summarise_obs_bool = False
summarise_state_bool = False
save_counters_bool = False
save_last = False

# Saves per summary (for summary metric aggregation)
saves_per_summary = int32(2)

# -------------------------------------------------------------------------
# Memory Location Configuration (for optimization experiments)
# -------------------------------------------------------------------------
# Toggle whether each internal array uses local or shared memory.
# 'local' = cuda.local.array (per-thread registers/local memory)
# 'shared' = cuda.shared.array (block-level shared memory)
#
# Note: These affect compile-time code generation. Changing these requires
# recompilation of the affected device functions.

# Loop buffers (main integration loop)
loop_state_buffer_memory = 'local'  # 'local' or 'shared'
loop_state_proposal_buffer_memory = 'local'  # 'local' or 'shared'
loop_parameters_buffer_memory = 'local'  # 'local' or 'shared'
loop_drivers_buffer_memory = 'local'  # 'local' or 'shared'
loop_drivers_proposal_buffer_memory = 'local'  # 'local' or 'shared'
loop_observables_buffer_memory = 'local'  # 'local' or 'shared'
loop_observables_proposal_buffer_memory = 'local'  # 'local' or 'shared'
loop_error_buffer_memory = 'local'  # 'local' or 'shared'
loop_counters_buffer_memory = 'local'  # 'local' or 'shared'
loop_state_summary_buffer_memory = 'local'  # 'local' or 'shared'
loop_observable_summary_buffer_memory = 'shared'  # 'local' or 'shared'

# This one doesn't really make sense - it'lls be shared(0) if algo doesn't
# request shared
loop_scratch_buffer_memory = 'local'  # 'local' or 'shared'

# Linear solver arrays (used in Krylov iteration)
linear_solver_preconditioned_vec_memory = 'local'  # 'local' or 'shared'
linear_solver_temp_memory = 'local'  # 'local' or 'shared'

# DIRK step arrays
dirk_stage_increment_memory = 'local'  # 'local' or 'shared'
dirk_stage_base_memory = 'local'  # 'local' or 'shared' (shared aliases
#                                    accumulator when multistage)
dirk_accumulator_memory = 'local'  # 'local' or 'shared'
dirk_solver_scratch_memory = 'local'  # 'local' or 'shared'

# ERK step arrays
erk_stage_rhs_memory = 'local'  # 'local' or 'shared'
erk_stage_accumulator_memory = 'local'  # 'local' or 'shared'
# Note: stage_cache aliases onto stage_rhs if shared, else onto accumulator
# if shared, else goes into persistent_local

# FIRK step arrays (Fully Implicit Runge-Kutta)
# FIRK solves all stages simultaneously as a coupled system
firk_solver_scratch_memory = 'local'  # 'local' or 'shared'
firk_stage_increment_memory = 'local'  # 'local' or 'shared'
firk_stage_driver_stack_memory = 'local'  # 'local' or 'shared'
firk_stage_state_memory = 'local'  # 'local' or 'shared'

# Rosenbrock step arrays (Rosenbrock-W methods)
# Rosenbrock methods use linearized implicit approach with cached Jacobian
rosenbrock_stage_rhs_memory = 'local'  # 'local' or 'shared'
rosenbrock_stage_store_memory = 'local'  # 'local' or 'shared'
rosenbrock_cached_auxiliaries_memory = 'local'  # 'local' or 'shared'

# -------------------------------------------------------------------------
# Kernel Launch Configuration
# -------------------------------------------------------------------------
# Maximum shared memory per block (hardware limit)
MAX_SHARED_MEMORY_PER_BLOCK = 32768

# Block size for kernel launch
blocksize = 64

# =========================================================================
# DERIVED CONFIGURATION (computed from above settings)
# =========================================================================

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
elif algorithm_type == 'firk':
    if algorithm_tableau_name not in FIRK_TABLEAU_REGISTRY:
        raise ValueError(
            f"Unknown FIRK tableau: '{algorithm_tableau_name}'. "
            f"Available: {list(FIRK_TABLEAU_REGISTRY.keys())}"
        )
    tableau = FIRK_TABLEAU_REGISTRY[algorithm_tableau_name]
elif algorithm_type == 'rosenbrock':
    if algorithm_tableau_name not in ROSENBROCK_TABLEAUS:
        raise ValueError(
            f"Unknown Rosenbrock tableau: '{algorithm_tableau_name}'. "
            f"Available: {list(ROSENBROCK_TABLEAUS.keys())}"
        )
    tableau = ROSENBROCK_TABLEAUS[algorithm_tableau_name]
else:
    raise ValueError(f"Unknown algorithm type: '{algorithm_type}'. "
                     "Use 'erk', 'dirk', 'firk', or 'rosenbrock'.")

# Numba/simsafe type conversions
numba_precision = numba_from_dtype(precision)
simsafe_precision = simsafe_dtype(precision)
simsafe_int32 = simsafe_dtype(np.int32)

# Typed constants for device code
typed_zero = numba_precision(0.0)

# Tableau stage count (for buffer sizing)
stage_count = tableau.stage_count

# Controller-dependent derived values
dt_min_ctrl = dt_min
dt_max_ctrl = dt_max
atol = np.full(n_states, atol_value, dtype=precision)
rtol = np.full(n_states, rtol_value, dtype=precision)

# Output dimensions
n_output_samples = int(floor(float(duration) / float(dt_save))) + 1

# Compile-time derived flags
summarise = summarise_obs_bool or summarise_state_bool
fixed_mode = (controller_type == 'fixed')
dt0 = dt if fixed_mode else np.sqrt(dt_min * dt_max)

# Memory location flags as booleans for compile-time branching
# Loop buffer flags
use_shared_loop_state = loop_state_buffer_memory == 'shared'
use_shared_loop_state_proposal = loop_state_proposal_buffer_memory == 'shared'
use_shared_loop_parameters = loop_parameters_buffer_memory == 'shared'
use_shared_loop_drivers = loop_drivers_buffer_memory == 'shared'
use_shared_loop_drivers_proposal = loop_drivers_proposal_buffer_memory == 'shared'
use_shared_loop_observables = loop_observables_buffer_memory == 'shared'
use_shared_loop_observables_proposal = (
    loop_observables_proposal_buffer_memory == 'shared'
)
use_shared_loop_error = loop_error_buffer_memory == 'shared'
use_shared_loop_counters = loop_counters_buffer_memory == 'shared'
use_shared_loop_state_summary = loop_state_summary_buffer_memory == 'shared'
use_shared_loop_observable_summary = (
    loop_observable_summary_buffer_memory == 'shared'
)
use_shared_loop_scratch = loop_scratch_buffer_memory == 'shared'

# Linear solver flags
use_shared_linear_preconditioned_vec = (
    linear_solver_preconditioned_vec_memory == 'shared'
)
use_shared_linear_temp = linear_solver_temp_memory == 'shared'

# DIRK step flags
use_shared_dirk_stage_increment = dirk_stage_increment_memory == 'shared'
use_shared_dirk_stage_base = dirk_stage_base_memory == 'shared'
use_shared_dirk_accumulator = dirk_accumulator_memory == 'shared'
use_shared_dirk_solver_scratch = dirk_solver_scratch_memory == 'shared'

# ERK step flags
use_shared_erk_stage_rhs = erk_stage_rhs_memory == 'shared'
use_shared_erk_stage_accumulator = erk_stage_accumulator_memory == 'shared'
# ERK stage_cache: aliases stage_rhs if shared, else accumulator if shared,
# else requires persistent_local storage
use_shared_erk_stage_cache = (use_shared_erk_stage_rhs or
                              use_shared_erk_stage_accumulator)

# FIRK step flags
use_shared_firk_solver_scratch = firk_solver_scratch_memory == 'shared'
use_shared_firk_stage_increment = firk_stage_increment_memory == 'shared'
use_shared_firk_stage_driver_stack = firk_stage_driver_stack_memory == 'shared'
use_shared_firk_stage_state = firk_stage_state_memory == 'shared'

# Rosenbrock step flags
use_shared_rosenbrock_stage_rhs = rosenbrock_stage_rhs_memory == 'shared'
use_shared_rosenbrock_stage_store = rosenbrock_stage_store_memory == 'shared'
use_shared_rosenbrock_cached_auxiliaries = (
    rosenbrock_cached_auxiliaries_memory == 'shared'
)


# =========================================================================
# STRIDE ORDERING UTILITIES
# =========================================================================

# Default stride order: (time, variable, run)
# Represented as indices: 0=time, 1=variable, 2=run
DEFAULT_STRIDE_ORDER = (0, 1, 2)


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
        0=time, 1=variable, 2=run. Defaults to (0, 1, 2).
    desired_order
        Tuple of indices describing the desired memory stride ordering.
        The last index changes fastest (contiguous). Defaults to (0, 1, 2)
        which matches cubie's current default: time, variable, run.

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
# AUTO-GENERATED DEVICE FUNCTION FACTORIES
# =========================================================================

def dxdt_factory(constants, prec):
    """Auto-generated dxdt factory."""
    sigma = prec(constants['sigma'])
    beta = prec(constants['beta'])
    numba_prec = numba_from_dtype(prec)

    @cuda.jit(
            (numba_prec[::1], numba_prec[::1], numba_prec[::1],
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

    @cuda.jit(
            (numba_prec[::1], numba_prec[::1], numba_prec[::1],
               numba_prec[::1], numba_prec),
              device=True, inline=True, **compile_kwargs)
    def get_observables(state, parameters, drivers, observables, t):
        pass
    return get_observables


# =========================================================================
# DRIVER INTERPOLATION INLINE DEVICE FUNCTIONS
# =========================================================================

def driver_function_inline_factory(interpolator):
    """Create inline evaluation function from ArrayInterpolator.
    
    Takes an ArrayInterpolator instance and creates an inline CUDA device
    function that evaluates driver polynomials at a given time. This is the
    critical device code for CUDA profiling tool alignment.
    
    Parameters
    ----------
    interpolator : ArrayInterpolator
        Configured ArrayInterpolator instance with computed coefficients.
        
    Returns
    -------
    callable
        Inline CUDA device function with signature:
        driver_function(time, coefficients, out)
    """
    prec = interpolator.precision
    numba_prec = numba_from_dtype(prec)
    order = int32(interpolator.order)
    num_drivers = int32(interpolator.num_inputs)
    num_segments = int32(interpolator.num_segments)
    inv_resolution = prec(prec(1.0) / interpolator.dt)
    start_time_val = prec(interpolator.t0)
    zero_value = prec(0.0)
    wrap = interpolator.wrap
    pad_clamped = (not wrap) and (interpolator.boundary_condition == 'clamped')
    evaluation_start = prec(start_time_val - (
        interpolator.dt if pad_clamped else prec(0.0)))
    
    @cuda.jit(
        (numba_prec, numba_prec[:, :, ::1], numba_prec[::1]),
        device=True,
        inline=True,
        **compile_kwargs
    )
    def driver_function(time, coefficients, out):
        """Evaluate all driver polynomials at time on the device.
        
        Parameters
        ----------
        time : float
            Query time for evaluation.
        coefficients : device array
            Segment-major coefficients with shape (segments, drivers, order+1).
        out : device array
            Output array to populate with evaluated driver values.
        """
        scaled = (time - evaluation_start) * inv_resolution
        scaled_floor = floor(scaled)
        idx = int32(scaled_floor)
        
        if wrap:
            seg = int32(idx % num_segments)
            tau = prec(scaled - scaled_floor)
            in_range = True
        else:
            in_range = (scaled >= 0.0) and (scaled <= num_segments)
            seg = selp(idx < 0, int32(0), idx)
            seg = selp(seg >= num_segments,
                      int32(num_segments - 1), seg)
            tau = scaled - float(seg)
        
        # Evaluate polynomials using Horner's rule
        for driver_idx in range(num_drivers):
            acc = zero_value
            for k in range(order, -1, -1):
                acc = acc * tau + coefficients[seg, driver_idx, k]
            out[driver_idx] = acc if in_range else zero_value
    
    return driver_function


def driver_derivative_inline_factory(interpolator):
    """Create inline derivative function from ArrayInterpolator.
    
    Takes an ArrayInterpolator instance and creates an inline CUDA device
    function that evaluates driver time derivatives at a given time. This is
    critical device code for CUDA profiling tool alignment.
    
    Parameters
    ----------
    interpolator : ArrayInterpolator
        Configured ArrayInterpolator instance with computed coefficients.
        
    Returns
    -------
    callable
        Inline CUDA device function with signature:
        driver_derivative(time, coefficients, out)
    """
    prec = interpolator.precision
    numba_prec = numba_from_dtype(prec)
    order = int32(interpolator.order)
    num_drivers = int32(interpolator.num_inputs)
    num_segments = int32(interpolator.num_segments)
    inv_resolution = prec(prec(1.0) / interpolator.dt)
    start_time_val = prec(interpolator.t0)
    zero_value = prec(0.0)
    wrap = interpolator.wrap
    pad_clamped = (not wrap) and (interpolator.boundary_condition == 'clamped')
    evaluation_start = prec(start_time_val - (
        interpolator.dt if pad_clamped else prec(0.0)))
    
    @cuda.jit(
        (numba_prec, numba_prec[:, :, ::1], numba_prec[::1]),
        device=True,
        inline=True,
        **compile_kwargs
    )
    def driver_derivative(time, coefficients, out):
        """Evaluate time derivative of each driver polynomial.
        
        Parameters
        ----------
        time : float
            Query time for evaluation.
        coefficients : device array
            Segment-major coefficients with shape (segments, drivers, order+1).
        out : device array
            Output array to populate with evaluated driver derivatives.
        """
        scaled = (time - evaluation_start) * inv_resolution
        scaled_floor = floor(scaled)
        idx = int32(scaled_floor)
        
        if wrap:
            seg = int32(idx % num_segments)
            tau = prec(scaled - scaled_floor)
            in_range = True
        else:
            in_range = (scaled >= 0.0) and (scaled <= num_segments)
            seg = selp(idx < 0, int32(0), idx)
            seg = selp(seg >= num_segments,
                      int32(num_segments - 1), seg)
            tau = scaled - float(seg)
        
        # Evaluate derivative using Horner's rule on derivative polynomial
        for driver_idx in range(num_drivers):
            acc = zero_value
            for k in range(order, 0, -1):
                acc = acc * tau + prec(k) * (
                    coefficients[seg, driver_idx, k]
                )
            out[driver_idx] = (
                acc * inv_resolution if in_range else zero_value
            )
    
    return driver_derivative


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
            (numba_prec[::1], numba_prec[::1], numba_prec[::1],
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
            j_11 = numba_prec(-1)
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

    @cuda.jit(
            (numba_prec[::1], numba_prec[::1], numba_prec[::1],
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
    gamma = prec(gamma)
    numba_prec = numba_from_dtype(prec)

    @cuda.jit(
            (numba_prec[::1], numba_prec[::1], numba_prec[::1],
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
        j_11 = numba_prec(-1)
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

def linear_solver_inline_factory(
        operator_apply, n, preconditioner, tolerance, max_iters, prec,
        correction_type
):
    """Create inline linear solver device function.

    Parameters
    ----------
    operator_apply
        The linear operator function.
    n
        Number of state variables.
    preconditioner
        The preconditioner function.
    tolerance
        Convergence tolerance.
    max_iters
        Maximum iterations.
    prec
        Precision dtype.
    correction_type
        Type of correction: 'steepest_descent' or 'minimal_residual'.

    Notes
    -----
    Memory location flags are read from global scope at compile time.
    The generated device function will use the selected memory type for
    each array. Currently only local memory is implemented; shared memory
    variants would require changes to the function signature to receive
    shared memory buffers.
    """
    numba_prec = numba_from_dtype(prec)
    typed_tol = numba_prec(tolerance)
    tol_squared = typed_tol * typed_tol
    n_arraysize = n
    n = int32(n)
    max_iters = int32(max_iters)
    sd_flag = int32(1) if correction_type == "steepest_descent" else int32(0)
    mr_flag = int32(1) if correction_type == "minimal_residual" else int32(0)

    @cuda.jit(
        (numba_prec[::1], numba_prec[::1], numba_prec[::1],
         numba_prec[::1], numba_prec, numba_prec, numba_prec,
         numba_prec[::1], numba_prec[::1]),
        device=True, inline=True, **compile_kwargs)
    def linear_solver(state, parameters, drivers, base_state, t, h, a_ij,
                      rhs, x):
        preconditioned_vec = cuda.local.array(n_arraysize, numba_prec)
        temp = cuda.local.array(n_arraysize, numba_prec)

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

            for i in range(n):
                rhs_i = rhs[i]
                temp_i = temp[i]
                if sd_flag:
                    zi = preconditioned_vec[i]
                    numerator += rhs_i * zi
                    denominator += temp_i * zi
                elif mr_flag:
                    numerator += temp_i * rhs_i
                    denominator += temp_i * temp_i

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
    n = int32(n)
    max_iters = int32(max_iters)
    max_backtracks = int32(max_backtracks)
    tol_squared = numba_prec(tolerance * tolerance)
    typed_zero = numba_prec(0.0)
    typed_one = numba_prec(1.0)
    typed_damping = numba_prec(damping)
    status_active = int32(-1)

    @cuda.jit(
            [(numba_prec[::1],
                numba_prec[::1],
                numba_prec[::1],
                numba_prec,
                numba_prec,
                numba_prec,
                numba_prec[::1],
                numba_prec[::1],
                int32[::1])],
              device=True,
              inline=True,
              **compile_kwargs
    )
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
        delta = shared_scratch[:n]
        residual = shared_scratch[n:int32(2 * n)]

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
            if all_sync(mask, status >= int32(0)):
                break
            if status < int32(0):
                iters_count += int32(1)
                lin_return = linear_solver(stage_increment, parameters,
                                           drivers, base_state, t, h, a_ij,
                                           residual, delta)
                krylov_iters = (lin_return >> int32(16)) & int32(0xFFFF)
                total_krylov_iters += krylov_iters
                lin_status = lin_return & int32(0xFFFF)
                if lin_status != int32(0):
                    status = int32(lin_status)

            scale = typed_one
            scale_applied = typed_zero
            found_step = False

            for _ in range(max_backtracks + int32(1)):
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

                    accept = (status < int32(0)) and (norm2_new < norm2_prev)
                    found_step = found_step or accept

                    for i in range(n):
                        residual[i] = selp(accept, -residual[i], residual[i])
                    norm2_prev = selp(accept, norm2_new, norm2_prev)

                if all_sync(mask, found_step or status >= int32(0)):
                    break
                scale *= typed_damping

            if (status < int32(0)) and (not found_step):
                for i in range(n):
                    stage_increment[i] -= scale_applied * delta[i]
                status = int32(1)

        if status < int32(0):
            status = int32(2)

        counters[0] = iters_count
        counters[1] = total_krylov_iters

        status |= iters_count << int16(16)
        return status
    return newton_krylov_solver


# =========================================================================
# INLINE DIRK STEP FACTORY (Generic DIRK with tableau)
# =========================================================================

def dirk_step_inline_factory(
    nonlinear_solver,
    dxdt_fn,
    observables_function,
    driver_function,
    n,
    prec,
    tableau,
):
    """Create inline DIRK step device function matching generic_dirk.py.

    Parameters
    ----------
    nonlinear_solver
        The Newton-Krylov solver function.
    dxdt_fn
        The derivative function.
    observables_function
        The observables function.
    driver_function
        The driver evaluation function (or None if no drivers).
    n
        Number of state variables.
    prec
        Precision dtype.
    tableau
        The DIRK tableau.

    Notes
    -----
    Memory location flags are read from global scope:
    - use_shared_dirk_stage_increment
    - use_shared_dirk_stage_base
    """
    numba_precision = numba_from_dtype(prec)
    typed_zero = numba_precision(0.0)

    # Extract tableau properties
    n_arraysize = n
    accumulator_length_arraysize = int(max(tableau.stage_count-1, 1) * n)
    double_n = 2 * n
    n = int32(n)
    stage_count = int32(tableau.stage_count)

    stages_except_first = stage_count - int32(1)

    # Compile-time toggles
    has_driver_function = driver_function is not None
    has_error = tableau.has_error_estimate
    multistage = stage_count > 1
    first_same_as_last = False  # SDIRK does not share first/last stage
    can_reuse_accepted_start = False

    stage_rhs_coeffs = tableau.typed_columns(tableau.a, numba_precision)
    explicit_a_coeffs = tableau.explicit_terms(numba_precision)
    solution_weights = tableau.typed_vector(tableau.b, numba_precision)
    typed_zero = numba_precision(0.0)
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

    # Memory location flags captured at compile time from global scope
    stage_increment_in_shared = use_shared_dirk_stage_increment
    stage_base_in_shared = use_shared_dirk_stage_base
    accumulator_in_shared = use_shared_dirk_accumulator
    solver_scratch_in_shared = use_shared_dirk_solver_scratch

    # Shared memory indices (only used when corresponding flag is True)
    # Accumulator comes first if shared
    acc_start = 0
    acc_end = accumulator_length if accumulator_in_shared else 0
    # Solver scratch follows accumulator if shared
    solver_start = acc_end
    solver_end = (acc_end + solver_shared_elements
                  if solver_scratch_in_shared else acc_end)

    @cuda.jit(
        [(
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
        )],
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
        # stage_increment: size n, shared or local memory.
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

        # ----------------------------------------------------------- #
        # Selective allocation from local or shared memory
        # ----------------------------------------------------------- #
        if stage_increment_in_shared:
            stage_increment = shared[solver_end:solver_end + n]
        else:
            stage_increment = cuda.local.array(n_arraysize, numba_precision)
            for _i in range(n_arraysize):
                stage_increment[_i] = numba_precision(0.0)

        if accumulator_in_shared:
            stage_accumulator = shared[acc_start:acc_end]
        else:
            stage_accumulator = cuda.local.array(accumulator_length_arraysize,
                                                 numba_precision)
            for _i in range(accumulator_length):
                stage_accumulator[_i] = numba_precision(0.0)

        if solver_scratch_in_shared:
            solver_scratch = shared[solver_start:solver_end]
        else:
            solver_scratch = cuda.local.array(double_n,numba_precision)
            for _i in range(solver_shared_elements):
                solver_scratch[_i] = numba_precision(0.0)

        # Alias stage base onto first stage accumulator or allocate locally
        if multistage:
            stage_base = stage_accumulator[:n]
        else:
            if stage_base_in_shared:
                stage_base = shared[:n]
            else:
                stage_base = cuda.local.array(n_arraysize, numba_precision)
                for _i in range(n_arraysize):
                    stage_base[_i] = numba_precision(0.0)

        # --------------------------------------------------------------- #

        current_time = time_scalar
        end_time = current_time + dt_scalar
        stage_rhs = solver_scratch[:n]

        # increment_cache and rhs_cache persist between steps for FSAL.
        # When solver_scratch is shared, slice from it; when local, use
        # persistent_local to maintain state between step invocations.
        if solver_scratch_in_shared:
            increment_cache = solver_scratch[n:int32(2)*n]
            rhs_cache = solver_scratch[:n]  # Aliases stage_rhs when shared
        else:
            increment_cache = persistent_local[:n]
            rhs_cache = persistent_local[n:int32(2)*n]

        for idx in range(n):
            if has_error and accumulates_error:
                error[idx] = typed_zero
            stage_increment[idx] = increment_cache[idx]  # cache spent

        status_code = int32(0)
        # --------------------------------------------------------------- #
        #            Stage 0: may reuse cached values                     #
        # --------------------------------------------------------------- #

        first_step = first_step_flag != int16(0)

        # Only use cache if all threads in warp can - otherwise no gain
        use_cached_rhs = False
        if first_same_as_last and multistage:
            if not first_step:
                mask = activemask()
                all_threads_accepted = all_sync(mask, accepted_flag != int16(0))
                use_cached_rhs = all_threads_accepted
        else:
            use_cached_rhs = False

        stage_time = current_time + dt_scalar * stage_time_fractions[0]
        diagonal_coeff = diagonal_coeffs[0]

        for idx in range(n):
            stage_base[idx] = state[idx]
            if accumulates_output:
                proposed_state[idx] = typed_zero

        if use_cached_rhs:
            # Load cached RHS from persistent storage (when solver_scratch
            # is local, rhs_cache points to persistent_local; when shared,
            # it aliases stage_rhs so this is a no-op)
            if not solver_scratch_in_shared:
                for idx in range(n):
                    stage_rhs[idx] = rhs_cache[idx]

        else:
            if can_reuse_accepted_start:
                for idx in range(int32(drivers_buffer.shape[0])):
                    # Use step-start driver values
                    proposed_drivers[idx] = drivers_buffer[idx]

            else:
                if has_driver_function:
                    driver_function(
                        stage_time,
                        driver_coeffs,
                        proposed_drivers,
                    )

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
            elif b_row == int32(0):
                # Direct assignment when stage 0 matches b_row
                proposed_state[idx] = stage_base[idx]
            if has_error:
                if accumulates_error:
                    # Standard accumulation
                    error[idx] += error_weight * rhs_value
                elif b_hat_row == int32(0):
                    # Direct assignment for error
                    error[idx] = stage_base[idx]

        for idx in range(accumulator_length):
            stage_accumulator[idx] = typed_zero

        # --------------------------------------------------------------- #
        #            Stages 1-s: must refresh all qtys                    #
        # --------------------------------------------------------------- #

        for prev_idx in range(stages_except_first):
            stage_offset = int32(prev_idx * n)
            stage_idx = prev_idx + int32(1)
            matrix_col = explicit_a_coeffs[prev_idx]

            # Stream previous stage's RHS into accumulators for successors
            for successor_idx in range(stages_except_first):
                coeff = matrix_col[successor_idx + int32(1)]
                row_offset = successor_idx * n
                for idx in range(n):
                    contribution = coeff * stage_rhs[idx] * dt_scalar
                    stage_accumulator[row_offset + idx] += contribution

            stage_time = (
                current_time + dt_scalar * stage_time_fractions[stage_idx]
            )

            if has_driver_function:
                driver_function(
                    stage_time,
                    driver_coeffs,
                    proposed_drivers,
                )

            for idx in range(n):
                stage_base[idx] = stage_accumulator[stage_offset + idx] + state[idx]

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
                proposed_state[idx] *= dt_scalar
                proposed_state[idx] += state[idx]
            if has_error:
                if accumulates_error:
                    error[idx] *= dt_scalar
                else:
                    error[idx] = proposed_state[idx] - error[idx]

        if has_driver_function:
            driver_function(
                end_time,
                driver_coeffs,
                proposed_drivers,
            )

        observables_function(
            proposed_state,
            parameters,
            proposed_drivers,
            proposed_observables,
            end_time,
        )

        # Cache increment and RHS for FSAL optimization
        for idx in range(n):
            increment_cache[idx] = stage_increment[idx]
            # Save RHS to cache (when solver_scratch is local, rhs_cache
            # points to persistent_local; when shared, aliases stage_rhs)
            if first_same_as_last:
                if not solver_scratch_in_shared:
                    rhs_cache[idx] = stage_rhs[idx]

        return status_code

    return step


# =========================================================================
# INLINE ERK STEP FACTORY (Generic ERK with tableau)
# =========================================================================

def erk_step_inline_factory(
    dxdt_fn,
    observables_function,
    driver_function,
    n,
    prec,
    tableau,
):
    """Create inline ERK step device function matching generic_erk.py.

    Parameters
    ----------
    dxdt_fn
        The derivative function.
    observables_function
        The observables function.
    driver_function
        The driver evaluation function (or None if no drivers).
    n
        Number of state variables.
    prec
        Precision dtype.
    tableau
        The ERK tableau.

    Notes
    -----
    Memory location flags are read from global scope:
    - use_shared_erk_stage_rhs
    - use_shared_erk_stage_accumulator
    """
    numba_precision = numba_from_dtype(prec)
    typed_zero = numba_precision(0.0)

    n_arraysize = n
    n = int32(n)
    stage_count = int32(tableau.stage_count)
    stages_except_first = stage_count - int32(1)
    accumulator_length = (tableau.stage_count - 1) * n_arraysize

    # Compile-time toggles
    has_driver_function = driver_function is not None
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

    # Memory location flags captured at compile time from global scope
    stage_rhs_in_shared = use_shared_erk_stage_rhs
    stage_accumulator_in_shared = use_shared_erk_stage_accumulator
    # stage_cache aliasing: prefers stage_rhs if shared, else accumulator
    # if shared, else needs persistent_local
    stage_cache_aliases_rhs = stage_rhs_in_shared
    stage_cache_aliases_accumulator = (not stage_rhs_in_shared and
                                       stage_accumulator_in_shared)

    @cuda.jit(
        [(
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
        )],
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
        # ----------------------------------------------------------- #
        # Shared and local buffer guide:
        # stage_accumulator: size (stage_count-1) * n.
        #   Memory location controlled by use_shared_stage_accumulator.
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
        # stage_rhs: size n.
        #   Memory location controlled by use_shared_stage_rhs.
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
        # Memory allocation based on configuration flags
        if stage_rhs_in_shared:
            stage_rhs = shared[:n]
        else:
            stage_rhs = cuda.local.array(n_arraysize, numba_precision)

        current_time = time_scalar
        end_time = current_time + dt_scalar

        if stage_accumulator_in_shared:
            stage_accumulator = shared[n:n + accumulator_length]
        else:
            stage_accumulator = cuda.local.array(accumulator_length,
                                                 dtype=precision)

        # stage_cache for FSAL: alias onto stage_rhs if shared, else
        # accumulator if shared (bottom n_states), else use persistent_local
        # persistent_local layout: [algo (4 elements), stage_cache (n elements)]
        if multistage:
            if stage_cache_aliases_rhs:
                stage_cache = stage_rhs
            elif stage_cache_aliases_accumulator:
                stage_cache = stage_accumulator[:n]
            else:
                # Neither shared - use persistent_local storage after algo
                stage_cache = persistent_local[int32(4):int32(4) + n]

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
                all_threads_accepted = all_sync(mask, accepted_flag != int16(0))
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
                    error[idx] = error[idx] + error_weights[0] * increment

        for idx in range(accumulator_length):
            stage_accumulator[idx] = typed_zero

        # ----------------------------------------------------------- #
        #            Stages 1-s: refresh observables and drivers       #
        # ----------------------------------------------------------- #
        for prev_idx in range(stages_except_first):
            stage_offset = prev_idx * n
            stage_idx = prev_idx + int32(1)
            matrix_col = stage_rhs_coeffs[prev_idx]

            for successor_idx in range(stages_except_first):
                coeff = matrix_col[successor_idx+int32(1)]
                row_offset = successor_idx * n
                for idx in range(n):
                    increment = stage_rhs[idx]
                    stage_accumulator[row_offset + idx] += coeff * increment

            base = stage_offset
            dt_stage = dt_scalar * stage_nodes[stage_idx]
            stage_time = current_time + dt_stage

            # Convert accumulated gradients sum(f(y_nj) into a state y_j
            for idx in range(n):
                stage_accumulator[base] = (stage_accumulator[base] *
                                           dt_scalar + state[idx])
                base += int32(1)

            # get rhs for next stage
            stage_drivers = proposed_drivers
            if has_driver_function:
                driver_function(
                    stage_time,
                    driver_coeffs,
                    stage_drivers,
                )

            observables_function(
                    stage_accumulator[stage_offset:stage_offset + n],
                    parameters,
                    stage_drivers,
                    proposed_observables,
                    stage_time,
            )

            dxdt_fn(
                stage_accumulator[stage_offset:stage_offset + n],
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

                if has_error:
                    if accumulates_error:
                        increment = stage_rhs[idx]
                        error[idx] += error_weight * increment

        if b_row is not None:
            for idx in range(n):
                proposed_state[idx] = stage_accumulator[(b_row-1) * n + idx]
        if b_hat_row is not None:
            for idx in range(n):
                error[idx] = stage_accumulator[(b_hat_row-1) * n + idx]
        # ----------------------------------------------------------- #
        for idx in range(n):

            # Scale and shift f(Y_n) value if accumulated
            if accumulates_output:
                proposed_state[idx] = (
                    proposed_state[idx] * dt_scalar + state[idx]
                )
            if has_error:
                # Scale error if accumulated
                if accumulates_error:
                    error[idx] *= dt_scalar
                #Or form error from difference if captured from a-row
                else:
                    error[idx] = proposed_state[idx] - error[idx]

        if has_driver_function:
            driver_function(
                end_time,
                driver_coeffs,
                proposed_drivers,
            )

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
# INLINE FIRK STEP FACTORY (Fully Implicit Runge-Kutta)
# =========================================================================

def firk_step_inline_factory(
    nonlinear_solver,
    dxdt_fn,
    observables_function,
    driver_function,
    n,
    prec,
    tableau,
):
    """Create inline FIRK step device function matching generic_firk.py.

    Parameters
    ----------
    nonlinear_solver
        The Newton-Krylov solver function for coupled stages.
    dxdt_fn
        The derivative function.
    observables_function
        The observables function.
    driver_function
        The driver evaluation function (or None if no drivers).
    n
        Number of state variables.
    prec
        Precision dtype.
    tableau
        The FIRK tableau.

    Notes
    -----
    Memory location flags are read from global scope:
    - use_shared_firk_solver_scratch
    - use_shared_firk_stage_increment
    - use_shared_firk_stage_driver_stack
    - use_shared_firk_stage_state
    """
    numba_precision = numba_from_dtype(prec)
    typed_zero = numba_precision(0.0)

    # Extract tableau properties
    # int32 versions for iterators
    n = int32(n)
    stage_count = int32(tableau.stage_count)
    all_stages_n = stage_count * n
    
    # int versions for cuda.local.array sizes
    n_int = int(n)
    stage_count_int = int(stage_count)
    all_stages_n_int = int(all_stages_n)

    # Compile-time toggles
    has_driver_function = driver_function is not None
    has_error = tableau.has_error_estimate

    stage_rhs_coeffs = tableau.a_flat(numba_precision)
    solution_weights = tableau.typed_vector(tableau.b, numba_precision)
    error_weights = tableau.error_weights(numba_precision)
    if error_weights is None or not has_error:
        error_weights = tuple(typed_zero for _ in range(stage_count_int))
    stage_time_fractions = tableau.typed_vector(tableau.c, numba_precision)

    # Last-step caching optimization
    accumulates_output = tableau.accumulates_output
    accumulates_error = tableau.accumulates_error
    b_row = tableau.b_matches_a_row
    b_hat_row = tableau.b_hat_matches_a_row
    if b_row is not None:
        b_row = int32(b_row)
    if b_hat_row is not None:
        b_hat_row = int32(b_hat_row)

    ends_at_one = stage_time_fractions[-1] == numba_precision(1.0)

    # Memory location flags captured at compile time from global scope
    solver_scratch_shared = use_shared_firk_solver_scratch
    stage_increment_shared = use_shared_firk_stage_increment
    stage_driver_stack_shared = use_shared_firk_stage_driver_stack
    stage_state_shared = use_shared_firk_stage_state

    solver_scratch_elements = 2 * all_stages_n
    stage_driver_stack_elements = stage_count * n_drivers
    
    # int versions for cuda.local.array sizes
    solver_scratch_elements_int = int(solver_scratch_elements)
    stage_driver_stack_elements_int = int(stage_driver_stack_elements)

    # Shared memory indices (only used when corresponding flag is True)
    shared_pointer = int32(0)

    # Solver scratch
    solver_scratch_start = shared_pointer
    solver_scratch_end = (solver_scratch_start + solver_scratch_elements
                          if solver_scratch_shared else solver_scratch_start)
    shared_pointer = solver_scratch_end

    # Stage increment
    stage_increment_start = shared_pointer
    stage_increment_end = (stage_increment_start + all_stages_n
                           if stage_increment_shared else stage_increment_start)
    shared_pointer = stage_increment_end

    # Stage driver stack
    stage_driver_stack_start = shared_pointer
    stage_driver_stack_end = (stage_driver_stack_start +
                              stage_driver_stack_elements
                              if stage_driver_stack_shared
                              else stage_driver_stack_start)
    shared_pointer = stage_driver_stack_end

    # Stage state
    stage_state_start = shared_pointer
    stage_state_end = (stage_state_start + n
                       if stage_state_shared else stage_state_start)
    shared_pointer = stage_state_end

    @cuda.jit(
        [(
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
        )],
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
        # Allocate buffers based on configuration flags
        if stage_state_shared:
            stage_state = shared[stage_state_start:stage_state_end]
        else:
            stage_state = cuda.local.array(n_int, numba_precision)

        if solver_scratch_shared:
            solver_scratch = shared[solver_scratch_start:solver_scratch_end]
        else:
            solver_scratch = cuda.local.array(solver_scratch_elements_int,
                                              numba_precision)

        if stage_increment_shared:
            stage_increment = shared[stage_increment_start:stage_increment_end]
        else:
            stage_increment = cuda.local.array(all_stages_n_int,
                                               numba_precision)

        if stage_driver_stack_shared:
            stage_driver_stack = shared[
                stage_driver_stack_start:stage_driver_stack_end
            ]
        else:
            stage_driver_stack = cuda.local.array(stage_driver_stack_elements_int,
                                                  numba_precision)

        current_time = time_scalar
        end_time = current_time + dt_scalar
        status_code = int32(0)
        stage_rhs_flat = solver_scratch[:all_stages_n]

        for idx in range(n):
            if accumulates_output:
                proposed_state[idx] = state[idx]
            if has_error and accumulates_error:
                error[idx] = typed_zero

        # Fill stage_drivers_stack if driver arrays provided
        if has_driver_function:
            for stage_idx in range(stage_count):
                stage_time = (
                    current_time + dt_scalar * stage_time_fractions[stage_idx]
                )
                driver_offset = stage_idx * n_drivers
                driver_slice = stage_driver_stack[
                    driver_offset:driver_offset + n_drivers
                ]
                driver_function(stage_time, driver_coeffs, driver_slice)

        status_code |= nonlinear_solver(
            stage_increment,
            parameters,
            stage_driver_stack,
            current_time,
            dt_scalar,
            typed_zero,
            state,
            solver_scratch,
            counters,
        )

        for stage_idx in range(stage_count):
            stage_time = (
                current_time + dt_scalar * stage_time_fractions[stage_idx]
            )

            if has_driver_function:
                stage_base = stage_idx * n_drivers
                stage_slice = stage_driver_stack[
                    stage_base:stage_base + n_drivers
                ]
                for idx in range(n_drivers):
                    proposed_drivers[idx] = stage_slice[idx]

            for idx in range(n):
                value = state[idx]
                for contrib_idx in range(stage_count):
                    flat_idx = stage_idx * stage_count + contrib_idx
                    coeff = stage_rhs_coeffs[flat_idx]
                    if coeff != typed_zero:
                        value += coeff * stage_increment[contrib_idx * n + idx]
                stage_state[idx] = value

            # Capture precalculated outputs if tableau allows
            if not accumulates_output:
                if b_row == stage_idx:
                    for idx in range(n):
                        proposed_state[idx] = stage_state[idx]
            if not accumulates_error:
                if b_hat_row == stage_idx:
                    for idx in range(n):
                        error[idx] = stage_state[idx]

            # Evaluate f at each stage for accumulation
            do_more_work = (has_error and accumulates_error) or accumulates_output

            if do_more_work:
                observables_function(
                    stage_state,
                    parameters,
                    proposed_drivers,
                    proposed_observables,
                    stage_time,
                )

                stage_rhs = stage_rhs_flat[
                    stage_idx * n:(stage_idx + int32(1)) * n
                ]
                dxdt_fn(
                    stage_state,
                    parameters,
                    proposed_drivers,
                    proposed_observables,
                    stage_rhs,
                    stage_time,
                )

        # Use Kahan summation algorithm to reduce floating point errors
        if accumulates_output:
            for idx in range(n):
                solution_acc = typed_zero
                for stage_idx in range(stage_count):
                    rhs_value = stage_rhs_flat[stage_idx * n + idx]
                    solution_acc += solution_weights[stage_idx] * rhs_value
                proposed_state[idx] = state[idx] + solution_acc * dt_scalar

        if has_error and accumulates_error:
            for idx in range(n):
                error_acc = typed_zero
                for stage_idx in range(stage_count):
                    rhs_value = stage_rhs_flat[stage_idx * n + idx]
                    error_acc += error_weights[stage_idx] * rhs_value
                error[idx] = dt_scalar * error_acc

        if not ends_at_one:
            if has_driver_function:
                driver_function(end_time, driver_coeffs, proposed_drivers)

            observables_function(
                proposed_state,
                parameters,
                proposed_drivers,
                proposed_observables,
                end_time,
            )

        if not accumulates_error:
            for idx in range(n):
                error[idx] = proposed_state[idx] - error[idx]

        return status_code

    return step


# =========================================================================
# INLINE ROSENBROCK STEP FACTORY (Rosenbrock-W methods)
# =========================================================================

def rosenbrock_step_inline_factory(
    linear_solver,
    prepare_jacobian,
    time_derivative_rhs,
    dxdt_fn,
    observables_function,
    driver_function,
    driver_del_t,
    n,
    prec,
    tableau,
):
    """Create inline Rosenbrock step device function.

    Parameters
    ----------
    linear_solver
        The linear solver function with cached Jacobian.
    prepare_jacobian
        Function to prepare Jacobian auxiliaries.
    time_derivative_rhs
        Function to compute time derivative terms.
    dxdt_fn
        The derivative function.
    observables_function
        The observables function.
    driver_function
        The driver evaluation function (or None if no drivers).
    driver_del_t
        The driver time derivative function (or None if no drivers).
    n
        Number of state variables.
    prec
        Precision dtype.
    tableau
        The Rosenbrock tableau.

    Notes
    -----
    Memory location flags are read from global scope:
    - use_shared_rosenbrock_stage_rhs
    - use_shared_rosenbrock_stage_store
    - use_shared_rosenbrock_cached_auxiliaries

    This is a simplified implementation for debugging purposes. Full
    Rosenbrock functionality requires system-specific Jacobian computation
    (prepare_jacobian) and time derivative evaluation (time_derivative_rhs)
    that are generated in production code by the solver_helpers system.
    The placeholder implementations here allow compilation and execution
    tracing but will not produce accurate results for real integrations.
    """
    numba_precision = numba_from_dtype(prec)
    typed_zero = numba_precision(0.0)

    # int32 versions for iterators
    n = int32(n)
    stage_count = int32(tableau.stage_count)
    stages_except_first = stage_count - int32(1)
    
    # int versions for cuda.local.array sizes
    n_int = int(n)
    stage_count_int = int(stage_count)
    stages_except_first_int = int(stages_except_first)

    has_driver_function = driver_function is not None
    has_error = tableau.has_error_estimate

    a_coeffs = tableau.typed_columns(tableau.a, numba_precision)
    C_coeffs = tableau.typed_columns(tableau.C, numba_precision)
    gamma_stages = tableau.typed_gamma_stages(numba_precision)
    gamma = numba_precision(tableau.gamma)
    solution_weights = tableau.typed_vector(tableau.b, numba_precision)
    error_weights = tableau.error_weights(numba_precision)
    if error_weights is None or not has_error:
        error_weights = tuple(typed_zero for _ in range(stage_count_int))
    stage_time_fractions = tableau.typed_vector(tableau.c, numba_precision)

    # Last-step caching optimization
    accumulates_output = tableau.accumulates_output
    accumulates_error = tableau.accumulates_error
    b_row = tableau.b_matches_a_row
    b_hat_row = tableau.b_hat_matches_a_row
    if b_row is not None:
        b_row = int32(b_row)
    if b_hat_row is not None:
        b_hat_row = int32(b_hat_row)

    # Memory location flags captured at compile time from global scope
    stage_rhs_shared = use_shared_rosenbrock_stage_rhs
    stage_store_shared = use_shared_rosenbrock_stage_store
    cached_auxiliaries_shared = use_shared_rosenbrock_cached_auxiliaries

    stage_store_elements = stage_count * n
    # For Rosenbrock, cached_auxiliary_count would be determined by the
    # solver helpers based on the system-specific Jacobian structure.
    # In production code, this is computed during helper construction.
    # For this debug script, we use 0 as a placeholder since the actual
    # Jacobian helpers are not implemented.
    cached_auxiliary_count = int32(0)  # Simplified for debug script
    
    # int versions for cuda.local.array sizes
    stage_store_elements_int = int(stage_store_elements)
    cached_auxiliary_count_int = int(cached_auxiliary_count)

    # Shared memory indices
    shared_pointer = int32(0)

    # Stage RHS
    stage_rhs_start = shared_pointer
    stage_rhs_end = (stage_rhs_start + n
                     if stage_rhs_shared else stage_rhs_start)
    shared_pointer = stage_rhs_end

    # Stage store
    stage_store_start = shared_pointer
    stage_store_end = (stage_store_start + stage_store_elements
                       if stage_store_shared else stage_store_start)
    shared_pointer = stage_store_end

    # Cached auxiliaries
    cached_aux_start = shared_pointer
    cached_aux_end = (cached_aux_start + cached_auxiliary_count
                      if cached_auxiliaries_shared else cached_aux_start)
    shared_pointer = cached_aux_end

    @cuda.jit(
        [(
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
        )],
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
        # Allocate buffers based on configuration flags
        if stage_rhs_shared:
            stage_rhs = shared[stage_rhs_start:stage_rhs_end]
        else:
            stage_rhs = cuda.local.array(n_int, numba_precision)

        if stage_store_shared:
            stage_store = shared[stage_store_start:stage_store_end]
        else:
            stage_store = cuda.local.array(stage_store_elements_int,
                                           numba_precision)

        if cached_auxiliaries_shared and cached_auxiliary_count > 0:
            cached_auxiliaries = shared[cached_aux_start:cached_aux_end]
        else:
            # Use a minimal local array as placeholder.
            # When cached_auxiliary_count is 0, we allocate size 1 to avoid
            # zero-sized array issues in Numba. The array won't be used since
            # the placeholder prepare_jacobian does nothing.
            cached_auxiliaries = cuda.local.array(max(1, cached_auxiliary_count_int),
                                                  numba_precision)

        current_time = time_scalar
        end_time = current_time + dt_scalar
        final_stage_base = n * (stage_count - int32(1))
        time_derivative = stage_store[final_stage_base:final_stage_base + n]

        inv_dt = numba_precision(1.0) / dt_scalar

        prepare_jacobian(
            state, parameters, drivers_buffer, current_time, cached_auxiliaries
        )

        # Evaluate del_t term at t_n, y_n
        if has_driver_function:
            driver_del_t(current_time, driver_coeffs, proposed_drivers)
        else:
            for idx in range(n_drivers):
                proposed_drivers[idx] = numba_precision(0.0)

        # Stage 0 slice copies the cached final increment as its guess
        stage_increment = stage_store[:n]

        for idx in range(n):
            stage_increment[idx] = time_derivative[idx]

        time_derivative_rhs(
            state,
            parameters,
            drivers_buffer,
            proposed_drivers,
            observables,
            time_derivative,
            current_time,
        )

        for idx in range(n):
            proposed_state[idx] = state[idx]
            time_derivative[idx] *= dt_scalar
            if has_error:
                error[idx] = typed_zero

        status_code = int32(0)
        stage_time = current_time + dt_scalar * stage_time_fractions[0]

        # Stage 0: uses starting values
        dxdt_fn(
            state,
            parameters,
            drivers_buffer,
            observables,
            stage_rhs,
            current_time,
        )

        for idx in range(n):
            f_value = stage_rhs[idx]
            rhs_value = (
                (f_value + gamma_stages[0] * time_derivative[idx]) * dt_scalar
            )
            stage_rhs[idx] = rhs_value * gamma

        # Create empty array slice as placeholder for signature compatibility.
        # The linear solver expects a base_state parameter, but Rosenbrock
        # doesn't use it. This empty slice satisfies the signature without
        # allocating memory.
        base_state_placeholder = shared[int32(0):int32(0)]

        status_code |= linear_solver(
            state,
            parameters,
            drivers_buffer,
            base_state_placeholder,
            cached_auxiliaries,
            stage_time,
            dt_scalar,
            numba_precision(1.0),
            stage_rhs,
            stage_increment,
            shared,
        )

        for idx in range(n):
            if accumulates_output:
                proposed_state[idx] += (
                    stage_increment[idx] * solution_weights[int32(0)]
                )
            if has_error and accumulates_error:
                error[idx] += stage_increment[idx] * error_weights[int32(0)]

        # Stages 1-s: must refresh all values
        for prev_idx in range(stages_except_first):
            stage_idx = prev_idx + int32(1)
            stage_offset = stage_idx * n
            stage_gamma = gamma_stages[stage_idx]
            stage_time = (
                current_time + dt_scalar * stage_time_fractions[stage_idx]
            )

            # Get base state
            stage_slice = stage_store[stage_offset:stage_offset + n]
            for idx in range(n):
                stage_slice[idx] = state[idx]

            # Accumulate contributions from predecessor stages
            for predecessor_idx in range(stages_except_first):
                a_col = a_coeffs[predecessor_idx]
                a_coeff = a_col[stage_idx]
                if predecessor_idx < stage_idx:
                    base_idx = predecessor_idx * n
                    for idx in range(n):
                        prior_val = stage_store[base_idx + idx]
                        stage_slice[idx] += a_coeff * prior_val

            if has_driver_function:
                driver_function(stage_time, driver_coeffs, proposed_drivers)

            observables_function(
                stage_slice,
                parameters,
                proposed_drivers,
                proposed_observables,
                stage_time,
            )

            dxdt_fn(
                stage_slice,
                parameters,
                proposed_drivers,
                proposed_observables,
                stage_rhs,
                stage_time,
            )

            # Capture precalculated outputs if tableau allows
            if b_row == stage_idx:
                for idx in range(n):
                    proposed_state[idx] = stage_slice[idx]
            if b_hat_row == stage_idx:
                for idx in range(n):
                    error[idx] = stage_slice[idx]

            # Recompute time-derivative for last stage
            if stage_idx == stage_count - int32(1):
                if has_driver_function:
                    driver_del_t(current_time, driver_coeffs, proposed_drivers)
                time_derivative_rhs(
                    state,
                    parameters,
                    drivers_buffer,
                    proposed_drivers,
                    observables,
                    time_derivative,
                    current_time,
                )
                for idx in range(n):
                    time_derivative[idx] *= dt_scalar

            # Add C_ij*K_j/dt + dt * gamma_i * d/dt terms to rhs
            for idx in range(n):
                correction = numba_precision(0.0)
                for predecessor_idx in range(stages_except_first):
                    c_col = C_coeffs[predecessor_idx]
                    c_coeff = c_col[stage_idx]
                    if predecessor_idx < stage_idx:
                        prior_idx = predecessor_idx * n + idx
                        prior_val = stage_store[prior_idx]
                        correction += c_coeff * prior_val

                f_stage_val = stage_rhs[idx]
                deriv_val = stage_gamma * time_derivative[idx]
                rhs_value = f_stage_val + correction * inv_dt + deriv_val
                stage_rhs[idx] = rhs_value * dt_scalar * gamma

            # Alias slice of stage storage for convenience
            stage_increment = stage_slice

            # Use previous stage's solution as a guess for this stage
            previous_base = prev_idx * n
            for idx in range(n):
                stage_increment[idx] = stage_store[previous_base + idx]

            status_code |= linear_solver(
                state,
                parameters,
                drivers_buffer,
                base_state_placeholder,
                cached_auxiliaries,
                stage_time,
                dt_scalar,
                numba_precision(1.0),
                stage_rhs,
                stage_increment,
                shared,
            )

            for idx in range(n):
                if accumulates_output:
                    proposed_state[idx] += (
                        stage_increment[idx] * solution_weights[stage_idx]
                    )
                if has_error and accumulates_error:
                    error[idx] += (
                        stage_increment[idx] * error_weights[stage_idx]
                    )

        if has_driver_function:
            driver_function(end_time, driver_coeffs, proposed_drivers)

        observables_function(
            proposed_state,
            parameters,
            proposed_drivers,
            proposed_observables,
            end_time,
        )

        if not accumulates_error:
            for idx in range(n):
                error[idx] = proposed_state[idx] - error[idx]

        return status_code

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
    for k in range(n_states32):
        cuda.stwt(output_states_slice, k, current_state[k])
    cuda.stwt(output_states_slice, n_states32, current_step)
    for i in range(n_counters32):
        cuda.stwt(output_counters_slice, i, current_counters[i])


# =========================================================================
# SUMMARY METRIC FUNCTIONS (Mean metric with chained pattern)
# =========================================================================

@cuda.jit(
    ["float32, float32[::1], int32, int32",
     "float64, float64[::1], int32, int32"],
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
    ["float32[::1], float32[::1], int32, int32",
     "float64[::1], float64[::1], int32, int32"],
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
    """Accumulate summary metrics from the current state sample."""
    total_buffer_size = int32(1)  # 1 slot for mean metric per variable
    for idx in range(n_states32):
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
    total_buffer_size = int32(1)  # 1 slot for mean metric per variable
    total_output_size = int32(1)  # 1 output for mean metric per variable
    for state_index in range(n_states32):
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

@cuda.jit(device=True, inline=True, **compile_kwargs)
def clamp(value, min_val, max_val):
    """Clamp a value between min and max."""
    return max(min_val, min(value, max_val))


@cuda.jit(
    [(
        numba_precision[::1],
        numba_precision[::1],
        numba_precision[::1],
        numba_precision[::1],
        int32,
        int32[::1],
        numba_precision[::1],
    )],
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
    n = int32(n)
    inv_n = precision(1.0 / n)
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

# Build driver function if drivers are present
if n_drivers > 0 and driver_input_dict is not None:
    # Create ArrayInterpolator instance to compute coefficients
    interpolator = ArrayInterpolator(precision, driver_input_dict)
    
    # Get coefficients from the interpolator
    driver_coefficients = interpolator.coefficients
    
    # Build inline driver evaluation function
    driver_function = driver_function_inline_factory(interpolator)
else:
    driver_function = None
    interpolator = None  # Define as None when drivers not present
    # Create dummy coefficients array for kernel signature compatibility
    driver_coefficients = np.zeros((1, max(n_drivers, 1), 6), dtype=precision)

# Build step function based on algorithm type
if algorithm_type == 'erk':
    # ERK step for explicit integration
    step_fn = erk_step_inline_factory(
        dxdt_fn,
        observables_function,
        driver_function,
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

    linear_solver_fn = linear_solver_inline_factory(
        operator_fn, n_states,
        preconditioner_fn,
        krylov_tolerance,
        max_linear_iters,
        precision,
        linear_correction_type,
    )

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
        driver_function,
        n_states,
        precision,
        tableau,
    )
elif algorithm_type == 'firk':
    # Build implicit solver components for FIRK (fully implicit)
    # FIRK requires n-stage coupled system solving
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
        operator_fn, n_states * tableau.stage_count,  # Note: all_stages_n
        preconditioner_fn,
        krylov_tolerance,
        max_linear_iters,
        precision,
        linear_correction_type,
    )

    newton_solver_fn = newton_krylov_inline_factory(
        residual_fn,
        linear_solver_fn,
        n_states * tableau.stage_count,  # Note: all_stages_n
        newton_tolerance,
        max_newton_iters,
        newton_damping,
        max_backtracks,
        precision,
    )

    step_fn = firk_step_inline_factory(
        newton_solver_fn,
        dxdt_fn,
        observables_function,
        driver_function,
        n_states,
        precision,
        tableau,
    )
elif algorithm_type == 'rosenbrock':
    # Build linear solver components for Rosenbrock (linearly implicit)
    # Rosenbrock uses cached Jacobian approximation
    preconditioner_fn = neumann_preconditioner_factory(
        constants,
        precision,
        beta=float(beta_solver),
        gamma=float(tableau.gamma),  # Use tableau gamma for Rosenbrock
        order=preconditioner_order,
    )
    operator_fn = linear_operator_factory(
        constants,
        precision,
        beta=float(beta_solver),
        gamma=float(tableau.gamma),  # Use tableau gamma for Rosenbrock
        order=preconditioner_order,
    )

    # For Rosenbrock, we need a linear solver with cached Jacobian
    # This is a simplified version - full implementation would need
    # cached Jacobian helpers
    linear_solver_fn = linear_solver_inline_factory(
        operator_fn, n_states,
        preconditioner_fn,
        krylov_tolerance,
        max_linear_iters,
        precision,
        linear_correction_type,
    )

    # Placeholder functions for Rosenbrock-specific operations
    # NOTE: These are simplified implementations for the debug script.
    # A full Rosenbrock implementation requires system-specific Jacobian
    # computation, which is generated by the solver_helpers system in
    # production code. For lineinfo debugging purposes, these placeholders
    # allow the step function to compile and trace execution flow.
    def prepare_jacobian_placeholder(*args):
        # Production code computes Jacobian approximation here
        pass

    def time_derivative_rhs_placeholder(*args):
        # Compute f_t = df/dt at current state
        # Production code evaluates time derivatives; simplified here as zero
        out_idx = len(args) - 2
        out_array = args[out_idx]
        for i in range(len(out_array)):
            out_array[i] = precision(0.0)

    # Driver time derivative (if drivers present)
    # Note: interpolator is guaranteed non-None when this condition is True,
    # as it's defined at module level in the driver setup block with the same condition
    if n_drivers > 0 and driver_input_dict is not None:
        driver_del_t = driver_derivative_inline_factory(interpolator)
    else:
        driver_del_t = None

    step_fn = rosenbrock_step_inline_factory(
        linear_solver_fn,
        prepare_jacobian_placeholder,
        time_derivative_rhs_placeholder,
        dxdt_fn,
        observables_function,
        driver_function,
        driver_del_t,
        n_states,
        precision,
        tableau,
    )
else:
    raise ValueError(f"Unknown algorithm type: '{algorithm_type}'. "
                     "Use 'erk', 'dirk', 'firk', or 'rosenbrock'.")

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

# Buffer layout - dynamic based on memory location configuration
# Pattern: x_size defines element count, x_start/x_end define shared offsets.
# Buffers only advance shared_pointer when their flag is True.

# Buffer sizes (unconditional)
state_buffer_size = int32(n_states)
proposed_state_size = int32(n_states)
params_size = int32(n_parameters)
drivers_size = int32(n_drivers)
proposed_drivers_size = int32(n_drivers)
obs_size = int32(n_observables)
proposed_obs_size = int32(n_observables)
error_size = int32(n_states)
counters_size = int32(n_counters) if save_counters_bool else int32(0)
proposed_counters_size = int32(2) if save_counters_bool else int32(0)
state_summ_size = int32(n_states) if summarise_state_bool else int32(0)
obs_summ_size = int32(n_observables) if summarise_obs_bool else int32(0)

# Scratch sizes depend on algorithm type
accumulator_size = int32((stage_count - 1) * n_states)
if algorithm_type == 'dirk':
    solver_scratch_size = 2 * n_states
    dirk_scratch_size = int(accumulator_size) + int(solver_scratch_size)
    erk_scratch_size = 0
    firk_scratch_size = 0
    rosenbrock_scratch_size = 0
elif algorithm_type == 'erk':
    solver_scratch_size = int32(0)
    dirk_scratch_size = 0
    firk_scratch_size = 0
    rosenbrock_scratch_size = 0
    # ERK: stage_rhs if shared, accumulator if shared
    erk_stage_rhs_size = n_states + 1 if use_shared_erk_stage_rhs else (
        0)
    erk_accumulator_shared_size = (accumulator_size
                                   if use_shared_erk_stage_accumulator
                                   else 0)
    erk_scratch_size = erk_stage_rhs_size + erk_accumulator_shared_size
elif algorithm_type == 'firk':
    # FIRK memory requirements:
    # - solver_scratch: 2 * stage_count * n_states (for Newton solver)
    # - stage_increment: stage_count * n_states (for stage increments)
    # - stage_driver_stack: stage_count * n_drivers (for driver values)
    # - stage_state: n_states (for state evaluation)
    all_stages_n = stage_count * n_states
    firk_solver_scratch_size = (
        2 * all_stages_n if use_shared_firk_solver_scratch else 0
    )
    firk_stage_increment_size = (
        all_stages_n if use_shared_firk_stage_increment else 0
    )
    firk_stage_driver_stack_size = (
        stage_count * n_drivers if use_shared_firk_stage_driver_stack else 0
    )
    firk_stage_state_size = (
        n_states if use_shared_firk_stage_state else 0
    )
    firk_scratch_size = (firk_solver_scratch_size + firk_stage_increment_size +
                         firk_stage_driver_stack_size + firk_stage_state_size)
    solver_scratch_size = int32(0)
    dirk_scratch_size = 0
    erk_scratch_size = 0
    rosenbrock_scratch_size = 0
elif algorithm_type == 'rosenbrock':
    # Rosenbrock memory requirements:
    # - stage_rhs: n_states (for RHS evaluation)
    # - stage_store: stage_count * n_states (for stage storage)
    # - cached_auxiliaries: 0 (simplified for debug script)
    rosenbrock_stage_rhs_size = (
        n_states if use_shared_rosenbrock_stage_rhs else 0
    )
    rosenbrock_stage_store_size = (
        stage_count * n_states if use_shared_rosenbrock_stage_store else 0
    )
    rosenbrock_cached_auxiliaries_size = (
        # Cached auxiliaries store precomputed Jacobian terms.
        # In production, size depends on system-specific structure.
        # Set to 0 here since placeholder Jacobian helpers are used.
        0
    )
    rosenbrock_scratch_size = (rosenbrock_stage_rhs_size +
                               rosenbrock_stage_store_size +
                               rosenbrock_cached_auxiliaries_size)
    solver_scratch_size = int32(0)
    dirk_scratch_size = 0
    erk_scratch_size = 0
    firk_scratch_size = 0
else:
    raise ValueError(f"Unknown algorithm type: '{algorithm_type}'")


# Shared memory pointer (advances for each shared buffer)
shared_pointer = int32(0)

# State buffer
state_shared_start = shared_pointer
state_shared_end = (state_shared_start + state_buffer_size
                    if use_shared_loop_state else state_shared_start)
shared_pointer = state_shared_end

# Proposed state buffer
proposed_state_start = shared_pointer
proposed_state_end = (proposed_state_start + proposed_state_size
                      if use_shared_loop_state_proposal else proposed_state_start)
shared_pointer = proposed_state_end

# Parameters buffer
params_start = shared_pointer
params_end = (params_start + params_size
              if use_shared_loop_parameters else params_start)
shared_pointer = params_end

# Drivers buffer
drivers_start = shared_pointer
drivers_end = (drivers_start + drivers_size
               if use_shared_loop_drivers else drivers_start)
shared_pointer = drivers_end

# Proposed drivers buffer
proposed_drivers_start = shared_pointer
proposed_drivers_end = (proposed_drivers_start + proposed_drivers_size
                        if use_shared_loop_drivers_proposal
                        else proposed_drivers_start)
shared_pointer = proposed_drivers_end

# Observables buffer
obs_start = shared_pointer
obs_end = (obs_start + obs_size
           if use_shared_loop_observables else obs_start)
shared_pointer = obs_end

# Proposed observables buffer
proposed_obs_start = shared_pointer
proposed_obs_end = (proposed_obs_start + proposed_obs_size
                    if use_shared_loop_observables_proposal
                    else proposed_obs_start)
shared_pointer = proposed_obs_end

# Error buffer
error_start = shared_pointer
error_end = (error_start + error_size
             if use_shared_loop_error else error_start)
shared_pointer = error_end

# Counters buffer
counters_start = shared_pointer
counters_end = (counters_start + counters_size
                if use_shared_loop_counters else counters_start)
shared_pointer = counters_end

# Proposed counters buffer
proposed_counters_start = shared_pointer
proposed_counters_end = (proposed_counters_start + proposed_counters_size
                         if use_shared_loop_counters else proposed_counters_start)
shared_pointer = proposed_counters_end

# Scratch buffer for step algorithms
scratch_start = shared_pointer
if algorithm_type == 'dirk':
    scratch_size = dirk_scratch_size if use_shared_loop_scratch else int32(0)
    local_scratch_size = dirk_scratch_size
elif algorithm_type == 'erk':
    scratch_size = erk_scratch_size if use_shared_loop_scratch else 0
    # ERK local scratch: accumulator_size + n_states
    local_scratch_size = accumulator_size + int32(n_states)
elif algorithm_type == 'firk':
    scratch_size = firk_scratch_size if use_shared_loop_scratch else int32(0)
    # FIRK local scratch: sum of all buffer sizes when not in shared memory
    # all_stages_n = stage_count * n_states (calculated in memory size section)
    local_scratch_size = (
        2 * stage_count * n_states +  # solver_scratch
        stage_count * n_states +      # stage_increment
        stage_count * n_drivers +  # stage_driver_stack
        n_states            # stage_state
    )
elif algorithm_type == 'rosenbrock':
    scratch_size = (rosenbrock_scratch_size if use_shared_loop_scratch
                    else int32(0))
    # Rosenbrock local scratch: sum of all buffer sizes when not in shared
    local_scratch_size = (
        n_states +           # stage_rhs
        stage_count * n_states +  # stage_store
        1  # cached_auxiliaries (minimum size 1 to avoid zero-size array in Numba)
    )
else:
    raise ValueError(f"Unknown algorithm type: '{algorithm_type}'")
scratch_end = scratch_start + scratch_size
shared_pointer = scratch_end

# State summary buffer
state_summ_start = shared_pointer
state_summ_end = (state_summ_start + state_summ_size
                  if use_shared_loop_state_summary else state_summ_start)
shared_pointer = state_summ_end

# Observable summary buffer
obs_summ_start = shared_pointer
obs_summ_end = (obs_summ_start + obs_summ_size
                if use_shared_loop_observable_summary else obs_summ_start)
shared_pointer = obs_summ_end

# Total shared memory elements required
shared_elements = shared_pointer


local_dt_slice = slice(0, 1)
local_accept_slice = slice(1, 2)
local_controller_slice = slice(2, 4)
local_algo_slice = slice(4, 8)
# Persistent local storage for step function includes algo (4 elements)
# plus stage_cache if needed for ERK (n_states elements)
base_local_elements = 8

# Add space for ERK stage_cache if it's not aliased to shared memory
# Uses previously computed use_shared_erk_stage_cache flag
if algorithm_type == 'erk':
    stage_cache_needs_local = not use_shared_erk_stage_cache
    stage_cache_local_size = n_states if stage_cache_needs_local else 0
else:
    stage_cache_local_size = 0

local_elements = base_local_elements + stage_cache_local_size
# Slice for step function persistent_local: algo + stage_cache
local_step_slice = slice(4, 8 + stage_cache_local_size)

status_mask = int32(0xFFFF)

obs_nonzero = max(n_observables, 1)
drv_nonzero = max(n_drivers, 1)
ncnt_nonzero = max(n_counters, 1)
@cuda.jit(
    [
        (
            numba_precision[::1],
            numba_precision[::1],
            numba_precision[:, :, ::1],
            numba_precision[::1],
            numba_precision[::1],
            numba_precision[:, ::1],
            numba_precision[:, ::1],
            numba_precision[:, ::1],
            numba_precision[:, ::1],
            numba_precision[:, ::1],
            float64,
            float64,
            float64,
        )
    ],
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
    t = float64(t0)
    t_prec = numba_precision(t)
    t_end = numba_precision(settling_time + t0 + duration)

    # Cap max iterations - all internal steps at dt_min, plus a bonus
    # end/start, plus one failure per successful step.
    # 64-bits required to get any reasonable duration with small step
    total_duration = duration + settling_time
    max_steps = min(
            int64(2**62), (int64(ceil(total_duration/dt_min)) + 2)
    )
    max_steps = max_steps << 1

    shared_scratch[:] = numba_precision(0.0)

    # Allocate buffers based on memory location configuration
    # Each buffer uses shared memory if its flag is True, otherwise local
    if use_shared_loop_state:
        state_buffer = shared_scratch[state_shared_start:state_shared_end]
    else:
        state_buffer = cuda.local.array(n_states, numba_precision)

    if use_shared_loop_state_proposal:
        state_proposal_buffer = shared_scratch[proposed_state_start:
                                               proposed_state_end]
    else:
        state_proposal_buffer = cuda.local.array(n_states, numba_precision)

    if use_shared_loop_observables:
        observables_buffer = shared_scratch[obs_start:obs_end]
    else:
        observables_buffer = cuda.local.array(obs_nonzero,
                                              numba_precision)

    if use_shared_loop_observables_proposal:
        observables_proposal_buffer = shared_scratch[proposed_obs_start:
                                                     proposed_obs_end]
    else:
        observables_proposal_buffer = cuda.local.array(obs_nonzero,
                                                       numba_precision)

    if use_shared_loop_parameters:
        parameters_buffer = shared_scratch[params_start:params_end]
    else:
        parameters_buffer = cuda.local.array(n_parameters, numba_precision)

    if use_shared_loop_drivers:
        drivers_buffer = shared_scratch[drivers_start:drivers_end]
    else:
        drivers_buffer = cuda.local.array(drv_nonzero, numba_precision)

    if use_shared_loop_drivers_proposal:
        drivers_proposal_buffer = shared_scratch[proposed_drivers_start:
                                                 proposed_drivers_end]
    else:
        drivers_proposal_buffer = cuda.local.array(drv_nonzero,
                                                   numba_precision)

    if use_shared_loop_state_summary:
        state_summary_buffer = shared_scratch[state_summ_start:state_summ_end]
    else:
        state_summary_buffer = cuda.local.array(n_states,
                                                numba_precision)

    if use_shared_loop_observable_summary:
        observable_summary_buffer = shared_scratch[obs_summ_start:obs_summ_end]
    else:
        observable_summary_buffer = cuda.local.array(obs_nonzero,
                                                     numba_precision)

    if use_shared_loop_scratch:
        remaining_shared_scratch = shared_scratch[scratch_start:scratch_end]
    else:
        # Local scratch sized for the algorithm (computed at module level)
        remaining_shared_scratch = cuda.local.array(local_scratch_size,
                                                    numba_precision)

    if use_shared_loop_counters:
        counters_since_save = shared_scratch[counters_start:counters_end]
    else:
        counters_since_save = cuda.local.array(ncnt_nonzero,
                                               simsafe_int32)

    if save_counters_bool and use_shared_loop_counters:
        # When enabled and shared, use shared memory buffers
        proposed_counters = shared_scratch[proposed_counters_start:
                                           proposed_counters_end]
    else:
        # When disabled or local, use a local "proposed_counters" buffer
        proposed_counters = cuda.local.array(2, dtype=simsafe_int32)

    dt = persistent_local[local_dt_slice]
    accept_step = persistent_local[local_accept_slice].view(simsafe_int32)

    if use_shared_loop_error:
        error = shared_scratch[error_start:error_end]
    else:
        error = cuda.local.array(n_states, numba_precision)

    controller_temp = persistent_local[local_controller_slice]
    step_persistent_local = persistent_local[local_step_slice]

    first_step_flag = int16(1)
    prev_step_accepted_flag = int16(1)

    # ----------------------------------------------------------------------- #
    #                       Seed t=0 values                                   #
    # ----------------------------------------------------------------------- #
    for k in range(n_states):
        state_buffer[k] = initial_states[k]
    for k in range(n_parameters):
        parameters_buffer[k] = parameters[k]

    # Seed initial observables from initial state.
    # driver_function not used in this test (n_drivers = 0)
    if n_observables > 0:
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
    next_save = numba_precision(settling_time + t0)
    if settling_time == 0.0:
        # Save initial state at t0, then advance to first interval save
        next_save += dt_save

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
    dt[0] = dt0
    dt_raw = dt0
    accept_step[0] = int32(0)

    # Initialize iteration counters
    for i in range(n_counters):
        counters_since_save[i] = int32(0)
        if i < 2:
            proposed_counters[i] = int32(0)

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
            at_last_save = finished and t_prec < t_end
            finished = selp(at_last_save, False, True)
            dt[0] = selp(at_last_save, numba_precision(t_end - t),
                         dt_raw)

        # also exit loop if min step size limit hit - things are bad
        finished |= (status & int32(0x8))

        if all_sync(mask, finished):
            return status

        if not finished:
            do_save = (t_prec + dt_raw) >= next_save
            dt_eff = selp(do_save, next_save - t_prec, dt_raw)

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
                step_persistent_local,
                proposed_counters,
            )
            first_step_flag = int16(0)

            niters = (step_status >> 16) & status_mask
            status |= step_status & status_mask

            # Adjust dt if step rejected - auto-accepts if fixed-step
            if not fixed_mode:
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
            dt_raw = dt[0]
            # Accumulate iteration counters if active
            if save_counters_bool:
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

            t_proposal = t + dt_eff
            t = selp(accept, t_proposal, t)
            t_prec = numba_precision(t)

            for i in range(n_states):
                newv = state_proposal_buffer[i]
                oldv = state_buffer[i]
                state_buffer[i] = selp(accept, newv, oldv)

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
                    observables_output[save_idx * save_obs_bool, :],
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
                if save_counters_bool:
                    for i in range(n_counters):
                        counters_since_save[i] = int32(0)

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
        [(
                numba_prec[:,::1],
                numba_prec[:, ::1],
                numba_prec[:, :, ::1],
                numba_prec[:, :, ::1],
                numba_prec[:, :, ::1],
                numba_prec[:, :, ::1],
                numba_prec[:, :, ::1],
                int32[:, :, ::1],
                int32[::1],
                float64,
                float64,
                float64,
                int32,
            )],
        **compile_kwargs)
def integration_kernel(inits, params, d_coefficients, state_output,
                       observables_output, state_summaries_output,
                       observables_summaries_output, iteration_counters_output,
                       status_codes_output, duration_k, warmup_k, t0_k,
                       n_runs_k):

    tx = cuda.threadIdx.x
    block_index = int32(cuda.blockIdx.x)
    runs_per_block = cuda.blockDim.x
    run_index = int32(runs_per_block * block_index + tx)
    if run_index >= n_runs_k:
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
    rx_state = state_output[:, :, run_index]
    rx_observables = observables_output[:, :, 0]
    rx_state_summaries = state_summaries_output[:, :, run_index]
    rx_observables_summaries = observables_summaries_output[:, :, 0]
    rx_iteration_counters = iteration_counters_output[:, :, run_index]

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
    algo_name = algorithm_type.upper()
    ctrl_name = controller_type.upper()
    print(f"Debug Integration - {algo_name} ({algorithm_tableau_name}) "
          f"with {ctrl_name} controller")
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

    # Create device arrays for inputs (BatchInputArrays pattern)
    d_inits = cuda.device_array((n_runs, n_states), dtype=precision)
    d_inits.copy_to_device(inits)
    d_params = cuda.to_device(params)
    d_driver_coefficients = cuda.to_device(driver_coefficients)

    # Create mapped arrays for outputs (BatchOutputArrays pattern)
    # Using stride ordering to match production batch solver memory layout.
    # Dimension indices: 0=time, 1=run, 2=variable
    n_summary_samples = int(ceil(n_output_samples / saves_per_summary))

    # state_output: shape=(samples, states+1, runs), native order=(time, var, run)
    state_shape = (n_output_samples, n_states + 1, n_runs)
    state_strides = get_strides(state_shape, precision, (0, 1, 2))
    state_output = cuda.device_array(state_shape, dtype=precision,
                                     strides=state_strides)

    # observables_output: shape=(samples, 1, 1), native order=(time, var, run)
    obs_shape = (n_output_samples, 1, 1)
    obs_strides = get_strides(obs_shape, precision, (0, 1, 2))
    observables_output = cuda.device_array(obs_shape, dtype=precision,
                                           strides=obs_strides)

    # state_summaries_output: shape=(summaries, states, runs)
    # native order=(time, var, run)
    state_summ_shape = (n_summary_samples, n_states, n_runs)
    state_summ_strides = get_strides(state_summ_shape, precision, (0, 1, 2))
    state_summaries_output = cuda.device_array(state_summ_shape, dtype=precision,
                                               strides=state_summ_strides)

    # observable_summaries_output: shape=(summaries, 1, 1)
    # native order=(time, var, run)
    obs_summ_shape = (n_summary_samples, 1, 1)
    obs_summ_strides = get_strides(obs_summ_shape, precision, (0, 1, 2))
    observable_summaries_output = cuda.device_array(obs_summ_shape,
                                                    dtype=precision,
                                                    strides=obs_summ_strides)

    # iteration_counters_output: shape=(runs, samples, counters)
    # native order=(run, time, var) - unchanged (special case)
    iter_shape = (n_output_samples, n_counters, n_runs)
    iter_strides = get_strides(iter_shape, np.int32, (0, 1, 2))
    iteration_counters_output = cuda.device_array(iter_shape, dtype=np.int32,
                                                  strides=iter_strides)

    # status_codes_output: 1D array, no custom strides needed
    status_codes_output = cuda.device_array((n_runs,), dtype=np.int32)

    print(f"State output shape: {state_output.shape}")

    # Kernel configuration - use global blocksize from config
    current_blocksize = blocksize
    runs_per_block = current_blocksize
    dynamic_sharedmem = int32(
        (f32_per_element * run_stride_f32) * 4 * runs_per_block)

    while dynamic_sharedmem > MAX_SHARED_MEMORY_PER_BLOCK:
        current_blocksize = current_blocksize // 2
        runs_per_block = current_blocksize
        dynamic_sharedmem = int(
            (f32_per_element * run_stride_f32) * 4 * runs_per_block)

    blocks_per_grid = int(ceil(n_runs / runs_per_block))

    print("\nKernel configuration:")
    print(f"  Block size: {current_blocksize}")
    print(f"  Blocks per grid: {blocks_per_grid}")
    print(f"  Shared memory per block: {dynamic_sharedmem} bytes")
    stream = cuda.stream()
    print("\nLaunching kernel...")
    kernel_launch_time = perf_counter()
    integration_kernel[blocks_per_grid, current_blocksize, stream,
                       dynamic_sharedmem](
        d_inits, d_params, d_driver_coefficients, state_output,
        observables_output, state_summaries_output,
        observable_summaries_output, iteration_counters_output,
        status_codes_output, duration, warmup, precision(0.0), n_runs)

    kernel_end_time = perf_counter()
    # Mapped arrays provide direct host access after synchronization
    # No explicit copy_to_host required
    print(f"\nKernel Execution time: {kernel_end_time - kernel_launch_time}")
    status_codes_output = status_codes_output.copy_to_host(stream=stream)
    state_output = state_output.copy_to_host(stream=stream)
    observables_output = observables_output.copy_to_host(stream=stream)
    state_summaries_output = state_summaries_output.copy_to_host(stream=stream)
    observables_summaries_output = observable_summaries_output.copy_to_host(stream=stream)
    cuda.synchronize()

    memcpy_time = perf_counter() - kernel_end_time
    print(f"\nMemcpy time: {memcpy_time:.3f} s")
    print("\nExecution complete.")
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


    print(f"Final state sample (run 0): {state_output[-1, :n_states, 0]}")
    print(f"Final state sample (run -1): {state_output[-1, :n_states, -1]}")

    return state_output, status_codes_output


if __name__ == "__main__":
    run_debug_integration(n_runs=int32(2**23))
