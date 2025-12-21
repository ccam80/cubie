"""All-in-one debug file for Numba lineinfo debugging.

All CUDA device functions are consolidated in this single file to enable
proper line-level debugging with Numba's lineinfo.
"""
# ruff: noqa: E402
from math import ceil, floor, sqrt, fabs
from time import perf_counter
from typing import Optional

import numpy as np
from numba import cuda, int32, float32, float64, bool_
from numba import from_dtype as numba_from_dtype
from cubie.cuda_simsafe import (
    activemask,
    all_sync,
    selp,
    compile_kwargs,
    syncwarp,
    any_sync,
)
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
from cubie.outputhandling.summarymetrics import summary_metrics

# =========================================================================
# CONFIGURATION
# =========================================================================

# -------------------------------------------------------------------------
# Algorithm Configuration
# -------------------------------------------------------------------------

script_start = perf_counter()
#
# algorithm_type = 'dirk'
# algorithm_tableau_name ='l_stable_sdirk_4'
algorithm_type = 'erk'
algorithm_tableau_name = 'tsit5'
# algorithm_type = 'firk'
# algorithm_tableau_name = 'radau'
# algorithm_type = 'rosenbrock'
# algorithm_tableau_name = 'ros3p'

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
constants = {'sigma_': 10.0, 'beta_': 8.0 / 3.0}

# System dimensions
n_states = 3
n_parameters = 1
n_observables = 0
n_drivers = 0
n_counters = 4

driver_input_dict = None

# -------------------------------------------------------------------------
# Time Parameters
# -------------------------------------------------------------------------
duration = precision(1.0)
warmup = precision(0.0)
dt = precision(1e-3) # TODO: should be able to set starting dt for adaptive
# runs
dt_save = precision(0.1)
dt_summarise = precision(0.5)
dt_max = precision(1e3)
dt_min = precision(1e-12)  # TODO: when 1e-15, infinite loop

output_types = ['state', 'mean', 'max', 'rms', 'peaks[3]']

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
algorithm_order = 5
kp = precision(0.7)
ki = precision(-0.4)
kd = precision(0.0)
min_gain = precision(0.1)
max_gain = precision(5.0)
deadband_min = precision(1.0)
deadband_max = precision(1.0)
safety = precision(0.9)

# Tolerances for adaptive step control
atol_value = precision(1e-8)
rtol_value = precision(1e-8)

saved_state_indices = np.arange(n_states, dtype=np.int_)
saved_observable_indices = np.arange(n_observables, dtype=np.int_)
summarised_state_indices = np.arange(n_states, dtype=np.int_)
summarised_observable_indices = np.arange(n_observables, dtype=np.int_)

save_last = False

# Saves per summary (for summary metric aggregation)
saves_per_summary = int(dt_summarise/dt_save)

# Summary save cadence (typed for device code)
#
# This is used by summary metric save functions to scale accumulated values.
# Keeping it as an `int32` and also precomputing the reciprocal avoids
# accidentally triggering fp64 division in device code.
summarise_every = int32(int(dt_summarise / dt_save))
inv_summarise_every = precision(1.0 / int(dt_summarise / dt_save))

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
loop_observable_summary_buffer_memory = 'local'  # 'local' or 'shared'

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
fixed_mode = controller_type == 'fixed'
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


# AUTO-GENERATED DXDT FACTORY
def dxdt_factory(constants, precision):
    sigma_ = precision(constants["sigma_"])
    beta_ = precision(constants["beta_"])

    @cuda.jit(
        # (precision[::1],
        #  precision[::1],
        #  precision[::1],
        #  precision[::1],
        #  precision[::1],
        #  precision),
        device=True,
        inline=True,
    )
    def dxdt(state, parameters, drivers, observables, out, t):
        out[2] = -beta_ * state[2] + state[0] * state[1]
        _cse0 = -state[1]
        out[1] = _cse0 + state[0] * (parameters[0] - state[2])
        out[0] = sigma_ * (-_cse0 - state[0])

    return dxdt


# AUTO-GENERATED OBSERVABLES FACTORY
def observables_factory(constants, precision):
    sigma_ = precision(constants['sigma_'])
    beta_ = precision(constants['beta_'])
    @cuda.jit(
            # (precision[::1],
            #  precision[::1],
            #  precision[::1],
            #  precision[::1],
            #  precision),
            device=True,
            inline=True)
    def get_observables(state, parameters, drivers, observables, t):
        pass

    return get_observables
# AUTO-GENERATED NEUMANN PRECONDITIONER FACTORY
def neumann_preconditioner(constants, precision, beta=1.0, gamma=1.0, order=1):
    n = int32(3)
    gamma = precision(gamma)
    beta = precision(beta)
    order = int32(order)
    beta_inv = precision(1.0 / beta)
    h_eff_factor = precision(gamma * beta_inv)
    sigma_ = precision(constants['sigma_'])
    beta_ = precision(constants['beta_'])
    @cuda.jit(
        # (precision[::1],
        #  precision[::1],
        #  precision[::1],
        #  precision[::1],
        #  precision,
        #  precision,
        #  precision,
        #  precision[::1],
        #  precision[::1],
        #  precision[::1]),
        device=True,
        inline=True)
    def preconditioner(
        state, parameters, drivers, base_state, t, h, a_ij, v, out, jvp
    ):
        # Horner form: S[m] = v + T S[m-1], T = ((gamma*a_ij)/beta) * h * J
        # Accumulator lives in `out`. Uses caller-provided `jvp` for JVP.
        for i in range(n):
            out[i] = v[i]
        h_eff = h * h_eff_factor * a_ij
        for _ in range(order):
            j_00 = -sigma_
            j_01 = sigma_
            j_10 = -a_ij*state[2] + parameters[0] - base_state[2]
            j_11 = precision(-1)
            j_12 = -a_ij*state[0] - base_state[0]
            j_20 = a_ij*state[1] + base_state[1]
            j_21 = a_ij*state[0] + base_state[0]
            j_22 = -beta_
            jvp[0] = j_00*out[0] + j_01*out[1]
            jvp[1] = j_10*out[0] + j_11*out[1] + j_12*out[2]
            jvp[2] = j_20*out[0] + j_21*out[1] + j_22*out[2]
            for i in range(n):
                out[i] = v[i] + h_eff * jvp[i]
        for i in range(n):
            out[i] = beta_inv * out[i]
    return preconditioner

# AUTO-GENERATED NONLINEAR RESIDUAL FACTORY
def stage_residual(constants, precision, beta=1.0, gamma=1.0, order=None):
    beta = precision(beta)
    gamma = precision(gamma)
    sigma_ = precision(constants['sigma_'])
    beta_ = precision(constants['beta_'])
    @cuda.jit(
        # (precision[::1],
        #  precision[::1],
        #  precision[::1],
        #  precision,
        #  precision,
        #  precision,
        #  precision[::1],
        #  precision[::1]),
        device=True,
        inline=True)
    def residual(u, parameters, drivers, t, h, a_ij, base_state, out):
        _cse0 = a_ij*u[1]
        _cse1 = a_ij*u[0] + base_state[0]
        _cse2 = a_ij*u[2] + base_state[2]
        _cse4 = precision(1.00000)*beta
        _cse5 = gamma*h
        _cse3 = _cse0 + base_state[1]
        dx_0 = sigma_*(_cse0 - _cse1 + base_state[1])
        dx_2 = _cse1*_cse3 - _cse2*beta_
        dx_1 = _cse1*(-_cse2 + parameters[0]) - _cse3
        out[0] = _cse4*u[0] - _cse5*dx_0
        out[2] = _cse4*u[2] - _cse5*dx_2
        out[1] = _cse4*u[1] - _cse5*dx_1
    return residual

# AUTO-GENERATED LINEAR OPERATOR FACTORY
def linear_operator(constants, precision, beta=1.0, gamma=1.0, order=None):
    beta = precision(beta)
    gamma = precision(gamma)
    sigma_ = precision(constants['sigma_'])
    beta_ = precision(constants['beta_'])
    @cuda.jit(
        # (precision[::1],
        #  precision[::1],
        #  precision[::1],
        #  precision[::1],
        #  precision,
        #  precision,
        #  precision,
        #  precision[::1],
        #  precision[::1]),
        device=True,
        inline=True)
    def operator_apply(state, parameters, drivers, base_state, t, h, a_ij, v, out):
        m_00 = precision(1.00000)
        m_11 = precision(1.00000)
        m_22 = precision(1.00000)
        j_00 = -sigma_
        j_01 = sigma_
        j_10 = -a_ij*state[2] + parameters[0] - base_state[2]
        j_11 = precision(-1)
        j_12 = -a_ij*state[0] - base_state[0]
        j_20 = a_ij*state[1] + base_state[1]
        j_21 = a_ij*state[0] + base_state[0]
        j_22 = -beta_
        out[0] = -a_ij*gamma*h*(j_00*v[0] + j_01*v[1]) + beta*m_00*v[0]
        out[1] = -a_ij*gamma*h*(j_10*v[0] + j_11*v[1] + j_12*v[2]) + beta*m_11*v[1]
        out[2] = -a_ij*gamma*h*(j_20*v[0] + j_21*v[1] + j_22*v[2]) + beta*m_22*v[2]
    return operator_apply


# AUTO-GENERATED N-STAGE RESIDUAL FACTORY
def n_stage_residual_3(constants, precision, beta=1.0, gamma=1.0, order=None):
    beta = precision(beta)
    gamma = precision(gamma)
    sigma_ = precision(constants['sigma_'])
    beta_ = precision(constants['beta_'])
    @cuda.jit(
        # (precision[::1],
        #  precision[::1],
        #  precision[::1],
        #  precision,
        #  precision,
        #  precision,
        #  precision[::1],
        #  precision[::1]),
        device=True,
        inline=True)
    def residual(u, parameters, drivers, t, h, a_ij, base_state, out):
        a_0_0 = precision(0.196815477223660)
        a_0_1 = precision(-0.0655354258501984)
        a_0_2 = precision(0.0237709743482202)
        a_1_0 = precision(0.394424314739087)
        a_1_1 = precision(0.292073411665228)
        a_1_2 = precision(-0.0415487521259979)
        a_2_0 = precision(0.376403062700467)
        a_2_1 = precision(0.512485826188422)
        a_2_2 = precision(0.111111111111111)
        _cse0 = -base_state[1]
        _cse5 = -parameters[0]
        _cse8 = precision(1.00000)*beta
        _cse9 = gamma*h
        _cse1 = a_0_0*u[1]
        _cse2 = a_0_1*u[4]
        _cse6 = a_0_0*u[2] + a_0_1*u[5] + a_0_2*u[8] + base_state[2]
        _cse4 = a_0_0*u[0] + a_0_1*u[3] + a_0_2*u[6] + base_state[0]
        _cse3 = a_0_2*u[7]
        _cse10 = a_1_0*u[1]
        _cse11 = a_1_1*u[4]
        _cse12 = a_1_2*u[7]
        _cse14 = a_1_0*u[2] + a_1_1*u[5] + a_1_2*u[8] + base_state[2]
        _cse13 = a_1_0*u[0] + a_1_1*u[3] + a_1_2*u[6] + base_state[0]
        _cse16 = a_2_0*u[1]
        _cse17 = a_2_1*u[4]
        _cse19 = a_2_0*u[0] + a_2_1*u[3] + a_2_2*u[6] + base_state[0]
        _cse18 = a_2_2*u[7]
        _cse20 = a_2_0*u[2] + a_2_1*u[5] + a_2_2*u[8] + base_state[2]
        dx_0_0 = sigma_*(-_cse0 + _cse1 + _cse2 + _cse3 - _cse4)
        _cse7 = _cse1 + _cse2 + _cse3 + base_state[1]
        _cse15 = _cse10 + _cse11 + _cse12 + base_state[1]
        dx_1_0 = sigma_*(-_cse0 + _cse10 + _cse11 + _cse12 - _cse13)
        dx_2_0 = sigma_*(-_cse0 + _cse16 + _cse17 + _cse18 - _cse19)
        _cse21 = _cse16 + _cse17 + _cse18 + base_state[1]
        out[0] = _cse8*u[0] - _cse9*dx_0_0
        dx_0_1 = _cse4*(-_cse5 - _cse6) - _cse7
        dx_0_2 = _cse4*_cse7 - _cse6*beta_
        dx_1_1 = _cse13*(-_cse14 - _cse5) - _cse15
        dx_1_2 = _cse13*_cse15 - _cse14*beta_
        out[3] = _cse8*u[3] - _cse9*dx_1_0
        out[6] = _cse8*u[6] - _cse9*dx_2_0
        dx_2_2 = _cse19*_cse21 - _cse20*beta_
        dx_2_1 = _cse19*(-_cse20 - _cse5) - _cse21
        out[1] = _cse8*u[1] - _cse9*dx_0_1
        out[2] = _cse8*u[2] - _cse9*dx_0_2
        out[4] = _cse8*u[4] - _cse9*dx_1_1
        out[5] = _cse8*u[5] - _cse9*dx_1_2
        out[8] = _cse8*u[8] - _cse9*dx_2_2
        out[7] = _cse8*u[7] - _cse9*dx_2_1
    return residual

# AUTO-GENERATED N-STAGE LINEAR OPERATOR FACTORY
def n_stage_linear_operator_3(constants, precision, beta=1.0, gamma=1.0, order=None):
    sigma_ = precision(constants['sigma_'])
    beta_ = precision(constants['beta_'])
    gamma = precision(gamma)
    beta = precision(beta)
    @cuda.jit(
        # (precision[::1],
        #  precision[::1],
        #  precision[::1],
        #  precision[::1],
        #  precision,
        #  precision,
        #  precision,
        #  precision[::1],
        #  precision[::1]),
        device=True,
        inline=True)
    def operator_apply(state, parameters, drivers, base_state, t, h, a_ij, v, out):
        a_0_0 = precision(0.196815477223660)
        a_0_1 = precision(-0.0655354258501984)
        a_0_2 = precision(0.0237709743482202)
        a_1_0 = precision(0.394424314739087)
        a_1_1 = precision(0.292073411665228)
        a_1_2 = precision(-0.0415487521259979)
        a_2_0 = precision(0.376403062700467)
        a_2_1 = precision(0.512485826188422)
        a_2_2 = precision(0.111111111111111)
        j_01_0 = sigma_
        j_11_0 = precision(-1)
        j_01_1 = sigma_
        j_11_1 = precision(-1)
        j_01_2 = sigma_
        j_11_2 = precision(-1)
        _cse5 = -parameters[0]
        _cse9 = -sigma_
        _cse10 = -beta_
        _cse11 = precision(1.00000)*beta
        _cse12 = gamma*h
        _cse1 = a_0_0*state[1]
        _cse2 = a_0_1*state[4]
        _cse6 = a_0_0*state[2] + a_0_1*state[5] + a_0_2*state[8] + base_state[2]
        _cse4 = a_0_0*state[0] + a_0_1*state[3] + a_0_2*state[6] + base_state[0]
        _cse3 = a_0_2*state[7]
        _cse13 = a_1_0*state[1]
        _cse14 = a_1_1*state[4]
        _cse17 = a_1_0*state[2] + a_1_1*state[5] + a_1_2*state[8] + base_state[2]
        _cse16 = a_1_0*state[0] + a_1_1*state[3] + a_1_2*state[6] + base_state[0]
        _cse15 = a_1_2*state[7]
        _cse20 = a_2_0*state[1]
        _cse21 = a_2_1*state[4]
        _cse22 = a_2_2*state[7]
        _cse24 = a_2_0*state[2] + a_2_1*state[5] + a_2_2*state[8] + base_state[2]
        _cse23 = a_2_0*state[0] + a_2_1*state[3] + a_2_2*state[6] + base_state[0]
        j_00_2 = _cse9
        j_00_0 = _cse9
        j_00_1 = _cse9
        j_22_2 = _cse10
        j_22_0 = _cse10
        j_22_1 = _cse10
        _cse7 = -_cse5 - _cse6
        j_12_0 = -_cse4
        j_21_0 = _cse4
        _cse8 = _cse1 + _cse2 + _cse3 + base_state[1]
        _cse18 = -_cse17 - _cse5
        j_21_1 = _cse16
        j_12_1 = -_cse16
        _cse19 = _cse13 + _cse14 + _cse15 + base_state[1]
        _cse26 = _cse20 + _cse21 + _cse22 + base_state[1]
        _cse25 = -_cse24 - _cse5
        j_12_2 = -_cse23
        j_21_2 = _cse23
        jvp_2_0 = j_00_2*v[0] + j_01_2*v[1]
        jvp_0_0 = j_00_0*v[0] + j_01_0*v[1]
        jvp_1_0 = j_00_1*v[0] + j_01_1*v[1]
        j_10_0 = _cse7
        j_20_0 = _cse8
        j_10_1 = _cse18
        j_20_1 = _cse19
        j_20_2 = _cse26
        j_10_2 = _cse25
        out[6] = _cse11*v[6] - _cse12*jvp_2_0
        out[0] = _cse11*v[0] - _cse12*jvp_0_0
        out[3] = _cse11*v[3] - _cse12*jvp_1_0
        jvp_0_1 = j_10_0*v[0] + j_11_0*v[1] + j_12_0*v[2]
        jvp_0_2 = j_20_0*v[0] + j_21_0*v[1] + j_22_0*v[2]
        jvp_1_1 = j_10_1*v[0] + j_11_1*v[1] + j_12_1*v[2]
        jvp_1_2 = j_20_1*v[0] + j_21_1*v[1] + j_22_1*v[2]
        jvp_2_2 = j_20_2*v[0] + j_21_2*v[1] + j_22_2*v[2]
        jvp_2_1 = j_10_2*v[0] + j_11_2*v[1] + j_12_2*v[2]
        out[1] = _cse11*v[1] - _cse12*jvp_0_1
        out[2] = _cse11*v[2] - _cse12*jvp_0_2
        out[4] = _cse11*v[4] - _cse12*jvp_1_1
        out[5] = _cse11*v[5] - _cse12*jvp_1_2
        out[8] = _cse11*v[8] - _cse12*jvp_2_2
        out[7] = _cse11*v[7] - _cse12*jvp_2_1
    return operator_apply

# AUTO-GENERATED N-STAGE NEUMANN PRECONDITIONER FACTORY
def n_stage_neumann_preconditioner_3(constants, precision, beta=1.0, gamma=1.0, order=1):
    sigma_ = precision(constants['sigma_'])
    beta_ = precision(constants['beta_'])
    total_n = int32(9)
    gamma = precision(gamma)
    beta = precision(beta)
    order = int32(order)
    beta_inv = precision(1.0 / beta)
    h_eff_factor = precision(gamma * beta_inv)
    stage_width = int32(3)
    @cuda.jit(
        # (precision[::1],
        #  precision[::1],
        #  precision[::1],
        #  precision[::1],
        #  precision,
        #  precision,
        #  precision,
        #  precision[::1],
        #  precision[::1],
        #  precision[::1]),
        device=True,
        inline=True)
    def preconditioner(state, parameters, drivers, base_state, t, h, a_ij, v, out, jvp):
        for i in range(total_n):
            out[i] = v[i]
        h_eff = h * h_eff_factor
        for _ in range(order):
            a_0_0 = precision(0.196815477223660)
            a_0_1 = precision(-0.0655354258501984)
            a_0_2 = precision(0.0237709743482202)
            a_1_0 = precision(0.394424314739087)
            a_1_1 = precision(0.292073411665228)
            a_1_2 = precision(-0.0415487521259979)
            a_2_0 = precision(0.376403062700467)
            a_2_1 = precision(0.512485826188422)
            a_2_2 = precision(0.111111111111111)
            j_01_0 = sigma_
            j_11_0 = precision(-1)
            j_01_1 = sigma_
            j_11_1 = precision(-1)
            j_01_2 = sigma_
            j_11_2 = precision(-1)
            _cse5 = -parameters[0]
            _cse9 = -sigma_
            _cse10 = -beta_
            _cse1 = a_0_0*state[1]
            _cse2 = a_0_1*state[4]
            _cse6 = a_0_0*state[2] + a_0_1*state[5] + a_0_2*state[8] + base_state[2]
            _cse4 = a_0_0*state[0] + a_0_1*state[3] + a_0_2*state[6] + base_state[0]
            _cse3 = a_0_2*state[7]
            _cse11 = a_1_0*state[1]
            _cse12 = a_1_1*state[4]
            _cse15 = a_1_0*state[2] + a_1_1*state[5] + a_1_2*state[8] + base_state[2]
            _cse14 = a_1_0*state[0] + a_1_1*state[3] + a_1_2*state[6] + base_state[0]
            _cse13 = a_1_2*state[7]
            _cse18 = a_2_0*state[1]
            _cse19 = a_2_1*state[4]
            _cse21 = a_2_0*state[0] + a_2_1*state[3] + a_2_2*state[6] + base_state[0]
            _cse22 = a_2_0*state[2] + a_2_1*state[5] + a_2_2*state[8] + base_state[2]
            _cse20 = a_2_2*state[7]
            j_00_2 = _cse9
            j_00_0 = _cse9
            j_00_1 = _cse9
            j_22_2 = _cse10
            j_22_0 = _cse10
            j_22_1 = _cse10
            _cse7 = -_cse5 - _cse6
            j_12_0 = -_cse4
            j_21_0 = _cse4
            _cse8 = _cse1 + _cse2 + _cse3 + base_state[1]
            _cse16 = -_cse15 - _cse5
            j_21_1 = _cse14
            j_12_1 = -_cse14
            _cse17 = _cse11 + _cse12 + _cse13 + base_state[1]
            j_12_2 = -_cse21
            j_21_2 = _cse21
            _cse23 = -_cse22 - _cse5
            _cse24 = _cse18 + _cse19 + _cse20 + base_state[1]
            jvp_2_0 = j_00_2*v[0] + j_01_2*v[1]
            jvp_0_0 = j_00_0*v[0] + j_01_0*v[1]
            jvp_1_0 = j_00_1*v[0] + j_01_1*v[1]
            j_10_0 = _cse7
            j_20_0 = _cse8
            j_10_1 = _cse16
            j_20_1 = _cse17
            j_10_2 = _cse23
            j_20_2 = _cse24
            jvp[6] = jvp_2_0
            jvp[0] = jvp_0_0
            jvp[3] = jvp_1_0
            jvp_0_1 = j_10_0*v[0] + j_11_0*v[1] + j_12_0*v[2]
            jvp_0_2 = j_20_0*v[0] + j_21_0*v[1] + j_22_0*v[2]
            jvp_1_1 = j_10_1*v[0] + j_11_1*v[1] + j_12_1*v[2]
            jvp_1_2 = j_20_1*v[0] + j_21_1*v[1] + j_22_1*v[2]
            jvp_2_1 = j_10_2*v[0] + j_11_2*v[1] + j_12_2*v[2]
            jvp_2_2 = j_20_2*v[0] + j_21_2*v[1] + j_22_2*v[2]
            jvp[1] = jvp_0_1
            jvp[2] = jvp_0_2
            jvp[4] = jvp_1_1
            jvp[5] = jvp_1_2
            jvp[7] = jvp_2_1
            jvp[8] = jvp_2_2
            for i in range(total_n):
                out[i] = v[i] + h_eff * jvp[i]
        for i in range(total_n):
            out[i] = beta_inv * out[i]
    return preconditioner
def neumann_preconditioner_cached(constants, precision, beta=1.0, gamma=1.0, order=1):
    n = int32(3)
    order = int32(order)
    gamma = precision(gamma)
    beta = precision(beta)
    beta_inv = precision(1.0 / beta)
    h_eff_factor = precision(gamma * beta_inv)
    sigma_ = precision(constants['sigma_'])
    beta_ = precision(constants['beta_'])
    @cuda.jit(
        # (precision[::1],
        #  precision[::1],
        #  precision[::1],
        #  precision[::1],
        #  precision[::1],
        #  precision,
        #  precision,
        #  precision,
        #  precision[::1],
        #  precision[::1],
        #  precision[::1]),
        device=True,
        inline=True)
    def preconditioner(
        state, parameters, drivers, cached_aux, base_state, t, h, a_ij, v, out, jvp
    ):
        for i in range(n):
            out[i] = v[i]
        h_eff = h * h_eff_factor * a_ij
        for _ in range(order):
            j_00 = -sigma_
            j_01 = sigma_
            j_10 = parameters[0] - state[2]
            j_11 = precision(-1)
            j_12 = -state[0]
            j_20 = state[1]
            j_21 = state[0]
            j_22 = -beta_
            jvp[0] = j_00*out[0] + j_01*out[1]
            jvp[1] = j_10*out[0] + j_11*out[1] + j_12*out[2]
            jvp[2] = j_20*out[0] + j_21*out[1] + j_22*out[2]
            for i in range(n):
                out[i] = v[i] + h_eff * jvp[i]
        for i in range(n):
            out[i] = beta_inv * out[i]
    return preconditioner

# AUTO-GENERATED CACHED LINEAR OPERATOR FACTORY
def linear_operator_cached(constants, precision, beta=1.0, gamma=1.0, order=None):
    beta = precision(beta)
    gamma = precision(gamma)
    sigma_ = precision(constants['sigma_'])
    beta_ = precision(constants['beta_'])
    @cuda.jit(
        # (precision[::1],
        #  precision[::1],
        #  precision[::1],
        #  precision[::1],
        #  precision[::1],
        #  precision,
        #  precision,
        #  precision,
        #  precision[::1],
        #  precision[::1]),
        device=True,
        inline=True)
    def operator_apply(
        state, parameters, drivers, cached_aux, base_state, t, h, a_ij, v, out
    ):
        m_00 = precision(1.00000)
        m_11 = precision(1.00000)
        m_22 = precision(1.00000)
        j_00 = -sigma_
        j_01 = sigma_
        j_10 = parameters[0] - state[2]
        j_11 = precision(-1)
        j_12 = -state[0]
        j_20 = state[1]
        j_21 = state[0]
        j_22 = -beta_
        out[0] = -a_ij*gamma*h*(j_00*v[0] + j_01*v[1]) + beta*m_00*v[0]
        out[1] = -a_ij*gamma*h*(j_10*v[0] + j_11*v[1] + j_12*v[2]) + beta*m_11*v[1]
        out[2] = -a_ij*gamma*h*(j_20*v[0] + j_21*v[1] + j_22*v[2]) + beta*m_22*v[2]
    return operator_apply

# AUTO-GENERATED JACOBIAN PREPARATION FACTORY
def prepare_jac(constants, precision):
    sigma_ = precision(constants['sigma_'])
    beta_ = precision(constants['beta_'])
    @cuda.jit(
        # (precision[::1],
        #  precision[::1],
        #  precision[::1],
        #  precision,
        #  precision[::1]),
        device=True,
        inline=True)
    def prepare_jac(state, parameters, drivers, t, cached_aux):
        pass
    return prepare_jac

# AUTO-GENERATED TIME-DERIVATIVE FACTORY
def time_derivative_rhs(constants, precision):
    sigma_ = precision(constants['sigma_'])
    beta_ = precision(constants['beta_'])
    @cuda.jit(
        # (precision[::1],
        #  precision[::1],
        #  precision[::1],
        #  precision[::1],
        #  precision[::1],
        #  precision[::1],
        #  precision),
        device=True,
        inline=True)
    def time_derivative_rhs(
        state, parameters, drivers, driver_dt, observables, out, t
    ):
        time_dx = precision(0)
        time_dy = precision(0)
        time_dz = precision(0)
        out[0] = time_dx
        out[1] = time_dy
        out[2] = time_dz

    return time_derivative_rhs
# =========================================================================
# DRIVER INTERPOLATION INLINE DEVICE FUNCTIONS
# =========================================================================

def driver_function_inline_factory(interpolator):
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
        # (numba_prec, numba_prec[:, :, ::1], numba_prec[::1]),
        device=True,
        inline=True,
        **compile_kwargs
    )
    def driver_function(time, coefficients, out):
        scaled = (time - evaluation_start) * inv_resolution
        scaled_floor = floor(scaled)
        idx = int32(scaled_floor)

        if wrap:
            seg = int32(idx % num_segments)
            tau = prec(scaled - scaled_floor)
            in_range = True
        else:
            in_range = (scaled >= prec(0.0)) and (scaled <= num_segments)
            seg = selp(idx < int32(0), int32(0), idx)
            seg = selp(seg >= num_segments,
                      int32(num_segments - 1), seg)
            tau = scaled - prec(seg)

        # Evaluate polynomials using Horner's rule
        for driver_idx in range(num_drivers):
            acc = zero_value
            for k in range(order, int32(-1), int32(-1)):
                acc = acc * tau + coefficients[seg, driver_idx, k]
            out[driver_idx] = acc if in_range else zero_value

    return driver_function


def driver_derivative_inline_factory(interpolator):
    prec = interpolator.precision
    if interpolator.num_inputs <= 0:
        return None
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
        # (numba_prec, numba_prec[:, :, ::1], numba_prec[::1]),
        device=True,
        inline=True,
        **compile_kwargs
    )
    def driver_derivative(time, coefficients, out):
        scaled = (time - evaluation_start) * inv_resolution
        scaled_floor = floor(scaled)
        idx = int32(scaled_floor)

        if wrap:
            seg = int32(idx % num_segments)
            tau = prec(scaled - scaled_floor)
            in_range = True
        else:
            in_range = (scaled >= precision(0.0)) and (scaled <= num_segments)
            seg = selp(idx < int32(0), int32(0), idx)
            seg = selp(seg >= num_segments,
                      int32(num_segments - int32(1)), seg)
            tau = scaled - precision(seg)

        # Evaluate derivative using Horner's rule on derivative polynomial
        for driver_idx in range(num_drivers):
            acc = zero_value
            for k in range(order, int32(0), int32(-1)):
                acc = acc * tau + prec(k) * (
                    coefficients[seg, driver_idx, k]
                )
            out[driver_idx] = (
                acc * inv_resolution if in_range else zero_value
            )

    return driver_derivative


def neumann_preconditioner_factory(constants, prec, beta, gamma, order):
    n = int32(3)
    order = int32(order)
    beta_inv = prec(1.0 / beta)
    gamma = prec(gamma)
    h_eff_factor = prec(gamma * beta_inv)
    sigma = prec(constants['sigma_'])
    beta_const = prec(constants['beta_'])
    numba_prec = numba_from_dtype(prec)

    @cuda.jit(
            # (numba_prec[::1], numba_prec[::1], numba_prec[::1],
            #    numba_prec[::1], numba_prec, numba_prec, numba_prec,
            #    numba_prec[::1], numba_prec[::1], numba_prec[::1]),
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
    sigma = prec(constants['sigma_'])
    beta_const = prec(constants['beta_'])
    numba_prec = numba_from_dtype(prec)
    beta_val = numba_prec(1.0) * numba_prec(beta)

    @cuda.jit(
            # (numba_prec[::1], numba_prec[::1], numba_prec[::1],
            #    numba_prec, numba_prec, numba_prec, numba_prec[::1],
            #    numba_prec[::1]),
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
    sigma = prec(constants['sigma_'])
    beta_const = prec(constants['beta_'])
    gamma = prec(gamma)
    numba_prec = numba_from_dtype(prec)

    @cuda.jit(
            # (numba_prec[::1], numba_prec[::1], numba_prec[::1],
            #    numba_prec[::1], numba_prec, numba_prec, numba_prec,
            #    numba_prec[::1], numba_prec[::1]),
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
    numba_prec = numba_from_dtype(prec)
    tol_squared = numba_prec(tolerance * tolerance)
    typed_zero = numba_prec(0.0)
    n_arraysize = n
    n_val = int32(n)
    max_iters = int32(max_iters)
    sd_flag = True if correction_type == "steepest_descent" else False
    mr_flag = True if correction_type == "minimal_residual" else False
    preconditioned=True
    @cuda.jit(
        # (numba_prec[::1], numba_prec[::1], numba_prec[::1],
        #  numba_prec[::1], numba_prec, numba_prec, numba_prec,
        #  numba_prec[::1], numba_prec[::1], int32[::1]),
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
        krylov_iters_out,
    ):
        preconditioned_vec = cuda.local.array(n_arraysize, numba_prec)
        temp = cuda.local.array(n_arraysize, numba_prec)

        operator_apply(
            state, parameters, drivers, base_state, t, h, a_ij, x, temp
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

            if denominator != typed_zero:
                alpha = numerator / denominator
            else:
                alpha = typed_zero

            acc = typed_zero
            if not converged:
                for i in range(n_val):
                    x[i] += alpha * preconditioned_vec[i]
                    rhs[i] -= alpha * temp[i]
                    residual_value = rhs[i]
                    acc += residual_value * residual_value
            else:
                for i in range(n_val):
                    residual_value = rhs[i]
                    acc += residual_value * residual_value

            converged = converged or (acc <= tol_squared)

        # Single exit point - status based on converged flag
        final_status = selp(converged, int32(0), int32(4))
        krylov_iters_out[0] = iter_count
        return final_status

        # no cover: end
    return linear_solver


def linear_solver_cached_inline_factory(
        operator_apply, n, preconditioner, tolerance, max_iters, prec,
        correction_type
):
    numba_prec = numba_from_dtype(prec)
    tol_squared = numba_prec(tolerance * tolerance)
    n_arraysize = n
    n_val = int32(n)
    max_iters = int32(max_iters)
    sd_flag = True if correction_type == "steepest_descent" else False
    mr_flag = True if correction_type == "minimal_residual" else False
    preconditioned = preconditioner is not None
    typed_zero_local = numba_prec(0.0)

    @cuda.jit(device=True, inline=True, **compile_kwargs)
    def linear_solver(
        state,
        parameters,
        drivers,
        base_state,
        cached_auxiliaries,
        t,
        h,
        a_ij,
        rhs,
        x,
        shared,
        krylov_iters_out,
    ):
        preconditioned_vec = cuda.local.array(n_arraysize, numba_prec)
        temp = cuda.local.array(n_arraysize, numba_prec)

        operator_apply(
            state,
            parameters,
            drivers,
            base_state,
            t,
            h,
            a_ij,
            x,
            temp,
        )
        acc = typed_zero_local
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
            numerator = typed_zero_local
            denominator = typed_zero_local
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

            alpha = selp(denominator != typed_zero_local,
                         numerator / denominator, typed_zero_local)
            alpha_effective = selp(converged, numba_prec(0.0), alpha)

            acc = typed_zero_local
            for i in range(n_val):
                x[i] += alpha_effective * preconditioned_vec[i]
                rhs[i] -= alpha_effective * temp[i]
                residual_value = rhs[i]
                acc += residual_value * residual_value
            converged = converged or (acc <= tol_squared)

        final_status = selp(converged, int32(0), int32(4))
        krylov_iters_out[0] = iter_count
        return int32(final_status)

    return linear_solver


# =========================================================================
# INLINE NEWTON-KRYLOV SOLVER FACTORY
# =========================================================================

def newton_krylov_inline_factory(residual_fn, linear_solver, n, tolerance,
                                 max_iters, damping, max_backtracks, prec):
    numba_prec = numba_from_dtype(prec)
    n_arraysize = int(n)
    n = int32(n)
    max_iters = int32(max_iters)
    max_backtracks = int32(max_backtracks + 1)
    tol_squared = numba_prec(tolerance * tolerance)
    typed_zero = numba_prec(0.0)
    typed_one = numba_prec(1.0)
    typed_damping = numba_prec(damping)

    @cuda.jit(
            # [
            #     (numba_prec[::1],
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
        delta = cuda.local.array(n_arraysize, numba_prec)
        residual_temp = cuda.local.array(n_arraysize, numba_prec)
        residual = cuda.local.array(n_arraysize, numba_prec)
        stage_base_bt = cuda.local.array(n_arraysize, numba_prec)

        residual_fn(
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
                krylov_iters_local,
            )

            lin_failed = lin_status != int32(0)
            has_error = has_error or lin_failed
            final_status = selp(
                lin_failed, int32(final_status | lin_status), final_status
            )

            total_krylov_iters += selp(active, krylov_iters_local[0], int32(0))

            # Backtracking loop
            for i in range(n):
                stage_base_bt[i] = stage_increment[i]

            found_step = False
            alpha = typed_one

            for _ in range(max_backtracks):
                active_bt = active and (not found_step) and (not converged)
                if not any_sync(mask, active_bt):
                    break

                if active_bt:
                    for i in range(n):
                        stage_increment[i] = stage_base_bt[i] + alpha * delta[i]

                    # residual_temp = cuda.local.array(n_arraysize, numba_prec)
                    residual_fn(
                            stage_increment,
                            parameters,
                            drivers,
                            t,
                            h,
                            a_ij,
                            base_state,
                            residual_temp
                    )

                    norm2_new = typed_zero
                    for i in range(n):
                        residual_value = residual_temp[i]
                        norm2_new += residual_value * residual_value

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
                    backtrack_failed,
                    int32(final_status | int32(1)),
                    final_status
            )

            # Revert state if backtrack failed
            if backtrack_failed:
                for i in range(n):
                    stage_increment[i] = stage_base_bt[i]

        # Max iterations exceeded without convergence
        max_iters_exceeded = (not converged) and (not has_error)
        final_status = selp(
                max_iters_exceeded,
                int32(final_status | int32(2)),
                final_status
        )

        counters[0] = iters_count
        counters[1] = total_krylov_iters

        # Return status without encoding iterations
        return int32(final_status)
    return newton_krylov_solver


# =========================================================================
# INLINE DIRK STEP FACTORY (Generic DIRK with tableau)
# =========================================================================

def dirk_step_inline_factory(
    nonlinear_solver,
    dxdt_fn,
    observables_function,
    driver_function,
    driver_del_t,
    n: int,
    prec,
    tableau,
):
    numba_precision = numba_from_dtype(prec)
    typed_zero = numba_precision(0.0)

    # Extract tableau properties
    n_arraysize = n
    accumulator_length_arraysize = int(max(tableau.stage_count-1, 1) * n)
    solver_scratch_local_size = 2 * n  # Python int for cuda.local.array
    n = int32(n)
    stage_count = int32(tableau.stage_count)

    stages_except_first = stage_count - int32(1)

    # Compile-time toggles
    has_driver_function = driver_function is not None
    has_error = tableau.has_error_estimate
    multistage = stage_count > 1
    first_same_as_last = tableau.first_same_as_last
    can_reuse_accepted_start = tableau.can_reuse_accepted_start

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
        # [
        #     (
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
        #     int32,
        #     int32,
        #     numba_precision[::1],
        #     numba_precision[::1],
        #     int32[::1],
        # )],
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
            stage_increment = cuda.local.array(n_arraysize,
                                               numba_precision)
            for _i in range(n_arraysize):
                stage_increment[_i] = numba_precision(0.0)

        if accumulator_in_shared:
            stage_accumulator = shared[acc_start:acc_end]
        else:
            stage_accumulator = cuda.local.array(
                accumulator_length_arraysize, numba_precision
            )
            for _i in range(accumulator_length):
                stage_accumulator[_i] = numba_precision(0.0)

        if solver_scratch_in_shared:
            solver_scratch = shared[solver_start:solver_end]
        else:
            solver_scratch = cuda.local.array(solver_scratch_local_size,
                                               numba_precision)
            for _i in range(solver_scratch_local_size):
                solver_scratch[_i] = numba_precision(0.0)

        # Alias stage base onto first stage accumulator or allocate locally
        if multistage:
            stage_base = stage_accumulator[:n]
        else:
            if stage_base_in_shared:
                stage_base = shared[:n]
            else:
                stage_base = cuda.local.array(n_arraysize,
                                              numba_precision)
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

        first_step = first_step_flag != int32(0)

        # Only use cache if all threads in warp can - otherwise no gain
        use_cached_rhs = False
        if first_same_as_last and multistage:
            if not first_step:
                mask = activemask()
                all_threads_accepted = all_sync(mask, accepted_flag != int32(0))
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
                solver_status = nonlinear_solver(
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
                status_code = int32(status_code | solver_status)

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
        mask = activemask()
        for prev_idx in range(stages_except_first):

            #DIRK is missing the instruction cache. The unrolled stage loop
            # is instruction dense, taking up most of the instruction space.
            # A block-wide sync hangs indefinitely, as some warps will
            # finish early and never reach it. We sync a warp to minimal
            # effect (it's a wash in the profiler) in case of divergence in
            # big systems.
            syncwarp(mask)
            stage_offset = int32(prev_idx * n)
            stage_idx = prev_idx + int32(1)
            matrix_col = explicit_a_coeffs[prev_idx]

            # Stream previous stage's RHS into accumulators for successors
            for successor_idx in range(stages_except_first):
                coeff = matrix_col[successor_idx + int32(1)]
                row_offset = successor_idx * n
                for idx in range(n):
                    contribution = coeff * stage_rhs[idx]
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
                stage_base[idx] = (stage_accumulator[stage_offset + idx] *
                                   dt_scalar + state[idx])

            diagonal_coeff = diagonal_coeffs[stage_idx]

            if stage_implicit[stage_idx]:
                solver_status = nonlinear_solver(
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
                status_code = int32(status_code | solver_status)

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

        return int32(status_code)

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
        # [
        #     int32(
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
        #     int32,
        #     int32,
        #     numba_precision[::1],
        #     numba_precision[::1],
        #     int32[::1],
        # )],
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

        if stage_rhs_in_shared:
            stage_rhs = shared[:n]
        else:
            stage_rhs = cuda.local.array(n_arraysize, numba_precision)

        current_time = time_scalar
        end_time = current_time + dt_scalar

        if stage_accumulator_in_shared:
            stage_accumulator = shared[n:n + accumulator_length]
        else:
            stage_accumulator = cuda.local.array(
                accumulator_length, dtype=numba_precision
            )

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
                all_threads_accepted = all_sync(mask, accepted_flag != int32(0))
                use_cached_rhs = all_threads_accepted
        else:
            use_cached_rhs = False

        stage_time = current_time
        stage_drivers = proposed_drivers
        if has_driver_function:
            driver_function(
                stage_time,
                driver_coeffs,
                stage_drivers,
            )

        observables_function(
            state,
            parameters,
            stage_drivers,
            proposed_observables,
            stage_time,
        )

        if multistage:
            if use_cached_rhs:
                for idx in range(n):
                    stage_rhs[idx] = stage_cache[idx]
            else:
                dxdt_fn(
                    state,
                    parameters,
                    stage_drivers,
                    proposed_observables,
                    stage_rhs,
                    stage_time,
                )
        else:
            dxdt_fn(
                state,
                parameters,
                stage_drivers,
                proposed_observables,
                stage_rhs,
                stage_time,
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
    numba_precision = numba_from_dtype(prec)
    typed_zero = numba_precision(0.0)

    # Extract tableau properties
    # int32 versions for iterators
    n = int32(n)
    stage_count = int32(tableau.stage_count)
    all_stages_n = int32(stage_count) * int32(n)
    solver_scratch_size = int32(2) * all_stages_n
    stage_driver_stack_size = int32(stage_count) * int32(n_drivers)
    stage_state_size = int32(n)

    # int versions for cuda.local.array sizes
    all_stages_n_ary = int(all_stages_n)
    solver_scratch_ary = int(solver_scratch_size)
    stage_driver_stack_local_size = max(int(stage_driver_stack_size), 1)
    stage_state_size_ary = int(stage_state_size)

    # Compile-time toggles
    has_driver_function = driver_function is not None and n_drivers > 0
    has_error = tableau.has_error_estimate

    stage_rhs_coeffs = tableau.a_flat(numba_precision)
    solution_weights = tableau.typed_vector(tableau.b, numba_precision)
    error_weights = tableau.error_weights(numba_precision)
    if error_weights is None or not has_error:
        error_weights = tuple(typed_zero for _ in range(stage_count))
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


    # Shared memory indices (only used when corresponding flag is True)
    shared_pointer = int32(0)

    # Solver scratch
    solver_scratch_start = shared_pointer
    solver_scratch_end = (
        solver_scratch_start + solver_scratch_size
        if solver_scratch_shared
        else solver_scratch_start
    )
    shared_pointer = solver_scratch_end

    # Stage increment
    stage_increment_start = shared_pointer
    stage_increment_end = (
        stage_increment_start + all_stages_n
        if stage_increment_shared
        else stage_increment_start
    )
    shared_pointer = stage_increment_end

    # Stage driver stack
    stage_driver_stack_start = shared_pointer
    stage_driver_stack_end = (
        stage_driver_stack_start + stage_driver_stack_local_size
        if stage_driver_stack_shared
        else stage_driver_stack_start
    )
    shared_pointer = stage_driver_stack_end

    # Stage state
    stage_state_start = shared_pointer
    stage_state_end = (
        stage_state_start + stage_state_size
        if stage_state_shared
        else stage_state_start
    )
    shared_pointer = stage_state_end

    @cuda.jit(
        # [
        #     int32(
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
        #     int32,
        #     int32,
        #     numba_precision[::1],
        #     numba_precision[::1],
        #     int32[::1],
        # )],
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
            stage_state = cuda.local.array(
                stage_state_size_ary, numba_precision
            )

        if solver_scratch_shared:
            solver_scratch = shared[solver_scratch_start:solver_scratch_end]
        else:
            solver_scratch = cuda.local.array(
                solver_scratch_ary, numba_precision
            )

        if stage_increment_shared:
            stage_increment = shared[stage_increment_start:stage_increment_end]
        else:
            stage_increment = cuda.local.array(
                all_stages_n_ary, numba_precision
            )

        if stage_driver_stack_shared:
            stage_driver_stack = shared[
                stage_driver_stack_start:stage_driver_stack_end
            ]
        else:
            stage_driver_stack = cuda.local.array(
                stage_driver_stack_local_size, numba_precision
            )

        current_time = time_scalar
        end_time = current_time + dt_scalar
        status_code = int32(0)

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

        status_temp = int32(
            nonlinear_solver(
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
        )
        status_code = int32(status_code | status_temp)

        for stage_idx in range(stage_count):

            if has_driver_function:
                stage_base = stage_idx * n_drivers
                for idx in range(n_drivers):
                    proposed_drivers[idx] = stage_driver_stack[stage_base + idx]

            for idx in range(n):
                value = state[idx]
                for contrib_idx in range(stage_count):
                    flat_idx = stage_idx * stage_count + contrib_idx
                    increment_idx = contrib_idx * n
                    coeff = stage_rhs_coeffs[flat_idx]
                    if coeff != typed_zero:
                        value += coeff * stage_increment[increment_idx + idx]
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

        # Kahan summation to reduce floating point errors
        if accumulates_output:
            for idx in range(n):
                solution_acc = typed_zero
                compensation = typed_zero
                for stage_idx in range(stage_count):
                    increment_value = stage_increment[stage_idx * n + idx]
                    weighted = solution_weights[stage_idx] * increment_value
                    term = weighted - compensation
                    temp = solution_acc + term
                    compensation = (temp - solution_acc) - term
                    solution_acc = temp
                proposed_state[idx] = state[idx] + solution_acc

        if has_error and accumulates_error:
            for idx in range(n):
                error_acc = typed_zero
                compensation = typed_zero
                for stage_idx in range(stage_count):
                    increment_value = stage_increment[stage_idx * n + idx]
                    weighted = error_weights[stage_idx] * increment_value
                    term = weighted - compensation
                    temp = error_acc + term
                    compensation = (temp - error_acc) - term
                    error_acc = temp
                error[idx] = error_acc

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
    cached_auxiliary_count,
):
    numba_precision = numba_from_dtype(prec)
    typed_zero = numba_precision(0.0)

    # int32 versions for iterators
    n = int32(n)
    stage_count = int32(tableau.stage_count)
    stages_except_first = stage_count - int32(1)

    # int versions for cuda.local.array sizes
    n_arraysize = int(n)
    stage_count_int = int(stage_count)


    has_driver_function = driver_function is not None and n_drivers > 0
    has_driver_derivative = driver_del_t is not None and n_drivers > 0
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
    cached_auxiliary_count = max(int32(cached_auxiliary_count), int32(1))

    # int versions for cuda.local.array sizes
    stage_store_elements = int(stage_store_elements)
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
        # [
        #     int32(
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
        #     int32,
        #     int32,
        #     numba_precision[::1],
        #     numba_precision[::1],
        #     int32[::1],
        # )],
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
            stage_rhs = cuda.local.array(n_arraysize, numba_precision)

        if stage_store_shared:
            stage_store = shared[stage_store_start:stage_store_end]
        else:
            stage_store = cuda.local.array(
                stage_store_elements, numba_precision
            )

        if cached_auxiliaries_shared:
            cached_auxiliaries = shared[cached_aux_start:cached_aux_end]
        else:
            cached_auxiliaries = cuda.local.array(
                cached_auxiliary_count_int, numba_precision
            )

        current_time = time_scalar
        end_time = current_time + dt_scalar
        final_stage_base = n * (stage_count - int32(1))
        time_derivative = stage_store[final_stage_base:final_stage_base + n]

        inv_dt = numba_precision(1.0) / dt_scalar

        stage_time = current_time + dt_scalar * stage_time_fractions[0]

        prepare_jacobian(
            state, parameters, drivers_buffer, current_time, cached_auxiliaries
        )

        if has_driver_function:
            driver_function(stage_time, driver_coeffs, drivers_buffer)
        else:
            for idx in range(n_drivers):
                drivers_buffer[idx] = numba_precision(0.0)

        if has_driver_derivative:
            driver_del_t(stage_time, driver_coeffs, proposed_drivers)
        else:
            for idx in range(n_drivers):
                proposed_drivers[idx] = numba_precision(0.0)

        for idx in range(n):
            time_derivative[idx] = typed_zero

        time_derivative_rhs(
            state,
            parameters,
            drivers_buffer,
            proposed_drivers,
            observables,
            time_derivative,
            stage_time,
        )

        # Stage 0 slice copies the cached final increment as its guess
        stage_increment = stage_store[:n]

        for idx in range(n):
            stage_increment[idx] = time_derivative[idx]
            proposed_state[idx] = state[idx]
            time_derivative[idx] *= dt_scalar
            if has_error:
                error[idx] = typed_zero

        status_code = int32(0)

        if has_driver_function:
            driver_function(stage_time, driver_coeffs, proposed_drivers)

        observables_function(
            state,
            parameters,
            proposed_drivers,
            proposed_observables,
            stage_time,
        )

        # Stage 0: uses starting values
        dxdt_fn(
            state,
            parameters,
            proposed_drivers,
            proposed_observables,
            stage_rhs,
            stage_time,
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
        krylov_iters_out = cuda.local.array(1, int32)
        krylov_iters_out[0] = int32(0)

        status_temp = int32(
            linear_solver(
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
                krylov_iters_out,
            )
        )
        status_code = int32(status_code | status_temp)
        if counters.shape[0] > 0:
            counters[0] += krylov_iters_out[0]
        if counters.shape[0] > 1:
            counters[1] += krylov_iters_out[0]

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
                driver_function(stage_time, driver_coeffs, drivers_buffer)
            else:
                for idx in range(n_drivers):
                    drivers_buffer[idx] = numba_precision(0.0)

            observables_function(
                stage_slice,
                parameters,
                drivers_buffer,
                proposed_observables,
                stage_time,
            )

            dxdt_fn(
                stage_slice,
                parameters,
                drivers_buffer,
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
                    driver_function(
                        stage_time,
                        driver_coeffs,
                        drivers_buffer,
                    )
                if has_driver_derivative:
                    driver_del_t(stage_time, driver_coeffs, proposed_drivers)
                else:
                    for idx in range(n_drivers):
                        proposed_drivers[idx] = numba_precision(0.0)
                time_derivative_rhs(
                    state,
                    parameters,
                    drivers_buffer,
                    proposed_drivers,
                    observables,
                    time_derivative,
                    stage_time,
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

            krylov_iters_out[0] = int32(0)

            status_temp = int32(
                linear_solver(
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
                    krylov_iters_out,
                )
            )
            status_code = int32(status_code | status_temp)
            if counters.shape[0] > 0:
                counters[0] += krylov_iters_out[0]
            if counters.shape[0] > 1:
                counters[1] += krylov_iters_out[0]

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
# SUMMARY METRICS REGISTRY SIMULATION
# =========================================================================
# SUMMARY METRIC FUNCTIONS (Mean metric with chained pattern)
# =========================================================================
@cuda.jit(
    # [
    #     "float32, float32[::1], int32, int32",
    #     "float64, float64[::1], int32, int32",
    # ],
    device=True,
    inline=True,
    forceinline=True,
    **compile_kwargs
)
def update_mean(
    value,
    buffer,
    offset,
    current_index,
    customisable_variable,
):
    buffer[offset + 0] += value

@cuda.jit(
    # [
    #     "float32[::1], float32[::1], int32, int32",
    #     "float64[::1], float64[::1], int32, int32",
    # ],
    device=True,
    inline=True,
    forceinline=True,
    **compile_kwargs
)
def save_mean(
    buffer,
    buffer_offset,
    output_array,
    output_offset,
    customisable_variable,
):
    output_array[output_offset + 0] = buffer[buffer_offset + 0] * inv_summarise_every
    buffer[buffer_offset + 0] = precision(0.0)

@cuda.jit(
    # [
    #     "float32, float32[::1], int32, int32",
    #     "float64, float64[::1], int32, int32",
    # ],
    device=True,
    inline=True,
    forceinline=True,
    **compile_kwargs
)
def update_max(
    value,
    buffer,
    offset,
    current_index,
    customisable_variable,
):
    update_flag = value > buffer[offset + 0]
    buffer[offset + 0] = selp(update_flag, value, buffer[offset + 0])

@cuda.jit(
    # [
    #     "float32[::1], float32[::1], int32, int32",
    #     "float64[::1], float64[::1], int32, int32",
    # ],
    device=True,
    inline=True,
    forceinline=True,
    **compile_kwargs
)
def save_max(
    buffer,
    buffer_offset,
    output_array,
    output_offset,
    customisable_variable,
):
    output_array[output_offset + 0] = buffer[buffer_offset + 0]
    buffer[buffer_offset + 0] = precision(-1.0e30)

@cuda.jit(
    # [
    #     "float32, float32[::1], int32, int32",
    #     "float64, float64[::1], int32, int32",
    # ],
    device=True,
    inline=True,
    forceinline=True,
    **compile_kwargs
)
def update_min(
    value,
    buffer,
    offset,
    current_index,
    customisable_variable,
):
    update_flag = value < buffer[offset + 0]
    buffer[offset + 0] = selp(update_flag, value, buffer[offset + 0])

@cuda.jit(
    # [
    #     "float32[::1], float32[::1], int32, int32",
    #     "float64[::1], float64[::1], int32, int32",
    # ],
    device=True,
    inline=True,
    forceinline=True,
    **compile_kwargs,
)
def save_min(
    buffer,
    buffer_offset,
    output_array,
    output_offset,
    customisable_variable,
):
    output_array[output_offset + 0] = buffer[buffer_offset + 0]
    buffer[buffer_offset + 0] = precision(1.0e30)

@cuda.jit(
    # [
    #     "float32, float32[::1], int32, int32",
    #     "float64, float64[::1], int32, int32",
    # ],
    device=True,
    inline=True,
    forceinline=True,
    **compile_kwargs
)
def update_rms(
    value,
    buffer,
    offset,
    current_index,
    customisable_variable,
):
    sum_of_squares = buffer[offset + 0]
    if current_index == 0:
        sum_of_squares = precision(0.0)
    sum_of_squares += value * value
    buffer[offset + 0] = sum_of_squares

@cuda.jit(
    # [
    #     "float32[::1], float32[::1], int32, int32",
    #     "float64[::1], float64[::1], int32, int32",
    # ],
    device=True,
    inline=True,
    forceinline=True,
    **compile_kwargs,
)
def save_rms(
    buffer,
    buffer_offset,
    output_array,
    output_offset,
    customisable_variable,
):
    output_array[output_offset + 0] = precision(sqrt(buffer[buffer_offset + 0] * inv_summarise_every))
    buffer[buffer_offset + 0] = precision(0.0)

@cuda.jit(
    # [
    #     "float32, float32[::1], int32, int32",
    #     "float64, float64[::1], int32, int32",
    # ],
    device=True,
    inline=True,
    forceinline=True,
    **compile_kwargs,
)
def update_std(
    value,
    buffer,
    offset,
    current_index,
    customisable_variable,
):
    if current_index == 0:
        buffer[offset + 0] = value  # Store shift value
        buffer[offset + 1] = precision(0.0)    # Reset sum
        buffer[offset + 2] = precision(0.0)    # Reset sum of squares
    
    shifted_value = value - buffer[offset + 0]
    buffer[offset + 1] += shifted_value
    buffer[offset + 2] += shifted_value * shifted_value

@cuda.jit(
    # [
    #     "float32[::1], float32[::1], int32, int32",
    #     "float64[::1], float64[::1], int32, int32",
    # ],
    device=True,
    inline=True,
    forceinline=True,
    **compile_kwargs,
)
def save_std(
    buffer,
    buffer_offset,
    output_array,
    output_offset,
    customisable_variable,
):
    mean_shifted = buffer[buffer_offset + 1] * inv_summarise_every
    mean_of_squares_shifted = buffer[buffer_offset + 2] * inv_summarise_every
    variance = mean_of_squares_shifted - (mean_shifted * mean_shifted)
    output_array[output_offset + 0] = sqrt(variance)
    mean = buffer[buffer_offset + 0] + mean_shifted
    buffer[buffer_offset + 0] = mean
    buffer[buffer_offset + 1] = precision(0.0)
    buffer[buffer_offset + 2] = precision(0.0)

@cuda.jit(
    # [
    #     "float32, float32[::1], int32, int32",
    #     "float64, float64[::1], int32, int32",
    # ],
    device=True,
    inline=True,
    forceinline=True,
    **compile_kwargs,
)
def update_mean_std(
    value,
    buffer,
    offset,
    current_index,
    customisable_variable,
):
    if current_index == 0:
        buffer[offset + 0] = value
        buffer[offset + 1] = precision(0.0)
        buffer[offset + 2] = precision(0.0)
    
    shifted_value = value - buffer[offset + 0]
    buffer[offset + 1] += shifted_value
    buffer[offset + 2] += shifted_value * shifted_value

@cuda.jit(
    # [
    #     "float32[::1], float32[::1], int32, int32",
    #     "float64[::1], float64[::1], int32, int32",
    # ],
    device=True,
    inline=True,
    forceinline=True,
    **compile_kwargs,
)
def save_mean_std(
    buffer,
    buffer_offset,
    output_array,
    output_offset,
    customisable_variable,
):
    shift = buffer[buffer_offset + 0]
    mean_shifted = buffer[buffer_offset + 1] * inv_summarise_every
    mean_of_squares_shifted = buffer[buffer_offset + 2] * inv_summarise_every

    mean = shift + mean_shifted
    variance = mean_of_squares_shifted - (mean_shifted * mean_shifted)
    
    output_array[output_offset + 0] = mean
    output_array[output_offset + 1] = sqrt(variance)
    
    buffer[buffer_offset + 0] = mean
    buffer[buffer_offset + 1] = precision(0.0)
    buffer[buffer_offset + 2] = precision(0.0)

@cuda.jit(
    # [
    #     "float32, float32[::1], int32, int32",
    #     "float64, float64[::1], int32, int32",
    # ],
    device=True,
    inline=True,
    forceinline=True,
    **compile_kwargs,
)
def update_mean_std_rms(
    value,
    buffer,
    offset,
    current_index,
    customisable_variable,
):
    if current_index == 0:
        buffer[offset + 0] = value
        buffer[offset + 1] = precision(0.0)
        buffer[offset + 2] = precision(0.0)
    
    shifted_value = value - buffer[offset + 0]
    buffer[offset + 1] += shifted_value
    buffer[offset + 2] += shifted_value * shifted_value

@cuda.jit(
    # [
    #     "float32[::1], float32[::1], int32, int32",
    #     "float64[::1], float64[::1], int32, int32",
    # ],
    device=True,
    inline=True,
    forceinline=True,
    **compile_kwargs,
)
def save_mean_std_rms(
    buffer,
    buffer_offset,
    output_array,
    output_offset,
    customisable_variable,
):
    shift = buffer[buffer_offset + 0]
    mean_shifted = buffer[buffer_offset + 1] * inv_summarise_every
    mean_of_squares_shifted = buffer[buffer_offset + 2] * inv_summarise_every

    # Mean: shift back to original scale
    mean = shift + mean_shifted
    
    # Variance: using shifted values
    variance = mean_of_squares_shifted - (mean_shifted * mean_shifted)
    std = sqrt(variance)
    
    # RMS: E[X^2] = E[(X-shift)^2] + 2*shift*E[X-shift] + shift^2
    mean_of_squares = (
        mean_of_squares_shifted
        + precision(2.0) * shift * mean_shifted
        + shift * shift
    )
    rms = sqrt(mean_of_squares)
    
    output_array[output_offset + 0] = mean
    output_array[output_offset + 1] = std
    output_array[output_offset + 2] = rms
    
    buffer[buffer_offset + 0] = mean
    buffer[buffer_offset + 1] = precision(0.0)
    buffer[buffer_offset + 2] = precision(0.0)

@cuda.jit(
    # [
    #     "float32, float32[::1], int32, int32",
    #     "float64, float64[::1], int32, int32",
    # ],
    device=True,
    inline=True,
    forceinline=True,
    **compile_kwargs,
)
def update_std_rms(
    value,
    buffer,
    offset,
    current_index,
    customisable_variable,
):
    if current_index == 0:
        buffer[offset + 0] = value
        buffer[offset + 1] = precision(0.0)
        buffer[offset + 2] = precision(0.0)
    
    shifted_value = value - buffer[offset + 0]
    buffer[offset + 1] += shifted_value
    buffer[offset + 2] += shifted_value * shifted_value

@cuda.jit(
    # [
    #     "float32[::1], float32[::1], int32, int32",
    #     "float64[::1], float64[::1], int32, int32",
    # ],
    device=True,
    inline=True,
    forceinline=True,
    **compile_kwargs,
)
def save_std_rms(
    buffer,
    buffer_offset,
    output_array,
    output_offset,
    customisable_variable,
):
    shift = buffer[buffer_offset + 0]
    mean_shifted = buffer[buffer_offset + 1] * inv_summarise_every
    mean_of_squares_shifted = buffer[buffer_offset + 2] * inv_summarise_every

    variance = mean_of_squares_shifted - (mean_shifted * mean_shifted)
    std = sqrt(variance)

    mean_of_squares = (
        mean_of_squares_shifted
        + precision(2.0) * shift * mean_shifted
        + shift * shift
    )
    rms = sqrt(mean_of_squares)
    
    output_array[output_offset + 0] = std
    output_array[output_offset + 1] = rms
    
    mean = shift + mean_shifted
    buffer[buffer_offset + 0] = mean
    buffer[buffer_offset + 1] = precision(0.0)
    buffer[buffer_offset + 2] = precision(0.0)

@cuda.jit(
    # [
    #     "float32, float32[::1], int32, int32",
    #     "float64, float64[::1], int32, int32",
    # ],
    device=True,
    inline=True,
    forceinline=True,
    **compile_kwargs,
)
def update_extrema(
    value,
    buffer,
    offset,
    current_index,
    customisable_variable,
):
    update_max = value > buffer[offset + 0]
    update_min = value < buffer[offset + 1]
    buffer[offset + 0] = selp(update_max, value, buffer[offset + 0])
    buffer[offset + 1] = selp(update_min, value, buffer[offset + 1])

@cuda.jit(
    # [
    #     "float32[::1], float32[::1], int32, int32",
    #     "float64[::1], float64[::1], int32, int32",
    # ],
    device=True,
    inline=True,
    forceinline=True,
    **compile_kwargs,
)
def save_extrema(
    buffer,
    buffer_offset,
    output_array,
    output_offset,
    customisable_variable,
):
    output_array[output_offset + 0] = buffer[buffer_offset + 0]
    output_array[output_offset + 1] = buffer[buffer_offset + 1]
    buffer[buffer_offset + 0] = precision(-1.0e30)
    buffer[buffer_offset + 1] = precision(1.0e30)

@cuda.jit(
    # [
    #     "float32, float32[::1], int32, int32",
    #     "float64, float64[::1], int32, int32",
    # ],
    device=True,
    inline=True,
    forceinline=True,
    **compile_kwargs,
)
def update_peaks(
    value,
    buffer,
    offset,
    current_index,
    customisable_variable,
):
    npeaks = customisable_variable
    prev = buffer[offset + 0]
    prev_prev = buffer[offset + 1]
    peak_counter = int32(buffer[offset + 2])

    is_valid = (
        (current_index >= int32(2))
        and (peak_counter < npeaks)
        and (prev_prev != precision(0.0))
    )
    is_peak = prev > value and prev_prev < prev
    should_record = is_valid and is_peak

    # Clamp index to valid range for predicated access
    safe_idx = peak_counter if peak_counter < npeaks else npeaks - 1

    # Predicated commit for peak recording
    new_counter = precision(int32(buffer[offset + 2]) + 1)
    buffer[offset + 3 + safe_idx] = selp(
        should_record,
        precision(current_index - 1),
        buffer[offset + 3 + safe_idx],
    )
    buffer[offset + 2] = selp(
        should_record, new_counter, buffer[offset + 2]
    )
    buffer[offset + 0] = value
    buffer[offset + 1] = prev

@cuda.jit(
    # [
    #     "float32[::1], float32[::1], int32, int32",
    #     "float64[::1], float64[::1], int32, int32",
    # ],
    device=True,
    inline=True,
    forceinline=True,
    **compile_kwargs,
)
def save_peaks(
    buffer,
    buffer_offset,
    output_array,
    output_offset,
    customisable_variable,
):
    n_peaks = int32(customisable_variable)
    for p in range(n_peaks):
        output_array[output_offset + p] = buffer[buffer_offset + 3 + p]
        buffer[buffer_offset + 3 + p] = precision(0.0)
    buffer[buffer_offset + 2] = precision(0.0)

@cuda.jit(
    # [
    #     "float32, float32[::1], int32, int32",
    #     "float64, float64[::1], int32, int32",
    # ],
    device=True,
    inline=True,
    forceinline=True,
    **compile_kwargs,
)
def update_negative_peaks(
    value,
    buffer,
    offset,
    current_index,
    customisable_variable,
):
    npeaks = customisable_variable
    prev = buffer[offset + 0]
    prev_prev = buffer[offset + 1]
    peak_counter = int32(buffer[offset + 2])

    is_valid = (
        (current_index >= int32(2))
        and (peak_counter < npeaks)
        and (prev_prev != precision(0.0))
    )
    is_peak = prev < value and prev_prev > prev
    should_record = is_valid and is_peak

    # Clamp index to valid range for predicated access
    safe_idx = peak_counter if peak_counter < npeaks else npeaks - 1

    # Predicated commit for peak recording
    new_counter = precision(int32(buffer[offset + 2]) + 1)
    buffer[offset + 3 + safe_idx] = selp(
        should_record,
        precision(current_index - 1),
        buffer[offset + 3 + safe_idx],
    )
    buffer[offset + 2] = selp(
        should_record, new_counter, buffer[offset + 2]
    )
    buffer[offset + 0] = value
    buffer[offset + 1] = prev

@cuda.jit(
    # [
    #     "float32[::1], float32[::1], int32, int32",
    #     "float64[::1], float64[::1], int32, int32",
    # ],
    device=True,
    inline=True,
    forceinline=True,
    **compile_kwargs,
)
def save_negative_peaks(
    buffer,
    buffer_offset,
    output_array,
    output_offset,
    customisable_variable,
):
    n_peaks = int32(customisable_variable)
    for p in range(n_peaks):
        output_array[output_offset + p] = buffer[buffer_offset + 3 + p]
        buffer[buffer_offset + 3 + p] = precision(0.0)
    buffer[buffer_offset + 2] = precision(0.0)

@cuda.jit(
    # [
    #     "float32, float32[::1], int32, int32",
    #     "float64, float64[::1], int32, int32",
    # ],
    device=True,
    inline=True,
    forceinline=True,
    **compile_kwargs,
)
def update_max_magnitude(
    value,
    buffer,
    offset,
    current_index,
    customisable_variable,
):
    abs_value = fabs(value)
    update_flag = abs_value > buffer[offset + 0]
    buffer[offset + 0] = selp(update_flag, abs_value, buffer[offset + 0])

@cuda.jit(
    # [
    #     "float32[::1], float32[::1], int32, int32",
    #     "float64[::1], float64[::1], int32, int32",
    # ],
    device=True,
    inline=True,
    forceinline=True,
    **compile_kwargs,
)
def save_max_magnitude(
    buffer,
    buffer_offset,
    output_array,
    output_offset,
    customisable_variable,
):
    output_array[output_offset + 0] = buffer[buffer_offset + 0]
    buffer[buffer_offset + 0] = precision(0.0)

@cuda.jit(
    # [
    #     "float32, float32[::1], int32, int32",
    #     "float64, float64[::1], int32, int32",
    # ],
    device=True,
    inline=True,
    forceinline=True,
    **compile_kwargs,
)
def update_dxdt_max(
    value,
    buffer,
    offset,
    current_index,
    customisable_variable,
):
    derivative_unscaled = value - buffer[offset + 0]
    update_flag = (derivative_unscaled > buffer[offset + 1]) and (buffer[offset + 0] != precision(0.0))
    buffer[offset + 1] = selp(update_flag, derivative_unscaled, buffer[offset + 1])
    buffer[offset + 0] = value

@cuda.jit(
    # [
    #     "float32[::1], float32[::1], int32, int32",
    #     "float64[::1], float64[::1], int32, int32",
    # ],
    device=True,
    inline=True,
    forceinline=True,
    **compile_kwargs,
)
def save_dxdt_max(
    buffer,
    buffer_offset,
    output_array,
    output_offset,
    customisable_variable,
):
    output_array[output_offset + 0] = buffer[buffer_offset + 1] / precision(dt_save)
    buffer[buffer_offset + 1] = precision(-1.0e30)

@cuda.jit(
    # [
    #     "float32, float32[::1], int32, int32",
    #     "float64, float64[::1], int32, int32",
    # ],
    device=True,
    inline=True,
    forceinline=True,
    **compile_kwargs,
)
def update_dxdt_min(
    value,
    buffer,
    offset,
    current_index,
    customisable_variable,
):
    derivative_unscaled = value - buffer[offset + 0]
    update_flag = (derivative_unscaled < buffer[offset + 1]) and (buffer[offset + 0] != precision(0.0))
    buffer[offset + 1] = selp(update_flag, derivative_unscaled, buffer[offset + 1])
    buffer[offset + 0] = value

@cuda.jit(
    # [
    #     "float32[::1], float32[::1], int32, int32",
    #     "float64[::1], float64[::1], int32, int32",
    # ],
    device=True,
    inline=True,
    forceinline=True,
    **compile_kwargs,
)
def save_dxdt_min(
    buffer,
    buffer_offset,
    output_array,
    output_offset,
    customisable_variable,
):
    output_array[output_offset + 0] = buffer[buffer_offset + 1] / precision(dt_save)
    buffer[buffer_offset + 1] = precision(1.0e30)

@cuda.jit(
    # [
    #     "float32, float32[::1], int32, int32",
    #     "float64, float64[::1], int32, int32",
    # ],
    device=True,
    inline=True,
    forceinline=True,
    **compile_kwargs,
)
def update_dxdt_extrema(
    value,
    buffer,
    offset,
    current_index,
    customisable_variable,
):
    derivative_unscaled = value - buffer[offset + 0]
    update_max = (derivative_unscaled > buffer[offset + 1]) and (buffer[offset + 0] != precision(0.0))
    update_min = (derivative_unscaled < buffer[offset + 2]) and (buffer[offset + 0] != precision(0.0))
    buffer[offset + 1] = selp(update_max, derivative_unscaled, buffer[offset + 1])
    buffer[offset + 2] = selp(update_min, derivative_unscaled, buffer[offset + 2])
    buffer[offset + 0] = value

@cuda.jit(
    # [
    #     "float32[::1], float32[::1], int32, int32",
    #     "float64[::1], float64[::1], int32, int32",
    # ],
    device=True,
    inline=True,
    forceinline=True,
    **compile_kwargs,
)
def save_dxdt_extrema(
    buffer,
    buffer_offset,
    output_array,
    output_offset,
    customisable_variable,
):
    output_array[output_offset + 0] = buffer[buffer_offset + 1] / precision(dt_save)
    output_array[output_offset + 1] = buffer[buffer_offset + 2] / precision(dt_save)
    buffer[buffer_offset + 1] = precision(-1.0e30)
    buffer[buffer_offset + 2] = precision(1.0e30)

@cuda.jit(
    # [
    #     "float32, float32[::1], int32, int32",
    #     "float64, float64[::1], int32, int32",
    # ],
    device=True,
    inline=True,
    forceinline=True,
    **compile_kwargs,
)
def update_d2xdt2_max(
    value,
    buffer,
    offset,
    current_index,
    customisable_variable,
):
    second_derivative_unscaled = value - precision(2.0) * buffer[offset + 0] + buffer[offset + 1]
    update_flag = (second_derivative_unscaled > buffer[offset + 2]) and (buffer[offset + 1] != precision(0.0))
    buffer[offset + 2] = selp(update_flag, second_derivative_unscaled, buffer[offset + 2])
    buffer[offset + 1] = buffer[offset + 0]
    buffer[offset + 0] = value

@cuda.jit(
    # [
    #     "float32[::1], float32[::1], int32, int32",
    #     "float64[::1], float64[::1], int32, int32",
    # ],
    device=True,
    inline=True,
    forceinline=True,
    **compile_kwargs,
)
def save_d2xdt2_max(
    buffer,
    buffer_offset,
    output_array,
    output_offset,
    customisable_variable,
):
    output_array[output_offset + 0] = buffer[buffer_offset + 2] / (precision(dt_save) * precision(dt_save))
    buffer[buffer_offset + 2] = precision(-1.0e30)

@cuda.jit(
    # [
    #     "float32, float32[::1], int32, int32",
    #     "float64, float64[::1], int32, int32",
    # ],
    device=True,
    inline=True,
    forceinline=True,
    **compile_kwargs,
)
def update_d2xdt2_min(
    value,
    buffer,
    offset,
    current_index,
    customisable_variable,
):
    second_derivative_unscaled = value - precision(2.0) * buffer[offset + 0] + buffer[offset + 1]
    update_flag = (second_derivative_unscaled < buffer[offset + 2]) and (buffer[offset + 1] != precision(0.0))
    buffer[offset + 2] = selp(update_flag, second_derivative_unscaled, buffer[offset + 2])
    buffer[offset + 1] = buffer[offset + 0]
    buffer[offset + 0] = value

@cuda.jit(
    # [
    #     "float32[::1], float32[::1], int32, int32",
    #     "float64[::1], float64[::1], int32, int32",
    # ],
    device=True,
    inline=True,
    forceinline=True,
    **compile_kwargs,
)
def save_d2xdt2_min(
    buffer,
    buffer_offset,
    output_array,
    output_offset,
    customisable_variable,
):
    output_array[output_offset + 0] = buffer[buffer_offset + 2] / (precision(dt_save) * precision(dt_save))
    buffer[buffer_offset + 2] = precision(1.0e30)

@cuda.jit(
    # [
    #     "float32, float32[::1], int32, int32",
    #     "float64, float64[::1], int32, int32",
    # ],
    device=True,
    inline=True,
    forceinline=True,
    **compile_kwargs,
)
def update_d2xdt2_extrema(
    value,
    buffer,
    offset,
    current_index,
    customisable_variable,
):
    second_derivative_unscaled = value - precision(2.0) * buffer[offset + 0] + buffer[offset + 1]
    update_max = (second_derivative_unscaled > buffer[offset + 2]) and (buffer[offset + 1] != precision(0.0))
    update_min = (second_derivative_unscaled < buffer[offset + 3]) and (buffer[offset + 1] != precision(0.0))
    buffer[offset + 2] = selp(update_max, second_derivative_unscaled, buffer[offset + 2])
    buffer[offset + 3] = selp(update_min, second_derivative_unscaled, buffer[offset + 3])
    buffer[offset + 1] = buffer[offset + 0]
    buffer[offset + 0] = value

@cuda.jit(
    # [
    #     "float32[::1], float32[::1], int32, int32",
    #     "float64[::1], float64[::1], int32, int32",
    # ],
    device=True,
    inline=True,
    forceinline=True,
    **compile_kwargs,
)
def save_d2xdt2_extrema(
    buffer,
    buffer_offset,
    output_array,
    output_offset,
    customisable_variable,
):
    output_array[output_offset + 0] = buffer[buffer_offset + 2] / (precision(dt_save) * precision(dt_save))
    output_array[output_offset + 1] = buffer[buffer_offset + 3] / (precision(dt_save) * precision(dt_save))
    buffer[buffer_offset + 2] = precision(-1.0e30)
    buffer[buffer_offset + 3] = precision(1.0e30)


# =========================================================================
# INLINE SUMMARY METRIC FUNCTION MAPPING
# =========================================================================

# Mapping from metric names to their inline update and save functions
INLINE_UPDATE_FUNCTIONS = {
    "mean": update_mean,
    "max": update_max,
    "min": update_min,
    "rms": update_rms,
    "std": update_std,
    "mean_std": update_mean_std,
    "mean_std_rms": update_mean_std_rms,
    "std_rms": update_std_rms,
    "extrema": update_extrema,
    "peaks": update_peaks,
    "negative_peaks": update_negative_peaks,
    "max_magnitude": update_max_magnitude,
    "dxdt_max": update_dxdt_max,
    "dxdt_min": update_dxdt_min,
    "dxdt_extrema": update_dxdt_extrema,
    "d2xdt2_max": update_d2xdt2_max,
    "d2xdt2_min": update_d2xdt2_min,
    "d2xdt2_extrema": update_d2xdt2_extrema,
}

INLINE_SAVE_FUNCTIONS = {
    'mean': save_mean,
    'max': save_max,
    'min': save_min,
    'rms': save_rms,
    'std': save_std,
    'mean_std': save_mean_std,
    'mean_std_rms': save_mean_std_rms,
    'std_rms': save_std_rms,
    'extrema': save_extrema,
    'peaks': save_peaks,
    'negative_peaks': save_negative_peaks,
    'max_magnitude': save_max_magnitude,
    'dxdt_max': save_dxdt_max,
    'dxdt_min': save_dxdt_min,
    'dxdt_extrema': save_dxdt_extrema,
    'd2xdt2_max': save_d2xdt2_max,
    'd2xdt2_min': save_d2xdt2_min,
    'd2xdt2_extrema': save_d2xdt2_extrema,
}


def inline_update_functions(summaries_list):
    """Get inline update functions for requested summary metrics.

    Parameters
    ----------
    summaries_list
        Metric names to get update functions for.

    Returns
    -------
    tuple[Callable, ...]
        Inline CUDA device update functions for the requested metrics.

    Raises
    ------
    KeyError
        If a metric in the parsed request is not in INLINE_UPDATE_FUNCTIONS.
    """
    parsed_request = summary_metrics.preprocess_request(list(summaries_list))
    result = []
    for metric in parsed_request:
        if metric not in INLINE_UPDATE_FUNCTIONS:
            raise KeyError(
                f"Metric '{metric}' not found in INLINE_UPDATE_FUNCTIONS. "
                f"Available: {list(INLINE_UPDATE_FUNCTIONS.keys())}"
            )
        result.append(INLINE_UPDATE_FUNCTIONS[metric])
    return tuple(result)


def inline_save_functions(summaries_list):
    """Get inline save functions for requested summary metrics.

    Parameters
    ----------
    summaries_list
        Metric names to get save functions for.

    Returns
    -------
    tuple[Callable, ...]
        Inline CUDA device save functions for the requested metrics.

    Raises
    ------
    KeyError
        If a metric in the parsed request is not in INLINE_SAVE_FUNCTIONS.
    """
    parsed_request = summary_metrics.preprocess_request(list(summaries_list))
    result = []
    for metric in parsed_request:
        if metric not in INLINE_SAVE_FUNCTIONS:
            raise KeyError(
                f"Metric '{metric}' not found in INLINE_SAVE_FUNCTIONS. "
                f"Available: {list(INLINE_SAVE_FUNCTIONS.keys())}"
            )
        result.append(INLINE_SAVE_FUNCTIONS[metric])
    return tuple(result)


@cuda.jit(
    device=True,
    inline=True,
    **compile_kwargs,
)
def do_nothing_update(
    value,
    buffer,
    buffer_offset,
    current_step,
):
    pass


def chain_metrics_update(metric_functions, buffer_offsets,
                         function_params, inner_chain=do_nothing_update):
    n = len(metric_functions)

    @cuda.jit(device=True, inline=True, forceinline=True, **compile_kwargs)
    def leaf(value, buffer, buffer_offset, current_step):
        inner_chain(value, buffer, buffer_offset, current_step)

    fn_chain = leaf
    for i in range(n - 1, -1, -1):
        fn = metric_functions[i]
        boff = int32(buffer_offsets[i])
        param = int32(function_params[i])
        prev = fn_chain

        @cuda.jit(device=True, inline=True, forceinline=True, **compile_kwargs)
        def wrapped(value, buffer, buffer_offset, current_step, fn=fn,
                    boff=boff, param=param, nxt=prev):
            fn(value, buffer, buffer_offset + boff, current_step, param)
            nxt(value, buffer, buffer_offset, current_step)

        fn_chain = wrapped

    return fn_chain

#
# def chain_metrics_update(
#     metric_functions,
#     buffer_offsets,
#     buffer_sizes,
#     function_params,
#     inner_chain=do_nothing_update,
# ):
#     if len(metric_functions) == 0:
#         return do_nothing_update
#
#     current_fn = metric_functions[0]
#     current_offset = buffer_offsets[0]
#     current_size = buffer_sizes[0]
#     current_param = function_params[0]
#
#     remaining_functions = metric_functions[1:]
#     remaining_offsets = buffer_offsets[1:]
#     remaining_sizes = buffer_sizes[1:]
#     remaining_params = function_params[1:]
#
#     @cuda.jit(
#         device=True,
#         inline=True,
#         **compile_kwargs,
#     )
#     def wrapper(
#         value,
#         buffer,
#         current_step,
#     ):
#         inner_chain(value, buffer, current_step)
#         current_fn(
#             value,
#             buffer[current_offset : current_offset + current_size],
#             current_step,
#             current_param,
#         )
#
#     if remaining_functions:
#         return chain_metrics_update(
#             remaining_functions,
#             remaining_offsets,
#             remaining_sizes,
#             remaining_params,
#             wrapper,
#         )
#     else:
#         return wrapper


def update_summary_factory(
    summaries_buffer_height_per_var,
    summarised_state_indices,
    summarised_observable_indices,
    summaries_list,
):
    num_summarised_states = int32(len(summarised_state_indices))
    num_summarised_observables = int32(len(summarised_observable_indices))
    buff_per_var = summaries_buffer_height_per_var
    total_buffer_size = int32(buff_per_var)
    buffer_offsets_list = summary_metrics.buffer_offsets(summaries_list)
    num_metrics = len(buffer_offsets_list)

    summarise_states = (num_summarised_states > 0) and (num_metrics > 0)
    summarise_observables = (num_summarised_observables > 0) and (
        num_metrics > 0
    )

    update_fns = inline_update_functions(summaries_list)
    params_list = summary_metrics.params(summaries_list)
    chain_fn = chain_metrics_update(
        update_fns, buffer_offsets_list, params_list
    )

    @cuda.jit(
        device=True,
        inline=True,
        forceinline=True,
        **compile_kwargs,
    )
    def update_summary_metrics_func(
        current_state,
        current_observables,
        state_summary_buffer,
        observable_summary_buffer,
        current_step,
    ):
        if summarise_states:
            for idx in range(num_summarised_states):
                base_offset = idx * total_buffer_size
                chain_fn(
                    current_state[summarised_state_indices[idx]],
                    state_summary_buffer,
                    base_offset,
                    current_step,
                )

        if summarise_observables:
            for idx in range(num_summarised_observables):
                base_offset = idx * total_buffer_size
                chain_fn(
                    current_observables[summarised_observable_indices[idx]],
                    observable_summary_buffer,
                    base_offset,
                    current_step,
                )

    return update_summary_metrics_func


@cuda.jit(
    device=True,
    inline=True,
    forceinline=True,
    **compile_kwargs,
)
def do_nothing_save(
    buffer,
    buffer_offset,
    output,
    output_offset,
):
    pass


def chain_metrics_save(
    metric_functions,
    buffer_offsets_list,
    output_offsets_list,
    function_params,
    inner_chain=do_nothing_save,
):
    if len(metric_functions) == 0:
        return do_nothing_save
    current_metric_fn = metric_functions[0]
    current_buffer_offset = buffer_offsets_list[0]
    current_output_offset = output_offsets_list[0]
    current_metric_param = function_params[0]

    remaining_metric_fns = metric_functions[1:]
    remaining_buffer_offsets = buffer_offsets_list[1:]
    remaining_output_offsets = output_offsets_list[1:]
    remaining_metric_params = function_params[1:]

    @cuda.jit(
        device=True,
        inline=True,
        **compile_kwargs,
    )
    def wrapper(
        buffer,
        buffer_offset,
        output,
        output_offset,
    ):
        inner_chain(
            buffer,
            buffer_offset,
            output,
            output_offset,
        )
        current_metric_fn(
            buffer,
            buffer_offset + current_buffer_offset,
            output,
            output_offset + current_output_offset,
            current_metric_param,
        )

    if remaining_metric_fns:
        return chain_metrics_save(
            remaining_metric_fns,
            remaining_buffer_offsets,
            remaining_output_offsets,
            remaining_metric_params,
            wrapper,
        )
    else:
        return wrapper


def save_summary_factory(
    summaries_buffer_height_per_var,
    summarised_state_indices,
    summarised_observable_indices,
    summaries_list,
):
    num_summarised_states = int32(len(summarised_state_indices))
    num_summarised_observables = int32(len(summarised_observable_indices))

    save_functions_list = inline_save_functions(summaries_list)

    buff_per_var = summaries_buffer_height_per_var
    total_buffer_size = int32(buff_per_var)
    total_output_size = int32(summary_metrics.summaries_output_height(summaries_list))

    buffer_offsets_list = summary_metrics.buffer_offsets(summaries_list)
    output_offsets_list = summary_metrics.output_offsets(summaries_list)
    params_list = summary_metrics.params(summaries_list)
    num_summary_metrics = len(output_offsets_list)

    summarise_states = (num_summarised_states > 0) and (
        num_summary_metrics > 0
    )
    summarise_observables = (num_summarised_observables > 0) and (
        num_summary_metrics > 0
    )

    summary_metric_chain = chain_metrics_save(
        save_functions_list,
        buffer_offsets_list,
        output_offsets_list,
        params_list,
    )

    @cuda.jit(
        device=True,
        inline=True,
        forceinline=True,
        **compile_kwargs,
    )
    def save_summary_metrics_func(
        buffer_state_summaries,
        buffer_observable_summaries,
        output_state_summaries_window,
        output_observable_summaries_window,
    ):
        if summarise_states:
            for state_index in range(num_summarised_states):
                buffer_base_offset = state_index * total_buffer_size
                output_base_offset = state_index * total_output_size

                summary_metric_chain(
                    buffer_state_summaries,
                    buffer_base_offset,
                    output_state_summaries_window,
                    output_base_offset,
                )

        if summarise_observables:
            for observable_index in range(num_summarised_observables):
                buffer_base_offset = observable_index * total_buffer_size
                output_base_offset = observable_index * total_output_size

                summary_metric_chain(
                    buffer_observable_summaries,
                    buffer_base_offset,
                    output_observable_summaries_window,
                    output_base_offset,
                )

    return save_summary_metrics_func


# -------------------------------------------------------------------------
# Output Configuration
# -------------------------------------------------------------------------

#TODO: summary metrics optimisations added:
# forceinline to individual and chaining functions
# wrap iterators in int32, floats in int32
# summarise_every and inv_summarise_every made constant.

if not output_types:
    summary_types = tuple()
    save_state_bool = False
    save_obs_bool = False
    save_time_bool = False
    save_counters_bool = False
else:
    save_state_bool = "state" in output_types
    save_obs_bool = "observables" in output_types
    save_time_bool = "time" in output_types
    save_counters_bool = "iteration_counters" in output_types
    
    summary_types_list = []
    for output_type in output_types:
        if any(
            (
                output_type.startswith(name)
                for name in summary_metrics.implemented_metrics
            )
        ):
            summary_types_list.append(output_type)
        elif output_type in ["state", "observables", "time", "iteration_counters"]:
            continue
        else:
            print(
                f"Warning: Summary type '{output_type}' is not implemented. "
                f"Ignoring."
            )
    
    summary_types = tuple(summary_types_list)

# Derive summarise booleans
save_summaries = len(summary_types) > 0
summarise_state_bool = save_summaries and n_states > 0
summarise_obs_bool = save_summaries and n_observables > 0
summarise = summarise_obs_bool or summarise_state_bool

# Calculate buffer and output sizes based on enabled metrics
if len(summary_types) > 0:
    summaries_buffer_height_per_var = summary_metrics.summaries_buffer_height(list(summary_types))
    summaries_output_height_per_var = summary_metrics.summaries_output_height(list(summary_types))
else:
    summaries_buffer_height_per_var = 0
    summaries_output_height_per_var = 0

# Generate chained update and save functions for enabled metrics
if len(summary_types) > 0:
    # Generate update chain
    update_summaries_chain = update_summary_factory(
        summaries_buffer_height_per_var,
        summarised_state_indices,
        summarised_observable_indices,
        summary_types,
    )
    
    # Generate save chain
    save_summaries_chain = save_summary_factory(
        summaries_buffer_height_per_var,
        summarised_state_indices,
        summarised_observable_indices,
        summary_types,
    )
else:
    # No metrics enabled, use do_nothing functions
    update_summaries_chain = do_nothing_update
    save_summaries_chain = do_nothing_save


@cuda.jit(device=True, inline=True, forceinline=True, **compile_kwargs)
def update_summaries_inline(
    current_state,
    current_observables,
    state_summary_buffer,
    obs_summary_buffer,
    current_step,
):
    update_summaries_chain(
        current_state,
        current_observables,
        state_summary_buffer,
        obs_summary_buffer,
        current_step,
    )


@cuda.jit(device=True, inline=True, forceinline=True, **compile_kwargs)
def save_summaries_inline(
    buffer_state,
    buffer_obs,
    output_state,
    output_obs,
):
    save_summaries_chain(
        buffer_state,
        buffer_obs,
        output_state,
        output_obs,
    )


# =========================================================================
# STEP CONTROLLER (Fixed and adaptive options)
# =========================================================================

@cuda.jit(device=True, inline=True, **compile_kwargs)
def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))


@cuda.jit(
    # [
    #     (
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
        accept_out[0] = int32(1) if nrm2 <= typed_one else int32(0)
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
    driver_del_t = driver_derivative_inline_factory(interpolator)

else:
    driver_function = None
    driver_del_t = None
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
        driver_del_t,
        n_states,
        precision,
        tableau,
    )
elif algorithm_type == 'firk':
    # Build implicit solver components for FIRK (fully implicit)
    # FIRK requires n-stage coupled system solving
    preconditioner_fn = n_stage_neumann_preconditioner_3(
        constants,
        precision,
        beta=float(beta_solver),
        gamma=float(gamma_solver),
        order=preconditioner_order,
    )
    residual_fn = n_stage_residual_3(
        constants,
        precision,
        beta=float(beta_solver),
        gamma=float(gamma_solver),
        order=preconditioner_order,
    )
    operator_fn = n_stage_linear_operator_3(
        constants,
        precision,
        beta=float(beta_solver),
        gamma=float(gamma_solver),
        order=preconditioner_order,
    )

    linear_solver_fn = linear_solver_inline_factory(
        operator_fn,
        n_states * tableau.stage_count,  # Note: all_stages_n
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

    linear_solver_cached = linear_solver_cached_inline_factory(
        operator_fn,
        n_states,
        preconditioner_fn,
        krylov_tolerance,
        max_linear_iters,
        precision,
        linear_correction_type,
    )

    # Placeholder implementations; codegen overwrites during generation.
    # Placeholder factory; generator injects the real cached auxiliary count.
    def cached_auxiliary_count_factory():
        return int32(1)

    @cuda.jit(device=True, inline=True, **compile_kwargs)
    def prepare_jacobian(
        state, parameters, drivers_buffer, time_value, cached_auxiliaries
    ):
        for idx in range(int(cached_auxiliary_count)):
            cached_auxiliaries[idx] = numba_precision(0.0)

    @cuda.jit(device=True, inline=True, **compile_kwargs)
    def time_derivative_rhs(
        state, parameters, drivers, driver_derivatives, observables, out, t
    ):
        for idx in range(out.shape[0]):
            out[idx] = numba_precision(0.0)

    # Driver time derivative (if drivers present)
    # Note: interpolator is guaranteed non-None when this condition is True,
    # as it's defined at module level in the driver setup block with the same condition
    if n_drivers > 0 and driver_input_dict is not None:
        driver_del_t = driver_derivative_inline_factory(interpolator)
    else:
        driver_del_t = None

    cached_auxiliary_count = int32(
        max(int32(cached_auxiliary_count_factory()), int32(1))
    )

    step_fn = rosenbrock_step_inline_factory(
        linear_solver_cached,
        prepare_jacobian,
        time_derivative_rhs,
        dxdt_fn,
        observables_function,
        driver_function,
        driver_del_t,
        n_states,
        precision,
        tableau,
        cached_auxiliary_count,
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
    all_stages_n = int32(stage_count) * int32(n_states)
    firk_solver_scratch_size = (
        int32(2) * all_stages_n if use_shared_firk_solver_scratch else int32(0)
    )
    firk_stage_increment_size = (
        all_stages_n if use_shared_firk_stage_increment else int32(0)
    )
    firk_stage_driver_stack_size = (
        int32(stage_count) * int32(n_drivers)
        if use_shared_firk_stage_driver_stack
        else int32(0)
    )
    firk_stage_state_size = (
        int32(n_states) if use_shared_firk_stage_state else int32(0)
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
    # - cached_auxiliaries: cached_auxiliary_count (placeholder)
    rosenbrock_stage_rhs_size = (
        n_states if use_shared_rosenbrock_stage_rhs else 0
    )
    rosenbrock_stage_store_size = (
        stage_count * n_states if use_shared_rosenbrock_stage_store else 0
    )
    rosenbrock_cached_auxiliaries_size = (
        int(cached_auxiliary_count)
        if use_shared_rosenbrock_cached_auxiliaries
        else 0
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
        2 * stage_count * n_states  # solver_scratch
        + stage_count * n_states  # stage_increment
        + stage_count * n_drivers  # stage_driver_stack
        + n_states  # stage_state
    )
elif algorithm_type == "rosenbrock":
    scratch_size = (rosenbrock_scratch_size if use_shared_loop_scratch
                    else int32(0))
    # Rosenbrock local scratch: sum of all buffer sizes when not in shared
    local_scratch_size = (
        n_states +           # stage_rhs
        stage_count * n_states +  # stage_store
        max(int(cached_auxiliary_count), 1)
        # cached_auxiliaries (minimum size 1 to avoid zero-size array)
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
n_arraysize = int(n_states)
n_params = int(n_parameters)
local_scratch_size = int(local_scratch_size)

@cuda.jit(
    # [
    #     (
    #         numba_precision[::1],
    #         numba_precision[::1],
    #         numba_precision[:, :, ::1],
    #         numba_precision[::1],
    #         numba_precision[::1],
    #         numba_precision[:, ::1],
    #         numba_precision[:, ::1],
    #         numba_precision[:, ::1],
    #         numba_precision[:, ::1],
    #         numba_precision[:, ::1],
    #         float64,
    #         float64,
    #         float64,
    #         int32,
    #     )
    # ],
    device=True,
    inline=True,
    **compile_kwargs,
)
def loop_fn(initial_states, parameters, driver_coefficients, shared_scratch,
            persistent_local, state_output, observables_output,
            state_summaries_output, observable_summaries_output,
            iteration_counters_output, duration, settling_time, t0):
    t = float64(t0)
    t_prec = numba_precision(t)
    t_end = numba_precision(settling_time + t0 + duration)

    stagnant_counts = int32(0)

    shared_scratch[:] = numba_precision(0.0)

    # Allocate buffers based on memory location configuration
    # Each buffer uses shared memory if its flag is True, otherwise local
    if use_shared_loop_state:
        state_buffer = shared_scratch[state_shared_start:state_shared_end]
    else:
        state_buffer = cuda.local.array(n_arraysize, numba_precision)

    if use_shared_loop_state_proposal:
        state_proposal_buffer = shared_scratch[proposed_state_start:
                                               proposed_state_end]
    else:
        state_proposal_buffer = cuda.local.array(
            n_arraysize, numba_precision
        )

    if use_shared_loop_observables:
        observables_buffer = shared_scratch[obs_start:obs_end]
    else:
        observables_buffer = cuda.local.array(
            obs_nonzero, numba_precision
        )

    if use_shared_loop_observables_proposal:
        observables_proposal_buffer = shared_scratch[proposed_obs_start:
                                                     proposed_obs_end]
    else:
        observables_proposal_buffer = cuda.local.array(
            obs_nonzero, numba_precision
        )

    if use_shared_loop_parameters:
        parameters_buffer = shared_scratch[params_start:params_end]
    else:
        parameters_buffer = cuda.local.array(
            n_parameters, numba_precision
        )

    if use_shared_loop_drivers:
        drivers_buffer = shared_scratch[drivers_start:drivers_end]
    else:
        drivers_buffer = cuda.local.array(drv_nonzero, numba_precision)

    if use_shared_loop_drivers_proposal:
        drivers_proposal_buffer = shared_scratch[proposed_drivers_start:
                                                 proposed_drivers_end]
    else:
        drivers_proposal_buffer = cuda.local.array(
            drv_nonzero, numba_precision
        )

    if use_shared_loop_state_summary:
        state_summary_buffer = shared_scratch[state_summ_start:state_summ_end]
    else:
        state_summary_buffer = cuda.local.array(
            n_arraysize, numba_precision
        )

    if use_shared_loop_observable_summary:
        observable_summary_buffer = shared_scratch[obs_summ_start:obs_summ_end]
    else:
        observable_summary_buffer = cuda.local.array(
            obs_nonzero, numba_precision
        )

    if use_shared_loop_scratch:
        remaining_shared_scratch = shared_scratch[scratch_start:scratch_end]
    else:
        # Local scratch sized for the algorithm (computed at module level)
        remaining_shared_scratch = cuda.local.array(
            local_scratch_size, numba_precision
        )

    if use_shared_loop_counters:
        counters_since_save = shared_scratch[counters_start:counters_end]
    else:
        counters_since_save = cuda.local.array(
            ncnt_nonzero, simsafe_int32
        )

    if use_shared_loop_error:
        error = shared_scratch[error_start:error_end]
    else:
        error = cuda.local.array(n_arraysize, numba_precision)
        for _i in range(n_arraysize):
            error[_i] = precision(0.0)

    proposed_counters = cuda.local.array(2, dtype=simsafe_int32)
    dt = persistent_local[local_dt_slice]
    accept_step = persistent_local[local_accept_slice].view(simsafe_int32)

    controller_temp = persistent_local[local_controller_slice]
    step_persistent_local = persistent_local[local_step_slice]

    first_step_flag = True
    prev_step_accepted_flag = True

    # ----------------------------------------------------------------------- #
    #                       Seed t=0 values                                   #
    # ----------------------------------------------------------------------- #
    for k in range(n_states):
        state_buffer[k] = initial_states[k]
    for k in range(n_parameters):
        parameters_buffer[k] = parameters[k]

    # Seed initial observables from initial state.
    if driver_function is not None and n_drivers > int32(0):
        driver_function(
                t_prec,
                driver_coefficients,
                drivers_buffer,
        )
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
            # Save initial summary state (typically zeros before any updates)
            save_summaries_inline(state_summary_buffer,
                                  observable_summary_buffer,
                                  state_summaries_output[
                                      summary_idx * summarise_state_bool, :
                                  ],
                                  observable_summaries_output[
                                      summary_idx * summarise_obs_bool, :
                                  ])
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
    while True:
        # Exit as soon as we've saved the final step
        finished = bool_(next_save > t_end)
        if save_last:
            # If last save requested, predicated commit dt, finished,
            # do_save
            at_last_save = finished and t_prec < t_end
            finished = selp(at_last_save, False, True)
            dt[0] = selp(at_last_save, numba_precision(t_end - t),
                         dt_raw)

        # also exit loop if min step size limit hit - things are bad
        # Similarly, if time doesn't change after we add a step, exit
        finished = finished or bool_(status & int32(0x8)) or bool_(
                status * int32(0x40))

        if all_sync(mask, finished):
            return status

        if not finished:
            do_save = bool_((t_prec + dt_raw) >= next_save)
            dt_eff = selp(do_save, next_save - t_prec, dt_raw)

            # Fixed mode auto-accepts all steps; adaptive uses controller

            step_status = int32(
                step_fn(
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
            )
            first_step_flag = False

            # Iterations now come from counters, not encoded in status
            niters = proposed_counters[0]
            status = int32(status | step_status)

            # Adjust dt if step rejected - auto-accepts if fixed-step
            if not fixed_mode:
                controller_status = step_controller_fn(
                    dt,
                    state_proposal_buffer,
                    state_buffer,
                    error,
                    niters,
                    accept_step,
                    controller_temp,
                )

                status = int32(status | controller_status)
                accept = bool_(accept_step[0] != int32(0))

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

            t_proposal = t + float64(dt_eff)

            if t_proposal == t:
                stagnant_counts += int32(1)
            else:
                stagnant_counts = int32(0)

            stagnant = bool_(stagnant_counts >= int32(2))
            status = selp(
                    stagnant,
                    int32(status | int32(0x40)),
                    status
            )

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
                int32(1),
                int32(0),
            )

            # Predicated update of next_save; update if save is accepted.
            do_save = bool_(accept and do_save)
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

                    if (save_idx % saves_per_summary == int32(0)):
                        save_summaries_inline(
                            state_summary_buffer,
                            observable_summary_buffer,
                            state_summaries_output[
                                summary_idx * summarise_state_bool, :
                            ],
                            observable_summaries_output[
                                summary_idx * summarise_obs_bool, :
                            ],
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
    1 if (shared_elems_per_run % 2 == 0 and f32_per_element == 1 and
          shared_elems_per_run > 0) else 0
)
run_stride_f32 = int32(
            (f32_per_element * shared_elems_per_run + f32_pad_perrun)
        )
numba_prec = numba_from_dtype(precision)

@cuda.jit(
        # [(
        #         numba_prec[:,::1],
        #         numba_prec[:, ::1],
        #         numba_prec[:, :, ::1],
        #         numba_prec[:, :, ::1],
        #         numba_prec[:, :, ::1],
        #         numba_prec[:, :, ::1],
        #         numba_prec[:, :, ::1],
        #         int32[:, :, ::1],
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

    tx = int32(cuda.threadIdx.x)
    block_index = int32(cuda.blockIdx.x)
    runs_per_block = int32(cuda.blockDim.x)
    run_index = int32(runs_per_block * block_index + tx)
    if run_index >= n_runs_k:
        return None

    shared_memory = cuda.shared.array(0, dtype=float32)
    local_scratch = cuda.local.array(
        local_elements_per_run, dtype=float32
    )
    c_coefficients = cuda.const.array_like(d_coefficients)

    run_idx_low = int32(tx * run_stride_f32)
    run_idx_high = int32(run_idx_low + f32_per_element * shared_elems_per_run
    )
    rx_shared_memory = shared_memory[run_idx_low:run_idx_high].view(
        simsafe_precision)

    rx_inits = inits[run_index, :]
    rx_params = params[run_index, :]
    rx_state = state_output[:, :, run_index]
    rx_observables = observables_output[:, :, run_index]
    rx_state_summaries = state_summaries_output[:, :, run_index]
    rx_observables_summaries = observables_summaries_output[:, :, run_index]
    rx_iteration_counters = iteration_counters_output[:, :, run_index]

    status = loop_fn(rx_inits, rx_params, c_coefficients, rx_shared_memory,
                     local_scratch, rx_state, rx_observables,
                     rx_state_summaries, rx_observables_summaries,
                     rx_iteration_counters, duration_k, warmup_k, t0_k)

    status_codes_output[run_index] = int32(status)


# =========================================================================
# MAIN EXECUTION
# =========================================================================

def run_debug_integration(n_runs=2**23, rho_min=0.0, rho_max=21.0,
                          start_time=0.0):
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
    observable_summaries_output = cuda.device_array(
        obs_summ_shape, dtype=precision, strides=obs_summ_strides
    )

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
    print("  Max registers: " +
    f"{tuple(reg for reg in integration_kernel.get_regs_per_thread().values())}"
    )

    stream = cuda.stream()
    print("\nLaunching kernel...")
    kernel_launch_time = perf_counter() if start_time == 0.0 else start_time
    integration_kernel[blocks_per_grid, current_blocksize, stream,
                       dynamic_sharedmem](
        d_inits, d_params, d_driver_coefficients, state_output,
        observables_output, state_summaries_output,
        observable_summaries_output, iteration_counters_output,
        status_codes_output, duration, warmup, precision(0.0), n_runs)

    kernel_end_time = perf_counter()
    # Mapped arrays provide direct host access after synchronization
    # No explicit copy_to_host required
    # print(f"\nKernel Execution time: {kernel_end_time - kernel_launch_time}")
    status_codes_output = status_codes_output.copy_to_host(stream=stream)
    state_output = state_output.copy_to_host(stream=stream)
    observables_output = observables_output.copy_to_host(stream=stream)
    state_summaries_output = state_summaries_output.copy_to_host(stream=stream)
    observables_summaries_output = observable_summaries_output.copy_to_host(stream=stream)
    stream.synchronize()

    memcpy_time = perf_counter() - kernel_end_time
    wall_clock_time = perf_counter() - kernel_launch_time
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
    print(f"Intial state sample (run -1): {state_output[0, :n_states, -1]}")
    print(f"Final state sample (run 0): {state_output[-1, :n_states, 0]}")
    print(f"Final state sample (run -1): {state_output[-1, :n_states, -1]}")
    print("\n\n\n")
    print("\n" + "=" * 70 + "\n")
    print("Type dump")
    print("\n" + "=" * 70 + "\n")
    # print(f"{integration_kernel.inspect_types()}")
    # print(f"{loop_fn.inspect_types()}")
    # print(f"{step_fn.inspect_types()}")
    # print(f"{save_state_inline.inspect_types()}")
    # print(f"{update_summaries_inline.inspect_types()}")
    # print(f"{save_summaries_inline.inspect_types()}")
    # print(f"{linear_solver_fn.inspect_types()}")
    # print(f"{newton_solver_fn.inspect_types()}")

    return state_output, status_codes_output, wall_clock_time


if __name__ == "__main__":
    _, _ , wall1 = run_debug_integration(n_runs=int32(2**23),
                                         start_time=script_start)
    _, _, wall2 = run_debug_integration(n_runs=int32(2**23))
    print(f"Wall clock time 1: {wall1*1000:.3f} ms")
    print(f"Wall clock time 2: {wall2*1000:.3f} ms")
    print(f"Implied Compile time: {(wall1 - wall2) * 1000:.3f}")
