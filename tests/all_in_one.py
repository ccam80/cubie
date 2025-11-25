"""All-in-one debug file for tracing Numba CUDA execution.

This file consolidates all CUDA device functions into a single location
to work around Numba's lineinfo limitations when debugging across
multiple source files.

Usage:
    1. Run this file once to trigger code generation
    2. Copy generated code from generated/<system_name>.py
    3. Paste into the GENERATED CODE PLACEHOLDER section
    4. Run again with all device functions visible in one file

Configuration:
    Edit the settings at the top of this file to adjust tolerances,
    step sizes, and integration parameters.

Note:
    Imports are placed near their usage (violating PEP8 E402) to
    improve readability and make this file easier to navigate during
    debugging sessions. This is intentional for this debug utility.
"""
# ruff: noqa: E402

import numpy as np

# =========================================================================
# CONFIGURATION SECTION - Edit these values for debugging
# =========================================================================

# Precision for all calculations
precision = np.float32

# Lorenz system parameters
LORENZ_SIGMA = 10.0  # σ parameter
LORENZ_RHO = 21.0    # ρ parameter (bifurcation around 24.74)
LORENZ_BETA = 8.0 / 3.0  # β parameter

# Initial conditions
INITIAL_X = 1.0
INITIAL_Y = 0.0
INITIAL_Z = 0.0

# Solver settings dictionary (matches conftest.py patterns)
solver_settings = {
    # Algorithm selection - use sdirk_2_2 which has embedded error estimate
    # for adaptive stepping with PID controller
    "algorithm": "sdirk_2_2",

    # Time parameters
    "duration": precision(1.0),
    "warmup": precision(0.0),

    # Step size parameters
    "dt": precision(0.01),
    "dt_min": precision(1e-7),
    "dt_max": precision(1.0),
    "dt_save": precision(0.1),
    "dt_summarise": precision(0.2),

    # Tolerance parameters
    "atol": precision(1e-6),
    "rtol": precision(1e-6),

    # Output configuration
    "saved_state_indices": [0, 1, 2],
    "saved_observable_indices": [],
    "summarised_state_indices": [0, 1, 2],
    "summarised_observable_indices": [],
    "output_types": ["state", "mean", "time", "iteration_counters"],

    # Controller parameters (PID)
    "step_controller": "pid",
    "kp": precision(1 / 18),
    "ki": precision(1 / 9),
    "kd": precision(1 / 18),
    "min_gain": precision(0.2),
    "max_gain": precision(2.0),
    "deadband_min": precision(1.0),
    "deadband_max": precision(1.2),

    # Newton-Krylov solver parameters
    "krylov_tolerance": precision(1e-6),
    "correction_type": "minimal_residual",
    "newton_tolerance": precision(1e-6),
    "preconditioner_order": 2,
    "max_linear_iters": 500,
    "max_newton_iters": 500,
    "newton_damping": precision(0.85),
    "newton_max_backtracks": 25,

    # CUDA parameters
    "blocksize": 32,
    "stream": 0,
    "profileCUDA": False,
}

# =========================================================================
# LORENZ SYSTEM DEFINITION
# =========================================================================

# Lorenz attractor equations:
#   dx/dt = σ(y - x)
#   dy/dt = x(ρ - z) - y
#   dz/dt = xy - βz

LORENZ_EQUATIONS = [
    "dx = sigma * (y - x)",
    "dy = x * (rho - z) - y",
    "dz = x * y - beta * z",
]

LORENZ_STATES = {"x": INITIAL_X, "y": INITIAL_Y, "z": INITIAL_Z}
LORENZ_PARAMETERS = {"rho": LORENZ_RHO}  # ρ as runtime parameter
LORENZ_CONSTANTS = {"sigma": LORENZ_SIGMA, "beta": LORENZ_BETA}
LORENZ_DRIVERS = []  # No external drivers
LORENZ_OBSERVABLES = []  # No observables beyond state


def build_lorenz_system(prec):
    """Build the Lorenz attractor ODE system.

    Parameters
    ----------
    prec : numpy dtype
        Floating-point precision for the system.

    Returns
    -------
    SymbolicODE
        Compiled ODE system ready for integration.
    """
    from cubie import create_ODE_system

    return create_ODE_system(
        dxdt=LORENZ_EQUATIONS,
        states=LORENZ_STATES,
        parameters=LORENZ_PARAMETERS,
        constants=LORENZ_CONSTANTS,
        drivers=LORENZ_DRIVERS,
        observables=LORENZ_OBSERVABLES,
        precision=prec,
        name="lorenz_attractor",
        strict=True,
    )


# =========================================================================
# UTILITY DEVICE FUNCTIONS (Inlined from src/cubie/_utils.py)
# =========================================================================

from numba import cuda, int32, from_dtype
from cubie.cuda_simsafe import compile_kwargs, selp


def clamp_factory_inline(prec):
    """Create a clamping device function.

    Parameters
    ----------
    prec : numpy dtype
        Floating-point precision.

    Returns
    -------
    callable
        CUDA device function: clamp(value, minimum, maximum) -> clamped
    """
    numba_precision = from_dtype(prec)

    @cuda.jit(
        numba_precision(numba_precision, numba_precision, numba_precision),
        device=True,
        inline=True,
        **compile_kwargs,
    )
    def clamp(value, minimum, maximum):
        clamped_high = selp(value > maximum, maximum, value)
        clamped = selp(clamped_high < minimum, minimum, clamped_high)
        return clamped

    return clamp


# =========================================================================
# GENERATED CODE PLACEHOLDER
# =========================================================================
#
# After running this file once, copy the generated code from:
#   generated/lorenz_attractor.py
#
# Paste the following factory functions here:
#   1. dxdt_factory(constants, numba_precision) -> dxdt device function
#   2. observables_factory(constants, numba_precision) -> observables device fn
#   3. linear_operator_factory(...) -> JVP-based operator
#   4. neumann_preconditioner_factory(...) -> polynomial preconditioner
#   5. stage_residual_factory(...) -> nonlinear residual function
#
# Example (replace with actual generated code):
# --------------------------------------------------------------------
# def dxdt_factory(constants, numba_precision):
#     sigma = constants['sigma']
#     beta = constants['beta']
#
#     @cuda.jit(device=True, inline=True, **compile_kwargs)
#     def dxdt(state, parameters, drivers, observables, dxdt_out, t):
#         x, y, z = state[0], state[1], state[2]
#         rho = parameters[0]
#         dxdt_out[0] = sigma * (y - x)
#         dxdt_out[1] = x * (rho - z) - y
#         dxdt_out[2] = x * y - beta * z
#     return dxdt
# --------------------------------------------------------------------
#
# [INSERT GENERATED CODE BELOW THIS LINE]
#

# Placeholder flag - set to True after pasting generated code
GENERATED_CODE_AVAILABLE = False

# =========================================================================
# END GENERATED CODE PLACEHOLDER
# =========================================================================

# =========================================================================
# MATRIX-FREE LINEAR SOLVER (Inlined from matrix_free_solvers/linear_solver.py)
# =========================================================================

from cubie.cuda_simsafe import activemask, all_sync


def linear_solver_factory_inline(
    operator_apply,
    n,
    preconditioner=None,
    correction_type="minimal_residual",
    tolerance=1e-6,
    max_iters=100,
    prec=np.float64,
):
    """Create a CUDA device function implementing steepest-descent or MR.

    Parameters
    ----------
    operator_apply
        Callback that overwrites its output vector with F @ v.
    n
        Length of the residual and search-direction vectors.
    preconditioner
        Approximate inverse preconditioner. If None, identity is used.
    correction_type
        "steepest_descent" or "minimal_residual".
    tolerance
        Target on the squared residual norm for convergence.
    max_iters
        Maximum number of iterations permitted.
    prec
        Floating-point precision.

    Returns
    -------
    callable
        CUDA device function returning 0 on convergence, 4 on limit.
    """
    sd_flag = 1 if correction_type == "steepest_descent" else 0
    mr_flag = 1 if correction_type == "minimal_residual" else 0
    preconditioned = 1 if preconditioner is not None else 0

    numba_precision = from_dtype(prec)
    typed_zero = numba_precision(0.0)
    tol_squared = tolerance * tolerance

    @cuda.jit(
        [
            (numba_precision[::1],
             numba_precision[::1],
             numba_precision[::1],
             numba_precision[::1],
             numba_precision,
             numba_precision,
             numba_precision,
             numba_precision[::1],
             numba_precision[::1],
            )
        ],
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
    ):
        preconditioned_vec = cuda.local.array(n, numba_precision)
        temp = cuda.local.array(n, numba_precision)

        operator_apply(
            state, parameters, drivers, base_state, t, h, a_ij, x, temp
        )
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
            if preconditioned:
                preconditioner(
                    state, parameters, drivers, base_state,
                    t, h, a_ij, rhs, preconditioned_vec, temp,
                )
            else:
                for i in range(n):
                    preconditioned_vec[i] = rhs[i]

            operator_apply(
                state, parameters, drivers, base_state,
                t, h, a_ij, preconditioned_vec, temp,
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
            alpha_effective = selp(converged, numba_precision(0.0), alpha)

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
# NEWTON-KRYLOV SOLVER (Inlined from matrix_free_solvers/newton_krylov.py)
# =========================================================================

def newton_krylov_solver_factory_inline(
    residual_function,
    linear_solver,
    n,
    tolerance,
    max_iters,
    damping=0.5,
    max_backtracks=8,
    prec=np.float32,
):
    """Create a damped Newton-Krylov solver device function.

    Parameters
    ----------
    residual_function
        Matrix-free residual evaluator.
    linear_solver
        Matrix-free linear solver from linear_solver_factory.
    n
        Size of the flattened residual and state vectors.
    tolerance
        Residual norm threshold for convergence.
    max_iters
        Maximum number of Newton iterations.
    damping
        Step shrink factor for backtracking.
    max_backtracks
        Maximum damping attempts per Newton step.
    prec
        Floating-point precision.

    Returns
    -------
    callable
        CUDA device function implementing damped Newton-Krylov.
    """
    numba_precision = from_dtype(prec)
    tol_squared = numba_precision(tolerance * tolerance)
    typed_zero = numba_precision(0.0)
    typed_one = numba_precision(1.0)
    typed_damping = numba_precision(damping)
    status_active = int32(-1)

    @cuda.jit(
        [(numba_precision[::1],
          numba_precision[::1],
          numba_precision[::1],
          numba_precision,
          numba_precision,
          numba_precision,
          numba_precision[::1],
          numba_precision[::1],
          int32[::1])],
        device=True,
        inline=True,
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
        residual = shared_scratch[n: 2 * n]

        residual_function(
            stage_increment, parameters, drivers,
            t, h, a_ij, base_state, residual,
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
        total_krylov_iters = int32(0)
        mask = activemask()
        for _ in range(max_iters):
            if all_sync(mask, status >= 0):
                break

            iters_count += int32(1)
            if status < 0:
                lin_return = linear_solver(
                    stage_increment, parameters, drivers, base_state,
                    t, h, a_ij, residual, delta,
                )
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

                    residual_function(
                        stage_increment, parameters, drivers,
                        t, h, a_ij, base_state, residual,
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
# DIRK STEP FUNCTION REFERENCE
# =========================================================================
#
# The DIRK step function is complex and tightly coupled to the tableau.
# For debugging purposes, reference:
#   src/cubie/integrators/algorithms/generic_dirk.py
#
# Key components:
# - Stage accumulator: stores intermediate RHS values
# - Newton solver: called for each implicit stage
# - Error estimate: computed from embedded method differences
#
# The step function signature:
#   step(state, proposed_state, parameters, driver_coeffs, drivers_buffer,
#        proposed_drivers, observables, proposed_observables, error,
#        dt_scalar, time_scalar, first_step_flag, accepted_flag,
#        shared, persistent_local, counters) -> status
#
# To inline the DIRK step, the full build_step method from DIRKStep
# would need to be copied. Due to its length (~350 lines), it is
# recommended to keep it as a reference rather than inlining.
#
# =========================================================================

# =========================================================================
# PID STEP CONTROLLER (Inlined from step_control/adaptive_PID_controller.py)
# =========================================================================

def pid_controller_factory_inline(
    prec,
    clamp_fn,
    kp,
    ki,
    kd,
    algorithm_order,
    n,
    atol,
    rtol,
    min_gain,
    max_gain,
    dt_min,
    dt_max,
    deadband_min,
    deadband_max,
    safety=0.9,
):
    """Create a PID step controller device function.

    Parameters
    ----------
    prec : numpy dtype
        Floating-point precision.
    clamp_fn : callable
        Clamping device function.
    kp, ki, kd : float
        PID gains (before scaling by order).
    algorithm_order : int
        Order of the integration algorithm.
    n : int
        Number of state variables.
    atol, rtol : array-like
        Absolute and relative tolerances.
    min_gain, max_gain : float
        Step size change factor bounds.
    dt_min, dt_max : float
        Absolute step size bounds.
    deadband_min, deadband_max : float
        Unity deadband thresholds.
    safety : float
        Safety factor for step size scaling.

    Returns
    -------
    callable
        CUDA device function implementing PID control.
    """
    numba_precision = from_dtype(prec)
    atol = np.asarray(atol, dtype=prec)
    rtol = np.asarray(rtol, dtype=prec)

    expo1 = numba_precision(kp / (2 * (algorithm_order + 1)))
    expo2 = numba_precision(ki / (2 * (algorithm_order + 1)))
    expo3 = numba_precision(kd / (2 * (algorithm_order + 1)))
    unity_gain = numba_precision(1.0)
    deadband_disabled = (deadband_min == unity_gain) and (
        deadband_max == unity_gain
    )

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
        local_temp
    ):
        err_prev = local_temp[0]
        err_prev_prev = local_temp[1]
        nrm2 = numba_precision(0.0)

        for i in range(n):
            error_i = max(abs(error[i]), numba_precision(1e-12))
            tol = atol[i] + rtol[i] * max(
                abs(state[i]), abs(state_prev[i])
            )
            ratio = tol / error_i
            nrm2 += ratio * ratio

        nrm2 = numba_precision(nrm2 / n)
        accept = nrm2 >= numba_precision(1.0)
        accept_out[0] = int32(1) if accept else int32(0)
        err_prev_safe = err_prev if err_prev > numba_precision(0.0) else nrm2
        err_prev_prev_safe = (
            err_prev_prev if err_prev_prev > numba_precision(0.0)
            else err_prev_safe
        )

        gain_new = numba_precision(
            safety
            * (nrm2 ** expo1)
            * (err_prev_safe ** expo2)
            * (err_prev_prev_safe ** expo3)
        )
        gain = numba_precision(clamp_fn(gain_new, min_gain, max_gain))
        if not deadband_disabled:
            within_deadband = (
                (gain >= deadband_min)
                and (gain <= deadband_max)
            )
            gain = selp(within_deadband, unity_gain, gain)

        dt_new_raw = dt[0] * gain
        dt[0] = clamp_fn(dt_new_raw, dt_min, dt_max)
        local_temp[1] = err_prev
        local_temp[0] = nrm2

        ret = int32(0) if dt_new_raw > dt_min else int32(8)
        return ret

    return controller_PID


# =========================================================================
# OUTPUT FUNCTIONS REFERENCE
# =========================================================================
#
# Output functions handle saving state snapshots and accumulating summaries.
# For debugging, reference:
#   src/cubie/outputhandling/output_functions.py
#   src/cubie/outputhandling/save_state.py
#   src/cubie/outputhandling/update_summaries.py
#   src/cubie/outputhandling/save_summaries.py
#
# Key functions:
# - save_state: writes state/observable values to output arrays
# - update_summaries: accumulates metrics (mean, max, etc.) in buffers
# - save_summaries: commits accumulated metrics to output arrays
#
# These are typically lightweight functions that index into arrays
# based on compile-time configuration (saved indices, metric types).
#
# =========================================================================

# =========================================================================
# INTEGRATION LOOP REFERENCE
# =========================================================================
#
# The main integration loop orchestrates:
# - Time stepping (dt management)
# - Step function calls (DIRK with Newton solver)
# - Controller updates (PID step adjustment)
# - Output saves (state snapshots, summaries)
#
# Reference: src/cubie/integrators/loops/ode_loop.py
#
# The loop is the outermost device function and calls all others.
# Inlining the loop requires inlining all its dependencies.
#
# =========================================================================

# =========================================================================
# EXECUTION SECTION
# =========================================================================

def run_integration():
    """Execute the Lorenz system integration.

    This function:
    1. Builds the Lorenz ODE system
    2. Creates the solver with configured settings
    3. Runs the integration
    4. Prints instructions for capturing generated code

    Returns
    -------
    SolveResult
        Integration results containing state trajectories and summaries.
    """
    from cubie import Solver
    from cubie.odesystems.symbolic.odefile import GENERATED_DIR

    print("=" * 70)
    print("All-in-One Debug File - Lorenz System Integration")
    print("=" * 70)

    # Step 1: Build system
    print("\n[1/4] Building Lorenz system...")
    system = build_lorenz_system(precision)
    print(f"      System: {system.sizes.states} states, "
          f"{system.sizes.parameters} parameters")

    # Step 2: Create solver
    print("\n[2/4] Creating solver with settings:")
    print(f"      Algorithm: {solver_settings['algorithm']}")
    print(f"      Controller: {solver_settings['step_controller']}")
    print(f"      Duration: {solver_settings['duration']}")
    print(f"      dt_save: {solver_settings['dt_save']}")

    solver = Solver(system, **solver_settings)

    # Step 3: Run integration
    print("\n[3/4] Running integration...")
    # Prepare initial values and parameters for solve()
    initial_values = {
        'x': np.array([INITIAL_X], dtype=precision),
        'y': np.array([INITIAL_Y], dtype=precision),
        'z': np.array([INITIAL_Z], dtype=precision),
    }
    parameters = {
        'rho': np.array([LORENZ_RHO], dtype=precision),
    }
    result = solver.solve(
        initial_values=initial_values,
        parameters=parameters,
        duration=solver_settings['duration'],
    )
    has_data = result.time_domain_array is not None and len(
        result.time_domain_array) > 0
    print(f"      Status: {'Success' if has_data else 'No data returned'}")

    # Step 4: Instructions for generated code
    print("\n[4/4] Generated code location:")
    print(f"      {GENERATED_DIR / 'lorenz_attractor.py'}")
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("1. Open the generated file at the path above")
    print("2. Copy the factory functions (dxdt_factory, etc.)")
    print("3. Paste them into the GENERATED CODE PLACEHOLDER section")
    print("4. Set GENERATED_CODE_AVAILABLE = True")
    print("5. Run this file again with all functions in one place")
    print("=" * 70)

    return result


def run_debug_integration(n_runs=2**23, rho_min=0.0, rho_max=21.0):
    """Execute integration with all device functions inlined.

    This function bypasses the normal CuBIE compilation path and uses
    inlined device functions for debugging with Numba's lineinfo.

    Parameters
    ----------
    n_runs : int
        Number of parallel integrations to run. Default is 2**23.
    rho_min : float
        Minimum value of rho parameter. Default is 0.0.
    rho_max : float
        Maximum value of rho parameter. Default is 21.0.

    Returns
    -------
    tuple
        (state_output, status_codes) arrays from the integration.
    """
    from math import ceil
    from numba import int16, int32, float32, float64
    from numba import from_dtype as numba_from_dtype
    from cubie.cuda_simsafe import activemask, all_sync, selp, compile_kwargs
    from cubie.cuda_simsafe import from_dtype as simsafe_dtype

    print("=" * 70)
    print("Debug Integration - All Functions Inlined")
    print("=" * 70)
    print(f"\nRunning {n_runs:,} integrations with rho in [{rho_min}, {rho_max}]")

    # ===================================================================
    # COMPILE-TIME CONSTANTS
    # ===================================================================
    numba_precision = numba_from_dtype(precision)
    simsafe_precision = simsafe_dtype(precision)
    simsafe_int32 = simsafe_dtype(np.int32)

    # System dimensions
    n_states = 3
    n_parameters = 1
    n_observables = 0
    n_drivers = 0
    n_counters = 4

    # Time parameters
    duration = precision(solver_settings['duration'])
    warmup = precision(solver_settings['warmup'])
    dt_save = precision(solver_settings['dt_save'])
    dt0 = precision(solver_settings['dt'])
    dt_min = precision(solver_settings['dt_min'])
    # dt_max available if adaptive stepping is needed
    _dt_max = precision(solver_settings['dt_max'])

    # Algorithm parameters (available for adaptive controllers)
    _atol = precision(solver_settings['atol'])
    _rtol = precision(solver_settings['rtol'])

    # Output dimensions
    n_output_samples = int(ceil(duration / dt_save)) + 1

    # Lorenz constants (sigma=10, beta=8/3)
    sigma = precision(LORENZ_SIGMA)
    beta = precision(LORENZ_BETA)

    # ===================================================================
    # INLINED DEVICE FUNCTIONS
    # ===================================================================

    # --- dxdt function (Lorenz system) ---
    @cuda.jit(device=True, inline=True, **compile_kwargs)
    def dxdt_inline(state, parameters, drivers, observables, dxdt_out, t):
        x = state[0]
        y = state[1]
        z = state[2]
        rho = parameters[0]
        dxdt_out[0] = sigma * (y - x)
        dxdt_out[1] = x * (rho - z) - y
        dxdt_out[2] = x * y - beta * z

    # --- observables function (no-op for Lorenz) ---
    @cuda.jit(device=True, inline=True, **compile_kwargs)
    def observables_inline(state, parameters, drivers, observables, t):
        pass

    # --- save_state function ---
    @cuda.jit(device=True, inline=True, **compile_kwargs)
    def save_state_inline(
        current_state,
        current_observables,
        current_counters,
        current_step,
        output_states_slice,
        output_observables_slice,
        output_counters_slice,
    ):
        for k in range(n_states):
            output_states_slice[k] = current_state[k]
        # Save time at the end
        output_states_slice[n_states] = current_step
        # Save counters
        for i in range(n_counters):
            output_counters_slice[i] = current_counters[i]

    # --- update_summaries function (mean accumulation) ---
    @cuda.jit(device=True, inline=True, **compile_kwargs)
    def update_summaries_inline(
        current_state,
        current_observables,
        state_summary_buffer,
        observable_summary_buffer,
        current_step,
    ):
        # Accumulate sum for mean calculation
        for idx in range(n_states):
            state_summary_buffer[idx] += current_state[idx]

    # --- save_summaries function ---
    @cuda.jit(device=True, inline=True, **compile_kwargs)
    def save_summaries_inline(
        buffer_state_summaries,
        buffer_observable_summaries,
        output_state_summaries_window,
        output_observable_summaries_window,
        summarise_every,
    ):
        # Compute mean and save
        for idx in range(n_states):
            mean_val = buffer_state_summaries[idx] / numba_precision(
                summarise_every)
            output_state_summaries_window[idx] = mean_val
            buffer_state_summaries[idx] = numba_precision(0.0)

    # --- Explicit Euler step (simplified for debugging) ---
    @cuda.jit(device=True, inline=True, **compile_kwargs)
    def step_euler_inline(
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
        dxdt_buffer = cuda.local.array(n_states, numba_precision)

        # Evaluate derivative
        dxdt_inline(
            state, parameters, drivers_buffer, observables, dxdt_buffer,
            time_scalar)

        # Euler step: x_new = x + dt * dxdt
        for idx in range(n_states):
            proposed_state[idx] = state[idx] + dt_scalar * dxdt_buffer[idx]

        # No error estimate for explicit Euler
        for idx in range(n_states):
            error[idx] = numba_precision(0.0)

        counters[0] = int32(1)
        counters[1] = int32(0)
        return int32(0)

    # --- Fixed step controller ---
    @cuda.jit(device=True, inline=True, **compile_kwargs)
    def step_controller_fixed_inline(
        dt,
        state,
        state_prev,
        error,
        niters,
        accept_out,
        local_temp
    ):
        # Always accept for fixed step
        accept_out[0] = int32(1)
        return int32(0)

    # ===================================================================
    # INLINED INTEGRATION LOOP (from IVPLoop)
    # ===================================================================

    # Buffer layout constants
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
    state_summ_start = proposed_counters_end
    state_summ_end = state_summ_start + n_states
    obs_summ_start = state_summ_end
    obs_summ_end = obs_summ_start + max(n_observables, 1)
    scratch_start = obs_summ_end
    shared_elements = scratch_start + 64  # Extra scratch space

    # Local memory layout
    local_dt_slice = slice(0, 1)
    local_accept_slice = slice(1, 2)
    local_controller_slice = slice(2, 4)
    local_algo_slice = slice(4, 8)
    local_elements = 8

    steps_per_save = int32(ceil(dt_save / dt0))
    saves_per_summary = int32(2)
    # equality_breaker available for adaptive stepping boundary checks
    _equality_breaker = precision(1e-7)
    status_mask = int32(0xFFFF)
    # fixed_mode flag for reference (using fixed step for simplicity)
    _fixed_mode = True

    @cuda.jit(device=True, inline=True, **compile_kwargs)
    def loop_fn_inline(
        initial_states,
        parameters,
        driver_coefficients,
        shared_scratch,
        persistent_local,
        state_output,
        observables_output,
        state_summaries_output,
        observable_summaries_output,
        iteration_counters_output,
        duration_arg,
        settling_time,
        t0_arg,
    ):
        t = numba_precision(t0_arg)
        t_end = numba_precision(settling_time + duration_arg)
        max_steps = (int32(ceil(t_end / dt_min)) + int32(2))
        max_steps = max_steps << 2

        n_output_samples_local = state_output.shape[0]
        shared_scratch[:] = numba_precision(0.0)

        # Buffer views
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
        state_summ_buffer = shared_scratch[state_summ_start:state_summ_end]
        obs_summ_buffer = shared_scratch[obs_summ_start:obs_summ_end]
        remaining_scratch = shared_scratch[scratch_start:]

        dt = persistent_local[local_dt_slice]
        accept_step = persistent_local[local_accept_slice].view(simsafe_int32)
        _controller_temp = persistent_local[local_controller_slice]
        algo_local = persistent_local[local_algo_slice]

        first_step_flag = int16(1)
        prev_step_accepted_flag = int16(1)

        # Initialize
        for k in range(n_states):
            state_buffer[k] = initial_states[k]
        for k in range(n_parameters):
            params_buffer[k] = parameters[k]

        save_idx = int32(0)
        summary_idx = int32(0)
        next_save = numba_precision(dt_save)

        # Save initial state
        save_state_inline(
            state_buffer, obs_buffer, counters_since_save, t,
            state_output[save_idx, :], observables_output[0, :],
            iteration_counters_output[save_idx, :])
        update_summaries_inline(
            state_buffer, obs_buffer, state_summ_buffer, obs_summ_buffer,
            save_idx)
        save_idx += int32(1)

        status = int32(0)
        dt[0] = dt0
        dt_eff = dt[0]
        accept_step[0] = int32(0)

        for i in range(n_counters):
            counters_since_save[i] = int32(0)

        step_counter = int32(0)
        mask = activemask()

        # Main integration loop
        for _ in range(max_steps):
            finished = save_idx >= n_output_samples_local
            if all_sync(mask, finished):
                return status

            if not finished:
                step_counter += 1
                do_save = (step_counter % steps_per_save) == 0
                if do_save:
                    step_counter = int32(0)

                step_status = step_euler_inline(
                    state_buffer, state_proposal, params_buffer,
                    driver_coefficients, drivers_buffer, drivers_proposal,
                    obs_buffer, obs_proposal, error_buffer, dt_eff, t,
                    first_step_flag, prev_step_accepted_flag,
                    remaining_scratch, algo_local, proposed_counters)

                first_step_flag = int16(0)
                status |= step_status & status_mask

                # Accept step
                t_proposal = t + dt_eff
                t = t_proposal

                for i in range(n_states):
                    state_buffer[i] = state_proposal[i]

                prev_step_accepted_flag = int16(1)
                next_save = selp(do_save, next_save + dt_save, next_save)

                # Accumulate counters
                for i in range(n_counters):
                    if i < 2:
                        counters_since_save[i] += proposed_counters[i]
                    elif i == 2:
                        counters_since_save[i] += int32(1)

                if do_save:
                    save_state_inline(
                        state_buffer, obs_buffer, counters_since_save, t,
                        state_output[save_idx, :], observables_output[0, :],
                        iteration_counters_output[save_idx, :])
                    update_summaries_inline(
                        state_buffer, obs_buffer, state_summ_buffer,
                        obs_summ_buffer, save_idx)

                    if (save_idx + 1) % saves_per_summary == 0:
                        save_summaries_inline(
                            state_summ_buffer, obs_summ_buffer,
                            state_summaries_output[summary_idx, :],
                            observable_summaries_output[0, :],
                            saves_per_summary)
                        summary_idx += 1

                    save_idx += 1
                    for i in range(n_counters):
                        counters_since_save[i] = int32(0)

        if status == int32(0):
            status = int32(32)
        return status

    # ===================================================================
    # INLINED KERNEL (from BatchSolverKernel)
    # ===================================================================

    local_elements_per_run = local_elements
    shared_elems_per_run = shared_elements
    f32_per_element = 2 if (numba_precision == float64) else 1
    # run_stride_f32 available for 2D grid configuration
    _run_stride_f32 = int(f32_per_element * shared_elems_per_run)

    @cuda.jit(**compile_kwargs)
    def integration_kernel(
        inits,
        params,
        d_coefficients,
        state_output,
        observables_output,
        state_summaries_output,
        observables_summaries_output,
        iteration_counters_output,
        status_codes_output,
        duration_k,
        warmup_k,
        t0_k,
        n_runs_k,
    ):
        # Use 1D grid for simpler indexing
        run_index = cuda.grid(1)
        if run_index >= n_runs_k:
            return

        # Dynamic shared memory - size specified in kernel launch
        shared_memory = cuda.shared.array(0, dtype=float32)
        # Local scratch uses float32 for compatibility
        local_scratch = cuda.local.array(local_elements_per_run, dtype=float32)
        c_coefficients = cuda.const.array_like(d_coefficients)

        run_idx_low = 0  # Simplified for 1D grid
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

        status = loop_fn_inline(
            rx_inits, rx_params, c_coefficients, rx_shared_memory,
            local_scratch, rx_state, rx_observables, rx_state_summaries,
            rx_observables_summaries, rx_iteration_counters,
            duration_k, warmup_k, t0_k)

        status_codes_output[run_index] = status

    # ===================================================================
    # ALLOCATE ARRAYS AND RUN
    # ===================================================================

    print(f"\nAllocating arrays for {n_runs:,} runs...")

    # Generate rho values
    rho_values = np.linspace(rho_min, rho_max, n_runs, dtype=precision)

    # Input arrays
    inits = np.zeros((n_runs, n_states), dtype=precision)
    inits[:, 0] = precision(INITIAL_X)
    inits[:, 1] = precision(INITIAL_Y)
    inits[:, 2] = precision(INITIAL_Z)

    params = np.zeros((n_runs, n_parameters), dtype=precision)
    params[:, 0] = rho_values

    # Driver coefficients (empty for Lorenz)
    d_coefficients = np.zeros((1, max(n_drivers, 1), 6), dtype=precision)

    # Output arrays
    state_output = np.zeros(
        (n_output_samples, n_runs, n_states + 1), dtype=precision)
    observables_output = np.zeros((n_output_samples, 1, 1), dtype=precision)
    n_summary_samples = int(ceil(n_output_samples / saves_per_summary))
    state_summaries_output = np.zeros(
        (n_summary_samples, n_runs, n_states), dtype=precision)
    observable_summaries_output = np.zeros(
        (n_summary_samples, 1, 1), dtype=precision)
    iteration_counters_output = np.zeros(
        (n_runs, n_output_samples, n_counters), dtype=np.int32)
    status_codes_output = np.zeros(n_runs, dtype=np.int32)

    print(f"State output shape: {state_output.shape}")
    print(f"Memory per run: ~{state_output.nbytes // n_runs} bytes")

    # Shared memory limit per SM for optimal occupancy (32 KB)
    MAX_SHARED_MEMORY_PER_BLOCK = 32768
    blocksize = 256
    runs_per_block = blocksize
    dynamic_sharedmem = int(
        (f32_per_element * shared_elems_per_run) * 4 * runs_per_block)

    # Limit shared memory to MAX_SHARED_MEMORY_PER_BLOCK
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

    # Launch kernel with 1D configuration
    # Parameters: kernel[grid_dim, block_dim, stream, shared_memory]
    integration_kernel[
        blocks_per_grid,
        blocksize,
        0,
        dynamic_sharedmem,
    ](
        inits, params, d_coefficients,
        state_output, observables_output,
        state_summaries_output, observable_summaries_output,
        iteration_counters_output, status_codes_output,
        duration, warmup, precision(0.0), n_runs,
    )

    cuda.synchronize()

    print("\n" + "=" * 70)
    print("Integration Complete")
    print("=" * 70)

    # Report results
    success_count = np.sum(status_codes_output == 0)
    print(f"\nSuccessful runs: {success_count:,} / {n_runs:,}")
    print(f"Final state sample (run 0): {state_output[-1, 0, :n_states]}")
    print(f"Final state sample (run -1): {state_output[-1, -1, :n_states]}")

    return state_output, status_codes_output


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--debug":
        run_debug_integration()
    else:
        result = run_integration()
        if result is not None:
            has_data = result.time_domain_array is not None and len(
                result.time_domain_array) > 0
            if has_data:
                print("\nIntegration completed successfully!")
                state_array = result.as_numpy.get('time_domain_array')
                if state_array is not None:
                    print(f"State array shape: {state_array.shape}")
