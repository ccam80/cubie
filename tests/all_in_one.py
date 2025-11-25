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
"""

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
    # Algorithm selection
    "algorithm": "dirk",

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
    result = solver.solve(n_runs=1)
    print(f"      Status: {'Success' if result.success else 'Failed'}")

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


def run_debug_integration():
    """Execute integration with all device functions inlined.

    This function is used after generated code has been pasted in.
    It bypasses the normal CuBIE compilation path and uses the
    inlined device functions for debugging.

    Note
    ----
    This is intentionally a stub. Full implementation requires manually
    constructing the integration loop with inlined device functions,
    which is beyond the scope of this debugging utility. Users should
    use this file's inlined functions with Numba's lineinfo feature
    to trace execution through cuda-gdb or nsys profiling.

    Returns
    -------
    None
        Prints debug information during execution.
    """
    if not GENERATED_CODE_AVAILABLE:
        print("ERROR: Generated code not available.")
        print("       Run run_integration() first, then paste generated code.")
        return

    print("=" * 70)
    print("Debug Integration - All Functions Inlined")
    print("=" * 70)
    print("\nNote: This function is a stub for future development.")
    print("      For debugging, use the inlined device functions with")
    print("      Numba's lineinfo feature via cuda-gdb or nsys profiling.")
    print("=" * 70)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--debug":
        run_debug_integration()
    else:
        result = run_integration()
        if result is not None and result.success:
            print("\nIntegration completed successfully!")
            print(f"Output shape: {result.state.shape}")
