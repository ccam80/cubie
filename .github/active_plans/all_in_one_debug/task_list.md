# Implementation Task List
# Feature: All-in-One Debug File
# Plan Reference: .github/active_plans/all_in_one_debug/agent_plan.md

## Task Group 1: Create Debug File Skeleton with Configuration - SEQUENTIAL
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: tests/conftest.py (lines 340-417) - solver_settings fixture pattern
- File: tests/system_fixtures.py (entire file) - system definition patterns
- File: src/cubie/__init__.py (entire file) - public API imports

**Input Validation Required**:
- None (configuration section is user-editable)

**Tasks**:
1. **Create tests/all_in_one.py with Configuration Section**
   - File: tests/all_in_one.py
   - Action: Create
   - Details:
     ```python
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
     ```
   - Edge cases: Precision must be consistent across all float values
   - Integration: Settings dict follows conftest.py patterns for compatibility

**Outcomes**: 
- Files Modified: 
  * tests/all_in_one.py (created, ~730 lines)
- Functions/Methods Added/Modified:
  * Configuration section with solver_settings dict
  * Lorenz system constants (LORENZ_SIGMA, LORENZ_RHO, LORENZ_BETA)
  * Initial conditions (INITIAL_X, INITIAL_Y, INITIAL_Z)
- Implementation Summary:
  Created tests/all_in_one.py with complete configuration section matching conftest.py patterns
- Issues Flagged: None

---

## Task Group 2: Add Lorenz System Definition - SEQUENTIAL
**Status**: [x]
**Dependencies**: Group 1

**Required Context**:
- File: tests/system_fixtures.py (lines 30-62, 67-99) - system definition patterns
- File: src/cubie/odesystems/symbolic/symbolicODE.py (lines 1-100) - create_ODE_system API

**Input Validation Required**:
- None (system definition is user-editable)

**Tasks**:
1. **Add Lorenz System Definition to all_in_one.py**
   - File: tests/all_in_one.py
   - Action: Modify (append after configuration)
   - Details:
     ```python
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
     ```
   - Edge cases: Parameter names must match equation variables
   - Integration: Uses create_ODE_system from cubie public API

**Outcomes**: 
- Files Modified: 
  * tests/all_in_one.py (updated)
- Functions/Methods Added/Modified:
  * LORENZ_EQUATIONS list
  * LORENZ_STATES, LORENZ_PARAMETERS, LORENZ_CONSTANTS dicts
  * build_lorenz_system(prec) function
- Implementation Summary:
  Added Lorenz system definition following system_fixtures.py patterns
- Issues Flagged: None

---

## Task Group 3: Add Utility Device Functions (Inline) - SEQUENTIAL
**Status**: [x]
**Dependencies**: Group 2

**Required Context**:
- File: src/cubie/_utils.py (lines 290-304) - clamp_factory implementation

**Input Validation Required**:
- None (device functions take validated inputs)

**Tasks**:
1. **Add Inline Utility Functions to all_in_one.py**
   - File: tests/all_in_one.py
   - Action: Modify (append after Lorenz system)
   - Details:
     ```python
     # =========================================================================
     # UTILITY DEVICE FUNCTIONS (Inlined from src/cubie/_utils.py)
     # =========================================================================

     from numba import cuda, int16, int32, from_dtype
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
     ```
   - Edge cases: selp must handle all comparison types correctly
   - Integration: Identical to src/cubie/_utils.py implementation

**Outcomes**: 
- Files Modified: 
  * tests/all_in_one.py (updated)
- Functions/Methods Added/Modified:
  * clamp_factory_inline(prec) factory function
- Implementation Summary:
  Added inlined clamp function identical to src/cubie/_utils.py
- Issues Flagged: None

---

## Task Group 4: Add Generated Code Placeholder Section - SEQUENTIAL
**Status**: [x]
**Dependencies**: Group 3

**Required Context**:
- File: src/cubie/odesystems/symbolic/odefile.py (lines 14-22) - HEADER pattern
- File: src/cubie/odesystems/symbolic/codegen/ (directory) - generator patterns

**Input Validation Required**:
- None (placeholder section)

**Tasks**:
1. **Add Generated Code Placeholder Section**
   - File: tests/all_in_one.py
   - Action: Modify (append after utility functions)
   - Details:
     ```python
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
     ```
   - Edge cases: File must be valid Python even without generated code
   - Integration: Instructions reference actual generated/ directory location

**Outcomes**: 
- Files Modified: 
  * tests/all_in_one.py (updated)
- Functions/Methods Added/Modified:
  * GENERATED_CODE_PLACEHOLDER section with instructions
  * GENERATED_CODE_AVAILABLE flag
- Implementation Summary:
  Added placeholder section with clear instructions for pasting generated code
- Issues Flagged: None

---

## Task Group 5: Add Linear Solver Device Function (Inline) - SEQUENTIAL
**Status**: [x]
**Dependencies**: Group 4

**Required Context**:
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py (lines 18-225) - full implementation

**Input Validation Required**:
- operator_apply: Must be a CUDA device function
- n: Must be positive integer
- tolerance: Must be positive float
- max_iters: Must be positive integer

**Tasks**:
1. **Add Inline Linear Solver Factory**
   - File: tests/all_in_one.py
   - Action: Modify (append after generated code placeholder)
   - Details:
     ```python
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

             operator_apply(state, parameters, drivers, base_state, t, h, a_ij, x, temp)
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
     ```
   - Edge cases: Empty rhs, zero denominator handled via selp
   - Integration: Identical to src implementation with inline closures

**Outcomes**: 
- Files Modified: 
  * tests/all_in_one.py (updated)
- Functions/Methods Added/Modified:
  * linear_solver_factory_inline() factory function
  * linear_solver() device function (nested)
- Implementation Summary:
  Added inlined linear solver identical to src/cubie/integrators/matrix_free_solvers/linear_solver.py
- Issues Flagged: None

---

## Task Group 6: Add Newton-Krylov Solver Device Function (Inline) - SEQUENTIAL
**Status**: [x]
**Dependencies**: Group 5

**Required Context**:
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py (lines 17-262) - full implementation

**Input Validation Required**:
- residual_function: Must be a CUDA device function
- linear_solver: Must be from linear_solver_factory
- n: Must be positive integer
- tolerance: Must be positive float

**Tasks**:
1. **Add Inline Newton-Krylov Solver Factory**
   - File: tests/all_in_one.py
   - Action: Modify (append after linear solver)
   - Details:
     ```python
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
         numba_precision = from_dtype(np.dtype(prec))
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
     ```
   - Edge cases: Already converged at start, backtracking exhaustion
   - Integration: Identical to src implementation

**Outcomes**: 
- Files Modified: 
  * tests/all_in_one.py (updated)
- Functions/Methods Added/Modified:
  * newton_krylov_solver_factory_inline() factory function
  * newton_krylov_solver() device function (nested)
- Implementation Summary:
  Added inlined Newton-Krylov solver identical to src/cubie/integrators/matrix_free_solvers/newton_krylov.py
- Issues Flagged: None

---

## Task Group 7: Add PID Controller Device Function (Inline) - SEQUENTIAL
**Status**: [x]
**Dependencies**: Group 6

**Required Context**:
- File: src/cubie/integrators/step_control/adaptive_PID_controller.py (lines 156-310) - build_controller method

**Input Validation Required**:
- kp, ki, kd: Must be non-negative floats
- dt_min, dt_max: dt_min < dt_max
- atol, rtol: Must be positive arrays

**Tasks**:
1. **Add Inline PID Controller Factory**
   - File: tests/all_in_one.py
   - Action: Modify (append after Newton-Krylov)
   - Details:
     ```python
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
     ```
   - Edge cases: Zero error, deadband logic, minimum dt rejection
   - Integration: Identical to src implementation

**Outcomes**: 
- Files Modified: 
  * tests/all_in_one.py (updated)
- Functions/Methods Added/Modified:
  * pid_controller_factory_inline() factory function
  * controller_PID() device function (nested)
- Implementation Summary:
  Added inlined PID controller identical to src/cubie/integrators/step_control/adaptive_PID_controller.py
- Issues Flagged: None

---

## Task Group 8: Add Execution Section - SEQUENTIAL
**Status**: [x]
**Dependencies**: Group 7

**Required Context**:
- File: tests/conftest.py (lines 135-145) - _build_solver_instance pattern
- File: src/cubie/batchsolving/solver.py (lines 1-100) - Solver class usage

**Input Validation Required**:
- None (execution uses validated settings from earlier groups)

**Tasks**:
1. **Add Main Execution Block**
   - File: tests/all_in_one.py
   - Action: Modify (append at end of file)
   - Details:
     ```python
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
         print("\nTODO: Implement inlined kernel execution")
         print("      This requires building the full loop with inlined functions")
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
     ```
   - Edge cases: Generated code not available, solver failure
   - Integration: Uses Solver from cubie public API

**Outcomes**: 
- Files Modified: 
  * tests/all_in_one.py (updated)
- Functions/Methods Added/Modified:
  * run_integration() function
  * run_debug_integration() function
  * if __name__ == "__main__" block
- Implementation Summary:
  Added execution section with main functions and CLI entry point
- Issues Flagged: None

---

## Task Group 9: Add Reference to DIRK Step (Documentation) - SEQUENTIAL
**Status**: [x]
**Dependencies**: Group 8

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_dirk.py (entire file) - DIRK implementation

**Input Validation Required**:
- None (documentation only)

**Tasks**:
1. **Add DIRK Step Reference Comment Block**
   - File: tests/all_in_one.py
   - Action: Modify (insert after Newton-Krylov, before PID controller)
   - Details:
     ```python
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
     ```
   - Edge cases: N/A (documentation)
   - Integration: Points to source location for manual inlining

**Outcomes**: 
- Files Modified: 
  * tests/all_in_one.py (updated)
- Functions/Methods Added/Modified:
  * DIRK STEP FUNCTION REFERENCE comment block
- Implementation Summary:
  Added documentation reference block for DIRK step function
- Issues Flagged: None

---

## Task Group 10: Add Output Function References (Documentation) - SEQUENTIAL
**Status**: [x]
**Dependencies**: Group 9

**Required Context**:
- File: src/cubie/outputhandling/output_functions.py (entire file) - output functions
- File: src/cubie/outputhandling/save_state.py (if exists) - save_state implementation
- File: src/cubie/outputhandling/update_summaries.py (if exists) - update implementation

**Input Validation Required**:
- None (documentation only)

**Tasks**:
1. **Add Output Functions Reference Comment Block**
   - File: tests/all_in_one.py
   - Action: Modify (insert after PID controller, before execution section)
   - Details:
     ```python
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
     ```
   - Edge cases: N/A (documentation)
   - Integration: Points to source locations for manual reference

**Outcomes**: 
- Files Modified: 
  * tests/all_in_one.py (updated)
- Functions/Methods Added/Modified:
  * OUTPUT FUNCTIONS REFERENCE comment block
  * INTEGRATION LOOP REFERENCE comment block
- Implementation Summary:
  Added documentation reference blocks for output functions and integration loop
- Issues Flagged: None

---

# Summary

## Total Task Groups: 10
## All Groups Completed: 10
## Dependency Chain:
```
Group 1 (Skeleton) 
    → Group 2 (Lorenz System)
        → Group 3 (Utilities)
            → Group 4 (Placeholder)
                → Group 5 (Linear Solver)
                    → Group 6 (Newton-Krylov)
                        → Group 7 (PID Controller)
                            → Group 8 (Execution)
                                → Group 9 (DIRK Reference)
                                    → Group 10 (Output Reference)
```

## Parallel Execution Opportunities:
- None - all groups are sequential dependencies

## Estimated Complexity:
- Groups 1-4: Low (configuration, definitions, placeholders)
- Groups 5-7: Medium (inlined device function factories)
- Group 8: Low (execution harness)
- Groups 9-10: Low (documentation/reference)

## Key Implementation Notes:
1. All device function factories are inlined copies from src/cubie
2. Generated code must be manually copied after first run
3. The file is intentionally standalone for debugging purposes
4. DIRK step and output functions are referenced, not inlined (too complex)
5. File location is tests/all_in_one.py for easy pytest integration

---

# Implementation Complete - Ready for Review

## Execution Summary
- Total Task Groups: 10
- Completed: 10
- Failed: 0
- Total Files Modified: 1

## Task Group Completion
- Group 1: [x] Create Debug File Skeleton with Configuration - Complete
- Group 2: [x] Add Lorenz System Definition - Complete
- Group 3: [x] Add Utility Device Functions (Inline) - Complete
- Group 4: [x] Add Generated Code Placeholder Section - Complete
- Group 5: [x] Add Linear Solver Device Function (Inline) - Complete
- Group 6: [x] Add Newton-Krylov Solver Device Function (Inline) - Complete
- Group 7: [x] Add PID Controller Device Function (Inline) - Complete
- Group 8: [x] Add Execution Section - Complete
- Group 9: [x] Add Reference to DIRK Step (Documentation) - Complete
- Group 10: [x] Add Output Function References (Documentation) - Complete

## All Modified Files
1. tests/all_in_one.py (817 lines created)

## Flagged Issues
None - all tasks completed as specified.

## Handoff to Reviewer
All implementation tasks complete. Task list updated with outcomes.
Ready for reviewer agent to validate against user stories and goals.
