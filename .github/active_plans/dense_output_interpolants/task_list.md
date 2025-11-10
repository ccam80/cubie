# Implementation Task List
# Feature: Dense Output Interpolants
# Plan Reference: .github/active_plans/dense_output_interpolants/agent_plan.md

## Overview

This task list implements dense output interpolants for Runge-Kutta methods in CuBIE. The implementation eliminates error estimate inflation by computing accurate errors from full steps while using interpolation to reach save points precisely.

**Key Design Principles:**
- Pass full `dt[0]` to step functions (no truncation)
- Use `selp()` for predicated commits (no branching)
- Pass `next_save` via shared memory (no signature changes)
- Only add coefficients from published literature

---

## Task Group 1: Loop Infrastructure - Shared Memory for next_save - SEQUENTIAL
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/loops/ode_loop.py (lines 1-600, full file)
- File: .github/context/cubie_internal_structure.md (sections: Loops, CUDAFactory Pattern)

**Input Validation Required**:
- None (internal refactoring only)

**Tasks**:

1. **Add shared memory allocation for next_save**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Details:
     * Locate the shared memory size calculation section (before kernel definition in `build()` method)
     * Add calculation for `shared_next_save_bytes`:
       ```python
       shared_next_save_bytes = precision_size  # 4 bytes for float32, 8 for float64
       ```
     * Add this to the running total of shared memory bytes
     * Update any relevant comments documenting shared memory layout
   - Location: In `IVPLoop.build()` method, shared memory allocation section
   - Edge cases: Ensure precision_size is correctly computed based on `self.precision`
   - Integration: Must be included in `total_shared_bytes` before kernel launch config

2. **Declare shared memory buffer in device kernel**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Details:
     * Locate the kernel device function definition (the `@cuda.jit` decorated function)
     * Find the section where shared memory arrays are declared (e.g., `cuda.shared.array(...)`)
     * Add declaration:
       ```python
       shared_next_save = cuda.shared.array(shape=(1,), dtype=precision)
       ```
     * Place after other shared memory declarations for clarity
   - Location: Inside the device kernel function, after existing shared memory array declarations
   - Edge cases: Ensure `precision` variable is correctly typed (numba_precision from closure)
   - Integration: This buffer will be written by thread 0 and read by all threads

3. **Populate next_save buffer in integration loop**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Details:
     * Locate the main integration loop (the `for _ in range(max_steps):` loop)
     * Find where `next_save` variable is used (around line 405-406 in adaptive mode)
     * Before the step function call, add:
       ```python
       # Write next_save to shared memory (only thread 0)
       if cuda.threadIdx.x == 0:
           shared_next_save[0] = next_save
       
       # Synchronize to ensure all threads see the value
       cuda.syncthreads()
       ```
     * Place this BEFORE the `step_function(...)` call
   - Location: Main integration loop, adaptive mode branch, before step function invocation
   - Edge cases: Only in adaptive mode (not fixed mode); thread 0 must write before sync
   - Integration: Ensures all threads in block have consistent `next_save` value

4. **Pass full dt[0] to step function (remove dt_eff truncation)**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Details:
     * Locate the line computing `dt_eff` (around line 406):
       ```python
       dt_eff = selp(do_save, next_save - t, dt[0])
       ```
     * Remove this line entirely
     * Change the step function call to pass `dt[0]` instead of `dt_eff`:
       ```python
       # OLD:
       step_status = step_function(..., dt_eff, ...)
       
       # NEW:
       step_status = step_function(..., dt[0], ...)
       ```
   - Location: Main integration loop, before step function call (line ~406-420)
   - Edge cases: Ensure `dt[0]` is always used, regardless of `do_save` flag
   - Integration: Step function will now handle interpolation internally

5. **Update do_save flag passing (if needed)**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify (if do_save is not already computed before step call)
   - Details:
     * Verify `do_save` flag is computed before step function call
     * Current location (line ~405):
       ```python
       do_save = (t + dt[0] + equality_breaker) >= next_save
       ```
     * This should remain unchanged and be computed BEFORE populating shared_next_save
     * No modification needed if already in correct order
   - Location: Main integration loop, adaptive mode
   - Edge cases: None
   - Integration: `do_save` is read-only by step function (not modified)

**Outcomes**: 
[To be filled by do_task agent]

---

## Task Group 2: DIRK Interpolation Logic - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Group 1 (requires shared_next_save buffer)

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_dirk.py (full file)
- File: src/cubie/integrators/algorithms/base_algorithm_step.py (lines 36-184, ButcherTableau class)
- File: .github/active_plans/dense_output_interpolants/agent_plan.md (sections 3.1-3.9)

**Input Validation Required**:
- theta: Clamp to [0.0, 1.0] to prevent floating-point edge cases
- needs_interp: Computed predicate, no validation needed (boolean logic)

**Tasks**:

1. **Capture compile-time interpolation flags**
   - File: src/cubie/integrators/algorithms/generic_dirk.py
   - Action: Modify
   - Details:
     * Locate the `build_step()` method (around line 304)
     * After capturing tableau properties (lines ~318-340), add:
       ```python
       # Interpolant support (compile-time constants)
       has_interpolant = tableau.has_interpolant
       b_interp_coeffs = (
           tableau.interpolant_coefficients(numba_precision) 
           if has_interpolant else None
       )
       interpolant_order = (
           len(b_interp_coeffs) - 1 if has_interpolant else 0
       )
       ```
     * These become compile-time constants in the closure
   - Location: `build_step()` method, after line 340 (after `diagonal_coeffs`)
   - Edge cases: Handle `b_interp_coeffs = None` when `has_interpolant = False`
   - Integration: Used in device function for conditional compilation

2. **Read do_save flag from loop**
   - File: src/cubie/integrators/algorithms/generic_dirk.py
   - Action: Modify
   - Details:
     * Locate the device step function signature (line ~385-400)
     * Find where loop-provided flags are read
     * The `do_save` flag should already be accessible from loop context
     * If not explicitly passed, need to derive from time comparison:
       ```python
       # Inside step device function, after unpacking parameters
       # Assuming shared_next_save[0] is accessible (passed via shared memory)
       do_save = (time_scalar + dt_scalar + equality_breaker) >= shared_next_save[0]
       ```
     * Note: Loop already computes `do_save`, but step function needs to re-compute
       it with same logic since it's not passed explicitly
   - Location: Device step function, parameter unpacking section
   - Edge cases: Use same `equality_breaker` value as loop for consistency
   - Integration: Required for `needs_interp` condition

3. **Read next_save from shared memory**
   - File: src/cubie/integrators/algorithms/generic_dirk.py
   - Action: Modify
   - Details:
     * Inside the device step function (after parameter unpacking)
     * Access the shared memory buffer created by loop:
       ```python
       # Read next_save from shared memory (all threads read)
       # Access shared array from loop's shared memory pool
       # Assuming shared buffer is accessible in remaining_shared_scratch
       next_save_value = shared_next_save[0]  # Will be defined in loop context
       ```
     * Note: The actual access mechanism depends on how shared memory is passed
     * If `shared` parameter contains the buffer, extract from there
     * Otherwise, may need to access from closure if loop makes it available
   - Location: Device step function, immediately after parameter declarations
   - Edge cases: Ensure shared memory is synchronized before read
   - Integration: Used for computing `theta` and `needs_interp`

4. **Compute interpolation conditions (predicated)**
   - File: src/cubie/integrators/algorithms/generic_dirk.py
   - Action: Modify
   - Details:
     * After the full RK step is computed (after proposed_state is calculated)
     * Before committing to observables:
       ```python
       # Interpolation conditions (always computed - predicated execution)
       # Recompute do_save locally with same logic as loop
       equality_breaker = numba_precision(1e-12)  # Match loop constant
       do_save = (time_scalar + dt_scalar + equality_breaker) >= next_save_value
       
       # Check if interpolation is needed
       needs_interp = (
           has_interpolant and 
           do_save and 
           (next_save_value >= time_scalar) and 
           (next_save_value <= time_scalar + dt_scalar)
       )
       
       # Compute theta (always, even if not used)
       theta = (next_save_value - time_scalar) / dt_scalar
       
       # Clamp theta to [0, 1] for floating-point safety
       theta = max(numba_precision(0.0), min(numba_precision(1.0), theta))
       ```
   - Location: After full RK step computation, before observables evaluation
   - Edge cases: Division by zero if dt_scalar == 0 (should not happen, but clamp helps)
   - Integration: Both `needs_interp` and `theta` used in subsequent interpolation

5. **Implement interpolant evaluation**
   - File: src/cubie/integrators/algorithms/generic_dirk.py
   - Action: Modify
   - Details:
     * After computing `needs_interp` and `theta`:
       ```python
       # Allocate local buffer for interpolated state
       y_interp = cuda.local.array(n, dtype=numba_precision)
       
       # Evaluate interpolant (always, even if not used - predicated commit)
       for i in range(n):
           y_interp[i] = state[i]  # Start with y(t)
           
           # Add stage contributions with polynomial weights
           if has_interpolant:  # Compile-time branch
               for stage_idx in range(stage_count):
                   # Evaluate polynomial: sum(b_interp[p][stage] * theta^p)
                   weight = numba_precision(0.0)
                   theta_power = numba_precision(1.0)
                   
                   for poly_idx in range(interpolant_order + 1):
                       weight += b_interp_coeffs[poly_idx][stage_idx] * theta_power
                       theta_power *= theta
                   
                   # Add weighted stage derivative contribution
                   # Note: stage_k is the existing stage derivative buffer
                   y_interp[i] += dt_scalar * weight * stage_k[stage_idx][i]
       ```
     * Note: `stage_k` is the buffer holding stage derivatives (k_i)
     * This must be computed after all stages are evaluated
   - Location: After stage computation loop, before proposed_state commit
   - Edge cases: Handle empty interpolant gracefully with compile-time `if has_interpolant`
   - Integration: Uses existing `stage_k` buffer, no new allocations needed

6. **Conditional commit to proposed_state using selp**
   - File: src/cubie/integrators/algorithms/generic_dirk.py
   - Action: Modify
   - Details:
     * After interpolant evaluation, before observables:
       ```python
       # Conditional commit (no branching - predicated execution)
       for i in range(n):
           # Use selp to choose interpolated vs full-step result
           proposed_state[i] = selp(
               needs_interp,
               y_interp[i],          # Interpolated state if saving
               proposed_state[i]     # Full-step result otherwise
           )
       ```
     * This replaces any existing direct assignment to `proposed_state`
     * Must import `selp` from `cubie.cuda_simsafe` if not already imported
   - Location: After interpolant evaluation, before observables computation
   - Edge cases: `selp` requires boolean first argument; ensure `needs_interp` is bool
   - Integration: Modifies `proposed_state` in-place using predicated commit

7. **Compute time for observables (t_obs)**
   - File: src/cubie/integrators/algorithms/generic_dirk.py
   - Action: Modify
   - Details:
     * Before calling observables function:
       ```python
       # Determine time for observables evaluation
       t_obs = selp(needs_interp, next_save_value, time_scalar + dt_scalar)
       ```
     * This time is used for both driver and observables evaluation
   - Location: Immediately before observables function call
   - Edge cases: None
   - Integration: Passed to observables and driver functions

8. **Evaluate drivers at t_obs (if driver function exists)**
   - File: src/cubie/integrators/algorithms/generic_dirk.py
   - Action: Modify
   - Details:
     * Locate driver function call (if `has_driver_function` is True)
     * Change time argument to `t_obs`:
       ```python
       # Evaluate drivers at correct time (either next_save or t+dt)
       if has_driver_function:
           driver_function(proposed_drivers, t_obs)
       ```
     * This ensures drivers are evaluated at the save point, not the step end
   - Location: Before observables function call
   - Edge cases: Only if driver function exists (guarded by `has_driver_function`)
   - Integration: Drivers must be evaluated before observables

9. **Evaluate observables at t_obs with correct state**
   - File: src/cubie/integrators/algorithms/generic_dirk.py
   - Action: Modify
   - Details:
     * Locate observables function call
     * Update to use `t_obs` instead of `time_scalar + dt_scalar`:
       ```python
       # Evaluate observables using (potentially interpolated) state
       observables_function(
           proposed_state,           # Either interpolated or full-step
           parameters,
           proposed_drivers,         # Evaluated at t_obs
           proposed_observables,     # Output buffer
           t_obs                     # Either next_save or t+dt
       )
       ```
   - Location: After driver evaluation
   - Edge cases: Ensure all inputs are consistent (state, drivers, time all at t_obs)
   - Integration: Final step before returning from step function

10. **Ensure error estimate comes from full step**
   - File: src/cubie/integrators/algorithms/generic_dirk.py
   - Action: Verify (no changes expected)
   - Details:
     * Verify that error computation happens BEFORE interpolation
     * Error should be computed from difference between `b` and `b_hat` weights
     * Current implementation should already compute error from full step
     * No modification needed if error is computed before proposed_state modifications
     * Verify error buffer is not affected by interpolation logic
   - Location: Error computation section (should be before interpolation)
   - Edge cases: None
   - Integration: Error must represent full step accuracy, not interpolated accuracy

**Outcomes**: 
[To be filled by do_task agent]

---

## Task Group 3: FIRK Interpolation Logic - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Group 1 (requires shared_next_save buffer), Task Group 2 (pattern reference)

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_firk.py (full file)
- File: src/cubie/integrators/algorithms/generic_dirk.py (completed implementation from Task Group 2)
- File: .github/active_plans/dense_output_interpolants/agent_plan.md (section 3, FIRK-specific notes)

**Input Validation Required**:
- Same as DIRK: theta clamped to [0.0, 1.0]

**Tasks**:

1. **Apply identical interpolation pattern to FIRK**
   - File: src/cubie/integrators/algorithms/generic_firk.py
   - Action: Modify
   - Details:
     * Follow the exact same pattern as DIRK (Task Group 2)
     * The steps are identical, just applied to FIRK's `build_step()` method
     * Key differences from DIRK:
       - FIRK solves all stages simultaneously (coupled system)
       - Stage derivatives `stage_k` buffer structure may differ
       - But interpolant evaluation logic is identical
     * Apply all 10 sub-tasks from Task Group 2:
       1. Capture compile-time flags
       2. Read do_save flag
       3. Read next_save from shared memory
       4. Compute interpolation conditions
       5. Implement interpolant evaluation
       6. Conditional commit with selp
       7. Compute t_obs
       8. Evaluate drivers at t_obs
       9. Evaluate observables at t_obs
       10. Verify error from full step
   - Location: `build_step()` method and device step function
   - Edge cases: Same as DIRK
   - Integration: FIRK stage structure is different, but interpolant evaluation is same

**Outcomes**: 
[To be filled by do_task agent]

---

## Task Group 4: Rosenbrock Interpolation Logic - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Group 1 (requires shared_next_save buffer), Task Group 2 (pattern reference)

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py (full file)
- File: src/cubie/integrators/algorithms/generic_dirk.py (completed implementation from Task Group 2)
- File: .github/active_plans/dense_output_interpolants/agent_plan.md (section 3, Rosenbrock-specific notes)

**Input Validation Required**:
- Same as DIRK: theta clamped to [0.0, 1.0]

**Tasks**:

1. **Apply identical interpolation pattern to Rosenbrock**
   - File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py
   - Action: Modify
   - Details:
     * Follow the exact same pattern as DIRK (Task Group 2)
     * The steps are identical, just applied to Rosenbrock's `build_step()` method
     * Key differences from DIRK:
       - Rosenbrock methods use linearization, not full implicit solve
       - Stage derivatives are k_i values from linear solves
       - But interpolant evaluation logic is identical
     * Apply all 10 sub-tasks from Task Group 2:
       1. Capture compile-time flags
       2. Read do_save flag
       3. Read next_save from shared memory
       4. Compute interpolation conditions
       5. Implement interpolant evaluation
       6. Conditional commit with selp
       7. Compute t_obs
       8. Evaluate drivers at t_obs
       9. Evaluate observables at t_obs
       10. Verify error from full step
   - Location: `build_step()` method and device step function
   - Edge cases: Same as DIRK
   - Integration: Rosenbrock stage structure is different, but interpolant evaluation is same

**Outcomes**: 
[To be filled by do_task agent]

---

## Task Group 5: DIRK Tableau Coefficients from Literature - PARALLEL
**Status**: [ ]
**Dependencies**: None (can be done independently of implementation)

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_dirk_tableaus.py (full file)
- Reference: Hairer & Wanner (1996), "Solving ODEs II", Chapter IV, Section 6
- Reference: .github/active_plans/dense_output_interpolants/agent_plan.md (section 4.1)

**Input Validation Required**:
- All b_interp coefficients must sum to b at theta=1.0 (validate in tableau tests)
- Coefficient precision: verify against published sources

**Tasks**:

1. **Replace TRAPEZOIDAL_DIRK_TABLEAU placeholder interpolant**
   - File: src/cubie/integrators/algorithms/generic_dirk_tableaus.py
   - Action: Modify
   - Details:
     * Locate `TRAPEZOIDAL_DIRK_TABLEAU` definition (around line 59)
     * Current `b_interp` is linear placeholder:
       ```python
       b_interp=(
           (0.0, 0.0),
           (0.5, 0.5),
       ),
       ```
     * Replace with Hermite cubic interpolant (3rd order):
       * Source: Hairer & Wanner (1996), Section IV.6 or Shampine (1985)
       * For methods with c=[0, 1], use cubic Hermite basis:
       ```python
       # Hermite cubic interpolant for trapezoidal rule
       # Based on Hermite basis functions for stages at c=[0, 1]
       b_interp=(
           (0.0, 0.0),                    # theta^0 coefficients
           (1.0, 0.0),                    # theta^1 coefficients
           (-1.5, 1.5),                   # theta^2 coefficients
           (0.5, -0.5),                   # theta^3 coefficients
       ),
       ```
     * Verify these coefficients in literature before committing
     * Update docstring to reference correct source
   - Location: Line ~59-71
   - Edge cases: Verify at theta=0 gives y(t), at theta=1 gives y(t+dt)
   - Integration: Must maintain 3rd-order accuracy for 2nd-order method

2. **Add interpolant to LOBATTO_IIIC_3_TABLEAU**
   - File: src/cubie/integrators/algorithms/generic_dirk_tableaus.py
   - Action: Modify
   - Details:
     * Locate `LOBATTO_IIIC_3_TABLEAU` definition (around line 95)
     * Currently has no `b_interp` field
     * Look up coefficients from Hairer & Wanner (1996), Table IV.6.X
     * Add `b_interp` field with 4th-order interpolant (method is 4th order):
       ```python
       LOBATTO_IIIC_3_TABLEAU = DIRKTableau(
           a=(...),
           b=(...),
           c=(...),
           order=4,
           b_interp=(
               # TODO: Look up from Hairer & Wanner (1996)
               # Should be 5 rows (4th order polynomial) x 3 stages
               (...),  # theta^0
               (...),  # theta^1
               (...),  # theta^2
               (...),  # theta^3
               (...),  # theta^4
           ),
       )
       ```
     * User must consult book to get exact coefficients
     * If coefficients not found, SKIP this tableau (not required for MVP)
   - Location: Line ~95-104
   - Edge cases: Validate polynomial order matches method order
   - Integration: Only add if coefficients are published and verifiable

3. **Document coefficient sources in tableau docstrings**
   - File: src/cubie/integrators/algorithms/generic_dirk_tableaus.py
   - Action: Modify
   - Details:
     * For each tableau with `b_interp` added, update docstring:
       ```python
       """Three-stage Lobatto IIIC DIRK tableau of order four.
       
       ... (existing description) ...
       
       Dense output uses 4th-order polynomial interpolation with
       coefficients from Hairer & Wanner (1996), Section IV.6, Table X.
       
       References
       ----------
       Hairer, E., & Wanner, G. (1996). *Solving Ordinary Differential
       Equations II: Stiff and Differential-Algebraic Problems* (2nd ed.).
       Springer. Chapter IV, Section 6, Table X.
       """
       ```
     * Include specific table number and page reference if available
   - Location: Docstrings for modified tableaus
   - Edge cases: None
   - Integration: Ensures traceability to literature

**Outcomes**: 
[To be filled by do_task agent]

---

## Task Group 6: FIRK Tableau Coefficients from Literature - PARALLEL
**Status**: [ ]
**Dependencies**: None (can be done independently)

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_firk_tableaus.py (full file)
- Reference: Hairer & Wanner (1996), "Solving ODEs II", Chapter IV, Section 6
- Reference: .github/active_plans/dense_output_interpolants/agent_plan.md (section 4.2)

**Input Validation Required**:
- Same validation as DIRK tableaus

**Tasks**:

1. **Add interpolant to GAUSS_LEGENDRE_2_TABLEAU**
   - File: src/cubie/integrators/algorithms/generic_firk_tableaus.py
   - Action: Modify
   - Details:
     * Locate `GAUSS_LEGENDRE_2_TABLEAU` definition
     * This is a 2-stage, 4th-order method
     * Look up coefficients from Hairer & Wanner (1996), Section IV.6
     * Add `b_interp` field:
       ```python
       GAUSS_LEGENDRE_2_TABLEAU = FIRKTableau(
           a=(...),
           b=(...),
           c=(...),
           order=4,
           b_interp=(
               # TODO: Look up from Hairer & Wanner (1996)
               # Should be 5 rows (4th order) x 2 stages
               (...),  # theta^0
               (...),  # theta^1
               (...),  # theta^2
               (...),  # theta^3
               (...),  # theta^4
           ),
       )
       ```
     * If coefficients not found in book, SKIP (not required for MVP)
   - Location: GAUSS_LEGENDRE_2_TABLEAU definition
   - Edge cases: Gauss-Legendre methods have stages in interior, not at boundaries
   - Integration: Must validate interpolant maintains symplectic properties

2. **Add interpolant to RADAU_IIA_5_TABLEAU (if exists)**
   - File: src/cubie/integrators/algorithms/generic_firk_tableaus.py
   - Action: Modify (conditional)
   - Details:
     * Check if RADAU_IIA_5_TABLEAU exists in file
     * If yes, add `b_interp` from Hairer & Wanner (1996), Table IV.6.X
     * 3-stage, 5th-order method requires 6 polynomial terms
     * If no such tableau exists, SKIP this task
   - Location: RADAU_IIA_5_TABLEAU definition (if exists)
   - Edge cases: Radau IIA methods are stiffly accurate (c_s = 1)
   - Integration: Same validation as other tableaus

3. **Document coefficient sources**
   - File: src/cubie/integrators/algorithms/generic_firk_tableaus.py
   - Action: Modify
   - Details:
     * Same pattern as DIRK tableaus (Task Group 5, task 3)
     * Add literature references to docstrings
     * Include table numbers and page references
   - Location: Docstrings for modified tableaus
   - Edge cases: None
   - Integration: Traceability to literature

**Outcomes**: 
[To be filled by do_task agent]

---

## Task Group 7: Rosenbrock Tableau Coefficients from Literature - PARALLEL
**Status**: [ ]
**Dependencies**: None (can be done independently)

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_rosenbrockw_tableaus.py (full file)
- Reference: Lang & Verwer (2001), "ROS3P Paper"
- Reference: .github/active_plans/dense_output_interpolants/agent_plan.md (section 4.3)

**Input Validation Required**:
- Same validation as DIRK tableaus

**Tasks**:

1. **Research ROS3P dense output availability**
   - File: None (research task)
   - Action: Research
   - Details:
     * Consult Lang & Verwer (2001) paper for dense output formula
     * Check if paper provides explicit interpolant coefficients
     * If coefficients are provided, convert to `b_interp` format
     * If not provided, check secondary sources:
       - Hairer & Wanner books
       - SciML/DifferentialEquations.jl implementation
       - scipy.integrate source code
     * Document findings in task outcome
   - Location: N/A
   - Edge cases: Dense output may not be published for ROS3P
   - Integration: Determines if implementation is possible

2. **Add interpolant to ROS3P_TABLEAU (if available)**
   - File: src/cubie/integrators/algorithms/generic_rosenbrockw_tableaus.py
   - Action: Modify (conditional on task 1 findings)
   - Details:
     * Only proceed if task 1 finds published coefficients
     * Locate `ROS3P_TABLEAU` definition
     * Add `b_interp` field with coefficients from literature:
       ```python
       ROS3P_TABLEAU = RosenbrockTableau(
           a=(...),
           b=(...),
           c=(...),
           order=3,
           b_interp=(
               # Coefficients from Lang & Verwer (2001) or other source
               (...),  # theta^0
               (...),  # theta^1
               (...),  # theta^2
               (...),  # theta^3 (for 3rd order)
           ),
       )
       ```
     * If no coefficients found, SKIP and note in outcome
   - Location: ROS3P_TABLEAU definition
   - Edge cases: Rosenbrock coefficients may use different formulation than RK
   - Integration: May require conversion between formulations

3. **Document ROS3P coefficient source (if added)**
   - File: src/cubie/integrators/algorithms/generic_rosenbrockw_tableaus.py
   - Action: Modify (conditional)
   - Details:
     * If `b_interp` was added, update docstring with reference
     * Include paper name, authors, year, and equation/table number
     * Same pattern as DIRK/FIRK tableaus
   - Location: ROS3P_TABLEAU docstring
   - Edge cases: None
   - Integration: Traceability to literature

**Outcomes**: 
[To be filled by do_task agent]

---

## Task Group 8: Integration Tests for Interpolation - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Groups 1-4 (requires implementation complete), Task Groups 5-7 (requires at least one tableau with b_interp)

**Required Context**:
- File: tests/integrators/algorithms/test_interpolant_concept.py (existing proof-of-concept)
- File: tests/conftest.py (fixture patterns)
- File: tests/system_fixtures.py (ODE systems)
- File: .github/active_plans/dense_output_interpolants/agent_plan.md (section 9)

**Input Validation Required**:
- Test tolerances: Set to method order (e.g., 1e-4 for 4th order)
- ODE systems: Use systems with known analytical solutions

**Tasks**:

1. **Create test for interpolant boundary conditions**
   - File: tests/integrators/algorithms/test_dense_output_interpolants.py
   - Action: Create
   - Details:
     * Test that interpolant at theta=0 gives y(t)
     * Test that interpolant at theta=1 gives y(t+dt)
     * Use linear ODE with known solution: dy/dt = lambda*y, y(t) = y0*exp(lambda*t)
     * Test pattern:
       ```python
       @pytest.mark.parametrize("algorithm", ["dirk_trapezoidal", "firk_gauss_2"])
       def test_interpolant_boundary_conditions(algorithm):
           # Set up ODE system
           # Take one step with tableau that has b_interp
           # Verify theta=0 matches start state
           # Verify theta=1 matches end state
           # Use tight tolerance (1e-12)
       ```
   - Location: New test file
   - Edge cases: Test with different dt values, different ODE systems
   - Integration: Validates tableau coefficients are correct

2. **Create test for interpolant accuracy**
   - File: tests/integrators/algorithms/test_dense_output_interpolants.py
   - Action: Modify (add to file created in task 1)
   - Details:
     * Test that interpolated solution matches analytical solution at mid-points
     * Use exponential ODE: dy/dt = lambda*y
     * Test at theta = 0.25, 0.5, 0.75
     * Verify error is within expected order of accuracy
     * Pattern:
       ```python
       @pytest.mark.parametrize("theta", [0.25, 0.5, 0.75])
       @pytest.mark.parametrize("algorithm", ["dirk_trapezoidal"])
       def test_interpolant_accuracy(algorithm, theta):
           # Set up ODE
           # Compute exact solution at t + theta*dt
           # Compute interpolated solution
           # Assert |exact - interpolated| < tolerance
           # Tolerance based on method order and dt
       ```
   - Location: Same test file
   - Edge cases: Test with stiff and non-stiff problems
   - Integration: Validates interpolant order of accuracy

3. **Create test for error estimate accuracy (not inflated)**
   - File: tests/integrators/algorithms/test_dense_output_interpolants.py
   - Action: Modify
   - Details:
     * Test that error estimates are accurate when save points occur
     * Compare error estimate with true error on full step
     * Verify error is not inflated by save point truncation
     * Pattern:
       ```python
       def test_error_estimate_not_inflated():
           # Set up adaptive solver with dense save points
           # Run integration with interpolants enabled
           # Track error estimates from step function
           # Compare with true error (from exact solution)
           # Verify error estimate is within 10% of true error
           # (Not 100%+ inflation as in old truncation method)
       ```
   - Location: Same test file
   - Edge cases: Test with different save point frequencies
   - Integration: Validates the main goal of the feature

4. **Create test for step acceptance rate improvement**
   - File: tests/integrators/algorithms/test_dense_output_interpolants.py
   - Action: Modify
   - Details:
     * Compare step acceptance rate with/without interpolants
     * Use stiff ODE with dense save points
     * Run same integration twice:
       - Once with tableau with b_interp (new behavior)
       - Once with tableau without b_interp (old truncation behavior)
     * Verify acceptance rate improves by 15-25%
     * Pattern:
       ```python
       def test_step_acceptance_rate_improvement():
           # Set up stiff ODE (van der Pol, Robertson)
           # Run with interpolant-enabled tableau
           # Count accepted vs rejected steps
           # Run with same tableau but b_interp=None (simulate old)
           # Compare acceptance rates
           # Assert improvement > 15%
       ```
   - Location: Same test file
   - Edge cases: Use multiple ODE systems for robustness
   - Integration: Validates performance benefit

5. **Create test for observables at save points**
   - File: tests/integrators/algorithms/test_dense_output_interpolants.py
   - Action: Modify
   - Details:
     * Verify observables are evaluated at correct time (next_save, not t+dt)
     * Use ODE with time-dependent observables
     * Check that observable values match expected values at save times
     * Pattern:
       ```python
       def test_observables_at_save_points():
           # Set up ODE with observable = t (or other time-dependent)
           # Run integration with dense save points
           # Extract saved observable values
           # Verify they match save times, not step times
           # Assert |observable_value - save_time| < tolerance
       ```
   - Location: Same test file
   - Edge cases: Test with driver functions that depend on time
   - Integration: Validates drivers and observables evaluated correctly

6. **Create edge case test: save point at step boundary**
   - File: tests/integrators/algorithms/test_dense_output_interpolants.py
   - Action: Modify
   - Details:
     * Test when next_save == t + dt (theta = 1.0)
     * Verify interpolant gives same result as full step
     * Pattern:
       ```python
       def test_save_at_step_boundary():
           # Set up ODE
           # Manually set dt and next_save so they align
           # Verify theta computes to 1.0
           # Verify interpolated state == full step state
       ```
   - Location: Same test file
   - Edge cases: Test with floating-point tolerance
   - Integration: Validates edge case handling

7. **Create edge case test: multiple saves per step**
   - File: tests/integrators/algorithms/test_dense_output_interpolants.py
   - Action: Modify
   - Details:
     * Test behavior when step size is large enough to span multiple saves
     * Current implementation only handles one save per step
     * This test documents expected behavior (only first save is hit)
     * Future enhancement could handle multiple saves
     * Pattern:
       ```python
       def test_multiple_saves_per_step():
           # Set up ODE with very small dt_save
           # Take step with large dt (spans multiple saves)
           # Verify only one save is recorded per step
           # Document this as expected behavior
       ```
   - Location: Same test file
   - Edge cases: May require loop logic changes for full support
   - Integration: Documents current limitation

**Outcomes**: 
[To be filled by do_task agent]

---

## Task Group 9: Documentation Updates - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Groups 1-8 (requires implementation complete)

**Required Context**:
- File: .github/active_plans/dense_output_interpolants/human_overview.md
- File: .github/active_plans/dense_output_interpolants/agent_plan.md
- File: README.md (if exists)

**Input Validation Required**:
- None (documentation only)

**Tasks**:

1. **Update ButcherTableau docstring with interpolant usage**
   - File: src/cubie/integrators/algorithms/base_algorithm_step.py
   - Action: Modify
   - Details:
     * Locate `ButcherTableau` class docstring (line ~38-74)
     * Expand `b_interp` parameter documentation:
       ```python
       b_interp
           Optional dense output polynomial coefficients for continuous
           extension. When provided, enables high-accuracy interpolation
           to arbitrary points within a completed step, eliminating error
           estimate inflation at save points.
           
           Structure: Tuple of stage weight vectors, where each vector
           provides coefficients for one power of theta in the interpolating
           polynomial. For a kth-order interpolant, b_interp contains k+1
           vectors corresponding to powers theta^0, theta^1, ..., theta^k.
           
           The interpolated solution is computed as:
               y(t + theta*dt) = y(t) + dt * sum_i sum_p b_interp[p][i] * theta^p * k_i
           
           where k_i are the stage derivatives from the Runge-Kutta step.
           
           When b_interp is provided, step functions automatically use
           interpolation to reach save points precisely while computing
           error estimates from full steps, avoiding the error inflation
           that occurs with step truncation.
       ```
   - Location: ButcherTableau class docstring, b_interp section
   - Edge cases: None
   - Integration: Helps users understand b_interp purpose

2. **Add user-facing documentation for dense output feature**
   - File: README.md or docs/features/dense_output.md
   - Action: Create or modify
   - Details:
     * Add section explaining dense output interpolants
     * Include:
       - What problem it solves (error estimate inflation)
       - Which tableaus support it (those with b_interp)
       - How to use it (automatic when tableau has b_interp)
       - Performance benefits (step acceptance rate)
     * Example:
       ```markdown
       ## Dense Output Interpolants
       
       CuBIE supports dense output (continuous extension) for Runge-Kutta
       methods with published interpolant coefficients. This feature
       eliminates error estimate inflation when integrating with frequent
       save points.
       
       ### How It Works
       
       When a tableau includes `b_interp` coefficients, CuBIE automatically:
       1. Takes full steps for accurate error estimates
       2. Interpolates to save points using polynomial interpolation
       3. Evaluates observables at exact save times
       
       ### Supported Methods
       
       - DIRK: Trapezoidal (Crank-Nicolson), Lobatto IIIC
       - FIRK: Gauss-Legendre
       - Rosenbrock: ROS3P (if available)
       
       ### Performance
       
       Dense output improves efficiency by 15-25% on problems with dense
       save points by reducing step rejections from inflated errors.
       ```
   - Location: README.md or new documentation file
   - Edge cases: None
   - Integration: User-facing documentation

3. **Update CHANGELOG or release notes**
   - File: CHANGELOG.md or HISTORY.md
   - Action: Modify
   - Details:
     * Add entry for dense output feature:
       ```markdown
       ## [Unreleased]
       
       ### Added
       - Dense output interpolants for Runge-Kutta methods with published
         coefficients. Eliminates error estimate inflation at save points,
         improving step acceptance rate by 15-25% on problems with dense
         output requirements. (#issue_number)
       
       ### Changed
       - Integration loop now passes full step size to step functions,
         with interpolation to save points handled internally.
       ```
   - Location: CHANGELOG.md, Unreleased section
   - Edge cases: None
   - Integration: Tracks feature for release

**Outcomes**: 
[To be filled by do_task agent]

---

## Summary

**Total Task Groups**: 9
**Dependency Chain**:
- Loop infrastructure (Group 1) must complete before step function changes (Groups 2-4)
- Tableau coefficients (Groups 5-7) can be done in parallel with implementation
- Tests (Group 8) require implementation + at least one tableau with b_interp
- Documentation (Group 9) should be done last

**Parallel Execution Opportunities**:
- Groups 5, 6, 7 (tableau coefficients) can run in parallel
- Groups 2, 3, 4 (step function modifications) are sequential but follow same pattern

**Estimated Complexity**:
- High complexity: Groups 2, 3, 4 (step function logic changes)
- Medium complexity: Group 1 (loop modifications), Group 8 (tests)
- Low complexity: Groups 5, 6, 7 (coefficient lookup), Group 9 (documentation)

**Critical Path**: Group 1 → Group 2 → Group 8 → Group 9

**Notes**:
- All implementation follows predicated execution pattern (no branching)
- Coefficients must come from published literature (no derivation)
- Error estimates must always come from full steps, never interpolated steps
- Backward compatibility maintained (tableaus without b_interp unchanged)
