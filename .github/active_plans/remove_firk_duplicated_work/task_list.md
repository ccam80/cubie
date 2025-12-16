# Implementation Task List
# Feature: Remove Duplicated Work in FIRK Device Functions
# Plan Reference: .github/active_plans/remove_firk_duplicated_work/agent_plan.md

## Task Group 1: Update Production FIRK Step - [SEQUENTIAL]
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_firk.py (lines 764-868)
- The post-solve stage loop (lines 764-821) currently:
  1. Reconstructs stage states `Y_i = y_n + Σ a_ij * u_j`
  2. Calls `observables_function()` and `dxdt_fn()` to compute `stage_rhs_flat`
  3. These are redundant when accumulating because `stage_increment` contains `h * k_i`

**Input Validation Required**:
- None - this is a pure refactoring that changes internal logic only

**Tasks**:

1. **Restructure Post-Solve Stage Loop (lines 764-821)**
   - File: src/cubie/integrators/algorithms/generic_firk.py
   - Action: Modify
   - Details:
     ```python
     # The stage loop should ONLY execute stage state reconstruction
     # when needed for stiffly-accurate shortcuts (not accumulates_output
     # or not accumulates_error).
     #
     # Current structure reconstructs ALL stages and evaluates dxdt_fn
     # for accumulation paths - this is the redundancy to eliminate.
     #
     # NEW LOGIC:
     # - Loop only reconstructs stage state when:
     #   (not accumulates_output and b_row == stage_idx) OR
     #   (not accumulates_error and b_hat_row == stage_idx)
     # - Remove observables_function() call inside loop
     # - Remove dxdt_fn() call inside loop
     # - Remove do_more_work logic entirely
     #
     # Implementation:
     for stage_idx in range(stage_count):
         # Only reconstruct stage state for stiffly-accurate shortcuts
         needs_stage_state = (
             (not accumulates_output and b_row == stage_idx) or
             (not accumulates_error and b_hat_row == stage_idx)
         )
         
         if needs_stage_state:
             for idx in range(n):
                 value = state[idx]
                 for contrib_idx in range(stage_count):
                     flat_idx = stage_idx * stage_count + contrib_idx
                     coeff = stage_rhs_coeffs[flat_idx]
                     if coeff != typed_zero:
                         value += (
                             coeff * stage_increment[contrib_idx * n + idx]
                         )
                 stage_state[idx] = value

             if not accumulates_output and b_row == stage_idx:
                 for idx in range(n):
                     proposed_state[idx] = stage_state[idx]
             if not accumulates_error and b_hat_row == stage_idx:
                 for idx in range(n):
                     error[idx] = stage_state[idx]
     ```
   - Edge cases:
     - Single-stage FIRK: Loop runs once, still works correctly
     - Both b_row and b_hat_row match same stage: Both assignments execute
     - Neither b_row nor b_hat_row match: Loop body skipped entirely
   - Integration: This replaces lines 764-821 in the current implementation

2. **Update Output Accumulation (lines 824-837)**
   - File: src/cubie/integrators/algorithms/generic_firk.py
   - Action: Modify
   - Details:
     ```python
     # Change source from stage_rhs_flat to stage_increment
     # Remove dt_scalar multiplication since stage_increment = h * k_i
     #
     # BEFORE:
     # if accumulates_output:
     #     for idx in range(n):
     #         ...
     #         rhs_value = stage_rhs_flat[stage_idx * n + idx]
     #         ...
     #         proposed_state[idx] = state[idx] + solution_acc * dt_scalar
     #
     # AFTER:
     if accumulates_output:
         for idx in range(n):
             solution_acc = typed_zero
             compensation = typed_zero
             for stage_idx in range(stage_count):
                 increment_value = stage_increment[stage_idx * n + idx]
                 term = (solution_weights[stage_idx] * increment_value -
                         compensation)
                 temp = solution_acc + term
                 compensation = (temp - solution_acc) - term
                 solution_acc += solution_weights[stage_idx] * increment_value
             proposed_state[idx] = state[idx] + solution_acc  # No dt_scalar
     ```
   - Edge cases:
     - Zero stage increments: Results in `proposed_state = state` (correct)
     - Very small increments: Kahan summation preserves precision
   - Integration: Preserves Kahan summation algorithm for numerical stability

3. **Update Error Accumulation (lines 839-846)**
   - File: src/cubie/integrators/algorithms/generic_firk.py
   - Action: Modify
   - Details:
     ```python
     # Change source from stage_rhs_flat to stage_increment
     # Remove dt_scalar multiplication since stage_increment = h * k_i
     #
     # BEFORE:
     # if has_error and accumulates_error:
     #     for idx in range(n):
     #         ...
     #         rhs_value = stage_rhs_flat[stage_idx * n + idx]
     #         error_acc += error_weights[stage_idx] * rhs_value
     #     error[idx] = dt_scalar * error_acc
     #
     # AFTER:
     if has_error and accumulates_error:
         for idx in range(n):
             error_acc = typed_zero
             for stage_idx in range(stage_count):
                 increment_value = stage_increment[stage_idx * n + idx]
                 error_acc += error_weights[stage_idx] * increment_value
             error[idx] = error_acc  # No dt_scalar multiplication
     ```
   - Edge cases:
     - Zero error weights: Results in zero error (correct)
     - All stages have same increment: Sum is weighted average of increments
   - Integration: Follows same pattern as output accumulation

4. **Remove Unused stage_rhs_flat Assignment**
   - File: src/cubie/integrators/algorithms/generic_firk.py
   - Action: Modify (optional cleanup)
   - Details:
     ```python
     # Line 725-726 currently has:
     # stage_rhs_flat = solver_scratch[:all_stages_n]
     #
     # This slice is no longer needed for accumulation since we use
     # stage_increment directly. However, it may still be used by
     # the nonlinear solver internally. Verify solver usage before removing.
     #
     # If stage_rhs_flat is only used for post-solve accumulation,
     # this line can be removed. Otherwise, keep it.
     #
     # RECOMMENDATION: Keep the line for now - it doesn't cause harm
     # and may be used by the solver scratch space.
     ```
   - Edge cases: None
   - Integration: This is optional cleanup; functionality works either way

**Outcomes**: 
- Files Modified:
  * src/cubie/integrators/algorithms/generic_firk.py (~50 lines changed)
- Functions/Methods Modified:
  * `firk_step()` - Post-solve stage loop restructured, output/error accumulation updated
- Implementation Summary:
  * Task 1.1: Restructured post-solve stage loop to only reconstruct stage states when needed for stiffly-accurate shortcuts (lines 764-791)
  * Task 1.2: Updated output accumulation to use `stage_increment` directly with Kahan summation, removed `dt_scalar` multiplication (lines 794-809)
  * Task 1.3: Updated error accumulation to use `stage_increment` directly, removed `dt_scalar` multiplication (lines 811-817)
  * Task 1.4: Left `stage_rhs_flat` variable in place as recommended - no action taken
- Issues Flagged: None

---

## Task Group 2: Update Instrumented Test FIRK Step - [SEQUENTIAL]
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: tests/integrators/algorithms/instrumented/generic_firk.py (lines 457-571)
- This file mirrors generic_firk.py but adds instrumentation logging arrays
- Changes must match Task Group 1 exactly, preserving instrumentation

**Input Validation Required**:
- None - this is a pure refactoring matching production code

**Tasks**:

1. **Restructure Post-Solve Stage Loop (lines 457-526)**
   - File: tests/integrators/algorithms/instrumented/generic_firk.py
   - Action: Modify
   - Details:
     ```python
     # Mirror the changes from Task Group 1, Task 1
     # PRESERVE instrumentation logging that records stage_states,
     # stage_increments, jacobian_updates, and stage_observables
     #
     # The current instrumentation records values even when do_more_work
     # is executed. After the change, some of these may not be populated.
     # 
     # IMPORTANT: Keep instrumentation logging for stage_increments since
     # that data is still valid. Only remove logging for values that
     # depended on the removed dxdt_fn calls.
     #
     # NEW LOGIC (with instrumentation):
     for stage_idx in range(stage_count):
         # Log stage increments (always valid after Newton solve)
         for idx in range(n):
             stage_increments[stage_idx, idx] = (
                 stage_increment[stage_idx * n + idx]
             )

         # Only reconstruct stage state for stiffly-accurate shortcuts
         needs_stage_state = (
             (not accumulates_output and b_row == stage_idx) or
             (not accumulates_error and b_hat_row == stage_idx)
         )
         
         if needs_stage_state:
             for idx in range(n):
                 value = state[idx]
                 for contrib_idx in range(stage_count):
                     coeff = stage_rhs_coeffs[
                         stage_idx * stage_count + contrib_idx
                     ]
                     if coeff != typed_zero:
                         value += (
                             coeff * stage_increment[contrib_idx * n + idx]
                         )
                 stage_state[idx] = value

             # Log reconstructed stage state
             for idx in range(n):
                 stage_states[stage_idx, idx] = stage_state[idx]

             if not accumulates_output and b_row == stage_idx:
                 for idx in range(n):
                     proposed_state[idx] = stage_state[idx]
             if not accumulates_error and b_hat_row == stage_idx:
                 for idx in range(n):
                     error[idx] = stage_state[idx]
     ```
   - Edge cases: Same as Task Group 1, Task 1
   - Integration: Instrumentation arrays are passed from test harness

2. **Update Output Accumulation (lines 528-541)**
   - File: tests/integrators/algorithms/instrumented/generic_firk.py
   - Action: Modify
   - Details:
     ```python
     # Mirror changes from Task Group 1, Task 2
     if accumulates_output:
         for idx in range(n):
             solution_acc = typed_zero
             compensation = typed_zero
             for stage_idx in range(stage_count):
                 increment_value = stage_increment[stage_idx * n + idx]
                 term = (solution_weights[stage_idx] * increment_value -
                         compensation)
                 temp = solution_acc + term
                 compensation = (temp - solution_acc) - term
                 solution_acc += solution_weights[stage_idx] * increment_value
             proposed_state[idx] = state[idx] + solution_acc  # No dt_scalar
     ```
   - Edge cases: Same as Task Group 1, Task 2
   - Integration: No new instrumentation needed for accumulation

3. **Update Error Accumulation (lines 543-550)**
   - File: tests/integrators/algorithms/instrumented/generic_firk.py
   - Action: Modify
   - Details:
     ```python
     # Mirror changes from Task Group 1, Task 3
     if has_error and accumulates_error:
         for idx in range(n):
             error_acc = typed_zero
             for stage_idx in range(stage_count):
                 increment_value = stage_increment[stage_idx * n + idx]
                 error_acc += error_weights[stage_idx] * increment_value
             error[idx] = error_acc  # No dt_scalar multiplication
     ```
   - Edge cases: Same as Task Group 1, Task 3
   - Integration: No new instrumentation needed for accumulation

**Outcomes**: 
- Files Modified:
  * tests/integrators/algorithms/instrumented/generic_firk.py (~55 lines changed)
- Functions/Methods Modified:
  * `step()` inside `build()` - Post-solve stage loop restructured, output/error accumulation updated
- Implementation Summary:
  * Task 2.1: Restructured post-solve stage loop with same pattern as production code, preserved stage_increments instrumentation logging
  * Task 2.2: Updated output accumulation to use `stage_increment` directly with Kahan summation, removed `dt_scalar` multiplication
  * Task 2.3: Updated error accumulation to use `stage_increment` directly, removed `dt_scalar` multiplication
- Issues Flagged: None

---

## Task Group 3: Update All-in-One Debug FIRK Step - [SEQUENTIAL]
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: tests/all_in_one.py (lines 2499-2595 in firk_step_inline_factory)
- This is a standalone debug version for Numba lineinfo debugging
- Changes must match Task Group 1 logic

**Input Validation Required**:
- None - this is a pure refactoring matching production code

**Tasks**:

1. **Restructure Post-Solve Stage Loop (lines 2499-2550)**
   - File: tests/all_in_one.py
   - Action: Modify
   - Details:
     ```python
     # Mirror changes from Task Group 1, Task 1
     # The all_in_one version uses slightly different variable naming
     # but the same logic structure
     #
     # BEFORE (lines 2499-2550):
     # for stage_idx in range(stage_count):
     #     stage_time = ...
     #     if has_driver_function: ...
     #     for idx in range(n):  # Stage state reconstruction
     #         value = state[idx]
     #         ...
     #         stage_state[idx] = value
     #     if not accumulates_output:
     #         if b_row == stage_idx: ...
     #     if not accumulates_error:
     #         if b_hat_row == stage_idx: ...
     #     do_more_work = ...
     #     if do_more_work:
     #         observables_function(...)
     #         stage_rhs = ...
     #         dxdt_fn(...)
     #
     # AFTER:
     for stage_idx in range(stage_count):
         # Only reconstruct stage state for stiffly-accurate shortcuts
         needs_stage_state = (
             (not accumulates_output and b_row == stage_idx) or
             (not accumulates_error and b_hat_row == stage_idx)
         )
         
         if needs_stage_state:
             for idx in range(n):
                 value = state[idx]
                 for contrib_idx in range(stage_count):
                     flat_idx = stage_idx * stage_count + contrib_idx
                     coeff = stage_rhs_coeffs[flat_idx]
                     if coeff != typed_zero:
                         value += coeff * stage_increment[contrib_idx * n + idx]
                 stage_state[idx] = value

             if not accumulates_output and b_row == stage_idx:
                 for idx in range(n):
                     proposed_state[idx] = stage_state[idx]
             if not accumulates_error and b_hat_row == stage_idx:
                 for idx in range(n):
                     error[idx] = stage_state[idx]
     ```
   - Edge cases: Same as Task Group 1, Task 1
   - Integration: Part of the debug file, no external dependencies

2. **Update Output Accumulation (lines 2552-2564)**
   - File: tests/all_in_one.py
   - Action: Modify
   - Details:
     ```python
     # Mirror changes from Task Group 1, Task 2
     # Note: all_in_one uses slightly different Kahan implementation
     #
     # BEFORE:
     # if accumulates_output:
     #     for idx in range(n):
     #         ...
     #         rhs_value = stage_rhs_flat[stage_idx * n + idx]
     #         weighted = solution_weights[stage_idx] * rhs_value
     #         ...
     #         proposed_state[idx] = state[idx] + solution_acc * dt_scalar
     #
     # AFTER:
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
             proposed_state[idx] = state[idx] + solution_acc  # No dt_scalar
     ```
   - Edge cases: Same as Task Group 1, Task 2
   - Integration: Part of the debug file, maintains same Kahan pattern

3. **Update Error Accumulation (lines 2566-2577)**
   - File: tests/all_in_one.py
   - Action: Modify
   - Details:
     ```python
     # Mirror changes from Task Group 1, Task 3
     # Note: all_in_one uses Kahan for error too
     #
     # BEFORE:
     # if has_error and accumulates_error:
     #     for idx in range(n):
     #         ...
     #         rhs_value = stage_rhs_flat[stage_idx * n + idx]
     #         weighted = error_weights[stage_idx] * rhs_value
     #         ...
     #         error[idx] = dt_scalar * error_acc
     #
     # AFTER:
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
             error[idx] = error_acc  # No dt_scalar multiplication
     ```
   - Edge cases: Same as Task Group 1, Task 3
   - Integration: Part of the debug file

**Outcomes**: 
- Files Modified:
  * tests/all_in_one.py (~45 lines changed)
- Functions/Methods Modified:
  * `step()` inside `firk_step_inline_factory()` - Post-solve stage loop restructured, output/error accumulation updated
- Implementation Summary:
  * Task 3.1: Restructured post-solve stage loop with same pattern as production code
  * Task 3.2: Updated output accumulation to use `stage_increment` directly with Kahan summation, removed `dt_scalar` multiplication
  * Task 3.3: Updated error accumulation to use `stage_increment` directly with Kahan summation, removed `dt_scalar` multiplication
- Issues Flagged: None

---

## Task Group 4: Run Verification Tests - [SEQUENTIAL]
**Status**: [ ]
**Dependencies**: Task Groups 1, 2, 3

**Required Context**:
- Existing test files that exercise FIRK algorithms
- Tests should pass without modification if changes are correct

**Input Validation Required**:
- None - tests validate the implementation

**Tasks**:

1. **Run Linting**
   - Action: Execute
   - Details:
     ```bash
     flake8 src/cubie/integrators/algorithms/generic_firk.py --count --select=E9,F63,F7,F82 --show-source --statistics
     flake8 tests/integrators/algorithms/instrumented/generic_firk.py --count --select=E9,F63,F7,F82 --show-source --statistics
     flake8 tests/all_in_one.py --count --select=E9,F63,F7,F82 --show-source --statistics
     ```
   - Edge cases: None
   - Integration: Standard CI linting check

2. **Run FIRK-Related Tests**
   - Action: Execute
   - Details:
     ```bash
     # Run tests that exercise FIRK algorithms
     pytest tests/integrators/algorithms/test_last_step_caching_integration.py -v
     pytest tests/integrators/algorithms/test_step_algorithms.py -v -k "firk or FIRK"
     
     # If available, run broader integration tests
     pytest tests/ -v -k "firk or FIRK" --ignore=tests/all_in_one.py
     ```
   - Edge cases:
     - CUDA simulator mode: Some tests may be marked nocudasim
     - If tests fail, verify numerical equivalence is maintained
   - Integration: Uses pytest fixtures from conftest.py

**Outcomes**: 
- Implementation Complete - Manual verification required by user
- Linting and tests could not be executed (bash tool not available)
- User should run:
  ```bash
  flake8 src/cubie/integrators/algorithms/generic_firk.py tests/integrators/algorithms/instrumented/generic_firk.py tests/all_in_one.py --count --select=E9,F63,F7,F82 --show-source --statistics
  pytest tests/integrators/algorithms/test_firk_step.py -v -x
  ```
- Issues Flagged: None

---

## Summary

### Total Task Groups: 4
### Dependency Chain:
```
Task Group 1 (Production FIRK)
       ↓
       ├─→ Task Group 2 (Instrumented FIRK) 
       └─→ Task Group 3 (All-in-One FIRK)
              ↓
       Task Group 4 (Verification)
```

### Parallel Execution Opportunities:
- Task Groups 2 and 3 can execute in PARALLEL after Group 1 completes
- Task Group 4 must wait for all prior groups

### Estimated Complexity:
- **Task Group 1**: Medium - Core logic changes with careful handling of edge cases
- **Task Group 2**: Low - Direct mirror of Group 1 with instrumentation preserved
- **Task Group 3**: Low - Direct mirror of Group 1 in standalone file
- **Task Group 4**: Low - Standard verification using existing tests

### Key Mathematical Insight:
At Newton convergence for FIRK: `u_i = h * f(Y_i) = h * k_i`

Therefore:
- `proposed_state = state + Σ b_i * stage_increment[i]` (NO dt_scalar)
- `error = Σ d_i * stage_increment[i]` (NO dt_scalar)

This eliminates:
1. Post-solve stage state reconstruction (except for stiffly-accurate shortcuts)
2. Post-solve `observables_function()` calls
3. Post-solve `dxdt_fn()` calls
4. The `dt_scalar` multiplication in accumulation formulas
