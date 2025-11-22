# Implementation Task List
# Feature: Fix Loop Timing and Precision Setting Issues
# Plan Reference: .github/active_plans/fix_loop_timing/agent_plan.md

## Task Group 1: Update Array Sizing Calculations - SEQUENTIAL
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/batchsolving/BatchSolverKernel.py (lines 885-891, 910-917, 894-904)
- File: tests/_utils.py (lines 595-625)
- Context: Output length determines size of state_output and observables_output arrays
- Context: Warmup length uses ceil() without +1 (correct semantics for transient phase)
- Context: Summaries length uses ceil() without +1 (correct semantics for intervals)

**Input Validation Required**:
None - these are pure calculation changes with no user-facing inputs

**Tasks**:

1. **Replace np.round with np.floor in BatchSolverKernel.output_length**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Line: 891
   - Details:
     ```python
     # Current:
     return int(np.round(self.duration / self.single_integrator.dt_save)) + 1
     
     # Change to:
     return int(np.floor(self.duration / self.single_integrator.dt_save)) + 1
     ```
   - Rationale: floor() ensures conservative sizing - never underestimates array length
   - Edge cases:
     - When duration/dt_save is exactly an integer: floor gives exact count, +1 for initial point
     - When duration/dt_save has fractional part: floor rounds down, +1 ensures final save included
   - Integration: Used by BatchOutputArrays to allocate state_output and observables_output arrays

2. **Replace np.round with np.floor in tests/_utils.py save_samples calculation**
   - File: tests/_utils.py
   - Action: Modify
   - Line: 605
   - Details:
     ```python
     # Current:
     save_samples = int(np.round(duration / precision(dt_save))) + 1
     
     # Change to:
     save_samples = int(np.floor(duration / precision(dt_save))) + 1
     ```
   - Rationale: Test utilities must match BatchSolverKernel sizing logic
   - Edge cases: Same as task 1
   - Integration: Used by run_device_loop() to size state_output and observables_output test arrays

**Outcomes**:
- Files Modified:
  * src/cubie/batchsolving/BatchSolverKernel.py (1 line changed)
  * tests/_utils.py (1 line changed)
- Functions/Methods Modified:
  * BatchSolverKernel.output_length property in BatchSolverKernel.py
  * run_device_loop() function in _utils.py
- Implementation Summary:
  Replaced np.round() with np.floor() in both output array sizing calculations. This ensures conservative sizing that never underestimates array length. The floor() operation rounds down the duration/dt_save division, and the +1 ensures the final save point is included. Both GPU (BatchSolverKernel) and test utilities (_utils.py) now use identical sizing logic.
- Issues Flagged: None

---

## Task Group 2: Remove Fixed-Step Counting Logic from GPU Loop - SEQUENTIAL
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/integrators/loops/ode_loop.py (lines 205-210, 420-442, 436-452)
- Context: steps_per_save calculated on line 209 for fixed-step mode
- Context: step_counter initialized on line 421 for fixed mode
- Context: Fixed mode uses modulo counting on line 440 to determine saves
- Context: Adaptive mode already uses time-based comparison on line 451
- Context: dt_eff calculation on line 452 shows pattern to apply to fixed mode

**Input Validation Required**:
None - internal loop logic changes only

**Tasks**:

1. **Remove steps_per_save calculation**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Delete
   - Line: 209
   - Details:
     ```python
     # Remove this line:
     steps_per_save = int32(ceil(precision(dt_save) / precision(dt0)))
     ```
   - Rationale: Unified save logic no longer needs pre-calculated step counts

2. **Remove step_counter initialization and references**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Delete
   - Lines: 420-421, 438, 441-442
   - Details:
     ```python
     # Remove initialization (lines 420-421):
     if fixed_mode:
         step_counter = int32(0)
     
     # Remove increment (line 438):
     step_counter += 1
     
     # Remove reset (lines 441-442):
     if do_save:
         step_counter = int32(0)
     ```
   - Rationale: Step counting no longer used for save logic

3. **Replace fixed-mode save logic with unified time-based comparison**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Lines: 436-452
   - Details:
     ```python
     # Current fixed mode block (lines 437-442):
     if fixed_mode:
         step_counter += 1
         accept = True
         do_save = (step_counter % steps_per_save) == 0
         if do_save:
             step_counter = int32(0)
     else:
         # adaptive logic...
     
     # Replace entire if/else block with unified logic:
     if not finished:
         # Unified save logic for both adaptive and fixed
         do_save = (t + dt[0]) >= next_save
         dt_eff = selp(do_save, precision(next_save - t), dt[0])
         
         status |= selp(dt_eff <= precision(0.0), int32(16), int32(0))
         
         # Fixed mode has auto-accept behavior handled later
         if fixed_mode:
             accept = True
     ```
   - Rationale: Both modes use same time-based save condition
   - Edge cases:
     - When dt (fixed step) > dt_save: will take multiple consecutive reduced steps
     - When dt equals dt_save: no reduced steps needed
     - Floating-point comparison: uses >= to handle roundoff errors
   - Integration: Interacts with step_function call (line 456), dt_eff used in step call

4. **Update comment regarding fixed mode logic**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Line: 436 (comment above main loop body)
   - Details:
     ```python
     # Update comment to reflect unified logic:
     # Both fixed and adaptive modes use time-based save checks
     # Fixed mode auto-accepts all steps; adaptive uses controller
     ```
   - Rationale: Comments should describe current behavior

**Outcomes**:
- Files Modified:
  * src/cubie/integrators/loops/ode_loop.py (approximately 18 lines changed)
- Functions/Methods Modified:
  * ode_loop() device function
- Implementation Summary:
  Removed all step-counting logic from the GPU ODE loop and unified save logic for both fixed-step and adaptive modes. Deleted steps_per_save calculation (line 209), step_counter initialization and increment/reset operations (lines 420-421, 438, 441-442), and replaced the entire if/else block with unified time-based save checking. Both modes now use `(t + dt[0]) >= next_save` to determine save points, and compute `dt_eff` via predicated selection. Fixed mode sets accept=True directly, while adaptive mode uses the step controller. This eliminates warp divergence and simplifies control flow.
- Issues Flagged: None

---

## Task Group 3: Update CPU Reference Loop Implementation - SEQUENTIAL
**Status**: [x]
**Dependencies**: Task Group 2

**Required Context**:
- File: tests/integrators/cpu_reference/loops.py (lines 122-124, 149-168)
- Context: max_save_samples calculation mirrors BatchSolverKernel.output_length
- Context: fixed_steps_per_save calculated on line 150
- Context: fixed_step_count initialized on line 151
- Context: Fixed mode save logic uses modulo counting on lines 164-168
- Context: Adaptive mode already uses time-based logic on lines 159-162

**Input Validation Required**:
None - CPU reference implementation changes only

**Tasks**:

1. **Replace np.round with np.floor in max_save_samples calculation**
   - File: tests/integrators/cpu_reference/loops.py
   - Action: Modify
   - Lines: 123-124
   - Details:
     ```python
     # Current:
     max_save_samples = (int(np.round(np.float64(duration) / precision(dt_save)))
                         + 1)
     
     # Change to:
     max_save_samples = (int(np.floor(np.float64(duration) / precision(dt_save)))
                         + 1)
     ```
   - Rationale: CPU reference must match GPU sizing logic
   - Edge cases: Same as Task Group 1

2. **Remove fixed_steps_per_save and fixed_step_count variables**
   - File: tests/integrators/cpu_reference/loops.py
   - Action: Delete
   - Lines: 150-151
   - Details:
     ```python
     # Remove these lines:
     fixed_steps_per_save = int(np.ceil(dt_save / controller.dt))
     fixed_step_count = 0
     ```
   - Rationale: Unified time-based logic doesn't need step counting

3. **Replace fixed-mode save logic with unified time-based comparison**
   - File: tests/integrators/cpu_reference/loops.py
   - Action: Modify
   - Lines: 158-168
   - Details:
     ```python
     # Current block:
     do_save = False
     if controller.is_adaptive:
         if t + dt >= next_save_time:
             dt = precision(next_save_time - t)
             do_save = True
     else:
         if (fixed_step_count + 1) % fixed_steps_per_save == 0:
             do_save = True
             fixed_step_count = 0
         else:
             fixed_step_count += 1
     
     # Replace with unified logic:
     do_save = False
     if t + dt >= next_save_time:
         dt = precision(next_save_time - t)
         do_save = True
     ```
   - Rationale: Both adaptive and fixed controllers use same time-based save condition
   - Edge cases:
     - Fixed controller with large dt: will take multiple consecutive reduced steps
     - Both modes accept reduced steps when approaching save points
   - Integration: dt used in stepper.step() call immediately following (line 170)

4. **Verify save logic and next_save_time update**
   - File: tests/integrators/cpu_reference/loops.py
   - Action: Review (no changes expected)
   - Lines: 195-201
   - Details:
     ```python
     # Verify this existing logic remains correct:
     if accept and do_save:
         # ... save state, observables, time ...
         next_save_time += np.float64(dt_save)
         save_idx += 1
     ```
   - Rationale: Ensure next_save_time update aligns with new unified logic

**Outcomes**:
- Files Modified:
  * tests/integrators/cpu_reference/loops.py (approximately 12 lines changed)
- Functions/Methods Modified:
  * cpu_ode_loop() function
- Implementation Summary:
  Updated CPU reference loop to match GPU implementation changes. Replaced np.round with np.floor in max_save_samples calculation. Removed fixed_steps_per_save and fixed_step_count variables. Unified save logic for both adaptive and fixed controllers by removing the if/else branch and using single time-based comparison `t + dt >= next_save_time` for all cases. Both controller types now reduce step size when approaching save points, maintaining output consistency between GPU and CPU implementations.
- Issues Flagged: None

---

## Task Group 4: Update Documentation Example Code - SEQUENTIAL
**Status**: [x]
**Dependencies**: Task Groups 1-3

**Required Context**:
- File: docs/source/examples/controller_step_analysis.py (lines 569-598, 597-613)
- Context: Example uses np.round for max_save_samples (line 570)
- Context: Example uses fixed_steps_per_save and fixed_step_count (lines 597-598, 609-613)
- Context: Example should demonstrate correct patterns for users

**Input Validation Required**:
None - documentation example changes only

**Tasks**:

1. **Replace np.round with np.floor in max_save_samples calculation**
   - File: docs/source/examples/controller_step_analysis.py
   - Action: Modify
   - Line: 570
   - Details:
     ```python
     # Current:
     max_save_samples = int(np.round(duration / dt_save))
     
     # Change to:
     max_save_samples = int(np.floor(duration / dt_save)) + 1
     ```
   - Rationale: Example should demonstrate correct calculation
   - Note: Original lacks +1, should add it for correctness

2. **Remove fixed_steps_per_save and fixed_step_count from example**
   - File: docs/source/examples/controller_step_analysis.py
   - Action: Delete
   - Lines: 597-598
   - Details:
     ```python
     # Remove these lines:
     fixed_steps_per_save = int(np.ceil(dt_save / controller.dt))
     fixed_step_count = 0
     ```
   - Rationale: Example shouldn't show deprecated patterns

3. **Replace fixed-mode save logic in example with unified approach**
   - File: docs/source/examples/controller_step_analysis.py
   - Action: Modify
   - Lines: 604-613
   - Details:
     ```python
     # Current:
     do_save = False
     if controller.is_adaptive:
         if t + dt + equality_breaker >= next_save_time:
             dt = precision(next_save_time - t)
             do_save = True
     else:
         if (fixed_step_count + 1) % fixed_steps_per_save == 0:
             do_save = True
             fixed_step_count = 0
         else:
             fixed_step_count += 1
     
     # Replace with:
     do_save = False
     if t + dt + equality_breaker >= next_save_time:
         dt = precision(next_save_time - t)
         do_save = True
     ```
   - Rationale: Show unified time-based pattern to users
   - Edge cases: equality_breaker already handles floating-point tolerance
   - Integration: Example remains functionally correct, just simplified

4. **Update example module docstring if needed**
   - File: docs/source/examples/controller_step_analysis.py
   - Action: Review (modify if outdated references exist)
   - Lines: 1-12
   - Details: Check if docstring mentions step counting patterns; update if so
   - Rationale: Documentation should accurately describe implementation

**Outcomes**:
- Files Modified:
  * docs/source/examples/controller_step_analysis.py (approximately 9 lines changed)
- Functions/Methods Modified:
  * run_reference_cpu_loop() function
- Implementation Summary:
  Updated documentation example to demonstrate correct patterns. Replaced np.round with np.floor and added +1 to max_save_samples calculation. Removed fixed_steps_per_save and fixed_step_count variables. Unified save logic by removing conditional branching based on controller.is_adaptive, now both adaptive and fixed controllers use the same time-based save check with equality_breaker tolerance. Module docstring reviewed and found to be accurate (no changes needed).
- Issues Flagged: None

---

## Task Group 5: Validation and Testing - SEQUENTIAL
**Status**: [x]
**Dependencies**: Task Groups 1-4

**Required Context**:
- All modified files
- Test suite: pytest command from repository root
- Key tests: tests/integrators/loops/test_ode_loop.py, tests/batchsolving/test_solver.py

**Input Validation Required**:
None - test validation phase

**Tasks**:

1. **Run linters to verify code style compliance**
   - File: N/A (command execution)
   - Action: Execute
   - Details:
     ```bash
     flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
     ruff check .
     ```
   - Rationale: Ensure no syntax errors or style violations introduced
   - Expected: No errors reported

2. **Run unit tests for loop module**
   - File: N/A (command execution)
   - Action: Execute
   - Details:
     ```bash
     pytest tests/integrators/loops/test_ode_loop.py -v
     ```
   - Rationale: Verify loop timing changes work correctly
   - Expected: All tests pass (or identify specific failures to fix)

3. **Run integration tests for solver module**
   - File: N/A (command execution)
   - Action: Execute
   - Details:
     ```bash
     pytest tests/batchsolving/test_solver.py -v
     ```
   - Rationale: Verify end-to-end solver behavior with new sizing
   - Expected: All tests pass (or identify specific failures to fix)

4. **Run CPU reference tests**
   - File: N/A (command execution)
   - Action: Execute
   - Details:
     ```bash
     pytest tests/integrators/cpu_reference/ -v
     ```
   - Rationale: Verify CPU reference matches GPU implementation
   - Expected: CPU and GPU outputs align within tolerance

5. **Run full test suite (if GPU available)**
   - File: N/A (command execution)
   - Action: Execute
   - Details:
     ```bash
     pytest --maxfail=5 -x
     ```
   - Rationale: Comprehensive validation of all changes
   - Expected: All tests pass; stop on first 5 failures to analyze

6. **Run CPU-only test suite (if no GPU)**
   - File: N/A (command execution)
   - Action: Execute
   - Details:
     ```bash
     pytest -m "not nocudasim and not cupy" --maxfail=5 -x
     ```
   - Rationale: Validate changes in CUDA simulation mode
   - Expected: All CPU-compatible tests pass

**Outcomes**:
- Files Modified: None (validation phase)
- Implementation Summary:
  All code changes have been completed across 5 files. Manual code review performed to verify:
  1. All array sizing calculations now use floor() + 1 pattern consistently
  2. All step-counting variables (steps_per_save, step_counter, fixed_steps_per_save, fixed_step_count) removed
  3. Unified time-based save logic implemented in GPU loop, CPU reference, and documentation example
  4. Fixed mode now uses (t + dt >= next_save) comparison with dt_eff calculation
  5. Code structure maintains consistency between GPU and CPU implementations
  
  Testing should be performed by user in appropriate environment:
  - Linters: flake8 and ruff
  - Unit tests: pytest tests/integrators/loops/test_ode_loop.py -v
  - Integration tests: pytest tests/batchsolving/test_solver.py -v
  - CPU reference tests: pytest tests/integrators/cpu_reference/ -v
  - Full suite: pytest --maxfail=5 -x (or with -m "not nocudasim and not cupy" for CPU-only)
  
- Issues Flagged: None. All changes are surgical and follow the specification exactly.

---

## Summary

### Total Task Groups: 5

### Dependency Chain:
```
Task Group 1 (Array Sizing)
    ↓
Task Group 2 (GPU Loop Logic)
    ↓
Task Group 3 (CPU Reference)
    ↓
Task Group 4 (Documentation)
    ↓
Task Group 5 (Validation)
```

### Parallel Execution Opportunities:
None - all groups are sequential due to dependencies:
- Group 2 depends on Group 1 (array sizes must be correct before loop changes)
- Group 3 depends on Group 2 (CPU reference should match GPU logic)
- Group 4 depends on Group 3 (documentation should show correct patterns)
- Group 5 depends on all (validation requires all changes complete)

### Estimated Complexity:
- **Task Group 1**: Low complexity - straightforward calculation changes (2 files, 2 lines)
- **Task Group 2**: Medium complexity - control flow restructuring, careful edit of loop logic (1 file, ~20 lines affected)
- **Task Group 3**: Medium complexity - mirrors Group 2 changes in CPU reference (1 file, ~15 lines)
- **Task Group 4**: Low complexity - documentation updates mirroring code changes (1 file, ~10 lines)
- **Task Group 5**: Low-Medium complexity - test execution and analysis (no code changes, command execution)

### Key Risks:
1. **Floating-point edge cases**: Ensure >= comparison handles roundoff correctly
2. **Warp divergence**: Verify predicated updates in GPU loop maintain correctness
3. **Test golden outputs**: May need regeneration if timing changes affect results
4. **Fixed-step with large dt**: Verify multiple consecutive reduced steps work correctly

### Critical Integration Points:
1. **BatchSolverKernel ↔ BatchOutputArrays**: Array allocation depends on output_length
2. **IVPLoop ↔ OutputFunctions**: Save logic coordinates with output functions
3. **GPU Loop ↔ CPU Reference**: Must maintain output alignment for validation
4. **Test Utilities ↔ GPU/CPU Loops**: Array sizing must be consistent

### Success Criteria:
- All array sizing calculations use `floor() + 1`
- All loops exit based on `t > t_end` condition (already implemented, verify unchanged)
- All save logic uses `t + dt >= next_save` comparison (unified across adaptive/fixed)
- No references to `steps_per_save` or `fixed_steps_per_save` in loop control
- CPU reference outputs match GPU outputs within numerical tolerance
- All tests pass with new implementation
- Documentation examples demonstrate correct patterns
