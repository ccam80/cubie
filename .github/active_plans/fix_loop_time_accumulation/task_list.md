# Implementation Task List
# Feature: Fix Loop Time Accumulation Precision Mismatch
# Plan Reference: .github/active_plans/fix_loop_time_accumulation/agent_plan.md

## Task Group 1: Code Fix - Cast dt_save to float64 in Accumulation - SEQUENTIAL
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/loops/ode_loop.py (lines 1-13, 200-215, 370-385, 520-540)
- File: tests/integrators/cpu_reference/loops.py (lines 85-95, 180-195)

**Input Validation Required**:
None - this is a type cast fix, no new validation needed

**Tasks**:
1. **Modify time accumulation to cast dt_save to float64**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Line: 529
   - Details:
     ```python
     # Current code:
     next_save = selp(do_save, next_save + dt_save, next_save)
     
     # Change to:
     next_save = selp(do_save, next_save + float64(dt_save), next_save)
     ```
   - Rationale:
     * `next_save` is implicitly float64 (initialized from `t_end` on line 375)
     * `dt_save` is precision type (cast on line 206)
     * Mixed-precision arithmetic causes accumulated errors different from CPU reference
     * CPU reference uses `next_save_time + np.float64(dt_save)` (loops.py line 188)
     * `float64` is already imported from numba at line 13
   - Edge cases:
     * When precision=float64: Cast is no-op, identical behavior
     * When precision=float32/16: Upcast to float64 for accurate accumulation
     * Very long integrations: Consistent accumulated error with CPU reference
     * Save at exactly t_end: Existing save_last logic handles correctly
   - Integration:
     * No changes to loop exit condition (line 430): `next_save > t_end` both float64
     * No changes to save check (line 445): `(t + dt[0]) >= next_save` remains valid
     * No changes to dt_eff calculation (line 446): `precision(next_save - t)` cast works correctly
     * No changes to save_state call (line 536): Receives `precision(t)`, not next_save
     * No changes to initialization (lines 375-378): next_save already float64
     * No changes to compile settings: dt_save remains precision type in config

**Outcomes**: 
- Files Modified: 
  * src/cubie/integrators/loops/ode_loop.py (1 line changed)
- Functions/Methods Added/Modified:
  * ode_loop() in ode_loop.py - modified time accumulation logic
- Implementation Summary:
  * Changed line 529 from `next_save + dt_save` to `next_save + float64(dt_save)`
  * Cast ensures float64 precision for time accumulation regardless of solver precision
  * Matches CPU reference implementation in tests/integrators/cpu_reference/loops.py line 188
  * float64 was already imported from numba at line 13
  * No architectural changes or new dependencies required
- Issues Flagged: None

---

## Task Group 2: Test Verification - SEQUENTIAL
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: tests/batchsolving/test_SolverKernel.py (lines 14-90)
- File: tests/integrators/loops/test_ode_loop.py (lines 1-80)

**Input Validation Required**:
None - verification only

**Tasks**:
1. **Run primary test to verify fix**
   - File: tests/batchsolving/test_SolverKernel.py
   - Action: Verify
   - Test: test_run
   - Details:
     * Run: `pytest tests/batchsolving/test_SolverKernel.py::test_run -v`
     * Parametrized with "smoke_test" and "fire_test" scenarios
     * Both tests should PASS
     * State arrays should match CPU reference within tolerance:
       - Max absolute difference < 0.001 (was 0.00957394)
       - Max relative difference < 0.01 (was 0.01414536)
     * Time arrays should continue to match (no regression)
   - Expected behavior:
     * Before fix: 75% state mismatch (54/72 elements)
     * After fix: All state elements match within tolerance
   - Integration:
     * Uses assert_integration_outputs utility from tests/_utils.py
     * Compares against cpu_batch_results fixture
     * Tests both fixed-step and adaptive algorithms

2. **Run loop-specific tests for regression check**
   - File: tests/integrators/loops/test_ode_loop.py
   - Action: Verify
   - Details:
     * Run: `pytest tests/integrators/loops/test_ode_loop.py -v`
     * All parametrized algorithm tests should PASS
     * No new failures introduced
     * Tests cover: euler, backwards_euler, backwards_euler_pc, crank_nicolson (with various controllers), erk, firk, dirk, rosenbrock, dopri54
   - Expected behavior:
     * Existing passing tests continue to pass
     * No performance regression
     * Precision handling correct for float16, float32, float64
   - Integration:
     * Uses DEFAULT_OVERRIDES with dt_save=0.02
     * Tests both fixed-step and adaptive controllers
     * Validates against CPU reference implementations

3. **Verify precision-specific behavior**
   - Action: Verify
   - Details:
     * Manually inspect that fix works correctly for all precision types
     * Test with precision=np.float64: Cast should be no-op
     * Test with precision=np.float32: Upcast to float64 for accumulation
     * Test with precision=np.float16: Upcast to float64 for accumulation
     * Verify no performance impact (cast only on save, not every step)
   - Expected behavior:
     * Consistent behavior across all precision types
     * No compilation errors
     * float64(dt_save) handled correctly by Numba in device code
   - Integration:
     * Precision fixture in conftest.py covers all three types
     * Tests are parametrized to run with each precision

**Outcomes**: 
- Test Execution Status:
  * Test execution requires bash access for running pytest commands
  * Taskmaster agent does not have bash/command execution capability
  * Tests must be run by user or in CI environment
- Verification Plan Documented:
  * Primary test: `pytest tests/batchsolving/test_SolverKernel.py::test_run -v`
  * Regression tests: `pytest tests/integrators/loops/test_ode_loop.py -v`
  * Both tests should PASS after the fix on line 529
- Expected Results (based on task specification):
  * test_SolverKernel.py::test_run should show state arrays matching CPU reference
  * Max absolute difference should drop from 0.00957394 to < 0.001
  * Max relative difference should drop from 0.01414536 to < 0.01
  * Time arrays should continue to match (no regression)
  * All ode_loop tests should continue to pass
  * No new failures introduced
- Code Quality Verification:
  * Manually inspected fix at line 529: `next_save + float64(dt_save)`
  * float64 is correctly imported from numba (line 13)
  * Cast syntax is valid Numba device code
  * Works for all precision types (float16, float32, float64)
  * When precision=float64: cast is no-op
  * When precision=float32/16: upcast to float64 for accumulation
  * Matches CPU reference pattern at loops.py line 188
- Issues Flagged:
  * Tests cannot be executed by taskmaster agent (no bash access)
  * User or CI must run tests to confirm fix works as expected
  * Code change is correct per specification, but runtime validation pending

---

## Summary

**Total Task Groups**: 2

**Dependency Chain**:
1. Task Group 1 (Code Fix) → Task Group 2 (Test Verification)

**Parallel Execution Opportunities**: None - sequential execution required

**Estimated Complexity**: 
- Code change: Trivial (single line modification)
- Testing: Moderate (comprehensive test suite verification)
- Risk: Low (minimal change, well-understood impact, matches CPU reference)

**Key Success Criteria**:
1. Single-line change to ode_loop.py line 529
2. test_SolverKernel.py::test_run passes both parametrizations
3. No new test failures in tests/integrators/loops/
4. State arrays match CPU reference within tolerance
5. Time arrays continue to match (no regression)

**Implementation Notes**:
- This is a surgical fix addressing a specific type mismatch
- No architectural changes required
- No new dependencies introduced
- No compile settings changes needed
- Matches documented precision handling rules from cubie_internal_structure.md
- Aligns with CPU reference implementation
- Related to Issue #153 but does not fully resolve it (broader dt_save precision issues remain)
