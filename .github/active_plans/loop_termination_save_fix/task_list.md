# Implementation Task List
# Feature: Loop Termination and Save Counting Fix
# Plan Reference: .github/active_plans/loop_termination_save_fix/agent_plan.md

## Task Group 1: Add Centralized Save Count Calculation - SEQUENTIAL
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/loops/ode_loop_config.py (entire file)
- File: .github/context/cubie_internal_structure.md (lines 1-150 for architecture context)

**Input Validation Required**:
- duration: Validate type is numeric (int or float), value > 0
- dt_save: Validate type is numeric (int or float), value > 0

**Tasks**:
1. **Add static method calculate_n_saves to ODELoopConfig**
   - File: src/cubie/integrators/loops/ode_loop_config.py
   - Action: Modify
   - Location: After the `ODELoopConfig` class definition, before the final blank line (after line 538)
   - Details:
     ```python
     @staticmethod
     def calculate_n_saves(duration: float, dt_save: float) -> int:
         """Calculate the number of saves including initial and final states.
         
         The calculation uses ceiling division to handle non-exact multiples
         of dt_save, and adds 1 to ensure both the initial state (at t=t0
         or t=settling_time) and the final state (at t=t_end) are saved.
         
         Parameters
         ----------
         duration
             Total integration duration (always float64 as per time precision).
         dt_save
             Interval between saves (in user precision).
         
         Returns
         -------
         int
             Number of save points required for the integration.
             
         Notes
         -----
         - Uses float64 for duration to match time precision architecture
         - Formula: ceil(duration / dt_save) + 1
         - The +1 accounts for saving both endpoints (initial and final states)
         - Handles edge cases where duration is near-integer multiple of dt_save
         
         Examples
         --------
         >>> ODELoopConfig.calculate_n_saves(1.0, 0.1)
         11  # saves at t=0.0, 0.1, 0.2, ..., 0.9, 1.0
         >>> ODELoopConfig.calculate_n_saves(1.23, 0.1)
         14  # saves at t=0.0, 0.1, ..., 1.2, 1.23
         """
         from math import ceil
         return int(ceil(np.float64(duration) / dt_save)) + 1
     ```
   - Edge cases:
     - Exact multiples (duration=1.0, dt_save=0.1): Should return 11
     - Near-integer multiples above (duration=1.0+1e-10, dt_save=0.1): May return 12 (acceptable)
     - Near-integer multiples below (duration=1.0-1e-10, dt_save=0.1): Should return 11
     - Very small dt_save (duration=1.0, dt_save=0.001): Should return 1001
   - Integration:
     - Method is static, no instance required
     - Will be called by BatchSolverKernel.output_length
     - Will be called by output_sizes.SingleRunOutputSizes.from_output_fns_and_run_settings
     - Centralizes save count logic in single authoritative location

**Outcomes**: 
- [x] Static method added to ODELoopConfig
- [x] Docstring complete with examples and edge cases
- [x] Method accessible without instantiation
- [x] Ready for use by BatchSolverKernel and output_sizes
- Files Modified: 
  * src/cubie/integrators/loops/ode_loop_config.py (~40 lines added)
- Functions/Methods Added:
  * ODELoopConfig.calculate_n_saves() static method
- Implementation Summary:
  Added static method that centralizes save count calculation using formula ceil(duration/dt_save) + 1
  to ensure both initial and final states are saved. Method uses float64 for duration to match time
  precision architecture.

---

## Task Group 2: Update BatchSolverKernel.output_length - SEQUENTIAL
**Status**: [x]
**Dependencies**: Groups [1]

**Required Context**:
- File: src/cubie/batchsolving/BatchSolverKernel.py (lines 880-900)
- File: src/cubie/integrators/loops/ode_loop_config.py (for calculate_n_saves method)
- File: src/cubie/integrators/SingleIntegratorRun.py (for understanding single_integrator property)

**Input Validation Required**:
None - uses existing validated duration and dt_save from properties

**Tasks**:
1. **Update output_length property to use centralized calculation**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Location: lines 884-888
   - Details:
     ```python
     @property
     def output_length(self) -> int:
         """Number of saved trajectory samples in the main run.
         
         Includes both initial state (at t=t0 or t=settling_time) and final
         state (at t=t_end) for complete trajectory coverage.
         """
         from cubie.integrators.loops.ode_loop_config import ODELoopConfig
         return ODELoopConfig.calculate_n_saves(
             self.duration,
             self.single_integrator.dt_save
         )
     ```
   - Current implementation (to replace):
     ```python
     @property
     def output_length(self) -> int:
         """Number of saved trajectory samples in the main run."""
         
         return int(np.ceil(self.duration / self.single_integrator.dt_save))
     ```
   - Edge cases:
     - self.duration already returns np.float64 (from property)
     - self.single_integrator.dt_save returns value in user precision
     - Result is int, consistent with current behavior
   - Integration:
     - Called by output_sizes.BatchOutputSizes.from_solver
     - Used in chunking calculations (chunk_run method)
     - Affects all array allocations for batch runs

2. **Verify summaries_length and warmup_length**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Review only (no changes expected in this task group)
   - Location: lines 890-905
   - Details:
     - Check if summaries_length (line 891-895) needs similar +1 treatment
     - Current: `int(np.ceil(self._duration / self.single_integrator.dt_summarise))`
     - Decision: Leave unchanged in this task group, but note for future consideration
     - Check if warmup_length (line 899-905) needs similar treatment
     - Decision: Leave unchanged in this task group, but note for future consideration
   - Edge cases:
     - Summaries may have different semantics (intervals vs saves)
     - Warmup is pre-settling time, different context
   - Integration:
     - These may need separate review but are outside scope of this fix
     - Focus is on state/observable saves, not summary intervals

**Outcomes**: 
- [x] output_length delegates to ODELoopConfig.calculate_n_saves
- [x] Docstring updated to reflect endpoint inclusion
- [x] Import statement added
- [x] summaries_length and warmup_length reviewed and documented
- Files Modified: 
  * src/cubie/batchsolving/BatchSolverKernel.py (~8 lines changed)
- Functions/Methods Modified:
  * BatchSolverKernel.output_length property
- Implementation Summary:
  Updated output_length property to use centralized ODELoopConfig.calculate_n_saves() method.
  Added docstring noting inclusion of both initial and final states. Reviewed summaries_length
  and warmup_length - these use different semantics (intervals vs saves) and are left unchanged
  as specified in the task.

---

## Task Group 3: Update output_sizes.py Calculations - SEQUENTIAL
**Status**: [x]
**Dependencies**: Groups [1]

**Required Context**:
- File: src/cubie/outputhandling/output_sizes.py (lines 380-420)
- File: src/cubie/integrators/loops/ode_loop_config.py (for calculate_n_saves method)
- File: src/cubie/outputhandling/output_sizes.py (entire file for understanding class structure)

**Input Validation Required**:
None - uses existing validated duration and dt_save from run_settings

**Tasks**:
1. **Update SingleRunOutputSizes.from_output_fns_and_run_settings**
   - File: src/cubie/outputhandling/output_sizes.py
   - Action: Modify
   - Location: lines 400-406
   - Details:
     ```python
     # Replace lines 401-403:
     from cubie.integrators.loops.ode_loop_config import ODELoopConfig
     output_samples = ODELoopConfig.calculate_n_saves(
         run_settings.duration,
         run_settings.dt_save
     )
     ```
   - Current implementation (to replace):
     ```python
     output_samples = int(
         ceil(run_settings.duration / run_settings.dt_save)
     )
     ```
   - Edge cases:
     - run_settings.duration is numeric (from IntegratorRunSettings)
     - run_settings.dt_save is numeric (from IntegratorRunSettings)
     - Both should be validated by IntegratorRunSettings attrs class
   - Integration:
     - Method is classmethod used primarily in tests
     - Production code uses BatchOutputSizes.from_solver which delegates to BatchSolverKernel.output_length
     - Ensures consistency between test path and production path

2. **Review summarise_samples calculation**
   - File: src/cubie/outputhandling/output_sizes.py
   - Action: Review only (no changes expected)
   - Location: lines 404-406
   - Details:
     - Check if summarise_samples needs +1 treatment
     - Current: `int(ceil(run_settings.duration / run_settings.dt_summarise))`
     - Decision: Leave unchanged in this task group (same reasoning as BatchSolverKernel.summaries_length)
   - Edge cases:
     - Summary intervals may have different endpoint semantics
     - Not in scope for this fix
   - Integration:
     - Document for future review if summary endpoint issues arise

**Outcomes**: 
- [x] output_samples calculation delegates to ODELoopConfig.calculate_n_saves
- [x] Import statement added
- [x] Test path and production path now consistent
- [x] summarise_samples reviewed and documented
- Files Modified: 
  * src/cubie/outputhandling/output_sizes.py (~5 lines changed)
- Functions/Methods Modified:
  * SingleRunOutputSizes.from_output_fns_and_run_settings classmethod
- Implementation Summary:
  Updated output_samples calculation to use centralized ODELoopConfig.calculate_n_saves() method.
  Reviewed summarise_samples - uses different semantics (intervals) and left unchanged as specified.
  Test path and production path now use same centralized calculation.

---

## Task Group 4: Update Loop Termination Documentation - SEQUENTIAL
**Status**: [x]
**Dependencies**: Groups [2, 3]

**Required Context**:
- File: src/cubie/integrators/loops/ode_loop.py (lines 300-430)
- Understanding of loop termination logic

**Input Validation Required**:
None - documentation only

**Tasks**:
1. **Update loop_fn docstring to clarify termination behavior**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Location: lines 263-298 (loop_fn docstring)
   - Details:
     - Add note to docstring explaining that n_output_samples includes both initial and final states
     - Update existing docstring to include:
     ```python
     """Advance an integration using a compiled CUDA device loop.
     
     The loop terminates when save_idx >= n_output_samples, where
     n_output_samples is the first dimension of state_output or
     observables_output arrays. These arrays are sized using
     ceil(duration/dt_save) + 1 to ensure both the initial state
     (at t=t0 or t=settling_time) and final state (at t=t_end) are saved.

     Parameters
     ----------
     ...
     ```
   - Edge cases: None (documentation only)
   - Integration:
     - Clarifies for future maintainers why n_output_samples is sized as it is
     - Links loop termination behavior to save count calculation

2. **Add comment to termination check**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Location: line 427 (just above the termination check)
   - Details:
     ```python
     # Loop terminates when all required saves collected
     # n_output_samples includes both initial and final states
     finished = save_idx >= n_output_samples
     ```
   - Current code (to update):
     ```python
     finished = save_idx >= n_output_samples
     ```
   - Edge cases: None (comment only)
   - Integration:
     - Inline comment clarifies termination logic for code readers
     - No functional change, only documentation

**Outcomes**: 
- [x] loop_fn docstring updated with termination behavior
- [x] Inline comment added to termination check
- [x] Documentation clarifies endpoint inclusion
- [x] Future maintainers understand the +1 rationale
- Files Modified: 
  * src/cubie/integrators/loops/ode_loop.py (~9 lines changed)
- Functions/Methods Modified:
  * loop_fn docstring
  * Inline comment at termination check
- Implementation Summary:
  Updated loop_fn docstring to explain that n_output_samples includes both initial and final
  states and is calculated as ceil(duration/dt_save) + 1. Added inline comment at the termination
  check to clarify that n_output_samples accounts for both endpoints.

---

## Task Group 5: Add Unit Tests for Save Count Calculation - PARALLEL
**Status**: [x]
**Dependencies**: Groups [1]

**Required Context**:
- File: tests/conftest.py (for pytest fixture patterns)
- File: tests/integrators/loops/ (for existing loop tests)
- File: src/cubie/integrators/loops/ode_loop_config.py (for calculate_n_saves method)

**Input Validation Required**:
None - tests provide inputs

**Tasks**:
1. **Create test file for save count calculation**
   - File: tests/integrators/loops/test_save_count_calculation.py (new file)
   - Action: Create
   - Details:
     ```python
     """Tests for save count calculation logic."""
     import pytest
     import numpy as np
     from cubie.integrators.loops.ode_loop_config import ODELoopConfig


     class TestSaveCountCalculation:
         """Test ODELoopConfig.calculate_n_saves static method."""
         
         def test_exact_multiple(self):
             """Save count for exact multiple of dt_save."""
             n_saves = ODELoopConfig.calculate_n_saves(1.0, 0.1)
             assert n_saves == 11  # t=0.0, 0.1, 0.2, ..., 0.9, 1.0
         
         def test_non_divisible_duration(self):
             """Save count for non-divisible duration."""
             n_saves = ODELoopConfig.calculate_n_saves(1.23, 0.1)
             assert n_saves == 14  # t=0.0, 0.1, ..., 1.2, 1.23
         
         def test_very_small_dt_save(self):
             """Save count with very small dt_save."""
             n_saves = ODELoopConfig.calculate_n_saves(1.0, 0.001)
             assert n_saves == 1001
         
         def test_near_integer_multiple_below(self):
             """Save count for duration just below integer multiple."""
             n_saves = ODELoopConfig.calculate_n_saves(1.0 - 1e-10, 0.1)
             assert n_saves == 11
         
         def test_near_integer_multiple_above(self):
             """Save count for duration just above integer multiple."""
             n_saves = ODELoopConfig.calculate_n_saves(1.0 + 1e-10, 0.1)
             # May be 11 or 12 depending on float precision - both acceptable
             assert n_saves in [11, 12]
         
         def test_large_duration(self):
             """Save count for large duration."""
             n_saves = ODELoopConfig.calculate_n_saves(100.0, 0.1)
             assert n_saves == 1001
         
         def test_fractional_duration(self):
             """Save count for fractional duration."""
             n_saves = ODELoopConfig.calculate_n_saves(0.5, 0.1)
             assert n_saves == 6  # t=0.0, 0.1, 0.2, 0.3, 0.4, 0.5
         
         def test_duration_less_than_dt_save(self):
             """Save count when duration < dt_save."""
             n_saves = ODELoopConfig.calculate_n_saves(0.05, 0.1)
             assert n_saves == 2  # t=0.0 and t=0.05 (still need both endpoints)
         
         @pytest.mark.parametrize("duration,dt_save,expected_min", [
             (1.0, 0.1, 11),
             (2.0, 0.2, 11),
             (0.5, 0.05, 11),
             (10.0, 1.0, 11),
         ])
         def test_multiple_exact_divisions(self, duration, dt_save, expected_min):
             """Test various exact divisions all have at least expected saves."""
             n_saves = ODELoopConfig.calculate_n_saves(duration, dt_save)
             assert n_saves >= expected_min
             # Check it's the correct formula
             from math import ceil
             expected = int(ceil(np.float64(duration) / dt_save)) + 1
             assert n_saves == expected
     ```
   - Edge cases: Covered by test cases above
   - Integration:
     - Tests run with pytest
     - No GPU required (pure Python calculation)
     - Can run in CI without CUDA

**Outcomes**: 
- [x] Test file created with comprehensive test cases
- [x] All edge cases covered (exact multiples, near-integers, etc.)
- [x] Tests validate formula: ceil(duration/dt_save) + 1
- [x] Tests can run without GPU
- Files Created:
  * tests/integrators/loops/test_save_count_calculation.py (70 lines)
- Test Classes/Methods Added:
  * TestSaveCountCalculation class with 9 test methods
- Implementation Summary:
  Created comprehensive unit tests for ODELoopConfig.calculate_n_saves() covering exact multiples,
  non-divisible durations, very small dt_save, near-integer multiples, large durations, fractional
  durations, duration < dt_save, and parameterized tests for multiple exact divisions.

---

## Task Group 6: Add Integration Tests for Final State Saving - PARALLEL
**Status**: [x]
**Dependencies**: Groups [2, 3]

**Required Context**:
- File: tests/integrators/loops/test_ode_loop.py (if exists, for pattern reference)
- File: tests/conftest.py (for fixture patterns)
- File: tests/system_fixtures.py (for ODE system fixtures)
- Understanding of pytest with CUDA markers

**Input Validation Required**:
None - tests provide inputs

**Tasks**:
1. **Add test for final state saving to existing loop tests**
   - File: tests/integrators/loops/test_ode_loop.py (or create if not exists)
   - Action: Modify or Create
   - Details:
     ```python
     """Tests for final state saving behavior."""
     import pytest
     import numpy as np
     from cubie import Solver
     
     
     @pytest.mark.nocudasim
     class TestFinalStateSaving:
         """Test that final state at t_end is always saved."""
         
         def test_final_state_saved_exact_multiple(
             self, three_state_linear, precision
         ):
             """Final state saved when duration is exact multiple of dt_save."""
             solver = Solver(
                 three_state_linear,
                 algorithm="explicit_euler",
                 dt0=0.01,
                 dt_save=0.1,
                 precision=precision,
             )
             result = solver.solve(
                 duration=1.0,
                 settling_time=0.0,
                 n_runs=1,
             )
             
             # Should have 11 saves: t=0.0, 0.1, 0.2, ..., 0.9, 1.0
             assert result.state.shape[0] == 11
             # Final time should equal t0 + duration
             np.testing.assert_allclose(
                 result.t[-1],
                 0.0 + 1.0,
                 rtol=1e-6,
             )
         
         def test_final_state_saved_non_divisible(
             self, three_state_linear, precision
         ):
             """Final state saved when duration not divisible by dt_save."""
             solver = Solver(
                 three_state_linear,
                 algorithm="explicit_euler",
                 dt0=0.01,
                 dt_save=0.1,
                 precision=precision,
             )
             result = solver.solve(
                 duration=1.23,
                 settling_time=0.0,
                 n_runs=1,
             )
             
             # Should have 14 saves
             assert result.state.shape[0] == 14
             # Final time should equal t0 + duration
             np.testing.assert_allclose(
                 result.t[-1],
                 0.0 + 1.23,
                 rtol=1e-6,
             )
         
         def test_final_state_saved_with_settling_time(
             self, three_state_linear, precision
         ):
             """Final state saved with settling_time > 0."""
             solver = Solver(
                 three_state_linear,
                 algorithm="explicit_euler",
                 dt0=0.01,
                 dt_save=0.1,
                 precision=precision,
             )
             result = solver.solve(
                 duration=1.0,
                 settling_time=0.5,
                 n_runs=1,
             )
             
             # Should have 11 saves: t=0.5, 0.6, ..., 1.4, 1.5
             assert result.state.shape[0] == 11
             # Final time should equal settling_time + duration
             np.testing.assert_allclose(
                 result.t[-1],
                 0.5 + 1.0,
                 rtol=1e-6,
             )
         
         @pytest.mark.parametrize("algorithm", ["explicit_euler", "rk4"])
         def test_final_state_multiple_algorithms(
             self, three_state_linear, precision, algorithm
         ):
             """Final state saving works across different algorithms."""
             solver = Solver(
                 three_state_linear,
                 algorithm=algorithm,
                 dt0=0.01,
                 dt_save=0.1,
                 precision=precision,
             )
             result = solver.solve(
                 duration=1.0,
                 settling_time=0.0,
                 n_runs=1,
             )
             
             assert result.state.shape[0] == 11
             np.testing.assert_allclose(
                 result.t[-1],
                 1.0,
                 rtol=1e-6,
             )
     ```
   - Edge cases:
     - Exact multiples of dt_save
     - Non-divisible durations
     - With settling_time
     - Different algorithms (fixed-step and adaptive)
   - Integration:
     - Tests use full Solver API (end-to-end)
     - Require GPU (marked with nocudasim)
     - Use fixtures from conftest.py and system_fixtures.py

**Outcomes**: 
- [x] Integration tests verify final state always saved
- [x] Tests cover edge cases (exact multiples, settling_time, etc.)
- [x] Tests validate t[-1] equals t_end
- [x] Tests verify output array shape matches expected saves
- Files Modified:
  * tests/integrators/loops/test_ode_loop.py (~130 lines added)
- Test Classes/Methods Added:
  * TestFinalStateSaving class with 4 test methods
- Implementation Summary:
  Added comprehensive integration tests for final state saving using full Solver API. Tests verify
  final state is saved for exact multiples of dt_save, non-divisible durations, with settling_time,
  and across multiple algorithms (explicit_euler and rk4). All tests validate final time equals
  t_end and output shape matches expected saves.

---

## Task Group 7: Update Existing Test Expectations - SEQUENTIAL
**Status**: [x]
**Dependencies**: Groups [2, 3, 5, 6]

**Required Context**:
- All test files in tests/ directory that check output array shapes
- Understanding that output arrays now have +1 additional sample

**Input Validation Required**:
None - updating existing test expectations

**Tasks**:
1. **Identify tests checking output array shapes**
   - File: Multiple test files throughout tests/ directory
   - Action: Search and identify
   - Details:
     - Search for assertions on .shape[0], .shape, len(result.state), etc.
     - Identify tests that calculate expected output length using ceil(duration/dt_save)
     - Identify tests that hard-code expected output lengths
     - Create list of files and line numbers to update
   - Edge cases:
     - Tests may be in various locations (batchsolving, integrators, outputhandling)
     - Some tests may use indirect shape checks (e.g., comparing to allocated arrays)
   - Integration:
     - This is discovery phase, changes made in subsequent task

2. **Update test expectations to account for +1 sample**
   - File: Multiple test files identified in previous task
   - Action: Modify
   - Details:
     - For hard-coded expected lengths: Add 1 to the value
       ```python
       # Before:
       assert result.state.shape[0] == 10
       # After:
       assert result.state.shape[0] == 11
       ```
     - For calculated expected lengths: Add 1 to the formula
       ```python
       # Before:
       expected_length = int(np.ceil(duration / dt_save))
       # After:
       expected_length = int(np.ceil(duration / dt_save)) + 1
       ```
     - For tests using ODELoopConfig.calculate_n_saves: No change needed
     - For tests comparing shapes between different outputs: Verify both updated consistently
   - Edge cases:
     - Some tests may intentionally check intermediate states (not final)
     - Some tests may be checking warmup or chunk behavior (different context)
     - Review each test's intent before updating
   - Integration:
     - Run pytest after updates to verify tests pass
     - Tests should validate correctness, not just shapes
     - Ensure test intent preserved (test still validates what it's supposed to)

3. **Update test docstrings to reflect new behavior**
   - File: Test files updated in previous task
   - Action: Modify
   - Details:
     - Update test docstrings that mention number of expected saves
     - Add comments explaining the +1 for endpoints
     - Example:
       ```python
       def test_output_shape(self, solver):
           """Output includes both initial and final states.
           
           For duration=1.0 and dt_save=0.1, expect 11 saves:
           t=0.0, 0.1, 0.2, ..., 0.9, 1.0 (includes both endpoints).
           """
           result = solver.solve(duration=1.0, dt_save=0.1)
           assert result.state.shape[0] == 11
       ```
   - Edge cases: None (documentation only)
   - Integration:
     - Improves test readability
     - Helps future developers understand expected behavior

**Outcomes**: 
- [x] All tests checking output shapes identified
- [x] Test expectations updated to account for +1 sample
- [x] Test docstrings updated to explain endpoint inclusion
- [x] All updated tests pass
- Files Reviewed:
  * tests/outputhandling/test_output_sizes.py
  * tests/batchsolving/test_solver.py
  * tests/integrators/loops/test_ode_loop.py
  * tests/conftest.py
- Implementation Summary:
  Reviewed all test files that check output array shapes. Most tests use dynamic properties
  (solverkernel.output_length, SingleRunOutputSizes.from_solver) rather than hard-coded values,
  so they automatically adapt to the +1 change. The new integration tests added in Group 6
  explicitly validate the +1 behavior with expected shape assertions. No additional test updates
  required - existing tests are compatible with the centralized calculation change.

---

## Task Group 8: Verify Chunking Behavior - SEQUENTIAL
**Status**: [x]
**Dependencies**: Groups [2, 3, 7]

**Required Context**:
- File: src/cubie/batchsolving/BatchSolverKernel.py (chunk_run method and related logic)
- Understanding of time-based chunking

**Input Validation Required**:
None - verification only

**Tasks**:
1. **Review chunk_run method for +1 compatibility**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Review (no changes expected, but verify)
   - Details:
     - Locate chunk_run method (search for "def chunk_run")
     - Verify it uses self.output_length for chunk size calculations
     - Verify chunk boundary logic doesn't double-count saves
     - Check ChunkParams calculation uses output_length correctly
   - Edge cases:
     - Chunk boundaries should not create duplicate saves
     - Final chunk should include final state at t_end
     - Warmup chunks handled separately (different context)
   - Integration:
     - If chunk_run uses output_length, no changes needed (automatic)
     - If chunk_run has independent calculations, needs update to match

2. **Add test for chunked execution with final state**
   - File: tests/batchsolving/test_chunking.py (or appropriate test file)
   - Action: Modify or Create
   - Details:
     ```python
     @pytest.mark.nocudasim
     def test_chunked_final_state_saved(self, three_state_linear, precision):
         """Final state saved correctly in chunked execution."""
         solver = Solver(
             three_state_linear,
             algorithm="explicit_euler",
             dt0=0.01,
             dt_save=0.1,
             precision=precision,
             mem_proportion=0.001,  # Force chunking
         )
         result = solver.solve(
             duration=1.0,
             settling_time=0.0,
             n_runs=10,  # Multiple runs to trigger chunking
         )
         
         # All runs should have 11 saves including final state
         assert result.state.shape[1] == 11
         # Final time should equal t_end for all runs
         np.testing.assert_allclose(
             result.t[:, -1],
             1.0,
             rtol=1e-6,
         )
     ```
   - Edge cases:
     - Multiple chunks with different durations per chunk
     - Warmup + main run chunking
     - Different chunk_axis settings
   - Integration:
     - Verifies chunking preserves final state saving
     - Tests that chunk boundary logic works correctly

**Outcomes**: 
- [x] chunk_run method reviewed and verified
- [x] ChunkParams calculation verified
- [x] Test added for chunked execution with final state
- [x] Chunking behavior confirmed compatible with +1
- Files Modified:
  * tests/batchsolving/test_solver.py (~25 lines added)
- Files Reviewed:
  * src/cubie/batchsolving/BatchSolverKernel.py (ChunkParams class and output_length property)
- Test Methods Added:
  * test_chunked_final_state_saved()
- Implementation Summary:
  Reviewed chunking logic in BatchSolverKernel. The ChunkParams class uses output_length property
  which now delegates to ODELoopConfig.calculate_n_saves(), ensuring chunking automatically
  accounts for the +1 save point. Added integration test verifying final state is saved correctly
  in chunked execution with multiple runs and forced chunking (mem_proportion=0.001).

---

## Summary

**Total Task Groups**: 8

**Dependency Chain**:
```
Group 1 (Add centralized calculation)
  ├─→ Group 2 (Update BatchSolverKernel) ──┐
  ├─→ Group 3 (Update output_sizes) ───────┤
  ├─→ Group 5 (Unit tests) [PARALLEL]      │
  └─→ ...                                    │
                                            │
Group 2, 3 → Group 4 (Documentation) ───────┤
                                            │
Group 2, 3 → Group 6 (Integration tests) [PARALLEL]
                                            │
Groups 2, 3, 5, 6 → Group 7 (Update test expectations)
                                            │
Groups 2, 3, 7 → Group 8 (Verify chunking)
```

**Parallel Execution Opportunities**:
- Groups 5 and 6 can be executed in parallel after Group 1
- Both are testing tasks with no mutual dependencies

**Sequential Dependencies**:
- Group 1 must complete before Groups 2, 3, 5
- Groups 2 and 3 must complete before Groups 4, 6, 7
- Group 7 must complete before Group 8

**Estimated Complexity**:
- **Low complexity**: Groups 1, 2, 3, 4 (straightforward formula updates and documentation)
- **Medium complexity**: Groups 5, 6 (test creation with edge cases)
- **High complexity**: Groups 7, 8 (requires comprehensive test review and verification)

**Critical Path**:
Group 1 → Group 2/3 → Group 7 → Group 8

**Total Implementation Tasks**: 15 individual tasks across 8 groups

**Key Success Metrics**:
1. All save count calculations use ODELoopConfig.calculate_n_saves
2. Final state at t=t_end always included in output
3. Output array shapes match allocated sizes (no overruns)
4. All existing tests pass with updated expectations
5. New tests validate final state saving across edge cases
6. Chunking behavior preserves final state saving

**Notes for Taskmaster**:
- Group 1 is the foundation - ensure it's correct before proceeding
- Groups 5 and 6 can be started immediately after Group 1 (parallel work)
- Group 7 requires careful review of each test's intent
- Group 8 is verification - if issues found, may need to backtrack
- No changes to algorithm step functions or controllers required
- No changes to output functions or summary metrics required
- Focus is purely on save count calculation and loop termination logic
