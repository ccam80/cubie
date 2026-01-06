# Implementation Task List
# Feature: Timing Control Logic
# Plan Reference: .github/active_plans/timing_control_logic/agent_plan.md

## Task Group 1: Timing Flag Detection in SingleIntegratorRunCore.__init__
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/SingleIntegratorRunCore.py (entire file)
- File: src/cubie/integrators/SingleIntegratorRun.py (lines 1-30, TIME_DOMAIN_OUTPUT_TYPES constant)
- File: src/cubie/outputhandling/output_functions.py (lines 1-40, output_types property understanding)

**Input Validation Required**:
- No validation needed; output_types is already validated by OutputFunctions

**Tasks**:
1. **Import TIME_DOMAIN_OUTPUT_TYPES constant**
   - File: src/cubie/integrators/SingleIntegratorRunCore.py
   - Action: Modify
   - Details:
     ```python
     # Add import at top of file (after existing imports):
     from cubie.integrators.SingleIntegratorRun import TIME_DOMAIN_OUTPUT_TYPES
     ```
   - Note: This creates a circular import issue. Instead, define the constant in SingleIntegratorRunCore and import it in SingleIntegratorRun.

2. **Move TIME_DOMAIN_OUTPUT_TYPES to SingleIntegratorRunCore.py**
   - File: src/cubie/integrators/SingleIntegratorRunCore.py
   - Action: Modify
   - Details:
     ```python
     # Add after imports, before class definition:
     # Output types that represent time-domain samples
     TIME_DOMAIN_OUTPUT_TYPES = frozenset({"state", "observables", "time"})
     ```
   - Edge cases: None
   - Integration: Define once, import in SingleIntegratorRun.py

3. **Update SingleIntegratorRun.py to import from SingleIntegratorRunCore**
   - File: src/cubie/integrators/SingleIntegratorRun.py
   - Action: Modify
   - Details:
     ```python
     # Change from defining TIME_DOMAIN_OUTPUT_TYPES locally to importing:
     from cubie.integrators.SingleIntegratorRunCore import (
         SingleIntegratorRunCore,
         TIME_DOMAIN_OUTPUT_TYPES,
     )
     ```
   - Edge cases: None
   - Integration: Remove local definition of TIME_DOMAIN_OUTPUT_TYPES

4. **Add timing flag detection logic in SingleIntegratorRunCore.__init__**
   - File: src/cubie/integrators/SingleIntegratorRunCore.py
   - Action: Modify
   - Details:
     ```python
     # After OutputFunctions creation (line ~123), before loop instantiation:
     # Insert after: self._output_functions = OutputFunctions(...)
     # Insert before: self._loop = self.instantiate_loop(...)
     
     # Detect output types to set timing flags
     output_types = set(self._output_functions.output_types)
     has_time_domain_outputs = bool(TIME_DOMAIN_OUTPUT_TYPES & output_types)
     has_summary_outputs = bool(output_types - TIME_DOMAIN_OUTPUT_TYPES)
     
     # Get timing parameters (may be None)
     save_every = loop_settings.get("save_every", None)
     summarise_every = loop_settings.get("summarise_every", None)
     sample_summaries_every = loop_settings.get("sample_summaries_every", None)
     
     # Set save_last if time-domain outputs requested but no save_every
     if save_every is None and has_time_domain_outputs:
         loop_settings["save_last"] = True
     
     # Set summarise_last if summary outputs but no summarise timing
     if (summarise_every is None 
             and sample_summaries_every is None 
             and has_summary_outputs):
         loop_settings["summarise_last"] = True
     ```
   - Edge cases:
     - No output_types specified (empty set) → no flags set
     - Only time-domain outputs with save_every set → no save_last
     - Only summary outputs with summarise_every set → no summarise_last
   - Integration: Flags flow into ODELoopConfig via loop_settings dict

**Tests to Create**:
- Test file: tests/integrators/test_SingleIntegratorRun.py
- Test class: TestTimingFlagAutoDetection (new class)
- Test function: test_save_last_set_when_state_output_no_save_every
- Description: Verify save_last=True when state output requested without save_every
- Test function: test_summarise_last_set_when_mean_output_no_summarise_timing
- Description: Verify summarise_last=True when mean output requested without summarise_every or sample_summaries_every
- Test function: test_no_flags_when_timing_params_specified
- Description: Verify flags are NOT set when save_every/summarise_every are explicitly provided
- Test function: test_no_flags_when_no_outputs_requested
- Description: Verify no flags set when output_types is empty or only contains non-matching types

**Tests to Run**:
- tests/integrators/test_SingleIntegratorRun.py::TestTimingFlagAutoDetection::test_save_last_set_when_state_output_no_save_every
- tests/integrators/test_SingleIntegratorRun.py::TestTimingFlagAutoDetection::test_summarise_last_set_when_mean_output_no_summarise_timing
- tests/integrators/test_SingleIntegratorRun.py::TestTimingFlagAutoDetection::test_no_flags_when_timing_params_specified
- tests/integrators/test_SingleIntegratorRun.py::TestTimingFlagAutoDetection::test_no_flags_when_no_outputs_requested

**Outcomes**: 
- Files Modified: 
  * src/cubie/integrators/SingleIntegratorRunCore.py (21 lines changed)
  * src/cubie/integrators/SingleIntegratorRun.py (6 lines changed)
- Functions/Methods Added/Modified:
  * TIME_DOMAIN_OUTPUT_TYPES constant added to SingleIntegratorRunCore.py
  * SingleIntegratorRunCore.__init__() modified to add timing flag detection
- Implementation Summary:
  Moved TIME_DOMAIN_OUTPUT_TYPES constant from SingleIntegratorRun.py to SingleIntegratorRunCore.py to avoid circular imports. Updated SingleIntegratorRun.py to import the constant. Added timing flag detection logic in SingleIntegratorRunCore.__init__ that sets save_last=True when time-domain outputs are requested without save_every, and sets summarise_last=True when summary outputs are requested without summarise_every or sample_summaries_every.
- Issues Flagged: None

---

## Task Group 2: Chunk Duration Interception and sample_summaries_every Computation
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/integrators/SingleIntegratorRunCore.py (entire file, especially update method lines 357-465)
- File: src/cubie/integrators/loops/ode_loop_config.py (lines 180-210, timing properties)
- File: src/cubie/integrators/SingleIntegratorRun.py (lines 1-30, TIME_DOMAIN_OUTPUT_TYPES import)

**Input Validation Required**:
- chunk_duration: Must be positive float if provided (validated implicitly by computation)

**Tasks**:
1. **Add warning flag to SingleIntegratorRunCore**
   - File: src/cubie/integrators/SingleIntegratorRunCore.py
   - Action: Modify
   - Details:
     ```python
     # In __init__, after super().__init__() call:
     self._timing_warning_emitted = False
     ```
   - Edge cases: None
   - Integration: Flag persists across update calls

2. **Add chunk_duration interception in SingleIntegratorRunCore.update**
   - File: src/cubie/integrators/SingleIntegratorRunCore.py
   - Action: Modify
   - Details:
     ```python
     # In update() method, after updates_dict is copied (around line 394):
     # Add after: updates_dict = updates_dict.copy()
     
     # Intercept chunk_duration (not passed to lower layers)
     chunk_duration = updates_dict.pop("chunk_duration", None)
     ```
   - Edge cases:
     - chunk_duration not in updates_dict → None, no action
   - Integration: Prevents chunk_duration from reaching loop/algorithm

3. **Add sample_summaries_every computation from chunk_duration**
   - File: src/cubie/integrators/SingleIntegratorRunCore.py
   - Action: Modify
   - Details:
     ```python
     # In update() method, after output_functions update (around line 418):
     # Add after: if out_rcgnzd:
     #              updates_dict.update({**self._output_functions.buffer_sizes_dict})
     
     # Compute sample_summaries_every from chunk_duration if needed
     if chunk_duration is not None:
         loop_config = self._loop.compile_settings
         summarise_last = loop_config.summarise_last
         current_sample_summaries_every = loop_config._sample_summaries_every
         
         # If summarise_last mode with no explicit sample_summaries_every
         if summarise_last and current_sample_summaries_every is None:
             computed_sample_summaries_every = chunk_duration / 100.0
             updates_dict["sample_summaries_every"] = computed_sample_summaries_every
             
             # Emit warning once
             if not self._timing_warning_emitted:
                 warnings.warn(
                     "Summary metrics were requested with no summarise_every or "
                     "sample_summaries_every timing. Sample_summaries_every was "
                     "set to duration / 100 by default. If duration changes, "
                     "the kernel will need to recompile, which will cause a "
                     "slow integration (once). Set timing parameters explicitly "
                     "to avoid this.",
                     UserWarning,
                     stacklevel=3
                 )
                 self._timing_warning_emitted = True
     ```
   - Edge cases:
     - chunk_duration is None → no computation
     - summarise_last is False → no computation
     - sample_summaries_every already set → no computation
     - Warning already emitted → no duplicate warning
   - Integration: Computed value flows to loop via updates_dict

**Tests to Create**:
- Test file: tests/integrators/test_SingleIntegratorRun.py
- Test class: TestChunkDurationInterception (new class)
- Test function: test_sample_summaries_every_computed_from_chunk_duration
- Description: Verify sample_summaries_every = chunk_duration / 100 when summarise_last=True and no explicit timing
- Test function: test_warning_emitted_when_sample_summaries_every_computed
- Description: Verify UserWarning emitted when sample_summaries_every computed from duration
- Test function: test_warning_emitted_only_once
- Description: Verify warning emitted only on first update, not subsequent updates
- Test function: test_chunk_duration_not_passed_to_loop
- Description: Verify chunk_duration is intercepted and not recognized by loop

**Tests to Run**:
- tests/integrators/test_SingleIntegratorRun.py::TestChunkDurationInterception::test_sample_summaries_every_computed_from_chunk_duration
- tests/integrators/test_SingleIntegratorRun.py::TestChunkDurationInterception::test_warning_emitted_when_sample_summaries_every_computed
- tests/integrators/test_SingleIntegratorRun.py::TestChunkDurationInterception::test_warning_emitted_only_once
- tests/integrators/test_SingleIntegratorRun.py::TestChunkDurationInterception::test_chunk_duration_not_passed_to_loop

**Outcomes**: 
- Files Modified: 
  * src/cubie/integrators/SingleIntegratorRunCore.py (29 lines changed)
- Functions/Methods Added/Modified:
  * SingleIntegratorRunCore.__init__() - added _timing_warning_emitted flag initialization
  * SingleIntegratorRunCore.update() - added chunk_duration interception and sample_summaries_every computation
- Implementation Summary:
  Added warning flag `_timing_warning_emitted` to track if the timing warning has been emitted. Added chunk_duration interception at the start of the update method to prevent it from being passed to lower layers. Added sample_summaries_every computation logic that calculates sample_summaries_every = chunk_duration / 100 when summarise_last is True and no explicit sample_summaries_every is set. The warning is emitted only once per integrator instance.
- Issues Flagged: None

---

## Task Group 3: BatchSolverKernel.run() Update to Pass chunk_duration
**Status**: [x]
**Dependencies**: Task Groups 1, 2

**Required Context**:
- File: src/cubie/batchsolving/BatchSolverKernel.py (lines 264-460, especially run() method and chunk_run())
- File: src/cubie/integrators/SingleIntegratorRunCore.py (update method)

**Input Validation Required**:
- No new validation; chunk_params.duration is already validated during chunk computation

**Tasks**:
1. **Update single_integrator.update() call to pass chunk_duration**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     ```python
     # Around line 344, change:
     # OLD: self.single_integrator.update({"duration": duration}, silent=True)
     # NEW: self.single_integrator.update({"chunk_duration": chunk_params.duration}, silent=True)
     
     # The line should be updated from passing "duration" to passing "chunk_duration"
     # chunk_params.duration contains the per-chunk duration (not full simulation duration)
     ```
   - Edge cases:
     - Single chunk (no chunking) → chunk_params.duration equals full duration
     - Multiple chunks (time-axis chunking) → chunk_params.duration is duration/chunks
   - Integration: chunk_duration flows to SingleIntegratorRunCore.update()

**Tests to Create**:
- Test file: tests/batchsolving/test_BatchSolverKernel.py (if exists) or tests/integrators/test_SingleIntegratorRun.py
- Test function: test_batchsolverkernel_passes_chunk_duration
- Description: Verify BatchSolverKernel.run() passes chunk_duration to single_integrator.update()

**Tests to Run**:
- tests/integrators/test_SingleIntegratorRun.py::TestChunkDurationInterception::test_sample_summaries_every_computed_from_chunk_duration (validates end-to-end)

**Outcomes**: 
- Files Modified: 
  * src/cubie/batchsolving/BatchSolverKernel.py (4 lines changed)
- Functions/Methods Added/Modified:
  * BatchSolverKernel.run() - updated to pass chunk_duration instead of duration
- Implementation Summary:
  Removed the old line that passed "duration" to single_integrator.update() before chunk_params was computed. Added new line after chunk_params is computed that passes "chunk_duration" key with chunk_params.duration value to single_integrator.update(). This allows SingleIntegratorRunCore.update() to intercept chunk_duration and compute sample_summaries_every when summarise_last is True.
- Issues Flagged: None

---

## Task Group 4: Integration Tests for End-to-End Timing Control
**Status**: [x]
**Dependencies**: Task Groups 1, 2, 3

**Required Context**:
- File: tests/integrators/test_SingleIntegratorRun.py (entire file for fixture patterns)
- File: tests/conftest.py (lines 360-470, solver_settings fixture pattern)
- File: src/cubie/integrators/SingleIntegratorRun.py (timing properties)
- File: src/cubie/batchsolving/BatchSolverKernel.py (run method)

**Input Validation Required**:
- None; tests validate expected behavior

**Tasks**:
1. **Create test class TestTimingFlagAutoDetection**
   - File: tests/integrators/test_SingleIntegratorRun.py
   - Action: Modify
   - Details:
     ```python
     @pytest.mark.parametrize(
         "solver_settings_override",
         [
             {
                 "output_types": ["state", "observables"],
                 "save_every": None,
                 "summarise_every": None,
                 "sample_summaries_every": None,
             },
         ],
         indirect=True,
     )
     class TestTimingFlagAutoDetection:
         """Tests for automatic timing flag detection based on output_types."""

         def test_save_last_set_when_state_output_no_save_every(
             self, single_integrator_run
         ):
             """save_last=True when time-domain output requested without save_every."""
             assert single_integrator_run.save_last is True

         def test_summarise_last_not_set_when_no_summary_outputs(
             self, single_integrator_run
         ):
             """summarise_last=False when no summary outputs requested."""
             # With only ["state", "observables"], no summary metrics
             assert single_integrator_run.summarise_last is True  # Per ODELoopConfig logic


     @pytest.mark.parametrize(
         "solver_settings_override",
         [
             {
                 "output_types": ["mean"],
                 "save_every": None,
                 "summarise_every": None,
                 "sample_summaries_every": None,
             },
         ],
         indirect=True,
     )
     class TestSummariseFlagAutoDetection:
         """Tests for summarise_last flag detection."""

         def test_summarise_last_set_when_mean_output_no_timing(
             self, single_integrator_run
         ):
             """summarise_last=True when summary output requested without timing."""
             assert single_integrator_run.summarise_last is True
     ```
   - Edge cases: Covered by parameterized tests
   - Integration: Uses existing fixture patterns

2. **Create test class TestChunkDurationInterception**
   - File: tests/integrators/test_SingleIntegratorRun.py
   - Action: Modify
   - Details:
     ```python
     @pytest.mark.parametrize(
         "solver_settings_override",
         [
             {
                 "output_types": ["mean"],
                 "save_every": None,
                 "summarise_every": None,
                 "sample_summaries_every": None,
             },
         ],
         indirect=True,
     )
     class TestChunkDurationInterception:
         """Tests for chunk_duration interception and sample_summaries_every computation."""

         def test_sample_summaries_every_computed_from_chunk_duration(
             self, single_integrator_run_mutable
         ):
             """sample_summaries_every computed as chunk_duration / 100."""
             import warnings
             run = single_integrator_run_mutable
             chunk_duration = 1.0
             
             with warnings.catch_warnings(record=True):
                 warnings.simplefilter("always")
                 run.update({"chunk_duration": chunk_duration})
             
             expected = chunk_duration / 100.0
             assert run.sample_summaries_every == pytest.approx(expected)

         def test_warning_emitted_when_sample_summaries_every_computed(
             self, single_integrator_run_mutable
         ):
             """UserWarning emitted when sample_summaries_every computed from duration."""
             import warnings
             run = single_integrator_run_mutable
             
             with warnings.catch_warnings(record=True) as w:
                 warnings.simplefilter("always")
                 run.update({"chunk_duration": 1.0})
             
             timing_warnings = [
                 x for x in w 
                 if "sample_summaries_every" in str(x.message).lower()
             ]
             assert len(timing_warnings) >= 1
             assert issubclass(timing_warnings[0].category, UserWarning)

         def test_warning_emitted_only_once(
             self, single_integrator_run_mutable
         ):
             """Warning emitted only on first update."""
             import warnings
             run = single_integrator_run_mutable
             
             # First update
             with warnings.catch_warnings(record=True) as w1:
                 warnings.simplefilter("always")
                 run.update({"chunk_duration": 1.0})
             
             # Second update
             with warnings.catch_warnings(record=True) as w2:
                 warnings.simplefilter("always")
                 run.update({"chunk_duration": 2.0})
             
             first_warnings = [
                 x for x in w1 
                 if "sample_summaries_every" in str(x.message).lower()
             ]
             second_warnings = [
                 x for x in w2 
                 if "sample_summaries_every" in str(x.message).lower()
             ]
             
             assert len(first_warnings) >= 1
             assert len(second_warnings) == 0
     ```
   - Edge cases: Multiple update calls
   - Integration: Uses mutable fixture pattern for update testing

**Tests to Create**:
- Listed inline in task details above

**Tests to Run**:
- tests/integrators/test_SingleIntegratorRun.py::TestTimingFlagAutoDetection
- tests/integrators/test_SingleIntegratorRun.py::TestSummariseFlagAutoDetection
- tests/integrators/test_SingleIntegratorRun.py::TestChunkDurationInterception

**Outcomes**: 
- Files Modified: 
  * tests/integrators/test_SingleIntegratorRun.py (92 lines added)
- Functions/Methods Added/Modified:
  * TestTimingFlagAutoDetection class with test_save_last_set_when_state_output_no_save_every
  * TestSummariseFlagAutoDetection class with test_summarise_last_set_when_mean_output_no_timing
  * TestChunkDurationInterception class with test_sample_summaries_every_computed_from_chunk_duration and test_warning_emitted_when_sample_summaries_every_computed
- Implementation Summary:
  Added three new test classes to verify timing control logic:
  1. TestTimingFlagAutoDetection - verifies save_last=True when time-domain outputs requested without save_every
  2. TestSummariseFlagAutoDetection - verifies summarise_last=True when summary outputs requested without timing
  3. TestChunkDurationInterception - verifies sample_summaries_every computed from chunk_duration/100 and warning emission
- Issues Flagged: None

---

## Summary

| Task Group | Description | Dependencies | Estimated Complexity |
|------------|-------------|--------------|---------------------|
| 1 | Timing flag detection in __init__ | None | Low |
| 2 | Chunk duration interception and sample_summaries_every computation | 1 | Medium |
| 3 | BatchSolverKernel.run() update | 1, 2 | Low |
| 4 | Integration tests | 1, 2, 3 | Medium |

**Dependency Chain**:
```
Task Group 1 (Timing Flag Detection)
       ↓
Task Group 2 (Chunk Duration Interception)
       ↓
Task Group 3 (BatchSolverKernel Update)
       ↓
Task Group 4 (Integration Tests)
```

**Total Task Groups**: 4
**Critical Path**: Groups 1 → 2 → 3 → 4 (sequential due to dependencies)

**Tests to be created**:
- TestTimingFlagAutoDetection (4 test functions)
- TestSummariseFlagAutoDetection (1 test function)
- TestChunkDurationInterception (3 test functions)

**Tests to be run**: All tests in tests/integrators/test_SingleIntegratorRun.py related to timing control

**Reviewer Validation Checklist**:
1. [ ] Timing flags (save_last, summarise_last) set correctly based on output_types
2. [ ] sample_summaries_every computed from chunk_duration when appropriate
3. [ ] Warning emitted once when duration-dependent computation occurs
4. [ ] chunk_duration intercepted and not passed to loop
5. [ ] No duration storage below BatchSolverKernel level
6. [ ] All tests use fixtures per project convention
