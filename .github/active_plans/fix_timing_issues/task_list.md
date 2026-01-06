# Implementation Task List
# Feature: Fix Timing Issues
# Plan Reference: .github/active_plans/fix_timing_issues/agent_plan.md

## Task Group 1: Fix OutputFunctions API Usage in Test
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: tests/batchsolving/test_SolverKernel.py (lines 182-203)
- File: src/cubie/outputhandling/output_functions.py (lines 28-38)

**Input Validation Required**:
- None - this is a test correction, not production code

**Tasks**:
1. **Remove save_every from output_settings dict**
   - File: tests/batchsolving/test_SolverKernel.py
   - Action: Modify
   - Details:
     ```python
     # Remove "save_every": 0.01 from the output_settings dict at line 187
     # The output_settings dict should only contain parameters from 
     # ALL_OUTPUT_FUNCTION_PARAMETERS (output_functions.py lines 28-38)
     # save_every is NOT in that set - it belongs in loop_settings
     
     # Before (line 182-196):
     output_settings = {
         "saved_state_indices": np.asarray([0, 1, 2]),
         "saved_observable_indices": np.asarray([0, 1, 2]),
         "summarised_state_indices": np.asarray([0]),
         "summarised_observable_indices": np.asarray([0]),
         "save_every": 0.01,  # <-- REMOVE THIS LINE
         "output_types": [
             "state",
             "observables",
             "mean",
             "max",
             "rms",
             "peaks[3]",
         ],
     }
     
     # After:
     output_settings = {
         "saved_state_indices": np.asarray([0, 1, 2]),
         "saved_observable_indices": np.asarray([0, 1, 2]),
         "summarised_state_indices": np.asarray([0]),
         "summarised_observable_indices": np.asarray([0]),
         "output_types": [
             "state",
             "observables",
             "mean",
             "max",
             "rms",
             "peaks[3]",
         ],
     }
     ```
   - Edge cases: None
   - Integration: The test already passes `save_every` in `loop_settings` at line 202

**Tests to Create**:
- None

**Tests to Run**:
- tests/batchsolving/test_SolverKernel.py::test_all_lower_plumbing

**Outcomes**:
- Files Modified: 
  * tests/batchsolving/test_SolverKernel.py (1 line removed)
- Functions/Methods Added/Modified:
  * test_all_lower_plumbing() - removed invalid `save_every` parameter from output_settings dict
- Implementation Summary:
  Removed `"save_every": 0.01,` from the `output_settings` dict in the test. This parameter is not valid for OutputFunctions (not in ALL_OUTPUT_FUNCTION_PARAMETERS) and belongs exclusively in `loop_settings`, where it is already correctly specified.
- Issues Flagged: None

---

## Task Group 2: Remove Sentinel from samples_per_summary Property
**Status**: [x]
**Dependencies**: None (can run in parallel with Group 1)

**Required Context**:
- File: src/cubie/integrators/loops/ode_loop_config.py (lines 316-324)
- File: src/cubie/integrators/loops/ode_loop.py (lines 370-379, 553-559, 804-817)

**Input Validation Required**:
- None - this is refactoring existing code logic

**Tasks**:
1. **Modify samples_per_summary property to return None instead of 2**30**
   - File: src/cubie/integrators/loops/ode_loop_config.py
   - Action: Modify
   - Details:
     ```python
     # Current implementation (lines 316-324):
     @property
     def samples_per_summary(self) -> Optional[int]:
         """Return the number of updates between summary outputs."""
         if self._summarise_every is None or self._sample_summaries_every is None:
             # In summarise_last mode, return large sentinel so modulo never triggers
             if self.summarise_last:
                 return 2**30
             return None
         return round(self.summarise_every / self.sample_summaries_every)
     
     # New implementation:
     @property
     def samples_per_summary(self) -> Optional[int]:
         """Return the number of updates between summary outputs.
         
         Returns None when timing is not configured; the loop build()
         method handles None by using a large value for summarise_last mode.
         """
         if self._summarise_every is None or self._sample_summaries_every is None:
             return None
         return round(self.summarise_every / self.sample_summaries_every)
     ```
   - Edge cases: summarise_last mode with no timing - now returns None instead of 2**30
   - Integration: ode_loop.py build() must handle None

2. **Handle None samples_per_summary in ode_loop.py build()**
   - File: src/cubie/integrators/loops/ode_loop.py  
   - Action: Modify
   - Details:
     ```python
     # Current implementation (line 373):
     samples_per_summary = config.samples_per_summary
     
     # New implementation (after line 373):
     samples_per_summary = config.samples_per_summary
     # For summarise_last mode, use large value so modulo never triggers
     # regular summary saves; summary is saved at end instead
     if samples_per_summary is None and config.summarise_last:
         samples_per_summary = 2**30
     ```
   - Edge cases: 
     - samples_per_summary is None and summarise_last is False - no change needed
     - samples_per_summary is None and summarise_last is True - use 2**30
   - Integration: This is used in the loop_fn device function for modulo checks

**Tests to Create**:
- Test file: tests/integrators/loops/test_ode_loop.py
- Test function: test_samples_per_summary_returns_none_in_summarise_last_mode
- Description: Verify that samples_per_summary returns None (not 2**30) when summarise_last is True and no timing is configured. Create an ODELoopConfig with summarise_last=True and _sample_summaries_every=None, verify samples_per_summary property returns None.

**Tests to Run**:
- tests/integrators/loops/test_ode_loop.py::test_samples_per_summary_returns_none_in_summarise_last_mode
- tests/batchsolving/test_SolverKernel.py::test_all_lower_plumbing

**Outcomes**:
- Files Modified: 
  * src/cubie/integrators/loops/ode_loop_config.py (6 lines changed)
  * src/cubie/integrators/loops/ode_loop.py (4 lines added)
  * tests/integrators/loops/test_ode_loop.py (28 lines added)
- Functions/Methods Added/Modified:
  * samples_per_summary property in ode_loop_config.py - returns None instead of 2**30 sentinel
  * build() method in ode_loop.py - handles None samples_per_summary for summarise_last mode
  * test_samples_per_summary_returns_none_in_summarise_last_mode() - new test function
- Implementation Summary:
  Removed the 2**30 sentinel value from the samples_per_summary property in ODELoopConfig. The property now returns None when timing is not configured. The sentinel value is now applied in the IVPLoop.build() method where it's captured in the loop closure, ensuring the compiled loop gets a valid integer while the property correctly returns None.
- Issues Flagged: None

---

## Task Group 3: Add is_duration_dependent Property
**Status**: [x]
**Dependencies**: Groups [1, 2]

**Required Context**:
- File: src/cubie/integrators/SingleIntegratorRun.py (lines 159-210)
- File: src/cubie/integrators/loops/ode_loop_config.py (lines 195-202, 189-192)

**Input Validation Required**:
- None - this is a read-only property

**Tasks**:
1. **Add is_duration_dependent property to SingleIntegratorRun**
   - File: src/cubie/integrators/SingleIntegratorRun.py
   - Action: Modify
   - Details:
     ```python
     # Add new property in the "Loop properties" section after summarise_last 
     # (after line 210):
     
     @property
     def is_duration_dependent(self) -> bool:
         """Return True when the loop is compile-dependent on duration.
         
         The loop function is duration-dependent when summarise_last mode
         is active but no explicit sample_summaries_every was provided.
         In this case, sample_summaries_every is computed from chunk_duration.
         """
         loop_config = self._loop.compile_settings
         return (loop_config.summarise_last 
                 and loop_config._sample_summaries_every is None)
     ```
   - Edge cases:
     - summarise_last=True, _sample_summaries_every=None -> True
     - summarise_last=True, _sample_summaries_every=0.01 -> False
     - summarise_last=False, _sample_summaries_every=None -> False
     - summarise_last=False, _sample_summaries_every=0.01 -> False
   - Integration: This property will be used by SingleIntegratorRunCore.update()

**Tests to Create**:
- Test file: tests/integrators/test_SingleIntegratorRun.py
- Test function: test_is_duration_dependent_true_when_summarise_last_and_no_timing
- Description: Create a SingleIntegratorRun with output_types that trigger summarise_last (e.g., ["mean"]) but no save_every/summarise_every/sample_summaries_every. Verify is_duration_dependent returns True.
- Test function: test_is_duration_dependent_false_when_explicit_timing
- Description: Create a SingleIntegratorRun with output_types=["mean"] and explicit sample_summaries_every. Verify is_duration_dependent returns False.
- Test function: test_is_duration_dependent_false_when_not_summarise_last
- Description: Create a SingleIntegratorRun with summarise_every set (which means summarise_last is False). Verify is_duration_dependent returns False.

**Tests to Run**:
- tests/integrators/test_SingleIntegratorRun.py::TestIsDurationDependentProperty::test_is_duration_dependent_true_when_summarise_last_and_no_timing
- tests/integrators/test_SingleIntegratorRun.py::TestIsDurationDependentFalseExplicitTiming::test_is_duration_dependent_false_when_explicit_timing
- tests/integrators/test_SingleIntegratorRun.py::TestIsDurationDependentFalseNotSummariseLast::test_is_duration_dependent_false_when_not_summarise_last

**Outcomes**:
- Files Modified: 
  * src/cubie/integrators/SingleIntegratorRun.py (11 lines added)
  * tests/integrators/test_SingleIntegratorRun.py (70 lines added)
- Functions/Methods Added/Modified:
  * is_duration_dependent property in SingleIntegratorRun.py
  * TestIsDurationDependentProperty test class with test_is_duration_dependent_true_when_summarise_last_and_no_timing
  * TestIsDurationDependentFalseExplicitTiming test class with test_is_duration_dependent_false_when_explicit_timing
  * TestIsDurationDependentFalseNotSummariseLast test class with test_is_duration_dependent_false_when_not_summarise_last
- Implementation Summary:
  Added the is_duration_dependent property to SingleIntegratorRun that returns True when summarise_last is True and no explicit sample_summaries_every was provided. This indicates that the loop compilation depends on duration. Created three parameterized test classes to verify all edge cases: True when summarise_last with no timing, False when explicit sample_summaries_every is set, and False when not in summarise_last mode.
- Issues Flagged: None

---

## Task Group 4: Gate chunk_duration Update with is_duration_dependent Check  
**Status**: [x]
**Dependencies**: Groups [1, 2, 3]

**Required Context**:
- File: src/cubie/integrators/SingleIntegratorRunCore.py (lines 449-475)
- File: src/cubie/integrators/SingleIntegratorRun.py (entire file for is_duration_dependent property location)

**Input Validation Required**:
- chunk_duration: Must be a positive float when provided (already validated by caller)

**Tasks**:
1. **Update loop before checking duration dependency and use property-based check**
   - File: src/cubie/integrators/SingleIntegratorRunCore.py
   - Action: Modify
   - Details:
     ```python
     # Current implementation (lines 449-475):
     # Compute sample_summaries_every from chunk_duration if needed
     if chunk_duration is not None:
         loop_config = self._loop.compile_settings
         summarise_last = loop_config.summarise_last
         current_sample_summaries_every = loop_config._sample_summaries_every
     
         # If summarise_last mode with no explicit sample_summaries_every
         if summarise_last and current_sample_summaries_every is None:
             computed_sample_summaries_every = chunk_duration / 100.0
             updates_dict["sample_summaries_every"] = (
                 computed_sample_summaries_every
             )
     
             # Emit warning once
             if not self._timing_warning_emitted:
                 warnings.warn(...)
                 self._timing_warning_emitted = True
     
     # New implementation:
     # The chunk_duration check should happen AFTER applying the updates_dict
     # to the loop, so that any new timing parameters are in place before
     # we check if duration dependency is still needed. However, since
     # we need to add sample_summaries_every TO the updates_dict before
     # loop.update() processes it, we need to check the condition before
     # passing to loop. The current logic is actually correct - we check
     # the CURRENT state and add to updates_dict if needed.
     #
     # The refinement is to use the is_duration_dependent logic directly
     # (same as the property in SingleIntegratorRun) instead of inline checks.
     # Since SingleIntegratorRunCore is the base class, we compute inline:
     
     if chunk_duration is not None:
         loop_config = self._loop.compile_settings
         # Check duration dependency: summarise_last with no explicit timing
         is_duration_dep = (loop_config.summarise_last 
                            and loop_config._sample_summaries_every is None)
     
         if is_duration_dep:
             computed_sample_summaries_every = chunk_duration / 100.0
             updates_dict["sample_summaries_every"] = (
                 computed_sample_summaries_every
             )
     
             # Emit warning once
             if not self._timing_warning_emitted:
                 warnings.warn(
                     "Summary metrics were requested with no "
                     "summarise_every or sample_summaries_every timing. "
                     "Sample_summaries_every was set to duration / 100 by "
                     "default. If duration changes, the kernel will need "
                     "to recompile, which will cause a slow integration "
                     "(once). Set timing parameters explicitly to avoid "
                     "this.",
                     UserWarning,
                     stacklevel=3
                 )
                 self._timing_warning_emitted = True
     ```
   - Edge cases:
     - chunk_duration is None -> skip entire block
     - chunk_duration provided but is_duration_dep is False -> skip computation
     - chunk_duration provided and is_duration_dep is True -> compute and warn once
   - Integration: The computed sample_summaries_every is added to updates_dict and processed by loop.update()

**Tests to Create**:
- Test file: tests/integrators/test_SingleIntegratorRun.py
- Test function: test_chunk_duration_computes_sample_summaries_every_when_duration_dependent
- Description: Verify that when chunk_duration is provided and is_duration_dependent is True, sample_summaries_every is computed. Create a SingleIntegratorRun with summarise_last=True and no timing, call update with chunk_duration, verify sample_summaries_every was set to chunk_duration/100.
- Test function: test_chunk_duration_skipped_when_not_duration_dependent
- Description: Verify that when is_duration_dependent is False (explicit sample_summaries_every provided), chunk_duration does not trigger computation
- Test function: test_timing_warning_emitted_once
- Description: Verify warning is only emitted once across multiple updates with chunk_duration

**Tests to Run**:
- tests/integrators/test_SingleIntegratorRun.py::TestChunkDurationInterception::test_sample_summaries_every_computed_from_chunk_duration
- tests/integrators/test_SingleIntegratorRun.py::TestChunkDurationSkipped::test_chunk_duration_skipped_when_not_duration_dependent
- tests/integrators/test_SingleIntegratorRun.py::TestChunkDurationInterception::test_timing_warning_emitted_once

**Outcomes**:
- Files Modified: 
  * src/cubie/integrators/SingleIntegratorRunCore.py (15 lines changed)
  * tests/integrators/test_SingleIntegratorRun.py (67 lines added)
- Functions/Methods Added/Modified:
  * __init__() in SingleIntegratorRunCore.py - added sentinel sample_summaries_every update for duration-dependent builds
  * update() in SingleIntegratorRunCore.py - refactored inline checks to use is_duration_dep variable
  * test_timing_warning_emitted_once() - new test in TestChunkDurationInterception
  * test_chunk_duration_skipped_when_not_duration_dependent() - new test in TestChunkDurationSkipped class
- Implementation Summary:
  Refactored the chunk_duration handling in update() to use `is_duration_dep` variable (matching is_duration_dependent property logic) instead of separate variable assignments. Added sentinel sample_summaries_every update (0.01) in __init__ when duration-dependent to avoid NaN during initial build. Created two new test functions: one to verify warning is only emitted once, and one to verify chunk_duration is skipped when explicit sample_summaries_every is set.
- Issues Flagged: None

---

## Summary

### Total Task Groups: 4

### Dependency Chain:
```
Group 1 (API fix) ────────────┐
                              ├──> Group 3 (is_duration_dependent) ──> Group 4 (gate chunk_duration)
Group 2 (sentinel removal) ───┘
```

Groups 1 and 2 are independent and can run in parallel.
Group 3 depends on Groups 1 and 2 being complete.
Group 4 depends on Group 3.

### Tests Summary:
- **Tests to Create**: 7 new test functions across 3 test files
- **Tests to Run**: 10 test function calls

### Estimated Complexity:
- **Group 1**: Low - Single line removal in test file
- **Group 2**: Low - Simple refactor, move sentinel logic from property to caller
- **Group 3**: Low - Add single read-only property
- **Group 4**: Low - Minor refactor of existing inline logic

**Overall Complexity**: Low - All changes are surgical modifications to existing code patterns.
