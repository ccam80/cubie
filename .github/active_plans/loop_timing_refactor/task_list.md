# Implementation Task List
# Feature: Loop Timing Refactor Completion
# Plan Reference: .github/active_plans/loop_timing_refactor/agent_plan.md

## Task Group 1: Replace selp with min for dt_eff Calculation
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/loops/ode_loop.py (lines 609-624)
- File: .github/context/cubie_internal_structure.md (for CUDAFactory pattern)

**Input Validation Required**:
- None - this is a pure optimization change

**Tasks**:
1. **Replace selp with min for next_event calculation**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Details:
     ```python
     # Current code (lines 614-618):
     next_event = selp(
         next_save < next_update_summary,
         next_save,
         next_update_summary
     )
     
     # Replace with:
     next_event = min(next_save, next_update_summary)
     ```
   - Edge cases: None - min() works identically to the selp pattern
   - Integration: No other changes required; min() is a Python builtin supported by Numba CUDA

**Tests to Create**:
- None - existing tests cover the loop behavior

**Tests to Run**:
- tests/integrators/loops/test_ode_loop.py

**Outcomes**:
- Files Modified:
  * src/cubie/integrators/loops/ode_loop.py (4 lines changed - removed 5 lines, added 1 line)
- Functions/Methods Modified:
  * build() method in ODELoopFactory class - modified next_event calculation
- Implementation Summary:
  Replaced the selp-based conditional pattern for computing next_event with the simpler min() builtin. Both produce identical results, but min() is cleaner and Numba CUDA supports it natively.
- Issues Flagged: None

---

## Task Group 2: Remove Timing Aliases in ode_loop.py
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/loops/ode_loop.py (lines 366-373, 546-548, 610-611, 748, 768-772)
- File: src/cubie/integrators/loops/ode_loop_config.py (lines 329-357 - property definitions)

**Input Validation Required**:
- None - refactoring only, no new validation

**Tasks**:
1. **Remove dt_save and dt_update_summaries alias declarations**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Details:
     ```python
     # Current code (lines 367-369):
     updates_per_summary = config.samples_per_summary
     dt_save = precision(config.save_every)
     dt_update_summaries = precision(config.sample_summaries_every)
     
     # Replace with (remove alias lines, keep updates_per_summary):
     updates_per_summary = config.samples_per_summary
     ```
   - Edge cases: None
   - Integration: Must update all references to dt_save and dt_update_summaries

2. **Replace dt_save references with config.save_every**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Details:
     Replace all occurrences:
     - Line 546: `next_save += dt_save` → `next_save += config.save_every`
     - Line 547: `next_update_summary += dt_update_summaries` → `next_update_summary += config.sample_summaries_every`
     - Line 748: `next_save = selp(do_save, next_save + dt_save, next_save)` → `next_save = selp(do_save, next_save + config.save_every, next_save)`
     - Line 770: `next_update_summary + dt_update_summaries` → `next_update_summary + config.sample_summaries_every`
   - Edge cases: config properties already return precision-cast values, so no additional casting needed
   - Integration: Properties return `self.precision(self._attribute)`, so values are already cast

**Tests to Create**:
- None - existing tests cover the loop behavior

**Tests to Run**:
- tests/integrators/loops/test_ode_loop.py

**Outcomes**:
- Files Modified:
  * src/cubie/integrators/loops/ode_loop.py (6 lines changed - removed 2 alias lines, updated 4 references)
- Functions/Methods Modified:
  * build() method in ODELoopFactory class - removed dt_save and dt_update_summaries aliases
- Implementation Summary:
  Removed the dt_save and dt_update_summaries alias variables and replaced all references with direct config property access (config.save_every and config.sample_summaries_every). The config properties already return precision-cast values, so no additional casting is needed.
- Issues Flagged: None

---

## Task Group 3: Tolerant Integer Multiple Validation
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/loops/ode_loop_config.py (lines 254-327)
- File: .github/copilot-instructions.md (for style guidelines)

**Input Validation Required**:
- ratio = summarise_every / sample_summaries_every
- If abs(ratio - round(ratio)) <= 0.01 (1%): auto-adjust with warning
- If deviation > 0.01: raise ValueError with clear message

**Tasks**:
1. **Implement tolerant validation with auto-adjustment**
   - File: src/cubie/integrators/loops/ode_loop_config.py
   - Action: Modify
   - Details:
     ```python
     # Current code (lines 317-327):
     if not self.summarise_last:
         tolerance = 1e-3 #0.1% timing tolerance (summary freq vs sampling)
         ratio = self._summarise_every / self._sample_summaries_every
         if abs(ratio - round(ratio)) > tolerance:
             raise ValueError(
                 f"sample_summaries_every ({self._sample_summaries_every}) "
                 f"must be an integer divisor of summarise_every "
                 f"({self._summarise_every}). Ratio: {ratio}"
             )
     
     # Replace with:
     if not self.summarise_last:
         from warnings import warn
         ratio = self._summarise_every / self._sample_summaries_every
         deviation = abs(ratio - round(ratio))
         
         if deviation <= 0.01:  # Within 1% - auto-adjust with warning
             rounded_ratio = round(ratio)
             adjusted = rounded_ratio * self._sample_summaries_every
             if adjusted != self._summarise_every:
                 warn(
                     f"summarise_every adjusted from {self._summarise_every} "
                     f"to {adjusted}, the nearest integer multiple of "
                     f"sample_summaries_every ({self._sample_summaries_every})"
                 )
                 self._summarise_every = adjusted
         else:  # More than 1% off - incompatible values
             raise ValueError(
                 f"summarise_every ({self._summarise_every}) must be an "
                 f"integer multiple of sample_summaries_every "
                 f"({self._sample_summaries_every}). The ratio {ratio:.4f} "
                 f"is not close to any integer."
             )
     ```
   - Edge cases: 
     - Float precision issues where values are slightly off
     - Truly incompatible values like 0.2 and 0.5
   - Integration: Import warnings at top of function block

**Tests to Create**:
- Test file: tests/integrators/loops/test_dt_update_summaries_validation.py
- Test function: test_tolerant_validation_auto_adjusts
- Description: Verify auto-adjustment when within 1% tolerance
- Test function: test_tolerant_validation_warns_on_adjustment
- Description: Verify warning is issued when adjustment made
- Test function: test_tolerant_validation_errors_on_incompatible
- Description: Verify ValueError for truly incompatible values

**Tests to Run**:
- tests/integrators/loops/test_dt_update_summaries_validation.py

**Outcomes**:
- Files Modified:
  * src/cubie/integrators/loops/ode_loop_config.py (14 lines changed - replaced 11 lines with 24 lines)
  * tests/integrators/loops/test_dt_update_summaries_validation.py (76 lines added)
- Functions/Methods Modified:
  * __attrs_post_init__() in ODELoopConfig class - replaced simple tolerance check with 1% tolerant validation with auto-adjustment and warning
- Implementation Summary:
  Replaced the existing 0.1% tolerance check with a new 1% tolerant validation. When summarise_every is within 1% of an integer multiple of sample_summaries_every, it auto-adjusts to the nearest multiple and emits a warning. Values more than 1% off raise a ValueError. Also updated existing test_sample_must_divide_summarise to match new error message and added three new tests.
- Issues Flagged: None

---

## Task Group 4: Remove Sentinel Values and Update Default Handling
**Status**: [x]
**Dependencies**: Task Group 3 (tolerant validation must be in place first)

**Required Context**:
- File: src/cubie/integrators/loops/ode_loop_config.py (lines 254-327)
- File: src/cubie/outputhandling/output_config.py (lines 141-185)
- File: src/cubie/integrators/loops/ode_loop.py (lines 366-380)

**Input Validation Required**:
- When all timing params are None: set flags only, no sentinel values
- Compile-time validation when outputs require timing

**Tasks**:
1. **Remove sentinel value assignment in ODELoopConfig**
   - File: src/cubie/integrators/loops/ode_loop_config.py
   - Action: Modify
   - Details:
     ```python
     # Current code (lines 261-273):
     if (
         self._save_every is None
         and self._summarise_every is None
         and self._sample_summaries_every is None
     ):
         self.save_last = True
         self.summarise_last = True
         # Set sentinel values for loop timing (will be overridden)
         self._save_every = 0.1
         self._summarise_every = 1.0
         self._sample_summaries_every = 0.1
         return  # Skip validation when using save_last/summarise_last
     
     # Replace with:
     if (
         self._save_every is None
         and self._summarise_every is None
         and self._sample_summaries_every is None
     ):
         self.save_last = True
         self.summarise_last = True
         return  # Skip validation when using save_last/summarise_last only
     ```
   - Edge cases: Loop build() must handle None timing values
   - Integration: ode_loop.py must check for None before using timing values

2. **Update output_config.py default for save_every**
   - File: src/cubie/outputhandling/output_config.py
   - Action: Modify
   - Details:
     ```python
     # Current code (lines 182-185):
     _save_every: float = attrs.field(
         default=0.01,
         validator=opt_gttype_validator(float, 0.0)
     )
     
     # Replace with:
     _save_every: Optional[float] = attrs.field(
         default=None,
         validator=opt_gttype_validator(float, 0.0)
     )
     ```
   - Edge cases: None values must be handled at compile time
   - Integration: Ensure compile_flags validation catches missing required values

3. **Handle None timing values in ode_loop.py build()**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Details:
     In the build() method (around line 366-373), add conditional handling:
     ```python
     # When save_last or summarise_last is True, use placeholder timing values
     # that will never trigger in the loop (loop exits on save_last/summarise_last logic)
     save_every_val = config.save_every if config.save_every is not None else precision(float('inf'))
     sample_summaries_val = config.sample_summaries_every if config.sample_summaries_every is not None else precision(float('inf'))
     ```
   - Edge cases: save_last and summarise_last flags indicate end-of-run-only behavior
   - Integration: Uses existing save_last/summarise_last flag logic

**Tests to Create**:
- Test file: tests/integrators/loops/test_dt_update_summaries_validation.py
- Test function: test_all_none_no_sentinels
- Description: Verify that all-None config does NOT set sentinel timing values
- Test function: test_all_none_sets_flags_only
- Description: Verify only save_last and summarise_last flags are set

**Tests to Run**:
- tests/integrators/loops/test_dt_update_summaries_validation.py

**Outcomes**:
- Files Modified:
  * src/cubie/integrators/loops/ode_loop_config.py (4 lines removed)
  * src/cubie/outputhandling/output_config.py (2 lines changed)
  * src/cubie/integrators/loops/ode_loop.py (17 lines added/modified)
  * tests/integrators/loops/test_dt_update_summaries_validation.py (55 lines added)
- Functions/Methods Added/Modified:
  * __attrs_post_init__() in ODELoopConfig - removed sentinel value assignments
  * _save_every field in OutputConfig - changed default from 0.01 to None
  * build() in ODELoopFactory - added save_every_val and sample_summaries_val with infinity fallback for None timing values
- Implementation Summary:
  Removed sentinel value assignments when all timing parameters are None. Now the timing attributes remain None and only save_last/summarise_last flags are set. In ode_loop.py, added conditional handling to use infinity as a placeholder when timing values are None, ensuring timing events never trigger when using end-of-run-only behavior. Updated output_config.py to default _save_every to None. Added two new tests and updated test_all_none_uses_defaults to verify new behavior.
- Issues Flagged: None

---

## Task Group 5: Implement summarise_last Logic in ode_loop
**Status**: [x]
**Dependencies**: Task Group 2 (alias removal should be complete)

**Required Context**:
- File: src/cubie/integrators/loops/ode_loop.py (lines 592-608, 767-793)
- File: .github/context/cubie_internal_structure.md (for CUDA device code patterns)

**Input Validation Required**:
- None - implementing existing flag behavior

**Tasks**:
1. **Implement summarise_last flag handling in main loop**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Details:
     After the existing save_last handling (around line 601), add parallel logic for summarise_last:
     ```python
     # After save_last block, add:
     if summarise_last:
         # Mirror save_last pattern for summary collection at end of run
         at_last_summarise = finished and t_prec < t_end
         # Force summary update on final step if using end-of-run summaries
     ```
     
     At the end of the loop where summaries are processed (around line 773-793), add final summary handling:
     ```python
     # Add condition to force final summary when summarise_last is True
     if summarise_last and finished:
         # Force final summary collection
         if summarise:
             statesumm_idx = summary_idx * summarise_state_bool
             obssumm_idx = summary_idx * summarise_obs_bool
             update_summaries(
                 state_buffer,
                 observables_buffer,
                 state_summary_buffer,
                 observable_summary_buffer,
                 update_idx
             )
             update_idx += int32(1)
             save_summaries(
                 state_summary_buffer,
                 observable_summary_buffer,
                 state_summaries_output[statesumm_idx,:],
                 observable_summaries_output[obssumm_idx,:],
                 updates_per_summary,
             )
     ```
   - Edge cases: 
     - Avoid double-write when regular summaries and last summary coincide
     - Ensure update_idx is correctly managed
   - Integration: Follows existing save_last pattern

**Tests to Create**:
- Test file: tests/integrators/loops/test_ode_loop.py
- Test function: test_summarise_last_collects_final_summary
- Description: Verify summaries collected at end of run with summarise_last=True
- Test function: test_summarise_last_with_summarise_every
- Description: Verify both can be used together without double-write

**Tests to Run**:
- tests/integrators/loops/test_ode_loop.py

**Outcomes**:
- Files Modified:
  * src/cubie/integrators/loops/ode_loop.py (38 lines changed)
- Functions/Methods Added/Modified:
  * build() method in IVPLoop class - implemented summarise_last flag handling
- Implementation Summary:
  1. Added at_last_summarise flag tracking in main loop (mirroring save_last pattern)
  2. Added logic to keep loop running when at_last_summarise is True
  3. Modified do_save and do_update_summary to include at_last flags for forced save/summary
  4. Updated dt_eff calculation to include t_end when on final step
  5. Modified summary save logic to force save when at_last_summarise is True (using OR condition to avoid double-write)
  6. Added None handling for samples_per_summary when timing params are None (defaults to 1)
  7. Reordered timing value extraction to handle None params before accessing samples_per_summary
- Tests Created:
  * test_summarise_last_collects_final_summary - verifies summaries collected at end of run
  * test_summarise_last_with_summarise_every - verifies both work together without double-write
- Issues Flagged: None

---

## Task Group 6: Remove Mutual Exclusivity Assumption
**Status**: [x]
**Dependencies**: Task Group 5 (summarise_last implementation)

**Required Context**:
- File: src/cubie/integrators/loops/ode_loop_config.py (entire file)
- File: src/cubie/integrators/loops/ode_loop.py (lines 592-608)

**Input Validation Required**:
- None - removing restrictions

**Tasks**:
1. **Search and remove any mutual exclusivity validation**
   - File: src/cubie/integrators/loops/ode_loop_config.py
   - Action: Modify (if needed)
   - Details:
     Search for any validation that prevents:
     - save_last=True when save_every is set
     - summarise_last=True when summarise_every is set
     
     If found, remove the validation. Based on code review, there appears to be no explicit mutual exclusivity check, but verify and document.
   - Edge cases: None
   - Integration: Allows more flexible user configurations

2. **Ensure loop logic handles combined flags correctly**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Verify/Modify
   - Details:
     Verify that the loop correctly handles:
     - save_last=True AND save_every set (periodic saves + final save)
     - summarise_last=True AND summarise_every set (periodic summaries + final summary)
     
     The existing code structure should already support this, but verify no conflicts.
   - Edge cases: Boundary conditions where regular save coincides with end of run
   - Integration: Uses predicated commits with selp()

**Tests to Create**:
- Test file: tests/integrators/loops/test_ode_loop.py
- Test function: test_save_last_with_save_every
- Description: Verify save_last and save_every can be used together
- Test function: test_summarise_last_with_summarise_every_combined
- Description: Verify summarise_last and summarise_every can be used together

**Tests to Run**:
- tests/integrators/loops/test_ode_loop.py
- tests/integrators/loops/test_dt_update_summaries_validation.py

**Outcomes**:
- Files Modified:
  * tests/integrators/loops/test_ode_loop.py (72 lines added)
- Functions/Methods Added:
  * test_save_last_with_save_every() - tests periodic saves with save_last enabled
  * test_summarise_last_with_summarise_every_combined() - tests periodic summaries with summarise_last enabled
- Implementation Summary:
  Searched ode_loop_config.py for mutual exclusivity validation - **none found**. The save_last and summarise_last boolean fields (lines 194-201) have no validators that prevent combining with save_every/summarise_every. The __attrs_post_init__ method (lines 261-269) only auto-sets these flags when all timing params are None, but does not enforce exclusivity. Verified ode_loop.py loop logic (lines 616-629) already supports combined usage: save_last forces final save via at_last_save flag, summarise_last forces final summary via at_last_summarise flag. The OR condition at line 819 (save_summary_now includes at_last_summarise) prevents double-writes when end time coincides with periodic summary. Added two tests to verify this combined behavior works correctly.
- Issues Flagged: None

---

## Task Group 7: Add sample_summaries_every to Config Plumbing Tests
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: tests/batchsolving/test_config_plumbing.py (lines 273-330)
- File: tests/conftest.py (lines 372-461)

**Input Validation Required**:
- None - adding test coverage

**Tasks**:
1. **Add sample_summaries_every to assert_ivploop_config**
   - File: tests/batchsolving/test_config_plumbing.py
   - Action: Modify
   - Details:
     ```python
     # Add to assert_ivploop_config function (around line 283):
     assert loop.sample_summaries_every == pytest.approx(
         settings["sample_summaries_every"], 
         rel=tolerance.rel_tight, 
         abs=tolerance.abs_tight
     )
     
     # Add compile_settings check:
     assert cs.sample_summaries_every == pytest.approx(
         settings["sample_summaries_every"], 
         rel=tolerance.rel_tight, 
         abs=tolerance.abs_tight
     )
     ```
   - Edge cases: None
   - Integration: Must add sample_summaries_every to solver_settings fixture

2. **Add sample_summaries_every to solver_settings fixture**
   - File: tests/conftest.py
   - Action: Modify
   - Details:
     ```python
     # Add to defaults dict (around line 388):
     "sample_summaries_every": precision(0.05),  # Match save_every default
     ```
     
     ```python
     # Add to float_keys set (around line 426):
     "sample_summaries_every",
     ```
   - Edge cases: None
   - Integration: Ensures fixture provides the value for all tests

3. **Add sample_summaries_every to test updates dict**
   - File: tests/batchsolving/test_config_plumbing.py
   - Action: Modify
   - Details:
     ```python
     # Add to updates dict in test_comprehensive_config_plumbing (around line 602):
     "sample_summaries_every": precision(0.05),
     ```
   - Edge cases: Must be integer multiple of summarise_every
   - Integration: Uses same value as save_every for consistency

**Tests to Create**:
- None - modifying existing test

**Tests to Run**:
- tests/batchsolving/test_config_plumbing.py

**Outcomes**:
- Files Modified:
  * tests/batchsolving/test_config_plumbing.py (11 lines changed)
  * tests/conftest.py (2 lines changed)
- Functions/Methods Added/Modified:
  * assert_ivploop_config() in test_config_plumbing.py - added sample_summaries_every assertions for loop and compile_settings
  * solver_settings fixture in conftest.py - added sample_summaries_every to defaults dict and float_keys set
  * test_comprehensive_config_plumbing in test_config_plumbing.py - added sample_summaries_every to updates dict
- Implementation Summary:
  Added sample_summaries_every parameter to the config plumbing tests. The parameter is now included in the solver_settings fixture defaults (0.05), added to the float_keys set for proper precision handling, and assertions added to verify it flows through to both the loop and compile_settings objects. In the comprehensive config plumbing test, sample_summaries_every is set to 0.05 which is an integer divisor of summarise_every (0.15), ensuring proper validation.
- Issues Flagged: None

---

## Task Group 8: Consolidate SolveResult Field Tests
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: tests/batchsolving/test_solveresult.py (lines 575-630)
- File: src/cubie/batchsolving/solveresult.py (lines 52-119 - SolveSpec class)

**Input Validation Required**:
- None - refactoring tests only

**Tasks**:
1. **Replace individual field tests with single comprehensive test**
   - File: tests/batchsolving/test_solveresult.py
   - Action: Modify
   - Details:
     ```python
     # Remove individual tests (lines 578-630):
     # - test_solvespec_save_every_field
     # - test_solvespec_summarise_every_field
     
     # Replace with single comprehensive test:
     def test_solvespec_has_all_expected_attributes(self):
         """Verify SolveSpec has all required attributes."""
         from cubie.batchsolving.solveresult import SolveSpec
     
         expected_attrs = [
             'dt', 'dt_min', 'dt_max', 'save_every', 'summarise_every',
             'atol', 'rtol', 'duration', 'warmup', 't0', 'algorithm',
             'saved_states', 'saved_observables', 'summarised_states',
             'summarised_observables', 'output_types', 'precision'
         ]
     
         spec = SolveSpec(
             dt=0.001,
             dt_min=0.0001,
             dt_max=0.01,
             save_every=0.1,
             summarise_every=1.0,
             atol=1e-6,
             rtol=1e-3,
             duration=10.0,
             warmup=0.0,
             t0=0.0,
             algorithm="euler",
             saved_states=["x0"],
             saved_observables=None,
             summarised_states=["x0"],
             summarised_observables=None,
             output_types=["state", "mean"],
             precision="float32",
         )
     
         for attr in expected_attrs:
             assert hasattr(spec, attr), f"SolveSpec missing attribute: {attr}"
     ```
   - Edge cases: None
   - Integration: Simpler, more maintainable test structure

**Tests to Create**:
- None - replacing existing tests

**Tests to Run**:
- tests/batchsolving/test_solveresult.py

**Outcomes**:
- Files Modified:
  * tests/batchsolving/test_solveresult.py (21 lines removed, 35 lines added)
- Functions/Methods Added/Modified:
  * Removed test_solvespec_save_every_field() in TestSolveSpecFields
  * Removed test_solvespec_summarise_every_field() in TestSolveSpecFields
  * Added test_solvespec_has_all_expected_attributes() in TestSolveSpecFields
- Implementation Summary:
  Consolidated two individual field tests into a single comprehensive test that verifies all 16 expected attributes of SolveSpec. The new test creates a SolveSpec instance with all required parameters and iterates through the expected_attrs list to verify each attribute exists on the spec object.
- Issues Flagged: None

---

## Task Group 9: Update Timing Validation Tests
**Status**: [x]
**Dependencies**: Task Groups 3, 4 (validation changes must be complete)

**Required Context**:
- File: tests/integrators/loops/test_dt_update_summaries_validation.py (entire file)

**Input Validation Required**:
- None - updating tests to match new behavior

**Tasks**:
1. **Update test_all_none_uses_defaults to remove sentinel assertions**
   - File: tests/integrators/loops/test_dt_update_summaries_validation.py
   - Action: Modify
   - Details:
     ```python
     # Current test (lines 8-27):
     def test_all_none_uses_defaults():
         """Test that all None sets save_last and summarise_last flags."""
         config = ODELoopConfig(...)
         
         # Sentinel values still set for loop timing calculations
         assert config.save_every == pytest.approx(0.1)  # REMOVE
         assert config.summarise_every == pytest.approx(1.0)  # REMOVE
         assert config.sample_summaries_every == pytest.approx(0.1)  # REMOVE
         # But flags indicate end-of-run-only behavior
         assert config.save_last is True
         assert config.summarise_last is True
     
     # Replace with:
     def test_all_none_uses_defaults():
         """Test that all None sets save_last and summarise_last flags."""
         config = ODELoopConfig(...)
         
         # No sentinel values - timing params remain None
         assert config._save_every is None
         assert config._summarise_every is None
         assert config._sample_summaries_every is None
         # Flags indicate end-of-run-only behavior
         assert config.save_last is True
         assert config.summarise_last is True
     ```
   - Edge cases: None
   - Integration: Matches new behavior without sentinels

2. **Add test for compile-time validation with outputs**
   - File: tests/integrators/loops/test_dt_update_summaries_validation.py
   - Action: Add
   - Details:
     ```python
     def test_none_timing_with_state_output_valid_with_save_last():
         """Test that None timing params are valid when save_last is set."""
         from cubie.outputhandling.output_config import OutputCompileFlags
         
         config = ODELoopConfig(
             n_states=3,
             n_parameters=0,
             n_drivers=0,
             n_observables=0,
             n_error=0,
             n_counters=0,
             state_summaries_buffer_height=0,
             observable_summaries_buffer_height=0,
             compile_flags=OutputCompileFlags(save_state=True),
         )
         
         # Should be valid because save_last defaults to True
         assert config.save_last is True
     ```
   - Edge cases: None
   - Integration: Validates compile-time behavior

**Tests to Create**:
- Test function: test_none_timing_with_state_output_valid_with_save_last (as described above)

**Tests to Run**:
- tests/integrators/loops/test_dt_update_summaries_validation.py

**Outcomes**:
- Files Modified:
  * tests/integrators/loops/test_dt_update_summaries_validation.py (18 lines added)
- Functions/Methods Added:
  * test_none_timing_with_state_output_valid_with_save_last() - tests that None timing params are valid when save_last is set with OutputCompileFlags(save_state=True)
- Implementation Summary:
  Task 1 (update test_all_none_uses_defaults) was already complete - the test already asserts timing params remain None instead of sentinel values. Added new test test_none_timing_with_state_output_valid_with_save_last that verifies ODELoopConfig with OutputCompileFlags(save_state=True) works correctly when timing params are None, with save_last defaulting to True.
- Issues Flagged: None

---

## Task Group 10: Update CPU Reference Implementation
**Status**: [x]
**Dependencies**: Task Groups 1-6 (core changes should be complete)

**Required Context**:
- File: tests/integrators/cpu_reference/loops.py (entire file)
- File: tests/_utils.py (calculate_expected_summaries function, if exists)

**Input Validation Required**:
- None - updating reference implementation

**Tasks**:
1. **Replace dt_save and dt_summarise with new names**
   - File: tests/integrators/cpu_reference/loops.py
   - Action: Modify
   - Details:
     ```python
     # Line 91-92 (current):
     dt_save = precision(solver_settings["save_every"])
     dt_summarise = precision(solver_settings["summarise_every"])
     
     # No change needed - already uses new names save_every and summarise_every!
     # Just verify variable naming matches:
     save_every = precision(solver_settings["save_every"])
     summarise_every = precision(solver_settings["summarise_every"])
     ```
   - Edge cases: None
   - Integration: Variable names should match source code

2. **Add sample_summaries_every handling**
   - File: tests/integrators/cpu_reference/loops.py
   - Action: Modify
   - Details:
     ```python
     # Add around line 93:
     sample_summaries_every = precision(
         solver_settings.get("sample_summaries_every", 
                             solver_settings["save_every"])
     )
     
     # Update line 211 (current):
     summarise_every = int(dt_summarise / dt_save)
     
     # Replace with:
     samples_per_summary = int(summarise_every / sample_summaries_every)
     ```
     
     Update calculate_expected_summaries call to use new variable name.
   - Edge cases: Backwards compatibility with tests not providing sample_summaries_every
   - Integration: Default to save_every if not provided

3. **Fix variable name clash (summarise_every as integer)**
   - File: tests/integrators/cpu_reference/loops.py
   - Action: Modify
   - Details:
     Line 211 reuses summarise_every as an integer (samples between summaries).
     Rename to samples_per_summary to match source code naming.
     ```python
     # Current (line 211):
     summarise_every = int(dt_summarise / dt_save)
     
     # Replace with:
     samples_per_summary = int(summarise_every / sample_summaries_every)
     ```
   - Edge cases: None
   - Integration: Clearer naming that matches source code

**Tests to Create**:
- None - CPU reference is used by other tests

**Tests to Run**:
- tests/integrators/cpu_reference/test_cpu_utils.py

**Outcomes**:
- Files Modified:
  * tests/integrators/cpu_reference/loops.py (8 lines changed)
- Functions/Methods Modified:
  * run_reference_loop() - renamed dt_save to save_every, dt_summarise to summarise_every, added sample_summaries_every handling with backward-compatible default, fixed variable name clash by renaming integer summarise_every to samples_per_summary
- Implementation Summary:
  1. Renamed dt_save to save_every and dt_summarise to summarise_every (lines 91-92)
  2. Added sample_summaries_every extraction with fallback to save_every for backwards compatibility (lines 93-96)
  3. Updated all references: max_save_samples calculation (line 127), next_save_time initialization (line 148), next_save_time update (line 194)
  4. Fixed variable name clash: renamed integer summarise_every to samples_per_summary (line 215)
  5. Updated calculate_expected_summaries call to use samples_per_summary and save_every (lines 222, 226)
- Issues Flagged: None

---

## Summary

**Total Task Groups**: 10
**Dependency Chain**:
1. Task Groups 1, 2, 7, 8 can run in parallel (no dependencies)
2. Task Group 3 must complete before Task Group 4
3. Task Group 2 should complete before Task Group 5 (alias removal)
4. Task Group 5 must complete before Task Group 6
5. Task Groups 3, 4 must complete before Task Group 9
6. Task Groups 1-6 should complete before Task Group 10

**Tests to Create (summary)**:
- test_tolerant_validation_auto_adjusts
- test_tolerant_validation_warns_on_adjustment
- test_tolerant_validation_errors_on_incompatible
- test_all_none_no_sentinels
- test_all_none_sets_flags_only
- test_summarise_last_collects_final_summary
- test_summarise_last_with_summarise_every
- test_save_last_with_save_every
- test_summarise_last_with_summarise_every_combined
- test_none_timing_with_state_output_valid_with_save_last
- test_solvespec_has_all_expected_attributes (replaces individual field tests)

**Estimated Complexity**: Medium
- Most changes are straightforward refactoring
- Tolerant validation and summarise_last implementation are the most complex
- Test updates are mechanical but numerous
