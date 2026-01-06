# Implementation Review Report
# Feature: Loop Timing Refactor
# Review Date: 2026-01-05
# Reviewer: Harsh Critic Agent

## Executive Summary

The loop-timing refactor implementation successfully addresses the majority of user stories. The core timing parameter changes (`save_every`, `summarise_every`, `sample_summaries_every`), tolerant validation, and `summarise_last` logic have been implemented. The code follows repository conventions and the architectural patterns are consistent.

However, **36 test failures** were reported, indicating critical issues. The root cause is **properties that crash when timing attributes are None**. The properties `save_every`, `summarise_every`, `sample_summaries_every` in `ODELoopConfig` (lines 354-366) call `self.precision(self._attribute)` but when all timing params are None (as happens with save_last/summarise_last), this raises `TypeError: unsupported operand type(s) for *: 'type' and 'NoneType'`. Similarly, `samples_per_summary` (line 339-341) divides potentially None values.

Additionally, the `SolveSpec` class is **missing the `sample_summaries_every` attribute**, which means the new timing parameter cannot be captured in solve results.

The implementation is approximately 85% complete. The remaining issues are property None-handling and missing attribute in SolveSpec.

## User Story Validation

**User Stories** (from human_overview.md):

- **US-1: Simplified Timing Configuration**: **Met** - None/0 defaults now work correctly, sentinel values removed, validation deferred to compile-time via `save_last`/`summarise_last` flags.

- **US-2: Decoupled Summary Sampling**: **Partial** - `sample_summaries_every` is independent from `save_every`, but the 36 test failures suggest the integration between sampling and summary output is broken.

- **US-3: Flexible Save/Summarise Combinations**: **Met** - `save_last` and `save_every` are no longer mutually exclusive. Tests `test_save_last_with_save_every` and `test_summarise_last_with_summarise_every_combined` added.

- **US-4: End-of-Run Summarise Logic**: **Partial** - `summarise_last` flag handling implemented in ode_loop.py, but test failures indicate the logic doesn't work correctly in all cases.

- **US-5: Updated CPU Reference Implementation**: **Met** - CPU reference updated with new naming scheme (`save_every`, `summarise_every`, `sample_summaries_every`, `samples_per_summary`).

**Acceptance Criteria Assessment**: 4 of 5 user stories are fully or mostly met. US-2 and US-4 have implementation gaps evidenced by test failures.

## Goal Alignment

**Original Goals** (from human_overview.md):

- **Replace selp with min for dt_eff**: **Achieved** - Line 647 uses `min(next_save, next_update_summary)`.

- **Remove timing aliases**: **Achieved** - `dt_save` and `dt_update_summaries` removed; direct `config.save_every` and `config.sample_summaries_every` access used.

- **Tolerant integer multiple validation**: **Achieved** - 1% tolerance with auto-adjustment and warnings implemented in `ode_loop_config.py` lines 316-336.

- **Remove sentinel values**: **Achieved** - `__attrs_post_init__` no longer sets sentinel values when all params are None.

- **Implement summarise_last**: **Partial** - Logic added but doesn't function correctly in all scenarios.

**Assessment**: The architectural goals are achieved, but runtime behavior has issues.

## Code Quality Analysis

### Duplication

- **Location**: None significant. The implementation follows DRY principles.

### Unnecessary Complexity

- **Location**: `src/cubie/integrators/loops/ode_loop.py`, lines 616-629

- **Issue**: The `at_last_save` and `at_last_summarise` logic uses nested `selp` calls that could be simplified. The condition at line 625-627 is complex:
  ```python
  at_last_summarise = bool_(
      (next_update_summary > t_end) and (t_prec < t_end)
  )
  ```
  This is computed before checking `summarise_last`, meaning unnecessary computation occurs when the flag is False.

- **Impact**: Minor performance overhead and reduced readability.

### Unnecessary Additions

- **Location**: None found. All additions serve the stated goals.

### Convention Violations

- **PEP8**: No violations found. Line lengths comply with 79 character limit.

- **Type Hints**: Properly placed in function signatures.

- **Repository Patterns**: Import style follows guidelines (explicit imports for CUDAFactory files).

## Performance Analysis

- **CUDA Efficiency**: The `min()` replacement for `selp` is a valid optimization.

- **Memory Patterns**: No new allocations introduced; existing buffer patterns maintained.

- **Buffer Reuse**: No new opportunities identified in this refactor.

- **Math vs Memory**: The timing calculations use direct arithmetic; no memory access concerns.

- **Optimization Opportunities**: None significant beyond the existing implementation.

## Architecture Assessment

- **Integration Quality**: Good. Changes integrate cleanly with existing CUDAFactory and buffer registry patterns.

- **Design Patterns**: Consistent with attrs-based configuration and device function compilation.

- **Future Maintainability**: The separation of timing validation in `ODELoopConfig.__attrs_post_init__` is clean and testable.

## Edge Case Coverage Issues

### Issue 1: Property Access on None Values (CRITICAL - Root Cause of Test Failures)

- **Location**: `src/cubie/integrators/loops/ode_loop_config.py`, lines 354-366
- **Issue**: Properties `save_every`, `summarise_every`, `sample_summaries_every` call `self.precision(self._attribute)` but when attributes are None, this raises `TypeError: unsupported operand type(s) for *: 'type' and 'NoneType'`.
- **Impact**: All tests that create an ODELoopConfig with None timing parameters and then access these properties crash. This is the root cause of the 36 test failures.
- **Fix**: Add None check in properties to return None when underlying attribute is None.

### Issue 2: samples_per_summary with None Values (CRITICAL)

- **Location**: `src/cubie/integrators/loops/ode_loop_config.py`, line 339-341
- **Issue**: `samples_per_summary` property calls `self.summarise_every / self.sample_summaries_every` which triggers the property accessors that crash on None.
- **Impact**: Property access crashes when timing params are None.
- **Fix**: Guard property with None check and return None.

### Issue 3: Missing sample_summaries_every in SolveSpec (CRITICAL)

- **Location**: `src/cubie/batchsolving/solveresult.py`, lines 52-119
- **Issue**: The `SolveSpec` attrs class has `save_every` and `summarise_every` but **does not have `sample_summaries_every`**.
- **Impact**: The new timing parameter cannot be captured in solve results, breaking the API contract for users expecting to retrieve this value.
- **Fix**: Add `sample_summaries_every: float = attrs.field(validator=getype_validator(float, 0.0))` to SolveSpec.

### Issue 4: Summary Index Tracking

- **Location**: `src/cubie/integrators/loops/ode_loop.py`, lines 803-829
- **Issue**: `summary_idx` is incremented after `save_summaries` call, but the `save_summary_now` condition includes `at_last_summarise` which may not correctly track whether a summary was already saved at the current time.
- **Impact**: Potential double-write or missed summaries when end time aligns with periodic summary.
- **Fix**: Track whether the final step summary was already saved to prevent duplicate writes.

## Suggested Edits

1. **[CRITICAL] Guard Property Access for None Timing Values**
   - Task Group: Reference to task group 4 in task_list.md
   - File: src/cubie/integrators/loops/ode_loop_config.py
   - Issue: Properties `save_every`, `summarise_every`, `sample_summaries_every` (lines 354-366) call `self.precision(self._attribute)` which crashes when internal attributes are None. This is the root cause of the 36 test failures.
   - Fix: Add None guard to properties:
     ```python
     @property
     def save_every(self) -> Optional[float]:
         """Return the output save interval, or None if not configured."""
         if self._save_every is None:
             return None
         return self.precision(self._save_every)
     
     @property
     def summarise_every(self) -> Optional[float]:
         """Return the summary interval, or None if not configured."""
         if self._summarise_every is None:
             return None
         return self.precision(self._summarise_every)
     
     @property
     def sample_summaries_every(self) -> Optional[float]:
         """Return the summary sampling interval, or None if not configured."""
         if self._sample_summaries_every is None:
             return None
         return self.precision(self._sample_summaries_every)
     ```
   - Rationale: Properties should handle None gracefully since None is now a valid state when save_last/summarise_last flags are used.
   - Status: ✅ APPLIED

2. **[CRITICAL] Guard samples_per_summary Property**
   - Task Group: Reference to task group 4 in task_list.md
   - File: src/cubie/integrators/loops/ode_loop_config.py
   - Issue: `samples_per_summary` property (line 339-341) divides `summarise_every / sample_summaries_every` but these call the properties which now return None, causing TypeError.
   - Fix: Add None check at start of property:
     ```python
     @property
     def samples_per_summary(self) -> Optional[int]:
         """Return the number of updates between summary outputs."""
         if self._summarise_every is None or self._sample_summaries_every is None:
             return None
         return round(self.summarise_every / self.sample_summaries_every)
     ```
   - Rationale: Prevents TypeError when timing params are None.
   - Status: ✅ APPLIED

3. **[CRITICAL] Add sample_summaries_every to SolveSpec**
   - Task Group: Not in original task list - should be added
   - File: src/cubie/batchsolving/solveresult.py
   - Issue: `SolveSpec` class is missing `sample_summaries_every` attribute. The new timing parameter is not captured in solve results.
   - Fix: Add attribute to SolveSpec class after line 96:
     ```python
     sample_summaries_every: float = attrs.field(
         validator=getype_validator(float, 0.0)
     )
     ```
     Also update the docstring to document the new attribute.
   - Rationale: Without this attribute, users cannot retrieve the sample_summaries_every value from solve results.
   - Status: ✅ APPLIED

4. **Add sample_summaries_every to SolveSpec Expected Attributes Test**
   - Task Group: Reference to task group 8 in task_list.md
   - File: tests/batchsolving/test_solveresult.py
   - Issue: `expected_attrs` list in `test_solvespec_has_all_expected_attributes` (line 582) is missing `sample_summaries_every`.
   - Fix: Add `'sample_summaries_every'` to the expected_attrs list.
   - Rationale: Ensures new timing parameter is validated in attribute test.
   - Status: ✅ APPLIED

5. **Fix at_last_summarise Condition Ordering**
   - Task Group: Reference to task group 5 in task_list.md
   - File: src/cubie/integrators/loops/ode_loop.py
   - Issue: `at_last_summarise` (lines 623-629) is computed regardless of `summarise_last` flag value, causing wasted computation.
   - Fix: Wrap `at_last_summarise` computation in conditional:
     ```python
     at_last_summarise = False
     if summarise_last:
         at_last_summarise = bool_(
             (next_update_summary > t_end) and (t_prec < t_end)
         )
         finished = selp(at_last_summarise, False, finished)
     ```
   - Rationale: Avoids unnecessary computation when `summarise_last=False`.
   - Status: ⏭️ SKIPPED - Code already initializes at_last_summarise to False (line 614) before the conditional block (line 623). The current implementation is correct.

6. **Update Type Hints for Optional Return Values**
   - Task Group: Reference to task group 4 in task_list.md
   - File: src/cubie/integrators/loops/ode_loop_config.py
   - Issue: Property return type hints say `float` but they now return `Optional[float]` after the None guard fix.
   - Fix: Update type hints from `float` to `Optional[float]` for `save_every`, `summarise_every`, `sample_summaries_every`, and `samples_per_summary` to `Optional[int]`.
   - Rationale: Type hints should match actual return values.
   - Status: ✅ APPLIED (included in EDIT 1 and EDIT 2) 
