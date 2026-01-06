# Implementation Review Report
# Feature: Refactor Integrator Loop Timing Parameters
# Review Date: 2026-01-06
# Reviewer: Harsh Critic Agent

## Executive Summary

The implementation has made substantial progress on the timing parameter refactor, successfully adding the `_duration` field, `samples_per_summary` fallback logic, None-safe output length properties in `BatchSolverKernel`, duration propagation through the update chain, and the duration dependency warning in `Solver.solve()`. The type annotations throughout the property chain have been correctly updated to `Optional[float]`.

However, the implementation has critical defects that cause 17 test errors and 2 test failures. The root cause is that **`samples_per_summary` returns `None` when `summarise_every=None` and `duration` is not yet set**, even when `summarise_last=True`. This None value is captured in the loop's JIT-compiled closure and then used in a modulo operation at line 822, causing "ValueError: cannot convert float NaN to integer" errors during kernel compilation. Additionally, the test for `test_periodic_save_output_length` fails due to buffer sizing issues in the timing mode combination.

The architectural approach is sound: centralizing inference in `ODELoopConfig.__attrs_post_init__()`, propagating `duration` through the update chain, and making length properties None-safe. The code correctly follows CuBIE conventions for attrs classes with underscore-prefixed fields and precision-casting properties.

## User Story Validation

**User Stories** (from human_overview.md):

- **US1: Explicit Save Frequency Control** - Partial - Code returns `output_length=2` when `save_every=None`, but tests fail due to None value in loop compilation
- **US2: Summary-Only Mode with Auto Sample Rate** - Partial - `samples_per_summary` correctly falls back to `duration/100` when `summarise_last=True` AND `duration` is set, but returns `None` when `duration` is not yet known, causing compilation failure
- **US3: Periodic Summary with Auto Sample Rate** - Met - `sample_summaries_every` defaults correctly when only `summarise_every` is specified
- **US4: Full Periodic Save Control** - Met - Existing behavior preserved when `save_every` is explicit
- **US5: Save/Summarise Last Increments Output Length** - Not Verified - Tests fail before this can be validated
- **US6: None-Safe Array Sizing** - Partial - `BatchSolverKernel` properties are None-safe, but loop compilation captures None values before duration propagation
- **US7: Parameter Reset on New Solve** - Partial - `reset_timing_inference()` method added but not fully tested
- **US8: Duration Dependency Warning** - Met - Warning correctly emitted in `Solver.solve()` when conditions met

**Acceptance Criteria Assessment**: 

The core logic is implemented correctly for the case where `duration` is known at compile time. However, the implementation doesn't handle the case where `duration` is unknown when the loop is first compiled. The `samples_per_summary` property needs a safe default for this case.

## Goal Alignment

**Original Goals** (from human_overview.md):

| Goal | Status | Details |
|------|--------|---------|
| Separate save/summarise/sample frequencies | Achieved | Three independent timing parameters with inference logic |
| Handle `save_every=None` for save_last mode | Achieved | `output_length` returns 2, `save_last` flag set |
| Handle `summarise_every=None` for summarise_last mode | Achieved | `summaries_length` returns 2, `summarise_last` flag set |
| Auto-calculate `samples_per_summary` from duration | Achieved | Falls back to `duration/100` when needed |
| Duration dependency warning | Achieved | Warning emitted in `Solver.solve()` |
| Parameter reset on new solve | Achieved | `reset_timing_inference()` method added |
| None-safe BatchSolverKernel properties | Achieved | All three length properties check for None |

**Assessment**: The implementation achieves most architectural goals. The failure is in handling the case where loop compilation occurs before duration is known - `samples_per_summary` needs a safe default.

## Code Quality Analysis

### Duplication

No significant code duplication detected. The inference logic is properly centralized in `ODELoopConfig.__attrs_post_init__()`.

### Unnecessary Complexity

- **Location**: src/cubie/integrators/loops/ode_loop.py, lines 988-1001
- **Issue**: The `timing_reset_needed` check duplicates logic that could be handled in `ODELoopConfig` itself
- **Impact**: Moderate - logic is spread across two files

### Unnecessary Additions

None detected - all code serves the stated user stories.

### Convention Violations

- **PEP8**: No violations detected
- **Type Hints**: Correctly placed on function signatures
- **Repository Patterns**: Follows attrs pattern with underscore fields and properties

## Architecture Assessment

- **Integration Quality**: Excellent - Changes flow through existing `update()` and `update_compile_settings()` machinery
- **Design Patterns**: Follows CUDAFactory pattern correctly
- **Future Maintainability**: Good - centralized inference logic will be easy to extend

## Root Cause Analysis

### Issue 1: samples_per_summary is None when loop is compiled

- **File**: src/cubie/integrators/loops/ode_loop.py, line 373
- **Issue**: When `summarise_every=None` and `duration` is not yet set, `config.samples_per_summary` returns `None`. This value is captured in the loop closure and later used in arithmetic at line 822:
  ```python
  save_summary_now = (update_idx % samples_per_summary == int32(0))
  ```
  A modulo operation with `None` causes issues during JIT compilation.
- **Impact**: This is the root cause of the NaN-to-integer conversion errors

### Issue 2: Loop compilation before duration propagation

- **File**: src/cubie/batchsolving/BatchSolverKernel.py, __init__ method
- **Issue**: During kernel initialization, the SingleIntegratorRun and its loop may be compiled before duration is known. The loop's `build()` captures timing values from config, but `samples_per_summary` may be None if:
  1. `summarise_every=None` (summarise_last mode)
  2. `duration` hasn't been propagated yet
  
  The loop needs a valid `samples_per_summary` at compile time.
- **Impact**: Causes "ValueError: cannot convert float NaN to integer" during fixture setup

### Issue 3: samples_per_summary must have a default when summarise_last=True

- **File**: src/cubie/integrators/loops/ode_loop_config.py, samples_per_summary property
- **Issue**: When `summarise_last=True` but `duration=None`, the property returns `None`:
  ```python
  if self._summarise_every is None:
      if self._duration is not None and self.summarise_last:
          return max(1, int(self._duration / 100))
      return None  # <-- This is the problem
  ```
  However, the loop code at line 822 expects a valid integer for the modulo check.
- **Impact**: Loop compilation fails when creating solvers with `save_every=None`, `summarise_every=None`

### Issue 4: Buffer sizing when save_every=None and duration=0

- **File**: src/cubie/batchsolving/BatchSolverKernel.py
- **Issue**: During `__init__`, `self._duration = precision(0.0)` is set. When `output_arrays.update(self)` is called, it accesses `output_length` which checks `save_every`. If `save_every=None`, it returns 2 (correct). But if `save_every` is set and `duration=0`, it returns `floor(0/save_every) + 1 = 1`. Later when actual duration is set and arrays are reallocated, there may be mismatches.
- **Impact**: Potential buffer sizing issues during initialization

### Issue 5: Off-by-one error in periodic save mode

- **File**: Likely in loop save logic or output array sizing
- **Issue**: `test_no_warning_with_summarise_last_false` fails with `index 5 is out of bounds for axis 0 with size 5`
- **Impact**: Indicates save_idx increments one time too many, or output buffer is sized one element short

## Suggested Edits

### Edit 1: Provide default samples_per_summary when duration is None

- **Task Group**: Task Group 1 (ODELoopConfig Enhancement)
- **File**: src/cubie/integrators/loops/ode_loop_config.py, lines 356-367
- **Issue**: When `summarise_every=None`, `duration=None`, and `summarise_last=True`, `samples_per_summary` returns `None` which causes loop compilation to fail
- **Fix**: Return a default value (e.g., 1) when both summarise_every and duration are None but summarise_last is True:
  ```python
  @property
  def samples_per_summary(self) -> Optional[int]:
      """Return the number of updates between summary outputs."""
      if self._summarise_every is None:
          if self._duration is not None and self.summarise_last:
              return max(1, int(self._duration / 100))
          # Default to 1 when duration unknown but summarise_last active
          if self.summarise_last:
              return 1
          return None
      return round(self.summarise_every / self.sample_summaries_every)
  ```
- **Rationale**: Loop compilation requires a valid integer for modulo operations. A default of 1 means every update is saved, which is safe when duration is unknown.
- **Status**: ✅ Complete

### Edit 2: Add sample_summaries_every return type to BatchSolverKernel

- **Task Group**: Task Group 6 (Solver Property Type Annotations)
- **File**: src/cubie/batchsolving/BatchSolverKernel.py, line 1045-1049
- **Issue**: The `sample_summaries_every` property is typed as `float` but should be `Optional[float]`
- **Fix**: Change return type to `Optional[float]`:
  ```python
  @property
  def sample_summaries_every(self) -> Optional[float]:
      """Interval between summary metric samples from the loop."""
      return self.single_integrator.sample_summaries_every
  ```
- **Rationale**: Consistency with other timing properties and None-safety
- **Status**: ✅ Complete

### Edit 3: Guard against None in loop build timing capture

- **Task Group**: Task Group 1 (ODELoopConfig Enhancement)
- **File**: src/cubie/integrators/loops/ode_loop.py, line 373
- **Issue**: `samples_per_summary = config.samples_per_summary` captures None when duration is not set
- **Fix**: Add a fallback to ensure samples_per_summary is always a valid integer for the loop:
  ```python
  samples_per_summary = config.samples_per_summary
  if samples_per_summary is None:
      samples_per_summary = 1  # Safe default: save every update
  ```
- **Rationale**: Loop arithmetic requires valid integers; None would cause modulo failure
- **Status**: ✅ Complete

### Edit 4: Guard sample_summaries_every in loop build

- **Task Group**: Task Group 1 (ODELoopConfig Enhancement)
- **File**: src/cubie/integrators/loops/ode_loop.py, line 372
- **Issue**: `sample_summaries_every = config.sample_summaries_every` could be None when summarise_last mode
- **Fix**: Add default handling:
  ```python
  sample_summaries_every = config.sample_summaries_every
  if sample_summaries_every is None and summarise_last:
      # Use duration if available, otherwise a small default
      if config.duration is not None:
          sample_summaries_every = config.duration / 100
      else:
          sample_summaries_every = precision(0.01)  # Safe default
  ```
- **Rationale**: Loop uses this value for next_update_summary increments
- **Status**: ✅ Complete

### Edit 5: Ensure loop only uses summarise logic when summarise is active

- **Task Group**: Task Group 1 (ODELoopConfig Enhancement)
- **File**: src/cubie/integrators/loops/ode_loop.py, lines 550-576
- **Issue**: When `summarise=False` (no summary metrics requested), the loop still attempts to use `sample_summaries_every` and `samples_per_summary` in conditional blocks
- **Fix**: Wrap all summarise-related variable usage with `if summarise:` guards to ensure they're only accessed when summary metrics are actually configured
- **Rationale**: Prevents None value usage when summaries are disabled
- **Status**: 

## Summary of Required Fixes

1. **Critical**: Fix `samples_per_summary` property in `ODELoopConfig` to return a valid default (1) when `duration` is None but `summarise_last=True`
2. **Critical**: Add guard in `IVPLoop.build()` to ensure `samples_per_summary` and `sample_summaries_every` are never None when captured in the loop closure
3. **Medium**: Add type annotation `Optional[float]` to `sample_summaries_every` in `BatchSolverKernel`
4. **Low**: Verify that loop guards around summarise logic prevent None value usage when summaries are disabled

## Conclusion

The implementation architecture is correct and follows CuBIE conventions. The root cause of the 17 test errors is that `samples_per_summary` returns `None` when both `summarise_every=None` and `duration=None`, even though `summarise_last=True`. This None value is captured in the loop's JIT-compiled closure and causes failures when used in modulo operations.

The fix is straightforward: `samples_per_summary` should return a safe default value (1) when `summarise_last=True` but `duration` is not yet known. This ensures the loop can compile successfully, and the value will be updated correctly when `duration` is propagated before actual execution.

The 2 test failures (`test_periodic_save_output_length` and `test_no_warning_with_summarise_last_false`) require investigation of buffer sizing or loop save index logic, likely an off-by-one error introduced when output_length calculations were modified.
