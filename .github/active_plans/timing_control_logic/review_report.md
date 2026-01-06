# Implementation Review Report
# Feature: Timing Control Logic
# Review Date: 2026-01-06
# Reviewer: Harsh Critic Agent

## Executive Summary

The timing control logic implementation has correctly identified the architectural approach: timing flags (`save_last`, `summarise_last`) are detected in `SingleIntegratorRunCore.__init__` based on output_types, and `chunk_duration` is intercepted in `SingleIntegratorRunCore.update()` to compute `sample_summaries_every` when needed. The warning emission logic is properly implemented to fire only once.

However, the implementation has a critical defect: it fails to handle the case where the CUDA device function is built **before** `chunk_duration` is provided via `update()`. When `summarise_last=True` and `sample_summaries_every` remains `None`, the `samples_per_summary` property returns `None`, which causes `ValueError: cannot convert float NaN to integer` when the Numba CUDA compiler tries to use this value in modulo and division operations within the loop device function.

The root cause is a timing mismatch: fixture setup creates `SingleIntegratorRun` instances and may trigger `build()` before `BatchSolverKernel.run()` calls `update()` with `chunk_duration`. The `ODELoopConfig.__attrs_post_init__` sets `summarise_last=True` when all timing is None, but does NOT provide a sentinel value for `samples_per_summary`, leaving the loop unbuildable.

## User Story Validation

**User Stories** (from human_overview.md):

- **US-1: Automatic Timing Flag Detection**: Partial - Flags are set correctly in `SingleIntegratorRunCore.__init__`, but the loop cannot be built until `chunk_duration` is provided
- **US-2: Automatic Sample Interval Inference**: Partial - Inference works correctly in `update()` when `chunk_duration` is provided, but fails before that
- **US-3: Duration-Based Warning**: Met - Warning is correctly emitted once when `sample_summaries_every` is computed from `chunk_duration`
- **US-4: Chunk Duration Interception**: Met - `chunk_duration` is correctly popped from updates_dict and used to compute `sample_summaries_every`

**Acceptance Criteria Assessment**: The implementation correctly identifies when timing flags should be set and correctly computes `sample_summaries_every` from `chunk_duration`. However, the acceptance criterion "When `summarise_every == None` and `sample_summaries_every == None` but summary metrics in output_types → `summarise_last=True`, `samples_per_summary` defaults to `duration/100`, `output_summaries_length=2`" is NOT met because the duration-based default only happens in `update()`, not during initialization.

## Goal Alignment

**Original Goals** (from human_overview.md):

- **Timing flag initialization in `__init__`**: Achieved - Flags are set based on output_types
- **Timing consolidation in `update`**: Achieved - `chunk_duration` intercepted and `sample_summaries_every` computed
- **Warning emission**: Achieved - Warning fires once per integrator instance
- **BatchSolverKernel updates**: Achieved - `chunk_duration` passed correctly

**Assessment**: All four goals are implemented, but the integration between them has a gap. The implementation assumes `build()` won't be called until after `update()` provides `chunk_duration`, but test fixtures and direct usage patterns can trigger `build()` earlier.

## Code Quality Analysis

### Duplication
- **Location**: No significant duplication found

### Unnecessary Complexity
- **Location**: `ODELoopConfig.__attrs_post_init__` (lines 255-314)
- **Issue**: The method handles many cases but doesn't establish a sentinel value for `samples_per_summary` when all timing is None
- **Impact**: Partial completion leaves the loop in an unbuildable state

### Unnecessary Additions
- None identified

### Convention Violations
- **PEP8**: No violations found
- **Type Hints**: Properly placed
- **Repository Patterns**: The implementation follows the CUDAFactory pattern correctly

## Performance Analysis

- **CUDA Efficiency**: Not evaluated (code doesn't build successfully)
- **Memory Patterns**: Not evaluated
- **Buffer Reuse**: Not applicable to this feature
- **Math vs Memory**: Not applicable to this feature

## Architecture Assessment

- **Integration Quality**: The implementation correctly integrates with existing components (SingleIntegratorRun, BatchSolverKernel, IVPLoop, ODELoopConfig). The issue is a timing/lifecycle problem, not an architectural one.
- **Design Patterns**: Correctly uses CUDAFactory pattern with update() for deferred configuration
- **Future Maintainability**: The separation of concerns is good; the fix is localized to the ODELoopConfig or IVPLoop build() method

## Root Cause Analysis

**The NaN Error Chain:**

1. Test fixture creates `SingleIntegratorRun` with `save_every=None`, `summarise_every=None`, `sample_summaries_every=None`, `output_types=["mean"]`
2. `SingleIntegratorRunCore.__init__` detects summary outputs and sets `loop_settings["summarise_last"] = True`
3. `IVPLoop.__init__` creates `ODELoopConfig` with all timing parameters as None
4. `ODELoopConfig.__attrs_post_init__` (Case 1) sets `save_last=True`, `summarise_last=True` but leaves `_sample_summaries_every=None`
5. Test accesses `single_integrator_run.device_function` or builds the loop
6. `IVPLoop.build()` reads `config.samples_per_summary` which returns `None` (since `_summarise_every=None` and `_sample_summaries_every=None`)
7. In the CUDA device function:
   - Line 558: `save_summaries(..., samples_per_summary)` - passes None
   - Line 805: `update_idx % samples_per_summary` - modulo with None
8. Numba CUDA compiler fails with "cannot convert float NaN to integer"

**Why Tests Fail:**
- The new tests (`TestTimingFlagAutoDetection`, `TestSummariseFlagAutoDetection`, `TestChunkDurationInterception`) create fixtures with all timing as None
- The existing tests (`test_save_last_flag_from_config`, etc.) also create fixtures with timing as None to test the flags
- All these tests trigger `build()` before `chunk_duration` is provided

## Suggested Edits

1. **Provide sentinel value for samples_per_summary when summarise_last is True**
   - Task Group: New (post-review fix)
   - File: src/cubie/integrators/loops/ode_loop_config.py
   - Issue: `samples_per_summary` property returns `None` when `summarise_last=True` but timing parameters are None
   - Fix: Modify the `samples_per_summary` property to return a large sentinel value (e.g., `2**30`) when `summarise_last=True` and both `_summarise_every` and `_sample_summaries_every` are None. This ensures the modulo check never fires during normal operation - only the `(summarise_last and at_end)` condition triggers the save.
   - Rationale: In `summarise_last` mode, the modulo check `update_idx % samples_per_summary == 0` should never trigger because we only save at the very end via the `at_end` condition. A large sentinel value ensures this behavior.
   - Status: [x] COMPLETE - Modified `samples_per_summary` property to return 2**30 when `summarise_last=True` 

2. **Alternatively: Guard sample_summaries_every usage in IVPLoop.build()**
   - Task Group: New (post-review fix)
   - File: src/cubie/integrators/loops/ode_loop.py
   - Issue: `sample_summaries_every` and `samples_per_summary` captured as None and used in CUDA device function
   - Fix: In `build()`, after line 373, add:
     ```python
     # Provide sentinel values for summarise_last mode
     if summarise_last and samples_per_summary is None:
         samples_per_summary = 2**30  # Large value so modulo never triggers
     if summarise_last and sample_summaries_every is None:
         sample_summaries_every = precision(1e30)  # Never triggers next_update_summary advance
     ```
   - Rationale: This guards the loop from undefined behavior when built before `chunk_duration` is provided. The actual timing values will be set correctly when `update()` is called with `chunk_duration`.
   - Status: 

3. **Consider: Move timing inference to ODELoopConfig.__attrs_post_init__**
   - Task Group: New (architectural consideration)
   - File: src/cubie/integrators/loops/ode_loop_config.py
   - Issue: The `__attrs_post_init__` method sets flags but doesn't provide usable defaults for device function compilation
   - Fix: When Case 1 applies (all timing None), also set `_sample_summaries_every` and `_summarise_every` to sentinel values that work with the loop logic. For example:
     ```python
     # Case 1: All timing None - set flags for end-of-run-only behavior
     if (
         self._save_every is None
         and self._summarise_every is None
         and self._sample_summaries_every is None
     ):
         self.save_last = True
         self.summarise_last = True
         # Set sentinel values for device function compilation
         self._sample_summaries_every = 1e30  # Never triggers periodic update
         self._summarise_every = 1e30  # Never triggers periodic save
         return
     ```
   - Rationale: This makes the ODELoopConfig self-consistent and buildable immediately after creation, without requiring a subsequent `update()` call.
   - Status: 

**Recommended Approach:**
Edit 1 (modifying `samples_per_summary` property) is the safest and most targeted fix. It doesn't change the data model, just the computed property. The property already handles the "return None" case, so changing it to return a sentinel instead is minimal and localized.

```python
# In ODELoopConfig, modify the samples_per_summary property:
@property
def samples_per_summary(self) -> Optional[int]:
    """Return the number of updates between summary outputs."""
    if self._summarise_every is None or self._sample_summaries_every is None:
        # In summarise_last mode, return large sentinel so modulo never triggers
        if self.summarise_last:
            return 2**30
        return None
    return round(self.summarise_every / self.sample_summaries_every)
```

This fix ensures that:
1. Normal timing mode continues to work exactly as before
2. `summarise_last` mode can be built immediately without `chunk_duration`
3. The modulo check `update_idx % samples_per_summary == 0` will never fire (2^30 is larger than any reasonable update count)
4. The actual save happens via `(summarise_last and at_end)` condition

## Reviewer Validation Checklist

1. [x] Timing flags (save_last, summarise_last) set correctly based on output_types - **YES, implemented correctly**
2. [x] sample_summaries_every computed from chunk_duration when appropriate - **YES, but only in update()**
3. [x] Warning emitted once when duration-dependent computation occurs - **YES, implemented correctly**
4. [x] chunk_duration intercepted and not passed to loop - **YES, implemented correctly**
5. [x] No duration storage below BatchSolverKernel level - **YES, correct**
6. [x] All tests use fixtures per project convention - **YES, correct**
7. [ ] Loop can be built before chunk_duration is provided - **NO, this is the bug**
