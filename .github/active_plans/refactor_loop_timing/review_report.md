# Implementation Review Report
# Feature: Refactor Loop Timing Parameters
# Review Date: 2026-01-05
# Reviewer: Harsh Critic Agent

## Executive Summary

The Loop Timing Parameters refactoring has been implemented competently across the codebase. The primary objectives—renaming `dt_save` → `save_every`, `dt_summarise` → `summarise_every`, `dt_update_summaries` → `sample_summaries_every`, and adding `save_last`/`summarise_last` flags—have been achieved. The deprecated parameter names have been successfully removed from the core components: ODELoopConfig, IVPLoop, OutputConfig, OutputFunctions, Solver, and SolveSpec.

The None-handling logic in ODELoopConfig's `__attrs_post_init__` correctly sets the `save_last` and `summarise_last` flags when all timing parameters are None, and appropriately sets `summarise_last` when only `save_every` is specified. The inference logic for deriving missing values from specified values is implemented correctly for all cases.

However, the implementation has one **critical omission**: **User Story US-4 (Duration-Dependent Recompile Warning) is NOT implemented**. When `summarise_every` must be derived from `duration`, no warning is issued. The agent_plan.md specified this warning should be issued at config creation time or kernel compile time, but this functionality is missing entirely.

## User Story Validation

**User Stories** (from human_overview.md):

- **US-1 (Rename Timing Parameters)**: ✅ **Met** - All deprecated names (`dt_save`, `dt_summarise`, `dt_update_summaries`) have been removed and replaced with new names (`save_every`, `summarise_every`, `sample_summaries_every`) across ODELoopConfig, IVPLoop, OutputConfig, OutputFunctions, Solver, and SolveSpec.

- **US-2 (Optional Timing Parameters with Smart Defaults)**: ✅ **Met** - All three timing parameters default to `None`. When all are `None`, `save_last=True` and `summarise_last=True` are set. No validation errors raised for individual `None` values.

- **US-3 (Intelligent Summary Timing Inference)**: ⚠️ **Partial** - The inference logic is implemented correctly:
  - If `sample_summaries_every` is `None` and `summarise_every` is set: `sample_summaries_every` defaults to `summarise_every / 10` ✅
  - If `summarise_every` is `None` and `sample_summaries_every` is set: `summarise_every` defaults to `10.0 * save_every` ⚠️ (hardcoded multiplier, not duration-dependent as specified)
  - Both None case sets `summarise_last` flag ✅
  
  However, the warning specified in the acceptance criteria is missing.

- **US-4 (Duration-Dependent Recompile Warning)**: ❌ **Not Met** - No warning is issued anywhere when `summarise_every` must be derived from `duration`. The acceptance criteria specified:
  > - Warning message explains that changing `duration` will force kernel recompilation
  > - Warning suggests setting an explicit `summarise_every` to avoid this overhead
  
  This is completely missing from the implementation.

**Acceptance Criteria Assessment**: 3 of 4 user stories are fully or partially met. US-4 is not implemented at all, which means users will not be warned about the recompilation overhead when using duration-dependent summary timing.

## Goal Alignment

**Original Goals** (from human_overview.md):

1. **Rename parameters for clarity**: ✅ Achieved - Names changed throughout
2. **Make all timing parameters optional with None defaults**: ✅ Achieved 
3. **Add save_last and summarise_last modes**: ⚠️ Partial - Flags are added and set correctly, but the actual loop behavior needs verification (see Architecture Assessment)

**Assessment**: The core refactoring is complete, but the recompile warning system described in the architecture overview is missing.

## Code Quality Analysis

### Duplication

No significant code duplication detected. The None-handling logic in ODELoopConfig is appropriately centralized in `__attrs_post_init__`.

### Unnecessary Complexity

- **Location**: src/cubie/integrators/loops/ode_loop_config.py, `__attrs_post_init__`
- **Issue**: The elif chain for 7 cases is somewhat complex but necessary for the inference logic. Each case is clearly commented.
- **Impact**: Moderate complexity, but acceptable given the requirements.

### Convention Violations

- **PEP8**: Lines 311-314 in ode_loop_config.py are at 79 characters which is the limit, acceptable.
- **Type Hints**: Present and correct in all modified files.
- **Repository Patterns**: Follows attrs pattern correctly with underscore-prefixed private fields and property wrappers.

### Unnecessary Additions

None detected. All changes serve the stated goals.

## Performance Analysis

- **CUDA Efficiency**: The `save_last` and `summarise_last` flags are captured at compile time via closure in `build()`, so there's no runtime overhead from checking these flags.
- **Memory Patterns**: No new memory allocations added; timing flags are compile-time constants.
- **Buffer Reuse**: Not applicable to this feature.
- **Math vs Memory**: Not applicable to this feature.
- **Optimization Opportunities**: None identified.

## Architecture Assessment

### Integration Quality

The implementation integrates well with the existing CuBIE architecture:
- ODELoopConfig correctly stores timing parameters with underscore prefix and exposes them via precision-wrapping properties
- IVPLoop correctly reads `save_last` and `summarise_last` from config and passes them to the loop closure
- The existing `save_last` variable in the loop device function (line 595 in ode_loop.py) now receives its value from config

### Design Patterns

The implementation correctly follows:
- Attrs classes pattern with underscore prefix for floats
- CUDAFactory pattern with compile settings
- Buffer registry update pattern

### Flag Wiring Verification

**Critical observation**: In ode_loop.py `build()` method (lines 371-373):
```python
save_last = config.save_last
summarise_last = config.summarise_last
```

These flags are used in the loop device function. The `save_last` flag appears to be wired correctly to the predicated commit logic in the loop (lines 595-601). However, **`summarise_last` is captured but never actually used in the loop logic**. Reviewing lines 767-793, the loop unconditionally calls `update_summaries` and `save_summaries` at regular intervals controlled by `do_update_summary`. There is no conditional check on `summarise_last` to skip intermediate summaries.

This means while the flag is set correctly when timing parameters are None, **the loop behavior for `summarise_last=True` is identical to `summarise_last=False`**. The flag currently has no effect on actual summarization behavior.

### Future Maintainability

Good - the inference logic is well-documented with comments indicating each case.

## Suggested Edits

1. **Implement Duration-Dependent Recompile Warning**
   - Task Group: N/A (not in original task list)
   - File: src/cubie/integrators/loops/ode_loop_config.py OR src/cubie/batchsolving/BatchSolverKernel.py
   - Issue: US-4 acceptance criteria requires a warning when `summarise_every` is derived from `duration`
   - Fix: Add a flag `_summarise_from_duration: bool = field(default=False, init=False)` in ODELoopConfig. Set this flag in `__attrs_post_init__` when summarise_every is inferred from only `save_every` being set. In BatchSolverKernel.run() or similar, check this flag and issue `warnings.warn()` with message:
     ```
     "Summarising only at the end of the run forces the CUDA kernel to 
     recompile whenever duration changes. Set an explicit summarise_every 
     value to avoid this overhead."
     ```
   - Rationale: US-4 explicitly requires this warning; currently missing
   - Status: 

2. **Implement summarise_last Loop Behavior**
   - Task Group: Task Group 2 (IVPLoop)
   - File: src/cubie/integrators/loops/ode_loop.py, lines 767-793
   - Issue: The `summarise_last` flag is captured in the loop closure (line 373) but **never used**. The loop unconditionally calls `update_summaries` at regular intervals. The flag has no effect on behavior.
   - Fix: Add predicated logic similar to `save_last` (lines 595-601). When `summarise_last=True`, skip intermediate summary updates and only compute/save summary at `t >= t_end`. Example approach:
     ```python
     # At line 767, modify the if condition:
     if do_update_summary:
         # When summarise_last, only update at final step
         skip_intermediate = bool_(summarise_last and (t_prec + dt_raw) < t_end)
         do_update_summary = selp(skip_intermediate, False, do_update_summary)
         # ... rest of logic
     ```
   - Rationale: US-2 acceptance criteria states "summarise_last flag... computes summaries only at run end" but this is currently not implemented
   - Status: 

3. **Add Test for Warning Emission**
   - Task Group: Task Group 6 (Tests)
   - File: tests/integrators/loops/test_dt_update_summaries_validation.py
   - Issue: No test verifies the duration-dependent recompile warning is emitted
   - Fix: After implementing the warning, add a test:
     ```python
     def test_summarise_only_save_every_warns():
         """Test that specifying only save_every issues recompile warning."""
         with pytest.warns(UserWarning, match="recompile"):
             # Trigger the warning by setting only save_every
             ...
     ```
   - Rationale: Warning behavior should be verified by tests
   - Status: 

4. **Consider Adding sample_summaries_every to SolveSpec**
   - Task Group: Task Group 5 (SolveSpec)
   - File: src/cubie/batchsolving/solveresult.py
   - Issue: The agent_plan.md suggested adding `sample_summaries_every` field to SolveSpec if not present, but it's still not there
   - Fix: Add field:
     ```python
     sample_summaries_every: float = attrs.field(
         validator=getype_validator(float, 0.0)
     )
     ```
     And update Solver.solve_info property to pass it.
   - Rationale: Completeness - all three timing parameters should be in SolveSpec for full transparency
   - Status: 

## Summary

The implementation successfully achieves the primary refactoring goals (renaming parameters, removing backward compatibility, adding flags). However, there are **two critical gaps**:

1. **US-4 (recompile warning) is not implemented** - This was explicitly specified in human_overview.md as a user story with acceptance criteria.
2. **`summarise_last` flag has no effect on loop behavior** - The flag is correctly set in ODELoopConfig, but the IVPLoop device function never uses it. When `summarise_last=True`, summaries are still computed at every interval, not just at run end.

**Priority of fixes:**
1. **Critical**: Implement `summarise_last` loop behavior (US-2 not fully met - flag has no effect)
2. **Critical**: Implement duration-dependent recompile warning (US-4 not met)
3. **Medium**: Add `sample_summaries_every` to SolveSpec (completeness)
4. **Low**: Add test for warning emission (testing completeness)
