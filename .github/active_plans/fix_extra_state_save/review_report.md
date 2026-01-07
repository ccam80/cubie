# Implementation Review Report
# Feature: Fix Extra State Save Bug
# Review Date: 2026-01-07
# Reviewer: Harsh Critic Agent

## Executive Summary

The implementation attempted to fix duplicate saves when `save_last` and `save_regularly` coincide at the integration end, but **introduced a critical bug** and **has incorrect test expectations**.

**Critical Bug**: The `at_end` condition on line 606 of `ode_loop.py` uses `t_prec <= t_end`, which causes `at_end` to fire twice when `save_regularly=False` and `save_last=True`. After the final save at `t_end`, on the next iteration `t_prec == t_end` still satisfies `t_prec <= t_end`, so `at_end` fires again, attempting to write to an array index beyond the allocated size.

**Test Expectation Bug**: The test `test_summarise_last_no_duplicate_at_aligned_end` expects 6 summaries but the correct behavior is 5 summaries. Unlike state saves (which include an initial save at t=0), summaries do NOT save at t=0 - the initialization code only resets the buffer, it doesn't count as a summary output.

The fix for `save_last` with `save_regularly=True` (Task Group 1) appears correct, as `test_save_last_no_duplicate_at_aligned_end` passes. The fundamental issue is that `at_end` can fire multiple times when the loop doesn't terminate after the final save.

## User Story Validation

**User Stories** (from human_overview.md):

- **Story 1 (Consistent Save Behavior)**: **Not Met** - The fix for `save_regularly=False` cases is broken. Tests with `save_every=None` and `save_last=True` fail with IndexError because `at_end` fires twice.

- **Story 2 (Consistent Summary Behavior)**: **Partial** - The logic changes are correct, but the test expectation is wrong. The test expects 6 summaries when only 5 should occur (summaries don't include an initial t=0 sample like state saves do).

- **Story 3 (Test Isolation for Timing State)**: **Unable to Verify** - The timing reset logic in conftest.py looks correct, but tests cannot run due to the IndexError bug.

**Acceptance Criteria Assessment**: The implementation fails to meet the primary goal of ensuring the loop saves exactly the expected number of snapshots without array overflows.

## Goal Alignment

**Original Goals** (from human_overview.md):

- **Prevent double save at end**: **Failed** - While the deduplication logic for `save_regularly & save_last` is correct (Task Group 1), the core `at_end` condition allows duplicate saves when `save_regularly=False`.

- **Prevent combination issue with summaries**: **Partial** - Logic appears correct but test expectations are wrong.

- **Fix test order timing leak**: **Unable to Verify** - Cannot test due to IndexError.

**Assessment**: The root cause was misdiagnosed. The issue is not just in the deduplication logic, but in the fundamental `at_end` condition which can fire multiple times.

## Code Quality Analysis

#### Critical Bug
- **Location**: src/cubie/integrators/loops/ode_loop.py, line 606
- **Issue**: `at_end = bool_(t_prec <= t_end) & finished` uses `<=` instead of `<`
- **Impact**: When `t_prec == t_end` after the final save, `at_end` still evaluates to True on the next iteration, causing a duplicate save attempt that overflows the array.

#### Incorrect Test Expectation
- **Location**: tests/integrators/loops/test_ode_loop.py, lines 574-588
- **Issue**: Test expects 6 summaries but correct behavior is 5 summaries
- **Impact**: Test will fail even when code is correct

#### Unnecessary Code Complexity
- **Location**: src/cubie/integrators/loops/ode_loop.py, lines 631-641
- **Issue**: The deduplication logic for `save_last` and `summarise_last` is redundant if the `at_end` condition is fixed
- **Impact**: The logic adds complexity without being the actual fix. Once `at_end` fires only once, no deduplication is needed because the regular save and `at_end` save cannot both trigger in the same iteration.

### Convention Violations
- **PEP8**: No violations detected
- **Type Hints**: Consistent with existing code
- **Repository Patterns**: Follows existing patterns

## Performance Analysis

- **CUDA Efficiency**: No performance concerns with the proposed fix
- **Memory Patterns**: No changes to memory access patterns
- **Buffer Reuse**: Not applicable
- **Math vs Memory**: Not applicable
- **Optimization Opportunities**: None identified

## Architecture Assessment

- **Integration Quality**: Changes integrate well with existing loop structure
- **Design Patterns**: Follows existing predicated-commit patterns
- **Future Maintainability**: The fix is simpler and more maintainable than the current approach

## Suggested Edits

1. **Fix at_end condition to use strict less-than**
   - Task Group: Modification to Task Group 1
   - File: src/cubie/integrators/loops/ode_loop.py
   - Issue: Line 606 uses `t_prec <= t_end` which causes `at_end` to fire twice
   - Fix: Change `at_end = bool_(t_prec <= t_end) & finished` to `at_end = bool_(t_prec < t_end) & finished`
   - Rationale: Using strict less-than ensures that once `t_prec == t_end` (after the final save), `at_end` becomes False and the loop can terminate normally. This prevents the duplicate save that causes the IndexError.
   - Status: [x] Applied
   - Task Group: Modification to Task Group 2 test
   - File: tests/integrators/loops/test_ode_loop.py
   - Issue: Test `test_summarise_last_no_duplicate_at_aligned_end` expects 6 summaries but should expect 5
   - Fix: Change `expected_summaries = 6` to `expected_summaries = 5` and update the comment explaining that summaries don't include t=0 (unlike state saves)
   - Rationale: The initialization code calls `save_summaries` but doesn't increment `summary_idx` - it's just a buffer reset. Therefore, summaries are only saved during the main loop at t=0.02, 0.04, 0.06, 0.08, 0.10 (5 saves), not at t=0.
   - Status: [x] Applied

3. **Simplify or remove redundant deduplication logic (optional)**
   - Task Group: Cleanup after fixes 1-2
   - File: src/cubie/integrators/loops/ode_loop.py
   - Issue: Lines 631-641 add deduplication logic that becomes unnecessary once the `at_end` condition is fixed
   - Fix: Consider removing the deduplication logic since `at_end` can only fire once, making it impossible for both regular save and `at_end` save to trigger in the same step
   - Rationale: Once `at_end` fires only on the final step before reaching `t_end`, and regular saves happen based on timing, these are mutually exclusive conditions. However, keeping the deduplication provides defense-in-depth and explicitly documents the behavior, so this is optional.
   - Status: 

## Root Cause Summary

The original bug description mentioned "Float64/precision rounding and misalignment" causing "an extra state save...before triggering". The actual root cause is simpler:

1. The `at_end` condition `t_prec <= t_end` uses `<=` (less-than-or-equal)
2. After saving at `t_end`, the time becomes exactly `t_end`
3. On the next iteration, `t_prec == t_end` still satisfies `t_prec <= t_end`
4. So `at_end` fires again, causing a second save attempt
5. This overflows the pre-allocated array, causing the IndexError

The fix is simply to use strict less-than: `t_prec < t_end`. This ensures that once we reach `t_end`, `at_end` becomes False and the loop terminates.
