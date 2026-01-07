# Agent Plan: Fix Extra State Save Bug

## Overview

This plan addresses a bug causing extra state saves under specific conditions in the IVPLoop device function. The issue manifests in three related ways:

1. Extra state save when `save_last=True` (default when no timing parameters specified)
2. Potential duplicate summaries when `summarise_regularly` and `summarise_last` are both true
3. Test order sensitivity due to `_user_timing` dictionary persistence

## Component Descriptions

### IVPLoop Device Function (`src/cubie/integrators/loops/ode_loop.py`)

The CUDA device function that executes the main integration loop. Key behavioral elements:

- **Initial save block** (lines 531-561): Saves initial state at t=0 when `settling_time == 0`
- **Termination logic** (lines 586-612): Determines when loop should exit
- **Event detection** (lines 614-648): Determines whether to save/summarize this step
- **Save execution** (lines 772-794): Performs state saves and counter resets
- **Summary execution** (lines 795-826): Performs summary updates and saves

### Expected Termination Behavior

The loop should terminate when:
- All outputs have been collected (no more saves/summaries needed)
- OR integration has reached t_end (for `save_last`/`summarise_last` modes)

The `at_end` flag should:
- Fire exactly once on the final step before termination
- Prevent premature termination to allow final save/summary
- NOT cause duplicate saves when regular saves already occurred at that time

### SingleIntegratorRunCore (`src/cubie/integrators/SingleIntegratorRunCore.py`)

Coordinates the integration loop and manages timing parameters. Key elements:

- **`_user_timing` dictionary** (lines 180-184): Tracks explicitly set timing parameters
- **`_process_loop_timing` method** (lines 197-270): Derives timing behavior from settings

### Test Fixtures (`tests/conftest.py`)

Session-scoped fixtures that create `SingleIntegratorRun` instances:
- **`single_integrator_run`** (lines 802-832): Creates session-scoped instance
- **`device_loop_outputs`** (lines 1036-1051): Executes device loop

## Architectural Changes Required

### Change 1: Prevent Duplicate Final Saves

**Current behavior**: When `save_last=True` and `save_regularly=True`, the final save may occur twice if `t_end` coincides with a regular save point.

**Expected behavior**: The final save should occur exactly once, whether triggered by `save_last` or regular interval.

**Integration points**:
- The `do_save` flag computation (lines 616-632)
- The save execution block (lines 772-794)

**Approach**: When `save_regularly=True` and `save_last=True`, detect when `at_end` coincides with a regular save point and suppress the duplicate.

### Change 2: Prevent Duplicate Final Summaries

**Current behavior**: When `summarise_regularly=True` and `summarise_last=True`, similar duplicate issue may occur.

**Expected behavior**: Summary should save exactly once at the final point.

**Integration points**:
- The `do_update_summary` flag computation (lines 623-635)
- The summary save logic (lines 811-826)

**Approach**: Similar deduplication logic for summaries.

### Change 3: Test Timing Isolation

**Current behavior**: The `_user_timing` dictionary persists within session-scoped fixtures, causing timing state leakage.

**Expected behavior**: Tests using default timing should not be affected by previous tests with explicit timing.

**Integration points**:
- The `single_integrator_run` fixture
- The `_process_loop_timing` method

**Approach**: Reset `_user_timing` to `{None, None, None}` before tests that don't explicitly specify timing parameters.

## Expected Interactions Between Components

### Loop Execution Flow

1. `SingleIntegratorRunCore` creates `IVPLoop` with derived timing flags
2. `_process_loop_timing` sets `save_last`, `summarise_last`, `save_regularly`, `summarise_regularly`
3. Loop `build()` captures these as compile-time constants
4. During execution:
   - Initial save occurs at t=0 (if `settling_time == 0`)
   - Main loop iterates, checking termination conditions
   - `at_end` fires on final step before termination
   - Save/summary triggered by appropriate flags
   - Loop returns when `finished` is true for all threads

### Timing Parameter Flow

```
User provides: save_every=None, summarise_every=None
    ↓
_process_loop_timing:
    - save_last = True (time-domain outputs requested, no save_every)
    - save_regularly = False
    - summarise_last = True (if summary outputs requested, no summarise_every)
    - summarise_regularly = False
    ↓
IVPLoop.build():
    - Captures flags as compile-time constants
    - Compiles loop with appropriate logic branches
```

## Data Structures and Their Purposes

### Loop Control Variables (device-side)

| Variable | Type | Purpose |
|----------|------|---------|
| `save_idx` | int32 | Current save position in output array |
| `summary_idx` | int32 | Current summary position in output array |
| `next_save` | precision | Time of next scheduled regular save |
| `next_update_summary` | precision | Time of next scheduled summary update |
| `at_end` | bool | True on final step before termination |
| `save_finished` | bool | True when all regular saves complete |
| `summary_finished` | bool | True when all regular summaries complete |
| `do_save` | bool | Whether to save state this step |
| `do_update_summary` | bool | Whether to update summary this step |

### Compile-Time Constants

| Constant | Type | Purpose |
|----------|------|---------|
| `save_last` | bool | Save final state at t_end |
| `summarise_last` | bool | Save final summary at t_end |
| `save_regularly` | bool | Save at regular intervals |
| `summarise_regularly` | bool | Update summaries at regular intervals |
| `save_every` | precision | Regular save interval |
| `sample_summaries_every` | precision | Summary update interval |

## Dependencies and Imports Required

No new dependencies required. All changes use existing:
- `numba.cuda` primitives
- `cubie.cuda_simsafe` helpers (`bool_`, `selp`, `all_sync`)
- NumPy types

## Edge Cases to Consider

### Edge Case 1: t_end Exactly Equals Regular Save Point
- When `save_every` divides `duration` evenly
- Both `save_regularly` and `save_last` would trigger
- Must save exactly once

### Edge Case 2: Floating-Point Precision at Boundaries
- `t_prec + dt_raw` may slightly exceed or fall short of `t_end` due to precision
- Comparisons like `t_prec <= t_end` may behave unexpectedly
- Consider using tolerance-based comparisons

### Edge Case 3: Zero Duration
- When `duration = 0`, only initial state should be saved
- `at_end` should not trigger additional saves

### Edge Case 4: Warmup with save_last
- When `settling_time > 0`, initial save at t=0 is skipped
- Final save should still occur at t_end

### Edge Case 5: Adaptive Stepping
- Step size may vary, affecting when `t_prec` reaches `t_end`
- `at_end` detection must work regardless of step size

### Edge Case 6: Test Parameter Combinations
- Session fixtures with different parameter overrides
- Ensure timing state doesn't leak between test classes/modules

## Implementation Notes for Detailed Implementer

### Loop Logic Fix

The core issue is likely in the interaction between:
1. The `finished` flag computation
2. The `at_end` flag computation
3. The `do_save` flag computation

The fix should ensure that when `save_regularly` has already saved at `t_end`, `save_last`'s `at_end` contribution to `do_save` is suppressed.

Consider:
```python
# Prevent save_last from triggering when regular save already occurred
if save_last:
    # Only add at_end contribution if not already covered by regular save
    do_save |= at_end & ~(save_regularly & (next_save > t_end))
```

### Test Fixture Fix

The simplest approach is to ensure `_user_timing` is reset appropriately. Options:
1. Reset in `_process_loop_timing` when no explicit timing provided
2. Add a fixture that resets timing before each test session parameter set
3. Make `single_integrator_run` check for timing in overrides and reset if not present

### Validation Strategy

After implementation:
1. Run failing tests individually to verify fix
2. Run full test suite to check for regressions
3. Add explicit test for the edge case of coincident final save
