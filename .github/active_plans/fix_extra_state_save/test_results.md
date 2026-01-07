# Test Results Summary

## Overview
- **Tests Run**: 7
- **Passed**: 7
- **Failed**: 0
- **Errors**: 0
- **Skipped**: 0

## New Tests Created for This Fix

| Test | Status |
|------|--------|
| `test_save_last_no_duplicate_at_aligned_end` | ✅ Passed |
| `test_summarise_last_no_duplicate_at_aligned_end` | ✅ Passed |
| `test_timing_state_isolation` | ✅ Passed |
| `test_save_last_only_produces_two_saves` | ✅ Passed |
| `test_save_and_summarise_last_no_duplicates` | ✅ Passed |

## Originally Failing Tests

| Test | Status |
|------|--------|
| `test_solve_ivp_with_summarise_variables` | ✅ Passed |
| `test_summarise_last_collects_final_summary` | ✅ Passed |

## Fixes Applied

### Fix 1: at_end condition (from reviewer)
- **Location**: src/cubie/integrators/loops/ode_loop.py, line 606
- **Change**: `at_end = bool_(t_prec <= t_end)` → `at_end = bool_(t_prec < t_end)`
- **Rationale**: Prevents `at_end` from firing twice when `t_prec == t_end`

### Fix 2: test expectation (from reviewer)
- **Location**: tests/integrators/loops/test_ode_loop.py
- **Change**: `expected_summaries = 6` → `expected_summaries = 5`
- **Rationale**: Summaries don't include t=0 (unlike state saves)

### Fix 3: dt_eff calculation (discovered during final testing)
- **Location**: src/cubie/integrators/loops/ode_loop.py, lines 649-653
- **Change**: 
  - `if do_save:` → `if do_save and save_regularly:`
  - `if do_update_summary:` → `if do_update_summary and summarise_regularly:`
- **Rationale**: When `save_regularly=False`, `next_save` is not being updated and remains at 0, causing incorrect `dt_eff` calculation

## Command Executed
```bash
NUMBA_ENABLE_CUDASIM=1 pytest -m "not nocudasim and not specific_algos" -v --tb=short [test_paths]
```

## Recommendations
All fixes have been verified. The implementation is ready for final review.
