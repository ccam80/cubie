# Test Results Summary (Post-Review Fix Run)

## Overview
- **Tests Run**: 103 (82 passed + 7 failed + 14 errors)
- **Passed**: 82
- **Failed**: 7
- **Errors**: 14
- **Skipped**: 0

## Import Error Fixed

Fixed `NameError: name 'BaseODE' is not defined` in `SingleIntegratorRun.py` line 132 by quoting the type hint (forward reference).

## Command Executed
```bash
NUMBA_ENABLE_CUDASIM=1 pytest -m "not nocudasim and not specific_algos" -v --tb=short tests/integrators/test_SingleIntegratorRun.py tests/integrators/loops/test_ode_loop.py tests/batchsolving/test_SolverKernel.py tests/batchsolving/test_solver.py
```

## Errors (14 total)

### NaN-Related Errors (9 errors)
**Type**: ValueError
**Message**: cannot convert float NaN to integer

| Test File | Test Name |
|-----------|-----------|
| test_SingleIntegratorRun.py | TestChunkDurationSkipped::test_chunk_duration_skipped_when_not_duration_dependent |
| test_SingleIntegratorRun.py | TestIsDurationDependentFalseExplicitTiming::test_is_duration_dependent_false_when_explicit_timing |
| test_SingleIntegratorRun.py | TestIsDurationDependentFalseNotSummariseLast::test_is_duration_dependent_false_when_not_summarise_last |
| test_SingleIntegratorRun.py | TestChunkDurationInterception::test_sample_summaries_every_computed_from_chunk_duration |
| test_ode_loop.py | test_samples_per_summary_returns_none_in_summarise_last_mode |
| test_ode_loop.py | test_save_last_flag_from_config |
| test_ode_loop.py | test_summarise_last_flag_from_config |
| test_ode_loop.py | test_summarise_last_collects_final_summary |

### Index Out of Bounds Errors (2 errors)

| Test File | Test Name | Message |
|-----------|-----------|---------|
| test_ode_loop.py | test_summarise_last_with_summarise_every_combined | index 3 is out of bounds for axis 0 with size 3 |
| test_SolverKernel.py | test_all_lower_plumbing | index 11 is out of bounds for axis 0 with size 11 |

## Failures (7 total)

### 1. test_all_summary_metrics_numerical_check[combined metrics]
**Type**: AssertionError
**Message**: Sentinel values (-4e32, 4e32) appearing in min/max outputs instead of expected values

### 2. test_all_summary_metrics_numerical_check[1st generation metrics]
**Type**: AssertionError
**Message**: Mismatch in sample count (expected 3, got 2)

### 3. test_all_summary_metrics_numerical_check[no combos]
**Type**: AssertionError
**Message**: Sentinel values (-4e32) in min outputs where 0 expected

### 4. test_adaptive_controller_with_float32
**Type**: AssertionError
**Message**: assert 0.0 == 1.0000799894332886 ± 1.0e-06 (integration returning 0.0)

### 5. test_save_at_settling_time_boundary
**Type**: AssertionError
**Message**: assert 1.0999999 == 1.1 (float32 precision issue)

### 6. test_loop[erk]
**Type**: AssertionError
**Message**: State summaries differ by ~0.008 (4% difference from expected)

### 7. test_all_lower_plumbing (also counted in errors)
**Type**: IndexError
**Message**: Buffer sizing issue in kernel plumbing

## Analysis

### NaN Errors Still Present
The NaN errors indicate that `sample_summaries_every` is still NaN when the loop tries to convert it to integer. The `_sample_summaries_auto_computed` tracking flag approach is not properly preventing NaN propagation.

### Sentinel Values Leaking to Output
The `-3.9999997e+32` and `3.9999997e+32` values in test failures suggest the min/max sentinel initialization is leaking into final outputs.

### Root Cause Hypothesis
1. The `_sample_summaries_auto_computed` flag may not be set correctly before NaN is assigned
2. The sentinel replacement in loop build() may not be covering all cases
3. `is_duration_dependent` may still have issues gating NaN usage

## Recommendations

1. Review where `sample_summaries_every` is set to NaN and ensure tracking flag is set first
2. Check if sentinel value is properly converted before use in computations  
3. Verify loop build() properly handles sentinel replacement for all affected metrics
4. Consider checking tracking flag in more locations

## Test Files Status
- tests/batchsolving/test_SolverKernel.py: 1 FAILED/ERROR
- tests/integrators/test_SingleIntegratorRun.py: 4 ERRORS
- tests/integrators/loops/test_ode_loop.py: 6 FAILED, 5 ERRORS
- tests/batchsolving/test_solver.py: All passed
