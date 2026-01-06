# Test Results Summary (Post-Review Fix Run)

## Overview
- **Tests Run**: 93
- **Passed**: 82
- **Failed**: 7
- **Errors**: 4
- **Skipped**: 0

## Status: NaN Errors Still Present

The 3 NaN errors that the reviewer identified are **still occurring**. The `samples_per_summary` fix either was not applied correctly or there are additional issues in the timing calculation chain.

## Command Executed
```bash
NUMBA_ENABLE_CUDASIM=1 pytest -m "not nocudasim and not specific_algos" -v --tb=short tests/integrators/test_SingleIntegratorRun.py tests/integrators/loops/test_ode_loop.py tests/batchsolving/test_SolverKernel.py tests/batchsolving/test_solver.py
```

## Errors (4)

### tests/integrators/loops/test_ode_loop.py::test_summarise_last_with_summarise_every_combined
**Type**: IndexError
**Message**: tid=[0, 0, 0] ctaid=[0, 0, 0]: index 3 is out of bounds for axis 0 with size 3

### tests/integrators/loops/test_ode_loop.py::test_save_last_flag_from_config
**Type**: ValueError
**Message**: cannot convert float NaN to integer

### tests/integrators/loops/test_ode_loop.py::test_summarise_last_flag_from_config
**Type**: ValueError
**Message**: cannot convert float NaN to integer

### tests/integrators/loops/test_ode_loop.py::test_summarise_last_collects_final_summary
**Type**: ValueError
**Message**: cannot convert float NaN to integer

## Failures (7)

### tests/integrators/loops/test_ode_loop.py::test_adaptive_controller_with_float32
**Type**: AssertionError
**Message**: assert 0.0 == 1.0000799894332886 ± 1.0e-06 - Obtained: 0.0, Expected: ~1.0

### tests/integrators/loops/test_ode_loop.py::test_save_at_settling_time_boundary
**Type**: AssertionError
**Message**: assert 1.0999999 == 1.1 - Float precision boundary issue

### tests/integrators/loops/test_ode_loop.py::test_loop[erk]
**Type**: AssertionError
**Message**: State summaries mismatch. Max absolute difference: 0.008452177, Max relative difference: 0.04125927

### tests/integrators/loops/test_ode_loop.py::test_all_summary_metrics_numerical_check[combined metrics]
**Type**: AssertionError
**Message**: State summaries mismatch. Mismatched elements: 5 / 64 (7.81%). Max absolute difference: 3.9999997e+32 (likely uninitialized peak values)

### tests/integrators/loops/test_ode_loop.py::test_all_summary_metrics_numerical_check[no combos]
**Type**: AssertionError
**Message**: State summaries mismatch. Mismatched elements: 2 / 40 (5%). Max absolute difference: 3.9999997e+32 (likely uninitialized peak values)

### tests/integrators/loops/test_ode_loop.py::test_all_summary_metrics_numerical_check[1st generation metrics]
**Type**: AssertionError
**Message**: State summaries mismatch. Mismatched elements: 1 / 24 (4.17%). Peak count expected 3 but got 2.

### tests/batchsolving/test_SolverKernel.py::test_all_lower_plumbing
**Type**: TypeError
**Message**: OutputFunctions.__init__() got an unexpected keyword argument 'save_every'

## Analysis

### Error Categories

1. **NaN to integer conversion (3 errors)**: The tests `test_save_last_flag_from_config`, `test_summarise_last_flag_from_config`, and `test_summarise_last_collects_final_summary` all fail with "cannot convert float NaN to integer". This suggests a timing calculation is returning NaN, likely from `output_length()` or `summaries_length()` methods.

2. **Index out of bounds (1 error)**: `test_summarise_last_with_summarise_every_combined` has an index error, suggesting array sizing is incorrect.

3. **Uninitialized peak values (3 failures)**: Several summary metrics tests show values of ±3.9999997e+32 vs expected 0.0, indicating peak min/max values aren't being properly initialized or updated.

4. **OutputFunctions API change (1 failure)**: `test_all_lower_plumbing` fails because `OutputFunctions.__init__()` no longer accepts `save_every` keyword argument.

5. **Numerical precision issues (2 failures)**: `test_adaptive_controller_with_float32` returns 0.0 instead of ~1.0, and `test_save_at_settling_time_boundary` has float32 precision issues.

## Recommendations

1. **Fix timing calculation NaN issue**: Investigate why `output_length()` or `summaries_length()` returns NaN. Check for division by zero or invalid inputs when calculating timing.

2. **Fix index bounds error**: Check array sizing in `test_summarise_last_with_summarise_every_combined` - the summarise_every calculation may be producing wrong buffer sizes.

3. **Update test_all_lower_plumbing**: The test passes `save_every` to OutputFunctions but the API appears to have changed. Update test to use current API.

4. **Investigate peak metric initialization**: The ±4e+32 values are likely uninitialized sentinels for peak min/max. Check if peaks are being properly computed when no peaks occur.

5. **Check adaptive controller integration**: The test expects ~1.0 but gets 0.0, suggesting the integration isn't running or results aren't being captured.
