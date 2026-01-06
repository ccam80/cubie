# Test Results Summary (Post-Review Fix Attempt)

## Overview
- **Tests Run**: 89
- **Passed**: 82
- **Failed**: 7
- **Errors**: 8
- **Skipped**: 0

## Status: NaN Errors NOT RESOLVED

The fix returning `2**30` as a sentinel value when `summarise_last=True` did NOT resolve the NaN errors. The errors persist.

## Errors (8 tests with setup/collection errors)

### tests/integrators/test_SingleIntegratorRun.py::TestTimingFlagAutoDetection::test_save_last_set_when_state_output_no_save_every
**Type**: ValueError
**Message**: cannot convert float NaN to integer

### tests/integrators/test_SingleIntegratorRun.py::TestSummariseFlagAutoDetection::test_summarise_last_set_when_mean_output_no_timing
**Type**: ValueError
**Message**: cannot convert float NaN to integer

### tests/integrators/test_SingleIntegratorRun.py::TestChunkDurationInterception::test_sample_summaries_every_computed_from_chunk_duration
**Type**: ValueError
**Message**: cannot convert float NaN to integer

### tests/integrators/test_SingleIntegratorRun.py::TestChunkDurationInterception::test_warning_emitted_when_sample_summaries_every_computed
**Type**: ValueError
**Message**: cannot convert float NaN to integer

### tests/integrators/loops/test_ode_loop.py::test_summarise_last_collects_final_summary
**Type**: ValueError
**Message**: cannot convert float NaN to integer

### tests/integrators/loops/test_ode_loop.py::test_save_last_flag_from_config
**Type**: ValueError
**Message**: cannot convert float NaN to integer

### tests/integrators/loops/test_ode_loop.py::test_summarise_last_flag_from_config
**Type**: ValueError
**Message**: cannot convert float NaN to integer

### tests/integrators/loops/test_ode_loop.py::test_summarise_last_with_summarise_every_combined
**Type**: IndexError
**Message**: index 3 is out of bounds for axis 0 with size 3

## Failures (7 tests failed assertions)

### tests/batchsolving/test_SolverKernel.py::test_all_lower_plumbing
**Type**: TypeError
**Message**: OutputFunctions.__init__() got an unexpected keyword argument 'save_every'
**Analysis**: The OutputFunctions class does not accept 'save_every' as a constructor argument.

### tests/integrators/loops/test_ode_loop.py::test_adaptive_controller_with_float32
**Type**: AssertionError
**Message**: assert 0.0 == 1.0000799894332886 ± 1.0e-06
**Analysis**: Adaptive controller test returning 0.0 instead of expected ~1.0.

### tests/integrators/loops/test_ode_loop.py::test_save_at_settling_time_boundary
**Type**: AssertionError
**Message**: assert 1.0999999 == 1.1 (float32 precision issue)
**Analysis**: Floating point precision mismatch.

### tests/integrators/loops/test_ode_loop.py::test_loop[erk]
**Type**: AssertionError
**Message**: state summaries mismatch - values differ by up to 0.008
**Analysis**: ERK loop producing different numerical results than reference.

### tests/integrators/loops/test_ode_loop.py::test_all_summary_metrics_numerical_check[no combos]
**Type**: AssertionError
**Message**: state summaries mismatch - sentinel values -4e+32 appearing
**Analysis**: Sentinel values leaking into results (related to 2**30 fix causing -4e+32 contamination).

### tests/integrators/loops/test_ode_loop.py::test_all_summary_metrics_numerical_check[1st generation metrics]
**Type**: AssertionError
**Message**: state summaries mismatch - count is 2 instead of expected 3
**Analysis**: Metric count off by 1.

### tests/integrators/loops/test_ode_loop.py::test_all_summary_metrics_numerical_check[combined metrics]
**Type**: AssertionError
**Message**: state summaries mismatch - sentinel values -4e+32/+4e+32 appearing, count 0 vs 3
**Analysis**: Multiple issues with combined metrics.

## Root Cause Analysis

1. **NaN source is NOT `samples_per_summary`**: The NaN is coming from elsewhere in the timing computation pipeline, likely:
   - `dt` computation when no timing info provided
   - `samples_per_chunk` computation
   - Division by zero or undefined operations

2. **Sentinel value contamination**: The `2**30` fix is causing new problems - values like `-3.9999997e+32` and `+3.9999997e+32` are appearing in summary results, indicating the sentinel is being used in calculations instead of being handled specially.

3. **OutputFunctions API mismatch**: Test uses `save_every` but constructor doesn't accept it.

## Recommendations

1. **Find actual NaN source**: Trace back from the error to find where NaN originates. Check:
   - `ODELoopConfig.from_solver_config()` - what happens when timing params are None
   - How `dt` is computed when no explicit timing provided
   - Division operations in timing computations

2. **Fix sentinel value handling**: When `samples_per_summary == 2**30`, the loop must NOT use it in actual summary computations - only use `summarise_last` to trigger final summary.

3. **Update OutputFunctions test**: Fix parameter name mismatch.
