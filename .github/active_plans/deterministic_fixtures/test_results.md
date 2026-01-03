# Test Results Summary

## Overview
- **Tests Run**: 33
- **Passed**: 33
- **Failed**: 0
- **Errors**: 0
- **Skipped**: 0

## Test Command Executed
```bash
NUMBA_ENABLE_CUDASIM=1 pytest -m "not nocudasim and not specific_algos" -v --tb=short tests/outputhandling/test_output_functions.py
```

## Fix Applied

The test file was updated to properly initialize summary metric buffers before the main test loop. The issue was that summary metrics like `max` and `min` require sentinel initialization values (`-1e30` for max, `1e30` for min) to work correctly with all input ranges, including negative values.

### Changes Made to `tests/outputhandling/test_output_functions.py`:

1. Added output height variables for dummy initialization arrays:
   ```python
   state_summary_output_height = output_functions.state_summaries_output_height
   obs_summary_output_height = output_functions.observable_summaries_output_height
   ```

2. Added handling for the `test_shared_mem is False` case to ensure non-zero local array sizes.

3. Added buffer initialization code in the test kernel that calls `save_summary_metrics_func` once before the main loop:
   ```python
   # Initialize summary buffers with proper sentinel values
   dummy_state_summary_out = cuda.local.array(state_summary_output_height, ...)
   dummy_obs_summary_out = cuda.local.array(obs_summary_output_height, ...)
   save_summary_metrics_func(
       state_summaries,
       observable_summaries,
       dummy_state_summary_out,
       dummy_obs_summary_out,
       summarise_every,
   )
   ```

This matches the behavior of the real ODE loop which calls `save_summaries` at t=0 to initialize buffers to proper sentinel values.

## All Tests Passed
- test_save_time[output_test_settings_overrides0]
- test_save_time[output_test_settings_overrides1]
- test_output_functions_build[no_summaries]
- test_output_functions_build[all_summaries]
- test_output_functions_build[single_saved]
- test_output_functions_build[saved_index_out_of_bounds]
- test_output_functions_build[saved_empty]
- test_output_functions_build[no_output_types]
- test_all_summaries_long_run[float32-large_dataset]
- test_all_summaries_long_run[float64-large_dataset]
- test_all_summaries_long_window[float32-large_dataset]
- test_all_summaries_long_window[float64-large_dataset]
- test_memory_types[local_mem]
- test_memory_types[shared_mem]
- test_input_value_ranges[tiny_values]
- test_input_value_ranges[large_values]
- test_input_value_ranges[wide_range]
- test_no_summarys[state_obs]
- test_no_summarys[obs_only]
- test_no_summarys[obs_time]
- test_no_summarys[time_only]
- test_no_summarys[state_obs_time]
- test_various_summaries[basic_summaries]
- test_various_summaries[peaks_only]
- test_various_summaries[all]
- test_various_summaries[state_and_mean]
- test_various_summaries[obs_and_mean]
- test_big_and_small_systems[1_1]
- test_big_and_small_systems[10_5]
- test_big_and_small_systems[50_20]
- test_big_and_small_systems[100_100]
- test_frequent_summaries
- test_output_array_heights_property
