# Test Results Summary

## Overview
- **Tests Run**: 54
- **Passed**: 35
- **Failed**: 2
- **Errors**: 17
- **Skipped**: 0

## Command Executed
```bash
NUMBA_ENABLE_CUDASIM=1 pytest -m "not nocudasim and not specific_algos" -v --tb=short \
  tests/integrators/loops/test_ode_loop_config_timing.py \
  tests/batchsolving/test_kernel_output_lengths.py \
  tests/batchsolving/test_duration_propagation.py \
  tests/batchsolving/test_solver_warnings.py \
  tests/batchsolving/test_solver_timing_properties.py \
  tests/batchsolving/test_timing_modes.py \
  tests/batchsolving/test_config_plumbing.py \
  tests/integrators/loops/test_buffer_settings.py
```

## Errors (17)

All errors are the same root cause: **ValueError: cannot convert float NaN to integer**

This occurs during fixture setup when creating solvers with timing parameters set to None. The error suggests that somewhere in the solver construction chain, a NaN value is being converted to an integer.

### Affected Tests:
| Test File | Test Name |
|-----------|-----------|
| test_kernel_output_lengths.py | TestWarmupLengthNoneSafe::test_warmup_length_with_save_every_none |
| test_kernel_output_lengths.py | TestTimingPropertyReturnTypes::test_save_every_returns_none |
| test_kernel_output_lengths.py | TestTimingPropertyReturnTypes::test_summarise_every_returns_none |
| test_kernel_output_lengths.py | TestOutputLengthNoneSafe::test_output_length_with_save_every_none |
| test_kernel_output_lengths.py | TestSummariesLengthNoneSafe::test_summaries_length_with_summarise_every_none |
| test_duration_propagation.py | TestDurationPropagation::test_samples_per_summary_uses_propagated_duration |
| test_duration_propagation.py | TestDurationPropagation::test_duration_propagates_to_loop_config |
| test_duration_propagation.py | TestDurationUpdateChain::test_duration_property_returns_value |
| test_solver_warnings.py | TestDurationDependencyWarning::test_no_warning_with_explicit_summarise_every |
| test_solver_warnings.py | TestDurationDependencyWarning::test_duration_dependency_warning_emitted |
| test_solver_timing_properties.py | TestSolverTimingPropertyReturnTypes::test_solver_summarise_every_returns_none_in_summarise_last_mode |
| test_solver_timing_properties.py | TestSolverTimingPropertyReturnTypes::test_solver_save_every_returns_none_in_save_last_mode |
| test_timing_modes.py | TestTimingModeOutputLengths::test_save_last_only_output_length |
| test_timing_modes.py | TestTimingModeOutputLengths::test_summarise_last_only_summaries_length |
| test_timing_modes.py | TestTimingModeOutputLengths::test_periodic_summarise_length |
| test_timing_modes.py | TestDurationDependencyWarning::test_warning_on_summarise_last_without_summarise_every |
| test_timing_modes.py | TestParameterReset::test_sample_summaries_every_recalculates_on_none |

## Failures (2)

### test_timing_modes.py::TestTimingModeOutputLengths::test_periodic_save_output_length
**Type**: IndexError
**Message**: `tid=[0, 0, 0] ctaid=[0, 0, 0]: index 0 is out of bounds for axis 0 with size 0`

This appears to be a CUDA kernel execution error where the output buffer has size 0, indicating the output_length calculation returned 0 or None when a positive value was expected.

### test_solver_warnings.py::TestDurationDependencyWarning::test_no_warning_with_summarise_last_false
**Type**: IndexError  
**Message**: `tid=[0, 0, 0] ctaid=[0, 0, 0]: index 5 is out of bounds for axis 0 with size 5`

This appears to be an off-by-one error in buffer allocation or indexing during kernel execution.

## Root Cause Analysis

### Primary Issue: NaN-to-Integer Conversion
The 17 errors all share the same root cause: when timing parameters (`save_every`, `summarise_every`) are set to `None`, somewhere in the initialization chain a `NaN` value is being used where an integer is expected.

**Likely location**: The issue is probably in one of these areas:
1. `ODELoopConfig` initialization or property calculations
2. `BatchSolverKernel` output length calculations
3. Buffer allocation code that uses timing parameters

The None-safety changes may not have been applied consistently throughout the initialization chain, allowing NaN values to propagate to integer conversion operations.

### Secondary Issue: Buffer Sizing
The 2 failures indicate that even when tests run, buffer sizing calculations are producing incorrect values (either 0 or off-by-one errors).

## Recommendations

1. **Trace the NaN source**: Add debugging to identify where NaN values are introduced when timing parameters are None. Check:
   - `ODELoopConfig.duration` property when duration is None
   - `samples_per_summary` calculations
   - Any `float('nan')` sentinel values

2. **Review None-safety in initialization chain**: Ensure that when `save_every=None` or `summarise_every=None`:
   - No calculations attempt to use these values before checking for None
   - Buffer allocation code handles None properly
   - Output length properties return None (not 0 or NaN)

3. **Fix buffer sizing**: Review the output_length and summaries_length calculations to ensure they:
   - Return None when appropriate timing parameters are None
   - Never return 0 for active output modes
   - Correctly calculate buffer sizes for all timing modes

4. **Check test fixtures**: Some test configurations may be incompatible (e.g., testing None timing parameters requires the solver to handle them gracefully during construction).
