# Test Results Summary (Post Buffer Sizing Fix Attempt)

## Overview
- **Tests Run**: 43
- **Passed**: 24
- **Failed**: 4
- **Errors**: 15
- **Skipped**: 0

## Command Executed
```bash
NUMBA_ENABLE_CUDASIM=1 pytest -m "not nocudasim and not specific_algos" -v --tb=short \
  tests/integrators/loops/test_ode_loop_config_timing.py \
  tests/batchsolving/test_kernel_output_lengths.py \
  tests/batchsolving/test_duration_propagation.py \
  tests/batchsolving/test_solver_warnings.py \
  tests/batchsolving/test_solver_timing_properties.py \
  tests/batchsolving/test_timing_modes.py
```

## Status: Test Run Has Regressed

The previous run had 33 passed with 10 failures. This run has 24 passed with 4 failures and 15 errors - indicating the fixes have introduced new problems.

---

## Errors (15)

All 15 errors share the same root cause:

**Type**: ValueError
**Message**: `cannot convert float NaN to integer`

**Affected Tests**:
1. test_kernel_output_lengths.py::TestOutputLengthNoneSafe::test_output_length_with_save_every_none
2. test_kernel_output_lengths.py::TestSummariesLengthNoneSafe::test_summaries_length_with_summarise_every_none
3. test_kernel_output_lengths.py::TestTimingPropertyReturnTypes::test_save_every_returns_none
4. test_kernel_output_lengths.py::TestTimingPropertyReturnTypes::test_summarise_every_returns_none
5. test_kernel_output_lengths.py::TestWarmupLengthNoneSafe::test_warmup_length_with_save_every_none
6. test_duration_propagation.py::TestDurationPropagation::test_samples_per_summary_uses_propagated_duration
7. test_duration_propagation.py::TestDurationUpdateChain::test_duration_property_returns_value
8. test_duration_propagation.py::TestDurationPropagation::test_duration_propagates_to_loop_config
9. test_solver_warnings.py::TestDurationDependencyWarning::test_duration_dependency_warning_emitted
10. test_solver_timing_properties.py::TestSolverTimingPropertyReturnTypes::test_solver_save_every_returns_none_in_save_last_mode
11. test_solver_timing_properties.py::TestSolverTimingPropertyReturnTypes::test_solver_summarise_every_returns_none_in_summarise_last_mode
12. test_timing_modes.py::TestTimingModeOutputLengths::test_save_last_only_output_length
13. test_timing_modes.py::TestTimingModeOutputLengths::test_summarise_last_only_summaries_length
14. test_timing_modes.py::TestParameterReset::test_sample_summaries_every_recalculates_on_none
15. test_timing_modes.py::TestDurationDependencyWarning::test_warning_on_summarise_last_without_summarise_every

**Analysis**: The original NaN conversion error has reappeared. The fix for `summaries_length` to check `summarise_last` flag directly appears to have been reverted or is not properly in place.

---

## Failures (4)

### 1. test_solver_warnings.py::TestDurationDependencyWarning::test_no_warning_with_summarise_last_false
**Type**: IndexError
**Message**: `tid=[0, 0, 0] ctaid=[0, 0, 0]: index 5 is out of bounds for axis 0 with size 5`

### 2. test_timing_modes.py::TestTimingModeOutputLengths::test_periodic_save_output_length
**Type**: IndexError
**Message**: `tid=[0, 0, 0] ctaid=[0, 0, 0]: index 0 is out of bounds for axis 0 with size 0`

### 3. test_solver_warnings.py::TestDurationDependencyWarning::test_no_warning_with_explicit_summarise_every
**Type**: IndexError
**Message**: `tid=[0, 0, 0] ctaid=[0, 0, 0]: index 5 is out of bounds for axis 0 with size 5`

### 4. test_timing_modes.py::TestTimingModeOutputLengths::test_periodic_summarise_length
**Type**: IndexError
**Message**: `tid=[0, 0, 0] ctaid=[0, 0, 0]: index 5 is out of bounds for axis 0 with size 5`

**Analysis**: These are buffer sizing issues unrelated to the NaN error. The buffer is being sized incorrectly.

---

## Root Cause Analysis

### Issue 1: NaN Conversion Errors Persist (15 errors)
The `summaries_length` property is still encountering NaN values during integer conversion. The fix to check `summarise_last` directly appears to not be working. This suggests:
- The fix may have been applied to the wrong property
- The fix may not cover all code paths
- There may be another location where NaN conversion occurs

### Issue 2: Buffer Sizing Off-by-One (4 failures)
Separate from the NaN issue, there are buffer sizing problems:
- "index 5 is out of bounds for axis 0 with size 5" - trying to access index 5 when only 0-4 are valid
- "index 0 is out of bounds for axis 0 with size 0" - buffer not allocated at all

---

## Recommendations

1. **Verify `summaries_length` fix is in place**: Check that the property correctly returns 1 when `summarise_last=True` and duration is not set.

2. **Check all NaN-to-integer conversion paths**: There may be multiple locations where this conversion happens:
   - `output_length` property
   - `warmup_length` property
   - `summaries_length` property
   - Buffer allocation code in `output_sizes.py`

3. **Investigate buffer sizing separately**: The IndexError failures indicate a separate issue in how buffers are sized for periodic save/summarise modes.

4. **Test expectation for samples_per_summary**: The previous test expectation issue (expecting 1 vs None) needs to be verified.

---

## Summary

**The test run shows regression from the previous state:**
- 15 tests error with "ValueError: cannot convert float NaN to integer"
- 4 tests fail with IndexError buffer sizing issues
- Only 24 tests pass (down from 33)

The taskmaster's fixes do not appear to have resolved the issues. The original NaN conversion errors are still occurring, and the buffer sizing issues persist.
