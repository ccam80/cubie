# Test Results Summary

## Overview
- **Tests Run**: 120
- **Passed**: 119
- **Failed**: 1
- **Errors**: 0
- **Skipped**: 0

## Command Executed
```bash
NUMBA_ENABLE_CUDASIM=1 pytest -m "not nocudasim and not specific_algos" -v --tb=short tests/outputhandling/test_output_functions.py tests/outputhandling/test_output_config.py tests/outputhandling/summarymetrics/test_summary_metrics.py
```

## Failures

### tests/outputhandling/test_output_config.py::TestSaveEveryProperty::test_save_every_default
**Type**: AssertionError
**Message**: `assert None == 0.01` - The `save_every` property returns `None` but the test expects `0.01`

**Details**: The test creates an OutputConfig with `_save_every=None` and `_sample_summaries_every=None`, then asserts that `save_every` should return `0.01`. However, the property is returning `None`.

This suggests that the `save_every` property's default behavior may have changed or is not correctly falling back to the expected default value of `0.01` when `_save_every` is `None`.

## Recommendations

1. **Investigate `save_every` property**: Check the implementation of the `save_every` property in `src/cubie/outputhandling/output_config.py` to understand when it should return a default value vs `None`.

2. **Verify test expectations**: The test `test_save_every_default` expects `save_every` to return `0.01` when `_save_every` is `None`. Determine if:
   - The property should have a default fallback value
   - The test expectation is incorrect
   - The refactor inadvertently changed this behavior

3. **Check related changes**: Since this is a rename refactor from `dt_save` to `sample_summaries_every`, verify that the `save_every` property logic wasn't accidentally modified during the refactor.
