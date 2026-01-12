# Test Results Summary

## Overview
- **Tests Run**: 344 (batchsolving directory)
- **Passed**: 341
- **Failed**: 2
- **Errors**: 0
- **Skipped**: 0

## New chunk_axis Property Tests
- **File**: `tests/batchsolving/test_chunk_axis_property.py`
- **Total**: 8 tests
- **Passed**: 7
- **Failed**: 1

## Failures

### tests/batchsolving/test_chunk_axis_property.py::TestChunkAxisInRun::test_chunk_axis_property_after_run
**Type**: UnboundLocalError  
**Message**: cannot access local variable 'chunksize' where it is not associated with a value  
**Location**: `src/cubie/batchsolving/BatchSolverKernel.py:653`

**Root Cause**: The test passes `chunk_axis="variable"` to `kernel.run()`, but the `chunk_run()` method only handles `"run"` and `"time"` values. When `chunk_axis="variable"` is passed, the `chunksize` variable is never assigned, causing the UnboundLocalError.

**Analysis**: The `run()` method documentation explicitly states that `chunk_axis` should be either `"run"` or `"time"`. The test is using an invalid value `"variable"`.

**Fix Required**: The test should use a valid `chunk_axis` value (either `"run"` or `"time"`), not `"variable"`. The test appears to be incorrectly testing an invalid scenario as if it were valid.

### tests/batchsolving/test_chunked_solver.py::TestChunkedSolverExecution::test_chunked_solve_produces_valid_output[run]
**Type**: UnboundLocalError  
**Message**: cannot access local variable 'chunksize' where it is not associated with a value  
**Location**: `src/cubie/batchsolving/BatchSolverKernel.py:653`

**Root Cause**: This appears to be the same underlying issue - either an invalid `chunk_axis` value is being passed, or there's an edge case where `chunk_axis` doesn't match `"run"` or `"time"`.

## Recommendations

1. **Fix test_chunk_axis_property_after_run**: Change line 114 in `tests/batchsolving/test_chunk_axis_property.py` from `chunk_axis="variable"` to `chunk_axis="time"` (or keep `"run"` default).

2. **Consider adding validation**: The `run()` method could validate that `chunk_axis` is one of the allowed values (`"run"` or `"time"`) before proceeding.

3. **Clarify "variable" usage**: The `BaseArrayManager` allows `"variable"` as a valid `_chunk_axis` value, but this is not supported by `kernel.run()`. Either:
   - Remove `"variable"` from the allowed values in `BaseArrayManager` if it's not a valid runtime option
   - OR add support for `"variable"` in the `chunk_run()` method if it should be valid

## Passing Tests (key ones)

- ✓ test_chunk_axis_property_returns_default_run
- ✓ test_chunk_axis_property_returns_consistent_value
- ✓ test_chunk_axis_property_raises_on_inconsistency
- ✓ test_chunk_axis_setter_updates_both_arrays
- ✓ test_chunk_axis_setter_allows_valid_values
- ✓ test_run_sets_chunk_axis_on_arrays
- ✓ test_update_from_solver_does_not_change_chunk_axis
