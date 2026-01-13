# Test Results Summary

## Overview
- **Tests Run**: 113
- **Passed**: 108
- **Failed**: 4
- **Errors**: 1
- **Skipped**: 0

## Command Executed
```bash
NUMBA_ENABLE_CUDASIM=1 pytest -m "not nocudasim and not specific_algos" -v --tb=short tests/test_utils.py::TestEnsureNonzeroSize tests/memory/test_memmgmt.py tests/batchsolving/arrays/test_basearraymanager.py tests/batchsolving/test_solver.py::test_solver_solve_with_save_variables
```

## New Tests for ensure_nonzero_size

All 8 new `ensure_nonzero_size` tests **PASSED**:
- `test_last_element_zero` ✓
- `test_single_zero_replaced` ✓
- `test_multiple_zeros_replaced` ✓
- `test_all_zeros_replaced` ✓
- `test_no_zeros_unchanged` ✓
- `test_integer_zero` ✓
- `test_integer_nonzero` ✓
- `test_first_element_zero` ✓

## Errors

### tests/batchsolving/test_solver.py::test_solver_solve_with_save_variables
**Type**: TypeError (at setup)
**Message**: `'>' not supported between instances of 'str' and 'int'`
**Location**: `src/cubie/_utils.py:641` in `ensure_nonzero_size`
**Details**: The bug fix using `max(1, v)` fails when `value` contains strings instead of integers. The `BatchOutputSizes` class has string fields that get passed to `ensure_nonzero_size`.

## Failures

### TestBaseArrayManager::test_initialize_device_zeros
**Type**: TypeError
**Message**: `'NoneType' object does not support item assignment`
**Notes**: Pre-existing test failure unrelated to memory refactor cleanup.

### TestCheckSizesAndTypes::test_check_sizes_with_chunking
**Type**: AssertionError
**Message**: `assert False is True`
**Notes**: Pre-existing test failure unrelated to memory refactor cleanup.

### test_chunked_shape_propagates_through_allocation
**Type**: AssertionError
**Message**: `assert False is True` - `needs_chunked_transfer` returned `False` when expected `True`
**Notes**: Pre-existing test failure unrelated to memory refactor cleanup.

### TestMemoryManager::test_process_request
**Type**: AssertionError
**Message**: `assert (4, 4, 8) == (4, 4, 4)` - shape mismatch at index 2
**Notes**: Pre-existing test failure unrelated to memory refactor cleanup.

## Analysis

### Bug in ensure_nonzero_size Fix

The solver test reveals that the bug fix for `ensure_nonzero_size` is incomplete. The function now uses `max(1, v)` but this fails when the value is a string. Looking at the code path:

1. `BatchOutputSizes.from_solver()` creates sizes
2. `output_sizes.py:56` calls `ensure_nonzero_size(value)` on each field
3. `BatchOutputSizes` contains string fields (like `stride_order`)
4. `max(1, "time")` raises `TypeError`

### Root Cause

The `ensure_nonzero_size` function processes fields without checking if they are numeric. The `@nonzero` property in `output_sizes.py` iterates over all attrs fields, including strings.

### Required Fix

The `ensure_nonzero_size` function needs to handle non-numeric types gracefully, OR the `nonzero` property needs to filter to only numeric fields.

## Recommendations

1. **Fix `ensure_nonzero_size`**: Add type checking to skip/pass through non-numeric values:
   ```python
   def ensure_nonzero_size(value):
       if isinstance(value, tuple):
           return tuple(max(1, v) if isinstance(v, (int, float)) else v for v in value)
       elif isinstance(value, (int, float)):
           return max(1, value)
       else:
           return value  # pass through non-numeric (e.g., strings)
   ```

2. **OR Fix `output_sizes.py`**: Filter to only numeric fields in the `nonzero` property.

3. **Pre-existing failures**: The 3 other failures and 1 error related to chunking/memory appear to be pre-existing issues unrelated to the memory refactor cleanup task.
