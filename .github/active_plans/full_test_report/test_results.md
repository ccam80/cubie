# Test Results Summary

## Overview
- **Tests Run**: 1395 (selected from 1592, 197 deselected due to markers)
- **Passed**: 1388
- **Failed**: 7
- **Errors**: 0
- **Skipped**: 0
- **Warnings**: 56537

## Test Command Executed
```bash
NUMBA_ENABLE_CUDASIM=1 pytest -m "not nocudasim and not specific_algos" -n0 --tb=line --no-cov --no-header -q
```

## Failures

### 1. test_basearraymanager.py::test_chunked_shape_propagates_through_allocation
**File**: `tests/batchsolving/arrays/test_basearraymanager.py:1763`
**Type**: AssertionError
**Message**: `assert False is True` - `ManagedArray.needs_chunked_transfer` returns `False` when it should be `True`
**Details**: The test expects `manager.host.arr1.needs_chunked_transfer` to be `True` after allocation, but it returns `False`. The `chunked_shape` is `None` when `is_chunked=True`.

---

### 2. test_batchinputarrays.py::TestBufferPoolIntegration::test_initialise_uses_buffer_pool_when_chunked
**File**: `tests/batchsolving/arrays/test_batchinputarrays.py:513`
**Type**: AssertionError
**Message**: `assert 0 > 0` - `_active_buffers` list is empty
**Details**: The test expects `len(input_arrays._active_buffers) > 0` after initialization, but the list is empty. The buffer pool integration is not populating `_active_buffers`.

---

### 3. test_batchinputarrays.py::TestBufferPoolIntegration::test_release_buffers_returns_to_pool
**File**: `tests/batchsolving/arrays/test_batchinputarrays.py:550`
**Type**: AssertionError
**Message**: `assert 0 > 0` - Buffer list is empty
**Details**: The test expects buffers to be present but finds an empty list.

---

### 4. test_batchinputarrays.py::TestBufferPoolIntegration::test_reset_clears_buffer_pool_and_active_buffers
**File**: `tests/batchsolving/arrays/test_batchinputarrays.py:618`
**Type**: AssertionError
**Message**: `assert 0 > 0` - `_active_buffers` list is empty
**Details**: Similar to #2, the test expects `len(input_arrays._active_buffers) > 0` initially, but the list is empty.

---

### 5. test_batchoutputarrays.py::TestNeedsChunkedTransferBranching::test_convert_host_to_numpy_uses_needs_chunked_transfer
**File**: `tests/batchsolving/arrays/test_batchoutputarrays.py:715`
**Type**: AssertionError
**Message**: `assert False is True` - `ManagedArray.needs_chunked_transfer` returns `False`
**Details**: The test expects `needs_chunked_transfer` to be `True` for an array with `is_chunked=True` and `chunked_shape=(51, 3, 5)`, but it returns `False`.

---

### 6. test_batchoutputarrays.py::TestNeedsChunkedTransferBranching::test_finalise_uses_needs_chunked_transfer
**File**: `tests/batchsolving/arrays/test_batchoutputarrays.py:786`
**Type**: AssertionError
**Message**: `assert 0 > 0` - `_pending_buffers` list is empty
**Details**: The test expects `len(output_arrays._pending_buffers) > 0`, but the list is empty.

---

### 7. test_SolverKernel.py::test_all_lower_plumbing
**File**: `tests/batchsolving/test_SolverKernel.py` (via CUDA simulator)
**Type**: IndexError
**Message**: `tid=[0, 2, 0] ctaid=[0, 0, 0]: index 2 is out of bounds for axis 1 with size 1`
**Details**: A CUDA kernel is accessing an array index out of bounds. The kernel is trying to access index `2` on axis `1` which only has size `1`. This is an actual bug in the kernel or its configuration.

---

## Failure Analysis

### Common Theme: Buffer Pool / Chunked Transfer Issues (Failures 1-6)
Six out of seven failures relate to:
1. `needs_chunked_transfer` property returning `False` when `True` is expected
2. Buffer pool not populating `_active_buffers` or `_pending_buffers` lists

This suggests a possible issue with:
- The `needs_chunked_transfer` property implementation in `ManagedArray`
- Buffer pool initialization logic in chunked array managers
- Missing setup steps in CUDA simulation mode

### Separate Issue: Array Index Out of Bounds (Failure 7)
The kernel in `test_all_lower_plumbing` is attempting to access invalid array indices. This could be:
- A configuration mismatch between array sizes and kernel launch parameters
- A bug in array shape calculation

## Recommendations

1. **Investigate `needs_chunked_transfer` property**: Check the logic in `ManagedArray` class that determines this property. It may not be correctly handling the case when `chunked_shape` is set.

2. **Check buffer pool initialization**: Review the `_initialise` method for `InputArrays` and `OutputArrays` to ensure buffers are correctly acquired from the pool.

3. **Fix kernel array bounds**: For `test_all_lower_plumbing`, verify that the kernel's output arrays have the correct dimensions for the number of observables being accessed.

4. **Review CUDA simulation compatibility**: Some of these failures may be specific to CUDA simulation mode and might pass on actual GPU hardware.

## Notable Warnings

- **Bitwise inversion deprecation**: Multiple warnings about `~` on bool being deprecated in Python 3.16
- **Numba deprecation**: `nopython=False` keyword argument deprecation warnings
- **PytestCollectionWarning**: Several test classes with `__init__` constructors cannot be collected as test classes

## Test Execution Time
Total: **284.79 seconds** (approximately 4 minutes 45 seconds)
