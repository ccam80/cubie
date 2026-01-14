# Test Results Summary

## Overview
- **Tests Run**: 10
- **Passed**: 3
- **Failed**: 7
- **Errors**: 0
- **Skipped**: 0

## Command Executed
```bash
NUMBA_ENABLE_CUDASIM=1 pytest -m "not nocudasim and not specific_algos" -v --tb=short \
  "tests/batchsolving/arrays/test_basearraymanager.py::test_chunked_shape_propagates_through_allocation" \
  "tests/batchsolving/arrays/test_batchinputarrays.py::TestBufferPoolIntegration::test_initialise_uses_buffer_pool_when_chunked" \
  "tests/batchsolving/arrays/test_batchinputarrays.py::TestBufferPoolIntegration::test_release_buffers_returns_to_pool" \
  "tests/batchsolving/arrays/test_batchinputarrays.py::TestBufferPoolIntegration::test_reset_clears_buffer_pool_and_active_buffers" \
  "tests/batchsolving/arrays/test_batchoutputarrays.py::TestNeedsChunkedTransferBranching::test_convert_host_to_numpy_uses_needs_chunked_transfer" \
  "tests/batchsolving/arrays/test_batchoutputarrays.py::TestNeedsChunkedTransferBranching::test_finalise_uses_needs_chunked_transfer" \
  "tests/batchsolving/test_SolverKernel.py::test_all_lower_plumbing" \
  "tests/batchsolving/test_chunk_axis_property.py::TestChunkAxisInRun::test_run_sets_chunk_axis_on_arrays" \
  "tests/batchsolving/test_chunk_axis_property.py::TestChunkAxisInRun::test_chunk_axis_property_after_run" \
  "tests/batchsolving/test_chunk_axis_property.py::TestUpdateFromSolverChunkAxis::test_update_from_solver_does_not_change_chunk_axis"
```

## Passed Tests

1. `tests/batchsolving/arrays/test_basearraymanager.py::test_chunked_shape_propagates_through_allocation`
2. `tests/batchsolving/test_SolverKernel.py::test_all_lower_plumbing`
3. `tests/batchsolving/test_chunk_axis_property.py::TestUpdateFromSolverChunkAxis::test_update_from_solver_does_not_change_chunk_axis`

## Failures

### tests/batchsolving/arrays/test_batchinputarrays.py::TestBufferPoolIntegration::test_initialise_uses_buffer_pool_when_chunked
**Type**: IndexError
**Message**: list assignment index out of range
**Location**: `test_batchinputarrays.py:515` - `chunked[run_idx] = chunk_size`
**Analysis**: Test fixture issue - `chunked` list is empty but test attempts to assign by index.

---

### tests/batchsolving/arrays/test_batchinputarrays.py::TestBufferPoolIntegration::test_release_buffers_returns_to_pool
**Type**: IndexError
**Message**: list assignment index out of range
**Location**: `test_batchinputarrays.py:573` - `chunked[run_idx] = chunk_size`
**Analysis**: Same fixture issue as above - `chunked` list is not properly sized before assignment.

---

### tests/batchsolving/arrays/test_batchinputarrays.py::TestBufferPoolIntegration::test_reset_clears_buffer_pool_and_active_buffers
**Type**: IndexError
**Message**: list assignment index out of range
**Location**: `test_batchinputarrays.py:660` - `chunked[run_idx] = chunk_size`
**Analysis**: Same fixture issue - test iterates with `run_idx` but `chunked` list isn't pre-allocated.

---

### tests/batchsolving/arrays/test_batchoutputarrays.py::TestNeedsChunkedTransferBranching::test_convert_host_to_numpy_uses_needs_chunked_transfer
**Type**: AssertionError
**Message**: `assert True is False` - `host_status.needs_chunked_transfer` returned `True` but expected `False`
**Location**: `test_batchoutputarrays.py:753`
**Analysis**: Test expects `needs_chunked_transfer` to be `False` for a host-only ManagedArray with `is_chunked=True`, but the property returns `True`. Either the test expectation is wrong or the `needs_chunked_transfer` logic needs fixing.

---

### tests/batchsolving/arrays/test_batchoutputarrays.py::TestNeedsChunkedTransferBranching::test_finalise_uses_needs_chunked_transfer
**Type**: AssertionError
**Message**: `assert 0 > 0` - `len(output_arrays_manager._pending_buffers)` is 0 but expected to be > 0
**Location**: `test_batchoutputarrays.py:845`
**Analysis**: Test expects pending buffers to be populated after some operation, but `_pending_buffers` remains empty. The chunked transfer logic may not be triggering as expected.

---

### tests/batchsolving/test_chunk_axis_property.py::TestChunkAxisInRun::test_run_sets_chunk_axis_on_arrays
**Type**: ValueError
**Message**: `input symbols in inputs_dict ({'placeholder'}) do not match drivers symbols in system ({'d0'})`
**Location**: `solver.py:429` → `array_interpolator.py:739`
**Analysis**: Test fixture uses placeholder symbols (`'placeholder'`) that don't match the actual driver symbols in the system (`'d0'`). The fixture needs to use valid driver names matching the system definition.

---

### tests/batchsolving/test_chunk_axis_property.py::TestChunkAxisInRun::test_chunk_axis_property_after_run
**Type**: ValueError
**Message**: `input symbols in inputs_dict ({'placeholder'}) do not match drivers symbols in system ({'d0'})`
**Location**: `solver.py:429` → `array_interpolator.py:739`
**Analysis**: Same issue as above - test fixture uses mismatched driver symbol names.

## Summary by Category

### Fixture Issues (5 failures)
- **IndexError in test_batchinputarrays.py** (3 tests): The test setup creates an empty `chunked` list but then attempts to assign by index (`chunked[run_idx] = chunk_size`). Should either:
  - Pre-allocate list with correct size: `chunked = [None] * expected_length`
  - Or use `chunked.append(chunk_size)` instead of index assignment

- **ValueError in test_chunk_axis_property.py** (2 tests): Tests use `{'placeholder': ...}` as inputs_dict but the system has driver symbol `'d0'`. Need to use `{'d0': ...}` or create matching fixtures.

### Logic/Assertion Issues (2 failures)
- **test_convert_host_to_numpy_uses_needs_chunked_transfer**: The `needs_chunked_transfer` property returns `True` for a host-only array with `is_chunked=True`. Review whether:
  - The property logic is correct (maybe host arrays with `is_chunked=True` should return `False`)
  - Or the test expectation is incorrect

- **test_finalise_uses_needs_chunked_transfer**: The `_pending_buffers` list is empty when expected to have entries. Either:
  - The test setup doesn't trigger the code path that populates `_pending_buffers`
  - Or there's a bug in the finalise logic for chunked transfers

## Recommendations

1. **Fix IndexError tests (test_batchinputarrays.py)**: Change from `chunked[run_idx] = chunk_size` to use list pre-allocation or append pattern.

2. **Fix ValueError tests (test_chunk_axis_property.py)**: Update the inputs_dict fixture to use the correct driver symbol name (`'d0'` instead of `'placeholder'`).

3. **Investigate needs_chunked_transfer logic**: Review the `needs_chunked_transfer` property to determine correct behavior for host-only arrays with `is_chunked=True`.

4. **Investigate pending_buffers population**: Trace the code path to understand when `_pending_buffers` should be populated and why it's empty in the test.
