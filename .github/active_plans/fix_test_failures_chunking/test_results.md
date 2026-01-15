# Test Results Summary - Chunking and Memory Management

**Test Run Date**: 2025-01-15
**CUDA Simulation**: Enabled (NUMBA_ENABLE_CUDASIM=1)
**Test Markers Excluded**: nocudasim, cupy

---

## Overall Results

| Test Suite | Total | Passed | Failed | Errors | Skipped |
|------------|-------|--------|--------|--------|---------|
| tests/memory/test_array_requests.py | 13 | 13 | 0 | 0 | 0 |
| tests/memory/test_memmgmt.py | 44 | 37 | 7 | 0 | 0 |
| tests/batchsolving/arrays/ (all) | 161 | 147 | 25 | 8 | 1 |
| tests/batchsolving/test_runparams.py | - | - | 3 | 0 | - |
| tests/batchsolving/test_runparams_integration.py | - | - | 3 | 0 | - |
| **TOTAL** | **218+** | **197** | **38** | **8** | **1** |

---

## ✅ Success: tests/memory/test_array_requests.py

**Status**: All 13 tests PASSED

This test suite validates:
- ArrayRequest accepts and validates `total_runs` field
- ArrayRequest `total_runs` defaults to None
- ArrayRequest validates `total_runs` is positive
- ArrayResponse has `chunked_shapes` field
- Chunk axis index validation

**No issues detected in this subsystem.**

---

## ⚠️ Partial Success: tests/memory/test_memmgmt.py

**Status**: 37 passed, 7 failed

### Failures

All 7 failures share the **same root cause**:

#### **Error Pattern**: `ValueError: No total_runs found in allocation requests`

**Failed Tests**:
1. `TestMemoryManager::test_process_request`
2. `TestMemoryManager::test_allocate_queue_single_instance`
3. `TestMemoryManager::test_allocate_queue_multiple_instances_group_limit`
4. `TestMemoryManager::test_allocate_queue_empty_queue`
5. `TestAllocateQueueExtractsNumRuns::test_allocate_queue_extracts_num_runs`
6. `TestAllocateQueueExtractsNumRuns::test_allocate_queue_chunks_correctly`
7. `test_allocate_queue_no_chunked_slices_in_response`

**Root Cause**:
The `MemoryManager._extract_num_runs()` method expects at least one `ArrayRequest` in the queue to have `total_runs` set, but these tests create ArrayRequests without specifying `total_runs`. The implementation now requires this field for chunking calculations.

**Error Location**: `src/cubie/memory/mem_manager.py:1210`

```python
if len(total_runs_values) == 0:
    raise ValueError(
        "No total_runs found in allocation requests. At least one "
        "request must specify total_runs for chunking calculations."
    )
```

**Issue**: Tests need to be updated to pass `total_runs` when creating `ArrayRequest` objects, OR the implementation needs to handle the case where `total_runs` is not provided (backward compatibility).

---

## ⚠️ Mixed Results: tests/batchsolving/arrays/

**Status**: 147 passed, 25 failed, 8 errors, 1 skipped

### Failure Categories

#### Category 1: Missing `total_runs` in ArrayRequests (3 failures)

**Error**: `ValueError: No total_runs found in allocation requests`

**Failed Tests**:
- `test_basearraymanager.py::TestBaseArrayManager::test_initialize_device_zeros`
- `test_basearraymanager.py::TestBaseArrayManager::test_request_allocation_auto`
- `test_basearraymanager.py::TestMemoryManagerIntegration::test_allocation_with_settings`

**Same root cause as memory tests above.**

---

#### Category 2: Inconsistent `total_runs` values (8 errors)

**Error**: `ValueError: Inconsistent total_runs in requests: found {1, 5}`

**Failed Tests** (all in `test_chunking.py`):
- `test_chunked_solver_produces_correct_results[100]`
- `test_chunked_solver_produces_correct_results[860]`
- `test_chunked_solver_produces_correct_results[1460]`
- `test_chunked_solver_produces_correct_results[2048]`
- `test_chunked_shape_equals_shape_when_not_chunking`
- `test_chunk_axis_index_in_array_requests`
- `test_chunked_shape_differs_from_shape_when_chunking`
- (and more)

**Root Cause**: Some arrays in the same allocation have `total_runs=1` (e.g., driver_coefficients) while others have `total_runs=5`. The new validation in `_extract_num_runs()` enforces consistency, but the existing code creates requests with different `total_runs` values.

**Error Location**: `src/cubie/memory/mem_manager.py:1219`

```python
if len(total_runs_values) > 1:
    raise ValueError(
        f"Inconsistent total_runs in requests: found {total_runs_values}. "
        "All requests with total_runs must have the same value."
    )
```

**Issue**: The implementation that creates ArrayRequests (likely in `BaseArrayManager.allocate()` or similar) is passing different `total_runs` values for different arrays. Arrays that don't chunk along the run axis (like driver_coefficients) should have `total_runs=None`, not a different value.

---

#### Category 3: Chunk Metadata Not Propagating (11 failures)

**Error Pattern**: Chunk parameters (length, axis_index) not being set correctly

**Failed Tests**:
- `TestChunkSliceMethod::test_chunk_slice_computes_correct_slices` - Expected shape `(10, 5, 25)` but got `(10, 5, 100)`
- `TestChunkSliceMethod::test_chunk_slice_handles_final_chunk_dynamically` - Expected shape `(10, 5, 25)` but got `(10, 5, 105)`
- `TestChunkSliceMethod::test_chunk_slice_none_parameters_returns_full_array` - `num_chunks` is 1 instead of None
- `TestChunkSliceMethod::test_chunk_slice_single_chunk` - Did not raise ValueError as expected
- `TestChunkSliceMethod::test_chunk_slice_validates_chunk_index` - Did not raise ValueError as expected
- `TestChunkSliceMethod::test_chunk_slice_different_axis_indices` - `_chunk_axis_index` is None instead of 0
- `TestChunkMetadataFlow::test_chunk_slice_uses_chunk_metadata_from_response` - Expected shape `(10, 5, 25)` but got `(10, 5, 100)`
- `TestChunkMetadataFlow::test_chunk_metadata_flow_integration` - Expected shape `(10, 5, 25)` but got `(10, 5, 100)`
- `test_managed_array.py::test_managed_array_chunk_fields_default_none` - `num_chunks` is 1 instead of None
- `test_chunk_axis_index_in_array_requests` - `_chunk_axis_index` is 1 instead of 2
- `TestWritebackWatcher::test_2d_array_slice` - Array shape mismatch `(0, 10)` vs `(5, 10)`

**Root Cause**: The chunk metadata (chunk_length, num_chunks, chunk_axis_index) from `ArrayResponse` is not being properly stored in `ManagedArray` objects. The `chunk_slice()` method cannot compute correct slices without this metadata.

**Likely Issues**:
1. `on_allocation_complete()` may not be extracting metadata from `ArrayResponse.chunked_shapes`
2. `ManagedArray` chunk fields might not be initialized properly
3. Default chunk parameters (when no chunking) should be `None`, not `1`

---

#### Category 4: Method API Changes (2 failures)

**Failed Tests**:
- `test_batchinputarrays.py::TestInputArrays::test_initialise_method` - `TypeError: chunk_index must be int, got slice`
- `test_batchinputarrays.py::TestInputArrays::test_call_method_size_change_triggers_reallocation` - Shape mismatch `(3, 1)` vs `(3, 7)`

**Root Cause**: Tests are calling methods with incorrect signatures after refactoring. The `chunk_slice()` method now expects an integer chunk_index, not a slice object.

---

#### Category 5: Missing Attributes (1 failure)

**Failed Test**:
- `TestGetTotalRuns::test_allocate_passes_total_runs_to_request` - `AttributeError: 'MemoryManager' object has no attribute 'queue'`

**Root Cause**: Test is accessing internal `MemoryManager.queue` attribute that doesn't exist. This is a test implementation issue.

---

#### Category 6: Test Infrastructure Issues (1 failure)

**Failed Test**:
- `TestWritebackTask::test_task_creation` - Buffer array identity check failed

**Root Cause**: Test is using `is` operator instead of `np.array_equal()` for array comparison.

---

## ⚠️ RunParams Tests

**Status**: 6 failed

### Failures in test_runparams.py (3 failures)

**Error Pattern**: `chunk_length` is 1 instead of expected value

**Failed Tests**:
- `test_runparams_update_from_allocation` - Expected `chunk_length=25`, got `1`
- `test_runparams_update_from_allocation_single_chunk` - Expected `chunk_length=50`, got `1`
- `test_runparams_update_from_allocation_dangling_chunk` - Expected `chunk_length=34`, got `1`

**Root Cause**: `RunParams.update_from_allocation()` is not extracting `chunk_length` from the `ArrayResponse`. The chunk metadata flow is broken.

---

### Failures in test_runparams_integration.py (3 failures)

**Error**: `AttributeError: 'SymbolicODE' object has no attribute 'num_params'`

**Failed Tests**:
- `test_runparams_single_chunk`
- `test_runparams_multiple_chunks`
- `test_runparams_exact_division`

**Root Cause**: Code is calling `ode_system.num_params` but the correct attribute is `num_parameters`. This is a simple typo/naming issue in the test or implementation.

---

## Summary of Issues

### Critical Issues (Blocking)

1. **Missing `total_runs` in test ArrayRequests** (10 failures)
   - Tests need to specify `total_runs` when creating ArrayRequest objects
   - OR implementation needs backward compatibility for None values

2. **Inconsistent `total_runs` validation** (8 errors)
   - Arrays that don't chunk (like driver_coefficients) should use `total_runs=None`
   - Current implementation passes different integer values causing validation errors

3. **Chunk metadata not propagating** (11 failures)
   - `ArrayResponse.chunked_shapes` data not being extracted properly
   - `ManagedArray` chunk fields not initialized correctly
   - `chunk_slice()` cannot work without proper metadata

### Medium Priority Issues

4. **RunParams chunk_length not updating** (3 failures)
   - `RunParams.update_from_allocation()` not extracting chunk_length from response

5. **Attribute naming mismatch** (3 failures)
   - Code uses `num_params` but should use `num_parameters`

### Low Priority Issues

6. **Test API mismatches** (3 failures)
   - Tests calling methods with wrong arguments after refactoring
   - Tests accessing non-existent internal attributes
   - Test using wrong comparison operators

---

## Recommendations

### Immediate Actions Required

1. **Fix `total_runs` propagation in `BaseArrayManager.allocate()`**
   - Ensure driver_coefficients and other non-chunked arrays get `total_runs=None`
   - Only chunkable arrays should get the actual runs value
   - Review `_get_total_runs()` implementation

2. **Fix chunk metadata extraction in `on_allocation_complete()`**
   - Extract chunked_shapes from ArrayResponse
   - Store chunk_length, num_chunks, chunk_axis_index in ManagedArray
   - Ensure defaults are None (not 1) when no chunking

3. **Update RunParams to extract chunk_length**
   - Modify `RunParams.update_from_allocation()` to read chunk_length from response

4. **Fix attribute naming**
   - Change `num_params` to `num_parameters` throughout

5. **Update failing tests**
   - Add `total_runs` parameter to ArrayRequest creation in tests
   - Fix method call signatures to match new API
   - Fix comparison operators in test assertions

### Testing Strategy

After fixes:
1. Re-run memory tests - should go from 37/44 to 44/44 passing
2. Re-run batchsolving arrays tests - should go from 147/161 to 161/161 passing
3. Re-run runparams tests - all should pass

---

## Test Commands Used

```bash
# Memory subsystem
NUMBA_ENABLE_CUDASIM=1 pytest tests/memory/test_array_requests.py -v -m "not nocudasim and not cupy"
NUMBA_ENABLE_CUDASIM=1 pytest tests/memory/test_memmgmt.py -v -m "not nocudasim and not cupy"

# Batchsolving arrays subsystem
NUMBA_ENABLE_CUDASIM=1 pytest tests/batchsolving/arrays/test_basearraymanager.py -v -m "not nocudasim and not cupy"
NUMBA_ENABLE_CUDASIM=1 pytest tests/batchsolving/arrays/test_batchinputarrays.py -v -m "not nocudasim and not cupy"
NUMBA_ENABLE_CUDASIM=1 pytest tests/batchsolving/arrays/test_batchoutputarrays.py -v -m "not nocudasim and not cupy"
NUMBA_ENABLE_CUDASIM=1 pytest tests/batchsolving/arrays/test_chunking.py -v -m "not nocudasim and not cupy"

# RunParams subsystem
NUMBA_ENABLE_CUDASIM=1 pytest tests/batchsolving/test_runparams.py -v -m "not nocudasim and not cupy"
NUMBA_ENABLE_CUDASIM=1 pytest tests/batchsolving/test_runparams_integration.py -v -m "not nocudasim and not cupy"
```
