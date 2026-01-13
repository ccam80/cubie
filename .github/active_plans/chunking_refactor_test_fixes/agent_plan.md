# Chunking Refactor Test Fixes - Agent Plan

## Overview

This plan addresses 87 failing tests across 13 categories. The failures stem from a major refactoring of CuBIE's chunking and array management system. Fixes are organized by root cause category and prioritized by impact (number of tests resolved per fix).

---

## Category 1: IndexError - Array Index Out of Bounds in Integration Kernel

**Affected Tests**: 31 (16 failures + 15 errors)  
**Location**: `src/cubie/batchsolving/BatchSolverKernel.py:791`  
**Root Cause Type**: Production Bug

### Problem Description
The integration kernel accesses `inits[:, run_index]` where `run_index` is the global thread index. After chunking, the device arrays are sized for one chunk, but `run_index` remains the global index across all chunks.

### Expected Behavior
When arrays are chunked:
- `run_index` should be relative to the current chunk (0 to chunk_size-1)
- The kernel should receive chunk-relative indices, not global batch indices

### Architectural Context
The kernel is launched per-chunk by `BatchSolverKernel.run()`. The chunk loop should pass the chunk offset to the kernel or the kernel should compute chunk-local indices.

### Integration Points
- `BatchSolverKernel.build()` generates the kernel
- `BatchSolverKernel.run()` launches kernels per-chunk
- `ChunkParams` provides chunk metadata (size, offset, etc.)

### Data Structures
- `ChunkParams`: Contains `runs_in_chunk`, `chunk_offset`, `chunk_index`
- The kernel receives `run_index = ty` (thread y index) but should also receive chunk offset

### Edge Cases
- Final chunk may have fewer runs than `runs_per_chunk`
- Non-chunked execution (chunks=1) should work unchanged

---

## Category 2: ValueError - Incompatible Shape in copy_to_host

**Affected Tests**: 11 (7 failures + 4 errors)  
**Location**: `src/cubie/memory/mem_manager.py:1139`  
**Root Cause Type**: Production Bug

### Problem Description
Device arrays have chunked shapes (e.g., `(10, 5, 25)`) but host arrays retain full shapes (e.g., `(10, 5, 100)`). The `from_device` method attempts direct copy without slicing.

### Expected Behavior
When `needs_chunked_transfer` is True:
- Device array is copied to a pinned staging buffer
- Staging buffer is copied to the appropriate slice of the host array
- This is already implemented in `OutputArrays.finalise()` but may have issues

### Architectural Context
The `BaseArrayManager.from_device()` delegates to `MemoryManager.from_device()` which does direct copy. Chunked transfers need the slice logic in `OutputArrays.finalise()`.

### Integration Points
- `MemoryManager.from_device()` - low-level copy
- `OutputArrays.finalise()` - should handle chunked slicing
- `ManagedArray.chunked_slice_fn` - provides slice for chunk index

### Dependencies
Depends on Category 1 fix (correct kernel execution) to produce valid data to copy.

---

## Category 3: ValueError - Can't Fit Single Run in GPU VRAM

**Affected Tests**: 6  
**Location**: `src/cubie/memory/mem_manager.py:1308`  
**Root Cause Type**: Production Bug

### Problem Description
Memory estimation calculates request sizes incorrectly, causing the "Can't fit a single run" error even when memory should be sufficient.

### Expected Behavior
The `get_chunk_parameters` method should:
- Correctly sum unchunkable and chunkable memory requirements
- Account for the per-run memory correctly
- Calculate a valid `max_chunk_size > 0` when memory is available

### Architectural Context
The `MockMemoryManager` in tests returns `(4096, 8192)` for `(free, total)`. The test arrays should fit but the calculation may be using wrong units or accumulating incorrectly.

### Data Structures
- `ArrayRequest.shape` and `ArrayRequest.dtype` determine per-array memory
- `chunkable_size` vs `unchunkable_size` partitioning

### Edge Cases
- All arrays unchunkable
- Mixed chunkable/unchunkable arrays
- Very small memory limits

---

## Category 4: ValueError - chunk_axis not in stride_order tuple

**Affected Tests**: 4 (3 failures + 1 from related category)  
**Location**: `src/cubie/memory/mem_manager.py:1397`  
**Root Cause Type**: Production Bug

### Problem Description
`compute_per_chunk_slice` calls `request.stride_order.index(chunk_axis)` without checking if `chunk_axis` exists in `stride_order`.

### Expected Behavior
Arrays that don't have the chunk axis in their stride order should:
- Be treated as unchunkable
- Return a full-slice function `(slice(None),) * ndim`

### Architectural Context
This affects arrays like `status_codes` with stride_order `("run",)` when chunking on "time" axis.

### Integration Points
- `compute_per_chunk_slice()` - needs guard before `.index()` call
- `is_request_chunkable()` - should be checked first

---

## Category 5: Device Arrays Returning None

**Affected Tests**: 18  
**Root Cause Type**: Test Update Required

### Problem Description
Tests call `allocate()` on array managers but don't call `allocate_queue()` to process the queued allocation requests.

### Expected Behavior
After `manager.allocate()`, tests must also call:
```python
memory_manager.allocate_queue(manager, chunk_axis="run")
```

### Test Files Affected
- `tests/batchsolving/arrays/test_batchinputarrays.py`
- `tests/batchsolving/arrays/test_batchoutputarrays.py`

### Fix Pattern
For each test that creates an array manager and expects device arrays:
1. Locate the `allocate()` call (may be inside `update()`)
2. Add `memory_manager.allocate_queue(manager)` after the call
3. Ensure the test's memory manager is accessible

---

## Category 6: Buffer Pool / Active Buffers Empty

**Affected Tests**: 5  
**Root Cause Type**: Test Update Required

### Problem Description
Tests expect `_active_buffers` to be populated after `initialise()` but the chunked transfer path requires `chunked_shape` and `chunked_slice_fn` to be set on managed arrays.

### Expected Behavior
For chunked mode tests:
1. `chunked_shape` must differ from `shape` for `needs_chunked_transfer` to be True
2. `chunked_slice_fn` must be set to provide slice tuples

### Test Files Affected
- `tests/batchsolving/arrays/test_batchinputarrays.py`
- `tests/batchsolving/test_pinned_memory_refactor.py`

### Fix Pattern
Before calling `initialise()`:
```python
for name, device_slot in manager.device.iter_managed_arrays():
    device_slot.chunked_shape = <smaller than shape>
    device_slot.chunked_slice_fn = lambda i: <appropriate slice>
```

---

## Category 7: TypeError - NoneType Issues

**Affected Tests**: 5  
**Root Cause Type**: Test Update Required

### Problem Description
Operations performed on None arrays due to allocation not completing before array access.

### Root Causes
1. `allocate_queue()` not called after `allocate()`
2. Allocation response doesn't include all expected arrays
3. Test fixtures don't set up arrays correctly

### Test Files Affected
- `tests/batchsolving/arrays/test_basearraymanager.py`
- `tests/batchsolving/arrays/test_batchinputarrays.py`
- `tests/batchsolving/arrays/test_batchoutputarrays.py`

### Fix Pattern
Same as Category 5 - ensure allocation queue is processed.

---

## Category 8: AttributeError - copy_to_host on None

**Affected Tests**: 3  
**Root Cause Type**: Test Update Required

### Problem Description
Tests call methods that attempt `copy_to_host()` on device arrays that are None.

### Root Cause
Same as Categories 5 and 7 - allocation queue not processed.

### Test Files Affected
- `tests/batchsolving/arrays/test_batchoutputarrays.py`
- `tests/batchsolving/test_pinned_memory_refactor.py`

---

## Category 9: Chunked Shape / needs_chunked_transfer Issues

**Affected Tests**: 2  
**Root Cause Type**: Test Update Required

### Problem Description
`needs_chunked_transfer` returns False because `chunked_shape` is not set, or is equal to `shape`.

### Expected Behavior
After allocation with chunking:
- `chunked_shape` should be set on all device ManagedArrays
- `chunked_shape != shape` for arrays that are actually chunked

### Architectural Context
`_on_allocation_complete` should propagate `chunked_shapes` from `ArrayResponse` to `ManagedArray` objects.

### Test Files Affected
- `tests/batchsolving/arrays/test_basearraymanager.py`

---

## Category 10: ensure_nonzero_size Behavior Change

**Affected Tests**: 2  
**Root Cause Type**: Code Review Required

### Problem Description
Tests expect `ensure_nonzero_size((1, 2, 0))` to return `(1, 2, 1)` but a different value is returned.

### Expected Behavior
Per the docstring:
- `ensure_nonzero_size((0, 2, 0))` → `(1, 2, 1)`
- `ensure_nonzero_size((1, 2, 0))` → `(1, 2, 1)`

### Current Behavior Investigation Needed
Need to run the function to see what it actually returns. The function looks correct in source, so this may be a test assertion issue.

### Test Files Affected
- `tests/test_utils.py`
- `tests/outputhandling/test_output_sizes.py`

---

## Category 11: Memory Manager Shape Calculation

**Affected Tests**: 1  
**Location**: `tests/memory/test_memmgmt.py`  
**Root Cause Type**: Production Bug or Test Update

### Problem Description
Memory manager processes a request with shape `(4, 4, 4)` but produces `(4, 4, 8)`.

### Investigation Needed
Examine the test to understand what operation is being tested and whether the expected shape needs updating or if the code has a bug.

---

## Category 12: Conditional Memory Conversion Issue

**Affected Tests**: 1  
**Location**: `tests/batchsolving/arrays/test_conditional_memory.py`  
**Root Cause Type**: Test Update or Production Bug

### Problem Description
Output arrays don't convert from pinned to host memory when chunked.

### Expected Behavior
Per `OutputArrays._on_allocation_complete()`:
- If `is_chunked` is True, call `_convert_host_to_numpy()`
- Host arrays should have `memory_type = "host"` not `"pinned"`

### Investigation Needed
Check if `_on_allocation_complete` is being called and if `is_chunked` is True.

---

## Category 13: Nonzero Functionality (driver_coefficients None)

**Affected Tests**: 1  
**Location**: `tests/outputhandling/test_output_sizes.py`  
**Root Cause Type**: Production Bug or Test Update

### Problem Description
`driver_coefficients` property returns None instead of expected value.

### Investigation Needed
Check if this is related to the `ensure_nonzero_size` changes or a separate issue with output sizing.

---

## Implementation Order

### Phase 1: Critical Production Bugs (Categories 1, 2, 3, 4)
These affect core functionality. Fix in this order:
1. Category 4 (chunk_axis not in tuple) - simple guard fix
2. Category 3 (memory calculation) - may be unit issue
3. Category 1 (kernel indexing) - core chunking bug
4. Category 2 (shape incompatibility) - depends on category 1

### Phase 2: Test Infrastructure Updates (Categories 5, 6, 7, 8, 9)
These are test-only fixes. Pattern:
1. Add `allocate_queue()` calls after `allocate()`
2. Set up `chunked_shape` and `chunked_slice_fn` for chunked tests
3. Update fixtures to use new factory patterns

### Phase 3: Cleanup and Edge Cases (Categories 10, 11, 12, 13)
Individual investigation needed for each.

---

## Dependencies Between Categories

```
Category 1 (Kernel Index) ← Category 2 (Shape Copy) depends on
Category 4 (chunk_axis guard) ← Category 3 (memory calc) may depend on
Category 5-9 (Test Updates) ← All depend on Phase 1 fixes
```

---

## Success Criteria

1. All 87 failing tests pass
2. No new test failures introduced
3. Full solver integration tests pass (test_solver.py, test_chunked_solver.py)
4. Memory management tests pass (test_memmgmt.py, test_pinned_memory_refactor.py)
