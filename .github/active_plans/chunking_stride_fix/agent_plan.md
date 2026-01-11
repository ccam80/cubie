# Chunking and Stride Fix - Agent Plan

## Overview

This plan addresses two bugs in the chunking subsystem and removes the
custom striding system to simplify the codebase.

### Bug 1: Run-axis Chunking Stride Mismatch

**Location**: `BatchOutputArrays.finalise()` → `mem_manager.from_device()`

**Symptom**: `ValueError: incompatible strides: (402653184, 134217728, 4)
vs. (805306368, 268435456, 4)`

**Root Cause**: When chunking on the run axis, the host array is larger
than the device array. The host slice (a view) has strides based on the
full host array, while the device array has strides based on its smaller
chunked shape. Numba requires matching strides for `copy_to_host`.

### Bug 2: Time-axis Chunking tuple.index Error

**Location**: `mem_manager.chunk_arrays()`

**Symptom**: `ValueError: tuple.index(x): x not in tuple`

**Root Cause**: Input arrays have `stride_order=("variable", "run")` with
no "time" axis. When `chunk_arrays` tries to find the time axis index in
these 2D arrays, it fails.

---

## Component Architecture Changes

### MemoryManager Changes

The `MemoryManager` class currently maintains a `_stride_order` attribute
and provides methods for custom striding. These will be removed:

**Attributes to Remove**:
- `_stride_order: tuple[str, str, str]`

**Methods to Remove**:
- `set_global_stride_ordering(ordering)`: Sets custom stride order
- `get_strides(request)`: Calculates custom strides for allocations

**Methods to Modify**:
- `create_host_array()`: Simplify to create C-contiguous arrays only
- `allocate()`: Remove strides parameter from cuda.device_array calls
- `allocate_all()`: Remove get_strides call
- `chunk_arrays()`: Add defensive check for missing chunk axis

### BaseArrayManager Changes

**Methods to Remove**:
- `_convert_to_device_strides()`: Converts arrays to match device strides

**Methods to Modify**:
- `_update_host_array()`: Remove stride conversion call

### BatchOutputArrays Changes

**Methods to Modify**:
- `finalise()`: Ensure host slices are made contiguous before copying.
  The slice of a larger host array is a view with the parent's strides.
  We need to either:
  a) Create contiguous copy of slice before passing to from_device, OR
  b) Modify from_device to handle this case

**Recommended approach**: Modify `finalise()` to ensure contiguous slices.

### BatchInputArrays Changes

**Methods to Modify**:
- `initialise()`: Similar to finalise - ensure slices are contiguous
  before host-to-device transfers

### ArrayRequest Changes

**Attributes to Remove**:
- `stride_order`: No longer needed if all arrays use default strides

**Note**: `stride_order` is still used to identify which axis corresponds
to "run", "time", "variable" for chunking purposes. It should remain as
metadata but not drive allocation strides.

---

## Integration Points

### ManagedArray

The `ManagedArray` class stores `stride_order` which describes the logical
meaning of each dimension. This should remain for documentation and
chunking logic, but should not affect physical memory layout.

### OutputArrayContainer / InputArrayContainer

These containers define default `stride_order` values for each array type:
- 3D output arrays: `("time", "variable", "run")`
- 2D input arrays: `("variable", "run")`

These remain unchanged as they describe logical structure, not physical layout.

---

## Expected Behavior After Changes

### Allocation Flow

1. `ArrayRequest` specifies shape, dtype, memory type
2. `MemoryManager.allocate()` creates C-contiguous device array
3. `MemoryManager.create_host_array()` creates C-contiguous host array
4. Arrays have matching strides when shapes match

### Chunking Flow

1. `chunk_arrays()` checks if array has the chunk axis in its stride_order
2. If axis not present, array is left unchanged (skip chunking)
3. If axis present, shape is divided along that axis
4. Allocations proceed with reduced shapes

### Transfer Flow (Run-axis Chunking)

1. Host array has shape `(time, variable, n_runs)`
2. Device array has shape `(time, variable, n_runs / chunks)`
3. For chunk `i`, host slice is `host[:, :, i*chunk_size:(i+1)*chunk_size]`
4. Slice is made contiguous via `.copy()` before transfer
5. `copy_to_host(slice.copy())` or `to_device(slice.copy())` succeeds

### Transfer Flow (Time-axis Chunking)

1. Host output array has shape `(time, variable, n_runs)`
2. Device array has shape `(time / chunks, variable, n_runs)`
3. Host input array has shape `(variable, n_runs)` - NOT chunked on time
4. For chunk `i`, output host slice is `host[i*chunk_size:(i+1)*chunk_size, :, :]`
5. Input arrays are copied in full (no time axis to slice)

---

## Data Structures

### ArrayRequest (Modified)

```python
@attrs.define
class ArrayRequest:
    dtype: type
    shape: tuple[int, ...]
    memory: str  # "device", "mapped", "pinned", "managed"
    stride_order: Optional[tuple[str, ...]]  # Logical dimension labels, kept for chunking
    unchunkable: bool  # Whether to skip chunking
```

**Change**: `stride_order` is kept but only used for chunk axis lookup, not
for stride calculation.

### MemoryManager (Modified)

Remove:
```python
_stride_order: tuple[str, str, str]  # REMOVED
```

### ManagedArray

No changes - continues to store logical stride_order for array metadata.

---

## Edge Cases

### Empty Chunk Axis

When `chunk_arrays()` encounters an array without the chunk axis in its
`stride_order`, it should skip that array. Example:
- Chunking on "time" with input array `stride_order=("variable", "run")`
- Input array has no "time" axis
- Skip chunking for this array

### Single Chunk

When `chunks == 1`, the code should behave identically to non-chunked
execution. Host and device arrays have matching shapes and strides.

### Unchunkable Arrays

Arrays with `is_chunked=False` (e.g., status_codes, driver_coefficients)
should be transferred in full regardless of chunk index.

### Zero-sized Dimensions

Arrays with `0` in any dimension should be handled gracefully. These can
occur when outputs are disabled (e.g., no observables).

---

## Dependencies

### Internal Dependencies

- `cubie.cuda_simsafe`: CUDA simulation compatibility layer
- `cubie._utils`: Precision handling, slice helpers
- `cubie.memory.stream_groups`: Stream management
- `cubie.outputhandling.output_sizes`: Array sizing calculations

### External Dependencies

- `numba.cuda`: CUDA array operations
- `numpy`: Array creation and manipulation
- `attrs`: Data class definitions

---

## Test Strategy

### Test 1: Chunked Run-axis Transfer

Create a test that:
1. Creates output arrays with a known pattern
2. Forces chunking by setting chunk count > 1
3. Calls `finalise()` for each chunk
4. Verifies host array contains expected values in correct positions

### Test 2: Chunked Time-axis Transfer

Create a test that:
1. Creates output arrays with time dimension
2. Forces time-axis chunking
3. Verifies input arrays (no time axis) are handled correctly
4. Verifies output arrays are correctly populated

### Test 3: Full Integration (Numerical Correctness)

Create a test that:
1. Runs a simple ODE with known solution
2. Forces chunking (mock VRAM limit or directly set chunks)
3. Compares chunked results to non-chunked results
4. Results should match within floating-point tolerance

### Test Environment

- Tests should run in CUDASIM mode for CI compatibility
- Use small array sizes that fit in memory but simulate chunking
- Directly manipulate `_chunks` attribute to force chunked code paths

---

## Implementation Order

1. **Fix chunk_arrays()**: Add defensive check for missing axis
2. **Fix finalise()/initialise()**: Make slices contiguous before transfer
3. **Remove custom striding from MemoryManager**: Clean up unused code
4. **Remove _convert_to_device_strides()**: Clean up BaseArrayManager
5. **Add regression tests**: Verify fixes work
6. **Remove set_stride_order() from BatchSolverKernel**: API cleanup

---

## Validation Criteria

### Functional

- [ ] Run-axis chunked transfers complete without stride errors
- [ ] Time-axis chunked transfers complete without tuple.index errors
- [ ] Chunked runs produce same numerical results as non-chunked
- [ ] All existing tests continue to pass

### Code Quality

- [ ] No unused striding code remains
- [ ] API surface is simplified
- [ ] New tests provide coverage for chunking scenarios
