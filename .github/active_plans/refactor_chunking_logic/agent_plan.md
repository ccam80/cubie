# Refactor Chunking Logic - Agent Plan

## Overview

This refactoring removes `stride_order` usage and the `compute_per_chunk_slice()` function from the memory manager. ManagedArray objects will compute their own per-chunk slices using their existing `_chunk_axis_index` field and simple chunk parameters (`chunks`, `chunk_length`) from ArrayResponse.

**Context After Merge:**
- `RunParams` class now handles chunking metadata (replaced `ChunkParams`)
- Memory manager gets `num_runs` from `triggering_instance.run_params.runs`
- `ArrayResponse` has `chunks` and `chunk_length` fields (no `axis_length` or `dangling_chunk_length`)
- Fallback logic exists but must be removed per review feedback

## Component Descriptions

### ManagedArray (src/cubie/batchsolving/arrays/BaseArrayManager.py)

**Current State:**
- Has `_chunk_axis_index` field set in `__attrs_post_init__` from stride_order.index("run")
- Has `chunked_slice_fn: Optional[Callable]` field storing closure from memory manager
- Has `chunk_slice(runslice: slice)` method that takes a pre-computed slice object and returns sliced array

**Expected Behavior After Refactoring:**
- Add `chunk_length: Optional[int]` and `num_chunks: Optional[int]` fields
- Remove `chunked_slice_fn` field entirely
- Change `chunk_slice()` signature to `chunk_slice(chunk_index: int) -> tuple[slice, ...]`
- Compute slice tuple using:
  - `_chunk_axis_index` to know which dimension (already set)
  - `chunk_length` to compute start/end: `start = chunk_index * chunk_length`
  - `num_chunks` to detect last chunk
  - For last chunk (chunk_index == num_chunks - 1), use `end = self.shape[_chunk_axis_index]`
  - For other chunks, use `end = start + chunk_length`
- Return tuple of slices, not the sliced array

**Edge Cases:**
- When `_chunk_axis_index` is None: return full-array slice tuple (all `slice(None)`)
- When `needs_chunked_transfer` is False: return full-array slice tuple
- Single chunk case (num_chunks=1): compute normally, will return full array slice
- Last chunk smaller than chunk_length: handled by using actual array dimension for end

### ArrayRequest (src/cubie/memory/array_requests.py)

**Current State:**
- Has `stride_order: Optional[tuple[str, ...]]` field
- `__attrs_post_init__()` sets default stride_order if None (line 82-83)
- Memory manager accesses `request.stride_order.index("run")` in compute_per_chunk_slice

**Expected Behavior After Refactoring:**
- **Option 1 (Preferred)**: Remove `stride_order` field entirely
- **Option 2**: Keep field but remove from __attrs_post_init__ default setting, make truly optional
- Memory manager must NOT access stride_order at all
- ManagedArray retains its own stride_order field (separate from ArrayRequest)

**Edge Cases:**
- If stride_order kept for compatibility, it must be ignored by memory manager
- No code should depend on ArrayRequest.stride_order for chunking

### ArrayResponse (src/cubie/memory/array_requests.py)

**Current State:**
- Has `chunked_slices: dict[str, Callable]` field (line 127-129)
- Has `chunks: int` and `chunk_length: int` fields
- Created in allocate_queue with chunked_slices populated from compute_per_chunk_slice

**Expected Behavior After Refactoring:**
- Remove `chunked_slices` field entirely
- Keep `arr`, `chunks`, `chunk_length`, `chunked_shapes` fields
- No callable/closure storage needed

### MemoryManager (src/cubie/memory/mem_manager.py)

**Current State:**
- `allocate_queue()` at line 1170:
  - Lines 1204-1212: Gets num_runs from triggering_instance.run_params.runs with fallback
  - Line 1224-1229: Calls `compute_per_chunk_slice()` and stores result in ArrayResponse
  - Line 1243: Passes chunked_slices to ArrayResponse
- `compute_per_chunk_slice()` function at lines 1382-1437:
  - Accesses `request.stride_order.index("run")` to find chunk axis
  - Generates closure functions that capture stride_order and chunk_index
  - Returns dict of callables

**Expected Behavior After Refactoring:**
- **Remove entire function**: `compute_per_chunk_slice()` (lines 1382-1437)
- **Modify allocate_queue()**:
  - Lines 1206-1212: **Remove fallback logic** - fail if run_params not available
  - Line 1204-1205: Keep getting num_runs from triggering_instance.run_params.runs
  - Lines 1224-1229: **Remove call to compute_per_chunk_slice()**
  - Line 1243: **Remove chunked_slices from ArrayResponse construction**
- **Keep unchanged**:
  - `get_chunk_parameters()` - computes chunk_length and num_chunks
  - `compute_chunked_shapes()` - computes per-chunk shapes
- **No access to stride_order** anywhere in memory manager

**Functions to Modify:**
- `allocate_queue()`: Remove lines 1206-1212 (fallback), 1224-1229 (slice generation), update line 1243
- Remove: `compute_per_chunk_slice()` entirely

**Edge Cases:**
- If triggering_instance lacks run_params attribute: raise AttributeError immediately (no fallback)
- Unchunkable arrays: still get chunks=1, chunk_length=<full size> in response

### BaseArrayManager (src/cubie/batchsolving/arrays/BaseArrayManager.py)

**Current State:**
- `_on_allocation_complete()` callback at ~line 318:
  - Stores `chunked_shapes[array_label]` in ManagedArray.chunked_shape
  - Stores `chunked_slices[array_label]` in ManagedArray.chunked_slice_fn

**Expected Behavior After Refactoring:**
- Store `response.chunks` in ManagedArray.num_chunks
- Store `response.chunk_length` in ManagedArray.chunk_length
- Store `chunked_shapes[array_label]` in ManagedArray.chunked_shape (unchanged)
- **Do not store** chunked_slice_fn (field will be removed from ManagedArray)
- Apply to both device and host containers

**Code Change:**
```python
for array_label in self._needs_reallocation:
    self.device.attach(array_label, arrays[array_label])
    if array_label in response.chunked_shapes:
        for container in (self.device, self.host):
            array = container.get_managed_array(array_label)
            array.chunked_shape = chunked_shapes[array_label]
            array.chunk_length = response.chunk_length
            array.num_chunks = response.chunks
```

**Edge Cases:**
- Arrays not in chunked_shapes: don't set chunk parameters
- Single chunk (chunks=1): still set parameters, ManagedArray will handle correctly

### InputArrays (src/cubie/batchsolving/arrays/BatchInputArrays.py)

**Current State:**
- `initialise(chunk_index: int)` at line 275
- Line 314: `slice_tuple = host_obj.chunked_slice_fn(chunk_index)`
- Uses slice_tuple to extract data: `host_slice = host_obj.array[slice_tuple]`

**Expected Behavior After Refactoring:**
- Change line 314 to: `slice_tuple = host_obj.chunk_slice(chunk_index)`
- ManagedArray.chunk_slice() now returns the tuple of slices (not the sliced array)
- Rest of logic unchanged

**Integration Points:**
- chunk_index comes from BatchSolverKernel.run() loop
- Chunk parameters already stored in ManagedArray from _on_allocation_complete

### OutputArrays (src/cubie/batchsolving/arrays/BatchOutputArrays.py)

**Current State:**
- `finalise(chunk_index: int)` at line 344
- Line 378: `slice_tuple = slot.chunked_slice_fn(chunk_index)`
- Uses slice_tuple to determine writeback location: `host_slice = host_array[slice_tuple]`

**Expected Behavior After Refactoring:**
- Change line 378 to: `slice_tuple = slot.chunk_slice(chunk_index)`
- ManagedArray.chunk_slice() now returns the tuple of slices (not the sliced array)
- Rest of logic unchanged

**Integration Points:**
- chunk_index comes from BatchSolverKernel.run() loop
- Chunk parameters already stored in ManagedArray from _on_allocation_complete

### BatchSolverKernel (src/cubie/batchsolving/BatchSolverKernel.py)

**Current State:**
- Has RunParams with runs, chunks, chunk_length fields
- `run()` method loops over chunks passing integer chunk_index to initialize/finalise
- Already compatible with target design

**Expected Behavior After Refactoring:**
- **No changes needed**
- Continues to pass integer chunk_index to array managers
- RunParams.update_from_allocation() updates chunk metadata from ArrayResponse

**Integration Points:**
- Array managers receive chunk parameters from ArrayResponse
- Kernel orchestrates loop, array managers handle slicing

## Architectural Changes

### Separation of Concerns
- **Before**: Memory manager generates slice functions that encode array structure knowledge
- **After**: Memory manager only computes allocation sizing, ManagedArray computes structure-aware slices

### Data Ownership
- **Before**: Slice logic generated in memory manager, stored as closure in ManagedArray
- **After**: Slice logic owned and executed by ManagedArray using simple parameters

### Coupling Reduction
- **Before**: ArrayRequest.stride_order → MemoryManager (tight coupling)
- **After**: No stride_order dependency in memory manager
- **Before**: MemoryManager → ManagedArray via closure in chunked_slice_fn
- **After**: MemoryManager → ManagedArray via simple integers (chunks, chunk_length)

## Expected Interactions Between Components

### Allocation Flow
```
1. InputArrays/OutputArrays create ArrayRequest (stride_order removed or unused)
2. MemoryManager.allocate_queue(triggering_instance) receives requests
3. MemoryManager:
   - Gets num_runs from triggering_instance.run_params.runs (NO fallback)
   - Calls get_chunk_parameters() → computes chunk_length, num_chunks
   - Calls compute_chunked_shapes() → computes per-chunk shapes
   - Does NOT call compute_per_chunk_slice() (function removed)
   - Returns ArrayResponse(chunks=num_chunks, chunk_length=chunk_length, chunked_shapes={...})
4. BaseArrayManager._on_allocation_complete():
   - Stores response.chunks in ManagedArray.num_chunks
   - Stores response.chunk_length in ManagedArray.chunk_length
   - Stores chunked_shapes in ManagedArray.chunked_shape
```

### Execution Flow (Per Chunk)
```
1. BatchSolverKernel.run() loops: for i in range(run_params.num_chunks)
2. Calls input_arrays.initialise(i) and output_arrays.initialise(i)
3. InputArrays.initialise(i):
   - For each managed array:
     - Call slice_tuple = managed_array.chunk_slice(i)
     - ManagedArray computes:
       * start = i * chunk_length
       * If i == num_chunks - 1: end = full_shape[_chunk_axis_index]
       * Else: end = start + chunk_length
       * Returns (slice(None), ..., slice(start, end), ...)
     - Use slice_tuple to extract chunk: host_slice = host_array[slice_tuple]
     - Transfer to device
4. Kernel executes
5. OutputArrays.finalise(i):
   - For each managed array:
     - Call slice_tuple = managed_array.chunk_slice(i)
     - ManagedArray computes slice (same as input)
     - Use slice_tuple to determine writeback location
     - Transfer from device
```

## Data Structures and Their Purposes

### ManagedArray New/Modified Fields
```python
# New fields to add:
chunk_length: Optional[int] = None  # Length of each chunk (except possibly last)
num_chunks: Optional[int] = None    # Total number of chunks

# Existing fields to keep:
_chunk_axis_index: Optional[int]    # Which axis to chunk (from stride_order.index("run"))
chunked_shape: Optional[tuple]      # Per-chunk shape

# Field to remove:
chunked_slice_fn: Optional[Callable]  # DELETE - no longer needed
```

**Purpose:** Enable ManagedArray to compute slices without external closures

### ManagedArray.chunk_slice() New Signature
```python
def chunk_slice(self, chunk_index: int) -> tuple[slice, ...]:
    """Compute per-chunk slice tuple for given chunk index.
    
    Parameters
    ----------
    chunk_index : int
        Zero-based chunk index.
    
    Returns
    -------
    tuple[slice, ...]
        Tuple of slice objects to index the full array for this chunk.
    """
    if self._chunk_axis_index is None or not self.needs_chunked_transfer:
        return tuple(slice(None) for _ in self.shape)
    
    start = chunk_index * self.chunk_length
    
    # Last chunk: use remaining elements
    if chunk_index == self.num_chunks - 1:
        end = self.shape[self._chunk_axis_index]
    else:
        end = start + self.chunk_length
    
    chunk_slice = [slice(None)] * len(self.shape)
    chunk_slice[self._chunk_axis_index] = slice(start, end)
    return tuple(chunk_slice)
```

### ArrayResponse Modified Structure
```python
@attrs.define
class ArrayResponse:
    arr: dict[str, DeviceNDArrayBase]  # Keep
    chunks: int  # Keep
    chunk_length: int  # Keep
    chunked_shapes: dict[str, tuple[int, ...]]  # Keep
    # chunked_slices: dict[str, Callable]  # REMOVE THIS FIELD
```

**Purpose:** Provide simple parameters rather than complex callables

## Dependencies and Imports

No new external dependencies required. Changes are internal refactoring.

**Import Changes:**
- ArrayResponse may remove `Callable` from type hints
- No new imports needed in ManagedArray (uses existing int, Optional, tuple types)

## Edge Cases to Consider

### Unchunkable Arrays
- Arrays with `is_chunked=False` or `needs_chunked_transfer=False`
- `chunk_slice()` returns full-array slice: `tuple(slice(None) for _ in shape)`
- Memory manager still returns chunks=1, chunk_length=full_size

### No Run Axis
- Arrays without "run" in stride_order have `_chunk_axis_index=None`
- `chunk_slice()` returns full-array slice regardless of chunk_index

### Single Chunk (No Chunking)
- When `num_chunks=1`, chunk_index=0 is the only valid index
- `chunk_slice(0)` computes: start=0, end=chunk_length=full_size
- Returns `slice(0, full_size)` which is equivalent to full array

### Last Chunk Smaller Than chunk_length
- When `chunk_index == num_chunks - 1`, use `end = self.shape[_chunk_axis_index]`
- Example: 1000 runs, chunk_length=300, num_chunks=4
  - Chunk 0: slice(0, 300) - 300 runs
  - Chunk 1: slice(300, 600) - 300 runs
  - Chunk 2: slice(600, 900) - 300 runs
  - Chunk 3: slice(900, 1000) - 100 runs (dangling chunk)

### Arrays with Different Shapes
- 2D arrays: (variable, run) → _chunk_axis_index=1
- 3D arrays: (time, variable, run) → _chunk_axis_index=2
- Each computes correct slice for its own dimensionality

### Missing run_params Attribute
- Memory manager must fail fast if `triggering_instance.run_params` doesn't exist
- No fallback to computing from array shapes
- AttributeError with clear message preferred

### chunk_slice() Return Value Changed
- **Old**: Returns sliced array (ndarray)
- **New**: Returns tuple of slice objects
- **Impact**: Calling code must use returned slice to index array
- **Compatibility**: InputArrays and OutputArrays already expect tuple (currently from closure)

## Implementation Order

1. **Add fields to ManagedArray** (src/cubie/batchsolving/arrays/BaseArrayManager.py)
   - Add `chunk_length: Optional[int] = None`
   - Add `num_chunks: Optional[int] = None`

2. **Modify ManagedArray.chunk_slice()** (src/cubie/batchsolving/arrays/BaseArrayManager.py)
   - Change signature to `chunk_slice(self, chunk_index: int) -> tuple[slice, ...]`
   - Implement slice computation logic using _chunk_axis_index, chunk_length, num_chunks
   - Return tuple of slices, not sliced array

3. **Update InputArrays.initialise()** (src/cubie/batchsolving/arrays/BatchInputArrays.py)
   - Line 314: Change to `slice_tuple = host_obj.chunk_slice(chunk_index)`
   - Verify slice_tuple is used correctly in line 315

4. **Update OutputArrays.finalise()** (src/cubie/batchsolving/arrays/BatchOutputArrays.py)
   - Line 378: Change to `slice_tuple = slot.chunk_slice(chunk_index)`
   - Verify slice_tuple is used correctly in line 379

5. **Update BaseArrayManager._on_allocation_complete()** (src/cubie/batchsolving/arrays/BaseArrayManager.py)
   - Store response.chunk_length in ManagedArray.chunk_length
   - Store response.chunks in ManagedArray.num_chunks
   - Remove storage of chunked_slice_fn

6. **Remove ArrayResponse.chunked_slices field** (src/cubie/memory/array_requests.py)
   - Delete lines 127-129

7. **Remove MemoryManager.compute_per_chunk_slice()** (src/cubie/memory/mem_manager.py)
   - Delete entire function at lines 1382-1437

8. **Update MemoryManager.allocate_queue()** (src/cubie/memory/mem_manager.py)
   - Delete lines 1206-1212 (fallback logic)
   - Delete lines 1224-1229 (compute_per_chunk_slice call)
   - Update line 1243: remove chunked_slices from ArrayResponse construction

9. **Remove ManagedArray.chunked_slice_fn field** (src/cubie/batchsolving/arrays/BaseArrayManager.py)
   - Delete lines 83-87

10. **Remove or update ArrayRequest.stride_order** (src/cubie/memory/array_requests.py)
    - Option 1: Delete field entirely
    - Option 2: Remove from __attrs_post_init__ default setting

11. **Update tests**
    - Remove tests for compute_per_chunk_slice() in test_memmgmt.py
    - Update ArrayResponse tests in test_array_requests.py
    - Update ManagedArray.chunk_slice() tests in test_basearraymanager.py
    - Review test_runparams.py and test_runparams_integration.py for fixture usage
    - Update test_chunking.py for new chunk_slice behavior

This order allows incremental testing and minimizes breaking changes.
