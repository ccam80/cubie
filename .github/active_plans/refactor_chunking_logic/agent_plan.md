# Refactor Chunking Logic - Agent Plan

## Overview

This refactoring removes array-indexing logic from the memory manager and localizes it in ManagedArray objects. The memory manager will return simple chunk parameters (axis_length, chunk_length, num_chunks, dangling_chunk_length), and ManagedArray objects will use these parameters along with their `_chunk_axis_index` field to compute per-chunk slices on demand.

## Component Descriptions

### ManagedArray (src/cubie/batchsolving/arrays/BaseArrayManager.py)

**Current State:**
- Has `_chunk_axis_index` field storing which axis to chunk (set from stride_order in `__attrs_post_init__`)
- Has `chunked_slice_fn` field storing a closure that generates slices
- Has `chunk_slice(runslice)` method that takes a pre-computed slice

**Expected Behavior After Refactoring:**
- Store chunk parameters (chunk_length, axis_length, num_chunks, dangling_chunk_length) as fields
- `chunk_slice(chunk_index: int)` method computes the slice tuple based on:
  - `_chunk_axis_index` to know which dimension to slice
  - `chunk_length` to compute start/end indices
  - `num_chunks` and `dangling_chunk_length` to handle the final chunk correctly
- No longer stores `chunked_slice_fn` - this field can be removed or deprecated
- `needs_chunked_transfer` property remains unchanged (compares full shape vs chunked shape)

**Edge Cases:**
- When `_chunk_axis_index` is None (no "run" axis), return full array regardless of chunk_index
- When `is_chunked` is False, return full array
- Final chunk may be shorter than chunk_length - use dangling_chunk_length

### ArrayRequest (src/cubie/memory/array_requests.py)

**Current State:**
- Has `stride_order` field used by memory manager for chunking logic
- `stride_order` is set in `__attrs_post_init__` if not provided

**Expected Behavior After Refactoring:**
- `stride_order` becomes optional (or is removed entirely)
- If kept, mark as deprecated and not used for chunking
- All chunking decisions based on presence of "run" in stride_order can be removed from memory manager
- ArrayRequest focuses on shape, dtype, memory type, and unchunkable flag

**Edge Cases:**
- Legacy code may still pass stride_order - handle gracefully if field is kept as deprecated
- Validation logic should not depend on stride_order structure

### ArrayResponse (src/cubie/memory/array_requests.py)

**Current State:**
- Contains `chunked_shapes` dict mapping array labels to per-chunk shapes
- Contains `chunked_slices` dict mapping array labels to slice-generating callables
- Contains `chunks` (number of chunks), `axis_length`, `chunk_length`, `dangling_chunk_length`

**Expected Behavior After Refactoring:**
- Keep `chunked_shapes` as before
- Remove `chunked_slices` field (or deprecate)
- Keep chunk parameters: `chunks`, `axis_length`, `chunk_length`, `dangling_chunk_length`
- These simple parameters are sufficient for ManagedArray to compute slices

### MemoryManager (src/cubie/memory/mem_manager.py)

**Current State:**
- `compute_per_chunk_slice()` function generates closure-based slice functions
- Uses `request.stride_order.index("run")` to find chunk axis
- Returns slice functions in ArrayResponse.chunked_slices
- `get_chunk_parameters()` computes chunk sizes and counts

**Expected Behavior After Refactoring:**
- Remove `compute_per_chunk_slice()` function entirely
- `allocate_queue()` should not populate `chunked_slices` in ArrayResponse
- Continue to compute and return chunk parameters (axis_length, chunk_length, num_chunks, dangling_chunk_length)
- Simplify logic that references `stride_order` in chunking calculations

**Functions to Modify:**
- `allocate_queue()`: Remove call to `compute_per_chunk_slice()`, remove population of `chunked_slices`
- `compute_per_chunk_slice()`: Remove entirely
- `is_request_chunkable()`: May need adjustment if stride_order is removed from ArrayRequest
- `get_chunk_axis_length()`: May need adjustment if stride_order is removed
- `replace_with_chunked_size()`: May need adjustment if stride_order is removed

**Edge Cases:**
- Handle requests where stride_order is not provided (if field becomes optional)
- Ensure backward compatibility during transition if stride_order is deprecated rather than removed

### BaseArrayManager (src/cubie/batchsolving/arrays/BaseArrayManager.py)

**Current State:**
- `_on_allocation_complete()` callback stores chunked_shapes and chunked_slice_fn from ArrayResponse
- Stores these in corresponding ManagedArray objects

**Expected Behavior After Refactoring:**
- Store chunk parameters (axis_length, chunk_length, num_chunks, dangling_chunk_length) in ManagedArray
- Do not store chunked_slice_fn (remove this logic)
- Chunk parameters should be available to all ManagedArray objects in the container

**Edge Cases:**
- Some arrays may not be chunkable - they should still receive chunk parameters but won't use them
- Ensure chunk parameters are accessible to both host and device ManagedArray objects

### InputArrays (src/cubie/batchsolving/arrays/BatchInputArrays.py)

**Current State:**
- `initialise(chunk_index: int)` method iterates arrays and calls `chunked_slice_fn(chunk_index)` to get slice
- Uses slice tuple to extract data from host array for transfer

**Expected Behavior After Refactoring:**
- `initialise(chunk_index: int)` calls `managed_array.chunk_slice(chunk_index)` instead
- `managed_array.chunk_slice()` computes and returns the slice tuple based on stored parameters
- Rest of logic remains the same (buffer pool acquisition, data transfer)

**Integration Points:**
- Must receive chunk parameters from allocation response (via BaseArrayManager)
- Chunk index comes from BatchSolverKernel loop

### OutputArrays (src/cubie/batchsolving/arrays/BatchOutputArrays.py)

**Current State:**
- `finalise(chunk_index: int)` method iterates arrays and calls `chunked_slice_fn(chunk_index)` to get slice
- Uses slice tuple to determine where to write data in host array

**Expected Behavior After Refactoring:**
- `finalise(chunk_index: int)` calls `managed_array.chunk_slice(chunk_index)` instead
- `managed_array.chunk_slice()` computes and returns the slice tuple based on stored parameters
- Rest of logic remains the same (buffer pool, writeback watcher)

**Integration Points:**
- Must receive chunk parameters from allocation response (via BaseArrayManager)
- Chunk index comes from BatchSolverKernel loop

### BatchSolverKernel (src/cubie/batchsolving/BatchSolverKernel.py)

**Current State:**
- `run()` method loops over chunks, calling `initialise(i)` and `finalise(i)` with chunk index
- Receives ArrayResponse with chunked metadata
- Stores chunk parameters in ChunkParams object

**Expected Behavior After Refactoring:**
- Continue passing simple chunk index `i` to `initialise()` and `finalise()`
- No changes needed to how it interacts with array managers
- ChunkParams handling remains the same

**Integration Points:**
- Array managers receive chunk parameters from MemoryManager via allocation response
- Kernel just orchestrates the loop and passes indices

## Architectural Changes

### Separation of Concerns
- **Before**: Memory manager knows about array structure (stride_order) and generates slicing logic
- **After**: Memory manager knows only about memory allocation. ManagedArray knows about array structure and slicing

### Data Ownership
- **Before**: Slice functions generated in memory manager, stored in ManagedArray
- **After**: Slice logic owned and executed by ManagedArray using its own metadata

### Coupling Reduction
- **Before**: ArrayRequest → MemoryManager (tight coupling via stride_order)
- **After**: ArrayRequest → MemoryManager (loose coupling, minimal metadata)
- **Before**: MemoryManager → ManagedArray (via closure in chunked_slice_fn)
- **After**: MemoryManager → ManagedArray (via simple parameters)

## Expected Interactions Between Components

### Allocation Flow
```
1. InputArrays/OutputArrays create ArrayRequest (no stride_order needed)
2. MemoryManager.allocate_queue() receives requests
3. MemoryManager computes chunk_length, num_chunks, axis_length, dangling_chunk_length
4. MemoryManager returns ArrayResponse with these parameters (no chunked_slices)
5. BaseArrayManager._on_allocation_complete() stores parameters in ManagedArray objects
```

### Execution Flow (Per Chunk)
```
1. BatchSolverKernel.run() loops: for i in range(chunks)
2. Calls input_arrays.initialise(i) and output_arrays.initialise(i)
3. InputArrays.initialise(i):
   - For each managed array:
     - Call managed_array.chunk_slice(i) → returns tuple of slice objects
     - Use slice to extract chunk from host array
     - Transfer to device
4. Kernel executes
5. OutputArrays.finalise(i):
   - For each managed array:
     - Call managed_array.chunk_slice(i) → returns tuple of slice objects
     - Use slice to determine where to write in host array
     - Transfer from device
```

## Data Structures and Their Purposes

### ManagedArray New Fields
```python
axis_length: Optional[int] = None  # Full length of run axis before chunking
chunk_length: Optional[int] = None  # Length of each chunk (except possibly last)
num_chunks: Optional[int] = None    # Total number of chunks
dangling_chunk_length: Optional[int] = None  # Length of final chunk if different
```

**Purpose:** Enable ManagedArray to compute slices without external closures

### ArrayResponse Modification
```python
# Remove:
chunked_slices: dict[str, Callable]  # No longer needed

# Keep:
chunks: int
axis_length: int
chunk_length: int
dangling_chunk_length: int
chunked_shapes: dict[str, tuple[int, ...]]
```

**Purpose:** Provide simple parameters rather than complex callables

## Dependencies and Imports

No new external dependencies required. Changes are internal refactoring.

**Potential Import Changes:**
- May remove `Callable` from type hints in ArrayResponse
- ManagedArray may need to import nothing new (just uses stored integers)

## Edge Cases to Consider

### Unchunkable Arrays
- Arrays with `is_chunked=False` should return full array regardless of chunk_index
- Memory manager still returns chunk parameters, but ManagedArray.chunk_slice() ignores them

### No Run Axis
- Arrays without "run" in stride_order have `_chunk_axis_index=None`
- `chunk_slice()` should return full array (slice(None) for all dimensions)

### Single Chunk (No Chunking)
- When `num_chunks=1`, slice should be full array
- `chunk_slice()` should handle this by computing slice(0, axis_length)

### Dangling Final Chunk
- Last chunk may be shorter than chunk_length
- When `chunk_index == num_chunks - 1`, use dangling_chunk_length instead of chunk_length
- Slice computation: `slice(chunk_index * chunk_length, chunk_index * chunk_length + actual_length)`

### Arrays with Different Shapes
- Some arrays are 2D (variable, run), others are 3D (time, variable, run)
- All must correctly identify their chunk axis via _chunk_axis_index
- Slice tuple must have correct number of dimensions

### Precision and Type Handling
- Chunk parameters (axis_length, etc.) should be integers
- No floating-point precision concerns for these parameters
- Existing precision handling for array data remains unchanged

### Memory Type Transitions
- Chunked mode uses pooled pinned buffers for staging
- Non-chunked mode uses directly allocated pinned arrays
- Chunk parameters valid in both cases, but chunked mode actually uses them

### Stream Synchronization
- Current async transfer behavior should be preserved
- Chunk slice computation is synchronous CPU operation
- No impact on CUDA stream handling

## Implementation Order Suggestion

1. Add chunk parameter fields to ManagedArray
2. Modify ManagedArray.chunk_slice() to compute based on parameters
3. Update ArrayResponse to not include chunked_slices
4. Modify MemoryManager to populate new ArrayResponse structure
5. Update BaseArrayManager._on_allocation_complete() to store parameters
6. Update InputArrays.initialise() to call chunk_slice()
7. Update OutputArrays.finalise() to call chunk_slice()
8. Remove chunked_slice_fn field from ManagedArray
9. Remove compute_per_chunk_slice() from MemoryManager
10. Remove or deprecate stride_order from ArrayRequest
11. Update tests to reflect new behavior

This order minimizes breaking changes and allows incremental testing.
