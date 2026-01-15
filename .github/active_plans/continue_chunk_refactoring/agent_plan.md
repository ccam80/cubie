# Continue Chunk Removal Refactoring - Detailed Implementation Plan

## Component Overview

This plan describes the architectural changes and expected behavior for consolidating run parameters and simplifying the memory allocation chain. The refactoring removes redundant data passing after the time-axis chunking removal.

## 1. RunParams Class (New Unified Class)

### Location
`src/cubie/batchsolving/BatchSolverKernel.py`

### Purpose
Replace both `FullRunParams` and `ChunkParams` with a single attrs class that manages run parameters and chunking metadata.

### Expected Attributes

```python
@define(frozen=True)
class RunParams:
    """Run parameters with optional chunking metadata.
    
    Chunking always occurs along the run axis.
    
    Attributes
    ----------
    duration : float
        Full duration of the simulation window.
    warmup : float
        Full warmup time before the main simulation.
    t0 : float
        Initial integration time.
    runs : int
        Total number of runs in the batch.
    num_chunks : int, default=1
        Number of chunks the batch is divided into.
    chunk_length : int, default=0
        Number of runs per chunk (except possibly the last).
    """
    duration: float
    warmup: float
    t0: float
    runs: int
    num_chunks: int = field(default=1, repr=False)
    chunk_length: int = field(default=0, repr=False)
```

### Expected Behavior

#### `__getitem__(self, index: int) -> RunParams`
Returns a new RunParams instance with the `runs` field set to the number of runs in chunk `index`.

**Logic:**
- If `index == num_chunks - 1` (last chunk):
  - `chunk_runs = runs - (num_chunks - 1) * chunk_length`
- Else:
  - `chunk_runs = chunk_length`
- Return `evolve(self, runs=chunk_runs)`

**Edge cases:**
- Should validate `0 <= index < num_chunks`
- Should handle `num_chunks=1` (no chunking) gracefully

#### `update_from_allocation(self, response: ArrayResponse) -> RunParams`
Returns a new RunParams instance updated with chunking metadata from the allocation response.

**Logic:**
- Extract `num_chunks`, `chunk_length` from `response`
- Return `evolve(self, num_chunks=num_chunks, chunk_length=chunk_length)`

**Note:** This could be a classmethod or instance method depending on immutability requirements (attrs frozen=True suggests returning new instance).

### Integration with BatchSolverKernel

**Initialization:**
```python
self.run_params = RunParams(duration=0.0, warmup=0.0, t0=0.0, runs=1)
```

**Update from user inputs:**
```python
# In run() method
self.run_params = RunParams(
    duration=duration,
    warmup=warmup,
    t0=t0,
    runs=inits.shape[1],
)
```

**On allocation callback:**
```python
def _on_allocation(self, response: ArrayResponse) -> None:
    self.run_params = self.run_params.update_from_allocation(response)
```

**During chunk iteration:**
```python
for i in range(self.run_params.num_chunks):
    chunk_params = self.run_params[i]
    runs = chunk_params.runs
    # ... use runs for this chunk
```

### Properties to Update in BatchSolverKernel

- `duration`, `warmup`, `t0` properties should access `self.run_params.*`
- `num_runs` property should return `self.run_params.runs`
- `chunks` property should return `self.run_params.num_chunks`

## 2. ArrayResponse Simplification

### Location
`src/cubie/memory/array_requests.py`

### Changes Required

**Remove fields:**
- `axis_length` (redundant - always equals num_runs from solver)
- `dangling_chunk_length` (computed in RunParams.__getitem__)

**Keep fields:**
- `arr` - allocated arrays dictionary
- `chunks` - number of chunks
- `chunk_length` - standard chunk size
- `chunked_shapes` - per-array chunked shapes
- `chunked_slices` - per-chunk slice functions

### Expected Behavior

ArrayResponse should contain only the information needed by array managers to perform chunked transfers:
- How many chunks were created
- Standard chunk size along run axis
- Actual allocated array shapes (after chunking)
- Slice functions for extracting per-chunk views

The **number of runs** is known by the solver/kernel and doesn't need to be echoed back.

## 3. MemoryManager Simplification

### Location
`src/cubie/memory/mem_manager.py`

### Methods to Remove or Modify

#### `get_chunk_axis_length()` - REMOVE
**Current purpose:** Extract axis length from first chunkable request.

**Why remove:** The solver already knows `num_runs`. Memory manager can receive it as a parameter instead of extracting it from array shapes.

**Migration:** Pass `num_runs` explicitly to `allocate_queue()` or extract from solver context.

#### `replace_with_chunked_size()` - EVALUATE
**Current purpose:** Replace "run" axis dimension with chunk_length.

**Possible outcomes:**
1. **Keep if:** Used in multiple places and provides clear abstraction
2. **Inline if:** Only used once in `compute_chunked_shapes()`
3. **Consolidate into:** Make it a simple lambda or local helper in `compute_chunked_shapes()`

**Decision criteria:** Does the function provide clarity or just add indirection?

#### `compute_per_chunk_slice()` - EVALUATE
**Current purpose:** Generate per-chunk slice functions for chunked transfers.

**Possible outcomes:**
1. **Keep if:** Array managers rely on this for correct chunked I/O
2. **Simplify if:** Logic can be moved inline or made simpler
3. **Remove if:** Chunked transfers can use simpler indexing

**Decision criteria:** Is this abstraction valuable for BaseArrayManager?

**Research required:** Check usage in:
- `BaseArrayManager.initialise()`
- `BaseArrayManager.finalise()`
- `InputArrays` / `OutputArrays` chunked transfer methods

### `allocate_queue()` Modifications

**Current signature:**
```python
def allocate_queue(self, triggering_instance: object) -> None:
```

**Needs:**
- Access to `num_runs` from triggering instance
- Removal of `axis_length` extraction loop
- Removal of `dangling_chunk_length` calculation

**Expected flow:**
1. Get stream group and queued requests
2. **Extract num_runs from triggering instance** (e.g., `triggering_instance.run_params.runs`)
3. Calculate chunk parameters: `chunk_length, num_chunks = self.get_chunk_parameters(queued_requests, num_runs, stream_group)`
4. Compute chunked shapes and slices
5. Allocate arrays
6. Build ArrayResponse **without** axis_length or dangling_chunk_length
7. Call allocation_ready_hook with response

### `get_chunk_parameters()` Modifications

**Current signature:**
```python
def get_chunk_parameters(
    self,
    requests: Dict[str, Dict],
    axis_length: int,
    stream_group: str,
) -> Tuple[int, int]:
```

**Change to:**
```python
def get_chunk_parameters(
    self,
    requests: Dict[str, Dict],
    num_runs: int,
    stream_group: str,
) -> Tuple[int, int]:
```

**Behavior:**
- Replace all references to `axis_length` with `num_runs`
- Return `(chunk_length, num_chunks)` as before
- No change to chunking algorithm, just parameter naming

### `compute_chunked_shapes()` - No Functional Changes

This method should continue to work as-is. If `replace_with_chunked_size()` is removed, inline the logic here:

```python
if is_request_chunkable(request):
    axis_index = request.stride_order.index("run")
    newshape = tuple(
        chunk_size if i == axis_index else dim 
        for i, dim in enumerate(request.shape)
    )
    chunked_shapes[key] = newshape
```

## 4. Array Manager Updates

### Locations
- `src/cubie/batchsolving/arrays/BaseArrayManager.py`
- `src/cubie/batchsolving/arrays/BatchInputArrays.py`
- `src/cubie/batchsolving/arrays/BatchOutputArrays.py`

### Expected Changes

#### Update to removal of ArrayResponse fields

Array managers receive ArrayResponse in their allocation hooks. They should:
- Stop accessing `response.axis_length` (use solver's `run_params.runs` instead)
- Stop accessing `response.dangling_chunk_length` (not needed if using slice functions)
- Continue using `response.chunks`, `response.chunk_length`, `response.chunked_shapes`, `response.chunked_slices`

#### Chunked Transfer Logic

**Current pattern (if using `compute_per_chunk_slice`):**
```python
# In initialise(chunk_index)
for label, managed in self.iter_managed_arrays():
    if managed.needs_chunked_transfer:
        slice_fn = managed.chunked_slice_fn
        chunk_slice = slice_fn(chunk_index)
        self.memory_manager.to_device(..., managed.array[chunk_slice], ...)
```

**Maintain this pattern** if `compute_per_chunk_slice` is kept. If removed, replace with direct slicing:
```python
# Direct slicing alternative
if managed.needs_chunked_transfer:
    start = chunk_index * chunk_length
    end = start + chunk_length
    if chunk_index == num_chunks - 1:
        end = total_runs  # Handle last chunk
    run_axis = managed.stride_order.index("run")
    # Build slice tuple
    chunk_slice = tuple(
        slice(start, end) if i == run_axis else slice(None)
        for i in range(len(managed.shape))
    )
```

**Decision:** Evaluate if the slice function abstraction is worth keeping based on how many places use it and how complex the logic becomes without it.

## 5. Testing Implications

### Tests to Update

**Location:** `tests/memory/test_memmgmt.py`

Tests for removed/modified functions:
- Tests calling `get_chunk_axis_length()` - Update to pass `num_runs` explicitly
- Tests calling `replace_with_chunked_size()` - Update if method is removed
- Tests for `compute_per_chunk_slice()` - Keep, modify, or remove based on decision

**Location:** `tests/batchsolving/`

Tests using `FullRunParams` or `ChunkParams`:
- Update to use `RunParams`
- Update assertions checking `.axis_length` or `.dangling_chunk_length`

### New Tests to Add

**For RunParams:**
- Test `__getitem__` with single chunk
- Test `__getitem__` with multiple chunks
- Test `__getitem__` with dangling chunk (last chunk smaller)
- Test `__getitem__` with exact division (no dangling)
- Test `update_from_allocation()` updates chunking metadata

**For Memory Manager:**
- Test that `num_runs` is correctly received and used
- Test chunking still works correctly without `axis_length`

## 6. Dependencies and Import Changes

### Files Importing FullRunParams or ChunkParams

All imports should change from:
```python
from cubie.batchsolving.BatchSolverKernel import FullRunParams, ChunkParams
```

To:
```python
from cubie.batchsolving.BatchSolverKernel import RunParams
```

### Expected Files to Update

- `src/cubie/batchsolving/BatchSolverKernel.py` (definition site)
- `tests/batchsolving/test_*.py` (test files)
- Any other modules that import these classes (check with grep)

## 7. Edge Cases and Error Handling

### RunParams Edge Cases

1. **Before allocation (num_chunks=1, chunk_length=0):**
   - `__getitem__(0)` should return runs unchanged
   - Attempting `__getitem__(i > 0)` should raise IndexError

2. **Single chunk after allocation:**
   - `num_chunks=1`, `chunk_length=runs`
   - `__getitem__(0)` returns runs unchanged

3. **Exact division:**
   - `runs=100`, `chunk_length=25`, `num_chunks=4`
   - All chunks have exactly 25 runs

4. **Dangling chunk:**
   - `runs=100`, `chunk_length=30`, `num_chunks=4`
   - Chunks 0-2: 30 runs each (90 total)
   - Chunk 3: 10 runs (dangling)

5. **Index validation:**
   - `__getitem__(-1)` - should raise ValueError or IndexError
   - `__getitem__(num_chunks)` - should raise IndexError

### Memory Manager Edge Cases

1. **No chunkable arrays:** Should still work with unchunkable-only allocations
2. **All arrays unchunkable but too large:** Should raise appropriate error
3. **Zero runs:** Should handle gracefully or raise clear error

## 8. Data Structures and Their Purposes

### RunParams
**Purpose:** Unified container for run parameters that can represent both full batch and individual chunks.

**Lifecycle:**
1. Created at BatchSolverKernel initialization with defaults
2. Updated when `run()` is called with actual parameters
3. Updated by `_on_allocation()` callback with chunking metadata
4. Indexed during chunk iteration to get per-chunk parameters

**Invariants:**
- `runs` is always the total number of runs for the full batch
- After allocation, `num_chunks >= 1` and `chunk_length > 0`
- `(num_chunks - 1) * chunk_length <= runs <= num_chunks * chunk_length`

### ArrayResponse
**Purpose:** Return allocated arrays and chunking metadata to allocation consumers.

**Contents:**
- `arr`: Dictionary of allocated device arrays
- `chunks`: Number of chunks created (1 if no chunking)
- `chunk_length`: Standard size of each chunk
- `chunked_shapes`: Per-array shapes after chunking applied
- `chunked_slices`: Functions to compute per-chunk slices (if kept)

**Not included (removed):**
- `axis_length`: Redundant with `num_runs` from solver
- `dangling_chunk_length`: Computed on-demand by RunParams

### ManagedArray (in ArrayContainer)
**Purpose:** Metadata wrapper for host/device arrays with chunking information.

**Chunking fields:**
- `chunked_shape`: Shape of the array after chunking (None if unchunked)
- `chunked_slice_fn`: Function to compute chunk slice (if kept)
- `needs_chunked_transfer`: Computed property comparing shape to chunked_shape

**Behavior:**
- If `chunked_shape == shape`, no chunked transfer needed
- If `chunked_shape != shape`, use slice function or direct slicing

## 9. Expected Interactions Between Components

### Initialization Flow
```
BatchSolverKernel.__init__()
  └─> Creates RunParams with defaults
      └─> Initializes InputArrays and OutputArrays
```

### Run Flow
```
BatchSolverKernel.run(inits, params, ...)
  └─> Updates RunParams with actual duration, warmup, t0, runs
      └─> Queues allocation requests (includes num_runs context)
          └─> MemoryManager.allocate_queue(triggering_instance)
              └─> Extracts num_runs from instance
              └─> Computes chunk_length, num_chunks
              └─> Allocates chunked arrays
              └─> Returns ArrayResponse (without axis_length, dangling)
                  └─> Triggers _on_allocation(response)
                      └─> Updates RunParams with chunking metadata
                          └─> Iteration over chunks:
                              └─> chunk_params = run_params[i]
                                  └─> Uses chunk_params.runs for this chunk
```

### Allocation Callback Flow
```
ArrayResponse received
  └─> RunParams.update_from_allocation(response)
      └─> New RunParams with num_chunks, chunk_length
          └─> Stored in BatchSolverKernel
```

### Chunk Iteration Flow
```
for i in range(run_params.num_chunks):
  └─> chunk_params = run_params[i]
      └─> If i < num_chunks - 1:
          └─> chunk_params.runs = chunk_length
      └─> Else (last chunk):
          └─> chunk_params.runs = total_runs - (num_chunks-1)*chunk_length
```

## 10. Migration Strategy

### Phase 1: Add RunParams Alongside Existing Classes
1. Create RunParams class with full implementation
2. Add tests for RunParams
3. Verify tests pass

### Phase 2: Update BatchSolverKernel to Use RunParams
1. Add `self.run_params` alongside existing `full_run_params` and `chunk_params`
2. Update `_on_allocation()` to populate run_params
3. Update chunk iteration to optionally use run_params
4. Verify existing tests still pass

### Phase 3: Update ArrayResponse
1. Remove `axis_length` and `dangling_chunk_length` fields
2. Update MemoryManager to not populate these fields
3. Update array managers to not access these fields
4. Update tests

### Phase 4: Clean Up Memory Manager
1. Remove or consolidate helper methods
2. Update `allocate_queue()` to use `num_runs` from solver
3. Update `get_chunk_parameters()` signature
4. Update tests

### Phase 5: Remove Old Classes
1. Remove `FullRunParams` class definition
2. Remove `ChunkParams` class definition
3. Remove `self.full_run_params` and `self.chunk_params` from BatchSolverKernel
4. Update all references to use `run_params`
5. Verify all tests pass

This phased approach allows validation at each step and makes it easier to debug issues.

---

*This plan provides the detailed specification for implementing the chunk removal refactoring. The detailed_implementer agent should use this to create specific function-level implementation tasks.*
