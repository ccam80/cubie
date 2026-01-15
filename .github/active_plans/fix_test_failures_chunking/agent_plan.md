# Agent Implementation Plan: Fix Chunking Test Failures

## Architectural Changes Required

### 1. ArrayRequest Enhancement

**File**: `src/cubie/memory/array_requests.py`

**Current Structure**:
```python
@attrs.define
class ArrayRequest:
    dtype = attrs.field(validator=...)
    shape: tuple[int, ...] = attrs.field(default=(1, 1, 1), ...)
    memory: str = attrs.field(default="device", ...)
    chunk_axis_index: Optional[int] = attrs.field(default=2, ...)
    unchunkable: bool = attrs.field(default=False, ...)
```

**Required Addition**:
- Add `total_runs: Optional[int]` field
- Validator: `opt_getype_validator(int, 1)` (optional, but if provided must be ≥1)
- Default: `None`
- Purpose: Carries the total number of runs to enable memory manager to compute chunking

**Expected Behavior**:
- When array managers create ArrayRequest for chunkable arrays, they set total_runs
- When total_runs is None, array is not intended for run-axis chunking (e.g., driver_coefficients)
- Memory manager extracts total_runs from any request in the group to determine chunking parameters

**Dependencies**:
- Imports from `cubie._utils` for validator
- No changes to ArrayResponse needed (chunk_length calculation happens in mem_manager)

---

### 2. Memory Manager Extraction Logic

**File**: `src/cubie/memory/mem_manager.py`

**Current Problem** (line 1203):
```python
def allocate_queue(self, triggering_instance: object) -> None:
    ...
    num_runs = triggering_instance.run_params.runs  # ❌ Assumes instance has run_params
```

**Required Changes**:

#### Change A: Extract num_runs from requests instead of instance
Replace line 1203 with call to new helper method:
```python
num_runs = self._extract_num_runs(queued_requests)
```

#### Change B: Add helper method `_extract_num_runs`
```python
def _extract_num_runs(
    self, 
    queued_requests: Dict[str, Dict[str, ArrayRequest]]
) -> int:
    """Extract total_runs from queued allocation requests.
    
    Iterates through all ArrayRequest objects in queued_requests and returns
    the first non-None total_runs value found. Validates that all requests
    with total_runs set have the same value.
    
    Parameters
    ----------
    queued_requests
        Nested dict: instance_id -> {array_label -> ArrayRequest}
    
    Returns
    -------
    int
        The total number of runs for chunking calculations
    
    Raises
    ------
    ValueError
        If no requests contain total_runs, or if inconsistent values found
    """
```

**Expected Behavior**:
- Iterate through `queued_requests` structure (instance_id → requests_dict → ArrayRequest)
- Collect all non-None `total_runs` values
- Verify all collected values are identical
- Return the value
- Raise descriptive error if no total_runs found or if inconsistent

**Edge Cases**:
- All requests have `total_runs=None`: Raise ValueError with helpful message
- Mixed None and integer values: Use the integer value (expected for driver_coefficients)
- Multiple different integers: Raise ValueError (indicates programming error)

---

### 3. Array Manager Request Creation

**File**: `src/cubie/batchsolving/arrays/BaseArrayManager.py`

**Current Code** (lines ~896-902):
```python
def allocate(self) -> None:
    requests = {}
    for array_label in list(set(self._needs_reallocation)):
        host_array_object = self.host.get_managed_array(array_label)
        host_array = host_array_object.array
        if host_array is None:
            continue
        device_array_object = self.device.get_managed_array(array_label)
        request = ArrayRequest(
            shape=host_array.shape,
            dtype=device_array_object.dtype,
            memory=device_array_object.memory_type,
            chunk_axis_index=host_array_object._chunk_axis_index,
            unchunkable=not host_array_object.is_chunked,
        )
        requests[array_label] = request
```

**Required Change**:
Add `total_runs` parameter to ArrayRequest construction:
```python
request = ArrayRequest(
    shape=host_array.shape,
    dtype=device_array_object.dtype,
    memory=device_array_object.memory_type,
    chunk_axis_index=host_array_object._chunk_axis_index,
    unchunkable=not host_array_object.is_chunked,
    total_runs=self._get_total_runs(),  # NEW
)
```

**Required Addition**: Helper method on BaseArrayManager:
```python
def _get_total_runs(self) -> Optional[int]:
    """Extract total runs from sizing metadata.
    
    Returns None if sizing metadata unavailable or doesn't contain runs.
    """
```

**Expected Behavior**:
- Check if `self._sizes` is not None
- Check if `self._sizes` has attribute `runs`
- Return `self._sizes.runs` if available
- Return `None` otherwise
- This allows graceful degradation for edge cases

---

### 4. Sizing Classes Enhancement

**Files**: 
- `src/cubie/outputhandling/output_sizes.py` (BatchOutputSizes, BatchInputSizes)

**Investigation Required**:
These classes track runs for computing array shapes. Verify they expose `runs` attribute:
- `BatchOutputSizes` should have `.runs` 
- `BatchInputSizes` should have `.runs`

**Expected Behavior**:
If these classes don't currently expose runs as a simple attribute:
- Add property that returns the run count used in shape calculations
- Or expose the underlying field that tracks runs

**Integration**:
`_get_total_runs()` in BaseArrayManager will access `self._sizes.runs`

---

### 5. Chunk Slice Logic Fix

**File**: `src/cubie/batchsolving/arrays/BaseArrayManager.py`

**Current Code** (lines 126-187, specifically around 175-186):
```python
def chunk_slice(self, chunk_index: int) -> Union[ndarray, DeviceNDArrayBase]:
    ...
    start = chunk_index * self.chunk_length
    
    if chunk_index == self.num_chunks - 1:
        end = None
    else:
        end = start + self.chunk_length  # This line exists but logic may be wrong
    
    chunk_slice_list = [slice(None)] * len(self.shape)
    chunk_slice_list[self._chunk_axis_index] = slice(start, end)
    
    return self.array[tuple(chunk_slice_list)]
```

**Problem**:
Test failures show slices returning full dimension instead of chunk size. This suggests either:
1. The slice construction is correct but self.chunk_length is wrong
2. The slice is being applied to wrong axis
3. The slice construction has a bug

**Investigation Required**:
Examine test case `test_chunk_slice_computes_correct_slices`:
- Expected: `(10, 5, 25)` for chunk 0
- Got: `(10, 5, 100)` 
- This suggests chunk_axis_index=2, chunk_length=25, but slice returning full 100

**Likely Issue**:
The chunk_length or num_chunks may not be properly set on the ManagedArray object when chunking is active.

**Required Investigation**:
- Verify ManagedArray.chunk_length is set during allocation
- Verify ManagedArray.num_chunks is set during allocation  
- Check where these are set from ArrayResponse
- Ensure BaseArrayManager propagates chunk metadata to ManagedArray instances

**Expected Flow**:
```
MemoryManager creates ArrayResponse(chunk_length=25, chunks=4)
    ↓
BaseArrayManager receives response in allocation_ready_hook
    ↓  
BaseArrayManager updates device ManagedArray objects:
    - managed_array.chunk_length = response.chunk_length
    - managed_array.num_chunks = response.chunks
    ↓
chunk_slice() can now use correct values
```

**Required Changes**:
Locate where BaseArrayManager processes ArrayResponse and ensure it sets:
- `device_array_object.chunk_length = response.chunk_length`
- `device_array_object.num_chunks = response.chunks`

This is likely in `allocation_ready_hook` or similar callback.

---

### 6. ManagedArray Chunk Metadata

**File**: `src/cubie/batchsolving/arrays/BaseArrayManager.py`

**Current ManagedArray** (lines ~40-90):
```python
@define(slots=False)
class ManagedArray:
    dtype: type = field(...)
    stride_order: tuple[str, ...] = field(...)
    default_shape: tuple[int, ...] = field(...)
    memory_type: str = field(...)
    is_chunked: bool = field(...)
    _chunk_axis_index: Optional[int] = field(...)
    _array: Optional[Union[NDArray, DeviceNDArrayBase]] = field(...)
    # ... potentially chunk_length and num_chunks fields
```

**Investigation Required**:
- Verify ManagedArray has `chunk_length` and `num_chunks` fields
- These should be settable attributes
- They should be initialized to None or sensible defaults
- They should be updated when allocation response received

**Expected Behavior**:
When BaseArrayManager receives ArrayResponse:
1. Extract chunk_length and chunks from response
2. Update each device ManagedArray with these values
3. chunk_slice() method reads these values from ManagedArray instance

**Integration Point**:
The ManagedArray `chunk_slice` method (lines 126-187) accesses:
- `self.chunk_length` - must be set from ArrayResponse
- `self.num_chunks` - must be set from ArrayResponse
- `self._chunk_axis_index` - already set from array metadata

---

### 7. ArrayResponse Chunk Length Calculation

**File**: `src/cubie/memory/mem_manager.py`

**Current Code** (around lines 1220-1228):
```python
arrays = self.allocate_all(chunked_requests, instance_id, stream=stream)
response = ArrayResponse(
    arr=arrays,
    chunks=num_chunks,
    chunk_length=chunk_length,
    chunked_shapes=chunked_shapes,
)
```

**Expected Behavior**:
The chunk_length is calculated by `get_chunk_parameters()` method:
```python
chunk_length, num_chunks = self.get_chunk_parameters(
    queued_requests, num_runs, stream_group
)
```

**Investigation Required**:
Verify `get_chunk_parameters` correctly calculates chunk_length as:
```python
chunk_length = math.ceil(num_runs / num_chunks)
```

**Integration**:
This chunk_length is already being passed to ArrayResponse. The issue is downstream:
- ArrayResponse contains correct chunk_length
- BaseArrayManager must extract and propagate to ManagedArray objects
- Tests expect ManagedArray.chunk_length to equal response.chunk_length

---

## Component Interactions

### Allocation Flow with Changes

```
1. BatchSolverKernel.solve()
   ↓
2. InputArrays.update(solver_instance, ...)
   - Calls update_from_solver(solver_instance)
     → Sets self._sizes = BatchInputSizes.from_solver(...)
     → self._sizes now has .runs attribute
   - Calls allocate()
     → Creates ArrayRequest with total_runs=self._sizes.runs
     → Calls request_allocation(requests)
       → mem_manager.queue_request(self, requests)
   ↓
3. OutputArrays.update(solver_instance)
   - Similar flow, creates ArrayRequest with total_runs
   ↓
4. One of them triggers: mem_manager.allocate_queue(triggering_instance)
   - Extracts num_runs from ANY ArrayRequest.total_runs (not from instance)
   - Calculates chunk_length = ceil(num_runs / num_chunks)
   - Creates ArrayResponse(chunks=num_chunks, chunk_length=chunk_length)
   - Calls allocation_ready_hook on each instance with response
   ↓
5. BaseArrayManager.allocation_ready_hook(response)
   - Updates self._chunks = response.chunks
   - Updates each ManagedArray:
     - device_array.chunk_length = response.chunk_length
     - device_array.num_chunks = response.chunks
   ↓
6. chunk_slice(chunk_index) can now work correctly
   - Has correct chunk_length and num_chunks from response
   - Returns properly sized slices
```

### Data Dependencies

**ArrayRequest Creation** depends on:
- BaseArrayManager has _sizes set (from update_from_solver)
- _sizes exposes .runs attribute
- _get_total_runs() can extract runs from _sizes

**Memory Manager Extraction** depends on:
- ArrayRequest has total_runs field
- At least one request in group has total_runs set
- _extract_num_runs() iterates nested dict structure correctly

**Chunk Slice Correctness** depends on:
- ArrayResponse contains correct chunk_length
- allocation_ready_hook propagates to ManagedArray objects
- ManagedArray stores chunk_length and num_chunks
- chunk_slice() reads from correct ManagedArray fields

---

## Testing Strategy

### Test Categories to Fix

**Category 1: Missing run_params** (8 tests)
- After adding total_runs to ArrayRequest
- After mem_manager uses _extract_num_runs
- These tests should pass without modification

**Category 2: Chunk slice shape** (6 tests)
- After fixing ManagedArray chunk metadata propagation
- After verifying chunk_slice logic uses correct values
- May need to update assertions if expected values were wrong

**Category 3: Chunk length in response** (6 tests)  
- After ensuring get_chunk_parameters calculates correctly
- After ArrayResponse carries chunk_length properly
- Tests verify RunParams.update_from_allocation works

**Category 4: Integration tests** (15 tests)
- Should pass once above three categories fixed
- May reveal additional edge cases in initialization

### Validation Points

**ArrayRequest Validation**:
- Can create with total_runs=100
- Can create with total_runs=None
- Validation rejects total_runs=0 or negative

**Memory Manager Extraction**:
- Finds total_runs from first request with it set
- Handles multiple requests, some with None
- Raises error if no total_runs found
- Raises error if inconsistent total_runs

**Chunk Metadata Propagation**:
- ArrayResponse contains chunk_length
- BaseArrayManager receives response
- ManagedArray has chunk_length set correctly
- chunk_slice returns correct shape

---

## Edge Cases to Handle

### Edge Case 1: No total_runs in Requests
**Scenario**: All ArrayRequest objects have total_runs=None
**Behavior**: mem_manager._extract_num_runs raises ValueError with message:
"No total_runs found in allocation requests. At least one request must specify total_runs for chunking."

### Edge Case 2: Inconsistent total_runs
**Scenario**: Request A has total_runs=100, Request B has total_runs=50
**Behavior**: Raise ValueError:
"Inconsistent total_runs in requests: found {100, 50}"

### Edge Case 3: Single Run
**Scenario**: total_runs=1, chunking not needed
**Behavior**: num_chunks=1, chunk_length=1, no actual chunking occurs

### Edge Case 4: Dangling Chunk
**Scenario**: total_runs=103, chunk_length=25, num_chunks=5
**Behavior**: 
- Chunks 0-3: 25 runs each
- Chunk 4: 3 runs
- chunk_slice handles via: `if chunk_index == num_chunks - 1: end = None`

### Edge Case 5: Unchunkable Arrays
**Scenario**: driver_coefficients has unchunkable=True, total_runs=None
**Behavior**: Not used for chunking calculation, ignored by _extract_num_runs

### Edge Case 6: Empty Allocation Queue
**Scenario**: allocate_queue called with no queued requests
**Behavior**: Should return early, not crash

---

## Implementation Order

**Phase 1: Data Carrier Enhancement**
1. Add total_runs to ArrayRequest with validator
2. Update ArrayRequest docstrings
3. Verify ArrayRequest tests pass

**Phase 2: Memory Manager Decoupling** 
4. Add _extract_num_runs helper to MemoryManager
5. Replace line 1203 with call to _extract_num_runs
6. Add tests for _extract_num_runs edge cases

**Phase 3: Array Manager Integration**
7. Add _get_total_runs to BaseArrayManager
8. Modify allocate() to pass total_runs to ArrayRequest
9. Verify sizing classes expose .runs attribute

**Phase 4: Chunk Metadata Propagation**
10. Locate allocation_ready_hook in BaseArrayManager
11. Ensure it sets chunk_length and num_chunks on ManagedArray
12. Verify chunk_slice uses these values

**Phase 5: Validation**
13. Run failing tests to verify fixes
14. Check for any remaining edge cases
15. Ensure no regressions in passing tests

---

## Success Criteria

**Functional**:
- All 35+ identified tests pass
- No new test failures introduced
- Chunking works correctly with various run counts

**Architectural**:
- MemoryManager doesn't access triggering_instance.run_params
- ArrayRequest serves as complete allocation specification  
- Array managers can be tested without BatchSolverKernel

**Code Quality**:
- Clear error messages for edge cases
- Docstrings updated for new parameters
- Type hints maintained throughout
