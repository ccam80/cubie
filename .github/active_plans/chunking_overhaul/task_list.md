# Implementation Task List
# Feature: Chunking Logic Overhaul
# Plan Reference: .github/active_plans/chunking_overhaul/agent_plan.md

---

## Task Group 1: Enhanced ArrayResponse with chunked_shapes
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/memory/array_requests.py (entire file, lines 1-130)
- File: .github/copilot-instructions.md (attrs classes pattern)

**Input Validation Required**:
- chunked_shapes: Must be dict with str keys and tuple[int, ...] values
- num_chunks: Must be positive integer >= 1

**Tasks**:
1. **Add chunked_shapes field to ArrayResponse**
   - File: src/cubie/memory/array_requests.py
   - Action: Modify
   - Details:
     ```python
     @attrs.define
     class ArrayResponse:
         # ... existing fields ...
         arr: dict[str, DeviceNDArrayBase] = attrs.field(
             default=attrs.Factory(dict), validator=val.instance_of(dict)
         )
         chunks: int = attrs.field(default=1)
         chunk_axis: str = attrs.field(
             default="run", validator=val.in_(["run", "variable", "time"])
         )
         # NEW FIELD:
         chunked_shapes: dict[str, tuple[int, ...]] = attrs.field(
             default=attrs.Factory(dict), validator=val.instance_of(dict)
         )
     ```
   - Edge cases: Empty dict when no chunking occurs
   - Integration: MemoryManager populates this during allocation

**Tests to Create**:
- Test file: tests/memory/test_array_requests.py
- Test function: test_array_response_has_chunked_shapes_field
- Description: Verify ArrayResponse can be instantiated with chunked_shapes dict
- Test function: test_array_response_chunked_shapes_default_empty
- Description: Verify chunked_shapes defaults to empty dict when not provided

**Tests to Run**:
- tests/memory/test_array_requests.py::TestArrayResponse::test_array_response_has_chunked_shapes_field
- tests/memory/test_array_requests.py::TestArrayResponse::test_array_response_chunked_shapes_default_empty

**Outcomes**: 
- Files Modified: 
  * src/cubie/memory/array_requests.py (10 lines changed)
  * tests/memory/test_array_requests.py (25 lines changed)
- Functions/Methods Added/Modified:
  * ArrayResponse class: added chunked_shapes field in array_requests.py
  * TestArrayResponse class: added test class with 2 test methods
- Implementation Summary:
  Added chunked_shapes field to ArrayResponse as a dict[str, tuple[int, ...]] with
  default empty dict and instance_of(dict) validator. Updated docstring to document
  the new field in both Parameters and Attributes sections. Created test class
  TestArrayResponse with tests for explicit instantiation and default empty dict.
- Issues Flagged: None 


---

## Task Group 2: ManagedArray with chunked_shape Storage
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/batchsolving/arrays/BaseArrayManager.py (lines 1-100, ManagedArray class)
- File: src/cubie/memory/array_requests.py (ArrayResponse class)

**Input Validation Required**:
- chunked_shape: Optional tuple of positive integers, or None before first allocation

**Tasks**:
1. **Add chunked_shape field to ManagedArray**
   - File: src/cubie/batchsolving/arrays/BaseArrayManager.py
   - Action: Modify
   - Details:
     ```python
     @define(slots=False)
     class ManagedArray:
         """Metadata wrapper for a single managed array."""
     
         dtype: type = field(...)
         stride_order: tuple[str, ...] = field(...)
         shape: tuple[Optional[int]] = field(...)
         memory_type: str = field(...)
         is_chunked: bool = field(...)  # Keep for now, will be deprecated
         _array: Optional[Union[NDArray, DeviceNDArrayBase]] = field(...)
         # NEW FIELD:
         chunked_shape: Optional[tuple[int, ...]] = field(
             default=None,
             validator=attrsval_optional(
                 attrsval_deep_iterable(
                     member_validator=attrsval_instance_of(int),
                     iterable_validator=attrsval_instance_of(tuple),
                 )
             ),
         )
     ```
   - Edge cases: chunked_shape is None until first allocation

2. **Add needs_chunked_transfer property to ManagedArray**
   - File: src/cubie/batchsolving/arrays/BaseArrayManager.py
   - Action: Modify
   - Details:
     ```python
     @property
     def needs_chunked_transfer(self) -> bool:
         """Return True if this array requires chunked transfers.
         
         Chunked transfers are needed when the array's full shape differs
         from its per-chunk shape. This comparison replaces complex
         is_chunked flag logic.
         """
         if self.chunked_shape is None:
             return False
         return self.shape != self.chunked_shape
     ```
   - Integration: Transfer functions will use this instead of is_chunked flag

**Tests to Create**:
- Test file: tests/batchsolving/arrays/test_basearraymanager.py
- Test function: test_managed_array_has_chunked_shape_field
- Description: Verify ManagedArray has chunked_shape field defaulting to None
- Test function: test_managed_array_needs_chunked_transfer_false_when_none
- Description: Verify needs_chunked_transfer returns False when chunked_shape is None
- Test function: test_managed_array_needs_chunked_transfer_false_when_equal
- Description: Verify needs_chunked_transfer returns False when shapes match
- Test function: test_managed_array_needs_chunked_transfer_true_when_different
- Description: Verify needs_chunked_transfer returns True when shapes differ

**Tests to Run**:
- tests/batchsolving/arrays/test_basearraymanager.py::TestManagedArrayChunkedShape::test_managed_array_has_chunked_shape_field
- tests/batchsolving/arrays/test_basearraymanager.py::TestManagedArrayChunkedShape::test_managed_array_needs_chunked_transfer_false_when_none
- tests/batchsolving/arrays/test_basearraymanager.py::TestManagedArrayChunkedShape::test_managed_array_needs_chunked_transfer_false_when_equal
- tests/batchsolving/arrays/test_basearraymanager.py::TestManagedArrayChunkedShape::test_managed_array_needs_chunked_transfer_true_when_different

**Outcomes**: 
- Files Modified: 
  * src/cubie/batchsolving/arrays/BaseArrayManager.py (19 lines added)
  * tests/batchsolving/arrays/test_basearraymanager.py (43 lines added)
- Functions/Methods Added/Modified:
  * ManagedArray class: added chunked_shape field in BaseArrayManager.py
  * ManagedArray class: added needs_chunked_transfer property in BaseArrayManager.py
  * TestManagedArrayChunkedShape class: added test class with 4 test methods
- Implementation Summary:
  Added chunked_shape field to ManagedArray as Optional[tuple[int, ...]] with
  default None and deep_iterable validator for tuple of ints. Added
  needs_chunked_transfer property that returns True when shape differs from
  chunked_shape, enabling simplified transfer logic based on shape comparison.
  Created test class TestManagedArrayChunkedShape with tests for default None,
  False when None, False when shapes match, and True when shapes differ.
- Issues Flagged: None 


---

## Task Group 3: Centralized Chunk Calculation in MemoryManager
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/memory/mem_manager.py (entire file, especially lines 946-1000 get_chunks, lines 1138-1184 chunk_arrays, lines 1186-1237 single_request, lines 1239-1322 allocate_queue)
- File: src/cubie/memory/array_requests.py (ArrayRequest, ArrayResponse)

**Input Validation Required**:
- requests: Must be dict[str, ArrayRequest]
- num_chunks: Must be >= 1
- chunk_axis: Must be in ["run", "variable", "time"]

**Tasks**:
1. **Add get_chunkable_request_size helper function**
   - File: src/cubie/memory/mem_manager.py
   - Action: Add new function after get_total_request_size (around line 1418)
   - Details:
     ```python
     def get_chunkable_request_size(
         requests: dict[str, ArrayRequest],
         chunk_axis: str,
     ) -> int:
         """Calculate total memory for chunkable arrays only.
         
         Parameters
         ----------
         requests
             Dictionary of array requests to analyze.
         chunk_axis
             Axis along which chunking would occur.
         
         Returns
         -------
         int
             Total bytes for arrays that can be chunked along chunk_axis.
         
         Notes
         -----
         Arrays are chunkable if:
         - request.unchunkable is False
         - chunk_axis is in request.stride_order
         """
         total = 0
         for request in requests.values():
             if request.unchunkable:
                 continue
             if request.stride_order is None:
                 continue
             if chunk_axis not in request.stride_order:
                 continue
             total += prod(request.shape) * request.dtype().itemsize
         return total
     ```
   - Edge cases: Returns 0 if all arrays are unchunkable

2. **Add compute_chunked_shapes helper method to MemoryManager**
   - File: src/cubie/memory/mem_manager.py
   - Action: Add new method to MemoryManager class (around line 1135)
   - Details:
     ```python
     def compute_chunked_shapes(
         self,
         requests: dict[str, ArrayRequest],
         num_chunks: int,
         chunk_axis: str,
     ) -> dict[str, tuple[int, ...]]:
         """Compute per-array chunked shapes using floor division.
         
         Parameters
         ----------
         requests
             Dictionary mapping labels to array requests.
         num_chunks
             Number of chunks to divide arrays into.
         chunk_axis
             Axis label along which to chunk.
         
         Returns
         -------
         dict[str, tuple[int, ...]]
             Mapping from array labels to their per-chunk shapes.
         
         Notes
         -----
         Uses floor division (not ceiling) to compute chunk size,
         ensuring chunk indices never exceed array bounds.
         Unchunkable arrays retain their original shape.
         """
         chunked_shapes = {}
         for key, request in requests.items():
             # Unchunkable arrays keep original shape
             if getattr(request, "unchunkable", False):
                 chunked_shapes[key] = request.shape
                 continue
             # Arrays without chunk_axis keep original shape
             if request.stride_order is None:
                 chunked_shapes[key] = request.shape
                 continue
             if chunk_axis not in request.stride_order:
                 chunked_shapes[key] = request.shape
                 continue
             # Compute chunked shape using floor division
             axis_index = request.stride_order.index(chunk_axis)
             axis_length = request.shape[axis_index]
             chunk_size = axis_length // num_chunks
             # Ensure at least 1 element per chunk
             chunk_size = max(1, chunk_size)
             newshape = tuple(
                 chunk_size if i == axis_index else dim
                 for i, dim in enumerate(request.shape)
             )
             chunked_shapes[key] = newshape
         return chunked_shapes
     ```
   - Edge cases: num_chunks == 1 returns original shapes; empty requests returns empty dict

3. **Refactor chunk_arrays to use floor division**
   - File: src/cubie/memory/mem_manager.py
   - Action: Modify chunk_arrays method (lines 1138-1184)
   - Details:
     ```python
     def chunk_arrays(
         self,
         requests: dict[str, ArrayRequest],
         numchunks: int,
         axis: str = "run",
     ) -> dict[str, ArrayRequest]:
         """Divide array requests into smaller chunks along a specified axis.
         
         Uses floor division to prevent chunk indices from exceeding bounds.
         """
         chunked_requests = deepcopy(requests)
         for key, request in chunked_requests.items():
             if getattr(request, "unchunkable", False):
                 continue
             if request.stride_order is None:
                 continue
             if axis not in request.stride_order:
                 continue
             run_index = request.stride_order.index(axis)
             axis_length = request.shape[run_index]
             # Use floor division to prevent overflow
             chunk_size = axis_length // numchunks
             chunk_size = max(1, chunk_size)
             newshape = tuple(
                 chunk_size if i == run_index else value
                 for i, value in enumerate(request.shape)
             )
             request.shape = newshape
             chunked_requests[key] = request
         return chunked_requests
     ```
   - Edge cases: numchunks > axis_length results in chunk_size = 1

4. **Update single_request to use chunkable memory and populate chunked_shapes**
   - File: src/cubie/memory/mem_manager.py
   - Action: Modify single_request method (lines 1186-1237)
   - Details:
     ```python
     def single_request(
         self,
         instance: Union[object, int],
         requests: dict[str, ArrayRequest],
         chunk_axis: str = "run",
     ) -> None:
         """Process a single allocation request with automatic chunking."""
         self._check_requests(requests)
         if isinstance(instance, int):
             instance_id = instance
         else:
             instance_id = id(instance)
     
         # Use only chunkable memory for chunk count calculation
         chunkable_size = get_chunkable_request_size(requests, chunk_axis)
         available_memory = self.get_available_single(id(instance))
         
         if chunkable_size == 0:
             numchunks = 1
         else:
             numchunks = self.get_chunks(chunkable_size, available_memory)
         
         # Compute chunked_shapes before modifying requests
         chunked_shapes = self.compute_chunked_shapes(
             requests, numchunks, chunk_axis
         )
         
         chunked_requests = self.chunk_arrays(
             requests, numchunks, axis=chunk_axis
         )
     
         arrays = self.allocate_all(
             chunked_requests, instance_id, self.get_stream(instance)
         )
         self.registry[instance_id].allocation_ready_hook(
             ArrayResponse(
                 arr=arrays,
                 chunks=numchunks,
                 chunk_axis=chunk_axis,
                 chunked_shapes=chunked_shapes,
             )
         )
     ```

5. **Update allocate_queue to use chunkable memory and populate chunked_shapes**
   - File: src/cubie/memory/mem_manager.py
   - Action: Modify allocate_queue method (lines 1239-1322)
   - Details:
     The key changes:
     - Use get_chunkable_request_size instead of get_total_request_size for chunk count
     - Call compute_chunked_shapes for each instance
     - Include chunked_shapes in ArrayResponse
     ```python
     # In the limit_type == "group" branch:
     chunkable_size = sum(
         get_chunkable_request_size(request, chunk_axis)
         for request in queued_requests.values()
     )
     if chunkable_size == 0:
         numchunks = 1
     else:
         numchunks = self.get_chunks(chunkable_size, available_memory)
     
     # For each instance allocation:
     chunked_shapes = self.compute_chunked_shapes(
         requests_dict, numchunks, chunk_axis
     )
     response = ArrayResponse(
         arr=arrays,
         chunks=numchunks,
         chunk_axis=chunk_axis,
         chunked_shapes=chunked_shapes,
     )
     ```

**Tests to Create**:
- Test file: tests/memory/test_memmgmt.py
- Test function: test_get_chunkable_request_size_excludes_unchunkable
- Description: Verify unchunkable arrays are excluded from size calculation
- Test function: test_get_chunkable_request_size_excludes_wrong_axis
- Description: Verify arrays without chunk_axis in stride_order are excluded
- Test function: test_compute_chunked_shapes_uses_floor_division
- Description: Verify chunk_size = axis_length // num_chunks (floor, not ceiling)
- Test function: test_compute_chunked_shapes_preserves_unchunkable
- Description: Verify unchunkable arrays keep original shape
- Test function: test_chunk_arrays_uses_floor_division
- Description: Verify chunk_arrays uses floor division
- Test function: test_single_request_populates_chunked_shapes
- Description: Verify single_request includes chunked_shapes in response
- Test function: test_allocate_queue_populates_chunked_shapes
- Description: Verify allocate_queue includes chunked_shapes in response

**Tests to Run**:
- tests/memory/test_memmgmt.py::test_get_chunkable_request_size_excludes_unchunkable
- tests/memory/test_memmgmt.py::test_get_chunkable_request_size_excludes_wrong_axis
- tests/memory/test_memmgmt.py::test_get_chunkable_request_size_returns_zero_all_unchunkable
- tests/memory/test_memmgmt.py::test_compute_chunked_shapes_uses_floor_division
- tests/memory/test_memmgmt.py::test_compute_chunked_shapes_preserves_unchunkable
- tests/memory/test_memmgmt.py::test_compute_chunked_shapes_empty_requests
- tests/memory/test_memmgmt.py::test_compute_chunked_shapes_single_chunk
- tests/memory/test_memmgmt.py::test_chunk_arrays_uses_floor_division
- tests/memory/test_memmgmt.py::test_chunk_arrays_min_one_element
- tests/memory/test_memmgmt.py::test_single_request_populates_chunked_shapes
- tests/memory/test_memmgmt.py::test_allocate_queue_populates_chunked_shapes

**Outcomes**: 
- Files Modified: 
  * src/cubie/memory/mem_manager.py (~90 lines changed)
  * tests/memory/test_memmgmt.py (~220 lines added)
- Functions/Methods Added/Modified:
  * get_chunkable_request_size() - new helper function in mem_manager.py
  * compute_chunked_shapes() - new method in MemoryManager class
  * chunk_arrays() - refactored to use floor division instead of ceiling
  * single_request() - updated to use chunkable memory and populate chunked_shapes
  * allocate_queue() - updated to use chunkable memory and populate chunked_shapes
- Implementation Summary:
  Implemented centralized chunk calculation using floor division to prevent
  chunk index overflow. Added get_chunkable_request_size helper to calculate
  memory for only chunkable arrays. Added compute_chunked_shapes method to
  compute per-array chunked shapes. Updated chunk_arrays to use floor division.
  Updated single_request and allocate_queue to use chunkable memory for chunk
  count calculation and populate chunked_shapes in ArrayResponse. Created
  11 comprehensive tests covering all new functions and edge cases.
- Issues Flagged: None

---

## Task Group 4: BaseArrayManager Updates for chunked_shape Propagation
**Status**: [x]
**Dependencies**: Task Groups 1, 2, 3

**Required Context**:
- File: src/cubie/batchsolving/arrays/BaseArrayManager.py (entire file, especially _on_allocation_complete lines 313-351, check_sizes lines 526-626)
- File: src/cubie/memory/array_requests.py (ArrayResponse)

**Input Validation Required**:
- response.chunked_shapes: Dict with matching keys to device arrays

**Tasks**:
1. **Update _on_allocation_complete to store chunked_shape in ManagedArrays**
   - File: src/cubie/batchsolving/arrays/BaseArrayManager.py
   - Action: Modify _on_allocation_complete method (lines 313-351)
   - Details:
     ```python
     def _on_allocation_complete(self, response: ArrayResponse) -> None:
         """Callback for when the allocation response is received."""
         for array_label in self._needs_reallocation:
             try:
                 self.device.attach(array_label, response.arr[array_label])
                 # Update chunked_shape from response
                 if array_label in response.chunked_shapes:
                     managed = self.device.get_managed_array(array_label)
                     managed.chunked_shape = response.chunked_shapes[array_label]
             except KeyError:
                 warn(
                     f"Device array {array_label} not found in allocation "
                     f"response. See "
                     f"BaseArrayManager._on_allocation_complete docstring "
                     f"for more info.",
                     UserWarning,
                 )
         self._chunks = response.chunks
         self._chunk_axis = response.chunk_axis
         self._needs_reallocation.clear()
     ```
   - Integration: ManagedArray.chunked_shape is now populated from allocation

2. **Update check_sizes to use chunked_shape instead of recalculating**
   - File: src/cubie/batchsolving/arrays/BaseArrayManager.py
   - Action: Modify check_sizes method (lines 526-626)
   - Details:
     Replace the chunk calculation logic (lines 595-610):
     ```python
     # OLD CODE using ceiling division:
     # if (location == "device" and self._chunks > 0 and managed.is_chunked):
     #     if chunk_axis_name in target_stride_order:
     #         chunk_axis_index = target_stride_order.index(chunk_axis_name)
     #         if expected_shape[chunk_axis_index] is not None:
     #             expected_shape[chunk_axis_index] = int(
     #                 np_ceil(expected_shape[chunk_axis_index] / self._chunks)
     #             )
     
     # NEW CODE using stored chunked_shape:
     if location == "device" and managed.chunked_shape is not None:
         # Use stored chunked_shape from allocation response
         expected_shape = list(managed.chunked_shape)
     ```
   - Edge cases: managed.chunked_shape may be None before allocation

**Tests to Create**:
- Test file: tests/batchsolving/arrays/test_basearraymanager.py
- Test function: test_on_allocation_complete_stores_chunked_shape
- Description: Verify _on_allocation_complete populates ManagedArray.chunked_shape
- Test function: test_check_sizes_uses_chunked_shape
- Description: Verify check_sizes uses stored chunked_shape instead of recalculating

**Tests to Run**:
- tests/batchsolving/arrays/test_basearraymanager.py::TestChunkedShapePropagation::test_on_allocation_complete_stores_chunked_shape
- tests/batchsolving/arrays/test_basearraymanager.py::TestChunkedShapePropagation::test_check_sizes_uses_chunked_shape

**Outcomes**: 
- Files Modified: 
  * src/cubie/batchsolving/arrays/BaseArrayManager.py (7 lines changed)
  * tests/batchsolving/arrays/test_basearraymanager.py (160 lines added)
- Functions/Methods Added/Modified:
  * _on_allocation_complete() in BaseArrayManager.py - added chunked_shape storage
  * check_sizes() in BaseArrayManager.py - replaced ceiling division with chunked_shape
  * TestChunkedShapePropagation class with 2 test methods
- Implementation Summary:
  Updated _on_allocation_complete to store chunked_shape from ArrayResponse into
  ManagedArray.chunked_shape for each allocated array. Updated check_sizes to use
  the stored chunked_shape instead of recalculating with ceiling division. Removed
  unused chunk_axis_name variable from check_sizes. Created TestChunkedShapePropagation
  test class with tests for allocation storage and check_sizes usage.
- Issues Flagged: None


---

## Task Group 5: Simplified Transfer Logic in BatchInputArrays
**Status**: [x]
**Dependencies**: Task Groups 2, 4

**Required Context**:
- File: src/cubie/batchsolving/arrays/BatchInputArrays.py (entire file, especially initialise method lines 279-343)
- File: src/cubie/batchsolving/arrays/BaseArrayManager.py (ManagedArray.needs_chunked_transfer)

**Input Validation Required**:
- None (uses existing validated fields)

**Tasks**:
1. **Simplify initialise method to use needs_chunked_transfer**
   - File: src/cubie/batchsolving/arrays/BatchInputArrays.py
   - Action: Modify initialise method (lines 279-343)
   - Details:
     Replace complex chunking checks with shape comparison:
     ```python
     def initialise(self, host_indices: Union[slice, NDArray]) -> None:
         """Copy a batch chunk of host data to device buffers."""
         from_ = []
         to_ = []
     
         if self._chunks <= 1:
             arrays_to_copy = [array for array in self._needs_overwrite]
             self._needs_overwrite = []
         else:
             arrays_to_copy = list(self.device.array_names())
     
         for array_name in arrays_to_copy:
             device_obj = self.device.get_managed_array(array_name)
             to_.append(device_obj.array)
             host_obj = self.host.get_managed_array(array_name)
             
             # Use needs_chunked_transfer for simple branching
             if not device_obj.needs_chunked_transfer:
                 from_.append(host_obj.array)
             else:
                 stride_order = host_obj.stride_order
                 if self._chunk_axis not in stride_order:
                     from_.append(host_obj.array)
                     continue
                 chunk_index = stride_order.index(self._chunk_axis)
                 slice_tuple = [slice(None)] * len(stride_order)
                 slice_tuple[chunk_index] = host_indices
                 host_slice = host_obj.array[tuple(slice_tuple)]
     
                 # Chunked mode: use buffer pool for pinned staging
                 device_shape = device_obj.array.shape
                 buffer = self._buffer_pool.acquire(
                     array_name, device_shape, host_slice.dtype
                 )
                 data_slice = tuple(slice(0, s) for s in host_slice.shape)
                 buffer.array[data_slice] = host_slice
                 from_.append(buffer.array)
                 self._active_buffers.append(buffer)
     
         self.to_device(from_, to_)
     ```
   - Edge cases: needs_chunked_transfer is False for unchunkable arrays

**Tests to Create**:
- Test file: tests/batchsolving/arrays/test_batchinputarrays.py
- Test function: test_initialise_uses_needs_chunked_transfer
- Description: Verify initialise uses needs_chunked_transfer for branching

**Tests to Run**:
- tests/batchsolving/arrays/test_batchinputarrays.py::TestNeedsChunkedTransferBranching::test_initialise_uses_needs_chunked_transfer
- tests/batchsolving/arrays/test_batchinputarrays.py::TestNeedsChunkedTransferBranching::test_initialise_no_buffers_when_needs_chunked_transfer_false

**Outcomes**: 
- Files Modified: 
  * src/cubie/batchsolving/arrays/BatchInputArrays.py (5 lines removed, net -5 lines)
  * tests/batchsolving/arrays/test_batchinputarrays.py (106 lines added)
- Functions/Methods Added/Modified:
  * initialise() in BatchInputArrays.py - simplified transfer logic
  * TestNeedsChunkedTransferBranching class with 2 test methods
- Implementation Summary:
  Simplified the initialise method in BatchInputArrays to use the new
  needs_chunked_transfer property from ManagedArray instead of the complex
  `self._chunks <= 1 or not device_obj.is_chunked` and nested `if self.is_chunked`
  checks. The new logic uses a single `if not device_obj.needs_chunked_transfer`
  branch, which determines transfer behavior by comparing shape vs chunked_shape.
  Removed the non-chunked branch that directly copied host slices since the
  needs_chunked_transfer property now encapsulates that logic. Created two tests
  to verify the new branching logic works correctly.
- Issues Flagged: None


---

## Task Group 6: Simplified Transfer Logic in BatchOutputArrays
**Status**: [x]
**Dependencies**: Task Groups 2, 4

**Required Context**:
- File: src/cubie/batchsolving/arrays/BatchOutputArrays.py (entire file, especially _on_allocation_complete lines 198-222, _convert_host_to_numpy lines 240-259, finalise lines 416-493)
- File: src/cubie/batchsolving/arrays/BaseArrayManager.py (ManagedArray.needs_chunked_transfer)

**Input Validation Required**:
- None (uses existing validated fields)

**Tasks**:
1. **Update _on_allocation_complete to store chunked_shapes**
   - File: src/cubie/batchsolving/arrays/BatchOutputArrays.py
   - Action: Modify _on_allocation_complete method (lines 198-222)
   - Details:
     Parent method now handles chunked_shape storage, so this override
     only needs to handle host array conversion:
     ```python
     def _on_allocation_complete(self, response: ArrayResponse) -> None:
         """Callback for when the allocation response is received."""
         super()._on_allocation_complete(response)
         if self.is_chunked:
             self._convert_host_to_numpy()
         else:
             self.host.set_memory_type("pinned")
     ```
   - Integration: Parent handles chunked_shape storage

2. **Simplify _convert_host_to_numpy to use needs_chunked_transfer**
   - File: src/cubie/batchsolving/arrays/BatchOutputArrays.py
   - Action: Modify _convert_host_to_numpy method (lines 240-259)
   - Details:
     ```python
     def _convert_host_to_numpy(self) -> None:
         """Convert pinned host arrays to regular numpy for chunked mode."""
         for name, slot in self.host.iter_managed_arrays():
             device_slot = self.device.get_managed_array(name)
             # Use needs_chunked_transfer instead of is_chunked flag
             if slot.memory_type == "pinned" and device_slot.needs_chunked_transfer:
                 old_array = slot.array
                 if old_array is not None:
                     new_array = self._memory_manager.create_host_array(
                         old_array.shape, old_array.dtype, "host"
                     )
                     slot.array = new_array
                     slot.memory_type = "host"
     ```

3. **Simplify finalise method to use needs_chunked_transfer**
   - File: src/cubie/batchsolving/arrays/BatchOutputArrays.py
   - Action: Modify finalise method (lines 416-493)
   - Details:
     Replace `self.is_chunked and slot.is_chunked` with needs_chunked_transfer:
     ```python
     def finalise(self, host_indices: ChunkIndices) -> None:
         """Queue device-to-host transfers for a chunk."""
         from_ = []
         to_ = []
         stream = self._memory_manager.get_stream(self)
     
         for array_name, slot in self.host.iter_managed_arrays():
             device_array = self.device.get_array(array_name)
             device_slot = self.device.get_managed_array(array_name)
             host_array = slot.array
             stride_order = slot.stride_order
     
             to_target = host_array
             from_target = device_array
             if self._chunk_axis in stride_order:
                 chunk_index = stride_order.index(self._chunk_axis)
                 slice_tuple = slice_variable_dimension(
                     host_indices, chunk_index, len(stride_order)
                 )
                 host_slice = host_array[slice_tuple]
                 # Use needs_chunked_transfer instead of is_chunked flags
                 if device_slot.needs_chunked_transfer:
                     buffer = self._buffer_pool.acquire(
                         array_name, device_array.shape, host_slice.dtype
                     )
                     to_target = buffer.array
                     self._pending_buffers.append(
                         PendingBuffer(
                             buffer=buffer,
                             target_array=host_array,
                             slice_tuple=slice_tuple,
                             array_name=array_name,
                             data_shape=host_slice.shape,
                             buffer_pool=self._buffer_pool,
                         )
                     )
     
             to_.append(to_target)
             from_.append(from_target)
     
         self.from_device(from_, to_)
     
         # Record events and submit to watcher for chunked mode
         if self._pending_buffers:
             if not CUDA_SIMULATION:
                 event = cuda.event()
                 event.record(stream)
             else:
                 event = None
     
             for buffer in self._pending_buffers:
                 self._watcher.submit_from_pending_buffer(
                     event=event,
                     pending_buffer=buffer,
                 )
     ```

**Tests to Create**:
- Test file: tests/batchsolving/arrays/test_batchoutputarrays.py
- Test function: test_convert_host_to_numpy_uses_needs_chunked_transfer
- Description: Verify _convert_host_to_numpy uses needs_chunked_transfer
- Test function: test_finalise_uses_needs_chunked_transfer
- Description: Verify finalise uses needs_chunked_transfer for branching

**Tests to Run**:
- tests/batchsolving/arrays/test_batchoutputarrays.py::TestNeedsChunkedTransferBranching::test_convert_host_to_numpy_uses_needs_chunked_transfer
- tests/batchsolving/arrays/test_batchoutputarrays.py::TestNeedsChunkedTransferBranching::test_finalise_uses_needs_chunked_transfer

**Outcomes**: 
- Files Modified: 
  * src/cubie/batchsolving/arrays/BatchOutputArrays.py (8 lines changed)
  * tests/batchsolving/arrays/test_batchoutputarrays.py (137 lines added)
- Functions/Methods Added/Modified:
  * _convert_host_to_numpy() in BatchOutputArrays.py - simplified to use needs_chunked_transfer
  * finalise() in BatchOutputArrays.py - replaced is_chunked flags with needs_chunked_transfer
  * TestNeedsChunkedTransferBranching class with 2 test methods
- Implementation Summary:
  Simplified transfer logic in BatchOutputArrays by using the needs_chunked_transfer
  property from ManagedArray (added in Task Group 2) instead of checking
  `self.is_chunked and slot.is_chunked` flags. The _convert_host_to_numpy method now
  checks `device_slot.needs_chunked_transfer` to determine if host arrays should be
  converted from pinned to regular numpy. The finalise method now uses
  `device_slot.needs_chunked_transfer` to determine if buffer pool should be used
  for chunked transfers. The final `if self.is_chunked and self._pending_buffers`
  check was simplified to just `if self._pending_buffers` since buffers only exist
  when chunked transfers are needed. The _on_allocation_complete method was already
  correctly implemented (calls super() which handles chunked_shape storage).
  Created TestNeedsChunkedTransferBranching test class with two tests to verify
  the new branching logic works correctly.
- Issues Flagged: None


---

## Task Group 7: BatchSolverKernel Chunking Delegation
**Status**: [x]
**Dependencies**: Task Groups 3, 4

**Required Context**:
- File: src/cubie/batchsolving/BatchSolverKernel.py (entire file, especially chunk_run lines 597-647, run method lines 342-551, _on_allocation lines 940-942)
- File: src/cubie/memory/array_requests.py (ArrayResponse)

**Input Validation Required**:
- None (uses values from ArrayResponse)

**Tasks**:
1. **Simplify chunk_run method to use floor division**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify chunk_run method (lines 597-647)
   - Details:
     Change from ceiling to floor division for run chunking:
     ```python
     def chunk_run(
         self,
         chunk_axis: str,
         duration: float,
         warmup: float,
         t0: float,
         numruns: int,
         chunks: int,
     ) -> ChunkParams:
         """Split the workload into chunks along the selected axis."""
         chunkruns = numruns
         chunk_warmup = warmup
         chunk_duration = duration
         chunk_t0 = t0
         if chunk_axis == "run":
             # Use floor division to match MemoryManager's calculation
             chunkruns = numruns // chunks
             chunkruns = max(1, chunkruns)
             chunksize = chunkruns
         elif chunk_axis == "time":
             chunk_duration = duration / chunks
             chunksize = self.output_length // chunks
             chunksize = max(1, chunksize)
             chunkruns = numruns
     
         return ChunkParams(
             duration=chunk_duration,
             warmup=chunk_warmup,
             t0=chunk_t0,
             size=chunksize,
             runs=chunkruns,
         )
     ```
   - Edge cases: chunks > numruns results in chunkruns = 1

2. **Update run method loop to handle final partial chunk correctly**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify run method chunk loop (around lines 500-546)
   - Details:
     The loop should compute actual indices for each chunk, handling
     the final chunk which may have fewer runs:
     ```python
     for i in range(self.chunks):
         # Compute actual start and end indices for this chunk
         start_idx = i * chunk_params.size
         # For final chunk, clamp to actual run count
         if chunk_axis == "run":
             end_idx = min((i + 1) * chunk_params.size, numruns)
         else:
             end_idx = min((i + 1) * chunk_params.size, self.output_length)
         indices = slice(start_idx, end_idx)
         
         # Use actual chunk run count for kernel launch
         if chunk_axis == "run":
             actual_chunk_runs = end_idx - start_idx
         else:
             actual_chunk_runs = kernel_runs
         
         # ... rest of loop with actual_chunk_runs for kernel launch
     ```
   - Edge cases: Final chunk with fewer runs than chunk_size

**Tests to Create**:
- Test file: tests/batchsolving/test_batchsolverkernel.py
- Test function: test_chunk_run_uses_floor_division
- Description: Verify chunk_run uses floor division for run count
- Test function: test_chunk_run_handles_uneven_division
- Description: Verify chunk_run handles numruns not divisible by chunks

**Tests to Run**:
- tests/batchsolving/test_batchsolverkernel.py::TestChunkRunFloorDivision::test_chunk_run_uses_floor_division
- tests/batchsolving/test_batchsolverkernel.py::TestChunkRunFloorDivision::test_chunk_run_handles_uneven_division
- tests/batchsolving/test_batchsolverkernel.py::TestChunkRunFloorDivision::test_chunk_run_minimum_one_run_per_chunk
- tests/batchsolving/test_batchsolverkernel.py::TestChunkRunFloorDivision::test_chunk_run_single_chunk_returns_all_runs
- tests/batchsolving/test_batchsolverkernel.py::TestChunkRunFloorDivision::test_chunk_run_time_axis_uses_floor_division

**Outcomes**: 
- Files Modified: 
  * src/cubie/batchsolving/BatchSolverKernel.py (25 lines changed)
  * tests/batchsolving/test_batchsolverkernel.py (96 lines added - new file)
- Functions/Methods Added/Modified:
  * chunk_run() in BatchSolverKernel.py - changed to use floor division
  * run() in BatchSolverKernel.py - updated loop to handle final partial chunks
  * TestChunkRunFloorDivision class with 5 test methods
- Implementation Summary:
  Changed chunk_run method to use floor division (`//`) instead of ceiling
  division for both run and time axis chunking. Added `max(1, ...)` to ensure
  at least 1 element per chunk. Updated run method loop to compute actual
  start/end indices for each chunk, clamping final chunk to actual run/output
  count. Removed unused BLOCKSPERGRID variable and now compute chunk_blocks
  inside the loop based on actual_chunk_runs. Created test file with 5 tests
  covering floor division, uneven division, minimum run per chunk, single chunk,
  and time axis chunking.
- Issues Flagged: None


---

## Task Group 8: Remove np_ceil Import and Cleanup
**Status**: [ ]
**Dependencies**: Task Groups 3, 4, 7

**Required Context**:
- File: src/cubie/batchsolving/arrays/BaseArrayManager.py (imports around line 24)
- File: src/cubie/batchsolving/BatchSolverKernel.py (imports around line 16)
- File: src/cubie/memory/mem_manager.py (imports around line 15)

**Input Validation Required**:
- None

**Tasks**:
1. **Remove np_ceil import from BaseArrayManager if no longer used**
   - File: src/cubie/batchsolving/arrays/BaseArrayManager.py
   - Action: Modify imports (around line 24)
   - Details:
     After Task Group 4, np_ceil should no longer be used in check_sizes.
     Remove from imports if unused:
     ```python
     from numpy import (
         array_equal as np_array_equal,
         # Remove: ceil as np_ceil,
         float32 as np_float32,
         zeros as np_zeros,
     )
     ```

2. **Remove np_ceil from chunk_arrays in mem_manager if no longer used**
   - File: src/cubie/memory/mem_manager.py
   - Action: Verify np_ceil usage (around line 15 and 1179)
   - Details:
     After refactoring chunk_arrays to use floor division, np_ceil may
     no longer be needed. Check and remove if unused.

3. **Update BatchSolverKernel imports if np_ceil no longer needed**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify imports (around line 16)
   - Details:
     After chunk_run uses floor division, np_ceil may be unused.
     Check and remove if so.

**Tests to Create**:
- None (cleanup task, run existing tests to verify no regressions)

**Tests to Run**:
- tests/memory/test_memmgmt.py
- tests/batchsolving/arrays/test_basearraymanager.py

**Outcomes**: 


---

## Task Group 9: Test Updates for Chunking Edge Cases
**Status**: [ ]
**Dependencies**: Task Groups 1-8

**Required Context**:
- File: tests/memory/test_memmgmt.py (entire file)
- File: tests/batchsolving/arrays/test_basearraymanager.py (entire file)

**Input Validation Required**:
- None (testing only)

**Tasks**:
1. **Add edge case tests for uneven chunk division**
   - File: tests/memory/test_memmgmt.py
   - Action: Add new test functions
   - Details:
     ```python
     def test_chunk_calculation_5_runs_4_chunks():
         """Test that 5 runs with 4 chunks produces correct chunk sizes.
         
         With floor division: chunk_size = 5 // 4 = 1
         Chunks process runs [0], [1], [2], [3], [4] (last chunk is partial)
         No index overflow should occur.
         """
         # Setup memory manager with limited memory forcing 4 chunks
         # Verify chunk_size calculation
         # Verify no indices exceed array bounds
     
     def test_all_arrays_unchunkable_produces_one_chunk():
         """Test that when all arrays are unchunkable, num_chunks = 1."""
         # All requests have unchunkable=True
         # Verify num_chunks == 1
     
     def test_final_chunk_has_correct_indices():
         """Test final chunk indices don't exceed array bounds."""
         # 5 runs, 2 chunks -> chunk_size = 2
         # Chunk 0: indices [0, 2)
         # Chunk 1: indices [2, 4) <- only 3 actual runs remain
         # Verify final chunk processes correct number of runs
     ```

2. **Add integration test for chunked_shape propagation**
   - File: tests/batchsolving/arrays/test_basearraymanager.py
   - Action: Add new test function
   - Details:
     ```python
     def test_chunked_shape_propagates_through_allocation():
         """Test chunked_shape flows from MemoryManager to ManagedArray."""
         # Create manager with arrays requiring chunking
         # Trigger allocation
         # Verify ManagedArray.chunked_shape matches expected
         # Verify needs_chunked_transfer returns correct value
     ```

**Tests to Create**:
- Test file: tests/memory/test_memmgmt.py
- Test function: test_chunk_calculation_5_runs_4_chunks
- Description: Verify 5 runs / 4 chunks handles remainder correctly
- Test function: test_all_arrays_unchunkable_produces_one_chunk
- Description: Verify unchunkable-only requests produce 1 chunk
- Test function: test_final_chunk_has_correct_indices
- Description: Verify final chunk indices don't overflow
- Test file: tests/batchsolving/arrays/test_basearraymanager.py
- Test function: test_chunked_shape_propagates_through_allocation
- Description: Verify chunked_shape propagates from MemoryManager to ManagedArray

**Tests to Run**:
- tests/memory/test_memmgmt.py::test_chunk_calculation_5_runs_4_chunks
- tests/memory/test_memmgmt.py::test_all_arrays_unchunkable_produces_one_chunk
- tests/memory/test_memmgmt.py::test_final_chunk_has_correct_indices
- tests/batchsolving/arrays/test_basearraymanager.py::test_chunked_shape_propagates_through_allocation

**Outcomes**: 


---

# Summary

## Total Task Groups: 9

## Dependency Chain Overview:
```
Task Group 1 (ArrayResponse) ─────┬───> Task Group 3 (MemoryManager) ───┐
                                  │                                      │
Task Group 2 (ManagedArray) ──────┼───> Task Group 4 (BaseArrayManager)──┤
                                  │                                      │
                                  │     Task Group 5 (BatchInputArrays) ─┤
                                  │                                      │
                                  │     Task Group 6 (BatchOutputArrays)─┤
                                  │                                      │
                                  └───> Task Group 7 (BatchSolverKernel)─┤
                                                                         │
                                        Task Group 8 (Cleanup) ──────────┤
                                                                         │
                                        Task Group 9 (Edge Case Tests) ──┘
```

## Tests to Create:
- 4 new tests in tests/memory/test_array_requests.py (or test_memmgmt.py)
- 4 new tests in tests/batchsolving/arrays/test_basearraymanager.py
- 7 new tests in tests/memory/test_memmgmt.py
- 1 new test in tests/batchsolving/arrays/test_batchinputarrays.py
- 2 new tests in tests/batchsolving/arrays/test_batchoutputarrays.py
- 2 new tests in tests/batchsolving/test_batchsolverkernel.py
- 4 edge case tests in tests/memory/test_memmgmt.py and test_basearraymanager.py

## Tests to Run (per group):
Each task group specifies exact pytest paths for the run_tests agent.

## Estimated Complexity:
- **Task Groups 1-2**: Low complexity (adding fields)
- **Task Group 3**: High complexity (core logic refactoring)
- **Task Groups 4-6**: Medium complexity (propagation and simplification)
- **Task Group 7**: Medium complexity (alignment with MemoryManager)
- **Task Groups 8-9**: Low complexity (cleanup and testing)

## Important Notes:
1. **DO NOT run solver.solve() or kernel.run() tests** - they hang in test environment
2. Only test array chunking mechanics directly
3. Use floor division throughout (not ceiling) to prevent overflow
4. chunked_shape in ManagedArray enables simple shape comparison for transfer logic
5. All legacy is_chunked flag usage should be replaced with needs_chunked_transfer property
