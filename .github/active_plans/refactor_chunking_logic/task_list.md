# Implementation Task List
# Feature: Refactor Chunking Logic
# Plan Reference: .github/active_plans/refactor_chunking_logic/agent_plan.md

## Task Group 1: Add Chunk Metadata Fields to ManagedArray
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: src/cubie/batchsolving/arrays/BaseArrayManager.py (lines 40-144)
- File: .github/context/cubie_internal_structure.md (entire file)

**Input Validation Required**:
- chunk_length: Validate type is int, value > 0 (only when not None)
- num_chunks: Validate type is int, value > 0 (only when not None)

**Tasks**:
1. **Add chunk_length field to ManagedArray**
   - File: src/cubie/batchsolving/arrays/BaseArrayManager.py
   - Action: Modify
   - Details:
     ```python
     # Add after line 82 (after chunked_shape field):
     chunk_length: Optional[int] = field(
         default=None,
         validator=attrsval_optional(attrsval_instance_of(int)),
     )
     ```
   - Edge cases: None value is valid (array not chunked)
   - Integration: Will be populated by BaseArrayManager._on_allocation_complete

2. **Add num_chunks field to ManagedArray**
   - File: src/cubie/batchsolving/arrays/BaseArrayManager.py
   - Action: Modify
   - Details:
     ```python
     # Add after chunk_length field:
     num_chunks: Optional[int] = field(
         default=None,
         validator=attrsval_optional(attrsval_instance_of(int)),
     )
     ```
   - Edge cases: None value is valid (array not chunked)
   - Integration: Will be populated by BaseArrayManager._on_allocation_complete

**Tests to Create**:
- Test file: tests/batchsolving/arrays/test_managed_array.py
- Test function: test_managed_array_chunk_fields_default_none
- Description: Verify that chunk_length and num_chunks default to None
- Test function: test_managed_array_chunk_fields_accept_valid_values
- Description: Verify that valid chunk_length and num_chunks can be set

**Tests to Run**:
- tests/batchsolving/arrays/test_managed_array.py::test_managed_array_chunk_fields_default_none
- tests/batchsolving/arrays/test_managed_array.py::test_managed_array_chunk_fields_accept_valid_values

**Outcomes**:

---

## Task Group 2: Modify ManagedArray.chunk_slice() Method
**Status**: [ ]
**Dependencies**: Group 1

**Required Context**:
- File: src/cubie/batchsolving/arrays/BaseArrayManager.py (lines 123-130, full ManagedArray class lines 40-144)
- File: .github/context/cubie_internal_structure.md (entire file)

**Input Validation Required**:
- chunk_index: Validate type is int, value >= 0
- chunk_index: Validate chunk_index < num_chunks when num_chunks is not None

**Tasks**:
1. **Update chunk_slice() signature and implementation**
   - File: src/cubie/batchsolving/arrays/BaseArrayManager.py
   - Action: Modify
   - Details:
     ```python
     # Replace existing chunk_slice method (lines 123-130):
     def chunk_slice(self, chunk_index: int) -> tuple[slice, ...]:
         )
         
         # Keep _chunk_axis_index as is
     ```
   - Edge cases: All fields are optional and default to None for non-chunked arrays
   - Integration: These fields will be populated by BaseArrayManager from ArrayResponse

**Tests to Create**:
- Test file: tests/batchsolving/arrays/test_basearraymanager.py
- Test function: test_managed_array_chunk_parameters_defaults
- Description: Verify chunk parameter fields default to None on instantiation

**Tests to Run**:
- tests/batchsolving/arrays/test_basearraymanager.py::test_managed_array_chunk_parameters_defaults

**Outcomes**:

---

## Task Group 2: Enhance ManagedArray.chunk_slice() Method
**Status**: [ ]
**Dependencies**: Groups [1]

**Required Context**:
- File: src/cubie/batchsolving/arrays/BaseArrayManager.py (lines 40-144)
- File: src/cubie/memory/mem_manager.py (lines 1381-1436)
- File: .github/context/cubie_internal_structure.md (lines 1-100)

**Input Validation Required**:
- chunk_index: Check type is int, 0 <= chunk_index < num_chunks (when chunking is active)

**Tasks**:
1. **Replace chunk_slice() signature and implementation**
   - File: src/cubie/batchsolving/arrays/BaseArrayManager.py
   - Action: Modify
   - Details:
     ```python
     def chunk_slice(self, chunk_index: int) -> Union[ndarray, DeviceNDArrayBase]:
         """Return a slice of the array for the specified chunk index.
         
         Parameters
         ----------
         chunk_index
             Zero-based index of the chunk to slice.
             
         Returns
         -------
         Union[ndarray, DeviceNDArrayBase]
             View or slice of the array for the specified chunk.
             
         Notes
         -----
         When chunking is inactive (is_chunked=False or _chunk_axis_index=None),
         returns the full array. Otherwise computes slice based on stored
         chunk parameters and _chunk_axis_index.
         """
         # Fast path: no chunking
         if self._chunk_axis_index is None or self.is_chunked is False:
             return self.array
         
         # Fast path: no chunk parameters set (shouldn't happen but safe)
         if self.num_chunks is None or self.chunk_length is None:
             return self.array
         
         # Validate chunk_index
         if not isinstance(chunk_index, int):
             raise TypeError(f"chunk_index must be int, got {type(chunk_index)}")
         if chunk_index < 0 or chunk_index >= self.num_chunks:
             raise ValueError(
                 f"chunk_index {chunk_index} out of range [0, {self.num_chunks})"
             )
         
         # Compute slice based on chunk parameters
         start = chunk_index * self.chunk_length
         
         # Final chunk may be shorter
         if chunk_index == self.num_chunks - 1 and self.dangling_chunk_length is not None:
             end = start + self.dangling_chunk_length
         else:
             end = start + self.chunk_length
         
         # Build slice tuple
         chunk_slice = [slice(None)] * len(self.shape)
         chunk_slice[self._chunk_axis_index] = slice(start, end)
         
         return self.array[tuple(chunk_slice)]
     ```
   - Edge cases:
     - chunk_index out of range → ValueError
     - _chunk_axis_index is None → return full array
     - is_chunked is False → return full array
     - Final chunk with dangling_chunk_length → use dangling length
     - Single chunk (num_chunks=1) → return slice(0, axis_length)
   - Integration: Called by InputArrays.initialise() and OutputArrays.finalise()

**Tests to Create**:
- Test file: tests/batchsolving/arrays/test_basearraymanager.py
- Test function: test_chunk_slice_no_chunking_returns_full_array
- Description: Verify chunk_slice returns full array when is_chunked=False
- Test function: test_chunk_slice_computes_correct_slices
- Description: Verify chunk_slice computes correct start/end for each chunk
- Test function: test_chunk_slice_handles_dangling_final_chunk
- Description: Verify final chunk uses dangling_chunk_length
- Test function: test_chunk_slice_validates_chunk_index
- Description: Verify chunk_slice raises ValueError for invalid chunk_index

**Tests to Run**:
- tests/batchsolving/arrays/test_basearraymanager.py::test_chunk_slice_no_chunking_returns_full_array
- tests/batchsolving/arrays/test_basearraymanager.py::test_chunk_slice_computes_correct_slices
- tests/batchsolving/arrays/test_basearraymanager.py::test_chunk_slice_handles_dangling_final_chunk
- tests/batchsolving/arrays/test_basearraymanager.py::test_chunk_slice_validates_chunk_index

**Outcomes**:

---

## Task Group 3: Update BaseArrayManager Allocation Callback
**Status**: [ ]
**Dependencies**: Groups [1, 2]

**Required Context**:
- File: src/cubie/batchsolving/arrays/BaseArrayManager.py (lines 337-388)
- File: src/cubie/memory/array_requests.py (lines 91-136)
- File: .github/context/cubie_internal_structure.md (lines 1-100)

**Input Validation Required**:
- None (ArrayResponse fields validated by attrs)

**Tasks**:
1. **Modify _on_allocation_complete() to store chunk parameters**
   - File: src/cubie/batchsolving/arrays/BaseArrayManager.py
   - Action: Modify
   - Details:
     ```python
     def _on_allocation_complete(self, response: ArrayResponse) -> None:
         """Callback for when the allocation response is received.
         
         Parameters
         ----------
         response
             Response object containing allocated arrays and metadata.
         
         Warns
         -----
         UserWarning
             If a device array is not found in the allocation response.
         
         Returns
         -------
         None
         
         Notes
         -----
         Warnings are only issued if the response contains some arrays but
         not the expected one, indicating a potential allocation mismatch.
         
         Stores chunk parameters from response in ManagedArray objects for
         both host and device containers.
         """
         chunked_shapes = response.chunked_shapes
         arrays = response.arr
         
         # Extract chunk parameters from response
         chunks = response.chunks
         axis_length = response.axis_length
         chunk_length = response.chunk_length
         dangling_chunk_length = response.dangling_chunk_length
         
         for array_label in self._needs_reallocation:
             try:
                 self.device.attach(array_label, arrays[array_label])
                 # Store chunked_shape and chunk parameters in ManagedArray
                 if array_label in response.chunked_shapes:
                     for container in (self.device, self.host):
                         array = container.get_managed_array(array_label)
                         array.chunked_shape = chunked_shapes[array_label]
                         # Store chunk parameters (remove chunked_slice_fn)
                         array.axis_length = axis_length
                         array.chunk_length = chunk_length
                         array.num_chunks = chunks
                         array.dangling_chunk_length = dangling_chunk_length
             except KeyError:
                 warn(
                     f"Device array {array_label} not found in allocation "
                     f"response. See "
                     f"BaseArrayManager._on_allocation_complete docstring "
                     f"for more info.",
                     UserWarning,
                 )
         
         self._chunks = response.chunks
         if self.is_chunked:
             self._convert_host_to_numpy()
         else:
             self._convert_host_to_pinned()
         self._needs_reallocation.clear()
     ```
   - Edge cases:
     - Response without chunked_shapes (unchunkable arrays) → parameters still stored
     - Missing array in response → warn but continue
   - Integration: Replaces storage of chunked_slice_fn with chunk parameters

**Tests to Create**:
- Test file: tests/batchsolving/arrays/test_basearraymanager.py
- Test function: test_on_allocation_complete_stores_chunk_parameters
- Description: Verify chunk parameters stored in ManagedArray from ArrayResponse

**Tests to Run**:
- tests/batchsolving/arrays/test_basearraymanager.py::test_on_allocation_complete_stores_chunk_parameters

**Outcomes**:

---

## Task Group 4: Update InputArrays.initialise() to Use New chunk_slice()
**Status**: [ ]
**Dependencies**: Groups [1, 2, 3]

**Required Context**:
- File: src/cubie/batchsolving/arrays/BatchInputArrays.py (lines 275-332)
- File: src/cubie/batchsolving/arrays/BaseArrayManager.py (lines 40-144)
- File: .github/context/cubie_internal_structure.md (lines 1-100)

**Input Validation Required**:
- chunk_index: Validated by ManagedArray.chunk_slice()

**Tasks**:
1. **Replace chunked_slice_fn call with chunk_slice method**
   - File: src/cubie/batchsolving/arrays/BatchInputArrays.py
   - Action: Modify
   - Details:
     ```python
     def initialise(self, chunk_index: int) -> None:
         """Copy a batch chunk of host data to device buffers.
         
         Parameters
         ----------
         chunk_index
             Index of the chunk being initialized.
         
         Returns
         -------
         None
             Host slices are staged into device arrays in place.
         
         Notes
         -----
         For chunked mode, pinned buffers are acquired from the pool for
         staging data before H2D transfer. Buffers are stored in
         _active_buffers and released after the H2D transfer completes.
         For non-chunked mode, pinned buffers are allocated directly.
         """
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
             
             # Direct transfer when shapes match; chunked transfer otherwise
             if not host_obj.needs_chunked_transfer:
                 from_.append(host_obj.array)
             else:
                 # NEW: Call chunk_slice method instead of chunked_slice_fn
                 host_slice = host_obj.chunk_slice(chunk_index)
                 
                 # Chunked mode: use buffer pool for pinned staging
                 # Buffer must match device array shape for H2D copy
                 device_shape = device_obj.array.shape
                 buffer = self._buffer_pool.acquire(
                     array_name, device_shape, host_slice.dtype
                 )
                 # Copy host slice into smallest indices of buffer,
                 # as the final host slice may be smaller than the buffer.
                 data_slice = tuple(slice(0, s) for s in host_slice.shape)
                 buffer.array[data_slice] = host_slice
                 from_.append(buffer.array)
                 # Record that we're using this buffer for later release.
                 self._active_buffers.append(buffer)
         
         self.to_device(from_, to_)
     ```
   - Edge cases:
     - Non-chunked mode → needs_chunked_transfer is False, direct copy
     - Final chunk smaller than buffer → data_slice handles this
   - Integration: chunk_index passed from BatchSolverKernel.run()

**Tests to Create**:
- Test file: tests/batchsolving/arrays/test_batchinputarrays.py
- Test function: test_initialise_uses_chunk_slice_method
- Description: Verify initialise calls chunk_slice instead of chunked_slice_fn

**Tests to Run**:
- tests/batchsolving/arrays/test_batchinputarrays.py::test_initialise_uses_chunk_slice_method
- tests/batchsolving/arrays/test_chunking.py::test_chunked_solve_produces_correct_results

**Outcomes**:

---

## Task Group 5: Update OutputArrays.finalise() to Use New chunk_slice()
**Status**: [ ]
**Dependencies**: Groups [1, 2, 3]

**Required Context**:
- File: src/cubie/batchsolving/arrays/BatchOutputArrays.py (lines 344-416)
- File: src/cubie/batchsolving/arrays/BaseArrayManager.py (lines 40-144)
- File: .github/context/cubie_internal_structure.md (lines 1-100)

**Input Validation Required**:
- chunk_index: Validated by ManagedArray.chunk_slice()

**Tasks**:
1. **Replace chunked_slice_fn call with chunk_slice method**
   - File: src/cubie/batchsolving/arrays/BatchOutputArrays.py
   - Action: Modify
   - Details:
     ```python
     def finalise(self, chunk_index: int) -> None:
         """Queue device-to-host transfers for a chunk.
         
         Parameters
         ----------
         chunk_index
             Index of the chunk being finalized.
         
         Returns
         -------
         None
             Queues async transfers. For chunked mode, submits writeback
             tasks to the watcher thread for non-blocking completion.
         
         Notes
         -----
         Host slices are made contiguous before transfer to ensure
         compatible strides with device arrays. For chunked mode, data
         is transferred to pooled pinned buffers and submitted to the
         watcher thread for async writeback. For non-chunked mode,
         the writeback call is made immediately (but will happen
         asynchronously).
         """
         from_ = []
         to_ = []
         stream = self._memory_manager.get_stream(self)
         
         for array_name, slot in self.host.iter_managed_arrays():
             device_array = self.device.get_array(array_name)
             host_array = slot.array
             
             to_target = host_array
             from_target = device_array
             if slot.needs_chunked_transfer:
                 # NEW: Call chunk_slice method instead of chunked_slice_fn
                 host_slice = slot.chunk_slice(chunk_index)
                 
                 # Chunked mode: use buffer pool and watcher
                 # Buffer must match device array shape for D2H copy
                 buffer = self._buffer_pool.acquire(
                     array_name, device_array.shape, host_slice.dtype
                 )
                 # Set pinned buffer as target and register for writeback
                 to_target = buffer.array
                 
                 # Compute slice tuple for writeback (same as chunk_slice internals)
                 chunk_slice_tuple = [slice(None)] * len(host_array.shape)
                 start = chunk_index * slot.chunk_length
                 if chunk_index == slot.num_chunks - 1 and slot.dangling_chunk_length is not None:
                     end = start + slot.dangling_chunk_length
                 else:
                     end = start + slot.chunk_length
                 chunk_slice_tuple[slot._chunk_axis_index] = slice(start, end)
                 slice_tuple = tuple(chunk_slice_tuple)
                 
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
             self._pending_buffers.clear()
     ```
   - Edge cases:
     - Non-chunked mode → needs_chunked_transfer is False, direct copy
     - Final chunk smaller than buffer → data_shape handles this
     - CUDA simulation mode → event is None
   - Integration: chunk_index passed from BatchSolverKernel.run()

**Tests to Create**:
- Test file: tests/batchsolving/arrays/test_batchoutputarrays.py
- Test function: test_finalise_uses_chunk_slice_method
- Description: Verify finalise calls chunk_slice and computes slice_tuple correctly

**Tests to Run**:
- tests/batchsolving/arrays/test_batchoutputarrays.py::test_finalise_uses_chunk_slice_method
- tests/batchsolving/arrays/test_chunking.py::test_chunked_solve_produces_correct_results

**Outcomes**:

---

## Task Group 6: Remove compute_per_chunk_slice from MemoryManager
**Status**: [ ]
**Dependencies**: Groups [1, 2, 3, 4, 5]

**Required Context**:
- File: src/cubie/memory/mem_manager.py (lines 1170-1260, 1381-1436)
- File: src/cubie/memory/array_requests.py (lines 91-136)
- File: .github/context/cubie_internal_structure.md (lines 1-100)

**Input Validation Required**:
- None

**Tasks**:
1. **Remove compute_per_chunk_slice function**
   - File: src/cubie/memory/mem_manager.py
   - Action: Delete
   - Details:
     ```python
     # DELETE ENTIRE FUNCTION (lines 1381-1436):
     # def compute_per_chunk_slice(...) -> dict[str, Callable]:
     #     ...
     ```
   - Edge cases: None (complete removal)
   - Integration: Function is no longer called after Group 4 and 5 changes

2. **Remove chunked_slices from allocate_queue()**
   - File: src/cubie/memory/mem_manager.py
   - Action: Modify
   - Details:
     ```python
     def allocate_queue(
         self,
         triggering_instance: object,
     ) -> None:
         """Process all queued requests for a stream group with coordinated chunking.
         
         Chunking is performed along the run axis when memory
         constraints require splitting the batch.
         
         Parameters
         ----------
         triggering_instance
             The instance that triggered queue processing.
         
         Notes
         -----
         Processes all pending requests in the same stream group, applying
         coordinated chunking based on available memory. Calls
         allocation_ready_hook for each instance with their results.
         
         Returns
         -------
         None
         """
         stream_group = self.get_stream_group(triggering_instance)
         stream = self.get_stream(triggering_instance)
         queued_requests = self._queued_allocations.pop(stream_group, {})
         
         # Get length of axis to chunk
         axis_length = 0
         for instance_id, requests_dict in queued_requests.items():
             axis_length = get_chunk_axis_length(requests_dict)
             if axis_length > 0:
                 break
         
         chunk_length, num_chunks = self.get_chunk_parameters(
             queued_requests, axis_length, stream_group
         )
         peers = self.stream_groups.get_instances_in_group(stream_group)
         notaries = set(peers) - set(queued_requests.keys())
         for instance_id, requests_dict in queued_requests.items():
             chunked_shapes = self.compute_chunked_shapes(
                 requests_dict,
                 chunk_length,
             )
             # REMOVE: chunked_slices = compute_per_chunk_slice(...)
             dangling_chunk_length = (
                 axis_length - (num_chunks - 1) * chunk_length
             )
             
             chunked_requests = deepcopy(requests_dict)
             for key, request in chunked_requests.items():
                 request.shape = chunked_shapes[key]
             
             arrays = self.allocate_all(
                 chunked_requests, instance_id, stream=stream
             )
             response = ArrayResponse(
                 arr=arrays,
                 chunks=num_chunks,
                 axis_length=axis_length,
                 chunk_length=chunk_length,
                 dangling_chunk_length=dangling_chunk_length,
                 chunked_shapes=chunked_shapes,
                 # REMOVE: chunked_slices=chunked_slices,
             )
             
             self.registry[instance_id].allocation_ready_hook(response)
             for peer in notaries:
                 self.registry[peer].allocation_ready_hook(
                     ArrayResponse(
                         arr={},
                         chunks=num_chunks,
                         axis_length=axis_length,
                         chunk_length=chunk_length,
                         dangling_chunk_length=dangling_chunk_length,
                         chunked_shapes={},
                         # REMOVE: chunked_slices={},
                     )
                 )
         
         return None
     ```
   - Edge cases: Ensure both regular allocation and notary allocation skip chunked_slices
   - Integration: ArrayResponse no longer contains chunked_slices field

**Tests to Create**:
- Test file: tests/memory/test_memmgmt.py
- Test function: test_allocate_queue_no_chunked_slices_in_response
- Description: Verify ArrayResponse from allocate_queue does not have chunked_slices

**Tests to Run**:
- tests/memory/test_memmgmt.py::test_allocate_queue_no_chunked_slices_in_response
- tests/batchsolving/arrays/test_chunking.py::test_chunked_solve_produces_correct_results

**Outcomes**:

---

## Task Group 7: Remove chunked_slices from ArrayResponse
**Status**: [ ]
**Dependencies**: Groups [6]

**Required Context**:
- File: src/cubie/memory/array_requests.py (lines 91-136)
- File: .github/context/cubie_internal_structure.md (lines 1-100)

**Input Validation Required**:
- None

**Tasks**:
1. **Remove chunked_slices field from ArrayResponse**
   - File: src/cubie/memory/array_requests.py
   - Action: Modify
   - Details:
     ```python
     @attrs.define
     class ArrayResponse:
         """Result of an array allocation containing buffers and chunking data.
         
         Parameters
         ----------
         arr
             Dictionary mapping array labels to allocated device arrays.
         chunks
             Number of chunks the allocation was divided into.
         axis_length
             Full length of the run axis before chunking.
         chunk_length
             Length of each chunk along the run axis (except possibly last).
         dangling_chunk_length
             Length of the final chunk if different from chunk_length.
         chunked_shapes
             Mapping from array labels to their per-chunk shapes. Empty dict when
             no chunking occurs.
         
         Attributes
         ----------
         arr
             Dictionary mapping array labels to allocated device arrays.
         chunks
             Number of chunks the allocation was divided into.
         axis_length
             Full length of the run axis before chunking.
         chunk_length
             Length of each chunk along the run axis.
         dangling_chunk_length
             Length of the final chunk if different from chunk_length.
         chunked_shapes
             Mapping from array labels to their per-chunk shapes.
         """
         
         arr: dict[str, DeviceNDArrayBase] = attrs.field(
             default=attrs.Factory(dict), validator=val.instance_of(dict)
         )
         chunks: int = attrs.field(
             default=1,
         )
         axis_length: int = attrs.field(
             default=1,
         )
         chunk_length: int = attrs.field(
             default=1,
         )
         chunked_shapes: dict[str, tuple[int, ...]] = attrs.field(
             default=attrs.Factory(dict), validator=val.instance_of(dict)
         )
         # REMOVE THIS FIELD:
         # chunked_slices: dict[str, Callable] = attrs.field(
         #     default=attrs.Factory(dict), validator=val.instance_of(dict)
         # )
         dangling_chunk_length: int = attrs.field(
             default=0,
         )
     ```
   - Edge cases: None (field removal)
   - Integration: No code should reference chunked_slices after Groups 4, 5, and 6

**Tests to Create**:
- Test file: tests/memory/test_array_requests.py
- Test function: test_array_response_no_chunked_slices_field
- Description: Verify ArrayResponse does not have chunked_slices attribute

**Tests to Run**:
- tests/memory/test_array_requests.py::test_array_response_no_chunked_slices_field
- tests/batchsolving/arrays/test_chunking.py::test_chunked_solve_produces_correct_results

**Outcomes**:

---

## Task Group 8: Remove chunked_slice_fn from ManagedArray
**Status**: [ ]
**Dependencies**: Groups [6, 7]

**Required Context**:
- File: src/cubie/batchsolving/arrays/BaseArrayManager.py (lines 40-144)
- File: .github/context/cubie_internal_structure.md (lines 1-100)

**Input Validation Required**:
- None

**Tasks**:
1. **Remove chunked_slice_fn field from ManagedArray**
   - File: src/cubie/batchsolving/arrays/BaseArrayManager.py
   - Action: Modify
   - Details:
     ```python
     @define(slots=False)
     class ManagedArray:
         """Metadata wrapper for a single managed array."""
         
         dtype: type = field(
             default=np_float32, validator=attrsval_instance_of(type)
         )
         stride_order: tuple[str, ...] = field(
             factory=tuple,
             validator=attrsval_deep_iterable(
                 member_validator=attrsval_instance_of(str),
                 iterable_validator=attrsval_instance_of(tuple),
             ),
         )
         default_shape: tuple[Optional[int], ...] = field(
             factory=tuple,
             validator=attrsval_deep_iterable(
                 member_validator=opt_gttype_validator(int, 0),
                 iterable_validator=attrsval_instance_of(tuple),
             ),
         )
         memory_type: str = field(
             default="device",
             validator=attrsval_in(
                 ["device", "mapped", "pinned", "managed", "host"]
             ),
         )
         is_chunked: bool = field(
             default=True, validator=attrsval_instance_of(bool)
         )
         _array: Optional[Union[NDArray, DeviceNDArrayBase]] = field(
             default=None,
             repr=False,
         )
         chunked_shape: Optional[tuple[int, ...]] = field(
             default=None,
             validator=attrsval_optional(
                 attrsval_deep_iterable(
                     member_validator=attrsval_instance_of(int),
                     iterable_validator=attrsval_instance_of(tuple),
                 )
             ),
         )
         # REMOVE THIS FIELD:
         # chunked_slice_fn: Optional[Callable] = field(
         #     default=None,
         #     repr=False,
         #     eq=False,
         # )
         _chunk_axis_index: Optional[int] = field(
             default=None,
             init=False,
             repr=False,
         )
         axis_length: Optional[int] = field(
             default=None,
             validator=attrsval_optional(attrsval_instance_of(int)),
         )
         chunk_length: Optional[int] = field(
             default=None,
             validator=attrsval_optional(attrsval_instance_of(int)),
         )
         num_chunks: Optional[int] = field(
             default=None,
             validator=attrsval_optional(attrsval_instance_of(int)),
         )
         dangling_chunk_length: Optional[int] = field(
             default=None,
             validator=attrsval_optional(attrsval_instance_of(int)),
         )
         
         # __attrs_post_init__ remains unchanged
     ```
   - Edge cases: None (field removal)
   - Integration: No code should reference chunked_slice_fn after Groups 4 and 5

**Tests to Create**:
- Test file: tests/batchsolving/arrays/test_basearraymanager.py
- Test function: test_managed_array_no_chunked_slice_fn_field
- Description: Verify ManagedArray does not have chunked_slice_fn attribute

**Tests to Run**:
- tests/batchsolving/arrays/test_basearraymanager.py::test_managed_array_no_chunked_slice_fn_field
- tests/batchsolving/arrays/test_chunking.py::test_chunked_solve_produces_correct_results

**Outcomes**:

---

## Task Group 9: Remove stride_order from ArrayRequest (Optional Deprecation Path)
**Status**: [ ]
**Dependencies**: Groups [1, 2, 3, 4, 5, 6, 7, 8]

**Required Context**:
- File: src/cubie/memory/array_requests.py (lines 18-89)
- File: src/cubie/memory/mem_manager.py (lines 1439-1572)
- File: src/cubie/batchsolving/arrays/BaseArrayManager.py (lines 40-144)
- File: .github/context/cubie_internal_structure.md (lines 1-100)

**Input Validation Required**:
- None (this is a field removal/deprecation)

**Tasks**:
1. **Verify stride_order is no longer needed for chunking**
   - File: src/cubie/memory/mem_manager.py
   - Action: Modify
   - Details:
     - Review is_request_chunkable() function (lines 1506-1543)
     - Review get_chunk_axis_length() function (lines 1439-1461)
     - Review replace_with_chunked_size() function (lines 1546-1572)
     - All three functions use request.stride_order.index("run")
     - Determine if these can be replaced with simpler logic
     
     Current issue: These functions are called BEFORE ManagedArray exists, so _chunk_axis_index
     is not yet available. stride_order is used during allocation planning.
     
     Decision: KEEP stride_order in ArrayRequest for now. It's used during memory planning
     phase, not just for slice generation. Removing it would require deeper architectural
     changes to how chunking decisions are made.
     
     Alternative approach:
     - Mark stride_order usage as "allocation planning only" in comments
     - Clarify that stride_order is NOT used for slice generation (that's ManagedArray's job)
     - Add comment explaining separation of concerns

2. **Update comments to clarify stride_order purpose**
   - File: src/cubie/memory/array_requests.py
   - Action: Modify
   - Details:
     ```python
     @attrs.define
     class ArrayRequest:
         """Specification for requesting array allocation.
         
         Parameters
         ----------
         shape
             Tuple describing the requested array shape. Defaults to ``(1, 1, 1)``.
         dtype
             NumPy precision constructor used to produce the allocation. Defaults to
             :func:`numpy.float64`. Integer status buffers use :func:`numpy.int32`.
         memory
             Memory placement option. Must be one of ``"device"``, ``"mapped"``,
             ``"pinned"``, or ``"managed"``.
         stride_order
             Optional tuple describing logical dimension labels in stride order.
             Used by memory manager for allocation planning and chunking decisions.
             Not used for slice generation (ManagedArray handles that).
         unchunkable
             Whether the memory manager is allowed to chunk the allocation.
         
         Attributes
         ----------
         shape
             Tuple describing the requested array shape.
         dtype
             NumPy precision constructor used to produce the allocation.
         memory
             Memory placement option.
         stride_order
             Tuple describing logical dimension labels in stride order.
             Used for allocation planning only; slice generation delegated to ManagedArray.
         unchunkable
             Flag indicating that chunking should be disabled.
         """
     ```
   - Edge cases: None (documentation update)
   - Integration: Clarifies that stride_order is for planning, not execution

3. **Add clarifying comments to memory manager functions**
   - File: src/cubie/memory/mem_manager.py
   - Action: Modify
   - Details:
     ```python
     def is_request_chunkable(request) -> bool:
         """Determine if a single ArrayRequest is chunkable along the run axis.
         
         Used during allocation planning phase to determine chunking strategy.
         Actual slice generation is delegated to ManagedArray.chunk_slice().
         
         Parameters
         ----------
         request
             The ArrayRequest to evaluate.
         
         Returns
         -------
         bool
             True if the request is chunkable along the "run" axis, False otherwise.
         
         Notes
         -----
         A request is considered chunkable if:
         - request.unchunkable is False
         - the array has a "run" axis
         """
         # ... implementation unchanged ...
     ```
   - Edge cases: None (documentation update)
   - Integration: Clarifies separation between planning (MemoryManager) and execution (ManagedArray)

**Tests to Create**:
- None (documentation-only changes)

**Tests to Run**:
- tests/batchsolving/arrays/test_chunking.py::test_chunked_solve_produces_correct_results
- tests/memory/test_memmgmt.py

**Outcomes**:

---

## Task Group 10: Update Existing Tests
**Status**: [ ]
**Dependencies**: Groups [1, 2, 3, 4, 5, 6, 7, 8, 9]

**Required Context**:
- File: tests/batchsolving/arrays/test_chunking.py (entire file)
- File: tests/batchsolving/arrays/test_basearraymanager.py (entire file)
- File: tests/batchsolving/arrays/test_batchinputarrays.py (entire file)
- File: tests/batchsolving/arrays/test_batchoutputarrays.py (entire file)
- File: tests/memory/test_memmgmt.py (entire file)
- File: tests/memory/test_array_requests.py (entire file)

**Input Validation Required**:
- None (test updates)

**Tasks**:
1. **Update tests that reference chunked_slice_fn**
   - File: Multiple test files
   - Action: Modify
   - Details:
     - Search for references to chunked_slice_fn in tests
     - Replace assertions about chunked_slice_fn with assertions about chunk parameters
     - Example:
       ```python
       # OLD:
       # assert managed_array.chunked_slice_fn is not None
       
       # NEW:
       assert managed_array.num_chunks is not None
       assert managed_array.chunk_length is not None
       ```
   - Edge cases: Tests that directly call chunked_slice_fn → update to call chunk_slice(i)
   - Integration: Ensures test suite passes with new architecture

2. **Update tests that inspect ArrayResponse.chunked_slices**
   - File: Multiple test files
   - Action: Modify
   - Details:
     - Search for references to response.chunked_slices
     - Remove or update these assertions
     - Example:
       ```python
       # OLD:
       # assert 'array_name' in response.chunked_slices
       
       # NEW:
       assert response.chunk_length > 0
       assert response.num_chunks > 0
       ```
   - Edge cases: None
   - Integration: Ensures test suite validates chunk parameters instead of slice functions

3. **Verify all chunking integration tests pass**
   - File: tests/batchsolving/arrays/test_chunking.py
   - Action: Verify (no modification needed if Groups 1-9 implemented correctly)
   - Details:
     - Run all tests in test_chunking.py
     - These are end-to-end tests that verify chunking produces correct results
     - Should pass without modification if refactoring is correct
   - Edge cases: Tests may need timeout adjustments for CI
   - Integration: Final validation of refactoring correctness

**Tests to Create**:
- None (updates to existing tests)

**Tests to Run**:
- tests/batchsolving/arrays/test_chunking.py
- tests/batchsolving/arrays/test_basearraymanager.py
- tests/batchsolving/arrays/test_batchinputarrays.py
- tests/batchsolving/arrays/test_batchoutputarrays.py
- tests/memory/test_memmgmt.py
- tests/memory/test_array_requests.py

**Outcomes**:

---

## Summary

### Task Groups Overview
1. **Group 1**: Add chunk parameter fields to ManagedArray (foundation)
2. **Group 2**: Enhance chunk_slice() to compute slices on-demand (core logic)
3. **Group 3**: Update allocation callback to store chunk parameters (integration)
4. **Group 4**: Update InputArrays to use new chunk_slice() (execution path 1)
5. **Group 5**: Update OutputArrays to use new chunk_slice() (execution path 2)
6. **Group 6**: Remove compute_per_chunk_slice from MemoryManager (cleanup)
7. **Group 7**: Remove chunked_slices from ArrayResponse (cleanup)
8. **Group 8**: Remove chunked_slice_fn from ManagedArray (cleanup)
9. **Group 9**: Clarify stride_order usage (documentation)
10. **Group 10**: Update existing tests (validation)

### Dependency Chain
```
Group 1 (foundation)
  ↓
Group 2 (core logic) ← depends on Group 1
  ↓
Group 3 (integration) ← depends on Groups 1, 2
  ↓
Groups 4, 5 (execution paths) ← depend on Groups 1, 2, 3
  ↓
Group 6 (cleanup MemoryManager) ← depends on Groups 1-5
  ↓
Group 7 (cleanup ArrayResponse) ← depends on Group 6
  ↓
Group 8 (cleanup ManagedArray) ← depends on Groups 6, 7
  ↓
Group 9 (documentation) ← depends on Groups 1-8
  ↓
Group 10 (test updates) ← depends on Groups 1-9
```

### Tests to Create Summary
- Group 1: 1 test (chunk parameter defaults)
- Group 2: 4 tests (chunk_slice behavior)
- Group 3: 1 test (allocation callback)
- Group 4: 1 test (InputArrays.initialise)
- Group 5: 1 test (OutputArrays.finalise)
- Group 6: 1 test (no chunked_slices in response)
- Group 7: 1 test (ArrayResponse field removal)
- Group 8: 1 test (ManagedArray field removal)
- Group 9: 0 tests (documentation only)
- Group 10: Updates to existing tests

**Total new tests**: 11
**Total test files to update**: 6

### Estimated Complexity
- **Low complexity**: Groups 1, 7, 8, 9 (data structure changes, cleanup, docs)
- **Medium complexity**: Groups 3, 4, 5, 6, 10 (integration, test updates)
- **High complexity**: Group 2 (core slicing logic with multiple edge cases)

### Key Architectural Improvements
1. **Separation of concerns**: MemoryManager handles allocation, ManagedArray handles slicing
2. **Simpler data flow**: Parameters flow one direction instead of complex closures
3. **Easier debugging**: Slice computation is explicit method call with visible parameters
4. **Better testability**: Can test slice computation independently of memory allocation
5. **Clearer responsibilities**: Each component has a single, well-defined role
