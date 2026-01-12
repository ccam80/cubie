# Implementation Task List
# Feature: Refactor Pinned Memory Usage in Chunking System
# Plan Reference: .github/active_plans/refactor_pinned_memory_chunking/agent_plan.md

## Task Group 1: Chunk Buffer Pool Infrastructure
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/memory/mem_manager.py (entire file - understand allocation patterns)
- File: src/cubie/batchsolving/arrays/BaseArrayManager.py (lines 1-100, understand ManagedArray)
- File: src/cubie/cuda_simsafe.py (lines 1-50, understand CUDA_SIMULATION flag)
- File: .github/context/cubie_internal_structure.md (entire file - architecture context)

**Input Validation Required**:
- `num_buffers`: Must be int >= 1
- `buffer_shape`: Must be tuple of positive ints
- `dtype`: Must be a valid numpy dtype

**Tasks**:
1. **Create ChunkBufferPool class**
   - File: src/cubie/memory/chunk_buffer_pool.py
   - Action: Create
   - Details:
     ```python
     """Reusable pinned buffer pool for chunked array transfers."""
     
     from typing import Dict, List, Optional, Tuple
     from threading import Lock
     
     from attrs import define, field
     from attrs.validators import instance_of as attrsval_instance_of
     from numba import cuda
     from numpy import ndarray
     from numpy import dtype as np_dtype
     
     from cubie.cuda_simsafe import CUDA_SIMULATION
     
     
     @define
     class PinnedBuffer:
         """Wrapper for a reusable pinned memory buffer.
         
         Attributes
         ----------
         buffer_id : int
             Unique identifier for this buffer.
         array : ndarray
             The pinned numpy array.
         in_use : bool
             Whether the buffer is currently in use.
         """
         
         buffer_id: int = field(validator=attrsval_instance_of(int))
         array: ndarray = field(validator=attrsval_instance_of(ndarray))
         in_use: bool = field(default=False, validator=attrsval_instance_of(bool))
     
     
     @define
     class ChunkBufferPool:
         """Pool of reusable pinned buffers for chunked transfers.
         
         Manages allocation and lifecycle of pinned memory buffers used
         for staging data during chunked device transfers. Buffers are
         sized for one chunk and reused across chunks.
         
         Attributes
         ----------
         _buffers : Dict[str, List[PinnedBuffer]]
             Pool of buffers organized by array name.
         _lock : Lock
             Thread-safe access to buffer pool.
         _next_id : int
             Counter for unique buffer IDs.
         """
         
         _buffers: Dict[str, List[PinnedBuffer]] = field(factory=dict)
         _lock: Lock = field(factory=Lock)
         _next_id: int = field(default=0)
         
         def acquire(
             self,
             array_name: str,
             shape: Tuple[int, ...],
             dtype: np_dtype,
         ) -> PinnedBuffer:
             """Acquire a pinned buffer for the given array.
             
             Parameters
             ----------
             array_name
                 Identifier for the array type (e.g., 'state', 'observables').
             shape
                 Required shape for the buffer.
             dtype
                 Data type for the buffer elements.
             
             Returns
             -------
             PinnedBuffer
                 A buffer ready for use, either reused or newly allocated.
             """
             # Implementation:
             # 1. Lock for thread safety
             # 2. Check pool for available buffer matching shape/dtype
             # 3. If found, mark in_use and return
             # 4. If not found, allocate new pinned buffer
             # 5. Add to pool and return
             pass
         
         def release(self, buffer: PinnedBuffer) -> None:
             """Release a buffer back to the pool.
             
             Parameters
             ----------
             buffer
                 The buffer to release.
             """
             # Implementation:
             # 1. Lock for thread safety
             # 2. Mark buffer as not in use
             pass
         
         def clear(self) -> None:
             """Clear all buffers from the pool.
             
             Should be called on cleanup or error to free pinned memory.
             """
             # Implementation:
             # 1. Lock for thread safety
             # 2. Clear all buffer lists
             pass
         
         def _allocate_buffer(
             self,
             shape: Tuple[int, ...],
             dtype: np_dtype,
         ) -> PinnedBuffer:
             """Allocate a new pinned buffer.
             
             Parameters
             ----------
             shape
                 Shape for the buffer.
             dtype
                 Data type for the buffer elements.
             
             Returns
             -------
             PinnedBuffer
                 Newly allocated pinned buffer.
             """
             # Implementation:
             # 1. Use cuda.pinned_array for real CUDA
             # 2. Use np.zeros for CUDASIM mode
             # 3. Increment _next_id
             # 4. Return wrapped PinnedBuffer
             pass
     ```
   - Edge cases: 
     - CUDASIM mode needs numpy fallback instead of cuda.pinned_array
     - Thread safety for concurrent acquire/release from watcher thread
     - Buffer reuse when shapes match exactly
   - Integration: Will be used by BatchInputArrays and BatchOutputArrays

**Tests to Create**:
- Test file: tests/memory/test_chunk_buffer_pool.py
- Test function: test_acquire_returns_pinned_buffer
- Description: Verify acquire returns a PinnedBuffer with correct shape and dtype
- Test function: test_release_marks_buffer_available
- Description: Verify released buffer can be reacquired
- Test function: test_acquire_reuses_released_buffer
- Description: Verify pool reuses buffers instead of allocating new ones
- Test function: test_clear_removes_all_buffers
- Description: Verify clear empties the pool
- Test function: test_thread_safety_concurrent_acquire_release
- Description: Verify concurrent operations don't cause race conditions

**Tests to Run**:
- tests/memory/test_chunk_buffer_pool.py::TestChunkBufferPool::test_acquire_returns_pinned_buffer
- tests/memory/test_chunk_buffer_pool.py::TestChunkBufferPool::test_release_marks_buffer_available
- tests/memory/test_chunk_buffer_pool.py::TestChunkBufferPool::test_acquire_reuses_released_buffer
- tests/memory/test_chunk_buffer_pool.py::TestChunkBufferPool::test_clear_removes_all_buffers
- tests/memory/test_chunk_buffer_pool.py::TestChunkBufferPool::test_thread_safety_concurrent_acquire_release

**Outcomes**:
- Files Modified:
  * src/cubie/memory/chunk_buffer_pool.py (135 lines, created)
  * tests/memory/test_chunk_buffer_pool.py (195 lines, created)
- Functions/Methods Added/Modified:
  * PinnedBuffer class in chunk_buffer_pool.py
  * ChunkBufferPool class in chunk_buffer_pool.py
  * ChunkBufferPool.acquire() in chunk_buffer_pool.py
  * ChunkBufferPool.release() in chunk_buffer_pool.py
  * ChunkBufferPool.clear() in chunk_buffer_pool.py
  * ChunkBufferPool._allocate_buffer() in chunk_buffer_pool.py
- Implementation Summary:
  Created ChunkBufferPool class with thread-safe buffer management for
  reusable pinned memory buffers. Uses Lock for thread safety, supports
  CUDASIM mode (falls back to np.zeros), and organizes buffers by array
  name. Also created comprehensive test suite with 13 test functions
  covering basic functionality, edge cases, and thread safety.
- Issues Flagged: None 

---

## Task Group 2: Writeback Watcher Thread
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/memory/chunk_buffer_pool.py (entire file - created in Group 1)
- File: src/cubie/time_logger.py (lines 36-120, understand CUDAEvent pattern)
- File: src/cubie/cuda_simsafe.py (lines 1-50, understand CUDA_SIMULATION)
- File: .github/context/cubie_internal_structure.md (entire file)

**Input Validation Required**:
- `event`: Must be a valid CUDA event object or None in CUDASIM
- `buffer`: Must be a PinnedBuffer instance
- `target_array`: Must be a numpy ndarray
- `slice_tuple`: Must be a tuple suitable for numpy indexing

**Tasks**:
1. **Create WritebackTask data container**
   - File: src/cubie/batchsolving/writeback_watcher.py
   - Action: Create
   - Details:
     ```python
     """Watcher thread for async writeback completion."""
     
     from queue import Queue, Empty
     from threading import Thread, Event
     from typing import Optional, Tuple, Callable
     from time import sleep
     
     from attrs import define, field
     from attrs.validators import instance_of as attrsval_instance_of
     from attrs.validators import optional as attrsval_optional
     from numpy import ndarray
     
     from cubie.cuda_simsafe import CUDA_SIMULATION
     from cubie.memory.chunk_buffer_pool import PinnedBuffer, ChunkBufferPool
     
     
     @define
     class WritebackTask:
         """Container for a pending writeback operation.
         
         Attributes
         ----------
         event
             CUDA event to query for completion (None in CUDASIM).
         buffer
             Pinned buffer containing data to copy.
         target_array
             Host array to write data into.
         slice_tuple
             Index tuple for target array slice.
         buffer_pool
             Pool to release buffer to after completion.
         array_name
             Name of the array for pool organization.
         """
         
         event: object = field()  # cuda.Event or None
         buffer: PinnedBuffer = field(validator=attrsval_instance_of(PinnedBuffer))
         target_array: ndarray = field(validator=attrsval_instance_of(ndarray))
         slice_tuple: tuple = field(validator=attrsval_instance_of(tuple))
         buffer_pool: ChunkBufferPool = field(
             validator=attrsval_instance_of(ChunkBufferPool)
         )
         array_name: str = field(validator=attrsval_instance_of(str))
     
     
     class WritebackWatcher:
         """Background thread for polling CUDA events and completing writebacks.
         
         Monitors a queue of WritebackTask objects, polls their associated
         CUDA events for completion, and copies data from pinned buffers
         to host arrays when ready. Releases buffers back to pool after copy.
         
         Attributes
         ----------
         _queue : Queue
             Thread-safe queue of pending WritebackTask objects.
         _thread : Thread or None
             Background polling thread.
         _stop_event : Event
             Signal to terminate the polling thread.
         _poll_interval : float
             Seconds between event polls.
         """
         
         def __init__(self, poll_interval: float = 0.0001) -> None:
             """Initialize the watcher.
             
             Parameters
             ----------
             poll_interval
                 Seconds between event polls. Default 0.1ms.
             """
             self._queue: Queue = Queue()
             self._thread: Optional[Thread] = None
             self._stop_event: Event = Event()
             self._poll_interval: float = poll_interval
             self._pending_count: int = 0
         
         def start(self) -> None:
             """Start the background polling thread."""
             # Implementation:
             # 1. Check if already running
             # 2. Clear stop event
             # 3. Create and start daemon thread
             pass
         
         def submit(
             self,
             event: object,
             buffer: PinnedBuffer,
             target_array: ndarray,
             slice_tuple: tuple,
             buffer_pool: ChunkBufferPool,
             array_name: str,
         ) -> None:
             """Submit a writeback task for async completion.
             
             Parameters
             ----------
             event
                 CUDA event to monitor for completion.
             buffer
                 Pinned buffer containing source data.
             target_array
                 Host array to write into.
             slice_tuple
                 Slice indices for target array.
             buffer_pool
                 Pool to release buffer to.
             array_name
                 Name of the array for pool organization.
             """
             # Implementation:
             # 1. Create WritebackTask
             # 2. Increment pending count
             # 3. Put task in queue
             # 4. Start thread if not running
             pass
         
         def wait_all(self, timeout: Optional[float] = None) -> None:
             """Block until all pending writebacks complete.
             
             Parameters
             ----------
             timeout
                 Maximum seconds to wait. None waits indefinitely.
             
             Raises
             ------
             TimeoutError
                 If timeout expires before completion.
             """
             # Implementation:
             # 1. Poll pending count until zero
             # 2. Check timeout
             pass
         
         def shutdown(self) -> None:
             """Stop the polling thread gracefully."""
             # Implementation:
             # 1. Signal stop event
             # 2. Wait for thread to join
             # 3. Clear thread reference
             pass
         
         def _poll_loop(self) -> None:
             """Main polling loop for the background thread."""
             # Implementation:
             # 1. While not stopped:
             #    a. Try to get task from queue (non-blocking)
             #    b. If task, check event.query()
             #    c. If complete, copy buffer to target, release buffer
             #    d. If not complete, requeue task
             #    e. Decrement pending count on completion
             #    f. Sleep poll_interval
             pass
         
         def _process_task(self, task: WritebackTask) -> bool:
             """Process a single writeback task.
             
             Parameters
             ----------
             task
                 Task to process.
             
             Returns
             -------
             bool
                 True if task completed, False if still pending.
             """
             # Implementation:
             # 1. Query event (or skip in CUDASIM)
             # 2. If complete:
             #    a. Copy buffer.array to target_array[slice_tuple]
             #    b. Release buffer to pool
             #    c. Return True
             # 3. If not complete, return False
             pass
     ```
   - Edge cases:
     - CUDASIM mode: events are None, treat as immediately complete
     - Thread shutdown during pending tasks
     - Timeout handling in wait_all
     - Empty queue handling in poll loop
   - Integration: Used by BatchOutputArrays.finalise_async()

**Tests to Create**:
- Test file: tests/batchsolving/test_writeback_watcher.py
- Test function: test_watcher_starts_and_stops
- Description: Verify watcher thread starts on first submit and stops on shutdown
- Test function: test_submit_and_wait_completes_writeback
- Description: Verify submitted task copies data to target array
- Test function: test_wait_all_blocks_until_complete
- Description: Verify wait_all blocks until all pending tasks finish
- Test function: test_cudasim_immediate_completion
- Description: Verify tasks complete immediately in CUDASIM mode
- Test function: test_multiple_concurrent_tasks
- Description: Verify multiple tasks can be queued and completed

**Tests to Run**:
- tests/batchsolving/test_writeback_watcher.py::TestWritebackWatcher::test_watcher_starts_and_stops
- tests/batchsolving/test_writeback_watcher.py::TestWritebackWatcher::test_submit_and_wait_completes_writeback
- tests/batchsolving/test_writeback_watcher.py::TestWritebackWatcher::test_wait_all_blocks_until_complete
- tests/batchsolving/test_writeback_watcher.py::TestWritebackWatcher::test_cudasim_immediate_completion
- tests/batchsolving/test_writeback_watcher.py::TestWritebackWatcher::test_multiple_concurrent_tasks

**Outcomes**:
- Files Modified:
  * src/cubie/batchsolving/writeback_watcher.py (228 lines, created)
  * tests/batchsolving/test_writeback_watcher.py (289 lines, created)
- Functions/Methods Added/Modified:
  * WritebackTask class in writeback_watcher.py
  * WritebackWatcher class in writeback_watcher.py
  * WritebackWatcher.__init__() in writeback_watcher.py
  * WritebackWatcher.start() in writeback_watcher.py
  * WritebackWatcher.submit() in writeback_watcher.py
  * WritebackWatcher.wait_all() in writeback_watcher.py
  * WritebackWatcher.shutdown() in writeback_watcher.py
  * WritebackWatcher._poll_loop() in writeback_watcher.py
  * WritebackWatcher._process_task() in writeback_watcher.py
- Implementation Summary:
  Created WritebackWatcher class with background daemon thread for polling
  CUDA events and completing async writebacks. Uses Queue for thread-safe
  task submission, Lock for pending count, and Event for graceful shutdown.
  Tasks with None events (CUDASIM mode) are treated as immediately complete.
  Poll loop maintains pending tasks list for tasks not yet complete and
  processes remaining tasks on shutdown to ensure all data is written.
  Created comprehensive test suite with 10 test functions covering lifecycle,
  data copying, synchronization, CUDASIM handling, concurrent tasks, timeout,
  buffer release, idempotent start, and 2D array slicing.
- Issues Flagged: None 

---

## Task Group 3: Conditional Memory Type Selection
**Status**: [x]
**Dependencies**: Task Group 1 (completed)

**Required Context**:
- File: src/cubie/batchsolving/arrays/BaseArrayManager.py (entire file)
- File: src/cubie/batchsolving/arrays/BatchOutputArrays.py (lines 1-180, understand host_factory and __attrs_post_init__)
- File: src/cubie/batchsolving/arrays/BatchInputArrays.py (lines 1-180, understand host_factory and __attrs_post_init__)
- File: src/cubie/memory/mem_manager.py (lines 835-872, understand create_host_array)
- File: .github/context/cubie_internal_structure.md (entire file)

**Input Validation Required**:
- `chunks`: Must be int >= 1 (validated elsewhere, just check value)
- No additional validation needed - this is internal logic

**Tasks**:
1. **Add is_chunked property to BaseArrayManager**
   - File: src/cubie/batchsolving/arrays/BaseArrayManager.py
   - Action: Modify
   - Details:
     ```python
     @property
     def is_chunked(self) -> bool:
         """Return True if arrays are being processed in multiple chunks."""
         return self._chunks > 1
     ```
   - Edge cases: Called before allocation complete (chunks == 0)
   - Integration: Used by host array creation logic

2. **Add method to select memory type based on chunking**
   - File: src/cubie/batchsolving/arrays/BaseArrayManager.py
   - Action: Modify
   - Details:
     ```python
     def get_host_memory_type(self, is_chunked: bool) -> str:
         """Determine host memory type based on chunking state.
         
         Parameters
         ----------
         is_chunked
             Whether the array will be processed in chunks.
         
         Returns
         -------
         str
             "pinned" for non-chunked, "host" for chunked arrays.
         
         Notes
         -----
         Non-chunked arrays use pinned memory for async transfers.
         Chunked arrays use regular numpy with per-chunk pinned buffers
         to limit total pinned memory to one chunk's worth.
         """
         return "host" if is_chunked else "pinned"
     ```
   - Edge cases: None
   - Integration: Called by update_from_solver in subclasses

3. **Modify OutputArrayContainer.host_factory to accept memory_type**
   - File: src/cubie/batchsolving/arrays/BatchOutputArrays.py
   - Action: Modify
   - Details:
     ```python
     @classmethod
     def host_factory(cls, memory_type: str = "pinned") -> "OutputArrayContainer":
         """Create a new host memory container.
         
         Parameters
         ----------
         memory_type
             Memory type for host arrays: "pinned" or "host".
             Default is "pinned" for non-chunked operation.
         
         Returns
         -------
         OutputArrayContainer
             A new container configured for the specified memory type.
         """
         container = cls()
         container.set_memory_type(memory_type)
         return container
     ```
   - Edge cases: Invalid memory_type (handled by ManagedArray validator)
   - Integration: Called during OutputArrays initialization

4. **Modify InputArrayContainer.host_factory to accept memory_type**
   - File: src/cubie/batchsolving/arrays/BatchInputArrays.py
   - Action: Modify
   - Details:
     ```python
     @classmethod
     def host_factory(cls, memory_type: str = "pinned") -> "InputArrayContainer":
         """Create a container for host memory transfers.
         
         Parameters
         ----------
         memory_type
             Memory type for host arrays: "pinned" or "host".
             Default is "pinned" for non-chunked operation.
         
         Returns
         -------
         InputArrayContainer
             Host-side container instance.
         """
         container = cls()
         container.set_memory_type(memory_type)
         return container
     ```
   - Edge cases: Invalid memory_type (handled by ManagedArray validator)
   - Integration: Called during InputArrays initialization

5. **Update OutputArrays to use conditional memory type**
   - File: src/cubie/batchsolving/arrays/BatchOutputArrays.py
   - Action: Modify
   - Details:
     In `_on_allocation_complete`, after setting `self._chunks`:
     ```python
     def _on_allocation_complete(self, response: ArrayResponse) -> None:
         # Call parent implementation
         super()._on_allocation_complete(response)
         
         # Update host memory type based on chunk count
         memory_type = self.get_host_memory_type(self.is_chunked)
         if self.is_chunked:
             self._convert_host_to_numpy()
     
     def _convert_host_to_numpy(self) -> None:
         """Convert pinned host arrays to regular numpy for chunked mode.
         
         When chunking is active, host arrays should be regular numpy
         to limit pinned memory usage. Per-chunk pinned buffers are
         used for staging during transfers.
         """
         for name, slot in self.host.iter_managed_arrays():
             if slot.memory_type == "pinned" and slot.is_chunked:
                 old_array = slot.array
                 if old_array is not None:
                     # Create numpy array with same data
                     new_array = np.array(old_array, dtype=slot.dtype)
                     slot.array = new_array
                     slot.memory_type = "host"
     ```
   - Edge cases: Arrays already allocated before chunk count known
   - Integration: Called after memory allocation response received

**Tests to Create**:
- Test file: tests/batchsolving/arrays/test_conditional_memory.py
- Test function: test_is_chunked_false_when_single_chunk
- Description: Verify is_chunked returns False when chunks <= 1
- Test function: test_is_chunked_true_when_multiple_chunks
- Description: Verify is_chunked returns True when chunks > 1
- Test function: test_get_host_memory_type_returns_pinned_non_chunked
- Description: Verify non-chunked arrays use pinned memory
- Test function: test_get_host_memory_type_returns_host_chunked
- Description: Verify chunked arrays use regular numpy
- Test function: test_output_arrays_converts_to_numpy_when_chunked
- Description: Verify OutputArrays converts pinned to numpy in chunked mode

**Tests to Run**:
- tests/batchsolving/arrays/test_conditional_memory.py::TestIsChunkedProperty::test_is_chunked_false_when_single_chunk
- tests/batchsolving/arrays/test_conditional_memory.py::TestIsChunkedProperty::test_is_chunked_true_when_multiple_chunks
- tests/batchsolving/arrays/test_conditional_memory.py::TestGetHostMemoryType::test_get_host_memory_type_returns_pinned_non_chunked
- tests/batchsolving/arrays/test_conditional_memory.py::TestGetHostMemoryType::test_get_host_memory_type_returns_host_chunked
- tests/batchsolving/arrays/test_conditional_memory.py::TestOutputArraysConvertToNumpyWhenChunked::test_output_arrays_converts_to_numpy_when_chunked
- tests/batchsolving/arrays/test_conditional_memory.py::TestOutputArraysConvertToNumpyWhenChunked::test_output_arrays_stays_pinned_when_not_chunked
- tests/batchsolving/arrays/test_conditional_memory.py::TestHostFactoryMemoryType::test_output_container_host_factory_default_pinned
- tests/batchsolving/arrays/test_conditional_memory.py::TestHostFactoryMemoryType::test_output_container_host_factory_accepts_host
- tests/batchsolving/arrays/test_conditional_memory.py::TestHostFactoryMemoryType::test_input_container_host_factory_default_pinned
- tests/batchsolving/arrays/test_conditional_memory.py::TestHostFactoryMemoryType::test_input_container_host_factory_accepts_host

**Outcomes**:
- Files Modified:
  * src/cubie/batchsolving/arrays/BaseArrayManager.py (26 lines added)
  * src/cubie/batchsolving/arrays/BatchOutputArrays.py (47 lines added/modified)
  * src/cubie/batchsolving/arrays/BatchInputArrays.py (10 lines modified)
  * tests/batchsolving/arrays/test_conditional_memory.py (175 lines, created)
- Functions/Methods Added/Modified:
  * is_chunked property in BaseArrayManager.py
  * get_host_memory_type() in BaseArrayManager.py
  * host_factory() in OutputArrayContainer (added memory_type parameter)
  * host_factory() in InputArrayContainer (added memory_type parameter)
  * _on_allocation_complete() override in OutputArrays
  * _convert_host_to_numpy() in OutputArrays
- Implementation Summary:
  Added is_chunked property that checks if _chunks > 1, and get_host_memory_type
  method that returns "host" for chunked or "pinned" for non-chunked arrays.
  Modified host_factory methods in both input/output containers to accept
  memory_type parameter with "pinned" default. Added _on_allocation_complete
  override in OutputArrays that converts pinned host arrays to regular numpy
  when chunking is active, limiting pinned memory to per-chunk buffers.
  Created comprehensive test suite with 10 test functions.
- Issues Flagged: None 

---

## Task Group 4: Integrate Buffer Pool and Watcher into BatchOutputArrays
**Status**: [x]
**Dependencies**: Task Groups 1, 2, 3

**Required Context**:
- File: src/cubie/batchsolving/arrays/BatchOutputArrays.py (entire file)
- File: src/cubie/memory/chunk_buffer_pool.py (entire file - from Group 1)
- File: src/cubie/batchsolving/writeback_watcher.py (entire file - from Group 2)
- File: src/cubie/cuda_simsafe.py (lines 1-50, CUDA_SIMULATION flag)
- File: src/cubie/_utils.py (lines 1-50, slice_variable_dimension utility)
- File: .github/context/cubie_internal_structure.md (entire file)

**Input Validation Required**:
- `host_indices`: Must be a valid slice or array for numpy indexing
- No additional validation - inputs are from internal calls

**Tasks**:
1. **Add buffer pool and watcher to OutputArrays**
   - File: src/cubie/batchsolving/arrays/BatchOutputArrays.py
   - Action: Modify
   - Details:
     Add imports at top:
     ```python
     from cubie.memory.chunk_buffer_pool import ChunkBufferPool
     from cubie.batchsolving.writeback_watcher import WritebackWatcher
     ```
     
     Add to class definition:
     ```python
     _buffer_pool: ChunkBufferPool = field(factory=ChunkBufferPool, init=False)
     _watcher: WritebackWatcher = field(factory=WritebackWatcher, init=False)
     ```
   - Edge cases: None
   - Integration: Used by finalise and wait_pending

2. **Replace finalise with event-based async version**
   - File: src/cubie/batchsolving/arrays/BatchOutputArrays.py
   - Action: Modify
   - Details:
     ```python
     def finalise(self, host_indices: ChunkIndices) -> None:
         """Queue device-to-host transfers for a chunk.
         
         Parameters
         ----------
         host_indices
             Indices for the chunk being finalized.
         
         Returns
         -------
         None
             Queues async transfers. For chunked mode, submits writeback
             tasks to the watcher thread for non-blocking completion.
         """
         from numba import cuda as numba_cuda
         from cubie.cuda_simsafe import CUDA_SIMULATION
         
         from_ = []
         to_ = []
         stream = self._memory_manager.get_stream(self)
         
         for array_name, slot in self.host.iter_managed_arrays():
             device_array = self.device.get_array(array_name)
             host_array = slot.array
             stride_order = slot.stride_order
             
             if self._chunk_axis in stride_order:
                 chunk_index = stride_order.index(self._chunk_axis)
                 slice_tuple = slice_variable_dimension(
                     host_indices, chunk_index, len(stride_order)
                 )
                 host_slice = host_array[slice_tuple]
                 
                 if self.is_chunked and slot.is_chunked:
                     # Chunked mode: use buffer pool and watcher
                     buffer = self._buffer_pool.acquire(
                         array_name, host_slice.shape, host_slice.dtype
                     )
                     to_.append(buffer.array)
                     from_.append(device_array)
                     
                     # Will submit to watcher after transfer
                     self._pending_buffers.append(
                         (buffer, host_array, slice_tuple, array_name)
                     )
                 else:
                     # Non-chunked mode: direct pinned transfer (legacy)
                     pinned_buffer = numba_cuda.pinned_array(
                         host_slice.shape, dtype=host_slice.dtype
                     )
                     self._deferred_writebacks.append(
                         (host_array, slice_tuple, pinned_buffer)
                     )
                     to_.append(pinned_buffer)
                     from_.append(device_array)
             else:
                 to_.append(host_array)
                 from_.append(device_array)
         
         self.from_device(from_, to_)
         
         # Record events and submit to watcher for chunked mode
         if self.is_chunked and self._pending_buffers:
             if not CUDA_SIMULATION:
                 event = numba_cuda.event()
                 event.record(stream)
             else:
                 event = None
             
             for buffer, host_array, slice_tuple, array_name in self._pending_buffers:
                 self._watcher.submit(
                     event=event,
                     buffer=buffer,
                     target_array=host_array,
                     slice_tuple=slice_tuple,
                     buffer_pool=self._buffer_pool,
                     array_name=array_name,
                 )
             self._pending_buffers.clear()
     ```
   - Edge cases:
     - Non-chunked arrays within chunked run (is_chunked on ManagedArray)
     - CUDASIM mode (event is None)
   - Integration: Called from BatchSolverKernel chunk loop

3. **Add _pending_buffers list attribute**
   - File: src/cubie/batchsolving/arrays/BatchOutputArrays.py
   - Action: Modify
   - Details:
     ```python
     _pending_buffers: List[Tuple] = field(factory=list, init=False)
     ```
   - Edge cases: None
   - Integration: Temporary storage between D2H transfer and event submission

4. **Replace complete_writeback with wait_pending**
   - File: src/cubie/batchsolving/arrays/BatchOutputArrays.py
   - Action: Modify
   - Details:
     ```python
     def wait_pending(self, timeout: Optional[float] = None) -> None:
         """Wait for all pending async writebacks to complete.
         
         Parameters
         ----------
         timeout
             Maximum seconds to wait. None waits indefinitely.
         
         Returns
         -------
         None
             Blocks until all pending operations complete.
         
         Notes
         -----
         Replaces complete_writeback(). For non-chunked mode, handles
         legacy deferred writebacks. For chunked mode, waits on watcher.
         """
         # Handle legacy deferred writebacks (non-chunked mode)
         for host_array, slice_tuple, contiguous_slice in self._deferred_writebacks:
             host_array[slice_tuple] = contiguous_slice
         self._deferred_writebacks.clear()
         
         # Handle watcher-based writebacks (chunked mode)
         if self.is_chunked:
             self._watcher.wait_all(timeout=timeout)
     
     # Keep complete_writeback as alias for backwards compatibility
     def complete_writeback(self) -> None:
         """Alias for wait_pending() for backwards compatibility."""
         self.wait_pending()
     ```
   - Edge cases: Mixed chunked/non-chunked in same run
   - Integration: Called from BatchSolverKernel after all chunks

5. **Add cleanup in reset method**
   - File: src/cubie/batchsolving/arrays/BatchOutputArrays.py
   - Action: Modify
   - Details:
     Add to existing reset or create if needed:
     ```python
     def reset(self) -> None:
         """Clear all cached arrays and reset allocation tracking."""
         super().reset()
         self._buffer_pool.clear()
         self._watcher.shutdown()
         self._pending_buffers.clear()
         self._deferred_writebacks.clear()
     ```
   - Edge cases: Called during error handling
   - Integration: Called on solver cleanup

**Tests to Create**:
- Test file: tests/batchsolving/arrays/test_batchoutputarrays.py (update existing)
- Test function: test_finalise_uses_buffer_pool_when_chunked
- Description: Verify chunked finalise acquires buffers from pool
- Test function: test_finalise_submits_to_watcher_when_chunked
- Description: Verify chunked finalise submits tasks to watcher
- Test function: test_wait_pending_blocks_until_complete
- Description: Verify wait_pending blocks until watcher completes
- Test function: test_complete_writeback_alias_works
- Description: Verify complete_writeback still works for compatibility
- Test function: test_non_chunked_uses_legacy_path
- Description: Verify non-chunked mode uses deferred writebacks

**Tests to Run**:
- tests/batchsolving/arrays/test_batchoutputarrays.py::TestBufferPoolAndWatcherIntegration::test_finalise_uses_buffer_pool_when_chunked
- tests/batchsolving/arrays/test_batchoutputarrays.py::TestBufferPoolAndWatcherIntegration::test_finalise_submits_to_watcher_when_chunked
- tests/batchsolving/arrays/test_batchoutputarrays.py::TestBufferPoolAndWatcherIntegration::test_wait_pending_blocks_until_complete
- tests/batchsolving/arrays/test_batchoutputarrays.py::TestBufferPoolAndWatcherIntegration::test_complete_writeback_alias_works
- tests/batchsolving/arrays/test_batchoutputarrays.py::TestBufferPoolAndWatcherIntegration::test_non_chunked_uses_legacy_path
- tests/batchsolving/arrays/test_batchoutputarrays.py::TestBufferPoolAndWatcherIntegration::test_reset_clears_buffer_pool_and_watcher

**Outcomes**:
- Files Modified:
  * src/cubie/batchsolving/arrays/BatchOutputArrays.py (84 lines added/modified)
  * tests/batchsolving/arrays/test_batchoutputarrays.py (145 lines added)
- Functions/Methods Added/Modified:
  * Added imports: ChunkBufferPool, WritebackWatcher, CUDA_SIMULATION
  * Added PendingBuffer type alias
  * Added _buffer_pool attribute (ChunkBufferPool)
  * Added _watcher attribute (WritebackWatcher)
  * Added _pending_buffers attribute (List[PendingBuffer])
  * Modified finalise() - event-based async version for chunked mode
  * Added wait_pending() - new method replacing complete_writeback
  * Modified complete_writeback() - now alias for wait_pending()
  * Added reset() - override to clear buffer pool, watcher, pending buffers
- Implementation Summary:
  Integrated ChunkBufferPool and WritebackWatcher into OutputArrays for
  chunked mode async writebacks. The finalise() method now acquires
  pooled pinned buffers, queues D2H transfers, records CUDA events, and
  submits tasks to the watcher thread. wait_pending() handles both legacy
  deferred writebacks (non-chunked) and watcher-based writebacks (chunked).
  complete_writeback() is kept as backwards-compatible alias. reset()
  clears buffer pool, shuts down watcher, and clears pending buffers.
  Created comprehensive test class with 6 tests covering buffer pool
  acquisition, watcher submission, wait_pending, alias compatibility,
  legacy path for non-chunked mode, and reset cleanup.
- Issues Flagged: None

---

## Task Group 5: Integrate Buffer Pool into BatchInputArrays
**Status**: [x]
**Dependencies**: Task Groups 1, 3

**Required Context**:
- File: src/cubie/batchsolving/arrays/BatchInputArrays.py (entire file)
- File: src/cubie/memory/chunk_buffer_pool.py (entire file - from Group 1)
- File: src/cubie/cuda_simsafe.py (lines 1-50)
- File: .github/context/cubie_internal_structure.md (entire file)

**Input Validation Required**:
- `host_indices`: Must be a valid slice or array for numpy indexing
- No additional validation - inputs are from internal calls

**Tasks**:
1. **Add buffer pool to InputArrays**
   - File: src/cubie/batchsolving/arrays/BatchInputArrays.py
   - Action: Modify
   - Details:
     Add import at top:
     ```python
     from cubie.memory.chunk_buffer_pool import ChunkBufferPool
     ```
     
     Add to class definition:
     ```python
     _buffer_pool: ChunkBufferPool = field(factory=ChunkBufferPool, init=False)
     ```
   - Edge cases: None
   - Integration: Used by initialise

2. **Update initialise to use buffer pool for chunked mode**
   - File: src/cubie/batchsolving/arrays/BatchInputArrays.py
   - Action: Modify
   - Details:
     Replace the cuda.pinned_array allocation in initialise:
     ```python
     def initialise(self, host_indices: Union[slice, NDArray]) -> None:
         """Copy a batch chunk of host data to device buffers.
         
         Parameters
         ----------
         host_indices
             Indices for the chunk being initialized.
         
         Returns
         -------
         None
             Host slices are staged into device arrays in place.
         """
         from numba import cuda as numba_cuda
         
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
             
             if self._chunks <= 1 or not device_obj.is_chunked:
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
                 
                 if self.is_chunked:
                     # Use buffer pool for chunked mode
                     buffer = self._buffer_pool.acquire(
                         array_name, host_slice.shape, host_slice.dtype
                     )
                     buffer.array[:] = host_slice
                     from_.append(buffer.array)
                     # Buffer released after H2D completes (sync at chunk end)
                     # Store for later release
                     self._active_buffers.append(buffer)
                 else:
                     # Non-chunked: allocate pinned buffer directly
                     pinned_buffer = numba_cuda.pinned_array(
                         host_slice.shape, dtype=host_slice.dtype
                     )
                     pinned_buffer[:] = host_slice
                     from_.append(pinned_buffer)
         
         self.to_device(from_, to_)
     ```
   - Edge cases:
     - Non-chunked arrays within chunked run
     - Chunk axis not in stride order
   - Integration: Called from BatchSolverKernel chunk loop

3. **Add _active_buffers attribute and release logic**
   - File: src/cubie/batchsolving/arrays/BatchInputArrays.py
   - Action: Modify
   - Details:
     ```python
     _active_buffers: List = field(factory=list, init=False)
     
     def release_buffers(self) -> None:
         """Release all active buffers back to the pool.
         
         Should be called after H2D transfer completes.
         """
         for buffer in self._active_buffers:
             self._buffer_pool.release(buffer)
         self._active_buffers.clear()
     ```
   - Edge cases: None
   - Integration: Called after kernel launch completes

4. **Add cleanup in reset method**
   - File: src/cubie/batchsolving/arrays/BatchInputArrays.py
   - Action: Modify
   - Details:
     ```python
     def reset(self) -> None:
         """Clear all cached arrays and reset allocation tracking."""
         super().reset()
         self._buffer_pool.clear()
         self._active_buffers.clear()
     ```
   - Edge cases: Called during error handling
   - Integration: Called on solver cleanup

**Tests to Create**:
- Test file: tests/batchsolving/arrays/test_batchinputarrays.py (update existing)
- Test function: test_initialise_uses_buffer_pool_when_chunked
- Description: Verify chunked initialise acquires buffers from pool
- Test function: test_release_buffers_returns_to_pool
- Description: Verify release_buffers returns buffers to pool
- Test function: test_non_chunked_uses_direct_pinned
- Description: Verify non-chunked mode allocates pinned directly

**Tests to Run**:
- tests/batchsolving/arrays/test_batchinputarrays.py::TestBufferPoolIntegration::test_input_arrays_has_buffer_pool
- tests/batchsolving/arrays/test_batchinputarrays.py::TestBufferPoolIntegration::test_input_arrays_has_active_buffers
- tests/batchsolving/arrays/test_batchinputarrays.py::TestBufferPoolIntegration::test_initialise_uses_buffer_pool_when_chunked
- tests/batchsolving/arrays/test_batchinputarrays.py::TestBufferPoolIntegration::test_release_buffers_returns_to_pool
- tests/batchsolving/arrays/test_batchinputarrays.py::TestBufferPoolIntegration::test_non_chunked_uses_direct_pinned
- tests/batchsolving/arrays/test_batchinputarrays.py::TestBufferPoolIntegration::test_reset_clears_buffer_pool_and_active_buffers
- tests/batchsolving/arrays/test_batchinputarrays.py::TestBufferPoolIntegration::test_buffers_reused_across_chunks

**Outcomes**:
- Files Modified:
  * src/cubie/batchsolving/arrays/BatchInputArrays.py (31 lines added/modified)
  * tests/batchsolving/arrays/test_batchinputarrays.py (172 lines added)
- Functions/Methods Added/Modified:
  * Added imports: ChunkBufferPool, PinnedBuffer, List
  * Added _buffer_pool attribute (ChunkBufferPool)
  * Added _active_buffers attribute (List[PinnedBuffer])
  * Modified initialise() - buffer pool integration for chunked mode
  * Added release_buffers() - returns pooled buffers after H2D
  * Added reset() - clears buffer pool and active buffers
- Implementation Summary:
  Integrated ChunkBufferPool into InputArrays for chunked mode H2D
  transfers. The initialise() method acquires pooled pinned buffers
  when is_chunked is True, copies host data slices to buffers, and
  stores buffers in _active_buffers. release_buffers() releases all
  active buffers back to the pool after H2D transfer completes.
  reset() clears both buffer pool and active buffers list. Created
  TestBufferPoolIntegration test class with 7 tests covering buffer
  pool presence, active buffers list, chunked mode acquisition,
  buffer release, non-chunked mode bypass, reset cleanup, and
  buffer reuse across chunks.
- Issues Flagged: None

---

## Task Group 6: Remove sync_stream from BatchSolverKernel and Solver
**Status**: [x]
**Dependencies**: Task Groups 4, 5

**Required Context**:
- File: src/cubie/batchsolving/BatchSolverKernel.py (lines 343-560, run method and chunk loop)
- File: src/cubie/batchsolving/solver.py (lines 329-450, solve method)
- File: src/cubie/batchsolving/arrays/BatchOutputArrays.py (entire file - modified in Group 4)
- File: src/cubie/batchsolving/arrays/BatchInputArrays.py (entire file - modified in Group 5)
- File: .github/context/cubie_internal_structure.md (entire file)

**Input Validation Required**:
- None - this is internal refactoring of existing validated flows

**Tasks**:
1. **Replace sync_stream with wait_pending in BatchSolverKernel.run**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     Replace lines 552-554:
     ```python
     # Old:
     # self.memory_manager.sync_stream(self)
     # self.output_arrays.complete_writeback()
     
     # New:
     self.output_arrays.wait_pending()
     ```
   - Edge cases: None - wait_pending handles both chunked and non-chunked
   - Integration: Called at end of run method

2. **Add input buffer release after kernel completes**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     After each chunk's kernel execution, release input buffers:
     ```python
     # After kernel launch (line ~537):
     kernel_event.record_end(stream)
     
     # Release input buffers after kernel (H2D complete)
     self.input_arrays.release_buffers()
     
     # d2h transfer timing
     d2h_event.record_start(stream)
     ```
   - Edge cases: Non-chunked mode (release_buffers is no-op)
   - Integration: Called in chunk loop

3. **Remove redundant sync_stream from Solver.solve**
   - File: src/cubie/batchsolving/solver.py
   - Action: Modify
   - Details:
     Remove line 440:
     ```python
     # Old:
     # self.kernel.run(...)
     # self.memory_manager.sync_stream(self.kernel)
     
     # New:
     self.kernel.run(...)
     # sync handled inside kernel.run() via wait_pending()
     ```
   - Edge cases: None
   - Integration: Solver.solve becomes non-blocking until wait_pending

**Tests to Create**:
- Test file: tests/batchsolving/test_chunked_solver.py (update existing)
- Test function: test_solver_completes_without_sync_stream
- Description: Verify solver works without explicit sync_stream call
- Test function: test_chunked_solver_produces_correct_results
- Description: Verify chunked execution produces same results as non-chunked
- Test function: test_input_buffers_released_after_kernel
- Description: Verify input buffers are released after each chunk

**Tests to Run**:
- tests/batchsolving/test_chunked_solver.py::TestSyncStreamRemoval::test_solver_completes_without_sync_stream
- tests/batchsolving/test_chunked_solver.py::TestSyncStreamRemoval::test_chunked_solver_produces_correct_results
- tests/batchsolving/test_chunked_solver.py::TestSyncStreamRemoval::test_input_buffers_released_after_kernel

**Outcomes**:
- Files Modified:
  * src/cubie/batchsolving/BatchSolverKernel.py (5 lines changed)
  * src/cubie/batchsolving/solver.py (1 line changed)
  * tests/batchsolving/test_chunked_solver.py (102 lines added)
- Functions/Methods Added/Modified:
  * run() in BatchSolverKernel.py - added release_buffers() call, replaced sync_stream/complete_writeback with wait_pending()
  * solve() in solver.py - removed redundant sync_stream call
- Implementation Summary:
  Replaced explicit sync_stream and complete_writeback calls with the new
  wait_pending() method that handles both chunked and non-chunked modes.
  Added release_buffers() call after kernel execution in the chunk loop
  to return pooled pinned buffers for reuse. Removed redundant sync_stream
  from Solver.solve since sync is now handled inside kernel.run() via
  wait_pending(). Created TestSyncStreamRemoval test class with 3 tests
  covering solver completion without sync_stream, result correctness
  between chunked and non-chunked modes, and buffer release verification.
- Issues Flagged: None

---

## Task Group 7: Integration Testing and Regression Verification
**Status**: [x]
**Dependencies**: Task Groups 1-6

**Required Context**:
- File: tests/batchsolving/test_solver.py (entire file - understand test patterns)
- File: tests/batchsolving/test_chunked_solver.py (entire file - chunked testing)
- File: tests/conftest.py (lines 1-100 - fixture patterns)
- File: .github/context/cubie_internal_structure.md (entire file)

**Input Validation Required**:
- None - testing only

**Tasks**:
1. **Create comprehensive integration test for two-tier memory**
   - File: tests/batchsolving/test_pinned_memory_refactor.py
   - Action: Create
   - Details:
     ```python
     """Integration tests for refactored pinned memory system."""
     
     import numpy as np
     import pytest
     
     
     class TestTwoTierMemoryStrategy:
         """Test that memory strategy changes based on chunking."""
         
         def test_non_chunked_uses_pinned_host(self, solver, small_batch):
             """Non-chunked runs use pinned host arrays."""
             # Verify host arrays are pinned when chunks == 1
             pass
         
         def test_chunked_uses_numpy_host(self, solver, large_batch):
             """Chunked runs use numpy host arrays with buffer pool."""
             # Verify host arrays are numpy when chunks > 1
             pass
         
         def test_total_pinned_memory_bounded(self, solver, large_batch):
             """Total pinned memory stays within one chunk's worth."""
             # Verify pinned memory usage is limited
             pass
     
     
     class TestEventBasedSynchronization:
         """Test CUDA event synchronization."""
         
         @pytest.mark.nocudasim
         def test_events_recorded_in_chunked_mode(self, solver, large_batch):
             """CUDA events are recorded for each chunk."""
             pass
         
         def test_wait_pending_blocks_correctly(self, solver, large_batch):
             """wait_pending blocks until all writebacks complete."""
             pass
     
     
     class TestWatcherThreadBehavior:
         """Test watcher thread lifecycle and behavior."""
         
         def test_watcher_starts_on_first_chunk(self, solver, large_batch):
             """Watcher thread starts when first task submitted."""
             pass
         
         def test_watcher_completes_all_tasks(self, solver, large_batch):
             """All submitted tasks are completed before solve returns."""
             pass
     
     
     class TestRegressionNonChunkedPath:
         """Verify non-chunked path unchanged."""
         
         def test_small_batch_produces_correct_results(self, solver):
             """Small batches work correctly with refactored code."""
             pass
         
         def test_non_chunked_performance_maintained(self, solver):
             """Non-chunked path has no performance regression."""
             pass
     
     
     class TestRegressionChunkedPath:
         """Verify chunked path produces correct results."""
         
         def test_large_batch_produces_correct_results(self, solver, large_batch):
             """Large batches produce same results as before."""
             pass
         
         def test_chunked_results_match_non_chunked(self, solver):
             """Chunked execution produces same results as non-chunked."""
             pass
     ```
   - Edge cases: Cover CUDASIM mode, various chunk counts
   - Integration: Full end-to-end testing

2. **Update existing tests for API changes**
   - File: tests/batchsolving/arrays/test_batchoutputarrays.py
   - Action: Modify
   - Details:
     Update tests that call complete_writeback to use wait_pending:
     - Line 327: `output_arrays_manager.complete_writeback()` stays (now alias)
     - Add test for wait_pending method directly
   - Edge cases: None
   - Integration: Ensure existing tests pass

**Tests to Create**:
- Test file: tests/batchsolving/test_pinned_memory_refactor.py
- (Multiple test functions as shown in task 1 details)

**Tests to Run**:
- tests/batchsolving/test_pinned_memory_refactor.py
- tests/batchsolving/test_solver.py
- tests/batchsolving/test_chunked_solver.py
- tests/batchsolving/arrays/test_batchoutputarrays.py
- tests/batchsolving/arrays/test_batchinputarrays.py

**Outcomes**:
- Files Modified:
  * tests/batchsolving/test_pinned_memory_refactor.py (325 lines, created)
- Functions/Methods Added/Modified:
  * TestTwoTierMemoryStrategy class with 3 tests
  * TestEventBasedSynchronization class with 2 tests
  * TestWatcherThreadBehavior class with 2 tests
  * TestRegressionNonChunkedPath class with 2 tests
  * TestRegressionChunkedPath class with 3 tests
- Implementation Summary:
  Created comprehensive integration test file covering the refactored
  pinned memory system. Tests verify two-tier memory strategy (pinned
  for non-chunked, numpy+buffer pool for chunked), event-based
  synchronization via wait_pending, watcher thread lifecycle, and
  regression testing for both chunked and non-chunked paths. Tests
  use MockMemoryManager to force chunking and verify buffer pool usage,
  memory type transitions, and result correctness. Existing tests in
  test_batchoutputarrays.py already cover wait_pending and complete_writeback
  alias, so no modifications needed there.
- Issues Flagged: None 

---

## Summary

**Total Task Groups**: 7

**Dependency Chain**:
```
Group 1 (Buffer Pool) ─────┬────> Group 3 (Memory Selection) ────┐
                           │                                      │
Group 2 (Watcher) ─────────┴────> Group 4 (Output Integration) ──┼──> Group 6 (Remove sync_stream) ──> Group 7 (Testing)
                                                                  │
                           Group 5 (Input Integration) ───────────┘
```

**Tests to Create**:
- tests/memory/test_chunk_buffer_pool.py (5 tests)
- tests/batchsolving/test_writeback_watcher.py (5 tests)
- tests/batchsolving/arrays/test_conditional_memory.py (5 tests)
- tests/batchsolving/arrays/test_batchoutputarrays.py (5 additional tests)
- tests/batchsolving/arrays/test_batchinputarrays.py (3 additional tests)
- tests/batchsolving/test_chunked_solver.py (3 additional tests)
- tests/batchsolving/test_pinned_memory_refactor.py (10 tests)

**Estimated Complexity**: Medium-High
- Core infrastructure (Groups 1-2): Well-defined, moderate complexity
- Integration (Groups 3-5): Requires careful handling of edge cases
- Cleanup (Group 6): Simple refactoring
- Testing (Group 7): Comprehensive validation needed
