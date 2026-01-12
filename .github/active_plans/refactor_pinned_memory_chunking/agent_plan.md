# Agent Plan: Refactor Pinned Memory Usage in Chunking System

## Problem Statement

The current chunking system allocates pinned memory for all host arrays regardless of chunking state. This leads to excessive pinned memory when chunking is active—pinned memory for the full batch is allocated even when only one chunk's worth is needed. Additionally, deferred writebacks rely on `sync_stream`, which blocks the main thread.

**Requirements:**
1. Non-chunked arrays: Use pinned host memory (current behavior)
2. Chunked arrays: Use regular numpy host arrays + per-chunk pinned buffers
3. Replace `sync_stream` with CUDA event polling + watcher thread
4. Maintain async I/O and chunked execution

---

## Component 1: Conditional Memory Type Selection

### Current Behavior
- `OutputArrayContainer.host_factory()` always sets memory_type to "pinned"
- `InputArrayContainer.host_factory()` always sets memory_type to "pinned"
- Host arrays allocated via `mem_manager.create_host_array()` with "pinned"

### Expected Behavior
- When chunking is NOT active (chunks == 1): use "pinned" memory
- When chunking IS active (chunks > 1): use "host" memory (regular numpy)
- Decision must happen AFTER `allocate_queue` determines chunk count

### Architectural Changes
1. Add property or method to `BaseArrayManager` to query chunking state
2. Modify host array creation in `update_from_solver()` methods
3. Memory type decision deferred until chunk count is known

### Integration Points
- `mem_manager.create_host_array()` already accepts `memory_type` parameter
- `ArrayResponse.chunks` carries the chunk count from allocation
- `BaseArrayManager._on_allocation_complete()` receives this response

### Dependencies
- Must know chunk count before creating host arrays
- May require restructuring array creation flow

---

## Component 2: Per-Chunk Pinned Buffer Pool

### Expected Behavior
When chunking is active:
1. Allocate a fixed set of pinned buffers (2-3 for double/triple buffering)
2. Each buffer sized to hold one chunk's worth of data
3. Buffers are reused across chunks
4. Input: copy numpy slice → pinned buffer → device
5. Output: device → pinned buffer → numpy slice (via watcher)

### Data Structures
**ChunkBufferPool** (new class or integration into existing):
- `_buffers: Dict[str, List[PinnedBuffer]]` - pool per array name
- `_in_use: Dict[str, Set[int]]` - track which buffers are busy
- `acquire(array_name, shape, dtype) -> PinnedBuffer`
- `release(buffer_id)`

**PinnedBuffer** (simple wrapper):
- `array: cuda.pinned_array`
- `id: int`
- `in_use: bool`

### Architectural Considerations
- Buffer pool can live in `BaseArrayManager` or as a separate component
- Must handle multiple array types (state, observables, summaries, etc.)
- Buffer sizing: use chunked shape from `ArrayRequest` after chunking

### Edge Cases
- Different array shapes per output type
- Dynamic chunk sizes (last chunk may be smaller)
- Clean up on exceptions

---

## Component 3: CUDA Event Integration

### Current Event Usage
The project already has `CUDAEvent` in `time_logger.py`:
- Uses `numba.cuda.event()` internally
- `record_start(stream)` and `record_end(stream)` methods
- `elapsed_time_ms()` for timing

### Expected Behavior for Synchronization
1. After D2H transfer for each chunk, record a CUDA event
2. Store the event with its associated writeback data
3. Query event completion without blocking
4. When complete, trigger writeback and release buffer

### New Event Pattern
```python
# After D2H transfer
event = cuda.event()
event.record(stream)

# Later, check completion
if event.query():  # Returns True if complete, False otherwise
    # Safe to copy buffer to host array
    host_array[slice_tuple] = pinned_buffer
```

### Integration with Existing Events
- Can reuse `CUDAEvent` class pattern or create simpler variant
- Events for writebacks are functional, not timing-focused

---

## Component 4: Watcher Thread for Async Writeback

### Expected Behavior
A background thread that:
1. Maintains a queue of pending writebacks
2. Polls associated CUDA events for completion
3. When complete, executes writeback copy
4. Releases pinned buffer back to pool
5. Thread-safe coordination with main thread

### Architecture
**WritebackWatcher** (new class):
- `_queue: Queue[WritebackTask]` - pending tasks
- `_thread: Thread` - polling thread
- `_running: bool` - control flag
- `submit(event, buffer, target_array, slice_tuple)`
- `wait_all()` - block until all pending complete
- `shutdown()` - clean termination

**WritebackTask** (data container):
- `event: cuda.event`
- `buffer: PinnedBuffer`
- `target_array: ndarray`
- `slice_tuple: tuple`
- `buffer_pool_ref` - for releasing buffer

### Thread Safety Considerations
- Queue operations are thread-safe in Python's `queue.Queue`
- NumPy array writes are safe if slices don't overlap
- Event.query() is thread-safe (CUDA driver handles)

### Lifecycle
1. Create watcher at solve start (or first chunked solve)
2. Main thread submits tasks after each chunk D2H
3. Watcher polls and processes in background
4. Before returning results, call `wait_all()` to ensure completion
5. Optionally shutdown or reuse for next solve

---

## Component 5: Refactored BatchOutputArrays

### Current Flow
```
update() → allocate() → 
for chunk:
    initialise() → kernel → finalise()
complete_writeback()
```

### Expected Flow (Chunked)
```
update() → allocate() [determines chunks] → 
    if chunks > 1: convert host to numpy, setup buffer pool
for chunk:
    initialise() → kernel → 
    finalise_async() [D2H to buffer, record event, submit to watcher]
wait_all() [blocks until all writebacks complete]
```

### Key Method Changes

**finalise()** becomes **finalise_async()**:
- Instead of storing deferred writebacks, submit to watcher thread
- Record event after D2H transfer
- Get buffer from pool, don't create new each time

**complete_writeback()** becomes **wait_pending()**:
- Call `watcher.wait_all()`
- Ensures all async operations finished before returning

### Data Structures
- `_buffer_pool: ChunkBufferPool` - shared buffer management
- `_watcher: WritebackWatcher` - async completion handler
- Remove `_deferred_writebacks: List[DeferredWriteback]`

---

## Component 6: Refactored BatchInputArrays

### Current Flow
- Creates pinned buffers in `initialise()` for each chunk slice
- Copies host slice to buffer, then buffer to device

### Expected Flow (Chunked)
- Use buffer pool instead of creating new each time
- Same staging pattern, just with pooled buffers

### Key Changes
- Share buffer pool with output arrays or have separate pool
- Acquire buffer, copy, transfer, release after H2D completion

---

## Component 7: BatchSolverKernel Integration

### Current sync_stream Usage
```python
# Line 553: After all chunks
self.memory_manager.sync_stream(self)
self.output_arrays.complete_writeback()
```

### Expected Changes
```python
# After all chunks
self.output_arrays.wait_pending()  # Uses event-based waiting
```

### Chunk Loop Changes
Each chunk iteration:
1. `initialise()` - H2D transfer
2. Kernel launch
3. `finalise_async()` - D2H transfer + event + submit watcher task

### Event Recording
Insert after D2H transfer in `finalise_async()`:
```python
event = cuda.event()
event.record(stream)
self._watcher.submit(event, buffer, target, slice_tuple)
```

---

## Component 8: Solver Level Changes

### Current
```python
self.kernel.run(...)
self.memory_manager.sync_stream(self.kernel)
```

### Expected
```python
self.kernel.run(...)
# sync_stream removed - kernel.run() handles via wait_pending()
```

### Alternative: Keep Solver-Level Wait
Could keep a final sync at solver level as safety net:
```python
self.kernel.run(...)
self.kernel.output_arrays.ensure_complete()
```

---

## Implementation Order

### Phase 1: Buffer Pool Infrastructure
1. Create `ChunkBufferPool` class
2. Integrate into `BaseArrayManager`
3. Unit test buffer acquire/release

### Phase 2: Event-Based Watcher
1. Create `WritebackWatcher` class
2. Test event polling in isolation
3. Test watcher thread lifecycle

### Phase 3: Conditional Memory Selection
1. Modify host array creation to defer memory type
2. Add logic to select pinned vs numpy based on chunks
3. Test both paths

### Phase 4: Integrate into ArrayManagers
1. Modify `BatchOutputArrays.finalise()` to use pool + watcher
2. Modify `BatchInputArrays.initialise()` to use pool
3. Update `complete_writeback()` → `wait_pending()`

### Phase 5: Kernel and Solver Integration
1. Replace `sync_stream` calls
2. Add event recording to chunk loop
3. Full integration test

### Phase 6: Testing and Validation
1. Test non-chunked path (regression)
2. Test chunked path (correct data)
3. Test memory usage is bounded
4. Test async behavior (main thread not blocked)

---

## Edge Cases to Handle

1. **Single chunk**: Should use pinned path, no buffer pool
2. **Last chunk smaller**: Buffer may be larger than needed (OK)
3. **Exception during execution**: Clean up buffers, terminate watcher
4. **Multiple concurrent solvers**: Each has own watcher/pool
5. **CUDASIM mode**: Events work differently, may need fallback

---

## Testing Strategy

### Unit Tests
- Buffer pool acquire/release
- Watcher thread lifecycle
- Event query simulation

### Integration Tests
- Small batch (non-chunked) produces correct results
- Large batch (chunked) produces correct results
- Memory usage stays bounded during chunked run
- Multiple sequential solves work correctly

### Regression Tests
- Existing batch solving tests pass
- Performance not degraded for non-chunked case

---

## Files to Create
1. `src/cubie/memory/chunk_buffer_pool.py` - Buffer pool implementation
2. `src/cubie/batchsolving/writeback_watcher.py` - Watcher thread

## Files to Modify
1. `src/cubie/batchsolving/arrays/BaseArrayManager.py` - Buffer pool integration
2. `src/cubie/batchsolving/arrays/BatchOutputArrays.py` - Async finalise, wait_pending
3. `src/cubie/batchsolving/arrays/BatchInputArrays.py` - Buffer pool usage
4. `src/cubie/batchsolving/BatchSolverKernel.py` - Remove sync_stream
5. `src/cubie/batchsolving/solver.py` - Remove redundant sync
6. `tests/batchsolving/arrays/test_batchoutputarrays.py` - Update tests
