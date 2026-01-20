# Deep Investigation: CUDA Event Issues in Chunked Tests

## Problem

Cleanup in `chunked_solved_solver` fixture didn't fix the xdist hanging issue. User suspects problems with `cuda.event()` - races, hangs, or stale references.

## CUDA Event Lifecycle Analysis

### Event Creation and Usage

**In BatchOutputArrays.finalise() (lines 404-416)**:
```python
if self._pending_buffers:
    if not CUDA_SIMULATION:
        event = cuda.event()  # Single event created
        event.record(stream)   # Recorded on stream
    else:
        event = None

    for buffer in self._pending_buffers:  # Multiple buffers
        self._watcher.submit_from_pending_buffer(
            event=event,    # SAME event reused for ALL buffers!
            pending_buffer=buffer,
        )
    self._pending_buffers.clear()
```

**CRITICAL ISSUE #1: Event Reuse**
- A single CUDA event is created and **shared across multiple writeback tasks**
- All pending buffers from one `finalise()` call share the same event
- Multiple watcher tasks query the same event object concurrently

### Event Query in Watcher

**In WritebackWatcher._process_task() (lines 291-295)**:
```python
if CUDA_SIMULATION or task.event is None:
    is_complete = True
else:
    is_complete = task.event.query()  # Queries shared event
```

**Potential Race Condition**:
1. Task A and Task B both have reference to `event_1`
2. Watcher thread processes Task A: `event_1.query()` returns False
3. Task A added back to pending list
4. Watcher processes Task B: `event_1.query()` returns False
5. Task B added back to pending list
6. Next poll: `event_1.query()` returns True
7. Both tasks process and release buffers

This seems okay for event query, but...

### Event Destruction Issue

**CRITICAL ISSUE #2: No Event Cleanup**
- Events are created but **never explicitly destroyed**
- Python GC will eventually clean them up, but timing is unpredictable
- On CUDA hardware, events are GPU resources that can leak
- Multiple tests = multiple accumulated events

**From CUDA documentation**:
- Events consume GPU memory
- Too many undestroyed events can exhaust resources
- Events should be destroyed with `event.destroy()` or context manager

### Watcher Shutdown Timing

**In WritebackWatcher.shutdown() (lines 219-224)**:
```python
def shutdown(self):
    self._stop_event.set()
    if self._thread is not None:
        self._thread.join(timeout=1.0)  # Only waits 1 second!
        self._thread = None
```

**CRITICAL ISSUE #3: Incomplete Shutdown**
- If watcher has tasks taking >1 second, thread abandoned
- Thread continues running as daemon
- `self._pending_count` may still be positive
- Events still being queried by orphaned thread

### Wait Timeout

**In WritebackWatcher.wait_all() (lines 193-217)**:
```python
def wait_all(self, timeout: Optional[float] = 10):
    start_time = perf_counter()
    while True:
        with self._lock:
            if self._pending_count == 0:
                return
        if timeout is not None:
            elapsed = perf_counter() - start_time
            if elapsed >= timeout:
                raise TimeoutError(...)
        sleep(self._poll_interval)
```

**CRITICAL ISSUE #4: Timeout Exception**
- Default timeout is 10 seconds
- On CUDA hardware with slow event completion, could timeout
- TimeoutError raised but tasks still pending
- `_pending_count > 0` but wait_all() exits
- Next test starts with polluted state

## Hypothesis: Event Query Hanging

### Scenario on CUDA Hardware

1. **Test 1** starts:
   - Creates solver, calls solve()
   - Creates CUDA events for async transfers
   - Events recorded on Stream A
   - Watcher polls events

2. **Stream synchronization**:
   - `stream.synchronize()` called
   - Waits for operations on Stream A
   - Returns when stream is idle

3. **Event query issue**:
   - Events were recorded on Stream A
   - `event.query()` checks if event is complete
   - **BUT**: If stream was recycled or context changed, event.query() might:
     - Hang waiting for non-existent operations
     - Return incorrect status
     - Block indefinitely

4. **Test 2** starts (parallel xdist):
   - Creates new solver
   - Gets same stream from memory manager (pooled?)
   - Reuses stream that has pending events from Test 1
   - New events recorded
   - **Conflict**: Old events + new events on same stream

### Stream Pooling Issue

**From mem_manager.py**:
```python
def get_stream(self, instance):
    # Returns stream from stream_groups
    # Streams might be reused across instances!
```

If streams are pooled and reused, events from previous test could still be referenced.

## Additional Issues Found

### Issue #5: _pending_count Race

**In _submit_task() (lines 131-133)**:
```python
with self._lock:
    self._pending_count += 1
self._queue.put(task)
```

**In _process_task() (lines 236-237, 246-247)**:
```python
if self._process_task(task):
    with self._lock:
        self._pending_count -= 1
```

**Potential race**:
1. Task submitted: `_pending_count = 1`
2. Watcher processes task: `_pending_count = 0`
3. `wait_all()` sees `_pending_count = 0`, returns
4. But `_process_task()` hasn't finished releasing buffer yet!
5. Buffer still marked `in_use`, not returned to pool
6. Next test tries to acquire buffer: **blocks forever**

### Issue #6: Buffer Pool Exhaustion

**From ChunkBufferPool**:
- Limited number of pinned buffers
- Buffers acquired during `finalise()`
- Released in `_process_task()` after event completes
- If events don't complete, buffers never released
- Pool exhausted → new tests block acquiring buffers

## Root Cause Theory

**Combining all issues**:

1. Events created but not destroyed → GPU resource leak
2. Events shared across multiple tasks → unclear ownership
3. Watcher shutdown timeout too short → orphaned threads
4. wait_all() timeout → tasks abandoned mid-processing
5. Buffer release after event check → buffers stuck in_use
6. Stream/event reuse across tests → stale references

**The Perfect Storm for xdist**:
- Test 1, 2, 3 start in parallel
- All create events, start watchers
- Some events don't complete in time (or hang)
- Watchers timeout or shutdown incompletely
- Buffers not released
- Tests 4+ try to acquire buffers → **deadlock**

## Recommended Fixes

### Fix 1: Proper Event Cleanup
```python
# In WritebackWatcher._process_task()
if is_complete:
    # ... existing code ...
    # Destroy event after all tasks using it complete
    if hasattr(task.event, 'destroy') and task.event is not None:
        task.event.destroy()
```

### Fix 2: Increase Shutdown Timeout
```python
def shutdown(self):
    self._stop_event.set()
    if self._thread is not None:
        self._thread.join(timeout=30.0)  # Wait longer
```

### Fix 3: Force Event Completion
```python
# Before querying event, synchronize stream
if not CUDA_SIMULATION and task.event is not None:
    # Ensure stream operations complete
    task.event.synchronize()  # Block until event complete
    is_complete = True
```

### Fix 4: One Event Per Task
```python
# Instead of sharing one event across all buffers:
for buffer in self._pending_buffers:
    if not CUDA_SIMULATION:
        event = cuda.event()
        event.record(stream)
    else:
        event = None
    self._watcher.submit_from_pending_buffer(event, buffer)
```

### Fix 5: Remove Watcher Timeout
```python
# In fixture cleanup, wait indefinitely
solver.kernel.wait_for_writeback(timeout=None)
```

### Fix 6: Stream Isolation Per Test
Ensure each test gets a fresh stream, not reused from pool.
