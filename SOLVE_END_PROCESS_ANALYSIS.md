# Chunked Test Investigation: Detailed Analysis of solve() End Process

## Problem Statement

Chunked tests run infinitely when executed with xdist on CUDA hardware. Approximately 3 tests complete, then remaining tests hang forever.

## solve() End Process - Detailed Flow

### 1. kernel.run() - The Integration Loop

```python
# solver.py line 446-455
self.kernel.run(
    inits=inits,
    params=params,
    driver_coefficients=self.driver_interpolator.coefficients,
    duration=duration,
    warmup=settling_time,
    t0=t0,
    blocksize=blocksize,
    stream=stream,
)
```

**Inside kernel.run()** (BatchSolverKernel.py lines 570-636):
```python
for i in range(chunks):  # Process each chunk
    # 1. Initialize arrays for chunk
    self.input_arrays.initialise(i)
    self.output_arrays.initialise(i)
    
    # 2. H2D transfers (host to device)
    self.input_arrays.prefill_from_host(i, stream)
    
    # 3. Launch CUDA kernel
    launch_kernel[grid, block, stream, dynamic_shared](...)
    
    # 4. D2H transfers (device to host) - LINE 631-632
    self.input_arrays.finalise(i)
    self.output_arrays.finalise(i)
```

### 2. output_arrays.finalise() - Queue Async Transfers

**BatchOutputArrays.finalise()** (lines 345-416):

```python
def finalise(self, chunk_index: int) -> None:
    stream = self._memory_manager.get_stream(self)
    
    # Prepare transfer lists
    for array_name, slot in self.host.iter_managed_arrays():
        device_array = self.device.get_array(array_name)
        
        if slot.needs_chunked_transfer:
            # Acquire pinned buffer from pool
            buffer = self._buffer_pool.acquire(...)
            
            # Queue D2H copy: device → pinned buffer
            from_.append(device_array.chunk_slice)
            to_.append(buffer.array)
            
            # Store for async writeback
            self._pending_buffers.append(PendingBuffer(...))
        else:
            # Direct copy: device → host
            from_.append(device_array)
            to_.append(host_array)
    
    # Execute all D2H copies on stream
    self.from_device(from_, to_)  # Async CUDA memcpy
    
    # Create event and submit to watcher (ORIGINAL CODE)
    if self._pending_buffers:
        if not CUDA_SIMULATION:
            event = cuda.event()     # Create ONE event
            event.record(stream)      # Record after D2H copies
        else:
            event = None
        
        # Submit all buffers with same event
        for buffer in self._pending_buffers:
            self._watcher.submit_from_pending_buffer(
                event=event,  # All tasks share this event
                pending_buffer=buffer,
            )
        self._pending_buffers.clear()
```

**Key Point**: Event is recorded AFTER all D2H transfers for this chunk are queued. The event represents "all D2H transfers for this chunk are complete".

### 3. sync_stream() - Wait for All CUDA Operations

**After kernel.run() completes** (solver.py line 458):
```python
self.memory_manager.sync_stream(self.kernel)
```

**mem_manager.sync_stream()** (mem_manager.py lines 1166-1168):
```python
def sync_stream(self, instance: object) -> None:
    stream = self.get_stream(instance)
    stream.synchronize()  # BLOCKS until ALL operations on stream complete
```

**What stream.synchronize() does**:
- Blocks CPU thread
- Waits for ALL queued operations on this stream to complete:
  - All kernel launches
  - All H2D transfers
  - All D2H transfers  
  - All event recordings
- Returns only when stream is completely idle

**Critical Insight**: After `sync_stream()` returns, ALL events recorded on that stream are GUARANTEED to be signaled/complete.

### 4. wait_for_writeback() - Process Pending Buffers

**After sync_stream()** (solver.py line 459):
```python
self.kernel.wait_for_writeback()
```

**wait_for_writeback()** (BatchSolverKernel.py):
```python
def wait_for_writeback(self):
    self.output_arrays.wait_pending()
```

**wait_pending()** (BatchOutputArrays.py line 436):
```python
def wait_pending(self, timeout: Optional[float] = None) -> None:
    self._watcher.wait_all(timeout=timeout)
```

**watcher.wait_all()** (writeback_watcher.py lines 206-217):
```python
def wait_all(self, timeout: Optional[float] = 10) -> None:
    start_time = perf_counter()
    while True:
        with self._lock:
            if self._pending_count == 0:
                return  # All tasks complete
        
        if timeout is not None:
            elapsed = perf_counter() - start_time
            if elapsed >= timeout:
                raise TimeoutError(...)
        
        sleep(self._poll_interval)  # Default 0.0001s
```

**What happens**: Main thread polls `_pending_count` until it reaches 0.

### 5. Watcher Thread - Background Processing

**Watcher._poll_loop()** runs in daemon thread (lines 226-276):
```python
def _poll_loop(self) -> None:
    pending_tasks = []
    
    while not self._stop_event.is_set():
        # Process pending tasks
        for task in pending_tasks:
            if self._process_task(task):  # Returns True if complete
                with self._lock:
                    self._pending_count -= 1
            else:
                still_pending.append(task)
        
        # Get new tasks from queue
        try:
            task = self._queue.get_nowait()
            if self._process_task(task):
                with self._lock:
                    self._pending_count -= 1
            else:
                pending_tasks.append(task)
        except Empty:
            pass
        
        sleep(self._poll_interval)
```

**_process_task()** (lines 277-309):
```python
def _process_task(self, task: WritebackTask) -> bool:
    # Check if event is complete
    if CUDA_SIMULATION or task.event is None:
        is_complete = True
    else:
        is_complete = task.event.query()  # Non-blocking query
    
    if is_complete:
        # Copy: pinned buffer → host array
        if task.data_shape is not None:
            buffer_slice = tuple(slice(0, s) for s in task.data_shape)
            task.target_array[:] = task.buffer.array[buffer_slice]
        else:
            task.target_array[:] = task.buffer.array
        
        # Release buffer back to pool
        task.buffer_pool.release(task.buffer)
        return True  # Task complete
    
    return False  # Task still pending
```

## Expected Behavior (Correct Flow)

### Single Test Execution

1. **solve() starts**: kernel.run() executes
2. **For each chunk**:
   - finalise() queues D2H transfers
   - finalise() records event on stream
   - finalise() submits tasks to watcher (all with same event)
3. **kernel.run() completes**
4. **sync_stream()**: Blocks until stream idle
   - **After sync_stream() returns**: All events are signaled
5. **wait_for_writeback()**: Polls `_pending_count`
6. **Watcher thread**:
   - Queries events (all return True immediately after sync)
   - Copies buffers to host arrays
   - Releases buffers
   - Decrements `_pending_count`
7. **`_pending_count` reaches 0**: wait_for_writeback() returns
8. **solve() returns**: SolveResult

**Expected duration**: Milliseconds after sync_stream()

### Multiple Events Per finalise() - Analysis

**Original Design**: One event per chunk's finalise() call
- Event recorded after all D2H transfers for chunk queued
- All buffers from that chunk share the event
- Makes sense: event = "this chunk's transfers are done"

**Alternative (My Incorrect "Fix")**: One event per buffer
- N events instead of 1 per chunk
- Each event recorded individually
- More GPU resources
- **NO BENEFIT**: After sync_stream(), ALL events complete regardless
- **POTENTIAL HARM**: More overhead, more resources

**User is correct**: Multiple events in same OutputArrays instance don't negatively affect behavior because sync_stream() ensures they're all complete before wait_for_writeback() even starts.

## Actual Problem - Why Tests Hang

If the above flow works correctly, tests should NOT hang. So what's actually broken?

### Hypothesis 1: Stream Not Actually Synchronized

**Possible Issue**: Different streams used for event recording vs synchronization?

**Check**:
- finalise() gets stream: `stream = self._memory_manager.get_stream(self)` (line 370)
- sync_stream() syncs: `stream = self.get_stream(instance)` (mem_manager line 1167)

Both use `get_stream()` on same instance → **same stream** → Not the issue.

### Hypothesis 2: Event Query Blocking/Failing

**Possible Issue**: On CUDA hardware, `event.query()` might:
- Block instead of returning immediately
- Return False even when event should be complete
- Hang if event in bad state

**Evidence Needed**: Actual CUDA hardware testing to see event.query() behavior

### Hypothesis 3: Watcher Thread Not Running

**Possible Issue**: Thread never starts or stops prematurely?

**Check start** (lines 115-121):
```python
def start(self) -> None:
    if self._thread is not None and self._thread.is_alive():
        return  # Already running
    self._stop_event.clear()
    self._thread = Thread(target=self._poll_loop, daemon=True)
    self._thread.start()
```

Thread starts on first submit. Daemon=True means it won't prevent process exit.

**Check stop** (lines 219-224):
```python
def shutdown(self) -> None:
    self._stop_event.set()
    if self._thread is not None:
        self._thread.join(timeout=1.0)
        self._thread = None
```

**ISSUE**: `join(timeout=1.0)` - only waits 1 second!

If watcher thread is still processing when shutdown() called:
- Thread continues as daemon
- `self._thread = None` - reference lost
- BUT thread still running
- `_pending_count` not managed correctly

### Hypothesis 4: Buffer Pool Exhaustion

**Possible Issue**: Pinned buffers not released, pool exhausted

**Flow**:
1. Test A acquires buffers
2. Event should signal → watcher copies → buffer released
3. But if watcher doesn't run or event doesn't signal:
   - Buffers stay "in use"
   - Pool exhausted
4. Test B tries to acquire buffer → **blocks forever**

### Hypothesis 5: xdist Process Isolation

**xdist runs tests in separate processes**

Potential issues:
- CUDA context not properly initialized per process?
- Stream pooling conflicts between processes?
- Event state not valid across process boundaries?

**Not likely** because user says first 3 tests complete successfully.

## Proposed Investigation Steps

1. **Add logging** to see exactly where tests hang:
   - Before/after sync_stream()
   - In wait_for_writeback()
   - In watcher._poll_loop()
   - In _process_task()

2. **Check buffer pool state** when hung:
   - How many buffers in use?
   - How many buffers free?
   - Which test owns which buffers?

3. **Check watcher thread state** when hung:
   - Is thread alive?
   - What is `_pending_count`?
   - How many tasks in queue?
   - What does event.query() return?

4. **Test without xdist**:
   - Run tests serially
   - If they pass → confirms concurrency issue
   - If they hang → different root cause

## Conclusion

The solve() end process should work correctly:
1. sync_stream() ensures all events are signaled
2. wait_for_writeback() should complete immediately
3. Multiple events per OutputArrays instance are fine

The bug is NOT in event sharing design. The bug is likely:
- **Buffer pool exhaustion** from incomplete cleanup
- **Watcher thread lifecycle** issues
- **Test fixture cleanup** not waiting properly
- **CUDA context/stream** issues specific to xdist

My previous "fixes" were addressing symptoms, not root cause. Need to identify why watcher doesn't complete tasks even though events should be signaled.
