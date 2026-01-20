# Chunked Test Concurrency Issue Analysis

## Problem Statement

Tests using `chunked_solved_solver` run infinitely (12+ hours) when executed with xdist parallel execution, but complete individually.

### Observed Behavior

**With xdist (parallel)**:
- Only 3 tests ever complete: `test_chunked_solve_produces_valid_output`, `test_convert_host_to_numpy_uses_needs_chunked_transfer`, `test_run_executes_with_chunking`
- All use `chunked_solved_solver` fixture
- Tests hang forever after these 3 complete

**Without xdist (serial)**:
- `test_chunked_shape_equals_shape_when_not_chunking` completes
- Uses `unchunked_solved_solver` (session-scoped)

## Key Observations

### Fixture Scoping

```python
@pytest.fixture(scope="function")
def chunked_solved_solver(system, precision, low_mem_solver, driver_settings):
    # ... creates solver and calls solve()
    result = solver.solve(...)  # Starts writeback watcher thread
    return solver, result

@pytest.fixture(scope="session")
def unchunked_solved_solver(...):
    # ... creates solver and calls solve()
    result = solver.solve(...)
    return solver, result
```

**Issue**: `chunked_solved_solver` is **function-scoped** but calls `solve()` which starts a **background thread** (WritebackWatcher).

### Threading in WritebackWatcher

```python
class WritebackWatcher:
    def start(self):
        self._thread = Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
    
    def wait_all(self, timeout=10):
        # Blocks until _pending_count == 0
        while True:
            with self._lock:
                if self._pending_count == 0:
                    return
            # ... timeout check ...
            sleep(self._poll_interval)
```

**Problem**: The watcher thread:
1. Is daemon (won't block process exit)
2. Polls a queue and processes async writeback tasks
3. Uses threading Lock for _pending_count

### Singleton State

**Module-level singletons**:
- `buffer_registry = BufferRegistry()` in `buffer_registry.py`
- `default_memmgr = MemoryManager()` in `memory/__init__.py`

With xdist, each worker is a **separate process**, so these should NOT be shared. However, within a single worker, they ARE shared across all tests.

## Hypothesis: Race Condition in WritebackWatcher

### Scenario

1. **Test A** uses `chunked_solved_solver`:
   - Creates solver instance A
   - Calls `solve()` → creates WritebackWatcher A
   - Watcher A starts thread
   - `solve()` calls `wait_for_writeback()` → `watcher.wait_all()`
   - Watcher processes tasks, `_pending_count` decrements
   - Test A completes when `_pending_count == 0`

2. **Test B** starts (parallel on xdist):
   - Creates solver instance B
   - Calls `solve()` → creates WritebackWatcher B
   - Watcher B starts thread
   - But what if there's contention?

### Potential Issues

**Issue 1: CUDA Event Polling in CUDASIM**

The watcher polls CUDA events:
```python
def _process_task(self, task):
    if task.event is None:  # CUDASIM
        completed = True
    else:
        completed = task.event.query()
```

In CUDASIM mode, `task.event is None`, so tasks complete immediately. But in real CUDA, event.query() polls hardware state.

**What if multiple watchers poll the same event?**

**Issue 2: Buffer Pool Contention**

Each solver has its own ChunkBufferPool, but they might contend for:
- Pinned memory allocation (global CUDA resource)
- CUDA context operations
- Stream synchronization

### Critical Clue

> "the same three run, and no others make it to completion"

This suggests:
1. Three tests start and complete successfully
2. Subsequent tests NEVER complete
3. This implies a **resource exhaustion** or **deadlock** after 3 concurrent operations

## Root Cause Theory

### Pinned Memory Exhaustion

When chunking, solvers use **pinned memory** for async transfers. If:
1. Test A allocates pinned memory
2. Test A's watcher holds buffers
3. Test B tries to allocate more pinned memory
4. System runs out of pinned memory
5. Test B blocks waiting for allocation
6. Test A's watcher can't release buffers because it's waiting on CUDA events
7. **Deadlock**

### Evidence

```python
# In chunked_solved_solver fixture:
solver = low_mem_solver  # MockMemoryManager with forced_free_mem
result = solver.solve(...)  # Calls solve, starts watcher
return solver, result  # Returns WITHOUT shutting down watcher
```

**The fixture returns the solver with an active watcher thread!**

Tests use the solver after fixture returns:
```python
def test_input_buffers_released_after_kernel(chunked_solved_solver):
    chunked_solver, result_chunked = chunked_solved_solver
    # Access chunked_solver internals
    input_arrays = chunked_solver.kernel.input_arrays
    assert len(input_arrays._active_buffers) == 0
```

If multiple tests run concurrently and all have active watchers, they could:
1. Compete for pinned memory
2. Hold buffers in "in_use" state
3. Block each other from proceeding

## Solution Approaches

### Option 1: Ensure Watcher Shutdown in Fixture

Modify `chunked_solved_solver` to shutdown watcher before returning:

```python
@pytest.fixture(scope="function")
def chunked_solved_solver(...):
    solver = low_mem_solver
    result = solver.solve(...)
    
    # Ensure watcher completes and shuts down
    solver.kernel.wait_for_writeback()
    solver.kernel.output_arrays._watcher.shutdown()
    
    return solver, result
```

### Option 2: Add Fixture Cleanup

Use yield to ensure cleanup:

```python
@pytest.fixture(scope="function")
def chunked_solved_solver(...):
    solver = low_mem_solver
    result = solver.solve(...)
    
    yield solver, result
    
    # Cleanup
    if hasattr(solver.kernel.output_arrays, '_watcher'):
        solver.kernel.output_arrays._watcher.shutdown()
    solver.kernel.output_arrays.reset()
```

### Option 3: Force Serial Execution for Chunked Tests

Mark chunked tests with xdist group to prevent parallel execution:

```python
@pytest.mark.xdist_group(name="chunked_tests")
def test_run_executes_with_chunking(...):
    ...
```

## Recommended Fix

**Immediate**: Add proper cleanup to `chunked_solved_solver` fixture using yield pattern.

**Long-term**: Review watcher lifecycle management and consider:
1. Automatic watcher shutdown when OutputArrays is destroyed
2. Context manager pattern for solver usage
3. Better resource tracking for pinned memory
