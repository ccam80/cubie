# Runtime Logging - Agent Technical Plan

## Overview

This plan details the implementation of runtime logging for CuBIE's BatchSolverKernel and Solver classes. The system instruments GPU kernel execution, memory transfers, and outer method timing using the existing TimeLogger infrastructure with a new CUDAEvent abstraction in cuda_simsafe.py.

**CRITICAL REQUIREMENTS FROM REVIEW:**
1. Consolidated transfer events (ONE h2d_transfer, ONE d2h_transfer)
2. NO stream synchronization during execution
3. CUDAEvent abstraction in cuda_simsafe.py
4. Deferred event retrieval after sync
5. Always log events even if duration is 0

## Component Specifications

### 1. CUDAEvent Class (cuda_simsafe.py)

**Location:** `src/cubie/cuda_simsafe.py`

**Purpose:** Abstract CUDA event timing with CUDASIM fallback

**Expected Behavior:**
- In CUDA mode: Uses `numba.cuda.event()` for GPU-accurate timing
- In CUDASIM mode: Falls back to `time.perf_counter()` wall-clock timing
- Self-registers with TimeLogger for deferred retrieval
- Provides `record_start()` and `record_end()` methods
- Provides `elapsed_time_ms()` for post-sync time calculation

**Component Interactions:**
- Imports: `numba.cuda`, `time`, `CUDA_SIMULATION` constant
- Interacts with: TimeLogger (registers self for retrieval)
- Used by: BatchSolverKernel.run(), InputArrays, OutputArrays

**Data Structures:**

```python
class CUDAEvent:
    """CUDA event wrapper with CUDASIM fallback.
    
    Attributes
    ----------
    name : str
        Event label for TimeLogger
    category : str
        TimeLogger category ('runtime')
    _start_event : cuda.Event or None
        CUDA start event (None in CUDASIM)
    _end_event : cuda.Event or None
        CUDA end event (None in CUDASIM)
    _start_time : float or None
        Wall-clock start time (CUDASIM only)
    _end_time : float or None
        Wall-clock end time (CUDASIM only)
    """
```

**Key Methods:**
- `__init__(name, category, timelogger)`: Create event and register with TimeLogger
- `record_start(stream=None)`: Record start timestamp/event
- `record_end(stream=None)`: Record end timestamp/event
- `elapsed_time_ms()`: Calculate elapsed time (blocks until events complete in CUDA mode)

### 2. TimeLogger Extensions

**Location:** `src/cubie/time_logger.py`

**New Methods Required:**

**`register_cuda_event(event: CUDAEvent)`**
- Stores CUDAEvent instance in internal list
- Called by CUDAEvent.__init__()
- No-op when verbosity is None

**`retrieve_cuda_events()`**
- Iterates through registered CUDAEvents
- Calls `elapsed_time_ms()` on each
- Records duration to TimeLogger events
- Called by Solver.solve() after sync_stream
- Clears the registered events list after retrieval

**Data Structures:**
```python
# Add to TimeLogger.__init__():
self._cuda_events: list[CUDAEvent] = []
```

### 3. Event Model

**Events Using CUDAEvent (GPU timing):**
- `kernel_launch_{chunk_idx}` - One per kernel launch per chunk
- `h2d_transfer` - ONE event for ALL host-to-device transfers
- `d2h_transfer` - ONE event for ALL device-to-host transfers

**Events Using Standard TimeLogger (wall-clock):**
- `solve_ivp_total` - Full solve_ivp() duration
- `solver_solve_total` - Full Solver.solve() duration
- `kernel_run_total` - Full BatchSolverKernel.run() duration

**Registration:**
All events registered with category='runtime'

### 4. solve_ivp Function Timing

**File:** `src/cubie/batchsolving/solver.py`

**Expected Behavior:**
- Register event at module level
- Start timing at function entry
- Stop timing before return
- Uses standard TimeLogger (wall-clock, not CUDAEvent)

**Integration Points:**
- Calls Solver.solve() internally
- Timing captures complete function including Solver creation

### 5. Solver.solve() Method Timing

**File:** `src/cubie/batchsolving/solver.py`

**Expected Behavior:**
- Register event in Solver.__init__()
- Start timing at method entry
- After sync_stream(), call `retrieve_cuda_events()` to collect GPU timings
- Stop timing before return
- Uses standard TimeLogger (wall-clock)

**Critical Integration:**
```python
def solve(self, ...):
    _default_timelogger.start_event('solver_solve_total')
    # ... existing code ...
    self.kernel.run(...)
    self.memory_manager.sync_stream(self.kernel)
    _default_timelogger.retrieve_cuda_events()  # NEW: collect GPU timings
    result = SolveResult.from_solver(self, ...)
    _default_timelogger.stop_event('solver_solve_total')
    return result
```

### 6. BatchSolverKernel.run() Method Timing

**File:** `src/cubie/batchsolving/BatchSolverKernel.py`

**Expected Behavior:**
- Register events in __init__()
- Outer wall-clock timing with TimeLogger
- Per-chunk CUDAEvent timing for kernel launches
- Consolidated CUDAEvent for h2d transfers
- Consolidated CUDAEvent for d2h transfers

**Consolidated Transfer Pattern:**

```python
def run(self, ...):
    _default_timelogger.start_event('kernel_run_total')
    
    # Create ONE CUDAEvent for ALL h2d transfers
    h2d_event = CUDAEvent('h2d_transfer', 'runtime', _default_timelogger)
    h2d_event.record_start(stream)
    
    for i in range(self.chunks):
        # All h2d transfers for this chunk
        self.input_arrays.initialise(indices)
    
    h2d_event.record_end(stream)  # After ALL transfers
    
    # Per-chunk kernel launches
    for i in range(self.chunks):
        kernel_event = CUDAEvent(f'kernel_launch_{i}', 'runtime', ...)
        kernel_event.record_start(stream)
        self.kernel[...]  # Launch
        kernel_event.record_end(stream)
    
    # Create ONE CUDAEvent for ALL d2h transfers
    d2h_event = CUDAEvent('d2h_transfer', 'runtime', _default_timelogger)
    d2h_event.record_start(stream)
    
    for i in range(self.chunks):
        self.output_arrays.finalise(indices)
    
    d2h_event.record_end(stream)  # After ALL transfers
    
    _default_timelogger.stop_event('kernel_run_total')
```

**CRITICAL: No sync_stream() calls during this method!**
The sync happens in Solver.solve() after run() returns.

### 7. InputArrays Transfer Integration

**File:** `src/cubie/batchsolving/arrays/BatchInputArrays.py`

**Expected Behavior:**
- NO individual timing in initialise()
- Transfers are covered by h2d_event created in BatchSolverKernel.run()
- No changes needed to this file for timing

### 8. OutputArrays Transfer Integration

**File:** `src/cubie/batchsolving/arrays/BatchOutputArrays.py`

**Expected Behavior:**
- NO individual timing in finalise()
- Transfers are covered by d2h_event created in BatchSolverKernel.run()
- No changes needed to this file for timing

## Edge Cases

### CUDASIM Mode
- CUDAEvent uses time.perf_counter() instead of cuda.event()
- elapsed_time_ms() returns wall-clock delta * 1000
- No stream parameter needed in CUDASIM
- All functionality works identically

### Verbosity=None Mode
- TimeLogger methods are no-ops
- CUDAEvent.register_cuda_event() is no-op
- retrieve_cuda_events() is no-op
- Zero overhead when timing disabled

### Zero-Duration Events
- Always record h2d_transfer even if no arrays to transfer
- Always record d2h_transfer even if no arrays to transfer
- Event with 0.0 duration is valid and logged

### Multi-Chunk Execution
- One h2d_transfer event spanning ALL chunks' h2d transfers
- One kernel_launch_{i} event PER chunk
- One d2h_transfer event spanning ALL chunks' d2h transfers

### Stream Synchronization
- NO sync during BatchSolverKernel.run()
- Sync happens in Solver.solve() after run() returns
- retrieve_cuda_events() called after sync
- elapsed_time_ms() blocks until events complete (implicit sync per event)

## Dependencies Between Components

### Component Creation Order
1. TimeLogger (module-level singleton exists)
2. CUDAEvent class (defined in cuda_simsafe.py)
3. Solver.__init__() registers events
4. BatchSolverKernel created by Solver

### Execution Flow
1. solve_ivp() → start_event('solve_ivp_total')
2. Solver.solve() → start_event('solver_solve_total')
3. BatchSolverKernel.run() → start_event('kernel_run_total')
4. Create h2d_event, record_start
5. For each chunk: initialise() (no timing here)
6. h2d_event.record_end
7. For each chunk: create kernel_event, record_start, launch, record_end
8. Create d2h_event, record_start
9. For each chunk: finalise() (no timing here)
10. d2h_event.record_end
11. stop_event('kernel_run_total')
12. run() returns to solve()
13. sync_stream()
14. retrieve_cuda_events() → calculates all GPU timings
15. stop_event('solver_solve_total')
16. solve() returns to solve_ivp()
17. stop_event('solve_ivp_total')

## File Modification Summary

**Files to Create/Modify:**

1. `src/cubie/cuda_simsafe.py`
   - ADD CUDAEvent class
   - Update __all__ list

2. `src/cubie/time_logger.py`
   - ADD `_cuda_events` list attribute
   - ADD `register_cuda_event()` method
   - ADD `retrieve_cuda_events()` method

3. `src/cubie/batchsolving/solver.py`
   - ADD module-level event registration
   - ADD timing in solve_ivp()
   - ADD timing in Solver.solve()
   - ADD retrieve_cuda_events() call after sync

4. `src/cubie/batchsolving/BatchSolverKernel.py`
   - ADD event registration in __init__()
   - ADD outer timing in run()
   - ADD consolidated CUDAEvent timing for transfers
   - ADD per-chunk CUDAEvent timing for kernel launches

**Files NOT Modified:**
- BatchInputArrays.py (no timing needed)
- BatchOutputArrays.py (no timing needed)
- BaseArrayManager.py (no timing needed)

## Testing Considerations

### Key Test Scenarios
- CUDAEvent in CUDA mode (requires GPU)
- CUDAEvent in CUDASIM mode (fallback path)
- TimeLogger verbosity=None (no-op)
- TimeLogger verbosity='verbose' (prints inline)
- Multi-chunk execution
- Zero-duration events (no arrays transferred)
- retrieve_cuda_events() after sync

### Test Fixtures Needed
- Simple ODE system for quick solve
- Verbosity level parameterization
- CUDASIM marker for GPU-less tests
