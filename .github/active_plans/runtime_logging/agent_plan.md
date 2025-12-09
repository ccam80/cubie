# Runtime Logging Technical Specification

## Component 1: CUDAEvent Class (cuda_simsafe.py)

### Purpose
Wrapper class for CUDA event pairs providing CUDASIM-compatible timing measurements.

### Location
`src/cubie/cuda_simsafe.py` - Add after existing helper classes

### Behavior
- Manages lifecycle of CUDA event pairs (start event, end event)
- Records events on specified CUDA streams
- Calculates elapsed time between events (post-sync only)
- Provides fallback behavior in CUDASIM mode using time.perf_counter()

### Attributes
- `name`: str - Event identifier (e.g., "kernel_chunk_0")
- `category`: str - TimeLogger category ('runtime', 'codegen', 'compile')
- `_start_event`: cuda.event or None - Start timestamp event (CUDA mode)
- `_end_event`: cuda.event or None - End timestamp event (CUDA mode)
- `_start_time`: float or None - Start timestamp (CUDASIM mode)
- `_end_time`: float or None - End timestamp (CUDASIM mode)

### Methods

**`__init__(self, name: str, category: str = 'runtime')`**
- Initialize event pair
- In CUDA mode: Create `cuda.event()` objects
- In CUDASIM mode: Initialize timestamp placeholders to None
- Store name and category for TimeLogger integration

**`record_start(self, stream)`**
- Record start timestamp on given stream
- In CUDA mode: Call `self._start_event.record(stream)`
- In CUDASIM mode: Store `time.perf_counter()`
- Must be called before corresponding `record_end()`

**`record_end(self, stream)`**
- Record end timestamp on given stream
- In CUDA mode: Call `self._end_event.record(stream)`
- In CUDASIM mode: Store `time.perf_counter()`
- Must be called after corresponding `record_start()`

**`elapsed_time_ms(self) -> float`**
- Calculate elapsed time in milliseconds
- **CRITICAL**: Must NOT block or synchronize
- In CUDA mode: Call `cuda.event_elapsed_time(self._start_event, self._end_event)`
- In CUDASIM mode: Return `(self._end_time - self._start_time) * 1000.0`
- Called AFTER stream has been synchronized externally
- Returns immediately with result

### Integration Points
- Imported by BatchSolverKernel for event creation
- Events registered with TimeLogger for summary reporting
- Must appear in `__all__` export list in cuda_simsafe.py

### Edge Cases
- Handle case where record_end() called without record_start()
- Handle case where elapsed_time_ms() called before both recordings
- CUDASIM fallback should produce reasonable timing estimates

## Component 2: TimeLogger Extensions (time_logger.py)

### New Attributes
- `_cuda_events`: list[CUDAEvent] - Storage for registered CUDA events

### Initialize in `__init__()`
```python
self._cuda_events: list[CUDAEvent] = []
```

### New Method: `register_cuda_event()`

**Signature:**
```python
def register_cuda_event(self, event: CUDAEvent) -> None
```

**Behavior:**
- Store CUDAEvent instance in `_cuda_events` list
- Register event name with existing `register_event()` method
- Use event's category and name for registration
- No-op when verbosity is None

**Purpose:**
- Collect CUDA events created during kernel execution
- Enable later retrieval and timing calculation

### New Method: `retrieve_cuda_events()`

**Signature:**
```python
def retrieve_cuda_events(self) -> None
```

**Behavior:**
- Iterate over all registered CUDAEvent instances in `_cuda_events`
- For each event:
  - Call `event.elapsed_time_ms()` to get duration
  - Create TimingEvent with duration in metadata
  - Store elapsed time in milliseconds
  - Append to `self.events` list
- Clear `_cuda_events` list after retrieval
- No-op when verbosity is None

**Timing Details:**
- Called AFTER stream synchronization completes
- `elapsed_time_ms()` returns immediately (no blocking)
- Converts GPU timeline to TimeLogger event format

**Purpose:**
- Convert CUDA event timings to TimeLogger events
- Enable unified reporting with compile/codegen timings

### Modified Method: `print_summary()`

**Changes:**
- Support per-chunk event display at appropriate verbosity
- When category='runtime' and verbosity='verbose' or 'debug':
  - Print individual chunk timings
  - Format: "TIMELOGGER   kernel_chunk_0: 10.345ms"
- When category='runtime' and verbosity='default':
  - Aggregate chunk timings by type
  - Format: "TIMELOGGER   kernel_total: 105.234ms"

**Pattern:**
- Check event name for "_chunk_" substring
- Group events by prefix (h2d_transfer, kernel, d2h_transfer)
- Sum durations for each group

### Integration Points
- Called by Solver.solve() after stream sync
- Events cleared after retrieval to avoid memory growth
- Must handle empty _cuda_events list gracefully

## Component 3: BatchSolverKernel Instrumentation

### New Attributes
- `_cuda_events`: list[CUDAEvent] - Temporary storage for current run's events
- `_gpu_workload_event`: CUDAEvent or None - Overall GPU timeline event

### Initialize in `__init__()`
```python
self._cuda_events: list[CUDAEvent] = []
self._gpu_workload_event: Optional[CUDAEvent] = None
```

### Modified Method: `run()`

#### Event Creation (before chunk loop)
```python
if _default_timelogger.verbosity is not None:
    # Create overall GPU workload event
    self._gpu_workload_event = CUDAEvent("gpu_workload", category="runtime")
    
    # Create per-chunk events
    for i in range(self.chunks):
        h2d_event = CUDAEvent(f"h2d_transfer_chunk_{i}", category="runtime")
        kernel_event = CUDAEvent(f"kernel_chunk_{i}", category="runtime")
        d2h_event = CUDAEvent(f"d2h_transfer_chunk_{i}", category="runtime")
        self._cuda_events.extend([h2d_event, kernel_event, d2h_event])
```

#### Event Recording Pattern

**Before chunk loop:**
```python
if self._gpu_workload_event is not None:
    self._gpu_workload_event.record_start(stream)
```

**Inside chunk loop (for each chunk i):**

Before `self.input_arrays.initialise(indices)`:
```python
if len(self._cuda_events) > 0:
    h2d_event = self._cuda_events[i * 3]
    h2d_event.record_start(stream)
```

After `self.input_arrays.initialise(indices)`:
```python
if len(self._cuda_events) > 0:
    h2d_event = self._cuda_events[i * 3]
    h2d_event.record_end(stream)
```

Before kernel launch:
```python
if len(self._cuda_events) > 0:
    kernel_event = self._cuda_events[i * 3 + 1]
    kernel_event.record_start(stream)
```

After kernel launch (before input/output finalise):
```python
if len(self._cuda_events) > 0:
    kernel_event = self._cuda_events[i * 3 + 1]
    kernel_event.record_end(stream)
```

Before `self.output_arrays.finalise(indices)`:
```python
if len(self._cuda_events) > 0:
    d2h_event = self._cuda_events[i * 3 + 2]
    d2h_event.record_start(stream)
```

After `self.output_arrays.finalise(indices)`:
```python
if len(self._cuda_events) > 0:
    d2h_event = self._cuda_events[i * 3 + 2]
    d2h_event.record_end(stream)
```

**After chunk loop:**
```python
if self._gpu_workload_event is not None:
    self._gpu_workload_event.record_end(stream)
    
    # Register all events with TimeLogger
    _default_timelogger.register_cuda_event(self._gpu_workload_event)
    for event in self._cuda_events:
        _default_timelogger.register_cuda_event(event)
```

### Behavior Details
- Events created only when verbosity is not None (zero overhead otherwise)
- All events record on the same stream (passed to run() method)
- Event list indexed by chunk: chunk i uses events at positions [i*3, i*3+1, i*3+2]
- Events registered with TimeLogger at end of run() for later retrieval

### Integration Points
- Import CUDAEvent from cuda_simsafe
- Import _default_timelogger from time_logger
- Events handed off to TimeLogger; BatchSolverKernel doesn't calculate timings

## Component 4: Solver Wall-Clock Timing

### Modified Method: `solve()`

#### Add at start of method:
```python
_default_timelogger.register_event("solver_solve", "runtime", 
                                   "Wall-clock time for Solver.solve()")
_default_timelogger.start_event("solver_solve")
```

#### After `self.memory_manager.sync_stream(self.kernel)`:
```python
# Retrieve CUDA event timings (after sync completes)
_default_timelogger.retrieve_cuda_events()
```

#### Before final return:
```python
_default_timelogger.stop_event("solver_solve")
_default_timelogger.print_summary(category='runtime')
```

### Behavior
- Wall-clock timing captures entire Solver.solve() duration
- Includes grid building, kernel execution, stream sync, result creation
- CUDA event retrieval happens after sync (events already completed)
- Summary printed before returning to user

### Integration Points
- Import _default_timelogger from time_logger
- No changes to return value or method signature
- Timing only active when verbosity is not None

## Component 5: solve_ivp() Wall-Clock Timing

### Modified Function: `solve_ivp()`

#### Add after Solver instantiation:
```python
_default_timelogger.register_event("solve_ivp", "runtime", 
                                   "Wall-clock time for solve_ivp()")
_default_timelogger.start_event("solve_ivp")
```

#### Before return:
```python
_default_timelogger.stop_event("solve_ivp")
# Note: Solver.solve() already prints summary, don't duplicate
```

### Behavior
- Captures full solve_ivp() wall-clock time
- Includes Solver construction and solve() call
- Summary printed by Solver.solve(), not duplicated here

### Integration Points
- Import _default_timelogger from time_logger
- No changes to return value or function signature

## Data Structures

### CUDAEvent Instance Example
```python
event = CUDAEvent("kernel_chunk_0", category="runtime")
# In CUDA mode:
event._start_event = cuda.event()  # CUDA event object
event._end_event = cuda.event()    # CUDA event object
# In CUDASIM mode:
event._start_time = None  # Will be set by record_start()
event._end_time = None    # Will be set by record_end()
```

### TimeLogger._cuda_events Structure
```python
# List of CUDAEvent instances registered during execution
_cuda_events = [
    CUDAEvent("gpu_workload", "runtime"),
    CUDAEvent("h2d_transfer_chunk_0", "runtime"),
    CUDAEvent("kernel_chunk_0", "runtime"),
    CUDAEvent("d2h_transfer_chunk_0", "runtime"),
    CUDAEvent("h2d_transfer_chunk_1", "runtime"),
    CUDAEvent("kernel_chunk_1", "runtime"),
    CUDAEvent("d2h_transfer_chunk_1", "runtime"),
    # ... more chunks
]
```

### BatchSolverKernel._cuda_events Structure
```python
# Created in run() before chunk loop
# 3 events per chunk: h2d, kernel, d2h
# Chunk i events at indices [i*3, i*3+1, i*3+2]
_cuda_events = [
    CUDAEvent("h2d_transfer_chunk_0", "runtime"),    # index 0
    CUDAEvent("kernel_chunk_0", "runtime"),          # index 1
    CUDAEvent("d2h_transfer_chunk_0", "runtime"),    # index 2
    CUDAEvent("h2d_transfer_chunk_1", "runtime"),    # index 3
    CUDAEvent("kernel_chunk_1", "runtime"),          # index 4
    CUDAEvent("d2h_transfer_chunk_1", "runtime"),    # index 5
    # ... pattern continues
]
```

## Expected Interactions

### Normal Execution Flow

1. **User calls solve_ivp() or Solver.solve()**
   - TimeLogger registers and starts wall-clock event

2. **Solver.solve() calls BatchSolverKernel.run()**
   - BatchSolverKernel creates CUDAEvent instances (if verbosity active)
   - Events stored in BatchSolverKernel._cuda_events list

3. **BatchSolverKernel.run() executes chunk loop**
   - Before loop: gpu_workload.record_start()
   - Each chunk: record_start/record_end for h2d, kernel, d2h
   - After loop: gpu_workload.record_end()
   - Events registered with TimeLogger

4. **Solver.solve() calls sync_stream()**
   - Stream blocks until all GPU work completes
   - CUDA events now contain valid timing data

5. **Solver.solve() calls TimeLogger.retrieve_cuda_events()**
   - Retrieves timing from each CUDAEvent
   - Creates TimingEvent entries in TimeLogger
   - Clears _cuda_events list

6. **Solver.solve() completes**
   - Stops wall-clock event
   - Prints runtime summary
   - Returns SolveResult to user

### Verbosity=None Flow

1. **User sets verbosity to None**
   - All event creation skipped
   - Empty _cuda_events lists
   - No performance overhead

2. **BatchSolverKernel.run() executes normally**
   - Checks `if _default_timelogger.verbosity is not None` → False
   - Skips event creation
   - No event recording calls

3. **Solver.solve() completes normally**
   - retrieve_cuda_events() is no-op (empty list)
   - No summary printed
   - Zero overhead

### CUDASIM Mode Flow

1. **Running with NUMBA_ENABLE_CUDASIM=1**
   - CUDAEvent uses time.perf_counter() fallback

2. **Event recording**
   - record_start() stores current time.perf_counter()
   - record_end() stores current time.perf_counter()

3. **Timing retrieval**
   - elapsed_time_ms() calculates delta and converts to ms
   - Rest of flow identical to CUDA mode

## Error Handling

### Invalid Event States
- Calling elapsed_time_ms() before both record_start() and record_end()
  - Return 0.0 or raise ValueError (to be decided in implementation)
  
### Empty Event Lists
- TimeLogger.retrieve_cuda_events() with empty _cuda_events
  - No-op, no errors

### Stream Mismatch
- All events in a run must record on same stream
  - Not enforced by code, documented requirement
  - Incorrect stream usage would give wrong timings

### CUDASIM Limitations
- CUDASIM timing is wall-clock, not GPU timeline
- Acceptable for testing, not for profiling
- Documented limitation

## Dependencies

### Imports Required

**cuda_simsafe.py:**
```python
import time  # For CUDASIM fallback
from numba import cuda  # Already imported
```

**time_logger.py:**
```python
from cubie.cuda_simsafe import CUDAEvent
from typing import List  # Already imported
```

**BatchSolverKernel.py:**
```python
from cubie.cuda_simsafe import CUDAEvent
from cubie.time_logger import _default_timelogger  # Already imported
```

**solver.py:**
```python
from cubie.time_logger import _default_timelogger  # Already imported
```

### Module Dependencies
- cuda_simsafe.py: No new dependencies
- time_logger.py: Depends on cuda_simsafe for CUDAEvent
- BatchSolverKernel.py: Depends on both cuda_simsafe and time_logger
- solver.py: Depends on time_logger (already does)

## Testing Considerations

### Unit Tests Needed
- CUDAEvent creation and recording (CUDA mode)
- CUDAEvent creation and recording (CUDASIM mode)
- CUDAEvent.elapsed_time_ms() (both modes)
- TimeLogger.register_cuda_event()
- TimeLogger.retrieve_cuda_events()
- BatchSolverKernel event creation and registration
- Solver timing retrieval and summary

### Integration Tests Needed
- Full solve_ivp() with timing enabled
- Full Solver.solve() with timing enabled
- Multi-chunk execution with per-chunk timing
- Verbosity=None (verify zero overhead)

### Test Markers
- Mark CUDA-specific tests with `@pytest.mark.nocudasim`
- Tests that work in both modes don't need marker
- CUDASIM-specific tests marked with `@pytest.mark.sim_only`

## Performance Characteristics

### Overhead When Enabled
- Event creation: ~1-2 microseconds per event
- Event recording: ~0.5-1 microseconds per call
- Total per chunk: ~10 microseconds (6 calls × 3 events)
- Negligible compared to kernel execution (milliseconds)

### Overhead When Disabled (verbosity=None)
- Zero: All instrumentation skipped via early returns
- No event objects created
- No event recording calls made

### Memory Usage
- ~100 bytes per CUDAEvent instance
- 3 events per chunk + 1 overall event
- For 10 chunks: ~3.1 KB temporary storage
- Cleared after retrieval

## Alignment with CuBIE Patterns

### CUDAFactory Pattern
- Not applicable: CUDAEvent is not a CUDAFactory subclass
- CUDAEvent wraps runtime events, not compiled functions

### Attrs Classes Pattern
- Consider making CUDAEvent an attrs class if beneficial
- Not required since it's a simple wrapper

### TimeLogger Integration
- Follows existing pattern: register_event(), start_event(), stop_event()
- Extends with new category: 'runtime'
- Consistent with 'codegen' and 'compile' categories

### CUDASIM Compatibility
- Follows cuda_simsafe.py patterns
- Conditional behavior based on CUDA_SIMULATION flag
- Fallback using Python standard library

### Zero-Overhead Principle
- All instrumentation guarded by verbosity checks
- No overhead when disabled
- Matches existing TimeLogger behavior
