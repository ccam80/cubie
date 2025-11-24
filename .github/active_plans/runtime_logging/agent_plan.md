# Runtime Logging - Agent Technical Plan

## Overview

This plan details the implementation of runtime logging for CuBIE's BatchSolverKernel and Solver classes. The system will instrument GPU kernel execution, memory transfers, and outer method timing using the existing TimeLogger infrastructure.

## Component Specifications

### 1. Event Registration

#### Location: Multiple components during initialization

**Events to Register:**
- `solve_ivp_execution` - Total time for solve_ivp() function
- `solver_solve_execution` - Total time for Solver.solve() method
- `kernel_run_execution` - Total time for BatchSolverKernel.run() method
- `kernel_launch` - GPU kernel execution time (per chunk)
- `h2d_transfer_initial_values` - Host-to-device transfer of initial values
- `h2d_transfer_parameters` - Host-to-device transfer of parameters
- `h2d_transfer_driver_coefficients` - Host-to-device transfer of driver coefficients
- `d2h_transfer_state` - Device-to-host transfer of state outputs
- `d2h_transfer_observables` - Device-to-host transfer of observable outputs
- `d2h_transfer_summaries` - Device-to-host transfer of summary outputs
- `d2h_transfer_status_codes` - Device-to-host transfer of status codes
- `d2h_transfer_iteration_counters` - Device-to-host transfer of iteration counters

**Registration Pattern:**
```python
_default_timelogger.register_event(
    label='kernel_launch',
    category='runtime',
    description='GPU kernel execution time per chunk'
)
```

**Registration Locations:**
- `solve_ivp()`: Register at module level (module load time)
- `Solver.__init__()`: Register solve events during initialization
- `BatchSolverKernel.__init__()`: Register kernel and transfer events during initialization

### 2. solve_ivp Function Timing

#### File: src/cubie/batchsolving/solver.py

**Instrumentation Points:**
- Start: Immediately after function entry, before Solver creation
- Stop: Before return statement, after results are ready

**Behavior:**
- Wrap entire function execution
- Include Solver creation and solve() call
- Call `print_summary(category='runtime')` before return if verbosity allows

**Implementation Approach:**
```python
def solve_ivp(...):
    _default_timelogger.start_event('solve_ivp_execution')
    try:
        # ... existing function body ...
        results = solver.solve(...)
        return results
    finally:
        _default_timelogger.stop_event('solve_ivp_execution')
        if _default_timelogger.verbosity == 'default':
            _default_timelogger.print_summary(category='runtime')
```

### 3. Solver.solve() Method Timing

#### File: src/cubie/batchsolving/solver.py

**Instrumentation Points:**
- Start: At method entry
- Stop: Before return statement

**Behavior:**
- Captures complete solve execution including:
  - Grid building
  - Driver interpolator updates
  - kernel.run() call
  - Stream synchronization
  - Result packaging
- Call `print_summary(category='runtime')` in default verbosity before return

**Integration Points:**
- Must complete before `SolveResult.from_solver()` is called
- Stream sync completes before stop timing

### 4. BatchSolverKernel.run() Method Timing

#### File: src/cubie/batchsolving/BatchSolverKernel.py

**Instrumentation Points:**
- Outer timing: Start at method entry, stop before method exit
- Per-chunk kernel timing: Start before kernel launch, stop after stream sync
- Memory transfer timing: Delegated to array managers

**Behavior:**
- `kernel_run_execution` event wraps entire run() method
- `kernel_launch` event repeated for each chunk
- CUDA events used for GPU kernel timing
- Wall-clock timing used for outer timing

**CUDA Event Usage:**
```python
# In CUDA mode (not CUDASIM)
start_event = cuda.event()
stop_event = cuda.event()

start_event.record(stream)
# kernel launch
stop_event.record(stream)

self.memory_manager.sync_stream(self)  # or stream.synchronize()
elapsed_ms = cuda.event_elapsed_time(start_event, stop_event)

_default_timelogger.stop_event(
    'kernel_launch',
    gpu_time_ms=elapsed_ms,
    chunk_index=i
)
```

**CUDASIM Fallback:**
```python
# In CUDASIM mode
_default_timelogger.start_event('kernel_launch')
# kernel launch
_default_timelogger.stop_event('kernel_launch', chunk_index=i)
```

**Chunking Considerations:**
- Kernel launch timing occurs once per chunk
- Transfer timing occurs once per chunk
- Outer timing encompasses all chunks

### 5. InputArrays Transfer Timing

#### File: src/cubie/batchsolving/arrays/BatchInputArrays.py

**Instrumentation Points:**
- In `initialise(indices)` method
- Time each array transfer separately

**Arrays to Time:**
- `initial_values` - Always transferred
- `parameters` - Always transferred
- `driver_coefficients` - Conditionally transferred (if not None)

**Behavior:**
- Start event before `copy_to_device()` or equivalent transfer
- Stop event after transfer completes
- Include array size in metadata for analysis
- Respect chunking - only time the sliced transfer

**Implementation Pattern:**
```python
def initialise(self, indices):
    # Transfer initial_values
    _default_timelogger.start_event('h2d_transfer_initial_values')
    # ... perform transfer ...
    _default_timelogger.stop_event(
        'h2d_transfer_initial_values',
        nbytes=self.host.initial_values.array.nbytes,
        chunk_slice=str(indices)
    )
    
    # Repeat for parameters and driver_coefficients
```

### 6. OutputArrays Transfer Timing

#### File: src/cubie/batchsolving/arrays/BatchOutputArrays.py

**Instrumentation Points:**
- In `finalise(indices)` method
- Time each array transfer separately

**Arrays to Time:**
- `state` - If active
- `observables` - If active
- `state_summaries` - If active
- `observable_summaries` - If active
- `status_codes` - Always active
- `iteration_counters` - If active

**Behavior:**
- Check if array is active before timing
- Start event before device-to-host copy
- Stop event after transfer completes
- Include array size and active status in metadata

**Conditional Timing Pattern:**
```python
def finalise(self, indices):
    if self.active_outputs.state:
        _default_timelogger.start_event('d2h_transfer_state')
        # ... perform transfer ...
        _default_timelogger.stop_event(
            'd2h_transfer_state',
            nbytes=self.device.state.array.nbytes,
            chunk_slice=str(indices)
        )
    # Repeat for other arrays
```

### 7. BaseArrayManager Integration

#### File: src/cubie/batchsolving/arrays/BaseArrayManager.py

**Considerations:**
- `initialise()` and `finalise()` are called from BaseArrayManager
- Actual transfers may be implemented in subclasses or base class
- Need to identify exact transfer locations in the inheritance chain

**Investigation Required:**
- Examine `copy_to_device()` and `copy_from_device()` methods
- Determine if transfers are chunked or full-array
- Understand memory_manager integration for async transfers

## Data Structures

### Event Metadata

**Kernel Launch Event:**
```python
{
    'gpu_time_ms': float,      # From cuda.event_elapsed_time()
    'chunk_index': int,         # Which chunk (0 to N-1)
    'numruns': int,            # Number of runs in this chunk
}
```

**Memory Transfer Event:**
```python
{
    'nbytes': int,             # Size of transferred data
    'chunk_slice': str,        # String representation of indices
    'is_chunked': bool,        # Whether array participates in chunking
}
```

**Outer Timing Event:**
```python
{
    'num_chunks': int,         # Total chunks executed
    'total_runs': int,         # Total integration runs
}
```

## Edge Cases

### CUDASIM Mode
- `numba.cuda.event` not available in simulator
- Fallback to `time.perf_counter()` for kernel timing
- Check `is_cudasim_enabled()` before using CUDA events
- Test both code paths

### Verbosity=None Mode
- All start_event/stop_event calls become no-ops
- Zero overhead when timing disabled
- No need for conditional checks before timing calls

### Chunked Execution
- Kernel launch timing repeated per chunk
- Transfer timing repeated per chunk
- Aggregate timing across chunks reported in outer events
- TimeLogger automatically sums durations for repeated event names

### Empty Output Arrays
- Check `active_outputs` flags before timing transfers
- Skip timing for disabled output types
- Status codes always active, so always timed

### Stream Synchronization
- Must sync stream before reading CUDA event elapsed time
- Sync already present in code after kernel launch
- No additional syncs required for timing

### Driver Coefficients
- May be None for systems without drivers
- Only time transfer if coefficients exist
- Check for None before starting timing event

## Testing Strategy

### Unit Tests Required
- Event registration verification
- Timing data collection in each component
- CUDASIM fallback behavior
- Verbosity level handling (None, default, verbose, debug)
- Chunked execution timing
- Conditional array transfer timing

### Integration Tests Required
- Complete solve_ivp() timing chain
- Multi-chunk kernel execution timing
- TimeLogger category filtering
- Summary printing with category='runtime'

### Test Fixtures
- Small test systems with known execution patterns
- Mock CUDA events for deterministic testing (if possible)
- Chunked and non-chunked execution scenarios

### Expected Test Behavior
- All events registered before use (no KeyError)
- Timing events have matching start/stop pairs
- Metadata includes expected keys
- Summary output includes runtime category
- No timing when verbosity=None

## Integration Considerations

### TimeLogger Compatibility
- Use existing `_default_timelogger` instance
- No changes to TimeLogger class required
- Category='runtime' distinguishes from compile/codegen
- `print_summary(category='runtime')` filters correctly

### Existing Timing Infrastructure
- Compile events already registered by CUDAFactory
- Codegen events already registered by codegen modules
- Runtime events follow same registration pattern
- All three categories can be printed independently or together

### Memory Manager Integration
- Array managers already integrate with memory manager
- Transfer timing doesn't affect memory management
- Stream synchronization already handled correctly
- No additional coordination required

### Error Handling
- TimeLogger already handles missing events gracefully
- start_event returns early if event not registered
- stop_event returns early if no active start
- Timing errors don't affect solver correctness

## Dependencies Between Components

### Registration Order
1. Module-level registration for solve_ivp (import time)
2. Solver.__init__() registration (instance creation)
3. BatchSolverKernel.__init__() registration (kernel creation)

### Execution Order
1. solve_ivp() starts
2. Solver.solve() starts
3. BatchSolverKernel.run() starts
4. For each chunk:
   - InputArrays.initialise() transfers (timed individually)
   - Kernel launch (CUDA event timing)
   - OutputArrays.finalise() transfers (timed individually)
5. BatchSolverKernel.run() stops
6. Solver.solve() stops
7. solve_ivp() stops and prints summary

### Timing Hierarchy
```
solve_ivp_execution
└── solver_solve_execution
    └── kernel_run_execution
        ├── h2d_transfer_* (per chunk)
        ├── kernel_launch (per chunk)
        └── d2h_transfer_* (per chunk)
```

## File Modification Summary

**Files to Modify:**
1. `src/cubie/batchsolving/solver.py`
   - Add module-level event registration for solve_ivp
   - Add event registration in Solver.__init__()
   - Add timing in solve_ivp() function
   - Add timing in Solver.solve() method
   - Add print_summary() calls

2. `src/cubie/batchsolving/BatchSolverKernel.py`
   - Add event registration in __init__()
   - Add outer timing in run() method
   - Add CUDA event timing for kernel launches
   - Add CUDASIM fallback logic

3. `src/cubie/batchsolving/arrays/BatchInputArrays.py`
   - Add transfer timing in initialise() method
   - Import _default_timelogger

4. `src/cubie/batchsolving/arrays/BatchOutputArrays.py`
   - Add transfer timing in finalise() method
   - Import _default_timelogger

**Files to Review:**
- `src/cubie/batchsolving/arrays/BaseArrayManager.py` - Understand transfer methods

**No Changes Required:**
- `src/cubie/time_logger.py` - Already supports runtime category
- `src/cubie/CUDAFactory.py` - Compile timing already implemented

## Implementation Notes

### Import Statements
Add to files requiring timing:
```python
from cubie.time_logger import _default_timelogger
from cubie.cuda_simsafe import is_cudasim_enabled
from numba import cuda  # For CUDA event timing
```

### CUDA Event Handling
```python
if not is_cudasim_enabled():
    # Use CUDA events
    start_evt = cuda.event()
    stop_evt = cuda.event()
    start_evt.record(stream)
    # ... kernel launch ...
    stop_evt.record(stream)
    stream.synchronize()  # or self.memory_manager.sync_stream(self)
    elapsed_ms = cuda.event_elapsed_time(start_evt, stop_evt)
    _default_timelogger.stop_event('kernel_launch', gpu_time_ms=elapsed_ms)
else:
    # Fallback to wall-clock timing in CUDASIM
    _default_timelogger.start_event('kernel_launch')
    # ... kernel launch ...
    _default_timelogger.stop_event('kernel_launch')
```

### Summary Printing Strategy
Print runtime summary in the outermost function (solve_ivp or Solver.solve if called directly):
- Check if verbosity is 'default'
- Call `_default_timelogger.print_summary(category='runtime')`
- Print after all runtime events complete
- One summary per user-facing function call

### Backward Compatibility
- No breaking changes to public API
- Timing is opt-in via verbosity parameter
- Default behavior (verbosity='default') prints summary automatically
- Existing code continues to work without modification
