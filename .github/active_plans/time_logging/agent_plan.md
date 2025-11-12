# Time Logging Infrastructure - Agent Implementation Plan

## Overview for Agents

This document provides detailed architectural specifications for the `time_logger` module and its integration with the CuBIE codebase. The `detailed_implementer` agent will use this to create function-level tasks, and the `reviewer` agent will validate implementation against user stories.

**Phase 1 Scope:** Create timing infrastructure and thread it through CUDAFactory hierarchy. Compilation and codegen timing will be implemented in future phases.

## Component Architecture

### Time Logger Module Structure

**Location:** `src/cubie/time_logger.py`

**Purpose:** Provide callback-based timing system with configurable verbosity

**Components:**

1. **TimeLogger Class**
   - Stores timing events with start/stop/progress records
   - Manages verbosity level (default/verbose/debug)
   - Provides callback functions for use by CUDAFactory subclasses
   - Formats and prints timing summaries
   - Uses `time.perf_counter()` for all time measurements

2. **Event Storage**
   - Each event has: name, event_type, timestamp, optional metadata
   - Event types: 'start', 'stop', 'progress'
   - Stored in chronological order
   - Duration calculated by matching start/stop events

3. **Callback Functions**
   - `start_event(event_name: str, **metadata) -> None`
   - `stop_event(event_name: str, **metadata) -> None`
   - `progress(event_name: str, message: str, **metadata) -> None`
   - Each callback records timestamp and calls appropriate logger method

4. **Output Formatting**
   - `print_summary()` - Prints timing based on verbosity
   - `get_event_duration(event_name: str) -> float` - Query individual durations
   - `get_aggregate_durations(category: str) -> dict` - Aggregate by category

**Data Structure:**
```python
@attrs.define
class TimingEvent:
    name: str
    event_type: str  # 'start', 'stop', 'progress'
    timestamp: float  # from time.perf_counter()
    metadata: dict = attrs.field(factory=dict)
```

**Expected Behavior:**
- When verbosity='default': Store all events, print only aggregates at end
- When verbosity='verbose': Print each component duration as it completes
- When verbosity='debug': Print every event (start/stop/progress) as it occurs
- When `time_logger=None` passed to factories: Use no-op callbacks (no overhead)

### Integration with CUDAFactory Hierarchy

**Pattern for All CUDAFactory Subclasses:**

Each `__init__` method should:
1. Accept optional `time_logger` parameter (default `None`)
2. Extract callbacks from time_logger if provided, else use no-ops
3. Store callbacks as instance attributes (or call directly)
4. Optionally call timing callbacks at appropriate points

**No-Op Callbacks:**
When `time_logger=None`, provide no-op callback functions:
```python
def noop_callback(*args, **kwargs):
    pass
```

This ensures calling code doesn't need conditional logic.

**CUDAFactory Subclasses to Modify (Phase 1):**

1. **BaseODE** (`src/cubie/odesystems/baseODE.py`)
   - Add `time_logger` parameter to `__init__`
   - Pass through to parent CUDAFactory if needed
   - Store callbacks for use in subclasses
   - No timing calls in Phase 1 (infrastructure only)

2. **SymbolicODE** (`src/cubie/odesystems/symbolic/symbolicODE.py`)
   - Add `time_logger` parameter to `__init__`
   - Pass to BaseODE parent `__init__`
   - No timing calls in Phase 1 (codegen timing is Phase 3)

3. **IVPLoop** (`src/cubie/integrators/loops/ode_loop.py`)
   - Add `time_logger` parameter to `__init__`
   - Store callbacks
   - No timing calls in Phase 1

4. **OutputFunctions** (`src/cubie/outputhandling/output_functions.py`)
   - Add `time_logger` parameter to `__init__`
   - Store callbacks
   - No timing calls in Phase 1

5. **BaseStepController** (`src/cubie/integrators/step_control/base_step_controller.py`)
   - Add `time_logger` parameter to `__init__`
   - Store callbacks
   - Pass to subclass implementations

6. **BaseAlgorithmStep** (if exists as abstract base)
   - Add `time_logger` parameter to concrete algorithm `__init__` methods
   - Store callbacks
   - No timing calls in Phase 1

7. **Metric Base Classes** (`src/cubie/outputhandling/summarymetrics/metrics.py`)
   - Add `time_logger` parameter to metric factory classes
   - Store callbacks
   - No timing calls in Phase 1

8. **BatchSolverKernel** (`src/cubie/batchsolving/BatchSolverKernel.py`)
   - Add `time_logger` parameter to `__init__`
   - Pass to SingleIntegratorRun
   - No timing calls in Phase 1

9. **SingleIntegratorRunCore** (`src/cubie/integrators/SingleIntegratorRunCore.py`)
   - Add `time_logger` parameter to `__init__`
   - Pass to all factories it creates (loop, algorithms, controllers)
   - No timing calls in Phase 1

10. **ArrayInterpolator** (`src/cubie/integrators/array_interpolator.py`)
    - Add `time_logger` parameter to `__init__` if it's a CUDAFactory
    - Store callbacks
    - No timing calls in Phase 1

**Instantiation Sites to Update:**

All locations that create these factories need to pass `time_logger`:

1. **Solver** (`src/cubie/batchsolving/solver.py`)
   - Create TimeLogger instance based on user-provided verbosity parameter
   - Add `verbosity` parameter to Solver `__init__` (default='default')
   - Pass time_logger to BatchSolverKernel

2. **solve_ivp** function
   - Accept `verbosity` parameter
   - Create TimeLogger and pass through to Solver

3. **BatchSolverKernel**
   - Accept time_logger from Solver
   - Pass to SingleIntegratorRun and system interface

4. **SingleIntegratorRun/SingleIntegratorRunCore**
   - Accept time_logger
   - Pass to all created factories (IVPLoop, algorithms, controllers, OutputFunctions)

5. **Test Fixtures** (`tests/conftest.py`)
   - Add `time_logger=None` to all factory instantiations
   - Ensures existing tests continue to work
   - Add new fixture for testing with TimeLogger

### Callback Usage Pattern

**Recommended pattern for Phase 1:**

```python
class SomeFactory(CUDAFactory):
    def __init__(self, ..., time_logger=None):
        super().__init__()
        
        # Extract callbacks or use no-ops
        if time_logger is not None:
            self._timing_start = time_logger.start_event
            self._timing_stop = time_logger.stop_event
            self._timing_progress = time_logger.progress
        else:
            self._timing_start = lambda *args, **kwargs: None
            self._timing_stop = lambda *args, **kwargs: None
            self._timing_progress = lambda *args, **kwargs: None
        
        # Phase 1: No actual calls yet
        # Future: self._timing_start('component_init')
        # ... initialization code ...
        # Future: self._timing_stop('component_init')
```

## Expected Interactions Between Components

### TimeLogger → Callbacks → CUDAFactory
1. User creates Solver with `verbosity='verbose'`
2. Solver creates TimeLogger instance with verbose mode
3. TimeLogger provides callbacks via attributes/methods
4. Solver passes TimeLogger to system and BatchSolverKernel
5. Each factory stores callbacks during `__init__`
6. (Future phases) Factories call callbacks at appropriate times
7. (Future phases) TimeLogger formats and prints based on verbosity

### Event Recording Flow
1. Factory calls `start_event('operation_name')`
2. TimeLogger records TimingEvent with current timestamp
3. Factory performs operation
4. Factory calls `stop_event('operation_name')`
5. TimeLogger finds matching start event, calculates duration
6. Based on verbosity, TimeLogger may print immediately or defer

### Summary Generation Flow
1. User operation completes (e.g., Solver.solve() finishes)
2. TimeLogger.print_summary() called (explicitly or in cleanup)
3. TimeLogger aggregates events by category
4. Format output based on verbosity level
5. Print to console (or return data structure for programmatic access)

## Data Structures and Their Purposes

### TimingEvent
- Represents single timing event (start/stop/progress)
- Immutable once created (use attrs frozen=True)
- Timestamp uses `time.perf_counter()` for wall-clock measurement
- Metadata allows extensibility (e.g., file names, sizes, counts)

### TimeLogger State
```python
class TimeLogger:
    verbosity: str  # 'default', 'verbose', 'debug'
    events: list[TimingEvent]  # Chronological event list
    categories: dict[str, list[str]]  # Maps categories to event names
```

### Callback Interface
```python
# Type hints for callbacks
CallbackFunc = Callable[[str, ...], None]

# TimeLogger provides these
start_event: CallbackFunc
stop_event: CallbackFunc
progress: CallbackFunc
```

## Dependencies and Imports

### time_logger.py Dependencies
```python
import time  # For time.perf_counter()
import attrs  # For TimingEvent data class
from typing import Optional, Callable, Dict, List, Any
```

### CUDAFactory Modifications
- No new imports needed
- `time_logger` parameter is `Optional[TimeLogger]` type
- Can use `Optional[Any]` to avoid circular imports if needed

### Solver Modifications
```python
from cubie.time_logger import TimeLogger  # New import
```

## Edge Cases to Consider

### 1. Mismatched Start/Stop Events
- What if `stop_event('X')` called without `start_event('X')`?
- **Behavior:** Log warning in debug mode, skip duration calculation
- Store orphaned stop event for diagnostics

### 2. Nested Events
- What if same event name used in nested calls?
- **Behavior:** Match most recent unmatched start event
- Consider using stack-based matching or unique IDs per event

### 3. Multiple Stop Calls
- What if `stop_event('X')` called twice?
- **Behavior:** Second call is no-op (warn in debug mode)

### 4. Long-Running Operations
- Progress callbacks for operations taking >10 seconds
- **Behavior:** Allow multiple progress calls per event
- Don't require matching start/stop for progress events

### 5. Thread Safety
- CuBIE is largely single-threaded Python code
- **Phase 1 Decision:** No thread safety required
- Future: Add threading.Lock if needed

### 6. Time Logger is None
- Most common case during development and testing
- **Behavior:** No-op callbacks have zero overhead
- No conditional checks in hot paths

### 7. Very Large Event Counts
- Pathological case: thousands of events
- **Mitigation:** Events stored in list (O(n) append)
- Future: Consider deque or chunked output for huge runs

### 8. Precision and Resolution
- `time.perf_counter()` has ~100ns resolution on modern systems
- Sub-millisecond operations may show zero duration
- **Behavior:** Accept and display actual measured values

## Integration Points with Current Codebase

### Entry Points (Where TimeLogger is Created)
1. **Solver.__init__**
   - Add `verbosity='default'` parameter
   - Create `self.time_logger = TimeLogger(verbosity=verbosity)`
   - Pass to all downstream components

2. **solve_ivp function**
   - Add `verbosity='default'` parameter
   - Pass through to Solver initialization

### Threading Through Hierarchy
```
Solver
  ├─> TimeLogger (created here)
  └─> BatchSolverKernel(time_logger=...)
        ├─> System (BaseODE/SymbolicODE)(time_logger=...)
        └─> SingleIntegratorRun(time_logger=...)
              ├─> IVPLoop(time_logger=...)
              ├─> OutputFunctions(time_logger=...)
              ├─> AlgorithmStep(time_logger=...)
              └─> StepController(time_logger=...)
```

### Test Infrastructure Updates
1. **tests/conftest.py**
   - Global fixtures need `time_logger=None` parameter
   - Add `time_logger_fixture` that returns test TimeLogger
   - Add parametrized fixture for verbosity levels

2. **Test Modifications**
   - Existing tests: no changes needed (None is default)
   - New tests: can inject TimeLogger to verify callbacks
   - Integration tests: can verify timing output format

### No Changes Required (Phase 1)
- CUDA device functions (timing is host-side only)
- Memory management (timing happens at Python level)
- Array operations (no timing in Phase 1)
- Symbolic codegen internals (Phase 3 scope)

## Architectural Constraints

### 1. Optional Parameter Pattern
All `time_logger` parameters must be:
- Optional (default `None`)
- Type hinted as `Optional[TimeLogger]` or `Optional[Any]`
- Handled gracefully when `None`

### 2. No Global State
- No module-level TimeLogger instance
- Each Solver creates its own instance
- Multiple solvers in same process must not interfere

### 3. Zero Overhead When Disabled
- `time_logger=None` should have no performance impact
- No string formatting until print time
- No time.perf_counter() calls unless timing enabled

### 4. No Breaking Changes
- All existing code must work without modification
- Adding optional parameter is non-breaking
- Tests default to `time_logger=None`

### 5. Python Standard Library Only
- No new external dependencies
- Use `time`, `attrs`, `typing` (already in project)
- Simple data structures (list, dict)

## Expected Behavior Specifications

### Default Verbosity
```python
solver = Solver(system, algorithm='RK45', verbosity='default')
result = solver.solve(...)

# Output at end:
# Codegen time: 2.35s
# Compile time: 0.87s
# Runtime: 0.12s
```

### Verbose Mode
```python
solver = Solver(system, algorithm='RK45', verbosity='verbose')
result = solver.solve(...)

# Output during execution:
# SymbolicODE initialization: 0.45s
# dxdt codegen: 1.20s
# Jacobian codegen: 0.50s
# Linear operator codegen: 0.20s
# Algorithm compilation: 0.35s
# Loop compilation: 0.28s
# Output functions compilation: 0.24s
# Integration runtime: 0.12s
```

### Debug Mode
```python
solver = Solver(system, algorithm='RK45', verbosity='debug')
result = solver.solve(...)

# Output during execution:
# [DEBUG] Started: SymbolicODE_init
# [DEBUG] Progress: Parsing equations (50/100 complete)
# [DEBUG] Progress: Parsing equations (100/100 complete)
# [DEBUG] Stopped: SymbolicODE_init (0.45s)
# [DEBUG] Started: dxdt_codegen
# [DEBUG] Progress: CSE reduction
# ... many more events ...
```

### No TimeLogger
```python
# Default behavior, no timing overhead
solver = Solver(system, algorithm='RK45')  # time_logger=None internally
result = solver.solve(...)
# No timing output
```

## Files to Create

### New Files
1. `src/cubie/time_logger.py`
   - TimeLogger class
   - TimingEvent data class
   - Helper functions for formatting

2. `tests/test_time_logger.py`
   - Unit tests for TimeLogger
   - Test all verbosity levels
   - Test edge cases (mismatched events, etc.)
   - Test no-op behavior when disabled

### Files to Modify (Phase 1)

**Source Files:**
1. `src/cubie/__init__.py` - Export TimeLogger if needed
2. `src/cubie/batchsolving/solver.py` - Create and pass TimeLogger
3. `src/cubie/batchsolving/BatchSolverKernel.py` - Accept and pass through
4. `src/cubie/integrators/SingleIntegratorRun.py` - Accept and pass through
5. `src/cubie/integrators/SingleIntegratorRunCore.py` - Accept and distribute
6. `src/cubie/odesystems/baseODE.py` - Accept in __init__
7. `src/cubie/odesystems/symbolic/symbolicODE.py` - Accept in __init__
8. `src/cubie/integrators/loops/ode_loop.py` - Accept in __init__
9. `src/cubie/outputhandling/output_functions.py` - Accept in __init__
10. `src/cubie/integrators/algorithms/` - Multiple files, accept in __init__
11. `src/cubie/integrators/step_control/` - Multiple files, accept in __init__
12. `src/cubie/outputhandling/summarymetrics/metrics.py` - Accept in metric classes
13. `src/cubie/integrators/array_interpolator.py` - Accept if CUDAFactory

**Test Files:**
1. `tests/conftest.py` - Add time_logger fixtures
2. Any existing tests that directly instantiate factories - May need `time_logger=None`

## Implementation Notes for Detailed Implementer

### Task Granularity
Break down into tasks like:
- Create TimingEvent data class
- Create TimeLogger class with verbosity handling
- Implement start_event/stop_event/progress methods
- Implement event matching and duration calculation
- Implement print_summary with formatting for each verbosity level
- Add time_logger parameter to BaseODE.__init__
- Add time_logger parameter to SymbolicODE.__init__
- ... (one task per factory modification)
- Update Solver to create TimeLogger
- Update solve_ivp to accept verbosity
- Update test fixtures

### Testing Strategy
- Unit tests for TimeLogger in isolation
- Integration tests that verify callbacks flow through hierarchy
- Test each verbosity level produces expected output
- Test no-op path has no overhead (timing comparison)
- Test edge cases (mismatched events, None handling)

### Code Review Checkpoints
- All CUDAFactory subclasses have optional time_logger parameter
- No breaking changes to existing code
- Tests pass with time_logger=None (default)
- TimeLogger correctly aggregates events
- Output formatting matches specifications
- No performance regression when timing disabled

## Future Phase Considerations

### Phase 2: Compilation Tracking (Not Implemented Now)
- Add `specialise_and_compile` method to CUDAFactory
- Takes CUDADispatcher (device function), extracts signature
- Defines dummy kernel that calls device function
- Triggers Numba JIT compilation with timing
- Called immediately after defining device function in build()
- Stores compiled function in cache alongside original

### Phase 3: Codegen Timing (Not Implemented Now)
- Add timing callbacks in SymbolicODE initialization
- Time each codegen stage (dxdt, jacobian, operators, etc.)
- Add progress callbacks in build_jvp machinery outer loop
- Time file I/O operations (write, import)
- Add timing to get_solver_helper and build_implicit_helpers

### Dependencies Between Phases
- Phase 1 must complete first (establishes infrastructure)
- Phase 2 can be done independently after Phase 1
- Phase 3 depends on Phase 1, independent of Phase 2
- All phases can be separate PRs for easier review
