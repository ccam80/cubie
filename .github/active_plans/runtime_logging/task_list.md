# Implementation Task List
# Feature: Runtime Logging for CuBIE
# Plan Reference: .github/active_plans/runtime_logging/agent_plan.md

## Task Group 1: CUDAEvent Class Foundation - SEQUENTIAL
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/cuda_simsafe.py (entire file)
- File: .github/active_plans/runtime_logging/agent_plan.md (Component 1)

**Input Validation Required**:
- name: Check non-empty string
- category: Check in {'codegen', 'runtime', 'compile'}
- stream: Accept any stream object (no validation)

**Tasks**:
1. **Add CUDAEvent class to cuda_simsafe.py**
   - File: src/cubie/cuda_simsafe.py
   - Action: Create
   - Location: After line 250 (after `is_cudasim_enabled()` function, before `__all__`)
   - Details:
     ```python
     class CUDAEvent:
         """CUDA event pair for timing measurements with CUDASIM fallback.
         
         Parameters
         ----------
         name : str
             Event identifier (e.g., "kernel_chunk_0")
         category : str, default='runtime'
             TimeLogger category: 'runtime', 'codegen', or 'compile'
         
         Attributes
         ----------
         name : str
             Event identifier
         category : str
             TimeLogger category
         _start_event : cuda.event or None
             Start event object (CUDA mode)
         _end_event : cuda.event or None
             End event object (CUDA mode)
         _start_time : float or None
             Start timestamp (CUDASIM mode)
         _end_time : float or None
             End timestamp (CUDASIM mode)
         """
         
         def __init__(self, name: str, category: str = 'runtime') -> None:
             # Validation
             if not name or not isinstance(name, str):
                 raise ValueError("name must be a non-empty string")
             if category not in {'codegen', 'runtime', 'compile'}:
                 raise ValueError(
                     f"category must be 'codegen', 'runtime', or 'compile', "
                     f"got '{category}'"
                 )
             
             self.name = name
             self.category = category
             
             if not CUDA_SIMULATION:
                 # CUDA mode: create event objects
                 self._start_event = cuda.event()
                 self._end_event = cuda.event()
                 self._start_time = None
                 self._end_time = None
             else:
                 # CUDASIM mode: use timestamps
                 self._start_event = None
                 self._end_event = None
                 self._start_time = None
                 self._end_time = None
         
         def record_start(self, stream) -> None:
             """Record start timestamp on given stream.
             
             Parameters
             ----------
             stream
                 CUDA stream on which to record event
             
             Notes
             -----
             Must be called before record_end().
             """
             if not CUDA_SIMULATION:
                 self._start_event.record(stream)
             else:
                 import time
                 self._start_time = time.perf_counter()
         
         def record_end(self, stream) -> None:
             """Record end timestamp on given stream.
             
             Parameters
             ----------
             stream
                 CUDA stream on which to record event
             
             Notes
             -----
             Must be called after record_start().
             """
             if not CUDA_SIMULATION:
                 self._end_event.record(stream)
             else:
                 import time
                 self._end_time = time.perf_counter()
         
         def elapsed_time_ms(self) -> float:
             """Calculate elapsed time in milliseconds.
             
             Returns
             -------
             float
                 Elapsed time in milliseconds
             
             Notes
             -----
             This method must NOT block or synchronize. It should be called
             AFTER the stream has been synchronized externally.
             In CUDA mode, uses cuda.event_elapsed_time() which returns
             immediately post-sync.
             In CUDASIM mode, calculates from stored timestamps.
             
             Returns 0.0 if both start and end have not been recorded.
             """
             if not CUDA_SIMULATION:
                 if self._start_event is None or self._end_event is None:
                     return 0.0
                 return cuda.event_elapsed_time(self._start_event, self._end_event)
             else:
                 if self._start_time is None or self._end_time is None:
                     return 0.0
                 return (self._end_time - self._start_time) * 1000.0
     ```
   - Edge cases:
     - Handle record_end() without record_start(): returns 0.0 in elapsed_time_ms()
     - Handle elapsed_time_ms() before both recordings: returns 0.0
     - CUDASIM fallback produces reasonable timing estimates
   - Integration: CUDAEvent will be imported by BatchSolverKernel and registered with TimeLogger

2. **Update __all__ export list in cuda_simsafe.py**
   - File: src/cubie/cuda_simsafe.py
   - Action: Modify
   - Location: Line 252-280 (existing `__all__` list)
   - Details:
     Add "CUDAEvent" to the `__all__` list in alphabetical order:
     ```python
     __all__ = [
         "CUDA_SIMULATION",
         "activemask",
         "all_sync",
         "BaseCUDAMemoryManager",
         "compile_kwargs",
         "CUDAEvent",  # ADD THIS LINE
         "DeviceNDArrayBase",
         # ... rest of exports
     ]
     ```
   - Edge cases: None
   - Integration: Enables `from cubie.cuda_simsafe import CUDAEvent`

**Outcomes**: 
- Files Modified:
  * src/cubie/cuda_simsafe.py (134 lines changed - added CUDAEvent class and updated __all__)
- Functions/Methods Added/Modified:
  * CUDAEvent class with __init__, record_start, record_end, elapsed_time_ms methods
  * Updated __all__ export list to include "CUDAEvent"
- Implementation Summary:
  Added complete CUDAEvent class with CUDA/CUDASIM mode branching. In CUDA mode, 
  creates cuda.event() objects for GPU timeline recording. In CUDASIM mode, uses 
  time.perf_counter() for wall-clock fallback. Validation ensures name is non-empty 
  string and category is in allowed set. elapsed_time_ms() returns 0.0 if events 
  not fully recorded.
- Issues Flagged: None

---

## Task Group 2: TimeLogger Extensions - SEQUENTIAL
**Status**: [x]
**Dependencies**: Group 1 (requires CUDAEvent class)

**Required Context**:
- File: src/cubie/time_logger.py (entire file)
- File: src/cubie/cuda_simsafe.py (CUDAEvent class)
- File: .github/active_plans/runtime_logging/agent_plan.md (Component 2)

**Input Validation Required**:
- event parameter in register_cuda_event: Check is instance of CUDAEvent
- No additional validation in retrieve_cuda_events (processes registered events)

**Tasks**:
1. **Add _cuda_events attribute to TimeLogger class**
   - File: src/cubie/time_logger.py
   - Action: Modify
   - Location: Line 62-74 (`__init__` method)
   - Details:
     After line 74 (`self._event_registry: dict[str, dict] = {}`), add:
     ```python
     self._cuda_events: list = []  # Stores CUDAEvent instances
     ```
   - Edge cases: None
   - Integration: Stores CUDAEvent instances for later timing retrieval

2. **Add import for CUDAEvent type**
   - File: src/cubie/time_logger.py
   - Action: Modify
   - Location: Line 1-5 (import section)
   - Details:
     Add to imports:
     ```python
     from typing import Optional, Any, TYPE_CHECKING
     
     if TYPE_CHECKING:
         from cubie.cuda_simsafe import CUDAEvent
     ```
   - Edge cases: None
   - Integration: Enables type hints without circular import issues

3. **Add register_cuda_event() method**
   - File: src/cubie/time_logger.py
   - Action: Create
   - Location: After line 393 (after `register_event()` method)
   - Details:
     ```python
     def register_cuda_event(self, event: "CUDAEvent") -> None:
         """Register a CUDA event for later timing retrieval.
         
         Parameters
         ----------
         event : CUDAEvent
             CUDA event instance to register
         
         Notes
         -----
         Events are stored until retrieve_cuda_events() is called.
         No-op when verbosity is None.
         The event's category and name are also registered with the
         standard register_event() mechanism.
         """
         if self.verbosity is None:
             return
         
         # Import here to avoid circular dependency
         from cubie.cuda_simsafe import CUDAEvent
         
         if not isinstance(event, CUDAEvent):
             raise TypeError(
                 f"event must be a CUDAEvent instance, got {type(event)}"
             )
         
         # Store event for later retrieval
         self._cuda_events.append(event)
         
         # Register with standard event registry
         self.register_event(
             event.name, 
             event.category, 
             f"GPU event: {event.name}"
         )
     ```
   - Edge cases:
     - Validate event is CUDAEvent instance
     - No-op when verbosity is None
   - Integration: Called by BatchSolverKernel.run() after event recording

4. **Add retrieve_cuda_events() method**
   - File: src/cubie/time_logger.py
   - Action: Create
   - Location: After register_cuda_event() method
   - Details:
     ```python
     def retrieve_cuda_events(self) -> None:
         """Retrieve timing from all registered CUDA events.
         
         Notes
         -----
         This method must be called AFTER stream synchronization.
         It converts CUDA event timings to TimeLogger events by calling
         elapsed_time_ms() on each registered CUDAEvent.
         
         The _cuda_events list is cleared after retrieval to avoid
         memory growth.
         
         No-op when verbosity is None or no events registered.
         """
         if self.verbosity is None:
             return
         
         if not self._cuda_events:
             return
         
         # Process each CUDA event
         for event in self._cuda_events:
             # Get elapsed time (returns immediately, no blocking)
             elapsed_ms = event.elapsed_time_ms()
             
             # Create a TimingEvent with the duration in metadata
             timing_event = TimingEvent(
                 name=event.name,
                 event_type='stop',
                 timestamp=0.0,  # Not used for CUDA events
                 metadata={'duration_ms': elapsed_ms}
             )
             self.events.append(timing_event)
         
         # Clear list after retrieval
         self._cuda_events.clear()
     ```
   - Edge cases:
     - Handle empty _cuda_events list gracefully
     - No-op when verbosity is None
   - Integration: Called by Solver.solve() after sync_stream()

5. **Modify print_summary() for per-chunk reporting**
   - File: src/cubie/time_logger.py
   - Action: Modify
   - Location: Line 313-339 (print_summary() method)
   - Details:
     Replace the print_summary() method with enhanced version:
     ```python
     def print_summary(self, category: Optional[str] = None) -> None:
         """Print timing summary for all events or specific category.
         
         Parameters
         ----------
         category : str, optional
             If provided, print summary only for events in this category
             ('codegen', 'runtime', or 'compile'). If None, print all events.
         
         Notes
         -----
         In 'default' verbosity mode, this method can be called with specific
         categories to print summaries at different stages:
         - Call with category='codegen' after parsing is complete
         - Call with category='compile' after compilation is complete
         - Call with category='runtime' after kernels return
         
         For runtime category with CUDA events:
         - 'verbose' or 'debug': prints individual chunk timings
         - 'default': aggregates chunk timings by type
         """
         if self.verbosity is None:
             return
         
         if self.verbosity == 'default':
             # Get standard durations from start/stop pairs
             durations = self.get_aggregate_durations(category=category)
             
             # Add CUDA event durations from metadata
             for event in self.events:
                 if event.event_type == 'stop' and 'duration_ms' in event.metadata:
                     # Filter by category if requested
                     if category is not None:
                         event_info = self._event_registry.get(event.name)
                         if event_info is None or event_info['category'] != category:
                             continue
                     
                     # Convert ms to seconds for consistency
                     durations[event.name] = event.metadata['duration_ms'] / 1000.0
             
             # For runtime category, aggregate chunk events
             if category == 'runtime':
                 aggregated = {}
                 chunk_events = {}
                 
                 for name, duration in durations.items():
                     if '_chunk_' in name:
                         # Extract prefix (h2d_transfer, kernel, d2h_transfer)
                         prefix = name.rsplit('_chunk_', 1)[0]
                         if prefix not in chunk_events:
                             chunk_events[prefix] = []
                         chunk_events[prefix].append(duration)
                     else:
                         aggregated[name] = duration
                 
                 # Sum chunk events by type
                 for prefix, durations_list in chunk_events.items():
                     aggregated[f"{prefix}_total"] = sum(durations_list)
                 
                 durations = aggregated
             
             if durations:
                 if category:
                     print(f"\nTIMELOGGER {category.capitalize()} Timing Summary:")
                 else:
                     print("\nTIMELOGGER Timing Summary:")
                 for name, duration in sorted(durations.items()):
                     print(f"TIMELOGGER   {name}: {duration:.3f}s")
         
         elif self.verbosity in ('verbose', 'debug'):
             # Print individual chunk timings for verbose/debug
             if category == 'runtime':
                 # Print CUDA events with ms precision
                 cuda_events = [
                     e for e in self.events 
                     if e.event_type == 'stop' and 'duration_ms' in e.metadata
                 ]
                 
                 if cuda_events:
                     print(f"\nTIMELOGGER {category.capitalize()} Timing (per-chunk):")
                     for event in cuda_events:
                         event_info = self._event_registry.get(event.name)
                         if event_info and event_info['category'] == category:
                             duration_ms = event.metadata['duration_ms']
                             print(f"TIMELOGGER   {event.name}: {duration_ms:.3f}ms")
         # verbose and debug already printed inline for start/stop events
     ```
   - Edge cases:
     - Handle events with and without duration_ms metadata
     - Aggregate chunk events correctly by prefix
     - Filter by category when requested
   - Integration: Called by Solver.solve() after event retrieval

**Outcomes**: 
- Files Modified:
  * src/cubie/time_logger.py (152 lines changed - added imports, attributes, methods)
- Functions/Methods Added/Modified:
  * Added TYPE_CHECKING import and CUDAEvent type hint
  * Added _cuda_events attribute to TimeLogger.__init__
  * Added register_cuda_event() method
  * Added retrieve_cuda_events() method
  * Modified print_summary() method for per-chunk aggregation
- Implementation Summary:
  Extended TimeLogger with CUDA event handling. Added _cuda_events list to store
  CUDAEvent instances. register_cuda_event() validates type and stores events.
  retrieve_cuda_events() calls elapsed_time_ms() on all events after sync and
  converts to TimingEvent format. print_summary() now aggregates chunk events
  in default mode and prints individual chunk timings in verbose/debug mode.
- Issues Flagged: None

---

## Task Group 3: BatchSolverKernel Event Recording - SEQUENTIAL
**Status**: [x]
**Dependencies**: Groups 1, 2 (requires CUDAEvent class and TimeLogger extensions)

**Required Context**:
- File: src/cubie/batchsolving/BatchSolverKernel.py (lines 1-372)
- File: src/cubie/cuda_simsafe.py (CUDAEvent class)
- File: src/cubie/time_logger.py (_default_timelogger)
- File: .github/active_plans/runtime_logging/agent_plan.md (Component 3)

**Input Validation Required**:
- No additional validation (uses existing stream parameter validation)

**Tasks**:
1. **Add imports for CUDAEvent and _default_timelogger**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Location: Line 13-14 (after existing cuda_simsafe imports)
   - Details:
     Change line 13-14 from:
     ```python
     from cubie.cuda_simsafe import is_cudasim_enabled, \
         compile_kwargs
     ```
     To:
     ```python
     from cubie.cuda_simsafe import is_cudasim_enabled, \
         compile_kwargs, CUDAEvent
     from cubie.time_logger import _default_timelogger
     ```
   - Edge cases: None
   - Integration: Enables event creation and registration

2. **Add _cuda_events attributes to BatchSolverKernel class**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Location: Line 136-138 (in __init__ method after chunks initialization)
   - Details:
     After line 138 (`self.num_runs = 1`), add:
     ```python
     # CUDA event tracking for timing
     self._cuda_events: list = []
     self._gpu_workload_event: Optional[CUDAEvent] = None
     ```
   - Edge cases: None
   - Integration: Storage for events created during run()

3. **Add Optional import for type hints**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Location: Line 4 (TYPE_CHECKING import line)
   - Details:
     Change from:
     ```python
     from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Union
     ```
     To:
     ```python
     from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
     ```
   - Edge cases: None
   - Integration: Enables List type hint for _cuda_events

4. **Instrument run() method with event creation**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Location: Line 299-332 (after memory allocation, before chunk loop)
   - Details:
     After line 332 (`if self.profileCUDA: cuda.profile_start()`), add:
     ```python
     # Create CUDA events for timing if verbosity is enabled
     if _default_timelogger.verbosity is not None:
         # Create overall GPU workload event
         self._gpu_workload_event = CUDAEvent("gpu_workload", category="runtime")
         
         # Create per-chunk events (3 events per chunk: h2d, kernel, d2h)
         self._cuda_events = []
         for i in range(self.chunks):
             h2d_event = CUDAEvent(f"h2d_transfer_chunk_{i}", category="runtime")
             kernel_event = CUDAEvent(f"kernel_chunk_{i}", category="runtime")
             d2h_event = CUDAEvent(f"d2h_transfer_chunk_{i}", category="runtime")
             self._cuda_events.extend([h2d_event, kernel_event, d2h_event])
         
         # Record start of overall GPU workload
         self._gpu_workload_event.record_start(stream)
     ```
   - Edge cases:
     - Only create events when verbosity is not None
     - Create exactly 3 events per chunk
   - Integration: Events stored in instance variables for recording

5. **Instrument run() method: record h2d_transfer events**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Location: Line 334-336 (chunk loop, before and after initialise calls)
   - Details:
     Before line 335 (`self.input_arrays.initialise(indices)`), add:
     ```python
             # Record start of h2d transfer
             if len(self._cuda_events) > 0:
                 h2d_event = self._cuda_events[i * 3]
                 h2d_event.record_start(stream)
     ```
     
     After line 336 (`self.output_arrays.initialise(indices)`), add:
     ```python
             # Record end of h2d transfer
             if len(self._cuda_events) > 0:
                 h2d_event = self._cuda_events[i * 3]
                 h2d_event.record_end(stream)
     ```
   - Edge cases:
     - Check _cuda_events is not empty
     - Index correctly: chunk i uses events at [i*3, i*3+1, i*3+2]
   - Integration: Records h2d timing around input array initialization

6. **Instrument run() method: record kernel execution events**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Location: Line 343-362 (before and after kernel launch)
   - Details:
     Before line 343 (`self.kernel[...](...)` kernel launch), add:
     ```python
             # Record start of kernel execution
             if len(self._cuda_events) > 0:
                 kernel_event = self._cuda_events[i * 3 + 1]
                 kernel_event.record_start(stream)
     ```
     
     After line 362 (kernel launch closing paren), add:
     ```python
             # Record end of kernel execution
             if len(self._cuda_events) > 0:
                 kernel_event = self._cuda_events[i * 3 + 1]
                 kernel_event.record_end(stream)
     ```
   - Edge cases:
     - Check _cuda_events is not empty
     - Index correctly: kernel event is at i*3+1
   - Integration: Records kernel timing around kernel launch

7. **Instrument run() method: record d2h_transfer events**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Location: Line 367-368 (before and after finalise calls)
   - Details:
     Before line 367 (`self.input_arrays.finalise(indices)`), add:
     ```python
             # Record start of d2h transfer
             if len(self._cuda_events) > 0:
                 d2h_event = self._cuda_events[i * 3 + 2]
                 d2h_event.record_start(stream)
     ```
     
     After line 368 (`self.output_arrays.finalise(indices)`), add:
     ```python
             # Record end of d2h transfer
             if len(self._cuda_events) > 0:
                 d2h_event = self._cuda_events[i * 3 + 2]
                 d2h_event.record_end(stream)
     ```
   - Edge cases:
     - Check _cuda_events is not empty
     - Index correctly: d2h event is at i*3+2
   - Integration: Records d2h timing around output array finalization

8. **Instrument run() method: finalize event recording and registration**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Location: Line 370-371 (after chunk loop, before profile_stop)
   - Details:
     Before line 370 (`if self.profileCUDA: cuda.profile_stop()`), add:
     ```python
         # Finalize GPU workload timing and register all events
         if self._gpu_workload_event is not None:
             self._gpu_workload_event.record_end(stream)
             
             # Register all events with TimeLogger
             _default_timelogger.register_cuda_event(self._gpu_workload_event)
             for event in self._cuda_events:
                 _default_timelogger.register_cuda_event(event)
     ```
   - Edge cases:
     - Only register if events were created (gpu_workload_event is not None)
   - Integration: Hands off all events to TimeLogger for later retrieval

**Outcomes**: 
- Files Modified:
  * src/cubie/batchsolving/BatchSolverKernel.py (82 lines changed)
- Functions/Methods Added/Modified:
  * Updated imports: Added List to typing, CUDAEvent, _default_timelogger
  * Added _cuda_events and _gpu_workload_event attributes to __init__
  * Instrumented run() method with event creation and recording
- Implementation Summary:
  Added comprehensive CUDA event instrumentation to BatchSolverKernel.run().
  Before chunk loop, creates gpu_workload event and per-chunk events (3 per chunk).
  Within chunk loop, records start/end for h2d_transfer, kernel execution, and
  d2h_transfer. After loop, records end of gpu_workload and registers all events
  with TimeLogger. All recording guards check verbosity and _cuda_events length
  to ensure zero overhead when disabled.
- Issues Flagged: None

---

## Task Group 4: Solver Wall-Clock Timing and Event Retrieval - SEQUENTIAL
**Status**: [x]
**Dependencies**: Groups 1, 2, 3 (requires complete event recording pipeline)

**Required Context**:
- File: src/cubie/batchsolving/solver.py (lines 41-113, 412-523)
- File: src/cubie/time_logger.py (_default_timelogger)
- File: .github/active_plans/runtime_logging/agent_plan.md (Components 4, 5)

**Input Validation Required**:
- No additional validation (uses existing method parameters)

**Tasks**:
1. **Add wall-clock timing to solve_ivp() function**
   - File: src/cubie/batchsolving/solver.py
   - Action: Modify
   - Location: Lines 93-114 (solve_ivp function body)
   - Details:
     After line 102 (solver instantiation), before line 104 (solver.solve() call), add:
     ```python
     # Register and start wall-clock timing
     _default_timelogger.register_event(
         "solve_ivp", 
         "runtime", 
         "Wall-clock time for solve_ivp()"
     )
     _default_timelogger.start_event("solve_ivp")
     ```
     
     Before line 114 (return statement), add:
     ```python
     # Stop wall-clock timing (summary printed by Solver.solve)
     _default_timelogger.stop_event("solve_ivp")
     ```
   - Edge cases:
     - No-op when verbosity is None
     - Summary not printed here (avoid duplication with Solver.solve)
   - Integration: Captures full solve_ivp() wall-clock duration

2. **Add wall-clock timing start to Solver.solve() method**
   - File: src/cubie/batchsolving/solver.py
   - Action: Modify
   - Location: Line 481-483 (start of solve() method)
   - Details:
     After line 482 (`if kwargs: self.update(kwargs, silent=True)`), add:
     ```python
         # Register and start wall-clock timing for solve
         _default_timelogger.register_event(
             "solver_solve",
             "runtime",
             "Wall-clock time for Solver.solve()"
         )
         _default_timelogger.start_event("solver_solve")
     ```
   - Edge cases:
     - No-op when verbosity is None
   - Integration: Starts timing at beginning of solve() method

3. **Add CUDA event retrieval and timing finalization to Solver.solve()**
   - File: src/cubie/batchsolving/solver.py
   - Action: Modify
   - Location: Line 522-523 (after sync_stream, before return)
   - Details:
     After line 522 (`self.memory_manager.sync_stream(self.kernel)`), add:
     ```python
         # Retrieve CUDA event timings (after sync completes)
         _default_timelogger.retrieve_cuda_events()
         
         # Stop wall-clock timing and print runtime summary
         _default_timelogger.stop_event("solver_solve")
         _default_timelogger.print_summary(category='runtime')
     ```
   - Edge cases:
     - retrieve_cuda_events() is no-op when no events registered
     - print_summary() is no-op when verbosity is None
   - Integration: Retrieves CUDA timings after sync, prints complete runtime summary

**Outcomes**: 
- Files Modified:
  * src/cubie/batchsolving/solver.py (35 lines changed)
- Functions/Methods Added/Modified:
  * Modified solve_ivp() function: Added wall-clock timing registration and start/stop
  * Modified Solver.solve() method: Added wall-clock timing start, CUDA event retrieval, and summary printing
- Implementation Summary:
  Added wall-clock timing to both solve_ivp() and Solver.solve(). solve_ivp() 
  registers and times the full function duration. Solver.solve() registers and 
  times its execution, then after sync_stream() calls retrieve_cuda_events() to
  convert GPU timings to TimeLogger events, stops the wall-clock timer, and prints
  the runtime summary. This completes the full timing pipeline from event creation
  to final reporting.
- Issues Flagged: None

---

## Task Group 5: Unit Tests for CUDAEvent - PARALLEL
**Status**: [ ]
**Dependencies**: Group 1 (requires CUDAEvent class)

**Required Context**:
- File: src/cubie/cuda_simsafe.py (CUDAEvent class)
- File: tests/test_cuda_simsafe.py (existing test patterns)

**Input Validation Required**:
- Test validates error handling for invalid inputs

**Tasks**:
1. **Create test_cuda_event.py test file**
   - File: tests/test_cuda_event.py
   - Action: Create
   - Details:
     ```python
     """Tests for CUDAEvent class."""
     
     import pytest
     from numba import cuda
     
     from cubie.cuda_simsafe import CUDAEvent, CUDA_SIMULATION
     
     
     class TestCUDAEventCreation:
         """Test CUDAEvent instantiation and validation."""
         
         def test_create_with_defaults(self):
             """Test CUDAEvent creation with default category."""
             event = CUDAEvent("test_event")
             assert event.name == "test_event"
             assert event.category == "runtime"
         
         def test_create_with_custom_category(self):
             """Test CUDAEvent creation with custom category."""
             event = CUDAEvent("compile_event", category="compile")
             assert event.name == "compile_event"
             assert event.category == "compile"
         
         def test_invalid_name_empty(self):
             """Test that empty name raises ValueError."""
             with pytest.raises(ValueError, match="non-empty string"):
                 CUDAEvent("")
         
         def test_invalid_name_type(self):
             """Test that non-string name raises ValueError."""
             with pytest.raises(ValueError, match="non-empty string"):
                 CUDAEvent(123)
         
         def test_invalid_category(self):
             """Test that invalid category raises ValueError."""
             with pytest.raises(ValueError, match="category must be"):
                 CUDAEvent("test", category="invalid")
     
     
     @pytest.mark.nocudasim
     class TestCUDAEventCUDAMode:
         """Test CUDAEvent in real CUDA mode."""
         
         def test_creates_cuda_events(self):
             """Test that CUDA event objects are created."""
             event = CUDAEvent("test")
             assert event._start_event is not None
             assert event._end_event is not None
             assert event._start_time is None
             assert event._end_time is None
         
         def test_record_on_stream(self):
             """Test recording events on a CUDA stream."""
             stream = cuda.stream()
             event = CUDAEvent("test_kernel")
             
             # Should not raise
             event.record_start(stream)
             event.record_end(stream)
         
         def test_elapsed_time_after_sync(self):
             """Test elapsed_time_ms returns value after sync."""
             stream = cuda.stream()
             event = CUDAEvent("test_timing")
             
             event.record_start(stream)
             # Simulate some work
             cuda.synchronize()
             event.record_end(stream)
             stream.synchronize()
             
             elapsed = event.elapsed_time_ms()
             assert isinstance(elapsed, float)
             assert elapsed >= 0.0
         
         def test_elapsed_time_before_recording_returns_zero(self):
             """Test elapsed_time_ms returns 0 before recording."""
             event = CUDAEvent("unrecorded")
             elapsed = event.elapsed_time_ms()
             assert elapsed == 0.0
     
     
     class TestCUDAEventSimMode:
         """Test CUDAEvent in CUDASIM mode."""
         
         def test_uses_timestamps(self):
             """Test that simulation mode uses timestamp fields."""
             if not CUDA_SIMULATION:
                 pytest.skip("Only runs in CUDASIM mode")
             
             event = CUDAEvent("test")
             assert event._start_event is None
             assert event._end_event is None
         
         def test_record_stores_timestamps(self):
             """Test that record methods store timestamps."""
             if not CUDA_SIMULATION:
                 pytest.skip("Only runs in CUDASIM mode")
             
             stream = None  # Stream not used in sim mode
             event = CUDAEvent("test_sim")
             
             event.record_start(stream)
             assert event._start_time is not None
             
             event.record_end(stream)
             assert event._end_time is not None
             assert event._end_time >= event._start_time
         
         def test_elapsed_time_calculation(self):
             """Test elapsed time calculation in sim mode."""
             if not CUDA_SIMULATION:
                 pytest.skip("Only runs in CUDASIM mode")
             
             import time
             stream = None
             event = CUDAEvent("test_elapsed")
             
             event.record_start(stream)
             time.sleep(0.01)  # Sleep 10ms
             event.record_end(stream)
             
             elapsed = event.elapsed_time_ms()
             assert isinstance(elapsed, float)
             assert elapsed >= 10.0  # At least 10ms
             assert elapsed < 100.0  # But not too much more
     ```
   - Edge cases:
     - Validation errors for invalid inputs
     - Returns 0.0 when elapsed_time_ms() called before recording
     - CUDASIM and CUDA mode differences
   - Integration: Provides unit test coverage for CUDAEvent class

**Outcomes**: 
[To be filled by taskmaster agent]

---

## Task Group 6: Unit Tests for TimeLogger Extensions - PARALLEL
**Status**: [ ]
**Dependencies**: Groups 1, 2 (requires CUDAEvent class and TimeLogger extensions)

**Required Context**:
- File: src/cubie/time_logger.py (register_cuda_event, retrieve_cuda_events methods)
- File: tests/test_time_logger.py (existing test patterns)

**Input Validation Required**:
- Test validates type checking in register_cuda_event

**Tasks**:
1. **Add tests for register_cuda_event() to test_time_logger.py**
   - File: tests/test_time_logger.py
   - Action: Modify
   - Location: End of file (new test class)
   - Details:
     Add new test class:
     ```python
     class TestTimeLoggerCUDAEvents:
         """Test TimeLogger CUDA event handling."""
         
         def test_register_cuda_event(self):
             """Test registering a CUDA event."""
             from cubie.cuda_simsafe import CUDAEvent
             
             logger = TimeLogger(verbosity='default')
             event = CUDAEvent("test_kernel", category="runtime")
             
             logger.register_cuda_event(event)
             
             assert len(logger._cuda_events) == 1
             assert logger._cuda_events[0] is event
             assert "test_kernel" in logger._event_registry
         
         def test_register_cuda_event_none_verbosity(self):
             """Test that registration is no-op with None verbosity."""
             from cubie.cuda_simsafe import CUDAEvent
             
             logger = TimeLogger(verbosity=None)
             event = CUDAEvent("test_kernel")
             
             logger.register_cuda_event(event)
             
             assert len(logger._cuda_events) == 0
         
         def test_register_cuda_event_invalid_type(self):
             """Test that non-CUDAEvent raises TypeError."""
             logger = TimeLogger(verbosity='default')
             
             with pytest.raises(TypeError, match="must be a CUDAEvent"):
                 logger.register_cuda_event("not an event")
         
         def test_retrieve_cuda_events_empty(self):
             """Test retrieve with no events is no-op."""
             logger = TimeLogger(verbosity='default')
             
             # Should not raise
             logger.retrieve_cuda_events()
             
             assert len(logger.events) == 0
         
         def test_retrieve_cuda_events_creates_timing_events(self):
             """Test that retrieve creates TimingEvent entries."""
             from cubie.cuda_simsafe import CUDAEvent
             
             logger = TimeLogger(verbosity='default')
             event = CUDAEvent("kernel_chunk_0", category="runtime")
             
             # Mock elapsed time by setting timestamps in sim mode
             if hasattr(event, '_start_time'):
                 event._start_time = 0.0
                 event._end_time = 0.015  # 15ms
             
             logger.register_cuda_event(event)
             logger.retrieve_cuda_events()
             
             # Should have created a timing event
             assert len(logger.events) == 1
             assert logger.events[0].name == "kernel_chunk_0"
             assert logger.events[0].event_type == "stop"
             assert 'duration_ms' in logger.events[0].metadata
         
         def test_retrieve_cuda_events_clears_list(self):
             """Test that retrieve clears the cuda_events list."""
             from cubie.cuda_simsafe import CUDAEvent
             
             logger = TimeLogger(verbosity='default')
             event = CUDAEvent("test", category="runtime")
             
             logger.register_cuda_event(event)
             assert len(logger._cuda_events) == 1
             
             logger.retrieve_cuda_events()
             assert len(logger._cuda_events) == 0
         
         def test_retrieve_cuda_events_none_verbosity(self):
             """Test that retrieve is no-op with None verbosity."""
             from cubie.cuda_simsafe import CUDAEvent
             
             logger = TimeLogger(verbosity=None)
             event = CUDAEvent("test")
             
             # Manually add to list (bypassing register which is also no-op)
             logger._cuda_events.append(event)
             
             logger.retrieve_cuda_events()
             
             # Should still be no-op
             assert len(logger.events) == 0
     ```
   - Edge cases:
     - No-op with None verbosity
     - Empty event list handled gracefully
     - List cleared after retrieval
   - Integration: Validates TimeLogger CUDA event handling

**Outcomes**: 
[To be filled by taskmaster agent]

---

## Task Group 7: Integration Tests for Runtime Logging - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Groups 1, 2, 3, 4 (requires complete implementation)

**Required Context**:
- File: tests/conftest.py (fixture patterns)
- File: tests/system_fixtures.py (system fixtures)
- File: tests/batchsolving/ (solver test patterns)

**Input Validation Required**:
- No additional validation (uses existing fixtures)

**Tasks**:
1. **Create test_runtime_logging.py integration test file**
   - File: tests/test_runtime_logging.py
   - Action: Create
   - Details:
     ```python
     """Integration tests for runtime logging feature."""
     
     import pytest
     import numpy as np
     
     from cubie import Solver, solve_ivp
     from cubie.time_logger import _default_timelogger
     from cubie.cuda_simsafe import CUDA_SIMULATION
     
     
     @pytest.fixture
     def simple_system(three_state_linear):
         """Provide a simple system for testing."""
         return three_state_linear
     
     
     class TestRuntimeLoggingIntegration:
         """Integration tests for complete runtime logging pipeline."""
         
         def test_solve_ivp_with_timing_enabled(self, simple_system):
             """Test solve_ivp records timing events."""
             # Set verbosity to default
             result = solve_ivp(
                 simple_system,
                 y0={"x1": [1.0], "x2": [0.0], "x3": [0.0]},
                 parameters={"p1": [1.0], "p2": [0.5], "p3": [0.1]},
                 duration=0.1,
                 dt_save=0.01,
                 method="euler",
                 time_logging_level='default'
             )
             
             # Check that events were recorded
             events = _default_timelogger.events
             event_names = [e.name for e in events]
             
             # Should have solve_ivp event
             assert "solve_ivp" in event_names
             
             # Should have solver_solve event
             assert "solver_solve" in event_names
         
         def test_solve_ivp_with_timing_disabled(self, simple_system):
             """Test solve_ivp with verbosity None has no overhead."""
             result = solve_ivp(
                 simple_system,
                 y0={"x1": [1.0], "x2": [0.0], "x3": [0.0]},
                 parameters={"p1": [1.0], "p2": [0.5], "p3": [0.1]},
                 duration=0.1,
                 dt_save=0.01,
                 method="euler",
                 time_logging_level=None
             )
             
             # Should have no events recorded
             events = _default_timelogger.events
             assert len(events) == 0
         
         @pytest.mark.nocudasim
         def test_cuda_events_recorded_single_chunk(self, simple_system):
             """Test CUDA events recorded for single chunk execution."""
             # Small problem should fit in one chunk
             result = solve_ivp(
                 simple_system,
                 y0={"x1": [1.0], "x2": [0.0], "x3": [0.0]},
                 parameters={"p1": [1.0], "p2": [0.5], "p3": [0.1]},
                 duration=0.1,
                 dt_save=0.01,
                 method="euler",
                 time_logging_level='verbose'
             )
             
             events = _default_timelogger.events
             event_names = [e.name for e in events]
             
             # Should have GPU workload event
             assert "gpu_workload" in event_names
             
             # Should have per-chunk events for chunk 0
             assert "h2d_transfer_chunk_0" in event_names
             assert "kernel_chunk_0" in event_names
             assert "d2h_transfer_chunk_0" in event_names
         
         @pytest.mark.nocudasim
         def test_cuda_events_multiple_chunks(self, simple_system):
             """Test CUDA events recorded for multiple chunk execution."""
             # Force multiple chunks with small memory proportion
             solver = Solver(
                 simple_system,
                 algorithm="euler",
                 time_logging_level='verbose',
                 mem_proportion=0.01  # Force chunking
             )
             
             # Run with many runs to trigger chunking
             result = solver.solve(
                 initial_values={"x1": np.ones(100), "x2": np.zeros(100), "x3": np.zeros(100)},
                 parameters={"p1": np.ones(100), "p2": np.ones(100)*0.5, "p3": np.ones(100)*0.1},
                 duration=0.1,
                 settling_time=0.0,
             )
             
             events = _default_timelogger.events
             event_names = [e.name for e in events]
             
             # Should have multiple chunk events
             chunk_events = [n for n in event_names if "_chunk_" in n]
             assert len(chunk_events) > 3  # More than one chunk's worth
         
         def test_timing_summary_printed_default_verbosity(self, simple_system, capsys):
             """Test that timing summary is printed at default verbosity."""
             result = solve_ivp(
                 simple_system,
                 y0={"x1": [1.0], "x2": [0.0], "x3": [0.0]},
                 parameters={"p1": [1.0], "p2": [0.5], "p3": [0.1]},
                 duration=0.1,
                 dt_save=0.01,
                 method="euler",
                 time_logging_level='default'
             )
             
             captured = capsys.readouterr()
             
             # Should have printed timing summary
             assert "TIMELOGGER Runtime Timing Summary:" in captured.out
             assert "solver_solve:" in captured.out
         
         def test_timing_summary_not_printed_none_verbosity(self, simple_system, capsys):
             """Test that no timing output with verbosity None."""
             result = solve_ivp(
                 simple_system,
                 y0={"x1": [1.0], "x2": [0.0], "x3": [0.0]},
                 parameters={"p1": [1.0], "p2": [0.5], "p3": [0.1]},
                 duration=0.1,
                 dt_save=0.01,
                 method="euler",
                 time_logging_level=None
             )
             
             captured = capsys.readouterr()
             
             # Should have no timing output
             assert "TIMELOGGER" not in captured.out
         
         def test_per_chunk_timing_verbose_mode(self, simple_system, capsys):
             """Test that per-chunk timing printed in verbose mode."""
             if CUDA_SIMULATION:
                 pytest.skip("CUDA events not meaningful in simulation")
             
             result = solve_ivp(
                 simple_system,
                 y0={"x1": [1.0], "x2": [0.0], "x3": [0.0]},
                 parameters={"p1": [1.0], "p2": [0.5], "p3": [0.1]},
                 duration=0.1,
                 dt_save=0.01,
                 method="euler",
                 time_logging_level='verbose'
             )
             
             captured = capsys.readouterr()
             
             # Should show individual chunk timings
             assert "kernel_chunk_0:" in captured.out
             assert "h2d_transfer_chunk_0:" in captured.out
             assert "d2h_transfer_chunk_0:" in captured.out
     ```
   - Edge cases:
     - Single vs. multiple chunk execution
     - Verbosity None vs. default vs. verbose
     - CUDASIM vs. CUDA mode
   - Integration: End-to-end validation of runtime logging feature

**Outcomes**: 
[To be filled by taskmaster agent]

---

## Summary

**Total Task Groups**: 7

**Dependency Chain**:
1. Group 1 (CUDAEvent Foundation) → Groups 2, 5
2. Group 2 (TimeLogger Extensions) → Groups 3, 6
3. Groups 1, 2 → Group 3 (BatchSolverKernel Instrumentation)
4. Groups 1, 2, 3 → Group 4 (Solver Timing)
5. Groups 1, 2, 3, 4 → Group 7 (Integration Tests)
6. Groups 5, 6 can execute in parallel with Groups 3, 4

**Parallel Execution Opportunities**:
- Groups 5 and 6 (unit tests) can run in parallel after their dependencies
- Group 5 depends only on Group 1
- Group 6 depends only on Groups 1, 2
- These can proceed while Group 3 is being implemented

**Estimated Complexity**:
- Group 1: Medium (80 lines, new class with CUDASIM fallback)
- Group 2: Medium (50 lines, extend existing class)
- Group 3: High (60 lines, careful instrumentation in multiple locations)
- Group 4: Low (30 lines, straightforward timing additions)
- Group 5: Medium (100 lines, comprehensive unit tests)
- Group 6: Medium (80 lines, TimeLogger extension tests)
- Group 7: Medium (120 lines, integration tests)

**Total Lines of Code**: ~520 lines (implementation: ~220, tests: ~300)

**Critical Implementation Details**:
1. **Per-chunk event recording**: Each chunk gets 3 CUDAEvents (h2d, kernel, d2h)
2. **Stream-aware recording**: All events record on solver's registered stream
3. **Deferred timing calculation**: elapsed_time_ms() called AFTER sync_stream()
4. **Zero overhead when disabled**: All instrumentation guarded by verbosity checks
5. **CUDASIM compatibility**: CUDAEvent provides time.perf_counter() fallback

**Key Integration Points**:
- CUDAEvent exported from cuda_simsafe module
- TimeLogger extensions maintain backward compatibility
- BatchSolverKernel event recording isolated to run() method
- Solver.solve() coordinates event retrieval and summary printing
- All changes respect existing verbosity controls
