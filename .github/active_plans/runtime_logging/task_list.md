# Implementation Task List
# Feature: Runtime Logging
# Plan Reference: .github/active_plans/runtime_logging/agent_plan.md

## Task Group 1: CUDAEvent Class in cuda_simsafe.py - SEQUENTIAL
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: src/cubie/cuda_simsafe.py (entire file for patterns)
- File: src/cubie/time_logger.py (for TimeLogger API)

**Input Validation Required**:
- name: Must be non-empty string
- category: Must be 'runtime' (or other valid category)
- timelogger: Must be TimeLogger instance (or None for default)

**Tasks**:
1. **Create CUDAEvent class in cuda_simsafe.py**
   - File: src/cubie/cuda_simsafe.py
   - Action: Add CUDAEvent class at end of file, before __all__
   - Details:
     ```python
     class CUDAEvent:
         """CUDA event wrapper with CUDASIM fallback for GPU timing.
         
         In CUDA mode, uses numba.cuda.event() for accurate GPU timing.
         In CUDASIM mode, falls back to time.perf_counter() wall-clock.
         Registers itself with a TimeLogger for deferred event retrieval.
         
         Parameters
         ----------
         name : str
             Event label for TimeLogger (e.g., 'kernel_launch', 'h2d_transfer')
         category : str
             TimeLogger category, typically 'runtime'
         timelogger : TimeLogger
             Logger instance to register with for deferred retrieval
         
         Attributes
         ----------
         name : str
             Event identifier
         category : str
             Event category for filtering
         _start_event : cuda.Event or None
             CUDA start event (None in CUDASIM mode)
         _end_event : cuda.Event or None
             CUDA end event (None in CUDASIM mode)
         _start_time : float or None
             Wall-clock start time (CUDASIM mode only)
         _end_time : float or None
             Wall-clock end time (CUDASIM mode only)
         """
         
         def __init__(
             self, name: str, category: str, timelogger: "TimeLogger"
         ) -> None:
             import time
             self.name = name
             self.category = category
             self._timelogger = timelogger
             
             # CUDA events (real GPU mode only)
             self._start_event = None
             self._end_event = None
             
             # Wall-clock times (CUDASIM fallback)
             self._start_time = None
             self._end_time = None
             
             # Register with TimeLogger for later retrieval
             if timelogger is not None:
                 timelogger.register_cuda_event(self)
         
         def record_start(self, stream=None) -> None:
             """Record the start of a timed GPU operation.
             
             Parameters
             ----------
             stream : cuda.Stream, optional
                 CUDA stream for event recording. Ignored in CUDASIM mode.
             """
             import time
             if CUDA_SIMULATION:
                 self._start_time = time.perf_counter()
             else:
                 self._start_event = cuda.event()
                 if stream is not None:
                     self._start_event.record(stream)
                 else:
                     self._start_event.record()
         
         def record_end(self, stream=None) -> None:
             """Record the end of a timed GPU operation.
             
             Parameters
             ----------
             stream : cuda.Stream, optional
                 CUDA stream for event recording. Ignored in CUDASIM mode.
             """
             import time
             if CUDA_SIMULATION:
                 self._end_time = time.perf_counter()
             else:
                 self._end_event = cuda.event()
                 if stream is not None:
                     self._end_event.record(stream)
                 else:
                     self._end_event.record()
         
         def elapsed_time_ms(self) -> float:
             """Calculate elapsed time in milliseconds.
             
             Returns
             -------
             float
                 Elapsed time in milliseconds between start and end events.
                 Returns 0.0 if events not properly recorded.
             
             Notes
             -----
             In CUDA mode, this call blocks until both events complete.
             Must be called after stream synchronization for accurate results.
             """
             if CUDA_SIMULATION:
                 if self._start_time is None or self._end_time is None:
                     return 0.0
                 return (self._end_time - self._start_time) * 1000.0
             else:
                 if self._start_event is None or self._end_event is None:
                     return 0.0
                 return cuda.event_elapsed_time(
                     self._start_event, self._end_event
                 )
     ```
   - Edge cases:
     - Handle None timelogger (don't register)
     - Handle missing start/end events (return 0.0)
     - CUDASIM fallback to wall-clock timing
   - Integration: Add after existing classes, before __all__

2. **Update __all__ export list**
   - File: src/cubie/cuda_simsafe.py
   - Action: Add "CUDAEvent" to __all__ list
   - Details:
     ```python
     __all__ = [
         # ... existing exports ...
         "CUDAEvent",
     ]
     ```
   - Edge cases: None
   - Integration: Append to existing __all__ list

**Outcomes**:

---

## Task Group 2: TimeLogger Extensions - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Group 1

**Required Context**:
- File: src/cubie/time_logger.py (entire file)
- File: src/cubie/cuda_simsafe.py (CUDAEvent class from Group 1)

**Input Validation Required**:
- register_cuda_event: event must be CUDAEvent instance
- retrieve_cuda_events: no user input (internal method)

**Tasks**:
1. **Add _cuda_events list attribute to TimeLogger.__init__**
   - File: src/cubie/time_logger.py
   - Action: Add new attribute in __init__ method
   - Details:
     ```python
     def __init__(self, verbosity: Optional[str] = None) -> None:
         # ... existing code ...
         self._event_registry: dict[str, dict] = {}
         # Add new attribute for CUDA event tracking
         self._cuda_events: list = []  # List of CUDAEvent instances
     ```
   - Edge cases: None
   - Integration: Add after existing attribute initialization

2. **Add register_cuda_event method**
   - File: src/cubie/time_logger.py
   - Action: Add new method to TimeLogger class
   - Details:
     ```python
     def register_cuda_event(self, event: "CUDAEvent") -> None:
         """Register a CUDAEvent for deferred timing retrieval.
         
         Parameters
         ----------
         event : CUDAEvent
             CUDA event wrapper to register for later retrieval
         
         Notes
         -----
         No-op when verbosity is None.
         CUDAEvent instances call this during their __init__.
         Registered events are processed by retrieve_cuda_events().
         """
         if self.verbosity is None:
             return
         self._cuda_events.append(event)
     ```
   - Edge cases: No-op when verbosity is None
   - Integration: Add after existing methods, before print_summary

3. **Add retrieve_cuda_events method**
   - File: src/cubie/time_logger.py
   - Action: Add new method to TimeLogger class
   - Details:
     ```python
     def retrieve_cuda_events(self) -> None:
         """Collect timing data from all registered CUDAEvents.
         
         Iterates through registered CUDAEvent instances, calculates
         elapsed time, and records as regular timing events.
         Called by Solver.solve() after stream synchronization.
         
         Notes
         -----
         No-op when verbosity is None.
         Clears the registered events list after processing.
         Events are always logged, even if duration is 0.
         """
         if self.verbosity is None:
             return
         
         for event in self._cuda_events:
             elapsed_ms = event.elapsed_time_ms()
             elapsed_s = elapsed_ms / 1000.0
             
             # Create start event record
             start_evt = TimingEvent(
                 name=event.name,
                 event_type='start',
                 timestamp=0.0,  # Relative timing only
                 metadata={'category': event.category, 'source': 'cuda_event'}
             )
             self.events.append(start_evt)
             
             # Create stop event record with duration
             stop_evt = TimingEvent(
                 name=event.name,
                 event_type='stop',
                 timestamp=elapsed_s,  # Store duration as timestamp
                 metadata={
                     'category': event.category,
                     'source': 'cuda_event',
                     'elapsed_ms': elapsed_ms
                 }
             )
             self.events.append(stop_evt)
             
             if self.verbosity == 'debug':
                 print(f"TIMELOGGER [DEBUG] CUDA Event: {event.name} "
                       f"({elapsed_ms:.3f}ms)")
             elif self.verbosity == 'verbose':
                 print(f"TIMELOGGER {event.name}: {elapsed_ms:.3f}ms")
         
         # Clear registered events
         self._cuda_events = []
     ```
   - Edge cases:
     - No-op when verbosity is None
     - Always log events even if 0 duration
     - Clear list after processing
   - Integration: Add after register_cuda_event method

**Outcomes**:

---

## Task Group 3: Event Registration - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Groups 1, 2

**Required Context**:
- File: src/cubie/batchsolving/solver.py (lines 1-50, 150-180)
- File: src/cubie/batchsolving/BatchSolverKernel.py (lines 1-50, 109-170)
- File: src/cubie/time_logger.py (register_event API)

**Input Validation Required**:
None - Event registration is internal, no user input

**Tasks**:
1. **Register solve_ivp_total event at module level**
   - File: src/cubie/batchsolving/solver.py
   - Action: Add module-level registration after imports
   - Details:
     ```python
     # After line 32 (after _default_timelogger import)
     _default_timelogger.register_event(
         label='solve_ivp_total',
         category='runtime',
         description='Total wall-clock time for solve_ivp() function'
     )
     ```
   - Edge cases: None
   - Integration: Place after _default_timelogger import

2. **Register solver_solve_total event in Solver.__init__**
   - File: src/cubie/batchsolving/solver.py
   - Action: Add event registration after set_verbosity call
   - Details:
     ```python
     # After line 179 (_default_timelogger.set_verbosity)
     _default_timelogger.register_event(
         label='solver_solve_total',
         category='runtime',
         description='Total wall-clock time for Solver.solve() method'
     )
     ```
   - Edge cases: None
   - Integration: Place after set_verbosity() call

3. **Register kernel_run_total event in BatchSolverKernel.__init__**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Add import and event registration
   - Details:
     ```python
     # Add import at top of file (after line 13)
     from cubie.time_logger import _default_timelogger
     
     # In __init__, after super().__init__() (after line 121)
     _default_timelogger.register_event(
         label='kernel_run_total',
         category='runtime',
         description='Total wall-clock time for BatchSolverKernel.run()'
     )
     _default_timelogger.register_event(
         label='kernel_launch',
         category='runtime',
         description='GPU kernel execution time per chunk'
     )
     _default_timelogger.register_event(
         label='h2d_transfer',
         category='runtime',
         description='Host-to-device transfer time for all arrays'
     )
     _default_timelogger.register_event(
         label='d2h_transfer',
         category='runtime',
         description='Device-to-host transfer time for all arrays'
     )
     ```
   - Edge cases: None
   - Integration: Add import at module level, registrations in __init__

**Outcomes**:

---

## Task Group 4: Outer Timing in solve_ivp and Solver.solve - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Groups 1, 2, 3

**Required Context**:
- File: src/cubie/batchsolving/solver.py (lines 35-109, 309-393)
- File: src/cubie/time_logger.py (start_event, stop_event, retrieve_cuda_events)

**Input Validation Required**:
None - Timing calls are internal, no user input

**Tasks**:
1. **Add timing to solve_ivp function**
   - File: src/cubie/batchsolving/solver.py
   - Action: Wrap function body with timing
   - Details:
     ```python
     def solve_ivp(...) -> SolveResult:
         """..."""
         _default_timelogger.start_event('solve_ivp_total')
         try:
             # Existing function body
             loop_settings = kwargs.pop("loop_settings", None)
             if dt_save is not None:
                 kwargs.setdefault("dt_save", dt_save)
             
             solver = Solver(...)
             results = solver.solve(...)
             return results
         finally:
             _default_timelogger.stop_event('solve_ivp_total')
     ```
   - Edge cases:
     - Use try/finally to ensure stop on exceptions
   - Integration: Wrap existing function body

2. **Add timing to Solver.solve method**
   - File: src/cubie/batchsolving/solver.py
   - Action: Add timing and retrieve_cuda_events call
   - Details:
     ```python
     def solve(self, ...) -> SolveResult:
         """..."""
         _default_timelogger.start_event('solver_solve_total')
         try:
             # Existing code...
             if kwargs:
                 self.update(kwargs, silent=True)
             
             inits, params = self.grid_builder(...)
             # ... driver updates ...
             
             self.kernel.run(...)
             self.memory_manager.sync_stream(self.kernel)
             
             # Collect CUDA event timings after sync
             _default_timelogger.retrieve_cuda_events()
             
             return SolveResult.from_solver(self, results_type=results_type)
         finally:
             _default_timelogger.stop_event('solver_solve_total')
     ```
   - Edge cases:
     - Use try/finally for exception safety
     - Call retrieve_cuda_events AFTER sync_stream
   - Integration: Modify existing method

**Outcomes**:

---

## Task Group 5: BatchSolverKernel.run() Timing with CUDAEvents - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Groups 1, 2, 3

**Required Context**:
- File: src/cubie/batchsolving/BatchSolverKernel.py (lines 215-371)
- File: src/cubie/cuda_simsafe.py (CUDAEvent class)
- File: src/cubie/time_logger.py (_default_timelogger)

**Input Validation Required**:
None - Timing is internal, no user input

**Tasks**:
1. **Add outer timing and CUDAEvent instrumentation to run()**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Add timing instrumentation to run() method
   - Details:
     ```python
     def run(self, ...) -> None:
         """..."""
         from cubie.cuda_simsafe import CUDAEvent
         
         _default_timelogger.start_event('kernel_run_total')
         try:
             if stream is None:
                 stream = self.stream
             
             # ... existing setup code (lines 264-299) ...
             
             # Create ONE CUDAEvent for ALL h2d transfers
             h2d_event = CUDAEvent(
                 'h2d_transfer', 'runtime', _default_timelogger
             )
             h2d_event.record_start(stream)
             
             # Process all chunks' h2d transfers first
             for i in range(self.chunks):
                 indices = slice(
                     i * chunk_params.size, (i + 1) * chunk_params.size
                 )
                 self.input_arrays.initialise(indices)
                 self.output_arrays.initialise(indices)
             
             h2d_event.record_end(stream)
             
             # Kernel launches with per-chunk CUDAEvents
             for i in range(self.chunks):
                 # Create per-chunk kernel event
                 kernel_event = CUDAEvent(
                     f'kernel_launch_{i}', 'runtime', _default_timelogger
                 )
                 
                 if (chunk_axis == "time") and (i != 0):
                     chunk_warmup = precision(0.0)
                     chunk_t0 = t0 + precision(i) * chunk_params.duration
                 
                 kernel_event.record_start(stream)
                 
                 self.kernel[
                     BLOCKSPERGRID,
                     (threads_per_loop, runsperblock),
                     stream,
                     dynamic_sharedmem,
                 ](
                     # ... existing kernel args ...
                 )
                 
                 kernel_event.record_end(stream)
             
             # Create ONE CUDAEvent for ALL d2h transfers
             d2h_event = CUDAEvent(
                 'd2h_transfer', 'runtime', _default_timelogger
             )
             d2h_event.record_start(stream)
             
             for i in range(self.chunks):
                 indices = slice(
                     i * chunk_params.size, (i + 1) * chunk_params.size
                 )
                 self.input_arrays.finalise(indices)
                 self.output_arrays.finalise(indices)
             
             d2h_event.record_end(stream)
             
             if self.profileCUDA:
                 cuda.profile_stop()
         finally:
             _default_timelogger.stop_event('kernel_run_total')
     ```
   - Edge cases:
     - Consolidated h2d_transfer and d2h_transfer (ONE event each)
     - Per-chunk kernel_launch_{i} events
     - NO stream.synchronize() during this method
     - Profile start/stop preserved if enabled
   - Integration: Major restructuring of run() method

**Outcomes**:

---

## Task Group 6: Tests - CUDAEvent Class - PARALLEL
**Status**: [ ]
**Dependencies**: Groups 1, 2

**Required Context**:
- File: tests/test_cuda_simsafe.py (existing tests for reference)
- File: src/cubie/cuda_simsafe.py (CUDAEvent implementation)

**Input Validation Required**:
None - Tests verify behavior

**Tasks**:
1. **Create test_cuda_event.py**
   - File: tests/test_cuda_event.py (create new)
   - Action: Test CUDAEvent class functionality
   - Details:
     ```python
     """Test CUDAEvent class for GPU timing."""
     import pytest
     import time
     from cubie.cuda_simsafe import CUDAEvent, CUDA_SIMULATION
     from cubie.time_logger import TimeLogger
     
     
     @pytest.fixture
     def test_timelogger():
         """Create fresh TimeLogger for tests."""
         return TimeLogger(verbosity='default')
     
     
     def test_cuda_event_creation(test_timelogger):
         """Test CUDAEvent can be created."""
         event = CUDAEvent('test_event', 'runtime', test_timelogger)
         assert event.name == 'test_event'
         assert event.category == 'runtime'
     
     
     def test_cuda_event_registers_with_timelogger(test_timelogger):
         """Test CUDAEvent registers with TimeLogger."""
         event = CUDAEvent('test_event', 'runtime', test_timelogger)
         assert event in test_timelogger._cuda_events
     
     
     def test_cuda_event_record_start_end(test_timelogger):
         """Test start/end recording."""
         event = CUDAEvent('test_event', 'runtime', test_timelogger)
         event.record_start()
         time.sleep(0.01)  # Small delay
         event.record_end()
         
         elapsed = event.elapsed_time_ms()
         assert elapsed >= 0.0  # Should be positive
     
     
     def test_cuda_event_elapsed_without_recording(test_timelogger):
         """Test elapsed_time_ms returns 0 if not recorded."""
         event = CUDAEvent('test_event', 'runtime', test_timelogger)
         assert event.elapsed_time_ms() == 0.0
     
     
     def test_cuda_event_none_timelogger():
         """Test CUDAEvent with None timelogger doesn't crash."""
         event = CUDAEvent('test_event', 'runtime', None)
         event.record_start()
         event.record_end()
         elapsed = event.elapsed_time_ms()
         assert elapsed >= 0.0
     
     
     def test_cuda_event_cudasim_uses_wallclock():
         """Verify CUDASIM mode uses wall-clock timing."""
         if not CUDA_SIMULATION:
             pytest.skip("Test only for CUDASIM mode")
         
         logger = TimeLogger(verbosity='default')
         event = CUDAEvent('test_event', 'runtime', logger)
         
         event.record_start()
         time.sleep(0.05)  # 50ms delay
         event.record_end()
         
         elapsed = event.elapsed_time_ms()
         # Should be around 50ms (allow tolerance)
         assert 40.0 <= elapsed <= 150.0
     ```
   - Edge cases:
     - None timelogger
     - Missing record calls
     - CUDASIM vs CUDA mode
   - Integration: New test file in tests/

**Outcomes**:

---

## Task Group 7: Tests - TimeLogger Extensions - PARALLEL
**Status**: [ ]
**Dependencies**: Groups 1, 2

**Required Context**:
- File: tests/test_time_logger.py (existing tests)
- File: src/cubie/time_logger.py (new methods)

**Input Validation Required**:
None - Tests verify behavior

**Tasks**:
1. **Add tests for register_cuda_event and retrieve_cuda_events**
   - File: tests/test_time_logger.py (modify existing)
   - Action: Add new test functions
   - Details:
     ```python
     def test_register_cuda_event():
         """Test register_cuda_event adds to list."""
         from cubie.cuda_simsafe import CUDAEvent
         
         logger = TimeLogger(verbosity='default')
         event = CUDAEvent('test', 'runtime', logger)
         
         assert len(logger._cuda_events) == 1
         assert logger._cuda_events[0] is event
     
     
     def test_register_cuda_event_noop_when_none_verbosity():
         """Test register_cuda_event is no-op when verbosity=None."""
         from cubie.cuda_simsafe import CUDAEvent
         
         logger = TimeLogger(verbosity=None)
         event = CUDAEvent('test', 'runtime', logger)
         
         assert len(logger._cuda_events) == 0
     
     
     def test_retrieve_cuda_events():
         """Test retrieve_cuda_events collects timing."""
         from cubie.cuda_simsafe import CUDAEvent
         import time
         
         logger = TimeLogger(verbosity='default')
         logger.register_event('test', 'runtime', 'Test event')
         
         event = CUDAEvent('test', 'runtime', logger)
         event.record_start()
         time.sleep(0.01)
         event.record_end()
         
         logger.retrieve_cuda_events()
         
         # Events should now be in logger.events
         event_names = [e.name for e in logger.events]
         assert 'test' in event_names
         
         # List should be cleared
         assert len(logger._cuda_events) == 0
     
     
     def test_retrieve_cuda_events_clears_list():
         """Test retrieve clears _cuda_events list."""
         from cubie.cuda_simsafe import CUDAEvent
         
         logger = TimeLogger(verbosity='default')
         event = CUDAEvent('test', 'runtime', logger)
         event.record_start()
         event.record_end()
         
         assert len(logger._cuda_events) == 1
         logger.retrieve_cuda_events()
         assert len(logger._cuda_events) == 0
     ```
   - Edge cases:
     - Verbosity None behavior
     - List clearing after retrieve
   - Integration: Add to existing test file

**Outcomes**:

---

## Task Group 8: Tests - Runtime Timing Integration - PARALLEL
**Status**: [ ]
**Dependencies**: Groups 1-5

**Required Context**:
- File: tests/conftest.py (fixtures)
- File: tests/system_fixtures.py (ODE systems)
- All modified source files

**Input Validation Required**:
None - Tests verify behavior

**Tasks**:
1. **Create test_runtime_logging.py**
   - File: tests/batchsolving/test_runtime_logging.py (create new)
   - Action: Test full runtime logging integration
   - Details:
     ```python
     """Test runtime logging integration."""
     import pytest
     import numpy as np
     from cubie import solve_ivp, Solver
     from cubie.time_logger import _default_timelogger
     
     
     @pytest.fixture(autouse=True)
     def reset_timelogger():
         """Reset timelogger before each test."""
         original_verbosity = _default_timelogger.verbosity
         _default_timelogger.events = []
         _default_timelogger._active_starts = {}
         _default_timelogger._cuda_events = []
         yield
         _default_timelogger.set_verbosity(original_verbosity)
         _default_timelogger.events = []
         _default_timelogger._cuda_events = []
     
     
     def test_solve_ivp_total_timing(three_state_linear):
         """Test solve_ivp_total event is recorded."""
         _default_timelogger.set_verbosity('default')
         
         y0 = np.array([[1.0, 0.0, 0.0]])
         result = solve_ivp(
             three_state_linear,
             y0,
             duration=0.1,
             dt_save=0.01,
             time_logging_level='default'
         )
         
         event_names = [e.name for e in _default_timelogger.events]
         assert 'solve_ivp_total' in event_names
     
     
     def test_solver_solve_total_timing(three_state_linear):
         """Test solver_solve_total event is recorded."""
         _default_timelogger.set_verbosity('default')
         
         solver = Solver(
             three_state_linear,
             algorithm='euler',
             time_logging_level='default'
         )
         y0 = np.array([[1.0, 0.0, 0.0]])
         result = solver.solve(y0, duration=0.1, dt_save=0.01)
         
         event_names = [e.name for e in _default_timelogger.events]
         assert 'solver_solve_total' in event_names
     
     
     def test_kernel_run_total_timing(three_state_linear):
         """Test kernel_run_total event is recorded."""
         _default_timelogger.set_verbosity('default')
         
         solver = Solver(
             three_state_linear,
             algorithm='euler',
             time_logging_level='default'
         )
         y0 = np.array([[1.0, 0.0, 0.0]])
         result = solver.solve(y0, duration=0.1, dt_save=0.01)
         
         event_names = [e.name for e in _default_timelogger.events]
         assert 'kernel_run_total' in event_names
     
     
     def test_cuda_events_collected_after_sync(three_state_linear):
         """Test CUDA events are collected after sync."""
         _default_timelogger.set_verbosity('default')
         
         solver = Solver(
             three_state_linear,
             algorithm='euler',
             time_logging_level='default'
         )
         y0 = np.array([[1.0, 0.0, 0.0]])
         result = solver.solve(y0, duration=0.1, dt_save=0.01)
         
         # CUDA events should be cleared (retrieved)
         assert len(_default_timelogger._cuda_events) == 0
         
         # Should have h2d_transfer and d2h_transfer events
         event_names = [e.name for e in _default_timelogger.events]
         assert 'h2d_transfer' in event_names
         assert 'd2h_transfer' in event_names
     
     
     def test_verbosity_none_no_events(three_state_linear):
         """Test no events when verbosity is None."""
         y0 = np.array([[1.0, 0.0, 0.0]])
         result = solve_ivp(
             three_state_linear,
             y0,
             duration=0.1,
             dt_save=0.01,
             time_logging_level=None
         )
         
         assert len(_default_timelogger.events) == 0
     
     
     def test_consolidated_transfer_events(three_state_linear):
         """Test ONE h2d and ONE d2h event (consolidated)."""
         _default_timelogger.set_verbosity('default')
         
         solver = Solver(
             three_state_linear,
             algorithm='euler',
             time_logging_level='default'
         )
         y0 = np.array([[1.0, 0.0, 0.0]])
         result = solver.solve(y0, duration=0.1, dt_save=0.01)
         
         h2d_events = [
             e for e in _default_timelogger.events
             if e.name == 'h2d_transfer'
         ]
         d2h_events = [
             e for e in _default_timelogger.events
             if e.name == 'd2h_transfer'
         ]
         
         # Should have exactly 2 events each (start + stop from retrieve)
         assert len(h2d_events) == 2
         assert len(d2h_events) == 2
     ```
   - Edge cases:
     - Verbosity None
     - Consolidated events
     - CUDA events cleared after retrieve
   - Integration: New test file using existing fixtures

**Outcomes**:

---

## Summary

**Total Task Groups**: 8
**Dependency Chain**:
- Group 1: CUDAEvent class (no dependencies)
- Group 2: TimeLogger extensions (depends on Group 1 for type hints)
- Group 3: Event registration (depends on Groups 1, 2)
- Groups 4, 5: Instrumentation (depends on Groups 1, 2, 3)
- Groups 6, 7, 8: Tests (depend on implementation groups)

**Parallel Execution Opportunities**:
- Groups 1 and 2 are mostly independent (can be parallel)
- Groups 6, 7, 8 (tests) can run in parallel after implementation

**Key Implementation Points**:
1. **CUDAEvent in cuda_simsafe.py** - CUDASIM-safe abstraction
2. **Consolidated transfer events** - ONE h2d_transfer, ONE d2h_transfer
3. **NO stream synchronization during BatchSolverKernel.run()**
4. **Deferred event retrieval** via retrieve_cuda_events() after sync
5. **Always log events** even if 0 duration

**Files Modified**:
- src/cubie/cuda_simsafe.py (add CUDAEvent class)
- src/cubie/time_logger.py (add register/retrieve methods)
- src/cubie/batchsolving/solver.py (timing, event registration)
- src/cubie/batchsolving/BatchSolverKernel.py (timing, CUDAEvents)

**Files Created**:
- tests/test_cuda_event.py
- tests/batchsolving/test_runtime_logging.py

**Estimated Complexity**: Medium
- Core abstractions are straightforward
- Main challenge is restructuring BatchSolverKernel.run() for consolidated events
- CUDASIM compatibility handled by CUDAEvent abstraction
