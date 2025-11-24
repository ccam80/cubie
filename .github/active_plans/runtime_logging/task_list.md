# Implementation Task List
# Feature: Runtime Logging
# Plan Reference: .github/active_plans/runtime_logging/agent_plan.md

## Task Group 1: Event Registration at Module/Class Level - SEQUENTIAL
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: src/cubie/batchsolving/solver.py (lines 1-20, 35-109, 111-236)
- File: src/cubie/batchsolving/BatchSolverKernel.py (lines 1-100, 134-213)
- File: src/cubie/time_logger.py (entire file for API reference)

**Input Validation Required**:
None - Event registration does not require input validation

**Tasks**:
1. **Register solve_ivp event at module level**
   - File: src/cubie/batchsolving/solver.py
   - Action: Add module-level registration after imports
   - Details:
     ```python
     # After line 32 (after _default_timelogger import)
     _default_timelogger.register_event(
         label='solve_ivp_execution',
         category='runtime',
         description='Total execution time for solve_ivp() function'
     )
     ```
   - Edge cases: None
   - Integration: Place after _default_timelogger import, before solve_ivp function definition

2. **Register Solver events in __init__**
   - File: src/cubie/batchsolving/solver.py
   - Action: Add event registration in Solver.__init__() after setting verbosity
   - Details:
     ```python
     # After line 179 (_default_timelogger.set_verbosity(time_logging_level))
     _default_timelogger.register_event(
         label='solver_solve_execution',
         category='runtime',
         description='Total execution time for Solver.solve() method'
     )
     ```
   - Edge cases: None
   - Integration: Place immediately after set_verbosity() call

3. **Register BatchSolverKernel events in __init__**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Add event registrations in __init__() method
   - Details:
     ```python
     # Add after super().__init__() (approximately after line 186)
     from cubie.time_logger import _default_timelogger
     
     # Register runtime events
     _default_timelogger.register_event(
         label='kernel_run_execution',
         category='runtime',
         description='Total execution time for BatchSolverKernel.run() method'
     )
     _default_timelogger.register_event(
         label='kernel_launch',
         category='runtime',
         description='GPU kernel execution time per chunk (CUDA event timing)'
     )
     _default_timelogger.register_event(
         label='h2d_transfer_initial_values',
         category='runtime',
         description='Host-to-device transfer of initial values'
     )
     _default_timelogger.register_event(
         label='h2d_transfer_parameters',
         category='runtime',
         description='Host-to-device transfer of parameters'
     )
     _default_timelogger.register_event(
         label='h2d_transfer_driver_coefficients',
         category='runtime',
         description='Host-to-device transfer of driver coefficients'
     )
     _default_timelogger.register_event(
         label='d2h_transfer_state',
         category='runtime',
         description='Device-to-host transfer of state outputs'
     )
     _default_timelogger.register_event(
         label='d2h_transfer_observables',
         category='runtime',
         description='Device-to-host transfer of observable outputs'
     )
     _default_timelogger.register_event(
         label='d2h_transfer_summaries',
         category='runtime',
         description='Device-to-host transfer of summary outputs'
     )
     _default_timelogger.register_event(
         label='d2h_transfer_status_codes',
         category='runtime',
         description='Device-to-host transfer of status codes'
     )
     _default_timelogger.register_event(
         label='d2h_transfer_iteration_counters',
         category='runtime',
         description='Device-to-host transfer of iteration counters'
     )
     ```
   - Edge cases: Ensure import is added at method level, not module level
   - Integration: Place after super().__init__() call in BatchSolverKernel.__init__()

**Outcomes**:

---

## Task Group 2: Add solve_ivp Timing - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Group 1

**Required Context**:
- File: src/cubie/batchsolving/solver.py (lines 35-109 for solve_ivp function)
- File: src/cubie/time_logger.py (lines 76-140 for start_event/stop_event API)

**Input Validation Required**:
None - Timing calls do not require additional validation

**Tasks**:
1. **Wrap solve_ivp function with timing**
   - File: src/cubie/batchsolving/solver.py
   - Action: Add start/stop timing calls in solve_ivp function
   - Details:
     ```python
     def solve_ivp(...) -> SolveResult:
         """..."""
         # Add at function entry (after line 48, before other logic)
         _default_timelogger.start_event('solve_ivp_execution')
         
         try:
             # Existing function body (lines 87-108)
             loop_settings = kwargs.pop("loop_settings", None)
             if dt_save is not None:
                 kwargs.setdefault("dt_save", dt_save)
             
             solver = Solver(...)
             results = solver.solve(...)
             
             return results
         finally:
             # Stop timing before return
             _default_timelogger.stop_event('solve_ivp_execution')
             # Print summary if verbosity is default
             if _default_timelogger.verbosity == 'default':
                 _default_timelogger.print_summary(category='runtime')
     ```
   - Edge cases: 
     - Ensure timing stops even if exception is raised (use try/finally)
     - Only print summary when verbosity='default' (not verbose or debug)
   - Integration: Wraps entire function body with try/finally block

**Outcomes**:

---

## Task Group 3: Add Solver.solve() Timing - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Group 1

**Required Context**:
- File: src/cubie/batchsolving/solver.py (lines 280-430 for Solver.solve method)
- File: src/cubie/time_logger.py (lines 76-140 for start_event/stop_event API)

**Input Validation Required**:
None - Timing calls do not require additional validation

**Tasks**:
1. **Add timing to Solver.solve() method**
   - File: src/cubie/batchsolving/solver.py
   - Action: Add start/stop timing calls at method entry/exit
   - Details:
     ```python
     def solve(self, ...) -> SolveResult:
         """..."""
         # Add at method entry (after line ~285, before other logic)
         _default_timelogger.start_event('solver_solve_execution')
         
         try:
             # Existing method body:
             # - Grid building
             # - Driver interpolator updates
             # - kernel.run() call
             # - Stream synchronization
             # - Result packaging
             
             # ... existing code ...
             
             # Before final return statement
             _default_timelogger.stop_event('solver_solve_execution')
             
             # Print summary if verbosity is default and NOT called from solve_ivp
             # (solve_ivp will print its own summary)
             # Check if this was called directly by checking call stack
             if _default_timelogger.verbosity == 'default':
                 # Only print if solve_ivp timing is not active
                 if 'solve_ivp_execution' not in _default_timelogger._active_starts:
                     _default_timelogger.print_summary(category='runtime')
             
             return result
         except Exception:
             _default_timelogger.stop_event('solver_solve_execution')
             raise
     ```
   - Edge cases:
     - Handle exceptions to ensure timing stops
     - Avoid duplicate print_summary when called from solve_ivp (check active starts)
   - Integration: Wraps entire method body with try/except/finally pattern

**Outcomes**:

---

## Task Group 4: Add BatchSolverKernel.run() Outer Timing - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Group 1

**Required Context**:
- File: src/cubie/batchsolving/BatchSolverKernel.py (lines 215-370 for run method)
- File: src/cubie/time_logger.py (lines 76-140 for timing API)

**Input Validation Required**:
None - Timing calls do not require additional validation

**Tasks**:
1. **Add outer timing to run() method**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Add timing at method entry and before method exit
   - Details:
     ```python
     def run(self, ...) -> None:
         """..."""
         # Add import at top of method
         from cubie.time_logger import _default_timelogger
         
         # Start timing at method entry (after line ~226, before validation)
         _default_timelogger.start_event('kernel_run_execution')
         
         try:
             # Existing method body (lines 264-370)
             # - Array updates
             # - Compile settings refresh
             # - Memory allocation
             # - Chunk processing loop
             # - Kernel launches
             
             # ... existing code ...
             
         finally:
             # Stop timing before method exit
             _default_timelogger.stop_event('kernel_run_execution')
     ```
   - Edge cases:
     - Must use try/finally to ensure timing stops
     - Do NOT print summary (Solver.solve handles that)
   - Integration: Wraps entire run() method body

**Outcomes**:

---

## Task Group 5: Add CUDA Kernel Launch Timing - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Groups 1, 4

**Required Context**:
- File: src/cubie/batchsolving/BatchSolverKernel.py (lines 332-368 for kernel launch loop)
- File: src/cubie/cuda_simsafe.py (for is_cudasim_enabled check)
- numba.cuda documentation for event API

**Input Validation Required**:
None - Timing is measurement only, no user input to validate

**Tasks**:
1. **Add CUDA event timing for kernel launches**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Add CUDA event timing around kernel launch in chunk loop
   - Details:
     ```python
     # In the chunk loop (around lines 332-368)
     for i in range(self.chunks):
         indices = slice(i * chunk_params.size, (i + 1) * chunk_params.size)
         self.input_arrays.initialise(indices)
         self.output_arrays.initialise(indices)
         
         # Adjust warmup for time-chunked runs
         if (chunk_axis == "time") and (i != 0):
             chunk_warmup = precision(0.0)
             chunk_t0 = t0 + precision(i) * chunk_params.duration
         
         # Add CUDA event timing
         if not is_cudasim_enabled():
             # Use CUDA events for accurate GPU timing
             start_evt = cuda.event()
             stop_evt = cuda.event()
             start_evt.record(stream)
         else:
             # Fallback to wall-clock timing in CUDASIM
             _default_timelogger.start_event(
                 'kernel_launch',
                 chunk_index=i,
                 chunk_runs=chunk_params.runs
             )
         
         # Existing kernel launch (lines 342-360)
         self.kernel[
             BLOCKSPERGRID,
             (threads_per_loop, runsperblock),
             stream,
             dynamic_sharedmem,
         ](
             self.input_arrays.device_initial_values,
             self.input_arrays.device_parameters,
             self.input_arrays.device_driver_coefficients,
             self.output_arrays.device_state,
             self.output_arrays.device_observables,
             self.output_arrays.device_state_summaries,
             self.output_arrays.device_observable_summaries,
             self.output_arrays.device_iteration_counters,
             self.output_arrays.device_status_codes,
             chunk_params.duration,
             chunk_warmup,
             chunk_t0,
             numruns,
         )
         
         # Record stop event and compute elapsed time
         if not is_cudasim_enabled():
             stop_evt.record(stream)
             # Sync stream to read event time
             self.memory_manager.sync_stream(self)
             elapsed_ms = cuda.event_elapsed_time(start_evt, stop_evt)
             _default_timelogger.stop_event(
                 'kernel_launch',
                 gpu_time_ms=elapsed_ms,
                 chunk_index=i,
                 chunk_runs=chunk_params.runs
             )
         else:
             # Stop wall-clock timing in CUDASIM
             _default_timelogger.stop_event(
                 'kernel_launch',
                 chunk_index=i,
                 chunk_runs=chunk_params.runs
             )
         
         # Existing finalise calls (lines 366-367)
         self.input_arrays.finalise(indices)
         self.output_arrays.finalise(indices)
     ```
   - Edge cases:
     - CUDASIM mode: Fall back to wall-clock timing
     - Stream synchronization must happen before reading event elapsed time
     - Multiple chunks: timing repeated for each chunk
   - Integration: Wraps kernel launch in chunk loop; requires is_cudasim_enabled import

**Outcomes**:

---

## Task Group 6: Add InputArrays Transfer Timing - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Group 1

**Required Context**:
- File: src/cubie/batchsolving/arrays/BatchInputArrays.py (lines 318-365 for initialise method)
- File: src/cubie/batchsolving/arrays/BaseArrayManager.py (lines 1-150, 300-450 for transfer methods)
- File: src/cubie/time_logger.py (lines 76-140 for timing API)

**Input Validation Required**:
None - Timing is measurement only

**Tasks**:
1. **Add import to BatchInputArrays**
   - File: src/cubie/batchsolving/arrays/BatchInputArrays.py
   - Action: Add import at module level
   - Details:
     ```python
     # After line 8 (after existing imports)
     from cubie.time_logger import _default_timelogger
     ```
   - Edge cases: None
   - Integration: Standard import at module level

2. **Add transfer timing in initialise() method**
   - File: src/cubie/batchsolving/arrays/BatchInputArrays.py
   - Action: Modify initialise() method to time individual array transfers
   - Details:
     ```python
     def initialise(self, host_indices: Union[slice, NDArray]) -> None:
         """Copy a batch chunk of host data to device buffers."""
         from_ = []
         to_ = []
         
         # Existing logic to determine arrays_to_copy (lines 339-350)
         if self._chunks <= 1:
             arrays_to_copy = [array for array in self._needs_overwrite]
             self._needs_overwrite = []
         else:
             arrays_to_copy = list(self.device.array_names())
         
         # Track which arrays are actually transferred for timing
         arrays_being_transferred = []
         
         for array_name in arrays_to_copy:
             device_obj = self.device.get_managed_array(array_name)
             to_.append(device_obj.array)
             host_obj = self.host.get_managed_array(array_name)
             if self._chunks <= 1 or not device_obj.is_chunked:
                 from_.append(host_obj.array)
                 arrays_being_transferred.append(array_name)
             else:
                 # Chunked slicing logic (existing code)
                 stride_order = host_obj.stride_order
                 slice_tuple = [slice(None)] * len(stride_order)
                 if self._chunk_axis in stride_order:
                     chunk_index = stride_order.index(self._chunk_axis)
                     slice_tuple[chunk_index] = host_indices
                     slice_tuple = tuple(slice_tuple)
                 from_.append(host_obj.array[slice_tuple])
                 arrays_being_transferred.append(array_name)
         
         # Time each array transfer separately
         for i, array_name in enumerate(arrays_being_transferred):
             event_name = f'h2d_transfer_{array_name}'
             nbytes = from_[i].nbytes if hasattr(from_[i], 'nbytes') else 0
             
             _default_timelogger.start_event(
                 event_name,
                 array_name=array_name,
                 nbytes=nbytes,
                 chunk_slice=str(host_indices)
             )
             
             # Transfer single array
             self.to_device([from_[i]], [to_[i]])
             
             _default_timelogger.stop_event(
                 event_name,
                 array_name=array_name,
                 nbytes=nbytes
             )
     ```
   - Edge cases:
     - Driver coefficients may be None (skip timing if not in arrays_to_copy)
     - Chunked vs non-chunked arrays handled by existing logic
     - Only time arrays that are actually transferred
   - Integration: Replace single bulk to_device() call with per-array transfers and timing

**Outcomes**:

---

## Task Group 7: Add OutputArrays Transfer Timing - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Group 1

**Required Context**:
- File: src/cubie/batchsolving/arrays/BatchOutputArrays.py (lines 1-100, 150-300 for structure)
- File: src/cubie/batchsolving/arrays/BaseArrayManager.py (lines 570-600 for finalise signature)
- File: src/cubie/time_logger.py (lines 76-140 for timing API)

**Input Validation Required**:
None - Timing is measurement only

**Tasks**:
1. **Implement finalise() method with transfer timing in OutputArrays**
   - File: src/cubie/batchsolving/arrays/BatchOutputArrays.py
   - Action: Implement finalise() method (currently abstract in BaseArrayManager)
   - Details:
     ```python
     # Add after line ~300 (at end of OutputArrays class)
     
     def finalise(self, host_indices: Union[slice, NDArray]) -> None:
         """Transfer output arrays from device to host with timing.
         
         Parameters
         ----------
         host_indices
             Chunk indices for the arrays being finalized.
         
         Returns
         -------
         None
             Device arrays are copied to host arrays in place.
         """
         from cubie.time_logger import _default_timelogger
         
         # Map of output array names to their active flags
         output_map = {
             'state': self.active_outputs.state,
             'observables': self.active_outputs.observables,
             'state_summaries': self.active_outputs.state_summaries,
             'observable_summaries': self.active_outputs.observable_summaries,
             'status_codes': self.active_outputs.status_codes,
             'iteration_counters': self.active_outputs.iteration_counters,
         }
         
         # Transfer each active array individually with timing
         for array_name, is_active in output_map.items():
             if not is_active:
                 continue
             
             event_name = f'd2h_transfer_{array_name}'
             device_obj = self.device.get_managed_array(array_name)
             host_obj = self.host.get_managed_array(array_name)
             
             # Determine slice for chunked arrays
             from_ = device_obj.array
             stride_order = host_obj.stride_order
             slice_tuple = [slice(None)] * len(stride_order)
             
             if self._chunks > 1 and device_obj.is_chunked:
                 if self._chunk_axis in stride_order:
                     chunk_index = stride_order.index(self._chunk_axis)
                     slice_tuple[chunk_index] = host_indices
                     slice_tuple = tuple(slice_tuple)
                 to_ = host_obj.array[slice_tuple]
             else:
                 to_ = host_obj.array
             
             nbytes = from_.nbytes if hasattr(from_, 'nbytes') else 0
             
             _default_timelogger.start_event(
                 event_name,
                 array_name=array_name,
                 nbytes=nbytes,
                 chunk_slice=str(host_indices),
                 is_active=is_active
             )
             
             # Transfer single array
             self.from_device([from_], [to_])
             
             _default_timelogger.stop_event(
                 event_name,
                 array_name=array_name,
                 nbytes=nbytes
             )
     ```
   - Edge cases:
     - Only transfer active outputs (check active_outputs flags)
     - status_codes always active, so always transferred
     - Handle chunked slicing for run dimension
     - Summaries may be combined (state_summaries, observable_summaries)
   - Integration: Implements abstract method from BaseArrayManager; called from BatchSolverKernel.run()

**Outcomes**:

---

## Task Group 8: Testing - Event Registration - PARALLEL
**Status**: [ ]
**Dependencies**: Groups 1, 2, 3, 4

**Required Context**:
- File: tests/conftest.py (for fixture patterns)
- File: src/cubie/time_logger.py (for TimeLogger API)
- All modified source files from Groups 1-4

**Input Validation Required**:
None - Tests verify behavior, not user input

**Tasks**:
1. **Create test_runtime_logging_registration.py**
   - File: tests/batchsolving/test_runtime_logging_registration.py (create new)
   - Action: Create test file for event registration
   - Details:
     ```python
     """Test runtime logging event registration."""
     import pytest
     from cubie.time_logger import _default_timelogger
     from cubie.batchsolving.solver import Solver
     from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel
     
     
     def test_solve_ivp_event_registered():
         """Verify solve_ivp event is registered at module load."""
         assert 'solve_ivp_execution' in _default_timelogger._event_registry
         event = _default_timelogger._event_registry['solve_ivp_execution']
         assert event['category'] == 'runtime'
         assert 'solve_ivp' in event['description'].lower()
     
     
     def test_solver_events_registered(three_state_linear):
         """Verify Solver events are registered during __init__."""
         solver = Solver(three_state_linear, algorithm='euler')
         assert 'solver_solve_execution' in _default_timelogger._event_registry
         event = _default_timelogger._event_registry['solver_solve_execution']
         assert event['category'] == 'runtime'
     
     
     def test_kernel_events_registered(three_state_linear):
         """Verify BatchSolverKernel events are registered."""
         kernel = BatchSolverKernel(
             three_state_linear,
             loop_settings={},
             profileCUDA=False,
             step_control_settings={},
             algorithm_settings={'algorithm': 'euler'},
             output_settings={},
             memory_settings={}
         )
         
         # Check all kernel events
         expected_events = [
             'kernel_run_execution',
             'kernel_launch',
             'h2d_transfer_initial_values',
             'h2d_transfer_parameters',
             'h2d_transfer_driver_coefficients',
             'd2h_transfer_state',
             'd2h_transfer_observables',
             'd2h_transfer_summaries',
             'd2h_transfer_status_codes',
             'd2h_transfer_iteration_counters',
         ]
         
         for event_name in expected_events:
             assert event_name in _default_timelogger._event_registry
             event = _default_timelogger._event_registry[event_name]
             assert event['category'] == 'runtime'
     ```
   - Edge cases: Test multiple Solver instances don't re-register events (should be idempotent)
   - Integration: Use existing system fixtures (three_state_linear)

**Outcomes**:

---

## Task Group 9: Testing - Timing Data Collection - PARALLEL  
**Status**: [ ]
**Dependencies**: Groups 1-7

**Required Context**:
- File: tests/conftest.py (for fixture patterns)
- File: tests/system_fixtures.py (for system fixtures)
- All modified source files

**Input Validation Required**:
None - Tests verify behavior

**Tasks**:
1. **Create test_runtime_logging_data.py**
   - File: tests/batchsolving/test_runtime_logging_data.py (create new)
   - Action: Create test file for timing data collection
   - Details:
     ```python
     """Test runtime logging data collection."""
     import pytest
     import numpy as np
     from cubie import solve_ivp, Solver
     from cubie.time_logger import _default_timelogger
     
     
     @pytest.fixture
     def reset_timelogger():
         """Reset timelogger before each test."""
         _default_timelogger.events = []
         _default_timelogger._active_starts = {}
         yield
         # Cleanup after test
         _default_timelogger.events = []
         _default_timelogger._active_starts = {}
     
     
     def test_solve_ivp_timing_collected(three_state_linear, reset_timelogger):
         """Verify solve_ivp timing events are collected."""
         _default_timelogger.set_verbosity('default')
         
         y0 = np.array([[1.0, 0.0, 0.0]])
         result = solve_ivp(
             three_state_linear,
             y0,
             duration=0.1,
             dt_save=0.01
         )
         
         # Check events were recorded
         event_names = [e.name for e in _default_timelogger.events]
         assert 'solve_ivp_execution' in event_names
         
         # Verify start and stop pairs
         starts = [e for e in _default_timelogger.events 
                   if e.name == 'solve_ivp_execution' and e.event_type == 'start']
         stops = [e for e in _default_timelogger.events 
                  if e.name == 'solve_ivp_execution' and e.event_type == 'stop']
         assert len(starts) == 1
         assert len(stops) == 1
         assert stops[0].timestamp > starts[0].timestamp
     
     
     def test_solver_solve_timing_collected(three_state_linear, reset_timelogger):
         """Verify Solver.solve() timing events are collected."""
         _default_timelogger.set_verbosity('default')
         
         solver = Solver(three_state_linear, algorithm='euler')
         y0 = np.array([[1.0, 0.0, 0.0]])
         result = solver.solve(y0, duration=0.1, dt_save=0.01)
         
         event_names = [e.name for e in _default_timelogger.events]
         assert 'solver_solve_execution' in event_names
     
     
     @pytest.mark.nocudasim
     def test_kernel_launch_timing_collected(three_state_linear, reset_timelogger):
         """Verify kernel launch timing with CUDA events."""
         _default_timelogger.set_verbosity('default')
         
         solver = Solver(three_state_linear, algorithm='euler')
         y0 = np.array([[1.0, 0.0, 0.0]])
         result = solver.solve(y0, duration=0.1, dt_save=0.01)
         
         # Find kernel_launch events
         kernel_events = [e for e in _default_timelogger.events 
                          if e.name == 'kernel_launch']
         
         # Should have start and stop for each chunk
         assert len(kernel_events) >= 2  # At least one start and stop
         
         # Check for gpu_time_ms metadata in stop events
         stop_events = [e for e in kernel_events if e.event_type == 'stop']
         for event in stop_events:
             assert 'gpu_time_ms' in event.metadata
             assert event.metadata['gpu_time_ms'] > 0
     
     
     def test_memory_transfer_timing_collected(three_state_linear, reset_timelogger):
         """Verify memory transfer timing events are collected."""
         _default_timelogger.set_verbosity('default')
         
         solver = Solver(three_state_linear, algorithm='euler')
         y0 = np.array([[1.0, 0.0, 0.0]])
         result = solver.solve(y0, duration=0.1, dt_save=0.01)
         
         event_names = [e.name for e in _default_timelogger.events]
         
         # Check for h2d transfer events
         assert 'h2d_transfer_initial_values' in event_names
         assert 'h2d_transfer_parameters' in event_names
         
         # Check for d2h transfer events  
         # Status codes should always be present
         assert 'd2h_transfer_status_codes' in event_names
         
         # Check metadata contains nbytes
         h2d_events = [e for e in _default_timelogger.events 
                       if e.name.startswith('h2d_transfer') and e.event_type == 'stop']
         for event in h2d_events:
             assert 'nbytes' in event.metadata
     ```
   - Edge cases:
     - CUDASIM mode has different behavior (skip gpu_time_ms check)
     - Multi-chunk execution repeats kernel_launch events
     - Optional arrays (drivers, observables) may not be transferred
   - Integration: Use existing fixtures and markers (nocudasim)

**Outcomes**:

---

## Task Group 10: Testing - Verbosity Levels - PARALLEL
**Status**: [ ]
**Dependencies**: Groups 1-7

**Required Context**:
- File: tests/conftest.py
- File: src/cubie/time_logger.py (lines 62-98 for verbosity handling)

**Input Validation Required**:
None - Tests verify behavior

**Tasks**:
1. **Create test_runtime_logging_verbosity.py**
   - File: tests/batchsolving/test_runtime_logging_verbosity.py (create new)
   - Action: Test verbosity level behavior
   - Details:
     ```python
     """Test runtime logging verbosity levels."""
     import pytest
     import numpy as np
     from cubie import solve_ivp
     from cubie.time_logger import _default_timelogger
     
     
     @pytest.fixture
     def reset_timelogger():
         """Reset timelogger before each test."""
         original_verbosity = _default_timelogger.verbosity
         _default_timelogger.events = []
         _default_timelogger._active_starts = {}
         yield
         _default_timelogger.set_verbosity(original_verbosity)
         _default_timelogger.events = []
         _default_timelogger._active_starts = {}
     
     
     def test_verbosity_none_no_events(three_state_linear, reset_timelogger):
         """Verify no events collected when verbosity=None."""
         _default_timelogger.set_verbosity(None)
         
         y0 = np.array([[1.0, 0.0, 0.0]])
         result = solve_ivp(
             three_state_linear,
             y0,
             duration=0.1,
             dt_save=0.01,
             time_logging_level=None
         )
         
         # No events should be recorded
         assert len(_default_timelogger.events) == 0
     
     
     def test_verbosity_default_events_collected(three_state_linear, reset_timelogger):
         """Verify events collected when verbosity='default'."""
         y0 = np.array([[1.0, 0.0, 0.0]])
         result = solve_ivp(
             three_state_linear,
             y0,
             duration=0.1,
             dt_save=0.01,
             time_logging_level='default'
         )
         
         # Events should be recorded
         assert len(_default_timelogger.events) > 0
         event_names = [e.name for e in _default_timelogger.events]
         assert 'solve_ivp_execution' in event_names
     
     
     def test_verbosity_verbose_events_collected(three_state_linear, reset_timelogger):
         """Verify events collected when verbosity='verbose'."""
         y0 = np.array([[1.0, 0.0, 0.0]])
         result = solve_ivp(
             three_state_linear,
             y0,
             duration=0.1,
             dt_save=0.01,
             time_logging_level='verbose'
         )
         
         assert len(_default_timelogger.events) > 0
     
     
     def test_verbosity_debug_events_collected(three_state_linear, reset_timelogger):
         """Verify events collected when verbosity='debug'."""
         y0 = np.array([[1.0, 0.0, 0.0]])
         result = solve_ivp(
             three_state_linear,
             y0,
             duration=0.1,
             dt_save=0.01,
             time_logging_level='debug'
         )
         
         assert len(_default_timelogger.events) > 0
     ```
   - Edge cases: Test all verbosity levels (None, default, verbose, debug)
   - Integration: Use existing fixtures

**Outcomes**:

---

## Task Group 11: Testing - Summary Output - PARALLEL
**Status**: [ ]
**Dependencies**: Groups 1-7

**Required Context**:
- File: tests/conftest.py
- File: src/cubie/time_logger.py (lines 170-250 for print_summary)

**Input Validation Required**:
None - Tests verify behavior

**Tasks**:
1. **Create test_runtime_logging_summary.py**
   - File: tests/batchsolving/test_runtime_logging_summary.py (create new)
   - Action: Test summary output functionality
   - Details:
     ```python
     """Test runtime logging summary output."""
     import pytest
     import numpy as np
     from io import StringIO
     import sys
     from cubie import solve_ivp
     from cubie.time_logger import _default_timelogger
     
     
     @pytest.fixture
     def reset_timelogger():
         """Reset timelogger before each test."""
         _default_timelogger.events = []
         _default_timelogger._active_starts = {}
         yield
         _default_timelogger.events = []
         _default_timelogger._active_starts = {}
     
     
     def test_summary_contains_runtime_events(three_state_linear, reset_timelogger, capsys):
         """Verify print_summary includes runtime events."""
         _default_timelogger.set_verbosity('default')
         
         y0 = np.array([[1.0, 0.0, 0.0]])
         result = solve_ivp(
             three_state_linear,
             y0,
             duration=0.1,
             dt_save=0.01
         )
         
         # Capture printed output
         captured = capsys.readouterr()
         
         # Verify runtime category appears
         assert 'runtime' in captured.out.lower() or 'Runtime' in captured.out
     
     
     def test_summary_category_filter(three_state_linear, reset_timelogger):
         """Verify category filtering in print_summary."""
         _default_timelogger.set_verbosity('default')
         
         y0 = np.array([[1.0, 0.0, 0.0]])
         # This will collect runtime events and print summary
         result = solve_ivp(
             three_state_linear,
             y0,
             duration=0.1,
             dt_save=0.01
         )
         
         # Manually call print_summary with category filter
         import io
         from contextlib import redirect_stdout
         
         f = io.StringIO()
         with redirect_stdout(f):
             _default_timelogger.print_summary(category='runtime')
         output = f.getvalue()
         
         # Should contain runtime category
         assert len(output) > 0
     ```
   - Edge cases:
     - Verify summary only prints once per solve_ivp call
     - Check that nested calls (solve_ivp -> solve) don't duplicate output
   - Integration: Use capsys fixture for capturing print output

**Outcomes**:

---

## Task Group 12: Testing - CUDASIM Fallback - PARALLEL
**Status**: [ ]
**Dependencies**: Groups 1-7

**Required Context**:
- File: tests/conftest.py (for sim_only marker)
- File: src/cubie/cuda_simsafe.py (for is_cudasim_enabled)

**Input Validation Required**:
None - Tests verify behavior

**Tasks**:
1. **Create test_runtime_logging_cudasim.py**
   - File: tests/batchsolving/test_runtime_logging_cudasim.py (create new)
   - Action: Test CUDASIM fallback behavior
   - Details:
     ```python
     """Test runtime logging CUDASIM fallback behavior."""
     import pytest
     import numpy as np
     from cubie import solve_ivp
     from cubie.time_logger import _default_timelogger
     from cubie.cuda_simsafe import is_cudasim_enabled
     
     
     @pytest.fixture
     def reset_timelogger():
         """Reset timelogger before each test."""
         _default_timelogger.events = []
         _default_timelogger._active_starts = {}
         yield
         _default_timelogger.events = []
         _default_timelogger._active_starts = {}
     
     
     @pytest.mark.sim_only
     def test_cudasim_timing_uses_wallclock(three_state_linear, reset_timelogger):
         """Verify CUDASIM mode uses wall-clock timing."""
         if not is_cudasim_enabled():
             pytest.skip("This test requires CUDASIM mode")
         
         _default_timelogger.set_verbosity('default')
         
         y0 = np.array([[1.0, 0.0, 0.0]])
         result = solve_ivp(
             three_state_linear,
             y0,
             duration=0.1,
             dt_save=0.01
         )
         
         # Find kernel_launch events
         kernel_events = [e for e in _default_timelogger.events 
                          if e.name == 'kernel_launch']
         
         # Should still have timing events
         assert len(kernel_events) > 0
         
         # But should NOT have gpu_time_ms metadata (uses wall-clock instead)
         stop_events = [e for e in kernel_events if e.event_type == 'stop']
         for event in stop_events:
             # In CUDASIM, gpu_time_ms should not be present
             assert 'gpu_time_ms' not in event.metadata
     
     
     @pytest.mark.sim_only
     def test_cudasim_all_events_collected(three_state_linear, reset_timelogger):
         """Verify all timing events work in CUDASIM mode."""
         if not is_cudasim_enabled():
             pytest.skip("This test requires CUDASIM mode")
         
         _default_timelogger.set_verbosity('default')
         
         y0 = np.array([[1.0, 0.0, 0.0]])
         result = solve_ivp(
             three_state_linear,
             y0,
             duration=0.1,
             dt_save=0.01
         )
         
         # All major events should be present
         event_names = [e.name for e in _default_timelogger.events]
         assert 'solve_ivp_execution' in event_names
         assert 'solver_solve_execution' in event_names
         assert 'kernel_run_execution' in event_names
         assert 'kernel_launch' in event_names
     ```
   - Edge cases:
     - Skip tests if not in CUDASIM mode
     - Verify fallback to wall-clock timing works correctly
   - Integration: Use sim_only marker for CUDASIM-specific tests

**Outcomes**:

---

## Task Group 13: Testing - Multi-Chunk Execution - PARALLEL
**Status**: [ ]
**Dependencies**: Groups 1-7

**Required Context**:
- File: tests/conftest.py
- File: src/cubie/batchsolving/BatchSolverKernel.py (lines 418-470 for chunking logic)

**Input Validation Required**:
None - Tests verify behavior

**Tasks**:
1. **Create test_runtime_logging_chunked.py**
   - File: tests/batchsolving/test_runtime_logging_chunked.py (create new)
   - Action: Test timing in multi-chunk scenarios
   - Details:
     ```python
     """Test runtime logging with chunked execution."""
     import pytest
     import numpy as np
     from cubie import Solver
     from cubie.time_logger import _default_timelogger
     
     
     @pytest.fixture
     def reset_timelogger():
         """Reset timelogger before each test."""
         _default_timelogger.events = []
         _default_timelogger._active_starts = {}
         yield
         _default_timelogger.events = []
         _default_timelogger._active_starts = {}
     
     
     def test_chunked_kernel_timing_repeated(three_state_linear, reset_timelogger):
         """Verify kernel timing is recorded for each chunk."""
         _default_timelogger.set_verbosity('default')
         
         # Create large batch to force chunking
         solver = Solver(
             three_state_linear,
             algorithm='euler',
             mem_proportion=0.1  # Small proportion to force chunking
         )
         
         # Large batch
         y0 = np.random.rand(1000, 3).astype(np.float32)
         result = solver.solve(y0, duration=0.1, dt_save=0.01)
         
         # Count kernel_launch events
         kernel_events = [e for e in _default_timelogger.events 
                          if e.name == 'kernel_launch']
         
         # Should have multiple start/stop pairs (one per chunk)
         starts = [e for e in kernel_events if e.event_type == 'start']
         stops = [e for e in kernel_events if e.event_type == 'stop']
         
         # Number of chunks depends on memory constraints
         # Just verify we have at least one pair
         assert len(starts) >= 1
         assert len(stops) >= 1
         assert len(starts) == len(stops)
         
         # Verify chunk_index in metadata increases
         stop_indices = [e.metadata.get('chunk_index', 0) for e in stops]
         assert stop_indices == sorted(stop_indices)
     
     
     def test_chunked_transfer_timing_repeated(three_state_linear, reset_timelogger):
         """Verify transfer timing is recorded for each chunk."""
         _default_timelogger.set_verbosity('default')
         
         solver = Solver(
             three_state_linear,
             algorithm='euler',
             mem_proportion=0.1
         )
         
         y0 = np.random.rand(1000, 3).astype(np.float32)
         result = solver.solve(y0, duration=0.1, dt_save=0.01)
         
         # Count h2d transfer events
         h2d_events = [e for e in _default_timelogger.events 
                       if e.name.startswith('h2d_transfer')]
         
         # Should have events for each chunk
         assert len(h2d_events) > 0
     ```
   - Edge cases:
     - Chunking depends on memory constraints (may not always chunk)
     - Verify chunk_index metadata is sequential
   - Integration: Use large batch sizes to force chunking

**Outcomes**:

---

## Summary

**Total Task Groups**: 13
**Dependency Chain**:
- Groups 2, 3, 4, 5, 6, 7 depend on Group 1 (event registration)
- Groups 8, 9, 10, 11, 12, 13 (tests) depend on Groups 1-7 (implementation)
- Groups 8-13 can execute in parallel once implementation is complete

**Parallel Execution Opportunities**:
- Groups 2, 3, 4 can be developed in parallel (different files, minimal overlap)
- Groups 5, 6, 7 can be developed in parallel (different files)
- Groups 8-13 (all tests) can be developed in parallel

**Estimated Complexity**: 
- Medium-High
- Core changes are straightforward (add timing calls)
- CUDA event API requires careful handling for CUDASIM fallback
- Array transfer timing requires understanding of chunking logic
- Comprehensive test coverage needed for all scenarios

**Key Integration Points**:
- TimeLogger API (start_event, stop_event, register_event, print_summary)
- CUDA event API (cuda.event(), event.record(), cuda.event_elapsed_time())
- is_cudasim_enabled() check for fallback behavior
- Array manager transfer methods (to_device, from_device)
- Chunking logic in BatchSolverKernel.run()

**Files Modified**: 4 source files, 6 new test files
**Files Created**: 6 test files
**No files deleted**
