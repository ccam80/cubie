# Implementation Task List
# Feature: Time Logging Infrastructure (Phase 1)
# Plan Reference: .github/active_plans/time_logging/agent_plan.md

## Overview

This task list implements Phase 1 of the time logging infrastructure: creating the TimeLogger module and threading it through all CUDAFactory subclasses. No actual timing calls are implemented in Phase 1 - only the infrastructure for future phases.

**Key Principle**: All `time_logger` parameters default to `None`, making this a non-breaking change. Existing code continues to work without modification.

---

## Task Group 1: Core TimeLogger Module - SEQUENTIAL
**Status**: [x]
**Dependencies**: None

**Required Context**:
- Review .github/active_plans/time_logging/agent_plan.md for complete specifications
- No existing files needed (creating new module)

**Input Validation Required**:
- verbosity: Must be one of {'default', 'verbose', 'debug'}
- event_type: Must be one of {'start', 'stop', 'progress'}
- event_name: Must be non-empty string

**Tasks**:

1. **Create TimingEvent data class**
   - File: src/cubie/time_logger.py
   - Action: Create
   - Details:
     ```python
     import time
     from typing import Optional, Dict, Any
     import attrs
     
     @attrs.define(frozen=True)
     class TimingEvent:
         """Record of a single timing event.
         
         Attributes
         ----------
         name : str
             Identifier for the event (e.g., 'dxdt_compilation')
         event_type : str
             Type of event: 'start', 'stop', or 'progress'
         timestamp : float
             Wall-clock time from time.perf_counter()
         metadata : dict
             Optional metadata (file names, sizes, counts, etc.)
         """
         name: str = attrs.field(validator=attrs.validators.instance_of(str))
         event_type: str = attrs.field(
             validator=attrs.validators.in_({'start', 'stop', 'progress'})
         )
         timestamp: float = attrs.field(validator=attrs.validators.instance_of(float))
         metadata: dict = attrs.field(factory=dict)
     ```
   - Edge cases: Frozen class prevents accidental modification
   - Integration: Used internally by TimeLogger class

2. **Create TimeLogger class with initialization**
   - File: src/cubie/time_logger.py
   - Action: Modify (continue in same file)
   - Details:
     ```python
     class TimeLogger:
         """Callback-based timing system for CuBIE operations.
         
         Parameters
         ----------
         verbosity : str, default='default'
             Output verbosity level. Options:
             - 'default': Aggregate times only
             - 'verbose': Component-level breakdown
             - 'debug': All events with start/stop/progress
         
         Attributes
         ----------
         verbosity : str
             Current verbosity level
         events : list[TimingEvent]
             Chronological list of all recorded events
         _active_starts : dict[str, float]
             Map of event names to their start timestamps (for matching)
         
         Notes
         -----
         Create one instance per Solver. Pass to factories via __init__.
         """
         
         def __init__(self, verbosity: str = 'default') -> None:
             if verbosity not in {'default', 'verbose', 'debug'}:
                 raise ValueError(
                     f"verbosity must be 'default', 'verbose', or 'debug', "
                     f"got '{verbosity}'"
                 )
             self.verbosity = verbosity
             self.events: list[TimingEvent] = []
             self._active_starts: dict[str, float] = {}
     ```
   - Edge cases: Validate verbosity on init, not on every call
   - Integration: Instantiated by Solver class

3. **Implement callback methods**
   - File: src/cubie/time_logger.py
   - Action: Modify (add to TimeLogger class)
   - Details:
     ```python
     def start_event(self, event_name: str, **metadata: Any) -> None:
         """Record the start of a timed operation.
         
         Parameters
         ----------
         event_name : str
             Unique identifier for this event
         **metadata : Any
             Optional metadata to store with event
         
         Notes
         -----
         If event_name already has an active start, logs warning in debug mode
         and treats as nested event (matches most recent).
         """
         if not event_name:
             raise ValueError("event_name cannot be empty")
         
         timestamp = time.perf_counter()
         event = TimingEvent(
             name=event_name,
             event_type='start',
             timestamp=timestamp,
             metadata=metadata
         )
         self.events.append(event)
         self._active_starts[event_name] = timestamp
         
         if self.verbosity == 'debug':
             print(f"[DEBUG] Started: {event_name}")
     
     def stop_event(self, event_name: str, **metadata: Any) -> None:
         """Record the end of a timed operation.
         
         Parameters
         ----------
         event_name : str
             Identifier matching a previous start_event call
         **metadata : Any
             Optional metadata to store with event
         
         Notes
         -----
         If no matching start event exists, logs warning in debug mode
         and stores orphaned stop event for diagnostics.
         """
         if not event_name:
             raise ValueError("event_name cannot be empty")
         
         timestamp = time.perf_counter()
         event = TimingEvent(
             name=event_name,
             event_type='stop',
             timestamp=timestamp,
             metadata=metadata
         )
         self.events.append(event)
         
         # Calculate and maybe print duration
         if event_name in self._active_starts:
             duration = timestamp - self._active_starts[event_name]
             del self._active_starts[event_name]
             
             if self.verbosity == 'debug':
                 print(f"[DEBUG] Stopped: {event_name} ({duration:.3f}s)")
             elif self.verbosity == 'verbose':
                 print(f"{event_name}: {duration:.3f}s")
         else:
             if self.verbosity == 'debug':
                 print(f"[DEBUG] Warning: stop_event('{event_name}') "
                       "without matching start")
     
     def progress(self, event_name: str, message: str, **metadata: Any) -> None:
         """Record a progress update within an operation.
         
         Parameters
         ----------
         event_name : str
             Identifier for the operation in progress
         message : str
             Progress message to log
         **metadata : Any
             Optional metadata to store with event
         
         Notes
         -----
         Progress events don't require matching start/stop.
         Only printed in debug mode.
         """
         if not event_name:
             raise ValueError("event_name cannot be empty")
         
         timestamp = time.perf_counter()
         metadata_with_msg = dict(metadata)
         metadata_with_msg['message'] = message
         event = TimingEvent(
             name=event_name,
             event_type='progress',
             timestamp=timestamp,
             metadata=metadata_with_msg
         )
         self.events.append(event)
         
         if self.verbosity == 'debug':
             print(f"[DEBUG] Progress: {event_name} - {message}")
     ```
   - Edge cases: 
     - Orphaned stops (no matching start)
     - Multiple starts for same name (nested events)
     - Empty event names
   - Integration: Called by CUDAFactory subclasses during build operations

4. **Implement summary and query methods**
   - File: src/cubie/time_logger.py
   - Action: Modify (add to TimeLogger class)
   - Details:
     ```python
     def get_event_duration(self, event_name: str) -> Optional[float]:
         """Query duration of a completed event.
         
         Parameters
         ----------
         event_name : str
             Name of event to query
         
         Returns
         -------
         float or None
             Duration in seconds, or None if no matching start/stop pair
         
         Notes
         -----
         Returns duration of most recent completed event with this name.
         """
         start_time = None
         stop_time = None
         
         # Search backwards for most recent pair
         for event in reversed(self.events):
             if event.name == event_name:
                 if event.event_type == 'stop' and stop_time is None:
                     stop_time = event.timestamp
                 elif event.event_type == 'start' and stop_time is not None:
                     start_time = event.timestamp
                     break
         
         if start_time is not None and stop_time is not None:
             return stop_time - start_time
         return None
     
     def get_aggregate_durations(self, category: Optional[str] = None) -> dict[str, float]:
         """Aggregate event durations by category or all events.
         
         Parameters
         ----------
         category : str, optional
             If provided, filter events by metadata['category']
         
         Returns
         -------
         dict[str, float]
             Mapping of event names to total durations
         
         Notes
         -----
         Sums all durations for events with the same name.
         Future phases will use metadata['category'] for grouping.
         """
         durations: dict[str, float] = {}
         event_starts: dict[str, float] = {}
         
         for event in self.events:
             # Filter by category if requested
             if category is not None:
                 if event.metadata.get('category') != category:
                     continue
             
             if event.event_type == 'start':
                 event_starts[event.name] = event.timestamp
             elif event.event_type == 'stop':
                 if event.name in event_starts:
                     duration = event.timestamp - event_starts[event.name]
                     durations[event.name] = durations.get(event.name, 0.0) + duration
                     del event_starts[event.name]
         
         return durations
     
     def print_summary(self) -> None:
         """Print timing summary based on verbosity level.
         
         Notes
         -----
         - default: Prints aggregate durations for major categories
         - verbose: Already printed during stop_event calls
         - debug: Already printed all events as they occurred
         
         Only performs new printing in 'default' mode.
         """
         if self.verbosity == 'default':
             durations = self.get_aggregate_durations()
             if durations:
                 print("\nTiming Summary:")
                 for name, duration in sorted(durations.items()):
                     print(f"  {name}: {duration:.3f}s")
         # verbose and debug already printed inline
     ```
   - Edge cases:
     - No events recorded (empty summary)
     - Mismatched start/stop pairs (skipped in aggregation)
     - Multiple events with same name (summed)
   - Integration: Called at end of solve operations

**Outcomes**: 
- Files Modified: 
  * src/cubie/time_logger.py (293 lines created)
- Functions/Methods Added:
  * TimingEvent data class with attrs (frozen)
  * TimeLogger.__init__() with verbosity validation
  * TimeLogger.start_event() with timestamp recording
  * TimeLogger.stop_event() with duration calculation
  * TimeLogger.progress() for progress tracking
  * TimeLogger.get_event_duration() for single event queries
  * TimeLogger.get_aggregate_durations() for aggregation
  * TimeLogger.print_summary() for output formatting
- Implementation Summary:
  Complete time_logger module with TimeLogger class and TimingEvent
  data structure. All callback methods implemented with proper
  verbosity handling (default/verbose/debug). No-op pattern will be
  implemented in CUDAFactory base class for None handling.
- Issues Flagged: None

---

## Task Group 2: Base CUDAFactory Infrastructure - SEQUENTIAL
**Status**: [x]
**Dependencies**: Task Group 1 (requires TimeLogger module)

**Required Context**:
- File: src/cubie/CUDAFactory.py (entire file)
- Note: CUDAFactory.__init__ currently has no parameters

**Input Validation Required**:
- time_logger: Must be TimeLogger instance or None (no validation needed - duck typing)

**Tasks**:

1. **Add time_logger parameter to CUDAFactory.__init__**
   - File: src/cubie/CUDAFactory.py
   - Action: Modify
   - Details:
     ```python
     # Current __init__ signature (line 67):
     def __init__(self):
     
     # Change to:
     def __init__(self, time_logger=None):
         """Initialize the CUDA factory.
         
         Parameters
         ----------
         time_logger : TimeLogger, optional
             Optional time logger for tracking compilation performance.
             When None, no timing overhead is incurred.
         """
         self._compile_settings = None
         self._cache_valid = True
         self._device_function = None
         self._cache = None
         
         # Extract callbacks or create no-ops
         if time_logger is not None:
             self._timing_start = time_logger.start_event
             self._timing_stop = time_logger.stop_event
             self._timing_progress = time_logger.progress
         else:
             # No-op callbacks with zero overhead
             self._timing_start = lambda *args, **kwargs: None
             self._timing_stop = lambda *args, **kwargs: None
             self._timing_progress = lambda *args, **kwargs: None
     ```
   - Edge cases: time_logger=None is most common case
   - Integration: All subclasses must call super().__init__(time_logger=time_logger)

**Outcomes**: 
- Files Modified:
  * src/cubie/CUDAFactory.py (23 lines modified in __init__)
- Functions/Methods Modified:
  * CUDAFactory.__init__() - added time_logger parameter with no-op
    callbacks
- Implementation Summary:
  Modified CUDAFactory base class to accept optional time_logger
  parameter. When time_logger is None, no-op lambda functions are
  created for _timing_start, _timing_stop, and _timing_progress,
  ensuring zero overhead. When time_logger is provided, callbacks
  are extracted from the TimeLogger instance.
- Issues Flagged: None

---

## Task Group 3: ODE System Factories - SEQUENTIAL
**Status**: [x]
**Dependencies**: Task Group 2

**Required Context**:
- File: src/cubie/odesystems/baseODE.py (lines 68-124)
- File: src/cubie/odesystems/symbolic/symbolicODE.py (lines 100-250)

**Input Validation Required**:
- None (time_logger validated by duck typing)

**Tasks**:

1. **Add time_logger to BaseODE.__init__**
   - File: src/cubie/odesystems/baseODE.py
   - Action: Modify
   - Details:
     ```python
     # Current signature starts at line 68:
     def __init__(
         self,
         initial_values: Optional[Dict[str, float]] = None,
         parameters: Optional[Dict[str, float]] = None,
         constants: Optional[Dict[str, float]] = None,
         observables: Optional[Dict[str, float]] = None,
         default_initial_values: Optional[Dict[str, float]] = None,
         default_parameters: Optional[Dict[str, float]] = None,
         default_constants: Optional[Dict[str, float]] = None,
         default_observable_names: Optional[Dict[str, float]] = None,
         precision: PrecisionDType = np.float64,
         num_drivers: int = 1,
         name: Optional[str] = None,
     ) -> None:
     
     # Add parameter after name:
     def __init__(
         self,
         initial_values: Optional[Dict[str, float]] = None,
         parameters: Optional[Dict[str, float]] = None,
         constants: Optional[Dict[str, float]] = None,
         observables: Optional[Dict[str, float]] = None,
         default_initial_values: Optional[Dict[str, float]] = None,
         default_parameters: Optional[Dict[str, float]] = None,
         default_constants: Optional[Dict[str, float]] = None,
         default_observable_names: Optional[Dict[str, float]] = None,
         precision: PrecisionDType = np.float64,
         num_drivers: int = 1,
         name: Optional[str] = None,
         time_logger = None,
     ) -> None:
     
     # Update docstring (after line 108 "name" parameter):
         time_logger
             Optional TimeLogger instance for tracking compilation timing.
             Defaults to None (no timing overhead).
     
     # Update super().__init__() call at line 110:
         super().__init__(time_logger=time_logger)
     ```
   - Edge cases: All existing BaseODE instantiations use positional or keyword args
   - Integration: SymbolicODE and other subclasses must pass through

2. **Add time_logger to SymbolicODE.create class method**
   - File: src/cubie/odesystems/symbolic/symbolicODE.py
   - Action: Modify
   - Details:
     ```python
     # Find the @classmethod create() method (look for signature with dxdt, states, etc.)
     # Add time_logger parameter to signature (after precision parameter)
     # Pass it through to SymbolicODE.__init__ call
     
     # Example modification pattern:
     @classmethod
     def create(
         cls,
         dxdt: Union[str, Iterable[str]],
         states: Optional[Union[dict[str, float], Iterable[str]]] = None,
         observables: Optional[Iterable[str]] = None,
         parameters: Optional[Union[dict[str, float], Iterable[str]]] = None,
         constants: Optional[Union[dict[str, float], Iterable[str]]] = None,
         drivers: Optional[Union[Iterable[str], dict[str, Any]]] = None,
         user_functions: Optional[dict[str, Callable]] = None,
         name: Optional[str] = None,
         precision: PrecisionDType = np.float32,
         strict: bool = False,
         time_logger = None,  # ADD THIS
     ) -> "SymbolicODE":
         # ... existing parsing code ...
         
         # In the return statement, add time_logger=time_logger
     ```
   - Edge cases: Ensure all return paths include time_logger
   - Integration: Called by create_ODE_system() function

3. **Add time_logger to SymbolicODE.__init__**
   - File: src/cubie/odesystems/symbolic/symbolicODE.py
   - Action: Modify
   - Details:
     ```python
     # Find __init__ method signature
     # Add time_logger parameter at end
     # Pass to super().__init__()
     
     # Pattern:
     def __init__(
         self,
         # ... existing parameters ...
         time_logger = None,
     ) -> None:
         # Call parent __init__ with time_logger
         super().__init__(
             # ... existing arguments ...
             time_logger=time_logger,
         )
     ```
   - Edge cases: Must preserve all existing parameter handling
   - Integration: BaseODE.__init__ receives time_logger

4. **Add time_logger to create_ODE_system function**
   - File: src/cubie/odesystems/symbolic/symbolicODE.py
   - Action: Modify
   - Details:
     ```python
     # Current signature at line 49:
     def create_ODE_system(
         dxdt: Union[str, Iterable[str]],
         states: Optional[Union[dict[str, float], Iterable[str]]] = None,
         observables: Optional[Iterable[str]] = None,
         parameters: Optional[Union[dict[str, float], Iterable[str]]] = None,
         constants: Optional[Union[dict[str, float], Iterable[str]]] = None,
         drivers: Optional[Union[Iterable[str], dict[str, Any]]] = None,
         user_functions: Optional[dict[str, Callable]] = None,
         name: Optional[str] = None,
         precision: PrecisionDType = np.float32,
         strict: bool = False,
     ) -> "SymbolicODE":
     
     # Add time_logger parameter:
     def create_ODE_system(
         dxdt: Union[str, Iterable[str]],
         states: Optional[Union[dict[str, float], Iterable[str]]] = None,
         observables: Optional[Iterable[str]] = None,
         parameters: Optional[Union[dict[str, float], Iterable[str]]] = None,
         constants: Optional[Union[dict[str, float], Iterable[str]]] = None,
         drivers: Optional[Union[Iterable[str], dict[str, Any]]] = None,
         user_functions: Optional[dict[str, Callable]] = None,
         name: Optional[str] = None,
         precision: PrecisionDType = np.float32,
         strict: bool = False,
         time_logger = None,
     ) -> "SymbolicODE":
     
     # Update docstring (add after strict parameter documentation):
         time_logger
             Optional TimeLogger instance for tracking compilation timing.
             Defaults to None.
     
     # Pass to SymbolicODE.create() call (around line 99):
         symbolic_ode = SymbolicODE.create(
             dxdt=dxdt,
             # ... other parameters ...
             time_logger=time_logger,
         )
     ```
   - Edge cases: Public API function - must maintain backward compatibility
   - Integration: Entry point for creating symbolic systems

**Outcomes**: 
- Files Modified:
  * src/cubie/odesystems/baseODE.py (3 lines modified)
  * src/cubie/odesystems/symbolic/symbolicODE.py (6 lines modified)
- Functions/Methods Modified:
  * BaseODE.__init__() - added time_logger parameter and docstring
  * SymbolicODE.__init__() - added time_logger parameter and pass to
    super()
  * SymbolicODE.create() - added time_logger parameter and pass to
    __init__()
  * create_ODE_system() - added time_logger parameter and pass to
    SymbolicODE.create()
- Implementation Summary:
  Added time_logger parameter to all ODE system factory classes and
  functions. Parameters default to None for backward compatibility.
  All changes properly thread time_logger from public API through
  inheritance chain to CUDAFactory base class.
- Issues Flagged: None

---

## Task Group 4: Integration Loop Factories - SEQUENTIAL
**Status**: [x]
**Dependencies**: Task Group 2

**Required Context**:
- File: src/cubie/integrators/loops/ode_loop.py (lines 80-100)
- File: src/cubie/integrators/SingleIntegratorRunCore.py (lines 76-99)

**Input Validation Required**:
- None (time_logger validated by duck typing)

**Tasks**:

1. **Add time_logger to IVPLoop.__init__**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Details:
     ```python
     # Current signature starts at line 80:
     def __init__(
         self,
         precision: PrecisionDType,
         shared_indices: LoopSharedIndices,
         local_indices: LoopLocalIndices,
         compile_flags: OutputCompileFlags,
         dt_save: float = 0.1,
         dt_summarise: float = 1.0,
         dt0: Optional[float]=None,
         dt_min: Optional[float]=None,
         dt_max: Optional[float]=None,
         is_adaptive: Optional[bool]=None,
         save_state_func: Optional[Callable] = None,
         update_summaries_func: Optional[Callable] = None,
         save_summaries_func: Optional[Callable] = None,
         step_controller_fn: Optional[Callable] = None,
         step_function: Optional[Callable] = None,
         driver_function: Optional[Callable] = None,
         observables_fn: Optional[Callable] = None,
     ) -> None:
     
     # Add time_logger parameter at end:
     def __init__(
         self,
         precision: PrecisionDType,
         shared_indices: LoopSharedIndices,
         local_indices: LoopLocalIndices,
         compile_flags: OutputCompileFlags,
         dt_save: float = 0.1,
         dt_summarise: float = 1.0,
         dt0: Optional[float]=None,
         dt_min: Optional[float]=None,
         dt_max: Optional[float]=None,
         is_adaptive: Optional[bool]=None,
         save_state_func: Optional[Callable] = None,
         update_summaries_func: Optional[Callable] = None,
         save_summaries_func: Optional[Callable] = None,
         step_controller_fn: Optional[Callable] = None,
         step_function: Optional[Callable] = None,
         driver_function: Optional[Callable] = None,
         observables_fn: Optional[Callable] = None,
         time_logger = None,
     ) -> None:
     
     # Update docstring (add parameter documentation):
         time_logger
             Optional TimeLogger instance for tracking compilation timing.
     
     # Update super().__init__() call at line 100:
         super().__init__(time_logger=time_logger)
     ```
   - Edge cases: Many function parameters - easy to miss in call sites
   - Integration: Created by SingleIntegratorRunCore

2. **Add time_logger to SingleIntegratorRunCore.__init__**
   - File: src/cubie/integrators/SingleIntegratorRunCore.py
   - Action: Modify
   - Details:
     ```python
     # Current signature starts at line 76:
     def __init__(
         self,
         system: "BaseODE",
         loop_settings: Optional[Dict[str, Any]] = None,
         output_settings: Optional[Dict[str, Any]] = None,
         driver_function: Optional[Callable] = None,
         driver_del_t: Optional[Callable] = None,
         algorithm_settings: Optional[Dict[str, Any]] = None,
         step_control_settings: Optional[Dict[str, Any]] = None,
     ) -> None:
     
     # Add time_logger parameter at end:
     def __init__(
         self,
         system: "BaseODE",
         loop_settings: Optional[Dict[str, Any]] = None,
         output_settings: Optional[Dict[str, Any]] = None,
         driver_function: Optional[Callable] = None,
         driver_del_t: Optional[Callable] = None,
         algorithm_settings: Optional[Dict[str, Any]] = None,
         step_control_settings: Optional[Dict[str, Any]] = None,
         time_logger = None,
     ) -> None:
     
     # Update super().__init__() call at line 86:
         super().__init__(time_logger=time_logger)
     
     # Pass time_logger to all factory instantiations:
     # - OutputFunctions (search for "OutputFunctions(")
     # - get_algorithm_step call (search for "get_algorithm_step(")
     # - get_controller call (search for "get_controller(")
     # - IVPLoop (search for "IVPLoop(")
     
     # Pattern for each:
     output_functions = OutputFunctions(
         # ... existing arguments ...
         time_logger=time_logger,
     )
     
     algorithm_step = get_algorithm_step(
         # ... existing arguments ...
         time_logger=time_logger,
     )
     
     controller = get_controller(
         # ... existing arguments ...
         time_logger=time_logger,
     )
     
     loop = IVPLoop(
         # ... existing arguments ...
         time_logger=time_logger,
     )
     ```
   - Edge cases: Multiple factory instantiations to update
   - Integration: Coordinates all integration components

**Outcomes**: 
- Files Modified:
  * src/cubie/integrators/loops/ode_loop.py (54 lines modified)
  * src/cubie/integrators/SingleIntegratorRunCore.py (10 lines modified)
- Functions/Methods Modified:
  * IVPLoop.__init__() - added time_logger parameter with full
    docstring
  * SingleIntegratorRunCore.__init__() - added time_logger parameter
    and pass to OutputFunctions, get_algorithm_step, get_controller,
    instantiate_loop
  * SingleIntegratorRunCore.instantiate_loop() - added time_logger
    parameter and pass to IVPLoop
  * SingleIntegratorRunCore.check_compatibility() - updated
    get_controller call to use self._time_logger
- Implementation Summary:
  Added time_logger threading through integration loop infrastructure.
  SingleIntegratorRunCore stores time_logger as instance variable and
  passes it to all factory instantiations: OutputFunctions,
  get_algorithm_step, get_controller, and IVPLoop. Also updated
  compatibility check to use stored time_logger when creating fixed
  controller.
- Issues Flagged: None

---

## Task Group 5: Output Handling Factories - SEQUENTIAL
**Status**: [x]
**Dependencies**: Task Group 2

**Required Context**:
- File: src/cubie/outputhandling/output_functions.py (lines 100-133)

**Input Validation Required**:
- None (time_logger validated by duck typing)

**Tasks**:

1. **Add time_logger to OutputFunctions.__init__**
   - File: src/cubie/outputhandling/output_functions.py
   - Action: Modify
   - Details:
     ```python
     # Current signature starts at line 100:
     def __init__(
         self,
         max_states: int,
         max_observables: int,
         output_types: list[str] = None,
         saved_state_indices: Union[Sequence[int], ArrayLike] = None,
         saved_observable_indices: Union[Sequence[int], ArrayLike] = None,
         summarised_state_indices: Union[Sequence[int], ArrayLike] = None,
         summarised_observable_indices: Union[Sequence[int], ArrayLike] = None,
         dt_save: Optional[float] = None,
         precision: Optional[np.dtype] = None,
     ):
     
     # Add time_logger parameter:
     def __init__(
         self,
         max_states: int,
         max_observables: int,
         output_types: list[str] = None,
         saved_state_indices: Union[Sequence[int], ArrayLike] = None,
         saved_observable_indices: Union[Sequence[int], ArrayLike] = None,
         summarised_state_indices: Union[Sequence[int], ArrayLike] = None,
         summarised_observable_indices: Union[Sequence[int], ArrayLike] = None,
         dt_save: Optional[float] = None,
         precision: Optional[np.dtype] = None,
         time_logger = None,
     ):
     
     # Update super().__init__() call at line 112:
         super().__init__(time_logger=time_logger)
     
     # Add parameter to docstring:
         time_logger
             Optional TimeLogger instance for tracking compilation timing.
     ```
   - Edge cases: None - straightforward addition
   - Integration: Created by SingleIntegratorRunCore

**Outcomes**: 
- Files Modified:
  * src/cubie/outputhandling/output_functions.py (35 lines modified)
- Functions/Methods Modified:
  * OutputFunctions.__init__() - added time_logger parameter with
    full docstring
- Implementation Summary:
  Added time_logger parameter to OutputFunctions.__init__ with proper
  documentation. Parameter is passed to CUDAFactory base class via
  super().__init__(). All existing functionality preserved with
  backward compatibility through default None value.
- Issues Flagged: None

---

## Task Group 6: Algorithm Step Factories - PARALLEL
**Status**: [x]
**Dependencies**: Task Group 2

**Required Context**:
- File: src/cubie/integrators/algorithms/base_algorithm_step.py (lines 1-150)
- File: src/cubie/integrators/algorithms/explicit_euler.py (lines 25-68)
- File: src/cubie/integrators/algorithms/__init__.py (search for get_algorithm_step)
- File: src/cubie/integrators/algorithms/ode_explicitstep.py
- File: src/cubie/integrators/algorithms/ode_implicitstep.py
- Directory: src/cubie/integrators/algorithms/ (all algorithm files)

**Input Validation Required**:
- None (time_logger validated by duck typing)

**Tasks**:

1. **Add time_logger to ODEExplicitStep.__init__**
   - File: src/cubie/integrators/algorithms/ode_explicitstep.py
   - Action: Modify
   - Details:
     ```python
     # Find __init__ signature
     # Add time_logger parameter at end
     # Pass to super().__init__()
     
     def __init__(
         self,
         # ... existing parameters ...
         time_logger = None,
     ) -> None:
         super().__init__(time_logger=time_logger)
     ```
   - Edge cases: Base class for explicit algorithms
   - Integration: Parent of ExplicitEulerStep and generic ERK

2. **Add time_logger to ODEImplicitStep.__init__**
   - File: src/cubie/integrators/algorithms/ode_implicitstep.py
   - Action: Modify
   - Details:
     ```python
     # Find __init__ signature
     # Add time_logger parameter at end
     # Pass to super().__init__()
     
     def __init__(
         self,
         # ... existing parameters ...
         time_logger = None,
     ) -> None:
         super().__init__(time_logger=time_logger)
     ```
   - Edge cases: Base class for implicit algorithms
   - Integration: Parent of BackwardsEulerStep, CrankNicolson, etc.

3. **Add time_logger to ExplicitEulerStep.__init__**
   - File: src/cubie/integrators/algorithms/explicit_euler.py
   - Action: Modify
   - Details:
     ```python
     # Current signature at line 25:
     def __init__(
         self,
         precision: PrecisionDType,
         n: int,
         dt: Optional[float] = None,
         dxdt_function: Optional[Callable] = None,
         observables_function: Optional[Callable] = None,
         driver_function: Optional[Callable] = None,
         get_solver_helper_fn: Optional[Callable] = None,
     ) -> None:
     
     # Add time_logger:
     def __init__(
         self,
         precision: PrecisionDType,
         n: int,
         dt: Optional[float] = None,
         dxdt_function: Optional[Callable] = None,
         observables_function: Optional[Callable] = None,
         driver_function: Optional[Callable] = None,
         get_solver_helper_fn: Optional[Callable] = None,
         time_logger = None,
     ) -> None:
     
     # Update super().__init__() call (around line 68):
         super().__init__(config, EE_DEFAULTS.copy(), time_logger=time_logger)
     ```
   - Edge cases: Simple algorithm, easy to update
   - Integration: Most common algorithm for testing

4. **Add time_logger to all other algorithm classes**
   - Files: 
     - src/cubie/integrators/algorithms/backwards_euler.py
     - src/cubie/integrators/algorithms/backwards_euler_predict_correct.py
     - src/cubie/integrators/algorithms/crank_nicolson.py
     - src/cubie/integrators/algorithms/generic_erk.py
     - src/cubie/integrators/algorithms/generic_dirk.py
     - src/cubie/integrators/algorithms/generic_firk.py
     - src/cubie/integrators/algorithms/generic_rosenbrock_w.py
   - Action: Modify
   - Details:
     ```python
     # For each algorithm class:
     # 1. Add time_logger parameter to __init__ signature (at end)
     # 2. Add parameter documentation to docstring
     # 3. Pass to super().__init__() or parent class
     
     # Pattern (apply to each):
     def __init__(
         self,
         # ... existing parameters ...
         time_logger = None,
     ) -> None:
         # ... existing logic ...
         super().__init__(
             # ... existing arguments ...
             time_logger=time_logger,
         )
     ```
   - Edge cases: Each algorithm has different parameters
   - Integration: All created via get_algorithm_step factory

5. **Add time_logger to get_algorithm_step factory function**
   - File: src/cubie/integrators/algorithms/__init__.py
   - Action: Modify
   - Details:
     ```python
     # Find get_algorithm_step function definition
     # Add time_logger parameter with default None
     # Pass to all algorithm instantiations
     
     def get_algorithm_step(
         algorithm: str,
         # ... existing parameters ...
         time_logger = None,
     ) -> BaseAlgorithmStep:
         # In each branch (if/elif for each algorithm):
         if algorithm == 'euler':
             return ExplicitEulerStep(
                 # ... existing arguments ...
                 time_logger=time_logger,
             )
         elif algorithm == 'backwards_euler':
             return BackwardsEulerStep(
                 # ... existing arguments ...
                 time_logger=time_logger,
             )
         # ... repeat for all algorithms ...
     ```
   - Edge cases: Many algorithm branches to update
   - Integration: Called by SingleIntegratorRunCore

**Outcomes**: 
- Files Modified:
  * src/cubie/integrators/algorithms/base_algorithm_step.py (3 lines)
  * src/cubie/integrators/algorithms/explicit_euler.py (3 lines)
  * src/cubie/integrators/algorithms/backwards_euler.py (3 lines)
  * src/cubie/integrators/algorithms/crank_nicolson.py (3 lines)
  * src/cubie/integrators/algorithms/generic_erk.py (3 lines)
  * src/cubie/integrators/algorithms/generic_dirk.py (3 lines)
  * src/cubie/integrators/algorithms/generic_firk.py (4 lines)
  * src/cubie/integrators/algorithms/generic_rosenbrock_w.py (3 lines)
  * src/cubie/integrators/algorithms/__init__.py (5 lines in
    get_algorithm_step)
- Functions/Methods Modified:
  * BaseAlgorithmStep.__init__() - added time_logger parameter
  * ExplicitEulerStep.__init__() - added time_logger parameter
  * BackwardsEulerStep.__init__() - added time_logger parameter
  * CrankNicolsonStep.__init__() - added time_logger parameter
  * ERKStep.__init__() - added time_logger parameter
  * DIRKStep.__init__() - added time_logger parameter
  * FIRKStep.__init__() - added time_logger parameter
  * GenericRosenbrockWStep.__init__() - added time_logger parameter
  * get_algorithm_step() - added time_logger parameter and pass to all
    algorithm instantiations
- Implementation Summary:
  Added time_logger threading through all algorithm step factories.
  Updated base class (BaseAlgorithmStep) to accept time_logger and
  pass to CUDAFactory. Updated all algorithm subclasses to accept
  time_logger in __init__ and pass to super(). Updated factory
  function get_algorithm_step to accept time_logger and inject it into
  all algorithm instantiations via filtered dict. BackwardsEulerPCStep
  inherits from BackwardsEulerStep so automatically gets time_logger.
- Issues Flagged: None

---

## Task Group 7: Step Controller Factories - PARALLEL
**Status**: [x]
**Dependencies**: Task Group 2

**Required Context**:
- File: src/cubie/integrators/step_control/base_step_controller.py (lines 93-99)
- File: src/cubie/integrators/step_control/fixed_step_controller.py
- File: src/cubie/integrators/step_control/adaptive_I_controller.py
- File: src/cubie/integrators/step_control/adaptive_PI_controller.py
- File: src/cubie/integrators/step_control/adaptive_PID_controller.py
- File: src/cubie/integrators/step_control/__init__.py (get_controller function)

**Input Validation Required**:
- None (time_logger validated by duck typing)

**Tasks**:

1. **Add time_logger to BaseStepController.__init__**
   - File: src/cubie/integrators/step_control/base_step_controller.py
   - Action: Modify
   - Details:
     ```python
     # Current signature at line 96:
     def __init__(self) -> None:
         """Initialise the base controller factory."""
         super().__init__()
     
     # Change to:
     def __init__(self, time_logger = None) -> None:
         """Initialise the base controller factory.
         
         Parameters
         ----------
         time_logger
             Optional TimeLogger instance for tracking compilation timing.
         """
         super().__init__(time_logger=time_logger)
     ```
   - Edge cases: Abstract base class - all subclasses must update
   - Integration: Parent of all controller classes

2. **Add time_logger to all controller classes**
   - Files:
     - src/cubie/integrators/step_control/fixed_step_controller.py
     - src/cubie/integrators/step_control/adaptive_I_controller.py
     - src/cubie/integrators/step_control/adaptive_PI_controller.py
     - src/cubie/integrators/step_control/adaptive_PID_controller.py
   - Action: Modify
   - Details:
     ```python
     # For each controller class:
     # 1. Add time_logger parameter to __init__ (at end)
     # 2. Pass to super().__init__()
     
     def __init__(
         self,
         # ... existing parameters ...
         time_logger = None,
     ) -> None:
         # ... existing logic ...
         super().__init__(time_logger=time_logger)
     ```
   - Edge cases: Each controller has different configuration parameters
   - Integration: All created via get_controller factory

3. **Add time_logger to get_controller factory function**
   - File: src/cubie/integrators/step_control/__init__.py
   - Action: Modify
   - Details:
     ```python
     # Find get_controller function
     # Add time_logger parameter
     # Pass to all controller instantiations
     
     def get_controller(
         step_controller: str,
         # ... existing parameters ...
         time_logger = None,
     ) -> BaseStepController:
         # In each branch:
         if step_controller == 'fixed':
             return FixedStepController(
                 # ... existing arguments ...
                 time_logger=time_logger,
             )
         elif step_controller == 'i':
             return AdaptiveIController(
                 # ... existing arguments ...
                 time_logger=time_logger,
             )
         # ... repeat for all controllers ...
     ```
   - Edge cases: Multiple controller types to update
   - Integration: Called by SingleIntegratorRunCore

**Outcomes**: 
- Files Modified:
  * src/cubie/integrators/step_control/base_step_controller.py (8 lines)
  * src/cubie/integrators/step_control/fixed_step_controller.py (4 lines)
  * src/cubie/integrators/step_control/adaptive_step_controller.py (4
    lines)
  * src/cubie/integrators/step_control/adaptive_I_controller.py (3
    lines)
  * src/cubie/integrators/step_control/adaptive_PI_controller.py (3
    lines)
  * src/cubie/integrators/step_control/adaptive_PID_controller.py (3
    lines)
  * src/cubie/integrators/step_control/gustafsson_controller.py (3
    lines)
  * src/cubie/integrators/step_control/__init__.py (5 lines in
    get_controller)
- Functions/Methods Modified:
  * BaseStepController.__init__() - added time_logger parameter
  * FixedStepController.__init__() - added time_logger parameter
  * BaseAdaptiveStepController.__init__() - added time_logger parameter
  * AdaptiveIController.__init__() - added time_logger parameter
  * AdaptivePIController.__init__() - added time_logger parameter
  * AdaptivePIDController.__init__() - added time_logger parameter
  * GustafssonController.__init__() - added time_logger parameter
  * get_controller() - added time_logger parameter and inject into all
    controller instantiations
- Implementation Summary:
  Added time_logger threading through all step controller factories.
  Updated base classes (BaseStepController and
  BaseAdaptiveStepController) to accept time_logger and pass to
  CUDAFactory. Updated all controller subclasses to accept time_logger
  in __init__ and pass to super(). Updated factory function
  get_controller to accept time_logger and inject it into all
  controller instantiations via filtered dict.
- Issues Flagged: None

---

## Task Group 8: Batch Solver Infrastructure - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Groups 3, 4, 5, 6, 7 (all component factories updated)

**Required Context**:
- File: src/cubie/batchsolving/BatchSolverKernel.py (lines 67-150)
- File: src/cubie/batchsolving/solver.py (lines 105-223)
- File: src/cubie/integrators/SingleIntegratorRun.py

**Input Validation Required**:
- None (time_logger validated by duck typing)

**Tasks**:

1. **Add time_logger to BatchSolverKernel.__init__**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     ```python
     # Find __init__ signature (around line 100+)
     # Add time_logger parameter at end
     # Pass to super().__init__() and SingleIntegratorRun
     
     def __init__(
         self,
         system: BaseODE,
         loop_settings: Optional[Dict[str, Any]] = None,
         driver_function: Optional[Callable] = None,
         profileCUDA: bool = False,
         step_control_settings: Optional[Dict[str, Any]] = None,
         algorithm_settings: Optional[Dict[str, Any]] = None,
         output_settings: Optional[Dict[str, Any]] = None,
         memory_settings: Optional[Dict[str, Any]] = None,
         time_logger = None,
     ) -> None:
         super().__init__(time_logger=time_logger)
         
         # When creating SingleIntegratorRun (search for "SingleIntegratorRun("):
         self.integrator_run = SingleIntegratorRun(
             # ... existing arguments ...
             time_logger=time_logger,
         )
     ```
   - Edge cases: Must pass to SingleIntegratorRun
   - Integration: Created by Solver class

2. **Add time_logger to SingleIntegratorRun wrapper**
   - File: src/cubie/integrators/SingleIntegratorRun.py
   - Action: Modify
   - Details:
     ```python
     # Find __init__ signature
     # Add time_logger parameter
     # Pass to SingleIntegratorRunCore
     
     def __init__(
         self,
         # ... existing parameters ...
         time_logger = None,
     ) -> None:
         # When creating SingleIntegratorRunCore:
         self._core = SingleIntegratorRunCore(
             # ... existing arguments ...
             time_logger=time_logger,
         )
     ```
   - Edge cases: Wrapper around SingleIntegratorRunCore
   - Integration: Created by BatchSolverKernel

3. **Add verbosity parameter to Solver.__init__**
   - File: src/cubie/batchsolving/solver.py
   - Action: Modify
   - Details:
     ```python
     # Add import at top of file:
     from cubie.time_logger import TimeLogger
     
     # Current signature at line 144:
     def __init__(
         self,
         system: BaseODE,
         algorithm: str = "euler",
         profileCUDA: bool = False,
         step_control_settings: Optional[Dict[str, object]] = None,
         algorithm_settings: Optional[Dict[str, object]] = None,
         output_settings: Optional[Dict[str, object]] = None,
         memory_settings: Optional[Dict[str, object]] = None,
         loop_settings: Optional[Dict[str, object]] = None,
         strict: bool = False,
         **kwargs: Any,
     ) -> None:
     
     # Add verbosity parameter:
     def __init__(
         self,
         system: BaseODE,
         algorithm: str = "euler",
         profileCUDA: bool = False,
         step_control_settings: Optional[Dict[str, object]] = None,
         algorithm_settings: Optional[Dict[str, object]] = None,
         output_settings: Optional[Dict[str, object]] = None,
         memory_settings: Optional[Dict[str, object]] = None,
         loop_settings: Optional[Dict[str, object]] = None,
         strict: bool = False,
         verbosity: str = 'default',
         **kwargs: Any,
     ) -> None:
     
     # Add to docstring (after strict parameter):
         verbosity
             Timing output verbosity. Options: 'default' (aggregate times),
             'verbose' (component breakdown), 'debug' (all events).
             Defaults to 'default'.
     
     # Create TimeLogger after line 168 (after super().__init__()):
         # Create TimeLogger based on verbosity
         if verbosity in {'default', 'verbose', 'debug'}:
             self.time_logger = TimeLogger(verbosity=verbosity)
         else:
             # Invalid verbosity - use None (no timing)
             self.time_logger = None
     
     # Pass to BatchSolverKernel (around line 207):
         self.kernel = BatchSolverKernel(
             system,
             loop_settings=loop_settings,
             profileCUDA=profileCUDA,
             step_control_settings=step_settings,
             algorithm_settings=algorithm_settings,
             output_settings=output_settings,
             memory_settings=memory_settings,
             time_logger=self.time_logger,
         )
     ```
   - Edge cases: 
     - Invalid verbosity values (use None)
     - Default verbosity='default' for backward compatibility
   - Integration: Entry point for user-facing timing API

4. **Add verbosity parameter to solve_ivp function**
   - File: src/cubie/batchsolving/solver.py
   - Action: Modify
   - Details:
     ```python
     # Current signature at line 34:
     def solve_ivp(
         system: BaseODE,
         y0: Union[np.ndarray, Dict[str, np.ndarray]],
         parameters: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
         drivers: Optional[Dict[str, object]] = None,
         dt_save: Optional[float] = None,
         method: str = "euler",
         duration: float = 1.0,
         settling_time: float = 0.0,
         t0: float = 0.0,
         grid_type: str = "combinatorial",
         **kwargs: Any,
     ) -> SolveResult:
     
     # Add verbosity parameter:
     def solve_ivp(
         system: BaseODE,
         y0: Union[np.ndarray, Dict[str, np.ndarray]],
         parameters: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
         drivers: Optional[Dict[str, object]] = None,
         dt_save: Optional[float] = None,
         method: str = "euler",
         duration: float = 1.0,
         settling_time: float = 0.0,
         t0: float = 0.0,
         grid_type: str = "combinatorial",
         verbosity: str = 'default',
         **kwargs: Any,
     ) -> SolveResult:
     
     # Add to docstring (after grid_type):
         verbosity
             Timing output verbosity. See :class:`Solver` for options.
             Defaults to 'default'.
     
     # Pass to Solver creation (around line 86):
         solver = Solver(
             system,
             algorithm=method,
             loop_settings=loop_settings,
             verbosity=verbosity,
             **kwargs,
         )
     ```
   - Edge cases: Public API - must be backward compatible
   - Integration: Convenience wrapper around Solver

**Outcomes**: 
(To be filled by taskmaster)

---

## Task Group 9: Array Interpolator - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Group 2

**Required Context**:
- File: src/cubie/integrators/array_interpolator.py (lines 1-100)

**Input Validation Required**:
- None (time_logger validated by duck typing)

**Tasks**:

1. **Add time_logger to ArrayInterpolator.__init__**
   - File: src/cubie/integrators/array_interpolator.py
   - Action: Modify
   - Details:
     ```python
     # Find __init__ signature (ArrayInterpolator inherits from CUDAFactory)
     # Add time_logger parameter at end
     # Pass to super().__init__()
     
     def __init__(
         self,
         # ... existing parameters (precision, input_dict, etc.) ...
         time_logger = None,
     ) -> None:
         super().__init__(time_logger=time_logger)
         # ... rest of initialization ...
     ```
   - Edge cases: May be None in many use cases
   - Integration: Created by Solver for driver interpolation

2. **Update ArrayInterpolator instantiation in Solver**
   - File: src/cubie/batchsolving/solver.py
   - Action: Modify
   - Details:
     ```python
     # Find ArrayInterpolator creation (around line 172):
     self.driver_interpolator = ArrayInterpolator(
         precision=precision,
         input_dict={
             "placeholder": np.zeros(6, dtype=precision),
             "dt": 0.1,
         },
         time_logger=self.time_logger,  # ADD THIS
     )
     ```
   - Edge cases: Solver may not have time_logger yet at this point
   - Integration: Ensure self.time_logger is created before this line

**Outcomes**: 
(To be filled by taskmaster)

---

## Task Group 10: Module Exports - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/__init__.py

**Input Validation Required**:
- None

**Tasks**:

1. **Export TimeLogger from cubie package**
   - File: src/cubie/__init__.py
   - Action: Modify
   - Details:
     ```python
     # Add to imports section:
     from cubie.time_logger import TimeLogger
     
     # Add to __all__ list (if it exists):
     __all__ = [
         # ... existing exports ...
         "TimeLogger",
     ]
     ```
   - Edge cases: Check if __all__ exists before modifying
   - Integration: Makes TimeLogger available as `from cubie import TimeLogger`

**Outcomes**: 
(To be filled by taskmaster)

---

## Task Group 11: Test Infrastructure - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Groups 1-10 (all implementation complete)

**Required Context**:
- File: tests/conftest.py (entire file)
- File: tests/system_fixtures.py

**Input Validation Required**:
- None (tests don't validate inputs)

**Tasks**:

1. **Create test_time_logger.py module**
   - File: tests/test_time_logger.py
   - Action: Create
   - Details:
     ```python
     """Tests for time_logger module."""
     
     import pytest
     from cubie.time_logger import TimeLogger, TimingEvent
     
     
     def test_timelogger_initialization():
         """Test TimeLogger initializes with valid verbosity."""
         logger = TimeLogger(verbosity='default')
         assert logger.verbosity == 'default'
         assert logger.events == []
         
         logger = TimeLogger(verbosity='verbose')
         assert logger.verbosity == 'verbose'
         
         logger = TimeLogger(verbosity='debug')
         assert logger.verbosity == 'debug'
     
     
     def test_timelogger_invalid_verbosity():
         """Test TimeLogger rejects invalid verbosity."""
         with pytest.raises(ValueError, match="verbosity must be"):
             TimeLogger(verbosity='invalid')
     
     
     def test_start_stop_event():
         """Test basic start/stop event recording."""
         logger = TimeLogger(verbosity='default')
         
         logger.start_event('test_event')
         assert len(logger.events) == 1
         assert logger.events[0].name == 'test_event'
         assert logger.events[0].event_type == 'start'
         
         logger.stop_event('test_event')
         assert len(logger.events) == 2
         assert logger.events[1].name == 'test_event'
         assert logger.events[1].event_type == 'stop'
     
     
     def test_progress_event():
         """Test progress event recording."""
         logger = TimeLogger(verbosity='default')
         
         logger.progress('operation', 'halfway done')
         assert len(logger.events) == 1
         assert logger.events[0].event_type == 'progress'
         assert logger.events[0].metadata['message'] == 'halfway done'
     
     
     def test_get_event_duration():
         """Test duration calculation."""
         import time
         
         logger = TimeLogger(verbosity='default')
         logger.start_event('timed_op')
         time.sleep(0.01)  # Small delay
         logger.stop_event('timed_op')
         
         duration = logger.get_event_duration('timed_op')
         assert duration is not None
         assert duration > 0.0
         assert duration < 1.0  # Should be small
     
     
     def test_orphaned_stop_event():
         """Test handling of stop without start."""
         logger = TimeLogger(verbosity='default')
         logger.stop_event('orphan')
         
         duration = logger.get_event_duration('orphan')
         assert duration is None
     
     
     def test_get_aggregate_durations():
         """Test aggregation of multiple events."""
         import time
         
         logger = TimeLogger(verbosity='default')
         
         # First operation
         logger.start_event('compile')
         time.sleep(0.01)
         logger.stop_event('compile')
         
         # Second operation
         logger.start_event('compile')
         time.sleep(0.01)
         logger.stop_event('compile')
         
         durations = logger.get_aggregate_durations()
         assert 'compile' in durations
         assert durations['compile'] > 0.02  # Sum of two sleeps
     
     
     def test_empty_event_name():
         """Test rejection of empty event names."""
         logger = TimeLogger(verbosity='default')
         
         with pytest.raises(ValueError, match="cannot be empty"):
             logger.start_event('')
         
         with pytest.raises(ValueError, match="cannot be empty"):
             logger.stop_event('')
         
         with pytest.raises(ValueError, match="cannot be empty"):
             logger.progress('', 'message')
     
     
     def test_print_summary_no_crash(capsys):
         """Test print_summary doesn't crash on empty logger."""
         logger = TimeLogger(verbosity='default')
         logger.print_summary()  # Should not crash
         
         # Add some events
         logger.start_event('test')
         logger.stop_event('test')
         logger.print_summary()  # Should print summary
         
         captured = capsys.readouterr()
         assert 'test' in captured.out or len(captured.out) == 0
     
     
     def test_metadata_storage():
         """Test metadata is stored with events."""
         logger = TimeLogger(verbosity='default')
         
         logger.start_event('compile', file='test.py', size=1024)
         assert logger.events[0].metadata['file'] == 'test.py'
         assert logger.events[0].metadata['size'] == 1024
     
     
     def test_timing_event_frozen():
         """Test TimingEvent is immutable."""
         event = TimingEvent(
             name='test',
             event_type='start',
             timestamp=0.0,
             metadata={}
         )
         
         with pytest.raises(attrs.exceptions.FrozenInstanceError):
             event.name = 'modified'
     ```
   - Edge cases: All major edge cases covered
   - Integration: Validates TimeLogger behavior in isolation

2. **Add time_logger fixture to conftest.py**
   - File: tests/conftest.py
   - Action: Modify
   - Details:
     ```python
     # Add import at top:
     from cubie.time_logger import TimeLogger
     
     # Add fixture (insert after other fixtures):
     @pytest.fixture
     def time_logger_none():
         """Provide None for time_logger parameter (default behavior)."""
         return None
     
     
     @pytest.fixture
     def time_logger_verbose():
         """Provide TimeLogger with verbose mode for testing."""
         return TimeLogger(verbosity='verbose')
     
     
     @pytest.fixture
     def time_logger_debug():
         """Provide TimeLogger with debug mode for testing."""
         return TimeLogger(verbosity='debug')
     ```
   - Edge cases: Provides fixtures for all verbosity levels
   - Integration: Available to all tests

3. **Update existing fixture factories with time_logger=None**
   - File: tests/conftest.py
   - Action: Modify
   - Details:
     ```python
     # Find all factory instantiations in fixture functions
     # Add time_logger=None to each
     
     # Examples to search for and update:
     # - "OutputFunctions(" -> add time_logger=None
     # - "IVPLoop(" -> add time_logger=None
     # - "get_algorithm_step(" -> add time_logger=None
     # - "get_controller(" -> add time_logger=None
     # - "SingleIntegratorRun(" -> add time_logger=None
     # - "BatchSolverKernel(" -> add time_logger=None
     # - "Solver(" -> add time_logger=None or verbosity='default'
     # - "ArrayInterpolator(" -> add time_logger=None
     
     # Pattern:
     factory_instance = FactoryClass(
         # ... existing parameters ...
         time_logger=None,
     )
     ```
   - Edge cases: Many fixture functions to update
   - Integration: Ensures existing tests pass without modification

4. **Create integration test for time_logger threading**
   - File: tests/test_time_logger.py
   - Action: Modify (add to existing file)
   - Details:
     ```python
     def test_time_logger_threads_through_solver(three_state_linear):
         """Test TimeLogger is passed through Solver to components."""
         from cubie import Solver
         
         logger = TimeLogger(verbosity='default')
         
         solver = Solver(
             three_state_linear,
             algorithm='euler',
             verbosity='default',  # This creates internal logger
         )
         
         # Solver should have created its own logger
         assert hasattr(solver, 'time_logger')
         # Note: We can't easily verify it was passed to all components
         # without exposing internal structure, but we can verify
         # the solver accepts verbosity parameter
     
     
     def test_solve_ivp_accepts_verbosity(three_state_linear):
         """Test solve_ivp accepts verbosity parameter."""
         from cubie import solve_ivp
         import numpy as np
         
         # Should not crash with verbosity parameter
         result = solve_ivp(
             three_state_linear,
             y0={'s0': [1.0], 's1': [0.0], 's2': [0.0]},
             method='euler',
             duration=0.1,
             dt_save=0.01,
             verbosity='default',
         )
         
         assert result is not None
     ```
   - Edge cases: Validates user-facing API
   - Integration: End-to-end test with real solver

**Outcomes**: 
(To be filled by taskmaster)

---

## Summary

**Total Task Groups**: 11

**Dependency Chain**:
```
Group 1 (TimeLogger Module)
    ↓
Group 2 (CUDAFactory Base)
    ↓
    ├→ Group 3 (ODE Systems) ──┐
    ├→ Group 4 (Integration)   │
    ├→ Group 5 (Output)        ├→ Group 8 (Batch Solver)
    ├→ Group 6 (Algorithms)    │       ↓
    ├→ Group 7 (Controllers) ──┘   Group 11 (Tests)
    ├→ Group 9 (Interpolator)
    └→ Group 10 (Exports)
```

**Parallel Execution Opportunities**:
- Groups 3-7, 9-10 can be done in parallel after Group 2 completes
- Group 6 (Algorithm Step Factories) has multiple independent algorithm files
- Group 7 (Step Controller Factories) has multiple independent controller files

**Estimated Complexity**:
- **Simple**: Groups 1, 2, 5, 9, 10 (single file or straightforward modifications)
- **Moderate**: Groups 3, 4, 8 (multiple files, coordination required)
- **Complex**: Groups 6, 7, 11 (many files, many instantiation sites)

**Key Risk Areas**:
1. **Signature changes**: Easy to miss a call site when adding optional parameter
2. **Test fixture updates**: Many fixtures instantiate factories - all need time_logger=None
3. **Import ordering**: TimeLogger must be importable when Solver tries to import it
4. **super().__init__() calls**: Must pass time_logger through inheritance chain

**Validation Checkpoints**:
1. After Group 1: TimeLogger unit tests pass
2. After Group 2: CUDAFactory accepts time_logger parameter
3. After Groups 3-7: All factories compile without errors
4. After Group 8: Solver creates TimeLogger and passes it down
5. After Group 11: Full test suite passes

**Phase 1 Completion Criteria**:
- [ ] All CUDAFactory subclasses accept time_logger parameter
- [ ] Solver creates TimeLogger based on verbosity
- [ ] solve_ivp accepts verbosity parameter
- [ ] All existing tests pass with time_logger=None
- [ ] New tests validate TimeLogger behavior
- [ ] No actual timing calls implemented (infrastructure only)
- [ ] No breaking changes to existing API
