"""Time logging infrastructure for tracking CuBIE compilation performance."""

import time
from typing import Optional, Any
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
    timestamp: float = attrs.field(
        validator=attrs.validators.instance_of(float)
    )
    metadata: dict = attrs.field(factory=dict)


class TimeLogger:
    """Callback-based timing system for CuBIE operations.
    
    Parameters
    ----------
    verbosity : str or None, default='default'
        Output verbosity level. Options:
        - 'default': Aggregate times only
        - 'verbose': Component-level breakdown
        - 'debug': All events with start/stop/progress
        - None or 'None': No-op callbacks with zero overhead
    
    Attributes
    ----------
    verbosity : str or None
        Current verbosity level
    events : list[TimingEvent]
        Chronological list of all recorded events
    _active_starts : dict[str, float]
        Map of event names to their start timestamps (for matching)
    _event_registry : dict[str, dict]
        Registry of event metadata (category, description) by label
    
    Notes
    -----
    A default instance is available as cubie.time_logger._default_timelogger.
    Use set_verbosity() to configure the global logger level.
    """
    
    def __init__(self, verbosity: Optional[str] = None) -> None:
        if verbosity not in {'default', 'verbose', 'debug', None, 'None'}:
            raise ValueError(
                f"verbosity must be 'default', 'verbose', 'debug', "
                f"None, or 'None', got '{verbosity}'"
            )
        # Normalize string 'None' to None
        if verbosity == 'None':
            verbosity = None
        self.verbosity = verbosity
        self.events: list[TimingEvent] = []
        self._active_starts: dict[str, float] = {}
        self._event_registry: dict[str, dict] = {}
    
    def start_event(self, event_name: str, **metadata: Any) -> None:
        """Record the start of a timed operation.
        
        Parameters
        ----------
        event_name : str
            Unique identifier for this event
        **metadata : Any
            Optional metadata to store with event
        
        Raises
        ------
        ValueError
            If event_name is empty, not registered, or already has an
            active start.
        
        Notes
        -----
        Events must be registered with _register_event before use.
        No-op when verbosity is None.
        """
        if self.verbosity is None:
            return
        
        if not event_name:
            raise ValueError("event_name cannot be empty")
        
        if event_name not in self._event_registry:
            raise ValueError(
                f"Event '{event_name}' not registered. "
                "Call _register_event() before using this event."
            )

        # If a job is logged twice for some reason, skip subsequent starts
        # to capture from first start to first return
        if event_name in self._active_starts:
            return

        
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
            print(f"TIMELOGGER [DEBUG] Started: {event_name}")
    
    def stop_event(self, event_name: str, **metadata: Any) -> None:
        """Record the end of a timed operation.
        
        Parameters
        ----------
        event_name : str
            Identifier matching a previous start_event call
        **metadata : Any
            Optional metadata to store with event
        
        Raises
        ------
        ValueError
            If event_name is empty, not registered, or has no active start.
        
        Notes
        -----
        Events must be registered with _register_event before use.
        A matching start_event must be called before stop_event.
        No-op when verbosity is None.
        """
        if self.verbosity is None:
            return
        
        if not event_name:
            raise ValueError("event_name cannot be empty")
        
        if event_name not in self._event_registry:
            raise ValueError(
                f"Event '{event_name}' not registered. "
                "Call _register_event() before using this event."
            )
        
        if event_name not in self._active_starts:
            return # skip extra stops if called twice

        
        timestamp = time.perf_counter()
        duration = timestamp - self._active_starts[event_name]
        del self._active_starts[event_name]
        
        event = TimingEvent(
            name=event_name,
            event_type='stop',
            timestamp=timestamp,
            metadata=metadata
        )
        self.events.append(event)
        
        if self.verbosity == 'debug':
            print(f"TIMELOGGER [DEBUG] Stopped: {event_name} "
                  f"({duration:.3f}s)")
        elif self.verbosity == 'verbose':
            print(f"TIMELOGGER {event_name}: {duration:.3f}s")
    
    def progress(
        self, event_name: str, message: str, **metadata: Any
    ) -> None:
        """Record a progress update within an operation.
        
        Parameters
        ----------
        event_name : str
            Identifier for the operation in progress
        message : str
            Progress message to log
        **metadata : Any
            Optional metadata to store with event
        
        Raises
        ------
        ValueError
            If event_name is empty or not registered.
        
        Notes
        -----
        Events must be registered with _register_event before use.
        Progress events don't require matching start/stop.
        Only printed in debug mode.
        No-op when verbosity is None.
        """
        if self.verbosity is None:
            return
        
        if not event_name:
            raise ValueError("event_name cannot be empty")
        
        if event_name not in self._event_registry:
            raise ValueError(
                f"Event '{event_name}' not registered. "
                "Call _register_event() before using this event."
            )
        
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
            print(f"TIMELOGGER [DEBUG] Progress: {event_name} - {message}")
    
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
    
    def get_aggregate_durations(
        self, category: Optional[str] = None
    ) -> dict[str, float]:
        """Aggregate event durations by category or all events.
        
        Parameters
        ----------
        category : str, optional
            If provided, filter events by their registered category
            ('codegen', 'runtime', or 'compile')
        
        Returns
        -------
        dict[str, float]
            Mapping of event names to total durations
        
        Notes
        -----
        Sums all durations for events with the same name.
        Uses the event registry to filter by category.
        """
        durations: dict[str, float] = {}
        event_starts: dict[str, float] = {}
        
        for event in self.events:
            # Filter by category using registry if requested
            if category is not None:
                event_info = self._event_registry.get(event.name)
                if event_info is None or event_info['category'] != category:
                    continue
            
            if event.event_type == 'start':
                event_starts[event.name] = event.timestamp
            elif event.event_type == 'stop':
                if event.name in event_starts:
                    duration = (
                        event.timestamp - event_starts[event.name]
                    )
                    durations[event.name] = (
                        durations.get(event.name, 0.0) + duration
                    )
                    del event_starts[event.name]
        
        return durations
    
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
        """
        if self.verbosity == 'default':
            durations = self.get_aggregate_durations(category=category)
            if durations:
                if category:
                    print(f"\nTIMELOGGER {category.capitalize()} Timing Summary:")
                else:
                    print("\nTIMELOGGER Timing Summary:")
                for name, duration in sorted(durations.items()):
                    print(f"TIMELOGGER   {name}: {duration:.3f}s")
        # verbose and debug already printed inline
    
    def set_verbosity(self, verbosity: Optional[str]) -> None:
        """Set the verbosity level for this logger.
        
        Parameters
        ----------
        verbosity : str or None
            New verbosity level. Options are 'default', 'verbose',
            'debug', None, or 'None'.
        
        Notes
        -----
        Changing verbosity does not clear existing events.
        """
        if verbosity not in {'default', 'verbose', 'debug', None, 'None'}:
            raise ValueError(
                f"verbosity must be 'default', 'verbose', 'debug', "
                f"None, or 'None', got '{verbosity}'"
            )
        # Normalize string 'None' to None
        if verbosity == 'None':
            verbosity = None
        self.verbosity = verbosity
    
    def register_event(
        self, label: str, category: str, description: str
    ) -> None:
        """Register an event with metadata for tracking and reporting.
        
        Parameters
        ----------
        label : str
            Event label used in start_event/stop_event calls
        category : str
            Event category: 'codegen', 'runtime', or 'compile'
        description : str
            Human-readable description included in printouts
        
        Notes
        -----
        This method is called by CUDAFactory subclasses to register
        timing events they will track. The category helps organize
        timing reports by operation type.
        """
        if category not in {'codegen', 'runtime', 'compile'}:
            raise ValueError(
                f"category must be 'codegen', 'runtime', or 'compile', "
                f"got '{category}'"
            )
        if label not in self._event_registry:
            self._event_registry[label] = {
                'category': category,
                'description': description
            }


# Default global logger instance
# Use set_verbosity() to configure, or access via cubie.time_logger
_default_timelogger = TimeLogger(verbosity=None)
