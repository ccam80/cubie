"""Time logging infrastructure for tracking CuBIE compilation performance."""

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
    timestamp: float = attrs.field(
        validator=attrs.validators.instance_of(float)
    )
    metadata: dict = attrs.field(factory=dict)


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
        If event_name already has an active start, logs warning in debug
        mode and treats as nested event (matches most recent).
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
                    duration = (
                        event.timestamp - event_starts[event.name]
                    )
                    durations[event.name] = (
                        durations.get(event.name, 0.0) + duration
                    )
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
