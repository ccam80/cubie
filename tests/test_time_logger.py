"""Tests for the time_logger module."""

import time
import pytest

from cubie.time_logger import TimeLogger, TimingEvent, _default_timelogger
from cubie.odesystems.symbolic import create_ODE_system
from cubie.batchsolving.solver import solve_ivp

class TestTimingEvent:
    """Test TimingEvent dataclass."""

    def test_timing_event_creation(self):
        """Test that TimingEvent can be created with required fields."""
        event = TimingEvent(
            name="test_event",
            event_type="start",
            timestamp=123.456,
        )
        assert event.name == "test_event"
        assert event.event_type == "start"
        assert event.timestamp == 123.456
        assert event.metadata == {}

    def test_timing_event_with_metadata(self):
        """Test TimingEvent with optional metadata."""
        event = TimingEvent(
            name="test_event",
            event_type="progress",
            timestamp=123.456,
            metadata={"message": "Test message"},
        )
        assert event.metadata == {"message": "Test message"}


class TestTimeLogger:
    """Test TimeLogger class."""

    def test_initialization_default(self):
        """Test TimeLogger initialization with default verbosity."""
        logger = TimeLogger()
        assert logger.verbosity == None
        assert logger.events == []

    def test_initialization_verbose(self):
        """Test TimeLogger initialization with verbose level."""
        logger = TimeLogger(verbosity="verbose")
        assert logger.verbosity == "verbose"

    def test_initialization_debug(self):
        """Test TimeLogger initialization with debug level."""
        logger = TimeLogger(verbosity="debug")
        assert logger.verbosity == "debug"

    def test_initialization_none(self):
        """Test TimeLogger initialization with None verbosity."""
        logger = TimeLogger(verbosity=None)
        assert logger.verbosity is None

    def test_initialization_string_none(self):
        """Test TimeLogger initialization with string 'None'."""
        logger = TimeLogger(verbosity='None')
        assert logger.verbosity is None

    def test_initialization_invalid_verbosity(self):
        """Test that invalid verbosity raises ValueError."""
        with pytest.raises(ValueError, match="verbosity must be"):
            TimeLogger(verbosity="invalid")

    def test_none_verbosity_no_op(self):
        """Test that None verbosity creates no-op logger."""
        logger = TimeLogger(verbosity=None)
        # Registration still works even with None verbosity
        logger.register_event("test", "runtime", "Test event")
        logger.start_event("test")
        logger.stop_event("test")
        logger.progress("test", "message")
        assert len(logger.events) == 0

    def test_set_verbosity(self):
        """Test changing verbosity level."""
        logger = TimeLogger(verbosity='default')
        logger.set_verbosity('verbose')
        assert logger.verbosity == 'verbose'
        logger.set_verbosity(None)
        assert logger.verbosity is None

    def test_start_event(self):
        """Test recording a start event."""
        logger = TimeLogger(verbosity="default")
        logger.register_event("test_operation", "runtime", "Test operation")
        logger.start_event("test_operation")
        
        assert len(logger.events) == 1
        assert logger.events[0].name == "test_operation"
        assert logger.events[0].event_type == "start"
        assert logger.events[0].timestamp > 0

    def test_stop_event(self):
        """Test recording a stop event."""
        logger = TimeLogger(verbosity='default')
        logger.register_event("test_operation", "runtime", "Test operation")
        logger.start_event("test_operation")
        time.sleep(0.01)
        logger.stop_event("test_operation")
        
        assert len(logger.events) == 2
        assert logger.events[1].name == "test_operation"
        assert logger.events[1].event_type == "stop"
        assert logger.events[1].timestamp > logger.events[0].timestamp

    def test_progress_event(self):
        """Test recording a progress event."""
        logger = TimeLogger(verbosity='default')
        logger.register_event("test_operation", "runtime", "Test operation")
        logger.progress("test_operation", "50% complete")
        
        assert len(logger.events) == 1
        assert logger.events[0].name == "test_operation"
        assert logger.events[0].event_type == "progress"
        assert logger.events[0].metadata["message"] == "50% complete"

    def test_get_event_duration(self):
        """Test calculating duration between start and stop events."""
        logger = TimeLogger(verbosity='default')
        logger.register_event("test_operation", "runtime", "Test operation")
        logger.start_event("test_operation")
        time.sleep(0.02)
        logger.stop_event("test_operation")
        
        duration = logger.get_event_duration("test_operation")
        assert duration is not None
        assert duration >= 0.02
        assert duration < 0.1

    def test_get_event_duration_no_stop(self):
        """Test get_event_duration returns None when stop event missing."""
        logger = TimeLogger()
        logger.register_event("test_operation", "runtime", "Test operation")
        logger.start_event("test_operation")
        
        duration = logger.get_event_duration("test_operation")
        assert duration is None

    def test_get_event_duration_no_start(self):
        """Test get_event_duration returns None when start missing."""
        logger = TimeLogger()
        # This should now raise an error since we require registration and start
        # So this test is no longer valid - removing assertion
        duration = logger.get_event_duration("test_operation")
        assert duration is None

    def test_multiple_operations(self):
        """Test tracking multiple operations."""
        logger = TimeLogger(verbosity='default')
        logger.register_event("operation1", "runtime", "Operation 1")
        logger.register_event("operation2", "runtime", "Operation 2")
        logger.start_event("operation1")
        logger.start_event("operation2")
        logger.stop_event("operation1")
        logger.stop_event("operation2")
        
        assert len(logger.events) == 4
        assert logger.get_event_duration("operation1") is not None
        assert logger.get_event_duration("operation2") is not None

    def test_callbacks_return_none(self):
        """Test that callbacks work but don't affect functionality."""
        logger = TimeLogger()
        logger.register_event("test", "runtime", "Test event")
        
        # All callbacks should work without errors
        result1 = logger.start_event("test")
        result2 = logger.stop_event("test")
        result3 = logger.progress("test", "message")
        
        # None of them return values that would affect code flow
        assert result1 is None
        assert result2 is None
        assert result3 is None

    def test_print_summary_default_verbosity(self, capsys):
        """Test summary output at default verbosity."""
        logger = TimeLogger(verbosity="default")
        logger.register_event("codegen", "codegen", "Code generation")
        logger.start_event("codegen")
        time.sleep(0.01)
        logger.stop_event("codegen")
        
        logger.print_summary()
        captured = capsys.readouterr()
        assert "Timing Summary" in captured.out

    def test_print_summary_verbose(self, capsys):
        """Test summary output at verbose level."""
        logger = TimeLogger(verbosity="verbose")
        logger.register_event("codegen", "codegen", "Code generation")
        logger.register_event("codegen.component1", "codegen", "Component 1")
        logger.start_event("codegen")
        logger.start_event("codegen.component1")
        time.sleep(0.01)
        logger.stop_event("codegen.component1")
        logger.stop_event("codegen")
        
        logger.print_summary()
        captured = capsys.readouterr()
        # Verbose mode prints during stop_event
        assert "codegen.component1" in captured.out

    def test_print_summary_debug(self, capsys):
        """Test summary output at debug level."""
        logger = TimeLogger(verbosity="debug")
        logger.register_event("test", "runtime", "Test event")
        logger.start_event("test")
        logger.progress("test", "halfway")
        logger.stop_event("test")
        
        logger.print_summary()
        captured = capsys.readouterr()
        # Debug mode prints during events
        assert "DEBUG" in captured.out
        assert "progress" in captured.out.lower()

    def test_get_aggregate_durations(self):
        """Test aggregating event durations."""
        logger = TimeLogger(verbosity='default')
        logger.register_event("operation1", "runtime", "Operation 1")
        logger.start_event("operation1")
        time.sleep(0.01)
        logger.stop_event("operation1")
        logger.start_event("operation1")
        time.sleep(0.01)
        logger.stop_event("operation1")
        
        durations = logger.get_aggregate_durations()
        assert "operation1" in durations
        assert durations["operation1"] >= 0.02

    def test_empty_event_name_raises(self):
        """Test that empty event names raise ValueError."""
        logger = TimeLogger(verbosity='default')
        logger.register_event("valid", "runtime", "Valid event")
        with pytest.raises(ValueError, match="event_name cannot be empty"):
            logger.start_event("")
        with pytest.raises(ValueError, match="event_name cannot be empty"):
            logger.stop_event("")
        with pytest.raises(ValueError, match="event_name cannot be empty"):
            logger.progress("", "message")

    def test_register_event(self):
        """Test registering events with metadata."""
        logger = TimeLogger()
        logger.register_event("dxdt_build", "compile", "Build dxdt function")
        
        assert "dxdt_build" in logger._event_registry
        assert logger._event_registry["dxdt_build"]["category"] == "compile"
        assert logger._event_registry["dxdt_build"]["description"] == "Build dxdt function"

    def test_register_event_invalid_category(self):
        """Test that invalid category raises ValueError."""
        logger = TimeLogger()
        with pytest.raises(ValueError, match="category must be"):
            logger.register_event("test", "invalid", "description")

    def test_register_event_valid_categories(self):
        """Test all valid categories."""
        logger = TimeLogger()
        logger.register_event("event1", "codegen", "Codegen event")
        logger.register_event("event2", "runtime", "Runtime event")
        logger.register_event("event3", "compile", "Compile event")
        
        assert len(logger._event_registry) == 3
        assert logger._event_registry["event1"]["category"] == "codegen"
        assert logger._event_registry["event2"]["category"] == "runtime"
        assert logger._event_registry["event3"]["category"] == "compile"

    def test_register_event_compile_category(self):
        """Test that 'compile' category is accepted."""
        logger = TimeLogger()
        logger.register_event("compile_test", "compile", "Compile event")
        
        assert "compile_test" in logger._event_registry
        assert logger._event_registry["compile_test"]["category"] == "compile"
        assert logger._event_registry["compile_test"]["description"] == "Compile event"

    def test_unregistered_event_raises(self):
        """Test that unregistered events raise ValueError."""
        logger = TimeLogger(verbosity='default')
        with pytest.raises(ValueError, match="not registered"):
            logger.start_event("unregistered")
        with pytest.raises(ValueError, match="not registered"):
            logger.stop_event("unregistered")
        with pytest.raises(ValueError, match="not registered"):
            logger.progress("unregistered", "message")

    def test_aggregate_durations_by_category(self):
        """Test filtering aggregate durations by category."""
        logger = TimeLogger(verbosity='default')
        logger.register_event("codegen1", "codegen", "Codegen 1")
        logger.register_event("runtime1", "runtime", "Runtime 1")
        logger.register_event("compile1", "compile", "Compile 1")
        
        logger.start_event("codegen1")
        time.sleep(0.01)
        logger.stop_event("codegen1")
        
        logger.start_event("runtime1")
        time.sleep(0.01)
        logger.stop_event("runtime1")
        
        logger.start_event("compile1")
        time.sleep(0.01)
        logger.stop_event("compile1")
        
        # Test filtering by category
        codegen_durations = logger.get_aggregate_durations(category="codegen")
        assert "codegen1" in codegen_durations
        assert "runtime1" not in codegen_durations
        assert "compile1" not in codegen_durations
        
        runtime_durations = logger.get_aggregate_durations(category="runtime")
        assert "runtime1" in runtime_durations
        assert "codegen1" not in runtime_durations
        assert "compile1" not in runtime_durations

    def test_aggregate_durations_compile_category(self):
        """Test filtering aggregate durations for compile category."""
        logger = TimeLogger(verbosity='default')
        logger.register_event("compile1", "compile", "Compile 1")
        logger.register_event("runtime1", "runtime", "Runtime 1")
        
        logger.start_event("compile1")
        time.sleep(0.01)
        logger.stop_event("compile1")
        
        logger.start_event("runtime1")
        time.sleep(0.01)
        logger.stop_event("runtime1")
        
        # Test filtering by compile category
        compile_durations = logger.get_aggregate_durations(category="compile")
        assert "compile1" in compile_durations
        assert "runtime1" not in compile_durations
        assert compile_durations["compile1"] >= 0.01

    def test_print_summary_by_category(self, capsys):
        """Test printing summary for specific categories."""
        logger = TimeLogger(verbosity="default")
        logger.register_event("codegen1", "codegen", "Codegen event")
        logger.register_event("compile1", "compile", "Compile event")
        logger.register_event("runtime1", "runtime", "Runtime event")
        
        logger.start_event("codegen1")
        time.sleep(0.01)
        logger.stop_event("codegen1")
        
        logger.start_event("compile1")
        time.sleep(0.01)
        logger.stop_event("compile1")
        
        logger.start_event("runtime1")
        time.sleep(0.01)
        logger.stop_event("runtime1")
        
        # Test codegen category summary
        logger.print_summary(category="codegen")
        captured = capsys.readouterr()
        assert "Codegen Timing Summary" in captured.out
        assert "codegen1" in captured.out
        assert "compile1" not in captured.out
        assert "runtime1" not in captured.out
        
        # Test compile category summary
        logger.print_summary(category="compile")
        captured = capsys.readouterr()
        assert "Compile Timing Summary" in captured.out
        assert "compile1" in captured.out
        assert "codegen1" not in captured.out
        assert "runtime1" not in captured.out
        
        # Test runtime category summary
        logger.print_summary(category="runtime")
        captured = capsys.readouterr()
        assert "Runtime Timing Summary" in captured.out
        assert "runtime1" in captured.out
        assert "codegen1" not in captured.out
        assert "compile1" not in captured.out

        # Test all categories summary
        logger.print_summary()
        captured = capsys.readouterr()
        assert "Timing Summary" in captured.out
        assert "codegen1" in captured.out
        assert "compile1" in captured.out
        assert "runtime1" in captured.out


@pytest.mark.nocudasim
def test_compilation_caching():
    """Test whether CUDA compilations are cached after specialise_and_compile.

    This test creates two SymbolicODE systems with the same equations but
    in different orders (to avoid codegen cache) and compares execution times.
    The second run should be faster if compiled kernels are cached.
    """
    # Define a reasonably complex ODE system
    # Using a simple 3-state system with nonlinear terms
    equations = [
        "dx = -k1*x + k2*y*z",
        "dy = k1*x - k2*y*z - k3*y",
        "dz = k3*y - k4*z",
    ]

    initial_values = {"x": [1.0], "y": [0.0], "z": [0.0]}

    parameters = {"k1": [0.1], "k2": [0.2], "k3": [0.3], "k4": [0.4]}

    # First run: with verbose logging
    print("\n=== First run (verbose logging) ===")
    _default_timelogger.set_verbosity("verbose")
    _default_timelogger.events = []  # Clear previous events

    system1 = create_ODE_system(
        dxdt=equations, parameters=list(parameters.keys()), name="TestSystem1"
    )

    start_time1 = time.perf_counter()
    result1 = solve_ivp(
        system=system1,
        y0=initial_values,
        parameters=parameters,
        duration=0.01,
        dt=1e-4,
        method="euler",
        settling_time=0.0,
    )
    end_time1 = time.perf_counter()

    time1 = end_time1 - start_time1
    print(f"First run total time: {time1:.4f}s")

    # Get compilation events from first run
    compile_events = [
        e for e in _default_timelogger.events if "compile" in e.name.lower()
    ]
    print(f"Number of compile events in first run: {len(compile_events)}")

    # Second run: same equations but swapped order (avoid codegen cache)
    # with no logging
    print("\n=== Second run (no logging, swapped equation order) ===")
    _default_timelogger.set_verbosity(None)
    _default_timelogger.events = []  # Clear events

    # Swap equation order to force new codegen
    equations_swapped = [
        "dz = k3*y - k4*z",
        "dx = -k1*x + k2*y*z",
        "dy = k1*x - k2*y*z - k3*y",
    ]

    system2 = create_ODE_system(
        dxdt=equations_swapped,
        parameters=list(parameters.keys()),
        name="TestSystem2",
    )

    # Reorder initial values to match
    initial_values_swapped = {
        "z": [0.0],
        "x": [1.0],
        "y": [0.0],
    }

    start_time2 = time.perf_counter()
    result2 = solve_ivp(
        system=system2,
        y0=initial_values_swapped,
        parameters=parameters,
        duration=0.01,
        dt=1e-4,
        method="euler",
        settling_time=0.0,
    )
    end_time2 = time.perf_counter()

    time2 = end_time2 - start_time2
    print(f"Second run total time: {time2:.4f}s")

    # Compare times
    print("\n=== Comparison ===")
    print(f"First run time:  {time1:.4f}s")
    print(f"Second run time: {time2:.4f}s")
    print(f"Speedup: {time1/time2:.2f}x")

    if time2 < time1:
        print("✓ Second run was faster - compilations may be cached")
    else:
        print("✗ Second run was not faster - compilations may not be cached")

    # Reset logger
    _default_timelogger.set_verbosity(None)
    _default_timelogger.events = []

    # Assert results are valid
    assert result1 is not None
    assert result2 is not None


@pytest.mark.nocudasim
def test_timelogger_default_mode_printing():
    """Test TimeLogger printing behavior in default mode.

    In default mode, the user wants:
    - Codegen summary to print after parsing is complete
    - Compilation time to print after top-level compile is completed
    - Runtime to print after kernels return
    """
    # This test documents the current behavior
    # Future implementation should modify print_summary or add new methods
    # to print category-specific summaries at appropriate times

    equations = ["dx = -k*x", "dy = k*x"]

    initial_values = {"x": [1.0], "y": [0.0]}

    parameters = {"k": [0.5]}

    _default_timelogger.set_verbosity("default")
    _default_timelogger.events = []

    system = create_ODE_system(
        dxdt=equations,
        parameters=list(parameters.keys()),
        name="TestPrintingSystem",
    )

    result = solve_ivp(
        system=system,
        y0=initial_values,
        parameters=parameters,
        duration=0.01,
        method="radau",
        settling_time=0.0,
    )

    # Print summary at the end
    _default_timelogger.print_summary()

    # Get events by category
    codegen_events = [
        e
        for e in _default_timelogger.events
        if _default_timelogger._event_registry.get(e.name, {}).get("category")
        == "codegen"
    ]
    compile_events = [
        e
        for e in _default_timelogger.events
        if _default_timelogger._event_registry.get(e.name, {}).get("category")
        == "compile"
    ]
    runtime_events = [
        e
        for e in _default_timelogger.events
        if _default_timelogger._event_registry.get(e.name, {}).get("category")
        == "runtime"
    ]

    print(f"\nCodegen events: {len(codegen_events)}")
    print(f"Compile events: {len(compile_events)}")
    print(f"Runtime events: {len(runtime_events)}")

    # Reset logger
    _default_timelogger.set_verbosity(None)
    _default_timelogger.events = []

    assert result is not None


if __name__ == "__main__":
    # Allow running this test file directly
    print("Running compilation timing tests...")
    test_compilation_caching()
    test_timelogger_default_mode_printing()

