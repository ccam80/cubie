"""Tests for the time_logger module."""

import time
import pytest

from cubie.time_logger import TimeLogger, TimingEvent


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
        assert logger.verbosity == "default"
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
        logger._register_event("test", "build", "Test event")
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
        logger = TimeLogger()
        logger._register_event("test_operation", "build", "Test operation")
        logger.start_event("test_operation")
        
        assert len(logger.events) == 1
        assert logger.events[0].name == "test_operation"
        assert logger.events[0].event_type == "start"
        assert logger.events[0].timestamp > 0

    def test_stop_event(self):
        """Test recording a stop event."""
        logger = TimeLogger()
        logger._register_event("test_operation", "build", "Test operation")
        logger.start_event("test_operation")
        time.sleep(0.01)
        logger.stop_event("test_operation")
        
        assert len(logger.events) == 2
        assert logger.events[1].name == "test_operation"
        assert logger.events[1].event_type == "stop"
        assert logger.events[1].timestamp > logger.events[0].timestamp

    def test_progress_event(self):
        """Test recording a progress event."""
        logger = TimeLogger()
        logger._register_event("test_operation", "build", "Test operation")
        logger.progress("test_operation", "50% complete")
        
        assert len(logger.events) == 1
        assert logger.events[0].name == "test_operation"
        assert logger.events[0].event_type == "progress"
        assert logger.events[0].metadata["message"] == "50% complete"

    def test_get_event_duration(self):
        """Test calculating duration between start and stop events."""
        logger = TimeLogger()
        logger._register_event("test_operation", "build", "Test operation")
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
        logger._register_event("test_operation", "build", "Test operation")
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
        logger = TimeLogger()
        logger._register_event("operation1", "build", "Operation 1")
        logger._register_event("operation2", "build", "Operation 2")
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
        logger._register_event("test", "build", "Test event")
        
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
        logger._register_event("codegen", "codegen", "Code generation")
        logger.start_event("codegen")
        time.sleep(0.01)
        logger.stop_event("codegen")
        
        logger.print_summary()
        captured = capsys.readouterr()
        assert "Timing Summary" in captured.out

    def test_print_summary_verbose(self, capsys):
        """Test summary output at verbose level."""
        logger = TimeLogger(verbosity="verbose")
        logger._register_event("codegen", "codegen", "Code generation")
        logger._register_event("codegen.component1", "codegen", "Component 1")
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
        logger._register_event("test", "build", "Test event")
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
        logger = TimeLogger()
        logger._register_event("operation1", "build", "Operation 1")
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
        logger = TimeLogger()
        logger._register_event("valid", "build", "Valid event")
        with pytest.raises(ValueError, match="event_name cannot be empty"):
            logger.start_event("")
        with pytest.raises(ValueError, match="event_name cannot be empty"):
            logger.stop_event("")
        with pytest.raises(ValueError, match="event_name cannot be empty"):
            logger.progress("", "message")

    def test_register_event(self):
        """Test registering events with metadata."""
        logger = TimeLogger()
        logger._register_event("dxdt_build", "build", "Build dxdt function")
        
        assert "dxdt_build" in logger._event_registry
        assert logger._event_registry["dxdt_build"]["category"] == "build"
        assert logger._event_registry["dxdt_build"]["description"] == "Build dxdt function"

    def test_register_event_invalid_category(self):
        """Test that invalid category raises ValueError."""
        logger = TimeLogger()
        with pytest.raises(ValueError, match="category must be"):
            logger._register_event("test", "invalid", "description")

    def test_register_event_valid_categories(self):
        """Test all valid categories."""
        logger = TimeLogger()
        logger._register_event("event1", "codegen", "Codegen event")
        logger._register_event("event2", "build", "Build event")
        logger._register_event("event3", "runtime", "Runtime event")
        
        assert len(logger._event_registry) == 3
        assert logger._event_registry["event1"]["category"] == "codegen"
        assert logger._event_registry["event2"]["category"] == "build"
        assert logger._event_registry["event3"]["category"] == "runtime"

    def test_unregistered_event_raises(self):
        """Test that unregistered events raise ValueError."""
        logger = TimeLogger()
        with pytest.raises(ValueError, match="not registered"):
            logger.start_event("unregistered")
        with pytest.raises(ValueError, match="not registered"):
            logger.stop_event("unregistered")
        with pytest.raises(ValueError, match="not registered"):
            logger.progress("unregistered", "message")

    def test_double_start_raises(self):
        """Test that starting an event twice raises ValueError."""
        logger = TimeLogger()
        logger._register_event("test", "build", "Test event")
        logger.start_event("test")
        with pytest.raises(ValueError, match="already has an active start"):
            logger.start_event("test")

    def test_stop_before_start_raises(self):
        """Test that stopping without start raises ValueError."""
        logger = TimeLogger()
        logger._register_event("test", "build", "Test event")
        with pytest.raises(ValueError, match="has no active start"):
            logger.stop_event("test")

    def test_aggregate_durations_by_category(self):
        """Test filtering aggregate durations by category."""
        logger = TimeLogger()
        logger._register_event("codegen1", "codegen", "Codegen 1")
        logger._register_event("build1", "build", "Build 1")
        logger._register_event("runtime1", "runtime", "Runtime 1")
        
        logger.start_event("codegen1")
        time.sleep(0.01)
        logger.stop_event("codegen1")
        
        logger.start_event("build1")
        time.sleep(0.01)
        logger.stop_event("build1")
        
        logger.start_event("runtime1")
        time.sleep(0.01)
        logger.stop_event("runtime1")
        
        # Test filtering by category
        codegen_durations = logger.get_aggregate_durations(category="codegen")
        assert "codegen1" in codegen_durations
        assert "build1" not in codegen_durations
        assert "runtime1" not in codegen_durations
        
        build_durations = logger.get_aggregate_durations(category="build")
        assert "build1" in build_durations
        assert "codegen1" not in build_durations
        assert "runtime1" not in build_durations

