"""
Comprehensive tests for OutputConfig class to ensure it integrates properly
with the rest of the system.
"""

import pytest
import numpy as np
from warnings import catch_warnings
from typing import Set, Tuple, List
from src.CuMC.ForwardSim.OutputHandling.output_config import OutputConfig
from src.CuMC.ForwardSim import summary_metrics


class TestOutputConfigInitialization:
    """Test initialization and validation of OutputConfig."""

    def test_minimal_initialization(self):
        """Test minimal valid initialization with defaults."""
        config = OutputConfig(max_states=10, max_observables=5)
        assert config.max_states == 10
        assert config.max_observables == 5
        assert config.save_state is True
        assert config.save_observables is True
        assert config.save_time is False
        assert len(config.summary_types) == 0
        # Check default indices
        np.testing.assert_array_equal(config.saved_state_indices, np.arange(10, dtype=np.int_))
        np.testing.assert_array_equal(config.saved_observable_indices, np.arange(5, dtype=np.int_))

    def test_full_initialization(self):
        """Test initialization with all parameters specified."""
        config = OutputConfig(
            max_states=20,
            max_observables=10,
            save_state=True,
            save_observables=True,
            save_time=True,
            saved_state_indices=[0, 1, 2],
            saved_observable_indices=[0, 1],
            summarised_state_indices=[0, 1],
            summarised_observable_indices=[0],
            summary_types=("mean", "max")
        )
        assert config.max_states == 20
        assert config.max_observables == 10
        assert config.save_state is True
        assert config.save_observables is True
        assert config.save_time is True
        assert config.summary_types == ("mean", "max")
        # Check indices
        np.testing.assert_array_equal(config.saved_state_indices, np.array([0, 1, 2], dtype=np.int_))
        np.testing.assert_array_equal(config.saved_observable_indices, np.array([0, 1], dtype=np.int_))

    def test_no_output_raises_error(self):
        """Test that requesting no output raises an error."""
        with pytest.raises(ValueError, match="At least one output type must be enabled"):
            OutputConfig(
                max_states=10,
                max_observables=5,
                save_state=False,
                save_observables=False,
                save_time=False
            )

    def test_out_of_bounds_indices_raise_error(self):
        """Test that out of bounds indices raise ValueError."""
        with pytest.raises(ValueError, match="Indices must be in the range"):
            OutputConfig(
                max_states=3,
                max_observables=5,
                saved_state_indices=[0, 1, 3]  # 3 is out of bounds for max_states=3
            )

        with pytest.raises(ValueError, match="Indices must be in the range"):
            OutputConfig(
                max_states=5,
                max_observables=2,
                saved_observable_indices=[0, 2]  # 2 is out of bounds for max_observables=2
            )

    def test_duplicate_indices_raise_error(self):
        """Test that duplicate indices raise ValueError."""
        with pytest.raises(ValueError, match="Duplicate indices found"):
            OutputConfig(
                max_states=5,
                max_observables=5,
                saved_state_indices=[0, 1, 1]  # 1 is duplicated
            )


class TestOutputConfigProperties:
    """Test property behavior in OutputConfig."""

    def test_max_dimensions_setters(self):
        """Test setters for max_states and max_observables."""
        config = OutputConfig(max_states=10, max_observables=5)

        config.max_states = 20
        assert config.max_states == 20
        # Default indices should be updated
        assert len(config.saved_state_indices) == 20

        config.max_observables = 8
        assert config.max_observables == 8
        assert len(config.saved_observable_indices) == 8

    def test_saved_indices_setters(self):
        """Test setters for saved indices."""
        config = OutputConfig(max_states=10, max_observables=5)

        # Set saved indices with list
        config.saved_state_indices = [1, 3, 5]
        np.testing.assert_array_equal(config.saved_state_indices, np.array([1, 3, 5], dtype=np.int_))

        # Set saved indices with numpy array
        config.saved_observable_indices = np.array([0, 2], dtype=np.int_)
        np.testing.assert_array_equal(config.saved_observable_indices, np.array([0, 2], dtype=np.int_))

    def test_summarised_indices_setters(self):
        """Test setters for summarised indices."""
        config = OutputConfig(max_states=10, max_observables=5, summary_types=("mean"))

        config.summarised_state_indices = [2, 4]
        np.testing.assert_array_equal(config.summarised_state_indices, np.array([2, 4], dtype=np.int_))

        config.summarised_observable_indices = [1]
        np.testing.assert_array_equal(config.summarised_observable_indices, np.array([1], dtype=np.int_))

    def test_summary_types_setter(self):
        """Test setter for summary_types."""
        config = OutputConfig(max_states=10, max_observables=5)

        # Test with list
        config.update_from_outputs_tuple(("mean", "max"))
        assert config.summary_types == ("mean", "max")

        # Test with tuple
        config.update_from_outputs_tuple(("rms",))
        assert config.summary_types == ("rms",)

        # Test with single string
        config.update_from_outputs_tuple(("mean",))
        assert config.summary_types == ("mean",)


class TestFlagBehavior:
    """Test how flags interact with indices."""

    def test_flag_affects_returned_indices(self):
        """Test that flag state affects which indices are returned."""
        config = OutputConfig(
            max_states=5,
            max_observables=3,
            save_state=True,
            save_observables=True,
            saved_state_indices=[0, 1, 2],
            saved_observable_indices=[0, 1]
        )

        # When flag is true, see real indices
        np.testing.assert_array_equal(config.saved_state_indices, np.array([0, 1, 2], dtype=np.int_))

        # Turn off flag, should see empty array but preserve the actual indices
        config._save_state = False
        np.testing.assert_array_equal(config.saved_state_indices, np.array([], dtype=np.int_))

        # Turn flag back on, should see original indices
        config._save_state = True
        np.testing.assert_array_equal(config.saved_state_indices, np.array([0, 1, 2], dtype=np.int_))

    def test_empty_indices_affect_flag_state(self):
        """Test that setting empty indices affects effective flag state."""
        config = OutputConfig(max_states=5, max_observables=3)

        # Initially flags are true
        assert config.save_state is True
        assert config.save_observables is True


class TestSummaryMetricsIntegration:
    """Test integration with summary metrics system."""

    def test_summary_parameters(self):
        """Test summary_parameters property."""
        config = OutputConfig(
            max_states=10,
            max_observables=5,
            summary_types=("mean", "max")
        )

        # Parameters should match what's defined in summary metrics
        params = config.summary_parameters
        assert len(params) == 2  # One for each summary type
        assert all(param == 0 for param in params)  # Default is 0 for these metrics

    def test_summary_memory_requirements(self):
        """Test memory calculation properties."""
        config = OutputConfig(
            max_states=10,
            max_observables=5,
            summary_types=("mean", "max")
        )

        # These should call into the summary_metrics system
        assert config.summary_temp_memory_per_var > 0
        assert config.summary_output_memory_per_var > 0

        # No summaries should mean no memory needed
        config.update_from_outputs_tuple(("state",))
        assert config.summary_temp_memory_per_var == 0
        assert config.summary_output_memory_per_var == 0


class TestUpdateFromOutputsList:
    """Test update_from_outputs_tuple method."""

    def test_basic_output_types(self):
        """Test setting basic output types."""
        config = OutputConfig(max_states=10, max_observables=5)

        config.update_from_outputs_tuple(("state", "time"))
        assert config._save_state is True
        assert config._save_observables is False
        assert config._save_time is True
        assert len(config.summary_types) == 0

    def test_summary_metrics(self):
        """Test setting summary metrics."""
        config = OutputConfig(max_states=10, max_observables=5)

        config.update_from_outputs_tuple(("mean", "max"))
        assert config._save_state is False
        assert config._save_observables is False
        assert config._save_time is False
        assert config.summary_types == ("mean", "max")

    def test_mixed_outputs(self):
        """Test setting mixed output types."""
        config = OutputConfig(max_states=10, max_observables=5)

        config.update_from_outputs_tuple(("state", "mean", "time"))
        assert config._save_state is True
        assert config._save_observables is False
        assert config._save_time is True
        assert config.summary_types == ("mean",)

    def test_unknown_summary_type_warning(self):
        """Test that unknown summary types produce a warning."""
        config = OutputConfig(max_states=10, max_observables=5)

        with catch_warnings(record=True) as w:
            config.update_from_outputs_tuple(("unknown_metric", "mean"))
            assert len(w) >= 1
            assert "is not implemented" in str(w[0].message)

        # Only valid metric should be included
        assert config.summary_types == ("mean",)


class TestFromLoopSettings:
    """Test the from_loop_settings factory method."""

    def test_basic_creation(self):
        """Test basic creation with defaults."""
        config = OutputConfig.from_loop_settings(
            output_types=["state", "observables"],
            max_states=10,
            max_observables=5
        )

        assert config.save_state is True
        assert config.save_observables is True
        assert config.n_saved_states == 10
        assert config.n_saved_observables == 5

    def test_with_specified_indices(self):
        """Test creation with specified indices."""
        config = OutputConfig.from_loop_settings(
            output_types=["state"],
            saved_states=[1, 3],
            max_states=10,
            max_observables=5
        )

        assert config.save_state is True
        assert config.save_observables is False
        np.testing.assert_array_equal(config.saved_state_indices, np.array([1, 3], dtype=np.int_))

    def test_with_summary_metrics(self):
        """Test creation with summary metrics."""
        config = OutputConfig.from_loop_settings(
            output_types=["mean", "max", "state"],
            saved_states=[0, 1],
            summarised_states=[1],
            max_states=10,
            max_observables=5
        )

        assert config.save_state is True
        assert "mean" in config.summary_types
        assert "max" in config.summary_types
        np.testing.assert_array_equal(config.summarised_state_indices, np.array([1], dtype=np.int_))

    def test_with_parametrized_metrics(self):
        """Test creation with parametrized metrics."""
        config = OutputConfig.from_loop_settings(
            output_types=["peaks[3]", "state"],
            max_states=10,
            max_observables=5
        )

        assert "peaks[3]" in config.summary_types


class TestUtilityProperties:
    """Test utility properties and convenience methods."""

    def test_summarise_states_property(self):
        """Test summarise_states property."""
        config = OutputConfig(
            max_states=10,
            max_observables=5,
            summary_types=("mean",)
        )
        assert config.summarise_states is True

        # Having summary types but no indices means no summarization
        config.update_from_outputs_tuple(("mean","state"))
        config.summarised_state_indices = []
        assert config.summarise_states is False

    def test_summarise_observables_property(self):
        """Test summarise_observables property."""
        config = OutputConfig(
            max_states=10,
            max_observables=5,
            summary_types=("mean",)
        )
        assert config.summarise_observables is True

        # No summary types means no summarization
        config.update_from_outputs_tuple(("state",))
        assert config.summarise_observables is False

        # Having summary types but no indices means no summarization
        config.update_from_outputs_tuple(("state",))
        config.summarised_observable_indices = []
        assert config.summarise_observables is False


if __name__ == "__main__":
    pytest.main([__file__])