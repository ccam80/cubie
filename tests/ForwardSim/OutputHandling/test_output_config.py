"""
Tests for OutputConfig class in OutputHandling module.
"""

import pytest
import numpy as np
from typing import Set
from src.CuMC.ForwardSim.OutputHandling.output_config import OutputConfig
from src.CuMC.ForwardSim import summary_metrics


class TestOutputConfig:
    """Test the OutputConfig class."""

    def test_minimal_initialization(self):
        """Test minimal valid initialization."""
        config = OutputConfig(max_states=10, max_observables=5)
        assert config.max_states == 10
        assert config.max_observables == 5
        assert config.save_state == True
        assert config.save_observables == True
        assert config.save_time == False
        assert len(config.summary_types) == 0

    def test_full_initialization(self):
        """Test full initialization with all parameters."""
        config = OutputConfig(
            max_states=20,
            max_observables=10,
            save_state=True,
            save_observables=True,
            save_time=True,
            saved_state_indices=np.array([0, 1, 2], dtype=np.int_),
            saved_observable_indices=np.array([0, 1], dtype=np.int_),
            summarised_state_indices=np.array([0, 1], dtype=np.int_),
            summarised_observable_indices=np.array([0], dtype=np.int_),
            summary_types=("mean", "max"),
        )
        assert config.max_states == 20
        assert config.max_observables == 10
        assert config.save_state == True
        assert config.save_observables == True
        assert config.save_time == True
        assert config.summary_types == ("mean", "max")

    def test_array_conversion(self):
        """Test that indices are properly converted to numpy arrays."""
        config = OutputConfig(
            max_states=10,
            max_observables=5,
            saved_state_indices=[0, 1, 2],
            saved_observable_indices=[0, 1]
        )
        assert isinstance(config.saved_state_indices, np.ndarray)
        assert isinstance(config.saved_observable_indices, np.ndarray)
        assert config.saved_state_indices.dtype == np.int_
        assert config.saved_observable_indices.dtype == np.int_

    def test_none_indices_conversion(self):
        """Test that None indices are converted to full range arrays."""
        config = OutputConfig(max_states=3, max_observables=2, save_state=True, save_observables=True)
        np.testing.assert_array_equal(config.saved_state_indices, np.array([0, 1, 2]))
        np.testing.assert_array_equal(config.saved_observable_indices, np.array([0, 1]))

    def test_indices_emptied_if_no_output(self):
        config = OutputConfig(max_states=3, max_observables=2, saved_state_indices=[0,1,2], saved_observable_indices=[
            0,1], save_state= False, save_observables=False, save_time=True)
        np.testing.assert_array_equal(config.saved_state_indices, np.array([]))
        np.testing.assert_array_equal(config.saved_observable_indices, np.array([]))

    def test_validation_no_output_error(self):
        """Test that error is raised when no output is requested."""
        with pytest.raises(ValueError, match="At least one output type must be enabled"):
            OutputConfig(
                max_states=10,
                max_observables=5,
                save_state=False,
                save_observables=False,
                save_time=False
            )

    def test_save_summaries_property(self):
        """Test save_summaries property."""
        config = OutputConfig(max_states=10, max_observables=5, summary_types=("mean", "max"))
        config.update_from_outputs_tuple(("time",))
        assert config.save_summaries == False

        config.update_from_outputs_tuple(("mean",))
        assert config.save_summaries == True

    def test_summarise_states_property(self):
        """Test summarise_states property."""
        config = OutputConfig(max_states=10, max_observables=5, summary_types=("mean",))
        assert config.summarise_states == True

        config.update_from_outputs_tuple(("time"))
        assert config.summarise_states == False

    def test_summarise_observables_property(self):
        """Test summarise_observables property."""
        config = OutputConfig(max_states=10, max_observables=5, summary_types=("mean",))
        assert config.summarise_observables == True

        config.update_from_outputs_tuple(("time",))
        assert config.summarise_observables == False

    def test_n_saved_states_property(self):
        """Test n_saved_states property."""
        config = OutputConfig(max_states=10, max_observables=5, save_state=True)
        assert config.n_saved_states == 10

        config.update_from_outputs_tuple(("mean",))
        assert config.n_saved_states == 0

        config.update_from_outputs_tuple(("state",))
        config.saved_state_indices = np.array([0, 1, 2], dtype=np.int_)
        assert config.n_saved_states == 3

    def test_n_saved_observables_property(self):
        """Test n_saved_observables property."""
        config = OutputConfig(max_states=10, max_observables=5, save_observables=True)
        assert config.n_saved_observables == 5

        config.update_from_outputs_tuple(("mean",))
        assert config.n_saved_observables == 0

        config.update_from_outputs_tuple(('mean',"observables"))
        config.saved_observable_indices = np.array([0, 1], dtype=np.int_)
        assert config.n_saved_observables == 2

    def test_n_summarised_states_property(self):
        """Test n_summarised_states property."""
        config = OutputConfig(max_states=10, max_observables=5, summary_types=("mean",))
        assert config.n_summarised_states == 10

        config.update_from_outputs_tuple(("time", "state"))
        assert config.n_summarised_states == 0

        config.update_from_outputs_tuple(("mean",))
        config.summarised_state_indices = np.array([0, 1], dtype=np.int_)
        assert config.n_summarised_states == 2

    def test_n_summarised_observables_property(self):
        """Test n_summarised_observables property."""
        config = OutputConfig(max_states=10, max_observables=5, summary_types=("mean",))
        assert config.n_summarised_observables == 5

        config.update_from_outputs_tuple(("time",))
        assert config.n_summarised_observables == 0

        config.update_from_outputs_tuple(("mean",))
        config.summarised_observable_indices = np.array([0], dtype=np.int_)
        assert config.n_summarised_observables == 1


class TestOutputConfigFromLoopSettings:
    """Test the from_loop_settings class method."""

    def test_basic_output_types(self):
        """Test basic output types."""
        config = OutputConfig.from_loop_settings(
            output_types=["state", "observables", "time"],
            max_states=10,
            max_observables=5
        )
        assert config.save_state == True
        assert config.save_observables == True
        assert config.save_time == True
        assert len(config.summary_types) == 0

    def test_summary_types(self):
        """Test summary types."""
        config = OutputConfig.from_loop_settings(
            output_types=["mean", "max", "rms"],
            max_states=10,
            max_observables=5
        )
        assert tuple(config.summary_types) == ("mean", "max", "rms")

    def test_peaks_with_value(self):
        """Test peaks with value specification."""
        config = OutputConfig.from_loop_settings(
            output_types=["peaks[5]",],
            max_states=10,
            max_observables=5
        )
        assert "peaks[5]" in config.summary_types

    def test_provided_indices(self):
        """Test with provided indices."""
        config = OutputConfig.from_loop_settings(
            output_types=["state", "observables"],
            saved_states=np.array([0, 1], dtype=np.int_),
            saved_observables=np.array([0], dtype=np.int_),
            max_states=10,
            max_observables=5
        )
        np.testing.assert_array_equal(config.saved_state_indices, np.array([0, 1]))
        np.testing.assert_array_equal(config.saved_observable_indices, np.array([0]))

    def test_summarised_indices_default_to_saved(self):
        """Test that summarised indices default to saved indices."""
        config = OutputConfig.from_loop_settings(
            output_types=["mean",],
            saved_states=np.array([0, 1], dtype=np.int_),
            saved_observables=np.array([0], dtype=np.int_),
            max_states=10,
            max_observables=5
        )
        np.testing.assert_array_equal(config.summarised_state_indices, np.array([0, 1]))
        np.testing.assert_array_equal(config.summarised_observable_indices, np.array([0]))

    def test_explicit_summarised_indices(self):
        """Test with explicit summarised indices."""
        config = OutputConfig.from_loop_settings(
            output_types=["mean",],
            saved_states=np.array([0, 1, 2], dtype=np.int_),
            saved_observables=np.array([0, 1], dtype=np.int_),
            summarised_states=np.array([0], dtype=np.int_),
            summarised_observables=np.array([1], dtype=np.int_),
            max_states=10,
            max_observables=5
        )
        np.testing.assert_array_equal(config.summarised_state_indices, np.array([0]))
        np.testing.assert_array_equal(config.summarised_observable_indices, np.array([1]))

    def test_mixed_output_types(self):
        """Test mixed output types."""
        config = OutputConfig.from_loop_settings(
            output_types=["state", "mean", "peaks[3]", "observables"],
            max_states=10,
            max_observables=5
        )
        assert config.save_state == True
        assert config.save_observables == True
        assert config.save_time == False
        assert tuple(config.summary_types) == ("mean", "peaks[3]")


class TestOutputConfigValidation:
    """Test validation scenarios."""

    def test_valid_summary_types(self):
        """Test that valid summary types are accepted."""
        for summary_type in summary_metrics._names:
            config = OutputConfig(max_states=10, max_observables=5, summary_types=(summary_type))
            assert summary_type in config.summary_types

    def test_multiple_summary_types(self):
        """Test multiple summary types."""
        config = OutputConfig(max_states=10, max_observables=5, summary_types=("mean", "max", "rms"))
        assert config.summary_types == ("mean", "max", "rms")

    def test_empty_summary_types(self):
        """Test empty summary types."""
        config = OutputConfig(max_states=10, max_observables=5, summary_types=set())
        assert len(config.summary_types) == 0
        assert config.save_summaries == False

