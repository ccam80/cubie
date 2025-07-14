"""
Tests for OutputConfig class in OutputHandling module.
"""

import pytest
import numpy as np
from typing import Set
from src.CuMC.ForwardSim.OutputHandling.output_config import (
    OutputConfig,
    ArrayHeights,
    parse_string_with_value,
    _ImplementedSummaries
)


class TestParseStringWithValue:
    """Test the parse_string_with_value function."""

    def test_string_with_value(self):
        """Test parsing string with square bracket notation."""
        result = parse_string_with_value("peaks[5]")
        assert result == ("peaks", 5)

    def test_string_with_large_value(self):
        """Test parsing string with large number."""
        result = parse_string_with_value("peaks[1000]")
        assert result == ("peaks", 1000)

    def test_string_without_value(self):
        """Test parsing string without square brackets."""
        result = parse_string_with_value("mean")
        assert result == ("mean", None)

    def test_string_with_extra_spaces(self):
        """Test parsing string with extra spaces."""
        result = parse_string_with_value("  peaks[3]  ")
        assert result == ("peaks", 3)

    def test_empty_string(self):
        """Test parsing empty string."""
        result = parse_string_with_value("")
        assert result == ("", None)


class TestArrayHeights:
    """Test the ArrayHeights class."""

    def test_basic_initialization(self):
        """Test basic ArrayHeights initialization."""
        heights = ArrayHeights(temp=10, output=5)
        assert heights.temp == 10
        assert heights.output == 5
        assert heights._CUDA_allocation_safe == False

    def testCUDA_safe_initialization(self):
        """Test CUDA-safe ArrayHeights initialization."""
        heights = ArrayHeights(temp=0, output=0, CUDA_allocation_safe=True)
        assert heights.temp == 1
        assert heights.output == 1

    def testCUDA_safe_with_nonzero_values(self):
        """Test CUDA-safe with non-zero values."""
        heights = ArrayHeights(temp=5, output=3, CUDA_allocation_safe=True)
        assert heights.temp == 5
        assert heights.output == 3

    def test_default_values(self):
        """Test default values."""
        heights = ArrayHeights()
        assert heights.temp == 0
        assert heights.output == 0
        assert heights._CUDA_allocation_safe == False


class TestOutputConfig:
    """Test the OutputConfig class."""

    def test_minimal_initialization(self):
        """Test minimal valid initialization."""
        config = OutputConfig(max_states=10, max_observables=5)
        assert config.max_states == 10
        assert config.max_observables == 5
        assert config.save_state == True
        assert config.save_observables == False
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
            summary_types={"mean", "max"},
            n_peaks=5
        )
        assert config.max_states == 20
        assert config.max_observables == 10
        assert config.save_state == True
        assert config.save_observables == True
        assert config.save_time == True
        assert config.summary_types == {"mean", "max"}
        assert config.n_peaks == 5

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
        config = OutputConfig(max_states=3, max_observables=2)
        np.testing.assert_array_equal(config.saved_state_indices, np.array([0, 1, 2]))
        np.testing.assert_array_equal(config.saved_observable_indices, np.array([0, 1]))

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
        config = OutputConfig(max_states=10, max_observables=5)
        assert config.save_summaries == False

        config.summary_types = {"mean"}
        assert config.save_summaries == True

    def test_summarise_states_property(self):
        """Test summarise_states property."""
        config = OutputConfig(max_states=10, max_observables=5, summary_types={"mean"})
        assert config.summarise_states == True

        config.summary_types = set()
        assert config.summarise_states == False

    def test_summarise_observables_property(self):
        """Test summarise_observables property."""
        config = OutputConfig(max_states=10, max_observables=5, summary_types={"mean"})
        assert config.summarise_observables == True

        config.summary_types = set()
        assert config.summarise_observables == False

    def test_summarise_peaks_property(self):
        """Test summarise_peaks property."""
        config = OutputConfig(max_states=10, max_observables=5, summary_types={"peaks"}, n_peaks=3)
        assert config.summarise_peaks == True

        config.n_peaks = 0
        assert config.summarise_peaks == False

        config.n_peaks = 3
        config.summary_types = {"mean"}
        assert config.summarise_peaks == False

    def test_individual_summary_properties(self):
        """Test individual summary type properties."""
        config = OutputConfig(max_states=10, max_observables=5, summary_types={"mean", "rms", "max"})
        assert config.summarise_mean == True
        assert config.summarise_rms == True
        assert config.summarise_max == True

        config.summary_types = {"min"}
        assert config.summarise_mean == False
        assert config.summarise_rms == False
        assert config.summarise_max == False

    def test_n_saved_states_property(self):
        """Test n_saved_states property."""
        config = OutputConfig(max_states=10, max_observables=5, save_state=True)
        assert config.n_saved_states == 10

        config.save_state = False
        assert config.n_saved_states == 0

        config.save_state = True
        config.saved_state_indices = np.array([0, 1, 2], dtype=np.int_)
        assert config.n_saved_states == 3

    def test_n_saved_observables_property(self):
        """Test n_saved_observables property."""
        config = OutputConfig(max_states=10, max_observables=5, save_observables=True)
        assert config.n_saved_observables == 5

        config.save_observables = False
        assert config.n_saved_observables == 0

        config.save_observables = True
        config.saved_observable_indices = np.array([0, 1], dtype=np.int_)
        assert config.n_saved_observables == 2

    def test_n_summarised_states_property(self):
        """Test n_summarised_states property."""
        config = OutputConfig(max_states=10, max_observables=5, summary_types={"mean"})
        assert config.n_summarised_states == 10

        config.summary_types = set()
        assert config.n_summarised_states == 0

        config.summary_types = {"mean"}
        config.summarised_state_indices = np.array([0, 1], dtype=np.int_)
        assert config.n_summarised_states == 2

    def test_n_summarised_observables_property(self):
        """Test n_summarised_observables property."""
        config = OutputConfig(max_states=10, max_observables=5, summary_types={"mean"})
        assert config.n_summarised_observables == 5

        config.summary_types = set()
        assert config.n_summarised_observables == 0

        config.summary_types = {"mean"}
        config.summarised_observable_indices = np.array([0], dtype=np.int_)
        assert config.n_summarised_observables == 1

    def test_get_array_sizes(self):
        """Test get_array_sizes method."""
        config = OutputConfig(max_states=10, max_observables=5, summary_types={"mean", "max"}, n_peaks=3)
        sizes = config.get_array_sizes()

        assert isinstance(sizes.state_summaries, ArrayHeights)
        assert isinstance(sizes.observable_summaries, ArrayHeights)
        assert isinstance(sizes.state, ArrayHeights)
        assert isinstance(sizes.observables, ArrayHeights)

    def test_get_array_sizes_with_peaks_and_min1(self):
        """Test get_array_sizes with peaks."""
        #TODO: modify for new 2d return.
        #fixme: fails as checking 1d results.
        config = OutputConfig(max_states=10, max_observables=5, saved_observable_indices = [], summary_types={"peaks"},
                              n_peaks=3)
        sizes = config.get_array_sizes(CUDA_allocation_safe=True)

        # peaks requires 3 + n_peaks temp and n_peaks output
        expected_temp = 3 + 3  # 3 + n_peaks
        expected_output = 3    # n_peaks

        assert sizes.state_summaries.temp == expected_temp * 10  # * n_summarised_states
        assert sizes.state_summaries.output == expected_output * 10

    # def test_get_output_function_settings(self):
    #     """Test get_output_function_settings property."""
    #     config = OutputConfig(max_states=10, max_observables=5, summary_types={"mean"}, n_peaks=3)
    #     settings = config.get_output_function_settings
    #
    #     assert isinstance(settings, dict)
    #     assert "summary_types" in settings
    #     assert "save_state" in settings
    #     assert "save_observables" in settings
    #     assert "save_time" in settings
    #     assert "n_peaks" in settings
    #     assert "saved_states" in settings
    #     assert "saved_observables" in settings
    #     assert "max_states" in settings
    #     assert "max_observables" in settings
    #
    #     assert settings["summary_types"] == {"mean"}
    #     assert settings["n_peaks"] == 3
    #     assert settings["max_states"] == 10
    #     assert settings["max_observables"] == 5


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
        assert config.summary_types == {"mean", "max", "rms"}

    def test_peaks_with_value(self):
        """Test peaks with value specification."""
        config = OutputConfig.from_loop_settings(
            output_types=["peaks[5]"],
            max_states=10,
            max_observables=5
        )
        assert "peaks" in config.summary_types
        assert config.n_peaks == 5

    def test_empty_indices_disable_flags(self):
        """Test that empty indices arrays disable corresponding flags."""
        config = OutputConfig.from_loop_settings(
            output_types=["state", "observables", "mean"],
            saved_states=np.array([], dtype=np.int_),
            saved_observables=np.array([], dtype=np.int_),
            summarised_states = np.array([0], dtype=np.int_),
            max_states=10,
            max_observables=5
        )
        assert config.save_state == False
        assert config.save_observables == False

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
            output_types=["mean"],
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
            output_types=["mean"],
            saved_states=np.array([0, 1, 2], dtype=np.int_),
            saved_observables=np.array([0, 1], dtype=np.int_),
            summarised_states=np.array([0], dtype=np.int_),
            summarised_observables=np.array([1], dtype=np.int_),
            max_states=10,
            max_observables=5
        )
        np.testing.assert_array_equal(config.summarised_state_indices, np.array([0]))
        np.testing.assert_array_equal(config.summarised_observable_indices, np.array([1]))

    def test_empty_summarised_indices_stay_empty(self):
        """Test that empty summarised indices become None."""
        config = OutputConfig.from_loop_settings(
            output_types=["mean"],
            saved_states=np.array([0, 1], dtype=np.int_),
            saved_observables=np.array([0], dtype=np.int_),
            summarised_states=np.array([], dtype=np.int_),
            summarised_observables=np.array([], dtype=np.int_),
            max_states=10,
            max_observables=5
        )
        np.testing.assert_array_equal(config.summarised_state_indices, np.array([], dtype=np.int_))
        np.testing.assert_array_equal(config.summarised_observable_indices, np.array([], dtype=np.int_))

    def test_invalid_output_type_with_value(self):
        """Test invalid output type with value raises error."""
        with pytest.raises(ValueError, match="Invalid output type with value"):
            OutputConfig.from_loop_settings(
                output_types=["invalid[5]"],
                max_states=10,
                max_observables=5
            )

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
        assert config.summary_types == {"mean", "peaks"}
        assert config.n_peaks == 3


class TestOutputConfigValidation:
    """Test validation scenarios."""

    def test_valid_summary_types(self):
        """Test that valid summary types are accepted."""
        for summary_type in _ImplementedSummaries:
            config = OutputConfig(max_states=10, max_observables=5, summary_types={summary_type})
            assert summary_type in config.summary_types

    def test_multiple_summary_types(self):
        """Test multiple summary types."""
        config = OutputConfig(max_states=10, max_observables=5, summary_types={"mean", "max", "rms"})
        assert config.summary_types == {"mean", "max", "rms"}

    def test_empty_summary_types(self):
        """Test empty summary types."""
        config = OutputConfig(max_states=10, max_observables=5, summary_types=set())
        assert len(config.summary_types) == 0
        assert config.save_summaries == False


if __name__ == "__main__":
    pytest.main([__file__])
