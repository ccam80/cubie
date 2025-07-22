import pytest
import numpy as np
from CuMC.ForwardSim.OutputHandling.output_sizes import ArrayHeights, OutputArraySizes

#TODO: Modify these placeholders from outpuf_configs to test this array sizing module properly
class TestArrayHeights:
    """Test the ArrayHeights class."""

    def test_basic_initialization(self):
        """Test basic ArrayHeights initialization."""
        pass
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


    def test_get_array_sizes(self):
        """Test get_array_sizes method."""
        # config = OutputConfig(max_states=10, max_observables=5, summary_types=("mean", "max"), n_peaks=3)
        # sizes = config.get_array_sizes()
        #
        # assert isinstance(sizes.state_summaries, ArrayHeights)
        # assert isinstance(sizes.observable_summaries, ArrayHeights)
        # assert isinstance(sizes.state, ArrayHeights)
        # assert isinstance(sizes.observables, ArrayHeights)

    def test_get_array_sizes_with_peaks_and_min1(self):
        """Test get_array_sizes with peaks."""
        #TODO: modify for new 2d return.
        #fixme: fails as checking 1d results.
        # config = OutputConfig(max_states=10, max_observables=5, saved_observable_indices = [], summary_types=("peaks"),
        #                       n_peaks=3)
        # sizes = config.get_array_sizes(CUDA_allocation_safe=True)
        #
        # # peaks requires 3 + n_peaks temp and n_peaks output
        # expected_temp = 3 + 3  # 3 + n_peaks
        # expected_output = 3    # n_peaks
        #
        # assert sizes.state_summaries.temp == expected_temp * 10  # * n_summarised_states
        # assert sizes.state_summaries.output == expected_output * 10
