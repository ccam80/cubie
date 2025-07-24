"""
Test suite for output_sizes.py module.
Tests the _nozeros functionality and size calculation classes.
"""

import pytest
import numpy as np
from unittest.mock import Mock
from numba import float32, float64

from CuMC.ForwardSim.OutputHandling.output_sizes import (
    _ensure_nonzero,
    SummariesBufferSizes,
    InnerLoopBufferSizes,
    LoopBufferSizes,
    OutputArrayHeights,
    SingleRunOutputSizes,
    BatchOutputSizes,
    BatchArrays
)
from CuMC.ForwardSim.OutputHandling.output_functions import OutputFunctions
from CuMC.SystemModels.Systems.ODEData import SystemSizes
from CuMC.ForwardSim.integrators.IntegratorRunSettings import IntegatorRunSettings


class TestEnsureNonzero:
    """Test the _ensure_nonzero helper function"""

    def test_ensure_nonzero_false_int(self):
        """Test that when nozeros=False, int values are returned unchanged"""
        assert _ensure_nonzero(0, False) == 0
        assert _ensure_nonzero(5, False) == 5
        assert _ensure_nonzero(-1, False) == -1

    def test_ensure_nonzero_true_int(self):
        """Test that when nozeros=True, zero int values become 1, others unchanged"""
        assert _ensure_nonzero(0, True) == 1
        assert _ensure_nonzero(5, True) == 5
        assert _ensure_nonzero(-1, True) == 1  # max(1, -1) = 1

    def test_ensure_nonzero_false_tuple(self):
        """Test that when nozeros=False, tuple values are returned unchanged"""
        assert _ensure_nonzero((0, 1, 2), False) == (0, 1, 2)
        assert _ensure_nonzero((5, 0, 3), False) == (5, 0, 3)

    def test_ensure_nonzero_true_tuple(self):
        """Test that when nozeros=True, zero tuple values become 1, others unchanged"""
        assert _ensure_nonzero((0, 1, 2), True) == (1, 1, 2)
        assert _ensure_nonzero((5, 0, 3), True) == (5, 1, 3)
        assert _ensure_nonzero((0, 0, 0), True) == (1, 1, 1)

    def test_ensure_nonzero_other_types(self):
        """Test that other types are returned unchanged"""
        test_str = "test"
        assert _ensure_nonzero(test_str, True) == test_str
        assert _ensure_nonzero(test_str, False) == test_str


@pytest.fixture
def mock_output_functions():
    """Create a mock OutputFunctions object for testing"""
    mock = Mock(spec=OutputFunctions)
    mock.state_summaries_output_height = 5
    mock.observable_summaries_output_height = 3
    mock.n_saved_states = 4
    mock.n_saved_observables = 6
    mock.save_time = True
    return mock


@pytest.fixture
def mock_output_functions_with_zeros():
    """Create a mock OutputFunctions object with zero values for testing nozeros"""
    mock = Mock(spec=OutputFunctions)
    mock.state_summaries_output_height = 0
    mock.observable_summaries_output_height = 0
    mock.n_saved_states = 0
    mock.n_saved_observables = 0
    mock.save_time = False
    return mock


@pytest.fixture
def mock_system_sizes():
    """Create a mock SystemSizes object for testing"""
    return SystemSizes(
        states=3,
        observables=4,
        parameters=2,
        constants=1,
        drivers=2
    )


@pytest.fixture
def mock_system_sizes_with_zeros():
    """Create a mock SystemSizes object with zero values for testing nozeros"""
    return SystemSizes(
        states=0,
        observables=0,
        parameters=0,
        constants=0,
        drivers=0
    )


@pytest.fixture
def mock_run_settings():
    """Create a mock IntegratorRunSettings object for testing"""
    mock = Mock(spec=IntegatorRunSettings)
    mock.output_samples = 10
    mock.summarise_samples = 5
    mock.precision = float32
    return mock


@pytest.fixture
def mock_run_settings_with_zeros():
    """Create a mock IntegratorRunSettings object with zero values for testing nozeros"""
    mock = Mock(spec=IntegatorRunSettings)
    mock.output_samples = 0
    mock.summarise_samples = 0
    mock.precision = float32
    return mock


class TestSummariesBufferSizes:
    """Test SummariesBufferSizes class"""

    def test_default_behavior(self, mock_output_functions):
        """Test default behavior without nozeros"""
        sizes = SummariesBufferSizes(mock_output_functions)
        assert sizes.state == 5
        assert sizes.observables == 3
        assert sizes._nozeros == False

    def test_nozeros_false_with_zeros(self, mock_output_functions_with_zeros):
        """Test that zeros are preserved when nozeros=False"""
        sizes = SummariesBufferSizes(mock_output_functions_with_zeros, nozeros=False)
        assert sizes.state == 0
        assert sizes.observables == 0

    def test_nozeros_true_with_zeros(self, mock_output_functions_with_zeros):
        """Test that zeros become ones when nozeros=True"""
        sizes = SummariesBufferSizes(mock_output_functions_with_zeros, nozeros=True)
        assert sizes.state == 1
        assert sizes.observables == 1

    def test_nozeros_true_with_nonzeros(self, mock_output_functions):
        """Test that non-zero values are preserved when nozeros=True"""
        sizes = SummariesBufferSizes(mock_output_functions, nozeros=True)
        assert sizes.state == 5
        assert sizes.observables == 3


class TestInnerLoopBufferSizes:
    """Test InnerLoopBufferSizes class"""

    def test_default_behavior(self, mock_system_sizes):
        """Test default behavior without nozeros"""
        sizes = InnerLoopBufferSizes(mock_system_sizes)
        assert sizes.state == 3
        assert sizes.observables == 4
        assert sizes.dxdt == 3  # should be states, not state
        assert sizes.parameters == 2
        assert sizes.drivers == 2
        assert sizes._nozeros == False

    def test_nozeros_false_with_zeros(self, mock_system_sizes_with_zeros):
        """Test that zeros are preserved when nozeros=False"""
        sizes = InnerLoopBufferSizes(mock_system_sizes_with_zeros, nozeros=False)
        assert sizes.state == 0
        assert sizes.observables == 0
        assert sizes.dxdt == 0
        assert sizes.parameters == 0
        assert sizes.drivers == 0

    def test_nozeros_true_with_zeros(self, mock_system_sizes_with_zeros):
        """Test that zeros become ones when nozeros=True"""
        sizes = InnerLoopBufferSizes(mock_system_sizes_with_zeros, nozeros=True)
        assert sizes.state == 1
        assert sizes.observables == 1
        assert sizes.dxdt == 1
        assert sizes.parameters == 1
        assert sizes.drivers == 1


class TestLoopBufferSizes:
    """Test LoopBufferSizes class"""

    def test_default_behavior(self, mock_system_sizes, mock_output_functions):
        """Test default behavior without nozeros"""
        sizes = LoopBufferSizes(mock_system_sizes, mock_output_functions)

        # Values from summary sizes
        assert sizes.state_summaries == 5
        assert sizes.observable_summaries == 3

        # Values from inner loop sizes
        assert sizes.state == 3
        assert sizes.observables == 4
        assert sizes.dxdt == 3
        assert sizes.parameters == 2
        assert sizes.drivers == 2

    def test_nozeros_propagation(self, mock_system_sizes_with_zeros, mock_output_functions_with_zeros):
        """Test that nozeros setting is properly propagated to sub-components"""
        sizes = LoopBufferSizes(mock_system_sizes_with_zeros, mock_output_functions_with_zeros, nozeros=True)

        # All values should be 1 due to nozeros=True
        assert sizes.state_summaries == 1
        assert sizes.observable_summaries == 1
        assert sizes.state == 1
        assert sizes.observables == 1
        assert sizes.dxdt == 1
        assert sizes.parameters == 1
        assert sizes.drivers == 1


class TestOutputArrayHeights:
    """Test OutputArrayHeights class"""

    def test_default_behavior(self, mock_output_functions):
        """Test default behavior without nozeros"""
        heights = OutputArrayHeights(mock_output_functions)
        assert heights.state == 5  # n_saved_states(4) + 1 * save_time(True)
        assert heights.observables == 6
        assert heights.state_summaries == 5
        assert heights.observable_summaries == 3

    def test_nozeros_true_with_zeros(self, mock_output_functions_with_zeros):
        """Test that zeros become ones when nozeros=True"""
        heights = OutputArrayHeights(mock_output_functions_with_zeros, nozeros=True)
        assert heights.state == 1  # max(1, 0 + 1 * False) = max(1, 0) = 1
        assert heights.observables == 1
        assert heights.state_summaries == 1
        assert heights.observable_summaries == 1


class TestSingleRunOutputSizes:
    """Test SingleRunOutputSizes class"""

    def test_default_behavior(self, mock_output_functions, mock_run_settings):
        """Test default behavior without nozeros"""
        sizes = SingleRunOutputSizes(mock_output_functions, mock_run_settings)

        assert sizes.state == (5, 10)  # (height, output_samples)
        assert sizes.observables == (6, 10)
        assert sizes.state_summaries == (5, 5)  # (height, summarise_samples)
        assert sizes.observable_summaries == (3, 5)

    def test_nozeros_true_with_zeros(self, mock_output_functions_with_zeros, mock_run_settings_with_zeros):
        """Test that zeros become ones when nozeros=True"""
        sizes = SingleRunOutputSizes(mock_output_functions_with_zeros, mock_run_settings_with_zeros, nozeros=True)

        assert sizes.state == (1, 1)
        assert sizes.observables == (1, 1)
        assert sizes.state_summaries == (1, 1)
        assert sizes.observable_summaries == (1, 1)


class TestBatchOutputSizes:
    """Test BatchOutputSizes class"""

    def test_default_behavior(self, mock_output_functions, mock_run_settings):
        """Test default behavior without nozeros"""
        single_run_sizes = SingleRunOutputSizes(mock_output_functions, mock_run_settings)
        batch_sizes = BatchOutputSizes(single_run_sizes, numruns=3)

        assert batch_sizes.state == (5, 3, 10)  # (height, numruns, samples)
        assert batch_sizes.observables == (6, 3, 10)
        assert batch_sizes.state_summaries == (5, 3, 5)
        assert batch_sizes.observable_summaries == (3, 3, 5)

    def test_nozeros_true_with_zeros(self, mock_output_functions_with_zeros, mock_run_settings_with_zeros):
        """Test that zeros become ones when nozeros=True"""
        single_run_sizes = SingleRunOutputSizes(mock_output_functions_with_zeros, mock_run_settings_with_zeros, nozeros=True)
        batch_sizes = BatchOutputSizes(single_run_sizes, numruns=0, nozeros=True)

        assert batch_sizes.state == (1, 1, 1)
        assert batch_sizes.observables == (1, 1, 1)
        assert batch_sizes.state_summaries == (1, 1, 1)
        assert batch_sizes.observable_summaries == (1, 1, 1)

    def test_numruns_zero_handling(self, mock_output_functions, mock_run_settings):
        """Test specific case where only numruns is zero"""
        single_run_sizes = SingleRunOutputSizes(mock_output_functions, mock_run_settings)
        batch_sizes = BatchOutputSizes(single_run_sizes, numruns=0, nozeros=True)

        # numruns should become 1, but other dimensions should remain from single_run_sizes
        assert batch_sizes.state == (5, 1, 10)
        assert batch_sizes.observables == (6, 1, 10)


class TestBatchArrays:
    """Test BatchArrays class"""

    def test_from_output_functions_and_run_settings_default(self, mock_output_functions, mock_run_settings):
        """Test creating BatchArrays from output functions and run settings"""
        batch_arrays = BatchArrays.from_output_functions_and_run_settings(
            mock_output_functions, mock_run_settings, numruns=2
        )

        assert batch_arrays._sizes.state == (5, 2, 10)
        assert batch_arrays._sizes.observables == (6, 2, 10)
        assert batch_arrays._precision == float32

    def test_from_output_functions_and_run_settings_with_nozeros(self, mock_output_functions_with_zeros, mock_run_settings_with_zeros):
        """Test creating BatchArrays with nozeros=True"""
        batch_arrays = BatchArrays.from_output_functions_and_run_settings(
            mock_output_functions_with_zeros, mock_run_settings_with_zeros,
            numruns=0, nozeros=True
        )

        assert batch_arrays._sizes.state == (1, 1, 1)
        assert batch_arrays._sizes.observables == (1, 1, 1)

    def test_allocate_new(self, mock_output_functions, mock_run_settings):
        """Test that _allocate_new creates arrays with correct shapes"""
        batch_arrays = BatchArrays.from_output_functions_and_run_settings(
            mock_output_functions, mock_run_settings, numruns=2
        )

        batch_arrays._allocate_new()

        assert batch_arrays.state.shape == (5, 2, 10)
        assert batch_arrays.observables.shape == (6, 2, 10)
        assert batch_arrays.state_summaries.shape == (5, 2, 5)
        assert batch_arrays.observable_summaries.shape == (3, 2, 5)
        # assert batch_arrays.state.dtype == float32


class TestIntegrationScenarios:
    """Test realistic integration scenarios"""

    def test_realistic_scenario_no_zeros(self):
        """Test a realistic scenario with typical non-zero values"""
        # Create realistic mock objects
        output_functions = Mock(spec=OutputFunctions)
        output_functions.state_summaries_output_height = 10
        output_functions.observable_summaries_output_height = 5
        output_functions.n_saved_states = 3
        output_functions.n_saved_observables = 2
        output_functions.save_time = True

        system_sizes = SystemSizes(
            states=3, observables=2, parameters=5, constants=2, drivers=1
        )

        run_settings = Mock(spec=IntegatorRunSettings)
        run_settings.output_samples = 100
        run_settings.summarise_samples = 20
        run_settings.precision = float64

        # Test the full chain
        single_run = SingleRunOutputSizes(output_functions, run_settings, nozeros=False)
        batch = BatchOutputSizes(single_run, numruns=50, nozeros=False)

        assert single_run.state == (4, 100)  # 3 + 1*True = 4 states, 100 samples
        assert single_run.observables == (2, 100)

        assert batch.state == (4, 50, 100)
        assert batch.observables == (2, 50, 100)

    def test_edge_case_all_zeros_with_nozeros(self):
        """Test edge case where everything is zero but nozeros=True"""
        # Create all-zero mock objects
        output_functions = Mock(spec=OutputFunctions)
        output_functions.state_summaries_output_height = 0
        output_functions.observable_summaries_output_height = 0
        output_functions.n_saved_states = 0
        output_functions.n_saved_observables = 0
        output_functions.save_time = False

        system_sizes = SystemSizes(
            states=0, observables=0, parameters=0, constants=0, drivers=0
        )

        run_settings = Mock(spec=IntegatorRunSettings)
        run_settings.output_samples = 0
        run_settings.summarise_samples = 0
        run_settings.precision = float32

        # Test with nozeros=True - everything should become 1
        single_run = SingleRunOutputSizes(output_functions, run_settings, nozeros=True)
        batch = BatchOutputSizes(single_run, numruns=0, nozeros=True)
        loop_buffer = LoopBufferSizes(system_sizes, output_functions, nozeros=True)

        assert single_run.state == (1, 1)
        assert single_run.observables == (1, 1)

        assert batch.state == (1, 1, 1)
        assert batch.observables == (1, 1, 1)

        assert loop_buffer.state == 1
        assert loop_buffer.observables == 1
        assert loop_buffer.parameters == 1
