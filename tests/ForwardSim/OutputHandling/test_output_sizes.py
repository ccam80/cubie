"""
Test suite for output_sizes.py module.
Tests the _nozeros functionality and size calculation classes using fixtures.
"""

import pytest
import numpy as np
from numba import float32, float64
import attrs

from CuMC.ForwardSim.OutputHandling.output_sizes import (
    _ensure_nonzero,
    SummariesBufferSizes,
    LoopBufferSizes,
    OutputArrayHeights,
    SingleRunOutputSizes,
    BatchOutputSizes,
    BatchArrays
    )


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


# Fixtures for run settings
@pytest.fixture(scope="function")
def run_settings_override(request):
    """Override for run settings, if provided."""
    return request.param if hasattr(request, 'param') else {}

@attrs.define
class IntegratorRunSettings:
    output_samples: int = attrs.field(default=1000, metadata={'description': 'Number of output samples'})
    summarise_samples: int = attrs.field(default=100, metadata={'description': 'Number of samples to summarise'})
    duration: float = attrs.field(default=10.0, metadata={'description': 'Total simulation duration'})

@pytest.fixture(scope="function")
def run_settings(run_settings_override):
    """Create a dictionary of run settings for testing"""
    default_settings = IntegratorRunSettings()
    for key, value in run_settings_override.items():
        if hasattr(default_settings, key):
            setattr(default_settings, key, value)
    return default_settings


class TestSummariesBufferSizes:
    """Test SummariesBufferSizes class"""

    @pytest.mark.parametrize("test_data", [
        pytest.param({'state': 5, 'observables': 3, 'nozeros': False}, id="normal_values"),
        pytest.param({'state': 0, 'observables': 0, 'nozeros': False}, id="zeros_no_fix"),
        pytest.param({'state': 0, 'observables': 0, 'nozeros': True}, id="zeros_with_fix"),
        pytest.param({'state': 10, 'observables': 7, 'nozeros': True}, id="normal_with_fix"),
    ])
    def test_explicit_initialization(self, test_data):
        """Test explicit initialization of SummariesBufferSizes"""
        sizes = SummariesBufferSizes(**test_data)
        if test_data['nozeros'] and test_data['state'] == 0:
            assert sizes.state == 1
            assert sizes.observables == 1
        else:
            assert sizes.state == test_data['state']
            assert sizes.observables == test_data['observables']
        assert sizes._nozeros == test_data['nozeros']

    @pytest.mark.parametrize("loop_compile_settings_overrides",
                             [{'output_functions': ["time", "state", "observables", "mean"]}]
                             , indirect=True)
    def test_from_output_fns_default(self, output_functions):
        """Test creating SummariesBufferSizes from output_functions"""
        sizes = SummariesBufferSizes.from_output_fns(output_functions)

        assert sizes.state == output_functions.state_summaries_buffer_height
        assert sizes.observables == output_functions.observable_summaries_buffer_height
        assert sizes._nozeros == False

    @pytest.mark.parametrize("loop_compile_settings_overrides", [
        {'output_functions': ["time", "state", "observables", "mean"], 'saved_states': [], 'saved_observables': []}
    ], indirect=True)
    def test_from_output_fns_with_nozeros(self, output_functions):
        """Test creating SummariesBufferSizes with nozeros=True"""
        sizes = SummariesBufferSizes.from_output_fns(output_functions, nozeros=True)

        # When output_functions is empty, buffer heights should be 0, but nozeros should make them 1
        expected_state = max(1, output_functions.state_summaries_buffer_height)
        expected_observables = max(1, output_functions.observable_summaries_buffer_height)

        assert sizes.state == expected_state
        assert sizes.observables == expected_observables
        assert sizes._nozeros == True

    @pytest.mark.parametrize("loop_compile_settings_overrides",
                             [{'output_functions': ["time", "state", "observables", "mean"]}]
                             , indirect=True)
    def test_explicit_vs_from_output_fns(self, output_functions):
        """Test that explicit initialization matches from_output_fns result"""
        from_fns = SummariesBufferSizes.from_output_fns(output_functions)
        explicit = SummariesBufferSizes(
            state=output_functions.state_summaries_buffer_height,
            observables=output_functions.observable_summaries_buffer_height,
            nozeros=False
        )

        assert from_fns.state == explicit.state
        assert from_fns.observables == explicit.observables
        assert from_fns._nozeros == explicit._nozeros


class TestLoopBufferSizes:
    """Test LoopBufferSizes class"""

    @pytest.mark.parametrize("test_data", [
        pytest.param({
            'state_summaries': 5, 'observable_summaries': 3, 'state': 3,
            'observables': 4, 'dxdt': 3, 'parameters': 2, 'drivers': 2,
            'nozeros': False
        }, id="normal_values"),
        pytest.param({
            'state_summaries': 0, 'observable_summaries': 0, 'state': 0,
            'observables': 0, 'dxdt': 0, 'parameters': 0, 'drivers': 0,
            'nozeros': True
        }, id="zeros_with_fix"),
    ])
    def test_explicit_initialization(self, test_data):
        """Test explicit initialization of LoopBufferSizes"""
        sizes = LoopBufferSizes(**test_data)

        if test_data['nozeros']:
            # All values should be at least 1
            assert sizes.state_summaries >= 1
            assert sizes.observable_summaries >= 1
            assert sizes.state >= 1
            assert sizes.observables >= 1
            assert sizes.dxdt >= 1
            assert sizes.parameters >= 1
            assert sizes.drivers >= 1
        else:
            assert sizes.state_summaries == test_data['state_summaries']
            assert sizes.observable_summaries == test_data['observable_summaries']
            assert sizes.state == test_data['state']
            assert sizes.observables == test_data['observables']
            assert sizes.dxdt == test_data['dxdt']
            assert sizes.parameters == test_data['parameters']
            assert sizes.drivers == test_data['drivers']

    @pytest.mark.parametrize("loop_compile_settings_overrides",
                             [{'output_functions': ["time", "state", "observables", "mean"]}]
                             , indirect=True)
    def test_from_system_and_output_fns_default(self, system, output_functions):
        """Test creating LoopBufferSizes from system and output_functions"""
        sizes = LoopBufferSizes.from_system_and_output_fns(system, output_functions)

        assert sizes.state_summaries == output_functions.state_summaries_buffer_height
        assert sizes.observable_summaries == output_functions.observable_summaries_buffer_height
        assert sizes.state == system.sizes.states
        assert sizes.observables == system.sizes.observables
        assert sizes.dxdt == system.sizes.states
        assert sizes.parameters == system.sizes.parameters
        assert sizes.drivers == system.sizes.drivers
        assert sizes._nozeros == False

    @pytest.mark.parametrize("loop_compile_settings_overrides",
                             [{'output_functions': ["time", "state", "observables", "mean"]}]
                             , indirect=True)
    def test_from_system_and_output_fns_with_nozeros(self, system, output_functions):
        """Test creating LoopBufferSizes with nozeros=True"""
        sizes = LoopBufferSizes.from_system_and_output_fns(system, output_functions, nozeros=True)

        # All values should be at least 1
        assert sizes.state_summaries >= 1
        assert sizes.observable_summaries >= 1
        assert sizes.state >= 1
        assert sizes.observables >= 1
        assert sizes.dxdt >= 1
        assert sizes.parameters >= 1
        assert sizes.drivers >= 1
        assert sizes._nozeros == True

    @pytest.mark.parametrize("loop_compile_settings_overrides",
                             [{'output_functions': ["time", "state", "observables", "mean"]}]
                             , indirect=True)
    def test_explicit_vs_from_system_and_output_fns(self, system, output_functions):
        """Test that explicit initialization matches from_system_and_output_fns result"""
        from_fns = LoopBufferSizes.from_system_and_output_fns(system, output_functions)
        explicit = LoopBufferSizes(
            state_summaries=output_functions.state_summaries_buffer_height,
            observable_summaries=output_functions.observable_summaries_buffer_height,
            state=system.sizes.states,
            observables=system.sizes.observables,
            dxdt=system.sizes.states,
            parameters=system.sizes.parameters,
            drivers=system.sizes.drivers,
            nozeros=False
        )

        assert from_fns.state_summaries == explicit.state_summaries
        assert from_fns.observable_summaries == explicit.observable_summaries
        assert from_fns.state == explicit.state
        assert from_fns.observables == explicit.observables
        assert from_fns.dxdt == explicit.dxdt
        assert from_fns.parameters == explicit.parameters
        assert from_fns.drivers == explicit.drivers


class TestOutputArrayHeights:
    """Test OutputArrayHeights class"""

    @pytest.mark.parametrize("test_data", [
        pytest.param({
            'state': 5, 'observables': 6, 'state_summaries': 5,
            'observable_summaries': 3, 'nozeros': False
        }, id="normal_values"),
        pytest.param({
            'state': 0, 'observables': 0, 'state_summaries': 0,
            'observable_summaries': 0, 'nozeros': True
        }, id="zeros_with_fix"),
    ])
    def test_explicit_initialization(self, test_data):
        """Test explicit initialization of OutputArrayHeights"""
        heights = OutputArrayHeights(**test_data)

        if test_data['nozeros']:
            # All values should be at least 1 due to __attrs_post_init__
            assert heights.state >= 1
            assert heights.observables >= 1
            assert heights.state_summaries >= 1
            assert heights.observable_summaries >= 1
        else:
            assert heights.state == test_data['state']
            assert heights.observables == test_data['observables']
            assert heights.state_summaries == test_data['state_summaries']
            assert heights.observable_summaries == test_data['observable_summaries']

    @pytest.mark.parametrize("loop_compile_settings_overrides",
                             [{'output_functions': ["time", "state", "observables", "mean"]}]
                             , indirect=True)
    def test_from_output_fns_default(self, output_functions):
        """Test creating OutputArrayHeights from output_functions"""
        heights = OutputArrayHeights.from_output_fns(output_functions)

        expected_state = output_functions.n_saved_states + (1 if output_functions.save_time else 0)
        assert heights.state == expected_state
        assert heights.observables == output_functions.n_saved_observables
        assert heights.state_summaries == output_functions.state_summaries_output_height
        assert heights.observable_summaries == output_functions.observable_summaries_output_height

    @pytest.mark.parametrize("loop_compile_settings_overrides",
                             [{'output_functions': ["time", "state", "observables", "mean"]}]
                             , indirect=True)
    def test_from_output_fns_with_nozeros(self, output_functions):
        """Test creating OutputArrayHeights with nozeros=True"""
        heights = OutputArrayHeights.from_output_fns(output_functions, nozeros=True)

        # All values should be at least 1
        assert heights.state >= 1
        assert heights.observables >= 1
        assert heights.state_summaries >= 1
        assert heights.observable_summaries >= 1
        assert heights._nozeros == True

    @pytest.mark.parametrize("loop_compile_settings_overrides",
                             [{'output_functions': ["time", "state", "observables", "mean"]}]
                             , indirect=True)
    def test_explicit_vs_from_output_fns(self, output_functions):
        """Test that explicit initialization matches from_output_fns result"""
        from_fns = OutputArrayHeights.from_output_fns(output_functions)

        expected_state = output_functions.n_saved_states + (1 if output_functions.save_time else 0)
        explicit = OutputArrayHeights(
            state=expected_state,
            observables=output_functions.n_saved_observables,
            state_summaries=output_functions.state_summaries_output_height,
            observable_summaries=output_functions.observable_summaries_output_height,
            nozeros=False
        )

        assert from_fns.state == explicit.state
        assert from_fns.observables == explicit.observables
        assert from_fns.state_summaries == explicit.state_summaries
        assert from_fns.observable_summaries == explicit.observable_summaries


class TestSingleRunOutputSizes:
    """Test SingleRunOutputSizes class"""

    @pytest.mark.parametrize("test_data", [
        pytest.param({
            'state': (5, 10), 'observables': (6, 10), 'state_summaries': (5, 5),
            'observable_summaries': (3, 5), 'nozeros': False
        }, id="normal_values"),
        pytest.param({
            'state': (0, 0), 'observables': (0, 0), 'state_summaries': (0, 0),
            'observable_summaries': (0, 0), 'nozeros': True
        }, id="zeros_with_fix"),
    ])
    def test_explicit_initialization(self, test_data):
        """Test explicit initialization of SingleRunOutputSizes"""
        sizes = SingleRunOutputSizes(**test_data)

        if test_data['nozeros']:
            # All tuple values should have elements >= 1
            assert all(v >= 1 for v in sizes.state)
            assert all(v >= 1 for v in sizes.observables)
            assert all(v >= 1 for v in sizes.state_summaries)
            assert all(v >= 1 for v in sizes.observable_summaries)
        else:
            assert sizes.state == test_data['state']
            assert sizes.observables == test_data['observables']
            assert sizes.state_summaries == test_data['state_summaries']
            assert sizes.observable_summaries == test_data['observable_summaries']

    @pytest.mark.parametrize("loop_compile_settings_overrides",
                             [{'output_functions': ["time", "state", "observables", "mean"]}]
                             , indirect=True)
    def test_from_output_fns_and_run_settings_default(self, output_functions, run_settings):
        """Test creating SingleRunOutputSizes from output_functions and run_settings"""
        sizes = SingleRunOutputSizes.from_output_fns_and_run_settings(output_functions, run_settings)

        expected_state_height = output_functions.n_saved_states + (1 if output_functions.save_time else 0)

        assert sizes.state == (expected_state_height, run_settings.output_samples)
        assert sizes.observables == (output_functions.n_saved_observables, run_settings.output_samples)
        assert sizes.state_summaries == (output_functions.state_summaries_output_height,
                                         run_settings.summarise_samples)
        assert sizes.observable_summaries == (output_functions.observable_summaries_output_height,
                                              run_settings.summarise_samples)

    @pytest.mark.parametrize("loop_compile_settings_overrides",
                             [{'output_functions': ["time", "state", "observables", "mean"]}]
                             , indirect=True)
    @pytest.mark.parametrize("run_settings_override", [{'output_samples': 0, 'summarise_samples': 0}], indirect=True)
    def test_from_output_fns_and_run_settings_with_nozeros(self, output_functions, run_settings):
        """Test creating SingleRunOutputSizes with nozeros=True"""
        sizes = SingleRunOutputSizes.from_output_fns_and_run_settings(output_functions, run_settings, nozeros=True)

        # All tuple values should have elements >= 1
        assert all(v >= 1 for v in sizes.state)
        assert all(v >= 1 for v in sizes.observables)
        assert all(v >= 1 for v in sizes.state_summaries)
        assert all(v >= 1 for v in sizes.observable_summaries)
        assert sizes._nozeros == True

    @pytest.mark.parametrize("loop_compile_settings_overrides",
                             [{'output_functions': ["time", "state", "observables", "mean"]}]
                             , indirect=True)
    def test_explicit_vs_from_output_fns_and_run_settings(self, output_functions, run_settings):
        """Test that explicit initialization matches from_output_fns_and_run_settings result"""
        from_fns = SingleRunOutputSizes.from_output_fns_and_run_settings(output_functions, run_settings)

        expected_state_height = output_functions.n_saved_states + (1 if output_functions.save_time else 0)
        explicit = SingleRunOutputSizes(
            state=(expected_state_height, run_settings.output_samples),
            observables=(output_functions.n_saved_observables, run_settings.output_samples),
            state_summaries=(output_functions.state_summaries_output_height, run_settings.summarise_samples),
            observable_summaries=(output_functions.observable_summaries_output_height,
                                  run_settings.summarise_samples),
            nozeros=False
        )

        assert from_fns.state == explicit.state
        assert from_fns.observables == explicit.observables
        assert from_fns.state_summaries == explicit.state_summaries
        assert from_fns.observable_summaries == explicit.observable_summaries


class TestBatchOutputSizes:
    """Test BatchOutputSizes class"""

    @pytest.mark.parametrize("test_data", [
        pytest.param({
            'state': (5, 3, 10), 'observables': (6, 3, 10), 'state_summaries': (5, 3, 5),
            'observable_summaries': (3, 3, 5), 'nozeros': False
        }, id="normal_values"),
        pytest.param({
            'state': (0, 0, 0), 'observables': (0, 0, 0), 'state_summaries': (0, 0, 0),
            'observable_summaries': (0, 0, 0), 'nozeros': True
        }, id="zeros_with_fix"),
    ])
    def test_explicit_initialization(self, test_data):
        """Test explicit initialization of BatchOutputSizes"""
        sizes = BatchOutputSizes(**test_data)

        if test_data['nozeros']:
            # All tuple values should have elements >= 1
            assert all(v >= 1 for v in sizes.state)
            assert all(v >= 1 for v in sizes.observables)
            assert all(v >= 1 for v in sizes.state_summaries)
            assert all(v >= 1 for v in sizes.observable_summaries)
        else:
            assert sizes.state == test_data['state']
            assert sizes.observables == test_data['observables']
            assert sizes.state_summaries == test_data['state_summaries']
            assert sizes.observable_summaries == test_data['observable_summaries']

    @pytest.mark.parametrize("loop_compile_settings_overrides",
                             [{'output_functions': ["time", "state", "observables", "mean"]}]
                             , indirect=True)
    def test_from_output_fns_and_run_settings_default(self, output_functions, run_settings):
        """Test creating BatchOutputSizes from output_functions and run_settings"""
        numruns = 3
        sizes = BatchOutputSizes.from_output_fns_and_run_settings(output_functions, run_settings, numruns)

        expected_state_height = output_functions.n_saved_states + (1 if output_functions.save_time else 0)

        assert sizes.state == (expected_state_height, numruns, run_settings.output_samples)
        assert sizes.observables == (output_functions.n_saved_observables, numruns, run_settings.output_samples)
        assert sizes.state_summaries == (output_functions.state_summaries_output_height, numruns,
                                         run_settings.summarise_samples)
        assert sizes.observable_summaries == (output_functions.observable_summaries_output_height, numruns,
                                              run_settings.summarise_samples)

    @pytest.mark.parametrize("loop_compile_settings_overrides",
                             [{'output_functions': ["time", "state", "observables", "mean"]}]
                             , indirect=True)
    @pytest.mark.parametrize("run_settings_override", [{'output_samples': 0, 'summarise_samples': 0}], indirect=True)
    def test_from_output_fns_and_run_settings_with_nozeros(self, output_functions, run_settings):
        """Test creating BatchOutputSizes with nozeros=True"""
        numruns = 0  # This should also become 1 with nozeros=True
        sizes = BatchOutputSizes.from_output_fns_and_run_settings(output_functions, run_settings, numruns, nozeros=True)

        # All tuple values should have elements >= 1
        assert all(v >= 1 for v in sizes.state)
        assert all(v >= 1 for v in sizes.observables)
        assert all(v >= 1 for v in sizes.state_summaries)
        assert all(v >= 1 for v in sizes.observable_summaries)
        assert sizes._nozeros == True

    @pytest.mark.parametrize("loop_compile_settings_overrides",
                             [{'output_functions': ["time", "state", "observables", "mean"]}]
                             , indirect=True)
    def test_explicit_vs_from_output_fns_and_run_settings(self, output_functions, run_settings):
        """Test that explicit initialization matches from_output_fns_and_run_settings result"""
        numruns = 3
        from_fns = BatchOutputSizes.from_output_fns_and_run_settings(output_functions, run_settings, numruns)

        expected_state_height = output_functions.n_saved_states + (1 if output_functions.save_time else 0)
        explicit = BatchOutputSizes(
            state=(expected_state_height, numruns, run_settings.output_samples),
            observables=(output_functions.n_saved_observables, numruns, run_settings.output_samples),
            state_summaries=(output_functions.state_summaries_output_height, numruns,
                             run_settings.summarise_samples),
            observable_summaries=(output_functions.observable_summaries_output_height, numruns,
                                  run_settings.summarise_samples),
            nozeros=False
        )

        assert from_fns.state == explicit.state
        assert from_fns.observables == explicit.observables
        assert from_fns.state_summaries == explicit.state_summaries
        assert from_fns.observable_summaries == explicit.observable_summaries


class TestBatchArrays:
    """Test BatchArrays class"""

    @pytest.mark.parametrize("loop_compile_settings_overrides",
                             [{'output_functions': ["time", "state", "observables", "mean"]}]
                             , indirect=True)
    def test_from_output_fns_and_run_settings_default(self, output_functions, run_settings):
        """Test creating BatchArrays from output_functions and run_settings"""
        numruns = 2
        batch_arrays = BatchArrays.from_output_fns_and_run_settings(
            output_functions, run_settings, numruns
        )

        expected_state_height = output_functions.n_saved_states + (1 if output_functions.save_time else 0)

        assert batch_arrays.sizes.state == (expected_state_height, numruns, run_settings.output_samples)
        assert batch_arrays.sizes.observables == (output_functions.n_saved_observables, numruns,
                                                  run_settings.output_samples)
        assert batch_arrays._precision == float32
        assert batch_arrays._nozeros == False

    @pytest.mark.parametrize("loop_compile_settings_overrides",
                             [{'output_functions': ["time", "state", "observables", "mean"]}]
                             , indirect=True)
    @pytest.mark.parametrize("run_settings_override", [{'output_samples': 0, 'summarise_samples': 0}], indirect=True)
    def test_from_output_fns_and_run_settings_with_nozeros(self, output_functions, run_settings):
        """Test creating BatchArrays with nozeros=True"""
        numruns = 0
        batch_arrays = BatchArrays.from_output_fns_and_run_settings(
            output_functions, run_settings, numruns, nozeros=True
        )

        # All dimensions should be at least 1
        assert all(v >= 1 for v in batch_arrays.sizes.state)
        assert all(v >= 1 for v in batch_arrays.sizes.observables)
        assert batch_arrays._nozeros == True

    @pytest.mark.parametrize("loop_compile_settings_overrides",
                             [{'output_functions': ["time", "state", "observables", "mean"]}]
                             , indirect=True)
    def test_from_output_fns_and_run_settings_precision(self, output_functions, run_settings):
        """Test creating BatchArrays with different precision"""
        numruns = 2
        batch_arrays = BatchArrays.from_output_fns_and_run_settings(
            output_functions, run_settings, numruns, precision=float64
        )

        assert batch_arrays._precision == float64


class TestIntegrationScenarios:
    """Test realistic integration scenarios"""

    @pytest.mark.parametrize("loop_compile_settings_overrides", [
        {'dt_save': 0.01, 'dt_summarise': 0.1, 'saved_states': [0, 1, 2], 'saved_observables': [0, 1],
         'output_functions': ["time", "state", "observables", "mean"]
         }
    ], indirect=True)
    def test_realistic_scenario_no_zeros(self, output_functions, run_settings):
        """Test a realistic scenario with typical non-zero values"""
        numruns = 50

        # Test the full chain
        single_run = SingleRunOutputSizes.from_output_fns_and_run_settings(output_functions, run_settings)
        batch = BatchOutputSizes.from_output_fns_and_run_settings(output_functions, run_settings, numruns)

        expected_state_height = output_functions.n_saved_states + (1 if output_functions.save_time else 0)

        assert single_run.state == (expected_state_height, run_settings.output_samples)
        assert single_run.observables == (output_functions.n_saved_observables, run_settings.output_samples)

        assert batch.state == (expected_state_height, numruns, run_settings.output_samples)
        assert batch.observables == (output_functions.n_saved_observables, numruns, run_settings.output_samples)

    @pytest.mark.parametrize("loop_compile_settings_overrides", [
        {'output_functions': ["time", "state", "observables", "mean"], 'saved_states': [], 'saved_observables': []}
    ], indirect=True)
    @pytest.mark.parametrize("run_settings_override", [{'output_samples': 0, 'summarise_samples': 0}], indirect=True)
    def test_edge_case_all_zeros_with_nozeros(self, system, output_functions, run_settings):
        """Test edge case where everything is zero but nozeros=True"""
        numruns = 0

        # Test with nozeros=True - everything should become at least 1
        single_run = SingleRunOutputSizes.from_output_fns_and_run_settings(output_functions, run_settings, nozeros=True)
        batch = BatchOutputSizes.from_output_fns_and_run_settings(output_functions, run_settings, numruns, nozeros=True)
        loop_buffer = LoopBufferSizes.from_system_and_output_fns(system, output_functions, nozeros=True)

        assert all(v >= 1 for v in single_run.state)
        assert all(v >= 1 for v in single_run.observables)

        assert all(v >= 1 for v in batch.state)
        assert all(v >= 1 for v in batch.observables)

        assert loop_buffer.state >= 1
        assert loop_buffer.observables >= 1
        assert loop_buffer.parameters >= 1

