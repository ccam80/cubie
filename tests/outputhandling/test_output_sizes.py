"""
Test suite for output_sizes.py module.
Tests the nonzero functionality and size calculation classes using fixtures.
"""

import pytest
import attrs

from cubie.outputhandling.output_sizes import (
    SummariesBufferSizes,
    LoopBufferSizes,
    OutputArrayHeights,
    SingleRunOutputSizes,
    BatchOutputSizes,
    BatchInputSizes,
    )


class TestNonzeroProperty:
    """Test the nonzero property functionality"""

    def test_nonzero_property_int_values(self):
        """Test that nonzero property converts zero int values to 1"""
        sizes = SummariesBufferSizes(state=0, observables=0, per_variable=5)
        nonzero_sizes = sizes.nonzero

        assert nonzero_sizes.state == 1
        assert nonzero_sizes.observables == 1
        assert nonzero_sizes.per_variable == 5  # unchanged

    def test_nonzero_property_tuple_values(self):
        """Test that nonzero property converts zero tuple values to 1"""
        sizes = SingleRunOutputSizes(
            state=(0, 5),
            observables=(3, 0),
            state_summaries=(0, 0),
            observable_summaries=(2, 4)
        )
        nonzero_sizes = sizes.nonzero

        assert nonzero_sizes.state == (1, 1)
        assert nonzero_sizes.observables == (1, 1)
        assert nonzero_sizes.state_summaries == (1, 1)
        assert nonzero_sizes.observable_summaries == (2, 4)

    def test_nonzero_property_preserves_original(self):
        """Test that nonzero property doesn't modify the original object"""
        original = SummariesBufferSizes(state=0, observables=3, per_variable=0)
        nonzero_copy = original.nonzero

        # Original should be unchanged
        assert original.state == 0
        assert original.observables == 3
        assert original.per_variable == 0

        # Copy should have zeros converted to ones
        assert nonzero_copy.state == 1
        assert nonzero_copy.observables == 3
        assert nonzero_copy.per_variable == 1




@attrs.define
class IntegratorRunSettings:
    output_samples: int = attrs.field(default=1000, metadata={'description': 'Number of output samples'})
    summarise_samples: int = attrs.field(default=100, metadata={'description': 'Number of samples to summarise'})
    duration: float = attrs.field(default=10.0, metadata={'description': 'Total simulation duration'})

# Fixtures for run settings
@pytest.fixture(scope="function")
def run_settings_override(request):
    """Override for run settings, if provided."""
    return request.param if hasattr(request, 'param') else {}

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
        pytest.param({'state': 5, 'observables': 3}, id="normal_values"),
        pytest.param({'state': 0, 'observables': 0}, id="zeros"),
        pytest.param({'state': 10, 'observables': 7}, id="larger_values"),
    ])
    def test_explicit_initialization(self, test_data):
        """Test explicit initialization of SummariesBufferSizes"""
        sizes = SummariesBufferSizes(**test_data)
        assert sizes.state == test_data['state']
        assert sizes.observables == test_data['observables']

    def test_nonzero_functionality(self):
        """Test that nonzero property works correctly"""
        sizes = SummariesBufferSizes(state=0, observables=0, per_variable=5)
        nonzero_sizes = sizes.nonzero

        assert nonzero_sizes.state == 1
        assert nonzero_sizes.observables == 1
        assert nonzero_sizes.per_variable == 5

    @pytest.mark.parametrize("loop_compile_settings_overrides",
                             [{'output_functions': ["time", "state", "observables", "mean"]}]
                             , indirect=True)
    def test_from_output_fns_default(self, output_functions):
        """Test creating SummariesBufferSizes from output_functions"""
        sizes = SummariesBufferSizes.from_output_fns(output_functions)

        assert sizes.state == output_functions.state_summaries_buffer_height
        assert sizes.observables == output_functions.observable_summaries_buffer_height

    @pytest.mark.parametrize("loop_compile_settings_overrides",
                             [{'output_functions': ["time", "state", "observables", "mean"]}]
                             , indirect=True)
    def test_explicit_vs_from_output_fns(self, output_functions):
        """Test that explicit initialization matches from_output_fns result"""
        from_fns = SummariesBufferSizes.from_output_fns(output_functions)
        explicit = SummariesBufferSizes(
            state=output_functions.state_summaries_buffer_height,
            observables=output_functions.observable_summaries_buffer_height,
        )

        assert from_fns.state == explicit.state
        assert from_fns.observables == explicit.observables


class TestLoopBufferSizes:
    """Test LoopBufferSizes class"""

    @pytest.mark.parametrize("test_data", [
        pytest.param({
            'state_summaries': 5, 'observable_summaries': 3, 'state': 3,
            'observables': 4, 'dxdt': 3, 'parameters': 2, 'drivers': 2,
        }, id="normal_values"),
        pytest.param({
            'state_summaries': 0, 'observable_summaries': 0, 'state': 0,
            'observables': 0, 'dxdt': 0, 'parameters': 0, 'drivers': 0,
        }, id="zeros"),
    ])
    def test_explicit_initialization(self, test_data):
        """Test explicit initialization of LoopBufferSizes"""
        sizes = LoopBufferSizes(**test_data)

        assert sizes.state_summaries == test_data['state_summaries']
        assert sizes.observable_summaries == test_data['observable_summaries']
        assert sizes.state == test_data['state']
        assert sizes.observables == test_data['observables']
        assert sizes.dxdt == test_data['dxdt']
        assert sizes.parameters == test_data['parameters']
        assert sizes.drivers == test_data['drivers']

    def test_nonzero_functionality(self):
        """Test that nonzero property works correctly"""
        sizes = LoopBufferSizes(
            state_summaries=0, observable_summaries=0, state=0,
            observables=0, dxdt=0, parameters=0, drivers=0
        )
        nonzero_sizes = sizes.nonzero

        # All values should be at least 1
        assert nonzero_sizes.state_summaries == 1
        assert nonzero_sizes.observable_summaries == 1
        assert nonzero_sizes.state == 1
        assert nonzero_sizes.observables == 1
        assert nonzero_sizes.dxdt == 1
        assert nonzero_sizes.parameters == 1
        assert nonzero_sizes.drivers == 1

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
            'observable_summaries': 3
        }, id="normal_values"),
        pytest.param({
            'state': 0, 'observables': 0, 'state_summaries': 0,
            'observable_summaries': 0
        }, id="zeros"),
    ])
    def test_explicit_initialization(self, test_data):
        """Test explicit initialization of OutputArrayHeights"""
        heights = OutputArrayHeights(**test_data)

        assert heights.state == test_data['state']
        assert heights.observables == test_data['observables']
        assert heights.state_summaries == test_data['state_summaries']
        assert heights.observable_summaries == test_data['observable_summaries']

    def test_nonzero_functionality(self):
        """Test that nonzero property works correctly"""
        heights = OutputArrayHeights(
            state=0, observables=0, state_summaries=0,
            observable_summaries=0, per_variable=0
        )
        nonzero_heights = heights.nonzero

        # All values should be at least 1
        assert nonzero_heights.state == 1
        assert nonzero_heights.observables == 1
        assert nonzero_heights.state_summaries == 1
        assert nonzero_heights.observable_summaries == 1
        assert nonzero_heights.per_variable == 1

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
    def test_explicit_vs_from_output_fns(self, output_functions):
        """Test that explicit initialization matches from_output_fns result"""
        from_fns = OutputArrayHeights.from_output_fns(output_functions)

        expected_state = output_functions.n_saved_states + (1 if output_functions.save_time else 0)
        explicit = OutputArrayHeights(
            state=expected_state,
            observables=output_functions.n_saved_observables,
            state_summaries=output_functions.state_summaries_output_height,
            observable_summaries=output_functions.observable_summaries_output_height,
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
            'observable_summaries': (3, 5)
        }, id="normal_values"),
        pytest.param({
            'state': (0, 0), 'observables': (0, 0), 'state_summaries': (0, 0),
            'observable_summaries': (0, 0)
        }, id="zeros"),
    ])
    def test_explicit_initialization(self, test_data):
        """Test explicit initialization of SingleRunOutputSizes"""
        sizes = SingleRunOutputSizes(**test_data)

        assert sizes.state == test_data['state']
        assert sizes.observables == test_data['observables']
        assert sizes.state_summaries == test_data['state_summaries']
        assert sizes.observable_summaries == test_data['observable_summaries']

    def test_nonzero_functionality(self):
        """Test that nonzero property works correctly"""
        sizes = SingleRunOutputSizes(
            state=(0, 0), observables=(0, 0),
            state_summaries=(0, 0), observable_summaries=(0, 0)
        )
        nonzero_sizes = sizes.nonzero

        # All tuple values should have elements >= 1
        assert all(v >= 1 for v in nonzero_sizes.state)
        assert all(v >= 1 for v in nonzero_sizes.observables)
        assert all(v >= 1 for v in nonzero_sizes.state_summaries)
        assert all(v >= 1 for v in nonzero_sizes.observable_summaries)

    @pytest.mark.parametrize("loop_compile_settings_overrides",
                             [{'output_functions': ["time", "state", "observables", "mean"]}]
                             , indirect=True)
    def test_from_output_fns_and_run_settings_default(self, output_functions, run_settings, solverkernel):
        """Test creating SingleRunOutputSizes from output_functions and run_settings"""
        sizes = SingleRunOutputSizes.from_solver(solverkernel)

        expected_state_height = output_functions.n_saved_states + (1 if output_functions.save_time else 0)

        assert sizes.state == (solverkernel.output_length, expected_state_height)
        assert sizes.observables == (solverkernel.output_length, output_functions.n_saved_observables)
        assert sizes.state_summaries == (solverkernel.summaries_length,
                                         output_functions.state_summaries_output_height)
        assert sizes.observable_summaries == (solverkernel.summaries_length,
                                              output_functions.observable_summaries_output_height
                                              )

    @pytest.mark.parametrize("loop_compile_settings_overrides",
                             [{'output_functions': ["time", "state", "observables", "mean"]}]
                             , indirect=True)
    @pytest.mark.parametrize("solver_settings_override", [{'duration': 0.0}], indirect=True)
    def test_from_solver_with_nonzero(self, solverkernel):
        """Test creating SingleRunOutputSizes and using nonzero property"""
        sizes = SingleRunOutputSizes.from_solver(solverkernel)
        nonzero_sizes = sizes.nonzero

        # All tuple values should have elements >= 1
        assert all(v >= 1 for v in nonzero_sizes.state)
        assert all(v >= 1 for v in nonzero_sizes.observables)
        assert all(v >= 1 for v in nonzero_sizes.state_summaries)
        assert all(v >= 1 for v in nonzero_sizes.observable_summaries)

    @pytest.mark.parametrize("loop_compile_settings_overrides",
                             [{'output_functions': ["time", "state", "observables", "mean"]}]
                             , indirect=True)
    def test_explicit_vs_from_solver(self, output_functions, run_settings, solverkernel):
        """Test that explicit initialization matches from_output_fns_and_run_settings result"""
        from_fns = SingleRunOutputSizes.from_solver(solverkernel)

        expected_state_height = output_functions.n_saved_states + (1 if output_functions.save_time else 0)
        explicit = SingleRunOutputSizes(
                state=(solverkernel.output_length, expected_state_height),
                observables=(solverkernel.output_length, output_functions.n_saved_observables),
                state_summaries=(solverkernel.summaries_length, output_functions.state_summaries_output_height),
                observable_summaries=(solverkernel.summaries_length,
                                      output_functions.observable_summaries_output_height
                                      ),
                )

        assert from_fns.state == explicit.state
        assert from_fns.observables == explicit.observables
        assert from_fns.state_summaries == explicit.state_summaries
        assert from_fns.observable_summaries == explicit.observable_summaries


class TestBatchInputSizes:
    """Test BatchInputSizes class"""

    @pytest.mark.parametrize("test_data", [
        pytest.param({
            'initial_values': (5, 3), 'parameters': (5, 2), 'forcing_vectors': (2, None)
        }, id="normal_values"),
        pytest.param({
            'initial_values': (0, 0), 'parameters': (0, 0), 'forcing_vectors': (0, None)
        }, id="zeros"),
    ])
    def test_explicit_initialization(self, test_data):
        """Test explicit initialization of BatchInputSizes"""
        sizes = BatchInputSizes(**test_data)

        assert sizes.initial_values == test_data['initial_values']
        assert sizes.parameters == test_data['parameters']
        assert sizes.forcing_vectors == test_data['forcing_vectors']

    def test_nonzero_functionality(self):
        """Test that nonzero property works correctly"""
        sizes = BatchInputSizes(
            initial_values=(0, 0), parameters=(0, 0), forcing_vectors=(0, None)
        )
        nonzero_sizes = sizes.nonzero

        # All tuple values should have elements >= 1
        assert all(v >= 1 for v in nonzero_sizes.initial_values)
        assert all(v >= 1 for v in nonzero_sizes.parameters)
        assert nonzero_sizes.forcing_vectors[0] == 1
        assert nonzero_sizes.forcing_vectors[1] == 1

    def test_stride_order_default(self):
        """Test that stride_order has correct default value"""
        sizes = BatchInputSizes()
        assert sizes.stride_order == ("run", "variable")


class TestBatchOutputSizes:
    """Test BatchOutputSizes class"""

    @pytest.mark.parametrize("test_data", [
        pytest.param({
            'state': (5, 3, 10), 'observables': (6, 3, 10), 'state_summaries': (5, 3, 5),
            'observable_summaries': (3, 3, 5)
        }, id="normal_values"),
        pytest.param({
            'state': (0, 0, 0), 'observables': (0, 0, 0), 'state_summaries': (0, 0, 0),
            'observable_summaries': (0, 0, 0)
        }, id="zeros"),
    ])
    def test_explicit_initialization(self, test_data):
        """Test explicit initialization of BatchOutputSizes"""
        sizes = BatchOutputSizes(**test_data)

        assert sizes.state == test_data['state']
        assert sizes.observables == test_data['observables']
        assert sizes.state_summaries == test_data['state_summaries']
        assert sizes.observable_summaries == test_data['observable_summaries']

    def test_nonzero_functionality(self):
        """Test that nonzero property works correctly"""
        sizes = BatchOutputSizes(
            state=(0, 0, 0), observables=(0, 0, 0),
            state_summaries=(0, 0, 0), observable_summaries=(0, 0, 0)
        )
        nonzero_sizes = sizes.nonzero

        # All tuple values should have elements >= 1
        assert all(v >= 1 for v in nonzero_sizes.state)
        assert all(v >= 1 for v in nonzero_sizes.observables)
        assert all(v >= 1 for v in nonzero_sizes.state_summaries)
        assert all(v >= 1 for v in nonzero_sizes.observable_summaries)


class TestIntegrationScenarios:
    """Test realistic integration scenarios"""

    @pytest.mark.parametrize("loop_compile_settings_overrides", [
        {'dt_save': 0.01, 'dt_summarise': 0.1, 'saved_state_indices': [0, 1, 2], 'saved_observable_indices': [0, 1],
         'output_functions': ["time", "state", "observables", "mean"]
         }
    ], indirect=True)
    def test_realistic_scenario_no_zeros(self, output_functions, run_settings, solverkernel):
        """Test a realistic scenario with typical non-zero values"""
        numruns = 1

        # Test the full chain
        single_run = SingleRunOutputSizes.from_solver(solverkernel)
        batch = BatchOutputSizes.from_solver(solverkernel)

        expected_state_height = output_functions.n_saved_states + (1 if output_functions.save_time else 0)

        assert single_run.state == (solverkernel.output_length, expected_state_height)
        assert single_run.observables == (solverkernel.output_length, output_functions.n_saved_observables)

        assert batch.state == (solverkernel.output_length, numruns, expected_state_height)
        assert batch.observables == (solverkernel.output_length, numruns, output_functions.n_saved_observables)

    @pytest.mark.parametrize("loop_compile_settings_overrides", [
        {'output_functions': ["time", "state", "observables", "mean"], 'saved_state_indices': [], 'saved_observable_indices': []}
    ], indirect=True)
    @pytest.mark.parametrize("solver_settings_override", [{'duration': 0.0}], indirect=True)
    def test_edge_case_all_zeros_with_nonzero(self, system, solverkernel, output_functions):
        """Test edge case where everything is zero but using nonzero property"""
        numruns = 0

        # Test with nonzero property - everything should become at least 1
        single_run = SingleRunOutputSizes.from_solver(solverkernel)
        batch = BatchOutputSizes.from_solver(solverkernel)
        loop_buffer = LoopBufferSizes.from_system_and_output_fns(system, output_functions)

        # Use nonzero property to get nonzero versions
        nonzero_single_run = single_run.nonzero
        nonzero_batch = batch.nonzero
        nonzero_loop_buffer = loop_buffer.nonzero

        assert all(v >= 1 for v in nonzero_single_run.state)
        assert all(v >= 1 for v in nonzero_single_run.observables)

        assert all(v >= 1 for v in nonzero_batch.state)
        assert all(v >= 1 for v in nonzero_batch.observables)

        assert nonzero_loop_buffer.state >= 1
        assert nonzero_loop_buffer.observables >= 1
        assert nonzero_loop_buffer.parameters >= 1
