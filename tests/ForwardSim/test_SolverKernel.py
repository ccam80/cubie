from CuMC.ForwardSim.BatchConfigurator import BatchConfigurator
from CuMC.ForwardSim.BatchSolverKernel import BatchSolverKernel
from CuMC.ForwardSim.OutputHandling.output_sizes import BatchOutputSizes
from CuMC.ForwardSim._utils import ensure_nonzero_size
from CuMC.ForwardSim.BatchOutputArrays import OutputArrays
import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_equal

@pytest.fixture(scope='function')
def batchconfig_instance(system):
    return BatchConfigurator.from_system(system)

@pytest.fixture(scope='function')
def square_drive(system, solver_settings, precision, request):
    """amplitude 1 square wave, request "cycles" to change default cycles per simulation from 5"""
    if hasattr(request, 'param'):
        if 'cycles' in request.param:
            cycles = request.getattr('cycles', 5)
    else:
        cycles = 5
    numvecs = system.sizes.drivers
    length = int(solver_settings['duration'] // solver_settings['dt_min'])
    driver = np.zeros((length, numvecs), dtype=precision)
    half_period = length//(2 * cycles)

    for i in range(cycles):
        driver[i*half_period:(i+1)*half_period, :] = 1.0

    return driver

@pytest.fixture(scope="function")
def batch_settings(request):
    """Fixture providing default batch settings."""
    defaults = {
        'num_state_vals_0': 2,
        'num_state_vals_1': 0,
        'num_param_vals_0': 2,
        'num_param_vals_1': 0,
        'kind': 'combinatorial',
    }

    if hasattr(request, 'param'):
        for key in request.param:
            if key in defaults:
                # Update only if the key exists in defaults
                defaults[key] = request.param[key]

    return defaults

@pytest.fixture(scope='function')
def batch_request(system, batch_settings):
    """Parametrized batch settings."""
    state_names = list(system.initial_values.names)
    param_names = list(system.parameters.names)
    return {
        state_names[0]: np.arange(batch_settings['num_state_vals_0']),
        state_names[1]: np.arange(batch_settings['num_state_vals_1']),
        param_names[0]: np.arange(batch_settings['num_param_vals_0']),
        param_names[1]: np.arange(batch_settings['num_param_vals_1']),
    }

@pytest.fixture(scope='function')
def batch_input_arrays(batch_request, batch_settings, batchconfig_instance):
    return batchconfig_instance.grid_arrays(batch_request,
                                                 kind=batch_settings['kind'])

def test_kernel_builds(solverkernel):
    """Test that the solver builds without errors."""
    kernelfunc = solverkernel.kernel

def test_run(system, solverkernel, batch_input_arrays, solver_settings, square_drive, batch_settings, precision):
    """Test that the solver can run with the provided inputs and settings."""
    inits, params = batch_input_arrays

    solverkernel.run(duration=solver_settings['duration'],
                 params=params,
                 inits=inits, # debug: inits has no varied parameters
                 forcing_vectors = square_drive,
                 blocksize=solver_settings['blocksize'],
                 stream=solver_settings['stream'],
                 warmup=solver_settings['warmup'])

    # Check that outputs are as expected
    output_length = int(solver_settings['duration'] / solver_settings['dt_save'])
    summaries_length = int(solver_settings['duration'] / solver_settings['dt_summarise'])
    numruns = ((batch_settings['num_state_vals_0'] if batch_settings['num_state_vals_0'] != 0 else 1) *
               (batch_settings['num_state_vals_1'] if batch_settings['num_state_vals_1'] != 0 else 1) *
               (batch_settings['num_param_vals_0'] if batch_settings['num_param_vals_0'] != 0 else 1) *
               (batch_settings['num_param_vals_1'] if batch_settings['num_param_vals_1'] != 0 else 1))

    active_output_arrays = solverkernel.active_output_arrays
    expected_state_output_shape = (output_length, numruns, len(solverkernel.saved_state_indices))
    expected_observables_output_shape = (output_length, numruns, len(solverkernel.saved_observable_indices))
    expected_state_summaries_shape = (summaries_length, numruns, len(solverkernel.summarised_state_indices))
    expected_observable_summaries_shape = (summaries_length, numruns, len(solverkernel.summarised_observable_indices))

    expected_state_output_shape = ensure_nonzero_size(expected_state_output_shape)
    expected_observables_output_shape = ensure_nonzero_size(expected_observables_output_shape)
    expected_state_summaries_shape = ensure_nonzero_size(expected_state_summaries_shape)
    expected_observable_summaries_shape = ensure_nonzero_size(expected_observable_summaries_shape)

    if active_output_arrays.state is False:
        expected_state_output_shape = (1, 1, 1)
    if active_output_arrays.observables is False:
        expected_observables_output_shape = (1, 1, 1)
    if active_output_arrays.state_summaries is False:
        expected_state_summaries_shape = (1, 1, 1)
    if active_output_arrays.observable_summaries is False:
        expected_observable_summaries_shape = (1, 1, 1)

    state = solverkernel.state_dev_array
    observables = solverkernel.observables_dev_array
    state_summaries = solverkernel.state_summaries_dev_array
    observable_summaries = solverkernel.observable_summaries_dev_array

    assert state.shape == expected_state_output_shape
    assert observables.shape == expected_observables_output_shape
    assert state_summaries.shape == expected_state_summaries_shape
    assert observable_summaries.shape == expected_observable_summaries_shape

    if active_output_arrays.state is True:
        with pytest.raises(AssertionError):
            assert_array_equal(state, np.zeros(state.shape, dtype=precision), err_msg="No output found")
    if active_output_arrays.observables is True:
        with pytest.raises(AssertionError):
            assert_array_equal(observables, np.zeros(observables.shape, dtype=precision), err_msg="No observables output found")
    if active_output_arrays.state_summaries is True:
        with pytest.raises(AssertionError):
            assert_array_equal(state_summaries, np.zeros(state_summaries.shape, dtype=precision), err_msg="No state summaries output "
                                                                                       "found")
    if active_output_arrays.observable_summaries is True:
        with ((pytest.raises(AssertionError))):
            assert_array_equal(observable_summaries, np.zeros(observable_summaries.shape, dtype=precision)),
            err_msg=("No observable summaries output found")


def test_algorithm_change(solverkernel):
    solverkernel.update({'algorithm': 'generic'})
    assert solverkernel.single_integrator._integrator_instance.shared_memory_required == 0

def test_getters_get(solverkernel):
    """Check for dead getters"""
    assert solverkernel.shared_memory_bytes_per_run is not None, ("BatchSolverKernel.shared_memory_bytes_per_run returning None")
    assert solverkernel.shared_memory_elements_per_run is not None, ("BatchSolverKernel.shared_memory_elements_per_run returning None")
    assert solverkernel.precision is not None, ("BatchSolverKernel.precision returning None")
    assert solverkernel.threads_per_loop is not None, ("BatchSolverKernel.threads_per_loop returning None")
    assert solverkernel.output_heights is not None, ("BatchSolverKernel.output_heights returning None")
    assert solverkernel.output_length is not None, ("BatchSolverKernel.output_length returning None")
    assert solverkernel.summaries_length is not None, ("BatchSolverKernel.summaries_length returning None")
    assert solverkernel.num_runs is not None, ("BatchSolverKernel.num_runs returning None")
    assert solverkernel.system is not None, ("BatchSolverKernel.system returning None")
    assert solverkernel.duration is not None, ("BatchSolverKernel.duration returning None")
    assert solverkernel.warmup is not None, ("BatchSolverKernel.warmup returning None")
    assert solverkernel.dt_save is not None, ("BatchSolverKernel.dt_save returning None")
    assert solverkernel.dt_summarise is not None, ("BatchSolverKernel.dt_summarise returning None")
    assert solverkernel.system_sizes is not None, ("BatchSolverKernel.system_sizes returning None")
    assert solverkernel.ouput_array_sizes_2d is not None, ("BatchSolverKernel.ouput_array_sizes_2d returning None")
    assert solverkernel.output_array_sizes_3d is not None, ("BatchSolverKernel.output_array_sizes_3d returning None")
    assert solverkernel.summary_legend_per_variable is not None, ("BatchSolverKernel.summary_legend_per_variable returning None")
    assert solverkernel.saved_state_indices is not None, ("BatchSolverKernel.saved_state_indices returning None")
    assert solverkernel.saved_observable_indices is not None, ("BatchSolverKernel.saved_observable_indices returning None")
    assert solverkernel.summarised_state_indices is not None, ("BatchSolverKernel.summarised_state_indices returning None")
    assert solverkernel.summarised_observable_indices is not None, ("BatchSolverKernel.summarised_observable_indices returning None")
    assert solverkernel.active_output_arrays is not None, ("BatchSolverKernel.active_output_arrays returning None")
    assert solverkernel.state_dev_array is not None, ("BatchSolverKernel.state_dev_array returning None")
    assert solverkernel.observables_dev_array is not None, ("BatchSolverKernel.observables_dev_array returning None")
    assert solverkernel.state_summaries_dev_array is not None, ("BatchSolverKernel.state_summaries_dev_array returning None")
    assert solverkernel.observable_summaries_dev_array is not None, ("BatchSolverKernel.observable_summaries_dev_array returning None")


def test_all_lower_plumbing(system, solverkernel):
    """Big plumbing integration check - check that config classes match exactly between an updated solver and one
    instantiated with the update settings."""
    new_settings = {
        'duration': 1.0,
        'dt_min': 0.0001,
        'dt_max': 0.01,
        'dt_save': 0.01,
        'dt_summarise': 0.1,
        'atol': 1e-2,
        'rtol': 1e-1,
        'saved_state_indices': [0,1,2],
        'saved_observable_indices': [0,1,2],
        'summarised_state_indices': [0,],
        'summarised_observable_indices': [0,],
        'output_types': ["state", "observables", "mean", "max", "rms", "peaks[3]"],
        'precision': np.float64,
    }
    solverkernel.update(new_settings)
    freshsolver = BatchSolverKernel(system,
                                          algorithm='euler',
                                          **new_settings)

    assert freshsolver.compile_settings == solverkernel.compile_settings, "BatchSolverConfig mismatch"
    assert freshsolver.single_integrator.config == solverkernel.single_integrator.config, "IntegratorRunSettings mismatch"
    assert freshsolver.single_integrator._output_functions.compile_settings == \
           solverkernel.single_integrator._output_functions.compile_settings, "OutputFunctions mismatch"
    assert freshsolver.single_integrator._system.compile_settings == \
           solverkernel.single_integrator._system.compile_settings, "SystemCompileSettings mismatch"
    assert BatchOutputSizes.from_solver(freshsolver) == BatchOutputSizes.from_solver(solverkernel), \
        "BatchOutputSizes mismatch"

def test_bogus_update_fails(solverkernel):
    solverkernel.update(dt_min=0.0001)
    with pytest.raises(KeyError):
        solverkernel.update(obviously_bogus_key="this should not work")
