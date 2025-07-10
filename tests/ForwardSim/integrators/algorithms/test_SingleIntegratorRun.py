import pytest
import numpy as np
from numba import from_dtype

from CuMC.ForwardSim.integrators.SingleIntegratorRun import SingleIntegratorRun
from OutputFunctions.output_functions import OutputFunctions


@pytest.fixture(scope="function")
def default_integrator_params(system, loop_compile_settings):
    """Default parameters for SingleIntegratorRun."""
    return {
        'system':            system,
        'algorithm':         'euler',
        'saved_states':      loop_compile_settings['saved_states'],
        'saved_observables': loop_compile_settings['saved_observables'],
        'dt_min':            loop_compile_settings['dt_min'],
        'dt_max':            loop_compile_settings['dt_max'],
        'dt_save':           loop_compile_settings['dt_save'],
        'dt_summarise':      loop_compile_settings['dt_summarise'],
        'atol':              loop_compile_settings['atol'],
        'rtol':              loop_compile_settings['rtol'],
        'output_functions':  loop_compile_settings['output_functions'],
        'n_peaks':           loop_compile_settings['n_peaks'],
        }


@pytest.fixture(scope="function")
def algorithm_override(request):
    """Override for integrator parameters."""
    return request.param if hasattr(request, 'param') else {}


@pytest.fixture(scope="function")
def integrator_params(default_integrator_params, algorithm_override):
    """Combine default and override parameters."""
    params = default_integrator_params.copy()
    params.update(algorithm_override)
    return params


@pytest.fixture(scope="function")
def single_integrator_loop(system, integrator_params):
    """Create a SingleIntegratorRun instance with the given parameters."""
    return SingleIntegratorRun(
            system=system,
            algorithm=integrator_params['algorithm'],
            dt_min=integrator_params['dt_min'],
            dt_max=integrator_params['dt_max'],
            dt_save=integrator_params['dt_save'],
            dt_summarise=integrator_params['dt_summarise'],
            atol=integrator_params['atol'],
            rtol=integrator_params['rtol'],
            saved_states=integrator_params['saved_states'],
            saved_observables=integrator_params['saved_observables'],
            output_functions=integrator_params['output_functions'],
            n_peaks=integrator_params['n_peaks'],
            )


def test_initialization(single_integrator_loop, system, integrator_params, precision):
    """Test that SingleIntegratorRun initializes correctly."""
    # Check that the system is stored correctly
    assert single_integrator_loop.system == system

    # Check that output settings are stored correctly
    assert np.array_equal(single_integrator_loop.output_settings['saved_states'], integrator_params['saved_states'])
    assert np.array_equal(single_integrator_loop.output_settings['saved_observables'],
                          integrator_params['saved_observables'],
                          )
    assert single_integrator_loop.output_settings['output_functions'] == integrator_params['output_functions']
    assert single_integrator_loop.output_settings['n_peaks'] == integrator_params['n_peaks']

    # Check that algorithm compile settings are stored correctly
    assert single_integrator_loop.algo_compile_settings['precision'] == from_dtype(precision)
    assert single_integrator_loop.algo_compile_settings['n_states'] == system.num_states
    assert single_integrator_loop.algo_compile_settings['n_obs'] == system.num_observables
    assert single_integrator_loop.algo_compile_settings['n_par'] == system.num_parameters
    assert single_integrator_loop.algo_compile_settings['n_drivers'] == system.num_drivers
    assert single_integrator_loop.algo_compile_settings['dt_min'] == integrator_params['dt_min']
    assert single_integrator_loop.algo_compile_settings['dt_max'] == integrator_params['dt_max']
    assert single_integrator_loop.algo_compile_settings['dt_save'] == integrator_params['dt_save']
    assert single_integrator_loop.algo_compile_settings['dt_summarise'] == integrator_params['dt_summarise']
    assert single_integrator_loop.algo_compile_settings['atol'] == integrator_params['atol']
    assert single_integrator_loop.algo_compile_settings['rtol'] == integrator_params['rtol']
    assert single_integrator_loop.algo_compile_settings['save_time'] == (
            "time" in integrator_params['output_functions'])
    assert single_integrator_loop.algo_compile_settings['n_saved_states'] == len(integrator_params['saved_states'])
    assert single_integrator_loop.algo_compile_settings['n_saved_observables'] == len(
            integrator_params['saved_observables'],
            )

    assert single_integrator_loop.algorithm_key == integrator_params['algorithm'].lower()

    # Check that build objects are None before building
    assert single_integrator_loop.integrator_instance is None
    assert single_integrator_loop.loop_function is None
    assert single_integrator_loop.output_functions.save_state_func is None


def test_instantiate_loop(single_integrator_loop, precision):
    """Test that _instantiate_loop passes config to loop algorithm."""
    # Build the dxdt function
    dxdt_function = single_integrator_loop._build_dxdt()

    # Instantiate the loop
    single_integrator_loop._instantiate_loop(dxdt_function)

    # Check that integrator_instance is not None after instantiation
    assert single_integrator_loop.integrator_instance is not None

    # Check that the algorithm object has the correct parameters
    algo_obj = single_integrator_loop.integrator_instance
    assert algo_obj.loop_parameters['precision'] == from_dtype(precision)
    assert algo_obj.loop_parameters['n_states'] == single_integrator_loop.algo_compile_settings['n_states']
    assert algo_obj.loop_parameters['n_obs'] == single_integrator_loop.algo_compile_settings['n_obs']
    assert algo_obj.loop_parameters['n_par'] == single_integrator_loop.algo_compile_settings['n_par']
    assert algo_obj.loop_parameters['n_drivers'] == single_integrator_loop.algo_compile_settings['n_drivers']
    assert algo_obj.loop_parameters['dt_min'] == single_integrator_loop.algo_compile_settings['dt_min']
    assert algo_obj.loop_parameters['dt_max'] == single_integrator_loop.algo_compile_settings['dt_max']
    assert algo_obj.loop_parameters['dt_save'] == single_integrator_loop.algo_compile_settings['dt_save']
    assert algo_obj.loop_parameters['dt_summarise'] == single_integrator_loop.algo_compile_settings['dt_summarise']
    assert algo_obj.loop_parameters['atol'] == single_integrator_loop.algo_compile_settings['atol']
    assert algo_obj.loop_parameters['rtol'] == single_integrator_loop.algo_compile_settings['rtol']
    assert algo_obj.loop_parameters['save_time'] == single_integrator_loop.algo_compile_settings['save_time']
    assert algo_obj.loop_parameters['n_saved_states'] == single_integrator_loop.algo_compile_settings['n_saved_states']
    assert algo_obj.loop_parameters['n_saved_observables'] == single_integrator_loop.algo_compile_settings[
        'n_saved_observables']
    assert algo_obj.loop_parameters['summary_temp_memory'] == single_integrator_loop.algo_compile_settings[
        'summary_temp_memory']

def test_build_output_functions(single_integrator_loop):
    """Test that _build_output_functions builds functions, and that algo summary temp memory is updated."""
    returned_output_functions = single_integrator_loop._build_output_functions()

    assert isinstance(returned_output_functions, OutputFunctions)
    assert single_integrator_loop.algo_compile_settings[
               'summary_temp_memory'] == returned_output_functions.temp_memory_requirements


def test_build_dxdt(single_integrator_loop, system):
    """Test that _build_dxdt builds a dxdt function successfully."""
    dxdt_function = single_integrator_loop._build_dxdt()

    assert callable(dxdt_function)




def test_build(single_integrator_loop, precision):
    """Test that build() builds a combined loop kernel of some sort."""
    built_loop_function, dynamic_shared_memory = single_integrator_loop.build()

    assert single_integrator_loop.integrator_instance is not None

    #Default has no summary memory, so this should return the thread shared memory. Default settings use Euler, so
    # we can check against the already-tested loop shared memory.
    datasize = precision().itemsize
    loop_internal_memory = single_integrator_loop.integrator_instance.get_loop_internal_shared_memory()
    assert dynamic_shared_memory == datasize * loop_internal_memory

    # Check that built_loop_function is callable
    assert callable(built_loop_function)

def test_update_and_rebuild_output_functions(single_integrator_loop):
    pass
def test_update_and_rebuild_timesettings(single_integrator_loop):
    pass

def test_update_and_rebuild_loop(single_integrator_loop):
    """Test that update_and_rebuild_loop updates and rebuilds the loop correctly."""
    pass
    #update parameters and check that result has flown through to memory requirements
    # First build the loop
    # single_integrator_loop.build()
    #
    #
    # # Update and rebuild the loop with the new dxdt function
    # rebuilt_loop_function = single_integrator_loop.update_and_rebuild_loop()
    #
    # # Check that rebuilt_loop_function is callable
    # assert callable(rebuilt_loop_function)
    #
    # # Check that the dxdt function in the algorithm object has been updated
    # # Note: We can't directly check this, but we can check that the update_settings method was called
    # # by verifying that the loop function was rebuilt




def test_summary_memory_change(single_integrator_loop):
    """Test that get_dynamic_memory_required returns the correct amount of memory."""
    # First build the loop
    # single_integrator_loop.build()

    # # Get the dynamic memory required
    # dynamic_memory = single_integrator_loop.get_dynamic_memory_required()
    #
    # # Calculate the expected memory
    # datasize = single_integrator_loop.algo_compile_settings['precision']().itemsize
    # n_summary_variables = (single_integrator_loop.algo_compile_settings['n_saved_states'] +
    #                        single_integrator_loop.algo_compile_settings['n_saved_observables'])
    # summary_memory_per_variable = single_integrator_loop.output_functions.temp_memory_requirements
    # summary_items_total = n_summary_variables * summary_memory_per_variable
    # loop_items = single_integrator_loop.integrator_instance.get_loop_internal_shared_memory()
    # expected_memory = (loop_items + summary_items_total) * datasize
    #
    # # Check that the dynamic memory matches the expected memory
    # assert dynamic_memory == expected_memory


@pytest.mark.parametrize("loop_compile_settings_overrides",[
    {'output_functions': ["state", "observables", "mean"]},
    {'output_functions': ["state", "observables", "max"]},
    {'output_functions': ["state", "observables", "rms"]},
    {'output_functions': ["state", "observables", "peaks"], 'n_peaks': 3},
    {'output_functions': ["state", "observables", "time"]},
    ], indirect=True,
                         )
def test_different_output_functions(single_integrator_loop, integrator_params):
    """Test that different output functions configurations work correctly."""
    pass
    # Build the loop
    # built_loop_function, dynamic_shared_memory = single_integrator_loop.build()
    #
    # # Check that built_loop_function is callable
    # assert callable(built_loop_function)
    #
    # # Check that dynamic_shared_memory is greater than 0
    # assert dynamic_shared_memory > 0
    #
    # # Check that the output functions match the expected configuration
    # output_functions = single_integrator_loop.output_functions
    # expected_outputs = integrator_params['output_functions']
    #
    # # Check save_time
    # assert output_functions.save_time == ("time" in expected_outputs)



@pytest.mark.parametrize("precision_override", [
(np.float32, np.float64)
], indirect=True,
)
def test_different_precisions(single_integrator_loop, system, precision, integrator_params):
    """Test that different precision configurations work correctly."""
    # Build the loop
    built_loop_function, dynamic_shared_memory = single_integrator_loop.build()

    # Check that built_loop_function is callable
    assert callable(built_loop_function)

    # Check that the algorithm object has the correct precision
    assert single_integrator_loop.integrator_instance.loop_parameters['precision'] == from_dtype(
    precision)
