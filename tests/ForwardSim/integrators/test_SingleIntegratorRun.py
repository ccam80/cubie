import pytest
import numpy as np
from numba import from_dtype

from CuMC.ForwardSim.integrators.SingleIntegratorRun import SingleIntegratorRun
from CuMC.ForwardSim.OutputHandling.output_functions import OutputFunctions


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
        'output_types':      loop_compile_settings['output_functions'],
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
            output_types=integrator_params['output_types'],
            n_peaks=integrator_params['n_peaks'],
            )


def test_initialization(single_integrator_loop, system, integrator_params, precision):
    """Test that SingleIntegratorRun initializes correctly."""
    # Check that the system is stored correctly
    assert single_integrator_loop._system == system

    # Check that output function parameters are stored (but not yet built)
    assert single_integrator_loop._output_functions is None
    assert single_integrator_loop._output_params['outputs_list'] == integrator_params['output_types']
    assert single_integrator_loop._output_params['saved_states'] is integrator_params['saved_states']
    assert single_integrator_loop._output_params['saved_observables'] is integrator_params['saved_observables']
    assert single_integrator_loop._output_params['n_peaks'] == integrator_params['n_peaks']

    # Check that algorithm parameters are stored correctly
    assert single_integrator_loop._algorithm_params['precision'] == from_dtype(precision)
    assert single_integrator_loop._algorithm_params['n_states'] == system.sizes.states
    assert single_integrator_loop._algorithm_params['n_obs'] == system.sizes.observables
    assert single_integrator_loop._algorithm_params['n_parameters'] == system.sizes.parameters
    assert single_integrator_loop._algorithm_params['n_drivers'] == system.sizes.drivers
    assert single_integrator_loop._algorithm_params['dt_min'] == integrator_params['dt_min']
    assert single_integrator_loop._algorithm_params['dt_max'] == integrator_params['dt_max']
    assert single_integrator_loop._algorithm_params['dt_save'] == integrator_params['dt_save']
    assert single_integrator_loop._algorithm_params['dt_summarise'] == integrator_params['dt_summarise']
    assert single_integrator_loop._algorithm_params['atol'] == integrator_params['atol']
    assert single_integrator_loop._algorithm_params['rtol'] == integrator_params['rtol']
    assert single_integrator_loop._algorithm_params['save_time'] == (
            "time" in integrator_params['output_types'])
    assert single_integrator_loop._algorithm_params['n_saved_states'] == len(integrator_params['saved_states'])
    assert single_integrator_loop._algorithm_params['n_saved_observables'] == len(
            integrator_params['saved_observables'],
            )

    assert single_integrator_loop.algorithm_key == integrator_params['algorithm'].lower()

    # Check that build objects are None before building
    assert single_integrator_loop._integrator_instance is None
    assert single_integrator_loop._compiled_loop is None
    assert single_integrator_loop._loop_cache_valid is False


def test_lazy_output_function_creation(single_integrator_loop):
    """Test that output functions are created lazily when needed."""
    # Initially should be None
    assert single_integrator_loop._output_functions is None

    # Access device_function should trigger creation
    device_func = single_integrator_loop.device_function

    # Now output functions should exist
    assert single_integrator_loop._output_functions is not None
    assert isinstance(single_integrator_loop._output_functions, OutputFunctions)
    assert callable(device_func)


def test_lazy_integrator_creation(single_integrator_loop, precision):
    """Test that the integrator instance is created lazily when needed."""
    # Initially should be None
    assert single_integrator_loop._integrator_instance is None

    # Access device_function should trigger creation
    device_func = single_integrator_loop.device_function

    # Now integrator instance should exist
    assert single_integrator_loop._integrator_instance is not None
    assert callable(device_func)

    # Check that the algorithm object has the correct parameters
    algo_obj = single_integrator_loop._integrator_instance
    assert algo_obj.compile_settings['precision'] == from_dtype(precision)
    assert algo_obj.compile_settings['n_states'] == single_integrator_loop._algorithm_params['n_states']
    assert algo_obj.compile_settings['n_obs'] == single_integrator_loop._algorithm_params['n_obs']
    assert algo_obj.compile_settings['n_parameters'] == single_integrator_loop._algorithm_params['n_parameters']
    assert algo_obj.compile_settings['n_drivers'] == single_integrator_loop._algorithm_params['n_drivers']
    assert algo_obj.compile_settings['dt_min'] == single_integrator_loop._algorithm_params['dt_min']
    assert algo_obj.compile_settings['dt_max'] == single_integrator_loop._algorithm_params['dt_max']
    assert algo_obj.compile_settings['dt_save'] == single_integrator_loop._algorithm_params['dt_save']
    assert algo_obj.compile_settings['dt_summarise'] == single_integrator_loop._algorithm_params['dt_summarise']
    assert algo_obj.compile_settings['atol'] == single_integrator_loop._algorithm_params['atol']
    assert algo_obj.compile_settings['rtol'] == single_integrator_loop._algorithm_params['rtol']
    assert algo_obj.compile_settings['save_time'] == single_integrator_loop._algorithm_params['save_time']
    assert algo_obj.compile_settings['n_saved_states'] == single_integrator_loop._algorithm_params['n_saved_states']
    assert algo_obj.compile_settings['n_saved_observables'] == single_integrator_loop._algorithm_params['n_saved_observables']


def test_output_functions_access(single_integrator_loop):
    """Test that output functions can be accessed and are built correctly."""
    # Initially should be None
    assert single_integrator_loop._output_functions is None

    single_integrator_loop.device_function

    # Now we can access output functions properties
    save_state_func = single_integrator_loop._output_functions.save_state_func
    update_summary_func = single_integrator_loop._output_functions.update_summary_metrics_func
    save_summary_func = single_integrator_loop._output_functions.save_summary_metrics_func

    assert callable(save_state_func)
    assert callable(update_summary_func)
    assert callable(save_summary_func)

    # Check memory requirements
    memory_reqs = single_integrator_loop._output_functions.memory_per_summarised_variable
    assert 'temporary' in memory_reqs
    assert 'output' in memory_reqs
    assert isinstance(memory_reqs['temporary'], int)
    assert isinstance(memory_reqs['output'], int)


def test_build_device_function(single_integrator_loop, precision):
    """Test that build() creates a device function successfully."""
    device_func = single_integrator_loop.build()

    assert callable(device_func)
    assert single_integrator_loop._integrator_instance is not None
    assert single_integrator_loop._compiled_loop is not None
    assert single_integrator_loop._loop_cache_valid is True

    # Test that build() returns the compiled function
    assert device_func is single_integrator_loop._compiled_loop

    # Test that subsequent calls to build() return the same cached version
    device_func2 = single_integrator_loop.device_function
    assert device_func is device_func2  # Should be same object


def test_cache_invalidation(single_integrator_loop):
    """Test that cache is properly invalidated when parameters change."""
    # Build initial version
    device_func1 = single_integrator_loop.device_function
    assert single_integrator_loop._loop_cache_valid is True

    # Update parameters should invalidate cache
    single_integrator_loop.update_parameters(dt_min=0.005)
    assert single_integrator_loop._loop_cache_valid is False

    # Rebuilding should create new version
    device_func2 = single_integrator_loop.device_function
    assert single_integrator_loop._loop_cache_valid is True

    # Should be different functions due to different parameters
    assert device_func1 is not device_func2


def test_parameter_routing(single_integrator_loop, system):
    """Test that parameters are correctly routed to appropriate components."""
    # Test system parameter routing
    if hasattr(system, 'compile_settings') and 'constants' in system.compile_settings:
        original_constants = system.compile_settings['constants']
        single_integrator_loop.update_parameters(constants=original_constants)
        # Should not raise an error

    # Test output function parameter routing
    single_integrator_loop.update_parameters(outputs_list=['state', 'time'])
    assert 'time' in single_integrator_loop._output_params['outputs_list']
    # If already built, should also update the instance
    if single_integrator_loop._output_functions is not None:
        assert 'time' in single_integrator_loop._output_functions.compile_settings['outputs_list']

    # Test algorithm parameter routing
    single_integrator_loop.update_parameters(dt_min=0.001, atol=1e-8)
    assert single_integrator_loop._algorithm_params['dt_min'] == 0.001
    assert single_integrator_loop._algorithm_params['atol'] == 1e-8

    # Test algorithm change
    single_integrator_loop.update_parameters(algorithm='euler')
    assert single_integrator_loop.algorithm_key == 'euler'


def test_cache_valid_property_replaces_is_current(single_integrator_loop):
    """Test that cache_valid property correctly tracks component states."""
    # Initially should be false (nothing built)
    assert single_integrator_loop.cache_valid is False

    # After building, should be true
    single_integrator_loop.build()
    assert single_integrator_loop.cache_valid is True

    # After updating parameters, should be false
    single_integrator_loop.update_parameters(dt_min=0.005)
    assert single_integrator_loop.cache_valid is False


def test_dynamic_memory_calculation(single_integrator_loop, precision):
    """Test that get_dynamic_memory_required returns correct memory amount."""
    # Build the loop first
    single_integrator_loop.build()

    # Get the dynamic memory required
    dynamic_memory = single_integrator_loop.get_dynamic_memory_required()

    # Calculate expected memory
    datasize = int(np.ceil(precision().itemsize))
    n_summary_variables = (single_integrator_loop._algorithm_params['n_saved_states'] +
                           single_integrator_loop._algorithm_params['n_saved_observables'])
    summary_memory_per_variable = single_integrator_loop._output_functions.memory_per_summarised_variable['temporary']
    summary_items_total = n_summary_variables * summary_memory_per_variable

    loop_items = single_integrator_loop._integrator_instance.get_cached_output('loop_shared_memory')

    expected_memory = (loop_items + summary_items_total) * datasize

    assert dynamic_memory == expected_memory


def test_algorithm_change(single_integrator_loop):
    """Test that algorithm can be changed and integrator is recreated."""
    # Build initial version
    single_integrator_loop.build()
    initial_instance = single_integrator_loop._integrator_instance

    # Change algorithm
    single_integrator_loop.update_parameters(algorithm='euler')

    # Instance should be reset to None
    assert single_integrator_loop._integrator_instance is None

    # Building again should create new instance
    single_integrator_loop.build()
    new_instance = single_integrator_loop._integrator_instance

    assert new_instance is not initial_instance


def test_output_function_updates(single_integrator_loop):
    """Test that output function parameters can be updated."""
    # Update output settings before building
    single_integrator_loop.update_parameters(
        outputs_list=['state', 'time', 'mean'],
        n_peaks=5
    )

    # Check that parameters were updated
    assert 'time' in single_integrator_loop._output_params['outputs_list']
    assert 'mean' in single_integrator_loop._output_params['outputs_list']
    assert single_integrator_loop._output_params['n_peaks'] == 5

    # Build should use updated parameters
    single_integrator_loop.build()

    # Check that output functions were created with updated parameters
    assert 'time' in single_integrator_loop._output_functions.compile_settings['outputs_list']
    assert 'mean' in single_integrator_loop._output_functions.compile_settings['outputs_list']
    assert single_integrator_loop._output_functions.compile_settings['n_peaks'] == 5

    # Further updates should update both parameter dict and instance
    single_integrator_loop.update_parameters(n_peaks=10)
    assert single_integrator_loop._output_params['n_peaks'] == 10
    # Cache should be invalidated
    assert single_integrator_loop._loop_cache_valid is False


def test_error_handling(single_integrator_loop):
    """Test error handling for invalid parameters."""
    # Test invalid algorithm
    with pytest.raises(KeyError):
        single_integrator_loop.update_parameters(algorithm='invalid_algorithm')
        single_integrator_loop.build()  # Should fail when trying to create algorithm instance


def test_system_update_propagation(single_integrator_loop, system):
    """Test that system updates properly propagate to the integrator instance."""
    # Build initial version
    device_func1 = single_integrator_loop.device_function
    initial_instance = single_integrator_loop._integrator_instance

    # Update system
    single_integrator_loop.update_parameters(c1= 42.0)

    # The integrator instance should still exist (not destroyed)
    assert single_integrator_loop._integrator_instance is initial_instance
    assert single_integrator_loop._loop_cache_valid is False

    # Getting device function should rebuild with fresh system function
    device_func2 = single_integrator_loop.device_function

    # Should be different functions due to updated system
    assert device_func2 is not device_func1
    # But same integrator instance (preserved)
    assert single_integrator_loop._integrator_instance is initial_instance


def test_output_function_update_propagation(single_integrator_loop):
    """Test that output function updates properly propagate to the integrator instance."""
    # Build initial version
    device_func1 = single_integrator_loop.device_function
    initial_instance = single_integrator_loop._integrator_instance

    # Update output functions (this should NOT destroy the integrator instance)
    single_integrator_loop.update_parameters(outputs_list=['state', 'time', 'mean'])

    # The integrator instance should still exist (not destroyed)
    assert single_integrator_loop._integrator_instance is initial_instance
    assert single_integrator_loop._loop_cache_valid is False

    # Getting device function should rebuild with fresh output functions
    device_func2 = single_integrator_loop.device_function

    # Should be different functions due to updated output functions
    assert device_func2 is not device_func1
    # But same integrator instance (preserved)
    assert single_integrator_loop._integrator_instance is initial_instance


def test_algorithm_change_destroys_instance(single_integrator_loop):
    """Test that algorithm changes still destroy and recreate the integrator instance."""
    # Build initial version
    single_integrator_loop.build()
    initial_instance = single_integrator_loop._integrator_instance

    # Change algorithm (this should still destroy the instance)
    single_integrator_loop.update_parameters(algorithm='euler')

    # Instance should be reset to None for algorithm changes
    assert single_integrator_loop._integrator_instance is None

    # Building again should create new instance
    single_integrator_loop.build()
    new_instance = single_integrator_loop._integrator_instance

    assert new_instance is not initial_instance


def test_fresh_functions_on_every_build(single_integrator_loop):
    """Test that fresh functions are always obtained from dependencies on every build."""
    # Build initial version
    single_integrator_loop.build()

    # Get references to current functions in the algorithm
    initial_dxdt = single_integrator_loop._integrator_instance.compile_settings['dxdt_func']
    initial_save_state = single_integrator_loop._integrator_instance.compile_settings['save_state_func']

    # Invalidate cache to force rebuild
    single_integrator_loop._invalidate_cache()

    # Rebuild should always get fresh functions from dependencies
    single_integrator_loop.build()

    # Functions should be the current ones from the dependencies
    current_dxdt = single_integrator_loop._integrator_instance.compile_settings['dxdt_func']
    current_save_state = single_integrator_loop._integrator_instance.compile_settings['save_state_func']

    # Should be the same as what the dependencies currently provide
    assert current_dxdt is single_integrator_loop._system.device_function
    assert current_save_state is single_integrator_loop._output_functions.save_state_func


def test_cache_valid_property(single_integrator_loop):
    """Test that cache_valid property only tracks our own cache state."""
    # Initially should be false (nothing built)
    assert single_integrator_loop.cache_valid is False

    # After building, should be true
    single_integrator_loop.build()
    assert single_integrator_loop.cache_valid is True

    # After updating parameters, should be false
    single_integrator_loop.update_parameters(dt_min=0.005)
    assert single_integrator_loop.cache_valid is False

    # After rebuilding, should be true again
    single_integrator_loop.build()
    assert single_integrator_loop.cache_valid is True


def test_dependency_functions_always_fresh_without_extra_checking(single_integrator_loop):
    """Test that dependency functions are always fresh due to CUDAFactory design, no extra checking needed."""
    # Build initial version
    single_integrator_loop.build()

    # Simulate system change by directly invalidating system cache
    single_integrator_loop._system._invalidate_cache()

    # Our cache_valid should still be True (we only track our own state)
    assert single_integrator_loop.cache_valid is True

    # But when we rebuild, we'll get fresh functions from the system automatically
    single_integrator_loop._invalidate_cache()  # Force our rebuild
    device_func = single_integrator_loop.build()

    # The functions in the algorithm should be fresh from the system
    assert single_integrator_loop._integrator_instance.compile_settings['dxdt_func'] is single_integrator_loop._system.device_function


def test_no_redundant_dependency_checking_needed(single_integrator_loop):
    """Test that the simplified cache_valid logic works correctly without dependency checking."""
    # Build initial version
    single_integrator_loop.build()
    assert single_integrator_loop.cache_valid is True

    # Change system via our update mechanism
    single_integrator_loop.update_parameters(dt_min=0.001)

    # Our cache should be invalidated
    assert single_integrator_loop.cache_valid is False

    # Rebuild should work correctly and get fresh functions
    device_func = single_integrator_loop.device_function
    assert single_integrator_loop.cache_valid is True

    # The design ensures fresh functions without extra checking


