import pytest
import numpy as np
from numba import cuda, int32, float32, float64

from CuMC.ForwardSim.integrators.algorithms.genericIntegratorAlgorithm import GenericIntegratorAlgorithm

# Test configurations for parameterized tests
INIT_TEST_CONFIGS = [
    # (precision, n_states, n_obs, n_par, n_drivers, dt_min, dt_max, dt_save, dt_summarise, atol, rtol, n_saved_states, n_saved_observables, summary_temp_memory, test_name)
    (float32, 3, 2, 5, 1, 0.001, 0.01, 0.01, 0.1, 1e-6, 1e-3, 10, 5, 20, "Basic float32 configuration"),
    (float64, 3, 2, 5, 1, 0.001, 0.01, 0.01, 0.1, 1e-6, 1e-3, 10, 5, 20, "Basic float64 configuration"),
    (float32, 1, 1, 1, 1, 0.001, 0.01, 0.01, 0.1, 1e-6, 1e-3, 1, 1, 5, "Minimal configuration"),
    (float32, 10, 8, 15, 3, 0.0001, 0.1, 0.05, 0.5, 1e-8, 1e-5, 50, 40, 100, "Large configuration"),
    (float32, 5, 0, 10, 2, 0.001, 0.01, 0.01, 0.1, 1e-6, 1e-3, 5, 0, 10, "No observables"),
    (float32, 3, 2, 5, 1, 0.01, 0.001, 0.01, 0.1, 1e-6, 1e-3, 10, 5, 20, "dt_min > dt_max"),
    (float32, 3, 2, 5, 1, 0.001, 0.01, 0.02, 0.1, 1e-6, 1e-3, 10, 5, 20, "dt_save > dt_max"),
]

REBUILD_TEST_CONFIGS = [
    # (initial_params, new_params, test_name)
    # Each tuple contains:
    # 1. Initial parameters (precision, n_states, n_obs, n_par, n_drivers, dt_min, dt_max, dt_save, dt_summarise, atol, rtol, n_saved_states, n_saved_observables, summary_temp_memory)
    # 2. Dictionary of parameters to update
    # 3. should_warn: if there's a bogus key included, it should raise a warning.
    # 3. Test name
    (
        (float32, 3, 2, 5, 1, 0.001, 0.01, 0.01, 0.1, 1e-6, 1e-3, 10, 5, 20),
        {"dt_min": 0.002},
        False,
        "Update dt_min"
    ),
    (
        (float64, 3, 2, 5, 1, 0.001, 0.01, 0.01, 0.1, 1e-6, 1e-3, 10, 5, 20),
        {"dt_max": 0.02, "dt_save": 0.015},
        False,
        "Update dt_max and dt_save"
    ),
    (
        (float32, 3, 2, 5, 1, 0.001, 0.01, 0.01, 0.1, 1e-6, 1e-3, 10, 5, 20),
        {"n_states": 4, "n_obs": 3},
        False,
        "Update n_states and n_obs"
    ),
    (
        (float64, 3, 2, 5, 1, 0.001, 0.01, 0.01, 0.1, 1e-6, 1e-3, 10, 5, 20),
        {"atol": 1e-7, "rtol": 1e-4},
        False,
        "Update tolerances"
    ),
    (
        (float64, 3, 2, 5, 1, 0.001, 0.01, 0.01, 0.1, 1e-6, 1e-3, 10, 5, 20),
        {"n_saved_states": 15, "n_saved_observables": 8, "summary_temp_memory": 30},
        False,
        "Update output parameters"
    ),
    (
        (float64, 3, 2, 5, 1, 0.001, 0.01, 0.01, 0.1, 1e-6, 1e-3, 10, 5, 20),
        {"non_existent_param": 100},
        True,
        "Update non-existent parameter"
    ),
]

# Mock functions for testing
@cuda.jit(device=True, inline=True)
def mock_dxdt_func(state, params, driver, observables, dxdt):
    """Mock dxdt function for testing."""
    for i in range(len(state)):
        dxdt[i] = state[i] + params[0]

@cuda.jit(device=True, inline=True)
def mock_save_state_func(state, observables, state_output, observables_output, time_idx):
    """Mock save state function for testing."""
    for i in range(len(state)):
        state_output[i] = state[i]
    for i in range(len(observables)):
        observables_output[i] = observables[i]

@cuda.jit(device=True, inline=True)
def mock_update_summary_func(state, observables, state_summaries, observables_summaries, time_idx):
    """Mock update summary function for testing."""
    for i in range(len(state)):
        state_summaries[i] = state[i]
    for i in range(len(observables)):
        observables_summaries[i] = observables[i]

@cuda.jit(device=True, inline=True)
def mock_save_summary_func(state_summaries, observables_summaries, state_summaries_output, observables_summaries_output, summarise_every):
    """Mock save summary function for testing."""
    for i in range(len(state_summaries)):
        state_summaries_output[i] = state_summaries[i]
    for i in range(len(observables_summaries)):
        observables_summaries_output[i] = observables_summaries[i]

@pytest.mark.parametrize(
    "precision, n_states, n_obs, n_par, n_drivers, dt_min, dt_max, dt_save, dt_summarise, atol, rtol, n_saved_states, n_saved_observables, summary_temp_memory, test_name",
    INIT_TEST_CONFIGS,
    ids=[config[-1] for config in INIT_TEST_CONFIGS]
)
def test_init(precision, n_states, n_obs, n_par, n_drivers, dt_min, dt_max, dt_save, dt_summarise, atol, rtol, n_saved_states, n_saved_observables, summary_temp_memory, test_name):
    """Test initialization of GenericIntegratorAlgorithm with various parameters."""
    # Create the algorithm
    algorithm = GenericIntegratorAlgorithm(
        precision=precision,
        dxdt_func=mock_dxdt_func,
        n_states=n_states,
        n_obs=n_obs,
        n_par=n_par,
        n_drivers=n_drivers,
        dt_min=dt_min,
        dt_max=dt_max,
        dt_save=dt_save,
        dt_summarise=dt_summarise,
        atol=atol,
        rtol=rtol,
        save_state_func=mock_save_state_func,
        update_summary_func=mock_update_summary_func,
        save_summary_func=mock_save_summary_func,
        n_saved_states=n_saved_states,
        n_saved_observables=n_saved_observables,
        summary_temp_memory=summary_temp_memory
    )
    


    # Check that loop_parameters were set correctly
    assert algorithm.loop_parameters['precision'] == precision
    assert algorithm.loop_parameters['n_states'] == n_states
    assert algorithm.loop_parameters['n_obs'] == n_obs
    assert algorithm.loop_parameters['n_par'] == n_par
    assert algorithm.loop_parameters['n_drivers'] == n_drivers
    assert algorithm.loop_parameters['dt_min'] == dt_min
    assert algorithm.loop_parameters['dt_max'] == dt_max
    assert algorithm.loop_parameters['dt_save'] == dt_save
    assert algorithm.loop_parameters['dt_summarise'] == dt_summarise
    assert algorithm.loop_parameters['atol'] == atol
    assert algorithm.loop_parameters['rtol'] == rtol
    assert algorithm.loop_parameters['n_saved_states'] == n_saved_states
    assert algorithm.loop_parameters['n_saved_observables'] == n_saved_observables
    assert algorithm.loop_parameters['summary_temp_memory'] == summary_temp_memory
    
    # Check that functions were set correctly
    assert algorithm.functions['dxdt_func'] == mock_dxdt_func
    assert algorithm.functions['save_state_func'] == mock_save_state_func
    assert algorithm.functions['update_summary_func'] == mock_update_summary_func
    assert algorithm.functions['save_summary_func'] == mock_save_summary_func

    # Check that the loop was built
    assert algorithm.loop_function is not None

@pytest.mark.parametrize(
    "precision, n_states, n_obs, n_par, n_drivers, dt_min, dt_max, dt_save, dt_summarise, atol, rtol, n_saved_states, n_saved_observables, summary_temp_memory, test_name",
    INIT_TEST_CONFIGS,
    ids=[config[-1] for config in INIT_TEST_CONFIGS]
)
def test_build(precision, n_states, n_obs, n_par, n_drivers, dt_min, dt_max, dt_save, dt_summarise, atol, rtol, n_saved_states, n_saved_observables, summary_temp_memory, test_name):
    """Test the build method of GenericIntegratorAlgorithm."""
    # Create the algorithm with basic parameters

    algorithm = GenericIntegratorAlgorithm(
        precision=precision,
        dxdt_func=mock_dxdt_func,
        n_states=n_states,
        n_obs=n_obs,
        n_par=n_par,
        n_drivers=n_drivers,
        dt_min=dt_min,
        dt_max=dt_max,
        dt_save=dt_save,
        dt_summarise=dt_summarise,
        atol=atol,
        rtol=rtol,
        save_state_func=mock_save_state_func,
        update_summary_func=mock_update_summary_func,
        save_summary_func=mock_save_summary_func,
        n_saved_states=n_saved_states,
        n_saved_observables=n_saved_observables,
        summary_temp_memory=summary_temp_memory
    )

    original_loop_function = algorithm.loop_function
    del algorithm.loop_function
    # Call build method
    algorithm.build()

    assert algorithm.loop_function is not None
    assert callable(algorithm.loop_function)

@pytest.mark.parametrize(
    "precision, n_states, n_obs, n_par, n_drivers, dt_min, dt_max, dt_save, dt_summarise, atol, rtol, n_saved_states, n_saved_observables, summary_temp_memory, test_name",
    INIT_TEST_CONFIGS,
    ids=[config[-1] for config in INIT_TEST_CONFIGS]
)
def test_build_loop(precision, n_states, n_obs, n_par, n_drivers, dt_min, dt_max, dt_save, dt_summarise, atol, rtol, n_saved_states, n_saved_observables, summary_temp_memory, test_name):
    """Test the build_loop method of GenericIntegratorAlgorithm."""
    # Create the algorithm with basic parameters
    algorithm = GenericIntegratorAlgorithm(
        precision=precision,
        dxdt_func=mock_dxdt_func,
        n_states=n_states,
        n_obs=n_obs,
        n_par=n_par,
        n_drivers=n_drivers,
        dt_min=dt_min,
        dt_max=dt_max,
        dt_save=dt_save,
        dt_summarise=dt_summarise,
        atol=atol,
        rtol=rtol,
        save_state_func=mock_save_state_func,
        update_summary_func=mock_update_summary_func,
        save_summary_func=mock_save_summary_func,
        n_saved_states=n_saved_states,
        n_saved_observables=n_saved_observables,
        summary_temp_memory=summary_temp_memory
    )
    
    # Call build_loop method directly
    loop_function = algorithm.build_loop(
        precision=precision,
        dxdt_func=mock_dxdt_func,
        n_states=n_states,
        n_obs=n_obs,
        n_par=n_par,
        n_drivers=n_drivers,
        dt_min=dt_min,
        dt_max=dt_max,
        dt_save=dt_save,
        dt_summarise=dt_summarise,
        atol=atol,
        rtol=rtol,
        save_state_func=mock_save_state_func,
        update_summary_func=mock_update_summary_func,
        save_summary_func=mock_save_summary_func,
        n_saved_states=n_saved_states,
        n_saved_observables=n_saved_observables,
        summary_temp_memory=summary_temp_memory
    )
    
    # Check that a loop function was returned
    assert loop_function is not None
    assert callable(loop_function)

@pytest.mark.parametrize(
    "precision, n_states, n_obs, n_par, n_drivers, dt_min, dt_max, dt_save, dt_summarise, atol, rtol, n_saved_states, n_saved_observables, summary_temp_memory, test_name",
    INIT_TEST_CONFIGS,
    ids=[config[-1] for config in INIT_TEST_CONFIGS]
)
def test_calculate_shared_memory(precision, n_states, n_obs, n_par, n_drivers, dt_min, dt_max, dt_save, dt_summarise, atol, rtol, n_saved_states, n_saved_observables, summary_temp_memory, test_name):
    """Test the calculate_shared_memory method of GenericIntegratorAlgorithm."""
    # Create the algorithm with basic parameters

    algorithm = GenericIntegratorAlgorithm(
        precision=precision,
        dxdt_func=mock_dxdt_func,
        n_states=n_states,
        n_obs=n_obs,
        n_par=n_par,
        n_drivers=n_drivers,
        dt_min=dt_min,
        dt_max=dt_max,
        dt_save=dt_save,
        dt_summarise=dt_summarise,
        atol=atol,
        rtol=rtol,
        save_state_func=mock_save_state_func,
        update_summary_func=mock_update_summary_func,
        save_summary_func=mock_save_summary_func,
        n_saved_states=n_saved_states,
        n_saved_observables=n_saved_observables,
        summary_temp_memory=summary_temp_memory
    )
    
    # Call calculate_shared_memory method
    shared_memory = algorithm._calculate_loop_internal_shared_memory()
    
    # Check that the result is as expected: n_states
    assert shared_memory == n_states

@pytest.mark.parametrize(
    "initial_params, new_params, should_warn, test_name",
    REBUILD_TEST_CONFIGS,
    ids=[config[-1] for config in REBUILD_TEST_CONFIGS]
)
def test_rebuild(initial_params, new_params, should_warn, test_name):
    """Test the rebuild method of GenericIntegratorAlgorithm with various parameter updates."""
    # Unpack initial parameters
    (precision, n_states, n_obs, n_par, n_drivers, dt_min, dt_max, dt_save, 
     dt_summarise, atol, rtol, n_saved_states, n_saved_observables, summary_temp_memory) = initial_params
    
    # Create the algorithm with initial parameters
    algorithm = GenericIntegratorAlgorithm(
        precision=precision,
        dxdt_func=mock_dxdt_func,
        n_states=n_states,
        n_obs=n_obs,
        n_par=n_par,
        n_drivers=n_drivers,
        dt_min=dt_min,
        dt_max=dt_max,
        dt_save=dt_save,
        dt_summarise=dt_summarise,
        atol=atol,
        rtol=rtol,
        save_state_func=mock_save_state_func,
        update_summary_func=mock_update_summary_func,
        save_summary_func=mock_save_summary_func,
        n_saved_states=n_saved_states,
        n_saved_observables=n_saved_observables,
        summary_temp_memory=summary_temp_memory
    )
    
    # Store the original loop function
    original_loop_function = algorithm.loop_function
    
    # Call rebuild method with new parameters
    if should_warn:
        with pytest.warns():
            algorithm.rebuild(**new_params)
    else:
        algorithm.rebuild(**new_params)
    
    # Check that parameters were updated correctly
    for key, value in new_params.items():
        if key in algorithm.loop_parameters:
            assert algorithm.loop_parameters[key] == value
        elif key in algorithm.functions:
            assert algorithm.functions[key] == value


