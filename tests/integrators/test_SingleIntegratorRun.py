import pytest

from cubie.integrators.SingleIntegratorRun import SingleIntegratorRun
from cubie.outputhandling import OutputFunctions
from cubie.integrators.IntegratorRunSettings import IntegratorRunSettings


@pytest.fixture(scope="function")
def default_integrator_params(system, solver_settings):
    """Default parameters for SingleIntegratorRun."""
    return {
        "system": system,
        "algorithm": "explicit_euler",
        "saved_state_indices": list(solver_settings["saved_state_indices"]),
        "saved_observable_indices": list(
            solver_settings["saved_observable_indices"]
        ),
        "summarised_state_indices": (
            list(solver_settings["summarised_state_indices"])
            if solver_settings.get("summarised_state_indices") is not None
            else None
        ),
        "summarised_observable_indices": (
            list(solver_settings["summarised_observable_indices"])
            if solver_settings.get("summarised_observable_indices")
            is not None
            else None
        ),
        "dt_min": solver_settings["dt_min"],
        "dt_max": solver_settings["dt_max"],
        "dt_save": solver_settings["dt_save"],
        "dt_summarise": solver_settings["dt_summarise"],
        "atol": solver_settings["atol"],
        "rtol": solver_settings["rtol"],
        "output_types": list(solver_settings["output_types"]),
    }


@pytest.fixture(scope="function")
def algorithm_override(request):
    """Override for integrator parameters."""
    return request.param if hasattr(request, "param") else {}


@pytest.fixture(scope="function")
def integrator_params(default_integrator_params, algorithm_override):
    """Combine default and override parameters."""
    params = default_integrator_params.copy()
    params.update(algorithm_override)
    return params


@pytest.fixture(scope="function")
def single_integrator_run(system, integrator_params):
    """Create a SingleIntegratorRun instance with the given parameters."""
    return SingleIntegratorRun(
        system=integrator_params["system"],
        algorithm=integrator_params["algorithm"],
        dt_min=integrator_params["dt_min"],
        dt_max=integrator_params["dt_max"],
        dt_save=integrator_params["dt_save"],
        dt_summarise=integrator_params["dt_summarise"],
        atol=integrator_params["atol"],
        rtol=integrator_params["rtol"],
        saved_state_indices=integrator_params["saved_state_indices"],
        saved_observable_indices=integrator_params["saved_observable_indices"],
        summarised_state_indices=integrator_params["summarised_state_indices"],
        summarised_observable_indices=integrator_params[
            "summarised_observable_indices"
        ],
        output_types=integrator_params["output_types"],
    )


def test_initialization(single_integrator_run, system, integrator_params):
    """Test that SingleIntegratorRun initializes correctly."""
    # Check that the system is stored correctly
    assert single_integrator_run._system == system

    # Check that config object is created correctly
    assert isinstance(single_integrator_run.config, IntegratorRunSettings)
    assert single_integrator_run.config.dt_min == integrator_params["dt_min"]
    assert single_integrator_run.config.dt_max == integrator_params["dt_max"]
    assert single_integrator_run.config.dt_save == integrator_params["dt_save"]
    assert (
        single_integrator_run.config.dt_summarise
        == integrator_params["dt_summarise"]
    )
    assert single_integrator_run.config.atol == integrator_params["atol"]
    assert single_integrator_run.config.rtol == integrator_params["rtol"]
    assert single_integrator_run.config.step_controller_kind == "fixed"

    # Check that output functions are created immediately
    assert single_integrator_run._output_functions is not None
    assert isinstance(single_integrator_run._output_functions, OutputFunctions)

    # Check algorithm key
    assert (
        single_integrator_run.algorithm_key
        == integrator_params["algorithm"].lower()
    )

    # Check that loop dependencies are created
    assert single_integrator_run._loop is not None
    assert single_integrator_run._step_controller is not None
    assert single_integrator_run._algo_step is not None

    # Check that compiled loop starts as None
    assert single_integrator_run._compiled_loop is None
    assert single_integrator_run._loop_cache_valid is False


def test_output_functions_immediate_creation(single_integrator_run, system):
    """Output functions exist immediately after initialisation."""
    # Output functions should exist immediately after initialization
    assert single_integrator_run._output_functions is not None
    assert isinstance(
        single_integrator_run._output_functions, OutputFunctions
    )

    # Should have correct system dimensions
    system_sizes = system.sizes
    compile_settings = single_integrator_run._output_functions.compile_settings
    assert compile_settings.max_states == system_sizes.states
    assert compile_settings.max_observables == system_sizes.observables
def test_property_access(single_integrator_run):
    """Test that properties can be accessed correctly."""
    # Test buffer sizes
    buffer_sizes = single_integrator_run.loop_buffer_sizes
    assert buffer_sizes is not None

    # Test output array heights
    array_heights = single_integrator_run.output_array_heights
    assert array_heights is not None

    # Test summaries_array buffer sizes
    summaries_sizes = single_integrator_run.summaries_buffer_sizes
    assert summaries_sizes is not None

    # Test precision
    precision = single_integrator_run.precision
    assert precision is not None

    # Test threads per loop
    threads = single_integrator_run.threads_per_loop
    assert isinstance(threads, int)
    assert threads > 0


def test_memory_requirements(single_integrator_run):
    """Shared and local memory calculations include dependencies."""

    base_shared = single_integrator_run.config.loop_shared_elements
    algo_shared = single_integrator_run._algo_step.shared_memory_required
    assert single_integrator_run.shared_memory_elements == base_shared + algo_shared
    assert (
        single_integrator_run.local_memory_elements
        == single_integrator_run.config.total_local_elements
    )


def test_function_access_properties(single_integrator_run):
    """Test that function access properties work correctly."""
    # Test dxdt function
    dxdt_function = single_integrator_run.dxdt_function
    assert callable(dxdt_function)

    # Test save state function
    save_state_func = single_integrator_run.save_state_func
    assert callable(save_state_func)

    # Test update summaries_array function
    update_summaries_func = single_integrator_run.update_summaries_func
    assert callable(update_summaries_func)

    # Test save summaries_array function
    save_summaries_func = single_integrator_run.save_summaries_func
    assert callable(save_summaries_func)

def test_build_device_function(single_integrator_run):
    """Test that build() creates a device function successfully."""
    device_func = single_integrator_run.build()

    assert callable(device_func)
    assert single_integrator_run._loop is not None
    assert single_integrator_run._compiled_loop is not None
    assert single_integrator_run._loop_cache_valid is True

    # Test that build() returns the compiled function
    assert device_func is single_integrator_run._compiled_loop

    # Test that subsequent calls to build() return the same cached version
    device_func2 = single_integrator_run.device_function
    assert device_func is device_func2  # Should be same object


def test_cache_invalidation(single_integrator_run):
    """Test that cache is properly invalidated when parameters change."""
    # Build initial version
    device_func1 = single_integrator_run.device_function
    assert single_integrator_run._loop_cache_valid is True

    # Update parameters should invalidate cache
    single_integrator_run.update(dt_min=0.005)
    assert single_integrator_run._loop_cache_valid is False

    # Rebuilding should create new version
    device_func2 = single_integrator_run.device_function
    assert single_integrator_run._loop_cache_valid is True

    # Should be different functions due to different parameters
    assert device_func1 is not device_func2


def test_parameter_updates_config_object(single_integrator_run):
    """Test that parameter updates correctly modify the config object."""
    original_dt_min = single_integrator_run.config.dt_min
    original_atol = single_integrator_run.config.atol

    # Update parameters
    single_integrator_run.update(dt_min=0.005, atol=1e-8)

    # Check that config object was updated
    assert single_integrator_run.config.dt_min == 0.005
    assert single_integrator_run.config.atol == 1e-8
    assert single_integrator_run.config.dt_min != original_dt_min
    assert single_integrator_run.config.atol != original_atol


def test_update_cache_invalidation(single_integrator_run, system):
    """Test that parameters are correctly routed to appropriate components."""
    # Test system parameter routing
    original_cache_valid = single_integrator_run.cache_valid
    single_integrator_run.update(c1=42.0)  # Assuming c1 is a system parameter

    # Should invalidate cache when system is updated
    assert (
        single_integrator_run.cache_valid != original_cache_valid
        or not original_cache_valid
    )

    # Test output function parameter routing
    single_integrator_run.update(output_types=["state", "time"])

    # Should invalidate cache
    assert single_integrator_run._loop_cache_valid is False


def test_algorithm_change(single_integrator_run):
    """Test that algorithm can be changed and integrator is recreated."""
    # Build initial version
    single_integrator_run.build()
    initial_algo = single_integrator_run._algo_step

    # Change algorithm
    single_integrator_run.update(algorithm="backwards_euler")

    # Should have new instance
    assert single_integrator_run._algo_step is not initial_algo

    # Algorithm key should be updated
    assert single_integrator_run.algorithm_key == "backwards_euler"


def test_cache_valid_property(single_integrator_run):
    """Test that cache_valid property correctly tracks component states."""
    # Initially should be false (nothing built)
    assert single_integrator_run.cache_valid is False

    # After building, should be true
    single_integrator_run.build()
    assert single_integrator_run.cache_valid is True

    # After updating parameters, should be false
    single_integrator_run.update(dt_min=0.005)
    assert single_integrator_run.cache_valid is False


def test_shared_memory_bytes(single_integrator_run):
    """Test that shared_memory_bytes property works correctly."""
    # Ensure everything is built
    single_integrator_run.build()

    # Get shared memory requirement
    shared_mem = single_integrator_run.shared_memory_bytes
    assert isinstance(shared_mem, int)
    assert shared_mem >= 0


def test_error_handling_unrecognized_parameters(single_integrator_run):
    """Test error handling for unrecognized parameters."""
    # Test invalid parameter that no component recognizes
    with pytest.raises(KeyError, match="Unrecognized parameters"):
        single_integrator_run.update(invalid_param_name=42)


def test_error_handling_invalid_algorithm(single_integrator_run):
    """Test error handling for invalid algorithm."""
    # Test invalid algorithm
    with pytest.raises(KeyError):
        single_integrator_run.update(algorithm="invalid_algorithm")


def test_empty_update_call(single_integrator_run):
    """Test that empty update calls are handled gracefully."""
    # Should not raise an error
    single_integrator_run.update()

    # Should not affect cache validity if nothing was built
    original_cache_valid = single_integrator_run.cache_valid
    single_integrator_run.update()
    assert single_integrator_run.cache_valid == original_cache_valid


def test_device_function_property_builds_automatically(single_integrator_run):
    """device_function property builds lazily when accessed."""
    # Initially cache should be invalid
    assert single_integrator_run.cache_valid is False

    # Accessing device_function should automatically build
    device_func = single_integrator_run.device_function

    # Should now be valid and callable
    assert single_integrator_run.cache_valid is True
    assert callable(device_func)
    assert device_func is single_integrator_run._compiled_loop
