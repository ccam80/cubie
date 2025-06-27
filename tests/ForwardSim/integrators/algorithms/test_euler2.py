import pytest
import numpy as np
from numba import cuda, from_dtype, float32, float64, int32
from scipy.stats.contingency import expected_freq
from tests.ForwardSim.integrators.algorithms.LoopAlgorithmTester import LoopAlgorithmTester

from CuMC.SystemModels.Systems.threeCM import ThreeChamberModel
from CuMC.SystemModels.Systems.decays import Decays
from CuMC.SystemModels.Systems.threeCM import GenericODE

from CuMC.ForwardSim.integrators.algorithms.euler import Euler

#Required system settings
# system_settings = (precision, state_names, parameter_names, observable_names, constants, num_drivers)

#Required loop compile settings
# compile_settings = {precision,
    #                  dxdt_func,
    #                  n_states,
    #                  n_obs,
    #                  n_par,
    #                  n_drivers,
    #                  dt_min,
    #                  dt_max,
    #                  dt_save,
    #                  dt_summarise,
    #                  atol,
    #                  rtol,
    #                  save_state_func,
    #                  update_summary_func,
    #                  save_summary_func,
    #                  n_saved_states,
    #                  n_saved_observables,
    #                  summary_temp_memory}
class TestEuler(LoopAlgorithmTester):
    """Testing class for the Euler algorithm. Checks the instantiation, compilation, and input/output for a range
    of cases, including incomplete inputs and random floats of different scales."""

    @pytest.fixture(scope="class", autouse=True)
    def system_instance(self, system_class, integrator_algorithm, system_settings, compile_settings, run_settings):
        self.system_instance = system_class(*system_settings)

    @pytest.fixture(scope="class")
    def integrator_algorithm(self, system_class, integrator_algorithm, system_settings, compile_settings, run_settings):
        return Euler

    def build_output_functions(self, system_class, integrator_algorithm, system_settings, compile_settings, run_settings):

    def expected_shared_memory(self):
        """Calculate the expected shared memory size for the Euler algorithm."""
        n_states = self.system_instance.num_states
        n_obs = self.system_instance.num_observables
        n_drivers = self.system_instance.num_drivers
        return n_states + n_states + n_obs + n_drivers

    def CPU_euler_loop(dxdt_func,
                       inits,
                       params,
                       driver_vec,
                       dt,
                       output_dt,
                       warmup,
                       duration,
                       saved_observables,
                       saved_states):
        """A simple CPU implementation of the Euler loop for testing."""
        t = 0.0
        save_every = int(round(output_dt / dt))
        n_saved_states = len(saved_states)
        n_saved_observables = len(saved_observables)
        total_samples = int((duration + warmup) / output_dt)
        state_output = np.zeros((n_saved_states,int(duration / output_dt)), dtype=inits.dtype)
        observables_output = np.zeros((n_saved_observables,int(duration / output_dt)), dtype=inits.dtype)
        state_output[:, 0] = inits
        state = inits

        for i in range(total_samples):

            for j in range(save_every):
                drivers = driver_vec[(i*save_every + j) % len(driver_vec)]
                t += dt
                dx = dxdt_func(state, params, drivers)
                state += dx[0] * dt
                if i == 0 and j == 0:
                    observables_output[:,0] = dx[1]

            state_output[saved_states, i+1] = dx[0]
            observables_output[saved_observables, i+1] = dx[1]

        return state_output, observables_output



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

# Test configurations for parameterized tests
SHARED_MEMORY_TEST_CONFIGS = [
    # (precision, n_states, n_obs, n_drivers, test_name)
    (float32, 3, 2, 1, "Basic configuration"),
    (float64, 5, 3, 2, "Larger configuration"),
    (float32, 10, 0, 3, "No observables"),
    (float32, 1, 1, 1, "Minimal configuration"),
]

TIME_TO_SAMPLES_TEST_CONFIGS = [
    # (dt_min, dt_save, dt_summarise, expected_save_every, expected_summarise_every, should_warn_save, should_warn_summarise, test_name)
    (0.001, 0.01, 0.05, 10, 5, False, False, "Exact multiples"),
    (0.001, 0.0105, 0.053, 11, 5, True, True, "Non-exact multiples"),
    (0.001, 0.001, 0.003, 1, 3, False, False, "Minimum save interval"),
]

TIME_TO_SAMPLES_ERROR_TEST_CONFIGS = [
    # (dt_min, dt_save, dt_summarise, test_name)
    (0.001, 0.0005, 0.01, "dt_save < dt_min"),
    (0.001, 0.01, 0.01, "dt_summarise = dt_save"),
    (0.001, 0.01, 0.005, "dt_summarise < dt_save"),
    (0.005, 0.02, 0.04, "4 samples per save, 2 per summarise, dt_min = 0.005")
]

REBUILD_TEST_CONFIGS = [
    # (initial_params, new_params, should_warn, test_name)
    (
        (float32, 3, 2, 5, 1, 0.001, 0.01, 0.01, 0.05, 1e-6, 1e-3, 10, 5, 20),
        {"dt_min": 0.002},
        False,
        "Update dt_min"
    ),
    (
        (float32, 3, 2, 5, 1, 0.001, 0.01, 0.01, 0.05, 1e-6, 1e-3, 10, 5, 20),
        {"n_states": 4, "n_obs": 3, "n_drivers": 2},
        False,
        "Update state dimensions"
    ),
]

# def test_euler_integration():
#     #  TODO: Get a general purpose algorithm tester up using the decays system.
#     """
#     Test the Euler integrator by comparing it with a non-GPU Euler loop.
#
#     The test verifies:
#     1. Result is close to a CPU implementation of the same algorithm
#     2. All numbers are of the right type
#     3. The GPU integrator loop succesfully compiles
#     """
#
#     precision = np.float32
#     numba_precision = from_dtype(precision)
#
#     # Create and build the system
#     sys = ThreeChamberModel(precision=precision)
#     sys.build()
#
#     # Set up integration parameters
#     internal_step = 0.001
#     save_step = 0.001
#     duration = 0.1
#     warmup = 0.0
#
#     output_samples = int(duration / save_step)
#     warmup_samples = int(warmup / save_step)
#     save_every = int(save_step / internal_step)
#
#     # Create and build the integrator under test
#     integrator = genericODEIntegrator(precision=precision)
#     integrator.build_loop(sys, internal_step, save_step)
#     intfunc = integrator.integratorLoop
#
#     # Define the test kernel
#     @cuda.jit()
#     def loop_test_kernel(inits, params, forcing_vector, output, observables):
#         c_forcing_vector = cuda.const.array_like(forcing_vector)
#         shared_memory = cuda.shared.array(0, dtype=numba_precision)
#
#         intfunc(
#             inits,
#             params,
#             c_forcing_vector,
#             shared_memory,
#             output,
#             observables,
#             output_samples,
#             warmup_samples
#         )
#
#     # Allocate memory for inputs and outputs
#     output = cuda.pinned_array((output_samples, sys.num_states), dtype=precision)
#     observables = cuda.pinned_array((output_samples, sys.num_observables), dtype=precision)
#     forcing_vector = cuda.pinned_array(((output_samples + warmup_samples)*save_every, sys.num_drivers), dtype=precision)
#
#     # Initialize arrays
#     output[:, :] = precision(0.0)
#     observables[:, :] = precision(0.0)
#     forcing_vector[:, :] = precision(0.0)
#     forcing_vector[0, :] = precision(1.0)
#
#     # Copy to device
#     d_forcing = cuda.to_device(forcing_vector)
#     d_inits = cuda.to_device(sys.init_values.values_array)
#     d_params = cuda.to_device(sys.parameters.values_array)
#     d_output = cuda.to_device(output)
#     d_observables = cuda.to_device(observables)
#
#     # Get shared memory requirements
#     sharedmem = integrator.get_shared_memory_requirements(sys)
#
#     # Run the kernel
#     loop_test_kernel[1, 1, 0, sharedmem](
#         d_inits,
#         d_params,
#         d_forcing,
#         d_output,
#         d_observables
#     )
#
#     # Synchronize and copy results back to host
#     cuda.synchronize()
#     cuda_output = d_output.copy_to_host()
#     cuda_obs = d_observables.copy_to_host()
#
#     # Verify all numbers are of the right type
#     assert cuda_output.dtype == precision, f"Output array has wrong dtype: {cuda_output.dtype} instead of {precision}"
#     assert cuda_obs.dtype == precision, f"Observables array has wrong dtype: {cuda_obs.dtype} instead of {precision}"
#
#     # Implement a scipy version for comparison
#     # Define the ODE function for scipy
#     def three_chamber_ode(state, parameters, driver_value):
#         """NON-CUDA version of the three chamber model"""
#         # Extract parameters
#         E_h = parameters[0]
#         E_a = parameters[1]
#         E_v = parameters[2]
#         R_i = parameters[3]
#         R_o = parameters[4]
#         R_c = parameters[5]
#
#         V_h = state[0]
#         V_a = state[1]
#         V_v = state[2]
#
#         # Calculate auxiliary values
#         P_a = E_a * V_a
#         P_v = E_v * V_v
#         P_h = E_h * V_h * driver_value[0]
#         Q_i = ((P_v - P_h) / R_i) if (P_v > P_h) else 0
#         Q_o = ((P_h - P_a) / R_o) if (P_h > P_a) else 0
#         Q_c = (P_a - P_v) / R_c
#
#         # Calculate gradient
#         dV_h = Q_i - Q_o
#         dV_a = Q_o - Q_c
#         dV_v = Q_c - Q_i
#
#         return np.asarray([dV_h, dV_a, dV_v])
#
#
#     # Get initial values and parameters
#     init_values = sys.init_values.values_array.copy()
#     params = sys.parameters.values_array.copy()
#
#     y = init_values.copy()
#     CPU_result = np.zeros((len(init_values), output_samples))
#
#     # Euler integration with fixed step size
#     dt = internal_step
#
#
#     for i in range(output_samples + warmup_samples):
#         for j in range(save_every):
#             # Calculate derivatives
#             driver_value = forcing_vector[i*save_every + j]
#             dydt = three_chamber_ode(y, params, driver_value)
#
#             # Euler step
#             y = y + dydt * dt
#
#
#         # Store the state at this output time
#         if i > (warmup_samples - 1):
#             # Store the state in the CPU result array:
#             CPU_result[:, i-warmup_samples] = y
#
#
#     # Compare results
#     # Note: We expect some differences due to floating point math
#     print("CUDA output (first few values):")
#     print(cuda_output[:5, :])
#     print("\nCPU output (first few values):")
#     print(CPU_result[:, :5].T)
#
#     for i in range(sys.num_states):
#         assert np.allclose(cuda_output[i,:], CPU_result[:,i], rtol=1e-3, atol=1e-6), \
#             f"State {i} results don't match between CUDA and scipy"
#
#     # The test passes if we get here, indicating successful compilation and execution
#     assert True, "Test completed successfully"

@pytest.mark.parametrize(
    "precision, n_states, n_obs, n_drivers, test_name",
    SHARED_MEMORY_TEST_CONFIGS,
    ids=[config[-1] for config in SHARED_MEMORY_TEST_CONFIGS]
)
def test_calculate_shared_memory(precision, n_states, n_obs, n_drivers, test_name):
    """Test the calculate_shared_memory method of Euler."""
    # Create the algorithm with basic parameters
    algorithm = Euler(
        precision=precision,
        dxdt_func=mock_dxdt_func,
        n_states=n_states,
        n_obs=n_obs,
        n_par=5,  # Not relevant for this test
        n_drivers=n_drivers,
        dt_min=0.001,  # Not relevant for this test
        dt_max=0.01,   # Not relevant for this test
        dt_save=0.01,  # Not relevant for this test
        dt_summarise=0.1,  # Not relevant for this test
        atol=1e-6,     # Not relevant for this test
        rtol=1e-3,     # Not relevant for this test
        save_state_func=mock_save_state_func,
        update_summary_func=mock_update_summary_func,
        save_summary_func=mock_save_summary_func,
        n_saved_states=10,  # Not relevant for this test
        n_saved_observables=5,  # Not relevant for this test
        summary_temp_memory=20  # Not relevant for this test
    )

    # Call calculate_shared_memory method
    shared_memory = algorithm.calculate_shared_memory()

    # Check that the result is as expected: n_states + n_states (for dxdt) + n_obs + n_drivers
    expected = n_states + n_states + n_obs + n_drivers
    assert shared_memory == expected, f"Expected {expected} shared memory items, got {shared_memory}"

@pytest.mark.parametrize(
    "dt_min, dt_save, dt_summarise, expected_save_every, expected_summarise_every, should_warn_save, should_warn_summarise, test_name",
    TIME_TO_SAMPLES_TEST_CONFIGS,
    ids=[config[-1] for config in TIME_TO_SAMPLES_TEST_CONFIGS]
)
def test_time_to_samples(dt_min, dt_save, dt_summarise, expected_save_every, expected_summarise_every, should_warn_save, should_warn_summarise, test_name):
    """Test the _time_to_samples method of Euler."""
    # Create the algorithm with basic parameters
    algorithm = Euler(
        precision=float32,
        dxdt_func=mock_dxdt_func,
        n_states=3,
        n_obs=2,
        n_par=5,
        n_drivers=1,
        dt_min=dt_min,
        dt_max=0.01,
        dt_save=dt_save,
        dt_summarise=dt_summarise,
        atol=1e-6,
        rtol=1e-3,
        save_state_func=mock_save_state_func,
        update_summary_func=mock_update_summary_func,
        save_summary_func=mock_save_summary_func,
        n_saved_states=10,
        n_saved_observables=5,
        summary_temp_memory=20
    )

    # Call _time_to_samples method with warning capture





    # Check warnings if expected
    if should_warn_save or should_warn_summarise:
        with pytest.warns(UserWarning):
            save_every_samples, summarise_every_samples, internal_step_size = algorithm._time_to_samples()
        assert algorithm.loop_parameters['dt_save'] == save_every_samples * dt_min, "dt_save was not updated correctly"
        assert algorithm.loop_parameters['dt_summarise'] == summarise_every_samples * algorithm.loop_parameters['dt_save'], "dt_summarise was not updated correctly"

    else:
        save_every_samples, summarise_every_samples, internal_step_size = algorithm._time_to_samples()
        # Check that the results are as expected
        assert save_every_samples == expected_save_every, f"Expected save_every_samples={expected_save_every}, got {save_every_samples}"
        assert summarise_every_samples == expected_summarise_every, f"Expected summarise_every_samples={expected_summarise_every}, got {summarise_every_samples}"
        assert internal_step_size == dt_min, f"Expected internal_step_size={dt_min}, got {internal_step_size}"


@pytest.mark.parametrize(
    "dt_min, dt_save, dt_summarise, test_name",
    TIME_TO_SAMPLES_ERROR_TEST_CONFIGS,
    ids=[config[-1] for config in TIME_TO_SAMPLES_ERROR_TEST_CONFIGS]
)
def test_time_to_samples_errors(dt_min, dt_save, dt_summarise, test_name):
    """Test that _time_to_samples raises appropriate errors."""
    # Create the algorithm with basic parameters
    algorithm = Euler(
        precision=float32,
        dxdt_func=mock_dxdt_func,
        n_states=3,
        n_obs=2,
        n_par=5,
        n_drivers=1,
        dt_min=0.01,
        dt_max=0.01,
        dt_save=0.1,
        dt_summarise=1.0,
        atol=1e-6,
        rtol=1e-3,
        save_state_func=mock_save_state_func,
        update_summary_func=mock_update_summary_func,
        save_summary_func=mock_save_summary_func,
        n_saved_states=10,
        n_saved_observables=5,
        summary_temp_memory=20
    )

    # Modify time values to match test parameters, then run _time_to_samples, checking for errors
    algorithm.loop_parameters['dt_min'] = dt_min
    algorithm.loop_parameters['dt_save'] = dt_save
    algorithm.loop_parameters['dt_summarise'] = dt_summarise

    if dt_save < dt_min:
        with pytest.raises(ValueError, match="dt_save.*"):
            algorithm._time_to_samples()
    elif dt_summarise <= dt_save:
        with pytest.raises(ValueError, match="dt_summarise.*"):
            algorithm._time_to_samples()
    else:
        save_every_samples, summarise_every_samples, step_size =  algorithm._time_to_samples()
        assert save_every_samples == int(np.round(dt_save / dt_min))
        assert summarise_every_samples == int(np.round(dt_summarise / dt_save))
        assert step_size == dt_min

@pytest.mark.parametrize(
    "initial_params, new_params, should_warn, test_name",
    REBUILD_TEST_CONFIGS,
    ids=[config[-1] for config in REBUILD_TEST_CONFIGS]
)
def test_rebuild(initial_params, new_params, should_warn, test_name):
    """Test the rebuild method of Euler with various parameter updates."""
    # Unpack initial parameters
    (precision, n_states, n_obs, n_par, n_drivers, dt_min, dt_max, dt_save, 
     dt_summarise, atol, rtol, n_saved_states, n_saved_observables, summary_temp_memory) = initial_params

    # Create the algorithm with initial parameters
    algorithm = Euler(
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

    # Calculate initial shared memory
    initial_shared_memory = algorithm.calculate_shared_memory()

    # Call rebuild method with new parameters
    if should_warn:
        with pytest.warns():
            algorithm.rebuild(**new_params)
    else:
        algorithm.rebuild(**new_params)

    # Check that parameters were updated correctly
    for key, value in new_params.items():
        if key in algorithm.loop_parameters:
            assert algorithm.loop_parameters[key] == value, f"Parameter {key} was not updated correctly"

    # Calculate new shared memory
    new_shared_memory = algorithm.calculate_shared_memory()

    # Check if shared memory changed when relevant parameters changed
    if any(param in new_params for param in ['n_states', 'n_obs', 'n_drivers']):
        assert new_shared_memory != initial_shared_memory, "Shared memory should have changed after rebuild"
    else:
        assert new_shared_memory == initial_shared_memory, "Shared memory should not have changed after rebuild"
