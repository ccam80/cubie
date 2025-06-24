import pytest
import numpy as np
from numpy.testing import assert_allclose
from numba import cuda, from_dtype
from CuMC.ForwardSim.integrators.output_functions import build_output_functions


def calculate_expected_summaries(state_input, num_states, num_observables, outputs_list, n_peaks,
                                 summarise_every, summary_output_length, precision):
    """Helper function to calculate expected summary values."""
    num_samples = state_input.shape[0]
    expected_state_summaries = np.zeros((int(num_samples / summarise_every), num_states * summary_output_length),
                                        dtype=precision)
    expected_obs_summaries = np.zeros((int(num_samples / summarise_every), num_observables * summary_output_length),
                                      dtype=precision)

    # Sort outputs list to match the order in build_output_functions
    types_order = ["mean", "peaks", "max", "rms"]
    sorted_outputs = sorted(outputs_list, key=lambda x: types_order.index(x) if x in types_order else len(types_order))

    # Calculate expected state summaries
    for j in range(num_states):
        for i in range(int(num_samples / summarise_every)):
            summary_index = 0
            for output_type in sorted_outputs:
                if output_type == 'mean':
                    expected_state_summaries[i, j * summary_output_length + summary_index] = np.mean(
                        state_input[i * summarise_every: (i + 1) * summarise_every, j], axis=0)
                    summary_index += 1
                if output_type == 'peaks':
                    #Use the last two samples, like the live version does
                    start_index = i * summarise_every - 2 if i > 0 else 0
                    end_index = (i+1) * summarise_every
                    maxima = local_maxima(
                        state_input[start_index: end_index, j])[:n_peaks] + start_index
                    expected_state_summaries[i, j * summary_output_length + summary_index:
                                                j * summary_output_length + summary_index + maxima.size] = maxima
                    summary_index += n_peaks
                if output_type == 'max':
                    expected_state_summaries[i, j * summary_output_length + summary_index] = np.max(
                        state_input[i * summarise_every: (i + 1) * summarise_every, j], axis=0)
                    summary_index += 1
                if output_type == 'rms':
                    expected_state_summaries[i, j * summary_output_length + summary_index] = np.sqrt(
                        np.mean(state_input[i * summarise_every: (i + 1) * summarise_every, j] ** 2, axis=0))
                    summary_index += 1

    # Calculate expected observable summaries
    for k in range(num_observables):
        j = k % num_states
        for i in range(int(num_samples / summarise_every)):
            summary_index = 0
            for output_type in sorted_outputs:
                if output_type == 'mean':
                    expected_obs_summaries[i, k * summary_output_length + summary_index] = np.mean(
                        state_input[i * summarise_every: (i + 1) * summarise_every, j] + 1, axis=0)
                    summary_index += 1
                if output_type == 'peaks':
                    start_index = i * summarise_every - 2 if i > 0 else 0
                    end_index = (i + 1) * summarise_every
                    maxima = local_maxima(state_input[start_index:end_index, j])[:n_peaks] + start_index
                    expected_obs_summaries[i, k * summary_output_length + summary_index:
                                              k * summary_output_length + summary_index + maxima.size] = maxima
                    summary_index += n_peaks
                if output_type == 'max':
                    expected_obs_summaries[i, k * summary_output_length + summary_index] = np.max(
                        state_input[i * summarise_every: (i + 1) * summarise_every, j] + 1, axis=0)
                    summary_index += 1
                if output_type == 'rms':
                    expected_obs_summaries[i, k * summary_output_length + summary_index] = np.sqrt(
                        np.mean((state_input[i * summarise_every: (i + 1) * summarise_every, j] + 1) ** 2,
                                axis=0))
                    summary_index += 1

    return expected_state_summaries, expected_obs_summaries

def test_input_output():
    """Test that output functions correctly save state and observable values."""
    precision = np.float32
    numba_precision = from_dtype(precision)

    # Create test data - a simple sine wave for testing
    num_samples = 1000
    summarise_every = 10
    num_states = 2
    num_observables = 1
    n_peaks = 3

    # Create input arrays with known patterns
    # State 0: sine wave
    # State 1: cosine wave
    # Observable 0: sine wave squared
    state_input = np.zeros((num_samples, num_states), dtype=precision)
    observable_input = np.zeros((num_samples, num_observables), dtype=precision)

    # Fill with test data
    for i in range(num_samples):
        t = i * 0.1  # time step
        state_input[i, 0] = np.sin(t)  # sine wave
        state_input[i, 1] = np.cos(t)  # cosine wave
        observable_input[i, 0] = np.sin(t) ** 2  # sine squared

    # Create output arrays
    state_output = np.zeros((num_samples, num_states), dtype=precision)
    observable_output = np.zeros((num_samples, num_observables), dtype=precision)

    # Create arrays for summary metrics
    state_summaries = np.zeros(num_states * (4 + n_peaks + 3), dtype=precision)  # Space for mean, max, rms, peaks
    observable_summaries = np.zeros(num_observables * (4 + n_peaks + 3), dtype=precision)
    state_summaries_output = np.zeros((int(num_samples / summarise_every), num_states * (4 + n_peaks)),
                                      dtype=precision)
    observable_summaries_output = np.zeros((int(num_samples / summarise_every), num_observables * (4 + n_peaks)),
                                           dtype=precision)

    # Create device arrays
    d_state_input = cuda.to_device(state_input)
    d_observable_input = cuda.to_device(observable_input)
    d_state_output = cuda.to_device(state_output)
    d_observable_output = cuda.to_device(observable_output)
    d_state_summaries = cuda.to_device(state_summaries)
    d_observable_summaries = cuda.to_device(observable_summaries)
    d_state_summaries_output = cuda.to_device(state_summaries_output)
    d_observable_summaries_output = cuda.to_device(observable_summaries_output)

    # Build output functions
    outputs_list = ["state", "observables", "mean", "max", "rms", "peaks"]
    saved_states = [0, 1]  # Save both states
    saved_observables = [0]  # Save the one observable

    output_funcs = build_output_functions(outputs_list, saved_states, saved_observables, numpeaks=3)
    save_state = output_funcs.save_state_func
    update_summary_metrics = output_funcs.update_summary_metrics_func
    save_summary_metrics = output_funcs.save_summary_metrics_func

    # Define a test kernel
    @cuda.jit()
    def output_functions_test_kernel(state_input, observable_input,
                                     state_output, observable_output,
                                     state_summaries, observable_summaries,
                                     state_summaries_output, observable_summaries_output):
        """Test kernel for output functions."""
        # Get thread ID
        tx = cuda.threadIdx.x
        bx = cuda.blockIdx.x

        # Each thread processes one time step
        tx = tx + bx * cuda.blockDim.x
        if tx != 0 or bx != 0:
            return

        # Local arrays for current state and observable
        current_state = cuda.local.array(num_states, dtype=numba_precision)
        current_observable = cuda.local.array(num_observables, dtype=numba_precision)

        for i in range(state_input.shape[0]):
            # Get current state and observable
            for j in range(num_states):
                current_state[j] = state_input[i, j]

            for j in range(num_observables):
                current_observable[j] = observable_input[i, j]

            # Call the output functions
            save_state(
                current_state,
                current_observable,
                state_output[i],
                observable_output[i]
            )

            update_summary_metrics(
                current_state,
                current_observable,
                state_summaries,
                observable_summaries,
                i
            )

            # Save summary metrics every summarise_every samples
            if i % summarise_every == 0:
                save_summary_metrics(
                    state_summaries,
                    observable_summaries,
                    state_summaries_output[int(i / summarise_every)],
                    observable_summaries_output[int(i / summarise_every)],
                    summarise_every
                )

    # Launch the kernel
    threads_per_block = 1
    blocks_per_grid = 1

    output_functions_test_kernel[blocks_per_grid, threads_per_block](
        d_state_input,
        d_observable_input,
        d_state_output,
        d_observable_output,
        d_state_summaries,
        d_observable_summaries,
        d_state_summaries_output,
        d_observable_summaries_output
    )

    # Synchronize and copy results back
    cuda.synchronize()

    state_output = d_state_output.copy_to_host()
    observable_output = d_observable_output.copy_to_host()
    state_summaries_output = d_state_summaries_output.copy_to_host()
    observable_summaries_output = d_observable_summaries_output.copy_to_host()

    # Assert that state values were saved correctly
    assert np.allclose(state_input, state_output), "State values were not saved correctly"

    # Assert that observable values were saved correctly
    assert np.allclose(observable_input, observable_output), "Observable values were not saved correctly"

    # Assert that summary metrics were calculated
    # Check that at least some values in the summary metrics are non-zero
    assert np.any(state_summaries_output != 0), "State summaries were not calculated"
    assert np.any(observable_summaries_output != 0), "Observable summaries were not calculated"

def local_maxima(signal: np.ndarray) -> np.ndarray:
    return np.flatnonzero((signal[1:-1] > signal[:-2]) & (signal[1:-1] > signal[2:])) + 1
TEST_CONFIGS = [
    # Basic state and observables tests with different precisions
    (2, 1, ["state", "observables"], None, np.float32, 100, 50, "State and observables, 32b, small and short"),
    (2, 1, ["state", "observables"], None, np.float64, 100, 50, "State and observables, 64b, small and short"),

    # Testing individual output types with different precisions
    (3, 2, ["state", "observables", "mean"], None, np.float32, 100, 50, "Mean only, 32b"),
    (3, 2, ["state", "observables", "max"], None, np.float32, 100, 50, "Max only, 32b"),
    (3, 2, ["state", "observables", "rms"], None, np.float32, 100, 50, "RMS only, 32b"),
    (3, 2, ["state", "observables", "peaks"], 3, np.float32, 100, 50, "Peaks only, 32b"),
    (3, 2, ["state", "observables", "mean"], None, np.float64, 100, 50, "Mean only, 64b"),

    # Testing combinations of output types
    (4, 3, ["state", "observables", "mean", "max"], None, np.float32, 100, 50, "Mean and max, 32b"),
    (4, 3, ["state", "observables", "mean", "rms"], None, np.float32, 100, 50, "Mean and rms, 32b"),
    (4, 3, ["state", "observables", "mean", "peaks"], 3, np.float32, 100, 50, "Mean and peaks, 32b"),
    (4, 3, ["state", "observables", "max", "rms"], None, np.float32, 100, 50, "Max and rms, 32b"),
    (4, 3, ["state", "observables", "max", "peaks"], 3, np.float32, 100, 50, "Max and peaks, 32b"),
    (4, 3, ["state", "observables", "rms", "peaks"], 3, np.float32, 100, 50, "RMS and peaks, 32b"),

    # Testing all output types together
    (5, 4, ["state", "observables", "mean", "max", "rms", "peaks"], 3, np.float32, 100, 50,
     "All metrics, 32b, small and short"),
    (5, 4, ["state", "observables", "mean", "max", "rms", "peaks"], 3, np.float64, 100, 50,
     "All metrics, 64b, small and short"),

    # Testing different state/observable sizes (states > observables)
    (10, 2, ["state", "observables", "mean", "max"], None, np.float32, 100, 50, "More states than observables, 32b"),
    (20, 5, ["state", "observables", "mean", "max", "rms"], None, np.float32, 100, 50,
     "Many more states than observables, 32b"),

    # Testing different state/observable sizes (observables > states)
    (2, 10, ["state", "observables", "mean", "max"], None, np.float32, 100, 50, "More observables than states, 32b"),
    (5, 20, ["state", "observables", "mean", "max", "rms"], None, np.float32, 100, 50,
     "Many more observables than states, 32b"),

    # Testing large equal numbers of states and observables
    (50, 50, ["state", "observables", "mean", "max", "rms", "peaks"], 3, np.float32, 100, 50,
     "Many states and observables, 32b"),
    (50, 50, ["state", "observables", "mean", "max", "rms", "peaks"], 3, np.float64, 100, 50,
     "Many states and observables, 64b"),

    # Testing different numbers of peaks
    (3, 3, ["state", "observables", "peaks"], 1, np.float32, 100, 50, "Single peak, 32b"),
    (3, 3, ["state", "observables", "peaks"], 5, np.float32, 100, 50, "Five peaks, 32b"),
    (3, 3, ["state", "observables", "peaks"], 10, np.float32, 100, 50, "Ten peaks, 32b"),

    # Testing long sample sizes with infrequent summarization
    (4, 4, ["state", "observables", "mean", "max"], None, np.float32, 10000, 10000,
     "Long samples, summarize once, 32b"),
    (4, 4, ["state", "observables", "mean", "max"], None, np.float64, 10000, 10000,
     "Long samples, summarize once, 64b"),

    # Testing long sample sizes with frequent summarization
    (4, 4, ["state", "observables", "mean", "max"], None, np.float32, 10000, 100,
     "Long samples, frequent summarization, 32b"),
    (4, 4, ["state", "observables", "mean", "max"], None, np.float64, 10000, 100,
     "Long samples, frequent summarization, 64b"),

    # Testing short sample sizes with very frequent summarization
    (4, 4, ["state", "observables", "mean", "max"], None, np.float32, 100, 5,
     "Short samples, summarize every 5 steps, 32b"),
    (4, 4, ["state", "observables", "mean", "max"], None, np.float64, 100, 5,
     "Short samples, summarize every 5 steps, 64b"),

    # Edge cases
    (1, 1, ["state", "observables", "mean", "max", "rms", "peaks"], 3, np.float32, 100, 50,
     "Minimal state and observables, 32b"),
    (1, 1, ["state", "observables", "mean", "max", "rms", "peaks"], 3, np.float64, 100, 50,
     "Minimal state and observables, 64b"),
    (100, 1, ["state", "observables", "mean"], None, np.float32, 100, 50, "Many states, single observable, 32b"),
    (1, 100, ["state", "observables", "mean"], None, np.float32, 100, 50, "Single state, many observables, 32b"),

    (1, 1, ["state", "observables", "mean", "max", "rms", "peaks"], 3, np.float32, 100, 50,
     "Minimal state and observables, 32b"),
    (1, 1, ["state", "observables", "mean", "max", "rms", "peaks"], 3, np.float64, 100, 50,
     "Minimal state and observables, 64b"),
    (100, 1, ["state", "observables", "mean"], None, np.float32, 100, 50, "Many states, single observable, 32b"),
    (1, 100, ["state", "observables", "mean"], None, np.float32, 100, 50, "Single state, many observables, 32b"),

    # Comprehensive tests with all metrics and different configurations
    (8, 4, ["state", "observables", "mean", "max", "rms"], None, np.float32, 1000, 500,
     "Medium test, all metrics except peaks, 32b"),
    (8, 4, ["state", "observables", "mean", "max", "rms", "peaks"], 3, np.float32, 1000, 500,
     "Medium test, all metrics with peaks, 32b"),
    (10, 5, ["state", "observables", "mean", "max", "rms", "peaks"], 5, np.float64, 1000, 500,
     "Medium test, all metrics with peaks, 64b"),

    # Extreme cases
    (100, 100, ["state", "observables", "mean"], None, np.float32, 10000, 1000,
     "Very large state and observables, mean only, 32b"),
    (100, 100, ["state", "observables", "peaks"], 20, np.float32, 10000, 1000,
     "Very large state and observables, many peaks, 32b")
]

@pytest.mark.parametrize(
    "num_states, num_observables, outputs_list, n_peaks, precision, num_samples, summarise_every, test_name",
    TEST_CONFIGS,
    ids=[config[7] for config in TEST_CONFIGS]  # Use test_name as the test ID
)
def test_shared_mem(num_states, num_observables, outputs_list, n_peaks, precision, num_samples, summarise_every, test_name):
    """General test harness/kernel, which allocates shared memory for temporary states and summaries based on the build_output_functions output."""

    numba_precision = from_dtype(precision)

    type_counter = 0

    # Create saved states and observables lists
    if n_peaks is None:
        n_peaks = 0

    saved_states = list(range(num_states))
    saved_observables = list(range(num_observables))

    state_input = np.zeros((num_samples, num_states), dtype=precision)
    expected_observables_out = np.zeros((num_samples, num_observables), dtype=precision)

    for j in range(num_states):
        for i in range(num_samples):
            t = i * 0.1  # time step
            state_input[i, j] = np.sin(t + j*np.pi/4)

    for j in range(num_observables):
        expected_observables_out[:, j] = state_input[:, j % num_states] + 1

    # Build output functions
    output_funcs = build_output_functions(outputs_list, saved_states, saved_observables, numpeaks=n_peaks)

    # Get shared memory requirements
    shared_mem_required_per_summarised_state = output_funcs.temp_memory_requirements
    summary_output_length = output_funcs.summary_output_length
    save_state = output_funcs.save_state_func
    update_summary_metrics = output_funcs.update_summary_metrics_func
    save_summary_metrics = output_funcs.save_summary_metrics_func


    # Create output arrays
    state_output = np.zeros((num_samples, num_states), dtype=precision)
    observable_output = np.zeros((num_samples, num_observables), dtype=precision)

    # Create arrays for summary metrics

    state_summaries_output = np.zeros((int(num_samples / summarise_every), num_states * summary_output_length),
                                      dtype=precision)
    observable_summaries_output = np.zeros((int(num_samples / summarise_every), num_observables * summary_output_length),
                                           dtype=precision)

    expected_state_summaries = np.zeros((int(num_samples / summarise_every), num_states * (summary_output_length)), dtype=precision)
    expected_obs_summaries = np.zeros((int(num_samples / summarise_every), num_observables * (summary_output_length)), dtype=precision)

    # Calculate expected summaries
    expected_state_summaries, expected_obs_summaries = calculate_expected_summaries(state_input, num_states,
                                                                                    num_observables, outputs_list,
                                                                                    n_peaks, summarise_every,
                                                                                    summary_output_length, precision)

    # Create device arrays
    d_state_input = cuda.to_device(state_input)
    d_state_output = cuda.to_device(state_output)
    d_observable_output = cuda.to_device(observable_output)
    d_state_summaries_output = cuda.to_device(state_summaries_output)
    d_observable_summaries_output = cuda.to_device(observable_summaries_output)

    shared_mem_required = shared_mem_required_per_summarised_state * (num_states + num_observables) #for temp summaries
    shared_mem_required += num_states + num_observables # for current state and observables
    shared_mem_required_bytes = shared_mem_required * precision().itemsize

    # Define a test kernel
    @cuda.jit()
    def output_functions_test_kernel(state_input,
                                     state_output, observable_output,
                                     state_summaries_output, observable_summaries_output):
        """Test kernel for output functions."""


        tx = cuda.threadIdx.x
        bx = cuda.blockIdx.x

        # Each thread processes one time step
        tx = tx + bx * cuda.blockDim.x
        if tx != 0 or bx != 0:
            return

        # Local arrays for current state and observable
        shared = cuda.shared.array(0, dtype=numba_precision)

        observables_start_idx = num_states
        state_summaries_start_idx = observables_start_idx + num_observables
        obs_summaries_start_idx = state_summaries_start_idx + num_states * shared_mem_required_per_summarised_state
        obs_summaries_end_idx = obs_summaries_start_idx + num_observables * shared_mem_required_per_summarised_state

        current_state = shared[:observables_start_idx]
        current_observable = shared[observables_start_idx:state_summaries_start_idx]
        state_summaries = shared[state_summaries_start_idx:obs_summaries_start_idx]
        observable_summaries = shared[obs_summaries_start_idx:obs_summaries_end_idx]

        current_state[:] = 0.0
        current_observable[:] = 0.0
        state_summaries[:] = 0.0
        observable_summaries[:] = 0.0

        for i in range(state_input.shape[0]):
            # Get current state and observable
            for j in range(num_states):
                current_state[j] = state_input[i, j]

            for j in range(num_observables):
                current_observable[j] = current_state[j % num_states] + 1

            # Call the output functions
            save_state(
                current_state,
                current_observable,
                state_output[i],
                observable_output[i]
            )


            update_summary_metrics(
                current_state,
                current_observable,
                state_summaries,
                observable_summaries,
                i
            )

            # Save summary metrics every summarise_every samples
            if (i+1) % summarise_every == 0:
                save_summary_metrics(
                    state_summaries,
                    observable_summaries,
                    state_summaries_output[i // summarise_every],
                    observable_summaries_output[i // summarise_every],
                    summarise_every
                )



    # Launch the kernel
    threads_per_block = 1
    blocks_per_grid = 1

    output_functions_test_kernel[blocks_per_grid, threads_per_block, 0, shared_mem_required_bytes](
        d_state_input,
        d_state_output,
        d_observable_output,
        d_state_summaries_output,
        d_observable_summaries_output,
    )

    # Synchronize and copy results back
    cuda.synchronize()

    state_output = d_state_output.copy_to_host()
    observable_output = d_observable_output.copy_to_host()
    state_summaries_output = d_state_summaries_output.copy_to_host()
    observable_summaries_output = d_observable_summaries_output.copy_to_host()

    del output_functions_test_kernel
    del save_state
    del update_summary_metrics
    del save_summary_metrics
    # Assert that state values were saved correctly
    assert_allclose(state_input, state_output, atol=1e-07, rtol=1e-07,
                    err_msg="State values were not saved correctly")

    # Assert that observable values were saved correctly
    assert_allclose(expected_observables_out, observable_output, atol=1e-07, rtol=1e-07,
                    err_msg= "Observable values were not saved correctly")

    # Assert that summary metrics were calculated correctly
    if any(summary in outputs_list for summary in ["mean", "max", "rms", "peaks"]):
        assert_allclose(expected_state_summaries, state_summaries_output, atol=1e-05, rtol=1e-05,
                        err_msg=f"State summaries didn't match expected values. Shapes: expected[{expected_state_summaries.shape}, actual[{state_summaries_output.shape}]")
        assert_allclose(expected_obs_summaries, observable_summaries_output, atol=1e-05, rtol=1e-05,
                        err_msg = f"Observable summaries didn't match expected values. Shapes: expected[{expected_obs_summaries.shape}, actual[{observable_summaries_output.shape}]")

