import pytest

if __name__ == "__main__":
    import os
    os.environ["NUMBA_ENABLE_CUDASIM"] = "0"
    os.environ["NUMBA_CUDA_DEBUGINFO"] = "1"
    os.environ["NUMBA_OPT"] = "0"

import numpy as np
from numba import cuda, from_dtype

from CuMC.ForwardSim.integrators.output_functions import build_output_functions

# from pdb import set_trace
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
    (4, 4, ["state", "observables", "mean", "max"], None, np.float32, 100, 1,
     "Short samples, summarize every step, 32b"),
    (4, 4, ["state", "observables", "mean", "max"], None, np.float64, 100, 1,
     "Short samples, summarize every step, 64b"),

    # Edge cases
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
    "num_states, num_observables, outputs_list, numpeaks, precision, num_samples, summarise_every, test_name",
    TEST_CONFIGS,
    ids=[config[7] for config in TEST_CONFIGS]  # Use test_name as the test ID
)
def test_shared_mem(num_states, num_observables, outputs_list, numpeaks, precision, num_samples, summarise_every, test_name):
    """Test that the shared memory requirements reported by build_output_functions are sufficient."""

    # TODO: This is where you're up to. The thing doesn't seem to be compiling the device functions again, and instead
    #  uses previous sizes for the arrays, which shits the bed. Perhaps
    numba_precision = from_dtype(precision)

    type_counter = 0

    # Create saved states and observables lists
    if numpeaks is None:
        numpeaks = 0

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
    output_funcs = build_output_functions(outputs_list, saved_states, saved_observables, numpeaks=numpeaks)

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

    state_summaries_output = np.zeros((int(np.ceil(num_samples / summarise_every)), num_states * summary_output_length),
                                      dtype=precision)
    observable_summaries_output = np.zeros((int(np.ceil(num_samples / summarise_every)), num_observables * summary_output_length),
                                           dtype=precision)

    expected_state_summaries = np.zeros((int(num_samples / summarise_every), num_states * (summary_output_length)), dtype=precision)
    expected_obs_summaries = np.zeros((int(num_samples / summarise_every), num_observables * (summary_output_length)), dtype=precision)

    # Fill expected summaries array - fairly horrific to look at, but calculates the ems, peaks, mean, max and slots
    # them in where they belong. The ordering of summaries is currently manually handled, which doesn't seem ideal.
    # Maybe we can handle this by assigning to a dict on output.
    types_order = ["mean", "peaks", "max", "rms"]
    outputs_list = sorted(outputs_list, key=lambda x: types_order.index(x) if x in types_order else len(types_order))

    for j in range(num_states): #the order in the build function is: mean, peaks, max, rms
        for i in range(int(num_samples / summarise_every)):
            summary_index = 0
            for output_type in outputs_list:
                if output_type == 'mean':
                    expected_state_summaries[i, j*summary_output_length + summary_index] = np.mean(state_input[i*summarise_every: (i+1)*summarise_every, j], axis=0)
                    summary_index += 1
                if output_type == 'peaks':
                    maxima = local_maxima(
                        state_input[i*summarise_every: (i+1)*summarise_every, j])[:numpeaks] + i*summarise_every
                    expected_state_summaries[i, j*summary_output_length + summary_index:
                                                j*summary_output_length + summary_index+maxima.size] = maxima
                    summary_index += numpeaks
                if output_type == 'max':
                    expected_state_summaries[i, j * summary_output_length + summary_index] = np.max(
                        state_input[i*summarise_every: (i+1)*summarise_every, j], axis=0)
                    summary_index += 1
                if output_type == 'rms':
                    expected_state_summaries[i, j * summary_output_length + summary_index] = np.sqrt(
                        np.mean(state_input[i * summarise_every: (i + 1) * summarise_every, j] ** 2, axis=0))
                    summary_index += 1


    for k in range(num_observables):
        j = k % num_states
        for i in range(int(num_samples / summarise_every)):
            summary_index = 0
            for output_type in outputs_list:
                if output_type == 'mean':
                    expected_obs_summaries[i, k * summary_output_length + summary_index] = np.mean(
                        state_input[i * summarise_every: (i + 1) * summarise_every, j]+1, axis=0)
                    summary_index += 1
                if output_type == 'peaks':
                    maxima = local_maxima(
                        state_input[i * summarise_every: (i + 1) * summarise_every, j])[
                             :numpeaks] + i * summarise_every
                    expected_obs_summaries[i, k * summary_output_length + summary_index:
                                                k * summary_output_length + summary_index + maxima.size] = maxima
                    summary_index += numpeaks
                if output_type == 'max':
                    expected_obs_summaries[i, k * summary_output_length + summary_index] = np.max(
                        state_input[i * summarise_every: (i + 1) * summarise_every, j]+1, axis=0)
                    summary_index += 1
                if output_type == 'rms':
                    expected_obs_summaries[i, k * summary_output_length + summary_index] = np.sqrt(
                        np.mean((state_input[i * summarise_every: (i + 1) * summarise_every, j]+1) ** 2,
                                axis=0))
                    summary_index += 1

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
    assert np.allclose(state_input, state_output), "State values were not saved correctly"

    # Assert that observable values were saved correctly
    assert np.allclose(expected_observables_out, observable_output), "Observable values were not saved correctly"

    # Assert that summary metrics were calculated
    # Check that at least some values in the summary metrics are non-zero
    print(f"State and observables worked! Shapes: {state_summaries_output.shape}, {observable_summaries_output.shape}")

    if any(summary in outputs_list for summary in ["mean", "max", "rms", "peaks"]):
        assert np.allclose(expected_state_summaries, state_summaries_output), "State summaries didn't match expected values"
        assert np.allclose(expected_obs_summaries, observable_summaries_output), "Observable summaries didn't match expected values"
        print(f"Summaries worked! Shapes: {state_summaries_output.shape}, {observable_summaries_output.shape}")



