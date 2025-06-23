import numpy as np
from numba import cuda, from_dtype
from CuMC.ForwardSim.integrators.output_functions import build_output_functions

def test_input_output():
    """Test that output functions correctly save state and observable values."""
    precision = np.float32
    numba_precision = from_dtype(precision)

    # Create test data - a simple sine wave for testing
    num_samples = 1000
    summarise_every = 10
    num_states = 2
    num_observables = 1
    numpeaks = 3

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
    state_summaries = np.zeros(num_states * (4 + numpeaks + 3), dtype=precision)  # Space for mean, max, rms, peaks
    observable_summaries = np.zeros(num_observables * (4 + numpeaks + 3), dtype=precision)
    state_summaries_output = np.zeros((int(num_samples / summarise_every), num_states * (4 + numpeaks)),
                                      dtype=precision)
    observable_summaries_output = np.zeros((int(num_samples / summarise_every), num_observables * (4 + numpeaks)),
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

def test_shared_mem_requests_and_summaries():
    """Test that the shared memory requirements reported by build_output_functions are sufficient."""
    precision = np.float32
    numba_precision = from_dtype(precision)
    # Test configurations with different numbers of states and output types
    test_configs = [
        # (num_states, num_observables, outputs_list, numpeaks)
        (2, 1, ["state", "observables"], None),
        (4, 2, ["state", "observables", "mean"], None),
        (6, 9, ["state", "observables", "mean", "max"], None),
        (8, 4, ["state", "observables", "mean", "max", "rms"], None),
        (10, 5, ["state", "observables", "mean", "max", "rms", "peaks"], 3),
        (1, 2, ["state", "observables", "mean", "max", "rms", "peaks"], 10)
    ]


    for num_states, num_observables, outputs_list, numpeaks in test_configs:
        # Create saved states and observables lists
        num_samples = 1000
        summarise_every = 500
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

        state_summaries_output = np.zeros((int(num_samples / summarise_every), num_states * summary_output_length),
                                          dtype=precision)
        observable_summaries_output = np.zeros((int(num_samples / summarise_every), num_observables * summary_output_length),
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
                        peaks = np.zeros(numpeaks, dtype=precision)
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
        # set_trace()
        # Define a test kernel
        @cuda.jit()
        def output_functions_test_kernel(state_input,
                                         state_output, observable_output,
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
            shared = cuda.shared.array(0, dtype=numba_precision)

            shared_index = num_states
            current_state = shared[:shared_index]
            current_observable = shared[shared_index:shared_index + num_observables]
            shared_index += num_observables
            state_summaries = shared[shared_index:shared_index + num_states * shared_mem_required_per_summarised_state]
            shared_index += num_states * shared_mem_required_per_summarised_state
            observable_summaries = shared[shared_index:shared_index + num_observables * shared_mem_required_per_summarised_state]

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
                    # set_trace()


        # Launch the kernel
        threads_per_block = 1
        blocks_per_grid = 1

        output_functions_test_kernel[blocks_per_grid, threads_per_block, 0, shared_mem_required](
            d_state_input,
            d_state_output,
            d_observable_output,
            d_state_summaries_output,
            d_observable_summaries_output
        )

        # Synchronize and copy results back
        cuda.synchronize()

        state_output = d_state_output.copy_to_host()
        observable_output = d_observable_output.copy_to_host()
        state_summaries_output = d_state_summaries_output.copy_to_host()
        observable_summaries_output = d_observable_summaries_output.copy_to_host()

        del output_functions_test_kernel
        # Assert that state values were saved correctly
        assert np.allclose(state_input, state_output), "State values were not saved correctly"

        # Assert that observable values were saved correctly
        assert np.allclose(expected_observables_out, observable_output), "Observable values were not saved correctly"

        # Assert that summary metrics were calculated
        # Check that at least some values in the summary metrics are non-zero

        if any(summary in outputs_list for summary in ["mean", "max", "rms", "peaks"]):
            # set_trace()
            assert np.allclose(expected_state_summaries, state_summaries_output), "State summaries didn't match expected values"
            assert np.allclose(expected_obs_summaries, observable_summaries_output), "Observable summaries didn't match expected values"
