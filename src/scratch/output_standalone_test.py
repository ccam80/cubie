# src/scratch/output_standalone_test.py
import os
os.environ.setdefault('NUMBA_ENABLE_CUDASIM', '1')
os.environ.setdefault('NUMBA_CUDA_LOG_LEVEL', 'DEBUG')
os.environ["NUMBA_CUDA_DEBUGINFO"] = "1"
os.environ["NUMBA_OPT"] = "0"

import numpy as np
from numba import cuda, from_dtype
from CuMC.ForwardSim.OutputHandling.output_functions import OutputFunctions

# 1. Test settings matching large_dataset-float32
test_configuration = {
    'num_samples': 1000,
    'num_summaries': 10,
    'num_states': 10,
    'num_observables': 10,
    'saved_states': [0, 1],
    'saved_observables': [0, 1],
    'random_scale': 1.0,
    'output_types': ["state", "observables", "mean", "max", "rms", "peaks[3]"],
    'precision_dtype': np.float32,
    'use_shared_memory': True,
}

precision_dtype = test_configuration['precision_dtype']
number_of_samples = test_configuration['num_samples']
number_of_states = test_configuration['num_states']
number_of_observables = test_configuration['num_observables']
random_value_scale = test_configuration['random_scale']
output_type_list = test_configuration['output_types']
use_shared_memory = test_configuration['use_shared_memory']
number_of_summaries = test_configuration['num_summaries']

# 2. Instantiate OutputFunctions
output_functions = OutputFunctions(
    number_of_states,
    number_of_observables,
    output_type_list,
    test_configuration['saved_states'],
    test_configuration['saved_observables']
)

# 3. Generate random inputs
np.random.seed(0)
host_state_input = (
    np.random.rand(number_of_samples, number_of_states) * random_value_scale
).astype(precision_dtype)
host_observable_input = (
    np.random.rand(number_of_samples, number_of_observables) * random_value_scale
).astype(precision_dtype)

# 4. Prepare empty output arrays
saved_state_count = output_functions.n_saved_states
if 'time' in output_type_list:
    saved_state_count += 1

saved_observable_count = output_functions.n_saved_observables

host_state_output = np.zeros(
    (number_of_samples, saved_state_count),
    dtype=precision_dtype
)
host_observable_output = np.zeros(
    (number_of_samples, saved_observable_count),
    dtype=precision_dtype
)

summary_output_height = output_functions.memory_per_summarised_variable['output']
host_state_summaries = np.zeros(
    (number_of_summaries,
     summary_output_height * output_functions.n_summarised_states),
    dtype=precision_dtype
)
host_observable_summaries = np.zeros(
    (number_of_summaries,
     summary_output_height * output_functions.n_summarised_observables),
    dtype=precision_dtype
)

# 5. Build the CUDA test kernel with descriptive names
samples_per_summary = number_of_samples // number_of_summaries
numba_precision = from_dtype(precision_dtype)

save_state_func = output_functions.save_state_func
update_summaries_func = output_functions.update_summaries_func
save_summary_metrics_func = output_functions.save_summary_metrics_func
@cuda.jit()
def output_functions_test_kernel(
    device_state_input,
    device_observable_input,
    device_state_output,
    device_observable_output,
    device_state_summaries,
    device_observable_summaries
):
    thread_x = cuda.threadIdx.x
    block_x = cuda.blockIdx.x
    if thread_x != 0 or block_x != 0:
        return

    # choose shared or local buffers
    if use_shared_memory:
        shared_buffer = cuda.shared.array(0, dtype=numba_precision)
        state_end_index = number_of_states
        observable_end_index = state_end_index + number_of_observables
        current_state_buffer = shared_buffer[:state_end_index]
        current_observable_buffer = shared_buffer[state_end_index:observable_end_index]
        state_summary_temp_buffer = shared_buffer[
            observable_end_index:
            observable_end_index + output_functions.memory_per_summarised_variable['temporary'] * output_functions.n_summarised_states
        ]
        observable_summary_temp_buffer = shared_buffer[
            observable_end_index + output_functions.memory_per_summarised_variable['temporary'] * output_functions.n_summarised_states:
            observable_end_index + output_functions.memory_per_summarised_variable['temporary'] * (
                output_functions.n_summarised_states + output_functions.n_summarised_observables
            )
        ]
    else:
        current_state_buffer = cuda.local.array(number_of_states, dtype=numba_precision)
        current_observable_buffer = cuda.local.array(number_of_observables, dtype=numba_precision)
        state_summary_temp_buffer = cuda.local.array(
            output_functions.memory_per_summarised_variable['temporary'] * output_functions.n_summarised_states,
            dtype=numba_precision
        )
        observable_summary_temp_buffer = cuda.local.array(
            output_functions.memory_per_summarised_variable['temporary'] * output_functions.n_summarised_observables,
            dtype=numba_precision
        )

    # initialize buffers to zero
    for idx in range(number_of_states):
        current_state_buffer[idx] = 0
    for idx in range(number_of_observables):
        current_observable_buffer[idx] = 0
    for idx in range(state_summary_temp_buffer.shape[0]):
        state_summary_temp_buffer[idx] = 0
    for idx in range(observable_summary_temp_buffer.shape[0]):
        observable_summary_temp_buffer[idx] = 0

    # main processing loop
    for sample_index in range(device_state_input.shape[0]):
        for state_index in range(number_of_states):
            current_state_buffer[state_index] = device_state_input[sample_index, state_index]
        for obs_index in range(number_of_observables):
            current_observable_buffer[obs_index] = device_observable_input[sample_index, obs_index]


        save_state_func(
            current_state_buffer,
            current_observable_buffer,
            device_state_output[sample_index, :],
            device_observable_output[sample_index, :],
            sample_index
        )

        update_summaries_func(
            current_state_buffer,
            current_observable_buffer,
            state_summary_temp_buffer,
            observable_summary_temp_buffer,
            sample_index
        )



        if (sample_index + 1) % samples_per_summary == 0:
            # if thread_x == 0 and block_x == 0:
            #     from pdb import set_trace as bp
            #     bp()
            summary_index = (sample_index + 1) // samples_per_summary - 1
            save_summary_metrics_func(
                state_summary_temp_buffer,
                observable_summary_temp_buffer,
                device_state_summaries[summary_index, :],
                device_observable_summaries[summary_index, :],
                samples_per_summary
            )

# 6. Copy data to device and launch the kernel
device_state_input = cuda.to_device(host_state_input)
device_observable_input = cuda.to_device(host_observable_input)
device_state_output = cuda.to_device(host_state_output)
device_observable_output = cuda.to_device(host_observable_output)
device_state_summaries = cuda.to_device(host_state_summaries)
device_observable_summaries = cuda.to_device(host_observable_summaries)

kernel_shared_size = number_of_states + number_of_observables
kernel_temp_shared = (
    output_functions.n_summarised_states + output_functions.n_summarised_observables
) * output_functions.memory_per_summarised_variable['temporary']
dynamic_shared_memory_bytes = (
    kernel_shared_size + kernel_temp_shared
) * np.dtype(precision_dtype).itemsize

output_functions_test_kernel[1, 1, 0, dynamic_shared_memory_bytes](
    device_state_input,
    device_observable_input,
    device_state_output,
    device_observable_output,
    device_state_summaries,
    device_observable_summaries
)
cuda.synchronize()

# 7. Retrieve outputs to host
host_state_output = device_state_output.copy_to_host()
host_observable_output = device_observable_output.copy_to_host()
host_state_summaries = device_state_summaries.copy_to_host()
host_observable_summaries = device_observable_summaries.copy_to_host()
