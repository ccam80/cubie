from numba import cuda, float64, int32, from_dtype, float32

"""Functions for saving values inside an integrator kernel, whether
they are states or summary metrics.

All functions should have the same signature and arguments, so that
they're easy to interchange without causing compilation headaches.

Each function will be called once for each state/output, rather than working
with arrays, because at the time of writing numba did not support the particular
slice assignment wanted in the integrator kernels.
"""


def build_output_function(funcs_list):

@cuda.jit(device=True, inline=True)
def save_state(current_index,
               current_state,
               current_observables,
               output,
               output_index,
               saved_states,
               saved_observables):
    """Save the current state value at the specified index.

    Args:
        current_index: Index in the saved_states array
        current_state: Array of current state values
        current_observables: Array of current observable values
        output: Output array to save to
        output_index: Index in the output array to save to
        saved_states: Array of indices of states to save
        saved_observables: Array of indices of observables to save

    Returns:
        The current state value at the specified index
    """
    return current_state[saved_states[current_index]]

@cuda.jit(device=True, inline=True)
def save_observable(current_index,
                   current_state,
                   current_observables,
                   output,
                   output_index,
                   saved_states,
                   saved_observables):
    """Save the current observable value at the specified index.

    Args:
        current_index: Index in the saved_observables array
        current_state: Array of current state values
        current_observables: Array of current observable values
        output: Output array to save to
        output_index: Index in the output array to save to
        saved_states: Array of indices of states to save
        saved_observables: Array of indices of observables to save

    Returns:
        The current observable value at the specified index
    """
    return current_observables[saved_observables[current_index]]


@cuda.jit(device=True, inline=True)
def running_mean(current_index,
                current_state,
                current_observables,
                output,
                output_index,
                saved_states,
                saved_observables):
    """Calculate the running mean of a state or observable.

    Args:
        current_index: Index of the state/observable to calculate mean for
        current_state: Array of current state values
        current_observables: Array of current observable values
        output: Output array containing the current running mean
        output_index: Index in the output array
        saved_states: Array of indices of states to save
        saved_observables: Array of indices of observables to save

    Returns:
        Updated running mean
    """
        old_mean = output[current_index]
        return old_mean + (value - old_mean) / (output_index + 1)