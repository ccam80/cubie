from numba import cuda
from numpy.typing import ArrayLike
from typing import Sequence
from CuMC.ForwardSim.OutputHandling import summary_metrics

"""This is a modification of the "chain" approach by sklam in https://github.com/numba/numba/issues/3405
to provide an alternative to an iterable of cuda.jit functions. This exists so that we can compile only the
device functions for the update metrics requested, and avoid wasting memory on empty (non-calculated) updates).

The process is made up of:

1. A "chain_metrics" function that takes a list of functions, memory offsets and sizes, and function params, 
and pops the first item off the lists, executing the function on the other arguments. Subsequent calls execute the 
saved function (which becomes the "inner chain"), then the next function on the list. Stepping through the list in 
this way, we end up with a recursive execution of each summary function. For the first iteration, where there are no 
functions in the inner chain, it calls a "do_nothing" function
2. An "update_summary_factory" function that loops through the requested states and observables, applying the
   chained function to each.

Things are pretty verbose in this module, because it was confusing to write, and therefore (I presume) will be 
confusing to read.
"""


@cuda.jit(device=True, inline=True)
def do_nothing(
        values,
        temp_array,
        current_step,
        ):
    """ no-op function for the first call to chain_metrics, when there are no metrics already chained. """
    pass


def chain_metrics(
        metric_functions: Sequence,
        temp_offsets,
        temp_sizes,
        function_params,
        inner_chain=do_nothing,
        ):
    """
    Take iterables of functions and compile-time constants, then step through recursively, executing the previously
    chained functions (the "inner chain" and then the top function in the iterable. Return the function which
    executes both, which becomes the "inner_chain" function for the next call, until we have a recursive  execution
    of all functions in the iterable.
    """
    if len(metric_functions) == 0:
        return do_nothing

    current_fn = metric_functions[0]
    current_offset = temp_offsets[0]
    current_size = temp_sizes[0]
    current_param = function_params[0]

    remaining_functions = metric_functions[1:]
    remaining_offsets = temp_offsets[1:]
    remaining_sizes = temp_sizes[1:]
    remaining_params = function_params[1:]

    @cuda.jit(device=True, inline=True)
    def wrapper(
            value,
            temp_array,
            current_step,
            ):
        inner_chain(value, temp_array, current_step)
        current_fn(value, temp_array[current_offset: current_offset + current_size], current_step, current_param)

    if remaining_functions:
        return chain_metrics(remaining_functions, remaining_offsets, remaining_sizes, remaining_params, wrapper)
    else:
        return wrapper

def update_summary_factory(
        summarised_states: Sequence[int] | ArrayLike,
        summarised_observables: Sequence[int] | ArrayLike,
        summaries_list: Sequence[str],
        ):
    """Loop through the requested states and observables, applying the chained function to each. Return a device
    function which updates all requested summaries."""
    num_summarised_states = len(summarised_states)
    num_summarised_observables = len(summarised_observables)
    total_temporary_size, temp_offsets = summary_metrics.temp_offsets(summaries_list)
    num_metrics = len(summary_metrics.temp_offsets(summaries_list)[1])

    summarise_states = (num_summarised_states > 0) and (num_metrics > 0)
    summarise_observables = (num_summarised_observables > 0) and (num_metrics > 0)

    update_fns = summary_metrics.update_functions(summaries_list)
    temp_sizes = summary_metrics.temp_sizes(summaries_list)
    params = summary_metrics.params(summaries_list)
    chain_fn = chain_metrics(update_fns, temp_offsets, temp_sizes, params)

    @cuda.jit(device=True, inline=True)
    def update_summary_metrics_func(
            current_state,
            current_observables,
            temp_state_summaries,
            temp_observable_summaries,
            current_step,
            ):
        if summarise_states:
            for state in range(num_summarised_states):
                single_variable_slice_start = state * total_temporary_size
                single_variable_slice_end = single_variable_slice_start + total_temporary_size
                chain_fn(
                        current_state[summarised_states[state]],
                        temp_state_summaries[single_variable_slice_start:single_variable_slice_end],
                        current_step,
                        )

        if summarise_observables:
            for observable in range(num_summarised_observables):
                single_variable_slice_start = observable * total_temporary_size
                single_variable_slice_end = single_variable_slice_start + total_temporary_size
                chain_fn(
                        current_observables[summarised_observables[observable]],
                        temp_observable_summaries[single_variable_slice_start:single_variable_slice_end],
                        current_step,
                        )

    return update_summary_metrics_func