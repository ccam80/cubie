from typing import Sequence

from numba import cuda
from numpy.typing import ArrayLike

from cubie.outputhandling import summary_metrics
from .output_sizes import SummariesBufferSizes

"""This is a modification of the "chain" approach by sklam in 
https://github.com/numba/numba/issues/3405
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
def do_nothing(values, buffer, current_step, ):
    """ no-op function for the first call to chain_metrics, when there are no metrics already chained. """
    pass


def chain_metrics(metric_functions: Sequence, buffer_offsets: Sequence[int],
        buffer_sizes, function_params, inner_chain=do_nothing, ):
    """
    Take iterables of functions and compile-time constants, then step through recursively, executing the previously
    chained functions (the "inner chain" and then the top function in the iterable. Return the function which
    executes both, which becomes the "inner_chain" function for the next call, until we have a recursive  execution
    of all functions in the iterable.
    """
    if len(metric_functions) == 0:
        return do_nothing

    current_fn = metric_functions[0]
    current_offset = buffer_offsets[0]
    current_size = buffer_sizes[0]
    current_param = function_params[0]

    remaining_functions = metric_functions[1:]
    remaining_offsets = buffer_offsets[1:]
    remaining_sizes = buffer_sizes[1:]
    remaining_params = function_params[1:]

    @cuda.jit(device=True, inline=True)
    def wrapper(value, buffer, current_step, ):
        inner_chain(value, buffer, current_step)
        current_fn(value,
                   buffer[current_offset: current_offset + current_size],
                   current_step, current_param)

    if remaining_functions:
        return chain_metrics(remaining_functions, remaining_offsets,
                             remaining_sizes, remaining_params, wrapper)
    else:
        return wrapper


def update_summary_factory(buffer_sizes: SummariesBufferSizes,
        summarised_state_indices: Sequence[int] | ArrayLike,
        summarised_observable_indices: Sequence[int] | ArrayLike,
        summaries_list: Sequence[str], ):
    """Loop through the requested states and observables, applying the chained function to each. Return a device
    function which updates all requested summaries_array."""
    num_summarised_states = len(summarised_state_indices)
    num_summarised_observables = len(summarised_observable_indices)
    total_buffer_size = buffer_sizes.per_variable  # Use from SummariesBufferSizes instead of manual calculation
    buffer_offsets = summary_metrics.buffer_offsets(summaries_list)
    num_metrics = len(buffer_offsets)

    summarise_states = (num_summarised_states > 0) and (num_metrics > 0)
    summarise_observables = (num_summarised_observables > 0) and (
            num_metrics > 0)

    update_fns = summary_metrics.update_functions(summaries_list)
    buffer_sizes_list = summary_metrics.buffer_sizes(summaries_list)
    params = summary_metrics.params(summaries_list)
    chain_fn = chain_metrics(update_fns, buffer_offsets, buffer_sizes_list,
                             params)

    @cuda.jit(device=True, inline=True)
    def update_summary_metrics_func(current_state, current_observables,
            state_summary_buffer, observable_summary_buffer, current_step, ):
        if summarise_states:
            for idx in range(num_summarised_states):
                start = idx * total_buffer_size
                end = start + total_buffer_size
                chain_fn(current_state[summarised_state_indices[idx]],
                        state_summary_buffer[start:end], current_step, )

        if summarise_observables:
            for idx in range(num_summarised_observables):
                start = idx * total_buffer_size
                end = start + total_buffer_size
                chain_fn(current_observables[
                    summarised_observable_indices[idx]],
                        observable_summary_buffer[start:end], current_step, )

    return update_summary_metrics_func
