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
def do_nothing(buffer, output, summarise_every, ):
    """ no-op function for the first call to chain_metrics, when there are no metrics already chained. """
    pass


def chain_metrics(metric_functions: Sequence, buffer_offsets, buffer_sizes,
        output_offsets, output_sizes, function_params,
        inner_chain=do_nothing, ):
    """
    Take an iterable of functions, then step through recursively, executing the last function in the iterable and the "inner_chain" function. Return the function which executes both, which becomes the "inner_chain" function for the next call, until we have a recursive execution of all functions in the iterable.
    """
    if len(metric_functions) == 0:
        return do_nothing
    current_metric_fn = metric_functions[0]
    current_buffer_offset = buffer_offsets[0]
    current_buffer_size = buffer_sizes[0]
    current_output_offset = output_offsets[0]
    current_output_size = output_sizes[0]
    current_metric_param = function_params[0]

    remaining_metric_fns = metric_functions[1:]
    remaining_buffer_offsets = buffer_offsets[1:]
    remaining_buffer_sizes = buffer_sizes[1:]
    remaining_output_offsets = output_offsets[1:]
    remaining_output_sizes = output_sizes[1:]
    remaining_metric_params = function_params[1:]

    @cuda.jit(device=True, inline=True)
    def wrapper(buffer, output, summarise_every, ):
        inner_chain(buffer, output, summarise_every, )
        current_metric_fn(buffer[
                          current_buffer_offset: current_buffer_offset + current_buffer_size],
                output[
                current_output_offset: current_output_offset + current_output_size],
                summarise_every, current_metric_param, )

    if remaining_metric_fns:
        return chain_metrics(remaining_metric_fns, remaining_buffer_offsets,
                             remaining_buffer_sizes, remaining_output_offsets,
                             remaining_output_sizes, remaining_metric_params,
                             wrapper, )
    else:
        return wrapper


def save_summary_factory(buffer_sizes: SummariesBufferSizes,
        summarised_state_indices: Sequence[int] | ArrayLike,
        summarised_observable_indices: Sequence[int] | ArrayLike,
        summaries_list: Sequence[str], ):
    """Loop through the requested states and observables, applying the chained function to each. Return a device
    function which saves all requested summaries_array."""
    num_summarised_states = len(summarised_state_indices)
    num_summarised_observables = len(summarised_observable_indices)

    save_functions = summary_metrics.save_functions(summaries_list)

    total_buffer_size = buffer_sizes.per_variable
    total_output_size = summary_metrics.summaries_output_height(summaries_list)

    buffer_offsets = summary_metrics.buffer_offsets(summaries_list)
    buffer_sizes_list = summary_metrics.buffer_sizes(summaries_list)
    output_offsets = summary_metrics.output_offsets(summaries_list)
    output_sizes = summary_metrics.output_sizes(summaries_list)
    params = summary_metrics.params(summaries_list)
    num_summary_metrics = len(output_offsets)

    summarise_states = (num_summarised_states > 0) and (
            num_summary_metrics > 0)
    summarise_observables = (num_summarised_observables > 0) and (
            num_summary_metrics > 0)

    summary_metric_chain = chain_metrics(save_functions, buffer_offsets,
                                         buffer_sizes_list, output_offsets,
                                         output_sizes, params, )

    @cuda.jit(device=True, inline=True)
    def save_summary_metrics_func(buffer_state_summaries,
            buffer_observable_summaries, output_state_summaries_window,
            output_observable_summaries_window, summarise_every, ):
        if summarise_states:
            for state_index in range(num_summarised_states):
                buffer_array_slice_start = state_index * total_buffer_size
                out_array_slice_start = state_index * total_output_size

                summary_metric_chain(buffer_state_summaries[
                                     buffer_array_slice_start:buffer_array_slice_start + total_buffer_size],
                        output_state_summaries_window[
                        out_array_slice_start:out_array_slice_start + total_output_size],
                        summarise_every, )

        if summarise_observables:
            for observable_index in range(num_summarised_observables):
                buffer_array_slice_start = observable_index * total_buffer_size
                out_array_slice_start = observable_index * total_output_size

                summary_metric_chain(buffer_observable_summaries[
                                     buffer_array_slice_start:buffer_array_slice_start + total_buffer_size],
                        output_observable_summaries_window[
                        out_array_slice_start:out_array_slice_start + total_output_size],
                        summarise_every, )

    return save_summary_metrics_func
