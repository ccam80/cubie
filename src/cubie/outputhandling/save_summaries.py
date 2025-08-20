"""
CUDA device function factory for saving summary metrics.

This module implements a recursive function chaining approach for CUDA device
functions that save summary metrics. It compiles only the requested summary
functions to optimize memory usage and performance.

Notes
-----
This implementation is based on the "chain" approach by sklam from
https://github.com/numba/numba/issues/3405. The approach allows iterating
through jitted functions without passing them as an iterable, which isn't
supported by Numba.

The process consists of:
1. A recursive chain_metrics function that builds a chain of summary functions
2. A save_summary_factory that applies the chained functions to each variable
"""

from typing import Sequence

from numba import cuda
from numpy.typing import ArrayLike

from cubie.outputhandling import summary_metrics
from .output_sizes import SummariesBufferSizes


@cuda.jit(device=True, inline=True)
def do_nothing(
    buffer,
    output,
    summarise_every,
):
    """
    No-operation function for empty metric chains.

    Parameters
    ----------
    buffer : array-like
        Buffer array (unused).
    output : array-like
        Output array (unused).
    summarise_every : int
        Summarization interval (unused).

    Notes
    -----
    This function serves as the base case for the recursive chain when
    no summary metrics are configured or as the initial inner_chain
    function.
    """
    pass


def chain_metrics(
    metric_functions: Sequence,
    buffer_offsets,
    buffer_sizes,
    output_offsets,
    output_sizes,
    function_params,
    inner_chain=do_nothing,
):
    """
    Recursively chain summary metric functions for CUDA execution.

    This function builds a recursive chain of summary metric functions,
    where each function in the sequence is wrapped with the previous
    functions to create a single callable that executes all metrics.

    Parameters
    ----------
    metric_functions : Sequence
        Sequence of CUDA device functions for summary metrics.
    buffer_offsets : Sequence
        Buffer memory offsets for each metric function.
    buffer_sizes : Sequence
        Buffer sizes required by each metric function.
    output_offsets : Sequence
        Output array offsets for each metric function.
    output_sizes : Sequence
        Output sizes for each metric function.
    function_params : Sequence
        Parameters required by each metric function.
    inner_chain : callable, default do_nothing
        Previously chained function to execute before current function.

    Returns
    -------
    callable
        CUDA device function that executes all chained metrics.

    Notes
    -----
    The function uses recursion to build a chain where each level executes
    the inner chain first, then the current metric function. This ensures
    all requested metrics are computed in the correct order.
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

    # no cover: start
    @cuda.jit(device=True, inline=True)
    def wrapper(
        buffer,
        output,
        summarise_every,
    ):
        inner_chain(
            buffer,
            output,
            summarise_every,
        )
        current_metric_fn(
            buffer[
                current_buffer_offset : current_buffer_offset
                + current_buffer_size
            ],
            output[
                current_output_offset : current_output_offset
                + current_output_size
            ],
            summarise_every,
            current_metric_param,
        )

    if remaining_metric_fns:
        return chain_metrics(
            remaining_metric_fns,
            remaining_buffer_offsets,
            remaining_buffer_sizes,
            remaining_output_offsets,
            remaining_output_sizes,
            remaining_metric_params,
            wrapper,
        )
    else:
        return wrapper
    # no cover: stop


def save_summary_factory(
    buffer_sizes: SummariesBufferSizes,
    summarised_state_indices: Sequence[int] | ArrayLike,
    summarised_observable_indices: Sequence[int] | ArrayLike,
    summaries_list: Sequence[str],
):
    """
    Factory function for creating CUDA device functions to save summary metrics.

    This factory generates a CUDA device function that applies chained
    summary metric calculations to all requested state and observable
    variables.

    Parameters
    ----------
    buffer_sizes : SummariesBufferSizes
        Object containing buffer size information for summary calculations.
    summarised_state_indices : Sequence[int] or ArrayLike
        Indices of state variables to include in summary calculations.
    summarised_observable_indices : Sequence[int] or ArrayLike
        Indices of observable variables to include in summary calculations.
    summaries_list : Sequence[str]
        List of summary metric names to compute.

    Returns
    -------
    callable
        Compiled CUDA device function for saving summary metrics.

    Notes
    -----
    The generated function iterates through all specified state and observable
    variables, applying the chained summary metrics to each variable's buffer
    and saving results to the appropriate output arrays.
    """
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
        num_summary_metrics > 0
    )
    summarise_observables = (num_summarised_observables > 0) and (
        num_summary_metrics > 0
    )

    summary_metric_chain = chain_metrics(
        save_functions,
        buffer_offsets,
        buffer_sizes_list,
        output_offsets,
        output_sizes,
        params,
    )

    # no cover: start
    @cuda.jit(device=True, inline=True)
    def save_summary_metrics_func(
        buffer_state_summaries,
        buffer_observable_summaries,
        output_state_summaries_window,
        output_observable_summaries_window,
        summarise_every,
    ):
        """
        Save summary metrics from buffers to output arrays.

        Parameters
        ----------
        buffer_state_summaries : array-like
            Buffer containing accumulated state summary data.
        buffer_observable_summaries : array-like
            Buffer containing accumulated observable summary data.
        output_state_summaries_window : array-like
            Output array window for state summary results.
        output_observable_summaries_window : array-like
            Output array window for observable summary results.
        summarise_every : int
            Number of steps between summary calculations.

        Notes
        -----
        This device function processes each state and observable variable
        by applying the chained summary metrics to extract final results
        from the accumulation buffers into the output arrays.
        """
        if summarise_states:
            for state_index in range(num_summarised_states):
                buffer_array_slice_start = state_index * total_buffer_size
                out_array_slice_start = state_index * total_output_size

                summary_metric_chain(
                    buffer_state_summaries[
                        buffer_array_slice_start : buffer_array_slice_start
                        + total_buffer_size
                    ],
                    output_state_summaries_window[
                        out_array_slice_start : out_array_slice_start
                        + total_output_size
                    ],
                    summarise_every,
                )

        if summarise_observables:
            for observable_index in range(num_summarised_observables):
                buffer_array_slice_start = observable_index * total_buffer_size
                out_array_slice_start = observable_index * total_output_size

                summary_metric_chain(
                    buffer_observable_summaries[
                        buffer_array_slice_start : buffer_array_slice_start
                        + total_buffer_size
                    ],
                    output_observable_summaries_window[
                        out_array_slice_start : out_array_slice_start
                        + total_output_size
                    ],
                    summarise_every,
                )

    # no cover: stop
    return save_summary_metrics_func
