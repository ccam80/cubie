"""Factories that build CUDA device functions to accumulate summary metrics.

This module chains summary metric update functions into a single callable that
is specialised for the requested metrics and tracked variables.

Notes
-----
This implementation is based on the "chain" approach by sklam from
https://github.com/numba/numba/issues/3405. The approach allows dynamic
compilation of only the summary metrics that are actually requested, avoiding
wasted space in working arrays.

The process consists of:
1. A recursive ``chain_metrics`` function that builds a chain of update
   functions.
2. An ``update_summary_factory`` that applies the chained functions to each
   variable.
"""

from typing import Callable, Sequence

from numba import cuda
from numpy.typing import ArrayLike

from cubie.outputhandling.summarymetrics import summary_metrics
from .output_sizes import SummariesBufferSizes


@cuda.jit(device=True, inline=True)
def do_nothing(
    values,
    buffer,
    current_step,
):
    """Provide a no-op device function for empty metric chains.

    Parameters
    ----------
    values
        device array containing the current scalar value (unused).
    buffer
        device array slice reserved for summary accumulation (unused).
    current_step
        Integer or scalar step identifier (unused).

    Returns
    -------
    None
        The device function intentionally performs no operations.

    Notes
    -----
    This function serves as the base case for the recursive chain when no
    summary metrics are configured or as the initial ``inner_chain`` function
    for update operations.
    """
    pass


def chain_metrics(
    metric_functions: Sequence[Callable],
    buffer_offsets: Sequence[int],
    buffer_sizes: Sequence[int],
    function_params: Sequence[object],
    inner_chain: Callable = do_nothing,
) -> Callable:
    """
    Recursively chain summary metric update functions for CUDA execution.

    This function builds a recursive chain of summary metric update functions,
    where each function in the sequence is wrapped with the previous
    functions to create a single callable that updates all metrics.

    Parameters
    ----------
    metric_functions
        Sequence of CUDA device functions for updating summary metrics.
    buffer_offsets
        Sequence of offsets into the metric buffer for each function.
    buffer_sizes
        Sequence of per-metric buffer lengths.
    function_params
        Sequence of parameter payloads passed to each metric function.
    inner_chain
        Callable executed before the current metric; defaults to ``do_nothing``.

    Returns
    -------
    Callable
        CUDA device function that executes all chained metric updates.

    Notes
    -----
    The function uses recursion to build a chain where each level executes
    the inner chain first, then the current metric update function. This
    ensures all requested metrics are updated in the correct order during
    each integration step.
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

    # no cover: start
    @cuda.jit(device=True, inline=True)
    def wrapper(
        value,
        buffer,
        current_step,
    ):
        """Apply the accumulated metric chain before invoking the current metric.

        Parameters
        ----------
        value
            device array element being summarised.
        buffer
            device array slice containing the metric working storage.
        current_step
            Integer or scalar step identifier passed through the chain.

        Returns
        -------
        None
            The device function mutates the metric buffer in place.
        """
        inner_chain(value, buffer, current_step)
        current_fn(
            value,
            buffer[current_offset : current_offset + current_size],
            current_step,
            current_param,
        )

    if remaining_functions:
        return chain_metrics(
            remaining_functions,
            remaining_offsets,
            remaining_sizes,
            remaining_params,
            wrapper,
        )
    else:
        return wrapper
    # no cover: stop


def update_summary_factory(
    buffer_sizes: SummariesBufferSizes,
    summarised_state_indices: Sequence[int] | ArrayLike,
    summarised_observable_indices: Sequence[int] | ArrayLike,
    summaries_list: Sequence[str],
) -> Callable:
    """
    Factory function for creating CUDA device functions to update summary
    metrics.

    This factory generates an optimized CUDA device function that applies
    chained summary metric updates to all requested state and observable
    variables during each integration step.

    Parameters
    ----------
    buffer_sizes
        ``SummariesBufferSizes`` instance that reports per-variable buffer
        lengths.
    summarised_state_indices
        Sequence of state indices to include in summary calculations.
    summarised_observable_indices
        Sequence of observable indices to include in summary calculations.
    summaries_list
        Ordered list of summary metric identifiers registered with
        :mod:`cubie.outputhandling.summarymetrics`.

    Returns
    -------
    Callable
        CUDA device function for updating summary metrics.

    Notes
    -----
    The generated function iterates through all specified state and observable
    variables, applying the chained summary metric updates to accumulate data
    in the appropriate buffer locations during each integration step.
    """
    num_summarised_states = len(summarised_state_indices)
    num_summarised_observables = len(summarised_observable_indices)
    total_buffer_size = buffer_sizes.per_variable
    buffer_offsets = summary_metrics.buffer_offsets(summaries_list)
    num_metrics = len(buffer_offsets)

    summarise_states = (num_summarised_states > 0) and (num_metrics > 0)
    summarise_observables = (num_summarised_observables > 0) and (
        num_metrics > 0
    )

    update_fns = summary_metrics.update_functions(summaries_list)
    buffer_sizes_list = summary_metrics.buffer_sizes(summaries_list)
    params = summary_metrics.params(summaries_list)
    chain_fn = chain_metrics(
        update_fns, buffer_offsets, buffer_sizes_list, params
    )

    # no cover: start
    @cuda.jit(device=True, inline=True)
    def update_summary_metrics_func(
        current_state,
        current_observables,
        state_summary_buffer,
        observable_summary_buffer,
        current_step,
    ):
        """Accumulate summary metrics from the current state sample.

        Parameters
        ----------
        current_state
            device array holding the latest integrator state values.
        current_observables
            device array holding the latest observable values.
        state_summary_buffer
            device array slice used to accumulate state summary data.
        observable_summary_buffer
            device array slice used to accumulate observable summary data.
        current_step
            Integer or scalar step identifier associated with the sample.

        Returns
        -------
        None
            The device function mutates the supplied summary buffers in place.

        Notes
        -----
        The chained metric function is executed for each selected state or
        observable entry, writing into the contiguous buffer segment assigned
        to that variable.
        """
        if summarise_states:
            for idx in range(num_summarised_states):
                start = idx * total_buffer_size
                end = start + total_buffer_size
                chain_fn(
                    current_state[summarised_state_indices[idx]],
                    state_summary_buffer[start:end],
                    current_step,
                )

        if summarise_observables:
            for idx in range(num_summarised_observables):
                start = idx * total_buffer_size
                end = start + total_buffer_size
                chain_fn(
                    current_observables[summarised_observable_indices[idx]],
                    observable_summary_buffer[start:end],
                    current_step,
                )

    # no cover: stop
    return update_summary_metrics_func
