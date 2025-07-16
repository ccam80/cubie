import attrs.validators
from numba import cuda
from math import sqrt
from numpy.typing import ArrayLike
from typing import Sequence
from CuMC.ForwardSim.OutputHandling import summary_metrics

def save_summary_factory(summarised_states: Sequence[int] | ArrayLike,
                         summarised_observables: Sequence[int] | ArrayLike,
                         summaries_list: Sequence[str],
                         ):
    """Factory function to create the save summary metrics device function."""
    n_summarised_states = len(summarised_states)
    n_summarised_observables = len(summarised_observables)

    # Get only the requested metrics data (no flags needed since we removed invalid metrics)
    save_functions = summary_metrics.save_functions(summaries_list)
    total_temp_size, temp_offsets_tuple = summary_metrics.temp_offsets(summaries_list)
    total_output_size, output_offsets_tuple = summary_metrics.output_offsets(summaries_list)
    temp_sizes = summary_metrics.temp_sizes(summaries_list)
    output_sizes = summary_metrics.output_sizes(summaries_list)
    params = summary_metrics.params(summaries_list)

    num_metrics = len(save_functions)
    summarise = num_metrics > 0  # We have metrics to summarise if any were requested
    summarise_observables = (n_summarised_observables > 0) and summarise
    summarise_state = (n_summarised_states > 0) and summarise

    # noinspection DuplicatedCode
    @cuda.jit(device=True, inline=True)
    def save_summary_metrics_func(temp_state_summaries,
                                  temp_observable_summaries,
                                  output_state_summaries_slice,
                                  output_observable_summaries_slice,
                                  summarise_every,
                                  ):
        """Update the summary metrics based on the current state and observables.
        Arguments:
            temp_state_summaries: temporary array for state summaries, updated by update_summaries_func
            temp_observable_summaries: temporary array for observable summaries, updated by update_summaries_func
            output_state_summaries_slice: current slice of output array for state summaries, to be updated
            output_observable_summaries_slice: current slice of output array for observable summaries, to be updated
            summarise_every: number of steps to average over for the summaries

        Returns:
            None, modifies the output_state_summaries_slice and output_observable_summaries_slice in-place.

        Efficiency note: this function contains duplicated code for each array type. Using a common
        function to handle the cases would be more efficient, but numba does not seem to recognise a change in the
        compile-time constants nstates/nobs for the same function. This is probably expected behaviour for numba,
        it seems reasonable.
        """
        if summarise_state:
            for k in range(n_summarised_states):
                for i in range(num_metrics):
                    # No need to check flags since we only have requested metrics
                    save_func = save_functions[i]
                    temp_array_index_start = temp_offsets_tuple[i] + k * total_temp_size
                    temp_array_index_end = temp_array_index_start + temp_sizes[i]
                    output_array_index_start = output_offsets_tuple[i] + k * total_output_size
                    output_array_index_end = output_array_index_start + output_sizes[i]

                    save_func(temp_state_summaries[temp_array_index_start:temp_array_index_end],
                             output_state_summaries_slice[output_array_index_start:output_array_index_end],
                             summarise_every,
                             params[i],
                             )

        if summarise_observables:
            for k in range(n_summarised_observables):
                for i in range(num_metrics):
                    # No need to check flags since we only have requested metrics
                    save_func = save_functions[i]
                    temp_array_index_start = temp_offsets_tuple[i] + k * total_temp_size
                    temp_array_index_end = temp_array_index_start + temp_sizes[i]
                    output_array_index_start = output_offsets_tuple[i] + k * total_output_size
                    output_array_index_end = output_array_index_start + output_sizes[i]

                    save_func(temp_observable_summaries[temp_array_index_start:temp_array_index_end],
                             output_observable_summaries_slice[output_array_index_start:output_array_index_end],
                             summarise_every,
                             params[i],
                             )

    return save_summary_metrics_func
