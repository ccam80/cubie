from numba import cuda
from CuMC.ForwardSim.OutputHandling import summary_metrics
from numpy.typing import ArrayLike
from typing import Sequence

def update_summary_factory(summarised_states: Sequence[int] | ArrayLike,
                           summarised_observables: Sequence[int] | ArrayLike,
                           summaries_list: Sequence[str],
                           ):
    """Factory function to create the update summary metrics device function."""
    n_summarised_states = len(summarised_states)
    n_summarised_observables = len(summarised_observables)
    update_functions = summary_metrics.update_functions(summaries_list)
    total_temp_size, temp_offsets_tuple = summary_metrics.temp_offsets(summaries_list)
    temp_sizes = summary_metrics.temp_sizes(summaries_list)
    params = summary_metrics.params(summaries_list)

    num_metrics = len(update_functions)
    summarise = num_metrics > 0  # We have metrics to summarise if any were requested
    summarise_observables = (n_summarised_observables > 0) and summarise
    summarise_state = (n_summarised_states > 0) and summarise

    @cuda.jit(device=True, inline=True)
    def update_summary_metrics_func(current_state,
                                    current_observables,
                                    temp_state_summaries,
                                    temp_observable_summaries,
                                    current_step,
                                    ):
        """Update the summary metrics based on the current state and observables.

        Arguments:
            current_state: current state array, containing the values of the states
            current_observables: current observables array, containing the values of the observables
            temp_state_summaries: temporary array for state summaries and working values, to be updated
            temp_observable_summaries: temporary array for observable summaries and working values, to be updated
            current_step: current step number, used for calculating peak values or other time-dependent summaries

        Returns:
            None, modifies the temp_state_summaries and temp_observable_summaries in-place.

            """
        # Efficiency note: this function contains duplicated code for each array type. Using a common
        # function to handle the cases would be more efficient, but the function can't be compiled twice with
        # different compile-time constants.

        if summarise_state:
            for k in range(n_summarised_states):
                for i in range(num_metrics):
                    # No need to check flags since we only have requested metrics
                    update_func = update_functions[i]
                    temp_array_index_start = temp_offsets_tuple[i] + k * total_temp_size
                    temp_array_index_end = temp_array_index_start + temp_sizes[i]
                    update_func(current_state[summarised_states[k]],
                                temp_state_summaries[temp_array_index_start:temp_array_index_end],
                                current_step,
                                params[i],
                                )

        if summarise_observables:
            for k in range(n_summarised_observables):
                for i in range(num_metrics):
                    # No need to check flags since we only have requested metrics
                    update_func = update_functions[i]
                    temp_array_index_start = temp_offsets_tuple[i] + k * total_temp_size
                    temp_array_index_end = temp_array_index_start + temp_sizes[i]
                    update_func(current_observables[summarised_observables[k]],
                                temp_observable_summaries[temp_array_index_start:temp_array_index_end],
                                current_step,
                                params[i],
                                )
        #         if summarise_mean:
        #             running_mean(current_state[summarised_states[k]], temp_state_summaries, temp_array_index)
        #             temp_array_index += 1  # Magic number - how can we get this out of the function without breaking Numba's rules?
        #         if summarise_peaks:
        #             running_peaks(current_state[summarised_states[k]], temp_state_summaries, temp_array_index,
        #                           current_step, n_peaks,
        #                           )
        #             temp_array_index += 3 + n_peaks
        #         if summarise_max:
        #             running_max(current_state[summarised_states[k]], temp_state_summaries, temp_array_index)
        #             temp_array_index += 1
        #         if summarise_rms:
        #             running_rms(current_state[summarised_states[k]], temp_state_summaries, temp_array_index,
        #                         current_step,
        #                         )
        #             temp_array_index += 1
        #
        # if summarise_observables:
        #     temp_array_index = 0
        #
        #     for k in range(nobs):
        #         if summarise_mean:
        #             running_mean(current_observables[summarised_observables[k]], temp_observable_summaries,
        #                          temp_array_index,
        #                          )
        #             temp_array_index += 1  # Magic number - how can we get this out of the function without breaking Numba's rules?
        #         if summarise_peaks:
        #             running_peaks(current_observables[summarised_observables[k]], temp_observable_summaries,
        #                           temp_array_index,
        #                           current_step, n_peaks,
        #                           )
        #             temp_array_index += 3 + n_peaks
        #         if summarise_max:
        #             running_max(current_observables[summarised_observables[k]], temp_observable_summaries,
        #                         temp_array_index,
        #                         )
        #             temp_array_index += 1
        #         if summarise_rms:
        #             running_rms(current_observables[summarised_observables[k]], temp_observable_summaries,
        #                         temp_array_index, current_step,
        #                         )
        #             temp_array_index += 1

    return update_summary_metrics_func