from numba import cuda
from math import sqrt


def save_summary_factory(nstates: int,
                         nobs: int,
                         summarise: bool,
                         summarise_mean: bool,
                         summarise_peaks: bool,
                         summarise_max: bool,
                         summarise_rms: bool,
                         save_observables: bool,
                         n_peaks: int,
                         ):
    """Factory function to create the save summary metrics device function."""

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
            temp_state_summaries: temporary array for state summaries, updated by update_summary_metrics_func
            temp_observable_summaries: temporary array for observable summaries, updated by update_summary_metrics_func
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
        if summarise:
            output_index = 0
            temp_index = 0

            for i in range(nstates):
                if summarise_mean:
                    output_state_summaries_slice[output_index] = temp_state_summaries[temp_index] / summarise_every
                    temp_state_summaries[temp_index] = 0.0
                    output_index += 1
                    temp_index += 1

                if summarise_peaks:
                    for p in range(n_peaks):
                        output_state_summaries_slice[output_index + p] = temp_state_summaries[temp_index + 3 + p]
                        temp_state_summaries[temp_index + 3 + p] = 0.0
                    temp_state_summaries[temp_index + 2] = 0.0  # Reset peak counter
                    output_index += n_peaks
                    temp_index += 3 + n_peaks

                if summarise_max:
                    output_state_summaries_slice[output_index] = temp_state_summaries[temp_index]
                    temp_state_summaries[
                        temp_index] = -1.0e30  # A very negative number, to allow us to capture max values greater than this
                    output_index += 1
                    temp_index += 1

                if summarise_rms:
                    output_state_summaries_slice[output_index] = sqrt(
                            temp_state_summaries[temp_index] / summarise_every,
                            )
                    temp_state_summaries[temp_index] = 0.0
                    output_index += 1
                    temp_index += 1
            if save_observables:
                output_index = 0
                temp_index = 0

                for i in range(nobs):

                    if summarise_mean:
                        output_observable_summaries_slice[output_index] = temp_observable_summaries[
                                                                              temp_index] / summarise_every
                        temp_observable_summaries[temp_index] = 0.0
                        output_index += 1
                        temp_index += 1

                    if summarise_peaks:
                        for p in range(n_peaks):
                            output_observable_summaries_slice[output_index + p] = temp_observable_summaries[
                                temp_index + 3 + p]
                            temp_observable_summaries[temp_index + 3 + p] = 0.0
                        temp_observable_summaries[temp_index + 2] = 0.0  # Reset peak counter
                        output_index += n_peaks
                        temp_index += 3 + n_peaks

                    if summarise_max:
                        output_observable_summaries_slice[output_index] = temp_observable_summaries[temp_index]
                        temp_observable_summaries[
                            temp_index] = -1.0e30  # A very negative number, to allow us to capture max values greater than this
                        output_index += 1
                        temp_index += 1

                    if summarise_rms:
                        output_observable_summaries_slice[output_index] = sqrt(
                                temp_observable_summaries[temp_index] / summarise_every,
                                )
                        temp_observable_summaries[temp_index] = 0.0
                        output_index += 1
                        temp_index += 1

    return save_summary_metrics_func

