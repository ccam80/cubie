import attrs.validators
from numba import cuda
from math import sqrt
from numpy.typing import ArrayLike
from typing import Sequence
from CuMC.ForwardSim.OutputHandling.summaries import OutputOffsets, SummaryFlags, TempOffsets

def save_summary_factory(summarised_states: Sequence[int] | ArrayLike,
                         summarised_observables: Sequence[int] | ArrayLike,
                         summaries_list: Sequence[str],
                         n_peaks: int,
                         ):
    """Factory function to create the save summary metrics device function."""
    output_offsets = OutputOffsets(peaks=n_peaks)
    (summarise_mean, summarise_peaks, summarise_max, summarise_rms) = SummaryFlags.unpack_from_list(summaries_list)

    (MEAN_TEMP_OFFSET, PEAKS_TEMP_OFFSET, MAX_TEMP_OFFSET, RMS_TEMP_OFFSET) = TempOffsets.unpack_from_n_peaks(n_peaks)
    (MEAN_OUTPUT_OFFSET, PEAKS_OUTPUT_OFFSET, MAX_OUTPUT_OFFSET, RMS_OUTPUT_OFFSET) = (
        OutputOffsets.unpack_from_n_peaks(n_peaks))

    n_summarised_states = len(summarised_states)
    n_summarised_observables = len(summarised_observables)

    summarise = summarise_mean or summarise_peaks or summarise_max or summarise_rms
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
            output_index = 0
            temp_index = 0

            for i in range(n_summarised_states):
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
        if summarise_observables:
            output_index = 0
            temp_index = 0

            for i in range(n_summarised_observables):

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

