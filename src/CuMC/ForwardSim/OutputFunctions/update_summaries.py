from numba import cuda
from CuMC.ForwardSim.OutputFunctions.summary_calculations import running_max, running_mean, running_peaks, running_rms
from numpy.typing import ArrayLike
from typing import Sequence


def update_summary_factory(nstates: int,
                           nobs: int,
                           saved_states: Sequence[int] | ArrayLike,
                           saved_observables: Sequence[int] | ArrayLike,
                           summarise: bool,
                           summarise_mean: bool,
                           summarise_peaks: bool,
                           summarise_max: bool,
                           summarise_rms: bool,
                           save_observables: bool,
                           n_peaks: int,
                           ):
    """Factory function to create the update summary metrics device function."""

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

        if summarise:
            temp_array_index = 0

            for k in range(nstates):
                if summarise_mean:
                    running_mean(current_state[saved_states[k]], temp_state_summaries, temp_array_index)
                    temp_array_index += 1  # Magic number - how can we get this out of the function without breaking Numba's rules?
                if summarise_peaks:
                    running_peaks(current_state[saved_states[k]], temp_state_summaries, temp_array_index,
                                  current_step, n_peaks,
                                  )
                    temp_array_index += 3 + n_peaks
                if summarise_max:
                    running_max(current_state[saved_states[k]], temp_state_summaries, temp_array_index)
                    temp_array_index += 1
                if summarise_rms:
                    running_rms(current_state[saved_states[k]], temp_state_summaries, temp_array_index,
                                current_step,
                                )
                    temp_array_index += 1

            if save_observables:
                temp_array_index = 0

                for k in range(nobs):
                    if summarise_mean:
                        running_mean(current_observables[saved_observables[k]], temp_observable_summaries,
                                     temp_array_index,
                                     )
                        temp_array_index += 1  # Magic number - how can we get this out of the function without breaking Numba's rules?
                    if summarise_peaks:
                        running_peaks(current_observables[saved_observables[k]], temp_observable_summaries,
                                      temp_array_index,
                                      current_step, n_peaks,
                                      )
                        temp_array_index += 3 + n_peaks
                    if summarise_max:
                        running_max(current_observables[saved_observables[k]], temp_observable_summaries,
                                    temp_array_index,
                                    )
                        temp_array_index += 1
                    if summarise_rms:
                        running_rms(current_observables[saved_observables[k]], temp_observable_summaries,
                                    temp_array_index, current_step,
                                    )
                        temp_array_index += 1

    return update_summary_metrics_func