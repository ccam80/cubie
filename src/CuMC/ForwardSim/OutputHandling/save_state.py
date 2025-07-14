from numba import cuda
from numpy.typing import ArrayLike
from typing import Sequence


def save_state_factory(nstates: int,
                       nobs: int,
                       saved_states: Sequence[int] | ArrayLike,
                       saved_observables: Sequence[int] | ArrayLike,
                       save_state: bool,
                       save_observables: bool,
                       save_time: bool,
                       ):
    @cuda.jit(device=True, inline=True)
    def save_state_func(current_state,
                        current_observables,
                        output_states_slice,
                        output_observables_slice,
                        current_step,
                        ):
        """Save the current state at the specified index.
        Arguments:
            current_state: current state array, containing the values of the states
            current_observables: current observables array, containing the values of the observables
            output_states_slice: current slice of output array for states, to be updated
            output_observables_slice: current slice of output array for observables, to be updated
            current_step: current step number, used for saving time if required

        Returns:
            None, modifies the output_states_slice and output_observables_slice in-place.
        """
        if save_state:
            for k in range(nstates):
                output_states_slice[k] = current_state[saved_states[k]]

        if save_observables:
            for m in range(nobs):
                output_observables_slice[m] = current_observables[saved_observables[m]]

        if save_time:
            # Append time at the end of the state output
            output_states_slice[nstates] = current_step

    return save_state_func