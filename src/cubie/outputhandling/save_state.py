from typing import Sequence

from numba import cuda
from numpy.typing import ArrayLike


def save_state_factory(saved_state_indices: Sequence[int] | ArrayLike,
                       saved_observable_indices: Sequence[int] | ArrayLike,
                       save_state: bool, save_observables: bool,
                       save_time: bool, ):
    # Extract sizes from heights object
    nobs = len(saved_observable_indices)
    nstates = len(saved_state_indices)

    @cuda.jit(device=True, inline=True)
    def save_state_func(current_state, current_observables,
                        output_states_slice, output_observables_slice,
                        current_step, ):
        """Save the current state at the specified index.
        Arguments:
            current_state: current state array, containing the values of the
            states
            current_observables: current observables array, containing the values of the observables
            output_states_slice: current slice of output array for states, to be updated
            output_observables_slice: current slice of output array for observables, to be updated
            current_step: current step number, used for saving time if required

        Returns:
            None, modifies the output_states_slice and output_observables_slice in-place.
        """
        if save_state:
            for k in range(nstates):
                output_states_slice[k] = current_state[saved_state_indices[k]]

        if save_observables:
            for m in range(nobs):
                output_observables_slice[m] = current_observables[
                    saved_observable_indices[m]]

        if save_time:
            # Append time at the end of the state output
            output_states_slice[nstates] = current_step

    return save_state_func
