"""
CUDA device function factory for saving state and observable values.

This module provides a factory function that generates CUDA device functions
for saving current state and observable values to output arrays during
integration.
"""

from typing import Sequence

from numba import cuda
from numpy.typing import ArrayLike


def save_state_factory(
    saved_state_indices: Sequence[int] | ArrayLike,
    saved_observable_indices: Sequence[int] | ArrayLike,
    save_state: bool,
    save_observables: bool,
    save_time: bool,
):
    """
    Factory function for creating CUDA device functions to save state data.

    This factory generates CUDA device functions that save current state and
    observable values to output arrays, with optional time saving. The
    generated function is specialized for the provided indices and flags.

    Parameters
    ----------
    saved_state_indices : Sequence[int] or ArrayLike
        Indices of state variables to save in the output.
    saved_observable_indices : Sequence[int] or ArrayLike
        Indices of observable variables to save in the output.
    save_state : bool
        Whether to save state variables.
    save_observables : bool
        Whether to save observable variables.
    save_time : bool
        Whether to append time values to the state output.

    Returns
    -------
    Callable
        Compiled CUDA device function for saving state data.

    Notes
    -----
    The generated function takes arguments (current_state,
    current_observables, output_states_slice, output_observables_slice,
    current_step) and modifies the output slices in-place. If save_time is
    True, the current step is appended to the state output after the state
    variables.
    """
    # Extract sizes from heights object
    nobs = len(saved_observable_indices)
    nstates = len(saved_state_indices)

    @cuda.jit(device=True, inline=True)
    def save_state_func(
        current_state,
        current_observables,
        output_states_slice,
        output_observables_slice,
        current_step,
    ):
        """
        Save current state and observables to output arrays.

        Parameters
        ----------
        current_state : array-like
            Current state values from the integrator.
        current_observables : array-like
            Current observable values computed from the state.
        output_states_slice : array-like
            Output array slice for state values, modified in-place.
        output_observables_slice : array-like
            Output array slice for observable values, modified in-place.
        current_step : int or float
            Current integration step number or time value.

        Notes
        -----
        The function modifies the output slices directly without returning
        values. If time saving is enabled, the current step/time is appended
        after the state variables in the state output slice.
        """
        # no cover: start
        if save_state:
            for k in range(nstates):
                output_states_slice[k] = current_state[saved_state_indices[k]]

        if save_observables:
            for m in range(nobs):
                output_observables_slice[m] = current_observables[
                    saved_observable_indices[m]
                ]

        if save_time:
            # Append time at the end of the state output
            output_states_slice[nstates] = current_step
        # no cover: stop

    return save_state_func
