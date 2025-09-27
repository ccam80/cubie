"""Factories that build CUDA device functions for saving solver state.

This module exposes a single factory that specialises a CUDA device function
for writing selected state, observable, and time values into output buffers
during integration.
"""

from typing import Callable, Sequence

from numba import cuda
from numpy.typing import ArrayLike


def save_state_factory(
    saved_state_indices: Sequence[int] | ArrayLike,
    saved_observable_indices: Sequence[int] | ArrayLike,
    save_state: bool,
    save_observables: bool,
    save_time: bool,
) -> Callable:
    """Build a CUDA device function that stores solver state and observables.

    Parameters
    ----------
    saved_state_indices
        Sequence of state indices to write into the state output window.
    saved_observable_indices
        Sequence of observable indices to write into the observable output
        window.
    save_state
        When ``True`` the generated function copies the current state slice.
    save_observables
        When ``True`` the generated function copies the current observable
        slice.
    save_time
        When ``True`` the generated function appends the current step to the
        end of the state output window.

    Returns
    -------
    Callable
        CUDA device function that writes state, observable, and optional time
        values into contiguous output buffers.

    Notes
    -----
    The generated device function expects ``current_state``,
    ``current_observables``, ``output_states_slice``,
    ``output_observables_slice``, and ``current_step`` arguments and mutates
    the output slices in place.
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
        """Write selected state, observable, and time values to device buffers.

        Parameters
        ----------
        current_state
            device array containing the latest integrator state values.
        current_observables
            device array containing the latest observable values.
        output_states_slice
            device array window that receives saved state (and optional time)
            values in place.
        output_observables_slice
            device array window that receives saved observable values in
            place.
        current_step
            Scalar step or time value associated with the current sample.

        Returns
        -------
        None
            The device function mutates the provided output buffers in place.

        Notes
        -----
        When ``save_time`` is ``True`` the current step value is stored at the
        first slot immediately after the copied state values.
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
