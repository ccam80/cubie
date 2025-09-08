"""Convenience interface for accessing system values.

This module provides the SystemInterface class which wraps SystemValues
instances for parameters, states, and observables. It exposes helper methods
for converting between user-facing labels/indices and internal representations.

Classes
-------
SystemInterface
    Convenient accessor for system values with label/index conversion methods.

Notes
-----
The SystemInterface allows updating default state or parameter values without
navigating the full system hierarchy, providing a simplified interface for
common operations.
"""

from typing import Dict, List, Optional, Union

import numpy as np

from cubie.odesystems.baseODE import BaseODE
from cubie.odesystems.SystemValues import SystemValues


class SystemInterface:
    """Convenient accessor for system values.

    This class wraps SystemValues instances for parameters, states, and
    observables and provides methods for converting between user-facing
    labels/indices and internal representations.

    Parameters
    ----------
    parameters : SystemValues
        System parameter values object.
    states : SystemValues
        System state values object.
    observables : SystemValues
        System observable values object.
    """

    def __init__(
        self,
        parameters: SystemValues,
        states: SystemValues,
        observables: SystemValues,
    ):
        self.parameters = parameters
        self.states = states
        self.observables = observables

    @classmethod
    def from_system(cls, system: BaseODE) -> "SystemInterface":
        """Create a SystemInterface from a system model.

        Parameters
        ----------
        system : BaseODE
            The system model to create an interface for.

        Returns
        -------
        SystemInterface
            A new SystemInterface instance wrapping the system's values.
        """
        return cls(
            system.parameters, system.initial_values, system.observables
        )

    def update(
        self,
        updates: Dict[str, float] | None = None,
        silent: bool = False,
        **kwargs,
    ) -> Optional[set]:
        """Update default parameter or state values.

        Parameters
        ----------
        updates : dict of str to float, optional
            Mapping of label to new value. If None, only keyword arguments
            are used for updates.
        silent : bool, default False
            If True, suppresses KeyError for unrecognized update keys.
        **kwargs
            Additional keyword arguments merged with ``updates``. Each key-value
            pair represents a label-value mapping for updating system values.

        Returns
        -------
        set or None
            Set of recognized update keys that were successfully applied.
            Returns None if no updates were provided.

        Raises
        ------
        KeyError
            If ``silent`` is False and unrecognized update keys are provided.


        Notes
        -----
        The method attempts to update both parameters and states. Updates are
        applied to whichever SystemValues object recognizes each key.
        """
        if updates is None:
            updates = {}
        if kwargs:
            updates.update(kwargs)
        if not updates:
            return

        all_unrecognized = set(updates.keys())
        for values_object in (self.parameters, self.states):
            recognized = values_object.update_from_dict(updates, silent=True)
            all_unrecognized -= recognized

        if all_unrecognized:
            if not silent:
                unrecognized_list = sorted(all_unrecognized)
                raise KeyError(
                    "The following updates were not recognized by the system. Was this a typo?: "
                    f"{unrecognized_list}"
                )

        recognized = set(updates.keys()) - all_unrecognized
        return recognized

    def state_indices(
        self,
        keys_or_indices: Optional[
            Union[List[Union[str, int]], str, int]
        ] = None,
        silent: bool = False,
    ) -> np.ndarray:
        """Convert state labels or indices to a numeric array.

        Parameters
        ----------
        keys_or_indices : list of {str, int} or str or int or None, default
        None.
            State names, indices, or a mix of both. Can be a single
            name/index or a list containing strings (state names) and/or
            integers (indices). None returns all state indices.
        silent : bool, default False
            If True, suppresses warnings for unrecognized keys or indices.

        Returns
        -------
        np.ndarray
            Array of integer indices corresponding to the provided state
            names or indices. Always returns a 1D array, even for single inputs.
        """
        if keys_or_indices is None:
            keys_or_indices = self.states.names
        return self.states.get_indices(keys_or_indices, silent=silent)

    def observable_indices(
        self,
        keys_or_indices: Union[List[Union[str, int]], str, int],
        silent: bool = False,
    ) -> np.ndarray:
        """Convert observable labels or indices to a numeric array.

        Parameters
        ----------
        keys_or_indices : list of {str, int} or str or int or None, default
        None.
            Observable names, indices, or a mix of both. Can be a single
            name/index or a list containing strings (observable names)
            and/or integers (indices). None returns all observable indices.
        silent : bool, default False
            If True, suppresses warnings for unrecognized keys or indices.
        Returns
        -------
        np.ndarray
            Array of integer indices corresponding to the provided observable
            names or indices. Always returns a 1D array, even for single inputs.

        """
        if keys_or_indices is None:
            keys_or_indices = self.observables.names
        return self.observables.get_indices(keys_or_indices, silent=silent)

    def parameter_indices(
        self,
        keys_or_indices: Union[List[Union[str, int]], str, int],
        silent: bool = False,
    ) -> np.ndarray:
        """Convert parameter labels or indices to a numeric array.

        Parameters
        ----------
        keys_or_indices : list of {str, int} or str or int
            Parameter names, indices, or a mix of both. Can be a single name/index
            or a list containing strings (parameter names) and/or integers (indices).
        silent : bool, default False
            If True, suppresses warnings for unrecognized keys or indices.

        Returns
        -------
        np.ndarray
            Array of integer indices corresponding to the provided parameter
            names or indices. Always returns a 1D array, even for single inputs.
        """
        return self.parameters.get_indices(keys_or_indices, silent=silent)

    def get_labels(
        self, values_object: SystemValues, indices: np.ndarray
    ) -> List[str]:
        """Return labels corresponding to the provided indices.

        Parameters
        ----------
        values_object : SystemValues
            The SystemValues object to retrieve labels from.
        indices : np.ndarray
            A 1D array of integer indices.

        Returns
        -------
        list of str
            List of labels corresponding to the provided indices.
        """
        return values_object.get_labels(indices)

    def state_labels(self, indices: Optional[np.ndarray] = None) -> List[str]:
        """Get the labels of the states corresponding to the provided indices.

        Parameters
        ----------
        indices : np.ndarray, optional
            A 1D array of state indices. If None, return all state labels.

        Returns
        -------
        list of str
            List of state labels corresponding to the provided indices.
            If indices is None, returns all state labels.
        """
        if indices is None:
            return self.states.names
        return self.get_labels(self.states, indices)

    def observable_labels(
        self, indices: Optional[np.ndarray] = None
    ) -> List[str]:
        """Get the labels of observables corresponding to the provided indices.

        Parameters
        ----------
        indices : np.ndarray, optional
            A 1D array of observable indices. If None, return all observable
            labels.

        Returns
        -------
        list of str
            List of observable labels corresponding to the provided indices.
            If indices is None, returns all observable labels.
        """
        if indices is None:
            return self.observables.names
        return self.get_labels(self.observables, indices)

    def parameter_labels(
        self, indices: Optional[np.ndarray] = None
    ) -> List[str]:
        """Get the labels of parameters corresponding to the provided indices.

        Parameters
        ----------
        indices : np.ndarray, optional
            A 1D array of parameter indices. If None, return all parameter
            labels.

        Returns
        -------
        list of str
            List of parameter labels corresponding to the provided indices.
            If indices is None, returns all parameter labels.
        """
        if indices is None:
            return self.parameters.names
        return self.get_labels(self.parameters, indices)

    @property
    def all_input_labels(self) -> List[str]:
        """Get all input labels, the union of state and parameter labels.

        Returns
        -------
        list of str
            List containing all state labels followed by all parameter labels.

        Notes
        -----
        This property provides a convenient way to access all system inputs
        (states and parameters) in a single list.
        """
        return self.state_labels() + self.parameter_labels()

    @property
    def all_output_labels(self) -> List[str]:
        """Get all output labels, the union of state and observable labels.

        Returns
        -------
        list of str
            List containing all state labels followed by all observable labels.

        Notes
        -----
        This property provides a convenient way to access all system outputs
        (states and observables) in a single list.
        """
        return self.state_labels() + self.observable_labels()


__all__ = ["SystemInterface"]
