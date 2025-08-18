"""Convenience interface for accessing system values.

This class wraps the :class:`~cubie.systemmodels.SystemValues` instances for
parameters, states, and observables and exposes helper methods for converting
between user-facing labels/indices and internal representations. It also
allows updating default state or parameter values without navigating the full
system hierarchy.
"""
from typing import Dict, List, Optional, Union

import numpy as np

from cubie.systemmodels.SystemValues import SystemValues
from cubie.systemmodels.systems.GenericODE import GenericODE


class SystemInterface:
    """Expose convenient accessors for system values."""

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
    def from_system(cls, system: GenericODE) -> "SystemInterface":
        """Create an accessor from a system model."""
        return cls(system.parameters, system.initial_values, system.observables)

    def update(self,
               updates: Dict[str, float] | None = None,
               silent=False,
               **kwargs) -> None:
        """Update default parameter or state values.

        Parameters
        ----------
        updates : dict, optional
            Mapping of label to new value. Keyword arguments are merged with
            ``updates``.
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

    def state_indices(self,
                      keys_or_indices: Union[List[Union[str, int]], str, int],
                      silent: bool = False
                      ) -> np.ndarray:
        """Convert state labels or indices to a numeric array.

        Parameters
        ----------
        keys_or_indices : Union[List[Union[str, int]], str, int]
            A list of parameter names or indices, or a single name or index.
            If a list is provided, it can contain strings (parameter names)
            or integers (indices).
        silent: bool
            If True, suppresses warnings for unrecognized keys or indices

        Returns
        -------
        indices: np.ndarray
            A numpy array of integer indices corresponding to the provided
            parameter names or indices.
            If a single name or index is provided, returns a 1D array with
            that single index.

        """

        return self.states.get_indices(keys_or_indices, silent=silent)

    def observable_indices(
            self,
            keys_or_indices: Union[List[Union[str, int]], str, int],
            silent: bool = False
            ) -> np.ndarray:
        """Convert observable labels or indices to a numeric array.

        Parameters
        ----------
        keys_or_indices : Union[List[Union[str, int]], str, int]
            A list of parameter names or indices, or a single name or index.
            If a list is provided, it can contain strings (parameter names)
            or integers (indices).
        silent: bool
            If True, suppresses warnings for unrecognized keys or indices

        Returns
        -------
        indices: np.ndarray
            A numpy array of integer indices corresponding to the provided
            parameter names or indices.
            If a single name or index is provided, returns a 1D array with
            that single index.

        """

        return self.observables.get_indices(keys_or_indices, silent=silent)

    def parameter_indices(
        self, keys_or_indices: Union[List[Union[str, int]], str, int],
            silent: bool = False
        ) -> np.ndarray:
        """Convert parameter labels or indices to a numeric array.

        Parameters
        ----------
        keys_or_indices : Union[List[Union[str, int]], str, int]
            A list of parameter names or indices, or a single name or index.
            If a list is provided, it can contain strings (parameter names)
            or integers (indices).
        silent: bool
            If True, suppresses warnings for unrecognized keys or indices

        Returns
        -------
        indices: np.ndarray
            A numpy array of integer indices corresponding to the provided
            parameter names or indices.
            If a single name or index is provided, returns a 1D array with
            that single index.
        """
        return self.parameters.get_indices(keys_or_indices, silent=silent)

    def get_labels(self,
                   values_object: SystemValues,
                   indices: np.ndarray
                   ) -> List[str]:
        """Return labels corresponding to the provided indices.

        Parameters
        ----------
        indices : np.ndarray
            A 1D array of state indices.

        Returns
        -------
        List[str]
            A list of state labels corresponding to the provided indices.
        """

        return values_object.get_labels(indices)

    def state_labels(self,
                     indices: Optional[np.ndarray] = None) -> List[str]:
        """
        Get the labels of the states corresponding to the provided indices.

        Parameters
        ----------
        indices : np.ndarray
            A 1D array of state indices. If None, return all state labels.

        Returns
        -------
        List[str]
            A list of state labels corresponding to the provided indices.
        """
        if indices is None:
            return self.states.names
        return self.get_labels(self.states, indices)

    def observable_labels(self,
                          indices: Optional[np.ndarray] = None) -> List[str]:
        """
        Get the labels of the observables corresponding to the provided
        indices.
        Parameters
        ----------
        indices : np.ndarray
            A 1D array of observable indices. If None, return all observable
            labels.

        Returns
        -------
        List[str]
            A list of observable labels corresponding to the provided indices.
        """
        if indices is None:
            return self.observables.names
        return self.get_labels(self.observables, indices)

    def parameter_labels(self,
                         indices: Optional[np.ndarray] = None) -> List[str]:
        """
        Get the labels of the parameters corresponding to the provided indices.
        Parameters
        ----------
        indices : np.ndarray
            A 1D array of parameter indices. If None, return all parameter
            labels.

        Returns
        -------
        List[str]
            A list of parameter labels corresponding to the provided indices.
        """
        if indices is None:
            return self.parameters.names
        return self.get_labels(self.parameters, indices)

    @property
    def all_input_labels(self) -> List[str]:
        """
        Get all input labels, the union of state and parameter labels.
        Returns
        -------
        List[str]
            A list of all input labels.
        """
        return self.state_labels() + self.parameter_labels()

    @property
    def all_output_labels(self) -> List[str]:
        """
        Get all output labels, the union of state and observable labels.

        Returns
        -------
        List[str]
            A list of all output labels.
        """
        return self.state_labels() + self.observable_labels()


__all__ = ["SystemInterface"]