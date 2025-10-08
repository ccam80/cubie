"""Convenience interface for accessing system values.

This module provides :class:`SystemInterface`, which wraps
:class:`cubie.odesystems.SystemValues` instances for parameters, states, and
observables. It exposes helper methods for converting between user-facing
labels or indices and internal representations.

Notes
-----
The interface allows updating default state or parameter values without
navigating the full system hierarchy, providing a simplified entry point for
common operations.
"""

from typing import Any, Dict, List, Optional, Set, Union

import numpy as np

from cubie.odesystems.baseODE import BaseODE
from cubie.odesystems.SystemValues import SystemValues


class SystemInterface:
    """Convenience accessor for system values.

    Parameters
    ----------
    parameters
        System parameter values object.
    states
        System state values object.
    observables
        System observable values object.

    Notes
    -----
    Instances mirror the structure of
    :class:`~cubie.odesystems.baseODE.BaseODE` components so that
    higher-level utilities can access names, indices, and default values from a
    single object.
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
        system
            The system model to create an interface for.

        Returns
        -------
        SystemInterface
            A new instance wrapping the system's values.
        """
        return cls(
            system.parameters, system.initial_values, system.observables
        )

    def update(
        self,
        updates: Optional[Dict[str, Any]] = None,
        silent: bool = False,
        **kwargs,
    ) -> Optional[Set[str]]:
        """Update default parameter or state values.

        Parameters
        ----------
        updates
            Mapping of label to new value. If ``None``, only keyword arguments
            are used for updates.
        silent
            If ``True``, suppresses ``KeyError`` for unrecognized update keys.
        **kwargs
            Additional keyword arguments merged with ``updates``. Each
            key-value pair represents a label-value mapping for updating system
            values.

        Returns
        -------
        set of str or None
            Set of recognized update keys that were successfully applied.
            Returns None if no updates were provided.

        Raises
        ------
        KeyError
            If ``silent`` is False and unrecognized update keys are provided.

        Notes
        -----
        The method attempts to update both parameters and states. Updates are
        applied to whichever :class:`SystemValues` object recognizes each key.
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
                    "The following updates were not recognized by the system. "
                    "Was this a typo?: "
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
        keys_or_indices
            State names, indices, or a mix of both. ``None`` returns all state
            indices.
        silent
            If ``True``, suppresses warnings for unrecognized keys or indices.

        Returns
        -------
        np.ndarray
            Array of integer indices corresponding to the provided identifiers.
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
        keys_or_indices
            Observable names, indices, or a mix of both. ``None`` returns all
            observable indices.
        silent
            If ``True``, suppresses warnings for unrecognized keys or indices.
        Returns
        -------
        np.ndarray
            Array of integer indices corresponding to the provided identifiers.

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
        keys_or_indices
            Parameter names, indices, or a mix of both.
        silent
            If ``True``, suppresses warnings for unrecognized keys or indices.

        Returns
        -------
        np.ndarray
            Array of integer indices corresponding to the provided identifiers.
        """
        return self.parameters.get_indices(keys_or_indices, silent=silent)

    def get_labels(
        self, values_object: SystemValues, indices: np.ndarray
    ) -> List[str]:
        """Return labels corresponding to the provided indices.

        Parameters
        ----------
        values_object
            The SystemValues object to retrieve labels from.
        indices
            A 1D array of integer indices.

        Returns
        -------
        list of str
            List of labels corresponding to the provided indices.
        """
        return values_object.get_labels(indices)

    def state_labels(self, indices: Optional[np.ndarray] = None) -> List[str]:
        """Return state labels corresponding to the provided indices.

        Parameters
        ----------
        indices
            A 1D array of state indices.
            If ``None``, return all state labels.

        Returns
        -------
        list of str
            List of state labels corresponding to the provided indices.
        """
        if indices is None:
            return self.states.names
        return self.get_labels(self.states, indices)

    def observable_labels(
        self, indices: Optional[np.ndarray] = None
    ) -> List[str]:
        """Return observable labels corresponding to the provided indices.

        Parameters
        ----------
        indices
            A 1D array of observable indices.
            If ``None``, return all observable labels.

        Returns
        -------
        list of str
            List of observable labels corresponding to the provided indices.
        """
        if indices is None:
            return self.observables.names
        return self.get_labels(self.observables, indices)

    def parameter_labels(
        self, indices: Optional[np.ndarray] = None
    ) -> List[str]:
        """Return parameter labels corresponding to the provided indices.

        Parameters
        ----------
        indices
            A 1D array of parameter indices.
            If ``None``, return all parameter labels.

        Returns
        -------
        list of str
            List of parameter labels corresponding to the provided indices.
        """
        if indices is None:
            return self.parameters.names
        return self.get_labels(self.parameters, indices)

    @property
    def all_input_labels(self) -> List[str]:
        """List all input labels (states followed by parameters)."""
        return self.state_labels() + self.parameter_labels()

    @property
    def all_output_labels(self) -> List[str]:
        """List all output labels (states followed by observables)."""
        return self.state_labels() + self.observable_labels()
