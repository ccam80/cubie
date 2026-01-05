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

from typing import Any, Dict, List, Optional, Set, Tuple, Union

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

    This class serves as the single source of truth for variable label
    handling. The variable resolution methods
    (:meth:`resolve_variable_labels`, :meth:`merge_variable_inputs`,
    :meth:`convert_variable_labels`) consolidate all label-to-index
    conversion logic. These methods interpret input values as follows:

    - ``None`` means "use all" (default behavior)
    - ``[]`` or empty array means "explicitly no variables"
    - When both labels and indices are provided, their union is used
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

    def resolve_variable_labels(
        self,
        labels: Optional[List[str]],
        silent: bool = False,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Resolve variable labels to separate state and observable indices.

        Parameters
        ----------
        labels
            Variable names that may be states or observables. If ``None``,
            returns ``(None, None)`` to signal "use defaults". If empty
            list, returns empty arrays for both.
        silent
            If ``True``, suppresses errors for unrecognized labels.

        Returns
        -------
        Tuple[Optional[np.ndarray], Optional[np.ndarray]]
            Tuple of (state_indices, observable_indices). Returns
            ``(None, None)`` when labels is None.

        Raises
        ------
        ValueError
            If any label is not found in states or observables and
            silent is False.
        """
        if labels is None:
            return (None, None)

        if len(labels) == 0:
            return (
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
            )

        # Resolve each label to state or observable index
        state_idxs = self.state_indices(labels, silent=True)
        obs_idxs = self.observable_indices(labels, silent=True)

        # Track which labels were resolved
        state_names_set = set(self.states.names)
        obs_names_set = set(self.observables.names)
        resolved_labels = state_names_set.union(obs_names_set)
        unresolved = [lbl for lbl in labels if lbl not in resolved_labels]

        # Validate all labels found (unless silent)
        if not silent and unresolved:
            raise ValueError(
                f"Variables not found in states or observables: "
                f"{unresolved}. "
                f"Available states: {self.states.names}. "
                f"Available observables: {self.observables.names}."
            )

        return (
            state_idxs.astype(np.int32),
            obs_idxs.astype(np.int32),
        )

    def merge_variable_inputs(
        self,
        var_labels: Optional[List[str]],
        state_indices: Optional[Union[List[int], np.ndarray]],
        observable_indices: Optional[Union[List[int], np.ndarray]],
        max_states: int,
        max_observables: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Merge label-based selections with index-based selections.

        Parameters
        ----------
        var_labels
            Variable names to resolve. None means "not provided".
        state_indices
            Direct state index selection. None means "not provided".
        observable_indices
            Direct observable index selection. None means "not provided".
        max_states
            Total number of states in the system.
        max_observables
            Total number of observables in the system.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Final (state_indices, observable_indices) arrays.

        Notes
        -----
        When all three inputs are None, returns full range arrays
        (all states, all observables). When any input is explicitly
        empty ([] or empty array), that emptiness is preserved.
        Union of resolved labels and provided indices is returned.
        """
        # Resolve var_labels using resolve_variable_labels()
        resolved_state, resolved_obs = self.resolve_variable_labels(var_labels)

        # Check if all three inputs are None (use defaults)
        all_none = (
            var_labels is None
            and state_indices is None
            and observable_indices is None
        )
        if all_none:
            return (
                np.arange(max_states, dtype=np.int32),
                np.arange(max_observables, dtype=np.int32),
            )

        # Convert None to empty arrays for union computation
        if resolved_state is None:
            resolved_state = np.array([], dtype=np.int32)
        if resolved_obs is None:
            resolved_obs = np.array([], dtype=np.int32)

        # Convert provided indices (None means empty for this computation)
        if state_indices is None:
            provided_state = np.array([], dtype=np.int32)
        else:
            provided_state = np.asarray(state_indices, dtype=np.int32)

        if observable_indices is None:
            provided_obs = np.array([], dtype=np.int32)
        else:
            provided_obs = np.asarray(observable_indices, dtype=np.int32)

        # Compute union of resolved label indices with provided indices
        final_state = np.union1d(resolved_state, provided_state).astype(
            np.int32
        )
        final_obs = np.union1d(resolved_obs, provided_obs).astype(np.int32)

        return (final_state, final_obs)

    def convert_variable_labels(
        self,
        output_settings: Dict[str, Any],
        max_states: int,
        max_observables: int,
    ) -> None:
        """Convert variable label settings to index arrays in-place.

        Parameters
        ----------
        output_settings
            Settings dict containing ``save_variables``,
            ``summarise_variables``, and their index counterparts.
            Modified in-place.
        max_states
            Total number of states in the system.
        max_observables
            Total number of observables in the system.

        Returns
        -------
        None
            Modifies output_settings in-place.

        Raises
        ------
        ValueError
            If any variable labels are not found in states or observables.

        Notes
        -----
        Pops ``save_variables`` and ``summarise_variables`` from the dict
        and replaces index parameters with final resolved arrays. For
        summarised indices, defaults to saved indices when both labels
        and indices are None.
        """
        # Extract save_variables and related indices
        save_vars = output_settings.pop("save_variables", None)
        saved_state_idxs = output_settings.get("saved_state_indices", None)
        saved_obs_idxs = output_settings.get("saved_observable_indices", None)

        # Call merge_variable_inputs for save variables
        final_saved_state, final_saved_obs = self.merge_variable_inputs(
            save_vars,
            saved_state_idxs,
            saved_obs_idxs,
            max_states,
            max_observables,
        )

        # Extract summarise_variables and related indices
        summarise_vars = output_settings.pop("summarise_variables", None)
        summ_state_idxs = output_settings.get(
            "summarised_state_indices", None
        )
        summ_obs_idxs = output_settings.get(
            "summarised_observable_indices", None
        )

        # Handle "summarised defaults to saved" when all summarise inputs None
        all_summarise_none = (
            summarise_vars is None
            and summ_state_idxs is None
            and summ_obs_idxs is None
        )
        if all_summarise_none:
            final_summ_state = final_saved_state.copy()
            final_summ_obs = final_saved_obs.copy()
        else:
            final_summ_state, final_summ_obs = self.merge_variable_inputs(
                summarise_vars,
                summ_state_idxs,
                summ_obs_idxs,
                max_states,
                max_observables,
            )

        # Update dict with final index arrays
        output_settings["saved_state_indices"] = final_saved_state
        output_settings["saved_observable_indices"] = final_saved_obs
        output_settings["summarised_state_indices"] = final_summ_state
        output_settings["summarised_observable_indices"] = final_summ_obs

    @property
    def all_input_labels(self) -> List[str]:
        """List all input labels (states followed by parameters)."""
        return self.state_labels() + self.parameter_labels()

    @property
    def all_output_labels(self) -> List[str]:
        """List all output labels (states followed by observables)."""
        return self.state_labels() + self.observable_labels()
