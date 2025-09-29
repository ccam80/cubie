"""
Output configuration management system for flexible, user-controlled output
selection.

This module provides configuration classes for managing output settings in
CUDA batch solvers, including validation of indices and output types,
and automatic configuration from user-specified parameters.
"""

from typing import List, Tuple, Union, Optional, Sequence
from warnings import warn

import attrs
import numpy as np
from numpy import array_equal
from numpy.typing import NDArray

from cubie.outputhandling.summarymetrics import summary_metrics


def _indices_validator(
    array: Optional[NDArray[np.int_]], max_index: int
) -> None:
    """Validate index arrays and enforce bounds.

    Parameters
    ----------
    array
        Array of indices to validate.
    max_index
        Maximum allowed index value (exclusive).

    Returns
    -------
    None
        Returns ``None``.

    Raises
    ------
    TypeError
        Raised when *array* is not an integer numpy array.
    ValueError
        Raised when indices are out of bounds or duplicated.
    """
    if array is not None:
        if not isinstance(array, np.ndarray) or array.dtype != np.int_:
            raise TypeError("Index array must be a numpy array of integers.")

        if np.any((array < 0) | (array >= max_index)):
            raise ValueError(f"Indices must be in the range [0, {max_index})")

        unique_array, duplicate_count = np.unique(array, return_counts=True)
        duplicates = unique_array[duplicate_count > 1]
        if len(duplicates) > 0:
            raise ValueError(f"Duplicate indices found: {duplicates.tolist()}")


@attrs.define
class OutputCompileFlags:
    """Boolean compile-time controls for CUDA output features.

    Attributes
    ----------
    save_state
        Whether to save state variables. Defaults to ``False``.
    save_observables
        Whether to save observable variables. Defaults to ``False``.
    summarise
        Whether to compute any summary statistics. Defaults to ``False``.
    summarise_observables
        Whether to compute summaries for observables. Defaults to ``False``.
    summarise_state
        Whether to compute summaries for state variables. Defaults to
        ``False``.
    """

    save_state: bool = attrs.field(
        default=False, validator=attrs.validators.instance_of(bool)
    )
    save_observables: bool = attrs.field(
        default=False, validator=attrs.validators.instance_of(bool)
    )
    summarise: bool = attrs.field(
        default=False, validator=attrs.validators.instance_of(bool)
    )
    summarise_observables: bool = attrs.field(
        default=False, validator=attrs.validators.instance_of(bool)
    )
    summarise_state: bool = attrs.field(
        default=False, validator=attrs.validators.instance_of(bool)
    )


@attrs.define
class OutputConfig:
    """Validated configuration for solver outputs and summaries.

    Parameters
    ----------
    max_states
        Maximum number of state variables.
    max_observables
        Maximum number of observable variables.
    saved_state_indices
        Indices of state variables to save. Defaults to an empty collection
        that resolves to all states.
    saved_observable_indices
        Indices of observable variables to save. Defaults to an empty
        collection that resolves to all observables.
    summarised_state_indices
        Indices of state variables to summarise. Defaults to
        *saved_state_indices*.
    summarised_observable_indices
        Indices of observable variables to summarise. Defaults to
        *saved_observable_indices*.
    output_types
        Requested output type names, including summary metric identifiers.

    Notes
    -----
    Private attributes store numpy arrays so that properties can manage
    circular dependencies between index validation and flag updates. The
    post-initialisation hook applies default indices, validates bounds, and
    ensures at least one output path is active.
    """

    # System dimensions, used to validate indices
    _max_states: int = attrs.field(validator=attrs.validators.instance_of(int))
    _max_observables: int = attrs.field(
        validator=attrs.validators.instance_of(int)
    )

    _saved_state_indices: Optional[Union[List[int], NDArray[np.int_]]] = attrs.field(
        default=attrs.Factory(list),
        eq=attrs.cmp_using(eq=array_equal),
    )
    _saved_observable_indices: Optional[
        Union[List[int], NDArray[np.int_]]
    ] = attrs.field(
        default=attrs.Factory(list),
        eq=attrs.cmp_using(eq=array_equal),
    )
    _summarised_state_indices: Optional[
        Union[List[int], NDArray[np.int_]]
    ] = attrs.field(
        default=attrs.Factory(list),
        eq=attrs.cmp_using(eq=array_equal),
    )
    _summarised_observable_indices: Optional[
        Union[List[int], NDArray[np.int_]]
    ] = attrs.field(
        default=attrs.Factory(list),
        eq=attrs.cmp_using(eq=array_equal),
    )

    _output_types: List[str] = attrs.field(default=attrs.Factory(list))
    _save_state: bool = attrs.field(default=True, init=False)
    _save_observables: bool = attrs.field(default=True, init=False)
    _save_time: bool = attrs.field(default=False, init=False)
    _summary_types: Tuple[str, ...] = attrs.field(
        default=attrs.Factory(tuple), init=False
    )

    def __attrs_post_init__(self) -> None:
        """Perform post-initialisation validation and setup.

        Returns
        -------
        None
            Returns ``None``.

        Notes
        -----
        Runs after object creation to populate default arrays, validate
        indices, and confirm that at least one output path is enabled.
        """
        self.update_from_outputs_list(self._output_types)
        self._check_saved_indices()
        self._check_summarised_indices()
        self._validate_index_arrays()
        self._check_for_no_outputs()

    def _validate_index_arrays(self) -> None:
        """Validate all index arrays for bounds and duplication.

        Returns
        -------
        None
            Returns ``None``.

        Notes
        -----
        Called post-initialisation so that ``None`` values can be replaced by
        defaults before validation.
        """
        index_arrays = [
            self._saved_state_indices,
            self._saved_observable_indices,
            self._summarised_state_indices,
            self._summarised_observable_indices,
        ]
        maxima = [
            self._max_states,
            self._max_observables,
            self._max_states,
            self._max_observables,
        ]
        for i, array in enumerate(index_arrays):
            _indices_validator(array, maxima[i])

    def _check_for_no_outputs(self) -> None:
        """Ensure that at least one output type is requested.

        Returns
        -------
        None
            Returns ``None``.

        Raises
        ------
        ValueError
            Raised when no output types are enabled.
        """
        any_output = (
            self._save_state
            or self._save_observables
            or self._save_time
            or self.save_summaries
        )
        if not any_output:
            raise ValueError(
                "At least one output type must be enabled (state, "
                "observables, time, summaries)"
            )

    def _check_saved_indices(self) -> None:
        """Convert saved indices to numpy arrays and provide defaults.

        Returns
        -------
        None
            Returns ``None``.

        Notes
        -----
        When no indices are provided the arrays are populated with the full
        range for the corresponding variable type.
        """
        if len(self._saved_state_indices) == 0:
            self._saved_state_indices = np.arange(
                self._max_states, dtype=np.int_
            )
        else:
            self._saved_state_indices = np.asarray(
                self._saved_state_indices, dtype=np.int_
            )
        if len(self._saved_observable_indices) == 0:
            self._saved_observable_indices = np.arange(
                self._max_observables, dtype=np.int_
            )
        else:
            self._saved_observable_indices = np.asarray(
                self._saved_observable_indices, dtype=np.int_
            )

    def _check_summarised_indices(self) -> None:
        """Default summarised indices to the saved selections when needed.

        Returns
        -------
        None
            Returns ``None``.

        Notes
        -----
        Empty summary selections inherit the saved selections for the matching
        variable type.
        """
        if len(self._summarised_state_indices) == 0:
            self._summarised_state_indices = self._saved_state_indices
        else:
            self._summarised_state_indices = np.asarray(
                self._summarised_state_indices, dtype=np.int_
            )
        if len(self._summarised_observable_indices) == 0:
            self._summarised_observable_indices = (
                self._saved_observable_indices
            )
        else:
            self._summarised_observable_indices = np.asarray(
                self._summarised_observable_indices, dtype=np.int_
            )

    @property
    def max_states(self) -> int:
        """Maximum number of states."""
        return self._max_states

    @max_states.setter
    def max_states(self, value: int) -> None:
        """Set the maximum number of states and refresh dependent indices.

        Parameters
        ----------
        value
            New maximum number of states.

        Returns
        -------
        None
            Returns ``None``.

        Notes
        -----
        When saved indices currently span the full range they are expanded to
        the new size before validation reruns.
        """
        if np.array_equal(
            self._saved_state_indices,
            np.arange(self.max_states, dtype=np.int_),
        ):
            self._saved_state_indices = np.arange(value, dtype=np.int_)
        self._max_states = value
        self.__attrs_post_init__()

    @property
    def max_observables(self) -> int:
        """Maximum number of observables."""
        return self._max_observables

    @max_observables.setter
    def max_observables(self, value: int) -> None:
        """Set the maximum number of observables and refresh saved indices.

        Parameters
        ----------
        value
            New maximum number of observables.

        Returns
        -------
        None
            Returns ``None``.

        Notes
        -----
        When saved indices span the full observable range they expand to the
        new size before validation reruns.
        """
        if np.array_equal(
            self._saved_observable_indices,
            np.arange(self.max_observables, dtype=np.int_),
        ):
            self._saved_observable_indices = np.arange(value, dtype=np.int_)
        self._max_observables = value
        self.__attrs_post_init__()

    @property
    def save_state(self) -> bool:
        """Whether state saving is enabled with valid indices."""
        return self._save_state and (len(self._saved_state_indices) > 0)

    @property
    def save_observables(self) -> bool:
        """Whether observable saving is enabled with valid indices."""
        return self._save_observables and (
            len(self._saved_observable_indices) > 0
        )

    @property
    def save_time(self) -> bool:
        """Whether solver time samples should be saved."""
        return self._save_time

    @property
    def save_summaries(self) -> bool:
        """Whether any summary metric is configured."""
        return len(self._summary_types) > 0

    @property
    def summarise_state(self) -> bool:
        """Whether state variables contribute to summaries."""
        return self.save_summaries and self.n_summarised_states > 0

    @property
    def summarise_observables(self) -> bool:
        """Whether observable variables contribute to summaries."""
        return self.save_summaries and self.n_summarised_observables > 0

    @property
    def compile_flags(self) -> OutputCompileFlags:
        """
        Get compile flags for this configuration.

        Returns
        -------
        OutputCompileFlags
            Flags indicating which output features should be compiled.
        """
        return OutputCompileFlags(
            save_state=self.save_state,
            save_observables=self.save_observables,
            summarise=self.save_summaries,
            summarise_observables=self.summarise_observables,
            summarise_state=self.summarise_state,
        )

    @property
    def saved_state_indices(self) -> NDArray[np.int_]:
        """State indices to save, or an empty array when disabled."""
        if not self._save_state:
            return np.asarray([], dtype=np.int_)
        return self._saved_state_indices

    @saved_state_indices.setter
    def saved_state_indices(
        self, value: Sequence[int] | NDArray[np.int_]
    ) -> None:
        """Set the state indices that will be saved.

        Parameters
        ----------
        value
            State indices to save.

        Returns
        -------
        None
            Returns ``None``.
        """
        self._saved_state_indices = np.asarray(value, dtype=np.int_)
        self._validate_index_arrays()
        self._check_for_no_outputs()

    @property
    def saved_observable_indices(self) -> NDArray[np.int_]:
        """Observable indices to save, or an empty array when disabled."""
        if not self._save_observables:
            return np.asarray([], dtype=np.int_)
        return self._saved_observable_indices

    @saved_observable_indices.setter
    def saved_observable_indices(
        self, value: Sequence[int] | NDArray[np.int_]
    ) -> None:
        """Set the observable indices that will be saved.

        Parameters
        ----------
        value
            Observable indices to save.

        Returns
        -------
        None
            Returns ``None``.
        """
        self._saved_observable_indices = np.asarray(value, dtype=np.int_)
        self._validate_index_arrays()
        self._check_for_no_outputs()

    @property
    def summarised_state_indices(self) -> NDArray[np.int_]:
        """State indices included in summary calculations."""
        return self._summarised_state_indices

    @summarised_state_indices.setter
    def summarised_state_indices(
        self, value: Sequence[int] | NDArray[np.int_]
    ) -> None:
        """Set the state indices used for summary calculations.

        Parameters
        ----------
        value
            State indices for summary calculations.

        Returns
        -------
        None
            Returns ``None``.
        """
        self._summarised_state_indices = np.asarray(value, dtype=np.int_)
        self._validate_index_arrays()
        self._check_for_no_outputs()

    @property
    def summarised_observable_indices(self) -> NDArray[np.int_]:
        """Observable indices included in summary calculations."""
        return self._summarised_observable_indices

    @summarised_observable_indices.setter
    def summarised_observable_indices(
        self, value: Sequence[int] | NDArray[np.int_]
    ) -> None:
        """Set the observable indices used for summary calculations.

        Parameters
        ----------
        value
            Observable indices for summary calculations.

        Returns
        -------
        None
            Returns ``None``.
        """
        self._summarised_observable_indices = np.asarray(value, dtype=np.int_)
        self._validate_index_arrays()
        self._check_for_no_outputs()

    @property
    def n_saved_states(self) -> int:
        """
        Get number of states that will be saved.

        Returns
        -------
        int
            Number of state variables to save in time-domain output.

        Notes
        -----
        Returns the length of saved_state_indices when save_state is True,
        otherwise 0.
        """
        return len(self._saved_state_indices) if self._save_state else 0

    @property
    def n_saved_observables(self) -> int:
        """
        Get number of observables that will be saved.

        Returns
        -------
        int
            Number of observable variables to save in time-domain output.
        """
        return (
            len(self._saved_observable_indices)
            if self._save_observables
            else 0
        )

    @property
    def n_summarised_states(self) -> int:
        """
        Get number of states that will be summarised.

        Returns
        -------
        int
            Number of state variables for summary calculations.

        Notes
        -----
        Returns the length of summarised_state_indices when save_summaries
        is active, otherwise 0.
        """
        return (
            len(self._summarised_state_indices) if self.save_summaries else 0
        )

    @property
    def n_summarised_observables(self) -> int:
        """
        Get number of observables that will be summarised.

        Returns
        -------
        int
            Number of observable variables for summary calculations.
        """
        return (
            len(self._summarised_observable_indices)
            if self.save_summaries
            else 0
        )

    @property
    def summary_types(self) -> Tuple[str, ...]:
        """Configured summary metric identifiers."""
        return self._summary_types

    @property
    def summary_legend_per_variable(self) -> dict[int, str]:
        """Map per-variable summary indices to metric names.

        Returns
        -------
        dict[int, str]
            Dictionary that assigns each per-variable summary slot to its
            metric identifier.
        """
        if not self._summary_types:
            return {}
        legend_tuple = summary_metrics.legend(self._summary_types)
        legend_dict = dict(zip(range(len(legend_tuple)), legend_tuple))
        return legend_dict

    @property
    def summary_parameters(self) -> dict[str, object]:
        """Collect parameters required by the registered summary metrics.

        Returns
        -------
        dict[str, object]
            Dictionary of metric-specific keyword arguments needed during
            compilation.
        """
        return summary_metrics.params(list(self._summary_types))

    @property
    def summaries_buffer_height_per_var(self) -> int:
        """
        Calculate buffer size per variable for summary calculations.

        Returns
        -------
        int
            Buffer height required per variable for summary metrics.
        """
        if not self.summary_types:
            return 0
        # Convert summary_types set to list for summarymetrics
        summary_list = list(self._summary_types)
        total_buffer_size = summary_metrics.summaries_buffer_height(
            summary_list
        )
        return total_buffer_size

    @property
    def summaries_output_height_per_var(self) -> int:
        """
        Calculate output array height per variable for summaries.

        Returns
        -------
        int
            Output height required per variable for summary results.
        """
        if not self._summary_types:
            return 0
        # Convert summary_types tuple to list for summarymetrics
        summary_list = list(self._summary_types)
        total_output_size = summary_metrics.summaries_output_height(
            summary_list
        )
        return total_output_size

    @property
    def state_summaries_buffer_height(self) -> int:
        """
        Calculate total buffer height for state summaries.

        Returns
        -------
        int
            Total buffer height for all state summary calculations.
        """
        return self.summaries_buffer_height_per_var * self.n_summarised_states

    @property
    def observable_summaries_buffer_height(self) -> int:
        """
        Calculate total buffer height for observable summaries.

        Returns
        -------
        int
            Total buffer height for all observable summary calculations.
        """
        return (
            self.summaries_buffer_height_per_var
            * self.n_summarised_observables
        )

    @property
    def total_summary_buffer_size(self) -> int:
        """
        Calculate total size of all summary buffers.

        Returns
        -------
        int
            Combined size of state and observable summary buffers.
        """
        return (
            self.state_summaries_buffer_height
            + self.observable_summaries_buffer_height
        )

    @property
    def state_summaries_output_height(self) -> int:
        """
        Calculate total output height for state summaries.

        Returns
        -------
        int
            Total output height for all state summary results.
        """
        return self.summaries_output_height_per_var * self.n_summarised_states

    @property
    def observable_summaries_output_height(self) -> int:
        """
        Calculate total output height for observable summaries.

        Returns
        -------
        int
            Total output height for all observable summary results.
        """
        return (
            self.summaries_output_height_per_var
            * self.n_summarised_observables
        )

    @property
    def output_types(self) -> List[str]:
        """Configured output type names in declaration order."""
        return self._output_types

    @output_types.setter
    def output_types(self, value: Sequence[str]) -> None:
        """Set output types and update configuration accordingly.

        Parameters
        ----------
        value
            Output types to configure. Accepts a list, tuple, or single string.

        Returns
        -------
        None
            Returns ``None``.

        Raises
        ------
        TypeError
            Raised when *value* is not a supported sequence type.
        """
        if isinstance(value, tuple):
            value = list(value)
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            raise TypeError(
                f"Output types must be a list or tuple of strings, "
                f"or a single string. Got {type(value)}"
            )

        self.update_from_outputs_list(value)
        self._check_for_no_outputs()

    def update_from_outputs_list(
        self,
        output_types: list[str],
    ) -> None:
        """Update configuration from a list of output type names.

        Parameters
        ----------
        output_types
            Output type names to configure.

        Returns
        -------
        None
            Returns ``None``.

        Notes
        -----
        Parses the list to set boolean flags and collect summary metric
        selections. Unknown entries trigger warnings but do not raise
        exceptions.
        """
        if not output_types:
            self._output_types = []
            self._summary_types = tuple()
            self._save_state = False
            self._save_observables = False
            self._save_time = False

        else:
            self._output_types = output_types
            self._save_state = "state" in output_types
            self._save_observables = "observables" in output_types
            self._save_time = "time" in output_types

            summary_types = []
            for output_type in output_types:
                if any(
                    (
                        output_type.startswith(name)
                        for name in summary_metrics.implemented_metrics
                    )
                ):
                    summary_types.append(output_type)
                elif output_type in ["state", "observables", "time"]:
                    continue
                else:
                    warn(
                        f"Summary type '{output_type}' is not implemented. "
                        f"Ignoring."
                    )

            self._summary_types = tuple(summary_types)

            self._check_for_no_outputs()

    @classmethod
    def from_loop_settings(
        cls,
        output_types: List[str],
        saved_state_indices: Sequence[int] | NDArray[np.int_] | None = None,
        saved_observable_indices: Sequence[int] | NDArray[np.int_] | None = None,
        summarised_state_indices: Sequence[int] | NDArray[np.int_] | None = None,
        summarised_observable_indices: Sequence[int] | NDArray[np.int_] | None = None,
        max_states: int = 0,
        max_observables: int = 0,
    ) -> "OutputConfig":
        """
        Create configuration from integrator-compatible specifications.

        Parameters
        ----------
        output_types
            Output types including ``"state"``, ``"observables"``, ``"time"``,
            and summary metric names such as ``"max"`` or ``"rms"``.
        saved_state_indices
            Indices of states to save in time-domain output.
        saved_observable_indices
            Indices of observables to save in time-domain output.
        summarised_state_indices
            Indices of states for summary calculations. Defaults to
            *saved_state_indices*.
        summarised_observable_indices
            Indices of observables for summary calculations. Defaults to
            *saved_observable_indices*.
        max_states
            Total number of state variables in the system.
        max_observables
            Total number of observable variables in the system.

        Returns
        -------
        OutputConfig
            Configured output configuration object.

        Notes
        -----
        This class method provides a convenient interface for creating
        OutputConfig objects from the parameter format used by integrator
        classes. It handles None values appropriately by converting them
        to empty arrays.
        """
        # Set boolean compile flags for output types
        output_types = output_types.copy()

        # OutputConfig doesn't play as nicely with Nones as the rest of python does
        if saved_state_indices is None:
            saved_state_indices = np.asarray([], dtype=np.int_)
        if saved_observable_indices is None:
            saved_observable_indices = np.asarray([], dtype=np.int_)
        if summarised_state_indices is None:
            summarised_state_indices = np.asarray([], dtype=np.int_)
        if summarised_observable_indices is None:
            summarised_observable_indices = np.asarray([], dtype=np.int_)

        return cls(
            max_states=max_states,
            max_observables=max_observables,
            saved_state_indices=saved_state_indices,
            saved_observable_indices=saved_observable_indices,
            summarised_state_indices=summarised_state_indices,
            summarised_observable_indices=summarised_observable_indices,
            output_types=output_types,
        )
