"""
Output configuration management system for flexible, user-controlled output
selection.
"""

from typing import List, Tuple, Union, Optional, Sequence
from warnings import warn

import attrs
import numpy as np
from numpy import array_equal
from numpy.typing import NDArray

from cubie.outputhandling import summary_metrics


# ************************************ standalone validators ************************************** #

def _indices_validator(array, max_index):
    """Validator to ensure indices are valid numpy arrays."""
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
    save_state: bool = attrs.field(default=False,
                                   validator=attrs.validators.instance_of(
                                           bool))
    save_observables: bool = attrs.field(default=False,
                                         validator=attrs.validators.instance_of(
                                                 bool))
    summarise: bool = attrs.field(default=False,
                                  validator=attrs.validators.instance_of(bool))
    summarise_observables: bool = attrs.field(default=False,
                                              validator=attrs.validators.instance_of(
                                                      bool))
    summarise_state: bool = attrs.field(default=False,
                                        validator=attrs.validators.instance_of(
                                                bool))


@attrs.define
class OutputConfig:
    """
    Data class to hold output configuration. Contains flags for compile-time toggling of different output types,
    and validation logic to ensure *some* output is requested, and that we're not requesting indices off the end of
    the state arrays.

    Went a bit eager on the private attributes and properties to get around a circular logic bug when setting indices or
     requested output flags. Many of them directly set/get a private attribute without additional logic, creating a
     public attribute instead, but it's non-breaking and allows for easier addition of checking should bugs arise.
    """

    # System dimensions, used to validate indices
    _max_states: int = attrs.field(validator=attrs.validators.instance_of(int))
    _max_observables: int = attrs.field(
            validator=attrs.validators.instance_of(int))

    _saved_state_indices: Optional[Union[List | NDArray]] = attrs.field(
            default=attrs.Factory(list), eq=attrs.cmp_using(eq=array_equal), )
    _saved_observable_indices: Optional[Union[List | NDArray]] = attrs.field(
            default=attrs.Factory(list), eq=attrs.cmp_using(eq=array_equal), )
    _summarised_state_indices: Optional[Union[List | NDArray]] = attrs.field(
            default=attrs.Factory(list), eq=attrs.cmp_using(eq=array_equal), )
    _summarised_observable_indices: Optional[
        Union[List | NDArray]] = attrs.field(default=attrs.Factory(list),
                                             eq=attrs.cmp_using(
                                                     eq=array_equal), )

    _output_types: List[str] = attrs.field(default=attrs.Factory(list))
    _save_state: bool = attrs.field(default=True, init=False)
    _save_observables: bool = attrs.field(default=True, init=False)
    _save_time: bool = attrs.field(default=False, init=False)
    _summary_types: Tuple[str] = attrs.field(default=attrs.Factory(tuple),
                                             init=False)

    # *********************************** post-init validators ***********************************
    def __attrs_post_init__(self):
        """Swap out None index arrays, check that all indices are within bounds, and check for a no-output request."""
        self.update_from_outputs_list(self._output_types)
        self._check_saved_indices()
        self._check_summarised_indices()
        self._validate_index_arrays()
        self._check_for_no_outputs()

    def _validate_index_arrays(self):
        """Ensure that saved indices arrays are valid and in bounds. This is called post-init to allow None arrays to be
        replaced with full arrays in the _indices_to_arrays step before checking.
        """
        index_arrays = [self._saved_state_indices,
                        self._saved_observable_indices,
                        self._summarised_state_indices,
                        self._summarised_observable_indices]
        maxima = [self._max_states, self._max_observables, self._max_states,
                  self._max_observables]
        for i, array in enumerate(index_arrays):
            _indices_validator(array, maxima[i])

    def _check_for_no_outputs(self):
        """Check if any output is requested."""
        any_output = (
                self._save_state or self._save_observables or self._save_time or self.save_summaries)
        if not any_output:
            raise ValueError(
                    "At least one output type must be enabled (state, observables, time, summaries_array)")

    def _check_saved_indices(self):
        """Convert indices iterables to numpy arrays for interface with device functions. If the array type is None,
        create an array of all possible indices."""
        if len(self._saved_state_indices) == 0:
            self._saved_state_indices = np.arange(self._max_states,
                                                  dtype=np.int_)
        else:
            self._saved_state_indices = np.asarray(self._saved_state_indices,
                                                   dtype=np.int_)
        if len(self._saved_observable_indices) == 0:
            self._saved_observable_indices = np.arange(self._max_observables,
                                                       dtype=np.int_)
        else:
            self._saved_observable_indices = np.asarray(
                    self._saved_observable_indices, dtype=np.int_)

    def _check_summarised_indices(self):
        """Set summarised indices to saved indices if not provided."""
        if len(self._summarised_state_indices) == 0:
            self._summarised_state_indices = self._saved_state_indices
        else:
            self._summarised_state_indices = np.asarray(
                    self._summarised_state_indices, dtype=np.int_)
        if len(self._summarised_observable_indices) == 0:
            self._summarised_observable_indices = self._saved_observable_indices
        else:
            self._summarised_observable_indices = np.asarray(
                    self._summarised_observable_indices, dtype=np.int_)

    # ************************************ Getters/Setters *************************************** #
    @property
    def max_states(self):
        return self._max_states

    @max_states.setter
    def max_states(self, value):
        """Set the maximum number of states. If the saved state indices are set to the default range, update them to
        the full range of the new maximum."""
        if np.array_equal(self._saved_state_indices,
                          np.arange(self.max_states, dtype=np.int_)):
            self._saved_state_indices = np.arange(value, dtype=np.int_)
        self._max_states = value
        self.__attrs_post_init__()

    @property
    def max_observables(self):
        return self._max_observables

    @max_observables.setter
    def max_observables(self, value):
        """Set the maximum number of observables. If the saved observable indices are set to the default range, update
        them to the full range of the new maximum."""
        if np.array_equal(self._saved_observable_indices,
                          np.arange(self.max_observables, dtype=np.int_)):
            self._saved_observable_indices = np.arange(value, dtype=np.int_)
        self._max_observables = value
        self.__attrs_post_init__()

    @property
    def save_state(self) -> bool:
        return self._save_state and (len(self._saved_state_indices) > 0)

    @property
    def save_observables(self):
        return self._save_observables and (
                len(self._saved_observable_indices) > 0)

    @property
    def save_time(self):
        return self._save_time

    @property
    def save_summaries(self) -> bool:
        """Do we need to summarise anything at all?"""
        return len(self._summary_types) > 0

    @property
    def summarise_state(self) -> bool:
        """Will any states be summarised?"""
        return self.save_summaries and self.n_summarised_states > 0

    @property
    def summarise_observables(self) -> bool:
        """Will any observables be summarised?"""
        return self.save_summaries and self.n_summarised_observables > 0

    @property
    def compile_flags(self) -> OutputCompileFlags:
        """Return the compile flags for this output configuration."""
        return OutputCompileFlags(save_state=self.save_state,
                save_observables=self.save_observables,
                summarise=self.save_summaries,
                summarise_observables=self.summarise_observables,
                summarise_state=self.summarise_state, )

    @property
    def saved_state_indices(self):
        if not self._save_state:
            return np.asarray([], dtype=np.int_)
        return self._saved_state_indices

    @saved_state_indices.setter
    def saved_state_indices(self, value):
        self._saved_state_indices = np.asarray(value, dtype=np.int_)
        self._validate_index_arrays()
        self._check_for_no_outputs()

    @property
    def saved_observable_indices(self):
        if not self._save_observables:
            return np.asarray([], dtype=np.int_)
        return self._saved_observable_indices

    @saved_observable_indices.setter
    def saved_observable_indices(self, value):
        self._saved_observable_indices = np.asarray(value, dtype=np.int_)
        self._validate_index_arrays()
        self._check_for_no_outputs()

    @property
    def summarised_state_indices(self):
        return self._summarised_state_indices

    @summarised_state_indices.setter
    def summarised_state_indices(self, value):
        self._summarised_state_indices = np.asarray(value, dtype=np.int_)
        self._validate_index_arrays()
        self._check_for_no_outputs()

    @property
    def summarised_observable_indices(self):
        return self._summarised_observable_indices

    @summarised_observable_indices.setter
    def summarised_observable_indices(self, value):
        self._summarised_observable_indices = np.asarray(value, dtype=np.int_)
        self._validate_index_arrays()
        self._check_for_no_outputs()

    @property
    def n_saved_states(self) -> int:
        """Number of states that will be saved (time-domain), which will the length of saved_state_indices as long as
        "save_state" is True."""
        return len(self._saved_state_indices) if self._save_state else 0

    @property
    def n_saved_observables(self) -> int:
        """Number of observables that will actually be saved."""
        return len(
                self._saved_observable_indices) if self._save_observables else 0

    @property
    def n_summarised_states(self) -> int:
        """Number of states that will be summarised, which is the length of summarised_state_indices as long as
        "save_summaries" is active."""
        return len(
                self._summarised_state_indices) if self.save_summaries else 0

    @property
    def n_summarised_observables(self) -> int:
        """Number of observables that will actually be summarised."""
        return len(
                self._summarised_observable_indices) if self.save_summaries else 0

    @property
    def summary_types(self):
        return self._summary_types

    @property
    def summary_legend_per_variable(self):
        """Returna a dict of index number to summary type for each variable saved."""
        if not self._summary_types:
            return {}
        legend_tuple = summary_metrics.legend(self._summary_types)
        legend_dict = dict(zip(range(len(self._summary_types)), legend_tuple))
        return legend_dict

    @property
    def summary_parameters(self):
        """Get parameters for summary metrics from the metrics system."""
        return summary_metrics.params(list(self._summary_types))

    @property
    def summaries_buffer_height_per_var(self) -> int:
        """Calculate buffer size per variable using summarymetrics system."""
        if not self.summary_types:
            return 0
        # Convert summary_types set to list for summarymetrics
        summary_list = list(self._summary_types)
        total_buffer_size = summary_metrics.summaries_buffer_height(
                summary_list)
        return total_buffer_size

    @property
    def summaries_output_height_per_var(self) -> int:
        """Calculate output memory per variable using summarymetrics system."""
        if not self._summary_types:
            return 0
        # Convert summary_types tuple to list for summarymetrics
        summary_list = list(self._summary_types)
        total_output_size = summary_metrics.summaries_output_height(
                summary_list)
        return total_output_size

    @property
    def state_summaries_buffer_height(self) -> int:
        """Calculate the height of the state summary buffer."""
        return self.summaries_buffer_height_per_var * self.n_summarised_states

    @property
    def observable_summaries_buffer_height(self) -> int:
        """Calculate the height of the observable summary buffer."""
        return self.summaries_buffer_height_per_var * self.n_summarised_observables

    @property
    def total_summary_buffer_size(self) -> int:
        """Calculate the total size of the summary buffer."""
        return (
                self.state_summaries_buffer_height + self.observable_summaries_buffer_height)

    @property
    def state_summaries_output_height(self) -> int:
        """Calculate the height of the state summary output."""
        return self.summaries_output_height_per_var * self.n_summarised_states

    @property
    def observable_summaries_output_height(self) -> int:
        """Calculate the height of the observable summary output."""
        return self.summaries_output_height_per_var * self.n_summarised_observables

    @property
    def output_types(self) -> List[str]:
        """Get the list of output types requested."""
        return self._output_types

    @output_types.setter
    def output_types(self, value: Sequence[str]):
        """Set the output types and update the compile flags accordingly."""
        if isinstance(value, tuple):
            value = list(value)
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            raise TypeError(
                    f"Output types must be a list or tuple of strings, or a single string. Got {type(value)}")

        self.update_from_outputs_list(value)
        self._check_for_no_outputs()

    # ***************************** Custom init methods for adapting to other components ***************************** #
    def update_from_outputs_list(self, output_types: list[str], ):
        """Update bools and summary types from a list of output types. Run the post_init validators to empty indices
        if not requested."""
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
                if any((output_type.startswith(name) for name in
                        summary_metrics.implemented_metrics)):
                    summary_types.append(output_type)
                elif output_type in ["state", "observables", "time"]:
                    continue
                else:
                    warn(f"Summary type '{output_type}' is not implemented. Ignoring.")

            self._summary_types = tuple(summary_types)

            self._check_for_no_outputs()

    @classmethod
    def from_loop_settings(cls, output_types: List[str],
                           saved_state_indices=None,
                           saved_observable_indices=None,
                           summarised_state_indices=None,
                           summarised_observable_indices=None,
                           max_states: int = 0,
                           max_observables: int = 0, ) -> "OutputConfig":
        """
        Create configuration from specifications in the format provided by the integrator classes.

        Args:
            output_types: List of strings specifying output types from ["state", "observables", "time", "max",
            "peaks", "mean", "rms", "min"]
            saved_state_indices: Indices of states to save
            saved_observable_indices: Indices of observables to save
            summarised_state_indices: Indices of states to summarise, if different from saved_state_indices, otherwise None
            summarised_observable_indices: Indices of observables to summarise, if different from saved_observable_indices, otherwise None
            n_peaks: Number of peaks to detect
            max_states: Total number of states in system
            max_observables: Total number of observables in system
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

        return cls(max_states=max_states, max_observables=max_observables,
                saved_state_indices=saved_state_indices,
                saved_observable_indices=saved_observable_indices,
                summarised_state_indices=summarised_state_indices,
                summarised_observable_indices=summarised_observable_indices,
                output_types=output_types, )
