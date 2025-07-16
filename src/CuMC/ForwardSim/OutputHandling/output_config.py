"""
Output configuration management system for flexible, user-controlled output selection.
"""

import attrs
from typing import List, Optional, Set, Tuple
import numpy as np
from CuMC.ForwardSim.OutputHandling._utils import process_outputs_list
from numpy.typing import NDArray, ArrayLike

import re

_ImplementedSummaries = {'max', 'min', 'mean', 'rms', 'peaks'}


@attrs.define
class ArrayHeights:
    """
    Shape information for arrays - array.heights_temp is the number of samples required for the working array,
    and ArrayHeights.output is the number of samples required for the output array. If you provide a boolean
    argument CUDA_allocation_safe = True, then heights return a minimum value of 1, to avoid allocating a size-0
    array."""
    _CUDA_allocation_safe: Optional[bool] = attrs.field(default=False, validator=attrs.validators.instance_of(bool))
    temp: int = attrs.field(default=0, validator=attrs.validators.instance_of(int))
    output: int = attrs.field(default=0, validator=attrs.validators.instance_of(int))

    #TODO: see if we end up using this from here - it might be easier to use the boolean toggles directly when
    # allocating in a higher-level module.
    def __attrs_post_init__(self):
        """If these are being generated for the purpose of filling a CUDA allocation slot, set any zero values to 1"""
        if self._CUDA_allocation_safe:
            self.temp = max(1, self.temp)
            self.output = max(1, self.output)


@attrs.define
class OutputArraySizes:
    """
    Class to hold the sizes of output arrays. This is used to ensure that the arrays are allocated with the correct
    sizes for the requested outputs.
    """
    state: ArrayHeights = attrs.field(default=ArrayHeights(temp=0, output=0),
                                      validator=attrs.validators.instance_of(ArrayHeights),
                                      )
    observables: ArrayHeights = attrs.field(default=ArrayHeights(temp=0, output=0),
                                            validator=attrs.validators.instance_of(ArrayHeights),
                                            )
    state_summaries: ArrayHeights = attrs.field(default=ArrayHeights(temp=0, output=0),
                                                validator=attrs.validators.instance_of(ArrayHeights),
                                                )
    observable_summaries: ArrayHeights = attrs.field(default=ArrayHeights(temp=0, output=0),
                                                     validator=attrs.validators.instance_of(ArrayHeights),
                                                     )
    _CUDA_allocation_safe: bool = attrs.field(default=False, validator=attrs.validators.instance_of(bool))

    @classmethod
    def from_sizes(cls,
                   summary_temp_per_var: int,
                   summary_output_per_var: int,
                   n_saved_states: int,
                   n_saved_observables: int,
                   n_summarised_states: int,
                   n_summarised_observables: int,
                   max_states: int,
                   max_observables: int,
                   save_time,
                   CUDA_allocation_safe=False,
                   ):
        return cls(state=ArrayHeights(temp=max_states,
                                      output=(n_saved_states + 1 * save_time),
                                      CUDA_allocation_safe=CUDA_allocation_safe,
                                      ),
                   observables=ArrayHeights(temp=max_observables,
                                            output=n_saved_observables,
                                            CUDA_allocation_safe=CUDA_allocation_safe,
                                            ),
                   state_summaries=ArrayHeights(temp=summary_temp_per_var * n_summarised_states,
                                                output=summary_output_per_var * n_summarised_states,
                                                CUDA_allocation_safe=CUDA_allocation_safe,
                                                ),
                   observable_summaries=ArrayHeights(temp=summary_temp_per_var * n_summarised_observables,
                                                     output=summary_output_per_var * n_summarised_observables,
                                                     CUDA_allocation_safe=CUDA_allocation_safe,
                                                     ),
                   )



def indices_validator(array, max_index):
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
class OutputConfig:
    """
    Attrs class to hold output configuration. Contains flags for compile-time toggling of different output types,
    and validation logic to ensure *some* output is requested, and that we're not requesting indices off the end of
    the state arrays. Returns information about the size of output arrays and compile-time flags for which output
    functions to build.

    To extend this class when adding new summary metrics, you will need to add:
    - A boolean property for the new output type (e.g.
        '''
        @property
        def summarise_new_output(self) -> bool):
            \"""Check if new output type is requested.\"""
            return "new_output" in self.summary_types
        '''
    - An entry in the _ImplementedSummaries set, e.g.
        '''_ImplementedSummaries = {'max', 'min', 'mean', 'rms', 'peaks', 'new_output'}'''


    This suggests a regrettable level of coupling, but we've got to tell the system *somewhere* what is acceptable.

    """
    # System dimensions, used to validate indices
    max_states: int = attrs.field(validator=attrs.validators.instance_of(int))
    max_observables: int = attrs.field(validator=attrs.validators.instance_of(int))

    save_state: bool = attrs.field(default=True)
    save_observables: bool = attrs.field(default=False)
    save_time: bool = attrs.field(default=False)

    # Which indices to save (None means save all)
    saved_state_indices = attrs.field(default=None)
    saved_observable_indices = attrs.field(default=None)

    summarised_state_indices = attrs.field(default=None)
    summarised_observable_indices = attrs.field(default=None)

    # Summary types to compute
    summary_types: Set[str] = attrs.field(default=attrs.Factory(set),
                                          validator=attrs.validators.deep_iterable(attrs.validators.in_(
                                                  _ImplementedSummaries,
                                                  ),
                                                  ),
                                          )
    n_peaks: int = attrs.field(default=0)


    def __attrs_post_init__(self):
        """Swap out None index arrays, check that all indices are within bounds, and check for a no-output request."""
        self._check_saved_indices()
        self._check_summarised_indices()
        self._validate_index_arrays()
        self._check_for_no_outputs()
        self._empty_indices_if_output_not_requested()

    def _validate_index_arrays(self):
        """Ensure that saved indices arrays are valid and in bounds. This is called post-init to allow None arrays to be
        replaced with full arrays in the _indices_to_arrays step before checking.
        """
        index_arrays = [self.saved_state_indices, self.saved_observable_indices,
                        self.summarised_state_indices, self.summarised_observable_indices]
        maxima = [self.max_states, self.max_observables, self.max_states, self.max_observables]
        for i, array in enumerate(index_arrays):
            indices_validator(array, maxima[i])

    def _check_for_no_outputs(self):
        """Check if any output is requested."""
        any_output = (self.save_state or self.save_observables or self.save_time or self.save_summaries)
        if not any_output:
            raise ValueError("At least one output type must be enabled (state, observables, time, summaries)")

    def _check_saved_indices(self):
        """Convert indices iterables to numpy arrays for interface with device functions. If the array type is None,
        create an array of all possible indices."""
        if self.saved_state_indices is None:
            self.saved_state_indices = np.arange(self.max_states, dtype=np.int_)
        else:
            self.saved_state_indices = np.asarray(self.saved_state_indices, dtype=np.int_)
        if self.saved_observable_indices is None:
            self.saved_observable_indices = np.arange(self.max_observables, dtype=np.int_)
        else:
            self.saved_observable_indices = np.asarray(self.saved_observable_indices, dtype=np.int_)

    def _check_summarised_indices(self):
        """Set summarised indices to saved indices if not provided."""
        if self.summarised_state_indices is None:
            self.summarised_state_indices = self.saved_state_indices
        else:
            self.summarised_state_indices = np.asarray(self.summarised_state_indices, dtype=np.int_)
        if self.summarised_observable_indices is None:
            self.summarised_observable_indices = self.saved_observable_indices
        else:
            self.summarised_observable_indices = np.asarray(self.summarised_observable_indices, dtype=np.int_)
    def _empty_indices_if_output_not_requested(self):
        """If the the user has requested some indices be saved/summarise, but the outputs list does not include the
        requested type, then replace the indices with an empty array. For example, if saved_state_indices = [0, 1,
        2], but output_types = ["observables"], set self.saved_state_indices= np.asarray([])."""
        if not self.save_state:
            self.saved_state_indices = np.asarray([], dtype=np.int_)
        if not self.save_observables:
            self.saved_observable_indices = np.asarray([], dtype=np.int_)
        if not self.save_summaries:
            self.summarised_state_indices = np.asarray([], dtype=np.int_)
            self.summarised_observable_indices = np.asarray([], dtype=np.int_)
    @property
    def _memory_per_output_type(self):
        return {"max":   {"temp": 1, "output": 1},
                "min":   {"temp": 1, "output": 1},
                "mean":  {"temp": 1, "output": 1},
                "rms":   {"temp": 1, "output": 1},
                "peaks": {"temp": 3 + self.n_peaks, "output": self.n_peaks},
                }

    @property
    def summary_temp_memory_per_var(self) -> int:
        return sum(self._memory_per_output_type[stype]["temp"] for stype in self.summary_types)

    @property
    def summary_output_memory_per_var(self) -> int:
        return sum(self._memory_per_output_type[stype]["output"] for stype in self.summary_types)

    @property
    def save_summaries(self) -> bool:
        """Do we need to summarise anything at all?"""
        return len(self.summary_types) > 0

    @property
    def summarise_states(self) -> bool:
        """Will any states be summarised?"""
        return len(self.summary_types) > 0 and self.n_summarised_states > 0

    @property
    def summarise_observables(self) -> bool:
        """Will any observables be summarised?"""
        return len(self.summary_types) > 0 and self.n_summarised_observables > 0

    @property
    def summarise_peaks(self) -> bool:
        """Do we detect peaks?"""
        return "peaks" in self.summary_types and self.n_peaks > 0

    @property
    def summarise_mean(self) -> bool:
        """Do we calculate a running mean?"""
        return "mean" in self.summary_types

    @property
    def summarise_rms(self) -> bool:
        """Do we calculate a running RMS?"""
        return "rms" in self.summary_types

    @property
    def summarise_max(self) -> bool:
        """Do we calculate a running max?"""
        return "max" in self.summary_types

    @property
    def n_saved_states(self) -> int:
        """Number of states that will be saved (time-domain), which will the length of saved_state_indices as long as
        "save_state" is True."""
        return len(self.saved_state_indices)

    @property
    def n_saved_observables(self) -> int:
        """Number of observables that will actually be saved."""
        return len(self.saved_observable_indices)

    @property
    def n_summarised_states(self) -> int:
        """Number of states that will be summarised, which is the length of summarised_state_indices as long as
        "save_summaries" is active."""
        return len(self.summarised_state_indices)

    @property
    def n_summarised_observables(self) -> int:
        """Number of observables that will actually be summarised."""
        return len(self.summarised_observable_indices)

    def get_array_sizes(self, CUDA_allocation_safe=False) -> OutputArraySizes:
        """Calculate the number of entries required in each array for the requested outputs. Optionally,
        pass CUDA_allocation_safe=True and the function will provide a minimum size of 1, avoiding zero-sized arrays"""

        return OutputArraySizes.from_sizes(self.summary_temp_memory_per_var,
                                           self.summary_output_memory_per_var,
                                           n_saved_states=self.n_saved_states,
                                           n_saved_observables=self.n_saved_observables,
                                           n_summarised_states=self.n_summarised_states,
                                           n_summarised_observables=self.n_summarised_observables,
                                           max_states=self.max_states,
                                           max_observables=self.max_observables,
                                           save_time=self.save_time,
                                           CUDA_allocation_safe=CUDA_allocation_safe,
                                           )

    # @property
    # def get_output_function_settings(self):
    #     return {'summary_types':     self.summary_types,
    #             "save_state":        self.save_state,
    #             "save_observables":  self.save_observables,
    #             "save_time":         self.save_time,
    #             "n_peaks":           self.n_peaks,
    #             "saved_states":      self.saved_state_indices,
    #             "saved_observables": self.saved_observable_indices,
    #             "max_states":        self.max_states,
    #             "max_observables":   self.max_observables
    #             }

    @classmethod
    def from_loop_settings(cls,
                           output_types: List[str],
                           saved_states=None,
                           saved_observables=None,
                           summarised_states=None,
                           summarised_observables=None,
                           max_states: int = 0,
                           max_observables: int = 0,
                           ) -> "OutputConfig":
        """
        Create configuration from specifications in the format provided by the integrator classes.

        Args:
            output_types: List of strings specifying output types from ["state", "observables", "time", "max",
            "peaks", "mean", "rms", "min"]
            saved_states: Indices of states to save
            saved_observables: Indices of observables to save
            summarised_states: Indices of states to summarise, if different from saved_states, otherwise None
            summarised_observables: Indices of observables to summarise, if different from saved_observables, otherwise None
            n_peaks: Number of peaks to detect
            max_states: Total number of states in system
            max_observables: Total number of observables in system
        """
        # Set boolean compile flags for output types
        save_state = "state" in output_types
        save_observables = "observables" in output_types
        save_time = "time" in output_types

        # set compile flags back off if the user has provided an empty indices array
        if saved_states is not None and len(saved_states) == 0:
            save_state = False
        if saved_observables is not None and len(saved_observables) == 0:
            save_observables = False

        # Extract summary types
        cleaned_output_types, n = process_outputs_list(output_types)
        summary_types = set()
        for output_type in cleaned_output_types:
            if output_type in _ImplementedSummaries:
                summary_types.add(output_type)


        return cls(
                max_states=max_states,
                max_observables=max_observables,
                save_state=save_state,
                save_observables=save_observables,
                save_time=save_time,
                saved_state_indices=saved_states,
                saved_observable_indices=saved_observables,
                summarised_state_indices=summarised_states,
                summarised_observable_indices=summarised_observables,
                summary_types=summary_types,
                **n
                )
