from dataclasses import dataclass
from numpy import asarray
from numpy.typing import ArrayLike
from typing import Sequence, Dict, Any, Callable
from CuMC.CUDAFactory import CUDAFactory
from CuMC.ForwardSim.OutputHandling.save_state import save_state_factory
from CuMC.ForwardSim.OutputHandling.save_summaries import save_summary_factory
from CuMC.ForwardSim.OutputHandling.update_summaries import update_summary_factory
from CuMC.ForwardSim.OutputHandling.output_config import OutputConfig
import attrs


#feature: max absolute
#feature: running std deviation
#feature: min
#feature: neg_peak
#feature: both_extrema
#feature: dxdt_extrema
#feature: d2xdt2 extrema
#feature: dxdt_max_peaks
#feature: raw dxdt

#TODO: Implement a "terminate" flag to communicate that a condition has been met - e.g. we have found a peak,
# so stop integrating.


#TODO: Replace all references to these with a call to get_memory_requirements
# _TempMemoryRequirements
# _OutputMemoryRequirements

@attrs.define
class OutputFunctionCache:
    save_state_function: Callable = attrs.field(validator=attrs.validators.instance_of(Callable))
    update_summaries_function: Callable = attrs.field(validator=attrs.validators.instance_of(Callable))
    save_summaries_function: Callable = attrs.field(validator=attrs.validators.instance_of(Callable))


class OutputFunctions(CUDAFactory):
    """Class to hold output functions and associated data, with automatic caching of built functions provided by the
    CUDAFactory base class.
    """

    def __init__(self,
                 max_states: int,
                 max_observables: int,
                 output_types: list[str] = None,
                 saved_states: Sequence[int] | ArrayLike = None,
                 saved_observables: Sequence[int] | ArrayLike = None,
                 summarised_states: Sequence[int] | ArrayLike = None,
                 summarised_observables: Sequence[int] | ArrayLike = None,
                 ):
        super().__init__()

        if output_types is None:
            output_types = ["state"]

        # Create and setup output configuration as compile settings
        config = OutputConfig.from_loop_settings(
                output_types=output_types,
                max_states=max_states,
                max_observables=max_observables,
                saved_states=saved_states,
                saved_observables=saved_observables,
                summarised_states=summarised_states,
                summarised_observables=summarised_observables,
                )
        self.setup_compile_settings(config)

    def update(self, **kwargs):
        """Update the configuration of the output functions with new parameters."""
        self.update_compile_settings(**kwargs)

    def build(self) -> OutputFunctionCache:
        """Compile three functions: Save state, update summary metrics, and save summaries.
        Calculate memory requirements for temporary and output arrays.

        Returns:
            A dictionary containing all compiled functions and memory requirements
        """
        config = self.compile_settings

        # Build functions using config attributes
        save_state_func = save_state_factory(
                config.n_saved_states,
                config.n_saved_observables,
                config.saved_state_indices,
                config.saved_observable_indices,
                config.save_state,
                config.save_observables,
                config.save_time,
                )

        update_summary_metrics_func = update_summary_factory(
                config.summarised_state_indices,
                config.summarised_observable_indices,
                config.save_summaries, config.summarise_mean,
                config.summarise_peaks, config.summarise_max,
                config.summarise_rms,
                )

        save_summary_metrics_func = save_summary_factory(
                config.save_summaries,
                config.summarise_mean,
                config.summarise_peaks,
                config.summarise_max,
                config.summarise_rms,
                config.save_observables,
                config.n_peaks,
                )

        return OutputFunctionCache(
                save_state_function=save_state_func,
                update_summaries_function=update_summary_metrics_func,
                save_summaries_function=save_summary_metrics_func,
                )

    @property
    def array_sizes(self):
        return self.compile_settings.get_array_sizes()

    @property
    def nonzero_array_sizes(self):
        return self.compile_settings.get_array_sizes(CUDA_allocation_safe=True)

    @property
    def save_state_func(self):
        """Return the save_state function. Will rebuild if necessary."""
        return self.get_cached_output('save_state_function')

    @property
    def update_summaries_func(self):
        """Return the update_summary_metrics function. Will rebuild if necessary."""
        return self.get_cached_output('update_summaries_function')

    @property
    def summary_types(self):
        """Return a set of the summaries requested/compiled into the functions"""
        return self.compile_settings.summary_types

    @property
    def save_summary_metrics_func(self):
        """Return the save_summary_metrics function. Will rebuild if necessary."""
        return self.get_cached_output('save_summaries_function')

    @property
    def memory_per_summarised_variable(self):
        """Return the memory requirements for temporary and output arrays."""
        return {
            'temporary': self.compile_settings.summary_temp_memory_per_var,
            'output':    self.compile_settings.summary_output_memory_per_var,
            }

    @property
    def save_time(self):
        """Return whether time is being saved."""
        return self.compile_settings.save_time

    @property
    def saved_state_indices(self):
        """Return array of saved state indices"""
        return self.compile_settings.saved_state_indices

    @property
    def saved_observable_indices(self):
        """Return array of saved ovservable indices"""
        return self.compile_settings.saved_observable_indices

    @property
    def summarised_state_indices(self):
        """Return array of saved state indices"""
        return self.compile_settings.summarised_state_indices

    @property
    def summarised_observable_indices(self):
        """Return array of saved ovservable indices"""
        return self.compile_settings.summarised_observable_indices

    @property
    def n_saved_states(self) -> int:
        """Number of states that will be saved (time-domain), which will the length of saved_state_indices as long as
        "save_state" is True."""
        return self.compile_settings.n_saved_states

    @property
    def n_saved_observables(self) -> int:
        """Number of observables that will actually be saved."""
        return self.compile_settings.n_saved_observables

    @property
    def n_summarised_states(self) -> int:
        """Number of states that will be summarised, which is the length of summarised_state_indices as long as
        "save_summaries" is active."""
        return self.compile_settings.n_summarised_states

    @property
    def n_summarised_observables(self) -> int:
        """Number of observables that will actually be summarised."""
        return self.compile_settings.n_summarised_observables

    def get_output_sizes(self, n_samples: int = 0, n_summary_samples: int = 0, numruns: int = 0, for_allocation=True) \
            -> dict[str, tuple]:
        """
        Get output array sizes for the current configuration. Call with no arguments for the heights of arrays (
        number of elements per sample), call with n_samples and n_summary_samples to get 2d "slice" shapes,
        and call with numruns as well to get the 3d full run array shapes.

        Args:
            n_samples: int
                Number of time-domain samples. Sets the first dimension of 2d and 3d time-domain arrays
            n_summary_samples: int
                Number of summary samples. Sets the first dimension of 2d and 3d summary arrays
            numruns: int
                Number of runs, used to set the "middle" dimension of 3d arrays
            for_allocation: If you're using this to allocate Memory, return minimum size 1 arrays to avoid breaking
                the memory allocator in numba.cuda.

        Returns:
            Dictionary with array names and their (samples, variables) shapes

        Example:
            '''
            >>> output_sizes = output_functions.get_output_sizes()
            >>> print(output_sizes)
            {
                'state': 5,
                'observables': 3,
                'state_summaries': 4,
                'observable_summaries': 2
            }
            >>> output_sizes = output_functions.get_output_sizes(n_samples=100, n_summary_samples=10)
            >>> print(output_sizes)
            {
                'state': (100, 5),
                'observables': (100, 3),
                'state_summaries': (10, 4),
                'observable_summaries': (10, 2)
            }
            >>> output_sizes = output_functions.get_output_sizes(n_samples=100, n_summary_samples=10, numruns=32)
            >>> print(output_sizes)
            {
                'state': (32, 100, 5),
                'observables': (32, 100, 3),
                'state_summaries': (32, 10, 4),
                'observable_summaries': (32, 10, 2)
            }

            '''
            if for_allocation is true, any shapes featuring a zero will be replaced with a tuple full of ones.

            #TODO: Move this into a kernel-level allocator function.
        """
        sizes = self.array_sizes()
        if n_samples == 0 and n_summary_samples == 0:
            state_shape = sizes.state.output
            observable_shape = sizes.observables.output
            state_summaries_shape = sizes.state_summaries.output
            observable_summaries_shape = sizes.observable_summaries.output
            one_element = 1
        elif numruns == 0:
            state_shape = (n_samples, sizes.state.output)
            observable_shape = (n_samples, sizes.observables.output)
            state_summaries_shape = (n_summary_samples, sizes.state_summaries.output)
            observable_summaries_shape = (n_summary_samples, sizes.observable_summaries.output)
            one_element = (1, 1)
        else:
            state_shape = (n_samples, numruns, sizes.state.output)
            observable_shape = (n_samples, numruns, sizes.observables.output)
            state_summaries_shape = (n_summary_samples, numruns, sizes.state_summaries.output)
            observable_summaries_shape = (n_summary_samples, numruns, sizes.observable_summaries.output)
            one_element = (1, 1, 1)

        array_size_dict = {'state':                state_shape,
                           'observables':          observable_shape,
                           'state_summaries':      state_summaries_shape,
                           'observable_summaries': observable_summaries_shape
                           }

        if for_allocation:
            # Replace any zero dimensions with ones to avoid breaking the memory allocator
            for key, value in array_size_dict.items():
                if 0 in value:
                    array_size_dict[key] = one_element

        return array_size_dict