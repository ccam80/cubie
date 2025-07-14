from dataclasses import dataclass
from numpy import asarray
from numpy.typing import ArrayLike
from typing import Sequence, Dict, Any
from CuMC.CUDAFactory import CUDAFactory
from CuMC.ForwardSim.OutputHandling.save_state import save_state_factory
from CuMC.ForwardSim.OutputHandling.save_summaries import save_summary_factory
from CuMC.ForwardSim.OutputHandling.update_summaries import update_summary_factory
from CuMC.ForwardSim.OutputHandling.output_config import OutputConfig

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
                 n_peaks: int = 0,
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
            n_peaks=n_peaks,
        )
        self.setup_compile_settings(config)

    def update(self, **kwargs):
        """Update the configuration of the output functions with new parameters."""
        #TODO: Remove this double-checking LBYL nonsense if it works

        # config_updates = {}
        # for key in ['output_types',
        #             'saved_states',
        #             'saved_observables',
        #             'summarised_peaks',
        #             'summarised_observables',
        #             'n_peaks']:
        #     if key in kwargs:
        #         config_updates[key] = kwargs[key]
        #
        # if config_updates:
        self.update_compile_settings(**kwargs)


    def build(self) -> Dict[str, Any]:
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
            config.n_summarised_states,
            config.n_summarised_observables,
            config.summarised_state_indices,
            config.summarised_observable_indices,
            config.save_summaries,
            config.summarise_mean,
            config.summarise_peaks,
            config.summarise_max,
            config.summarise_rms,
            config.save_observables,
            config.n_peaks,
        )

        save_summary_metrics_func = save_summary_factory(
            config.n_summarised_states,
            config.n_summarised_observables,
            config.save_summaries,
            config.summarise_mean,
            config.summarise_peaks,
            config.summarise_max,
            config.summarise_rms,
            config.save_observables,
            config.n_peaks,
        )

        return {
            'save_state_function': save_state_func,
            'update_summary_metrics_function': update_summary_metrics_func,
            'save_summary_metrics_function': save_summary_metrics_func,
        }

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
    def update_summary_metrics_func(self):
        """Return the update_summary_metrics function. Will rebuild if necessary."""
        return self.get_cached_output('update_summary_metrics_function')

    @property
    def save_summary_metrics_func(self):
        """Return the save_summary_metrics function. Will rebuild if necessary."""
        return self.get_cached_output('save_summary_metrics_function')

    @property
    def memory_per_summarised_variable(self):
        """Return the memory requirements for temporary and output arrays."""
        return {
            'temporary': self.get_cached_output('temp_memory_requirements'),
            'output': self.get_cached_output('summary_output_length')
        }

    @property
    def save_time(self):
        """Return whether time is being saved."""
        return self.compile_settings.save_time

    def get_output_sizes(self, n_samples: int, n_summary_samples: int, for_allocation=True) -> Dict[str, tuple]:
        """
        Get output array sizes for the current configuration.

        Args:
            n_samples: Number of time-domain samples
            n_summary_samples: Number of summary samples
            for_allocation: If you're using this to allocate Memory, return minimum size 1 arrays to avoid breaking
                the memory allocator in numba.cuda.

        Returns:
            Dictionary with array names and their (samples, variables) shapes

            #TODO: Rejig this to live inside outputconfig. It should also have a class as keys aren't modifiable.
            #fixme: If for_allocation is True, this will return 1xnum_summaries arrays instead of the desired 1x1.
        """
        if for_allocation:
            sizes = self.nonzero_array_sizes()
        else:
            sizes = self.array_sizes()
        return {
            'state': (n_samples, sizes.state.output),
            'observables': (n_samples, sizes.observables.output),
            'state_summaries': (n_summary_samples, sizes.state_summaries.output),
            'observable_summaries': (n_summary_samples, sizes.observable_summaries.output)
        }
