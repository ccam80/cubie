from dataclasses import dataclass
from numpy.typing import ArrayLike
from typing import Sequence, Callable
from CuMC.CUDAFactory import CUDAFactory
from CuMC.ForwardSim.OutputHandling.save_state import save_state_factory
from CuMC.ForwardSim.OutputHandling.save_summaries import save_summary_factory
from CuMC.ForwardSim.OutputHandling.update_summaries import update_summary_factory
from CuMC.ForwardSim.OutputHandling.output_config import OutputConfig
from CuMC.ForwardSim.OutputHandling.output_sizes import OutputArrayHeights, SummariesBufferSizes
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
        """Compile three functions: Save state, update summaries metrics, and save summaries.
        Calculate memory requirements for buffer and output arrays.

        Returns:
            A dictionary containing all compiled functions and memory requirements
        """
        config = self.compile_settings

        heights = OutputArrayHeights.from_output_fns(self)
        buffer_sizes = SummariesBufferSizes.from_output_fns(self)

        # Build functions using output sizes objects
        save_state_func = save_state_factory(
                heights,
                config.saved_state_indices,
                config.saved_observable_indices,
                config.save_state,
                config.save_observables,
                config.save_time,
                )

        update_summary_metrics_func = update_summary_factory(
                buffer_sizes,
                config.summarised_state_indices,
                config.summarised_observable_indices,
                config.summary_types,
                )

        save_summary_metrics_func = save_summary_factory(
                buffer_sizes,
                config.summarised_state_indices,
                config.summarised_observable_indices,
                config.summary_types,
                )

        return OutputFunctionCache(
                save_state_function=save_state_func,
                update_summaries_function=update_summary_metrics_func,
                save_summaries_function=save_summary_metrics_func,
                )



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
        """Return the memory requirements for buffer and output arrays."""
        return {
            'buffer': self.compile_settings.summaries_buffer_height_per_var,
            'output':    self.compile_settings.summaries_output_height_per_var,
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
    def state_summaries_output_height(self) -> int:
        """Height of the output array for state summaries."""
        return self.compile_settings.state_summaries_output_height

    @property
    def observable_summaries_output_height(self) -> int:
        """Height of the output array for observable summaries."""
        return self.compile_settings.observable_summaries_output_height

    @property
    def summaries_buffer_height_per_var(self) -> int:
        """Calculate the height of the state summaries buffer."""
        return self.compile_settings.summaries_buffer_height_per_var

    @property
    def state_summaries_buffer_height(self) -> int:
        """Calculate the height of the state summaries buffer."""
        return self.compile_settings.state_summaries_buffer_height

    @property
    def observable_summaries_buffer_height(self) -> int:
        """Calculate the height of the observable summaries buffer."""
        return self.compile_settings.observable_summaries_buffer_height

    @property
    def summaries_output_height_per_var(self) -> int:
        """Calculate the height of the state summaries output."""
        return self.compile_settings.summaries_output_height_per_var

    @property
    def n_summarised_states(self) -> int:
        """Number of states that will be summarised, which is the length of summarised_state_indices as long as
        "save_summaries" is active."""
        return self.compile_settings.n_summarised_states

    @property
    def n_summarised_observables(self) -> int:
        """Number of observables that will actually be summarised."""
        return self.compile_settings.n_summarised_observables
