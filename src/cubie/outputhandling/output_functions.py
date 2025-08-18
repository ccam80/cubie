from typing import Sequence, Callable

import attrs
from numpy.typing import ArrayLike

from cubie.CUDAFactory import CUDAFactory
from cubie.outputhandling.output_config import OutputConfig
from cubie.outputhandling.output_sizes import SummariesBufferSizes, \
    OutputArrayHeights
from cubie.outputhandling.save_state import save_state_factory
from cubie.outputhandling.save_summaries import save_summary_factory
from cubie.outputhandling.update_summaries import \
    update_summary_factory


@attrs.define
class OutputFunctionCache:
    save_state_function: Callable = attrs.field(
            validator=attrs.validators.instance_of(Callable))
    update_summaries_function: Callable = attrs.field(
            validator=attrs.validators.instance_of(Callable))
    save_summaries_function: Callable = attrs.field(
            validator=attrs.validators.instance_of(Callable))


class OutputFunctions(CUDAFactory):
    """Class to hold output functions and associated data, with automatic
    caching of built functions provided by the
    CUDAFactory base class.
    """

    def __init__(self, max_states: int, max_observables: int,
                 output_types: list[str] = None,
                 saved_state_indices: Sequence[int] | ArrayLike = None,
                 saved_observable_indices: Sequence[int] | ArrayLike = None,
                 summarised_state_indices: Sequence[int] | ArrayLike = None,
                 summarised_observable_indices: Sequence[
                                                    int] | ArrayLike = None, ):
        super().__init__()

        if output_types is None:
            output_types = ["state"]

        # Create and setup output configuration as compile settings
        config = OutputConfig.from_loop_settings(output_types=output_types,
                max_states=max_states, max_observables=max_observables,
                saved_state_indices=saved_state_indices,
                saved_observable_indices=saved_observable_indices,
                summarised_state_indices=summarised_state_indices,
                summarised_observable_indices=summarised_observable_indices, )
        self.setup_compile_settings(config)

    def update(self, updates_dict=None, silent=False, **kwargs):
        """        Pass updates to compile settings through the CUDAFactory interface, which will invalidate cache if an update
        is successful. Pass silent=True if doing a bulk update with other component's params to suppress warnings
        about keys not found.

        Args:
            silent (bool): If True, suppress warnings about unrecognized parameters
            **kwargs: Parameter updates to apply

        Returns:
            list: recognized_params
            """
        if updates_dict is None:
            updates_dict = {}
        if kwargs:
            updates_dict.update(kwargs)
        if updates_dict == {}:
            return []
        unrecognised = set(updates_dict.keys())

        recognised_params = set()
        recognised_params |= self.update_compile_settings(updates_dict,
                                                          silent=True)
        unrecognised -= recognised_params

        if not silent and unrecognised:
            raise KeyError(
                    f"Unrecognized parameters in update: {unrecognised}. "
                    "These parameters were not updated.", )
        return set(recognised_params)

    def build(self) -> OutputFunctionCache:
        """Compile three functions: Save state, update summaries_array metrics, and save summaries_array.
        Calculate memory requirements for buffer and output arrays.

        Returns:
            A dictionary containing all compiled functions and memory requirements
        """
        config = self.compile_settings

        buffer_sizes = self.summaries_buffer_sizes

        # Build functions using output sizes objects
        save_state_func = save_state_factory(config.saved_state_indices,
                config.saved_observable_indices, config.save_state,
                config.save_observables, config.save_time, )

        update_summary_metrics_func = update_summary_factory(buffer_sizes,
                config.summarised_state_indices,
                config.summarised_observable_indices, config.summary_types, )

        save_summary_metrics_func = save_summary_factory(buffer_sizes,
                config.summarised_state_indices,
                config.summarised_observable_indices, config.summary_types, )

        return OutputFunctionCache(save_state_function=save_state_func,
                update_summaries_function=update_summary_metrics_func,
                save_summaries_function=save_summary_metrics_func, )

    @property
    def save_state_func(self):
        """Exposes :attr:`~cubie.batchsolving.outputhandling.output_functions.OutputFunctionCache.save_state_function` from the child OutputFunctionCache object."""
        return self.get_cached_output('save_state_function')

    @property
    def update_summaries_func(self):
        """Exposes :attr:`~cubie.batchsolving.outputhandling.output_functions.OutputFunctionCache.update_summaries_function` from the child OutputFunctionCache object."""
        return self.get_cached_output('update_summaries_function')

    @property
    def output_types(self):
        """Return a set of the summaries_array requested/compiled into the functions"""
        return self.compile_settings.output_types

    @property
    def save_summary_metrics_func(self):
        """Exposes :attr:`~cubie.batchsolving.outputhandling.output_functions.OutputFunctionCache.save_summaries_function` from the child OutputFunctionCache object."""
        return self.get_cached_output('save_summaries_function')

    @property
    def compile_flags(self):
        """Return the compile flags for the output functions."""
        return self.compile_settings.compile_flags

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
        """Height of the output array for state summaries_array."""
        return self.compile_settings.state_summaries_output_height

    @property
    def observable_summaries_output_height(self) -> int:
        """Height of the output array for observable summaries_array."""
        return self.compile_settings.observable_summaries_output_height

    @property
    def summaries_buffer_height_per_var(self) -> int:
        """Calculate the height of the state summaries_array buffer."""
        return self.compile_settings.summaries_buffer_height_per_var

    @property
    def state_summaries_buffer_height(self) -> int:
        """Calculate the height of the state summaries_array buffer."""
        return self.compile_settings.state_summaries_buffer_height

    @property
    def observable_summaries_buffer_height(self) -> int:
        """Calculate the height of the observable summaries_array buffer."""
        return self.compile_settings.observable_summaries_buffer_height

    @property
    def total_summary_buffer_size(self) -> int:
        """Calculate the total size of the summaries_array buffer."""
        return self.compile_settings.total_summary_buffer_size

    @property
    def summaries_output_height_per_var(self) -> int:
        """Calculate the height of the state summaries_array output."""
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

    @property
    def summaries_buffer_sizes(self) -> SummariesBufferSizes:
        """Exposes :class:`~cubie.batchsolving.outputhandling.output_sizes.SummariesBufferSizes` from the child SummariesBufferSizes object."""
        return SummariesBufferSizes.from_output_fns(self)

    @property
    def output_array_heights(self) -> OutputArrayHeights:
        """Exposes :class:`~cubie.batchsolving.outputhandling.output_sizes.OutputArrayHeights` from the child OutputArrayHeights object."""
        return OutputArrayHeights.from_output_fns(self)

    @property
    def summary_legend_per_variable(self) -> dict[str, int]:
        """Return a dictionary mapping summary names to their heights per variable."""
        return self.compile_settings.summary_legend_per_variable
