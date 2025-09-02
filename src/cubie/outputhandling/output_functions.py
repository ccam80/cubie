"""
Output function management for CUDA-accelerated batch solving.

This module provides classes for managing output functions that handle state
saving, summary calculations, and writing to memory for CUDA batch solvers.
The functions are automatically cached and compiled on demand through the
CUDAFactory base class.
"""

from typing import Sequence, Callable, Union

import attrs
from numpy.typing import ArrayLike

from cubie.CUDAFactory import CUDAFactory
from cubie.outputhandling.output_config import OutputConfig
from cubie.outputhandling.output_sizes import (
    SummariesBufferSizes,
    OutputArrayHeights,
)
from cubie.outputhandling.save_state import save_state_factory
from cubie.outputhandling.save_summaries import save_summary_factory
from cubie.outputhandling.update_summaries import update_summary_factory


@attrs.define
class OutputFunctionCache:
    """
    Cache container for compiled output functions.

    This class holds the three main compiled functions used in output
    handling: state saving, summary updates, and summary saving. It serves
    as a data container returned by the OutputFunctions.build() method.

    Attributes
    ----------
    save_state_function : Callable
        Compiled CUDA function for saving state values.
    update_summaries_function : Callable
        Compiled CUDA function for updating summary metrics.
    save_summaries_function : Callable
        Compiled CUDA function for saving summary results.
    """

    save_state_function: Callable = attrs.field(
        validator=attrs.validators.instance_of(Callable)
    )
    update_summaries_function: Callable = attrs.field(
        validator=attrs.validators.instance_of(Callable)
    )
    save_summaries_function: Callable = attrs.field(
        validator=attrs.validators.instance_of(Callable)
    )


class OutputFunctions(CUDAFactory):
    """
    Output function factory with automatic caching.

    This class manages the creation and caching of CUDA-compiled output
    functions for state saving and summary calculations. It extends
    CUDAFactory to provide automatic function caching and invalidation when
    settings change.

    Parameters
    ----------
    max_states : int
        Maximum number of state variables in the system.
    max_observables : int
        Maximum number of observable variables in the system.
    output_types : list[str], optional
        Types of output to generate. Default is ["state"].
    saved_state_indices : Sequence[int] or ArrayLike, optional
        Indices of state variables to save in time-domain output.
    saved_observable_indices : Sequence[int] or ArrayLike, optional
        Indices of observable variables to save in time-domain output.
    summarised_state_indices : Sequence[int] or ArrayLike, optional
        Indices of state variables to include in summary calculations.
    summarised_observable_indices : Sequence[int] or ArrayLike, optional
        Indices of observable variables to include in summary calculations.

    Notes
    -----
    The class automatically creates an OutputConfig object from the provided
    parameters and sets it up as compile settings through the CUDAFactory
    interface. Functions are compiled lazily when first accessed.
    """

    def __init__(
        self,
        max_states: int,
        max_observables: int,
        output_types: list[str] = None,
        saved_state_indices: Union[Sequence[int], ArrayLike] = None,
        saved_observable_indices: Union[Sequence[int], ArrayLike] = None,
        summarised_state_indices: Union[Sequence[int], ArrayLike] = None,
        summarised_observable_indices: Union[Sequence[int], ArrayLike] = None,
    ):
        super().__init__()

        if output_types is None:
            output_types = ["state"]

        # Create and setup output configuration as compile settings
        config = OutputConfig.from_loop_settings(
            output_types=output_types,
            max_states=max_states,
            max_observables=max_observables,
            saved_state_indices=saved_state_indices,
            saved_observable_indices=saved_observable_indices,
            summarised_state_indices=summarised_state_indices,
            summarised_observable_indices=summarised_observable_indices,
        )
        self.setup_compile_settings(config)

    def update(self, updates_dict=None, silent=False, **kwargs):
        """
        Update compile settings through the CUDAFactory interface.

        Pass updates to compile settings, which will invalidate the function
        cache if an update is successful. This allows dynamic reconfiguration
        of output parameters.

        Parameters
        ----------
        updates_dict : dict, optional
            Dictionary of parameter updates to apply.
        silent : bool, optional
            If True, suppress warnings about unrecognized parameters.
            Default is False.
        **kwargs
            Additional parameter updates to apply.

        Returns
        -------
        set
            Set of recognized parameter names that were successfully updated.

        Raises
        ------
        KeyError
            If unrecognized parameters are provided and silent=False.

        Notes
        -----
        This method is useful for bulk updates with other component parameters
        when silent=True is used to suppress warnings about keys not found in
        this component.
        """
        if updates_dict is None:
            updates_dict = {}
        if kwargs:
            updates_dict.update(kwargs)
        if updates_dict == {}:
            return []
        unrecognised = set(updates_dict.keys())

        recognised_params = set()
        recognised_params |= self.update_compile_settings(
            updates_dict, silent=True
        )
        unrecognised -= recognised_params

        if not silent and unrecognised:
            raise KeyError(
                f"Unrecognized parameters in update: {unrecognised}. "
                "These parameters were not updated.",
            )
        return set(recognised_params)

    def build(self) -> OutputFunctionCache:
        """
        Compile output functions and calculate memory requirements.

        Compiles three main functions: save state, update summary metrics,
        and save summaries. Also calculates memory requirements for buffer
        and output arrays based on the current configuration.

        Returns
        -------
        OutputFunctionCache
            Container with all compiled functions and memory requirements.

        Notes
        -----
        This method is called automatically by the CUDAFactory when functions
        are first accessed. The compiled functions are optimized for the
        current configuration settings.
        """
        config = self.compile_settings

        buffer_sizes = self.summaries_buffer_sizes

        # Build functions using output sizes objects
        save_state_func = save_state_factory(
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
        """
        Access the compiled state saving function.

        Returns
        -------
        Callable
            Compiled CUDA function for saving state values.

        Notes
        -----
        Exposes save_state_function from the cached OutputFunctionCache
        object. The function is compiled on first access if not already cached.
        """
        return self.get_cached_output("save_state_function")

    @property
    def update_summaries_func(self):
        """
        Access the compiled summary update function.

        Returns
        -------
        Callable
            Compiled CUDA function for updating summary metrics.

        Notes
        -----
        Exposes update_summaries_function from the cached OutputFunctionCache
        object. The function is compiled on first access if not already
        cached.
        """
        return self.get_cached_output("update_summaries_function")

    @property
    def output_types(self):
        """
        Get the output types configured for compilation.

        Returns
        -------
        set
            Set of output types requested/compiled into the functions.
        """
        return self.compile_settings.output_types

    @property
    def save_summary_metrics_func(self):
        """
        Access the compiled summary saving function.

        Returns
        -------
        Callable
            Compiled CUDA function for saving summary results.

        Notes
        -----
        Exposes save_summaries_function from the cached OutputFunctionCache
        object. The function is compiled on first access if not already
        cached.
        """
        return self.get_cached_output("save_summaries_function")

    @property
    def compile_flags(self):
        """
        Get the compile flags for output functions.

        Returns
        -------
        OutputCompileFlags
            Compile flags indicating which output types are enabled.
        """
        return self.compile_settings.compile_flags

    @property
    def save_time(self):
        """
        Check if time saving is enabled.

        Returns
        -------
        bool
            True if time values are being saved with state data.
        """
        return self.compile_settings.save_time

    @property
    def saved_state_indices(self):
        """
        Get indices of states to save in time-domain output.

        Returns
        -------
        np.ndarray
            Array of state variable indices for time-domain saving.
        """
        return self.compile_settings.saved_state_indices

    @property
    def saved_observable_indices(self):
        """
        Get indices of observables to save in time-domain output.

        Returns
        -------
        np.ndarray
            Array of observable variable indices for time-domain saving.
        """
        return self.compile_settings.saved_observable_indices

    @property
    def summarised_state_indices(self):
        """
        Get indices of states to include in summary calculations.

        Returns
        -------
        np.ndarray
            Array of state variable indices for summary calculations.
        """
        return self.compile_settings.summarised_state_indices

    @property
    def summarised_observable_indices(self):
        """
        Get indices of observables to include in summary calculations.

        Returns
        -------
        np.ndarray
            Array of observable variable indices for summary calculations.
        """
        return self.compile_settings.summarised_observable_indices

    @property
    def n_saved_states(self) -> int:
        """
        Get number of states that will be saved in time-domain output.

        Returns
        -------
        int
            Number of state variables to save, which equals the length of
            saved_state_indices when save_state is True.
        """
        return self.compile_settings.n_saved_states

    @property
    def n_saved_observables(self) -> int:
        """
        Get number of observables that will be saved in time-domain output.

        Returns
        -------
        int
            Number of observable variables that will actually be saved.
        """
        return self.compile_settings.n_saved_observables

    @property
    def state_summaries_output_height(self) -> int:
        """
        Get height of the output array for state summaries.

        Returns
        -------
        int
            Height of the output array for state summary data.
        """
        return self.compile_settings.state_summaries_output_height

    @property
    def observable_summaries_output_height(self) -> int:
        """
        Get height of the output array for observable summaries.

        Returns
        -------
        int
            Height of the output array for observable summary data.
        """
        return self.compile_settings.observable_summaries_output_height

    @property
    def summaries_buffer_height_per_var(self) -> int:
        """
        Get height of the summary buffer per variable.

        Returns
        -------
        int
            Height of the summary buffer required per variable.
        """
        return self.compile_settings.summaries_buffer_height_per_var

    @property
    def state_summaries_buffer_height(self) -> int:
        """
        Get height of the state summaries buffer.

        Returns
        -------
        int
            Total height of the buffer for state summary calculations.
        """
        return self.compile_settings.state_summaries_buffer_height

    @property
    def observable_summaries_buffer_height(self) -> int:
        """
        Get height of the observable summaries buffer.

        Returns
        -------
        int
            Total height of the buffer for observable summary calculations.
        """
        return self.compile_settings.observable_summaries_buffer_height

    @property
    def total_summary_buffer_size(self) -> int:
        """
        Get total size of the summaries buffer.

        Returns
        -------
        int
            Total size required for all summary buffers combined.
        """
        return self.compile_settings.total_summary_buffer_size

    @property
    def summaries_output_height_per_var(self) -> int:
        """
        Get height of the summary output per variable.

        Returns
        -------
        int
            Height of the summary output array per variable.
        """
        return self.compile_settings.summaries_output_height_per_var

    @property
    def n_summarised_states(self) -> int:
        """
        Get number of states that will be summarised.

        Returns
        -------
        int
            Number of state variables included in summary calculations,
            which equals the length of summarised_state_indices when
            save_summaries is active.
        """
        return self.compile_settings.n_summarised_states

    @property
    def n_summarised_observables(self) -> int:
        """
        Get number of observables that will be summarised.

        Returns
        -------
        int
            Number of observable variables that will actually be summarised.
        """
        return self.compile_settings.n_summarised_observables

    @property
    def summaries_buffer_sizes(self) -> SummariesBufferSizes:
        """
        Get summary buffer size information.

        Returns
        -------
        SummariesBufferSizes
            Object containing buffer size information for summary
            calculations.

        Notes
        -----
        Exposes SummariesBufferSizes from the child SummariesBufferSizes
        object.
        """
        return SummariesBufferSizes.from_output_fns(self)

    @property
    def output_array_heights(self) -> OutputArrayHeights:
        """
        Get output array height information.

        Returns
        -------
        OutputArrayHeights
            Object containing height information for output arrays.

        Notes
        -----
        Exposes OutputArrayHeights from the child OutputArrayHeights object.
        """
        return OutputArrayHeights.from_output_fns(self)

    @property
    def summary_legend_per_variable(self) -> dict[str, int]:
        """
        Get mapping of summary names to their heights per variable.

        Returns
        -------
        dict[str, int]
            Dictionary mapping summary metric names to their required
            heights per variable.
        """
        return self.compile_settings.summary_legend_per_variable
