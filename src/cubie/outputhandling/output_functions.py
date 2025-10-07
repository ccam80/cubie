"""Factories that compile and cache CUDA output management routines.

The module provides a single entry point, :class:`OutputFunctions`, that uses
:class:`cubie.CUDAFactory` to build CUDA callables for saving state values,
updating summary metrics, and writing summary data back to host memory. All
helper factories consume an :class:`~cubie.outputhandling.output_config.OutputConfig`
instance so the compiled functions always reflect the active configuration.
"""

from typing import Callable, Sequence, Union

import attrs
import numpy as np
from numpy.typing import ArrayLike, NDArray

from cubie.CUDAFactory import CUDAFactory
from cubie.outputhandling.output_config import OutputCompileFlags, OutputConfig
from cubie.outputhandling.output_sizes import (
    SummariesBufferSizes,
    OutputArrayHeights,
)
from cubie.outputhandling.save_state import save_state_factory
from cubie.outputhandling.save_summaries import save_summary_factory
from cubie.outputhandling.update_summaries import update_summary_factory


# Define the complete set of recognised configuration keys so callers can
# filter keyword arguments consistently before instantiating the factory.
ALL_OUTPUT_FUNCTION_PARAMETERS = {
    "output_types",
    "saved_states", "saved_observables",           # Solver-level aliases
    "summarised_states", "summarised_observables", # Solver-level aliases
    "saved_state_indices",
    "saved_observable_indices",
    "summarised_state_indices",
    "summarised_observable_indices",
}


@attrs.define
class OutputFunctionCache:
    """Cache container for compiled output functions.

    Attributes
    ----------
    save_state_function
        Compiled CUDA function for saving state values.
    update_summaries_function
        Compiled CUDA function for updating summary metrics.
    save_summaries_function
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
    """Factory that compiles and caches output management functions.

    Parameters
    ----------
    max_states
        Maximum number of state variables in the system.
    max_observables
        Maximum number of observable variables in the system.
    output_types
        Types of output to generate. Defaults to ["state"].
    saved_state_indices
        Indices of state variables to save in time-domain output.
    saved_observable_indices
        Indices of observable variables to save in time-domain output.
    summarised_state_indices
        Indices of state variables to include in summary calculations.
    summarised_observable_indices
        Indices of observable variables to include in summary calculations.

    Notes
    -----
    The constructor converts the provided options into an
    :class:`~cubie.outputhandling.output_config.OutputConfig` instance and
    installs it as the compile settings. CUDA callables are only built once
    per configuration and cached by :class:`cubie.CUDAFactory`.
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

    def update(
        self,
        updates_dict: dict[str, object] | None = None,
        silent: bool = False,
        **kwargs: object,
    ) -> set[str]:
        """Update compile settings through the factory interface.

        Parameters
        ----------
        updates_dict
            Dictionary of parameter updates to apply.
        silent
            When ``True``, suppress warnings about unrecognised parameters.
        **kwargs
            Additional parameter updates to apply.

        Returns
        -------
        set[str]
            Recognised parameter names that were successfully updated.

        Raises
        ------
        KeyError
            If unrecognised parameters are provided and ``silent`` is ``False``.

        Notes
        -----
        Use this method for coordinated configuration updates alongside other
        components by passing ``silent=True`` so unrelated keys fall through
        without raising.
        """
        if updates_dict is None:
            updates_dict = {}
        updates_dict = updates_dict.copy()
        if kwargs:
            updates_dict.update(kwargs)
        if updates_dict == {}:
            return set()
        unrecognised = set(updates_dict.keys())

        recognised_params = set()
        recognised_params |= self.update_compile_settings(
            updates_dict, silent=True
        )
        self.compile_settings.__attrs_post_init__()  # call validation funcs
        unrecognised -= recognised_params

        if not silent and unrecognised:
            raise KeyError(
                f"Unrecognized parameters in update: {unrecognised}. "
                "These parameters were not updated.",
            )
        return set(recognised_params)

    def build(self) -> OutputFunctionCache:
        """Compile output functions and calculate memory requirements.

        Returns
        -------
        OutputFunctionCache
            Container with compiled functions that target the current
            configuration.

        Notes
        -----
        This method is invoked lazily by :class:`cubie.CUDAFactory` the first
        time a compiled function is requested. The resulting cache is reused
        until configuration settings change.
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
    def save_state_func(self) -> Callable:
        """Compiled state saving function."""
        return self.get_cached_output("save_state_function")

    @property
    def update_summaries_func(self) -> Callable:
        """Compiled summary update function."""
        return self.get_cached_output("update_summaries_function")

    @property
    def output_types(self) -> set[str]:
        """Configured output types."""
        return self.compile_settings.output_types

    @property
    def save_summary_metrics_func(self) -> Callable:
        """Compiled summary saving function."""
        return self.get_cached_output("save_summaries_function")

    @property
    def compile_flags(self) -> OutputCompileFlags:
        """Compile flags for the active configuration."""
        return self.compile_settings.compile_flags

    @property
    def save_time(self) -> bool:
        """Whether time samples are saved alongside states."""
        return self.compile_settings.save_time

    @property
    def saved_state_indices(self) -> NDArray[np.int_]:
        """Indices of states saved in time-domain output."""
        return self.compile_settings.saved_state_indices

    @property
    def saved_observable_indices(self) -> NDArray[np.int_]:
        """Indices of observables saved in time-domain output."""
        return self.compile_settings.saved_observable_indices

    @property
    def summarised_state_indices(self) -> NDArray[np.int_]:
        """Indices of states tracked by summary metrics."""
        return self.compile_settings.summarised_state_indices

    @property
    def summarised_observable_indices(self) -> NDArray[np.int_]:
        """Indices of observables tracked by summary metrics."""
        return self.compile_settings.summarised_observable_indices

    @property
    def n_saved_states(self) -> int:
        """Number of state variables saved in time-domain output."""
        return self.compile_settings.n_saved_states

    @property
    def n_saved_observables(self) -> int:
        """Number of observable variables saved in time-domain output."""
        return self.compile_settings.n_saved_observables

    @property
    def state_summaries_output_height(self) -> int:
        """Height of the state summary output array."""
        return self.compile_settings.state_summaries_output_height

    @property
    def observable_summaries_output_height(self) -> int:
        """Height of the observable summary output array."""
        return self.compile_settings.observable_summaries_output_height

    @property
    def summaries_buffer_height_per_var(self) -> int:
        """Height of the summary buffer required per variable."""
        return self.compile_settings.summaries_buffer_height_per_var

    @property
    def state_summaries_buffer_height(self) -> int:
        """Total height of the buffer for state summary calculations."""
        return self.compile_settings.state_summaries_buffer_height

    @property
    def observable_summaries_buffer_height(self) -> int:
        """Total height of the buffer for observable summary calculations."""
        return self.compile_settings.observable_summaries_buffer_height

    @property
    def total_summary_buffer_size(self) -> int:
        """Total size required for all summary buffers combined."""
        return self.compile_settings.total_summary_buffer_size

    @property
    def summaries_output_height_per_var(self) -> int:
        """Height of the summary output array per variable."""
        return self.compile_settings.summaries_output_height_per_var

    @property
    def n_summarised_states(self) -> int:
        """Number of states included in summary calculations."""
        return self.compile_settings.n_summarised_states

    @property
    def n_summarised_observables(self) -> int:
        """Number of observables included in summary calculations."""
        return self.compile_settings.n_summarised_observables

    @property
    def summaries_buffer_sizes(self) -> SummariesBufferSizes:
        """Summary buffer size helper built from the active configuration."""
        return SummariesBufferSizes.from_output_fns(self)

    @property
    def output_array_heights(self) -> OutputArrayHeights:
        """Output array height helper built from the active configuration."""
        return OutputArrayHeights.from_output_fns(self)

    @property
    def summary_legend_per_variable(self) -> dict[str, int]:
        """Mapping of summary metric names to their per-variable heights."""
        return self.compile_settings.summary_legend_per_variable
