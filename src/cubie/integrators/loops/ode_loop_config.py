"""Configuration helpers for CUDA-based integration loops.

The objects defined here capture shared and local buffer layouts alongside
compile-critical metadata such as precision, save cadence, and device
callbacks. They centralise validation so that loop factories receive
consistent, ready-to-compile settings.
"""
from typing import Callable, MutableMapping, Optional, Union

from attrs import define, field, validators
import numba
from numpy import float32

from cubie._utils import (
    PrecisionDType,
    getype_validator,
    gttype_validator,
    is_device_validator,
    precision_converter,
    precision_validator,
    opt_getype_validator,
    opt_gttype_validator,
)
from cubie.cuda_simsafe import from_dtype as simsafe_dtype
from cubie.outputhandling.output_config import OutputCompileFlags

valid_opt_slice = validators.optional(validators.instance_of(slice))

@define
class LoopLocalIndices:
    """Index layout for persistent local memory buffers.

    Attributes
    ----------
    timestep_slice
        Slice pointing to the timestep storage element.
    acceptance_flag_slice
        Slice pointing to the acceptance flag storage element.
    controller_state_slice
        Slice covering scratch space reserved for the controller state.
    algorithm_state_slice
        Slice covering scratch space reserved for the algorithm state.
    loop_buffer_end_offset
        Offset of the end of loop-managed storage.
    total_buffer_end_offset
        Offset of the end of the persistent local buffer.
    all
        Slice that spans the entire persistent local buffer.
    """

    timestep_slice: Optional[slice] = field(default=None, validator=valid_opt_slice)
    acceptance_flag_slice: Optional[slice] = field(default=None, validator=valid_opt_slice)
    controller_state_slice: Optional[slice] = field(
        default=None, validator=valid_opt_slice
    )
    algorithm_state_slice: Optional[slice] = field(default=None, validator=valid_opt_slice)
    loop_buffer_end_offset: Optional[int] = field(
        default=None, validator=opt_getype_validator(int, 0)
    )
    total_buffer_end_offset: Optional[int] = field(
        default=None, validator=opt_getype_validator(int, 0)
    )
    all: Optional[slice] = field(default=None, validator=valid_opt_slice)

    @classmethod
    def empty(cls) -> "LoopLocalIndices":
        """Build an empty local-memory layout.

        Returns
        -------
        LoopLocalIndices
            Layout with zero-length slices for all buffers.
        """

        zero = slice(0, 0)
        return cls(
            timestep_slice=zero,
            acceptance_flag_slice=zero,
            controller_state_slice=zero,
            algorithm_state_slice=zero,
            loop_buffer_end_offset=0,
            total_buffer_end_offset=0,
            all=slice(None),
        )

    @classmethod
    def from_sizes(
        cls, controller_len: int, algorithm_len: int
    ) -> "LoopLocalIndices":
        """Build index slices from component memory requirements.

        Parameters
        ----------
        controller_len
            Number of persistent elements reserved for the controller.
        algorithm_len
            Number of persistent elements reserved for the algorithm.

        Returns
        -------
        LoopLocalIndices
            Layout sized to cover the requested buffer lengths.
        """

        controller_len = max(int(controller_len), 0)
        algorithm_len = max(int(algorithm_len), 0)

        timestep_slice_value = slice(0, 1)
        acceptance_flag_slice_value = slice(1, 2)
        controller_start = acceptance_flag_slice_value.stop
        controller_stop = controller_start + controller_len
        controller_state_slice_value = slice(controller_start, controller_stop)

        algorithm_start = controller_stop
        algorithm_stop = algorithm_start + algorithm_len
        algorithm_state_slice_value = slice(algorithm_start, algorithm_stop)

        return cls(
            timestep_slice=timestep_slice_value,
            acceptance_flag_slice=acceptance_flag_slice_value,
            controller_state_slice=controller_state_slice_value,
            algorithm_state_slice=algorithm_state_slice_value,
            loop_buffer_end_offset=acceptance_flag_slice_value.stop,
            total_buffer_end_offset=algorithm_state_slice_value.stop,
            all=slice(None),
        )

    @property
    def loop_element_count(self) -> int:
        """Return the loop's intrinsic persistent local requirement."""
        return int(self.loop_buffer_end_offset or 0)

@define
class LoopSharedIndices:
    """Slice container describing shared-memory buffer layouts.

    Attributes
    ----------
    state
        Slice covering the primary state buffer.
    proposed_state
        Slice covering the proposed state buffer.
    observables
        Slice covering observable work buffers.
    proposed_observables
        Slice covering the proposed observable buffer.
    parameters
        Slice covering parameter storage.
    drivers
        Slice covering driver storage.
    proposed_drivers
        Slice covering the proposed driver storage.
    state_summaries
        Slice covering aggregated state summaries.
    observable_summaries
        Slice covering aggregated observable summaries.
    error
        Slice covering the shared error buffer reused by adaptive algorithms.
    counters
        Slice covering the iteration counters buffer.
    proposed_counters
        Slice covering the proposed iteration counters buffer.
    local_end
        Offset of the end of loop-managed shared memory.
    scratch
        Slice covering any remaining shared-memory scratch space.
    all
        Slice that spans the full shared-memory buffer.
    """

    state_buffer_slice:  Optional[slice] = field(
            default=None,
            validator=valid_opt_slice
    )
    proposed_state_buffer_slice: Optional[slice] = field(
            default=None,
            validator=valid_opt_slice
    )
    observables_buffer_slice: Optional[slice] = field(
            default=None,
            validator=valid_opt_slice
    )
    proposed_observables_buffer_slice: Optional[slice] = field(
            default=None,
            validator=valid_opt_slice
    )
    parameters_buffer_slice: Optional[slice] = field(
            default=None,
            validator=valid_opt_slice
    )
    driver_buffer_slice: Optional[slice] = field(
            default=None,
            validator=valid_opt_slice
    )
    proposed_driver_buffer_slice: Optional[slice] = field(
            default=None,
            validator=valid_opt_slice
    )
    state_summary_buffer_slice: Optional[slice] = field(
            default=None,
            validator=valid_opt_slice
    )
    observable_summary_buffer_slice: Optional[slice] = field(
            default=None,
            validator=valid_opt_slice
    )
    error_buffer_slice: Optional[slice] = field(
            default=None,
            validator=valid_opt_slice
    )
    counter_buffer_slice: Optional[slice] = field(
            default=None,
            validator=valid_opt_slice
    )
    proposed_counter_buffer_slice: Optional[slice] = field(
            default=None,
            validator=valid_opt_slice
    )
    loop_buffer_end_offset: Optional[int] = field(
            default=None,
            validator=opt_getype_validator(int, 0)
    )
    scratch_buffer_slice: Optional[slice] = field(
            default=None,
            validator=valid_opt_slice
    )
    full_buffer_slice: Optional[slice] = field(
            default=None,
            validator=valid_opt_slice
    )

    def load_indices_from_mapping(
        self,
        indices_dict: MutableMapping[str, Union[slice, tuple[int, ...]]],
    ) -> None:
        """Load indices from a mapping.

        Parameters
        ----------
        indices_dict
            Mapping of attribute names to slices or ``slice`` constructor
            arguments.
        """
        for key, value in indices_dict.items():
            if isinstance(value, slice):
                setattr(self, key, value)
            else:
                setattr(self, key, slice(*value))

        self.local_end = self.scratch.stop

    @classmethod
    def from_sizes(cls,
                   n_states: int,
                   n_observables: int,
                   n_parameters: int,
                   n_drivers: int,
                   state_summaries_buffer_height: int,
                   observable_summaries_buffer_height: int,
                   n_error: int = 0,
                   save_counters: bool = False,
                   ) -> "LoopSharedIndices":
        """Build index slices from component sizes.

        Parameters
        ----------
        n_states
            Number of state elements.
        n_observables
            Number of observable elements.
        n_parameters
            Number of parameter elements.
        n_drivers
            Number of driver elements.
        state_summaries_buffer_height
            Number of state summary buffer elements.
        observable_summaries_buffer_height
            Number of observable summary buffer elements.
        save_counters
            Whether to allocate space for iteration counters.

        Returns
        -------
        LoopSharedIndices
            Layout sized to cover the requested shared-memory partitions.
        """

        state_start_idx = 0
        state_proposal_start_idx = state_start_idx + n_states
        observables_start_index = state_proposal_start_idx + n_states
        observables_proposal_start_idx = (
            observables_start_index + n_observables
        )
        parameters_start_index = (
            observables_proposal_start_idx + n_observables
        )
        drivers_start_index = parameters_start_index + n_parameters
        drivers_proposal_start_idx = drivers_start_index + n_drivers
        state_summ_start_index = drivers_proposal_start_idx + n_drivers
        obs_summ_start_index = (
            state_summ_start_index + state_summaries_buffer_height
        )
        error_start_index = (
            obs_summ_start_index + observable_summaries_buffer_height
        )
        error_stop_index = error_start_index + n_error
        
        # Add counters if enabled
        # counters_since_save has 4 elements: newton, krylov, steps, rejections
        # proposed_counters has 2 elements: newton, krylov (from step function)
        counters_since_save_size = 4 if save_counters else 0
        proposed_counters_size = 2 if save_counters else 0
        counters_start_index = error_stop_index
        counters_stop_index = counters_start_index + counters_since_save_size
        proposed_counters_start_index = counters_stop_index
        proposed_counters_stop_index = proposed_counters_start_index + proposed_counters_size
        
        final_stop_index = proposed_counters_stop_index

        return cls(
            state_buffer_slice=slice(state_start_idx, state_proposal_start_idx),
            proposed_state_buffer_slice=slice(state_proposal_start_idx, observables_start_index),
            observables_buffer_slice=slice(
                observables_start_index, observables_proposal_start_idx
            ),
            proposed_observables_buffer_slice=slice(
                observables_proposal_start_idx, parameters_start_index
            ),
            parameters_buffer_slice=slice(parameters_start_index, drivers_start_index),
            driver_buffer_slice=slice(drivers_start_index, drivers_proposal_start_idx),
            proposed_driver_buffer_slice=slice(
                drivers_proposal_start_idx, state_summ_start_index
            ),
            state_summary_buffer_slice=slice(state_summ_start_index, obs_summ_start_index),
            observable_summary_buffer_slice=slice(obs_summ_start_index, error_start_index),
            error_buffer_slice=slice(error_start_index, error_stop_index),
            counter_buffer_slice=slice(counters_start_index, counters_stop_index),
            proposed_counter_buffer_slice=slice(proposed_counters_start_index, proposed_counters_stop_index),
            loop_buffer_end_offset=final_stop_index,
            scratch_buffer_slice=slice(final_stop_index, None),
            full_buffer_slice=slice(None),
        )


    @property
    def loop_shared_element_count(self) -> int:
        """Return the number of shared memory elements."""
        return int(self.loop_buffer_end_offset or 0)

    @property
    def state_count(self) -> int:
        """Return the number of states."""
        return int(self.state_buffer_slice.stop - self.state_buffer_slice.start)

    @property
    def parameter_count(self) -> int:
        """Return the number of parameters."""
        return int(self.parameters_buffer_slice.stop - self.parameters_buffer_slice.start)

    @property
    def driver_count(self) -> int:
        """Return the number of drivers."""
        return int(self.driver_buffer_slice.stop - self.driver_buffer_slice.start)

    @property
    def observable_count(self) -> int:
        """Return the number of observables."""
        return int(self.observables_buffer_slice.stop - self.observables_buffer_slice.start)

    @property
    def counter_count(self) -> int:
        """Return the number of counter elements (4 if enabled, 0 if not)."""
        return int(self.counter_buffer_slice.stop - self.counter_buffer_slice.start)


@define
class ODELoopConfig:
    """Compile-critical settings for an integrator loop.

    Attributes
    ----------
    shared_buffer_indices
        Shared-memory layout describing scratch buffers and outputs.
    local_indices
        Persistent local-memory layout describing private buffers.
    precision
        Precision used for all loop-managed computations.
    compile_flags
        Output configuration governing save and summary cadence.
    _dt_save
        Interval between accepted saves.
    _dt_summarise
        Interval between summary accumulations.
    state_saving_function
        Device function that records state and observable snapshots.
    summary_update_function
        Device function that accumulates summary statistics.
    summary_saving_function
        Device function that writes summary statistics to output buffers.
    step_controller_fn
        Device function that updates the timestep and acceptance flag.
    step_function
        Device function that advances the solution by one tentative step.
    driver_function
        Device function that evaluates driver signals for a given time.
    observables_function
        Device function that evaluates observables for the current state.
    _dt0
        Initial timestep prior to controller feedback.
    _dt_min
        Minimum allowable timestep.
    _dt_max
        Maximum allowable timestep.
    is_adaptive
        Whether the loop operates with an adaptive controller.
    """

    shared_buffer_indices: LoopSharedIndices = field(
        validator=validators.instance_of(LoopSharedIndices)
    )
    local_indices: LoopLocalIndices = field(
        validator=validators.instance_of(LoopLocalIndices)
    )

    precision: PrecisionDType = field(
        default=float32,
        converter=precision_converter,
        validator=precision_validator,
    )
    compile_flags: OutputCompileFlags = field(
        default=OutputCompileFlags(),
        validator=validators.instance_of(OutputCompileFlags),
    )
    _dt_save: float = field(
        default=0.1,
        validator=gttype_validator(float, 0)
    )
    _dt_summarise: float = field(
        default=1.0,
        validator=gttype_validator(float, 0)
    )
    state_saving_function: Optional[Callable] = field(
        default=None,
        validator=validators.optional(is_device_validator)
    )
    summary_update_function: Optional[Callable] = field(
        default=None,
        validator=validators.optional(is_device_validator)
    )
    summary_saving_function: Optional[Callable] = field(
        default=None,
        validator=validators.optional(is_device_validator)
    )
    step_controller_fn: Optional[Callable] = field(
        default=None,
        validator=validators.optional(is_device_validator)
    )
    step_function: Optional[Callable] = field(
        default=None,
        validator=validators.optional(is_device_validator)
    )
    driver_function: Optional[Callable] = field(
        default=None,
        validator=validators.optional(is_device_validator)
    )
    observables_function: Optional[Callable] = field(
        default=None,
        validator=validators.optional(is_device_validator)
    )
    _dt0: Optional[float] = field(
        default=0.01,
        validator=opt_gttype_validator(float, 0),
    )
    _dt_min: Optional[float] = field(
        default=0.01,
        validator=opt_gttype_validator(float, 0),
    )
    _dt_max: Optional[float] = field(
        default=0.1,
        validator=opt_gttype_validator(float, 0),
    )
    is_adaptive: Optional[bool] = field(
            default=False,
            validator=validators.optional(validators.instance_of(bool)))

    @property
    def save_count_per_summary(self) -> int:
        """Return the number of saves between summary outputs."""
        return int(self.dt_summarise // self.dt_save)

    @property
    def numba_precision(self) -> type:
        """Return the Numba precision type."""
        return numba.from_dtype(self.precision)

    @property
    def simsafe_precision(self) -> type:
        """Return the simulator safe precision."""
        return simsafe_dtype(self.precision)

    @property
    def dt_save(self) -> float:
        """Return the output save interval."""
        return self.precision(self._dt_save)

    @property
    def dt_summarise(self) -> float:
        """Return the summary interval."""
        return self.precision(self._dt_summarise)

    @property
    def dt0(self) -> float:
        """Return the initial timestep."""
        return self.precision(self._dt0)

    @property
    def dt_min(self) -> float:
        """Return the minimum allowable timestep."""
        return self.precision(self._dt_min)

    @property
    def dt_max(self) -> float:
        """Return the maximum allowable timestep."""
        return self.precision(self._dt_max)

    @property
    def loop_shared_element_count(self) -> int:
        """Return the loop's shared-memory contribution."""

        local_end = getattr(self.shared_buffer_indices, "local_end", None)
        return int(local_end or 0)

    @property
    def loop_local_element_count(self) -> int:
        """Return the loop's persistent local-memory contribution."""

        return self.local_indices.loop_element_count

