"""
Integrator configuration management with validation and adapter patterns.

This module provides the IntegratorLoopSettings class for managing compile-critical
settings for integrator loops, including timing parameters, buffer sizes, and
function references. It uses validation and adapter patterns to ensure
configuration consistency.
"""
from typing import Optional, Callable

from attrs import define, field, validators
import numba
from numpy import float32, float16, float64

from cubie._utils import (is_device_validator, getype_validator,
                          gttype_validator)
from cubie.cudasim_utils import from_dtype as simsafe_dtype
from cubie.outputhandling.output_config import OutputCompileFlags
from cubie.outputhandling.output_sizes import LoopBufferSizes

valid_opt_slice = validators.optional(validators.instance_of(slice))

@define
class LoopLocalIndices:
    """Index slices for persistent local memory used by the loop."""

    dt: Optional[slice] = field(default=None, validator=valid_opt_slice)
    accept: Optional[slice] = field(default=None, validator=valid_opt_slice)
    error: Optional[slice] = field(default=None, validator=valid_opt_slice)
    controller: Optional[slice] = field(
        default=None, validator=valid_opt_slice
    )
    algorithm: Optional[slice] = field(default=None, validator=valid_opt_slice)
    loop_end: Optional[int] = field(
        default=None, validator=validators.optional(getype_validator(int, 0))
    )
    total_end: Optional[int] = field(
        default=None, validator=validators.optional(getype_validator(int, 0))
    )
    all: Optional[slice] = field(default=None, validator=valid_opt_slice)

    @classmethod
    def empty(cls) -> "LoopLocalIndices":
        """Return an empty local-memory layout."""

        zero = slice(0, 0)
        return cls(
            dt=zero,
            error_integral=zero,
            accept=zero,
            error=zero,
            controller=zero,
            algorithm=zero,
            loop_end=0,
            total_end=0,
            all=slice(None),
        )

    @classmethod
    def from_sizes(
        cls, n_states: int, controller_len: int, algorithm_len: int
    ) -> "LoopLocalIndices":
        """Build index slices from component memory requirements."""

        n_states = max(int(n_states), 0)
        controller_len = max(int(controller_len), 0)
        algorithm_len = max(int(algorithm_len), 0)

        dt_slice = slice(0, 1)
        accept_slice = slice(1, 2)
        error_start = 2
        error_stop = error_start + n_states
        error_slice = slice(error_start, error_stop)

        controller_start = error_stop
        controller_stop = controller_start + controller_len
        controller_slice = slice(controller_start, controller_stop)

        algorithm_start = controller_stop
        algorithm_stop = algorithm_start + algorithm_len
        algorithm_slice = slice(algorithm_start, algorithm_stop)

        return cls(
            dt=dt_slice,
            accept=accept_slice,
            error=error_slice,
            controller=controller_slice,
            algorithm=algorithm_slice,
            loop_end=error_slice.stop,
            total_end=algorithm_slice.stop,
            all=slice(None),
        )

    @property
    def loop_elements(self) -> int:
        """Return the loop's intrinsic persistent local requirement."""
        return int(self.loop_end or 0)

    # @property
    # def total_elements(self) -> int:
    #     """Return the total persistent local requirement including children."""
    #
    #     return int(self.total_end or 0)
    #
    # @property
    # def controller_elements(self) -> int:
    #     """Return the controller contribution to persistent local memory."""
    #
    #     if self.controller is None:
    #         return 0
    #     return int(self.controller.stop - self.controller.start)
    #
    # @property
    # def algorithm_elements(self) -> int:
    #     """Return the algorithm contribution to persistent local memory."""
    #
    #     if self.algorithm is None:
    #         return 0
    #     return int(self.algorithm.stop - self.algorithm.start)

@define
class LoopSharedIndices:
    """General container for array indices used in integrator loops. Each
    attribute is a slice, for indexing arrays directly"""

    dxdt: Optional[slice] = field(
            default=None,
            validator=valid_opt_slice
    )
    state:  Optional[slice] = field(
            default=None,
            validator=valid_opt_slice
    )
    proposed_state: Optional[slice] = field(
            default=None,
            validator=valid_opt_slice
    )
    observables: Optional[slice] = field(
            default=None,
            validator=valid_opt_slice
    )
    parameters: Optional[slice] = field(
            default=None,
            validator=valid_opt_slice
    )
    drivers: Optional[slice] = field(
            default=None,
            validator=valid_opt_slice
    )
    state_summaries: Optional[slice] = field(
            default=None,
            validator=valid_opt_slice
    )
    observable_summaries: Optional[slice] = field(
            default=None,
            validator=valid_opt_slice
    )
    local_end: Optional[int] = field(
            default=None,
            validator=validators.optional(getype_validator(int, 0))
    )
    scratch: Optional[slice] = field(
            default=None,
            validator=valid_opt_slice
    )
    all: Optional[slice] = field(
            default=None,
            validator=valid_opt_slice
    )

    def from_dict(self, indices_dict):
        for key, value in indices_dict.items():
            if isinstance(value, slice):
                setattr(self, key, value)
            else:
                setattr(self, key, slice(*value))

        self.local_end = self.scratch.stop

    @classmethod
    def from_buffer_sizes(cls, sizes: LoopBufferSizes) -> "LoopSharedIndices":
        state_start_idx = 0
        state_proposal_start_idx = state_start_idx + sizes.state
        dxdt_start_index = state_proposal_start_idx + sizes.state
        observables_start_index = dxdt_start_index + sizes.dxdt
        parameters_start_index = observables_start_index + sizes.observables
        drivers_start_index = parameters_start_index + sizes.parameters
        state_summaries_start_index = drivers_start_index + sizes.drivers
        obs_summaries_start_index = (state_summaries_start_index
                                     + sizes.state_summaries)
        end_index = obs_summaries_start_index + sizes.observable_summaries

        return cls(
            state=slice(state_start_idx, state_proposal_start_idx),
            proposed_state=slice(state_proposal_start_idx, dxdt_start_index),
            dxdt=slice(dxdt_start_index, observables_start_index),
            observables=slice(observables_start_index, parameters_start_index),
            parameters=slice(parameters_start_index, drivers_start_index),
            drivers=slice(drivers_start_index, state_summaries_start_index),
            state_summaries=slice(state_summaries_start_index,
                                  obs_summaries_start_index),
            observable_summaries=slice(obs_summaries_start_index, end_index),
            local_end=end_index,
            scratch=slice(end_index, None),
            all=slice(None),
        )

    @property
    def loop_shared_elements(self) -> int:
        """Return the number of shared memory elements."""
        return int(self.local_end or 0)

    @property
    def n_states(self) -> int:
        """Return the number of states."""
        return int(self.state.stop - self.state.start)

    @property
    def n_parameters(self) -> int:
        """Return the number of proposed states."""
        return int(self.parameters.stop - self.parameters.start)

    @property
    def n_drivers(self) -> int:
        """Return the number of drivers."""
        return int(self.drivers.stop - self.drivers.start)


@define
class ODELoopConfig:
    """
    Compile-critical settings for an integrator loop.

    This class manages configuration settings that are critical for compiling
    integrator loops, including timing parameters, buffer sizes, precision,
    and function references. The integrator loop is not the source of truth
    for these settings, so minimal setters are provided. Instead, there are
    update_from methods which extract relevant settings from other objects.

    """

    shared_buffer_indices: LoopSharedIndices = field(
        validator=validators.instance_of(LoopSharedIndices)
    )
    local_indices: LoopLocalIndices = field(
            validator=validators.instance_of(LoopLocalIndices)
    )
    save_state_func: Callable = field(validator=is_device_validator)
    update_summaries_func: Callable = field(validator=is_device_validator)
    save_summaries_func: Callable = field(validator=is_device_validator)
    step_controller_fn: Callable = field(validator=is_device_validator)
    step_fn: Callable = field(validator=is_device_validator)

    precision: type = field(
        default=float32,
        validator=validators.in_([float32, float64, float16]),
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
    _dt0: float = field(
        default=0.01,
        validator=gttype_validator(float, 0),
    )
    _dt_min: float = field(
        default=0.01,
        validator=gttype_validator(float, 0),
    )
    _dt_max: float = field(
        default=0.1,
        validator=gttype_validator(float, 0),
    )
    is_adaptive: bool = field(default=False, validator=validators.instance_of(bool))

    @property
    def saves_per_summary(self) -> int:
        """Return the number of saves between summary outputs."""
        return int(self.dt_summarise // self.dt_save)

    @property
    def numba_precision(self) -> type:
        """Returns numba precision type."""
        return numba.from_dtype(self.precision)

    @property
    def simsafe_precision(self) -> type:
        """Returns simulator safe precision."""
        return simsafe_dtype(self.precision)

    @property
    def dt_save(self) -> float:
        """Returns output save interval."""
        return self.precision(self._dt_save)

    @property
    def dt_summarise(self) -> float:
        """Returns summary interval."""
        return self.precision(self._dt_summarise)

    @property
    def dt0(self) -> float:
        """Return the initial timestep."""
        return float(self._dt0)

    @property
    def dt_min(self) -> float:
        """Return the minimum allowable timestep."""
        return float(self._dt_min)

    @property
    def dt_max(self) -> float:
        """Return the maximum allowable timestep."""
        return float(self._dt_max)

    @property
    def loop_shared_elements(self) -> int:
        """Return the loop's shared-memory contribution."""

        local_end = getattr(self.shared_buffer_indices, "local_end", None)
        return int(local_end or 0)

    @property
    def loop_local_elements(self) -> int:
        """Return the loop's persistent local-memory contribution."""

        return self.local_indices.loop_elements

