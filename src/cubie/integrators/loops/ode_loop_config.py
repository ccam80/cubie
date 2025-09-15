"""
Integrator configuration management with validation and adapter patterns.

This module provides the IntegratorLoopSettings class for managing compile-critical
settings for integrator loops, including timing parameters, buffer sizes, and
function references. It uses validation and adapter patterns to ensure
configuration consistency.
"""
from typing import Optional, Callable

from attrs import define, field, validators
from numpy import float32, float16, float64

from cubie import is_device_validator
from cubie.outputhandling.output_config import OutputCompileFlags
from cubie.outputhandling.output_sizes import LoopBufferSizes

valid_opt_slice = validators.optional(validators.instance_of(slice))
valid_float = validators.instance_of(float)
valid_int = validators.instance_of(int)
valid_bool = validators.instance_of(bool)
valid_callable = validators.is_callable

@define
class LoopIndices:
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
    # Alias this to state indices for shared mem in fixed-step mode
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
            validator=validators.optional(valid_int)
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
    def from_buffer_sizes(cls,
                          sizes: LoopBufferSizes,
                          step_type: str="adaptive"
                          ):
        state_start_idx = 0
        #Alias state proposal to state for always-accepted fixed steps
        if step_type == "adaptive":
            state_end_idx = state_start_idx + sizes.state
            state_proposal_start_idx = state_end_idx
        else:
            state_end_idx = state_start_idx + sizes.state
            state_proposal_start_idx = state_start_idx

        dxdt_start_index = state_proposal_start_idx + sizes.state
        observables_start_index = dxdt_start_index + sizes.dxdt
        parameters_start_index = observables_start_index + sizes.observables
        drivers_start_index = parameters_start_index + sizes.parameters
        state_summaries_start_index = drivers_start_index + sizes.drivers
        obs_summaries_start_index = (state_summaries_start_index
                                     + sizes.state_summaries)
        end_index = obs_summaries_start_index + sizes.observable_summaries

        return cls(
            state=slice(state_start_idx, state_end_idx),
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
    buffer_sizes: LoopBufferSizes = field(
        validator=validators.instance_of(LoopBufferSizes)
    )
    buffer_indices: LoopIndices = field(
        validator=validators.instance_of(LoopIndices)
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
    # constant_memory_indices: LoopIndices = field(
    #     factory=LoopIndices, validator=validators.instance_of(LoopIndices)
    # )
    # local_memory_indices: LoopIndices = field(
    #     factory=LoopIndices, validator=validators.instance_of(LoopIndices)
    # )
    dt_save: float = field(default=0.1, validator=valid_float)
    dt_summarise: float = field(default=1.0, validator=valid_float)


    @property
    def saves_per_summary(self) -> int:
        """Return the number of saves between summary outputs."""
        return int(self.dt_summarise // self.dt_save)



    # @classmethod
    # def from_single_integrator_run(cls, run_object):
    #     """Create configuration from a :class:`SingleIntegratorRun`.
    #
    #     Parameters
    #     ----------
    #     run_object : SingleIntegratorRun
    #         Integration run supplying configuration values.
    #
    #     Returns
    #     -------
    #     ODELoopConfig
    #         Configuration populated with values from ``run_object``.
    #     """
    #     return cls(
    #         loop_step_config=run_object.loop_step_config,
    #         buffer_sizes=run_object.loop_buffer_sizes,
    #         precision=run_object.precision,
    #         compile_flags=run_object.compile_flags,
    #         save_state_func=run_object.save_state_func,
    #         update_summaries_func=run_object.update_summaries_func,
    #         save_summaries_func=run_object.save_summaries_func,
    #         step_controller_fn=None,  # TODO: supply controller function
    #         step_fn=run_object._integrator_instance.device_function,
    #         is_fixed_step=False,  # TODO: set based on algorithm
    #         dt0=run_object.fixed_step_size,  # TODO: confirm dt0 source
    #         saves_per_summary=1,  # TODO: derive from loop_step_config
    #         total_saved_samples=0,  # TODO: derive from output sizes
    #     )
