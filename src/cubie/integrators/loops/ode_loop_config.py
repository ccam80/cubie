"""
Integrator configuration management with validation and adapter patterns.

This module provides the IntegratorLoopSettings class for managing compile-critical
settings for integrator loops, including timing parameters, buffer sizes, and
function references. It uses validation and adapter patterns to ensure
configuration consistency.
"""
from math import ceil
from typing import Optional, Callable

from attrs import define, field, validators
from numpy import float32, float16, float64

from cubie import is_device_validator
from cubie.integrators.algorithms.LoopStepConfig import LoopStepConfig
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
        stops = [value.stop for value in self.__dict__.values()
                 if value is not None]
        self.end = slice(max(stops))


@define
class IVPLoopConfig:
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
    shared_memory_indices: LoopIndices = field(
        factory=LoopIndices, validator=validators.instance_of(LoopIndices)
    )
    # constant_memory_indices: LoopIndices = field(
    #     factory=LoopIndices, validator=validators.instance_of(LoopIndices)
    # )
    # local_memory_indices: LoopIndices = field(
    #     factory=LoopIndices, validator=validators.instance_of(LoopIndices)
    # )
    dt0: float = field(default=1e-3, validator=valid_float)
    dt_save: float = field(default=0.1, validator=valid_float)
    dt_summarise: float = field(default=1.0, validator=valid_float)
    duration: float = field(default=1.0, validator=valid_float)
    settling_time: float = field(default=0.0, validator=valid_float)
    total_saved_samples: int = field(default=0, validator=valid_int)
    # saves_per_summary: int = field(default=1, validator=valid_int)

    @property
    def output_length(self) -> int:
        """Return the length of the output array."""
        return ceil(self.duration / self.dt_save)

    @property
    def saves_per_summary(self) -> int:
        """Return the number of saves between summary outputs."""
        return int(self.dt_summarise // self.dt_save)

    def _validate_timing(self, dt_min, dt_max) -> bool:
        """TODO: Add timing validation."""
        # TODO: implement validation logic
        True

    @classmethod
    def from_single_integrator_run(cls, run_object):
        """Create configuration from a :class:`SingleIntegratorRun`.

        Parameters
        ----------
        run_object : SingleIntegratorRun
            Integration run supplying configuration values.

        Returns
        -------
        IVPLoopConfig
            Configuration populated with values from ``run_object``.
        """
        return cls(
            loop_step_config=run_object.loop_step_config,
            buffer_sizes=run_object.loop_buffer_sizes,
            precision=run_object.precision,
            compile_flags=run_object.compile_flags,
            save_state_func=run_object.save_state_func,
            update_summaries_func=run_object.update_summaries_func,
            save_summaries_func=run_object.save_summaries_func,
            step_controller_fn=None,  # TODO: supply controller function
            step_fn=run_object._integrator_instance.device_function,
            is_fixed_step=False,  # TODO: set based on algorithm
            dt0=run_object.fixed_step_size,  # TODO: confirm dt0 source
            saves_per_summary=1,  # TODO: derive from loop_step_config
            total_saved_samples=0,  # TODO: derive from output sizes
        )
