"""
Integrator configuration management with validation and adapter patterns.

This module provides the IntegratorLoopSettings class for managing compile-critical
settings for integrator loops, including timing parameters, buffer sizes, and
function references. It uses validation and adapter patterns to ensure
configuration consistency.
"""
from abc import abstractmethod
from typing import Optional

from attrs import define, field, validators
from numpy import float32, float16, float64

from cubie.outputhandling.output_config import OutputCompileFlags

valid_opt_slice = validators.optional(validators.instance_of(slice))
valid_float = validators.instance_of(float)
valid_int = validators.instance_of(int)

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
    end: Optional[slice] = field(
            default=None,
            validator=valid_opt_slice
    )

    def from_dict(self, indices_dict):
        for key, value in indices_dict.items():
            if isinstance(value, slice):
                setattr(self, key, value)
            else:
                setattr(self, key, slice(*value))

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
    # Bury this in step
    # buffer_sizes: LoopBufferSizes = field(
    #     validator=validators.instance_of(LoopBufferSizes)
    # )
    precision: type = field(
        default=float32,
        validator=validators.and_(
            validators.instance_of(type),
            validators.in_(
                [float32, float64, float16],
            ),
        ),
    )
    compile_flags: OutputCompileFlags = field(
        default=OutputCompileFlags(),
        validator=validators.instance_of(
            OutputCompileFlags,
        ),
    )
    shared_memory_indices: LoopIndices = field(
            factory=LoopIndices,
            validator=validators.instance_of(LoopIndices)
    )
    constant_memory_indices: LoopIndices = field(
            factory=LoopIndices,
            validator=validators.instance_of(LoopIndices)
    )
    local_memory_indices: LoopIndices = field(
            factory=LoopIndices,
            validator=validators.instance_of(LoopIndices)
    )

    @abstractmethod
    def _validate_timing(self):
        raise NotImplementedError

    @abstractmethod
    @classmethod
    def from_integrator_run(cls, run_object):
        """
        Create an IntegratorLoopSettings instance from a SingleIntegratorRun object.

        Parameters
        ----------
        run_object : SingleIntegratorRun
            The SingleIntegratorRun object containing configuration parameters.

        Returns
        -------
        IntegratorLoopSettings
            New instance configured with parameters from the run object.
        """
        raise NotImplementedError