"""Configuration utilities for the batch solver.

This module provides :class:`BatchSolverConfig`, a small container that holds
settings used when compiling and running the CUDA integration kernel.
"""
from typing import Optional, Callable

import attrs
import numba
from numpy import float32

from cubie._utils import (
    PrecisionDType,
    getype_validator,
    precision_converter,
    precision_validator,
    is_device_validator
)
from cubie.batchsolving.arrays.BatchOutputArrays import ActiveOutputs
from cubie.cuda_simsafe import from_dtype as simsafe_dtype


@attrs.define
class BatchSolverConfig:
    """Compile-critical settings for the batch solver kernel.

    Attributes
    ----------
    precision
        Data type used for computation. Defaults to ``float32``.
    loop_fn
        The loop function constructed by singleintegratorrun. This is
        compiled into the kernel.
    local_memory_elements
        size of the local memory buffer, allocated at build time.
    shared_memory_elements
        size of the shared memory buffer, allocated at build time
    ActiveOutputs
        Which array outputs are active/not.

    """

    precision: PrecisionDType = attrs.field(
        default=float32,
        converter=precision_converter,
        validator=precision_validator,
    )
    loop_fn: Optional[Callable] = attrs.field(
            default=None,
            validator=attrs.validators.optional(is_device_validator)
    )
    local_memory_elements: int = attrs.field(
            default=0,
            validator=getype_validator(int, 0)
    )
    shared_memory_elements: int = attrs.field(
            default=0,
            validator=getype_validator(int, 0)
    )
    ActiveOutputs: ActiveOutputs = attrs.field(
            factory=ActiveOutputs,
            validator=attrs.validators.instance_of(ActiveOutputs)
    )

    @property
    def numba_precision(self) -> type:
        """Returns numba precision type."""
        return numba.from_dtype(self.precision)

    @property
    def simsafe_precision(self) -> type:
        """Returns simulator safe precision."""
        return simsafe_dtype(self.precision)
