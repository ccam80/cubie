"""Configuration utilities for the batch solver.

This module provides :class:`BatchSolverConfig`, a small container that holds
settings used when compiling and running the CUDA integration kernel.
"""

import attrs
import numba
from numpy import float32

from cubie._utils import getype_validator
from cubie.cuda_simsafe import from_dtype as simsafe_dtype


@attrs.define
class BatchSolverConfig:
    """Settings for configuring a batch solver kernel.

    Attributes
    ----------
    precision : type, optional
        Data type used for computation. Defaults to ``float32``.
    algorithm : str, optional
        Name of the integration algorithm. Defaults to ``'euler'``.
    duration : float, optional
        Total integration duration in seconds. Defaults to ``1.0``.
    warmup : float, optional
        Length of the warm-up period before outputs are stored. Defaults to
        ``0.0``.
    stream : int, optional
        Identifier for the CUDA stream to execute on. ``None`` defaults to the
        solver's stream. Defaults to ``0``.
    profileCUDA : bool, optional
        If ``True`` CUDA profiling is enabled. Defaults to ``False``.
    """

    precision: type = attrs.field(
        default=float32, validator=attrs.validators.instance_of(type)
    )
    _duration: float = attrs.field(
        default=1.0, validator=getype_validator(float, 0)
    )
    _warmup: float = attrs.field(
        default=0.0, validator=getype_validator(float, 0)
    )
    stream: int = attrs.field(
        default=0,
        validator=attrs.validators.optional(
            attrs.validators.instance_of(
                int,
            ),
        ),
    )
    profileCUDA: bool = False

    @property
    def numba_precision(self) -> type:
        """Returns numba precision type."""
        return numba.from_dtype(self.precision)

    @property
    def simsafe_precision(self) -> type:
        """Returns simulator safe precision."""
        return simsafe_dtype(self.precision)

    @property
    def duration(self) -> float:
        """Returns integration duration."""
        return self.precision(self._duration)

    @property
    def warmup(self) -> float:
        """Returns warm-up duration."""
        return self.precision(self._warmup)
