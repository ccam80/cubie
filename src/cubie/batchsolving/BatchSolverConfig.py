"""Configuration utilities for the batch solver.

This module provides :class:`BatchSolverConfig`, a small container that holds
settings used when compiling and running the CUDA integration kernel.
"""

import attrs
from numpy import float32


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
    algorithm: str = "euler"
    duration: float = 1.0
    warmup: float = 0.0
    stream: int = attrs.field(
        default=0,
        validator=attrs.validators.optional(
            attrs.validators.instance_of(
                int,
            ),
        ),
    )
    profileCUDA: bool = False
