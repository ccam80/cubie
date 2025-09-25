"""Runtime configuration settings for numerical integration algorithms.

This module provides the :class:`IntegratorRunSettings` class which manages
timing, tolerance, and controller configuration for ODE integration runs.
It performs light dependency injection by instantiating algorithm step
objects and step-size controllers used by the modular IVP loop.
"""

import attrs
import numba
from numpy import float32

from cubie._utils import precision_converter, precision_validator


@attrs.define
class IntegratorRunSettings:
    """Container for runtime/timing settings grouped for IVP loops.

    Parameters
    ----------
    precision
        Numerical precision used for timing comparisons.
    algorithm
        Name of the integration step algorithm.
    step_controller_kind
        Name of the step-size controller.
    """

    precision: type = attrs.field(
        default=float32,
        converter=precision_converter,
        validator=precision_validator,
    )
    algorithm: str = attrs.field(
        default="euler",
        validator=attrs.validators.instance_of(str),
    )
    step_controller_kind: str = attrs.field(
        default="fixed",
        validator=attrs.validators.instance_of(str),
    )

    @property
    def numba_precision(self) -> type:
        """Return the Numba-compatible precision."""

        return numba.from_dtype(self.precision)
