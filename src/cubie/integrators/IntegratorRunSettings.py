"""Runtime configuration settings for numerical integration algorithms.

This module provides :class:`IntegratorRunSettings`, an attrs-based
container that centralises precision, algorithm, and controller
configuration for the CUDA IVP loop orchestrators.
"""

import attrs
import numba
from numpy import float32

from cubie._utils import PrecisionDType, precision_converter, precision_validator


@attrs.define
class IntegratorRunSettings:
    """Container for runtime and controller settings used by IVP loops.

    Attributes
    ----------
    precision
        Numerical precision used for timing comparisons.
    algorithm
        Name of the integration step algorithm.
    step_controller
        Name of the step-size controller.
    """

    precision: PrecisionDType = attrs.field(
        default=float32,
        converter=precision_converter,
        validator=precision_validator,
    )
    algorithm: str = attrs.field(
        default="euler",
        validator=attrs.validators.instance_of(str),
    )
    step_controller: str = attrs.field(
        default="fixed",
        validator=attrs.validators.instance_of(str),
    )

    @property
    def numba_precision(self) -> type:
        """Return the Numba-compatible precision."""

        return numba.from_dtype(self.precision)
