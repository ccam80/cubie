"""Runtime configuration settings for numerical integration algorithms.

This module provides :class:`IntegratorRunSettings`, an attrs-based
container that centralises precision, algorithm, and controller
configuration for the CUDA IVP loop orchestrators.
"""

import attrs
import numba

from cubie.CUDAFactory import CUDAFactoryConfig


@attrs.define
class IntegratorRunSettings(CUDAFactoryConfig):
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

    algorithm: str = attrs.field(
        default="euler",
        validator=attrs.validators.instance_of(str),
    )
    step_controller: str = attrs.field(
        default="fixed",
        validator=attrs.validators.instance_of(str),
    )


