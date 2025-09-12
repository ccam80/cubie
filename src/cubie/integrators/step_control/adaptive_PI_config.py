from typing import Optional
from warnings import warn

from attrs import define, field, validators
from numpy import float32, float64, float16
from cubie.integrators.step_control.adaptive_step_config import \
    AdaptiveStepControlConfig

@define
class PIStepControlConfig(AdaptiveStepControlConfig):
    """
    Configuration for an adaptive step size controller using a simplified PI
    algorithm. More efficient than the traditional I controller used in
    non-stiff systems.
    """
    kp: float = field(
        default=0.075,
        validator=validators.instance_of(float)
    )
    ki: float = field(
        default=0.175,
        validator=validators.instance_of(float)
    )
