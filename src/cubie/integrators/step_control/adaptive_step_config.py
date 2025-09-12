"""
Integrator configuration management with validation and adapter patterns.

This module provides the IntegratorLoopSettings class for managing compile-critical
settings for integrator loops, including timing parameters, buffer sizes, and
function references. It uses validation and adapter patterns to ensure
configuration consistency.
"""
from typing import Optional
from warnings import warn

import numpy as np
from attr import validators
from attrs import define, field

from cubie._utils import inrangetype_validator, getype_validator
from cubie.integrators.step_control.base_step_controller_config import \
    BaseStepControllerConfig


@define
class AdaptiveStepControlConfig(BaseStepControllerConfig):
    """
    Configuration parameters for a general adaptive step controller.

    This class can be used as-is for simple adaptive step controllers,
    or subclassed for more complex controllers. BaseStepControllerConfig
    defines the interface with the integrator loop; This class provides
    default logic for API properties.

    Any parameters required for subclassed controllers should live in here
    if they are used as compile-time constants - this will trigger the
    device function to recompile when they change.
    """
    _dt_min: float = field(
            default=1e-6,
            validator=getype_validator(float, 0)
    )
    _dt_max: Optional[float] = field(
            default=None,
            validator=getype_validator(float, 0)
    )
    atol: np.ndarray = field(
            default=1e-6,
            validator=validators.deep_iterable(
                    iterable_validator=validators.instance_of(np.ndarray),
                    member_validator=getype_validator(float, 0))
    )
    rtol: np.ndarray = field(
            default=1e-6,
            validator=validators.deep_iterable(
                    iterable_validator=validators.instance_of(np.ndarray),
                    member_validator=getype_validator(float, 0))
    )
    algorithm_order: int = field(
            default=1,
            validator=getype_validator(int, 1)
    )
    min_gain: float = field(
            default=0.3,
            validator=inrangetype_validator(float, 0, 1)
    )
    max_gain: float = field(
            default=2.0,
            validator=getype_validator(float, 1)
    )
    safety: float = field(
            default=0.9,
            validator=inrangetype_validator(float, 0, 1)
    )

    def __attrs_post_init__(self):
        self._validate_config()

    def _validate_config(self):
        if self._dt_max is None:
            self._dt_max = self.dt_min * 100
        if self.dt_max < self.dt_min:
            warn("dt_max ({self.dt_max}) < dt_min ({self.dt_min}). Setting "
                 "dt_max = dt_min * 100")
            self._dt_max = self.dt_min * 100
        
    @property
    def dt_min(self) -> float:
        """
        Returns the minimum time step size.

        Returns
        -------
        float
            Minimum time step size from the loop step configuration.
        """
        return self._dt_min

    @property
    def dt_max(self) -> float:
        """Returns the maximum time step size."""
        return self._dt_max

    @property
    def dt0(self) -> float:
        """Returns initial step size at start of loop."""
        return (self.dt_min + self.dt_max) / 2

    @property
    def is_adaptive(self) -> bool:
        """Returns whether the step controller is adaptive."""
        return True

