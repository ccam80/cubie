"""
Integrator configuration management with validation and adapter patterns.

This module provides the IntegratorLoopSettings class for managing compile-critical
settings for integrator loops, including timing parameters, buffer sizes, and
function references. It uses validation and adapter patterns to ensure
configuration consistency.
"""

from attrs import define, field
from numba import cuda, int32

from cubie._utils import getype_validator
from cubie.integrators.step_control.base_step_controller import (
    BaseStepControllerConfig, BaseStepController)

@define
class FixedStepControlConfig(BaseStepControllerConfig):
    """ Configuration for fixed-step integrator loops.

    There's not much to it.

    Parameters
    ----------
    precision:
        numpy datatype to work in: select from (float32, float64, float16)
    dt:
        fixed step size

    Attributes
    ----------
    precision:
        numpy datatype to work in: select from (float32, float64, float16)
    dt:
        fixed step size

    Methods
    -------
    dt_min: self.precision
        alias for dt
    dt_max: self.precision
        alias for dt
    dt0: self.precision
        alias for dt
    is_adaptive: bool
        False

    """
    dt: float = field(default=1e-3, validator=getype_validator(float, 0))

    def __attrs_post_init__(self):
        self._validate_config()

    def _validate_config(self):
        return True

    @property
    def dt_min(self) -> float:
        """Returns the minimum time step size."""
        return self.dt

    @property
    def dt_max(self) -> float:
        """Returns the maximum step size."""
        return self.dt
    @property
    def dt0(self) -> float:
        """Returns initial step size at start of loop."""
        return self.dt

    @property
    def is_adaptive(self) -> bool:
        """Returns whether the step controller is adaptive."""
        return False


class FixedStepController(BaseStepController):
    """Placeholder controller for a fixed step loop - does nothing but
    returns the right properties to satisfy the interface """
    def __init__(self,
                 precision: type,
                 dt: float):
        self.dt = dt
        super().__init__()
        config = FixedStepControlConfig(precision=precision, n=1, dt=dt)
        self.setup_compile_settings(config)

    def build(self):
        cuda.jit(device=True, inline=True, fastmath=True)
        def controller_fixed_step(
            dt, state, state_prev, error, accept_out, scaled_error, local_temp
        ):
            return int32(0)
        return controller_fixed_step

    @property
    def local_memory_required(self) -> int:
        return 0
