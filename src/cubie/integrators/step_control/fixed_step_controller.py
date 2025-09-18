"""Fixed step-size controller implementations."""

from typing import Callable

from attrs import define, field
from numba import cuda, int32

from cubie._utils import getype_validator
from cubie.integrators.step_control.base_step_controller import (
    BaseStepControllerConfig, BaseStepController)

@define
class FixedStepControlConfig(BaseStepControllerConfig):
    """Configuration for fixed-step integrator loops.

    Parameters
    ----------
    precision : type
        Floating point precision.
    dt : float
        Fixed step size.
    """
    _dt: float = field(
        default=1e-3, validator=getype_validator(float, 0)
    )

    def __attrs_post_init__(self) -> None:
        """Validate configuration after initialisation."""
        self._validate_config()

    def _validate_config(self) -> None:
        return True

    @property
    def dt(self) -> float:
        """Returns fixed step size."""
        return self.precision(self._dt)

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
    """Controller that enforces a constant time step."""

    def __init__(self, precision: type, dt: float) -> None:
        """Initialise the fixed step controller."""
        self.dt = dt
        super().__init__()
        config = FixedStepControlConfig(precision=precision, n=1, dt=dt)
        self.setup_compile_settings(config)

    def build(self) -> Callable:
        """Return a device function that always accepts with fixed step."""
        @cuda.jit(device=True, inline=True, fastmath=True)
        def controller_fixed_step(
            dt, state, state_prev, error, accept_out, local_temp
        ):
            accept_out[0] = int32(1)
            return int32(0)
        return controller_fixed_step

    @property
    def local_memory_required(self) -> int:
        """Amount of local memory required by the controller."""
        return 0
