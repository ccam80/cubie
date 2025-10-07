"""Fixed step-size controller implementations."""

from typing import Callable

from attrs import define, field
from numba import cuda, int32

from cubie._utils import PrecisionDType, getype_validator
from cubie.integrators.step_control.base_step_controller import (
    BaseStepControllerConfig, BaseStepController)

@define
class FixedStepControlConfig(BaseStepControllerConfig):
    """Configuration for fixed-step integrator loops.

    Attributes
    ----------
    precision
        Precision used for numerical operations.
    n
        Number of state variables controlled per step.
    """
    _dt: float = field(
        default=1e-3, validator=getype_validator(float, 0)
    )

    def __attrs_post_init__(self) -> None:
        """Validate configuration after initialisation."""
        self._validate_config()

    def _validate_config(self) -> None:
        """Confirm that the configuration is internally consistent."""

        return True

    @property
    def dt(self) -> float:
        """Return the fixed step size."""
        return self.precision(self._dt)

    @property
    def dt_min(self) -> float:
        """Return the minimum time step size."""
        return self.dt

    @property
    def dt_max(self) -> float:
        """Return the maximum step size."""
        return self.dt
    @property
    def dt0(self) -> float:
        """Return the initial step size used at loop start."""
        return self.dt

    @property
    def is_adaptive(self) -> bool:
        """Return ``False`` because the controller is not adaptive."""
        return False

    @property
    def settings_dict(self) -> dict[str, object]:
        """Return the configuration as a dictionary."""
        settings_dict = super().settings_dict
        settings_dict.update({'dt': self.dt})
        return settings_dict

class FixedStepController(BaseStepController):
    """Controller that enforces a constant time step."""

    def __init__(
        self,
        precision: PrecisionDType,
        dt: float,
        n: int = 1,
    ) -> None:
        """Initialise the fixed step controller.

        Parameters
        ----------
        precision
            Precision used for controller calculations.
        dt
            Fixed step size to apply on every iteration.
        n
            Number of state variables advanced by the integrator.
        """

        super().__init__()
        config = FixedStepControlConfig(precision=precision, n=n, dt=dt)
        self.setup_compile_settings(config)

    def build(self) -> Callable:
        """Return a device function that always accepts with fixed step.

        Returns
        -------
        Callable
            CUDA device function that keeps the step size constant.
        """

        @cuda.jit(device=True, inline=True, fastmath=True)
        def controller_fixed_step(
            dt, state, state_prev, error, accept_out, local_temp
        ):  # pragma: no cover - CUDA
            """Fixed-step controller device function.

            Parameters
            ----------
            dt : device array
                Current integration step size.
            state : device array
                Current state vector.
            state_prev : device array
                Previous state vector.
            error : device array
                Estimated local error vector.
            accept_out : device array
                Output flag indicating acceptance of the step.
            local_temp : device array
                Scratch space provided by the integrator.

            Returns
            -------
            int32
                Zero, indicating that the current step size should be kept.
            """

            accept_out[0] = int32(1)
            return int32(0)

        return controller_fixed_step

    @property
    def local_memory_elements(self) -> int:
        """Amount of local memory required by the controller."""
        return 0

    @property
    def dt(self) -> float:
        """Return the fixed step size used by the controller."""

        return self.compile_settings.dt
