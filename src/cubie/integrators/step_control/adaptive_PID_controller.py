"""Adaptive proportional–integral–derivative step controller."""

from typing import Optional, Union, Callable

import numpy as np
from numba import cuda, int32
from numpy._typing import ArrayLike
from attrs import define, field, validators

from cubie.integrators.step_control.adaptive_step_controller import (
    BaseAdaptiveStepController,
)
from cubie.integrators.step_control.adaptive_PI_controller import (
    PIStepControlConfig)


@define
class PIDStepControlConfig(PIStepControlConfig):
    """Configuration for a proportional–integral–derivative controller."""

    _kd: float = field(
        default=0.0, validator=validators.instance_of(float)
    )

    @property
    def kd(self) -> float:
        """Returns derivative gain."""
        return self.precision(self._kd)


class AdaptivePIDController(BaseAdaptiveStepController):
    """Adaptive PID step size controller."""

    def __init__(
        self,
        precision: type,
        dt_min: float,
        dt_max: Optional[float] = None,
        atol: Optional[Union[float, np.ndarray, ArrayLike]] = 1e-6,
        rtol: Optional[Union[float, np.ndarray, ArrayLike]] = 1e-6,
        algorithm_order: int = 2,
        n: int = 1,
        kp: float = 0.7,
        ki: float = 0.4,
        kd: float = 0.2,
        min_gain: float = 0.2,
        max_gain: float = 5.0,
        norm: str = "l2",
        norm_kwargs: Optional[dict] = None,
    ) -> None:
        """Initialise a proportional–integral–derivative controller."""

        config = PIDStepControlConfig(
            precision=precision,
            dt_min=dt_min,
            dt_max=dt_max,
            atol=atol,
            rtol=rtol,
            algorithm_order=algorithm_order,
            min_gain=min_gain,
            max_gain=max_gain,
            kp=kp,
            ki=ki,
            kd=kd,
            n=n,
        )

        super().__init__(config, norm, norm_kwargs)

    @property
    def kp(self) -> float:
        """Proportional gain."""
        return self.compile_settings.kp

    @property
    def ki(self) -> float:
        """Integral gain."""
        return self.compile_settings.ki

    @property
    def kd(self) -> float:
        """Derivative gain."""
        return self.compile_settings.kd

    @property
    def local_memory_elements(self) -> int:
        """Amount of local memory required by the controller."""
        return 2

    def settings_dict(self) -> dict:
        """Returns a dictionary of settings."""
        settings_dict = super().settings_dict
        settings_dict.update({'kp': self.kp,
                              'ki': self.ki,
                              'kd': self.kd})
        return settings_dict

    def build_controller(
        self,
        precision: type,
        clamp: Callable,
        norm_func: Callable,
        min_gain: float,
        max_gain: float,
        dt_min: float,
        dt_max: float,
        n: int,
        atol: np.ndarray,
        rtol: np.ndarray,
        order: np.ndarray,
        safety: float,
    ) -> Callable:
        """Create the device function for the PID controller."""

        kp = self.kp
        ki = self.ki
        kd = self.kd
        expo1 = precision(kp / (2 * (order + 1)))
        expo2 = precision(ki / (2 * (order + 1)))
        expo3 =     precision(kd / (2 * (order + 1)))

        @cuda.jit(device=True, inline=True, fastmath=True)
        def controller_PID(
            dt,
            state,
            state_prev,
            error,
            niters,
            accept_out,
            local_temp
        ):
            err_prev = local_temp[0]
            err_prev_inv = local_temp[1]
            nrm2 = precision(0.0)
            for i in range(n):
                tol = atol[i] + rtol[i] * max(abs(state[i]),
                                              abs(state_prev[i]))
                nrm2 += (tol*tol) / (error[i]*error[i])

            nrm2 = precision(nrm2/n)
            accept = nrm2 >= precision(1.0)
            accept_out[0] = int32(1) if accept else int32(0)
            err_prev_safe = err_prev if err_prev > precision(0.0) else nrm2
            err_prev_inv_safe = err_prev_inv if err_prev_inv > precision(0.0) \
                else nrm2

            gain_new = precision(safety * ((nrm2 ** expo1) *
                                 (err_prev_safe ** expo2) *
                                 ((nrm2*err_prev_inv_safe) ** expo3)))
            gain = precision(clamp(gain_new, min_gain, max_gain))

            dt_new_raw = dt[0] * gain
            dt[0] = clamp(dt_new_raw, dt_min, dt_max)
            local_temp[1] = 1/nrm2
            local_temp[0] = nrm2

            ret = int32(0) if dt_new_raw > dt_min else int32(1)
            return ret

        return controller_PID
