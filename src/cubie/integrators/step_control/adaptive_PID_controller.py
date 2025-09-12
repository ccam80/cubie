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

    kd: float = field(default=0.0, validator=validators.instance_of(float))


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
        norm: str = "hairer",
        norm_kwargs: Optional[dict] = None,
    ) -> None:

        atol = self.sanitise_tol_array(atol, n, precision)
        rtol = self.sanitise_tol_array(rtol, n, precision)

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
        return self.compile_settings.kp

    @property
    def ki(self) -> float:
        return self.compile_settings.ki

    @property
    def kd(self) -> float:
        return self.compile_settings.kd

    @property
    def local_memory_required(self) -> int:
        return 2

    def build_controller(self,
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
                         safety: float):

        kp = self.kp
        ki = self.ki
        kd = self.kd
        expo1 = precision(kp / (2 * (order + 1)))
        expo2 = precision(ki / (2 * (order + 1)))
        expo3 = precision(kd / (2 * (order + 1)))

        @cuda.jit(device=True, inline=True, fastmath=True)
        def controller_PID(
            dt, state, state_prev, error, accept_out, scaled_error, local_temp
        ):
            err_prev = local_temp[0]
            err_prev2 = local_temp[1]
            for i in range(n):
                tol = atol[i] + rtol[i] * max(abs(state[i]),
                                              abs(state_prev[i]))
                scaled_error[i] = tol / error[i]

            nrm2 = norm_func(scaled_error)
            accept = nrm2 >= precision(1.0)
            accept_out[0] = int32(1) if accept else int32(0)

            gain_new = safety * ((nrm2 ** expo1) *
                                 (err_prev ** expo2) *
                                 (err_prev2 ** expo3))
            gain = clamp(gain_new, max_gain, min_gain)

            dt_new_raw = dt[0] * gain
            dt[0] = clamp(dt_new_raw, dt_max, dt_min)
            local_temp[1] = err_prev
            local_temp[0] = nrm2

            ret = int32(0) if dt_new_raw > dt_min else int32(1)
            return ret

        return controller_PID
