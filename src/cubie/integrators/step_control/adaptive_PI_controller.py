"""Adaptive proportional–integral step controller."""

from typing import Optional, Union, Callable

from numba import cuda, int32
import numpy as np
from numpy._typing import ArrayLike
from attrs import field, define, validators

from cubie.integrators.step_control.adaptive_step_controller import (
    AdaptiveStepControlConfig, BaseAdaptiveStepController
)


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


class AdaptivePIController(BaseAdaptiveStepController):
    """Proportional–integral step-size controller."""

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
        min_gain: float = 0.2,
        max_gain: float = 5.0,
        norm: str = "hairer",
        norm_kwargs: Optional[dict] = None,
    ) -> None:
        """Initialise a proportional–integral step controller."""
        atol = self.sanitise_tol_array(atol, n, precision)
        rtol = self.sanitise_tol_array(rtol, n, precision)

        config = PIStepControlConfig(
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
            n=n,
        )

        super().__init__(config,
                         norm,
                         norm_kwargs)


    @property
    def kp(self) -> float:
        """Returns proportional gain."""
        return self.compile_settings.kp

    @property
    def ki(self) -> float:
        """Returns integral gain."""
        return self.compile_settings.ki

    @property
    def local_memory_required(self) -> int:
        """Amount of local memory required by the controller."""
        return 1

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
        """Create the device function for the PI controller."""
        kp = self.kp / (order + 1)
        ki = self.ki / (order + 1)

        # step sizes and norms can be approximate - fastmath is fine
        @cuda.jit(device=True, inline=True, fastmath=True)
        def controller_PI(
            dt, state, state_prev, error, accept_out, scaled_error, local_temp
        ):
            """Proportional–integral accept/step-size controller."""
            err_prev = local_temp[0]
            for i in range(n):
                tol = atol[i] + rtol[i] * max(
                    abs(state[i]), abs(state_prev[i])
                )
                scaled_error[i] = tol / error[i]

            #consider fusing into this function to include in scaled error
            # calculation without a buffer
            nrm2 = norm_func(scaled_error)
            accept = nrm2 >= precision(1.0)
            accept_out[0] = int32(1) if accept else int32(0)

            pgain = nrm2 ** (kp / 2)
            igain = err_prev ** (ki / 2)
            gain_new = safety * pgain * igain
            gain = clamp(gain_new, max_gain, min_gain)

            dt_new_raw = dt[0] * gain
            dt[0] = clamp(dt_new_raw, dt_max, dt_min)
            local_temp[0] = nrm2

            ret = int32(0) if dt_new_raw > dt_min else int32(1)
            return ret

        return controller_PI
   