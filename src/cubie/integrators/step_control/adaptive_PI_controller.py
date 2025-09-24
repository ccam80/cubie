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
    _kp: float = field(
        default=0.075,
        validator=validators.instance_of(float),
    )
    _ki: float = field(
        default=0.175,
        validator=validators.instance_of(float),
    )

    @property
    def kp(self) -> float:
        """Returns proportional gain."""
        return self.precision(self._kp)

    @property
    def ki(self) -> float:
        """Returns integral gain."""
        return self.precision(self._ki)


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
        norm: str = "l2",
        norm_kwargs: Optional[dict] = None,
    ) -> None:
        """Initialise a proportional–integral step controller."""

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
    def local_memory_elements(self) -> int:
        """Amount of local memory required by the controller."""
        return 1

    @property
    def settings_dict(self) -> dict:
        """Returns settings as a dictionary."""
        settings_dict = super().settings_dict
        settings_dict.update({'kp': self.kp,
                              'ki': self.ki})
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
        order: int,
        safety: float,
    ) -> Callable:
        """Create the device function for the PI controller."""
        kp = precision(self.kp / ((order + 1) * 2))
        ki = precision(self.ki / ((order + 1) * 2))

        # step sizes and norms can be approximate - fastmath is fine
        @cuda.jit(device=True, inline=True, fastmath=True)
        def controller_PI(
            dt, state, state_prev, error, niters, accept_out, local_temp
        ):
            """Proportional–integral accept/step-size controller."""
            err_prev = local_temp[0]
            nrm2 = precision(0.0)
            for i in range(n):
                tol = atol[i] + rtol[i] * max(
                    abs(state[i]), abs(state_prev[i])
                )
                ratio = tol / error[i]
                nrm2 += ratio * ratio

            nrm2 = precision(nrm2/n)
            accept = nrm2 >= precision(1.0)
            accept_out[0] = int32(1) if accept else int32(0)

            pgain = precision(nrm2 ** (kp))
            # Handle uninitialized err_prev by using current error as fallback
            igain = precision((err_prev if err_prev > precision(0.0) else nrm2) ** (ki))
            gain_new = safety * pgain * igain
            gain = clamp(gain_new, min_gain, max_gain)

            dt_new_raw = dt[0] * gain
            dt[0] = clamp(dt_new_raw, dt_min, dt_max)
            local_temp[0] = nrm2

            ret = int32(0) if dt_new_raw > dt_min else int32(1)
            return ret

        return controller_PI
