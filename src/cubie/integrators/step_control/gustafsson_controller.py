"""Gustafsson predictive step controller."""

from typing import Optional, Union, Callable

import numpy as np
from numba import cuda, int32
from numpy._typing import ArrayLike

from cubie.integrators.step_control.adaptive_step_controller import (
    BaseAdaptiveStepController, AdaptiveStepControlConfig
)


class GustafssonController(BaseAdaptiveStepController):
    """Adaptive controller using Gustafsson acceleration."""

    def __init__(
        self,
        precision: type,
        dt_min: float,
        dt_max: Optional[float] = None,
        atol: Optional[Union[float, np.ndarray, ArrayLike]] = 1e-6,
        rtol: Optional[Union[float, np.ndarray, ArrayLike]] = 1e-6,
        algorithm_order: int = 2,
        n: int = 1,
        min_gain: float = 0.2,
        max_gain: float = 5.0,
        norm: str = "l2",
        norm_kwargs: Optional[dict] = None,
    ) -> None:
        """Initialise a Gustafsson predictive controller."""

        atol = self.sanitise_tol_array(atol, n, precision)
        rtol = self.sanitise_tol_array(rtol, n, precision)

        config = AdaptiveStepControlConfig(
            precision=precision,
            dt_min=dt_min,
            dt_max=dt_max,
            atol=atol,
            rtol=rtol,
            algorithm_order=algorithm_order,
            min_gain=min_gain,
            max_gain=max_gain,
            n=n,
        )

        super().__init__(config, norm, norm_kwargs)

    @property
    def local_memory_required(self) -> int:
        """Amount of local memory required by the controller."""
        return 2  # previous dt and error

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
        """Create the device function for the Gustafsson controller."""
        expo = precision(1.0 / (2 * (order + 1)))

        @cuda.jit(device=True, inline=True, fastmath=True)
        def controller_gustafsson(
            dt, state, state_prev, error, accept_out, scaled_error, local_temp
        ):
            dt_prev = local_temp[0]
            err_prev = local_temp[1]
            for i in range(n):
                tol = atol[i] + rtol[i] * max(abs(state[i]),
                                              abs(state_prev[i]))
                scaled_error[i] = tol / error[i]

            nrm2 = norm_func(scaled_error)
            accept = nrm2 >= precision(1.0)
            accept_out[0] = int32(1) if accept else int32(0)

            gain_basic = precision(safety * (nrm2 ** expo))
            if accept and dt_prev > precision(0.0) and err_prev > precision(0.0):
                ratio = nrm2 / err_prev
                gain_gus = safety * (dt[0] / dt_prev) * (ratio ** expo)
                gain = gain_gus if gain_gus < gain_basic else gain_basic
                local_temp[0] = dt[0]
                local_temp[1] = max(nrm2, precision(1e-4))
            else:
                gain = gain_basic

            gain = clamp(gain, max_gain, min_gain)
            dt_new_raw = dt[0] * gain
            dt[0] = clamp(dt_new_raw, dt_max, dt_min)

            ret = int32(0) if dt_new_raw > dt_min else int32(1)
            return ret

        return controller_gustafsson
