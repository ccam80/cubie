"""Gustafsson predictive step controller."""

from typing import Optional, Union, Callable

import numpy as np
from numba import cuda, int32
from numpy._typing import ArrayLike
from attrs import define, field

from cubie.integrators.step_control.adaptive_step_controller import (
    BaseAdaptiveStepController, AdaptiveStepControlConfig
)
from cubie._utils import getype_validator, inrangetype_validator

@define
class GustafssonStepControlConfig(AdaptiveStepControlConfig):
    """Configuration for Gustafsson-like predictive controller.

    Adds gamma damping factor and maximum Newton iterations for implicit solves.
    """
    _gamma: float = field(default=0.9, validator=inrangetype_validator(float, 0, 1))
    _max_newton_iters: int = field(default=0, validator=getype_validator(int, 0))

    @property
    def gamma(self) -> float:
        return self.precision(self._gamma)

    @property
    def max_newton_iters(self) -> int:
        return int(self._max_newton_iters)


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
        gamma: float = 0.9,
        max_newton_iters: int = 0,
    ) -> None:
        """Initialise a Gustafsson predictive controller."""

        config = GustafssonStepControlConfig(
            precision=precision,
            dt_min=dt_min,
            dt_max=dt_max,
            atol=atol,
            rtol=rtol,
            algorithm_order=algorithm_order,
            min_gain=min_gain,
            max_gain=max_gain,
            n=n,
            gamma=gamma,
            max_newton_iters=max_newton_iters,
        )

        super().__init__(config, norm, norm_kwargs)

    @property
    def gamma(self) -> float:
        return self.compile_settings.gamma

    @property
    def max_newton_iters(self) -> int:
        return self.compile_settings.max_newton_iters

    @property
    def local_memory_elements(self) -> int:
        """Amount of local memory required by the controller."""
        return 2

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
        gamma = precision(self.gamma)
        max_newton_iters = int(self.max_newton_iters)
        gain_numerator = precision((1 + 2 * max_newton_iters)) * gamma

        @cuda.jit(device=True, inline=True, fastmath=True)
        def controller_gustafsson(
            dt, state, state_prev, error, niters, accept_out, local_temp
        ):
            dt_prev = max(local_temp[0],precision(1e-16))
            err_prev = max(local_temp[1], precision(1e-16))

            nrm2 = precision(0.0)
            for i in range(n):
                tol = atol[i] + rtol[i] * max(abs(state[i]),
                                              abs(state_prev[i]))
                nrm2 += (tol*tol) / (error[i]*error[i])

            nrm2 = precision(nrm2/n)
            accept = nrm2 >= precision(1.0)
            accept_out[0] = int32(1) if accept else int32(0)

            niters_step = niters[0]
            denom = precision(niters_step + 2 * max_newton_iters)
            tmp = gain_numerator / denom
            fac = gamma if gamma < tmp else tmp
            gain_basic = precision(safety * fac * (nrm2 ** expo))

            ratio = (nrm2*nrm2) / err_prev
            gain_gus = precision(safety * (dt[0] /dt_prev) * (ratio ** expo) *
                                 gamma)
            gain = gain_gus if gain_gus < gain_basic else gain_basic
            gain = gain if (accept and dt_prev > precision(1e-16)) else (
                gain_basic)

            gain = clamp(gain, min_gain, max_gain)
            dt_new_raw = dt[0] * gain
            dt[0] = clamp(dt_new_raw, dt_min, dt_max)

            local_temp[0] = dt[0]
            local_temp[1] = min(nrm2, precision(1e4))
            ret = int32(0) if dt_new_raw > dt_min else int32(1)
            return ret

        return controller_gustafsson
