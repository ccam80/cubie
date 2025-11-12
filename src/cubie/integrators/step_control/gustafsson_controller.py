"""Gustafsson predictive step controller."""

from typing import Callable, Optional, Union

import numpy as np
from numba import cuda, int32
from numpy._typing import ArrayLike
from attrs import define, field

from cubie.integrators.step_control.adaptive_step_controller import (
    BaseAdaptiveStepController, AdaptiveStepControlConfig
)
from cubie._utils import PrecisionDType, getype_validator, inrangetype_validator
from cubie.cuda_simsafe import selp

@define
class GustafssonStepControlConfig(AdaptiveStepControlConfig):
    """Configuration for Gustafsson-like predictive controller.

    Notes
    -----
    Includes damping and Newton iteration limits used by Gustafsson's
    predictor for implicit integrators.
    """
    _gamma: float = field(
        default=0.9,
        validator=inrangetype_validator(float, 0, 1),
    )
    _max_newton_iters: int = field(
        default=0,
        validator=getype_validator(int, 0),
    )

    @property
    def gamma(self) -> float:
        """Return the damping factor applied to the gain."""

        return self.precision(self._gamma)

    @property
    def max_newton_iters(self) -> int:
        """Return the maximum number of Newton iterations considered."""
        return int(self._max_newton_iters)

    @property
    def settings_dict(self) -> dict[str, object]:
        """Return the configuration as a dictionary."""
        settings_dict = super().settings_dict
        settings_dict.update({'gamma': self.gamma,
                              'max_newton_iters': self.max_newton_iters})
        return settings_dict

class GustafssonController(BaseAdaptiveStepController):
    """Adaptive controller using Gustafsson acceleration."""

    def __init__(
        self,
        precision: PrecisionDType,
        dt_min: float = 1e-6,
        dt_max: float = 1.0,
        atol: Optional[Union[float, np.ndarray, ArrayLike]] = 1e-6,
        rtol: Optional[Union[float, np.ndarray, ArrayLike]] = 1e-6,
        algorithm_order: int = 2,
        n: int = 1,
        min_gain: float = 0.2,
        max_gain: float = 5.0,
        gamma: float = 0.9,
        max_newton_iters: int = 0,
        deadband_min: float = 1.0,
        deadband_max: float = 1.2,
    ) -> None:
        """Initialise a Gustafsson predictive controller.

        Parameters
        ----------
        precision
            Precision used for controller calculations.
        dt_min
            Minimum allowed step size.
        dt_max
            Maximum allowed step size.
        atol
            Absolute tolerance specification.
        rtol
            Relative tolerance specification.
        algorithm_order
            Order of the integration algorithm.
        n
            Number of state variables.
        min_gain
            Lower bound for the step size change factor.
        max_gain
            Upper bound for the step size change factor.
        gamma
            Gustafsson damping factor applied to the gain.
        max_newton_iters
            Maximum number of Newton iterations expected during solves.
        deadband_min
            Lower gain threshold for holding the previous step size.
        deadband_max
            Upper gain threshold for holding the previous step size.
        """

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
            deadband_min=deadband_min,
            deadband_max=deadband_max,
        )

        super().__init__(config)

    @property
    def gamma(self) -> float:
        """Return the damping factor applied to the gain."""

        return self.compile_settings.gamma

    @property
    def max_newton_iters(self) -> int:
        """Return the maximum number of Newton iterations considered."""

        return self.compile_settings.max_newton_iters

    @property
    def local_memory_elements(self) -> int:
        """Return the number of local memory slots required."""

        return 2

    def build_controller(
        self,
        precision: PrecisionDType,
        clamp: Callable,
        min_gain: float,
        max_gain: float,
        dt_min: float,
        dt_max: float,
        n: int,
        atol: np.ndarray,
        rtol: np.ndarray,
        algorithm_order: int,
        safety: float,
    ) -> Callable:
        """Create the device function for the Gustafsson controller.

        Parameters
        ----------
        precision
            Precision callable used to coerce scalars on device.
        clamp
            Callable that clamps proposed step sizes.
        min_gain
            Minimum allowed gain when adapting the step size.
        max_gain
            Maximum allowed gain when adapting the step size.
        dt_min
            Minimum permissible step size.
        dt_max
            Maximum permissible step size.
        n
            Number of state variables controlled per step.
        atol
            Absolute tolerance vector.
        rtol
            Relative tolerance vector.
        algorithm_order
            Order of the integration algorithm.
        safety
            Safety factor used when scaling the step size.

        Returns
        -------
        Callable
            CUDA device function implementing the Gustafsson controller.
        """
        expo = precision(1.0 / (2 * (algorithm_order + 1)))
        gamma = precision(self.gamma)
        max_newton_iters = int(self.max_newton_iters)
        gain_numerator = precision((1 + 2 * max_newton_iters)) * gamma
        unity_gain = precision(1.0)
        deadband_min = precision(self.deadband_min)
        deadband_max = precision(self.deadband_max)
        deadband_disabled = (
            (deadband_min == unity_gain)
            and (deadband_max == unity_gain)
        )

        @cuda.jit(device=True, inline=True, fastmath=True)
        def controller_gustafsson(
            dt, state, state_prev, error, niters, accept_out, local_temp
        ):  # pragma: no cover - CUDA
            """Gustafsson accept/step controller.

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
            niters : int32
                Iteration counters from the integrator loop.
            accept_out : device array
                Output flag indicating acceptance of the step.
            local_temp : device array
                Scratch space provided by the integrator.

            Returns
            -------
            int32
                Non-zero when the step is rejected at the minimum size.
            """

            current_dt = dt[0]
            dt_prev = max(local_temp[0], precision(1e-16))
            err_prev = max(local_temp[1], precision(1e-16))

            nrm2 = precision(0.0)
            for i in range(n):
                error_i = max(abs(error[i]), precision(1e-12))
                tol = atol[i] + rtol[i] * max(
                    abs(state[i]), abs(state_prev[i])
                )
                ratio = tol / error_i
                nrm2 += ratio * ratio

            nrm2 = precision(nrm2/n)
            accept = nrm2 >= precision(1.0)
            accept_out[0] = int32(1) if accept else int32(0)

            denom = precision(niters + 2 * max_newton_iters)
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
            if not deadband_disabled:
                within_deadband = (
                    (gain >= deadband_min)
                    and (gain <= deadband_max)
                )
                gain = selp(within_deadband, unity_gain, gain)
            dt_new_raw = current_dt * gain
            dt[0] = clamp(dt_new_raw, dt_min, dt_max)

            local_temp[0] = current_dt
            local_temp[1] = nrm2
            ret = int32(0) if dt_new_raw > dt_min else int32(8)
            return ret

        return controller_gustafsson
