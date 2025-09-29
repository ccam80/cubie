"""Adaptive proportional–integral–derivative controller implementations."""

from typing import Callable, Optional, Union

import numpy as np
from numba import cuda, int32
from numpy._typing import ArrayLike
from attrs import define, field

from cubie._utils import gttype_validator
from cubie.integrators.step_control.adaptive_step_controller import (
    BaseAdaptiveStepController,
)
from cubie.integrators.step_control.adaptive_PI_controller import (
    PIStepControlConfig)


@define
class PIDStepControlConfig(PIStepControlConfig):
    """Configuration for a proportional–integral–derivative controller."""

    _kd: float = field(
        default=0.0, validator=gttype_validator(float, 0.0)
    )

    @property
    def kd(self) -> float:
        """Return the derivative gain."""
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
    ) -> None:
        """Initialise a proportional–integral–derivative controller.

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
        kp
            Proportional gain before scaling for controller order.
        ki
            Integral gain before scaling for controller order.
        kd
            Derivative gain before scaling for controller order.
        min_gain
            Lower bound for the step size change factor.
        max_gain
            Upper bound for the step size change factor.
        """

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

        super().__init__(config)

    @property
    def kp(self) -> float:
        """Return the proportional gain."""
        return self.compile_settings.kp

    @property
    def ki(self) -> float:
        """Return the integral gain."""
        return self.compile_settings.ki

    @property
    def kd(self) -> float:
        """Return the derivative gain."""
        return self.compile_settings.kd

    @property
    def local_memory_elements(self) -> int:
        """Return the number of local memory slots required."""

        return 2

    @property
    def settings_dict(self) -> dict[str, object]:
        """Return the configuration as a dictionary."""
        settings_dict = super().settings_dict
        settings_dict.update({'kp': self.kp,
                              'ki': self.ki,
                              'kd': self.kd})
        return settings_dict

    def build_controller(
        self,
        precision: type,
        clamp: Callable,
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
        """Create the device function for the PID controller.

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
        order
            Order of the integration algorithm.
        safety
            Safety factor used when scaling the step size.

        Returns
        -------
        Callable
            CUDA device function implementing the PID controller.
        """

        kp = self.kp
        ki = self.ki
        kd = self.kd
        expo1 = precision(kp / (2 * (order + 1)))
        expo2 = precision(ki / (2 * (order + 1)))
        expo3 = precision(kd / (2 * (order + 1)))

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
            """Proportional–integral–derivative accept/step controller.

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
            niters : device array
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
            err_prev = local_temp[0]
            err_prev_prev = local_temp[1]
            nrm2 = precision(0.0)
            for i in range(n):
                error[i] = max(error[i], precision(1e-30))
                tol = atol[i] + rtol[i] * max(
                    abs(state[i]), abs(state_prev[i])
                )
                ratio = tol / error[i]
                nrm2 += ratio * ratio

            nrm2 = precision(nrm2/n)
            accept = nrm2 >= precision(1.0)
            accept_out[0] = int32(1) if accept else int32(0)
            err_prev_safe = err_prev if err_prev > precision(0.0) else nrm2
            err_prev_prev_safe = (
                err_prev_prev if err_prev_prev > precision(0.0) else err_prev_safe
            )

            gain_new = precision(
                safety
                * (nrm2 ** expo1)
                * (err_prev_safe ** (-expo2))
                * (err_prev_prev_safe ** (-expo3))
            )
            gain = precision(clamp(gain_new, min_gain, max_gain))

            dt_new_raw = dt[0] * gain
            dt[0] = clamp(dt_new_raw, dt_min, dt_max)
            local_temp[1] = err_prev
            local_temp[0] = nrm2

            ret = int32(0) if dt_new_raw > dt_min else int32(1)
            return ret

        return controller_PID
