"""Adaptive proportional–integral–derivative controller implementations."""

from typing import Callable, Optional, Union

import numpy as np
from numba import cuda, int32
from numpy._typing import ArrayLike
from attrs import define, field, validators

from cubie._utils import PrecisionDType, _expand_dtype
from cubie.integrators.step_control.adaptive_step_controller import (
    BaseAdaptiveStepController,
)
from cubie.integrators.step_control.adaptive_PI_controller import (
    PIStepControlConfig)
from cubie.cuda_simsafe import selp


@define
class PIDStepControlConfig(PIStepControlConfig):
    """Configuration for a proportional–integral–derivative controller."""

    _kd: float = field(
        default=0.0,
        validator=validators.instance_of(_expand_dtype(float)),
    )

    @property
    def kd(self) -> float:
        """Return the derivative gain."""
        return self.precision(self._kd)

class AdaptivePIDController(BaseAdaptiveStepController):
    """Adaptive PID step size controller."""

    def __init__(
        self,
        precision: PrecisionDType,
        dt_min: float = 1e-6,
        dt_max: float = 1.0,
        atol: Optional[Union[float, np.ndarray, ArrayLike]] = 1e-6,
        rtol: Optional[Union[float, np.ndarray, ArrayLike]] = 1e-6,
        algorithm_order: int = 2,
        n: int = 1,
        kp: float = 1/18,
        ki: float = 1/9,
        kd: float = 1/18,
        min_gain: float = 0.2,
        max_gain: float = 5.0,
        deadband_min: float = 1.0,
        deadband_max: float = 1.2,
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
        deadband_min
            Lower gain threshold for holding the previous step size.
        deadband_max
            Upper gain threshold for holding the previous step size.
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
            deadband_min=deadband_min,
            deadband_max=deadband_max,
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
    def deadband_min(self) -> float:
        """Return the lower gain threshold for the unity deadband."""

        return self.compile_settings.deadband_min

    @property
    def deadband_max(self) -> float:
        """Return the upper gain threshold for the unity deadband."""

        return self.compile_settings.deadband_max

    @property
    def local_memory_elements(self) -> int:
        """Return the number of local memory slots required."""

        return 2

    @property
    def settings_dict(self) -> dict[str, object]:
        """Return the configuration as a dictionary."""
        settings_dict = super().settings_dict
        settings_dict.update(
            {
                'kp': self.kp,
                'ki': self.ki,
                'kd': self.kd,
                'deadband_min': self.deadband_min,
                'deadband_max': self.deadband_max,
            }
        )
        return settings_dict

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
        algorithm_order
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
        expo1 = precision(kp / (2 * (algorithm_order + 1)))
        expo2 = precision(ki / (2 * (algorithm_order + 1)))
        expo3 = precision(kd / (2 * (algorithm_order + 1)))
        unity_gain = precision(1.0)
        deadband_min = precision(self.deadband_min)
        deadband_max = precision(self.deadband_max)
        deadband_disabled = (
            (deadband_min == unity_gain)
            and (deadband_max == unity_gain)
        )

        @cuda.jit(device=True, inline=True, fastmath=True)
        def controller_PID(
            dt,
            state,
            state_prev,
            error,
            niters,
            accept_out,
            local_temp
        ):  # pragma: no cover - CUDA
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
                error_i = max(abs(error[i]), precision(1e-12))
                tol = atol[i] + rtol[i] * max(
                    abs(state[i]), abs(state_prev[i])
                )
                ratio = tol / error_i
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
                * (nrm2 ** (expo1))
                * (err_prev_safe ** (expo2))
                * (err_prev_prev_safe ** (expo3))
            )
            gain = precision(clamp(gain_new, min_gain, max_gain))
            if not deadband_disabled:
                within_deadband = (
                    (gain >= deadband_min)
                    and (gain <= deadband_max)
                )
                gain = selp(within_deadband, unity_gain, gain)

            dt_new_raw = dt[0] * gain
            dt[0] = clamp(dt_new_raw, dt_min, dt_max)
            local_temp[1] = err_prev
            local_temp[0] = nrm2

            ret = int32(0) if dt_new_raw > dt_min else int32(8)
            return ret

        return controller_PID
