"""Adaptive proportional–integral–derivative controller implementations."""
from typing import Callable, Optional, Union

import numpy as np
from numba import cuda, int32
from attrs import define, field, validators

from cubie._utils import PrecisionDType, _expand_dtype, build_config
from cubie.buffer_registry import buffer_registry
from cubie.integrators.step_control.adaptive_step_controller import (
    BaseAdaptiveStepController,
)
from cubie.integrators.step_control.adaptive_PI_controller import (
    PIStepControlConfig,
)
from cubie.cuda_simsafe import compile_kwargs, selp
from cubie.integrators.step_control.base_step_controller import ControllerCache


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
        n: int = 1,
        **kwargs,
    ) -> None:
        """Initialise a proportional–integral–derivative controller.

        Parameters
        ----------
        precision
            Precision used for controller calculations.
        n
            Number of state variables.
        **kwargs
            Optional parameters passed to PIDStepControlConfig. See
            PIDStepControlConfig for available parameters including dt_min,
            dt_max, atol, rtol, algorithm_order, kp, ki, kd, min_gain,
            max_gain, deadband_min, deadband_max. None values are ignored.
        """
        config = build_config(
            PIDStepControlConfig,
            required={'precision': precision, 'n': n},
            **kwargs
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
    ) -> ControllerCache:
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
        alloc_timestep_buffer = buffer_registry.get_allocator(
            'timestep_buffer', self
        )

        kp = self.kp
        ki = self.ki
        kd = self.kd
        expo1 = precision(kp / (2 * (algorithm_order + 1)))
        expo2 = precision(ki / (2 * (algorithm_order + 1)))
        expo3 = precision(kd / (2 * (algorithm_order + 1)))
        safety = precision(safety)
        typed_one = precision(1.0)
        typed_zero = precision(0.0)
        min_gain = precision(min_gain)
        max_gain = precision(max_gain)
        dt_min = precision(dt_min)
        dt_max = precision(dt_max)
        deadband_min = precision(self.deadband_min)
        deadband_max = precision(self.deadband_max)
        deadband_disabled = (deadband_min == typed_one) and (
                deadband_max == typed_one
        )
        precision = self.compile_settings.numba_precision
        n = int32(n)
        inv_n = precision(1.0 / n)
        # step sizes and norms can be approximate - fastmath is fine
        @cuda.jit(
            device=True,
            inline=True,
            **compile_kwargs,
        )
        def controller_PID(
            dt,
            state,
            state_prev,
            error,
            niters,
            accept_out,
            shared_scratch,
            persistent_local,
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
            shared_scratch : device array
                Shared memory scratch space.
            persistent_local : device array
                Persistent local memory for controller state.

            Returns
            -------
            int32
                Non-zero when the step is rejected at the minimum size.
            """
            timestep_buffer = alloc_timestep_buffer(
                shared_scratch, persistent_local
            )

            err_prev = timestep_buffer[0]
            err_prev_prev = timestep_buffer[1]
            nrm2 = typed_zero

            for i in range(n):
                error_i = max(abs(error[i]), precision(1e-16))
                tol = atol[i] + rtol[i] * max(
                    abs(state[i]), abs(state_prev[i])
                )
                ratio = error_i / tol
                nrm2 += ratio * ratio

            nrm2 = nrm2 * inv_n
            accept = nrm2 <= typed_one
            accept_out[0] = int32(1) if accept else int32(0)
            err_prev_safe = err_prev if err_prev > typed_zero else nrm2
            err_prev_prev_safe = (
                err_prev_prev if err_prev_prev > typed_zero else err_prev_safe
            )

            gain_new = precision(
                safety
                * (nrm2 ** (-expo1))
                * (err_prev_safe ** (-expo2))
                * (err_prev_prev_safe ** (-expo3))
            )
            gain = clamp(gain_new, min_gain, max_gain)
            if not deadband_disabled:
                within_deadband = (
                    (gain >= deadband_min)
                    and (gain <= deadband_max)
                )
                gain = selp(within_deadband, typed_one, gain)

            dt_new_raw = dt[0] * gain
            dt[0] = clamp(dt_new_raw, dt_min, dt_max)
            timestep_buffer[1] = err_prev
            timestep_buffer[0] = nrm2

            ret = int32(0) if dt_new_raw > dt_min else int32(8)
            return ret

        return ControllerCache(device_function=controller_PID)
