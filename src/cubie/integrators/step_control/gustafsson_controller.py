"""Gustafsson predictive step controller."""
from typing import Callable, Optional, Union

import numpy as np
from numba import cuda, int32
from numpy._typing import ArrayLike
from attrs import define, field

from cubie.buffer_registry import buffer_registry
from cubie.integrators.step_control.adaptive_step_controller import (
    BaseAdaptiveStepController, AdaptiveStepControlConfig
)
from cubie._utils import (
    PrecisionDType,
    getype_validator,
    inrangetype_validator,
    build_config,
)
from cubie.cuda_simsafe import compile_kwargs, selp
from cubie.integrators.step_control.base_step_controller import ControllerCache


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
        default=20,
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
        n: int = 1,
        **kwargs,
    ) -> None:
        """Initialise a Gustafsson predictive controller.

        Parameters
        ----------
        precision
            Precision used for controller calculations.
        n
            Number of state variables.
        **kwargs
            Optional parameters passed to GustafssonStepControlConfig. See
            GustafssonStepControlConfig for available parameters including
            dt_min, dt_max, atol, rtol, algorithm_order, min_gain, max_gain,
            gamma, max_newton_iters, deadband_min, deadband_max. None values
            are ignored.
        """
        config = build_config(
            GustafssonStepControlConfig,
            required={'precision': precision, 'n': n},
            **kwargs
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
    ) -> ControllerCache:
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
        alloc_timestep_buffer = buffer_registry.get_allocator(
            'timestep_buffer', self
        )

        expo = precision(1.0 / (2 * (algorithm_order + 1)))
        gamma = precision(self.gamma)
        max_newton_iters = int(self.max_newton_iters)
        gain_numerator = precision((1 + 2 * max_newton_iters)) * gamma
        typed_one = precision(1.0)
        typed_zero = precision(0.0)
        deadband_min = precision(self.deadband_min)
        deadband_max = precision(self.deadband_max)
        min_gain = precision(min_gain)
        max_gain = precision(max_gain)
        deadband_disabled = (deadband_min == typed_one) and (
                deadband_max == typed_one
        )
        numba_precision = self.compile_settings.numba_precision
        n = int32(n)
        inv_n = precision(1.0 / n)
        # step sizes and norms can be approximate - fastmath is fine
        @cuda.jit(
            device=True,
            inline=True,
            **compile_kwargs,
        )
        def controller_gustafsson(
            dt, state, state_prev, error, niters, accept_out,
            shared_scratch, persistent_local
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

            current_dt = dt[0]
            dt_prev = max(timestep_buffer[0], precision(1e-16))
            err_prev = max(timestep_buffer[1], precision(1e-16))

            nrm2 = typed_zero
            for i in range(n):
                error_i = max(abs(error[i]), precision(1e-12))
                tol = atol[i] + rtol[i] * max(
                    abs(state[i]), abs(state_prev[i])
                )
                ratio = error_i / tol
                nrm2 += ratio * ratio

            nrm2 = nrm2 * inv_n
            accept = nrm2 <= typed_one
            accept_out[0] = int32(1) if accept else int32(0)

            denom = precision(niters + 2 * max_newton_iters)
            tmp = gain_numerator / denom
            fac = gamma if gamma < tmp else tmp
            gain_basic = precision(fac * (nrm2 ** (-expo)))

            ratio = nrm2 * nrm2  / err_prev
            gain_gus = precision(safety * (dt[0] /dt_prev) * (ratio ** -expo) *
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
                gain = selp(within_deadband, typed_one, gain)
            dt_new_raw = current_dt * gain
            dt[0] = clamp(dt_new_raw, dt_min, dt_max)

            timestep_buffer[0] = current_dt
            timestep_buffer[1] = nrm2
            ret = int32(0) if dt_new_raw > dt_min else int32(8)
            return ret

        return ControllerCache(device_function=controller_gustafsson)
