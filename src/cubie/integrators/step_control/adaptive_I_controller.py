"""Adaptive integral step controller."""
from typing import Callable, Optional, Union

from numba import cuda, int32
from numpy._typing import ArrayLike

from cubie._utils import PrecisionDType, build_config
from cubie.integrators.step_control.adaptive_step_controller import (
    BaseAdaptiveStepController,
    AdaptiveStepControlConfig,
)
from cubie.cuda_simsafe import compile_kwargs, selp

import numpy as np

from cubie.integrators.step_control.base_step_controller import ControllerCache


class AdaptiveIController(BaseAdaptiveStepController):
    """Integral step-size controller using only previous error."""

    def __init__(
        self,
        precision: PrecisionDType,
        n: int = 1,
        **kwargs,
    ) -> None:
        """Initialise an integral step controller.

        Parameters
        ----------
        precision
            Precision used for controller calculations.
        n
            Number of state variables.
        **kwargs
            Optional parameters passed to AdaptiveStepControlConfig. See
            AdaptiveStepControlConfig for available parameters including
            dt_min, dt_max, atol, rtol, algorithm_order, min_gain, max_gain,
            deadband_min, deadband_max. None values are ignored.
        """
        config = build_config(
            AdaptiveStepControlConfig,
            required={'precision': precision, 'n': n},
            **kwargs
        )

        super().__init__(config)

    @property
    def local_memory_elements(self) -> int:
        """Return the number of local memory slots required."""

        return 0

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
        """Create the device function for the integral controller.

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
            CUDA device function implementing the integral controller.
        """
        order_exponent = precision(1.0 / (2 * (1 + algorithm_order)))
        typed_one = precision(1.0)
        typed_zero = precision(0.0)
        deadband_min = precision(self.deadband_min)
        deadband_max = precision(self.deadband_max)
        safety = precision(safety)
        min_gain = precision(min_gain)
        max_gain = precision(max_gain)
        deadband_disabled = (
            (deadband_min == typed_one)
            and (deadband_max == typed_one)
        )
        n = int32(n)
        inv_n = precision(1.0 / n)

        precision = self.compile_settings.numba_precision
        # step sizes and norms can be approximate - fastmath is fine
        @cuda.jit(
            device=True,
            inline=True,
            **compile_kwargs,
        )
        def controller_I(
            dt,
            state,
            state_prev,
            error,
            niters,
            accept_out,
            shared_scratch,
            persistent_local,
        ):  # pragma: no cover - CUDA
            """Integral accept/step-size controller.

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
            nrm2 = typed_zero
            for i in range(n):
                error_i = max(abs(error[i]), precision(1e-16))
                tol = (
                    atol[i] + rtol[i] * max(abs(state[i]), abs(state_prev[i]))
                )
                ratio = error_i / tol
                nrm2 += ratio * ratio

            nrm2 = nrm2 * inv_n
            accept = nrm2 <= typed_one
            accept_out[0] = int32(1) if accept else int32(0)

            gaintmp = precision(safety * nrm2 ** (-order_exponent))
            gain = clamp(gaintmp, min_gain, max_gain)
            if not deadband_disabled:
                within_deadband = (
                    (gain >= deadband_min)
                    and (gain <= deadband_max)
                )
                gain = selp(within_deadband, typed_one, gain)

            # Update step from the current dt
            dt_new_raw = dt[0] * gain
            dt[0] = clamp(dt_new_raw, dt_min, dt_max)

            ret = int32(0) if dt_new_raw > dt_min else int32(8)
            return ret

        return ControllerCache(device_function=controller_I)
