"""Adaptive integral step controller."""

from typing import Callable, Optional, Union

from numba import cuda, int32
from numpy._typing import ArrayLike

from cubie._utils import PrecisionDType
from cubie.integrators.step_control.adaptive_step_controller import (
    BaseAdaptiveStepController, AdaptiveStepControlConfig
)
from cubie.cuda_simsafe import selp

import numpy as np

class AdaptiveIController(BaseAdaptiveStepController):
    """Integral step-size controller using only previous error."""

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
        max_gain: float = 2.0,
        deadband_min: float = 1.0,
        deadband_max: float = 1.2,
    ) -> None:
        """Initialise an integral step controller.

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
        deadband_min
            Lower gain threshold for holding the previous step size.
        deadband_max
            Upper gain threshold for holding the previous step size.
        """

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
            deadband_min=deadband_min,
            deadband_max=deadband_max,
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
    ) -> Callable:
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
        unity_gain = precision(1.0)
        deadband_min = precision(self.deadband_min)
        deadband_max = precision(self.deadband_max)
        deadband_disabled = (
            (deadband_min == unity_gain)
            and (deadband_max == unity_gain)
        )

        # step sizes and norms can be approximate - fastmath is fine
        @cuda.jit(device=True, inline=True, fastmath=True)
        def controller_I(
            dt,
            state,
            state_prev,
            error,
            niters,
            accept_out,
            local_temp,
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
            local_temp : device array
                Scratch space provided by the integrator.

            Returns
            -------
            int32
                Non-zero when the step is rejected at the minimum size.
            """
            nrm2 = precision(0.0)
            for i in range(n):
                error_i = max(abs(error[i]), precision(1e-12))
                tol = atol[i] + rtol[i] * max(
                    abs(state[i]), abs(state_prev[i])
                )
                nrm2 += (tol * tol) / (error_i * error_i)

            nrm2 = precision(nrm2/n)
            accept = nrm2 >= precision(1.0)
            accept_out[0] = int32(1) if accept else int32(0)

            gaintmp = safety * (nrm2 ** order_exponent)
            gain = clamp(gaintmp, min_gain, max_gain)
            if not deadband_disabled:
                within_deadband = (
                    (gain >= deadband_min)
                    and (gain <= deadband_max)
                )
                gain = selp(within_deadband, unity_gain, gain)

            # Update step from the current dt
            dt_new_raw = dt[0] * gain
            dt[0] = clamp(dt_new_raw, dt_min, dt_max)

            ret = int32(0) if dt_new_raw > dt_min else int32(8)
            return ret

        return controller_I
