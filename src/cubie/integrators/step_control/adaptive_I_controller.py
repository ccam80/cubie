"""Adaptive integral step controller."""

from typing import Optional, Union, Callable

from numba import cuda, int32
from numpy._typing import ArrayLike

from cubie.integrators.step_control.adaptive_step_controller import (
    BaseAdaptiveStepController, AdaptiveStepControlConfig
)

import numpy as np

class AdaptiveIController(BaseAdaptiveStepController):
    """Integral step-size controller using only previous error."""

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
        norm: str = "hairer",
        norm_kwargs: Optional[dict] = None,
    ) -> None:
        """Initialise an integral step controller.

        Parameters
        ----------
        precision
            Data type used for calculations.
        dt_min
            Minimum allowed step size.
        dt_max
            Maximum allowed step size.
        atol, rtol
            Absolute and relative tolerances.
        algorithm_order
            Order of the integration algorithm.
        n
            Number of state variables.
        min_gain, max_gain
            Bounds on the step size change factor.
        norm
            Error norm to use.
        norm_kwargs
            Additional keyword arguments passed to the norm factory.
        """
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

        super().__init__(config,
                         norm,
                         norm_kwargs)

    @property
    def local_memory_required(self) -> int:
        """Amount of local memory required by the controller."""
        return 0

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
        """Create the device function for the integral controller."""
        order_exponent = precision(1.0 / (2 * (1 + order)))

        # step sizes and norms can be approximate - fastmath is fine
        @cuda.jit(device=True, inline=True, fastmath=True)
        def controller_I(
            dt, state, state_prev, error, accept_out, scaled_error, local_temp
        ):
            """Integral accept/step-size controller."""
            for i in range(n):
                tol = atol[i] + rtol[i] * max(
                    abs(state[i]), abs(state_prev[i])
                )
                scaled_error[i] = tol / error[i]

            nrm2 = norm_func(scaled_error)
            accept = nrm2 >= precision(1.0)
            accept_out[0] = int32(1) if accept else int32(0)

            gaintmp = safety * (nrm2 ** order_exponent)
            gain = clamp(gaintmp, max_gain, min_gain)

            # Update step from the current dt
            dt_new_raw = dt[0] * gain
            dt[0] = clamp(dt_new_raw, dt_max, dt_min)

            ret = int32(0) if dt_new_raw > dt_min else int32(1)
            return ret

        return controller_I
   