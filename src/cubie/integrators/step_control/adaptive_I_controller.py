from typing import Optional, Union

from numba import cuda, int32
from numba import from_dtype
from numpy._typing import ArrayLike

from cubie import clamp_factory
from cubie.errornorms import get_norm_factory
from cubie.integrators.step_control.adaptive_step_config import \
    AdaptiveStepControlConfig
from cubie.integrators.step_control.base_adaptive_controller import \
    BaseAdaptiveStepController

import numpy as np

class AdaptiveIIController(BaseAdaptiveStepController):


    def __init__(self,
                 precision: type,
                 dt_min: float,
                 dt_max: Optional[float] = None,
                 atol: Optional[Union[float, np.ndarray, ArrayLike]] = 1e-6,
                 rtol: Optional[Union[float, np.ndarray, ArrayLike]] = 1e-6,
                 algorithm_order: int = 2,
                 n: int = 1,
                 min_gain: float = 0.2,
                 max_gain: float = 5.0,
                 norm: str = 'hairer',
                 norm_kwargs: Optional[dict] = None):
        """
        Adaptive PI controller for time step adjustment in numerical integration.

        This controller adjusts the time step based on the error estimates
        from the current and previous steps using a Proportional-Integral (PI)
        control strategy.

        """
        super().__init__()

        if isinstance(atol, float):
            atol = np.asarray([atol]* n, dtype=precision)
        else:
            atol = np.asarray(atol, dtype=self.precision)
            if atol.shape[0] != n:
                raise ValueError("atol must have shape (n,).")

        if isinstance(rtol, float):
            rtol = np.asarray([rtol]* n, dtype=precision)
        else:
            rtol = np.asarray(rtol, dtype=precision)
            if rtol.shape[0] != n:
                raise ValueError("atol must have shape (n,).")

        config = AdaptiveStepControlConfig(precision=precision,
                                     dt_min=dt_min,
                                     dt_max=dt_max,
                                     atol=atol,
                                     rtol=rtol,
                                     algorithm_order=algorithm_order,
                                     min_gain=min_gain,
                                     max_gain=max_gain,
                                     n=n
)

        self.setup_compile_settings(config)

        norm_factory = get_norm_factory(norm)
        if norm_kwargs is None:
            norm_kwargs = {}
        try:
            self.norm_func = norm_factory(precision, n, **norm_kwargs)
        except TypeError:
            raise AttributeError("Invalid parameters for chosen norm: "
                                 f"{norm_kwargs}. Check the norm function for "
                                 "expected parameters.")


    @property
    def kp(self) -> float:
        """Returns proportional gain."""
        return self.compile_settings.kp

    @property
    def ki(self) -> float:
        """Returns integral gain."""
        return self.compile_settings.ki

    @property
    def min_gain(self) -> float:
        """Returns minimum gain."""
        return self.compile_settings.min_gain

    @property
    def max_gain(self) -> float:
        """Returns maximum gain."""
        return self.compile_settings.max_gain

    @property
    def atol(self) -> np.ndarray:
        """ Returns absolute tolerance vector."""
        return self.compile_settings.atol

    @property
    def rtol(self) -> np.ndarray:
        """ Returns relative tolerance vector."""
        return self.compile_settings.rtol

    @property
    def local_memory_required(self) -> int:
        return 0

    def build(self):
        precision = from_dtype(self.precision)
        clamp = clamp_factory(precision)
        min_gain = self.min_gain
        max_gain = self.max_gain
        dt_min = self.dt_min
        dt_max = self.dt_max
        n = self.compile_settings.n
        norm_func = self.norm_func
        order = self.compile_settings.algorithm_order
        order_exponent = 1/ ((1 + order) * 2) # *2 as the norm isn't rooted

        safety = self.compile_settings.safety_factor

        # step sizes and norms can be approximate - fastmath is fine
        @cuda.jit(device=True, inline=True, fastmath=True)
        def controller_I(
            dt, state, state_prev, error, accept_out, scaled_error, local_temp
        ):
            """
            PI-like accept/step-size controller:

              - Computes a tentative step unconditionally (into state_tmp).
              - Measures an error norm (here: L2 of `err` row).
              - Accepts if tol - norm >= 0.
              - Updates dt using a PI term and clamps to [dt_min, dt_max].

            Writes single element arrays in place:
            dt[0] = dt_new (clamped)
            accept[0] = accept (as Int32: 1 for True, 0 for False)
            error_integral_array[0] = running error integral

            Returns retcode as int32.
            """
            for i in range(n):
                tol = self.atol[i] + self.rtol[i] * abs(
                    max(state[i], state_prev[i])
                )
                scaled_error[i] = tol/error[i]

            nrm2 = norm_func(scaled_error)
            accept = nrm2 >= precision(0.0)
            accept_out[0] = int32(1) if accept else int32(0)

            gaintmp = nrm2 ** order_exponent * safety
            gain = clamp(gaintmp, min_gain, max_gain)

            # Update step from the current dt
            dt_new_raw = dt[0] * gain
            dt[0] = clamp(dt_new_raw, dt_min, dt_max)

            ret = int32(0) if dt_new_raw > dt_min else int32(1)
            return ret

        return controller_I
   