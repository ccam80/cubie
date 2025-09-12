from numba import cuda, int32, from_dtype
from cubie._utils import clamp_factory


def PI_factory(tol2, kp, ki, dt_min, dt_max, precision, norm_func):
    precision = from_dtype(precision)
    clamp = clamp_factory(precision)
    @cuda.jit(device=True, inline=True)
    def controller_PI(dt,
                      state,
                      state_tmp,
                      accept_array,
                      error_integral_array):
        """
        PI-like accept/step-size controller:

          - Computes a tentative step unconditionally (into state_tmp).
          - Measures an error norm (here: L2 of `err` row).
          - Accepts if tol - norm >= 0.
          - Updates dt using a PI term and clamps to [dt_min, dt_max].

        Writes single element arrays in place:
        dt[0] = dt_new (clamped)
        accept_array[0] = accept (as Int32: 1 for True, 0 for False)
        error_integral_array[0] = running error integral

        Returns retcode as int32.
        """

        nrm = norm_func(state, state_tmp)
        ctrl_err = tol2 - nrm

        error_integral_array[0] = error_integral_array[0] + ctrl_err
        gain = kp * ctrl_err + ki * error_integral_array[0]

        # Update step from the current dt
        dt_new_raw = dt[0] * (precision(1.0) + gain)
        dt[0] = clamp(dt_new_raw, dt_min, dt_max)

        accept = ctrl_err >= precision(0.0)

        ret = int32(0)
        ret |= int32(dt[0] <= dt_min)

        # Write results to separate typed arrays in-place
        accept_array[0] = int32(1) if accept else int32(0)

        return ret

    return controller_PI