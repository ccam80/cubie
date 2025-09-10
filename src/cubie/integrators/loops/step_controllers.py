from numba import cuda, int32, int8, from_dtype
from cubie._utils import clamp_factory

def l2_squared_norm_factory(precision, n):
    precision = from_dtype(precision)

    @cuda.jit(device=True, inline=True)
    def norm_l2_squared(i, in1, in2):
        acc = precision(0.0)
        for j in range(n):
            val = in1[i, j] - in2[i, j]
            acc += val * val
        return acc

def PI_factory(tol2, kp, ki, dt_min, dt_max, precision, norm_func):
    precision = from_dtype(precision)
    clamp = clamp_factory(precision)
    @cuda.jit(device=True, inline=True)
    def controller_PI(i,
                      dt_current,
                      state,
                      state_tmp,
                      acc_int_in):
        """
        PI-like accept/step-size controller:
          - Computes a tentative step unconditionally (into state_tmp).
          - Measures an error norm (here: L2 of `err` row).
          - Accepts if tol - norm >= 0.
          - Updates dt using a PI term and clamps to [dt_min, dt_max].
        Returns (accept, dt_new, acc_int_out, retmask).
        """

        nrm = norm_func(i, state, state_tmp)
        ctrl_err = tol2 - nrm

        acc_int_out = acc_int_in + ctrl_err
        gain = kp * ctrl_err + ki * acc_int_out

        # Update step from the current dt
        dt_new_raw = dt_current * (precision(1.0) + gain)
        dt_new = clamp(dt_new_raw, dt_min, dt_max)

        accept = ctrl_err >= precision(0.0)

        ret = int32(0)
        ret |= int32(dt_new <= dt_min) << int8(0)

        return accept, dt_new, acc_int_out, ret

    return controller_PI