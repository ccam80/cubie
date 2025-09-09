
from numba import cuda

def fixed_step_controller_factory(step_fn,
                                  tolerance,
                                  norm_func,
                                  dt_min,
                                  kp,
                                  ki):

    kp = kp / tolerance
    ki = ki / tolerance
    @cuda.jit(
        (
            "float64, float64, float64[::1], float64[::1], float64[::1], "
            "float64[::1], int32, int16"
        ),
        device=True,
        inline=True,
    )
    def fixed_step_controller(
        t,
        dt,
        state_in,
        state_out,
        error,
        scratch,
        retcode,
        tx
    ):
        t = t + dt
        # overwrite state_in for fixed-step as there's no need to keep it for
        # a second try
        code = step_fn(dt, state_in, state_in, error, scratch)

    def adaptive_PI_controller(
            t,
            dt,
            accumulator,
            state_in,
            state_out,
            error,
            norm_func,
            scratch,
            next_save,
            retcode,
            tx
    ):
        code = step_fn(dt, state_in, state_out, error, scratch)
        current_norm = norm_func(state_out, error)
        control_error = tolerance - current_norm # should protect this from
        # going above one or calculate next dt differently
        accumulator += control_error
        ctrl_output = kp*control_error + ki*accumulator
        dt = dt * ctrl_output
        retcode &= (dt < dt_min) * 5 # let's say 5 is the error code for
        # step size too small

        # Accept/reject needn't be a branch - we can selp it, the work is
        # already done anyway
        t = cuda.selp(control_error > 0, t + dt, t)
        for i in range(state_in.shape[0]):
            state_out[i] = cuda.selp(control_error > 0, state_out[i],
                                     state_in[i])

        # Control flow is legit branching: we must call save and update
        # functions. we want to almost never branch:

        if t == next_save:
            return
        # Favour an every-iteration selp over a branch
        t = cuda.selp(t > next_save, next_save, t)

# then loop looks like:
def loop_factory(step_controller,
                     save_state_fn,
                     update_summaries_fn,
                     save_summaries_fn,
                     duration,
                     saves_per_summary):
    @cuda.jit()
    def loop(inputs,
             step_controller,# has step_fn baked in)
             t_start):
        # process_inputs
        t = cuda.local.array(1)
        retcode = cuda.local.array(1)
        t[0] = t_start
        while True:
            if t[0] >= duration:
                break
            for i in range(saves_per_summary):
                step_controller(retcode)
                save_state_fn()
                update_summaries_fn()
                if retcode[0] != 0:
                    return
            save_summaries_fn()


# Robot's sketchpython
from numba import int32
import math

@cuda.jit(device=True, inline=True, fastmath=True)
def step_fn(dt, state_in, state_out, err, scratch):
    # Compute next state and error estimate unconditionally
    # ... toy body ...
    for i in range(state_in.shape[0]):
        state_out[i] = state_in[i] + dt * err[i]
    return 0  # status

@cuda.jit(device=True, inline=True, fastmath=True)
def norm_fn(state, err):
    # Example norm
    acc = 0.0
    for i in range(state.shape[0]):
        acc = acc + err[i] * err[i]
    return math.sqrt(acc)

@cuda.jit(device=True, inline=True, fastmath=True)
def controller_PI(t, dt, state_in, state_out, err, scratch,
                  kp, ki, tol, dt_min, dt_max, acc_int):
    code = step_fn(dt, state_in, state_out, err, scratch)
    n = norm_fn(state_out, err)
    ctrl_err = tol - n
    acc_int += ctrl_err
    gain = kp * ctrl_err + ki * acc_int

    # Clamp dt branchlessly
    new_dt = dt * gain
    new_dt = dt_min if new_dt < dt_min else new_dt
    new_dt = dt_max if new_dt > dt_max else new_dt

    accept = ctrl_err >= 0.0

    # Predicated commit of t and state; rejection keeps inputs
    t = t + new_dt if accept else t
    for i in range(state_in.shape[0]):
        s = state_out[i]
        state_out[i] = s if accept else state_in[i]

    # Branchless retcode bit for too-small dt (keep bitmask idea)
    ret = 0
    ret |= int32(new_dt <= dt_min) << 0
    ret |= int32(code != 0) << 1

    return t, new_dt, acc_int, ret

@cuda.jit(fastmath=True)
def integrate_kernel(state_in, state_out, err, scratch,
                     t0, dt0, steps, save_every,
                     kp, ki, tol, dt_min, dt_max,
                     saves, retcode_out):
    i = cuda.grid(1)
    if i >= state_in.shape[0]:
        return

    # Per-thread scalars in registers
    t = t0
    dt = dt0
    acc_int = 0.0
    save_ctr = save_every
    ret = 0

    for _ in range(steps):
        if ret != 0:
            break

        t, dt, acc_int, r = controller_PI(t, dt, state_in, state_out, err, scratch,
                                          kp, ki, tol, dt_min, dt_max, acc_int)
        ret |= r

        # Save scheduling via integer countdown (no float equality)
        save_ctr -= 1
        do_save = save_ctr == 0
        if do_save:
            # write to `saves` buffer at an index derived from i and step
            save_ctr = save_every

    retcode_out[i] = ret