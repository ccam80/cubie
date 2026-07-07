"""Structural tests for step controllers."""

import numpy as np
import pytest
from numba import cuda


_CONTROLLER_SETTINGS = {
    controller: {"step_controller": controller, "atol": 1e-3, "rtol": 0.0}
    for controller in ("i", "pi", "pid", "gustafsson")
}

_DT_CLAMP_LIMITS = {"dt": 0.15, "dt_min": 0.1, "dt_max": 0.2}
_DT_CLAMP_CASES = {
    "max_limit": {"dt0": 1.0, "error": np.asarray([1e-12, 1e-12, 1e-12])},
    "min_limit": {"dt0": 0.001, "error": np.asarray([1e12, 1e12, 1e12])},
}


@pytest.mark.parametrize(
    "solver_settings_override, step_setup",
    [
        (dict(settings, **_DT_CLAMP_LIMITS), case)
        for settings in _CONTROLLER_SETTINGS.values()
        for case in _DT_CLAMP_CASES.values()
    ],
    ids=[
        f"{controller}-{case}"
        for controller in _CONTROLLER_SETTINGS
        for case in _DT_CLAMP_CASES
    ],
    indirect=True,
)
def test_dt_clamps(
    step_controller_settings,
    step_setup,
    device_step_results,
    tolerance,
):
    dt0 = step_setup["dt0"]
    dt_min = step_controller_settings["dt_min"]
    dt_max = step_controller_settings["dt_max"]
    if dt0 < dt_min:
        expected = dt_min
    elif dt0 > dt_max:
        expected = dt_max
    else:
        expected = dt0
    assert device_step_results.dt == pytest.approx(
        expected,
        rel=tolerance.rel_tight,
        abs=tolerance.abs_tight,
    )


@pytest.mark.parametrize(
    "solver_settings_override",
    list(_CONTROLLER_SETTINGS.values()),
    ids=list(_CONTROLLER_SETTINGS),
    indirect=True,
)
class TestControllers:
    def test_controller_builds(self, step_controller, precision):
        assert callable(step_controller.device_function)

    def test_rejected_step_never_grows_dt(
        self, step_controller, precision, system
    ):
        """A rejected step shrinks dt despite a huge stored error history.

        The error history from a failed step (the loop stores 1e16 on
        solver failure) feeds the PI/PID history term on the next call;
        the gain on a rejected step must stay below one regardless.
        """
        device_func = step_controller.device_function
        n = system.sizes.states

        dt = np.asarray([0.017], dtype=precision)
        accept = np.zeros(1, dtype=np.int32)
        niters = np.int32(1)
        shared_scratch = np.zeros(1, dtype=precision)
        persistent_local = np.zeros(4, dtype=precision)
        state = np.ones(n, dtype=precision)
        state_prev = np.ones(n, dtype=precision)

        @cuda.jit
        def kernel(dt_val, state_val, state_prev_val, err_val,
                   niters_val, accept_val, shared_val, persistent_val):
            device_func(dt_val, state_val, state_prev_val, err_val,
                        niters_val, accept_val, shared_val,
                        persistent_val)

        # First step: solver-failure error injection (loop uses 1e16).
        huge_error = np.full(n, 1e16, dtype=precision)
        kernel[1, 1](dt, state, state_prev, huge_error, niters, accept,
                     shared_scratch, persistent_local)
        assert int(accept[0]) == 0

        # Second step: moderate rejection (nrm2 just above one). The
        # stored history from the failed step must not grow dt.
        dt_before = precision(0.017)
        dt[0] = dt_before
        moderate_error = np.full(n, 1.23e-3, dtype=precision)
        kernel[1, 1](dt, state, state_prev, moderate_error, niters,
                     accept, shared_scratch, persistent_local)
        assert int(accept[0]) == 0
        assert float(dt[0]) < float(dt_before)
