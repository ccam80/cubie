"""Structural tests for step controllers."""

import numpy as np
import pytest
from numba import cuda

from cubie.result_codes import CUBIE_RESULT_CODES
from tests._utils import run_controller_device_step


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

    @pytest.mark.parametrize(
        "flag, value",
        [("afn", False), ("lto", False), ("lineinfo", True)],
    )
    def test_jit_flag_updates_reach_compile_settings(
        self, step_controller, flag, value
    ):
        """Jit-flag settings route into the controller's config."""
        original = getattr(step_controller.compile_settings.jit_flags, flag)
        recognized = step_controller.update_compile_settings({flag: value})
        try:
            assert flag in recognized
            assert getattr(
                step_controller.compile_settings.jit_flags, flag
            ) == value
        finally:
            step_controller.update_compile_settings({flag: original})

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
                   niters_val, truncated_flag, accept_val, shared_val,
                   persistent_val):
            device_func(dt_val, state_val, state_prev_val, err_val,
                        niters_val, truncated_flag, accept_val,
                        shared_val, persistent_val)

        # First step: solver-failure error injection (loop uses 1e16).
        huge_error = np.full(n, 1e16, dtype=precision)
        kernel[1, 1](dt, state, state_prev, huge_error, niters, False,
                     accept, shared_scratch, persistent_local)
        assert int(accept[0]) == 0

        # Second step: moderate rejection (nrm2 just above one). The
        # stored history from the failed step must not grow dt.
        dt_before = precision(0.017)
        dt[0] = dt_before
        moderate_error = np.full(n, 1.23e-3, dtype=precision)
        kernel[1, 1](dt, state, state_prev, moderate_error, niters,
                     False, accept, shared_scratch, persistent_local)
        assert int(accept[0]) == 0
        assert float(dt[0]) < float(dt_before)

    def test_truncated_accepted_step_freezes_controller(
        self, step_controller, precision, system
    ):
        """An accepted truncated step rescales nothing: dt and the
        error history are unchanged."""
        device_func = step_controller.device_function
        n = system.sizes.states
        dt0 = precision(0.017)
        tiny_error = np.full(n, 1e-12, dtype=precision)
        state = np.ones(n, dtype=precision)

        frozen = run_controller_device_step(
            device_func,
            precision,
            dt0,
            tiny_error,
            state=state,
            state_prev=state,
            truncated=True,
        )
        assert frozen.accepted == 1
        assert frozen.dt == dt0
        assert np.all(frozen.local_mem == precision(0.0))

        unforced = run_controller_device_step(
            device_func,
            precision,
            dt0,
            tiny_error,
            state=state,
            state_prev=state,
            truncated=False,
        )
        assert unforced.accepted == 1
        assert unforced.dt > dt0

    def test_truncated_rejected_step_still_shrinks_dt(
        self, step_controller, precision, system
    ):
        """A rejected truncated step still walks dt down."""
        device_func = step_controller.device_function
        n = system.sizes.states
        dt0 = precision(0.017)
        huge_error = np.full(n, 1e16, dtype=precision)
        state = np.ones(n, dtype=precision)

        result = run_controller_device_step(
            device_func,
            precision,
            dt0,
            huge_error,
            state=state,
            state_prev=state,
            truncated=True,
        )
        assert result.accepted == 0
        assert result.dt < dt0

    def test_truncated_accepted_step_at_dt_min_returns_success(
        self, step_controller, precision, system
    ):
        """An accepted truncated step at dt_min reports SUCCESS.

        Its sub-unity gain would otherwise propose dt <= dt_min and
        end the run as irrecoverable.
        """
        device_func = step_controller.device_function
        n = system.sizes.states
        dt_min = precision(step_controller.dt_min)
        # Error norm just below 1: accepted, with gain < 1.
        near_unity_error = np.full(n, 0.999e-3, dtype=precision)
        state = np.ones(n, dtype=precision)

        result = run_controller_device_step(
            device_func,
            precision,
            dt_min,
            near_unity_error,
            state=state,
            state_prev=state,
            truncated=True,
        )
        assert result.accepted == 1
        assert result.dt == dt_min
        assert result.status == int(CUBIE_RESULT_CODES.SUCCESS)
