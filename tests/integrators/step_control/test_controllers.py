"""Structural tests for step controllers."""

import numpy as np
import pytest
from numba import cuda


@pytest.fixture(scope='function')
def step_setup(request, precision, system):
    n = system.sizes.states
    setup_dict = {'dt0': 0.05,
                  'error': np.asarray([0.01]*system.sizes.states,
                                    dtype=precision),
                  'state': np.ones(n, dtype=precision),
                  'state_prev': np.ones(n, dtype=precision),
                  'local_mem': np.zeros(2, dtype=precision)
                  }
    if hasattr(request, 'param'):
        for key, value in request.param.items():
            if key in setup_dict:
                setup_dict[key] = value
    return setup_dict

class StepResult:
    def __init__(self, dt, accepted, local_mem):
        self.dt = dt
        self.accepted = accepted
        self.local_mem = local_mem

def _run_device_step(
    device_func,
    precision,
    dt0,
    error,
    *,
    local_mem=None,
    state=None,
    state_prev=None,
    niters: int = 1,
):
    """Execute a controller device function once."""

    err = np.asarray(error, dtype=precision)
    state_arr = np.asarray(state, dtype=precision) if state is not None else np.zeros_like(err)
    state_prev_arr = np.asarray(state_prev, dtype=precision) if state_prev is not None else np.zeros_like(err)

    dt = np.asarray([dt0], dtype=precision)
    accept = np.zeros(1, dtype=np.int32)
    temp = np.asarray(local_mem, dtype=precision) if local_mem is not None \
        else np.zeros(2, dtype=precision)
    niters_val = np.int32(niters)
    shared_scratch = np.zeros(1, dtype=precision)
    persistent_local = np.zeros(2, dtype=precision)

    @cuda.jit
    def kernel(dt_val, state_val, state_prev_val, err_val, niters_val,
               accept_val, shared_val, persistent_val):
        device_func(dt_val, state_val, state_prev_val, err_val, niters_val,
                    accept_val, shared_val, persistent_val)

    kernel[1, 1](dt, state_arr, state_prev_arr, err, niters_val, accept,
                 shared_scratch, persistent_local)
    return StepResult(float(dt[0]), int(accept[0]), persistent_local.copy())

@pytest.fixture(scope='function')
def device_step_results(step_controller, precision, step_setup):
    return _run_device_step(
        step_controller.device_function,
        precision,
        step_setup['dt0'],
        step_setup['error'],
        state=step_setup['state'],
        state_prev=step_setup['state_prev'],
        local_mem=step_setup['local_mem'],
    )


@pytest.mark.parametrize(
    "solver_settings_override2",
    [
        ({"step_controller": "i", 'atol':1e-3,'rtol':0.0}),
        ({"step_controller": "pi", 'atol':1e-3,'rtol':0.0}),
        ({"step_controller": "pid", 'atol':1e-3,'rtol':0.0}),
        ({"step_controller": "gustafsson", 'atol':1e-3,'rtol':0.0}),
    ],
    ids=("i", "pi", "pid", "gustafsson"),
    indirect=True
)
class TestControllers:
    def test_controller_builds(self, step_controller, precision):
        assert callable(step_controller.device_function)

    @pytest.mark.parametrize('solver_settings_override, step_setup',
                             (({'dt': 0.15, 'dt_min': 0.1, 'dt_max': 0.2},
                               {'dt0': 1.0, 'error': np.asarray([1e-12, 1e-12, 1e-12])}),
                              ({'dt': 0.15, 'dt_min': 0.1, 'dt_max': 0.2},
                               {'dt0': 0.001, 'error': np.asarray([1e12, 1e12, 1e12])})),
                             ids=("max_limit", "min_limit"),
                             indirect=True)
    def test_dt_clamps(
        self,
        step_controller_settings,
        step_setup,
        device_step_results,
        tolerance,
    ):
        dt0 = step_setup['dt0']
        dt_min = step_controller_settings['dt_min']
        dt_max = step_controller_settings['dt_max']
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
