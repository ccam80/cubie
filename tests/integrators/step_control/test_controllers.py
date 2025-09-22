import numpy as np
import pytest
from numba import cuda


@pytest.fixture(scope='function')
def step_setup(request, precision, system):
    n = system.sizes.states
    setup_dict = {'dt0': 0.01,
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
):
    """Execute a controller device function once."""

    err = np.asarray(error, dtype=precision)
    if state is None:
        state_arr = np.zeros_like(err)
    else:
        state_arr = np.asarray(state, dtype=precision)
    if state_prev is None:
        state_prev_arr = np.zeros_like(err)
    else:
        state_prev_arr = np.asarray(state_prev, dtype=precision)

    dt = np.asarray([dt0], dtype=precision)
    accept = np.zeros(1, dtype=np.int32)
    if local_mem is None:
        temp = np.empty(0, dtype=precision)
    else:
        temp = np.asarray(local_mem, dtype=precision)

    @cuda.jit
    def kernel(dt_val, state_val, state_prev_val, err_val, accept_val,
               temp_val):
        device_func(
            dt_val,
            state_val,
            state_prev_val,
            err_val,
            accept_val,
            temp_val,
        )

    kernel[1, 1](dt, state_arr, state_prev_arr, err, accept, temp)
    return StepResult(dt=float(dt[0]),
                      accepted=int(accept[0]),
                      local_mem=temp.copy())

@pytest.fixture(scope='function')
def device_step_results(step_controller, precision, step_setup):
    """Run device step and return results."""
    return _run_device_step(
        step_controller.device_function,
        precision,
        step_setup['dt0'],
        step_setup['error'],
        state=step_setup['state'],
        state_prev=step_setup['state_prev'],
        local_mem=step_setup['local_mem'],
    )

@pytest.fixture(scope='function')
def cpu_step_results(cpu_step_controller, precision, step_setup):
    """Simulate one CPU controller step, syncing local memory in/out.

    Responsibilities:
    - Ingest provided local_mem (vector sized for max controller) and map to
      CPU controller internal history/state.
    - Compute acceptance & error estimate analogous to device function logic.
    - Update dt via cpu_step_controller.propose_dt(error_estimate, accept).
    - Export updated local_mem vector matching device layout for that kind.
    """
    controller = cpu_step_controller
    kind = controller.kind.lower()
    controller.dt = step_setup['dt0']
    state = np.asarray(step_setup['state'], dtype=precision)
    state_prev = np.asarray(step_setup['state_prev'], dtype=precision)
    error_vec = np.asarray(step_setup['error'], dtype=precision)
    provided_local = np.asarray(step_setup['local_mem'], dtype=precision)

    history = []
    dt_history = []
    if kind == 'pi':
        prev_nrm2 = float(provided_local[0])
        history.append(prev_nrm2)
    elif kind == 'pid':
        prev_nrm2 = float(provided_local[0])
        prev_nrm2 = max(prev_nrm2, 1e-12)
        controller._prev_nrm2 = prev_nrm2
        controller._prev_inv_nrm2 = 1.0 / prev_nrm2
    # elif kind == 'gustafsson':
    #     dt_prev = float(provided_local[0])
    #     err_prev_nrm2 = float(provided_local[1])
    #     history.append(err_prev_nrm2)
    #     controller._dt_history[] = dt_history[-2:]
    controller._history = history[-3:]

    errornorm = controller.error_norm(state_prev, state, error_vec)
    accept = True
    controller.propose_dt(errornorm, accept)

    if kind == 'i':
        out_local = np.zeros(2, dtype=precision)
    elif kind == 'pi':
        out_local = np.array([errornorm, 0.0], dtype=precision)
    elif kind == 'pid':
        prev_nrm2 = float(provided_local[0]) if provided_local.size > 0 else 0.0
        out_local = np.array([errornorm, 1/errornorm], dtype=precision)
    # elif kind == 'gustafsson':
    #     out_local = np.array([controller.dt, max(errornorm, 1e-4)],
    #                          dtype=precision)
    else:
        out_local = np.zeros(2, dtype=precision)

    return StepResult(dt=controller.dt, accepted=int(accept), local_mem=out_local)

@pytest.mark.parametrize(
    "solver_settings_override",
    [
        ({"step_controller": "i",
          'atol':1e-3,'rtol':0.0}),
        ({"step_controller": "pi",
          'atol':1e-3,'rtol':0.0}),
        ({"step_controller": "pid",
          'atol':1e-3,'rtol':0.0}),
        # ({"step_controller": "gustafsson",
        #   'atol':1e-3,'rtol':0.0}),
    ],
    ids=("i", "pi", "pid"),
    indirect=True
)
class TestControllers:
    def test_controller_builds(self, step_controller, precision):
        assert callable(step_controller.device_function)
    @pytest.mark.parametrize('step_controller_settings_override, step_setup',
                             (({'dt_min': 0.1, 'dt_max': 0.2},
                               {'dt0': 1.0, 'error': np.asarray(
                                       [1e-12, 1e-12, 1e-12])}),
                              ({'dt_min': 0.1, 'dt_max': 0.2},
                              {'dt0': 0.001, 'error': np.asarray(
                                       [1e12, 1e12, 1e12])})),
                             ids=("max_limit", "min_limit"),
                             indirect=True)
    def test_dt_clamps(self, step_controller_settings, step_setup,
                    device_step_results):
        dt0 = step_setup['dt0']
        dt_min = step_controller_settings['dt_min']
        dt_max = step_controller_settings['dt_max']
        if dt0 < dt_min:
            expected = dt_min
        elif dt0 > dt_max:
            expected = dt_max
        else:
            expected = dt0
        assert device_step_results.dt == pytest.approx(expected,
                                                       abs=1e-3,
                                                       rel=1e-3)

    @pytest.mark.parametrize(
        'step_controller_settings_override, step_setup',
        (
            (
                {'dt_min': 1e-4, 'dt_max': 1.0, 'min_gain': 0.5, 'max_gain': 1.5},
                {'dt0': 0.1, 'error': np.asarray([1e-12, 1e-12, 1e-12])}
            ),
            (
                {'dt_min': 1e-4, 'dt_max': 1.0, 'min_gain': 0.5, 'max_gain': 1.5},
                {'dt0': 0.1, 'error': np.asarray([1e12, 1e12, 1e12])}
            ),
        ),
        ids=("gain_max_clamp", "gain_min_clamp"),
        indirect=True,
    )
    def test_gain_clamps(self, step_controller_settings, step_setup, device_step_results):
        """Similar to test_dt_clamps, set narrow gain bounds, set error very high or very low, confirm that
        dt_out is either max_gain or min_gain * dt0."""
        dt0 = step_setup['dt0']
        min_gain = step_controller_settings['min_gain']
        max_gain = step_controller_settings['max_gain']
        # Decide direction based on error magnitude
        if float(step_setup['error'][0]) < 1e-6:
            expected = dt0 * max_gain
        else:
            expected = dt0 * min_gain
        assert device_step_results.dt == pytest.approx(expected, rel=1e-4, abs=1e-6)

    @pytest.mark.parametrize(
        'step_setup',
        (
            {'dt0': 0.005, 'error': np.asarray([5e-4, 5e-4, 5e-4])},  # low error
            {'dt0': 0.005, 'error': np.asarray([5e-3, 5e-3, 5e-3])},  # high error
            {'dt0': 0.005, 'error': np.asarray([5e-4, 5e-4, 5e-4]),
             'local_mem': np.asarray([0.005, 0.8])},
            {'dt0': 0.005, 'error': np.asarray([5e-3, 5e-3, 5e-3]),
             'local_mem': np.asarray([0.005, 0.8])},
        ),
        ids=("low_err", "high_err", "low_err_with_mem", "high_err_with_mem"),
        indirect=True,
    )
    def test_matches_cpu(self, step_controller,step_controller_settings,
                        step_setup, cpu_step_results, device_step_results):
        """The test should just assert approximate (1e-6) equality between
        the fields of the cpu_step_results and device_step_results classes.
        Parameterise to:
        - Setup two "normal" situations, with high and low error
        and dt0 in the middle of the range. Confirm that device results
        match cpu results.
        Set values for local_mem that somewhat match a realistic scenario.
        Test that they match again. (same local mem input for all
        controllers; some will ignore parts of it and this is fine) """
        # Compare dt and local memory contents
        assert device_step_results.dt == pytest.approx(cpu_step_results.dt, abs=1e-6, rel=1e-6)
        valid_localmem = step_controller.local_memory_elements
        assert np.allclose(device_step_results.local_mem[:valid_localmem],
                           cpu_step_results.local_mem[:valid_localmem], rtol=1e-6, atol=1e-6)
