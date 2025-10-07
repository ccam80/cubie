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

    @cuda.jit
    def kernel(dt_val, state_val, state_prev_val, err_val, niters_val, accept_val, temp_val):
        device_func(dt_val, state_val, state_prev_val, err_val, niters_val, accept_val, temp_val)

    kernel[1, 1](dt, state_arr, state_prev_arr, err, niters_val, accept, temp)
    return StepResult(float(dt[0]), int(accept[0]), temp.copy())

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

@pytest.fixture(scope='function')
def cpu_step_results(cpu_step_controller, precision, step_setup):
    """CPU analogue for one controller step (niters=1)."""
    controller = cpu_step_controller
    kind = controller.kind.lower()
    controller.dt = step_setup['dt0']
    state = np.asarray(step_setup['state'], dtype=precision)
    state_prev = np.asarray(step_setup['state_prev'], dtype=precision)
    error_vec = np.asarray(step_setup['error'], dtype=precision)
    provided_local = np.asarray(step_setup['local_mem'], dtype=precision)

    if kind == 'pi':
        controller._prev_nrm2 = float(provided_local[0])
    elif kind == 'pid':
        controller._prev_nrm2 = float(provided_local[0])
        controller._prev_prev_nrm2 = float(provided_local[1])
    elif kind == 'gustafsson':
        controller._prev_dt = float(provided_local[0])
        controller._prev_nrm2 = float(provided_local[1])


    accept = controller.propose_dt(prev_state=state_prev,
                          new_state=state,
                          error_vector=error_vec,
                          niters=1)
    errornorm = controller.error_norm(state_prev, state, error_vec)

    if kind == 'i':
        out_local = np.zeros(0, dtype=precision)
    elif kind == 'pi':
        out_local = np.array([errornorm], dtype=precision)
    elif kind == 'pid':
        out_local = np.array([
            controller._prev_nrm2,
            controller._prev_prev_nrm2,
        ], dtype=precision)
    elif kind == 'gustafsson':
        out_local = np.array([
            controller.dt,
            max(errornorm, 1e-4),
        ], dtype=precision)
    else:
        out_local = np.zeros(0, dtype=precision)

    return StepResult(controller.dt, int(accept), out_local)

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
                             (({'dt_min': 0.1, 'dt_max': 0.2},
                               {'dt0': 1.0, 'error': np.asarray([1e-12, 1e-12, 1e-12])}),
                              ({'dt_min': 0.1, 'dt_max': 0.2},
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

    @pytest.mark.parametrize(
        'solver_settings_override, step_setup',
        (
            (
                {
                    'dt_min': 1e-4,
                    'dt_max': 1.0,
                    'min_gain': 0.5,
                    'max_gain': 1.5,
                },
                {
                    'dt0': 0.1,
                    'error': np.asarray([1e-12, 1e-12, 1e-12]),
                    'local_mem': np.asarray([1e6, 1e-6]),
                },
            ),
            (
                {
                    'dt_min': 1e-4,
                    'dt_max': 1.0,
                    'min_gain': 0.5,
                    'max_gain': 1.5,
                },
                {
                    'dt0': 0.1,
                    'error': np.asarray([1e12, 1e12, 1e12]),
                    'local_mem': np.asarray([1e-6, 1e6]),
                },
            ),
        ),
        ids=("gain_max_clamp", "gain_min_clamp"),
        indirect=True,
    )
    def test_gain_clamps(
        self,
        step_controller_settings,
        step_setup,
        device_step_results,
        tolerance,
    ):
        dt0 = step_setup['dt0']
        min_gain = step_controller_settings['min_gain']
        max_gain = step_controller_settings['max_gain']
        expected = dt0 * (max_gain if float(step_setup['error'][0]) < 1e-6 else min_gain)
        assert device_step_results.dt == pytest.approx(
            expected,
            rel=tolerance.rel_tight,
            abs=tolerance.abs_tight,
        )

    @pytest.mark.parametrize(
        'step_setup',
        (
            {'dt0': 0.005, 'error': np.asarray([5e-4, 5e-4, 5e-4])},
            {'dt0': 0.005, 'error': np.asarray([5e-3, 5e-3, 5e-3])},
            {'dt0': 0.005, 'error': np.asarray([5e-4, 5e-4, 5e-4]), 'local_mem': np.asarray([0.005, 0.8])},
            {'dt0': 0.005, 'error': np.asarray([5e-3, 5e-3, 5e-3]), 'local_mem': np.asarray([0.005, 0.8])},
        ),
        ids=("low_err", "high_err", "low_err_with_mem", "high_err_with_mem"),
        indirect=True,
    )
    @pytest.mark.parametrize('solver_settings_override',
                             [{'dt_min': 1e-6}], indirect=True)
    def test_matches_cpu(
        self,
        step_controller,
        step_controller_settings,
        step_setup,
        cpu_step_results,
        device_step_results,
        tolerance,
    ):
        assert device_step_results.dt == pytest.approx(
            cpu_step_results.dt,
            rel=tolerance.rel_tight,
            abs=tolerance.abs_tight,
        )
        valid_localmem = step_controller.local_memory_elements
        assert np.allclose(
            device_step_results.local_mem[:valid_localmem],
            cpu_step_results.local_mem[:valid_localmem],
            rtol=tolerance.rel_tight,
            atol=tolerance.abs_tight,
        )

    @pytest.mark.parametrize('solver_settings_override',
                             ({'dt_min': 1e-4, 'dt_max': 0.2},),
                             indirect=True)
    def test_cpu_gpu_sequence_agree(
        self,
        step_controller,
        cpu_step_controller,
        precision,
        system,
        tolerance,
    ):
        n_states = system.sizes.states
        dtype = precision
        cpu_controller = cpu_step_controller
        current_dt_cpu = float(step_controller.dt0)
        current_dt_gpu = float(step_controller.dt0)
        local_mem = np.zeros(
            step_controller.local_memory_elements,
            dtype=dtype,
        )
        base_state = np.linspace(
            0.5,
            0.5 + 0.1 * (n_states - 1),
            n_states,
            dtype=dtype,
        )

        error_values = (
            dtype(0.0),
            dtype(5.0e-4),
            dtype(1.4e-3),
            dtype(6.0e-4),
            dtype(1.6e-3),
        )
        delta_values = (
            dtype(2.0e-2),
            dtype(1.5e-2),
            dtype(1.0e-2),
            dtype(2.5e-2),
            dtype(1.5e-2),
        )
        niters_values = (1, 2, 1, 3, 2)

        current_state = base_state.copy()

        for i, (error_val, delta_val, niters) in enumerate(zip(
            error_values,
            delta_values,
            niters_values,
        )):
            state_prev = current_state.copy()
            state = state_prev + np.full(n_states, delta_val, dtype=dtype)
            error_vec = np.full(n_states, error_val, dtype=dtype)

            cpu_controller.dt = dtype(current_dt_cpu)
            accept_cpu = cpu_controller.propose_dt(
                prev_state=state_prev,
                new_state=state,
                error_vector=error_vec,
                niters=niters,
            )
            dt_cpu = float(cpu_controller.dt)

            device_result = _run_device_step(
                step_controller.device_function,
                dtype,
                dtype(current_dt_gpu),
                error_vec,
                state=state,
                state_prev=state_prev,
                local_mem=local_mem,
                niters=niters,
            )
            local_mem = device_result.local_mem.copy()
            current_dt_gpu = device_result.dt

            assert device_result.accepted == int(accept_cpu), f"Step {i} accept mismatch"
            assert current_dt_gpu == pytest.approx(
                dt_cpu,
                rel=tolerance.rel_tight,
                abs=tolerance.abs_tight,
            ), f"Step {i} dt mismatch"

            if accept_cpu:
                current_state = state

            current_dt_cpu = dt_cpu
