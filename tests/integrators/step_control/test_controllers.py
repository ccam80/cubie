import numpy as np
import pytest
from numba import cuda

from cubie.integrators.step_control import (
    AdaptiveIController,
    AdaptivePIController,
    AdaptivePIDController,
    GustafssonController,
    get_controller,
)

from tests.integrators.step_control.cpu_controllers import (
    CPUAdaptiveIController,
    CPUAdaptivePIController,
    CPUAdaptivePIDController,
    CPUGustafssonController,
)


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
    return float(dt[0]), int(accept[0]), temp.copy()


def _make_controller_pair(factory, precision, tolerances, params=None):
    """Create matching device and CPU controllers."""

    if params is None:
        params = {}
    dt_min = 1e-6
    dt_max = 1e-2
    base_device = {
        "precision": precision,
        "dt_min": dt_min,
        "dt_max": dt_max,
        "n": 1,
        "atol": tolerances["atol"],
        "rtol": tolerances["rtol"],
        "algorithm_order": tolerances["order"],
        "min_gain": tolerances["min_gain"],
        "max_gain": tolerances["max_gain"],
    }
    base_device.update(params)
    device = factory(**base_device)

    base_cpu = {
        "precision": precision,
        "dt_min": dt_min,
        "dt_max": dt_max,
        "atol": tolerances["atol"],
        "rtol": tolerances["rtol"],
        "order": tolerances["order"],
        "safety": tolerances["safety"],
        "min_gain": tolerances["min_gain"],
        "max_gain": tolerances["max_gain"],
    }
    if factory is AdaptivePIController:
        kp = params.get("kp", 0.7)
        ki = params.get("ki", 0.4)
        cpu = CPUAdaptivePIController(kp=kp, ki=ki, **base_cpu)
    elif factory is AdaptivePIDController:
        kp = params.get("kp", 0.7)
        ki = params.get("ki", 0.4)
        kd = params.get("kd", 0.2)
        cpu = CPUAdaptivePIDController(kp=kp, ki=ki, kd=kd, **base_cpu)
    elif factory is GustafssonController:
        cpu = CPUGustafssonController(**base_cpu)
    else:
        cpu = CPUAdaptiveIController(**base_cpu)
    return device, cpu, dt_min, dt_max


def _execute_pair(device, cpu, precision, dt0, error, *, local_mem=None):
    """Run CPU and device controllers and assert identical behaviour."""

    err = np.asarray(error, dtype=precision)
    cpu_local = None
    device_local = None
    if local_mem is not None:
        device_local = np.asarray(local_mem, dtype=precision)
        cpu_local = device_local.copy()

    cpu_result = cpu.step(
        dt=float(dt0),
        error=err,
        state=None,
        state_prev=None,
        local_mem=cpu_local,
    )
    dt_device, accept_device, local_device = _run_device_step(
        device.device_function,
        precision,
        precision(dt0),
        err,
        local_mem=device_local,
    )
    assert accept_device == cpu_result.accepted
    assert dt_device == pytest.approx(cpu_result.dt)
    np.testing.assert_allclose(
        local_device, cpu_result.local_mem, rtol=1e-6, atol=0.0
    )
    return cpu_result


def _copy_local(local):
    """Return a copy of local memory or ``None``."""

    if local is None:
        return None
    return [float(val) for val in np.asarray(local).tolist()]


@pytest.mark.parametrize(
    "name, expected",
    [
        ("i", AdaptiveIController),
        ("pi", AdaptivePIController),
        ("pid", AdaptivePIDController),
        ("gustafsson", GustafssonController),
    ],
)
def test_get_controller(name, expected, precision):
    controller = get_controller(
        name, precision=precision, dt_min=1e-6, dt_max=1e-3, n=1
    )
    assert isinstance(controller, expected)


@pytest.mark.parametrize(
    "name, expected",
    [
        ("i", AdaptiveIController),
        ("pi", AdaptivePIController),
        ("pid", AdaptivePIDController),
        ("gustafsson", GustafssonController),
    ],
)
def test_controller_builds(name, expected, precision):
    controller = get_controller(
        name, precision=precision, dt_min=1e-6, dt_max=1e-3, n=1
    )
    assert callable(controller.device_function)


def test_i_controller_matches_device(precision, controller_tolerances):
    device, cpu, _, _ = _make_controller_pair(
        AdaptiveIController, precision, controller_tolerances
    )
    dt0 = float(precision(1e-4))
    error = [float(precision(2e-6))]
    result = _execute_pair(device, cpu, precision, dt0, error)
    assert result.accepted == 0
    assert result.dt < dt0


@pytest.mark.parametrize(
    ("error_value", "factor", "expected_accept"),
    [
        (5e-3, "min_gain", 0),
        (1e-10, "max_gain", 1),
    ],
)
def test_i_controller_gain_bounds(
    error_value, factor, expected_accept, precision, controller_tolerances
):
    device, cpu, _, _ = _make_controller_pair(
        AdaptiveIController, precision, controller_tolerances
    )
    dt0 = float(precision(1e-4))
    error = [float(precision(error_value))]
    result = _execute_pair(device, cpu, precision, dt0, error)
    expected = dt0 * controller_tolerances[factor]
    assert result.dt == pytest.approx(expected)
    assert result.accepted == expected_accept


def test_i_controller_clamps_dt_min(precision, controller_tolerances):
    device, cpu, dt_min, _ = _make_controller_pair(
        AdaptiveIController, precision, controller_tolerances
    )
    error = [float(precision(5e-3))]
    result = _execute_pair(device, cpu, precision, dt_min, error)
    assert result.dt == pytest.approx(dt_min)
    assert result.accepted == 0


def test_pi_controller_previous_error(precision, controller_tolerances):
    params = {"kp": 0.7, "ki": 0.4}
    device, cpu, _, _ = _make_controller_pair(
        AdaptivePIController, precision, controller_tolerances, params
    )
    dt0 = float(precision(1e-4))
    error = [float(precision(2e-6))]
    high = _execute_pair(device, cpu, precision, dt0, error, local_mem=[1.0])
    low = _execute_pair(device, cpu, precision, dt0, error, local_mem=[0.25])
    assert high.dt > low.dt
    assert high.accepted == low.accepted == 0
    np.testing.assert_allclose(high.local_mem, low.local_mem)


@pytest.mark.parametrize(
    ("error_value", "factor", "expected_accept"),
    [
        (5e-3, "min_gain", 0),
        (1e-10, "max_gain", 1),
    ],
)
def test_pi_controller_gain_bounds(
    error_value, factor, expected_accept, precision, controller_tolerances
):
    params = {"kp": 0.7, "ki": 0.4}
    device, cpu, _, _ = _make_controller_pair(
        AdaptivePIController, precision, controller_tolerances, params
    )
    dt0 = float(precision(1e-4))
    error = [float(precision(error_value))]
    result = _execute_pair(device, cpu, precision, dt0, error, local_mem=[1.0])
    expected = dt0 * controller_tolerances[factor]
    assert result.dt == pytest.approx(expected)
    assert result.accepted == expected_accept


def test_pid_controller_error_history(precision, controller_tolerances):
    params = {"kp": 0.7, "ki": 0.4, "kd": 0.2}
    device, cpu, _, _ = _make_controller_pair(
        AdaptivePIDController, precision, controller_tolerances, params
    )
    dt0 = float(precision(1e-4))
    error = [float(precision(2e-6))]
    full_history = _execute_pair(
        device, cpu, precision, dt0, error, local_mem=[1.0, 1.0]
    )
    damped_history = _execute_pair(
        device, cpu, precision, dt0, error, local_mem=[1.0, 0.25]
    )
    assert full_history.dt > damped_history.dt
    assert full_history.accepted == damped_history.accepted == 0
    assert full_history.local_mem[1] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("error_value", "factor", "expected_accept"),
    [
        (5e-3, "min_gain", 0),
        (1e-10, "max_gain", 1),
    ],
)
def test_pid_controller_gain_bounds(
    error_value, factor, expected_accept, precision, controller_tolerances
):
    params = {"kp": 0.7, "ki": 0.4, "kd": 0.2}
    device, cpu, _, _ = _make_controller_pair(
        AdaptivePIDController, precision, controller_tolerances, params
    )
    dt0 = float(precision(1e-4))
    error = [float(precision(error_value))]
    result = _execute_pair(
        device, cpu, precision, dt0, error, local_mem=[1.0, 1.0]
    )
    expected = dt0 * controller_tolerances[factor]
    assert result.dt == pytest.approx(expected)
    assert result.accepted == expected_accept


def test_gustafsson_predictive_history(precision, controller_tolerances):
    device, cpu, _, _ = _make_controller_pair(
        GustafssonController, precision, controller_tolerances
    )
    dt0 = float(precision(1e-4))
    tol = float(controller_tolerances["atol"][0])
    error_first = [float(precision(tol / 2.0))]
    history = _execute_pair(
        device, cpu, precision, dt0, error_first, local_mem=[0.0, 0.0]
    )
    dt1 = history.dt
    error_second = [float(precision(tol / np.sqrt(2.0)))]
    predictive = _execute_pair(
        device,
        cpu,
        precision,
        dt1,
        error_second,
        local_mem=history.local_mem.copy(),
    )
    baseline = _execute_pair(
        device,
        cpu,
        precision,
        dt1,
        error_second,
        local_mem=[0.0, 0.0],
    )
    assert predictive.accepted == baseline.accepted == 1
    assert predictive.dt <= baseline.dt


@pytest.mark.parametrize(
    ("error_value", "factor", "expected_accept"),
    [
        (5e-3, "min_gain", 0),
        (5e-9, "max_gain", 1),
    ],
)
def test_gustafsson_gain_bounds(
    error_value, factor, expected_accept, precision, controller_tolerances
):
    device, cpu, _, _ = _make_controller_pair(
        GustafssonController, precision, controller_tolerances
    )
    dt0 = float(precision(1e-4))
    error = [float(precision(error_value))]
    result = _execute_pair(
        device, cpu, precision, dt0, error, local_mem=[0.0, 0.0]
    )
    expected = dt0 * controller_tolerances[factor]
    assert result.dt == pytest.approx(expected)
    assert result.accepted == expected_accept


@pytest.mark.parametrize(
    "factory, params, template",
    [
        (AdaptiveIController, {}, lambda dt: None),
        (AdaptivePIController, {"kp": 0.7, "ki": 0.4}, lambda dt: [1.0]),
        (
            AdaptivePIDController,
            {"kp": 0.7, "ki": 0.4, "kd": 0.2},
            lambda dt: [1.0, 1.0],
        ),
        (GustafssonController, {}, lambda dt: [dt, 1.0]),
    ],
)
def test_controllers_converge(
    factory, params, template, precision, controller_tolerances
):
    device, cpu, _, _ = _make_controller_pair(
        factory, precision, controller_tolerances, params
    )
    dt0 = float(precision(1e-4))
    err_high = [float(precision(5e-6))]
    err_low = [float(precision(5e-7))]
    local_high = _copy_local(template(dt0))
    local_low = _copy_local(template(dt0))
    down = _execute_pair(
        device, cpu, precision, dt0, err_high, local_mem=local_high
    )
    up = _execute_pair(
        device, cpu, precision, dt0, err_low, local_mem=local_low
    )
    assert down.dt < dt0
    assert up.dt > dt0
