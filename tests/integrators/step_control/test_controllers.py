
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


def _run_device_step(
    device_func,
    precision,
    dt0,
    error,
    local_mem=None,
):
    """Execute controller device function once.

    Parameters
    ----------
    device_func : Callable
        Compiled controller device function.
    precision : type
        Floating point precision for arrays.
    dt0 : float
        Initial step size.
    error : float
        Estimated local error.
    local_mem : list[float] | None, optional
        Controller local memory.

    Returns
    -------
    tuple[float, int, np.ndarray]
        Updated step, accept flag and local memory.
    """

    state = np.zeros(1, dtype=precision)
    state_prev = np.zeros(1, dtype=precision)
    err = np.array([error], dtype=precision)
    dt = np.array([dt0], dtype=precision)
    accept = np.zeros(1, dtype=np.int32)
    scaled = np.zeros(1, dtype=precision)
    if local_mem is None:
        temp = np.empty(0, dtype=precision)
    else:
        temp = np.array(local_mem, dtype=precision)

    @cuda.jit
    def kernel(dt, state, state_prev, err, accept, scaled, temp):
        device_func(dt, state, state_prev, err, accept, scaled, temp)

    kernel[1, 1](dt, state, state_prev, err, accept, scaled, temp)
    return float(dt[0]), int(accept[0]), temp


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


def test_i_controller_step(precision):
    controller = AdaptiveIController(
        precision=precision, dt_min=1e-6, dt_max=1e-2, n=1
    )
    dt0 = precision(1e-4)
    error = precision(2e-6)
    tol = controller.atol[0]
    order = controller.compile_settings.algorithm_order
    order_exp = precision(1.0 / (2 * (1 + order)))
    nrm2 = (tol / error) ** 2
    gain_tmp = controller.compile_settings.safety * (nrm2 ** order_exp)
    gain = max(controller.min_gain, min(gain_tmp, controller.max_gain))
    expected = dt0 * gain
    dt_new, accepted, _ = _run_device_step(
        controller.device_function, precision, dt0, error
    )
    assert accepted == 0
    assert dt_new == pytest.approx(expected)


def test_pi_controller_uses_previous_error(precision):
    controller = AdaptivePIController(
        precision=precision, dt_min=1e-6, dt_max=1e-2, n=1
    )
    dt0 = precision(1e-4)
    error = precision(2e-6)
    tol = controller.atol[0]
    order = controller.compile_settings.algorithm_order
    kp = controller.kp / (order + 1)
    ki = controller.ki / (order + 1)
    nrm2 = (tol / error) ** 2
    pgain = nrm2 ** (kp / 2)
    igain1 = precision(1.0) ** (ki / 2)
    igain2 = precision(0.25) ** (ki / 2)
    gain1 = controller.compile_settings.safety * pgain * igain1
    gain2 = controller.compile_settings.safety * pgain * igain2
    expected1 = dt0 * max(
        controller.min_gain, min(gain1, controller.max_gain)
    )
    expected2 = dt0 * max(
        controller.min_gain, min(gain2, controller.max_gain)
    )
    dt1, _, _ = _run_device_step(
        controller.device_function, precision, dt0, error, [1.0]
    )
    dt2, _, _ = _run_device_step(
        controller.device_function, precision, dt0, error, [0.25]
    )
    assert dt1 > dt2
    assert dt1 == pytest.approx(expected1)
    assert dt2 == pytest.approx(expected2)


def test_pid_controller_uses_two_errors(precision):
    controller = AdaptivePIDController(
        precision=precision, dt_min=1e-6, dt_max=1e-2, n=1, kd=0.2
    )
    dt0 = precision(1e-4)
    error = precision(2e-6)
    tol = controller.atol[0]
    order = controller.compile_settings.algorithm_order
    expo1 = controller.kp / (2 * (order + 1))
    expo2 = controller.ki / (2 * (order + 1))
    expo3 = controller.kd / (2 * (order + 1))
    nrm2 = (tol / error) ** 2
    gain1 = controller.compile_settings.safety * (
        (nrm2 ** expo1) * (1.0 ** expo2) * (1.0 ** expo3)
    )
    gain2 = controller.compile_settings.safety * (
        (nrm2 ** expo1) * (1.0 ** expo2) * (0.25 ** expo3)
    )
    expected1 = dt0 * max(
        controller.min_gain, min(gain1, controller.max_gain)
    )
    expected2 = dt0 * max(
        controller.min_gain, min(gain2, controller.max_gain)
    )
    dt1, _, _ = _run_device_step(
        controller.device_function, precision, dt0, error, [1.0, 1.0]
    )
    dt2, _, _ = _run_device_step(
        controller.device_function, precision, dt0, error, [1.0, 0.25]
    )
    assert dt1 > dt2
    assert dt1 == pytest.approx(expected1)
    assert dt2 == pytest.approx(expected2)


def test_gustafsson_predictive_step(precision):
    controller = GustafssonController(
        precision=precision, dt_min=1e-6, dt_max=1e-2, n=1
    )
    dt0 = precision(1e-4)
    error = precision(5e-7)
    tol = controller.atol[0]
    order = controller.compile_settings.algorithm_order
    expo = precision(1.0 / (2 * (1 + order)))
    nrm2 = (tol / error) ** 2
    gain_basic = controller.compile_settings.safety * (nrm2 ** expo)
    gain_basic = max(
        controller.min_gain, min(gain_basic, controller.max_gain)
    )
    expected_basic = dt0 * gain_basic
    dt_basic, _, _ = _run_device_step(
        controller.device_function, precision, dt0, error, [0.0, 0.0]
    )
    dt_pred, _, _ = _run_device_step(
        controller.device_function, precision, dt0, error, [dt0, nrm2]
    )
    assert dt_basic == pytest.approx(expected_basic)
    assert dt_pred < dt_basic


@pytest.mark.parametrize(
    "factory",
    [
        AdaptiveIController,
        AdaptivePIController,
        AdaptivePIDController,
        GustafssonController,
    ],
)
def test_controllers_converge(factory, precision):
    controller = factory(precision=precision, dt_min=1e-6, dt_max=1e-2, n=1)
    dt0 = precision(1e-4)
    err_high = precision(5e-6)
    err_low = precision(5e-7)
    if isinstance(controller, GustafssonController):
        local_high = [dt0, 1.0]
        local_low = [dt0, 1.0]
    else:
        local_high = [1.0] * controller.local_memory_required
        local_low = [1.0] * controller.local_memory_required
    dt_down, _, _ = _run_device_step(
        controller.device_function, precision, dt0, err_high, local_high
    )
    dt_up, _, _ = _run_device_step(
        controller.device_function, precision, dt0, err_low, local_low
    )
    assert dt_down < dt0
    assert dt_up > dt0
