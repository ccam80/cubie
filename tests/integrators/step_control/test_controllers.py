import pytest

from cubie import is_devfunc
from cubie.integrators.step_control import (
    AdaptiveIController,
    AdaptivePIController,
    AdaptivePIDController,
    GustafssonController,
    get_controller,
)


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
    assert is_devfunc(controller.device_function)
