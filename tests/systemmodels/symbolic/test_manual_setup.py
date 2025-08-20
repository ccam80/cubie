import numpy as np
import pytest

from cubie.systemmodels.symbolic import setup_system


def manual_settings_defaults():
    return {
        "observables": ["obs"],
        "parameters": {"k": 1.0},
        "constants": {},
        "drivers": ["d"],
        "states": {"x": 1.0},
        "dxdt": ["obs = k * x * d", "dx = obs"],
    }


@pytest.fixture(scope="function")
def manual_settings_override(request):
    return request.param if hasattr(request, "param") else {}


@pytest.fixture(scope="function")
def manual_settings(manual_settings_override):
    settings = manual_settings_defaults()
    settings.update(manual_settings_override)
    return settings


def test_setup_system(manual_settings, precision):
    system = setup_system(**manual_settings, precision=precision)
    system.build()
    states = np.array([1.0], dtype=precision)
    params = np.array([2.0], dtype=precision)
    drivers = np.array([3.0], dtype=precision)
    dxdt, obs = system.correct_answer_python(states, params, drivers)
    assert dxdt[0] == pytest.approx(6.0)
    assert obs[0] == pytest.approx(6.0)
