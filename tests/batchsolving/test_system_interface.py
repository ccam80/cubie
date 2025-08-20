import numpy as np
import pytest

from cubie.batchsolving.SystemInterface import SystemInterface


@pytest.fixture(scope="function")
def interface(system):
    return SystemInterface.from_system(system)


def test_from_system(interface):
    assert hasattr(interface, "parameters")
    assert hasattr(interface, "states")
    assert hasattr(interface, "observables")


@pytest.mark.parametrize(
    "updates",
    [
        {"x0": 42.0},
        {"p0": 3.14},
        {"x0": 1.23, "p1": 4.56},
    ],
)
def test_update(interface, updates):
    interface.update(updates)
    for k, v in updates.items():
        if k in interface.states:
            assert np.isclose(interface.states[k], v)
        elif k in interface.parameters:
            assert np.isclose(interface.parameters[k], v)


@pytest.mark.parametrize(
    "updates",
    [{"not_a_key": 1.0}, {"x0": 1.0, "bad_param": 2.0}],
)
def test_update_invalid(interface, updates):
    with pytest.raises(KeyError):
        interface.update(updates)


def test_state_indices(interface, system):
    idx_by_name = interface.state_indices(system.initial_values.names)
    idx_by_idx = interface.state_indices(list(range(system.sizes.states)))
    assert np.all(idx_by_name == np.arange(system.sizes.states))
    assert np.all(idx_by_idx == np.arange(system.sizes.states))


def test_observable_indices(interface, system):
    idx_by_name = interface.observable_indices(system.observables.names)
    idx_by_idx = interface.observable_indices(
        list(range(system.sizes.observables))
    )
    assert np.all(idx_by_name == np.arange(system.sizes.observables))
    assert np.all(idx_by_idx == np.arange(system.sizes.observables))


def test_parameter_indices(interface, system):
    idx_by_name = interface.parameter_indices(system.parameters.names)
    idx_by_idx = interface.parameter_indices(
        list(range(system.sizes.parameters))
    )
    assert np.all(idx_by_name == np.arange(system.sizes.parameters))
    assert np.all(idx_by_idx == np.arange(system.sizes.parameters))
