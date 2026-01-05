import numpy as np
import pytest

from cubie.batchsolving.SystemInterface import SystemInterface


@pytest.fixture(scope="session")
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


# =============================================================================
# Tests for resolve_variable_labels
# =============================================================================

def test_resolve_variable_labels_none_returns_none_tuple(interface):
    state_idx, obs_idx = interface.resolve_variable_labels(None)
    assert state_idx is None
    assert obs_idx is None


def test_resolve_variable_labels_empty_returns_empty_arrays(interface):
    state_idx, obs_idx = interface.resolve_variable_labels([])
    assert len(state_idx) == 0
    assert len(obs_idx) == 0
    assert state_idx.dtype == np.int32
    assert obs_idx.dtype == np.int32


def test_resolve_variable_labels_states_only(interface, system):
    state_names = list(system.initial_values.names)
    state_idx, obs_idx = interface.resolve_variable_labels(state_names)
    assert len(state_idx) == len(state_names)
    assert len(obs_idx) == 0


def test_resolve_variable_labels_observables_only(interface, system):
    obs_names = list(system.observables.names)
    state_idx, obs_idx = interface.resolve_variable_labels(obs_names)
    assert len(state_idx) == 0
    assert len(obs_idx) == len(obs_names)


def test_resolve_variable_labels_mixed(interface, system):
    state_names = list(system.initial_values.names)
    obs_names = list(system.observables.names)
    mixed = [state_names[0], obs_names[0]]
    state_idx, obs_idx = interface.resolve_variable_labels(mixed)
    assert len(state_idx) == 1
    assert len(obs_idx) == 1
    assert 0 in state_idx
    assert 0 in obs_idx


def test_resolve_variable_labels_invalid_raises(interface):
    with pytest.raises(ValueError, match="Variables not found"):
        interface.resolve_variable_labels(["not_a_real_variable"])


def test_resolve_variable_labels_silent_mode(interface):
    state_idx, obs_idx = interface.resolve_variable_labels(
        ["not_a_real_variable"], silent=True
    )
    assert len(state_idx) == 0
    assert len(obs_idx) == 0


def test_resolve_variable_labels_partial_invalid_raises(interface, system):
    state_names = list(system.initial_values.names)
    mixed_labels = [state_names[0], "invalid_label"]
    with pytest.raises(ValueError, match="Variables not found"):
        interface.resolve_variable_labels(mixed_labels, silent=False)


# =============================================================================
# Tests for merge_variable_inputs
# =============================================================================

def test_merge_variable_inputs_all_none_returns_full(interface, system):
    max_states = system.sizes.states
    max_obs = system.sizes.observables
    state_idx, obs_idx = interface.merge_variable_inputs(
        None, None, None, max_states, max_obs
    )
    assert np.array_equal(state_idx, np.arange(max_states, dtype=np.int32))
    assert np.array_equal(obs_idx, np.arange(max_obs, dtype=np.int32))


def test_merge_variable_inputs_empty_labels(interface, system):
    max_states = system.sizes.states
    max_obs = system.sizes.observables
    state_idx, obs_idx = interface.merge_variable_inputs(
        [], None, None, max_states, max_obs
    )
    assert len(state_idx) == 0
    assert len(obs_idx) == 0


def test_merge_variable_inputs_empty_indices(interface, system):
    max_states = system.sizes.states
    max_obs = system.sizes.observables
    state_idx, obs_idx = interface.merge_variable_inputs(
        None, [], None, max_states, max_obs
    )
    assert len(state_idx) == 0
    assert np.array_equal(obs_idx, np.array([], dtype=np.int32))


def test_merge_variable_inputs_union(interface, system):
    max_states = system.sizes.states
    max_obs = system.sizes.observables
    state_names = list(system.initial_values.names)
    state_idx, obs_idx = interface.merge_variable_inputs(
        [state_names[0]], [1], None, max_states, max_obs
    )
    assert 0 in state_idx
    assert 1 in state_idx
    assert len(state_idx) == 2


def test_merge_variable_inputs_deduplication(interface, system):
    max_states = system.sizes.states
    max_obs = system.sizes.observables
    state_names = list(system.initial_values.names)
    state_idx, obs_idx = interface.merge_variable_inputs(
        [state_names[0]], [0], None, max_states, max_obs
    )
    assert len(state_idx) == 1
    assert state_idx[0] == 0


# =============================================================================
# Tests for convert_variable_labels
# =============================================================================

def test_convert_variable_labels_mutates_dict(interface, system):
    max_states = system.sizes.states
    max_obs = system.sizes.observables
    settings = {}
    interface.convert_variable_labels(settings, max_states, max_obs)
    assert "saved_state_indices" in settings
    assert "saved_observable_indices" in settings
    assert "summarised_state_indices" in settings
    assert "summarised_observable_indices" in settings


def test_convert_variable_labels_pops_label_keys(interface, system):
    max_states = system.sizes.states
    max_obs = system.sizes.observables
    state_names = list(system.initial_values.names)
    settings = {
        "save_variables": [state_names[0]],
        "summarise_variables": [state_names[0]],
    }
    interface.convert_variable_labels(settings, max_states, max_obs)
    assert "save_variables" not in settings
    assert "summarise_variables" not in settings


def test_convert_variable_labels_summarised_defaults_to_saved(interface,
                                                               system):
    max_states = system.sizes.states
    max_obs = system.sizes.observables
    state_names = list(system.initial_values.names)
    settings = {"save_variables": [state_names[0]]}
    interface.convert_variable_labels(settings, max_states, max_obs)
    assert np.array_equal(
        settings["saved_state_indices"],
        settings["summarised_state_indices"]
    )
    assert np.array_equal(
        settings["saved_observable_indices"],
        settings["summarised_observable_indices"]
    )
