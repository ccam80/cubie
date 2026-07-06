"""Tests for cubie.batchsolving.SystemInterface."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from cubie.batchsolving.SystemInterface import SystemInterface
from cubie.odesystems.SystemValues import SystemValues


# ── __init__ and from_system ─────────────────────────────── #

def test_init_stores_system_values(system_interface, system):
    """__init__ stores parameters, states, observables as SystemValues."""
    assert system_interface.parameters is system.parameters
    assert system_interface.states is system.initial_values
    assert system_interface.observables is system.observables


def test_from_system_creates_interface(system):
    """from_system wraps system.parameters, initial_values, observables."""
    si = SystemInterface.from_system(system)
    assert si.parameters is system.parameters
    assert si.states is system.initial_values
    assert si.observables is system.observables


# ── update ───────────────────────────────────────────────── #

def test_update_returns_none_when_no_updates(system_interface):
    """Returns None when updates is None and no kwargs."""
    result = system_interface.update(None)
    assert result is None


def test_update_merges_kwargs(system_interface_mutable, system):
    """Merges kwargs into updates dict and applies them."""
    name = system.parameters.names[0]
    original = system_interface_mutable.parameters.values_dict[name]
    new_val = original + 1.0
    recognized = system_interface_mutable.update({}, **{name: new_val})
    assert name in recognized
    assert system_interface_mutable.parameters.values_dict[name] == new_val


def test_update_applies_to_parameters_and_states(
    system_interface_mutable, system,
):
    """Attempts update on both parameters and states."""
    param_name = system.parameters.names[0]
    state_name = system.initial_values.names[0]
    recognized = system_interface_mutable.update(
        {param_name: 99.0, state_name: 88.0}
    )
    assert param_name in recognized
    assert state_name in recognized


def test_update_raises_on_unrecognized_keys(system_interface):
    """Raises KeyError when unrecognized keys and silent=False."""
    with pytest.raises(KeyError, match="not recognized"):
        system_interface.update({"__bogus_key__": 1.0})


def test_update_silent_returns_recognized(system_interface_mutable, system):
    """Returns recognized keys set when silent=True."""
    param_name = system.parameters.names[0]
    recognized = system_interface_mutable.update(
        {param_name: 1.0, "__bogus__": 2.0}, silent=True
    )
    assert param_name in recognized
    assert "__bogus__" not in recognized


# ── state_indices / observable_indices / parameter_indices ── #

def test_state_indices_none_returns_all(system_interface, system):
    """Returns all state indices when keys_or_indices is None."""
    result = system_interface.state_indices(None)
    expected = np.arange(len(system.initial_values.names), dtype=np.int32)
    assert_array_equal(result, expected)


def test_state_indices_by_name(system_interface, system):
    """Delegates to states.get_indices for named keys."""
    names = system.initial_values.names[:1]
    result = system_interface.state_indices(names)
    assert_array_equal(result, np.array([0], dtype=np.int32))


def test_observable_indices_none_returns_all(system_interface, system):
    """Returns all observable indices when keys_or_indices is None."""
    result = system_interface.observable_indices(None)
    expected = np.arange(len(system.observables.names), dtype=np.int32)
    assert_array_equal(result, expected)


def test_observable_indices_by_name(system_interface, system):
    """Delegates to observables.get_indices for named keys."""
    names = system.observables.names[:1]
    result = system_interface.observable_indices(names)
    assert_array_equal(result, np.array([0], dtype=np.int32))


def test_parameter_indices_by_name(system_interface, system):
    """Delegates to parameters.get_indices."""
    names = system.parameters.names[:1]
    result = system_interface.parameter_indices(names)
    assert_array_equal(result, np.array([0], dtype=np.int32))


# ── get_labels / state_labels / observable_labels / parameter_labels ── #

def test_get_labels_delegates(system_interface, system):
    """Delegates to values_object.get_labels(indices)."""
    indices = np.array([0], dtype=np.int32)
    result = system_interface.get_labels(system_interface.states, indices)
    assert result == [system.initial_values.names[0]]


@pytest.mark.parametrize(
    "method, expected_source",
    [
        ("state_labels", "initial_values"),
        ("observable_labels", "observables"),
        ("parameter_labels", "parameters"),
    ],
    ids=["states", "observables", "parameters"],
)
def test_labels_none_returns_all(
    system_interface, system, method, expected_source,
):
    """Returns all names when indices is None."""
    result = getattr(system_interface, method)(None)
    source = getattr(system, expected_source)
    # initial_values is the states source
    if expected_source == "initial_values":
        assert result == source.names
    else:
        assert result == source.names


@pytest.mark.parametrize(
    "method, expected_source",
    [
        ("state_labels", "initial_values"),
        ("observable_labels", "observables"),
        ("parameter_labels", "parameters"),
    ],
    ids=["states", "observables", "parameters"],
)
def test_labels_with_indices(
    system_interface, system, method, expected_source,
):
    """Delegates to get_labels with provided indices."""
    source = getattr(system, expected_source)
    indices = np.array([0], dtype=np.int32)
    result = getattr(system_interface, method)(indices)
    assert result == [source.names[0]]


# ── resolve_variable_labels ──────────────────────────────── #

def test_resolve_variable_labels_none(system_interface):
    """Returns (None, None) when labels is None."""
    result = system_interface.resolve_variable_labels(None)
    assert result == (None, None)


def test_resolve_variable_labels_empty(system_interface):
    """Returns (empty, empty) int32 arrays when labels is empty list."""
    s, o = system_interface.resolve_variable_labels([])
    assert s.dtype == np.int32
    assert o.dtype == np.int32
    assert len(s) == 0
    assert len(o) == 0


def test_resolve_variable_labels_resolves(system_interface, system):
    """Resolves labels to state and observable indices."""
    state_name = system.initial_values.names[0]
    obs_name = system.observables.names[0]
    s, o = system_interface.resolve_variable_labels(
        [state_name, obs_name]
    )
    assert 0 in s
    assert 0 in o
    assert s.dtype == np.int32
    assert o.dtype == np.int32


def test_resolve_variable_labels_error_on_unknown(system_interface):
    """Raises ValueError for unresolved labels when silent=False."""
    with pytest.raises(ValueError, match="not found"):
        system_interface.resolve_variable_labels(["__nonexistent__"])


# ── merge_variable_inputs ────────────────────────────────── #

def test_merge_variable_inputs_all_none(system_interface, system):
    """Returns full range when all three inputs None."""
    s, o = system_interface.merge_variable_inputs(None, None, None)
    expected_s = np.arange(len(system.initial_values.names), dtype=np.int32)
    expected_o = np.arange(len(system.observables.names), dtype=np.int32)
    assert_array_equal(s, expected_s)
    assert_array_equal(o, expected_o)


def test_merge_variable_inputs_union(system_interface, system):
    """Computes union of label-resolved and directly-provided indices."""
    state_name = system.initial_values.names[0]
    # Provide index 1 directly, label resolves to index 0
    s, o = system_interface.merge_variable_inputs(
        [state_name], np.array([1], dtype=np.int32), None,
    )
    # Union should contain both 0 and 1
    assert 0 in s
    assert 1 in s


def test_merge_variable_inputs_none_replaced_with_empty(
    system_interface, system,
):
    """Replaces None inputs with empty arrays (not full range)."""
    # Only provide state indices, no labels, no obs indices
    s, o = system_interface.merge_variable_inputs(
        None, np.array([0], dtype=np.int32), None,
    )
    # var_labels=None but state_indices provided => not all_none path
    # obs should be empty since obs_from_labels=None->empty, obs_from_indices=None->empty
    assert_array_equal(s, np.array([0], dtype=np.int32))
    assert len(o) == 0


# ── merge_variable_labels_and_idxs ───────────────────────── #

def test_merge_variable_labels_and_idxs_pops_keys(system_interface):
    """Pops save_variables and summarise_variables from dict."""
    settings = {
        "save_variables": None,
        "summarise_variables": None,
    }
    system_interface.merge_variable_labels_and_idxs(settings)
    assert "save_variables" not in settings
    assert "summarise_variables" not in settings


def test_merge_variable_labels_and_idxs_defaults_summarise_to_saved(
    system_interface, system,
):
    """Defaults summarise indices to saved indices when all None."""
    settings = {}
    system_interface.merge_variable_labels_and_idxs(settings)
    # All None => full range for save, summarise defaults to save
    expected_s = np.arange(len(system.initial_values.names), dtype=np.int32)
    expected_o = np.arange(len(system.observables.names), dtype=np.int32)
    assert_array_equal(settings["saved_state_indices"], expected_s)
    assert_array_equal(settings["saved_observable_indices"], expected_o)
    assert_array_equal(settings["summarised_state_indices"], expected_s)
    assert_array_equal(settings["summarised_observable_indices"], expected_o)


def test_merge_variable_labels_and_idxs_separate_summarise(
    system_interface, system,
):
    """Calls merge_variable_inputs for summarise when not all None."""
    settings = {
        "summarised_state_indices": np.array([0], dtype=np.int32),
    }
    system_interface.merge_variable_labels_and_idxs(settings)
    # Summarised should use explicit index, not default to saved
    assert_array_equal(
        settings["summarised_state_indices"],
        np.array([0], dtype=np.int32),
    )


def test_merge_variable_labels_and_idxs_updates_all_four_keys(
    system_interface,
):
    """Updates dict in-place with final index arrays."""
    settings = {}
    system_interface.merge_variable_labels_and_idxs(settings)
    for key in (
        "saved_state_indices",
        "saved_observable_indices",
        "summarised_state_indices",
        "summarised_observable_indices",
    ):
        assert key in settings
        assert settings[key].dtype == np.int32


# ── Properties ───────────────────────────────────────────── #

def test_all_input_labels(system_interface, system):
    """all_input_labels = state_labels() + parameter_labels()."""
    expected = (
        list(system.initial_values.names) + list(system.parameters.names)
    )
    assert system_interface.all_input_labels == expected


def test_all_output_labels(system_interface, system):
    """all_output_labels = state_labels() + observable_labels()."""
    expected = (
        list(system.initial_values.names) + list(system.observables.names)
    )
    assert system_interface.all_output_labels == expected
