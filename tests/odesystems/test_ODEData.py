"""Tests for cubie.odesystems.ODEData."""

from __future__ import annotations

import numpy as np
import pytest

from cubie.odesystems.ODEData import ODEData, SystemSizes
from cubie.odesystems.SystemValues import SystemValues


# ── SystemSizes ───────────────────────────────────────────────── #

def test_system_sizes_construction():
    """All fields stored correctly on frozen attrs class."""
    sizes = SystemSizes(
        states=3, observables=2, parameters=4, constants=5, drivers=1,
    )
    assert sizes.states == 3
    assert sizes.observables == 2
    assert sizes.parameters == 4
    assert sizes.constants == 5
    assert sizes.drivers == 1


@pytest.mark.parametrize(
    "field, bad_value",
    [
        ("states", 1.5),
        ("observables", "x"),
        ("parameters", None),
        ("constants", [1]),
        ("drivers", 2.0),
    ],
    ids=["states", "observables", "parameters", "constants", "drivers"],
)
def test_system_sizes_validates_int(field, bad_value):
    """Each field rejects non-int values."""
    kwargs = dict(states=1, observables=1, parameters=1, constants=1, drivers=1)
    kwargs[field] = bad_value
    with pytest.raises(TypeError):
        SystemSizes(**kwargs)


# ── ODEData construction ──────────────────────────────────────── #

def _make_odedata(precision=np.float32, num_drivers=1):
    """Helper to create ODEData via from_BaseODE_initargs."""
    return ODEData.from_BaseODE_initargs(
        precision=precision,
        default_initial_values={"x": 0.0, "y": 1.0},
        default_parameters={"a": 0.5, "b": 0.3},
        default_constants={"g": 9.81},
        default_observable_names={"v": 0.0, "w": 0.0},
        num_drivers=num_drivers,
    )


def test_odedata_construction():
    """ODEData stores SystemValues for each component."""
    data = _make_odedata()
    assert data.initial_states.n == 2
    assert data.parameters.n == 2
    assert data.constants.n == 1
    assert data.observables.n == 2


# ── ODEData.update_precisions ─────────────────────────────────── #

def test_update_precisions_updates_all():
    """update_precisions propagates precision to all SystemValues."""
    data = _make_odedata(precision=np.float32)
    data.update_precisions({"precision": np.float64})
    assert data.parameters.precision == np.float64
    assert data.constants.precision == np.float64
    assert data.initial_states.precision == np.float64
    assert data.observables.precision == np.float64


def test_update_precisions_noop_without_key():
    """update_precisions leaves precision unchanged when key absent."""
    data = _make_odedata(precision=np.float32)
    data.update_precisions({"unrelated": 42})
    assert data.parameters.precision == np.float32


# ── ODEData properties ────────────────────────────────────────── #

@pytest.mark.parametrize(
    "prop, expected",
    [
        ("num_states", 2),
        ("num_observables", 2),
        ("num_parameters", 2),
        ("num_constants", 1),
    ],
)
def test_odedata_count_properties(prop, expected):
    """Count properties delegate to the correct SystemValues.n."""
    data = _make_odedata()
    assert getattr(data, prop) == expected


def test_odedata_sizes_returns_system_sizes():
    """sizes property returns SystemSizes with all counts."""
    data = _make_odedata(num_drivers=3)
    sizes = data.sizes
    assert sizes.states == 2
    assert sizes.observables == 2
    assert sizes.parameters == 2
    assert sizes.constants == 1
    assert sizes.drivers == 3


def test_odedata_mass_returns_stored_value():
    """mass property returns the _mass field."""
    data = _make_odedata()
    assert data.mass is None


# ── ODEData.from_BaseODE_initargs ─────────────────────────────── #

def test_from_base_ode_initargs_handles_none_optional():
    """Factory handles None for optional arguments gracefully."""
    data = ODEData.from_BaseODE_initargs(
        precision=np.float32,
        default_initial_values={"x": 1.0},
        default_parameters=None,
        default_constants=None,
        default_observable_names=None,
        num_drivers=0,
    )
    assert data.num_states == 1
    assert data.num_drivers == 0
    assert data.parameters.n == 0
    assert data.constants.n == 0
    assert data.observables.n == 0


def test_from_base_ode_initargs_overrides_defaults():
    """User values override defaults in from_BaseODE_initargs."""
    data = ODEData.from_BaseODE_initargs(
        precision=np.float64,
        initial_values={"x": 5.0},
        default_initial_values={"x": 0.0, "y": 1.0},
        default_parameters={"a": 0.5},
    )
    # User override for x should apply; y keeps default
    assert data.num_states == 2
    val = data.initial_states.values_dict["x"]
    assert float(val) == pytest.approx(5.0)
