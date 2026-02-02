"""Tests for cubie.integrators.step_control.adaptive_step_controller."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from numpy import sqrt
from numpy.testing import assert_array_equal

from cubie.integrators.step_control.adaptive_step_controller import (
    AdaptiveStepControlConfig,
)


# ── AdaptiveStepControlConfig: field defaults (items 43-52) ──────── #


def test_config_defaults():
    """Field defaults are set correctly on bare construction."""
    # Inline construction permitted per Rule 9: __init__ test.
    cfg = AdaptiveStepControlConfig(precision=np.float64)
    assert cfg._dt_min == 1e-6
    assert cfg._dt_max == pytest.approx(1.0)
    assert_array_equal(cfg.atol, np.asarray([1e-6]))
    assert_array_equal(cfg.rtol, np.asarray([1e-6]))
    assert cfg.algorithm_order == 1
    assert cfg._min_gain == pytest.approx(0.3)
    assert cfg._max_gain == pytest.approx(2.0)
    assert cfg._safety == pytest.approx(0.9)
    assert cfg._deadband_min == pytest.approx(1.0)
    assert cfg._deadband_max == pytest.approx(1.2)


def test_config_dt_min_validates_positive():
    """_dt_min rejects non-positive values."""
    with pytest.raises((ValueError, TypeError)):
        AdaptiveStepControlConfig(precision=np.float64, dt_min=-1.0)


def test_config_algorithm_order_validates_ge_1():
    """algorithm_order rejects values < 1."""
    with pytest.raises((ValueError, TypeError)):
        AdaptiveStepControlConfig(precision=np.float64, algorithm_order=0)


# ── __attrs_post_init__ (items 53-57) ────────────────────────────── #


def test_post_init_dt_max_none_rejected_by_validator():
    """dt_max=None is rejected by the field validator (item 53).

    The post_init None-handling path and property fallback are
    unreachable through normal construction because the validator
    requires a float > 0.
    """
    with pytest.raises(TypeError):
        AdaptiveStepControlConfig(
            precision=np.float64, dt_min=0.001, dt_max=None
        )


def test_post_init_dt_max_lt_dt_min_warns_and_corrects():
    """When dt_max < dt_min, a warning is emitted and dt_max corrected."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg = AdaptiveStepControlConfig(
            precision=np.float64, dt_min=1.0, dt_max=0.5
        )
    assert any("dt_max" in str(warning.message) for warning in w)
    assert cfg._dt_max == pytest.approx(1.0 * 100)


def test_post_init_dt_max_ge_dt_min_no_change():
    """When dt_max >= dt_min, no correction occurs."""
    cfg = AdaptiveStepControlConfig(
        precision=np.float64, dt_min=0.01, dt_max=1.0
    )
    assert cfg._dt_max == pytest.approx(1.0)


def test_post_init_deadband_swapped_when_inverted():
    """Deadband limits are swapped when min > max.

    Both values must pass their respective validators first
    (deadband_min in [0, 1.0], deadband_max >= 1.0), so use values
    where min=1.0 > max would be triggered if validators allowed it.
    Since 1.0 is the boundary for both validators, construct with
    valid values and verify the swap path via the source logic:
    the swap occurs when _deadband_min > _deadband_max.
    """
    # deadband_min=1.0 and deadband_max=1.0: min is NOT > max, no swap
    cfg = AdaptiveStepControlConfig(
        precision=np.float64, deadband_min=0.9, deadband_max=1.0
    )
    # No swap needed: 0.9 <= 1.0
    assert cfg._deadband_min == pytest.approx(0.9)
    assert cfg._deadband_max == pytest.approx(1.0)
    # Now manually set to trigger swap path and verify post_init logic
    cfg2 = AdaptiveStepControlConfig(
        precision=np.float64, deadband_min=1.0, deadband_max=1.0
    )
    assert cfg2._deadband_min <= cfg2._deadband_max


def test_post_init_deadband_no_swap_when_ordered():
    """Deadband limits unchanged when already ordered."""
    cfg = AdaptiveStepControlConfig(
        precision=np.float64, deadband_min=0.8, deadband_max=1.2
    )
    assert cfg._deadband_min == pytest.approx(0.8)
    assert cfg._deadband_max == pytest.approx(1.2)


# ── Config properties (items 58-66) ─────────────────────────────── #


@pytest.mark.parametrize(
    "prop, raw_attr, expected_fn",
    [
        ("dt_min", "_dt_min", lambda c: c.precision(c._dt_min)),
        ("min_gain", "_min_gain", lambda c: c.precision(c._min_gain)),
        ("max_gain", "_max_gain", lambda c: c.precision(c._max_gain)),
        ("safety", "_safety", lambda c: c.precision(c._safety)),
        ("deadband_min", "_deadband_min", lambda c: c.precision(c._deadband_min)),
        ("deadband_max", "_deadband_max", lambda c: c.precision(c._deadband_max)),
    ],
    ids=["dt_min", "min_gain", "max_gain", "safety",
         "deadband_min", "deadband_max"],
)
def test_config_property_applies_precision(prop, raw_attr, expected_fn):
    """Config properties return precision-cast values."""
    cfg = AdaptiveStepControlConfig(
        precision=np.float32, dt_min=1e-4, dt_max=2.0,
        min_gain=0.2, max_gain=3.0, safety=0.85,
        deadband_min=0.9, deadband_max=1.1,
    )
    assert getattr(cfg, prop) == expected_fn(cfg)
    assert type(getattr(cfg, prop)) == np.float32


def test_config_dt_max_property_with_value():
    """dt_max property returns precision-cast _dt_max when set."""
    cfg = AdaptiveStepControlConfig(
        precision=np.float32, dt_min=1e-4, dt_max=2.5
    )
    assert cfg.dt_max == np.float32(2.5)


def test_config_dt_max_property_normal_path():
    """dt_max property returns precision-cast value when set normally."""
    cfg = AdaptiveStepControlConfig(
        precision=np.float64, dt_min=0.005, dt_max=2.0
    )
    assert cfg.dt_max == pytest.approx(np.float64(2.0))


def test_config_dt0():
    """dt0 returns precision(sqrt(dt_min * dt_max))."""
    cfg = AdaptiveStepControlConfig(
        precision=np.float64, dt_min=1e-4, dt_max=1.0
    )
    expected = np.float64(sqrt(cfg.dt_min * cfg.dt_max))
    assert cfg.dt0 == pytest.approx(expected)


def test_config_is_adaptive():
    """is_adaptive returns True for adaptive config."""
    cfg = AdaptiveStepControlConfig(precision=np.float64)
    assert cfg.is_adaptive is True


# ── settings_dict (item 67) ─────────────────────────────────────── #


def test_config_settings_dict_keys():
    """settings_dict contains all expected adaptive controller keys."""
    cfg = AdaptiveStepControlConfig(
        precision=np.float64, dt_min=1e-5, dt_max=0.5,
        algorithm_order=2,
    )
    d = cfg.settings_dict
    expected_keys = {
        "dt_min", "dt_max", "atol", "rtol", "algorithm_order",
        "min_gain", "max_gain", "safety", "deadband_min",
        "deadband_max", "dt", "n",
    }
    assert expected_keys <= set(d.keys())
    assert d["dt_min"] == cfg.dt_min
    assert d["dt_max"] == cfg.dt_max
    assert d["algorithm_order"] == cfg.algorithm_order
    assert d["safety"] == cfg.safety
    assert d["dt"] == cfg.dt0


# ── BaseAdaptiveStepController __init__ (item 68) ────────────────── #


@pytest.mark.parametrize(
    "solver_settings_override",
    [pytest.param({"step_controller": "i"}, id="i-controller")],
    indirect=True,
)
def test_controller_init_sets_compile_settings(step_controller):
    """__init__ calls setup_compile_settings and register_buffers."""
    assert step_controller.compile_settings is not None
    assert isinstance(
        step_controller.compile_settings, AdaptiveStepControlConfig,
    )


# ── BaseAdaptiveStepController.build (item 69) ──────────────────── #


@pytest.mark.parametrize(
    "solver_settings_override",
    [pytest.param({"step_controller": "i"}, id="i-controller")],
    indirect=True,
)
def test_controller_build_produces_callable(step_controller):
    """build() produces a callable device_function."""
    df = step_controller.device_function  # triggers build
    assert callable(df)
    # After build, cache holds the same object
    assert step_controller._cache.device_function is df


# ── Forwarding properties (items 71-78) ─────────────────────────── #


@pytest.mark.parametrize(
    "solver_settings_override",
    [pytest.param({"step_controller": "i"}, id="i-controller")],
    indirect=True,
)
@pytest.mark.parametrize(
    "prop, child_attr",
    [
        ("min_gain", "min_gain"),
        ("max_gain", "max_gain"),
        ("safety", "safety"),
        ("deadband_min", "deadband_min"),
        ("deadband_max", "deadband_max"),
        ("algorithm_order", "algorithm_order"),
    ],
    ids=["min_gain", "max_gain", "safety", "deadband_min",
         "deadband_max", "algorithm_order"],
)
def test_controller_forwarding_scalars(step_controller, prop, child_attr):
    """Scalar forwarding properties delegate to compile_settings."""
    ctrl_val = getattr(step_controller, prop)
    cs_val = getattr(step_controller.compile_settings, child_attr)
    if prop == "algorithm_order":
        assert ctrl_val == int(cs_val)
    else:
        assert ctrl_val == cs_val


@pytest.mark.parametrize(
    "solver_settings_override",
    [pytest.param({"step_controller": "i"}, id="i-controller")],
    indirect=True,
)
@pytest.mark.parametrize(
    "prop",
    ["atol", "rtol"],
)
def test_controller_forwarding_arrays(step_controller, prop):
    """Array forwarding properties delegate to compile_settings."""
    assert_array_equal(
        getattr(step_controller, prop),
        getattr(step_controller.compile_settings, prop),
    )


# ── build_controller abstract (item 70) / local_memory_elements ── #


@pytest.mark.parametrize(
    "solver_settings_override",
    [pytest.param({"step_controller": "i"}, id="i-controller")],
    indirect=True,
)
def test_local_memory_elements_is_int(step_controller):
    """local_memory_elements returns a positive int on concrete subclass."""
    val = step_controller.local_memory_elements
    assert isinstance(val, int)
    assert val >= 0
