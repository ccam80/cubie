"""Tests for the Gustafsson controller configuration surface."""

import numpy as np
import pytest

from cubie.integrators.step_control.adaptive_step_controller import (
    AdaptiveStepControlConfig,
)
from cubie.integrators.step_control.gustafsson_controller import (
    GustafssonStepControlConfig,
)


# ── GustafssonStepControlConfig: defaults and validation ─────────── #


def test_config_defaults():
    """Gustafsson-specific fields default correctly."""
    # Inline construction permitted per Rule 9: __init__ test.
    cfg = GustafssonStepControlConfig(precision=np.float64)
    assert cfg.gamma == pytest.approx(0.9)
    assert cfg.newton_max_iters == 20


def test_config_gamma_applies_precision():
    """gamma is returned in the configured precision."""
    cfg = GustafssonStepControlConfig(precision=np.float32, gamma=0.8)
    assert cfg.gamma == np.float32(0.8)
    assert isinstance(cfg.gamma, np.float32)


@pytest.mark.parametrize("bad_gamma", [-0.1, 1.5])
def test_config_gamma_validates_range(bad_gamma):
    """gamma rejects values outside [0, 1]."""
    with pytest.raises((ValueError, TypeError)):
        GustafssonStepControlConfig(
            precision=np.float64, gamma=bad_gamma
        )


def test_config_newton_max_iters_returns_int():
    """newton_max_iters is exposed as a plain int."""
    cfg = GustafssonStepControlConfig(
        precision=np.float64, newton_max_iters=7
    )
    assert cfg.newton_max_iters == 7
    assert isinstance(cfg.newton_max_iters, int)


def test_config_newton_max_iters_validates_nonnegative():
    """newton_max_iters rejects negative values."""
    with pytest.raises((ValueError, TypeError)):
        GustafssonStepControlConfig(
            precision=np.float64, newton_max_iters=-1
        )


def test_config_settings_dict_extends_parent():
    """settings_dict adds gamma and newton_max_iters to parent keys."""
    cfg = GustafssonStepControlConfig(
        precision=np.float64, gamma=0.85, newton_max_iters=11
    )
    settings = cfg.settings_dict
    assert settings["gamma"] == pytest.approx(0.85)
    assert settings["newton_max_iters"] == 11
    parent_keys = set(
        AdaptiveStepControlConfig(precision=np.float64).settings_dict
    )
    assert parent_keys <= set(settings)


# ── GustafssonController: property forwarding ─────────────────────── #


@pytest.mark.parametrize(
    "solver_settings_override",
    [{"step_controller": "gustafsson"}],
    ids=["gustafsson"],
    indirect=True,
)
def test_controller_forwards_config_properties(step_controller):
    """Controller gamma/newton_max_iters mirror compile_settings."""
    assert (
        step_controller.gamma
        == step_controller.compile_settings.gamma
    )
    assert (
        step_controller.newton_max_iters
        == step_controller.compile_settings.newton_max_iters
    )
    assert isinstance(step_controller.newton_max_iters, int)
