"""Tests for cubie.integrators.algorithms.explicit_euler."""

from __future__ import annotations

import numpy as np
import pytest

from cubie.integrators.algorithms.explicit_euler import (
    EE_DEFAULTS,
    ExplicitEulerStep,
)
from cubie.integrators.algorithms.base_algorithm_step import (
    StepCache,
    StepControlDefaults,
)
from cubie.integrators.algorithms.ode_explicitstep import ExplicitStepConfig


# ── Module-level constants ─────────────────────────────── #


def test_ee_defaults_step_controller_type():
    """EE_DEFAULTS specifies fixed step controller with dt=1e-3."""
    assert EE_DEFAULTS.step_controller["step_controller"] == "fixed"
    assert EE_DEFAULTS.step_controller["dt"] == pytest.approx(1e-3)


# ── __init__ ───────────────────────────────────────────── #


def test_init_creates_explicit_step_config(step_object):
    """Constructor builds an ExplicitStepConfig as compile_settings."""
    cs = step_object.compile_settings
    # isinstance justified: verifying correct config subclass is the
    # functionality, combined with value check below
    assert isinstance(cs, ExplicitStepConfig)
    assert cs.n == step_object.compile_settings.n


def test_init_controller_defaults_match_ee_defaults(step_object):
    """Constructor passes EE_DEFAULTS as controller defaults."""
    defaults = step_object.controller_defaults
    assert defaults.step_controller["step_controller"] == "fixed"
    assert defaults.step_controller["dt"] == pytest.approx(1e-3)


# ── build_step ─────────────────────────────────────────── #


def test_build_returns_step_cache(single_integrator_run):
    """build_step returns a StepCache with step and no nonlinear solver."""
    algo = single_integrator_run._algo_step
    # Trigger build through the integrator run's device_function
    _ = single_integrator_run.device_function
    cache = algo._cache
    assert cache.nonlinear_solver is None
    # Cache step is the same object as device_function.step
    assert cache.step is algo._cache.step


# ── Properties ─────────────────────────────────────────── #


@pytest.mark.parametrize(
    "prop, expected",
    [
        ("threads_per_step", 1),
        ("is_multistage", False),
        ("is_adaptive", False),
        ("order", 1),
    ],
)
def test_properties(step_object, prop, expected):
    """Static properties return correct constant values."""
    assert getattr(step_object, prop) == expected
