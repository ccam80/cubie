"""Test ERK dynamic controller defaults selection.

This test module validates the automatic selection of step controller
defaults in :class:`~cubie.integrators.algorithms.generic_erk.ERKStep` based
on whether the tableau has an embedded error estimate.

The tests ensure that:

1. Errorless tableaus (e.g., Classical RK4, Heun) default to fixed-step
   controllers.
2. Adaptive tableaus (e.g., Dormand-Prince) default to adaptive controllers
   (PI).
3. The default tableau (Dormand-Prince) selects adaptive defaults.
4. Default selection is based on ``tableau.has_error_estimate`` property.

This automatic selection prevents users from accidentally pairing an
errorless tableau with an adaptive controller, which would fail at runtime
since adaptive controllers require error estimates.

Notes
-----
These tests instantiate real :class:`ERKStep` objects with various tableaus
and verify the controller defaults without using mocks. This ensures the
logic works correctly with actual tableau objects.
"""

import pytest
import numpy as np

from cubie.integrators.algorithms.generic_erk import ERKStep
from cubie.integrators.algorithms.generic_erk_tableaus import (
    CLASSICAL_RK4_TABLEAU,
    DORMAND_PRINCE_54_TABLEAU,
    HEUN_21_TABLEAU,
)


def test_erk_errorless_tableau_defaults_to_fixed():
    """ERK with errorless tableau defaults to fixed controller."""

    step = ERKStep(
        precision=np.float32,
        n=3,
        dt=None,
        tableau=CLASSICAL_RK4_TABLEAU,
    )

    defaults = step.controller_defaults.step_controller
    assert defaults["step_controller"] == "fixed"
    assert "dt" in defaults


def test_erk_adaptive_tableau_defaults_to_adaptive():
    """ERK with adaptive tableau defaults to adaptive controller."""

    step = ERKStep(
        precision=np.float32,
        n=3,
        dt=None,
        tableau=DORMAND_PRINCE_54_TABLEAU,
    )

    defaults = step.controller_defaults.step_controller
    assert defaults["step_controller"] == "pi"
    assert "dt_min" in defaults
    assert "dt_max" in defaults


def test_erk_heun_tableau_defaults_to_fixed():
    """ERK with Heun tableau (errorless) defaults to fixed."""

    step = ERKStep(
        precision=np.float32,
        n=3,
        dt=None,
        tableau=HEUN_21_TABLEAU,
    )

    defaults = step.controller_defaults.step_controller
    assert defaults["step_controller"] == "fixed"


def test_erk_default_tableau_defaults_to_adaptive():
    """ERK with default tableau defaults to adaptive controller."""

    step = ERKStep(
        precision=np.float32,
        n=3,
        dt=None,
    )

    defaults = step.controller_defaults.step_controller
    assert defaults["step_controller"] == "pi"
