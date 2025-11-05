"""Test algorithm-controller compatibility validation.

This test module validates the compatibility checking logic in
:class:`~cubie.integrators.SingleIntegratorRunCore.SingleIntegratorRunCore`.
The tests ensure that:

1. Incompatible configurations (adaptive controller + errorless algorithm)
   raise informative ValueError exceptions.
2. Compatible configurations (matching adaptive/fixed modes) succeed.
3. Error messages contain specific algorithm and controller names.
4. Error messages provide actionable guidance for fixing the issue.

The validation prevents silent failures where users accidentally pair an
adaptive step controller with an algorithm that cannot provide the error
estimate required for adaptive stepping.

Notes
-----
These tests use real system fixtures and actual integrator components
rather than mocks, ensuring that the validation behaves correctly in
real-world usage scenarios.
"""

import pytest
import numpy as np

from cubie.integrators.SingleIntegratorRunCore import (
    SingleIntegratorRunCore
)
from cubie.integrators.algorithms.generic_erk_tableaus import (
    CLASSICAL_RK4_TABLEAU,
    DORMAND_PRINCE_54_TABLEAU,
)


def test_errorless_euler_with_adaptive_raises(system):
    """Errorless explicit Euler with adaptive PI raises ValueError."""

    algorithm_settings = {
        "algorithm": "euler",
    }
    step_control_settings = {
        "step_controller": "pi",
        "dt_min": 1e-6,
        "dt_max": 1e-1,
    }

    with pytest.raises(ValueError) as exc_info:
        SingleIntegratorRunCore(
            system=system,
            algorithm_settings=algorithm_settings,
            step_control_settings=step_control_settings,
        )

    error_msg = str(exc_info.value).lower()
    assert "euler" in error_msg
    assert "pi" in error_msg
    assert "adaptive" in error_msg
    assert "fixed" in error_msg or "error estimate" in error_msg


def test_errorless_rk4_tableau_with_adaptive_raises(system):
    """Errorless RK4 tableau with adaptive PI raises ValueError."""

    algorithm_settings = {
        "algorithm": "erk",
        "tableau": CLASSICAL_RK4_TABLEAU,
    }
    step_control_settings = {
        "step_controller": "pi",
        "dt_min": 1e-6,
        "dt_max": 1e-1,
    }

    with pytest.raises(ValueError) as exc_info:
        SingleIntegratorRunCore(
            system=system,
            algorithm_settings=algorithm_settings,
            step_control_settings=step_control_settings,
        )

    error_msg = str(exc_info.value).lower()
    assert "erk" in error_msg
    assert "pi" in error_msg


def test_adaptive_tableau_with_adaptive_succeeds(system):
    """Adaptive Dormand-Prince with PI controller succeeds."""

    algorithm_settings = {
        "algorithm": "erk",
        "tableau": DORMAND_PRINCE_54_TABLEAU,
    }
    step_control_settings = {
        "step_controller": "pi",
        "dt_min": 1e-6,
        "dt_max": 1e-1,
    }

    core = SingleIntegratorRunCore(
        system=system,
        algorithm_settings=algorithm_settings,
        step_control_settings=step_control_settings,
    )

    assert core._algo_step.is_adaptive
    assert core._step_controller.is_adaptive


def test_errorless_euler_with_fixed_succeeds(system):
    """Errorless explicit Euler with fixed controller succeeds."""

    algorithm_settings = {
        "algorithm": "euler",
    }
    step_control_settings = {
        "step_controller": "fixed",
        "dt": 1e-3,
    }

    core = SingleIntegratorRunCore(
        system=system,
        algorithm_settings=algorithm_settings,
        step_control_settings=step_control_settings,
    )

    assert not core._algo_step.is_adaptive
    assert not core._step_controller.is_adaptive


def test_error_message_contains_algorithm_and_controller(system):
    """Error message includes algorithm and controller names."""

    algorithm_settings = {
        "algorithm": "euler",
    }
    step_control_settings = {
        "step_controller": "pid",
        "dt_min": 1e-6,
        "dt_max": 1e-1,
    }

    with pytest.raises(ValueError) as exc_info:
        SingleIntegratorRunCore(
            system=system,
            algorithm_settings=algorithm_settings,
            step_control_settings=step_control_settings,
        )

    error_msg = str(exc_info.value)
    assert "euler" in error_msg
    assert "pid" in error_msg
    assert "error estimate" in error_msg.lower()
    assert "fixed" in error_msg.lower()
