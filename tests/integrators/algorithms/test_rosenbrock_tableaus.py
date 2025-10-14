"""Tests for Rosenbrock-W tableau registry integration."""

import pytest

from cubie.integrators.algorithms.generic_rosenbrockw_tableaus import (
    DEFAULT_ROSENBROCK_TABLEAU,
    DEFAULT_ROSENBROCK_TABLEAU_NAME,
    ROS3P_TABLEAU,
    ROSENBROCK_TABLEAUS,
)
from tests.integrators.cpu_reference import get_ref_step_function


def test_rosenbrock_registry_contains_expected_tableaus():
    """Registry should expose documented Rosenbrock-W tableaus."""

    registered = set(ROSENBROCK_TABLEAUS)
    assert {"ros3p", "rosenbrock_w6s4os"}.issubset(registered)
    assert DEFAULT_ROSENBROCK_TABLEAU_NAME == "ros3p"
    assert DEFAULT_ROSENBROCK_TABLEAU is ROS3P_TABLEAU


def test_rosenbrock_step_function_accepts_registry_key():
    """CPU reference stepper should resolve registry keys."""

    stepper = get_ref_step_function("rosenbrock", tableau="ros3p")
    assert stepper.keywords["tableau"] is ROSENBROCK_TABLEAUS["ros3p"]

@pytest.mark.parametrize(
    "solver_settings_override, system_override",
    [
        (
            {
                "algorithm": "rosenbrock",
                "tableau": DEFAULT_ROSENBROCK_TABLEAU,
                "step_controller": "pi",
                "krylov_tolerance": 1e-7,
            },
            {},
        )
    ],
    indirect=True,
)
def test_rosenbrock_step_accepts_registry_tableau(step_object):
    """GPU Rosenbrock step should load tableaus from the registry."""

    assert step_object.tableau is ROSENBROCK_TABLEAUS["ros3p"]
