"""Tests covering DIRK tableau registration and selection."""

import pytest

from cubie.integrators.algorithms.generic_dirk import DIRKStep
from cubie.integrators.algorithms.generic_dirk_tableaus import (
    DEFAULT_DIRK_TABLEAU,
    DEFAULT_DIRK_TABLEAU_NAME,
    DIRK_TABLEAU_REGISTRY,
    DIRKTableau,
    L_STABLE_SDIRK4_TABLEAU,
)


@pytest.mark.parametrize(
    "expected_key",
    [
        "implicit_midpoint",
        "trapezoidal_dirk",
        "lobatto_iiic_3",
        "sdirk_2_2",
    ],
)
def test_dirk_tableau_registry_contains_expected_entries(expected_key):
    """Registry must expose the documented DIRK tableaus."""

    assert expected_key in DIRK_TABLEAU_REGISTRY
    assert isinstance(DIRK_TABLEAU_REGISTRY[expected_key], DIRKTableau)


def test_dirk_tableau_default_matches_registry():
    """Default DIRK tableau should coincide with the registry entry."""

    assert DIRK_TABLEAU_REGISTRY[DEFAULT_DIRK_TABLEAU_NAME] is DEFAULT_DIRK_TABLEAU


def test_l_stable_sdirk4_fourth_stage_is_consistent():
    """Hairer4's fourth-stage row sum must equal its abscissa."""

    assert sum(L_STABLE_SDIRK4_TABLEAU.a[3]) == pytest.approx(
        L_STABLE_SDIRK4_TABLEAU.c[3]
    )


def test_dirk_step_accepts_tableau_instance(precision):
    """DIRKStep should consume explicit tableau instances."""

    custom_name = "lobatto_iiic_3"
    custom_tableau = DIRK_TABLEAU_REGISTRY[custom_name]
    step = DIRKStep(precision=precision, n=2, tableau=custom_tableau)
    assert step.compile_settings.tableau is custom_tableau
