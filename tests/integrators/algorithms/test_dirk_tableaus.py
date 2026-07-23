"""Tests covering DIRK tableau registration and selection."""

import numpy as np
import pytest

from cubie.integrators.algorithms.generic_dirk import DIRKStep
from cubie.integrators.algorithms.generic_dirk_tableaus import (
    DEFAULT_DIRK_TABLEAU,
    DEFAULT_DIRK_TABLEAU_NAME,
    DIRK_TABLEAU_REGISTRY,
    DIRKTableau,
    KVAERNO3_TABLEAU,
    KVAERNO5_TABLEAU,
    L_STABLE_SDIRK4_TABLEAU,
)


@pytest.mark.parametrize(
    "expected_key",
    [
        "implicit_midpoint",
        "trapezoidal_dirk",
        "lobatto_iiic_3",
        "sdirk_2_2",
        "kvaerno3",
        "kvaerno5",
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


@pytest.mark.parametrize("precision_value", [np.float32, np.float64])
def test_first_stage_is_explicit_classifies_by_diagonal(
    precision_value,
):
    """ESDIRK tableaus report an explicit stage zero; SDIRK and
    single-stage tableaus report an implicit one."""

    assert KVAERNO3_TABLEAU.first_stage_is_explicit(precision_value)
    assert KVAERNO5_TABLEAU.first_stage_is_explicit(precision_value)
    assert not DIRK_TABLEAU_REGISTRY[
        "lobatto_iiic_3"
    ].first_stage_is_explicit(precision_value)
    assert not DIRK_TABLEAU_REGISTRY[
        "implicit_midpoint"
    ].first_stage_is_explicit(precision_value)


def test_first_stage_is_explicit_uses_typed_diagonal():
    """The classification compares in the requested precision, so a
    subnormal-below-float32 coefficient counts as explicit there and
    implicit at float64."""

    tiny_diagonal = DIRKTableau(
        a=((1e-46, 0.0), (0.5, 0.5)),
        b=(0.5, 0.5),
        c=(1e-46, 1.0),
        order=1,
    )
    assert tiny_diagonal.first_stage_is_explicit(np.float32)
    assert not tiny_diagonal.first_stage_is_explicit(np.float64)


def test_dirk_step_accepts_tableau_instance(precision):
    """DIRKStep should consume explicit tableau instances."""

    custom_name = "lobatto_iiic_3"
    custom_tableau = DIRK_TABLEAU_REGISTRY[custom_name]
    step = DIRKStep(precision=precision, n=2, tableau=custom_tableau)
    assert step.compile_settings.tableau is custom_tableau


@pytest.mark.parametrize(
    "tableau,stage_count,b_row,b_hat_row,expected_d",
    [
        (
            KVAERNO3_TABLEAU,
            4,
            3,
            2,
            (
                -0.18175341844607201,
                1.4169932981732141,
                -1.671106401227145,
                0.4358665215,
            ),
        ),
        (
            KVAERNO5_TABLEAU,
            7,
            6,
            5,
            (
                -0.0019588905362793174,
                0.0,
                -0.012515715947863326,
                -0.06565284626324187,
                0.010502658265357234,
                -0.1903752055179727,
                0.26,
            ),
        ),
    ],
)
def test_kvaerno_tableau_invariants(
    tableau,
    stage_count,
    b_row,
    b_hat_row,
    expected_d,
):
    """Kvaerno pairs expose stiff-accuracy and embedded-row invariants."""

    assert tableau.stage_count == stage_count
    assert tableau.b_matches_a_row == b_row
    assert tableau.b_hat_matches_a_row == b_hat_row
    assert tableau.first_same_as_last
    assert tableau.can_reuse_accepted_start
    assert tableau.has_error_estimate
    assert tableau.d == expected_d
