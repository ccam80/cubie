"""Tests covering DIRK tableau registration and selection."""

import pytest

from cubie.integrators.algorithms.generic_dirk import DIRKStep
from cubie.integrators.algorithms.generic_dirk_tableaus import (
    DEFAULT_DIRK_TABLEAU,
    DEFAULT_DIRK_TABLEAU_NAME,
    DIRK_TABLEAU_REGISTRY,
    DIRKTableau,
    KVAERNO3_GAMMA,
    KVAERNO3_TABLEAU,
    KVAERNO5_GAMMA,
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


def test_dirk_step_accepts_tableau_instance(precision):
    """DIRKStep should consume explicit tableau instances."""

    custom_name = "lobatto_iiic_3"
    custom_tableau = DIRK_TABLEAU_REGISTRY[custom_name]
    step = DIRKStep(precision=precision, n=2, tableau=custom_tableau)
    assert step.compile_settings.tableau is custom_tableau


def test_kvaerno3_coefficients_are_canonical():
    """Kvaerno3 coefficients match the published SciML tableau."""

    assert KVAERNO3_TABLEAU.a == (
        (0.0, 0.0, 0.0, 0.0),
        (KVAERNO3_GAMMA, KVAERNO3_GAMMA, 0.0, 0.0),
        (
            0.490563388419108,
            0.073570090080892,
            KVAERNO3_GAMMA,
            0.0,
        ),
        (
            0.308809969973036,
            1.490563388254106,
            -1.235239879727145,
            KVAERNO3_GAMMA,
        ),
    )
    assert KVAERNO3_TABLEAU.b == KVAERNO3_TABLEAU.a[3]
    assert KVAERNO3_TABLEAU.b_hat == KVAERNO3_TABLEAU.a[2]
    assert KVAERNO3_TABLEAU.c == (
        0.0,
        2.0 * KVAERNO3_GAMMA,
        1.0,
        1.0,
    )
    assert KVAERNO3_TABLEAU.order == 3


def test_kvaerno5_coefficients_are_canonical():
    """Kvaerno5 coefficients match the published SciML tableau."""

    assert KVAERNO5_TABLEAU.a == (
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (KVAERNO5_GAMMA, KVAERNO5_GAMMA, 0.0, 0.0, 0.0, 0.0, 0.0),
        (
            0.13,
            0.84033320996790809,
            KVAERNO5_GAMMA,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            0.22371961478320505,
            0.47675532319799699,
            -0.06470895363112615,
            KVAERNO5_GAMMA,
            0.0,
            0.0,
            0.0,
        ),
        (
            0.16648564323248321,
            0.1045001884159172,
            0.03631482272098715,
            -0.13090704451073998,
            KVAERNO5_GAMMA,
            0.0,
            0.0,
        ),
        (
            0.13855640231268224,
            0.0,
            -0.04245337201752043,
            0.02446657898003141,
            0.61943039072480676,
            KVAERNO5_GAMMA,
            0.0,
        ),
        (
            0.13659751177640291,
            0.0,
            -0.05496908796538376,
            -0.04118626728321046,
            0.62993304899016403,
            0.06962479448202728,
            KVAERNO5_GAMMA,
        ),
    )
    assert KVAERNO5_TABLEAU.b == KVAERNO5_TABLEAU.a[6]
    assert KVAERNO5_TABLEAU.b_hat == KVAERNO5_TABLEAU.a[5]
    assert KVAERNO5_TABLEAU.c == (
        0.0,
        2.0 * KVAERNO5_GAMMA,
        1.230333209967908,
        0.895765984350076,
        0.436393609858648,
        1.0,
        1.0,
    )
    assert KVAERNO5_TABLEAU.order == 5


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
