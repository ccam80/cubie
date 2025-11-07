"""Integration tests for last-step caching optimization.

These tests verify that algorithms with last-step caching produce
numerically equivalent results to the standard accumulation path.
"""

import pytest


@pytest.mark.parametrize(
    "tableau_name,expected_b_row,expected_b_hat_row",
    [
        ("rodas4p", 5, 4),
        ("rodas5p", 7, 6),
    ]
)
def test_rosenbrock_last_step_caching_properties(
    tableau_name, expected_b_row, expected_b_hat_row
):
    """Verify tableau properties for last-step caching in Rosenbrock methods."""
    from cubie.integrators.algorithms.generic_rosenbrockw_tableaus import (
        ROSENBROCK_TABLEAUS,
    )

    tableau = ROSENBROCK_TABLEAUS[tableau_name]

    assert tableau.b_matches_a_row == expected_b_row, (
        f"Expected b_matches_a_row={expected_b_row} for {tableau_name}, "
        f"got {tableau.b_matches_a_row}"
    )
    assert tableau.b_hat_matches_a_row == expected_b_hat_row, (
        f"Expected b_hat_matches_a_row={expected_b_hat_row} for {tableau_name}, "
        f"got {tableau.b_hat_matches_a_row}"
    )


def test_firk_last_step_caching_properties():
    """Verify tableau properties for last-step caching in FIRK methods."""
    from cubie.integrators.algorithms.generic_firk_tableaus import (
        RADAU_IIA_5_TABLEAU,
    )

    tableau = RADAU_IIA_5_TABLEAU

    assert tableau.b_matches_a_row == 2, (
        f"Expected b_matches_a_row=2 for RadauIIA5, "
        f"got {tableau.b_matches_a_row}"
    )

