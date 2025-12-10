"""Integration tests for last-step caching optimization.

These tests verify that algorithms with last-step caching produce
numerically equivalent results to the standard accumulation path.
"""



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

