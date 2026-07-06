"""Numerical correctness test for chunked batch solving.

Verifies chunked execution produces same results as non-chunked
for various memory constraints.
"""

import numpy as np
import pytest


@pytest.mark.parametrize(
    "forced_free_mem",
    [
        860,
        1024,
        1240,
        1460,
        2048,  # unchunked to verify
    ],  # magic numbers explained in arrays/conftest.py
    indirect=True,
)
def test_chunked_solver_produces_correct_results(
    chunked_solved_solver, unchunked_solved_solver, forced_free_mem
):
    """Verify chunked execution produces same results as
    non-chunked."""
    chunked_solver, result_chunked = chunked_solved_solver
    unchunked_solver, result_normal = unchunked_solved_solver

    # Let the deliberate one-chunk test fall through
    if forced_free_mem < 2048:
        assert chunked_solver.chunks > 1
    assert unchunked_solver.chunks == 1

    # Results should match (within floating point tolerance)
    np.testing.assert_allclose(
        result_chunked.time_domain_array,
        result_normal.time_domain_array,
        rtol=1e-5,
        atol=1e-7,
        err_msg=(
            " ################################### \n"
            " Delta \n"
            f"{result_chunked.time_domain_array - result_normal.time_domain_array} \n"
            " ------------------------------------ \n"
            " Chunked output: \n"
            f"{result_chunked.time_domain_array} \n"
            " ------------------------------------ \n"
            " Unchunked output: \n"
            f"{result_normal.time_domain_array} \n"
            " ################################### "
        ),
    )
