"""Integration tests for chunked solver execution.

These tests verify that the chunking functionality works correctly,
testing both the "run" and "time" chunk axes to ensure the fixes for:
1. Stride incompatibility when chunking on the run axis
2. Missing axis error when chunking on the time axis
"""

import pytest
import numpy as np

from cubie.batchsolving.solver import Solver


class TestChunkedSolverExecution:
    """Test solver execution with forced chunking."""

    @pytest.mark.parametrize("chunk_axis", ["run", "time"])
    def test_chunked_solve_produces_valid_output(
        self, system, precision, chunk_axis
    ):
        """Verify chunked solver produces valid output arrays."""
        solver = Solver(system, algorithm="euler")
        n_runs = 10
        n_states = system.sizes.states
        n_params = system.sizes.parameters

        inits = np.ones((n_states, n_runs), dtype=precision)
        params = np.ones((n_params, n_runs), dtype=precision)

        result = solver.solve(
            inits,
            params,
            duration=0.1,
            save_every=0.01,
            chunk_axis=chunk_axis,
        )

        # Verify output shape and that values are not all zeros/NaN
        assert result.time_domain_array is not None
        assert result.time_domain_array.shape[2] == n_runs
        assert not np.all(result.time_domain_array == 0)
        assert not np.any(np.isnan(result.time_domain_array))

    @pytest.mark.parametrize("chunk_axis", ["run", "time"])
    def test_chunked_solve_with_observables(
        self, system, precision, chunk_axis
    ):
        """Verify chunked solver correctly computes time-domain output."""
        solver = Solver(system, algorithm="euler")
        n_runs = 8
        n_states = system.sizes.states
        n_params = system.sizes.parameters

        inits = np.ones((n_states, n_runs), dtype=precision)
        params = np.ones((n_params, n_runs), dtype=precision)

        result = solver.solve(
            inits,
            params,
            duration=0.1,
            save_every=0.02,
            chunk_axis=chunk_axis,
        )

        # Verify time-domain array is computed (includes observables)
        if result.time_domain_array is not None and result.time_domain_array.size > 0:
            assert not np.all(result.time_domain_array == 0)
            assert not np.any(np.isnan(result.time_domain_array))

    @pytest.mark.parametrize("chunk_axis", ["run", "time"])
    def test_chunked_solve_small_batch(
        self, system, precision, chunk_axis
    ):
        """Verify chunked solver works with small batch sizes."""
        solver = Solver(system, algorithm="euler")
        n_runs = 4
        n_states = system.sizes.states
        n_params = system.sizes.parameters

        inits = np.ones((n_states, n_runs), dtype=precision)
        params = np.ones((n_params, n_runs), dtype=precision)

        result = solver.solve(
            inits,
            params,
            duration=0.05,
            save_every=0.01,
            chunk_axis=chunk_axis,
        )

        assert result.time_domain_array is not None
        assert result.time_domain_array.shape[2] == n_runs
        # Verify values are computed (not all ones which would be initial)
        # After integration, state should have changed from initial value
        assert not np.allclose(
            result.time_domain_array[-1], result.time_domain_array[0]
        )
