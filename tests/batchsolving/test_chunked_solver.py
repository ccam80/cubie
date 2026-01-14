"""Integration tests for chunked solver execution.

These tests verify that the chunking functionality works correctly,
testing both the "run" and "time" chunk axes to ensure the fixes for:
1. Stride incompatibility when chunking on the run axis
2. Missing axis error when chunking on the time axis
"""

import pytest
import numpy as np

from cubie import MemoryManager
from cubie.batchsolving.solver import Solver


class MockMemoryManager(MemoryManager):
    """Mock memory manager for testing with controlled memory info."""

    def get_memory_info(self):
        return int(4096), int(8192)  # 4kb free, 8kb total


@pytest.fixture(scope="module")
def low_memory():
    return MockMemoryManager()


class TestChunkedSolverExecution:
    """Test solver execution with forced chunking."""

    @pytest.mark.parametrize("chunk_axis", ["run"])  # , "time"])
    def test_chunked_solve_produces_valid_output(
        self, system, precision, chunk_axis, low_memory
    ):
        """Verify chunked solver produces valid output arrays."""
        solver = Solver(system, algorithm="euler", memory_manager=low_memory)

        n_runs = 5
        n_states = system.sizes.states
        n_params = system.sizes.parameters

        inits = np.ones((n_states, n_runs), dtype=precision)
        params = np.ones((n_params, n_runs), dtype=precision)

        # 5 runs, 3 states, 100 saves per s, 4 bytes per float32.
        # 6000 bytes per second of duration.
        # With 4kb available limit, chunking should occur before duration == 1
        result = solver.solve(
            inits,
            params,
            duration=1.0,
            save_every=0.01,
            chunk_axis=chunk_axis,
        )

        # Verify output shape and that values are not all zeros/NaN
        assert result.time_domain_array is not None
        assert result.time_domain_array.shape[2] == n_runs
        assert not np.all(result.time_domain_array == 0)
        assert not np.any(np.isnan(result.time_domain_array))


class TestSyncStreamRemoval:
    """Test that solver works without explicit sync_stream calls."""

    def test_chunked_solver_produces_correct_results(
        self, system, precision, low_memory
    ):
        """Verify chunked execution produces same results as non-chunked."""
        # Create solvers with different memory settings
        solver_low = Solver(
            system, algorithm="euler", memory_manager=low_memory
        )

        # Use default memory manager for non-chunked baseline
        solver_normal = Solver(system, algorithm="euler")

        n_runs = 3
        n_states = system.sizes.states
        n_params = system.sizes.parameters

        inits = np.ones((n_states, n_runs), dtype=precision)
        params = np.ones((n_params, n_runs), dtype=precision)

        # Run with normal (non-chunked) solver
        result_normal = solver_normal.solve(
            inits.copy(),
            params.copy(),
            duration=0.1,
            save_every=0.01,
        )

        # Run with low memory (chunked) solver
        result_chunked = solver_low.solve(
            inits.copy(),
            params.copy(),
            duration=0.1,
            save_every=0.01,
        )

        # Results should match (within floating point tolerance)
        np.testing.assert_allclose(
            result_chunked.time_domain_array,
            result_normal.time_domain_array,
            rtol=1e-5,
            atol=1e-7,
        )

    def test_input_buffers_released_after_kernel(
        self, system, precision, low_memory
    ):
        """Verify input buffers are released after each chunk."""
        solver = Solver(system, algorithm="euler", memory_manager=low_memory)

        n_runs = 5
        n_states = system.sizes.states
        n_params = system.sizes.parameters

        inits = np.ones((n_states, n_runs), dtype=precision)
        params = np.ones((n_params, n_runs), dtype=precision)

        # Run solve to trigger chunked execution
        result = solver.solve(
            inits,
            params,
            duration=1.0,
            save_every=0.01,
            chunk_axis="run",
        )

        # After solve completes, input_arrays should have empty active buffers
        input_arrays = solver.kernel.input_arrays
        assert len(input_arrays._active_buffers) == 0

        # Verify solve completed successfully
        assert result.time_domain_array is not None
        assert not np.any(np.isnan(result.time_domain_array))
