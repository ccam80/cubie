"""Integration tests for refactored pinned memory system.

These tests verify the two-tier memory strategy for chunked vs non-chunked
batch solving:
- Non-chunked: Uses pinned host arrays for async transfers
- Chunked: Uses regular numpy host arrays with per-chunk pinned buffers
  from the ChunkBufferPool to limit total pinned memory usage.
"""

import numpy as np

from cubie.batchsolving.solver import Solver
from cubie.memory.chunk_buffer_pool import ChunkBufferPool


class TestTwoTierMemoryStrategy:
    """Test that memory strategy changes based on chunking."""

    def test_non_chunked_uses_pinned_host(
        self, system, solver, precision, driver_settings
    ):
        """Non-chunked runs use pinned host arrays."""
        # Create solver with default memory (non-chunked)

        n_runs = 3
        n_states = system.sizes.states
        n_params = system.sizes.parameters

        inits = np.ones((n_states, n_runs), dtype=precision)
        params = np.ones((n_params, n_runs), dtype=precision)

        # Run with non-chunked settings (small data, should fit)
        result = solver.solve(
            inits,
            params,
            drivers=driver_settings,
            duration=0.05,
            save_every=0.01,
        )

        # Verify run completed successfully
        assert result.time_domain_array is not None
        assert not np.any(np.isnan(result.time_domain_array))

        # Verify chunks == 1 for non-chunked mode
        assert solver.kernel.output_arrays._chunks == 1

        # Verify host arrays are pinned when non-chunked
        for (
            name,
            slot,
        ) in solver.kernel.output_arrays.host.iter_managed_arrays():
            assert slot.memory_type == "pinned"

    def test_chunked_uses_numpy_host(
        self, system, precision, low_mem_solver, driver_settings
    ):
        """Chunked runs use numpy host arrays with buffer pool."""
        solver = low_mem_solver

        n_runs = 5
        n_states = system.sizes.states
        n_params = system.sizes.parameters

        inits = np.ones((n_states, n_runs), dtype=precision)
        params = np.ones((n_params, n_runs), dtype=precision)

        # Run with forced chunking (low memory)
        result = solver.solve(
            inits,
            params,
            drivers=driver_settings,
            duration=1.0,
            save_every=0.01,
            chunk_axis="run",
        )

        # Verify run completed successfully
        assert result.time_domain_array is not None
        assert not np.any(np.isnan(result.time_domain_array))

        # Verify chunks > 1 for chunked mode
        assert solver.chunks > 1

        # When chunked, host arrays should be numpy (not pinned)
        # to limit total pinned memory to buffer pool only
        found_one = False
        for (
            name,
            slot,
        ) in solver.kernel.output_arrays.host.iter_managed_arrays():
            if slot.needs_chunked_transfer:
                assert slot.memory_type == "host"
                found_one = True
        assert found_one, "No chunked-transfer arrays found"

    def test_total_pinned_memory_bounded(
        self, system, precision, low_mem_solver
    ):
        """Total pinned memory stays within one chunk's worth."""
        solver = low_mem_solver

        n_runs = 5
        n_states = system.sizes.states
        n_params = system.sizes.parameters

        inits = np.ones((n_states, n_runs), dtype=precision)
        params = np.ones((n_params, n_runs), dtype=precision)

        # Run with forced chunking
        result = solver.solve(
            inits,
            params,
            duration=1.0,
            save_every=0.01,
            chunk_axis="run",
        )

        # Verify successful completion
        assert result.time_domain_array is not None

        # Buffer pool should exist and be used
        buffer_pool = solver.kernel.output_arrays._buffer_pool
        assert isinstance(buffer_pool, ChunkBufferPool)

        # After solve completes, buffers should be released back to pool
        # (in_use should be False for all)
        for buffer_list in buffer_pool._buffers.values():
            for buf in buffer_list:
                assert buf.in_use is False


class TestWatcherThreadBehavior:
    """Test watcher thread lifecycle and behavior."""

    def test_watcher_completes_all_tasks(self, system, precision, low_memory):
        """All submitted tasks are completed before solve returns."""
        solver = Solver(system, algorithm="euler", memory_manager=low_memory)

        n_runs = 5
        n_states = system.sizes.states
        n_params = system.sizes.parameters

        inits = np.ones((n_states, n_runs), dtype=precision)
        params = np.ones((n_params, n_runs), dtype=precision)

        # Run with forced chunking
        result = solver.solve(
            inits,
            params,
            duration=1.0,
            save_every=0.01,
            chunk_axis="run",
        )

        # Verify all tasks completed
        output_arrays = solver.kernel.output_arrays
        assert output_arrays._watcher._pending_count == 0

        # Verify data is valid
        assert not np.any(np.isnan(result.time_domain_array))


class TestChunkedVsNonChunkedResults:
    def test_chunked_results_match_non_chunked(
        self, system, solver, precision, low_mem_solver
    ):
        """Chunked execution produces same results as non-chunked."""
        # Create two solvers - one normal, one with low memory forcing chunks
        solver_normal = solver
        solver_low = low_mem_solver

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
            dt=solver.dt,
        )

        # Results should match (within floating point tolerance)
        np.testing.assert_allclose(
            result_chunked.time_domain_array,
            result_normal.time_domain_array,
            rtol=1e-5,
            atol=1e-7,
        )
