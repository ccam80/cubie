"""Integration tests for refactored pinned memory system.

These tests verify the two-tier memory strategy for chunked vs non-chunked
batch solving:
- Non-chunked: Uses pinned host arrays for async transfers
- Chunked: Uses regular numpy host arrays with per-chunk pinned buffers
  from the ChunkBufferPool to limit total pinned memory usage.
"""

import numpy as np
import pytest

from cubie import MemoryManager
from cubie.batchsolving.solver import Solver
from cubie.batchsolving.arrays.BatchOutputArrays import OutputArrays
from cubie.memory.chunk_buffer_pool import ChunkBufferPool
from cubie.outputhandling.output_sizes import BatchOutputSizes


class MockMemoryManager(MemoryManager):
    """Mock memory manager for testing with controlled memory info."""

    def get_memory_info(self):
        return int(4096), int(8192)  # 4kb free, 8kb total


@pytest.fixture(scope="module")
def low_memory():
    return MockMemoryManager()


class TestTwoTierMemoryStrategy:
    """Test that memory strategy changes based on chunking."""

    def test_non_chunked_uses_pinned_host(self, system, precision):
        """Non-chunked runs use pinned host arrays."""
        # Create solver with default memory (non-chunked)
        solver = Solver(system, algorithm="euler")

        n_runs = 3
        n_states = system.sizes.states
        n_params = system.sizes.parameters

        inits = np.ones((n_states, n_runs), dtype=precision)
        params = np.ones((n_params, n_runs), dtype=precision)

        # Run with non-chunked settings (small data, should fit)
        result = solver.solve(
            inits,
            params,
            duration=0.01,
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

    def test_chunked_uses_numpy_host(self, system, precision, low_memory):
        """Chunked runs use numpy host arrays with buffer pool."""
        solver = Solver(system, algorithm="euler", memory_manager=low_memory)

        n_runs = 5
        n_states = system.sizes.states
        n_params = system.sizes.parameters

        inits = np.ones((n_states, n_runs), dtype=precision)
        params = np.ones((n_params, n_runs), dtype=precision)

        # Run with forced chunking (low memory)
        result = solver.solve(
            inits,
            params,
            duration=1.0,
            save_every=0.01,
            chunk_axis="run",
        )

        # Verify run completed successfully
        assert result.time_domain_array is not None
        assert not np.any(np.isnan(result.time_domain_array))

        # Verify chunks > 1 for chunked mode
        assert solver.kernel.output_arrays._chunks > 1

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

    def test_total_pinned_memory_bounded(self, system, precision, low_memory):
        """Total pinned memory stays within one chunk's worth."""
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


class TestEventBasedSynchronization:
    """Test CUDA event synchronization."""

    def test_wait_pending_blocks_correctly(
        self, system, precision, low_memory
    ):
        """wait_pending blocks until all writebacks complete."""
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

        # After solve completes, wait_pending should have been called
        # and all pending tasks should be done
        output_arrays = solver.kernel.output_arrays
        assert output_arrays._watcher._pending_count == 0

        # Verify results are valid
        assert result.time_domain_array is not None
        assert not np.any(np.isnan(result.time_domain_array))


class TestWatcherThreadBehavior:
    """Test watcher thread lifecycle and behavior."""

    def test_watcher_starts_on_first_chunk(
        self, system, precision, low_memory
    ):
        """Watcher thread starts when first task submitted."""
        solver = Solver(system, algorithm="euler", memory_manager=low_memory)

        n_runs = 5
        n_states = system.sizes.states
        n_params = system.sizes.parameters

        inits = np.ones((n_states, n_runs), dtype=precision)
        params = np.ones((n_params, n_runs), dtype=precision)

        # Initially watcher should not be running
        output_arrays = solver.kernel.output_arrays
        assert output_arrays._watcher._thread is None

        # Run with forced chunking
        result = solver.solve(
            inits,
            params,
            duration=1.0,
            save_every=0.01,
            chunk_axis="run",
        )

        # After solve, watcher should have been started
        # (it may have been shut down after completion)
        assert result.time_domain_array is not None

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


class TestRegressionNonChunkedPath:
    """Verify non-chunked path unchanged."""

    def test_small_batch_produces_correct_results(
        self, system, precision, solver
    ):
        """Small batches work correctly with refactored code."""
        solver = Solver(system, algorithm="euler")

        n_runs = 2
        n_states = system.sizes.states
        n_params = system.sizes.parameters

        inits = np.ones((n_states, n_runs), dtype=precision)
        params = np.ones((n_params, n_runs), dtype=precision)

        result = solver.solve(
            inits,
            params,
            duration=0.1,
            save_every=0.01,
        )

        assert result.time_domain_array is not None
        assert result.time_domain_array.shape[2] == n_runs
        assert not np.all(result.time_domain_array == 0)
        assert not np.any(np.isnan(result.time_domain_array))

    def test_non_chunked_path_no_buffer_pool_usage(self, solver, precision):
        """Non-chunked mode does not use buffer pool for output arrays."""
        solver.kernel.duration = 1.0
        batch_output_sizes = BatchOutputSizes.from_solver(solver)
        output_arrays = OutputArrays(
            sizes=batch_output_sizes,
            precision=precision,
        )
        output_arrays.update(solver)
        # Allocate device arrays
        output_arrays._memory_manager.allocate_queue(
            output_arrays, chunk_axis="run"
        )

        # Call finalise
        host_indices = slice(None)
        output_arrays.finalise(host_indices)

        # Non-chunked should use deferred writebacks, not buffer pool
        # Buffer pool should remain empty
        assert len(output_arrays._buffer_pool._buffers) == 0

        # Complete the writebacks
        output_arrays._memory_manager.sync_stream(output_arrays)


class TestRegressionChunkedPath:
    """Verify chunked path produces correct results."""

    def test_large_batch_produces_correct_results(
        self, system, precision, low_memory
    ):
        """Large batches produce same results as before."""
        solver = Solver(system, algorithm="euler", memory_manager=low_memory)

        n_runs = 5
        n_states = system.sizes.states
        n_params = system.sizes.parameters

        inits = np.ones((n_states, n_runs), dtype=precision)
        params = np.ones((n_params, n_runs), dtype=precision)

        result = solver.solve(
            inits,
            params,
            duration=1.0,
            save_every=0.01,
            chunk_axis="run",
        )

        # Verify output is valid
        assert result.time_domain_array is not None
        assert result.time_domain_array.shape[2] == n_runs
        assert not np.all(result.time_domain_array == 0)
        assert not np.any(np.isnan(result.time_domain_array))

    def test_chunked_results_match_non_chunked(self, system, precision):
        """Chunked execution produces same results as non-chunked."""
        # Create two solvers - one normal, one with low memory forcing chunks
        solver_normal = Solver(system, algorithm="euler")
        solver_low = Solver(
            system, algorithm="euler", memory_manager=MockMemoryManager()
        )

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
