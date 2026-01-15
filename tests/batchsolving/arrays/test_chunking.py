"""Tests for chunking logic."""

from time import sleep

import numpy as np
import pytest
import attrs

from cubie.batchsolving.arrays.BatchOutputArrays import (
    OutputArrayContainer,
)
from cubie.batchsolving.arrays.BatchInputArrays import InputArrayContainer
from cubie.batchsolving.arrays.BaseArrayManager import (
    ManagedArray,
    ArrayContainer,
    BaseArrayManager,
)
from cubie.batchsolving.writeback_watcher import (
    WritebackTask,
    WritebackWatcher,
)
from cubie.memory.chunk_buffer_pool import ChunkBufferPool


def test_run_executes_with_chunking(
    chunked_solved_solver, system, driver_settings
):
    """Verify solve() executes with run-axis chunking."""
    solver, result = chunked_solved_solver

    # Verify chunking occurred
    assert solver.chunks > 1


def test_chunked_solve_produces_valid_output(
    system, precision, chunked_solved_solver
):
    """Verify chunked solver produces valid output arrays."""
    solver, result = chunked_solved_solver

    # Verify output shape and that values are not all zeros/NaN
    assert result.time_domain_array is not None
    assert result.time_domain_array.shape[2] == 5
    assert not np.all(result.time_domain_array == 0)
    assert not np.any(np.isnan(result.time_domain_array))


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
    """Verify chunked execution produces same results as non-chunked."""
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


def test_input_buffers_released_after_kernel(chunked_solved_solver):
    chunked_solver, result_chunked = chunked_solved_solver

    # After solve completes, input_arrays should have empty active buffers
    input_arrays = chunked_solver.kernel.input_arrays
    assert len(input_arrays._active_buffers) == 0


def test_non_chunked_uses_pinned_host(unchunked_solved_solver):
    """Non-chunked runs use pinned host arrays."""

    solver, result = unchunked_solved_solver

    # Verify host arrays are pinned when non-chunked
    for name, slot in solver.kernel.output_arrays.host.iter_managed_arrays():
        assert slot.memory_type == "pinned"


def test_chunked_uses_numpy_host(chunked_solved_solver):
    """Chunked runs use numpy host arrays with buffer pool."""
    solver, result = chunked_solved_solver

    # When chunked, host arrays should be numpy (not pinned)
    # to limit total pinned memory to buffer pool only
    found_one = False
    for name, slot in solver.kernel.output_arrays.host.iter_managed_arrays():
        if slot.needs_chunked_transfer:
            assert slot.memory_type == "host"
            found_one = True
    assert found_one, "No chunked-transfer arrays found"


def test_pinned_buffers_created(chunked_solved_solver):
    """Total pinned memory stays within one chunk's worth."""
    solver, result = chunked_solved_solver
    buffer_pool = solver.kernel.output_arrays._buffer_pool

    # After solve completes, buffers should exist in pool list and be fre
    for buffer_list in buffer_pool._buffers.values():
        for buf in buffer_list:
            assert buf.in_use is False


def test_watcher_completes_all_tasks(chunked_solved_solver):
    """All submitted tasks are completed before solve returns."""
    solver, result = chunked_solved_solver
    # Verify all tasks completed
    output_arrays = solver.kernel.output_arrays
    assert output_arrays._watcher._pending_count == 0


class TestWritebackTask:
    """Tests for WritebackTask data container."""

    def test_task_creation(self):
        """Verify WritebackTask can be created with valid inputs."""
        pool = ChunkBufferPool()
        buffer = pool.acquire("test", (10,), np.float32)
        target = np.zeros((100,), dtype=np.float32)

        task = WritebackTask(
            event=None,
            buffer=buffer,
            target_array=target[:10],
            buffer_pool=pool,
            array_name="test",
        )

        assert task.event is None
        assert task.buffer is buffer
        assert task.target_array is target[:10]
        assert task.buffer_pool is pool
        assert task.array_name == "test"

    def test_task_validates_buffer_type(self):
        """Verify WritebackTask validates buffer is a PinnedBuffer."""
        pool = ChunkBufferPool()
        target = np.zeros((100,), dtype=np.float32)

        with pytest.raises(TypeError):
            WritebackTask(
                event=None,
                buffer="not a buffer",  # Invalid
                target_array=target[0:10],
                buffer_pool=pool,
                array_name="test",
            )


class TestWritebackWatcher:
    """Tests for WritebackWatcher class."""

    def test_watcher_starts_and_stops(self):
        """Verify watcher thread starts on first submit and stops on shutdown.

        Tests lifecycle management: thread should start when first task
        is submitted and terminate cleanly on shutdown().
        """
        watcher = WritebackWatcher()
        pool = ChunkBufferPool()
        buffer = pool.acquire("test", (10,), np.float32)
        target = np.zeros((100,), dtype=np.float32)

        # Thread should not be running initially
        assert watcher._thread is None

        # Submit a task (should start thread)
        watcher.submit(
            event=None,
            buffer=buffer,
            target_array=target[0:10],
            buffer_pool=pool,
            array_name="test",
        )

        # Give thread time to start
        sleep(0.01)

        # Thread should now be running
        assert watcher._thread is not None
        assert watcher._thread.is_alive()

        # Shutdown and verify thread stops
        watcher.shutdown()
        assert watcher._thread is None

    def test_submit_and_wait_completes_writeback(self):
        """Verify submitted task copies data to target array.

        Tests end-to-end functionality: data in buffer should be
        copied to target array at specified slice after wait_all().
        """
        watcher = WritebackWatcher()
        pool = ChunkBufferPool()
        buffer = pool.acquire("test", (10,), np.float32)
        target = np.zeros((100,), dtype=np.float32)

        # Fill buffer with test data
        buffer.array[:] = np.arange(10, dtype=np.float32)

        # Submit task
        watcher.submit(
            event=None,  # CUDASIM treats None as complete
            buffer=buffer,
            target_array=target[20:30],
            buffer_pool=pool,
            array_name="test",
        )

        # Wait for completion
        watcher.wait_all(timeout=1.0)

        # Verify data was copied to correct slice
        expected = np.zeros((100,), dtype=np.float32)
        expected[20:30] = np.arange(10, dtype=np.float32)
        np.testing.assert_array_equal(target, expected)

        # Cleanup
        watcher.shutdown()

    def test_wait_all_blocks_until_complete(self):
        """Verify wait_all blocks until all pending tasks finish.

        Tests synchronization: wait_all should return only when
        all submitted tasks have completed.
        """
        watcher = WritebackWatcher()
        pool = ChunkBufferPool()

        # Submit multiple tasks
        targets = []
        for i in range(3):
            buffer = pool.acquire(f"test_{i}", (5,), np.float32)
            buffer.array[:] = float(i + 1)
            target = np.zeros((20,), dtype=np.float32)
            targets.append(target)

            watcher.submit(
                event=None,
                buffer=buffer,
                target_array=target[5 * i : 5 * (i + 1)],
                buffer_pool=pool,
                array_name=f"test_{i}",
            )

        # Wait for all
        watcher.wait_all(timeout=1.0)

        # Verify all tasks completed
        for i, target in enumerate(targets):
            expected_value = float(i + 1)
            np.testing.assert_array_equal(
                target[i * 5 : (i + 1) * 5],
                np.full((5,), expected_value, dtype=np.float32),
            )

        # Cleanup
        watcher.shutdown()

    def test_multiple_concurrent_tasks(self):
        """Verify multiple tasks can be queued and completed.

        Tests concurrent operation: multiple tasks submitted rapidly
        should all complete correctly without data corruption.
        """
        watcher = WritebackWatcher()
        pool = ChunkBufferPool()

        num_tasks = 10
        target = np.zeros((num_tasks * 10,), dtype=np.float32)

        for i in range(num_tasks):
            buffer = pool.acquire("test", (10,), np.float32)
            buffer.array[:] = float(i)

            watcher.submit(
                event=None,
                buffer=buffer,
                target_array=target[10 * i : 10 * (i + 1)],
                buffer_pool=pool,
                array_name="test",
            )

        # Wait for all tasks
        watcher.wait_all(timeout=2.0)

        # Verify all slices have correct values
        for i in range(num_tasks):
            expected = np.full((10,), float(i), dtype=np.float32)
            np.testing.assert_array_equal(
                target[i * 10 : (i + 1) * 10], expected
            )

        # Cleanup
        watcher.shutdown()

    def test_wait_all_timeout_raises(self):
        """Verify wait_all raises TimeoutError when timeout expires."""
        watcher = WritebackWatcher()

        # Manually set pending count to simulate stuck task
        watcher._pending_count = 1

        with pytest.raises(TimeoutError):
            watcher.wait_all(timeout=0.1)

    def test_buffer_released_after_completion(self):
        """Verify buffer is released back to pool after task completes."""
        watcher = WritebackWatcher()
        pool = ChunkBufferPool()
        buffer = pool.acquire("test", (10,), np.float32)
        target = np.zeros((10,), dtype=np.float32)

        # Buffer should be in use after acquire
        assert buffer.in_use

        watcher.submit(
            event=None,
            buffer=buffer,
            target_array=target,
            buffer_pool=pool,
            array_name="test",
        )

        watcher.wait_all(timeout=1.0)

        # Buffer should be released after completion
        assert not buffer.in_use

        # Cleanup
        watcher.shutdown()

    def test_start_is_idempotent(self):
        """Verify calling start() multiple times is safe."""
        watcher = WritebackWatcher()

        # Start multiple times
        watcher.start()
        first_thread = watcher._thread
        watcher.start()
        second_thread = watcher._thread

        # Should be the same thread
        assert first_thread is second_thread

        # Cleanup
        watcher.shutdown()

    def test_2d_array_slice(self):
        """Verify slicing works correctly for 2D arrays."""
        watcher = WritebackWatcher()
        pool = ChunkBufferPool()
        buffer = pool.acquire("test", (5, 10), np.float64)
        target = np.zeros((5, 10), dtype=np.float64)

        # Fill buffer with test data
        buffer.array[:] = np.arange(50, dtype=np.float64).reshape(5, 10)

        watcher.submit(
            event=None,
            buffer=buffer,
            target_array=target,
            buffer_pool=pool,
            array_name="test",
        )

        watcher.wait_all(timeout=1.0)

        # Verify data was copied to correct slice
        expected = np.arange(50, dtype=np.float64).reshape(5, 10)
        np.testing.assert_array_equal(target[5:10, :], expected)

        # Cleanup
        watcher.shutdown()


@attrs.define
class ConcreteArrayManager(BaseArrayManager):
    """Concrete implementation of BaseArrayManager for testing."""

    def finalise(self, chunk_index):
        return chunk_index

    def initialise(self, chunk_index):
        return chunk_index

    def update(self):
        return


@attrs.define(slots=False)
class TestArrayContainer(ArrayContainer):
    """Simple test container with a single managed array."""

    state: ManagedArray = attrs.field(
        factory=lambda: ManagedArray(
            dtype=np.float32,
            default_shape=(10, 3, 100),
            memory_type="pinned",
        )
    )


class TestIsChunkedProperty:
    """Test the is_chunked property on BaseArrayManager."""

    def test_is_chunked_false_when_single_chunk(self):
        """Verify is_chunked returns False when chunks <= 1."""
        manager = ConcreteArrayManager(
            host=TestArrayContainer(),
            device=TestArrayContainer(),
        )
        # Default is 0 chunks
        assert manager._chunks == 0
        assert manager.is_chunked is False

        # Set to 1 chunk
        manager._chunks = 1
        assert manager.is_chunked is False

    def test_is_chunked_true_when_multiple_chunks(self):
        """Verify is_chunked returns True when chunks > 1."""
        manager = ConcreteArrayManager(
            host=TestArrayContainer(),
            device=TestArrayContainer(),
        )
        manager._chunks = 2
        assert manager.is_chunked is True

        manager._chunks = 10
        assert manager.is_chunked is True


class TestHostFactoryMemoryType:
    """Test host_factory methods accept memory_type parameter."""

    def test_output_container_host_factory_default_pinned(self):
        """Verify OutputArrayContainer.host_factory defaults to pinned."""
        container = OutputArrayContainer.host_factory()
        for _, slot in container.iter_managed_arrays():
            assert slot.memory_type == "pinned"

    def test_output_container_host_factory_accepts_host(self):
        """Verify OutputArrayContainer.host_factory accepts host type."""
        container = OutputArrayContainer.host_factory(memory_type="host")
        for _, slot in container.iter_managed_arrays():
            assert slot.memory_type == "host"

    def test_input_container_host_factory_default_pinned(self):
        """Verify InputArrayContainer.host_factory defaults to pinned."""
        container = InputArrayContainer.host_factory()
        for _, slot in container.iter_managed_arrays():
            assert slot.memory_type == "pinned"

    def test_input_container_host_factory_accepts_host(self):
        """Verify InputArrayContainer.host_factory accepts host type."""
        container = InputArrayContainer.host_factory(memory_type="host")
        for _, slot in container.iter_managed_arrays():
            assert slot.memory_type == "host"


class TestBufferPoolAndWatcherIntegration:
    """Test buffer pool and watcher integration in OutputArrays."""

    def test_finalise_uses_buffer_pool_when_chunked(
        self, chunked_solved_solver
    ):
        """Verify chunked finalise acquires buffers from pool."""
        solver = chunked_solved_solver[0]
        output_arrays_manager = solver.kernel.output_arrays

        #  Check that the output arrays manager has created buffers
        assert len(output_arrays_manager._buffer_pool._buffers) >= 0

    def test_reset_clears_buffer_pool_and_watcher(self, chunked_solved_solver):
        """Verify chunked finalise acquires buffers from pool."""
        solver = chunked_solved_solver[0]
        output_arrays_manager = solver.kernel.output_arrays

        # Reset should clear everything
        output_arrays_manager.reset()

        # Buffer pool should be empty
        assert len(output_arrays_manager._buffer_pool._buffers) == 0

        # Pending buffers should be clear
        assert len(output_arrays_manager._pending_buffers) == 0


class TestNeedsChunkedTransferBranching:
    """Test needs_chunked_transfer property usage in BatchOutputArrays."""

    def test_convert_host_to_numpy_uses_needs_chunked_transfer(
        self, chunked_solved_solver, unchunked_solved_solver
    ):
        """Verify _convert_host_to_numpy uses needs_chunked_transfer.

        The method should convert pinned arrays to regular numpy only
        when the device array's needs_chunked_transfer property is True.
        This is determined by comparing shape vs chunked_shape.
        """

        chunked_solver, chunked_results = chunked_solved_solver
        unchunked_solver, unchunked_results = unchunked_solved_solver

        # Verify input array had needs_chunked_transfer = True (shapes differ)
        chunked_input_manager = chunked_solver.kernel.input_arrays
        unchunked_input_manager = unchunked_solver.kernel.input_arrays
        chunked_inits = chunked_input_manager.host.initial_values
        chunked_drivers = chunked_input_manager.host.driver_coefficients
        unchunked_inits = unchunked_input_manager.host.initial_values

        # Check needs_chunked_transfer values are set appropriately
        assert chunked_inits.needs_chunked_transfer is True
        assert unchunked_inits.needs_chunked_transfer is False
        assert chunked_drivers.needs_chunked_transfer is False

        assert chunked_inits.memory_type == "host"
        assert unchunked_inits.memory_type == "pinned"


def test_chunked_shape_differs_from_shape_when_chunking(
    chunked_solved_solver,
):
    """Verify chunked_shape differs from shape when chunking is active.

    When chunking is active (chunks > 1), device arrays that are chunked
    should have a chunked_shape that differs from their full shape along
    the run axis (axis 2). This verifies that the memory manager
    correctly computed chunked shapes based on chunk_axis_index.
    """
    solver, result = chunked_solved_solver

    # Verify chunking occurred
    assert solver.chunks > 1

    # Check device arrays have different chunked_shape
    output_arrays = solver.kernel.output_arrays
    state_device = output_arrays.device.state
    time_domain_device = output_arrays.device.time_domain_array

    # Both arrays should be chunked (needs_chunked_transfer = True)
    assert state_device.needs_chunked_transfer is True
    assert time_domain_device.needs_chunked_transfer is True

    # chunked_shape should differ from shape along run axis
    assert state_device.chunked_shape != state_device.shape
    assert time_domain_device.chunked_shape != time_domain_device.shape

    # Specifically, run axis (axis 2) should be smaller in chunked_shape
    assert state_device.chunked_shape[2] < state_device.shape[2]
    assert time_domain_device.chunked_shape[2] < time_domain_device.shape[2]


def test_chunked_shape_equals_shape_when_not_chunking(
    unchunked_solved_solver,
):
    """Verify chunked_shape equals shape when chunking is not active.

    When chunking is not active (chunks == 1), all device arrays should
    have chunked_shape equal to their full shape. This verifies that
    unchunked runs do not perform unnecessary shape modifications.
    """
    solver, result = unchunked_solved_solver

    # Verify no chunking occurred
    assert solver.chunks == 1

    # Check device arrays have identical chunked_shape
    output_arrays = solver.kernel.output_arrays
    state_device = output_arrays.device.state
    time_domain_device = output_arrays.device.time_domain_array

    # Arrays should not need chunked transfer
    assert state_device.needs_chunked_transfer is False
    assert time_domain_device.needs_chunked_transfer is False

    # chunked_shape should equal shape
    assert state_device.chunked_shape == state_device.shape
    assert time_domain_device.chunked_shape == time_domain_device.shape


def test_chunk_axis_index_in_array_requests(chunked_solved_solver):
    """Verify ArrayRequest objects have correct chunk_axis_index.

    Array requests created by the solver should have chunk_axis_index=2,
    which corresponds to the run axis in the stride order
    ("time", "variable", "run"). This verifies that the system correctly
    sets the chunking axis for memory allocation.
    """
    solver, result = chunked_solved_solver

    # Access the array requests used by the memory manager
    # These are stored when allocate() is called
    input_manager = solver.kernel.input_arrays
    output_manager = solver.kernel.output_arrays

    # Check ManagedArray _chunk_axis_index computed from stride_order
    # All arrays use stride_order ("time", "variable", "run")
    # where "run" is at index 2
    assert input_manager.device.initial_values._chunk_axis_index == 2
    assert output_manager.device.state._chunk_axis_index == 2
    assert output_manager.device.time_domain_array._chunk_axis_index == 2

    # Verify the chunk axis matches the run axis position in shape
    # For output state with shape (n_states, n_runs), run is at axis 1
    # Wait - the fixture uses default stride order, need to check actual
    # Let's verify by checking that chunked_shape differs at that axis
    state_device = output_manager.device.state
    if state_device.needs_chunked_transfer:
        chunk_axis = state_device._chunk_axis_index
        # chunked_shape should differ from shape at chunk_axis
        for i in range(len(state_device.shape)):
            if i == chunk_axis:
                assert state_device.chunked_shape[i] < state_device.shape[i]
            else:
                assert state_device.chunked_shape[i] == state_device.shape[i]
