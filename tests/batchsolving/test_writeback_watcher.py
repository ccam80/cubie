"""Tests for WritebackWatcher and WritebackTask."""

import numpy as np
import pytest
from time import sleep

from cubie.memory.chunk_buffer_pool import ChunkBufferPool, PinnedBuffer
from cubie.batchsolving.writeback_watcher import WritebackTask, WritebackWatcher


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
            target_array=target,
            slice_tuple=(slice(0, 10),),
            buffer_pool=pool,
            array_name="test",
        )

        assert task.event is None
        assert task.buffer is buffer
        assert task.target_array is target
        assert task.slice_tuple == (slice(0, 10),)
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
                target_array=target,
                slice_tuple=(slice(0, 10),),
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
            target_array=target,
            slice_tuple=(slice(0, 10),),
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
            target_array=target,
            slice_tuple=(slice(20, 30),),
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
                target_array=target,
                slice_tuple=(slice(i * 5, (i + 1) * 5),),
                buffer_pool=pool,
                array_name=f"test_{i}",
            )

        # Wait for all
        watcher.wait_all(timeout=1.0)

        # Verify all tasks completed
        for i, target in enumerate(targets):
            expected_value = float(i + 1)
            np.testing.assert_array_equal(
                target[i * 5:(i + 1) * 5],
                np.full((5,), expected_value, dtype=np.float32)
            )

        # Cleanup
        watcher.shutdown()

    def test_cudasim_immediate_completion(self):
        """Verify tasks complete immediately in CUDASIM mode.

        Tests CUDASIM handling: when event is None, tasks should be
        treated as immediately complete.
        """
        watcher = WritebackWatcher()
        pool = ChunkBufferPool()
        buffer = pool.acquire("test", (10,), np.float32)
        target = np.zeros((10,), dtype=np.float32)

        # Fill buffer with test data
        buffer.array[:] = 42.0

        # Submit task with None event (simulates CUDASIM)
        watcher.submit(
            event=None,
            buffer=buffer,
            target_array=target,
            slice_tuple=(slice(None),),
            buffer_pool=pool,
            array_name="test",
        )

        # Should complete very quickly since event is None
        watcher.wait_all(timeout=0.5)

        # Verify data was copied
        np.testing.assert_array_equal(target, np.full((10,), 42.0))

        # Verify buffer was released
        assert not buffer.in_use

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
                target_array=target,
                slice_tuple=(slice(i * 10, (i + 1) * 10),),
                buffer_pool=pool,
                array_name="test",
            )

        # Wait for all tasks
        watcher.wait_all(timeout=2.0)

        # Verify all slices have correct values
        for i in range(num_tasks):
            expected = np.full((10,), float(i), dtype=np.float32)
            np.testing.assert_array_equal(
                target[i * 10:(i + 1) * 10], expected
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
            slice_tuple=(slice(None),),
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
        target = np.zeros((20, 10), dtype=np.float64)

        # Fill buffer with test data
        buffer.array[:] = np.arange(50, dtype=np.float64).reshape(5, 10)

        watcher.submit(
            event=None,
            buffer=buffer,
            target_array=target,
            slice_tuple=(slice(5, 10), slice(None)),
            buffer_pool=pool,
            array_name="test",
        )

        watcher.wait_all(timeout=1.0)

        # Verify data was copied to correct slice
        expected = np.arange(50, dtype=np.float64).reshape(5, 10)
        np.testing.assert_array_equal(target[5:10, :], expected)

        # Cleanup
        watcher.shutdown()
