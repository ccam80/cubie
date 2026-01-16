"""Tests for ChunkBufferPool and PinnedBuffer classes."""

import threading
import numpy as np

from cubie.memory.chunk_buffer_pool import ChunkBufferPool, PinnedBuffer


class TestPinnedBuffer:
    """Tests for PinnedBuffer class."""

    def test_pinned_buffer_creation(self):
        """Verify PinnedBuffer can be created with required attributes."""
        arr = np.zeros((10, 20), dtype=np.float32)
        buffer = PinnedBuffer(buffer_id=0, array=arr)

        assert buffer.buffer_id == 0
        assert buffer.array is arr
        assert buffer.in_use is False

    def test_pinned_buffer_in_use_flag(self):
        """Verify in_use flag can be set."""
        arr = np.zeros((10,), dtype=np.float64)
        buffer = PinnedBuffer(buffer_id=1, array=arr, in_use=True)

        assert buffer.in_use is True


class TestChunkBufferPool:
    """Tests for ChunkBufferPool class."""

    def test_acquire_returns_pinned_buffer(self):
        """Verify acquire returns a PinnedBuffer with correct shape and dtype.
        """
        pool = ChunkBufferPool()
        shape = (100, 50)
        dtype = np.float32

        buffer = pool.acquire("state", shape, dtype)

        assert isinstance(buffer, PinnedBuffer)
        assert buffer.array.shape == shape
        assert buffer.array.dtype == dtype
        assert buffer.in_use is True

    def test_release_marks_buffer_available(self):
        """Verify released buffer can be reacquired."""
        pool = ChunkBufferPool()
        shape = (10, 20)
        dtype = np.float64

        buffer = pool.acquire("observables", shape, dtype)
        assert buffer.in_use is True

        pool.release(buffer)
        assert buffer.in_use is False

    def test_acquire_reuses_released_buffer(self):
        """Verify pool reuses buffers instead of allocating new ones."""
        pool = ChunkBufferPool()
        shape = (50, 30)
        dtype = np.float32

        # Acquire and release a buffer
        buffer1 = pool.acquire("state", shape, dtype)
        buffer1_id = buffer1.buffer_id
        pool.release(buffer1)

        # Acquire again - should reuse the same buffer
        buffer2 = pool.acquire("state", shape, dtype)

        assert buffer2.buffer_id == buffer1_id
        assert buffer2 is buffer1

    def test_acquire_allocates_new_when_all_in_use(self):
        """Verify new buffer allocated when existing ones are in use."""
        pool = ChunkBufferPool()
        shape = (25, 25)
        dtype = np.float32

        buffer1 = pool.acquire("test", shape, dtype)
        buffer2 = pool.acquire("test", shape, dtype)

        assert buffer1.buffer_id != buffer2.buffer_id
        assert buffer1 is not buffer2

    def test_acquire_allocates_new_for_different_shape(self):
        """Verify new buffer allocated when shape differs."""
        pool = ChunkBufferPool()
        dtype = np.float32

        buffer1 = pool.acquire("test", (10, 10), dtype)
        pool.release(buffer1)

        # Different shape should allocate new buffer
        buffer2 = pool.acquire("test", (20, 10), dtype)

        assert buffer1.buffer_id != buffer2.buffer_id

    def test_acquire_allocates_new_for_different_dtype(self):
        """Verify new buffer allocated when dtype differs."""
        pool = ChunkBufferPool()
        shape = (15, 15)

        buffer1 = pool.acquire("test", shape, np.float32)
        pool.release(buffer1)

        # Different dtype should allocate new buffer
        buffer2 = pool.acquire("test", shape, np.float64)

        assert buffer1.buffer_id != buffer2.buffer_id

    def test_clear_removes_all_buffers(self):
        """Verify clear empties the pool."""
        pool = ChunkBufferPool()

        # Allocate some buffers
        pool.acquire("state", (10, 10), np.float32)
        pool.acquire("observables", (20, 5), np.float64)

        pool.clear()

        # Pool should be empty
        assert len(pool._buffers) == 0
        assert pool._next_id == 0

    def test_buffers_organized_by_array_name(self):
        """Verify buffers are organized by array name in pool."""
        pool = ChunkBufferPool()

        pool.acquire("state", (10,), np.float32)
        pool.acquire("observables", (20,), np.float64)
        pool.acquire("state", (10,), np.float32)

        assert "state" in pool._buffers
        assert "observables" in pool._buffers
        assert len(pool._buffers["state"]) == 2
        assert len(pool._buffers["observables"]) == 1

    def test_thread_safety_concurrent_acquire_release(self):
        """Verify concurrent operations don't cause race conditions."""
        pool = ChunkBufferPool()
        shape = (100, 100)
        dtype = np.float32
        num_threads = 10
        iterations_per_thread = 50
        errors = []

        def worker(thread_id):
            try:
                for _ in range(iterations_per_thread):
                    buffer = pool.acquire(f"array_{thread_id}", shape, dtype)
                    assert buffer.in_use is True
                    assert buffer.array.shape == shape
                    pool.release(buffer)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=(i,))
            for i in range(num_threads)
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"

    def test_concurrent_acquire_same_array_name(self):
        """Verify concurrent acquire on same array name is thread-safe."""
        pool = ChunkBufferPool()
        shape = (50, 50)
        dtype = np.float32
        num_threads = 5
        acquired_buffers = []
        lock = threading.Lock()

        def worker():
            buffer = pool.acquire("shared_array", shape, dtype)
            with lock:
                acquired_buffers.append(buffer)

        threads = [
            threading.Thread(target=worker) for _ in range(num_threads)
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # All buffers should be unique since none were released
        buffer_ids = [b.buffer_id for b in acquired_buffers]
        assert len(buffer_ids) == len(set(buffer_ids))

    def test_buffer_id_uniqueness(self):
        """Verify each allocated buffer has a unique ID."""
        pool = ChunkBufferPool()

        buffers = [
            pool.acquire(f"array_{i}", (10,), np.float32)
            for i in range(10)
        ]

        buffer_ids = [b.buffer_id for b in buffers]
        assert len(buffer_ids) == len(set(buffer_ids))
