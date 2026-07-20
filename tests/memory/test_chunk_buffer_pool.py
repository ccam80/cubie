"""Tests for cubie.memory.chunk_buffer_pool."""

from __future__ import annotations

import threading

import numpy as np

from cubie.memory.chunk_buffer_pool import ChunkBufferPool, PinnedBuffer


# ── PinnedBuffer ──────────────────────────────────────────────── #

def test_pinned_buffer_construction():
    """PinnedBuffer stores buffer_id, array, and in_use (default False)."""
    arr = np.zeros((10, 20), dtype=np.float32)
    buf = PinnedBuffer(buffer_id=0, array=arr)
    assert buf.buffer_id == 0
    assert buf.array is arr
    assert buf.in_use is False


def test_pinned_buffer_in_use_override():
    """in_use can be set to True at construction."""
    arr = np.zeros((5,), dtype=np.float64)
    buf = PinnedBuffer(buffer_id=1, array=arr, in_use=True)
    assert buf.in_use is True


# ── acquire ───────────────────────────────────────────────────── #

def test_acquire_returns_buffer_with_correct_shape_dtype():
    """acquire returns a PinnedBuffer matching requested shape and dtype."""
    pool = ChunkBufferPool()
    buf = pool.acquire("state", (100, 4), np.float32)
    assert buf.array.shape == (100, 4)
    assert buf.array.dtype == np.float32
    assert buf.in_use is True


def test_acquire_reuses_released_buffer():
    """acquire reuses a released buffer with matching shape/dtype."""
    pool = ChunkBufferPool()
    buf1 = pool.acquire("state", (50, 3), np.float32)
    bid = buf1.buffer_id
    pool.release(buf1)
    buf2 = pool.acquire("state", (50, 3), np.float32)
    assert buf2.buffer_id == bid
    assert buf2 is buf1


def test_acquire_allocates_new_when_all_in_use():
    """acquire grows the pool when in-use buffers block reuse.

    Growth is forced open so the assertion does not depend on the
    machine's free RAM at test time; the headroom-exhausted branch
    is exercised by the blocking tests below.
    """
    pool = _UnthrottledPool()
    buf1 = pool.acquire("x", (10,), np.float32)
    buf2 = pool.acquire("x", (10,), np.float32)
    assert buf1.buffer_id != buf2.buffer_id


def test_acquire_allocates_new_for_different_shape():
    """acquire allocates new buffer when shape differs."""
    pool = ChunkBufferPool()
    buf1 = pool.acquire("x", (10,), np.float32)
    pool.release(buf1)
    buf2 = pool.acquire("x", (20,), np.float32)
    assert buf1.buffer_id != buf2.buffer_id


def test_acquire_allocates_new_for_different_dtype():
    """acquire allocates new buffer when dtype differs."""
    pool = ChunkBufferPool()
    buf1 = pool.acquire("x", (10,), np.float32)
    pool.release(buf1)
    buf2 = pool.acquire("x", (10,), np.float64)
    assert buf1.buffer_id != buf2.buffer_id


def test_acquire_creates_new_array_name_entry():
    """First acquire for a name creates a new entry in _buffers."""
    pool = ChunkBufferPool()
    pool.acquire("new_name", (5,), np.float32)
    assert "new_name" in pool._buffers
    assert len(pool._buffers["new_name"]) == 1


# ── release ───────────────────────────────────────────────────── #

def test_release_marks_not_in_use():
    """release sets buffer.in_use to False."""
    pool = ChunkBufferPool()
    buf = pool.acquire("x", (10,), np.float32)
    assert buf.in_use is True
    pool.release(buf)
    assert buf.in_use is False


# ── clear ─────────────────────────────────────────────────────── #

def test_clear_empties_pool_and_resets_id():
    """clear removes all buffers and resets _next_id to 0."""
    pool = ChunkBufferPool()
    pool.acquire("a", (10,), np.float32)
    pool.acquire("b", (20,), np.float64)
    pool.clear()
    assert len(pool._buffers) == 0
    assert pool._next_id == 0


# ── _allocate_buffer ──────────────────────────────────────────── #

def test_allocate_buffer_increments_id():
    """Each allocated buffer gets an incrementing buffer_id."""
    pool = ChunkBufferPool()
    ids = []
    for i in range(5):
        buf = pool.acquire(f"arr_{i}", (3,), np.float32)
        ids.append(buf.buffer_id)
    assert ids == [0, 1, 2, 3, 4]


# ── Thread safety ─────────────────────────────────────────────── #

def test_thread_safe_concurrent_acquire_release():
    """Concurrent acquire/release does not raise or corrupt state."""
    pool = ChunkBufferPool()
    errors = []

    def worker(tid):
        try:
            for _ in range(20):
                buf = pool.acquire(f"arr_{tid}", (10,), np.float32)
                pool.release(buf)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(errors) == 0


# ── headroom-bounded growth ───────────────────────────────────── #

class _ThrottledPool(ChunkBufferPool):
    """Pool whose headroom check is forced closed for testing."""

    def _headroom_allows(self, shape, dtype):
        return False


class _UnthrottledPool(ChunkBufferPool):
    """Pool whose headroom check is forced open for testing."""

    def _headroom_allows(self, shape, dtype):
        return True


def test_acquire_grows_first_buffer_even_without_headroom():
    """A label with nothing in flight always gets one buffer."""
    pool = _ThrottledPool()
    buf = pool.acquire("state", (10,), np.float32)
    assert buf.in_use is True


def test_acquire_blocks_until_release_when_headroom_exhausted():
    """With headroom exhausted, acquire waits for an in-flight
    buffer to be released rather than growing the pool."""
    pool = _ThrottledPool()
    first = pool.acquire("state", (10,), np.float32)
    acquired = []
    started = threading.Event()

    def worker():
        started.set()
        acquired.append(pool.acquire("state", (10,), np.float32))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    started.wait(timeout=2.0)
    thread.join(timeout=0.2)
    assert thread.is_alive()
    assert acquired == []

    pool.release(first)
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert acquired[0] is first
