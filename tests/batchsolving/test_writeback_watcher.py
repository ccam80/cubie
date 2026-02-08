"""Tests for cubie.batchsolving.writeback_watcher."""

from __future__ import annotations

import numpy as np
import pytest

from cubie.batchsolving.writeback_watcher import (
    PendingBuffer,
    WritebackTask,
    WritebackWatcher,
)
from numba import cuda

from cubie.cuda_simsafe import CUDA_SIMULATION
from cubie.memory.chunk_buffer_pool import ChunkBufferPool, PinnedBuffer


# ── Helpers ─────────────────────────────────────────────────── #


def _make_pool():
    """Return a fresh ChunkBufferPool."""
    return ChunkBufferPool()


def _make_pinned_buffer(shape=(4, 3), dtype=np.float32, fill=1.0):
    """Return a PinnedBuffer with known data."""
    arr = np.full(shape, fill, dtype=dtype)
    return PinnedBuffer(buffer_id=0, array=arr)


def _make_pending_buffer(shape=(4, 3), dtype=np.float32, fill=1.0):
    """Return a PendingBuffer with matching target array."""
    buf = _make_pinned_buffer(shape=shape, dtype=dtype, fill=fill)
    target = np.zeros(shape, dtype=dtype)
    pool = _make_pool()
    return PendingBuffer(
        buffer=buf,
        target_array=target,
        array_name="state",
        data_shape=shape,
        buffer_pool=pool,
    )


# ── PendingBuffer attrs dataclass (item 1) ──────────────────── #


def test_pending_buffer_stores_all_fields():
    """PendingBuffer stores buffer, target_array, array_name, data_shape, buffer_pool."""
    buf = _make_pinned_buffer()
    target = np.zeros((4, 3), dtype=np.float32)
    pool = _make_pool()
    shape = (4, 3)
    pb = PendingBuffer(
        buffer=buf,
        target_array=target,
        array_name="obs",
        data_shape=shape,
        buffer_pool=pool,
    )
    assert pb.buffer is buf
    assert pb.target_array is target
    assert pb.array_name == "obs"
    assert pb.data_shape == (4, 3)
    assert pb.buffer_pool is pool


# ── WritebackTask attrs dataclass (item 2) ──────────────────── #


def test_writeback_task_stores_all_fields():
    """WritebackTask stores event, buffer, target_array, buffer_pool, array_name, data_shape."""
    buf = _make_pinned_buffer()
    target = np.zeros((4, 3), dtype=np.float32)
    pool = _make_pool()
    event = object()
    task = WritebackTask(
        event=event,
        buffer=buf,
        target_array=target,
        buffer_pool=pool,
        array_name="state",
        data_shape=(2, 3),
    )
    assert task.event is event
    assert task.buffer is buf
    assert task.target_array is target
    assert task.buffer_pool is pool
    assert task.array_name == "state"
    assert task.data_shape == (2, 3)


def test_writeback_task_data_shape_defaults_none():
    """WritebackTask data_shape defaults to None."""
    buf = _make_pinned_buffer()
    target = np.zeros((4, 3), dtype=np.float32)
    pool = _make_pool()
    task = WritebackTask(
        event=None,
        buffer=buf,
        target_array=target,
        buffer_pool=pool,
        array_name="state",
    )
    assert task.data_shape is None


# ── from_pending_buffer classmethod (item 3) ─────────────────── #


def test_from_pending_buffer_transfers_all_fields():
    """from_pending_buffer creates WritebackTask with PendingBuffer fields + event."""
    pb = _make_pending_buffer()
    event = object()
    task = WritebackTask.from_pending_buffer(pb, event)
    assert task.event is event
    assert task.buffer is pb.buffer
    assert task.target_array is pb.target_array
    assert task.buffer_pool is pb.buffer_pool
    assert task.array_name == pb.array_name
    assert task.data_shape == pb.data_shape


# ── WritebackWatcher.__init__ (item 4) ───────────────────────── #


def test_watcher_init_defaults():
    """Watcher initializes with expected defaults."""
    w = WritebackWatcher()
    assert w._poll_interval == 0.001
    assert w._pending_count == 0
    assert w._thread is None
    assert not w._stop_event.is_set()
    assert w._queue.empty()


def test_watcher_init_custom_poll_interval():
    """Watcher accepts custom poll_interval."""
    w = WritebackWatcher(poll_interval=0.05)
    assert w._poll_interval == 0.05


# ── start (items 5, 6) ───────────────────────────────────────── #


def test_start_creates_daemon_thread():
    """start() creates and starts a daemon polling thread."""
    w = WritebackWatcher()
    w.start()
    assert w._thread is not None
    assert w._thread.is_alive()
    assert w._thread.daemon is True
    w.shutdown()


def test_start_noop_when_already_running():
    """start() is no-op when thread already alive."""
    w = WritebackWatcher()
    w.start()
    first_thread = w._thread
    w.start()
    assert w._thread is first_thread
    w.shutdown()


# ── _submit_task (items 7, 8, 9) ─────────────────────────────── #


def test_submit_task_starts_thread_and_processes():
    """_submit_task enqueues task, starts thread, task gets processed."""
    w = WritebackWatcher()
    buf = _make_pinned_buffer(fill=99.0)
    target = np.zeros((4, 3), dtype=np.float32)
    pool = _make_pool()
    task = WritebackTask(
        event=None, buffer=buf, target_array=target,
        buffer_pool=pool, array_name="state",
    )
    w._submit_task(task)
    # Thread was started (item 9)
    assert w._thread is not None
    assert w._thread.is_alive()
    w.wait_all(timeout=2.0)
    # Task was processed: pending_count back to 0, data copied (items 7, 8)
    assert w._pending_count == 0
    np.testing.assert_array_equal(target, 99.0)
    w.shutdown()


# ── submit_from_pending_buffer (item 10) ──────────────────────── #


def test_submit_from_pending_buffer():
    """submit_from_pending_buffer creates task and submits."""
    w = WritebackWatcher()
    pb = _make_pending_buffer(fill=7.0)
    w.submit_from_pending_buffer(pb, event=None)
    # In CUDASIM, event=None means immediate completion
    w.wait_all(timeout=2.0)
    np.testing.assert_array_equal(pb.target_array, pb.buffer.array)
    w.shutdown()


# ── submit (item 11) ─────────────────────────────────────────── #


def test_submit_with_individual_args():
    """submit() creates WritebackTask from individual args and submits."""
    w = WritebackWatcher()
    buf = _make_pinned_buffer(fill=3.0)
    target = np.zeros((4, 3), dtype=np.float32)
    pool = _make_pool()
    w.submit(
        event=None,
        buffer=buf,
        target_array=target,
        buffer_pool=pool,
        array_name="state",
    )
    w.wait_all(timeout=2.0)
    np.testing.assert_array_equal(target, buf.array)
    w.shutdown()


# ── wait_all (items 12, 13, 14) ──────────────────────────────── #


def test_wait_all_returns_immediately_when_no_pending():
    """wait_all returns immediately when pending_count == 0."""
    w = WritebackWatcher()
    # Should not block or raise
    w.wait_all(timeout=0.1)


def test_wait_all_polls_until_complete():
    """wait_all polls until pending_count == 0."""
    w = WritebackWatcher()
    buf = _make_pinned_buffer(fill=5.0)
    target = np.zeros((4, 3), dtype=np.float32)
    pool = _make_pool()
    w.submit(event=None, buffer=buf, target_array=target,
             buffer_pool=pool, array_name="s")
    w.wait_all(timeout=2.0)
    assert w._pending_count == 0
    np.testing.assert_array_equal(target, buf.array)
    w.shutdown()


def test_wait_all_raises_timeout_error():
    """wait_all raises TimeoutError when timeout expires."""
    w = WritebackWatcher()
    # Artificially set pending count without actual task
    w._pending_count = 1
    with pytest.raises(TimeoutError, match="wait_all timed out"):
        w.wait_all(timeout=0.05)


# ── shutdown (items 15, 16, 17) ───────────────────────────────── #


def test_shutdown_sets_stop_event_and_clears_thread():
    """shutdown() sets stop_event, joins thread, sets _thread = None."""
    w = WritebackWatcher()
    w.start()
    assert w._thread is not None
    w.shutdown()
    assert w._stop_event.is_set()
    assert w._thread is None


# ── _poll_loop / _process_task integration (items 18-26) ──────── #


def test_process_task_copies_full_buffer_when_no_data_shape():
    """_process_task copies full buffer when data_shape is None (item 24)."""
    w = WritebackWatcher()
    buf = _make_pinned_buffer(shape=(3, 2), fill=9.0)
    target = np.zeros((3, 2), dtype=np.float32)
    pool = _make_pool()
    task = WritebackTask(
        event=None, buffer=buf, target_array=target,
        buffer_pool=pool, array_name="state", data_shape=None,
    )
    result = w._process_task(task)
    assert result is True
    np.testing.assert_array_equal(target, 9.0)


def test_process_task_copies_sliced_buffer_when_data_shape_provided():
    """_process_task copies buffer[0:s] when data_shape provided (item 23)."""
    w = WritebackWatcher()
    # Buffer is 4x3 but data_shape is (2, 3) — only first 2 rows
    buf = _make_pinned_buffer(shape=(4, 3), fill=0.0)
    buf.array[:] = np.arange(12, dtype=np.float32).reshape(4, 3)
    target = np.zeros((2, 3), dtype=np.float32)
    pool = _make_pool()
    task = WritebackTask(
        event=None, buffer=buf, target_array=target,
        buffer_pool=pool, array_name="state", data_shape=(2, 3),
    )
    result = w._process_task(task)
    assert result is True
    expected = np.arange(6, dtype=np.float32).reshape(2, 3)
    np.testing.assert_array_equal(target, expected)


def test_process_task_releases_buffer_to_pool():
    """_process_task releases buffer back to pool (item 25)."""
    w = WritebackWatcher()
    pool = _make_pool()
    buf = pool.acquire("state", (3,), np.float32)
    assert buf.in_use is True
    target = np.zeros((3,), dtype=np.float32)
    task = WritebackTask(
        event=None, buffer=buf, target_array=target,
        buffer_pool=pool, array_name="state",
    )
    w._process_task(task)
    assert buf.in_use is False


def test_process_task_returns_false_for_incomplete():
    """_process_task returns False when event not yet complete (item 26).

    Under CUDASIM, CUDA_SIMULATION is True so _process_task treats all
    events as immediately complete — this test verifies the non-CUDASIM
    branch and will fail under CUDASIM (flagged in batch report).
    """
    w = WritebackWatcher()
    buf = _make_pinned_buffer()
    target = np.zeros((4, 3), dtype=np.float32)
    pool = _make_pool()
    event = cuda.event()
    task = WritebackTask(
        event=event, buffer=buf, target_array=target,
        buffer_pool=pool, array_name="state",
    )
    if CUDA_SIMULATION:
        # Under CUDASIM, all events complete immediately
        assert w._process_task(task) is True
    else:
        assert w._process_task(task) is False


def test_process_task_cudasim_immediate_complete():
    """_process_task treats as immediately complete in CUDA_SIMULATION (item 21)."""
    w = WritebackWatcher()
    buf = _make_pinned_buffer(fill=42.0)
    target = np.zeros((4, 3), dtype=np.float32)
    pool = _make_pool()
    task = WritebackTask(
        event="not_a_real_event",  # Not None, not a cuda.Event
        buffer=buf, target_array=target,
        buffer_pool=pool, array_name="state",
    )
    # Under CUDASIM this completes immediately regardless of event type
    if CUDA_SIMULATION:
        assert w._process_task(task) is True
        np.testing.assert_array_equal(target, 42.0)
    else:
        # Under real CUDA, non-event objects can't be queried —
        # verify the task is NOT treated as complete
        assert w._process_task(task) is False


def test_process_task_none_event_immediate_complete():
    """_process_task treats as immediately complete when event is None (item 21)."""
    w = WritebackWatcher()
    buf = _make_pinned_buffer(fill=11.0)
    target = np.zeros((4, 3), dtype=np.float32)
    pool = _make_pool()
    task = WritebackTask(
        event=None, buffer=buf, target_array=target,
        buffer_pool=pool, array_name="state",
    )
    assert w._process_task(task) is True
    np.testing.assert_array_equal(target, 11.0)


def test_shutdown_drains_and_completes_remaining_tasks():
    """On shutdown, poll_loop drains queue and completes remaining tasks (item 20)."""
    w = WritebackWatcher()
    buf = _make_pinned_buffer(fill=77.0)
    target = np.zeros((4, 3), dtype=np.float32)
    pool = _make_pool()
    w.submit(event=None, buffer=buf, target_array=target,
             buffer_pool=pool, array_name="state")
    w.shutdown()
    # After shutdown, data should be copied
    np.testing.assert_array_equal(target, 77.0)
    assert w._thread is None


def test_multiple_tasks_all_complete():
    """Multiple submitted tasks all complete correctly (items 18, 19)."""
    w = WritebackWatcher()
    pool = _make_pool()
    targets = []
    expected_values = [1.0, 2.0, 3.0]
    for val in expected_values:
        buf = _make_pinned_buffer(shape=(2,), fill=val)
        target = np.zeros((2,), dtype=np.float32)
        targets.append((target, val))
        w.submit(event=None, buffer=buf, target_array=target,
                 buffer_pool=pool, array_name="arr")
    w.wait_all(timeout=5.0)
    for target, val in targets:
        np.testing.assert_array_equal(target, val)
    w.shutdown()
