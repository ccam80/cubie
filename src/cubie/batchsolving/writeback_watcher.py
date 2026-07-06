"""Watcher thread for async writeback completion.

Published Classes
-----------------
:class:`PendingBuffer`
    Data structure pairing a pinned buffer with its writeback target.

:class:`WritebackTask`
    Container associating a CUDA event with a pinned buffer, host target,
    and pool for release after completion.

:class:`WritebackWatcher`
    Background daemon thread that polls CUDA events and copies completed
    pinned-buffer data to host arrays.

See Also
--------
:class:`~cubie.batchsolving.arrays.BatchOutputArrays.OutputArrays`
    Consumer that submits writeback tasks during chunked transfers.
:class:`~cubie.memory.chunk_buffer_pool.ChunkBufferPool`
    Pool managing pinned buffer allocation and reuse.
"""

from queue import Queue, Empty
from threading import Thread, Event, Lock
from typing import Optional
from time import sleep, perf_counter

from attrs import define, field
from attrs.validators import (
    instance_of as attrsval_instance_of,
    optional as attrsval_optional,
)
from numpy import ndarray
from numpy.typing import NDArray

from cubie.cuda_simsafe import CUDA_SIMULATION
from cubie.memory.chunk_buffer_pool import PinnedBuffer, ChunkBufferPool


@define
class PendingBuffer:
    """Data structure for pending writeback buffers."""

    buffer: PinnedBuffer
    target_array: NDArray
    array_name: str
    data_shape: tuple
    buffer_pool: ChunkBufferPool


@define
class WritebackTask:
    """Container for a pending writeback operation.

    Attributes
    ----------
    event
        CUDA event to query for completion (None in CUDASIM).
    buffer
        Pinned buffer containing data to copy.
    target_array
        Host array to write data into.
    buffer_pool
        Pool to release buffer to after completion.
    array_name
        Name of the array for pool organization.
    data_shape
        Shape of actual data in buffer (may be smaller than buffer size).
    """

    event: object = field()  # cuda.Event or None
    buffer: PinnedBuffer = field(validator=attrsval_instance_of(PinnedBuffer))
    target_array: ndarray = field(validator=attrsval_instance_of(ndarray))
    buffer_pool: ChunkBufferPool = field(
        validator=attrsval_instance_of(ChunkBufferPool)
    )
    array_name: str = field(validator=attrsval_instance_of(str))
    data_shape: tuple = field(
        default=None, validator=attrsval_optional(attrsval_instance_of(tuple))
    )

    @classmethod
    def from_pending_buffer(
        cls, pending_buffer: "PendingBuffer", event
    ) -> "WritebackTask":
        """Create WritebackTask from PendingBuffer and event."""
        return cls(
            event=event,
            buffer=pending_buffer.buffer,
            target_array=pending_buffer.target_array,
            buffer_pool=pending_buffer.buffer_pool,
            array_name=pending_buffer.array_name,
            data_shape=pending_buffer.data_shape,
        )


class WritebackWatcher:
    """Background thread for polling CUDA events and completing writebacks.

    Monitors a queue of WritebackTask objects, polls their associated
    CUDA events for completion, and copies data from pinned buffers
    to host arrays when ready. Releases buffers back to pool after copy.

    Attributes
    ----------
    _queue : Queue
        Thread-safe queue of pending WritebackTask objects.
    _thread : Thread or None
        Background polling thread.
    _stop_event : Event
        Signal to terminate the polling thread.
    _poll_interval : float
        Seconds between event polls.
    _pending_count : int
        Number of tasks still awaiting completion.
    _lock : Lock
        Thread-safe access to pending count.
    """

    def __init__(self, poll_interval: float = 0.001) -> None:
        """Initialize the watcher.

        Parameters
        ----------
        poll_interval
            Seconds between event polls. Default 0.1ms.
        """
        self._queue: Queue = Queue()
        self._thread: Optional[Thread] = None
        self._stop_event: Event = Event()
        self._poll_interval: float = poll_interval
        self._pending_count: int = 0
        self._lock: Lock = Lock()

    def start(self) -> None:
        """Start the background polling thread."""
        if self._thread is not None and self._thread.is_alive():
            return  # Already running
        self._stop_event.clear()
        self._thread = Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def _submit_task(self, task: WritebackTask) -> None:
        """Submit a writeback task to the queue.

        Parameters
        ----------
        task
            WritebackTask to submit.
        """
        with self._lock:
            self._pending_count += 1
        self._queue.put(task)
        # Start thread if not running
        self.start()

    def submit_from_pending_buffer(
        self,
        pending_buffer: "PendingBuffer",
        event: object,
    ) -> None:
        """Submit a writeback task from a PendingBuffer.

        Parameters
        ----------
        pending_buffer
            PendingBuffer containing writeback info.
        event
            CUDA event to monitor for completion.
        """
        self._submit_task(
            WritebackTask.from_pending_buffer(pending_buffer, event)
        )

    def submit(
        self,
        event: object,
        buffer: PinnedBuffer,
        target_array: ndarray,
        buffer_pool: ChunkBufferPool,
        array_name: str,
        data_shape: Optional[tuple] = None,
    ) -> None:
        """Submit a writeback task for async completion.

        Parameters
        ----------
        event
            CUDA event to monitor for completion.
        buffer
            Pinned buffer containing source data.
        target_array
            Host array to write into.
        buffer_pool
            Pool to release buffer to.
        array_name
            Name of the array for pool organization.
        data_shape
            Shape of actual data in buffer. When provided, only this
            portion of the buffer is copied to target. Used when buffer
            is larger than actual data (e.g., last chunk).
        """
        task = WritebackTask(
            event=event,
            buffer=buffer,
            target_array=target_array,
            buffer_pool=buffer_pool,
            array_name=array_name,
            data_shape=data_shape,
        )
        self._submit_task(task)

    def wait_all(self, timeout: Optional[float] = 10) -> None:
        """Block until all pending writebacks complete.

        Parameters
        ----------
        timeout
            Maximum seconds to wait. None waits indefinitely.

        Raises
        ------
        TimeoutError
            If timeout expires before completion.
        """
        start_time = perf_counter()
        while True:
            with self._lock:
                if self._pending_count == 0:
                    return
            if timeout is not None:
                elapsed = perf_counter() - start_time
                if elapsed >= timeout:
                    raise TimeoutError(
                        f"wait_all timed out after {timeout} seconds"
                    )
            sleep(self._poll_interval)

    def shutdown(self) -> None:
        """Stop the polling thread gracefully."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _poll_loop(self) -> None:
        """Main polling loop for the background thread."""
        # Collect tasks that are not yet complete for re-queuing
        pending_tasks = []

        while not self._stop_event.is_set():
            # Process all tasks in the pending list first
            still_pending = []
            for task in pending_tasks:
                if self._process_task(task):
                    with self._lock:
                        self._pending_count -= 1
                else:
                    still_pending.append(task)
            pending_tasks = still_pending

            # Try to get new tasks from queue (non-blocking)
            try:
                task = self._queue.get_nowait()
                if self._process_task(task):
                    with self._lock:
                        self._pending_count -= 1
                else:
                    pending_tasks.append(task)
            except Empty:
                pass
                # No tasks available in queue; continue polling loop.
            sleep(self._poll_interval)

        # On shutdown, process remaining tasks synchronously
        # to ensure all data is written
        while not self._queue.empty():
            try:
                task = self._queue.get_nowait()
                pending_tasks.append(task)
            except Empty:
                break

        # Complete all pending tasks before shutdown
        while pending_tasks:
            still_pending = []
            for task in pending_tasks:
                if self._process_task(task):
                    with self._lock:
                        self._pending_count -= 1
                else:
                    still_pending.append(task)
            pending_tasks = still_pending
            if still_pending:
                sleep(self._poll_interval)

    def _process_task(self, task: WritebackTask) -> bool:
        """Process a single writeback task.

        Parameters
        ----------
        task
            Task to process.

        Returns
        -------
        bool
            True if task completed, False if still pending.
        """
        # In CUDASIM mode or event is None, treat as immediately complete
        if CUDA_SIMULATION or task.event is None:
            is_complete = True
        else:
            # Query event for completion (returns True when complete)
            is_complete = task.event.query()

        if is_complete:
            # Copy buffer data to target array at specified slice
            # If data_shape provided, only copy that portion of the buffer
            if task.data_shape is not None:
                buffer_slice = tuple(slice(0, s) for s in task.data_shape)
                task.target_array[:] = task.buffer.array[buffer_slice]
            else:
                task.target_array[:] = task.buffer.array
            # Release buffer back to pool
            task.buffer_pool.release(task.buffer)
            return True

        return False
