"""Reusable pinned buffer pool for chunked array transfers."""

from typing import Dict, List, Tuple
from threading import Lock

from attrs import define, field
from attrs.validators import instance_of as attrsval_instance_of
from numba import cuda
from numpy import ndarray, zeros as np_zeros
from numpy import dtype as np_dtype

from cubie.cuda_simsafe import CUDA_SIMULATION


@define
class PinnedBuffer:
    """Wrapper for a reusable pinned memory buffer.

    Attributes
    ----------
    buffer_id : int
        Unique identifier for this buffer.
    array : ndarray
        The pinned numpy array.
    in_use : bool
        Whether the buffer is currently in use.
    """

    buffer_id: int = field(validator=attrsval_instance_of(int))
    array: ndarray = field(validator=attrsval_instance_of(ndarray))
    in_use: bool = field(default=False, validator=attrsval_instance_of(bool))


@define
class ChunkBufferPool:
    """Pool of reusable pinned buffers for chunked transfers.

    Manages allocation and lifecycle of pinned memory buffers used
    for staging data during chunked device transfers. Buffers are
    sized for one chunk and reused across chunks.

    Attributes
    ----------
    _buffers : Dict[str, List[PinnedBuffer]]
        Pool of buffers organized by array name.
    _lock : Lock
        Thread-safe access to buffer pool.
    _next_id : int
        Counter for unique buffer IDs.
    """

    _buffers: Dict[str, List[PinnedBuffer]] = field(factory=dict)
    _lock: Lock = field(factory=Lock)
    _next_id: int = field(default=0)

    def acquire(
        self,
        array_name: str,
        shape: Tuple[int, ...],
        dtype: np_dtype,
    ) -> PinnedBuffer:
        """Acquire a pinned buffer for the given array.

        Parameters
        ----------
        array_name
            Identifier for the array type (e.g., 'state', 'observables').
        shape
            Required shape for the buffer.
        dtype
            Data type for the buffer elements.

        Returns
        -------
        PinnedBuffer
            A buffer ready for use, either reused or newly allocated.
        """
        with self._lock:
            # Check pool for available buffer matching shape/dtype
            if array_name in self._buffers:
                for buf in self._buffers[array_name]:
                    if (not buf.in_use and
                            buf.array.shape == shape and
                            buf.array.dtype == dtype):
                        buf.in_use = True
                        return buf

            # Allocate new buffer since no suitable one found
            new_buffer = self._allocate_buffer(shape, dtype)
            new_buffer.in_use = True

            # Add to pool
            if array_name not in self._buffers:
                self._buffers[array_name] = []
            self._buffers[array_name].append(new_buffer)

            return new_buffer

    def release(self, buffer: PinnedBuffer) -> None:
        """Release a buffer back to the pool.

        Parameters
        ----------
        buffer
            The buffer to release.
        """
        with self._lock:
            buffer.in_use = False

    def clear(self) -> None:
        """Clear all buffers from the pool.

        Should be called on cleanup or error to free pinned memory.
        """
        with self._lock:
            self._buffers.clear()
            self._next_id = 0

    def _allocate_buffer(
        self,
        shape: Tuple[int, ...],
        dtype: np_dtype,
    ) -> PinnedBuffer:
        """Allocate a new pinned buffer.

        Parameters
        ----------
        shape
            Shape for the buffer.
        dtype
            Data type for the buffer elements.

        Returns
        -------
        PinnedBuffer
            Newly allocated pinned buffer.
        """
        # Use np.zeros for CUDASIM mode, cuda.pinned_array otherwise
        if CUDA_SIMULATION:
            arr = np_zeros(shape, dtype=dtype)
        else:
            arr = cuda.pinned_array(shape, dtype=dtype)

        buffer_id = self._next_id
        self._next_id += 1

        return PinnedBuffer(buffer_id=buffer_id, array=arr)
