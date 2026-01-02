"""Factory for creating buffer allocator device functions."""

import numpy as np
from numba import cuda


class AllocatorFactory:
    """Factory that generates buffer allocator device functions.

    Parameters
    ----------
    buffer_size : int
        Size of the local array to allocate.

    Attributes
    ----------
    buffer_size : int
        Size of the local array to allocate.
    """

    def __init__(self, buffer_size: int):
        """Initialize the allocator factory.

        Parameters
        ----------
        buffer_size : int
            Size of the local array to allocate.
        """
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        self.buffer_size = buffer_size

    def build(self):
        """Build and return a CUDA device function allocator.

        Returns
        -------
        callable
            CUDA device function that allocates a local array.
            Signature: () -> local_array
        """
        _buffer_size = self.buffer_size

        @cuda.jit(device=True, inline=True)
        def allocate_buffer():
            """Allocate a local array buffer."""
            return cuda.local.array(_buffer_size, dtype=np.float32)

        return allocate_buffer
