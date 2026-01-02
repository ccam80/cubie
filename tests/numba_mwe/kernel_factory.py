"""Factory for creating CUDA kernels that use the allocator."""

from numba import cuda

from tests.numba_mwe.allocator_factory import AllocatorFactory


class KernelFactory:
    """Factory that builds CUDA kernels using an allocator factory.

    Parameters
    ----------
    allocator_factory : AllocatorFactory
        Factory for creating the buffer allocator device function.

    Attributes
    ----------
    allocator_factory : AllocatorFactory
        Factory for creating the buffer allocator device function.
    """

    def __init__(self, allocator_factory: AllocatorFactory):
        """Initialize the kernel factory.

        Parameters
        ----------
        allocator_factory : AllocatorFactory
            Factory for creating the buffer allocator device function.
        """
        if not isinstance(allocator_factory, AllocatorFactory):
            raise TypeError(
                "allocator_factory must be an AllocatorFactory instance"
            )
        self.allocator_factory = allocator_factory

    def build(self):
        """Build and return a CUDA kernel.

        The kernel:
        1. Gets thread index via cuda.grid(1)
        2. Returns early if thread index >= n_threads
        3. Calls allocator to get local array
        4. Assigns 1 to first element of local array
        5. Copies value from local array to output array

        Returns
        -------
        callable
            CUDA kernel function.
            Signature: (output_array, n_threads) -> None
        """
        # Get allocator and capture in closure
        allocator = self.allocator_factory.build()

        @cuda.jit
        def kernel(output, n_threads):
            """CUDA kernel that uses local array allocation.

            Parameters
            ----------
            output : array
                Output array to write results.
            n_threads : int
                Number of threads that should do work.
            """
            thread_idx = cuda.grid(1)

            # Early return for excess threads
            # BUG: In CUDASIM, threads may continue past this return
            if thread_idx >= n_threads:
                return

            # Allocate local array (fails if thread continues past return)
            local_buffer = allocator()

            # Use the local buffer
            local_buffer[0] = 1.0

            # Copy to output
            output[thread_idx] = local_buffer[0]

        return kernel
