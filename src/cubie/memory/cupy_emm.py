"""
CuPy async/sync memory pool External Memory Manager plugin for Numba.

This module provides Numba External Memory Manager (EMM) plugins that integrate
CuPy's memory pools for GPU memory allocation. It supports both synchronous and
asynchronous memory pools and provides stream-ordered allocations when using
the async allocator.
"""

from contextlib import contextmanager
import logging
from os import environ

from numba import cuda
import ctypes

# no cover: start
if environ.get("NUMBA_ENABLE_CUDASIM", "0") == "1":
    from cubie.cudasim_utils import FakeGetIpcHandleMixin as GetIpcHandleMixin
    from cubie.cudasim_utils import (
        FakeHostOnlyCUDAManager as HostOnlyCUDAMemoryManager,
    )
    from cubie.cudasim_utils import FakeMemoryPointer as MemoryPointer
    from cubie.cudasim_utils import FakeMemoryInfo as MemoryInfo
# no cover: end
else:
    from numba.cuda import (
        GetIpcHandleMixin,
        HostOnlyCUDAMemoryManager,
        MemoryPointer,
        MemoryInfo,
    )


logger = logging.getLogger(__name__)


def _numba_stream_ptr(nb_stream):
    """
    Extract CUstream pointer from a numba.cuda.cudadrv.driver.Stream.

    Parameters
    ----------
    nb_stream : numba.cuda.cudadrv.driver.Stream or None
        Numba stream object to extract pointer from.

    Returns
    -------
    int or None
        CUstream pointer as integer, or None if extraction fails.

    Notes
    -----
    Tries common layouts across Numba versions to maintain compatibility.
    """
    if nb_stream is None:
        return None
    h = getattr(nb_stream, "handle", None)
    if h is None:
        return None
    # ctypes.c_void_p or int-like
    if isinstance(h, ctypes.c_void_p):
        return int(h.value) if h.value is not None else None
    try:
        return int(getattr(h, "value", h))
    except Exception:
        return None


class current_cupy_stream:
    """
    Context manager to override CuPy's current stream with a Numba stream.

    Parameters
    ----------
    nb_stream : numba.cuda.cudadrv.driver.Stream
        Numba stream to use as CuPy's current stream.

    Attributes
    ----------
    nb_stream : numba.cuda.cudadrv.driver.Stream
        The Numba stream being used.
    cupy_ext_stream : cupy.cuda.ExternalStream or None
        CuPy external stream wrapper around the Numba stream.

    Notes
    -----
    This context manager only has effect when the current CUDA memory manager
    is a CuPy-based manager. Otherwise, it acts as a no-op context manager.
    """

    def __init__(self, nb_stream):
        try:
            import cupy as cp
        except ImportError: # pragma: no cover
            raise ImportError(
                "To use Cupy memory managers, you must install cupy: pip "
                "install cupy-cuda12x (assuming CUDA toolkit 12.x installed)]"
            )
        self.nb_stream = nb_stream

        self.cupy_ext_stream = None
        try:
            self._mgr_is_cupy = cuda.current_context().memory_manager.is_cupy
        except AttributeError:  # Numba allocators have no such attribute
            self._mgr_is_cupy = False

    def __enter__(self):
        """
        Enter the context and set up CuPy external stream if applicable.

        Returns
        -------
        self
            Returns self for use in with statement.
        """
        try:
            import cupy as cp
        except ImportError: # pragma: no cover
            raise ImportError(
                "To use Cupy memory managers, you must install cupy: pip "
                "install cupy-cuda12x (assuming CUDA toolkit 12.x installed)]"
            )
        if self._mgr_is_cupy:
            ptr = _numba_stream_ptr(self.nb_stream)
            if ptr:
                self.cupy_ext_stream = cp.cuda.ExternalStream(ptr)
                self.cupy_ext_stream.__enter__()
            return self
        else:
            return self

    def __exit__(self, exc_type, exc, tb):
        """
        Exit the context and clean up CuPy external stream.

        Parameters
        ----------
        exc_type : type or None
            Exception type if an exception occurred.
        exc : Exception or None
            Exception instance if an exception occurred.
        tb : traceback or None
            Traceback object if an exception occurred.
        """
        if self._mgr_is_cupy:
            if self.cupy_ext_stream is not None:
                self.cupy_ext_stream.__exit__(exc_type, exc, tb)
                self.cupy_ext_stream = None


class CuPyNumbaManager(GetIpcHandleMixin, HostOnlyCUDAMemoryManager):
    """
    Base Numba EMM plugin for using CuPy memory pools to allocate.

    Parameters
    ----------
    context : numba.cuda.cudadrv.driver.Context
        CUDA context for memory management.

    Attributes
    ----------
    is_cupy : bool
        Flag indicating this is a CuPy-based memory manager.

    Notes
    -----
    Drawn from the tutorial example at:
    https://github.com/numba/nvidia-cuda-tutorial/blob/main/session-5/examples/cupy_emm_plugin.py

    Extended to handle passing numba-generated streams as CuPy external
    streams, such that the allocations are stream-ordered when using the async
    allocator.
    """

    def __init__(self, context):
        try:
            import cupy as cp
        except ImportError:
            raise ImportError(
                "To use Cupy memory managers, you must install cupy: pip "
                "install cupy-cuda12x (assuming CUDA toolkit 12.x installed)]"
            )
        super().__init__(context=context)
        # We keep a record of all allocations, and remove each allocation
        # record in the finalizer, which results in it being returned back to
        # the CuPy memory pool.
        self._allocations = {}
        # The CuPy memory pool.
        self._mp = None
        self._ctx = context
        self.is_cupy = True

        # These provide a way for tests to check who's allocating what.
        self._testing = False
        self._testout = None

    def memalloc(self, nbytes):
        """
        Allocate memory from the CuPy pool.

        Parameters
        ----------
        nbytes : int
            Number of bytes to allocate.

        Returns
        -------
        MemoryPointer
            Numba memory pointer wrapping the CuPy allocation.
        """
        try:
            import cupy as cp
        except ImportError: # pragma: no cover
            raise ImportError(
                "To use Cupy memory managers, you must install cupy: pip "
                "install cupy-cuda12x (assuming CUDA toolkit 12.x installed)]"
            )
        # Allocate from the CuPy pool and wrap the result in a MemoryPointer as
        # required by Numba.
        cp_mp = self._mp.malloc(nbytes)
        logger.debug("Allocated %d bytes at %x" % (nbytes, cp_mp.ptr))
        logger.debug("on stream %s" % (cp.cuda.get_current_stream()))
        self._allocations[cp_mp.ptr] = cp_mp
        return MemoryPointer(
            cuda.current_context(),
            ctypes.c_void_p(int(cp_mp.ptr)),
            nbytes,
            finalizer=self._make_finalizer(cp_mp, nbytes),
        )

    def _make_finalizer(self, cp_mp, nbytes):
        """
        Create a finalizer function for memory cleanup.

        Parameters
        ----------
        cp_mp : cupy memory pool allocation
            CuPy memory pool allocation to be cleaned up.
        nbytes : int
            Number of bytes in the allocation.

        Returns
        -------
        callable
            Finalizer function that removes the allocation reference.
        """
        try:
            import cupy as cp
        except ImportError: # pragma: no cover
            raise ImportError(
                "To use Cupy memory managers, you must install cupy: pip "
                "install cupy-cuda12x (assuming CUDA toolkit 12.x installed)]"
            )
        allocations = self._allocations
        ptr = cp_mp.ptr

        def finalizer():
            logger.debug("Freeing %d bytes at %x" % (nbytes, ptr))
            logger.debug("on stream %s" % (cp.cuda.get_current_stream()))
            # Removing the last reference to the allocation causes it to be
            # garbage-collected and returned to the pool.
            allocations.pop(ptr)

        return finalizer

    def get_memory_info(self):
        """
        Get memory information from the CuPy memory pool.

        Returns
        -------
        MemoryInfo
            Object containing free and total memory in bytes from the pool.

        Notes
        -----
        Returns information from the CuPy memory pool, not the whole device.
        """
        return MemoryInfo(
            free=self._mp.free_bytes(), total=self._mp.total_bytes()
        )

    def initialize(self):
        """Initialize the memory manager."""
        super().initialize()

    def reset(self, stream=None):
        """
        Free all blocks with optional stream for async operations.

        Parameters
        ----------
        stream : cupy.cuda.Stream or None, optional
            Stream for async operations. If None, operates synchronously.

        Notes
        -----
        This is called without a stream argument when the context is reset. To
        run the operation in one stream, call this function by itself using
        cuda.current_context().memory_manager.reset(stream)
        """
        super().reset()
        if self._mp:
            self._mp.free_all_blocks(stream=stream)

    @contextmanager
    def defer_cleanup(self):
        """
        Context manager for deferring memory cleanup operations.

        Yields
        ------
        None

        Notes
        -----
        This doesn't actually defer returning memory back to the pool, but
        returning memory to the pool will not interrupt async operations like
        an actual cudaFree / cuMemFree would.
        """
        with super().defer_cleanup():
            yield

    @property
    def interface_version(self):
        """
        Get the EMM interface version.

        Returns
        -------
        int
            Interface version number.
        """
        return 1


class CuPyAsyncNumbaManager(CuPyNumbaManager):
    """
    Numba EMM plugin using CuPy MemoryAsyncPool for allocation and freeing.

    Parameters
    ----------
    context : numba.cuda.cudadrv.driver.Context
        CUDA context for memory management.

    Notes
    -----
    Uses CuPy's asynchronous memory pool which provides stream-ordered
    memory operations.
    """

    def __init__(self, context):
        super().__init__(context=context)

    def initialize(self):
        """Initialize the async memory pool."""
        try:
            import cupy as cp
        except ImportError: # pragma: no cover
            raise ImportError(
                "To use Cupy memory managers, you must install cupy: pip "
                "install cupy-cuda12x (assuming CUDA toolkit 12.x installed)]"
            )
        super().initialize()
        self._mp = cp.cuda.MemoryAsyncPool()

    def memalloc(self, nbytes):
        """
        Allocate memory from the async CuPy pool.

        Parameters
        ----------
        nbytes : int
            Number of bytes to allocate.

        Returns
        -------
        MemoryPointer
            Numba memory pointer wrapping the CuPy allocation.
        """
        if self._testing:
            self._testout = "async"
        return super().memalloc(nbytes)


class CuPySyncNumbaManager(CuPyNumbaManager):
    """
    Numba EMM plugin using CuPy MemoryPool for allocation and freeing.

    Parameters
    ----------
    context : numba.cuda.cudadrv.driver.Context
        CUDA context for memory management.

    Notes
    -----
    Uses CuPy's synchronous memory pool which provides standard
    memory operations.
    """

    def __init__(self, context):
        super().__init__(context=context)

    def initialize(self):
        """Initialize the sync memory pool."""
        try:
            import cupy as cp
        except ImportError: # pragma: no cover
            raise ImportError(
                "To use Cupy memory managers, you must install cupy: pip "
                "install cupy-cuda12x (assuming CUDA toolkit 12.x installed)]"
            )
        super().initialize()
        # Get the default memory pool for this context.
        self._mp = cp.get_default_memory_pool()

    def memalloc(self, nbytes):
        """
        Allocate memory from the sync CuPy pool.

        Parameters
        ----------
        nbytes : int
            Number of bytes to allocate.

        Returns
        -------
        MemoryPointer
            Numba memory pointer wrapping the CuPy allocation.
        """
        if self._testing:
            self._testout = "sync"
        return super().memalloc(nbytes)
