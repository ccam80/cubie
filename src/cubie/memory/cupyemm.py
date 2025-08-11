""" Cupy async- or sync-memory pool External Memory Manager plugin for Numba"""

from contextlib import contextmanager
from typing import TYPE_CHECKING
import logging
from os import environ

from numba import cuda
import ctypes

if environ.get("NUMBA_ENABLE_CUDASIM", "0") == "1":
    from cubie.cudasim_utils import FakeGetIpcHandleMixin as GetIpcHandleMixin
    from cubie.cudasim_utils import (FakeHostOnlyCUDAManager as
                                     HostOnlyCUDAMemoryManager)
    from cubie.cudasim_utils import FakeMemoryPointer as MemoryPointer
    from cubie.cudasim_utils import FakeMemoryInfo as MemoryInfo
else:
    from numba.cuda import (
        GetIpcHandleMixin,
        HostOnlyCUDAMemoryManager,
        MemoryPointer,
        MemoryInfo
    )
try:
    import cupy as cp
except ImportError:
    cp = None



logger = logging.getLogger(__name__)

def _numba_stream_ptr(nb_stream):
    """
    Returns CUstream pointer (int) from a numba.cuda.cudadrv.driver.Stream.
    Tries common layouts across Numba versions.
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
    Context manager to override CuPy's *current* stream with a Numba stream.
    """
    def __init__(self, nb_stream):
        if cp is None:
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
        if self._mgr_is_cupy:
            ptr = _numba_stream_ptr(self.nb_stream)
            if ptr:
                self.cupy_ext_stream = cp.cuda.ExternalStream(ptr)
                self.cupy_ext_stream.__enter__()
            return self
        else:
            return self

    def __exit__(self, exc_type, exc, tb):
        if self._mgr_is_cupy:
            if self.cupy_ext_stream is not None:
                self.cupy_ext_stream.__exit__(exc_type, exc, tb)
                self.cupy_ext_stream = None

class CuPyNumbaManager(GetIpcHandleMixin, HostOnlyCUDAMemoryManager):
    """Numba EMM plugin for using cupy memory pools to allocate.

    Drawn from the tutorial example at:
    https://github.com/numba/nvidia-cuda-tutorial/blob/main/session-5/examples/cupy_emm_plugin.py

    Extended to handle passing numba-generated streams as CuPy external
    streams, such that the allocations are stream-ordered when using the async
    allocator.
    """
    def __init__(self, context):
        if cp is None:
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
            finalizer=self._make_finalizer(cp_mp, nbytes)
        )

    def _make_finalizer(self, cp_mp, nbytes):
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
        """Returns the free and total memory in bytes from the CuPy memory pool
        (not the whole device, necessarily!)
        """
        return MemoryInfo(free=self._mp.free_bytes(),
                          total=self._mp.total_bytes())

    def initialize(self):
        super().initialize()

    def reset(self, stream=None):
        """ Free all blocks with optional stream for async operations

        This is called without a stream argument when the context is reset. To
        run the operation in one stream, call this function by itself using
        cuda.current_context().memory_manager.reset(stream)
        """
        super().reset()
        if self._mp:
            self._mp.free_all_blocks(stream=stream)

    @contextmanager
    def defer_cleanup(self):
        # This doesn't actually defer returning memory back to the pool, but
        # returning memory to the pool will not interrupt async operations like
        # an actual cudaFree / cuMemFree would.
        with super().defer_cleanup():
            yield

    @property
    def interface_version(self):
        return 1


class CuPyAsyncNumbaManager(CuPyNumbaManager):
    """Numba EMM plugin for using CuPy MemoryAsyncPool to allocate and free."""
    def __init__(self, context):
        super().__init__(context=context)

    def initialize(self):
        super().initialize()
        self._mp = cp.cuda.MemoryAsyncPool()

    def memalloc(self, nbytes):
        if self._testing:
            self._testout = "async"
        return super().memalloc(nbytes)


class CuPySyncNumbaManager(CuPyNumbaManager):
    """Numba EMM plugin for using CuPy MemoryPool to allocate and free."""
    def __init__(self, context):
        super().__init__(context=context)

    def initialize(self):
        super().initialize()
        # Get the default memory pool for this context.
        self._mp = cp.get_default_memory_pool()

    def memalloc(self, nbytes):
        if self._testing:
            self._testout = "sync"
        return super().memalloc(nbytes)

