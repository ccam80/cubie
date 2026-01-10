"""Simulation-safe CUDA helpers and stand-ins.

This module centralises compatibility utilities for environments running with
``NUMBA_ENABLE_CUDASIM=1``.  It exposes a consistent surface so callers can
import CUDA-facing helpers without branching on simulator state.
"""
from __future__ import annotations

from contextlib import contextmanager
from ctypes import c_void_p
import os
from typing import Any, Callable, Tuple, Union

from numba import cuda
from numba import from_dtype as numba_from_dtype
from numpy import dtype


CUDA_SIMULATION: bool = os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"

# Compile kwargs for cuda.jit decorators
# lineinfo is not supported in CUDASIM mode
compile_kwargs: dict[str, bool] = (
        {} if CUDA_SIMULATION
        else {
            'lineinfo': True,
            # 'debug':True,
            # 'opt':False,
            'fastmath': {
                'nsz': True,
                   'contract': True,
                   'arcp': True,
              },
        }
)


class FakeBaseCUDAMemoryManager: # pragma: no cover - placeholder
    """Minimal stub of a CUDA memory manager."""

    def __init__(self, context: Union[Any, None] = None):
        self.context = context

    def initialize(self) -> None:
        """Placeholder initialize method."""

    def reset(self) -> None:
        """Placeholder reset method."""

    def defer_cleanup(self):
        """Return a no-op context manager."""

        return contextmanager(lambda: (yield))()


class FakeNumbaCUDAMemoryManager(FakeBaseCUDAMemoryManager): # pragma: no cover - placeholder
    """Minimal fake of a CUDA memory manager."""

    handle: int = 0
    ptr: int = 0
    free: int = 0
    total: int = 0

    def __init__(self) -> None:
        super().__init__()


class FakeGetIpcHandleMixin:  # pragma: no cover - placeholder
    """Return a fake IPC handle object."""

    def get_ipc_handle(self):
        class FakeIpcHandle:
            """Trivial stand-in for an IPC handle."""

            def __init__(self) -> None:
                super().__init__()

        return FakeIpcHandle()


class FakeStream:  # pragma: no cover - placeholder
    """Placeholder CUDA stream."""

    handle = c_void_p(0)


class FakeHostOnlyCUDAManager(FakeBaseCUDAMemoryManager):  # pragma: no cover - placeholder
    """Host-only manager used in simulation environments."""


class FakeMemoryPointer:  # pragma: no cover - placeholder
    """Lightweight pointer-like object used in simulation."""

    def __init__(
        self,
        context: Any,
        device_pointer: int,
        size: int,
        finalizer: Union[Any, None] = None,
    ) -> None:
        self.context = context
        self.device_pointer = device_pointer
        self.size = size
        self._cuda_memsize = size
        self.handle = self.device_pointer
        self._finalizer = finalizer


class FakeMemoryInfo:  # pragma: no cover - placeholder
    """Container for fake memory statistics."""

    free = 1024 ** 3
    total = 8 * 1024 ** 3


if CUDA_SIMULATION:  # pragma: no cover - simulated
    from numba.cuda.simulator.cudadrv.devicearray import FakeCUDAArray

    NumbaCUDAMemoryManager = FakeNumbaCUDAMemoryManager
    BaseCUDAMemoryManager = FakeBaseCUDAMemoryManager
    HostOnlyCUDAMemoryManager = FakeHostOnlyCUDAManager
    GetIpcHandleMixin = FakeGetIpcHandleMixin
    MemoryPointer = FakeMemoryPointer
    MemoryInfo = FakeMemoryInfo
    Stream = FakeStream
    DeviceNDArrayBase = FakeCUDAArray
    DeviceNDArray = FakeCUDAArray
    MappedNDArray = FakeCUDAArray
    # local = LocalArrayFactory()

    def current_mem_info() -> Tuple[int, int]:
        """Return fake free and total memory values."""

        fakemem = FakeMemoryInfo()
        return fakemem.free, fakemem.total

    def set_cuda_memory_manager(manager: Any) -> None:
        """Stub for setting a memory manager."""

else:  # pragma: no cover - exercised in GPU environments
    from numba.cuda import (  # type: ignore[attr-defined]
        HostOnlyCUDAMemoryManager,
        MemoryPointer,
        MemoryInfo,
        set_memory_manager as set_cuda_memory_manager,
        is_cuda_array as _is_cuda_array,
    )
    from numba.cuda.cudadrv.driver import (  # type: ignore[attr-defined]
        BaseCUDAMemoryManager,
        NumbaCUDAMemoryManager,
        Stream,
    )
    from numba.cuda.cudadrv.devicearray import (  # type: ignore[attr-defined]
        DeviceNDArrayBase,
        DeviceNDArray,
        MappedNDArray,
    )
    from numba.cuda.cudadrv.driver import GetIpcHandleMixin  # type: ignore[attr-defined]

    def current_mem_info() -> Tuple[int, int]:
        """Return free and total memory from the active CUDA context."""

        return cuda.current_context().get_memory_info()


# --- Caching infrastructure ---
# In CUDASIM mode, provide stub classes since numba caching is unavailable


class _StubCacheLocator:
    """Stub for _CacheLocator in CUDASIM mode.

    Provides minimal interface for CUBIECacheLocator to inherit from
    when running under CUDA simulator.
    """

    def ensure_cache_path(self):
        """Create cache directory if it does not exist."""
        import os
        path = self.get_cache_path()
        os.makedirs(path, exist_ok=True)

    def get_cache_path(self):
        """Return cache directory path. Must be overridden."""
        raise NotImplementedError

    def get_source_stamp(self):
        """Return source freshness stamp. Must be overridden."""
        raise NotImplementedError

    def get_disambiguator(self):
        """Return disambiguator string. Must be overridden."""
        raise NotImplementedError


class _StubCacheImpl:
    """Stub for CacheImpl in CUDASIM mode.

    Provides minimal interface for CUBIECacheImpl to inherit from.
    Serialization methods raise NotImplementedError since CUDA
    context is not available.
    """

    _locator_classes = []

    def reduce(self, data):
        """Reduce data for serialization. Not available in CUDASIM."""
        raise NotImplementedError("Cannot reduce in CUDASIM mode")

    def rebuild(self, target_context, payload):
        """Rebuild from cached payload. Not available in CUDASIM."""
        raise NotImplementedError("Cannot rebuild in CUDASIM mode")

    def check_cachable(self, data):
        """Check if data is cachable. Always False in CUDASIM."""
        return False


class _StubIndexDataCacheFile:
    """Stub for IndexDataCacheFile in CUDASIM mode.

    Provides minimal interface for cache file operations.
    Save/load operations are no-ops since CUDA caching is unavailable.
    """

    def __init__(self, cache_path, filename_base, source_stamp):
        self._cache_path = cache_path
        self._filename_base = filename_base
        self._source_stamp = source_stamp

    def flush(self):
        """Clear the index. No-op in CUDASIM mode."""
        pass

    def save(self, key, data):
        """Save data to cache. Not available in CUDASIM."""
        raise NotImplementedError("Cannot save in CUDASIM mode")

    def load(self, key):
        """Load data from cache. Always returns None in CUDASIM."""
        return None


class _StubCUDACache:
    """Stub for CUDACache in CUDASIM mode.

    Provides minimal interface for CUBIECache to inherit from.
    Cache operations are no-ops or return cache misses.
    """

    _impl_class = None

    def __init__(self, *args, **kwargs):
        """Accept any arguments for compatibility with CUBIECache."""
        self._enabled = False

    def enable(self):
        """Enable caching. Sets internal flag."""
        self._enabled = True

    def disable(self):
        """Disable caching. Clears internal flag."""
        self._enabled = False

    def load_overload(self, sig, target_context):
        """Load cached overload. Always returns None in CUDASIM."""
        return None

    def save_overload(self, sig, data):
        """Save overload to cache. No-op in CUDASIM."""
        pass


if CUDA_SIMULATION:  # pragma: no cover - simulated
    _CacheLocator = _StubCacheLocator
    CacheImpl = _StubCacheImpl
    IndexDataCacheFile = _StubIndexDataCacheFile
    CUDACache = _StubCUDACache
    _CACHING_AVAILABLE = False
else:  # pragma: no cover - exercised in GPU environments
    from numba.cuda.core.caching import (
        _CacheLocator,
        CacheImpl,
        IndexDataCacheFile,
    )
    from numba.cuda.dispatcher import CUDACache
    _CACHING_AVAILABLE = True


def is_cuda_array(value: Any) -> bool:
    """Check whether ``value`` should be treated as a CUDA array."""

    if CUDA_SIMULATION:
        return hasattr(value, "shape")
    return _is_cuda_array(value)


def from_dtype(dt: dtype):
    """Return a CUDA-ready dtype or a simulator-safe placeholder.

    Parameters
    ----------
    dt
        NumPy dtype to adapt for use with CUDA or the simulator.

    Returns
    -------
    dtype
        A Numba CUDA-compatible dtype when running on a real GPU, or
        the original dtype unchanged when running in CUDA simulation
        mode.
    """

    if not CUDA_SIMULATION:
        return numba_from_dtype(dt)
    return dt


def is_devfunc(func: Callable[..., Any]) -> bool:
    """Test whether ``func`` represents a Numba CUDA device function.

    Parameters
    ----------
    func
        Callable object to inspect for CUDA device metadata.

    Returns
    -------
    bool
        ``True`` when ``func`` is tagged as a CUDA device function.
    """

    if CUDA_SIMULATION:  # pragma: no cover - simulated
        return bool(getattr(func, "_device", False))
    target_options = getattr(func, "targetoptions", None)
    if isinstance(target_options, dict):
        return bool(target_options.get("device", False))
    return False


if CUDA_SIMULATION:  # pragma: no cover - simulated
    @cuda.jit(
        device=True,
        inline=True,
    )
    def selp(pred, true_value, false_value):
        return true_value if pred else false_value

    @cuda.jit(
        device=True,
        inline=True,
    )
    def activemask():
        return 0xFFFFFFFF

    @cuda.jit(
        device=True,
        inline=True,
    )
    def all_sync(mask, predicate):
        return predicate

    @cuda.jit(
        device=True,
        inline=True,
    )
    def any_sync(mask, predicate):
        return predicate

    @cuda.jit(
            device=True,
            inline=True,
    )
    def syncwarp(mask):
        pass

    @cuda.jit(
            device=True,
            inline=True,
    )
    def stwt(array, index, value):
        array[index] = value

else:  # pragma: no cover - relies on GPU runtime
    @cuda.jit(
        device=True,
        inline=True,
        **compile_kwargs,
    )
    def selp(pred, true_value, false_value):
        return cuda.selp(pred, true_value, false_value)

    @cuda.jit(
        device=True,
        inline=True,
        **compile_kwargs,
    )
    def activemask():
        return cuda.activemask()

    @cuda.jit(
        device=True,
        inline=True,
        **compile_kwargs,
    )
    def all_sync(mask, predicate):
        return cuda.all_sync(mask, predicate)

    @cuda.jit(
        device=True,
        inline=True,
        **compile_kwargs,
    )
    def any_sync(mask, predicate):
        return cuda.any_sync(mask, predicate)

    @cuda.jit(
        device=True,
        inline=True,
        **compile_kwargs,
    )
    def syncwarp(mask):
        return cuda.syncwarp(mask)

    @cuda.jit(
            device=True,
            inline=True,
            **compile_kwargs,
    )
    def stwt(array, index, value):
        cuda.stwt(array, index, value)


def is_cudasim_enabled() -> bool:
    """Return ``True`` when running under the CUDA simulator."""

    return CUDA_SIMULATION


__all__ = [
    "_CacheLocator",
    "_CACHING_AVAILABLE",
    "activemask",
    "all_sync",
    "BaseCUDAMemoryManager",
    "CacheImpl",
    "compile_kwargs",
    "CUDACache",
    "CUDA_SIMULATION",
    "current_mem_info",
    "DeviceNDArray",
    "DeviceNDArrayBase",
    "FakeBaseCUDAMemoryManager",
    "FakeGetIpcHandleMixin",
    "FakeHostOnlyCUDAManager",
    "FakeMemoryInfo",
    "FakeMemoryPointer",
    "FakeNumbaCUDAMemoryManager",
    "FakeStream",
    "from_dtype",
    "GetIpcHandleMixin",
    "HostOnlyCUDAMemoryManager",
    "IndexDataCacheFile",
    "is_cuda_array",
    "is_cudasim_enabled",
    "is_devfunc",
    "MappedNDArray",
    "MemoryInfo",
    "MemoryPointer",
    "NumbaCUDAMemoryManager",
    "selp",
    "set_cuda_memory_manager",
    "Stream",
    "stwt",
    "syncwarp",
]
