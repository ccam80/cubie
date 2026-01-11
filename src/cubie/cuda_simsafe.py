"""Simulation-safe CUDA helpers and stand-ins.

This module centralises compatibility utilities for environments running with
``NUMBA_ENABLE_CUDASIM=1``.  It exposes a consistent surface so callers can
import CUDA-facing helpers without branching on simulator state.
"""
from __future__ import annotations

from abc import ABCMeta, abstractmethod
import contextlib
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
# Vendored from numba-cuda on 2026-01-11 for CUDASIM compatibility.
# Source: numba_cuda/numba/cuda/core/caching.py

try:
    from numba.cuda.serialize import dumps as _numba_dumps
except ImportError:
    import pickle

    def _numba_dumps(obj):
        return pickle.dumps(obj)


class IndexDataCacheFile:
    """Vendored from numba-cuda for CUDASIM compatibility.

    Manages index (.nbi) and data (.nbc) files for cache storage.
    Source: numba_cuda/numba/cuda/core/caching.py (2026-01-11)
    """

    def __init__(self, cache_path, filename_base, source_stamp):
        import numba
        self._cache_path = cache_path
        self._index_name = "%s.nbi" % (filename_base,)
        self._index_path = os.path.join(self._cache_path, self._index_name)
        self._data_name_pattern = "%s.{number:d}.nbc" % (filename_base,)
        self._source_stamp = source_stamp
        self._version = numba.__version__

    def flush(self):
        self._save_index({})

    def save(self, key, data):
        import itertools
        overloads = self._load_index()
        try:
            data_name = overloads[key]
        except KeyError:
            existing = set(overloads.values())
            for i in itertools.count(1):
                data_name = self._data_name(i)
                if data_name not in existing:
                    break
            overloads[key] = data_name
            self._save_index(overloads)
        self._save_data(data_name, data)

    def load(self, key):
        overloads = self._load_index()
        data_name = overloads.get(key)
        if data_name is None:
            return
        try:
            return self._load_data(data_name)
        except OSError:
            return

    def _load_index(self):
        import pickle
        try:
            with open(self._index_path, "rb") as f:
                version = pickle.load(f)
                data = f.read()
        except FileNotFoundError:
            return {}
        if version != self._version:
            return {}
        stamp, overloads = pickle.loads(data)
        if stamp != self._source_stamp:
            return {}
        return overloads

    def _save_index(self, overloads):
        import pickle
        data = self._source_stamp, overloads
        data = self._dump(data)
        with self._open_for_write(self._index_path) as f:
            pickle.dump(self._version, f, protocol=-1)
            f.write(data)

    def _load_data(self, name):
        import pickle
        path = self._data_path(name)
        with open(path, "rb") as f:
            data = f.read()
        return pickle.loads(data)

    def _save_data(self, name, data):
        data = self._dump(data)
        path = self._data_path(name)
        with self._open_for_write(path) as f:
            f.write(data)

    def _data_name(self, number):
        return self._data_name_pattern.format(number=number)

    def _data_path(self, name):
        return os.path.join(self._cache_path, name)

    def _dump(self, obj):
        return _numba_dumps(obj)

    @contextlib.contextmanager
    def _open_for_write(self, filepath):
        import uuid
        uid = uuid.uuid4().hex[:16]
        tmpname = "%s.tmp.%s" % (filepath, uid)
        try:
            with open(tmpname, "wb") as f:
                yield f
            os.replace(tmpname, filepath)
        except Exception:
            try:
                os.unlink(tmpname)
            except OSError:
                pass
            raise


class _CacheLocator(metaclass=ABCMeta):
    """Vendored from numba-cuda for CUDASIM compatibility.

    Abstract base for filesystem locators for caching functions.
    Source: numba_cuda/numba/cuda/core/caching.py (2026-01-11)
    """

    def ensure_cache_path(self):
        import tempfile
        path = self.get_cache_path()
        os.makedirs(path, exist_ok=True)
        tempfile.TemporaryFile(dir=path).close()

    @abstractmethod
    def get_cache_path(self):
        """Return the directory the function is cached in."""

    @abstractmethod
    def get_source_stamp(self):
        """Get a timestamp representing source code freshness."""

    @abstractmethod
    def get_disambiguator(self):
        """Get a string disambiguator for this locator's function."""

    @classmethod
    def from_function(cls, py_func, py_file):
        """Create a locator instance for the given function."""
        raise NotImplementedError


class CacheImpl(metaclass=ABCMeta):
    """Vendored from numba-cuda for CUDASIM compatibility.

    Provides core machinery for caching serialization.
    Source: numba_cuda/numba/cuda/core/caching.py (2026-01-11)
    """

    _locator_classes = []

    @property
    def filename_base(self):
        return self._filename_base

    @property
    def locator(self):
        return self._locator

    @abstractmethod
    def reduce(self, data):
        """Returns the serialized form of the data."""

    @abstractmethod
    def rebuild(self, target_context, reduced_data):
        """Returns the de-serialized form of the reduced_data."""

    @abstractmethod
    def check_cachable(self, data):
        """Returns True if data is cachable; otherwise False."""


class _Cache(metaclass=ABCMeta):
    """Vendored from numba-cuda for CUDASIM compatibility.

    Abstract base for per-function compilation cache.
    Source: numba_cuda/numba/cuda/core/caching.py (2026-01-11)
    """

    @property
    @abstractmethod
    def cache_path(self):
        """The base filesystem path of this cache."""

    @abstractmethod
    def load_overload(self, sig, target_context):
        """Load an overload for the given signature."""

    @abstractmethod
    def save_overload(self, sig, data):
        """Save the overload for the given signature."""

    @abstractmethod
    def enable(self):
        """Enable the cache."""

    @abstractmethod
    def disable(self):
        """Disable the cache."""

    @abstractmethod
    def flush(self):
        """Flush the cache."""


class CUDACache(_Cache):
    """Vendored from numba-cuda for CUDASIM compatibility.

    Cache that saves and loads CUDA kernels and compile results.
    Source: numba_cuda/numba/cuda/dispatcher.py (2026-01-11)
    """

    _impl_class = None

    def __init__(self, py_func):
        import errno
        self._name = repr(py_func)
        self._py_func = py_func
        self._impl = self._impl_class(py_func)
        self._cache_path = self._impl.locator.get_cache_path()
        source_stamp = self._impl.locator.get_source_stamp()
        filename_base = self._impl.filename_base
        self._cache_file = IndexDataCacheFile(
            cache_path=self._cache_path,
            filename_base=filename_base,
            source_stamp=source_stamp,
        )
        self.enable()

    def __repr__(self):
        return "<%s py_func=%r>" % (self.__class__.__name__, self._name)

    @property
    def cache_path(self):
        return self._cache_path

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def flush(self):
        self._cache_file.flush()

    def load_overload(self, sig, target_context):
        import errno
        with self._guard_against_spurious_io_errors():
            return self._load_overload(sig, target_context)

    def _load_overload(self, sig, target_context):
        if not self._enabled:
            return
        key = self._index_key(sig, target_context.codegen())
        data = self._cache_file.load(key)
        if data is not None:
            data = self._impl.rebuild(target_context, data)
        return data

    def save_overload(self, sig, data):
        with self._guard_against_spurious_io_errors():
            self._save_overload(sig, data)

    def _save_overload(self, sig, data):
        if not self._enabled:
            return
        if not self._impl.check_cachable(data):
            return
        self._impl.locator.ensure_cache_path()
        key = self._index_key(sig, data.codegen)
        data = self._impl.reduce(data)
        self._cache_file.save(key, data)

    @contextlib.contextmanager
    def _guard_against_spurious_io_errors(self):
        import errno
        if os.name == "nt":
            try:
                yield
            except OSError as e:
                if e.errno != errno.EACCES:
                    raise
        else:
            yield

    def _index_key(self, sig, codegen):
        """Compute index key for the given signature and codegen."""
        import hashlib
        codebytes = self._py_func.__code__.co_code
        if self._py_func.__closure__ is not None:
            cvars = tuple([x.cell_contents for x in self._py_func.__closure__])
            cvarbytes = _numba_dumps(cvars)
        else:
            cvarbytes = b""

        def hasher(x):
            return hashlib.sha256(x).hexdigest()

        return (
            sig,
            codegen.magic_tuple(),
            (hasher(codebytes), hasher(cvarbytes)),
        )


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
    "_Cache",
    "_CacheLocator",
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
