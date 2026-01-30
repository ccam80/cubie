"""Simulation-safe CUDA helpers and stand-ins.

This module centralises compatibility utilities for environments
running with ``NUMBA_ENABLE_CUDASIM=1``. It exposes a consistent
surface so callers can import CUDA-facing helpers without branching
on simulator state.

Published Functions
-------------------
:func:`from_dtype`
    Return a CUDA-ready or simulator-safe dtype.
:func:`is_devfunc`
    Test whether a callable is a Numba CUDA device function.
:func:`is_cuda_array`
    Check whether a value should be treated as a CUDA array.
:func:`is_cudasim_enabled`
    Return whether the CUDA simulator is active.

Published Device Functions
--------------------------
``selp``, ``activemask``, ``all_sync``, ``any_sync``,
``syncwarp``, ``stwt``
    Wrappers around CUDA intrinsics with CUDASIM fallbacks.

Published Constants
-------------------
:data:`CUDA_SIMULATION`
    ``True`` when ``NUMBA_ENABLE_CUDASIM=1``.
:data:`compile_kwargs`
    Default keyword arguments for ``@cuda.jit`` decorators.

See Also
--------
:mod:`cubie._utils`
    Imports ``compile_kwargs`` and ``is_devfunc`` from this module.
:mod:`cubie.memory.mem_manager`
    Uses memory manager classes exported here.
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
    {}
    if CUDA_SIMULATION
    else {
        # "lineinfo": True,
        # 'debug':True,
        # 'opt':False,
        "fastmath": {
            "nsz": True,
            "contract": True,
            "arcp": True,
        },
    }
)


class FakeBaseCUDAMemoryManager:  # pragma: no cover - placeholder
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


class FakeNumbaCUDAMemoryManager(
    FakeBaseCUDAMemoryManager
):  # pragma: no cover - placeholder
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


class FakeHostOnlyCUDAManager(
    FakeBaseCUDAMemoryManager
):  # pragma: no cover - placeholder
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

    free = 1024**3
    total = 8 * 1024**3


if CUDA_SIMULATION:  # pragma: no cover - simulated
    from numba.cuda.simulator.cudadrv.devicearray import FakeCUDAArray
    from cubie.vendored.numba_cuda_cache import CUDACache

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
    from numba.cuda.dispatcher import CUDACache  # noqa: Linter can't find
    # cuda.dispatcher

    def current_mem_info() -> Tuple[int, int]:
        """Return free and total memory from the active CUDA context."""

        return cuda.current_context().get_memory_info()


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
        """Select ``true_value`` or ``false_value`` based on predicate.

        Parameters
        ----------
        pred : bool
            Condition to evaluate.
        true_value : numba scalar
            Value returned when ``pred`` is true.
        false_value : numba scalar
            Value returned when ``pred`` is false.

        Returns
        -------
        numba scalar
            Selected value.
        """
        return true_value if pred else false_value

    @cuda.jit(
        device=True,
        inline=True,
    )
    def activemask():
        """Return the active thread mask for the current warp.

        Returns
        -------
        int32
            Bitmask of active threads (all-ones in CUDASIM).
        """
        return 0xFFFFFFFF

    @cuda.jit(
        device=True,
        inline=True,
    )
    def all_sync(mask, predicate):
        """Return whether all threads in ``mask`` satisfy ``predicate``.

        Parameters
        ----------
        mask : int32
            Active thread mask.
        predicate : bool
            Per-thread condition.

        Returns
        -------
        bool
            ``True`` if all masked threads satisfy ``predicate``.
        """
        return predicate

    @cuda.jit(
        device=True,
        inline=True,
    )
    def any_sync(mask, predicate):
        """Return whether any thread in ``mask`` satisfies ``predicate``.

        Parameters
        ----------
        mask : int32
            Active thread mask.
        predicate : bool
            Per-thread condition.

        Returns
        -------
        bool
            ``True`` if any masked thread satisfies ``predicate``.
        """
        return predicate

    @cuda.jit(
        device=True,
        inline=True,
    )
    def syncwarp(mask):
        """Synchronise threads within a warp.

        Parameters
        ----------
        mask : int32
            Active thread mask.
        """
        pass

    @cuda.jit(
        device=True,
        inline=True,
    )
    def stwt(array, index, value):
        """Store-through write: write ``value`` to ``array[index]``.

        Parameters
        ----------
        array : device array
            Target array.
        index : int32
            Element index.
        value : numba scalar
            Value to write.
        """
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
    "activemask",
    "all_sync",
    "BaseCUDAMemoryManager",
    "compile_kwargs",
    "CUDA_SIMULATION",
    "CUDACache",
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
