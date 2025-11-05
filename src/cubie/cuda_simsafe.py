"""Simulation-safe CUDA helpers and stand-ins.

This module centralises compatibility utilities for environments running with
``NUMBA_ENABLE_CUDASIM=1``.  It exposes a consistent surface so callers can
import CUDA-facing helpers without branching on simulator state.
"""
from __future__ import annotations

from contextlib import contextmanager
from ctypes import c_void_p
import os
from typing import Any, Callable, Tuple

import numba
from numba import cuda
import numpy as np


CUDA_SIMULATION: bool = os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"


class FakeBaseCUDAMemoryManager: # pragma: no cover - placeholder
    """Minimal stub of a CUDA memory manager."""

    def __init__(self, context: Any | None = None):
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
        finalizer: Any | None = None,
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
        MappedNDArray,
    )
    from numba.cuda.cudadrv.driver import GetIpcHandleMixin  # type: ignore[attr-defined]

    def current_mem_info() -> Tuple[int, int]:
        """Return free and total memory from the active CUDA context."""

        return cuda.current_context().get_memory_info()


def is_cuda_array(value: Any) -> bool:
    """Check whether ``value`` should be treated as a CUDA array."""

    if CUDA_SIMULATION:
        return hasattr(value, "shape")
    return _is_cuda_array(value)


def from_dtype(dtype: np.dtype):
    """Return a CUDA-ready dtype or a simulator-safe placeholder."""

    if not CUDA_SIMULATION:
        return numba.from_dtype(dtype)
    return dtype


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
    @cuda.jit(device=True, inline=True)
    def selp(pred, true_value, false_value):
        """
        Select value based on predicate (simulator version).

        Parameters
        ----------
        pred : bool
            Boolean predicate for selection.
        true_value : any
            Value returned when predicate is True.
        false_value : any
            Value returned when predicate is False.

        Returns
        -------
        any
            ``true_value`` if ``pred`` is True, otherwise ``false_value``.

        Notes
        -----
        This is a CUDA device function called from kernels only.
        Simulator version provides simple ternary operator semantics.
        """
        return true_value if pred else false_value

    @cuda.jit(device=True, inline=True)
    def activemask():
        """
        Return bitmask of active threads in the warp (simulator version).

        Returns
        -------
        int
            Bitmask with bits set for active threads. Simulator returns
            0xFFFFFFFF (all 32 threads active).

        Notes
        -----
        This is a CUDA device function called from kernels only.
        Real GPU version uses PTX instruction; simulator returns fixed mask.
        """
        return 0xFFFFFFFF

    @cuda.jit(device=True, inline=True)
    def all_sync(mask, predicate):
        """
        Check if predicate is true for all threads in mask (simulator version).

        Parameters
        ----------
        mask : int
            Bitmask specifying threads to include in the vote.
        predicate : bool
            Condition to evaluate for each thread.

        Returns
        -------
        bool
            ``True`` if predicate is true for all threads in mask.
            Simulator returns predicate value directly.

        Notes
        -----
        This is a CUDA device function called from kernels only.
        Real GPU version synchronizes warp and performs collective vote.
        Simulator skips synchronization and vote, returning predicate unchanged.
        """
        return predicate

else:  # pragma: no cover - relies on GPU runtime
    @cuda.jit(device=True, inline=True)
    def selp(pred, true_value, false_value):
        """
        Select value based on predicate (GPU version).

        Parameters
        ----------
        pred : bool
            Boolean predicate for selection.
        true_value : any
            Value returned when predicate is True.
        false_value : any
            Value returned when predicate is False.

        Returns
        -------
        any
            ``true_value`` if ``pred`` is True, otherwise ``false_value``.

        Notes
        -----
        This is a CUDA device function called from kernels only.
        Uses PTX ``selp`` instruction for efficient predicated selection.
        """
        return cuda.selp(pred, true_value, false_value)

    @cuda.jit(device=True, inline=True)
    def activemask():
        """
        Return bitmask of active threads in the warp (GPU version).

        Returns
        -------
        int
            Bitmask with bits set for threads currently active in the warp.
            Bit i corresponds to thread i in the warp (0-31).

        Notes
        -----
        This is a CUDA device function called from kernels only.
        Uses PTX ``activemask.b32`` instruction to query warp state.
        Essential for correct warp-synchronous programming and divergence
        avoidance in conditional blocks.
        """
        return cuda.activemask()

    @cuda.jit(device=True, inline=True)
    def all_sync(mask, predicate):
        """
        Check if predicate is true for all threads in mask (GPU version).

        Parameters
        ----------
        mask : int
            Bitmask specifying which threads participate in the vote.
        predicate : bool
            Condition to evaluate for each thread.

        Returns
        -------
        bool
            ``True`` if and only if predicate evaluates to true for all
            threads specified in mask. ``False`` otherwise.

        Notes
        -----
        This is a CUDA device function called from kernels only.
        Uses PTX ``vote.all.sync`` instruction to perform a warp-wide
        collective vote. All threads in mask are synchronized before
        the vote. Critical for ensuring uniform control flow decisions
        within a warp to avoid divergence penalties.

        Used in FSAL caching decisions (issue #149) to ensure all threads
        in a warp agree on whether to reuse cached derivatives.
        """
        return cuda.all_sync(mask, predicate)


def is_cudasim_enabled() -> bool:
    """Return ``True`` when running under the CUDA simulator."""

    return CUDA_SIMULATION


__all__ = [
    "CUDA_SIMULATION",
    "activemask",
    "all_sync",
    "BaseCUDAMemoryManager",
    "DeviceNDArrayBase",
    "FakeBaseCUDAMemoryManager",
    "FakeGetIpcHandleMixin",
    "FakeHostOnlyCUDAManager",
    "FakeMemoryInfo",
    "FakeMemoryPointer",
    "FakeNumbaCUDAMemoryManager",
    "FakeStream",
    "GetIpcHandleMixin",
    "HostOnlyCUDAMemoryManager",
    "MappedNDArray",
    "MemoryInfo",
    "MemoryPointer",
    "NumbaCUDAMemoryManager",
    "Stream",
    "current_mem_info",
    "from_dtype",
    "is_devfunc",
    "is_cuda_array",
    "is_cudasim_enabled",
    "set_cuda_memory_manager",
    "selp"
]
