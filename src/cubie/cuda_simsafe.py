"""CUDA helper surface shared by device-facing modules.

This module centralises CUDA-facing helpers that were previously
switchable between the real GPU target and Numba's CUDA simulator.
numba-cuda-mlir has no simulator, so the helpers now always target
the real GPU; setting ``NUMBA_ENABLE_CUDASIM=1`` raises at import.

Published Functions
-------------------
:func:`from_dtype`
    Return a CUDA-ready dtype.
:func:`is_devfunc`
    Test whether a callable is a CUDA device function.
:func:`is_cuda_array`
    Check whether a value should be treated as a CUDA array.
:func:`is_cudasim_enabled`
    Always ``False``; retained for callers that branch on it.

Published Device Functions
--------------------------
``selp``, ``activemask``, ``all_sync``, ``any_sync``,
``syncwarp``, ``stwt``
    Wrappers around CUDA intrinsics.

Published Constants
-------------------
:data:`CUDA_SIMULATION`
    Always ``False``; retained for callers that branch on it.
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

import os
from typing import Any, Callable, Tuple

from numpy import dtype

if os.environ.get("NUMBA_ENABLE_CUDASIM") == "1":
    raise ImportError(
        "NUMBA_ENABLE_CUDASIM=1 is set, but numba-cuda-mlir has no "
        "CUDA simulator. Unset the variable and run on a real GPU."
    )

from numba_cuda_mlir import cuda
from numba_cuda_mlir.cuda import (
    HostOnlyCUDAMemoryManager,
    MemoryPointer,
    MemoryInfo,
    set_memory_manager as set_cuda_memory_manager,
    is_cuda_array as _is_cuda_array,
)
from numba_cuda_mlir.numba_cuda.cudadrv.driver import (
    BaseCUDAMemoryManager,
    GetIpcHandleMixin,
    NumbaCUDAMemoryManager,
    Stream,
)
from numba_cuda_mlir.numba_cuda.cudadrv.devicearray import (
    DeviceNDArrayBase,
    DeviceNDArray,
    MappedNDArray,
)
from numba_cuda_mlir.numba_cuda.np.numpy_support import (
    from_dtype as numba_from_dtype,
)


CUDA_SIMULATION: bool = False

# Compile kwargs for cuda.jit decorators.
# numba-cuda-mlir only accepts fastmath as a boolean, so the previous
# per-flag fastmath selection ({"nsz", "contract", "arcp"}) cannot be
# expressed; leave fastmath at its default (off) rather than enable
# the full set of approximations.
compile_kwargs: dict[str, bool] = {
    # "lineinfo": True,
}


def current_mem_info() -> Tuple[int, int]:
    """Return free and total memory from the active CUDA context."""

    return cuda.current_context().get_memory_info()


def is_cuda_array(value: Any) -> bool:
    """Check whether ``value`` should be treated as a CUDA array."""

    return _is_cuda_array(value)


def from_dtype(dt: dtype):
    """Return a CUDA-ready dtype.

    Parameters
    ----------
    dt
        NumPy dtype to adapt for use with CUDA.

    Returns
    -------
    dtype
        A Numba CUDA-compatible dtype.
    """

    return numba_from_dtype(dt)


def is_devfunc(func: Callable[..., Any]) -> bool:
    """Test whether ``func`` represents a CUDA device function.

    Parameters
    ----------
    func
        Callable object to inspect for CUDA device metadata.

    Returns
    -------
    bool
        ``True`` when ``func`` is tagged as a CUDA device function.
    """

    target_options = getattr(func, "targetoptions", None)
    if isinstance(target_options, dict):
        return bool(target_options.get("device", False))
    return False


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
    # Requires the memref pointer-offset shim in cubie._mlir_compat
    # (or a numba-cuda-mlir build carrying upstream #73): without it
    # the cache-hint store lowering drops the offset of array views.
    cuda.stwt(array, index, value)


def is_cudasim_enabled() -> bool:
    """Return ``False``; numba-cuda-mlir has no CUDA simulator."""

    return CUDA_SIMULATION


def max_shared_memory_per_block() -> int:
    """Return the device's dynamic shared-memory limit per block.

    Returns
    -------
    int
        Per-block shared-memory limit in bytes. numba-cuda does not
        set ``CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES``, so
        the default (non-opt-in) device limit applies to every
        launch. Under CUDASIM the ubiquitous 48 kiB default is
        returned.
    """
    if CUDA_SIMULATION:
        return 49152
    return int(
        cuda.get_current_device().MAX_SHARED_MEMORY_PER_BLOCK
    )


__all__ = [
    "activemask",
    "all_sync",
    "any_sync",
    "BaseCUDAMemoryManager",
    "compile_kwargs",
    "CUDA_SIMULATION",
    "current_mem_info",
    "DeviceNDArray",
    "DeviceNDArrayBase",
    "from_dtype",
    "GetIpcHandleMixin",
    "HostOnlyCUDAMemoryManager",
    "is_cuda_array",
    "is_cudasim_enabled",
    "max_shared_memory_per_block",
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
