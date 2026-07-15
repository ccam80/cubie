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

Published Classes
-----------------
:class:`JITFlags`
    Managed ``cuda.jit`` compile options stored on every factory's
    compile settings and rendered to decorator kwargs by
    :func:`get_jit_kwargs`.

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
    Uses the ``Stream`` stand-in, ``current_mem_info``, and the
    ``cupy``/``cupyx`` imports exported here. This module owns the
    single conditional import of ``cupy``/``cupyx``: both are
    imported eagerly on a real GPU (CuPy is CuBIE's device
    allocation provider, so it is a hard requirement there) and are
    ``None`` under the CUDA simulator, which never touches device
    memory. Consumers import them from here rather than importing
    CuPy directly.
"""

from __future__ import annotations

from ctypes import c_void_p
import os
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Tuple, Union

from attrs import Factory, define, field
from attrs import validators as attrs_validators
from numba import cuda
from numba import from_dtype as numba_from_dtype
from numpy import dtype

from cubie._env import lineinfo_default


CUDA_SIMULATION: bool = os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"


@define
class JITFlags:
    """Per-factory ``cuda.jit`` compile flags.

    Every managed jit option travels the same path: stored on the
    factory's compile settings (hashed into the config, so a change
    triggers a rebuild), then rendered to decorator keyword arguments
    by :func:`get_jit_kwargs`. New jit options are added here as new
    fields.

    Attributes
    ----------
    lineinfo
        Compile with source-line correlation data. Defaults to the
        ``CUBIE_LINEINFO`` environment variable.
    nsz
        Treat signed zero as insignificant in floating-point ops.
    contract
        Allow floating-point contraction (fused multiply-add).
    arcp
        Allow reciprocal approximation of division.
    afn
        Allow approximate transcendental functions (``LG2``/``EX2``
        hardware paths for ``log``/``exp``/``pow``).
    lto
        Enable link-time optimisation across device functions.
    """

    lineinfo: bool = field(
        default=Factory(lineinfo_default),
        validator=attrs_validators.instance_of(bool),
    )
    nsz: bool = field(
        default=True, validator=attrs_validators.instance_of(bool)
    )
    contract: bool = field(
        default=True, validator=attrs_validators.instance_of(bool)
    )
    arcp: bool = field(
        default=True, validator=attrs_validators.instance_of(bool)
    )
    afn: bool = field(
        default=True, validator=attrs_validators.instance_of(bool)
    )
    lto: bool = field(
        default=True, validator=attrs_validators.instance_of(bool)
    )

    @property
    def fastmath(self) -> set:
        """Return the set of enabled LLVM fast-math flag names."""
        enabled = {
            "nsz": self.nsz,
            "contract": self.contract,
            "arcp": self.arcp,
            "afn": self.afn,
        }
        return {name for name, on in enabled.items() if on}

    def update(self, updates_dict=None, **kwargs):
        """Update flag fields, following the config-update contract.

        Parameters
        ----------
        updates_dict
            Mapping of flag names to new boolean values. Unknown keys
            are ignored so composite configs can broadcast one updates
            dict to every nested attrs class.
        **kwargs
            Additional flag updates.

        Returns
        -------
        tuple[set[str], set[str]]
            Names of recognised settings and names of changed settings.
        """
        if updates_dict is None:
            updates_dict = {}
        updates_dict = {**updates_dict, **kwargs}
        recognized = set()
        changed = set()
        flag_names = {
            "lineinfo",
            "nsz",
            "contract",
            "arcp",
            "afn",
            "lto",
        }
        for key, value in updates_dict.items():
            if key not in flag_names:
                continue
            recognized.add(key)
            if getattr(self, key) != value:
                setattr(self, key, bool(value))
                changed.add(key)
        return recognized, changed


# Base compile kwargs for cuda.jit decorators, from the default flag
# set. Applies to device functions decorated at import time, which
# never see a factory config; factory builds render their config's
# JITFlags through get_jit_kwargs instead. GPU-only options are
# omitted under the CUDA simulator.
compile_kwargs: Mapping[str, Any] = MappingProxyType(
    {}
    if CUDA_SIMULATION
    else {
        "fastmath": JITFlags().fastmath,
        "lineinfo": lineinfo_default(),
        "lto": JITFlags().lto,
    }
)


def get_jit_kwargs(
    jit_flags: Optional[Union["JITFlags", bool]] = None,
) -> dict[str, Any]:
    """Return per-build ``cuda.jit`` keyword arguments.

    Parameters
    ----------
    jit_flags
        Flags for the build. A :class:`JITFlags` instance renders all
        of its fields; a bare boolean is accepted as the ``lineinfo``
        value with default fast-math flags (the form generated system
        modules use); ``None`` uses the default flag set.

    Returns
    -------
    dict
        ``{"fastmath": set, "lineinfo": bool, "lto": bool}``
        rendered from the flags. Under the CUDA simulator every
        GPU-only option is omitted and an empty dict is returned,
        regardless of the flags passed.
    """
    if CUDA_SIMULATION:
        return {}
    if jit_flags is None:
        jit_flags = JITFlags()
    elif isinstance(jit_flags, bool):
        jit_flags = JITFlags(lineinfo=jit_flags)
    return {
        "fastmath": jit_flags.fastmath,
        "lineinfo": jit_flags.lineinfo,
        "lto": jit_flags.lto,
    }


class FakeStream:  # pragma: no cover - placeholder
    """Placeholder CUDA stream."""

    handle = c_void_p(0)


class FakeMemoryInfo:  # pragma: no cover - placeholder
    """Container for fake memory statistics."""

    free = 1024**3
    total = 8 * 1024**3


if CUDA_SIMULATION:  # pragma: no cover - simulated
    from numba.cuda.simulator.cudadrv.devicearray import FakeCUDAArray
    from cubie.vendored.numba_cuda_cache import CUDACache

    # The simulator never touches real device memory, so CuPy is not
    # required; code paths guarded by CUDA_SIMULATION never use these.
    cupy = None
    cupyx = None

    Stream = FakeStream
    DeviceNDArrayBase = FakeCUDAArray
    DeviceNDArray = FakeCUDAArray
    MappedNDArray = FakeCUDAArray

    def current_mem_info() -> Tuple[int, int]:
        """Return fake free and total memory values."""

        fakemem = FakeMemoryInfo()
        return fakemem.free, fakemem.total

else:  # pragma: no cover - exercised in GPU environments
    try:
        import cupy
        import cupyx
    except ImportError as e:
        raise ImportError(
            "CuPy is required for CuBIE's device memory allocations "
            "on a real GPU. Install it via the cuda12/cuda13 extra "
            "(pip install cubie[cuda12]) or pip install cupy-cuda12x "
            "directly (assuming CUDA toolkit 12.x)."
        ) from e

    from numba.cuda import (  # type: ignore[attr-defined]
        is_cuda_array as _is_cuda_array,
    )
    from numba.cuda.cudadrv.driver import (  # type: ignore[attr-defined]
        Stream,
    )
    from numba.cuda.cudadrv.devicearray import (  # type: ignore[attr-defined]
        DeviceNDArrayBase,
        DeviceNDArray,
        MappedNDArray,
    )
    # Linter can't find cuda.dispatcher.
    from numba.cuda.dispatcher import CUDACache  # noqa: F401

    def current_mem_info() -> Tuple[int, int]:
        """Return free and total memory from the active CUDA context."""

        return cuda.current_context().get_memory_info()


def is_cuda_array(value: Any) -> bool:
    """Check whether ``value`` should be treated as a CUDA array."""

    if CUDA_SIMULATION:  # pragma: no cover - simulated
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
    return dt  # pragma: no cover - simulated


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

    # no cover: start
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

    # no cover: end

else:  # pragma: no cover - relies on GPU runtime

    # no cover: start
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

    # no cover: end


def is_cudasim_enabled() -> bool:
    """Return ``True`` when running under the CUDA simulator."""

    return CUDA_SIMULATION


def compute_capability_code() -> Optional[str]:
    """Return the current device's compute capability as ``"M.m"``.

    Returns
    -------
    str or None
        Architecture code such as ``"8.9"``, or ``None`` under
        CUDASIM, where no physical architecture exists.
    """
    if CUDA_SIMULATION:  # pragma: no cover - simulated
        return None
    major, minor = cuda.get_current_device().compute_capability
    return f"{major}.{minor}"


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
    if CUDA_SIMULATION:  # pragma: no cover - simulated
        return 49152
    return int(
        cuda.get_current_device().MAX_SHARED_MEMORY_PER_BLOCK
    )


__all__ = [
    "activemask",
    "all_sync",
    "compile_kwargs",
    "compute_capability_code",
    "get_jit_kwargs",
    "JITFlags",
    "CUDA_SIMULATION",
    "CUDACache",
    "cupy",
    "cupyx",
    "current_mem_info",
    "DeviceNDArray",
    "DeviceNDArrayBase",
    "FakeMemoryInfo",
    "FakeStream",
    "from_dtype",
    "is_cuda_array",
    "is_cudasim_enabled",
    "max_shared_memory_per_block",
    "is_devfunc",
    "MappedNDArray",
    "selp",
    "Stream",
    "stwt",
    "syncwarp",
]
