"""Read process-wide ``CUBIE_*`` defaults.

Explicit arguments override environment values. Values are read when the
relevant object is created.

``CUBIE_LINEINFO`` enables line information. ``CUBIE_CACHE_DIR`` sets the
root for every disk cache. ``CUBIE_KERNEL_CACHE_DIR`` moves only compiled
kernels. ``CUBIE_MAX_CACHE_ENTRIES`` sets their per-system LRU limit; zero
disables eviction. ``CUBIE_CUDA_BACKEND`` selects ``numba-cuda`` or ``mlir``.
"""

import os
from typing import Optional

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"0", "false", "no", "off", ""})


def env_bool(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable.

    Parameters
    ----------
    name
        Environment variable name.
    default
        Value returned when the variable is unset.

    Returns
    -------
    bool
        Parsed value.

    Raises
    ------
    ValueError
        If the variable is set to an unrecognised value.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _TRUTHY:
        return True
    if value in _FALSY:
        return False
    raise ValueError(
        f"Environment variable {name}={raw!r} is not a recognised boolean; "
        f"use one of {sorted(_TRUTHY | _FALSY - {''})}."
    )


def lineinfo_default() -> bool:
    """Return the default ``lineinfo`` compile setting.

    Reads ``CUBIE_LINEINFO`` from the environment; explicit
    ``Solver(lineinfo=...)`` arguments override this value.
    """
    return env_bool("CUBIE_LINEINFO", False)


def cache_dir_default() -> Optional[str]:
    """Return the environment-supplied on-disk cache root, if any.

    Reads ``CUBIE_CACHE_DIR`` from the environment; an explicit
    :func:`cubie.cache_root.set_cache_root` call overrides this value.
    Empty and whitespace-only values are treated as unset.

    Returns
    -------
    Optional[str]
        The configured directory, or ``None`` when the variable is
        unset or blank.
    """
    raw = os.environ.get("CUBIE_CACHE_DIR")
    if raw is None or not raw.strip():
        return None
    return raw


def kernel_cache_dir_default() -> Optional[str]:
    """Return ``CUBIE_KERNEL_CACHE_DIR``, or ``None`` when unset."""
    raw = os.environ.get("CUBIE_KERNEL_CACHE_DIR")
    if raw is None or not raw.strip():
        return None
    return raw


def max_cache_entries_default() -> int:
    """Return the non-negative kernel-cache limit; zero disables eviction."""
    raw = os.environ.get("CUBIE_MAX_CACHE_ENTRIES")
    if raw is None or not raw.strip():
        return 10
    try:
        value = int(raw.strip())
    except ValueError:
        raise ValueError(
            f"CUBIE_MAX_CACHE_ENTRIES={raw!r} must be a non-negative integer."
        ) from None
    if value < 0:
        raise ValueError(
            f"CUBIE_MAX_CACHE_ENTRIES={raw!r} must be non-negative."
        )
    return value


def cuda_backend_requested() -> Optional[str]:
    """Return the explicitly requested CUDA backend, if any.

    Reads ``CUBIE_CUDA_BACKEND`` from the environment; empty and
    whitespace-only values are treated as unset.
    :mod:`cubie.cuda_backend` resolves the active backend from this
    value and the installed packages.

    Returns
    -------
    Optional[str]
        ``"numba-cuda"`` or ``"mlir"``, or ``None`` when unset.

    Raises
    ------
    ValueError
        If the variable is set to an unrecognised value.
    """
    raw = os.environ.get("CUBIE_CUDA_BACKEND")
    if raw is None or not raw.strip():
        return None
    value = raw.strip().lower()
    if value not in ("numba-cuda", "mlir"):
        raise ValueError(
            f"CUBIE_CUDA_BACKEND={raw!r} is not recognised; valid "
            "values are 'numba-cuda' and 'mlir'."
        )
    return value
