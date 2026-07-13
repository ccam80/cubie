"""Single source of truth for cubie's on-disk cache root directory.

Three disk cache layers persist artefacts between sessions: generated
CUDA source modules (:class:`~cubie.odesystems.symbolic.odefile.ODEFile`),
pickled CellML parse results
(:class:`~cubie.odesystems.symbolic.parsing.cellml_cache.CellMLCache`),
and compiled kernels (:mod:`cubie.cubie_cache`). All three resolve
their base directory through :func:`get_cache_root`, so redirecting
the root (for example in tests) relocates every layer together.

Published Functions
-------------------
:func:`get_cache_root`
    Return the active cache root directory.
:func:`set_cache_root`
    Override the cache root process-wide, or restore the default.
"""

from os import getcwd
from pathlib import Path
from typing import Optional, Union

_cache_root_override: Optional[Path] = None


def get_cache_root() -> Path:
    """Return the root directory for cubie's on-disk caches.

    Returns
    -------
    Path
        The override set through :func:`set_cache_root` when one is
        active, otherwise ``<current working directory>/generated``,
        evaluated at call time.
    """
    if _cache_root_override is not None:
        return _cache_root_override
    return Path(getcwd()) / "generated"


def set_cache_root(path: Optional[Union[str, Path]]) -> None:
    """Override the cache root for every disk cache layer.

    Parameters
    ----------
    path
        New root directory for generated source, CellML parse, and
        compiled-kernel caches. Pass ``None`` to restore the default
        ``<cwd>/generated`` behaviour.
    """
    global _cache_root_override
    _cache_root_override = None if path is None else Path(path)
