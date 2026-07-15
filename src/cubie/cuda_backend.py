"""Resolve which CUDA backend package CuBIE compiles against.

CuBIE supports two CUDA backends: ``numba-cuda`` (the default NVIDIA
Numba target) and ``numba-cuda-mlir`` (the MLIR-based compiler). The
backend is resolved once at import time. An explicit
``CUBIE_CUDA_BACKEND`` environment value (``"numba-cuda"`` or
``"mlir"``, read through :mod:`cubie._env`) always wins; otherwise
whichever backend is installed is used. When both are installed and
no explicit choice is made, numba-cuda is selected and a warning
explains how to pick the MLIR backend. Under the CUDA simulator
(``NUMBA_ENABLE_CUDASIM=1``) numba-cuda is preferred when installed,
because numba-cuda-mlir has no simulator.

Every CUDA-facing symbol (the ``cuda`` module object, scalar types,
``from_dtype``, driver internals, cache base classes) is re-exported
by :mod:`cubie.cuda_simsafe`, which consumes :data:`CUDA_BACKEND`
from here. Library code never imports a backend package directly.

Published Constants
-------------------
:data:`CUDA_BACKEND`
    The resolved backend name, ``"numba-cuda"`` or ``"mlir"``.
:data:`IS_MLIR`
    ``True`` when the MLIR backend is active.

See Also
--------
:mod:`cubie.cuda_simsafe`
    The import hub that maps backend-specific modules onto a single
    shared surface.
:mod:`cubie._env`
    Registry of ``CUBIE_*`` environment variables, including
    ``CUBIE_CUDA_BACKEND``.
"""

import os
from importlib.util import find_spec
from warnings import warn

from cubie._env import cuda_backend_requested

NUMBA_CUDA_BACKEND = "numba-cuda"
MLIR_BACKEND = "mlir"

_INSTALL_HINT = (
    "Install a CUDA backend: 'pip install cubie[cuda12]' or "
    "'cubie[cuda13]' for numba-cuda, 'pip install "
    "cubie[mlir-cuda12]' or 'cubie[mlir-cuda13]' for numba-cuda-mlir "
    "(the bare 'cuda'/'mlir' extras skip the toolkit wheels when a "
    "system CUDA install is present)."
)


def _resolve_backend() -> str:
    """Return the active backend name from environment and installs.

    An explicit ``CUBIE_CUDA_BACKEND`` value wins and its package
    must be installed. Otherwise the installed backend is used; under
    ``NUMBA_ENABLE_CUDASIM=1`` numba-cuda is preferred (the MLIR
    backend has no simulator), and when both backends are installed
    numba-cuda is selected with a warning.

    Returns
    -------
    str
        ``"numba-cuda"`` or ``"mlir"``.

    Raises
    ------
    ImportError
        If no backend is installed, or if the requested backend's
        package is missing.
    ValueError
        If ``CUBIE_CUDA_BACKEND`` is set to an unrecognised value.
    """
    mlir_installed = find_spec("numba_cuda_mlir") is not None
    numba_cuda_installed = find_spec("numba_cuda") is not None
    cudasim = os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"

    requested = cuda_backend_requested()
    if requested is not None:
        available = (
            mlir_installed
            if requested == MLIR_BACKEND
            else numba_cuda_installed
        )
        if not available:
            raise ImportError(
                f"CUBIE_CUDA_BACKEND={requested!r} is set but the "
                f"matching backend package is not installed. "
                + _INSTALL_HINT
            )
        return requested

    if cudasim and numba_cuda_installed:
        return NUMBA_CUDA_BACKEND
    if mlir_installed and numba_cuda_installed:
        warn(
            "Both numba-cuda and numba-cuda-mlir are installed; "
            "auto-selecting numba-cuda. Set CUBIE_CUDA_BACKEND='mlir' "
            "to use the MLIR backend."
        )
        return NUMBA_CUDA_BACKEND
    if mlir_installed:
        return MLIR_BACKEND
    if numba_cuda_installed:
        return NUMBA_CUDA_BACKEND
    raise ImportError(
        "No CUDA backend is installed. " + _INSTALL_HINT
    )


CUDA_BACKEND: str = _resolve_backend()
IS_MLIR: bool = CUDA_BACKEND == MLIR_BACKEND

__all__ = [
    "CUDA_BACKEND",
    "IS_MLIR",
    "MLIR_BACKEND",
    "NUMBA_CUDA_BACKEND",
]
