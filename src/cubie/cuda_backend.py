"""Resolve which CUDA frontend package CuBIE compiles against.

CuBIE supports two mutually exclusive CUDA frontends: ``numba-cuda``
(the default NVIDIA Numba target) and ``numba-cuda-mlir`` (the
MLIR-based compiler). Exactly one must be installed; the backend is
resolved once at import time from the installed packages, with the
``CUBIE_CUDA_BACKEND`` environment variable as an explicit override
(values ``"numba-cuda"`` or ``"mlir"``).

Every CUDA-facing symbol (the ``cuda`` module object, scalar types,
``from_dtype``, driver internals, cache base classes) is re-exported
by :mod:`cubie.cuda_simsafe`, which consumes :data:`CUDA_BACKEND`
from here. Library code never imports a frontend package directly.

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

NUMBA_CUDA_BACKEND = "numba-cuda"
MLIR_BACKEND = "mlir"
_VALID_BACKENDS = (NUMBA_CUDA_BACKEND, MLIR_BACKEND)

_INSTALL_HINT = (
    "Install exactly one CUDA frontend: 'pip install cubie[cuda12]' "
    "or 'cubie[cuda13]' for numba-cuda, 'pip install "
    "cubie[mlir-cuda12]' or 'cubie[mlir-cuda13]' for numba-cuda-mlir "
    "(the bare 'cuda'/'mlir' extras skip the toolkit wheels when a "
    "system CUDA install is present)."
)


def _resolve_backend() -> str:
    """Return the active backend name from installs and environment.

    Returns
    -------
    str
        ``"numba-cuda"`` or ``"mlir"``.

    Raises
    ------
    ImportError
        If no frontend is installed, if both are installed without an
        explicit ``CUBIE_CUDA_BACKEND`` choice, or if the requested
        backend's package is missing.
    ValueError
        If ``CUBIE_CUDA_BACKEND`` is set to an unrecognised value.
    """
    mlir_installed = find_spec("numba_cuda_mlir") is not None
    numba_cuda_installed = find_spec("numba_cuda") is not None

    requested = os.environ.get("CUBIE_CUDA_BACKEND")
    if requested is not None:
        requested = requested.strip().lower()
        if requested not in _VALID_BACKENDS:
            raise ValueError(
                f"CUBIE_CUDA_BACKEND={requested!r} is not recognised; "
                f"valid values are {_VALID_BACKENDS}."
            )
        available = (
            mlir_installed
            if requested == MLIR_BACKEND
            else numba_cuda_installed
        )
        if not available:
            raise ImportError(
                f"CUBIE_CUDA_BACKEND={requested!r} is set but the "
                f"matching frontend package is not installed. "
                + _INSTALL_HINT
            )
        return requested

    if mlir_installed and numba_cuda_installed:
        raise ImportError(
            "Both numba-cuda and numba-cuda-mlir are installed. The "
            "CUDA frontends are mutually exclusive: uninstall one, or "
            "set CUBIE_CUDA_BACKEND to 'numba-cuda' or 'mlir' to pick "
            "one explicitly."
        )
    if mlir_installed:
        return MLIR_BACKEND
    if numba_cuda_installed:
        return NUMBA_CUDA_BACKEND
    raise ImportError(
        "No CUDA frontend is installed. " + _INSTALL_HINT
    )


CUDA_BACKEND: str = _resolve_backend()
IS_MLIR: bool = CUDA_BACKEND == MLIR_BACKEND

__all__ = [
    "CUDA_BACKEND",
    "IS_MLIR",
    "MLIR_BACKEND",
    "NUMBA_CUDA_BACKEND",
]
