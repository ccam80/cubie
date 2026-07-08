"""CuPy stream interop helpers for CuBIE's memory manager.

CuPy is CuBIE's single GPU allocation provider on a real device. This
module forwards Numba-generated CUDA streams into CuPy so that
CuPy-side allocations and host/device copies stay ordered on the same
stream as the Numba-launched integration kernel.

Published Classes
-----------------
:class:`current_cupy_stream`
    Context manager forwarding a Numba stream into CuPy APIs.

    >>> from numba import cuda
    >>> stream = cuda.stream()
    >>> with current_cupy_stream(stream):  # doctest: +SKIP
    ...     pass

See Also
--------
:class:`~cubie.memory.mem_manager.MemoryManager`
    Performs CuPy allocations and copies inside this context manager.
:mod:`~cubie.cuda_simsafe`
    Provides the ``Stream`` typing stand-in and CUDASIM shims.
"""

from types import TracebackType
from typing import Optional
import ctypes

from cubie.cuda_simsafe import Stream


def _numba_stream_ptr(
    nb_stream: Optional[Stream],
) -> Optional[int]:
    """
    Extract a ``CUstream`` pointer from a Numba stream wrapper.

    Parameters
    ----------
    nb_stream
        Numba CUDA stream whose ``CUstream`` pointer should be extracted. When
        ``None``, pointer extraction is skipped.

    Returns
    -------
    int or None
        Pointer value compatible with CuPy external streams, or ``None`` when
        extraction fails.

    Notes
    -----
    The function checks common attribute layouts across supported Numba
    versions to maintain compatibility.
    """
    if nb_stream is None:
        return None
    h = getattr(nb_stream, "handle", None)
    if h is None:
        return None
    # ctypes.c_void_p or int-like
    if isinstance(h, ctypes.c_void_p):
        return int(h.value) if h.value is not None else None
    try:
        return int(getattr(h, "value", h))
    except Exception:
        return None


class current_cupy_stream:
    """Context manager that forwards a Numba stream into CuPy APIs.

    Parameters
    ----------
    nb_stream
        Numba CUDA stream to expose to CuPy.

    Attributes
    ----------
    nb_stream
        The Numba stream being forwarded.
    cupy_ext_stream
        CuPy external stream wrapper around the Numba stream.

    Raises
    ------
    ImportError
        If CuPy is not installed. CuPy is CuBIE's only real-GPU
        allocation provider, so it must be installed to allocate,
        copy, or otherwise touch device memory.

    Notes
    -----
    Numba's default stream (handle ``0``) is left as CuPy's ambient
    current stream rather than wrapped, matching Numba's own default
    stream semantics.
    """

    def __init__(self, nb_stream: Stream) -> None:
        try:
            import cupy  # noqa: F401
        except ImportError:  # pragma: no cover
            raise ImportError(
                "CuPy is required for CuBIE's device memory allocations "
                "on a real GPU. Install it via the cuda12/cuda13 extra "
                "(pip install cubie[cuda12]) or pip install "
                "cupy-cuda12x directly (assuming CUDA toolkit 12.x)."
            )
        self.nb_stream = nb_stream
        self.cupy_ext_stream = None

    def __enter__(self) -> "current_cupy_stream":
        """
        Enter the context and set up a CuPy external stream.

        Returns
        -------
        current_cupy_stream
            The active context manager instance.
        """
        import cupy as cp

        ptr = _numba_stream_ptr(self.nb_stream)
        if ptr:
            self.cupy_ext_stream = cp.cuda.ExternalStream(ptr)
            self.cupy_ext_stream.__enter__()
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        """Exit the context and clean up the CuPy external stream.

        Parameters
        ----------
        exc_type
            Exception type if an exception occurred.
        exc
            Exception instance if an exception occurred.
        tb
            Traceback object if an exception occurred.
        """
        if self.cupy_ext_stream is not None:
            self.cupy_ext_stream.__exit__(exc_type, exc, tb)
            self.cupy_ext_stream = None
