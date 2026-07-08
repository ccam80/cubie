import pytest

from cubie.memory.cupy_emm import _numba_stream_ptr
from cubie.memory.cupy_emm import current_cupy_stream
from numba import cuda
import cupy as cp


@pytest.fixture(scope="session")
def stream1():
    return cuda.stream()


@pytest.fixture(scope="session")
def stream2():
    return cuda.stream()


@pytest.mark.nocudasim
def test_numba_stream_ptr(stream1):
    try:
        expected_ptr = int(stream1.handle.value)
    except AttributeError:
        expected_ptr = int(stream1.handle)
    assert _numba_stream_ptr(stream1) == expected_ptr


@pytest.mark.nocudasim
@pytest.mark.cupy
def test_cupy_stream_wrapper(stream1, stream2):
    """Verify current_cupy_stream always forwards a Numba stream.

    CuPy is CuBIE's single device allocation provider, so the
    forwarding context manager is unconditional and no longer gated
    on detecting a CuPy-backed Numba External Memory Manager plugin.
    """
    with current_cupy_stream(stream1) as cupy_stream:
        assert isinstance(cupy_stream.cupy_ext_stream, cp.cuda.ExternalStream)
        assert cupy_stream.cupy_ext_stream.ptr == _numba_stream_ptr(stream1)
        assert cp.cuda.get_current_stream().ptr == _numba_stream_ptr(stream1)

    with current_cupy_stream(stream2) as cupy_stream:
        assert isinstance(cupy_stream.cupy_ext_stream, cp.cuda.ExternalStream)
        assert cupy_stream.cupy_ext_stream.ptr == _numba_stream_ptr(stream2)
        assert cp.cuda.get_current_stream().ptr == _numba_stream_ptr(stream2)

    # Check that the default current stream is untouched
    assert cp.cuda.get_current_stream().ptr != _numba_stream_ptr(stream1)
    assert cp.cuda.get_current_stream().ptr != _numba_stream_ptr(stream2)
