import pytest
from cubie.memory.cupyemm import _numba_stream_ptr
from cubie.memory.cupyemm import current_cupy_stream, CuPyAsyncNumbaManager
from cubie.memory.cupyemm import CupySyncNumbaManager
from numba.cuda.cudadrv.driver import NumbaCUDAMemoryManager
from numba import cuda
import numpy as np
import cupy as cp

# @pytest.fixture(scope="function")
# def asyncManager():
#     return CuPyAsyncNumbaManager()
#
# def syncmanager():
#     return CupySyncNumbaManager()

@pytest.fixture(scope="module")
def stream1():
    return cuda.stream()

@pytest.fixture(scope="module")
def stream2():
    return cuda.stream()

def test_numba_stream_ptr(stream1):
    try:
        expected_ptr = int(stream1.handle.value)
    except:
        expected_ptr = int(stream1.handle)
    assert _numba_stream_ptr(stream1) == expected_ptr

@pytest.fixture(scope="module")
def cp_stream_nocheck():
    class monkeypatch_cp_stream(current_cupy_stream):
        """current_cupy_stream without check for a cupy memory manager"""
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._mgr_is_cupy = True
    return monkeypatch_cp_stream

def test_cupy_stream_wrapper(stream1, stream2, cp_stream_nocheck):

    with cp_stream_nocheck(stream1) as cupy_stream:
        assert isinstance(cupy_stream._cupy_ext_stream, cp.cuda.ExternalStream)
        assert cupy_stream._cupy_ext_stream.ptr == _numba_stream_ptr(stream1)
        assert cp.cuda.get_current_stream().ptr == _numba_stream_ptr(stream1)

    with cp_stream_nocheck(stream2) as cupy_stream:
        assert isinstance(cupy_stream._cupy_ext_stream, cp.cuda.ExternalStream)
        assert cupy_stream._cupy_ext_stream.ptr == _numba_stream_ptr(stream2)
        assert cp.cuda.get_current_stream().ptr == _numba_stream_ptr(stream2)

    # Check that the default current stream is untouched
    assert cp.cuda.get_current_stream().ptr != _numba_stream_ptr(stream1)
    assert cp.cuda.get_current_stream().ptr != _numba_stream_ptr(stream2)

def test_cupy_wrapper_mgr_check(stream1, stream2):
    cuda.set_memory_manager(CuPyAsyncNumbaManager)
    cuda.close()
    with current_cupy_stream(stream1) as cupy_stream:
        assert cupy_stream._mgr_is_cupy is True, "Async manager not detected"

    cuda.set_memory_manager(CupySyncNumbaManager)
    cuda.close()
    with current_cupy_stream(stream2) as cupy_stream:
        assert cupy_stream._mgr_is_cupy is True, "Sync manager not detected"

    cuda.set_memory_manager(NumbaCUDAMemoryManager)
    cuda.close()
    with current_cupy_stream(stream1) as cupy_stream:
        assert cupy_stream._mgr_is_cupy is False, "Default manager not detected"


def test_correct_memalloc():
    cuda.set_memory_manager(CuPyAsyncNumbaManager)
    cuda.close()
    mgr = cuda.current_context().memory_manager
    mgr._testing = True
    newstream1 = cuda.stream()

    with current_cupy_stream(newstream1):
        testarr = cuda.device_array((10,10,10), dtype=np.float32)
        cuda.synchronize()
        assert mgr._testout == "async"
    del testarr
    cuda.synchronize()

    cuda.set_memory_manager(CupySyncNumbaManager)
    cuda.close()
    mgr = cuda.current_context().memory_manager
    mgr._testing = True
    newstream2 = cuda.stream()

    with current_cupy_stream(newstream2):
        testarr = cuda.device_array((10,10,10), dtype=np.float32)
        cuda.synchronize()
        assert mgr._testout == "sync"
    del testarr
    cuda.synchronize()

    cuda.set_memory_manager(NumbaCUDAMemoryManager)
    cuda.close()
    mgr = cuda.current_context().memory_manager
    newstream3 = cuda.stream()
    with current_cupy_stream(newstream3):
        testarr = cuda.device_array((10,10,10), dtype=np.float32)
        cuda.synchronize()
        with pytest.raises(AttributeError):
            test = mgr._testout
    del testarr
    cuda.synchronize()