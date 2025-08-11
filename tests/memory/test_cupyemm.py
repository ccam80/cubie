import pytest
try:
    from cubie.memory.cupyemm import _numba_stream_ptr
    from cubie.memory.cupyemm import current_cupy_stream, CuPyAsyncNumbaManager
    from cubie.memory.cupyemm import CuPySyncNumbaManager
    from numba.cuda.cudadrv.driver import NumbaCUDAMemoryManager
except ImportError:
    pytest.skip("CUDA Simulator doesn't have memory features",
                allow_module_level=True)

from numba import cuda
import numpy as np
import cupy as cp

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

@pytest.mark.cupy
@pytest.fixture(scope="module")
def cp_stream_nocheck():
    class monkeypatch_cp_stream(current_cupy_stream):
        """current_cupy_stream without check for a cupy memory manager"""
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._mgr_is_cupy = True
    return monkeypatch_cp_stream

@pytest.mark.cupy
def test_cupy_stream_wrapper(stream1, stream2, cp_stream_nocheck):

    with cp_stream_nocheck(stream1) as cupy_stream:
        assert isinstance(cupy_stream.cupy_ext_stream, cp.cuda.ExternalStream)
        assert cupy_stream.cupy_ext_stream.ptr == _numba_stream_ptr(stream1)
        assert cp.cuda.get_current_stream().ptr == _numba_stream_ptr(stream1)

    with cp_stream_nocheck(stream2) as cupy_stream:
        assert isinstance(cupy_stream.cupy_ext_stream, cp.cuda.ExternalStream)
        assert cupy_stream.cupy_ext_stream.ptr == _numba_stream_ptr(stream2)
        assert cp.cuda.get_current_stream().ptr == _numba_stream_ptr(stream2)

    # Check that the default current stream is untouched
    assert cp.cuda.get_current_stream().ptr != _numba_stream_ptr(stream1)
    assert cp.cuda.get_current_stream().ptr != _numba_stream_ptr(stream2)

@pytest.mark.cupy
def test_cupy_wrapper_mgr_check(stream1, stream2):
    cuda.set_memory_manager(CuPyAsyncNumbaManager)
    cuda.close()
    with current_cupy_stream(stream1) as cupy_stream:
        assert cupy_stream._mgr_is_cupy is True, "Async manager not detected"

    cuda.set_memory_manager(CuPySyncNumbaManager)
    cuda.close()
    with current_cupy_stream(stream2) as cupy_stream:
        assert cupy_stream._mgr_is_cupy is True, "Sync manager not detected"

    cuda.set_memory_manager(NumbaCUDAMemoryManager)
    cuda.close()
    with current_cupy_stream(stream1) as cupy_stream:
        assert cupy_stream._mgr_is_cupy is False, "Default manager not detected"

@pytest.mark.cupy
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

    cuda.set_memory_manager(CuPySyncNumbaManager)
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

@pytest.mark.cupy
@pytest.mark.parametrize("mgr", [NumbaCUDAMemoryManager,
                                 CuPySyncNumbaManager,
                                 CuPyAsyncNumbaManager])
def test_allocation(mgr):
    cuda.set_memory_manager(mgr)
    cuda.close()

    @cuda.jit()
    def test_kernel(arr):
        i = cuda.grid(1)
        arr[i] = i

    testarr = np.zeros((256), dtype=np.float32)
    d_testarr = cuda.device_array_like(testarr)
    d_testarr.copy_to_device(testarr)
    test_kernel[1, 256,0 ,0](d_testarr)
    d_testarr.copy_to_device(testarr)
    assert not np.array_equal(testarr, np.zeros((256*256), dtype=np.float32))

