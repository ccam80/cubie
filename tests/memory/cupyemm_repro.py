"""Reproducer from https://github.com/numba/numba/issues/5130 to reproduce
async malloc instead. Timing results are hard to interpret. The async malloc
is using the correct API calls (cudaMallocAsync) and the profile looks a
little different in this test (shorter duration malloc, and for some reason
shorter memcopies."""

from numba import cuda, NumbaPerformanceWarning
import numpy as np
import time
from numba.cuda.cudadrv.driver import NumbaCUDAMemoryManager
from cubie.memory import (
    current_cupy_stream,
    CuPyAsyncNumbaManager,
    CuPySyncNumbaManager,
)
import cupy as cp
import logging
from warnings import filterwarnings

# logging.basicConfig(level=logging.DEBUG)

filterwarnings("ignore", category=NumbaPerformanceWarning)
filterwarnings("ignore", category=FutureWarning)


def main(mgr):
    dev = cuda.current_context().device
    # print('CUDA device [%s]' % dev.name)
    print("Memory Manager [%s]" % mgr.__name__)

    # set async memory manager, reset context
    cuda.set_memory_manager(mgr)
    cuda.close()

    # define kernel after context switch to avoid RESOURCE_ERROR
    @cuda.jit
    def increment_kernel(g_data, inc_value, factor):
        i = cuda.grid(1)
        # Loop to make things take longer to run
        for j in range(factor):
            g_data[i] = g_data[i] + inc_value[i]

    # find the default stream (should perform synchronous mallocs and copys)
    if mgr is NumbaCUDAMemoryManager:
        default_stream = cuda.default_stream()
    else:
        cp_default_stream_ptr = cp.cuda.get_current_stream().ptr
        default_stream = cuda.external_stream(cp_default_stream_ptr)

    # Create streams & steam events
    stream1 = cuda.stream()
    stream2 = cuda.stream()
    stream3 = cuda.stream()

    start_dev = cuda.event(timing=True)
    start1 = cuda.event(timing=True)
    start2 = cuda.event(timing=True)
    start3 = cuda.event(timing=True)
    stop1 = cuda.event(timing=True)
    stop2 = cuda.event(timing=True)
    stop3 = cuda.event(timing=True)
    stop_dev = cuda.event(timing=True)

    # Kernel parameters and inputs
    nthreads = 512
    n_kernel = 128 * nthreads
    nblocks = n_kernel // nthreads
    value = 26
    factor = 1500000
    inc_value = cuda.pinned_array(n_kernel, dtype=np.int32) + value
    d_inc_value = cuda.to_device(inc_value, stream=default_stream)

    # One-time-use arrays for compilation
    comp_array = np.zeros(n_kernel, dtype=np.int32)
    d_comp = cuda.to_device(comp_array, stream=default_stream)

    # Get compilation out of the way
    increment_kernel[nblocks, nthreads, default_stream](
        d_comp, d_inc_value, factor
    )
    increment_kernel[nblocks, nthreads, default_stream](
        d_comp, d_inc_value, factor
    )

    # Allocate host memory
    n = 2 * 1024 * 1024 * 256  # 2GB Each
    a = cuda.pinned_array(n, dtype=np.int32)
    b = cuda.pinned_array(n, dtype=np.int32)
    c = cuda.pinned_array(n, dtype=np.int32)
    cuda.synchronize()

    # Execute allocation, copy, and kernel execution in each stream
    start_time = time.time()
    start_dev.record(stream=default_stream)
    with current_cupy_stream(stream1):
        start1.record(stream=stream1)
        d_inc1 = cuda.to_device(inc_value, stream1)
        d_a = cuda.device_array_like(a, stream1)
    with current_cupy_stream(stream2):
        start2.record(stream=stream2)
        d_inc2 = cuda.to_device(inc_value, stream2)
        d_b = cuda.to_device(b, stream2)
    with current_cupy_stream(stream3):
        start3.record(stream=stream3)
        d_inc3 = cuda.to_device(inc_value, stream3)
        d_c = cuda.to_device(c, stream3)
    increment_kernel[nblocks, nthreads, stream1](d_a, d_inc1, factor)
    increment_kernel[nblocks, nthreads, stream2](d_b, d_inc2, factor)
    increment_kernel[nblocks, nthreads, stream3](d_c, d_inc3, factor)
    stop1.record(stream=stream1)
    stop2.record(stream=stream2)
    stop3.record(stream=stream3)
    stop_dev.record(stream=default_stream)

    stop_time = time.time()

    stream1.synchronize()
    stream2.synchronize()
    stream3.synchronize()
    default_stream.synchronize()
    allsync_time = time.time()

    stream1_time = cuda.event_elapsed_time(start1, stop1)
    stream2_time = cuda.event_elapsed_time(start2, stop2)
    stream3_time = cuda.event_elapsed_time(start3, stop3)
    gpu_time = cuda.event_elapsed_time(start_dev, stop_dev)

    print("time spent by stream1: %.2f" % stream1_time)
    print("time spent by stream2: %.2f" % stream2_time)
    print("time spent by stream3: %.2f" % stream3_time)
    print("total time spent by GPU: %.2f" % gpu_time)
    print(
        "time spent by CPU in CUDA calls: %.2f"
        % ((stop_time - start_time) * 1000)
    )
    print(
        "CPU time spent from async start to all-synced: %.2f"
        % ((allsync_time - start_time) * 1000)
    )
    print("==")


if __name__ == "__main__":
    for mgr in [
        NumbaCUDAMemoryManager,
        CuPyAsyncNumbaManager,
        CuPySyncNumbaManager,
        NumbaCUDAMemoryManager,
    ]:
        main(mgr)
