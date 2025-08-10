"""Reproducer from https://github.com/numba/numba/issues/5130 to reproduce
async malloc instead. Timing results are hard to interpret. The async malloc
is using the correct API calls (cudaMallocAsync) and the profile looks a
little"""

from numba import cuda, NumbaPerformanceWarning
import numpy as np
import time
from numba.cuda.cudadrv.driver import NumbaCUDAMemoryManager
from cubie.memory import current_cupy_stream, CuPyAsyncNumbaManager, \
    CuPySyncNumbaManager
import cupy as cp
import logging

# logging.basicConfig(level=logging.DEBUG)
from warnings import filterwarnings

filterwarnings("ignore", category=NumbaPerformanceWarning)
filterwarnings("ignore", category=FutureWarning)



def main(mgr):

    dev = cuda.current_context().device
    # print('CUDA device [%s]' % dev.name)
    print('Memory Manager [%s]' % mgr.__name__)

    # set async memory manager, reset context
    cuda.set_memory_manager(mgr)
    cuda.close()
    # cuda.select_device(0)

    @cuda.jit
    def increment_kernel(g_data, inc_value, factor):
        i = cuda.grid(1)
        # Loop to make things take longer to run
        for j in range(factor):
            g_data[i] = g_data[i] + inc_value[i]


    # Create streams
    stream1 = cuda.stream()
    stream2 = cuda.stream()
    stream3 = cuda.stream()

    if mgr is NumbaCUDAMemoryManager:
        default_stream = cuda.default_stream()
    else:
        cp_default_stream_ptr = cp.cuda.get_current_stream().ptr
        default_stream = cuda.external_stream(cp_default_stream_ptr)

    nthreads = 512
    n_kernel = 32 * 512 # Few enough to not occupy all blocks and allow
    # concurrent execution
    nblocks = n_kernel // nthreads
    value = 26
    factor = 150000 # leads to a roughly equivalent execution time to a
    # 3GB malloc
    n_kernel = 16 * 1024 * 1024
    inc_value1 = cuda.pinned_array(n_kernel, dtype=np.int32) + value
    kernelarray1 = cuda.pinned_array(n_kernel, dtype=np.int32)
    d_kernelarray1 =  cuda.to_device(kernelarray1)
    d_inc_value1 = cuda.to_device(inc_value1)
    inc_value2 = cuda.pinned_array(n_kernel, dtype=np.int32) + value
    kernelarray2 = cuda.pinned_array(n_kernel, dtype=np.int32)
    d_kernelarray2 =  cuda.to_device(kernelarray2)
    d_inc_value2 = cuda.to_device(inc_value2)
    # wait in case context creation is delayed until a transfer
    mgr_instance = cuda.current_context().memory_manager

    #run kernel twice to make sure compilation is completed before the
    # timed portion
    increment_kernel[nblocks, nthreads, stream2](d_kernelarray1,
                                                d_inc_value1, factor)
    increment_kernel[nblocks, nthreads, stream3](d_kernelarray2,
                                                d_inc_value2, factor)
    cuda.synchronize()

    # Allocate host memory
    n = 2 * 1024 * 1024 * 256 # 3GB
    a = cuda.pinned_array(n, dtype=np.int32)

    # Create event handles
    start_dev = cuda.event(timing=True)
    start1 = cuda.event(timing=True)
    start2 = cuda.event(timing=True)
    start3 = cuda.event(timing=True)
    stop1 = cuda.event(timing=True)
    stop2 = cuda.event(timing=True)
    stop3 = cuda.event(timing=True)
    stop_dev = cuda.event(timing=True)


    # Stage 1: Asynchronously issue work
    cuda.synchronize()

    start_time = time.time()
    # print("==== START ASYNC ====")
    start_dev.record(stream=default_stream)
    with current_cupy_stream(stream1):
        start1.record(stream=stream1)
        d_a = cuda.device_array_like(a, stream1)
        del d_a
        stop1.record(stream=stream1)
    start2.record(stream=stream2)
    increment_kernel[nblocks, nthreads, stream2](d_kernelarray1,
                                                 d_inc_value1, factor)
    increment_kernel[nblocks, nthreads, stream2](d_kernelarray1,
                                                 d_inc_value1, factor)
    stop2.record(stream=stream2)
    start3.record(stream=stream3)
    increment_kernel[nblocks, nthreads, stream3](d_kernelarray2,
                                                 d_inc_value2, factor)
    increment_kernel[nblocks, nthreads, stream3](d_kernelarray2,
                                                 d_inc_value2, factor)
    stop3.record(stream=stream3)

    stop_dev.record(stream=default_stream)
    # free_all_streams_memory([stream1, stream2, stream3, default_stream],
    #                         mgr_instance)
    # print("==== STOP ASYNC ====")
    stop_time = time.time()
    stream1.synchronize()
    stream2.synchronize()
    stream3.synchronize()
    default_stream.synchronize()
    allsync_time = time.time()

    stream1_time = cuda.event_elapsed_time(start1, stop1)
    stream2_time = cuda.event_elapsed_time(start2, stop2)
    stream3_time = cuda.event_elapsed_time(start2, stop2)

    gpu_time = cuda.event_elapsed_time(start_dev, stop_dev)
    print("time spent allocating by stream1: %.2f" % stream1_time)
    print("time spent executing by stream2: %.2f" % stream2_time)
    print("time spent executing by stream3: %.2f" % stream3_time)
    print("time spent by CPU in CUDA calls: %.2f" % ((stop_time -
                                                      start_time) * 1000))
    print("time spent from async to all-synced: %.2f" % ((allsync_time -
                                                        start_time)*1000))

    print("time spent by GPU: %.2f" % gpu_time)
    print("==")
    print("==")


if __name__ == '__main__':
    for mgr in [CuPyAsyncNumbaManager, NumbaCUDAMemoryManager,
                CuPySyncNumbaManager, NumbaCUDAMemoryManager]:
        main(mgr)

        #Can't really make sense of times, but can confirm that the async
        # calls are being used successfully