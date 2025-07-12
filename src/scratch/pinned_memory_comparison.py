from numba import cuda, float32
from CuMC._utils import timing
import numpy as np

@cuda.jit()
def kerneltester(outarray):
    """
    A simple CUDA kernel that writes a value to the output array.
    This is used to test if the CUDA environment is set up correctly.
    """
    idx = cuda.grid(1)
    if idx < outarray.size:
        outarray[idx] = float32(idx)


threads_per_block = 1024
blocks_per_grid = 512000
size = threads_per_block * blocks_per_grid
devicearray = cuda.device_array((size,), dtype=np.float32)

@timing
def run_pinned():
    pinnedarray = cuda.pinned_array((size,), dtype=np.float32)
    kerneltester[blocks_per_grid, threads_per_block](pinnedarray)
    cuda.synchronize()
    # print(f"Pinned: {pinnedarray[10]}")
    return pinnedarray
    del pinnedarray


@timing
def run_mapped():
    mappedarray = cuda.mapped_array((size,), dtype=np.float32)
    kerneltester[blocks_per_grid, threads_per_block](mappedarray)
    cuda.synchronize()
    # print(f"Mapped: {mappedarray[10]}")
    del mappedarray


@timing
def run_device():
    devicearray = cuda.device_array((size,), dtype=np.float32)
    kerneltester[blocks_per_grid, threads_per_block](devicearray)
    cuda.synchronize()
    device_on_host = devicearray.copy_to_host()
    # print(f"device: {devicearray[10]}")
    del devicearray

run_pinned()
run_device()
run_mapped()
