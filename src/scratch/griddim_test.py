from numba import cuda, float32
from CuMC._utils import timing
import numpy as np

@cuda.jit()
def kerneltester(outarray, nruns):
    """
    A simple CUDA kernel that writes a value to the output array.
    This is used to test if the CUDA environment is set up correctly.
    """
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    run_index = tx + ty * cuda.blockDim.x + bx * cuda.blockDim.x * cuda.gridDim.y + by * cuda.blockDim.y * cuda.gridDim.x

    if run_index >= nruns:
        return None

    outarray[run_index, 0] = float32(tx)
    outarray[run_index, 1] = float32(ty)
    outarray[run_index, 2] = float32(bx)
    outarray[run_index, 3] = float32(by)

def run(blockdim,
        griddim):
    """
    Run the kernel with the specified block and grid dimensions.
    """
    if isinstance(blockdim, int):
        grids = blockdim
    else:
        grids = griddim[0] * griddim[1]
    if isinstance(griddim, int):
        blocks = griddim
    else:
        blocks = blockdim[0] * blockdim[1]
    runs = grids * blocks
    mappedarray = cuda.mapped_array((runs, 4), dtype=np.float32)

    kerneltester[griddim, blockdim](mappedarray, runs)
    cuda.synchronize()
    print(blockdim, griddim)
    print(mappedarray)

run((2, 2), (2, 2))
run((2, 1), (2, 1))

run((1, 1), (1, 1))
run(2,2)
run(1,1)

