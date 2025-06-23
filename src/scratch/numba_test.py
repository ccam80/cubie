if __name__ == "__main__":
    import os
    os.environ["NUMBA_ENABLE_CUDASIM"] = "0"
    os.environ["NUMBA_CUDA_DEBUGINFO"] = "1"
    os.environ["NUMBA_OPT"] = "0"

from numba import cuda, int32
import numpy as np


@cuda.jit(debug=True)
def sm_slice(x):
    dynsmem = cuda.shared.array(0, dtype=int32)
    sm1 = dynsmem[0:1]
    sm2 = dynsmem[1:2]

    sm1[0] = 1
    sm2[0] = 2
    x[0] = sm1[0]
    x[1] = sm2[0]


arr = np.zeros(2, dtype=np.int32)
nshared = 2 * arr.dtype.itemsize
sm_slice[1, 1, 0, nshared](arr)
print(arr)