"""Utility device functions for matrix-free solvers."""

from numba import cuda


@cuda.jit(device=True)
def vector_norm(vector, out):
    """Compute the Euclidean norm of ``vector`` in place.

    Parameters
    ----------
    vector : numba.cuda.cudadrv.devicearray.DeviceNDArray
        Input vector.
    out : numba.cuda.cudadrv.devicearray.DeviceNDArray
        Single-element array receiving the norm.
    """
    norm = 0.0
    for i in range(vector.shape[0]):
        norm += vector[i] * vector[i]
    out[0] = norm ** 0.5
