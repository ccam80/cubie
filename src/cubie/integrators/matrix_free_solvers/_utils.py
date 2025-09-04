"""Utility device functions for matrix-free solvers."""

from numba import cuda


@cuda.jit(device=True)
def vector_norm(vector):
    """Return the Euclidean norm of ``vector``.

    Parameters
    ----------
    vector : numba.cuda.cudadrv.devicearray.DeviceNDArray
        Input vector.

    Returns
    -------
    float
        Euclidean norm of ``vector``.
    """
    norm = 0.0
    for i in range(vector.shape[0]):
        norm += vector[i] * vector[i]
    return norm ** 0.5
