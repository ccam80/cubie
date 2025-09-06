import numpy as np
import pytest
from numba import cuda

from cubie.integrators.matrix_free_solvers._utils import vector_norm


@pytest.fixture(scope="function")
def norm_kernel(precision):
    """Kernel exposing vector_norm."""

    def factory(n):
        @cuda.jit
        def kernel(vec, out):
            vector_norm(vec, out)

        return kernel

    return factory


def test_vector_norm(norm_kernel, precision):
    """Evaluate Euclidean norm on device."""

    kernel = norm_kernel(3)
    vec = cuda.to_device(np.array([3.0, 4.0, 0.0], dtype=precision))
    out = cuda.device_array(1, precision)
    kernel[1, 1](vec, out)
    assert np.allclose(out.copy_to_host()[0], 5.0, atol=1e-7)

