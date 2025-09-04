import numpy as np
import pytest
from numba import cuda

from cubie.integrators.matrix_free_solvers.minimal_residual import (
    minimal_residual_solver_factory)


@pytest.fixture(scope="function")
def mr_factory_settings_override(request):
    return request.param if hasattr(request, "param") else {}


@pytest.fixture(scope="function")
def mr_factory_settings(mr_factory_settings_override):
    settings = {"tolerance": 1e-6, "max_iters": 8}
    settings.update(mr_factory_settings_override)
    return settings


@pytest.fixture(scope="function")
def mr_solver_device(mr_factory_settings):
    return minimal_residual_solver_factory(**mr_factory_settings)


def test_minimal_residual_solver_linear_system(mr_solver_device, precision):
    n = 3
    b = np.array([1.0, 2.0, 3.0], dtype=precision)

    @cuda.jit(device=True)
    def jvp(state, vec, out):
        for i in range(n):
            out[i] = precision(0.5) * vec[i]

    @cuda.jit
    def kernel(x, rhs):
        residual = cuda.local.array(n, precision)
        z_vec = cuda.local.array(n, precision)
        v_vec = cuda.local.array(n, precision)
        precond_temp = cuda.local.array(n, precision)
        state = cuda.local.array(n, precision)
        for i in range(n):
            state[i] = precision(0.0)
            x[i] = precision(0.0)
        mr_solver_device(
            jvp, state, rhs, x, residual, z_vec, v_vec, None, precond_temp
        )

    x = cuda.to_device(np.zeros(n, dtype=precision))
    rhs = cuda.to_device(b)
    kernel[1, 1](x, rhs)
    assert np.allclose(x.copy_to_host(), 2 * b, atol=1e-4)