import numpy as np
import pytest
from numba import cuda

from cubie.integrators.matrix_free_solvers.gmres import gmres_solver_factory


@pytest.fixture(scope="function")
def gmres_factory_settings_override(request):
    return request.param if hasattr(request, "param") else {}


@pytest.fixture(scope="function")
def gmres_factory_settings(gmres_factory_settings_override):
    settings = {"tolerance": 1e-6, "max_iters": 3}
    settings.update(gmres_factory_settings_override)
    return settings


@pytest.fixture(scope="function")
def gmres_solver_device(gmres_factory_settings):
    return gmres_solver_factory(**gmres_factory_settings)


def test_gmres_solver_linear_system(
    gmres_solver_device, gmres_factory_settings, precision
):
    n = 3
    m = gmres_factory_settings["max_iters"]
    b = np.array([1.0, 2.0, 3.0], dtype=precision)

    @cuda.jit(device=True)
    def jvp(state, vec, out):
        for i in range(n):
            out[i] = precision(0.5) * vec[i]

    @cuda.jit
    def kernel(x, rhs):
        basis = cuda.local.array((m + 1, n), precision)
        hess = cuda.local.array((m + 1, m), precision)
        giv = cuda.local.array((2, m), precision)
        gvec = cuda.local.array(m + 1, precision)
        wvec = cuda.local.array(n, precision)
        state = cuda.local.array(n, precision)
        for i in range(n):
            state[i] = precision(0.0)
        gmres_solver_device(jvp, state, rhs, x, basis, hess, giv, gvec, wvec)

    x = cuda.to_device(np.zeros(n, dtype=precision))
    rhs = cuda.to_device(b)
    kernel[1, 1](x, rhs)
    assert np.allclose(x.copy_to_host(), 2 * b, atol=1e-4)