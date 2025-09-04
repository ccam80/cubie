import numpy as np
import pytest
from numba import cuda

from cubie.integrators.matrix_free_solvers.minimal_residual import (
    minimal_residual_solver_factory
)
from cubie.integrators.matrix_free_solvers.newton_krylov import (
    neumann_preconditioner_factory,
    newton_krylov_solver_factory,
)


@pytest.fixture(scope="function")
def preconditioner_settings_override(request):
    return request.param if hasattr(request, "param") else {}


@pytest.fixture(scope="function")
def preconditioner_settings(preconditioner_settings_override):
    settings = {"order": 2}
    settings.update(preconditioner_settings_override)
    return settings


@pytest.fixture(scope="function")
def preconditioner_device(preconditioner_settings):
    return neumann_preconditioner_factory(**preconditioner_settings)


@pytest.fixture(scope="function")
def solver_factory_settings_override(request):
    return request.param if hasattr(request, "param") else {}


@pytest.fixture(scope="function")
def solver_factory_settings(solver_factory_settings_override):
    settings = {"tolerance": 1e-6, "max_iters": 8}
    settings.update(solver_factory_settings_override)
    return settings


@pytest.fixture(scope="function")
def linear_solver_device():
    settings = {"tolerance": 1e-6, "max_iters": 8}
    return minimal_residual_solver_factory(**settings)


@pytest.fixture(scope="function")
def solver_device(solver_factory_settings, linear_solver_device):
    return newton_krylov_solver_factory(
        linear_solver=linear_solver_device, **solver_factory_settings
    )


@pytest.mark.parametrize(
    "preconditioner_settings_override, expected",
    [({"order": 1}, np.asarray([1.1, 1.1, 1.1])),
     ({"order": 2}, np.asarray([1.21, 1.21, 1.21]))],
    indirect=["preconditioner_settings_override"],
)
def test_neumann_preconditioner(
    preconditioner_device, preconditioner_settings, expected, precision
):
    n = 3

    @cuda.jit(device=True)
    def jvp(state, vec, out):
        for i in range(n):
            out[i] = precision(0.1) * vec[i]

    @cuda.jit
    def kernel(out_array):
        vec = cuda.local.array(n, precision)
        tmp = cuda.local.array(n, precision)
        state = cuda.local.array(n, precision)
        for i in range(n):
            vec[i] = precision(1.0)
            state[i] = precision(0.0)
        preconditioner_device(jvp, state, vec, out_array, tmp)

    out = cuda.to_device(np.zeros(n, dtype=precision))
    kernel[1, 1](out)
    assert np.allclose(out.copy_to_host(), expected.astype(precision))


@pytest.mark.parametrize(
    "preconditioner_settings_override",
    [({"order": 2})],
    indirect=True,
)
def test_newton_krylov_solver_linear_system(
    solver_device, preconditioner_device, precision
):
    n = 3
    b = np.array([1.0, 2.0, 3.0], dtype=precision)
    b0 = precision(b[0])
    b1 = precision(b[1])
    b2 = precision(b[2])

    @cuda.jit(device=True)
    def jvp(state, vec, out):
        for i in range(n):
            out[i] = precision(0.5) * vec[i]

    @cuda.jit(device=True)
    def residual(state, out):
        out[0] = precision(0.5) * state[0] - b0
        out[1] = precision(0.5) * state[1] - b1
        out[2] = precision(0.5) * state[2] - b2

    @cuda.jit
    def kernel(x):
        res = cuda.local.array(n, precision)
        rhs = cuda.local.array(n, precision)
        delta = cuda.local.array(n, precision)
        z_vec = cuda.local.array(n, precision)
        v_vec = cuda.local.array(n, precision)
        precond_temp = cuda.local.array(n, precision)
        solver_device(
            jvp,
            residual,
            x,
            res,
            rhs,
            delta,
            z_vec,
            v_vec,
            preconditioner_device,
            precond_temp,
        )

    x = cuda.to_device(np.zeros(n, dtype=precision))
    kernel[1, 1](x)
    assert np.allclose(x.copy_to_host(), 2 * b, atol=1e-4)