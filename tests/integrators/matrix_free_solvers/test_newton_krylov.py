import numpy as np
import pytest
from numba import cuda

from cubie.integrators.matrix_free_solvers.linear_solver import (
    linear_solver_factory,
    neumann_preconditioner_factory,
)
from cubie.integrators.matrix_free_solvers.newton_krylov import (
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
    settings = {"tolerance": 1e-6, "max_iters": 64}
    settings.update(solver_factory_settings_override)
    return settings


@pytest.fixture(scope="function")
def solver_device(solver_factory_settings, system_setup):
    operator = system_setup["operator"]
    residual_func = system_setup["residual"]
    linear_solver = linear_solver_factory(
        operator,
        correction_type="minimal_residual",
        tolerance=1e-6,
        max_iters=32,
    )
    return newton_krylov_solver_factory(
        residual_function=residual_func,
        linear_solver=linear_solver,
        **solver_factory_settings,
    )


@pytest.mark.parametrize(
    "preconditioner_settings_override, expected",
    [
        ({"order": 1}, np.asarray([1.1, 1.1, 1.1])),
        ({"order": 2}, np.asarray([1.21, 1.21, 1.21])),
    ],
    indirect=["preconditioner_settings_override"],
)
def test_neumann_preconditioner(
    preconditioner_settings, expected, precision
):
    n = 3

    @cuda.jit(device=True)
    def neumann_operator(state, parameters, drivers, h, residual, out):
        for i in range(n):
            out[i] = out[i] + h * precision(0.1) * out[i]

    preconditioner = neumann_preconditioner_factory(
        neumann_operator, **preconditioner_settings
    )

    @cuda.jit
    def kernel(out_array):
        vec = cuda.local.array(n, precision)
        state = cuda.local.array(n, precision)
        parameters = cuda.local.array(1, precision)
        drivers = cuda.local.array(1, precision)
        h = precision(1.0)
        for i in range(n):
            vec[i] = precision(1.0)
            state[i] = precision(0.0)
        preconditioner(state, parameters, drivers, h, vec, out_array)

    out = cuda.to_device(np.zeros(n, dtype=precision))
    kernel[1, 1](out)
    assert np.allclose(out.copy_to_host(), expected.astype(precision))


@pytest.mark.parametrize(
    "system_setup",
    [
        "linear",
        "nonlinear",
    ],
    indirect=True,
)
def test_newton_krylov_solver(system_setup, solver_device, precision):
    n = system_setup["n"]
    expected = system_setup["nk_expected"]

    @cuda.jit
    def kernel(state):
        res = cuda.local.array(n, precision)
        rhs = cuda.local.array(n, precision)
        delta = cuda.local.array(n, precision)
        z_vec = cuda.local.array(n, precision)
        v_vec = cuda.local.array(n, precision)
        parameters = cuda.local.array(1, precision)
        drivers = cuda.local.array(1, precision)
        h = precision(1.0)
        for i in range(n):
            rhs[i] = precision(0.0)
        solver_device(
            state,
            parameters,
            drivers,
            h,
            rhs,
            delta,
            res,
            z_vec,
            v_vec,
        )

    x = cuda.to_device(np.zeros(n, dtype=precision))
    kernel[1, 1](x)
    assert np.allclose(x.copy_to_host(), expected, atol=1e-4)


@pytest.mark.parametrize("system_setup", ["stiff"], indirect=True)
@pytest.mark.xfail(reason="solver struggles with stiff systems")
def test_newton_krylov_stiff_failure(system_setup, solver_device, precision):
    n = system_setup["n"]
    expected = system_setup["nk_expected"]

    @cuda.jit
    def kernel(state):
        res = cuda.local.array(n, precision)
        rhs = cuda.local.array(n, precision)
        delta = cuda.local.array(n, precision)
        z_vec = cuda.local.array(n, precision)
        v_vec = cuda.local.array(n, precision)
        parameters = cuda.local.array(1, precision)
        drivers = cuda.local.array(1, precision)
        h = precision(1.0)
        for i in range(n):
            rhs[i] = precision(0.0)
        solver_device(
            state,
            parameters,
            drivers,
            h,
            rhs,
            delta,
            res,
            z_vec,
            v_vec,
        )

    x = cuda.to_device(np.zeros(n, dtype=precision))
    kernel[1, 1](x)
    assert np.allclose(x.copy_to_host(), expected, atol=1e-4)
