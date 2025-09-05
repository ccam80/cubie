import numpy as np
import pytest
from numba import cuda

from cubie.integrators.matrix_free_solvers.linear_solver import (
    linear_solver_factory,
    neumann_preconditioner_factory,
)


@pytest.fixture(scope="function")
def placeholder_operator(precision):
    """Device operator applying a simple SPD matrix."""

    @cuda.jit(device=True)
    def operator(state, parameters, drivers, h, vec, out):
        out[0] = precision(4.0) * vec[0] + precision(1.0) * vec[1]
        out[1] = precision(1.0) * vec[0] + precision(3.0) * vec[1]
        out[2] = precision(2.0) * vec[2]

    return operator


@pytest.fixture(scope="function")
def placeholder_neumann_operator(precision):
    """Operator used inside the Neumann preconditioner tests."""

    c0, c1, c2 = precision(0.1), precision(0.2), precision(0.3)

    @cuda.jit(device=True)
    def neumann(state, parameters, drivers, h, residual, out):
        out[0] = residual[0] + h * c0 * out[0]
        out[1] = residual[1] + h * c1 * out[1]
        out[2] = residual[2] + h * c2 * out[2]

    return neumann


@pytest.fixture(scope="function")
def solver_device(request, placeholder_operator):
    """Return solver device for the requested correction type."""

    return linear_solver_factory(
        placeholder_operator,
        correction_type=request.param,
        tolerance=1e-12,
        max_iters=32,
    )


def  toftest_neumann_preconditioner(
    placeholder_neumann_operator, neumann_kernel, precision
):
    """Validate Neumann preconditioner against a polynomial expansion."""

    precond = neumann_preconditioner_factory(
        placeholder_neumann_operator, order=2
    )
    kernel = neumann_kernel(precond, 3)
    residual = cuda.to_device(np.ones(3, dtype=precision))
    out = cuda.device_array(3, precision)
    state = cuda.to_device(np.zeros(3, dtype=precision))
    kernel[1, 1](state, residual, out)
    expected = np.array([1.11, 1.24, 1.39], dtype=precision)
    assert np.allclose(out.copy_to_host(), expected, atol=1e-7)


@pytest.mark.parametrize(
    "solver_device",
    [
        "steepest_descent",
        pytest.param(
            "minimal_residual",
            marks=pytest.mark.xfail(reason="minimal residual variant fails"),
        ),
    ],
    indirect=True,
)
def test_linear_solver_placeholder(solver_device, solver_kernel, precision):
    """Solve a simple linear system with the placeholder operator."""

    rhs = np.array([1.0, 2.0, 3.0], dtype=precision)
    matrix = np.array(
        [[4.0, 1.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, 2.0]],
        dtype=precision,
    )
    expected = np.linalg.solve(matrix, rhs)
    kernel = solver_kernel(solver_device, 3)
    state = cuda.to_device(np.zeros(3, dtype=precision))
    rhs_dev = cuda.to_device(rhs)
    x_dev = cuda.to_device(np.zeros(3, dtype=precision))
    residual = cuda.device_array(3, precision)
    z_vec = cuda.device_array(3, precision)
    temp = cuda.device_array(3, precision)
    kernel[1, 1](state, rhs_dev, x_dev, residual, z_vec, temp)
    assert np.allclose(x_dev.copy_to_host(), expected, atol=1e-5)


@pytest.mark.parametrize(
    "system_setup",
    [
        "linear",
        pytest.param(
            "nonlinear",
            marks=pytest.mark.xfail(reason="nonlinear case diverges"),
        ),
    ],
    indirect=True,
)
def test_linear_solver_symbolic(system_setup, solver_kernel, precision):
    """Solve systems built from symbolic expressions."""

    n = system_setup["n"]
    operator = system_setup["jvp"]
    rhs_vec = system_setup["mr_rhs"]
    expected = system_setup["mr_expected"]
    solver = linear_solver_factory(
        operator,
        correction_type="steepest_descent",
        tolerance=1e-8,
        max_iters=64,
    )
    kernel = solver_kernel(solver, n)
    state = cuda.to_device(expected)
    rhs_dev = cuda.to_device(rhs_vec)
    x_dev = cuda.to_device(np.zeros(n, dtype=precision))
    residual = cuda.device_array(n, precision)
    z_vec = cuda.device_array(n, precision)
    temp = cuda.device_array(n, precision)
    kernel[1, 1](state, rhs_dev, x_dev, residual, z_vec, temp)
    assert np.allclose(x_dev.copy_to_host(), expected, atol=1e-4)


@pytest.mark.parametrize("system_setup", ["stiff"], indirect=True)
@pytest.mark.xfail(reason="solver struggles with stiff systems")
def test_linear_solver_stiff_failure(system_setup, solver_kernel, precision):
    """Demonstrate failure on a stiff system without preconditioning."""

    n = system_setup["n"]
    operator = system_setup["jvp"]
    rhs_vec = system_setup["mr_rhs"]
    expected = system_setup["mr_expected"]
    solver = linear_solver_factory(
        operator,
        correction_type="minimal_residual",
        tolerance=1e-8,
        max_iters=4,
    )
    kernel = solver_kernel(solver, n)
    state = cuda.to_device(expected)
    rhs_dev = cuda.to_device(rhs_vec)
    x_dev = cuda.to_device(np.zeros(n, dtype=precision))
    residual = cuda.device_array(n, precision)
    z_vec = cuda.device_array(n, precision)
    temp = cuda.device_array(n, precision)
    kernel[1, 1](state, rhs_dev, x_dev, residual, z_vec, temp)
    assert np.allclose(x_dev.copy_to_host(), expected, atol=1e-4)
