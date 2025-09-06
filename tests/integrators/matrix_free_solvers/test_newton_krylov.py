import numpy as np
import pytest
from numba import cuda

from cubie.integrators.matrix_free_solvers.linear_solver import (
    linear_solver_factory,
)
from cubie.integrators.matrix_free_solvers.newton_krylov import (
    newton_krylov_solver_factory,
)


@pytest.fixture(scope="function")
def placeholder_system(precision):
    """Provide residual and operator for a scalar nonlinear equation."""

    @cuda.jit(device=True)
    def residual(state, parameters, drivers, h, base_state, work, out):
        # Ignore base_state and work in this placeholder
        out[0] = state[0] * state[0] - precision(2.0)

    @cuda.jit(device=True)
    def operator(state, parameters, drivers, h, vec, out):
        out[0] = precision(2.0) * state[0] * vec[0]

    # Provide a trivial base_state for the test kernel to pass through
    base = cuda.to_device(np.array([0.0], dtype=precision))
    return residual, operator, base


def test_newton_krylov_placeholder(placeholder_system, precision):
    """Solve a simple nonlinear system using Newton-Krylov."""

    residual, operator, base_state = placeholder_system
    linear_solver = linear_solver_factory(
        operator, tolerance=1e-12, max_iters=32
    )
    solver = newton_krylov_solver_factory(
        residual_function=residual,
        linear_solver=linear_solver,
        tolerance=1e-10,
        max_iters=16,
    )

    @cuda.jit
    def kernel(state, base):
        res = cuda.local.array(1, precision)
        delta = cuda.local.array(1, precision)
        z_vec = cuda.local.array(1, precision)
        v_vec = cuda.local.array(1, precision)
        params = cuda.local.array(1, precision)
        drivers = cuda.local.array(1, precision)
        h = precision(1.0)
        solver(state, params, drivers, h, base, delta, res, z_vec, v_vec)

    x = cuda.to_device(np.array([1.0], dtype=precision))
    kernel[1, 1](x, base_state)
    expected = np.array([np.sqrt(2.0)], dtype=precision)
    assert np.allclose(x.copy_to_host(), expected, atol=1e-4)


@pytest.mark.parametrize(
    "system_setup",
    [
        "linear",
        "nonlinear",
        "stiff",
        "coupled_linear",
        "coupled_nonlinear",
    ],
    indirect=True,
)
def test_newton_krylov_symbolic(system_setup, precision):
    """Solve systems built from symbolic expressions."""

    n = system_setup["n"]
    operator = system_setup["operator"]
    residual_func = system_setup["residual"]
    base_state = system_setup["base_state"]
    expected = system_setup["nk_expected"]
    linear_solver = linear_solver_factory(
        operator,
        correction_type="minimal_residual",
        tolerance=1e-6,
        max_iters=1000,
    )
    solver = newton_krylov_solver_factory(
        residual_function=residual_func,
        linear_solver=linear_solver,
        tolerance=1e-8,
        max_iters=32,
    )

    @cuda.jit
    def kernel(state, base):
        res = cuda.local.array(n, precision)
        delta = cuda.local.array(n, precision)
        z_vec = cuda.local.array(n, precision)
        v_vec = cuda.local.array(n, precision)
        params = cuda.local.array(1, precision)
        drivers = cuda.local.array(1, precision)
        h = precision(1.0)
        solver(state, params, drivers, h, base, delta, res, z_vec, v_vec)

    x = cuda.to_device(np.zeros(n, dtype=precision))
    kernel[1, 1](x, base_state)
    assert np.allclose(x.copy_to_host(), expected, atol=1e-4)
