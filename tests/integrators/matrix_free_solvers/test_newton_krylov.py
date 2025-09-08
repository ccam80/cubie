import numpy as np
import pytest
from numba import cuda

from cubie.integrators.matrix_free_solvers.linear_solver import (
    linear_solver_factory,
)
from cubie.integrators.matrix_free_solvers.newton_krylov import (
    newton_krylov_solver_factory,
)
from cubie.integrators.matrix_free_solvers import SolverRetCodes


@pytest.fixture(scope="function")
def placeholder_system(precision):
    """Provide residual and operator for a scalar ODE step."""

    @cuda.jit(device=True)
    def residual(state, parameters, drivers, h, a_ij, base_state, out):
        out[0] = state[0] - base_state[0] - h * state[0]

    @cuda.jit(device=True)
    def operator(state, parameters, drivers, h, vec, out):
        out[0] = (precision(1.0) - h) * vec[0]

    base = cuda.to_device(np.array([1.0], dtype=precision))
    return residual, operator, base


def test_newton_krylov_placeholder(placeholder_system, precision):
    """Solve a simple implicit Euler step using Newton-Krylov."""

    residual, operator, base_state = placeholder_system
    n = 1
    linear_solver = linear_solver_factory(
        operator, n, tolerance=1e-8, max_iters=32
    )
    solver = newton_krylov_solver_factory(
        residual_function=residual,
        linear_solver=linear_solver,
        n=n,
        tolerance=1e-6,
        max_iters=16,
    )

    @cuda.jit
    def kernel(state, base, flag, h):
        res = cuda.local.array(1, precision)
        delta = cuda.local.array(1, precision)
        z_vec = cuda.local.array(1, precision)
        v_vec = cuda.local.array(1, precision)
        params = cuda.local.array(1, precision)
        drivers = cuda.local.array(1, precision)
        a_ij = precision(1.0)
        flag[0] = solver(
            state, params, drivers, h, a_ij, base, delta, res, z_vec, v_vec
        )

    h = precision(0.01)
    expected = np.array([base_state.copy_to_host()[0] / (1.0 - h)], dtype=precision)
    x0 = expected * precision(0.99)
    x = cuda.to_device(x0)
    out_flag = cuda.to_device(np.array([0], dtype=np.int32))
    kernel[1, 1](x, base_state, out_flag, h)
    assert out_flag.copy_to_host()[0] == SolverRetCodes.SUCCESS
    assert np.allclose(x.copy_to_host(), expected, atol=1e-5)


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
@pytest.mark.parametrize("precond_order", [0, 1, 2])
def test_newton_krylov_symbolic(system_setup, precision, precond_order):
    """Solve a symbolic system with optional preconditioning provided by fixture."""

    n = system_setup["n"]
    operator = system_setup["operator"]
    residual_func = system_setup["residual"]
    base_state = system_setup["base_state"]
    expected = system_setup["nk_expected"]
    h = system_setup["h"]
    # Use the real Neumann preconditioner factory from the fixture
    precond = (
        None if precond_order == 0 else system_setup["preconditioner"](precond_order)
    )
    linear_solver = linear_solver_factory(
        operator,
        n,
        preconditioner=precond,
        correction_type="minimal_residual",
        tolerance=1e-6,
        max_iters=1000,
    )
    solver = newton_krylov_solver_factory(
        residual_function=residual_func,
        linear_solver=linear_solver,
        n=n,
        tolerance=1e-8,
        max_iters=1000,
    )

    @cuda.jit
    def kernel(state, base, flag, h):
        res = cuda.local.array(n, precision)
        delta = cuda.local.array(n, precision)
        z_vec = cuda.local.array(n, precision)
        v_vec = cuda.local.array(n, precision)
        params = cuda.local.array(1, precision)
        drivers = cuda.local.array(1, precision)
        a_ij = precision(1.0)
        flag[0] = solver(
            state, params, drivers, h, a_ij, base, delta, res, z_vec, v_vec
        )

    x = system_setup["state_init"]
    out_flag = cuda.to_device(np.array([0], dtype=np.int32))
    kernel[1, 1](x, base_state, out_flag, h)
    retcode = out_flag.copy_to_host()
    #Nonlinear system needs preconditioning.
    if system_setup["id"] == 'nonlinear' and precond_order == 0:
        assert retcode == SolverRetCodes.NO_SUITABLE_STEP_FOUND
    else:
        assert retcode == SolverRetCodes.SUCCESS
        assert np.allclose(x.copy_to_host(), expected, atol=1e-4)


def test_newton_krylov_failure(precision):
    """Solver returns NO_SUITABLE_STEP_FOUND when residual cannot be reduced."""

    @cuda.jit(device=True)
    def residual(state, parameters, drivers, h, a_ij, base_state, out):
        out[0] = precision(1.0)

    @cuda.jit(device=True)
    def operator(state, parameters, drivers, h, vec, out):
        out[0] = vec[0]

    n = 1
    linear_solver = linear_solver_factory(operator, n, tolerance=1e-12, max_iters=8)
    solver = newton_krylov_solver_factory(
        residual_function=residual,
        linear_solver=linear_solver,
        n=n,
        tolerance=1e-8,
        max_iters=2,
    )

    @cuda.jit
    def kernel(flag, h):
        state = cuda.local.array(1, precision)
        res = cuda.local.array(1, precision)
        delta = cuda.local.array(1, precision)
        z_vec = cuda.local.array(1, precision)
        v_vec = cuda.local.array(1, precision)
        params = cuda.local.array(1, precision)
        drivers = cuda.local.array(1, precision)
        a_ij = precision(1.0)
        base = cuda.local.array(1, precision)
        flag[0] = solver(
            state, params, drivers, h, a_ij, base, delta, res, z_vec, v_vec
        )

    out_flag = cuda.to_device(np.array([1], dtype=np.int32))
    kernel[1, 1](out_flag, precision(0.01))
    assert out_flag.copy_to_host()[0] == SolverRetCodes.NO_SUITABLE_STEP_FOUND


def test_newton_krylov_max_newton_iters_exceeded(placeholder_system, precision):
    """Returns MAX_NEWTON_ITERATIONS_EXCEEDED when max_iters=0 and residual>tolerance."""

    residual, operator, base_state = placeholder_system
    n = 1
    linear_solver = linear_solver_factory(operator, n, tolerance=1e-8, max_iters=32)
    solver = newton_krylov_solver_factory(
        residual_function=residual,
        linear_solver=linear_solver,
        n=n,
        tolerance=1e-6,
        max_iters=0,  # force no Newton iterations
    )

    @cuda.jit
    def kernel(state, base, flag, h):
        res = cuda.local.array(1, precision)
        delta = cuda.local.array(1, precision)
        z_vec = cuda.local.array(1, precision)
        v_vec = cuda.local.array(1, precision)
        params = cuda.local.array(1, precision)
        drivers = cuda.local.array(1, precision)
        a_ij = precision(1.0)
        flag[0] = solver(
            state, params, drivers, h, a_ij, base, delta, res, z_vec, v_vec
        )

    h = precision(0.01)
    x = cuda.to_device(np.array([0.0], dtype=precision))  # ensures residual>tol
    out_flag = cuda.to_device(np.array([0], dtype=np.int32))
    kernel[1, 1](x, base_state, out_flag, h)
    assert out_flag.copy_to_host()[0] == SolverRetCodes.MAX_NEWTON_ITERATIONS_EXCEEDED


def test_newton_krylov_linear_solver_failure_propagates(precision):
    """Newton-Krylov returns MAX_LINEAR_ITERATIONS_EXCEEDED when inner solver fails."""

    @cuda.jit(device=True)
    def residual(state, parameters, drivers, h, a_ij, base_state, out):
        # Simple residual: nonzero so that a linear solve is attempted
        out[0] = precision(1.0)

    @cuda.jit(device=True)
    def zero_operator(state, parameters, drivers, h, vec, out):
        # Linear operator always zero => inner solver cannot make progress
        out[0] = precision(0.0)

    # Inner linear solver will return MAX_LINEAR_ITERATIONS_EXCEEDED
    n = 1
    linear_solver = linear_solver_factory(
        zero_operator,
        n,
        correction_type="minimal_residual",
        tolerance=1e-20,
        max_iters=8,
    )
    solver = newton_krylov_solver_factory(
        residual_function=residual,
        linear_solver=linear_solver,
        n=n,
        tolerance=1e-8,
        max_iters=4,
    )

    @cuda.jit
    def kernel(flag, h):
        state = cuda.local.array(1, precision)
        res = cuda.local.array(1, precision)
        delta = cuda.local.array(1, precision)
        z_vec = cuda.local.array(1, precision)
        v_vec = cuda.local.array(1, precision)
        params = cuda.local.array(1, precision)
        drivers = cuda.local.array(1, precision)
        a_ij = precision(1.0)
        base = cuda.local.array(1, precision)
        flag[0] = solver(
            state, params, drivers, h, a_ij, base, delta, res, z_vec, v_vec
        )

    out_flag = cuda.to_device(np.array([0], dtype=np.int32))
    kernel[1, 1](out_flag, precision(0.01))
    assert (
        out_flag.copy_to_host()[0]
        == SolverRetCodes.MAX_LINEAR_ITERATIONS_EXCEEDED
    )