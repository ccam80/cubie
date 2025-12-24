import numpy as np
import pytest
from numba import cuda
from numpy.testing import assert_allclose

from cubie.integrators.matrix_free_solvers.linear_solver import (
    LinearSolver,
)
from cubie.integrators.matrix_free_solvers import SolverRetCodes


@pytest.fixture(scope="function")
def placeholder_operator(precision):
    """Device operator applying a simple SPD matrix."""

    @cuda.jit(device=True)
    def operator(state, parameters, drivers, base_state, t, h, a_ij, vec, out):
        out[0] = precision(4.0) * vec[0] + precision(1.0) * vec[1]
        out[1] = precision(1.0) * vec[0] + precision(3.0) * vec[1]
        out[2] = precision(2.0) * vec[2]

    return operator


# Removed placeholder Neumann factory usage; use real generated preconditioner via system_setup
@pytest.mark.parametrize("solver_settings_override",
                         [{"precision": np.float64}],
                         ids=[""],
                         indirect=True)
@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("system_setup", ["linear"], indirect=True)
def test_neumann_preconditioner(
    order,
    system_setup,
    neumann_kernel,
    precision,
    tolerance,
):
    """Validate Neumann preconditioner equals truncated series on the linear system.

    Uses the real generated preconditioner from system_setup and applies it to a
    vector of ones. For the 'linear' system, J is diagonal with 0.5 entries,
    beta=1, stage coefficient a_ij=1, and h=1, so the truncated series is
    sum_{k=0..order} (h*J)^k v.
    """

    n = system_setup["n"]
    h = system_setup["h"]
    precond = system_setup["preconditioner"](order)
    kernel = neumann_kernel(precond, n, h)

    residual = cuda.to_device(np.ones(n, dtype=precision))
    out = cuda.device_array(n, precision)
    state = system_setup["state_init"]
    empty_base = cuda.to_device(np.empty(0, dtype=precision))

    kernel[1, 1](state, residual, empty_base, out)

    expected_scalar = sum((h * precision(0.5)) ** k for k in range(order + 1))
    expected = np.full(n, expected_scalar, dtype=precision)
    assert_allclose(
        out.copy_to_host(),
        expected,
        rtol=tolerance.rel_tight,
        atol=tolerance.abs_tight,
    )


@pytest.fixture(scope="function")
def solver_device(request, placeholder_operator, precision):
    """Return solver device for the requested correction type."""

    solver = LinearSolver(
        precision=precision,
        n=3,
        linear_correction_type=request.param,
        krylov_tolerance=1e-12,
        max_linear_iters=32,
    )
    solver.update(operator_apply=placeholder_operator)
    return solver.device_function

@pytest.mark.parametrize(
    "solver_device", ["steepest_descent", "minimal_residual"], indirect=True
)
def test_linear_solver_placeholder(
    solver_device,
    solver_kernel,
    precision,
    tolerance,
):
    """Solve a simple linear system with the placeholder operator."""

    rhs = np.array([1.0, 2.0, 3.0], dtype=precision)
    matrix = np.array(
        [[4.0, 1.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, 2.0]],
        dtype=precision,
    )
    expected = np.linalg.solve(matrix, rhs)
    h = precision(0.01)
    kernel = solver_kernel(solver_device, 3, h, precision)
    base_state = np.array([1.0, -1.0, 0.5], dtype=precision)
    state = cuda.to_device(base_state + h * np.array([0.1, -0.2, 0.3], dtype=precision))
    rhs_dev = cuda.to_device(rhs)
    x_dev = cuda.to_device(np.zeros(3, dtype=precision))
    flag = cuda.to_device(np.array([0], dtype=np.int32))
    empty_base = cuda.to_device(np.empty(0, dtype=precision))
    kernel[1, 1](state, rhs_dev, empty_base, x_dev, flag)
    code = flag.copy_to_host()[0] & 0xFF
    assert code == SolverRetCodes.SUCCESS
    assert np.allclose(
        x_dev.copy_to_host(),
        expected,
        rtol=tolerance.rel_tight,
        atol=tolerance.abs_tight,
    )

@pytest.mark.parametrize(
    "system_setup", ["linear", "coupled_linear"], indirect=True
)
@pytest.mark.parametrize("linear_correction_type", ["steepest_descent", "minimal_residual"])
@pytest.mark.parametrize("precond_order", [0, 1, 2])
def test_linear_solver_symbolic(
    system_setup,
    solver_kernel,
    precision,
    linear_correction_type,
    precond_order,
    tolerance,
):
    """Solve systems built from symbolic expressions."""

    n = system_setup["n"]
    operator = system_setup["operator"]
    rhs_vec = system_setup["mr_rhs"]
    expected = system_setup["mr_expected"]
    h = system_setup["h"]
    precond = (
        None if precond_order == 0 else system_setup["preconditioner"](precond_order)
    )
    
    solver = LinearSolver(
        precision=precision,
        n=n,
        linear_correction_type=linear_correction_type,
        krylov_tolerance=1e-8,
        max_linear_iters=1000,
    )
    solver.update(operator_apply=operator, preconditioner=precond)
    solver = solver.device_function
    
    kernel = solver_kernel(solver, n, h, precision)
    state = system_setup["state_init"]
    rhs_dev = cuda.to_device(rhs_vec)
    x_dev = cuda.to_device(np.zeros(n, dtype=precision))
    flag = cuda.to_device(np.array([0], dtype=np.int32))
    empty_base = cuda.to_device(np.empty(0, dtype=precision))
    kernel[1, 1](state, rhs_dev, empty_base, x_dev, flag)
    code = flag.copy_to_host()[0] & 0xFF
    assert code == SolverRetCodes.SUCCESS
    assert np.allclose(
        x_dev.copy_to_host(),
        expected,
        rtol=tolerance.rel_tight,
        atol=tolerance.abs_tight,
    )


def test_linear_solver_max_iters_exceeded(solver_kernel, precision):
    """Linear solver returns MAX_LINEAR_ITERATIONS_EXCEEDED when operator is zero."""

    @cuda.jit(device=True)
    def zero_operator(state, parameters, drivers, base_state, t, h, a_ij, vec, out):
        # F z = 0 for all z -> no progress in line search
        for i in range(out.shape[0]):
            out[i] = precision(0.0)

    n = 3
    solver = LinearSolver(
        precision=precision,
        n=n,
        linear_correction_type="minimal_residual",
        krylov_tolerance=1e-20,
        max_linear_iters=16,
    )
    solver.update(operator_apply=zero_operator)
    solver = solver.device_function

    h = precision(0.01)
    kernel = solver_kernel(solver, n, h, precision)
    state = cuda.to_device(np.zeros(n, dtype=precision))
    rhs_dev = cuda.to_device(np.ones(n, dtype=precision))
    x_dev = cuda.to_device(np.zeros(n, dtype=precision))
    flag = cuda.to_device(np.array([0], dtype=np.int32))
    empty_base = cuda.to_device(np.empty(0, dtype=precision))
    kernel[1, 1](state, rhs_dev, empty_base, x_dev, flag)
    code = flag.copy_to_host()[0] & 0xFF
    assert code == SolverRetCodes.MAX_LINEAR_ITERATIONS_EXCEEDED
