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

STATUS_MASK = 0xFFFF


@pytest.fixture(scope="function")
def placeholder_system(precision):
    """Provide residual and operator for a scalar ODE step."""

    @cuda.jit(device=True)
    def residual(state, parameters, drivers, t, h, a_ij, base_state, out):
        out[0] = state[0] - h * (base_state[0] + a_ij * state[0])

    @cuda.jit(device=True)
    def operator(state, parameters, drivers, base_state, t, h, a_ij, vec, out):
        out[0] = (precision(1.0) - h * a_ij)  * vec[0]

    base = cuda.to_device(np.array([1.0], dtype=precision))
    return residual, operator, base


def test_newton_krylov_placeholder(placeholder_system, precision, tolerance):
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

    scratch_len = 3 * n

    @cuda.jit
    def kernel(state, base, flag, h):
        params = cuda.local.array(1, precision)
        drivers = cuda.local.array(1, precision)
        counters = cuda.local.array(2, np.int32)
        a_ij = precision(1.0)
        shared = cuda.shared.array(scratch_len, precision)
        time_scalar = precision(0.0)
        flag[0] = solver(
            state,
            params,
            drivers,
            time_scalar,
            h,
            a_ij,
            base,
            shared,
            counters,
        )

    h = precision(0.01)
    base_val = base_state.copy_to_host()[0]
    expected_final = precision(base_val / (1.0 - h))
    expected_increment = np.array([expected_final - base_val], dtype=precision)
    x0 = expected_increment * precision(0.99)
    x = cuda.to_device(x0)
    out_flag = cuda.to_device(np.array([0], dtype=np.int32))
    kernel[1, 1](x, base_state, out_flag, h)
    status_code = int(out_flag.copy_to_host()[0]) & STATUS_MASK

    assert status_code == SolverRetCodes.SUCCESS
    result_increment = x.copy_to_host()
    assert np.allclose(
        result_increment,
        expected_increment,
        rtol=tolerance.rel_tight,
        atol=tolerance.abs_tight,
    )

@pytest.mark.parametrize("precision_override", [np.float64], indirect=True,
                         ids=[""])
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
def test_newton_krylov_symbolic(system_setup, precision, precond_order, tolerance):
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

    scratch_len = 3 * n

    @cuda.jit
    def kernel(state, base, flag, h):
        params = cuda.local.array(1, precision)
        drivers = cuda.local.array(1, precision)
        counters = cuda.local.array(2, np.int32)
        a_ij = precision(1.0)
        shared = cuda.shared.array(scratch_len, precision)
        time_scalar = precision(0.0)
        flag[0] = solver(
            state,
            params,
            drivers,
            time_scalar,
            h,
            a_ij,
            base,
            shared,
            counters,
        )

    base_vals = system_setup["base_state"].copy_to_host()
    expected_increment = expected - base_vals
    x = system_setup["state_init"]
    out_flag = cuda.to_device(np.array([0], dtype=np.int32))
    kernel[1, 1](x, base_state, out_flag, h)
    status_code = int(out_flag.copy_to_host()[0]) & STATUS_MASK
    # Nonlinear system needs preconditioning.
    if system_setup["id"] == "nonlinear" and precond_order == 0:
        assert (
            status_code == SolverRetCodes.NEWTON_BACKTRACKING_NO_SUITABLE_STEP
        )
    else:
        assert status_code == SolverRetCodes.SUCCESS
        assert np.allclose(
            x.copy_to_host(),
            expected_increment,
            rtol=tolerance.rel_tight,
            atol=tolerance.abs_tight,
        )


def test_newton_krylov_failure(precision):
    """Solver returns NEWTON_BACKTRACKING_NO_SUITABLE_STEP when residual cannot be reduced."""

    @cuda.jit(device=True)
    def residual(state, parameters, drivers, t, h, a_ij, base_state, out):
        out[0] = precision(1.0)

    @cuda.jit(device=True)
    def operator(state, parameters, drivers, base_state, t, h, a_ij, vec, out):
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

    scratch_len = 3 * n

    @cuda.jit
    def kernel(flag, h):
        state = cuda.local.array(1, precision)
        params = cuda.local.array(1, precision)
        drivers = cuda.local.array(1, precision)
        counters = cuda.local.array(2, np.int32)
        a_ij = precision(1.0)
        base = cuda.local.array(1, precision)
        shared = cuda.shared.array(scratch_len, precision)
        time_scalar = precision(0.0)
        flag[0] = solver(
            state,
            params,
            drivers,
            time_scalar,
            h,
            a_ij,
            base,
            shared,
            counters,
        )

    out_flag = cuda.to_device(np.array([1], dtype=np.int32))
    kernel[1, 1](out_flag, precision(0.01))
    status_code = int(out_flag.copy_to_host()[0]) & STATUS_MASK
    assert (
            status_code
            == SolverRetCodes.NEWTON_BACKTRACKING_NO_SUITABLE_STEP
    )

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

    scratch_len = 3 * n

    @cuda.jit
    def kernel(state, base, flag, h):
        params = cuda.local.array(1, precision)
        drivers = cuda.local.array(1, precision)
        counters = cuda.local.array(2, np.int32)
        a_ij = precision(1.0)
        shared = cuda.shared.array(scratch_len, precision)
        time_scalar = precision(0.0)
        flag[0] = solver(
            state,
            params,
            drivers,
            time_scalar,
            h,
            a_ij,
            base,
            shared,
            counters,
        )

    h = precision(0.01)
    x = cuda.to_device(np.array([0.0], dtype=precision))  # ensures residual>tol
    out_flag = cuda.to_device(np.array([0], dtype=np.int32))
    kernel[1, 1](x, base_state, out_flag, h)
    status_code = int(out_flag.copy_to_host()[0]) & STATUS_MASK
    assert status_code == SolverRetCodes.MAX_NEWTON_ITERATIONS_EXCEEDED


def test_newton_krylov_linear_solver_failure_propagates(precision):
    """Newton-Krylov returns MAX_LINEAR_ITERATIONS_EXCEEDED when inner solver fails."""

    @cuda.jit(device=True)
    def residual(state, parameters, drivers, t, h, a_ij, base_state, out):
        # Simple residual: nonzero so that a linear solve is attempted
        out[0] = precision(1.0)

    @cuda.jit(device=True)
    def zero_operator(state, parameters, drivers, base_state, t, h, a_ij, vec, out):
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

    scratch_len = 3 * n

    @cuda.jit
    def kernel(flag, h):
        state = cuda.local.array(1, precision)
        params = cuda.local.array(1, precision)
        drivers = cuda.local.array(1, precision)
        counters = cuda.local.array(2, np.int32)
        a_ij = precision(1.0)
        base = cuda.local.array(1, precision)
        shared = cuda.shared.array(scratch_len, precision)
        time_scalar = precision(0.0)
        flag[0] = solver(
            state,
            params,
            drivers,
            time_scalar,
            h,
            a_ij,
            base,
            shared,
            counters,
        )

    out_flag = cuda.to_device(np.array([0], dtype=np.int32))
    kernel[1, 1](out_flag, precision(0.01))
    status_code = int(out_flag.copy_to_host()[0]) & STATUS_MASK
    assert (
        status_code == SolverRetCodes.MAX_LINEAR_ITERATIONS_EXCEEDED
    )
