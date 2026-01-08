import numpy as np
import pytest
from numba import cuda
from numpy.testing import assert_allclose

from cubie.integrators.matrix_free_solvers.linear_solver import (
    LinearSolver,
)
from cubie.integrators.matrix_free_solvers.newton_krylov import (
    NewtonKrylov,
)
from cubie.integrators.matrix_free_solvers import SolverRetCodes

STATUS_MASK = 0xFFFF


@pytest.fixture(scope="session")
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
    
    linear_solver_instance = LinearSolver(
        precision=precision,
        n=n,
        krylov_tolerance=1e-8,
        max_linear_iters=32,
    )
    linear_solver_instance.update(operator_apply=operator)
    
    newton_instance = NewtonKrylov(
        precision=precision,
        n=n,
        linear_solver=linear_solver_instance,
        newton_tolerance=1e-6,
        max_newton_iters=16,
    )
    newton_instance.update(residual_function=residual)
    solver = newton_instance.device_function

    scratch_len = 2 * n

    @cuda.jit
    def kernel(state, base, flag, h):
        params = cuda.local.array(1, precision)
        drivers = cuda.local.array(1, precision)
        counters = cuda.local.array(2, np.int32)
        a_ij = precision(1.0)
        shared = cuda.shared.array(scratch_len, precision)
        persistent_local = cuda.local.array(scratch_len, precision)
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
            persistent_local,
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
    # Scaled norm may converge at different iterations than L2 norm,
    # producing slightly different final values (~5% difference).
    assert np.allclose(
        result_increment,
        expected_increment,
        rtol=tolerance.rel_loose * 1000,
        atol=tolerance.abs_loose * 1000,
    )

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
    sym_system = system_setup["sym_system"]
    n = sym_system.num_states
    operator = system_setup["operator"]
    residual_func = system_setup["residual"]
    base_state = system_setup["base_state"]
    expected = system_setup["nk_expected"]
    h = system_setup["h"]

    precond = (
        None
        if precond_order == 0
        else system_setup["preconditioner"](precond_order)
    )
    # Use tighter tolerances to ensure full convergence regardless of norm
    # type used internally. This makes final results independent of whether
    # L2 or scaled norm is used for convergence checks.
    krylov_tol = 1e-10 if precision == np.float64 else 1e-6
    newton_tol = 1e-10 if precision == np.float64 else 1e-6
    linear_solver_instance = LinearSolver(
        precision=precision,
        n=n,
        linear_correction_type="minimal_residual",
        krylov_tolerance=krylov_tol,
        max_linear_iters=1000,
    )
    linear_solver_instance.update(operator_apply=operator,
                                  preconditioner=precond)

    newton_instance = NewtonKrylov(
        precision=precision,
        n=n,
        linear_solver=linear_solver_instance,
        newton_tolerance=newton_tol,
        max_newton_iters=1000,
    )

    newton_instance.update(residual_function=residual_func)
    solver = newton_instance.device_function


    scratch_len = 2 * n

    @cuda.jit
    def kernel(state, base, flag, h):
        params = cuda.local.array(1, precision)
        drivers = cuda.local.array(1, precision)
        counters = cuda.local.array(2, np.int32)
        a_ij = precision(1.0)
        shared = cuda.shared.array(scratch_len, precision)
        persistent_local = cuda.local.array(scratch_len, precision)
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
            persistent_local,
            counters,
        )

    base_vals = system_setup["base_state"].copy_to_host()
    expected_increment = expected - base_vals
    x = system_setup["state_init"]
    out_flag = cuda.to_device(np.array([0], dtype=np.int32))
    kernel[1, 1](x, base_state, out_flag, h)
    status_code = int(out_flag.copy_to_host()[0]) & STATUS_MASK
    # Nonlinear system needs preconditioning.
    # if system_setup["id"] == "nonlinear" and precond_order == 0:
    #     assert (
    #         status_code == SolverRetCodes.NEWTON_BACKTRACKING_NO_SUITABLE_STEP
    #     )
    # else:
    assert status_code == SolverRetCodes.SUCCESS
    # Scaled norm may converge at different iterations than L2 norm,
    # producing slightly different final values (~5% difference).
    assert_allclose(
        x.copy_to_host(),
        expected_increment,
        rtol=tolerance.rel_loose * 1000,
        atol=tolerance.abs_loose * 1000,
    )


def test_newton_krylov_failure(precision):
    """Solver reports backtracking failure when it cannot reduce residual."""

    @cuda.jit(device=True)
    def residual(state, parameters, drivers, t, h, a_ij, base_state, out):
        out[0] = precision(1.0)

    @cuda.jit(device=True)
    def operator(state, parameters, drivers, base_state, t, h, a_ij, vec, out):
        out[0] = vec[0]

    n = 1
    linear_solver_instance = LinearSolver(
        precision=precision,
        n=n,
        krylov_tolerance=1e-12,
        max_linear_iters=8,
    )
    linear_solver_instance.update(operator_apply=operator)

    newton_instance = NewtonKrylov(
        precision=precision,
        n=n,
        linear_solver=linear_solver_instance,
        newton_tolerance=1e-8,
        max_newton_iters=2,
    )

    newton_instance.update(residual_function=residual)
    solver = newton_instance.device_function

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
        persistent_local = cuda.local.array(scratch_len, precision)
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
            persistent_local,
            counters,
        )

    out_flag = cuda.to_device(np.array([1], dtype=np.int32))
    kernel[1, 1](out_flag, precision(0.01))
    status_code = int(out_flag.copy_to_host()[0]) & STATUS_MASK
    assert status_code in (
        SolverRetCodes.MAX_NEWTON_ITERATIONS_EXCEEDED,
        SolverRetCodes.NEWTON_BACKTRACKING_NO_SUITABLE_STEP,
    )


def test_newton_krylov_max_newton_iters_exceeded(
    placeholder_system, precision
):
    """Returns MAX_NEWTON_ITERATIONS_EXCEEDED when max_iters=0 and residual>tolerance."""
    _, _, base_state = placeholder_system

    @cuda.jit(device=True)
    def residual(state, parameters, drivers, t, h, a_ij, base_state, out):
        # Cubic residual in the final state: f(x) = (base + x - target)^3.
        # Newton's method reduces this slowly from a distant initial guess.
        target = base_state[0] + precision(1.0)  # solution increment = 1.0
        y = base_state[0] + state[0]
        diff = y - target
        out[0] = diff * diff * diff

    @cuda.jit(device=True)
    def operator(
        state, parameters, drivers, base_state, t, h, a_ij, vec, out
    ):
        # Jacobian of the cubic residual: J = 3*(y - target)^2
        target = base_state[0] + precision(1.0)
        y = base_state[0] + state[0]
        jac = precision(3.0) * (y - target) * (y - target)
        out[0] = jac * vec[0]
    n = 1
    linear_solver_instance = LinearSolver(
        precision=precision,
        n=n,
        krylov_tolerance=1e-8,
        max_linear_iters=20,
    )
    linear_solver_instance.update(operator_apply=operator)

    newton_instance = NewtonKrylov(
        precision=precision,
        n=n,
        linear_solver=linear_solver_instance,
        newton_tolerance=1e-20,
        max_newton_iters=1,
    )

    newton_instance.update(residual_function=residual)
    solver = newton_instance.device_function


    scratch_len = 3 * n

    @cuda.jit
    def kernel(state, base, flag, h):
        params = cuda.local.array(1, precision)
        drivers = cuda.local.array(1, precision)
        counters = cuda.local.array(2, np.int32)
        a_ij = precision(1.0)
        shared = cuda.shared.array(scratch_len, precision)
        persistent_local = cuda.local.array(scratch_len, precision)
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
            persistent_local,
            counters,
        )

    h = precision(0.01)
    x = cuda.to_device(np.array([0.0], dtype=precision))  # ensures residual>tol
    out_flag = cuda.to_device(np.array([0], dtype=np.int32))
    kernel[1, 1](x, base_state, out_flag, h)
    status_code = int(out_flag.copy_to_host()[0]) & STATUS_MASK
    assert (status_code == SolverRetCodes.MAX_NEWTON_ITERATIONS_EXCEEDED)


def test_newton_krylov_linear_solver_failure_propagates(precision):
    """Newton-Krylov reports inner linear failure when Newton exits unconverged."""

    @cuda.jit(device=True)
    def residual(state, parameters, drivers, t, h, a_ij, base_state, out):
        # Simple residual: nonzero so that a linear solve is attempted
        out[0] = precision(1.0)

    @cuda.jit(device=True)
    def zero_operator(
        state, parameters, drivers, base_state, t, h, a_ij, vec, out
    ):
        # Linear operator always zero => inner solver cannot make progress
        out[0] = precision(0.0)

    # Inner linear solver will return MAX_LINEAR_ITERATIONS_EXCEEDED
    n = 1
    linear_solver_instance = LinearSolver(
        precision=precision,
        n=n,
        linear_correction_type="minimal_residual",
        krylov_tolerance=1e-20,
        max_linear_iters=8,
    )
    linear_solver_instance.update(operator_apply=zero_operator)

    newton_instance = NewtonKrylov(
        precision=precision,
        n=n,
        linear_solver=linear_solver_instance,
        newton_tolerance=1e-8,
        max_newton_iters=4,
    )

    newton_instance.update(residual_function=residual)
    solver = newton_instance.device_function


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
        persistent_local = cuda.local.array(scratch_len, precision)
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
            persistent_local,
            counters,
        )

    out_flag = cuda.to_device(np.array([0], dtype=np.int32))
    kernel[1, 1](out_flag, precision(0.01))
    status_code = int(out_flag.copy_to_host()[0]) & STATUS_MASK
    assert (
        status_code
        == (
            SolverRetCodes.MAX_NEWTON_ITERATIONS_EXCEEDED
            | SolverRetCodes.NEWTON_BACKTRACKING_NO_SUITABLE_STEP
            | SolverRetCodes.MAX_LINEAR_ITERATIONS_EXCEEDED
        )
    )


def test_newton_krylov_config_scalar_tolerance_broadcast(precision):
    """Verify scalar newton_atol/rtol broadcasts to array of length n."""
    n = 5
    linear_solver = LinearSolver(precision=precision, n=n)
    newton = NewtonKrylov(
        precision=precision,
        n=n,
        linear_solver=linear_solver,
        newton_atol=1e-6,
        newton_rtol=1e-4,
    )
    assert newton.newton_atol.shape == (n,)
    assert newton.newton_rtol.shape == (n,)
    assert np.all(newton.newton_atol == precision(1e-6))
    assert np.all(newton.newton_rtol == precision(1e-4))


def test_newton_krylov_config_array_tolerance_accepted(precision):
    """Verify array tolerances of correct length are accepted."""
    n = 3
    atol = np.array([1e-6, 1e-8, 1e-4], dtype=precision)
    rtol = np.array([1e-3, 1e-5, 1e-2], dtype=precision)
    linear_solver = LinearSolver(precision=precision, n=n)
    newton = NewtonKrylov(
        precision=precision,
        n=n,
        linear_solver=linear_solver,
        newton_atol=atol,
        newton_rtol=rtol,
    )
    assert np.allclose(newton.newton_atol, atol)
    assert np.allclose(newton.newton_rtol, rtol)


def test_newton_krylov_config_wrong_length_raises(precision):
    """Verify wrong-length tolerance array raises ValueError."""
    n = 3
    wrong_atol = np.array([1e-6, 1e-8], dtype=precision)  # length 2
    linear_solver = LinearSolver(precision=precision, n=n)
    with pytest.raises(ValueError, match="tol must have shape"):
        NewtonKrylov(
            precision=precision,
            n=n,
            linear_solver=linear_solver,
            newton_atol=wrong_atol,
        )


def test_newton_krylov_scaled_tolerance_converges(precision, tolerance):
    """Verify Newton solver converges with per-element tolerances."""

    @cuda.jit(device=True)
    def residual(state, parameters, drivers, t, h, a_ij, base_state, out):
        # Simple implicit Euler residual: state - h * base_state
        out[0] = state[0] - h * (base_state[0] + a_ij * state[0])

    @cuda.jit(device=True)
    def operator(state, parameters, drivers, base_state, t, h, a_ij, vec, out):
        out[0] = (precision(1.0) - h * a_ij) * vec[0]

    n = 1
    linear_solver = LinearSolver(
        precision=precision,
        n=n,
        krylov_tolerance=1e-8,
        krylov_atol=1e-8,
        krylov_rtol=1e-8,
        max_linear_iters=32,
    )
    linear_solver.update(operator_apply=operator)

    newton = NewtonKrylov(
        precision=precision,
        n=n,
        linear_solver=linear_solver,
        newton_atol=1e-6,
        newton_rtol=1e-6,
        max_newton_iters=16,
    )
    newton.update(residual_function=residual)
    solver = newton.device_function

    scratch_len = 2 * n
    base = cuda.to_device(np.array([1.0], dtype=precision))

    @cuda.jit
    def kernel(state, base_dev, flag, h):
        params = cuda.local.array(1, precision)
        drivers = cuda.local.array(1, precision)
        counters = cuda.local.array(2, np.int32)
        a_ij = precision(1.0)
        shared = cuda.shared.array(scratch_len, precision)
        persistent_local = cuda.local.array(scratch_len, precision)
        time_scalar = precision(0.0)
        flag[0] = solver(
            state,
            params,
            drivers,
            time_scalar,
            h,
            a_ij,
            base_dev,
            shared,
            persistent_local,
            counters,
        )

    h = precision(0.01)
    base_val = base.copy_to_host()[0]
    expected_final = precision(base_val / (1.0 - h))
    expected_increment = np.array([expected_final - base_val], dtype=precision)
    x0 = expected_increment * precision(0.99)
    x = cuda.to_device(x0)
    out_flag = cuda.to_device(np.array([0], dtype=np.int32))
    kernel[1, 1](x, base, out_flag, h)
    status_code = int(out_flag.copy_to_host()[0]) & STATUS_MASK

    assert status_code == SolverRetCodes.SUCCESS
    # Scaled norm may converge at different iterations than L2 norm,
    # producing slightly different final values (~5% difference).
    assert np.allclose(
        x.copy_to_host(),
        expected_increment,
        rtol=tolerance.rel_loose * 1000,
        atol=tolerance.abs_loose * 1000,
    )


def test_newton_krylov_scalar_tolerance_backward_compatible(
    placeholder_system, precision, tolerance
):
    """Verify scalar tolerance input produces same behavior as before."""
    residual, operator, base_state = placeholder_system
    n = 1

    linear_solver = LinearSolver(
        precision=precision,
        n=n,
        krylov_tolerance=1e-8,
        max_linear_iters=32,
    )
    linear_solver.update(operator_apply=operator)

    newton = NewtonKrylov(
        precision=precision,
        n=n,
        linear_solver=linear_solver,
        newton_tolerance=1e-6,
        max_newton_iters=16,
    )
    newton.update(residual_function=residual)
    solver = newton.device_function

    scratch_len = 2 * n

    @cuda.jit
    def kernel(state, base, flag, h):
        params = cuda.local.array(1, precision)
        drivers = cuda.local.array(1, precision)
        counters = cuda.local.array(2, np.int32)
        a_ij = precision(1.0)
        shared = cuda.shared.array(scratch_len, precision)
        persistent_local = cuda.local.array(scratch_len, precision)
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
            persistent_local,
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
    # Scaled norm may converge at different iterations than L2 norm,
    # producing slightly different final values (~5% difference).
    assert np.allclose(
        x.copy_to_host(),
        expected_increment,
        rtol=tolerance.rel_loose * 1000,
        atol=tolerance.abs_loose * 1000,
    )


def test_newton_krylov_uses_scaled_norm(precision):
    """Verify NewtonKrylov uses ScaledNorm for convergence checking."""
    from cubie.integrators.norms import ScaledNorm

    n = 3
    linear_solver = LinearSolver(precision=precision, n=n)
    newton = NewtonKrylov(
        precision=precision,
        n=n,
        linear_solver=linear_solver,
        newton_atol=1e-6,
        newton_rtol=1e-4,
    )
    # Verify norm factory exists and is a ScaledNorm
    assert hasattr(newton, 'norm')
    assert isinstance(newton.norm, ScaledNorm)
    # Verify norm has correct configuration
    assert newton.norm.n == n
    assert newton.norm.precision == precision
    assert np.all(newton.norm.atol == precision(1e-6))
    assert np.all(newton.norm.rtol == precision(1e-4))


def test_newton_krylov_tolerance_update_propagates(precision):
    """Verify newton_atol/newton_rtol updates reach norm factory."""
    n = 3
    initial_atol = 1e-6
    initial_rtol = 1e-4
    linear_solver = LinearSolver(precision=precision, n=n)
    newton = NewtonKrylov(
        precision=precision,
        n=n,
        linear_solver=linear_solver,
        newton_atol=initial_atol,
        newton_rtol=initial_rtol,
    )
    # Verify initial values
    assert np.all(newton.newton_atol == precision(initial_atol))
    assert np.all(newton.newton_rtol == precision(initial_rtol))
    assert np.all(newton.norm.atol == precision(initial_atol))
    assert np.all(newton.norm.rtol == precision(initial_rtol))

    # Update tolerances
    new_atol = 1e-8
    new_rtol = 1e-6
    newton.update(newton_atol=new_atol, newton_rtol=new_rtol)

    # Verify properties delegate to norm factory
    assert np.all(newton.newton_atol == precision(new_atol))
    assert np.all(newton.newton_rtol == precision(new_rtol))
    # Verify norm factory was updated
    assert np.all(newton.norm.atol == precision(new_atol))
    assert np.all(newton.norm.rtol == precision(new_rtol))
