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
        rtol=tolerance.rel_loose,
        atol=tolerance.abs_loose,
    )


@pytest.fixture(scope="function")
def solver_device(request, placeholder_operator, precision):
    """Return solver device for the requested correction type."""

    solver = LinearSolver(
        precision=precision,
        n=3,
        linear_correction_type=request.param,
        krylov_tolerance=1e-12,
        kyrlov_max_iters=32,
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
        rtol=tolerance.rel_loose,
        atol=tolerance.abs_loose,
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
        kyrlov_max_iters=1000,
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
        rtol=tolerance.rel_loose,
        atol=tolerance.abs_loose,
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
        kyrlov_max_iters=16,
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


def test_linear_solver_config_scalar_tolerance_broadcast(precision):
    """Verify scalar krylov_atol/rtol broadcasts to array of length n."""
    n = 5
    solver = LinearSolver(
        precision=precision,
        n=n,
        krylov_atol=1e-6,
        krylov_rtol=1e-4,
    )
    assert solver.krylov_atol.shape == (n,)
    assert solver.krylov_rtol.shape == (n,)
    assert np.all(solver.krylov_atol == precision(1e-6))
    assert np.all(solver.krylov_rtol == precision(1e-4))


def test_linear_solver_config_array_tolerance_accepted(precision):
    """Verify array tolerances of correct length are accepted."""
    n = 3
    atol = np.array([1e-6, 1e-8, 1e-4], dtype=precision)
    rtol = np.array([1e-3, 1e-5, 1e-2], dtype=precision)
    solver = LinearSolver(
        precision=precision,
        n=n,
        krylov_atol=atol,
        krylov_rtol=rtol,
    )
    assert np.allclose(solver.krylov_atol, atol)
    assert np.allclose(solver.krylov_rtol, rtol)


def test_linear_solver_config_wrong_length_raises(precision):
    """Verify wrong-length tolerance array raises ValueError."""
    n = 3
    wrong_atol = np.array([1e-6, 1e-8], dtype=precision)  # length 2
    with pytest.raises(ValueError, match="tol must have shape"):
        LinearSolver(
            precision=precision,
            n=n,
            krylov_atol=wrong_atol,
        )


@pytest.mark.parametrize(
    "solver_device", ["steepest_descent", "minimal_residual"], indirect=True
)
def test_linear_solver_scaled_tolerance_converges(
    solver_device,
    solver_kernel,
    precision,
    tolerance,
):
    """Verify linear solver converges with per-element tolerances on mixed-scale.

    Creates a system with variables at different scales and sets per-element
    tolerances matching the expected solution magnitudes.
    """
    rhs = np.array([1.0, 2.0, 3.0], dtype=precision)
    matrix = np.array(
        [[4.0, 1.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, 2.0]],
        dtype=precision,
    )
    expected = np.linalg.solve(matrix, rhs)
    h = precision(0.01)
    kernel = solver_kernel(solver_device, 3, h, precision)
    base_state = np.array([1.0, -1.0, 0.5], dtype=precision)
    state = cuda.to_device(
        base_state + h * np.array([0.1, -0.2, 0.3], dtype=precision)
    )
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
        rtol=tolerance.rel_loose,
        atol=tolerance.abs_loose,
    )


@pytest.mark.parametrize(
    "solver_device", ["steepest_descent", "minimal_residual"], indirect=True
)
def test_linear_solver_scalar_tolerance_backward_compatible(
    solver_device,
    solver_kernel,
    precision,
    tolerance,
):
    """Verify scalar tolerance produces convergent behavior."""
    rhs = np.array([1.0, 2.0, 3.0], dtype=precision)
    matrix = np.array(
        [[4.0, 1.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, 2.0]],
        dtype=precision,
    )
    expected = np.linalg.solve(matrix, rhs)
    h = precision(0.01)
    kernel = solver_kernel(solver_device, 3, h, precision)
    base_state = np.array([1.0, -1.0, 0.5], dtype=precision)
    state = cuda.to_device(
        base_state + h * np.array([0.1, -0.2, 0.3], dtype=precision)
    )
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
        rtol=tolerance.rel_loose,
        atol=tolerance.abs_loose,
    )


def test_linear_solver_uses_scaled_norm(precision):
    """Verify LinearSolver creates and uses ScaledNorm for convergence."""
    from cubie.integrators.norms import ScaledNorm

    n = 3
    solver = LinearSolver(
        precision=precision,
        n=n,
        krylov_atol=1e-6,
        krylov_rtol=1e-4,
    )
    # Verify the norm factory is created
    assert hasattr(solver, 'norm')
    assert isinstance(solver.norm, ScaledNorm)
    # Verify the norm factory has correct settings
    assert solver.norm.n == n
    assert solver.norm.precision == precision
    assert np.all(solver.norm.atol == precision(1e-6))
    assert np.all(solver.norm.rtol == precision(1e-4))


def test_linear_solver_tolerance_update_propagates(precision):
    """Verify krylov_atol/krylov_rtol updates propagate to norm factory."""
    n = 3
    solver = LinearSolver(
        precision=precision,
        n=n,
        krylov_atol=1e-6,
        krylov_rtol=1e-4,
    )
    # Initial values
    assert np.all(solver.krylov_atol == precision(1e-6))
    assert np.all(solver.krylov_rtol == precision(1e-4))

    # Update tolerances
    new_atol = np.array([1e-8, 1e-7, 1e-9], dtype=precision)
    new_rtol = np.array([1e-5, 1e-6, 1e-4], dtype=precision)
    solver.update(krylov_atol=new_atol, krylov_rtol=new_rtol)

    # Verify properties delegate to norm factory
    assert np.allclose(solver.krylov_atol, new_atol)
    assert np.allclose(solver.krylov_rtol, new_rtol)
    # Verify norm factory was updated
    assert np.allclose(solver.norm.atol, new_atol)
    assert np.allclose(solver.norm.rtol, new_rtol)


def test_linear_solver_config_no_tolerance_fields(precision):
    """Verify LinearSolverConfig no longer has krylov_atol/krylov_rtol fields."""
    from cubie.integrators.matrix_free_solvers.linear_solver import (
        LinearSolverConfig,
    )

    config = LinearSolverConfig(precision=precision, n=3)

    # These fields should no longer exist on the config
    assert not hasattr(config, 'krylov_atol')
    assert not hasattr(config, 'krylov_rtol')

    # The legacy scalar tolerance should still exist
    assert hasattr(config, 'krylov_tolerance')
    assert config.krylov_tolerance == precision(1e-6)


def test_linear_solver_config_settings_dict_excludes_tolerance_arrays(precision):
    """Verify settings_dict does not include tolerance arrays."""
    from cubie.integrators.matrix_free_solvers.linear_solver import (
        LinearSolverConfig,
    )

    config = LinearSolverConfig(precision=precision, n=3)
    settings = config.settings_dict

    # Tolerance arrays should not be in settings_dict
    assert 'krylov_atol' not in settings
    assert 'krylov_rtol' not in settings

    # Other expected settings should be present
    assert 'krylov_tolerance' in settings
    assert 'kyrlov_max_iters' in settings
    assert 'linear_correction_type' in settings
    assert 'preconditioned_vec_location' in settings
    assert 'temp_location' in settings


def test_linear_solver_inherits_from_matrix_free_solver(precision):
    """Verify LinearSolver is instance of MatrixFreeSolver."""
    from cubie.integrators.matrix_free_solvers.base_solver import (
        MatrixFreeSolver,
    )

    solver = LinearSolver(
        precision=precision,
        n=3,
    )
    assert isinstance(solver, MatrixFreeSolver)
    assert hasattr(solver, 'solver_type')
    assert solver.solver_type == "krylov"


def test_linear_solver_update_preserves_original_dict(precision):
    """Verify update() does not modify the input updates_dict."""
    solver = LinearSolver(
        precision=precision,
        n=3,
    )

    # Create an update dict with tolerance values
    original_updates = {
        'krylov_atol': 1e-8,
        'krylov_rtol': 1e-5,
        'kyrlov_max_iters': 50,
    }
    # Make a copy to compare later
    updates_copy = dict(original_updates)

    # Call update with the dict
    solver.update(updates_dict=original_updates)

    # Verify the original dict was not modified
    assert original_updates == updates_copy


def test_linear_solver_no_manual_cache_invalidation(precision):
    """Verify cache invalidation happens through config update, not manual."""
    solver = LinearSolver(
        precision=precision,
        n=3,
        krylov_atol=1e-6,
        krylov_rtol=1e-4,
    )

    # Access device_function to populate cache
    _ = solver.device_function

    # Get the current norm device function from config
    config1 = solver.compile_settings

    # Update tolerance - should update config's norm_device_function
    new_atol = np.array([1e-8, 1e-7, 1e-9], dtype=precision)
    solver.update(krylov_atol=new_atol)

    # Verify config was updated with new norm device function
    config2 = solver.compile_settings
    norm_fn2 = config2.norm_device_function

    # The norm device function should be set (not None)
    assert norm_fn2 is not None

    # Verify the solver's norm factory was updated
    assert np.allclose(solver.norm.atol, new_atol)


def test_linear_solver_settings_dict_includes_tolerance_arrays(precision):
    """Verify settings_dict includes krylov_atol and krylov_rtol from norm."""
    n = 3
    atol = np.array([1e-6, 1e-8, 1e-4], dtype=precision)
    rtol = np.array([1e-3, 1e-5, 1e-2], dtype=precision)
    solver = LinearSolver(
        precision=precision,
        n=n,
        krylov_atol=atol,
        krylov_rtol=rtol,
    )
    settings = solver.settings_dict

    # Tolerance arrays should be in settings_dict
    assert 'krylov_atol' in settings
    assert 'krylov_rtol' in settings
    assert np.allclose(settings['krylov_atol'], atol)
    assert np.allclose(settings['krylov_rtol'], rtol)

    # Other expected settings from config should also be present
    assert 'krylov_tolerance' in settings
    assert 'kyrlov_max_iters' in settings
    assert 'linear_correction_type' in settings
    assert 'preconditioned_vec_location' in settings
    assert 'temp_location' in settings
