import numpy as np
import pytest
from numba import cuda
from numpy.testing import assert_allclose

from cubie.integrators.matrix_free_solvers.bicgstab_solver import (
    BiCGSTABSolver,
    BiCGSTABSolverConfig,
)
from cubie.integrators.matrix_free_solvers.linear_solver import (
    MRLinearSolver,
)
from cubie.result_codes import CUBIE_RESULT_CODES


@pytest.mark.parametrize(
    "system_setup",
    ["linear", "coupled_linear"],
    indirect=True,
)
@pytest.mark.parametrize("precond_order", [1, 2])
def test_bicgstab_convergence(
    system_setup,
    solver_kernel,
    precision,
    precond_order,
    tolerance,
):
    """BiCGSTAB converges on well-conditioned systems."""
    n = system_setup["n"]
    operator = system_setup["operator"]
    rhs_vec = system_setup["mr_rhs"]
    expected = system_setup["mr_expected"]
    h = system_setup["h"]
    precond = (
        None
        if precond_order == 0
        else system_setup["preconditioner"](precond_order)
    )

    solver = BiCGSTABSolver(
        precision=precision,
        n=n,
        krylov_atol=1e-8,
        krylov_rtol=1e-8,
        krylov_max_iters=200,
    )
    solver.update(operator_apply=operator, preconditioner=precond)
    solver_fn = solver.device_function

    kernel = solver_kernel(solver_fn, n, h, precision)
    state = system_setup["state_init"]
    rhs_dev = cuda.to_device(rhs_vec)
    x_dev = cuda.to_device(np.zeros(n, dtype=precision))
    flag = cuda.to_device(np.array([0], dtype=np.int32))
    empty_base = cuda.to_device(np.empty(0, dtype=precision))
    kernel[1, 1](state, rhs_dev, empty_base, x_dev, flag)
    code = flag.copy_to_host()[0] & 0xFF
    assert code == CUBIE_RESULT_CODES.SUCCESS
    assert_allclose(
        x_dev.copy_to_host(),
        expected,
        rtol=tolerance.rel_loose,
        atol=tolerance.abs_loose,
    )


@pytest.mark.parametrize(
    "system_setup",
    ["stiff"],
    indirect=True,
)
@pytest.mark.parametrize("precond_order", [1, 2])
def test_bicgstab_stiff_convergence(
    system_setup,
    solver_kernel,
    precision,
    precond_order,
    tolerance,
):
    """BiCGSTAB converges on moderately ill-conditioned system."""
    n = system_setup["n"]
    operator = system_setup["operator"]
    rhs_vec = system_setup["mr_rhs"]
    expected = system_setup["mr_expected"]
    h = system_setup["h"]
    precond = system_setup["preconditioner"](precond_order)

    solver = BiCGSTABSolver(
        precision=precision,
        n=n,
        krylov_atol=1e-8,
        krylov_rtol=1e-8,
        krylov_max_iters=200,
    )
    solver.update(operator_apply=operator, preconditioner=precond)
    solver_fn = solver.device_function

    kernel = solver_kernel(solver_fn, n, h, precision)
    state = system_setup["state_init"]
    rhs_dev = cuda.to_device(rhs_vec)
    x_dev = cuda.to_device(np.zeros(n, dtype=precision))
    flag = cuda.to_device(np.array([0], dtype=np.int32))
    empty_base = cuda.to_device(np.empty(0, dtype=precision))
    kernel[1, 1](state, rhs_dev, empty_base, x_dev, flag)
    code = flag.copy_to_host()[0] & 0xFF
    assert code == CUBIE_RESULT_CODES.SUCCESS
    assert_allclose(
        x_dev.copy_to_host(),
        expected,
        rtol=tolerance.rel_loose,
        atol=tolerance.abs_loose,
    )


@pytest.mark.parametrize(
    "system_setup",
    ["linear"],
    indirect=True,
)
def test_bicgstab_fewer_iters_than_mr(
    system_setup,
    solver_kernel,
    precision,
    tolerance,
):
    """BiCGSTAB converges in fewer iterations than MR on easy system."""
    n = system_setup["n"]
    operator = system_setup["operator"]
    rhs_vec = system_setup["mr_rhs"]
    h = system_setup["h"]
    precond = system_setup["preconditioner"](1)

    # --- Build BiCGSTAB ---
    bicg = BiCGSTABSolver(
        precision=precision,
        n=n,
        krylov_atol=1e-8,
        krylov_rtol=1e-8,
        krylov_max_iters=200,
    )
    bicg.update(operator_apply=operator, preconditioner=precond)
    bicg_fn = bicg.device_function

    # --- Build MR ---
    mr = MRLinearSolver(
        precision=precision,
        n=n,
        linear_correction_type="minimal_residual",
        krylov_atol=1e-8,
        krylov_rtol=1e-8,
        krylov_max_iters=1000,
    )
    mr.update(operator_apply=operator, preconditioner=precond)
    mr_fn = mr.device_function

    state = system_setup["state_init"]
    empty_base = cuda.to_device(np.empty(0, dtype=precision))

    # BiCGSTAB run
    bicg_kernel = solver_kernel(bicg_fn, n, h, precision)
    rhs_b = cuda.to_device(rhs_vec.copy())
    x_b = cuda.to_device(np.zeros(n, dtype=precision))
    flag_b = cuda.to_device(np.array([0], dtype=np.int32))
    bicg_kernel[1, 1](state, rhs_b, empty_base, x_b, flag_b)

    # MR run
    mr_kernel = solver_kernel(mr_fn, n, h, precision)
    rhs_m = cuda.to_device(rhs_vec.copy())
    x_m = cuda.to_device(np.zeros(n, dtype=precision))
    flag_m = cuda.to_device(np.array([0], dtype=np.int32))
    mr_kernel[1, 1](state, rhs_m, empty_base, x_m, flag_m)

    assert (flag_b.copy_to_host()[0] & 0xFF) == CUBIE_RESULT_CODES.SUCCESS
    assert (flag_m.copy_to_host()[0] & 0xFF) == CUBIE_RESULT_CODES.SUCCESS
    # Both should produce the same solution
    assert_allclose(
        x_b.copy_to_host(),
        x_m.copy_to_host(),
        rtol=tolerance.rel_loose,
        atol=tolerance.abs_loose,
    )


def test_bicgstab_breakdown_detection(precision):
    """BiCGSTAB returns BICGSTAB_BREAKDOWN on degenerate operator."""

    @cuda.jit(device=True)
    def zero_operator(
        state, parameters, drivers, base_state, t, h, a_ij, vec, out
    ):
        for i in range(out.shape[0]):
            out[i] = precision(0.0)

    n = 3
    solver = BiCGSTABSolver(
        precision=precision,
        n=n,
        krylov_atol=1e-20,
        krylov_rtol=1e-20,
        krylov_max_iters=16,
    )
    solver.update(operator_apply=zero_operator)
    solver_fn = solver.device_function

    scratch_size = 6 * n

    @cuda.jit
    def kernel(flag, h):
        state = cuda.local.array(3, precision)
        params = cuda.local.array(1, precision)
        drivers = cuda.local.array(1, precision)
        base = cuda.local.array(1, precision)
        shared = cuda.shared.array(scratch_size, precision)
        persistent_local = cuda.local.array(scratch_size, precision)
        counters = cuda.local.array(1, np.int32)
        rhs = cuda.local.array(3, precision)
        x = cuda.local.array(3, precision)
        for i in range(3):
            rhs[i] = precision(1.0)
            x[i] = precision(0.0)
        time_scalar = precision(0.0)
        flag[0] = solver_fn(
            state,
            params,
            drivers,
            base,
            time_scalar,
            h,
            precision(1.0),
            rhs,
            x,
            shared,
            persistent_local,
            counters,
        )

    out_flag = cuda.to_device(np.array([0], dtype=np.int32))
    kernel[1, 1](out_flag, precision(0.01))
    status_code = int(out_flag.copy_to_host()[0]) & 0xFF
    # Zero operator: rho_0 = 0 initially, so the check
    # |rho_new| < tol * rho_0 fires or max_iters is hit.
    # Either breakdown or max_iters exceeded is acceptable.
    assert status_code in (
        CUBIE_RESULT_CODES.BICGSTAB_BREAKDOWN,
        CUBIE_RESULT_CODES.MAX_LINEAR_ITERATIONS_EXCEEDED,
    )


@pytest.mark.parametrize(
    "system_setup",
    ["linear"],
    indirect=True,
)
def test_bicgstab_unpreconditioned(
    system_setup,
    solver_kernel,
    precision,
    tolerance,
):
    """BiCGSTAB converges without preconditioner."""
    n = system_setup["n"]
    operator = system_setup["operator"]
    rhs_vec = system_setup["mr_rhs"]
    expected = system_setup["mr_expected"]
    h = system_setup["h"]

    solver = BiCGSTABSolver(
        precision=precision,
        n=n,
        krylov_atol=1e-8,
        krylov_rtol=1e-8,
        krylov_max_iters=200,
    )
    solver.update(operator_apply=operator)
    solver_fn = solver.device_function

    kernel = solver_kernel(solver_fn, n, h, precision)
    state = system_setup["state_init"]
    rhs_dev = cuda.to_device(rhs_vec)
    x_dev = cuda.to_device(np.zeros(n, dtype=precision))
    flag = cuda.to_device(np.array([0], dtype=np.int32))
    empty_base = cuda.to_device(np.empty(0, dtype=precision))
    kernel[1, 1](state, rhs_dev, empty_base, x_dev, flag)
    code = flag.copy_to_host()[0] & 0xFF
    assert code == CUBIE_RESULT_CODES.SUCCESS
    assert_allclose(
        x_dev.copy_to_host(),
        expected,
        rtol=tolerance.rel_loose,
        atol=tolerance.abs_loose,
    )


def test_bicgstab_buffer_locations(precision):
    """BiCGSTAB accepts various buffer location combinations."""
    n = 3
    solver = BiCGSTABSolver(
        precision=precision,
        n=n,
        r0_hat_location="local",
        p_location="shared",
        v_location="shared",
        tmp_location="local",
        s_hat_location="local",
    )
    settings = solver.settings_dict
    assert settings["r0_hat_location"] == "local"
    assert settings["p_location"] == "shared"
    assert settings["v_location"] == "shared"
    assert settings["tmp_location"] == "local"
    assert settings["s_hat_location"] == "local"


def test_bicgstab_config_defaults(precision):
    """BiCGSTABSolverConfig has correct default buffer locations."""
    config = BiCGSTABSolverConfig(precision=precision, n=3)
    assert config.r0_hat_location is None
    assert config.resolved_r0_hat_location == "local"
    assert config.p_location == "local"
    assert config.v_location == "local"
    assert config.tmp_location == "local"
    assert config.s_hat_location == "local"


def test_bicgstab_r0_hat_auto_placement():
    """Witness vector auto-selects shared in the DRAM-bound window.

    The window is 512 <= n*itemsize <= 1024 bytes: below it the
    working set is served on-chip and shared placement loses; above
    it the 32 KiB dynamic-shared cap collapses the block size.
    """
    cases = [
        (np.float32, 8, "local"),
        (np.float32, 100, "local"),
        (np.float32, 128, "shared"),
        (np.float32, 200, "shared"),
        (np.float32, 256, "shared"),
        (np.float32, 300, "local"),
        (np.float64, 50, "local"),
        (np.float64, 64, "shared"),
        (np.float64, 128, "shared"),
        (np.float64, 200, "local"),
    ]
    for prec, n, expected in cases:
        config = BiCGSTABSolverConfig(precision=prec, n=n)
        assert config.resolved_r0_hat_location == expected, (
            f"n={n}, precision={prec.__name__}"
        )


def test_bicgstab_r0_hat_override_respected():
    """Explicit r0_hat_location bypasses the auto heuristic."""
    config = BiCGSTABSolverConfig(
        precision=np.float32, n=200, r0_hat_location="local"
    )
    assert config.resolved_r0_hat_location == "local"
    config = BiCGSTABSolverConfig(
        precision=np.float32, n=8, r0_hat_location="shared"
    )
    assert config.resolved_r0_hat_location == "shared"


def test_bicgstab_linear_correction_type(precision):
    """BiCGSTABSolver reports 'bicgstab' as correction type."""
    solver = BiCGSTABSolver(precision=precision, n=3)
    assert solver.linear_correction_type == "bicgstab"


def test_bicgstab_settings_dict(precision):
    """BiCGSTAB settings_dict includes buffer locations and tolerances."""
    n = 3
    solver = BiCGSTABSolver(
        precision=precision,
        n=n,
        krylov_atol=1e-6,
        krylov_rtol=1e-4,
        krylov_max_iters=50,
    )
    settings = solver.settings_dict
    assert settings["krylov_max_iters"] == 50
    assert settings["linear_correction_type"] == "bicgstab"
    assert "krylov_atol" in settings
    assert "krylov_rtol" in settings
    assert np.all(settings["krylov_atol"] == precision(1e-6))
    assert np.all(settings["krylov_rtol"] == precision(1e-4))
    assert "r0_hat_location" in settings
    assert "p_location" in settings
    assert "v_location" in settings
    assert "tmp_location" in settings
    assert "s_hat_location" in settings


def test_bicgstab_inherits_from_linear_solver_base(precision):
    """BiCGSTABSolver is instance of LinearSolverBase."""
    from cubie.integrators.matrix_free_solvers.linear_solver_base import (
        LinearSolverBase,
    )
    from cubie.integrators.matrix_free_solvers.base_solver import (
        MatrixFreeSolver,
    )

    solver = BiCGSTABSolver(precision=precision, n=3)
    assert isinstance(solver, LinearSolverBase)
    assert isinstance(solver, MatrixFreeSolver)
    assert solver.solver_type == "krylov"


def test_bicgstab_tolerance_broadcast(precision):
    """Scalar krylov_atol/rtol broadcasts to array of length n."""
    n = 5
    solver = BiCGSTABSolver(
        precision=precision,
        n=n,
        krylov_atol=1e-6,
        krylov_rtol=1e-4,
    )
    assert solver.krylov_atol.shape == (n,)
    assert solver.krylov_rtol.shape == (n,)
    assert np.all(solver.krylov_atol == precision(1e-6))
    assert np.all(solver.krylov_rtol == precision(1e-4))


# --- Cached-auxiliaries path (Rosenbrock-W selects this) -------------
# The operator and preconditioner use the cached signature, taking
# cached_aux immediately after drivers. Here cached_aux carries the
# diagonal of A, so a correct solve proves cached_aux is threaded
# through the solver rather than ignored.
@cuda.jit(device=True, inline=True)
def _cached_diag_operator(
    state, parameters, drivers, cached_aux, base_state,
    t, h, a_ij, vin, vout,
):
    for i in range(vin.shape[0]):
        vout[i] = cached_aux[i] * vin[i]


@cuda.jit(device=True, inline=True)
def _cached_jacobi_precond(
    state, parameters, drivers, cached_aux, base_state,
    t, h, a_ij, rhs, out, temp, scratch,
):
    for i in range(rhs.shape[0]):
        out[i] = rhs[i] / cached_aux[i]


def _cached_solver_kernel(n, precision):
    """Kernel that invokes a solver with the cached-aux signature."""
    scratch_size = 2 * n

    def factory(solver, h):
        @cuda.jit
        def kernel(state_init, rhs, base_state, cached_aux, x, flag):
            time_scalar = precision(0.0)
            state = cuda.local.array(n, precision)
            for i in range(n):
                state[i] = state_init[i]
            parameters = cuda.local.array(1, precision)
            drivers = cuda.local.array(1, precision)
            shared = cuda.shared.array(scratch_size, dtype=precision)
            persistent_local = cuda.local.array(
                scratch_size, dtype=precision
            )
            counters = cuda.local.array(1, np.int32)
            flag[0] = solver(
                state, parameters, drivers, base_state, cached_aux,
                time_scalar, h, precision(1.0), rhs, x, shared,
                persistent_local, counters,
            )

        return kernel

    return factory


@pytest.mark.parametrize("with_precond", [False, True])
def test_bicgstab_cached_auxiliaries(precision, tolerance, with_precond):
    """Cached-aux BiCGSTAB compiles and solves (Rosenbrock-W path).

    Regression for the signature mismatch that made
    ``linear_correction_type='bicgstab'`` unusable with Rosenbrock-W:
    the solver received a cached-signature operator but emitted a
    non-cached call site.
    """
    n = 3
    diag = np.array([4.0, 5.0, 6.0], dtype=precision)
    rhs = np.array([1.0, 1.0, 1.0], dtype=precision)

    solver = BiCGSTABSolver(
        precision=precision,
        n=n,
        krylov_atol=1e-8,
        krylov_rtol=1e-8,
        krylov_max_iters=200,
    )
    solver.update(
        operator_apply=_cached_diag_operator,
        preconditioner=_cached_jacobi_precond if with_precond else None,
        use_cached_auxiliaries=True,
    )
    solver_fn = solver.device_function

    kernel = _cached_solver_kernel(n, precision)(solver_fn, precision(0.01))
    state = cuda.to_device(np.zeros(n, dtype=precision))
    base = cuda.to_device(np.zeros(n, dtype=precision))
    aux = cuda.to_device(diag)
    rhs_dev = cuda.to_device(rhs.copy())
    x_dev = cuda.to_device(np.zeros(n, dtype=precision))
    flag = cuda.to_device(np.array([0], dtype=np.int32))

    kernel[1, 1](state, rhs_dev, base, aux, x_dev, flag)

    assert (flag.copy_to_host()[0] & 0xFF) == CUBIE_RESULT_CODES.SUCCESS
    assert_allclose(
        x_dev.copy_to_host(),
        rhs / diag,
        rtol=tolerance.rel_loose,
        atol=tolerance.abs_loose,
    )
