import numpy as np
import pytest
from cubie.cuda_simsafe import cuda

from cubie.odesystems.symbolic.symbolicODE import create_ODE_system


@pytest.fixture(scope="function")
def system_setup(request, precision):
    """Generate symbolic systems for solver tests.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Provides the system identifier via ``param``.
    precision : np.dtype
        Floating point precision for the system.

    Returns
    -------
    dict
        Problem definition including helper functions and reference
        solutions computed for a small implicit Euler step.
    """
    system = request.param
    if system == "linear":
        dxdt = [
            "dx0 = 0.5*x0 - 1.0",
            "dx1 = 0.5*x1 - 2.0",
            "dx2 = 0.5*x2 - 3.0",
        ]
        mr_rhs = np.array([1.0, 2.0, 3.0], dtype=precision)
    elif system == "nonlinear":
        dxdt = [
            "dx0 = 0.5*x0 - 1.0",
            "dx1 = x1**3 - 1.0",
            "dx2 = -50.0*x2 + x2**3 - 1.0",
        ]
        mr_rhs = np.array([1.0, 1.0, 1.0], dtype=precision)
    elif system == "stiff":
        dxdt = [
            "dx0 = 1e-6*x0 - 1e-6",
            "dx1 = 0.5*x1 - 0.5",
            "dx2 = 1e6*x2 - 1e6",
        ]
        mr_rhs = np.array([1.0, 1.0, 1.0], dtype=precision)
    elif system == "coupled_linear":
        dxdt = [
            "dx0 = 0.5*x0 + 0.1*x1 - 1.0",
            "dx1 = 0.2*x0 + 0.3*x1 - 1.0",
            "dx2 = 0.1*x0 + 0.2*x1 + 0.4*x2 - 1.0",
        ]
        mr_rhs = np.array([1.0, 1.0, 1.0], dtype=precision)
    elif system == "coupled_nonlinear":
        dxdt = [
            "dx0 = 0.5*x0 - x1**2 - 1.0",
            "dx1 = x0*x1 - x1**3 - 1.0",
            "dx2 = x0 + x1**2 - x2**2 - 1.0",
        ]
        mr_rhs = np.array([1.0, 1.0, 1.0], dtype=precision)
    else:
        raise ValueError(f"Unknown system: {system}")


    #Construct system, generate helper functions
    sym_system = create_ODE_system(dxdt,
                                   states=[f"x{i}" for i in range(3)],
                                   precision=precision)
    sym_system.build()
    dxdt_func = sym_system.evaluate_f
    operator = sym_system.get_solver_helper("linear_operator")
    # Use helper interface for residual and preconditioner generation
    residual_func = sym_system.get_solver_helper("stage_residual")

    def make_precond(order):
        return sym_system.get_solver_helper(
            "neumann_preconditioner", preconditioner_order=order
        )

    # start system from a non-equilibrium position, generate initial guesses
    # using Euler
    if system == "stiff":
        h = precision(1e-4)
        base_host = np.ones(3, dtype=precision)
    else:
        h = precision(0.01)
        base_host = np.zeros(3, dtype=precision)

    base_state = cuda.to_device(base_host)
    params = np.zeros(1, dtype=precision)
    drivers = np.zeros(1, dtype=precision)
    observables = np.zeros(3, dtype=precision)
    deriv = np.zeros(3, dtype=precision)

    @cuda.jit()
    def dxdt_kernel(state, params, drivers, observables, deriv, time_scalar):
        dxdt_func(state, params, drivers, observables, deriv, time_scalar)

    zero_time = precision(0.0)
    dxdt_kernel[1, 1](base_host, params, drivers, observables, deriv, zero_time)
    state_init_host = base_host + h * deriv * precision(1.05)

    # Step forward until we converge onto the solution
    state_fp = state_init_host.copy()
    for _ in range(32):
        dxdt_kernel[1, 1](state_fp, params, drivers, observables, deriv, zero_time)
        new_state = base_host + h * deriv
        if np.max(np.abs(new_state - state_fp)) < precision(1e-7):
            state_fp = new_state
            break
        state_fp = new_state
    nk_expected = state_fp

    F = np.zeros((3, 3), dtype=precision)
    temp_in = np.zeros(3, dtype=precision)
    temp_out = np.zeros(3, dtype=precision)

    @cuda.jit()
    def operator_kernel(state, params, drivers, base_state, time_scalar, h, in_vec, out_vec):
        operator(
            state,
            params,
            drivers,
            base_state,
            time_scalar,
            h,
            precision(1.0),
            in_vec,
            out_vec,
        )

    for j in range(3):
        temp_in.fill(0)
        temp_in[j] = precision(1.0)
        operator_kernel[1, 1](state_fp, params, drivers, base_state, zero_time, h, temp_in, temp_out)
        F[:, j] = temp_out
    try:
        mr_expected = np.linalg.solve(F, mr_rhs)
    except np.linalg.LinAlgError:
        # Headless launches leave F singular; later tests still compile.
        mr_expected = np.full_like(mr_rhs, np.nan)


    return {
        "id": system,
        "n": 3,
        "h": h,
        "operator": operator,
        "residual": residual_func,
        "base_state": base_state,
        "state_init": cuda.to_device(state_init_host - base_host),
        "preconditioner": make_precond,
        "mr_rhs": mr_rhs,
        "mr_expected": mr_expected,
        "nk_expected": nk_expected,
        "sym_system": sym_system,
    }


@pytest.fixture(scope="function")
def neumann_kernel(precision):
    """Compile a kernel for the Neumann preconditioner.

    Parameters
    ----------
    precision : np.dtype
        Floating point precision used for arrays.

    Returns
    -------
    callable
        Factory producing kernels of the form
        ``(state_init, residual, out)``.
    """

    def factory(precond, n, h):
        scratch_size = n

        @cuda.jit
        def kernel(state_init, residual, base_state, out):
            time_scalar = precision(0.0)
            state = cuda.local.array(n, precision)
            for i in range(n):
                state[i] = state_init[i]
            parameters = cuda.local.array(1, precision)
            drivers = cuda.local.array(1, precision)
            temp = cuda.shared.array(scratch_size, dtype=precision)
            scratch = cuda.shared.array(scratch_size, dtype=precision)
            precond(
                state,
                parameters,
                drivers,
                base_state,
                time_scalar,
                h,
                precision(1.0),
                residual,
                out,
                temp,
                scratch,
            )

        return kernel

    return factory


@pytest.fixture(scope="function")
def solver_kernel():
    """Compile a kernel for linear solver device functions.

    Parameters
    ----------
    precision : np.dtype
        Floating point precision used for arrays.

    Returns
    -------
    callable
        Factory producing kernels executing ``(state_init, rhs, x)``.
    """
    def factory(solver, n, h, precision):
        scratch_size = 2 * n
        @cuda.jit
        def kernel(state_init, rhs, base_state, x, flag):
            time_scalar = precision(0.0)
            state = cuda.local.array(n, precision)
            for i in range(n):
                state[i] = state_init[i]
            parameters = cuda.local.array(1, precision)
            drivers = cuda.local.array(1, precision)
            # Allocate shared memory for solver buffers
            shared = cuda.shared.array(scratch_size, dtype=precision)
            persistent_local = cuda.local.array(scratch_size, dtype=precision)
            counters = cuda.local.array(1, np.int32)
            flag[0] = solver(
                state,
                parameters,
                drivers,
                base_state,
                time_scalar,
                h,
                precision(1.0),
                rhs,
                x,
                shared,
                persistent_local,
                counters
            )

        return kernel

    return factory


@pytest.fixture(scope="function")
def linear_solver_instance(solver_settings, system_setup, precision):
    """Build the linear solver selected by the central solver settings.

    Routes ``linear_correction_type`` from the session ``solver_settings``
    fixture to the matching solver class, so parameterizing
    ``solver_settings_override`` with ``"bicgstab"`` exercises
    :class:`BiCGSTABSolver` through the same tests as the
    minimal-residual and steepest-descent solvers.
    """
    from cubie.integrators.matrix_free_solvers.bicgstab_solver import (
        BiCGSTABSolver,
    )
    from cubie.integrators.matrix_free_solvers.linear_solver import (
        MRLinearSolver,
    )

    order = solver_settings["preconditioner_order"]
    if order == 0:
        preconditioner = None
    else:
        preconditioner = system_setup["preconditioner"](order)

    correction_type = solver_settings["linear_correction_type"]
    common = {
        "precision": precision,
        "n": system_setup["n"],
        "krylov_atol": solver_settings["krylov_atol"],
        "krylov_rtol": solver_settings["krylov_rtol"],
        "krylov_max_iters": solver_settings["krylov_max_iters"],
    }
    if correction_type == "bicgstab":
        solver = BiCGSTABSolver(**common)
    else:
        solver = MRLinearSolver(
            linear_correction_type=correction_type, **common
        )
    solver.update(
        operator_apply=system_setup["operator"],
        preconditioner=preconditioner,
    )
    return solver


@pytest.fixture(scope="function")
def newton_solver_instance(
    solver_settings, linear_solver_instance, system_setup, precision
):
    """Wrap the configured linear solver in a NewtonKrylov instance."""
    from cubie.integrators.matrix_free_solvers.newton_krylov import (
        NewtonKrylov,
    )

    solver = NewtonKrylov(
        precision=precision,
        n=system_setup["n"],
        linear_solver=linear_solver_instance,
        newton_atol=solver_settings["newton_atol"],
        newton_rtol=solver_settings["newton_rtol"],
        newton_max_iters=solver_settings["newton_max_iters"],
        newton_damping=solver_settings["newton_damping"],
        newton_max_backtracks=solver_settings["newton_max_backtracks"],
    )
    solver.update(residual_function=system_setup["residual"])
    return solver
