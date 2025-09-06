import numpy as np
import pytest
from numba import cuda, from_dtype

from cubie.systemmodels.symbolic.operator_apply import (
    generate_residual_end_state_code,
    generate_neumann_preconditioner_code,
)
from cubie.systemmodels.symbolic.symbolicODE import create_ODE_system


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

    sym_system = create_ODE_system(dxdt, states=[f"x{i}" for i in range(3)])
    sym_system.build()
    dxdt_func = sym_system.dxdt_function
    operator = sym_system.get_solver_helper("operator")
    neumann_code = generate_neumann_preconditioner_code(
        sym_system.equations, sym_system.indices
    )
    neumann_factory = sym_system.gen_file.import_function(
        "neumann_preconditioner_factory", neumann_code
    )
    code = generate_residual_end_state_code(sym_system.indices)
    res_factory = sym_system.gen_file.import_function(
        "residual_end_state_factory", code
    )
    residual_func = res_factory(
        sym_system.constants.values_dict,
        from_dtype(sym_system.precision),
        dxdt_func,
    )

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
    def dxdt_kernel(state, params, drivers, observables, deriv):
        dxdt_func(state, params, drivers, observables, deriv)

    dxdt_kernel[1, 1](base_host, params, drivers, observables, deriv)
    state_init_host = base_host + h * deriv * precision(1.01)
    if system == "stiff":
        state_init_host += precision(1e-3)

    state_fp = state_init_host.copy()
    for _ in range(32):
        dxdt_kernel[1, 1](state_fp, params, drivers, observables, deriv)
        new_state = base_host + h * deriv
        if np.max(np.abs(new_state - state_fp)) < precision(1e-12):
            state_fp = new_state
            break
        state_fp = new_state
    nk_expected = state_fp

    F = np.zeros((3, 3), dtype=precision)
    temp_in = np.zeros(3, dtype=precision)
    temp_out = np.zeros(3, dtype=precision)

    @cuda.jit()
    def operator_kernel(state, params, drivers, h, in_vec, out_vec):
        operator(state, params, drivers, h, in_vec, out_vec)

    for j in range(3):
        temp_in.fill(0)
        temp_in[j] = precision(1.0)
        operator_kernel[1, 1](state_fp, params, drivers, h, temp_in, temp_out)
        F[:, j] = temp_out
    mr_expected = np.linalg.solve(F, mr_rhs)

    def make_precond(order):
        return neumann_factory(
            sym_system.constants.values_dict,
            from_dtype(sym_system.precision),
            order=order,
        )

    return {
        "id": system,
        "n": 3,
        "h": h,
        "operator": operator,
        "residual": residual_func,
        "base_state": base_state,
        "state_init": cuda.to_device(state_init_host),
        "preconditioner": make_precond,
        "mr_rhs": mr_rhs,
        "mr_expected": mr_expected,
        "nk_expected": nk_expected,
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
        @cuda.jit
        def kernel(state_init, residual, out):
            state = cuda.local.array(n, precision)
            for i in range(n):
                state[i] = state_init[i]
            parameters = cuda.local.array(1, precision)
            drivers = cuda.local.array(1, precision)
            scratch = cuda.shared.array(n, precision)
            precond(state, parameters, drivers, h, residual, out, scratch)

        return kernel

    return factory


@pytest.fixture(scope="function")
def solver_kernel(precision):
    """Compile a kernel for linear solver device functions.

    Parameters
    ----------
    precision : np.dtype
        Floating point precision used for arrays.

    Returns
    -------
    callable
        Factory producing kernels executing
        ``(state_init, rhs, x, residual, z_vec, temp)``.
    """

    def factory(solver, n, h):
        @cuda.jit
        def kernel(state_init, rhs, x, residual, z_vec, temp, flag):
            state = cuda.local.array(n, precision)
            for i in range(n):
                state[i] = state_init[i]
            parameters = cuda.local.array(1, precision)
            drivers = cuda.local.array(1, precision)
            flag[0] = solver(
                state, parameters, drivers, h, rhs, x, z_vec, temp
            )

        return kernel

    return factory
