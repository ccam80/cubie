import numpy as np
import pytest
from numba import cuda

from cubie.systemmodels.symbolic.operator_apply import residual_end_state_factory
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
        solutions.
    """
    system = request.param
    if system == "linear":
        dxdt = [
            "dx0 = 0.5*x0 - 1.0",
            "dx1 = 0.5*x1 - 2.0",
            "dx2 = 0.5*x2 - 3.0",
        ]
        mr_rhs = np.array([1.0, 2.0, 3.0], dtype=precision)
        mr_expected = 2 * mr_rhs
        nk_expected = mr_expected.copy()
    elif system == "nonlinear":
        dxdt = [
            "dx0 = 0.5*x0 - 1.0",
            "dx1 = x1**3 - 1.0",
            "dx2 = -50.0*x2 + x2**3 - 1.0",
        ]
        mr_rhs = np.array([1.0, 1.0, 1.0], dtype=precision)
        j0 = precision(0.5)
        j1 = precision(3.0)
        sol2 = precision(7.08104667829272)
        j2 = precision(-50.0 + 3.0 * (sol2 ** 2))
        mr_expected = np.array(
            [1.0 / (1.0 - j0), 1.0 / (1.0 - j1), 1.0 / (1.0 - j2)],
            dtype=precision,
        )
        nk_expected = np.array([2.0, 1.0, sol2], dtype=precision)
    elif system == "stiff":
        dxdt = [
            "dx0 = 1e-6*x0 - 1e-6",
            "dx1 = 0.5*x1 - 0.5",
            "dx2 = 1e6*x2 - 1e6",
        ]
        mr_rhs = np.array([1.0, 1.0, 1.0], dtype=precision)
        mr_expected = np.array(
            [1.0 / (1.0 - 1e-6), 1.0 / (1.0 - 0.5), 1.0 / (1.0 - 1e6)],
            dtype=precision,
        )
        nk_expected = np.array([1.0, 1.0, 1.0], dtype=precision)
    elif system == "coupled_linear":
        dxdt = [ "dx0 = 0.5*x0 + 0.1*x1 - 1.0",
                 "dx1 = 0.2*x0 + 0.3*x1 - 1.0",
                 "dx2 = 0.1*x0 + 0.2*x1 + 0.4*x2 - 1.0",
                 ]
        mr_rhs = np.array([1.0, 1.0, 1.0], dtype=precision)
        # Solution values for linear coupled system
        sol0 = precision(1.923)
        sol1 = precision(1.538)
        sol2 = precision(1.346)
        # Jacobian diagonal elements
        j0 = precision(0.5)
        j1 = precision(0.3)
        j2 = precision(0.4)
        mr_expected = np.array( [1.0 / (1.0 - j0),
                                 1.0 / (1.0 - j1),
                                 1.0 / (1.0 - j2)],
                                dtype=precision,
                                )
        nk_expected = np.array([sol0, sol1, sol2], dtype=precision)
    elif system == "coupled_nonlinear":
        dxdt = [ "dx0 = 0.5*x0 - x1**2 - 1.0",
                 "dx1 = x0*x1 - x1**3 - 1.0",
                 "dx2 = x0 + x1**2 - x2**2 - 1.0",
                 ]
        mr_rhs = np.array([1.0, 1.0, 1.0], dtype=precision)
        # Approximate solution values for coupled nonlinear system
        sol0 = precision(2.618)
        sol1 = precision(1.618)
        sol2 = precision(2.236)
        # Jacobian diagonal elements at solution
        j0 = precision(0.5 - 2.0 *  sol1)
        j1 = precision(sol0 - 3.0 * (sol1 ** 2))
        j2 = precision(-2.0 * sol2)
        mr_expected = np.array( [1.0 / (1.0 - j0),
                                 1.0 / (1.0 - j1),
                                 1.0 / (1.0 - j2)],
                                dtype=precision,
                                )
        nk_expected = np.array([sol0, sol1, sol2], dtype=precision)
    else:
        raise ValueError(f"Unknown system: {system}")

    sym_system = create_ODE_system(dxdt, states=[f"x{i}" for i in range(3)])
    sym_system.build()
    dxdt_func = sym_system.dxdt_function
    operator = sym_system.get_solver_helper("operator")
    base_state = cuda.to_device(np.zeros(3, dtype=precision))
    residual_func = residual_end_state_factory(base_state, dxdt_func)

    return {
        "n": 3,
        "operator": operator,
        "residual": residual_func,
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

    def factory(precond, n):
        @cuda.jit
        def kernel(state_init, residual, out):
            state = cuda.local.array(n, precision)
            for i in range(n):
                state[i] = state_init[i]
            parameters = cuda.local.array(1, precision)
            drivers = cuda.local.array(1, precision)
            h = precision(1.0)
            precond(state, parameters, drivers, h, residual, out)

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

    def factory(solver, n):
        @cuda.jit
        def kernel(state_init, rhs, x, residual, z_vec, temp):
            state = cuda.local.array(n, precision)
            for i in range(n):
                state[i] = state_init[i]
            parameters = cuda.local.array(1, precision)
            drivers = cuda.local.array(1, precision)
            h = precision(1.0)
            solver(
                state, parameters, drivers, h, rhs, x, z_vec, temp
            )

        return kernel

    return factory
