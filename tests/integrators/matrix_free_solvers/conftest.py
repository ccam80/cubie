import numpy as np
import pytest
from numba import cuda

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
    else:
        raise ValueError(f"Unknown system: {system}")

    sym_system = create_ODE_system(dxdt, states=[f"x{i}" for i in range(3)])
    i_minus_hj = sym_system.get_solver_helper("i-hj")
    residual_plus_i_minus_hj = sym_system.get_solver_helper("r+i-hj")

    return {
        "n": 3,
        "jvp": i_minus_hj,
        "residual": residual_plus_i_minus_hj,
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
                state, parameters, drivers, h, rhs, x, residual, z_vec, temp
            )

        return kernel

    return factory
