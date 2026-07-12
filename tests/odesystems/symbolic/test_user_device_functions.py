"""End-to-end tests for user device functions in generated systems.

The generated module is imported standalone, so user device callables
must be injected into its namespace before the factory compiles. These
tests build systems whose ``dxdt`` calls a real
``@cuda.jit(device=True)`` function and solve them, comparing against
an identical system written without user functions.
"""

import numpy as np
import pytest
from numba import cuda

from cubie import create_ODE_system, solve_ivp
from cubie.odesystems.symbolic.codegen.neumann_convergence import (
    check_neumann_convergence,
)


@pytest.fixture(scope="module")
def cubed():
    @cuda.jit(device=True, inline="always")
    def cubed(x):
        return x * x * x

    return cubed


@pytest.fixture(scope="module")
def d_cubed():
    @cuda.jit(device=True, inline="always")
    def d_cubed(x, index):
        return 3.0 * x * x

    return d_cubed


def _solve(system, method):
    result = solve_ivp(
        system,
        y0={"x": 2.0},
        method=method,
        duration=0.5,
        dt=0.01,
        save_every=0.05,
    )
    assert not np.any(result.status_codes)
    return result.time_domain_array


@pytest.fixture(scope="module")
def reference_explicit(precision):
    reference = create_ODE_system(
        "dx = -x*x*x",
        states={"x": 2.0},
        precision=precision,
        name="userfunc_reference_explicit",
    )
    return _solve(reference, "euler")


def test_string_system_device_function_solves(cubed, precision,
                                              reference_explicit):
    """String-form dxdt calling a device function compiles and solves."""
    system = create_ODE_system(
        "dx = -cubed(x)",
        states={"x": 2.0},
        user_functions={"cubed": cubed},
        precision=precision,
        name="userfunc_string_explicit",
    )
    state = _solve(system, "euler")
    np.testing.assert_allclose(state, reference_explicit, rtol=1e-6)


def test_callable_system_device_function_solves(cubed, precision,
                                                reference_explicit):
    """Callable-form dxdt calling a device function compiles and solves."""

    def rhs(t, y):
        dx = -cubed(y.x)  # noqa: F821
        return [dx]

    system = create_ODE_system(
        rhs,
        states={"x": 2.0},
        user_functions={"cubed": cubed},
        precision=precision,
        name="userfunc_callable_explicit",
    )
    state = _solve(system, "euler")
    np.testing.assert_allclose(state, reference_explicit, rtol=1e-6)


def test_device_function_with_derivative_implicit_solve(
    cubed, d_cubed, precision
):
    """Jacobian-based helpers resolve the derivative device function."""
    system = create_ODE_system(
        "dx = -cubed(x)",
        states={"x": 2.0},
        user_functions={"cubed": cubed},
        user_function_derivatives={"cubed": d_cubed},
        precision=precision,
        name="userfunc_string_implicit",
    )
    reference = create_ODE_system(
        "dx = -x*x*x",
        states={"x": 2.0},
        precision=precision,
        name="userfunc_reference_implicit",
    )
    state = _solve(system, "backwards_euler")
    expected = _solve(reference, "backwards_euler")
    np.testing.assert_allclose(state, expected, rtol=1e-5)


def test_check_neumann_convergence_evaluates_device_function(
    cubed, d_cubed, precision
):
    """The diagnostic evaluates the compiled ``dxdt`` on the device.

    For ``dx = -cubed(x)`` at ``x = 2`` the Jacobian is ``-12``; a
    single-state system is diagonally dominant, so the check returns
    a genuine verdict even though the user function is a device-only
    callable.
    """
    system = create_ODE_system(
        "dx = -cubed(x)",
        states={"x": 2.0},
        user_functions={"cubed": cubed},
        user_function_derivatives={"cubed": d_cubed},
        precision=precision,
        name="userfunc_neumann_device",
    )
    result = check_neumann_convergence(
        system.indices,
        system._get_neumann_evaluator(),
    )
    assert result["converges"] is True
    np.testing.assert_allclose(
        result["J_numeric"], [[-12.0]], rtol=5e-2
    )
