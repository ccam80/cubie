import numpy as np
import pytest
from sympy import symbols, Eq

from cubie.systemmodels.symbolic import SymbolicODE
from cubie.systemmodels.symbolic.math_functions import exp_


def test_symbolic_basic():
    x, y, k = symbols('x y k')
    equations = [Eq(x, -k * x), Eq(y, k * x)]


def test_symbolic_basic_and_jacobian():
    x, y, k = symbols("x y k")
    equations = [Eq(x, -k * x), Eq(y, k * x)]
    system = SymbolicODE(
        states=[x, y], parameters={k: 1.0}, equations=equations
    )
    system.build()
    assert system.device_function is not None
    assert system.jac_v is not None
    states = np.array([1.0, 0.0], dtype=system.precision)
    params = np.array([2.0], dtype=system.precision)
    drivers = np.zeros(1, dtype=system.precision)
    dxdt, obs = system.correct_answer_python(states, params, drivers)
    assert dxdt[0] == pytest.approx(-2.0)
    assert dxdt[1] == pytest.approx(2.0)
    assert obs.size == 0
    jac = system.correct_jacobian_python(states, params, drivers)
    assert jac.shape == (2, 2)
    assert jac[0, 0] == pytest.approx(-2.0)
    assert jac[1, 0] == pytest.approx(2.0)
    assert jac[0, 1] == pytest.approx(0.0)
    assert jac[1, 1] == pytest.approx(0.0)
    assert obs.size == 0


def test_allowed_operator():
    x, k = symbols("x k")
    equations = [Eq(x, -k * exp_(x))]
    system = SymbolicODE(states=[x], parameters={k: 1.0}, equations=equations)
    states = np.array([1.0], dtype=system.precision)
    params = np.array([2.0], dtype=system.precision)
    drivers = np.zeros(1, dtype=system.precision)
    dxdt, _ = system.correct_answer_python(states, params, drivers)
    assert dxdt[0] == pytest.approx(-2.0 * np.exp(1.0))
