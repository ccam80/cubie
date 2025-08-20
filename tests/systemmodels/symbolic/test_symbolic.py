import numpy as np
import pytest
from sympy import symbols, Eq

from cubie.systemmodels.symbolic.symbolic import SymbolicODESystem


def test_symbolic_basic():
    x, y, k = symbols('x y k')
    equations = [Eq(x, -k*x), Eq(y, k*x)]
    system = SymbolicODESystem(states=[x, y], parameters={k: 1.0}, equations=equations)
    system.build()
    assert system.device_function is not None
    states = np.array([1.0, 0.0], dtype=system.precision)
    params = np.array([2.0], dtype=system.precision)
    drivers = np.zeros(1, dtype=system.precision)
    dxdt, obs = system.correct_answer_python(states, params, drivers)
    assert dxdt[0] == pytest.approx(-2.0)
    assert dxdt[1] == pytest.approx(2.0)
    assert obs.size == 0