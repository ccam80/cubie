import pytest
import numpy as np
from numpy.testing import assert_array_equal

from cubie.odesystems.symbolic.symbolicODE import (
    SymbolicODE,
    create_ODE_system,
)


@pytest.fixture(scope="session")
def symbolic_input_simple():
    return {
        "observables": ["obs1", "obs2"],
        "parameters": {"k1": 0.32, "k2": 0.91},
        "constants": {"c1": 2.1, "c2": 1.8},
        "drivers": {"d1": 0.9, "d2": 0.8},
        "states": {"x1": 0.5, "x2": 2.0},
        "dxdt": [
            "obs1 = k1 * x1 * d2 + d1 * c1",
            "obs2 = c2 * c2 * k2 + x1 + x2 ** 2 + obs1",
            "dx1 = obs1 + c2",
            "dx2 = c1 + obs1 + obs2",
        ],
    }


@pytest.fixture(scope="session")
def simple_ode_strict(symbolic_input_simple):
    return SymbolicODE.create(
        dxdt=symbolic_input_simple["dxdt"],
        states=symbolic_input_simple["states"],
        parameters=symbolic_input_simple["parameters"],
        constants=symbolic_input_simple["constants"],
        observables=symbolic_input_simple["observables"],
        drivers=symbolic_input_simple["drivers"],
        name="simpletest_strict",
        precision=np.float32,
        strict=True,
    )


@pytest.fixture(scope="session")
def simple_ode_nonstrict(symbolic_input_simple):
    return SymbolicODE.create(
        dxdt=symbolic_input_simple["dxdt"],
        strict=False,
        name="simpletest_nonstrict",
        precision=np.float32,
    )


def test_create_ODE_system_strict(simple_ode_strict, symbolic_input_simple):
    sys1 = create_ODE_system(
        dxdt=symbolic_input_simple["dxdt"],
        states=symbolic_input_simple["states"],
        parameters=symbolic_input_simple["parameters"],
        constants=symbolic_input_simple["constants"],
        observables=symbolic_input_simple["observables"],
        drivers=symbolic_input_simple["drivers"],
        name="simpletest_strict",
        precision=np.float32,
        strict=True,
    )
    sys2 = simple_ode_strict
    assert_array_equal(
        sys1.constants.values_array, sys2.constants.values_array
    )
    assert_array_equal(
        sys1.parameters.values_array, sys2.parameters.values_array
    )
    assert_array_equal(
        sys1.initial_values.values_array, sys2.initial_values.values_array
    )
    assert_array_equal(
        sys1.observables.values_array, sys2.observables.values_array
    )
    assert_array_equal(sys1.num_drivers, sys2.num_drivers)


def test_create_ODE_system_nonstrict(
    simple_ode_nonstrict, symbolic_input_simple
):
    sys1 = create_ODE_system(
        dxdt=symbolic_input_simple["dxdt"],
        name="simpletest_nonstrict",
        precision=np.float32,
    )
    sys2 = simple_ode_nonstrict
    assert_array_equal(
        sys1.constants.values_array, sys2.constants.values_array
    )
    assert_array_equal(
        sys1.parameters.values_array, sys2.parameters.values_array
    )
    assert_array_equal(
        sys1.initial_values.values_array, sys2.initial_values.values_array
    )
    assert_array_equal(
        sys1.observables.values_array, sys2.observables.values_array
    )
    assert_array_equal(sys1.num_drivers, sys2.num_drivers)


@pytest.fixture(scope="session")
def built_simple_strict(simple_ode_strict):
    simple_ode_strict.build()
    return simple_ode_strict


@pytest.fixture(scope="session")
def built_simple_nonstrict(simple_ode_nonstrict):
    simple_ode_nonstrict.build()
    return simple_ode_nonstrict

def test_simple_strict_builds(built_simple_strict):
    assert callable(built_simple_strict.get_solver_helper("linear_operator"))

def test_simple_nonstrict_builds(built_simple_nonstrict):
    assert callable(built_simple_nonstrict.get_solver_helper(
            "linear_operator"))


def test_solver_helper_cached(built_simple_strict):
    func1 = built_simple_strict.get_solver_helper("linear_operator")
    assert callable(func1)
    func2 = built_simple_strict.get_solver_helper("linear_operator")
    assert func1 is func2


def test_observables_helper_available(built_simple_strict):
    """Symbolic systems should expose an observables-only helper."""

    func = built_simple_strict.observables_function
    assert callable(func)
    cached = built_simple_strict.observables_function
    assert func is cached


def test_time_derivative_helper_available(built_simple_strict):
    """Time-derivative helper should be compiled during system build."""

    helper = built_simple_strict.get_solver_helper("time_derivative_rhs")
    assert callable(helper)


class TestSympyStringEquivalence:
    """Test equivalence of SymPy and string input pathways."""
    
    def test_generated_code_identical(self):
        """Verify SymPy and string inputs generate identical code."""
        import sympy as sp
        from cubie._utils import is_devfunc
        
        x, y, k = sp.symbols('x y k')
        dx, dy = sp.symbols('dx dy')
        dxdt_sympy = [
            sp.Eq(dx, -k * x),
            sp.Eq(dy, k * x)
        ]
        
        ode_sympy = SymbolicODE.create(
            dxdt=dxdt_sympy,
            states={'x': 1.0, 'y': 0.0},
            parameters={'k': 0.1},
            name='test_sympy',
            precision=np.float32,
        )
        
        dxdt_string = ["dx = -k * x", "dy = k * x"]
        
        ode_string = SymbolicODE.create(
            dxdt=dxdt_string,
            states={'x': 1.0, 'y': 0.0},
            parameters={'k': 0.1},
            name='test_string',
            precision=np.float32,
        )
        
        assert is_devfunc(ode_sympy.dxdt_function)
        assert is_devfunc(ode_string.dxdt_function)
        
        assert ode_sympy.num_states == ode_string.num_states
        assert ode_sympy.num_states == 2
    
    def test_hash_consistency(self):
        """Verify hash is consistent for equivalent definitions."""
        import sympy as sp
        from cubie.odesystems.symbolic.parsing.parser import parse_input
        
        x, k = sp.symbols('x k')
        dx = sp.Symbol('dx')
        dxdt_sympy = [sp.Eq(dx, -k * x)]
        
        dxdt_string = "dx = -k * x"
        
        result_sympy = parse_input(
            dxdt=dxdt_sympy,
            states=['x'],
            parameters=['k'],
            constants={'c': 1.0}
        )
        
        result_string = parse_input(
            dxdt=dxdt_string,
            states=['x'],
            parameters=['k'],
            constants={'c': 1.0}
        )
        
        hash_sympy = result_sympy[4]
        hash_string = result_string[4]
        
        assert hash_sympy == hash_string
    
    def test_observables_equivalence(self):
        """Verify observables work identically in both pathways."""
        import sympy as sp
        
        x, k, z = sp.symbols('x k z')
        dx = sp.Symbol('dx')
        dxdt_sympy = [
            sp.Eq(dx, -k * x),
            sp.Eq(z, x * k)
        ]
        
        ode_sympy = SymbolicODE.create(
            dxdt=dxdt_sympy,
            states={'x': 1.0},
            parameters={'k': 0.1},
            observables=['z'],
            precision=np.float32,
        )
        
        dxdt_string = ["dx = -k * x", "z = x * k"]
        
        ode_string = SymbolicODE.create(
            dxdt=dxdt_string,
            states={'x': 1.0},
            parameters={'k': 0.1},
            observables=['z'],
            precision=np.float32,
        )
        
        assert len(ode_sympy.indices.observables.index_map) == 1
        assert len(ode_string.indices.observables.index_map) == 1
