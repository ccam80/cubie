import pytest
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
def simple_ode_strict(symbolic_input_simple, precision):
    return SymbolicODE.create(
        precision=precision,
        dxdt=symbolic_input_simple["dxdt"],
        states=symbolic_input_simple["states"],
        parameters=symbolic_input_simple["parameters"],
        constants=symbolic_input_simple["constants"],
        observables=symbolic_input_simple["observables"],
        drivers=symbolic_input_simple["drivers"],
        name="simpletest_strict",
        strict=True,
    )


@pytest.fixture(scope="session")
def simple_ode_nonstrict(symbolic_input_simple, precision):
    return SymbolicODE.create(
        precision=precision,
        dxdt=symbolic_input_simple["dxdt"],
        strict=False,
        name="simpletest_nonstrict",
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

    func = built_simple_strict.evaluate_observables
    assert callable(func)
    cached = built_simple_strict.evaluate_observables
    assert func is cached


def test_time_derivative_helper_available(built_simple_strict):
    """Time-derivative helper should be compiled during system build."""

    helper = built_simple_strict.get_solver_helper("time_derivative_rhs")
    assert callable(helper)


class TestSympyStringEquivalence:
    """Test equivalence of SymPy and string input pathways."""
    
    def test_generated_code_identical(self, precision):
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
            precision=precision,
            states={"x": 1.0, "y": 0.0},
            parameters={"k": 0.1},
            name="test_sympy",
        )

        dxdt_string = ["dx = -k * x", "dy = k * x"]
        
        ode_string = SymbolicODE.create(
            dxdt=dxdt_string,
            precision=precision,
            states={'x': 1.0, 'y': 0.0},
            parameters={'k': 0.1},
            name='test_string'
        )
        
        assert is_devfunc(ode_sympy.evaluate_f)
        assert is_devfunc(ode_string.evaluate_f)
        
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
    
    def test_observables_equivalence(self, precision):
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
            precision=precision,
            parameters={'k': 0.1},
            observables=['z']
        )
        
        dxdt_string = ["dx = -k * x", "z = x * k"]
        
        ode_string = SymbolicODE.create(
            dxdt=dxdt_string,
            precision=precision,
            states={'x': 1.0},
            parameters={'k': 0.1},
            observables=['z']
        )
        
        assert len(ode_sympy.indices.observables.index_map) == 1
        assert len(ode_string.indices.observables.index_map) == 1


class TestSymbolicODEHash:
    """Test hash handling in SymbolicODE."""

    def test_symbolic_ode_hash_determinism(self, precision):
        """Verify identical SymbolicODE systems produce identical fn_hash."""
        dxdt = ["dx = -k * x", "dy = k * x"]
        states = {"x": 1.0, "y": 0.0}
        parameters = {"k": 0.1}

        ode1 = SymbolicODE.create(
            dxdt=dxdt,
            precision=precision,
            states=states,
            parameters=parameters,
            name="hash_test_1",
        )

        ode2 = SymbolicODE.create(
            dxdt=dxdt,
            precision=precision,
            states=states,
            parameters=parameters,
            name="hash_test_2",
        )

        assert ode1.fn_hash == ode2.fn_hash

    def test_symbolic_ode_hash_fallback(self, precision):
        """Verify __init__ correctly computes hash when fn_hash=None."""
        from cubie.odesystems.symbolic.parsing.parser import parse_input

        dxdt = ["dx = -k * x", "dy = k * x"]
        states = {"x": 1.0, "y": 0.0}
        parameters = {"k": 0.1}
        constants = {"c": 2.0}

        # Create via parse_input (provides fn_hash)
        parsed_result = parse_input(
            dxdt=dxdt,
            states=list(states.keys()),
            parameters=list(parameters.keys()),
            constants=constants,
        )
        expected_hash = parsed_result[4]

        # Create via SymbolicODE.create (computes hash internally)
        ode = SymbolicODE.create(
            dxdt=dxdt,
            precision=precision,
            states=states,
            parameters=parameters,
            constants=constants,
            name="hash_fallback_test",
        )

        assert ode.fn_hash == expected_hash


class TestCacheSkipsCodegen:
    """Tests verifying codegen is skipped when file cache exists."""

    def test_prepare_jac_aux_count_cached(self, precision):
        """Verify prepare_jac aux_count is retrieved from cached factory."""
        ode = SymbolicODE.create(
            dxdt=["dx = -k * x", "dy = k * x"],
            precision=precision,
            states={"x": 1.0, "y": 0.0},
            parameters={"k": 0.1},
            name="cache_test_prepare_jac",
        )
        ode.build()

        # First call generates and caches prepare_jac
        helper1 = ode.get_solver_helper("prepare_jac")
        assert callable(helper1)
        aux_count_initial = ode._jacobian_aux_count
        assert aux_count_initial is not None

        # Create a new ODE instance with the same definition and name
        # to exercise retrieval of prepare_jac from the file cache.
        ode_cached = SymbolicODE.create(
            dxdt=["dx = -k * x", "dy = k * x"],
            precision=precision,
            states={"x": 1.0, "y": 0.0},
            parameters={"k": 0.1},
            name="cache_test_prepare_jac",
        )
        ode_cached.build()

        # Second call should retrieve from file cache (no fresh codegen)
        # and restore aux_count from the cached factory attribute.
        helper2 = ode_cached.get_solver_helper("prepare_jac")
        assert callable(helper2)
        assert ode_cached._jacobian_aux_count == aux_count_initial

    def test_codegen_skipped_on_cache_hit(self, precision):
        """Verify that code generation is skipped when function is cached."""
        ode = SymbolicODE.create(
            dxdt=["dx = -k * x", "dy = k * x + c"],
            precision=precision,
            states={"x": 1.0, "y": 0.0},
            parameters={"k": 0.1},
            constants={"c": 0.5},
            name="cache_skip_codegen_test",
        )
        ode.build()

        # First call generates linear_operator
        helper1 = ode.get_solver_helper("linear_operator")
        assert callable(helper1)

        # Verify function is marked as cached in file
        factory_name = "linear_operator"
        assert ode.gen_file.function_is_cached(factory_name)

        # Create a new ODE instance with the same definition and name
        # to exercise retrieval from the file cache.
        ode_cached = SymbolicODE.create(
            dxdt=["dx = -k * x", "dy = k * x + c"],
            precision=precision,
            states={"x": 1.0, "y": 0.0},
            parameters={"k": 0.1},
            constants={"c": 0.5},
            name="cache_skip_codegen_test",
        )
        ode_cached.build()

        # Second call should skip codegen (uses file cache)
        helper2 = ode_cached.get_solver_helper("linear_operator")
        assert callable(helper2)
