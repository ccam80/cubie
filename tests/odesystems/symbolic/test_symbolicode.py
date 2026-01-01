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


def test_generate_dummy_args_returns_all_keys(simple_ode_strict):
    """Verify _generate_dummy_args returns all expected solver helper keys."""
    dummy_args = simple_ode_strict._generate_dummy_args()

    expected_keys = {
        'dxdt',
        'observables',
        'linear_operator',
        'linear_operator_cached',
        'prepare_jac',
        'calculate_cached_jvp',
        'neumann_preconditioner',
        'neumann_preconditioner_cached',
        'stage_residual',
        'n_stage_residual',
        'n_stage_linear_operator',
        'n_stage_neumann_preconditioner',
        'time_derivative_rhs',
    }

    assert set(dummy_args.keys()) == expected_keys


def test_generate_dummy_args_correct_arities(simple_ode_strict):
    """Verify each argument tuple has the correct arity."""
    dummy_args = simple_ode_strict._generate_dummy_args()

    # Expected arities based on codegen template signatures
    expected_arities = {
        'dxdt': 6,  # state, params, drivers, obs, out, t
        'observables': 5,  # state, params, drivers, obs, t
        'linear_operator': 9,  # state, params, drivers, base, t, h, a, v, out
        'linear_operator_cached': 10,  # +cached_aux
        'prepare_jac': 5,  # state, params, drivers, t, cached_aux
        'calculate_cached_jvp': 7,  # state, params, drivers, aux, t, v, out
        'neumann_preconditioner': 10,  # state, params, drivers, base, ...
        'neumann_preconditioner_cached': 11,  # +cached_aux
        'stage_residual': 8,  # u, params, drivers, t, h, a, base, out
        'n_stage_residual': 8,  # u, params, drivers, t, h, a, base, out
        'n_stage_linear_operator': 9,  # state, params, drivers, base, ...
        'n_stage_neumann_preconditioner': 10,  # state, params, drivers, ...
        'time_derivative_rhs': 7,  # state, params, drivers, driver_dt, ...
    }

    for key, expected_arity in expected_arities.items():
        assert len(dummy_args[key]) == expected_arity, \
            f"{key} has arity {len(dummy_args[key])}, expected {expected_arity}"


def test_generate_dummy_args_array_shapes(simple_ode_strict):
    """Verify array arguments have shapes consistent with system sizes."""
    import numpy as np

    dummy_args = simple_ode_strict._generate_dummy_args()
    sizes = simple_ode_strict.sizes
    n_states = int(sizes.states)
    n_params = int(sizes.parameters)
    n_drivers = max(1, int(sizes.drivers))
    n_obs = max(1, int(sizes.observables))
    n_stages = 2  # Default stage count

    # Check dxdt arrays
    dxdt = dummy_args['dxdt']
    assert dxdt[0].shape == (n_states,)  # state
    assert dxdt[1].shape == (n_params,)  # parameters
    assert dxdt[2].shape == (n_drivers,)  # drivers
    assert dxdt[3].shape == (n_obs,)  # observables
    assert dxdt[4].shape == (n_states,)  # out
    assert np.isscalar(dxdt[5]) or dxdt[5].shape == ()  # t

    # Check n_stage arrays use flattened sizes
    n_stage_res = dummy_args['n_stage_residual']
    assert n_stage_res[0].shape == (n_stages * n_states,)  # u
    assert n_stage_res[2].shape == (n_stages * n_drivers,)  # drivers
    assert n_stage_res[6].shape == (n_states,)  # base_state
    assert n_stage_res[7].shape == (n_stages * n_states,)  # out


def test_generate_dummy_args_precision(simple_ode_strict):
    """Verify all arrays and scalars use the correct precision dtype."""
    import numpy as np

    dummy_args = simple_ode_strict._generate_dummy_args()
    precision = simple_ode_strict.precision

    for key, args in dummy_args.items():
        for i, arg in enumerate(args):
            if isinstance(arg, np.ndarray):
                assert arg.dtype == precision, \
                    f"{key}[{i}] has dtype {arg.dtype}, expected {precision}"
            elif np.isscalar(arg):
                # Scalar should be numpy scalar with correct dtype
                assert type(arg) == precision or \
                    (hasattr(arg, 'dtype') and arg.dtype == precision), \
                    f"{key}[{i}] scalar has wrong precision"
