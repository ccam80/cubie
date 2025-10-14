import sympy as sp

from cubie.odesystems.symbolic.jacobian import (
    clear_cache,
    generate_analytical_jvp,
    generate_jacobian,
    get_cache_counts,
)
from cubie.odesystems.symbolic.parser import IndexedBases, ParsedEquations


def test_generate_jacobian_with_auxiliary():
    """Jacobian matches expected expressions with auxiliaries."""

    index_map = IndexedBases.from_user_inputs(
        states=["x", "y"],
        parameters=[],
        constants={"a": 0.0, "b": 0.0},
        observables=[],
        drivers=[],
    )
    x, y = list(index_map.states.ref_map.keys())
    a = index_map.constants.symbol_map["a"]
    b = index_map.constants.symbol_map["b"]
    dx, dy = list(index_map.dxdt.ref_map.keys())
    aux = sp.Symbol("aux", real=True)
    equations = [
        (aux, a * x + b * y),
        (dx, aux - x),
        (dy, -aux + y),
    ]
    parsed = ParsedEquations.from_equations(equations, index_map)
    jac = generate_jacobian(
        parsed,
        index_map.states.index_map,
        index_map.dxdt.index_map,
    )
    expected = sp.Matrix([[a - 1, b], [-a, -b + 1]])
    assert jac.equals(expected)


def test_generate_jacobian_coupled_nonlinear():
    """Jacobian of a coupled nonlinear system matches full expression."""

    index_map = IndexedBases.from_user_inputs(
        states=["x0", "x1"],
        parameters=[],
        constants={},
        observables=[],
        drivers=[],
    )
    x0, x1 = list(index_map.states.ref_map.keys())
    dx0, dx1 = list(index_map.dxdt.ref_map.keys())
    equations = [
        (dx0, sp.sin(x0) + x0 * x1 ** 2),
        (dx1, x0 ** 2 + sp.exp(x1)),
    ]
    parsed = ParsedEquations.from_equations(equations, index_map)
    jac = generate_jacobian(
        parsed,
        index_map.states.index_map,
        index_map.dxdt.index_map,
    )
    expected = sp.Matrix(
        [
            [sp.cos(x0) + x1 ** 2, 2 * x0 * x1],
            [2 * x0, sp.exp(x1)],
        ]
    )
    assert jac.equals(expected)


def test_jacobian_caching():
    """Repeated calls reuse cached Jacobian and JVP."""

    index_map = IndexedBases.from_user_inputs(
        states=["x", "y"],
        parameters=[],
        constants={},
        observables=[],
        drivers=[],
    )
    x, y = list(index_map.states.ref_map.keys())
    dx = next(iter(index_map.dxdt.ref_map.keys()))
    equations = [(dx, x + y)]
    parsed = ParsedEquations.from_equations(equations, index_map)
    clear_cache()
    generate_jacobian(parsed, index_map.states.index_map, index_map.dxdt.index_map)
    generate_analytical_jvp(
        parsed, index_map.states.index_map, index_map.dxdt.index_map
    )
    counts = get_cache_counts()
    assert counts == {"jac": 1, "jvp": 1}
    generate_jacobian(parsed, index_map.states.index_map, index_map.dxdt.index_map)
    generate_analytical_jvp(
        parsed, index_map.states.index_map, index_map.dxdt.index_map
    )
    counts2 = get_cache_counts()
    assert counts2 == counts
