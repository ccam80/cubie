import sympy as sp

from cubie.systemmodels.symbolic.jacobian import (
    clear_cache,
    generate_analytical_jvp,
    generate_jacobian,
    get_cache_counts,
)


def test_generate_jacobian_with_auxiliary():
    """Jacobian matches expected expressions with auxiliaries."""

    x, y = sp.symbols("x y")
    a, b = sp.symbols("a b")
    equations = [
        (sp.Symbol("aux"), a * x + b * y),
        (sp.Symbol("dx"), sp.Symbol("aux") - x),
        (sp.Symbol("dy"), -sp.Symbol("aux") + y),
    ]
    input_order = {x: 0, y: 1}
    output_order = {sp.Symbol("dx"): 0, sp.Symbol("dy"): 1}
    jac = generate_jacobian(equations, input_order, output_order)
    expected = sp.Matrix([[a - 1, b], [-a, -b + 1]])
    assert jac.equals(expected)


def test_generate_jacobian_coupled_nonlinear():
    """Jacobian of a coupled nonlinear system matches full expression."""

    x0, x1 = sp.symbols("x0 x1")
    equations = [
        (sp.Symbol("dx0"), sp.sin(x0) + x0 * x1 ** 2),
        (sp.Symbol("dx1"), x0 ** 2 + sp.exp(x1)),
    ]
    input_order = {x0: 0, x1: 1}
    output_order = {sp.Symbol("dx0"): 0, sp.Symbol("dx1"): 1}
    jac = generate_jacobian(equations, input_order, output_order)
    expected = sp.Matrix(
        [
            [sp.cos(x0) + x1 ** 2, 2 * x0 * x1],
            [2 * x0, sp.exp(x1)],
        ]
    )
    assert jac.equals(expected)


def test_jacobian_caching():
    """Repeated calls reuse cached Jacobian and JVP."""

    x, y = sp.symbols("x y")
    equations = [(sp.Symbol("dx"), x + y)]
    input_order = {x: 0, y: 1}
    output_order = {sp.Symbol("dx"): 0}
    clear_cache()
    generate_jacobian(equations, input_order, output_order)
    generate_analytical_jvp(equations, input_order, output_order)
    counts = get_cache_counts()
    assert counts == {"jac": 1, "jvp": 1}
    generate_jacobian(equations, input_order, output_order)
    generate_analytical_jvp(equations, input_order, output_order)
    counts2 = get_cache_counts()
    assert counts2 == counts
