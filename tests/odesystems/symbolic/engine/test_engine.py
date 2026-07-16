"""Unit tests for the IR expression engine.

Verifies interning and algebraic folding, differentiation against
SymPy ground truth, simultaneous substitution, CSE numeric
equivalence, topological ordering, pruning, and SymPy round-trips.
"""

import math
import random

import sympy as sp

from cubie.odesystems.symbolic.engine import (
    TRUE,
    add,
    arr,
    call,
    count_ops,
    cse_and_stack,
    diff,
    div,
    free_atoms,
    from_sympy,
    is_one,
    is_zero,
    mul,
    neg,
    num,
    piecewise,
    pow_,
    prune_unused,
    rel,
    sub,
    sym,
    to_sympy,
    topological_sort,
    xreplace,
)


class TestInterningAndFolding:
    def test_structural_interning(self):
        x, y = sym("x"), sym("y")
        assert add(x, y) is add(y, x)
        assert mul(x, y) is mul(y, x)
        assert sym("x") is x
        assert arr("state", 1) is arr("state", 1)

    def test_like_term_collection(self):
        x = sym("x")
        assert add(x, x) is mul(num(2), x)
        assert sub(x, x) is num(0)

    def test_power_collection(self):
        x = sym("x")
        assert mul(x, x) is pow_(x, num(2))
        assert div(x, x) is num(1)

    def test_identity_folds(self):
        x = sym("x")
        assert add(x, num(0)) is x
        assert mul(x, num(1)) is x
        assert is_zero(mul(x, num(0)))
        assert is_one(pow_(x, num(0)))
        assert pow_(x, num(1)) is x

    def test_numeric_folding(self):
        assert mul(num(2), num(3)) is num(6)
        assert add(num(0.5), num(0.25)) is num(0.75)
        assert pow_(pow_(sym("x"), num(2)), num(3)) is pow_(
            sym("x"), num(6)
        )

    def test_int_and_float_zero_distinct_nodes(self):
        assert num(0) is not num(0.0)
        assert is_zero(num(0.0))

    def test_piecewise_collapses_after_true(self):
        x = sym("x")
        collapsed = piecewise((x, TRUE))
        assert collapsed is x


class TestDifferentiation:
    def test_matches_sympy_on_rule_table(self):
        x = sp.Symbol("x", real=True)
        y = sp.Symbol("y", real=True)
        cases = [
            x * sp.sin(x),
            sp.exp(2 * x) / (1 + x**2),
            sp.sqrt(x + y) * sp.cos(x * y),
            sp.tan(x) + sp.atan(x) + sp.tanh(x) + sp.atanh(x),
            sp.log(x) * sp.asin(y) + sp.acos(x * y),
            x ** sp.Rational(3, 2) + y**-2,
            sp.erf(x) + sp.erfc(2 * x),
            sp.atan2(y, x),
            sp.sinh(x) * sp.cosh(y) + sp.asinh(x) + sp.acosh(x + 2),
        ]
        rng = random.Random(7)
        for case in cases:
            for var in (x, y):
                expected = sp.lambdify(
                    (x, y), sp.diff(case, var), "math"
                )
                produced = sp.lambdify(
                    (x, y),
                    to_sympy(
                        diff(from_sympy(case), from_sympy(var))
                    ),
                    "math",
                )
                # Domain 0.15..0.85 keeps every argument valid:
                # |atanh/asin args| < 1 and acosh sees x + 2 > 1.
                for _ in range(8):
                    px = rng.uniform(0.15, 0.85)
                    py = rng.uniform(0.15, 0.85)
                    want = expected(px, py)
                    got = produced(px, py)
                    assert math.isclose(
                        got, want, rel_tol=1e-10, abs_tol=1e-12
                    ), (case, var, px, py, got, want)

    def test_piecewise_differentiates_by_branch(self):
        x = sym("x")
        expr = piecewise(
            (mul(x, x), rel("<", x, num(0))),
            (mul(num(3), x), TRUE),
        )
        result = diff(expr, x)
        expected = piecewise(
            (mul(num(2), x), rel("<", x, num(0))),
            (num(3), TRUE),
        )
        assert result is expected

    def test_min_differentiates_to_selection(self):
        x, y = sym("x"), sym("y")
        result = diff(call("Min", x, y), x)
        assert result is piecewise(
            (num(1), rel("<=", x, y)), (num(0), TRUE)
        )

    def test_user_function_derivative_placeholder(self):
        x, y = sym("x"), sym("y")
        result = diff(call("foo_", x, y), x)
        assert result is call("d_foo", x, y, num(0))
        renamed = diff(
            call("foo_", x, y),
            y,
            derivative_names={"foo_": "dfoo_dx"},
        )
        assert renamed is call("dfoo_dx", x, y, num(1))

    def test_array_reference_is_constant(self):
        x = sym("x")
        assert is_zero(diff(arr("v", 0), x))
        v0 = arr("v", 0)
        assert diff(mul(x, v0), x) is v0


class TestSubstitution:
    def test_simultaneous_replacement(self):
        x, y = sym("x"), sym("y")
        expr = add(x, y)
        swapped = xreplace(expr, {x: y, y: x})
        assert swapped is expr  # commutative interning

    def test_no_rescan_of_images(self):
        x = sym("x")
        image = add(x, num(1))
        replaced = xreplace(mul(x, x), {x: image})
        assert replaced is pow_(image, num(2))

    def test_array_replacement(self):
        combo = add(arr("base", 0), mul(sym("a"), arr("u", 0)))
        replaced = xreplace(
            mul(arr("v", 0), sym("k")), {arr("v", 0): combo}
        )
        assert replaced is mul(combo, sym("k"))


def _evaluate_lines(assignments, bindings):
    """Execute printed assignments and return the environment."""
    from cubie.odesystems.symbolic.engine import print_cuda_multiple

    env = {"math": math, "precision": float}
    env.update(bindings)
    for line in print_cuda_multiple(assignments):
        exec(line, env)
    return env


class TestCseAndStack:
    def test_numeric_equivalence(self):
        x, y, k = sym("x"), sym("y"), sym("k")
        shared = mul(call("exp", mul(k, x)), add(x, y))
        assignments = [
            (sym("r0"), add(shared, x)),
            (sym("r1"), mul(num(2), shared)),
            (sym("r2"), mul(shared, y, num(-3))),
        ]
        bindings = {"x": 0.37, "y": -1.2, "k": 2.5}
        direct = {
            "r0": (
                math.exp(2.5 * 0.37) * (0.37 - 1.2) + 0.37
            ),
            "r1": 2 * math.exp(2.5 * 0.37) * (0.37 - 1.2),
            "r2": -3 * math.exp(2.5 * 0.37) * (0.37 - 1.2) * -1.2,
        }
        env = _evaluate_lines(cse_and_stack(assignments), bindings)
        for name, expected in direct.items():
            assert math.isclose(env[name], expected, rel_tol=1e-12)

    def test_coefficient_scaled_products_share(self):
        u, w, z = sym("u"), sym("w"), sym("z")
        shared = mul(u, call("exp", w))
        assignments = [
            (sym("p0"), mul(num(2), shared)),
            (sym("p1"), mul(num(-3), shared)),
            (arr("out", 0), mul(shared, z)),
        ]
        stacked = cse_and_stack(assignments)
        names = [
            lhs.name
            for lhs, _ in stacked
            if hasattr(lhs, "name") and lhs.name.startswith("_cse")
        ]
        assert names, "no shared subexpression extracted"
        # exp(w)*u must be computed exactly once.
        from cubie.odesystems.symbolic.engine import (
            print_cuda_multiple,
        )

        text = "\n".join(print_cuda_multiple(stacked))
        assert text.count("math.exp(w)") == 1

    def test_numbering_continues_after_existing(self):
        x = sym("x")
        shared = call("exp", mul(x, x))
        assignments = [
            (sym("_cse4"), add(x, num(1))),
            (sym("a"), add(shared, sym("_cse4"))),
            (sym("b"), mul(shared, num(2))),
        ]
        stacked = cse_and_stack(assignments)
        new_names = {
            lhs.name
            for lhs, _ in stacked
            if hasattr(lhs, "name") and lhs.name.startswith("_cse")
        }
        assert "_cse4" in new_names
        assert "_cse5" in new_names


class TestOrderingAndPruning:
    def test_topological_sort_orders_dependencies(self):
        a, b = sym("a"), sym("b")
        ordered = topological_sort(
            [(b, add(a, num(1))), (a, num(2))]
        )
        assert [lhs for lhs, _ in ordered] == [a, b]

    def test_topological_sort_detects_cycles(self):
        a, b = sym("a"), sym("b")
        try:
            topological_sort([(a, b), (b, a)])
        except ValueError as error:
            assert "Circular" in str(error)
        else:
            raise AssertionError("cycle not detected")

    def test_prune_drops_dead_assignments(self):
        a, dead = sym("a"), sym("dead")
        pruned = prune_unused(
            [
                (a, num(2)),
                (dead, num(9)),
                (arr("out", 0), mul(a, num(3))),
            ],
            output_name="out",
        )
        assert (dead, num(9)) not in pruned

    def test_prune_without_outputs_is_noop(self):
        a = sym("a")
        assignments = [(a, num(2))]
        assert prune_unused(assignments, output_name="out") == (
            assignments
        )

    def test_free_atoms_and_count_ops(self):
        x, k = sym("x"), sym("k")
        expr = mul(x, call("sin", mul(k, x)))
        assert free_atoms(expr) == frozenset((x, k))
        assert count_ops(expr) == 3


class TestSympyRoundTrip:
    def test_round_trip_preserves_value(self):
        x, y = sp.symbols("x y", real=True)
        cases = [
            x**2 * sp.sin(y) + sp.Rational(1, 2) * x / y,
            sp.Piecewise((x, x > 0), (-x, True)),
            sp.Min(x, y) + sp.Max(x, 2 * y),
            sp.exp(-(x**2)) * sp.erf(y),
        ]
        rng = random.Random(42)
        for case in cases:
            round_tripped = to_sympy(from_sympy(case))
            for _ in range(5):
                point = {
                    x: rng.uniform(0.1, 2.0),
                    y: rng.uniform(0.1, 2.0),
                }
                expected = float(case.subs(point))
                produced = float(round_tripped.subs(point))
                assert math.isclose(
                    produced, expected, rel_tol=1e-12
                )

    def test_indexed_becomes_arr(self):
        base = sp.IndexedBase("state")
        node = from_sympy(base[2])
        assert node is arr("state", 2)

    def test_bracket_named_symbol_becomes_arr(self):
        node = from_sympy(sp.Symbol("jvp[3]"))
        assert node is arr("jvp", 3)

    def test_negation_prints_through(self):
        x = sym("x")
        assert to_sympy(neg(x)) == -sp.Symbol("x", real=True)
