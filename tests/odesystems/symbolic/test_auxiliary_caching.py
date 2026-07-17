"""Unit tests for the greedy auxiliary cache planner.

Each test builds a real JVPEquations instance from small assignment
lists whose operation counts steer the planner into a specific
selection outcome. Expressions are built directly on the engine IR.
"""

from cubie.odesystems.symbolic.engine import expr as ir
from cubie.odesystems.symbolic.parsing import JVPEquations
from cubie.odesystems.symbolic.parsing.auxiliary_caching import (
    plan_auxiliary_cache,
)


def _chain(symbol, functions, extra=None):
    """Sum of ``functions`` applied to ``symbol`` plus an optional term."""
    terms = [ir.call(name, symbol) for name in functions]
    if extra is not None:
        terms.append(extra)
    return ir.add(*terms)


def test_zero_slot_limit_selects_nothing():
    """max_cached_terms=0 disables caching and keeps all nodes runtime."""
    x0 = ir.sym("x0")
    aux = ir.sym("aux_a")
    exprs = [
        (aux, _chain(x0, ("sin", "cos", "exp", "tan"))),
        (ir.arr("jvp", 0), ir.mul(aux, ir.arr("v", 0))),
    ]
    equations = JVPEquations(
        exprs, max_cached_terms=0, min_ops_threshold=1
    )
    selection = plan_auxiliary_cache(equations)
    assert selection.groups == ()
    assert selection.cached_leaves == ()
    assert selection.runtime_nodes == tuple(equations.non_jvp_order)


def test_dead_auxiliary_is_never_a_seed():
    """An assignment feeding no jvp term stays runtime and uncached."""
    x0 = ir.sym("x0")
    dead = ir.sym("aux_dead")
    live = ir.sym("aux_live")
    exprs = [
        (dead, _chain(x0, ("sin", "cos", "exp", "tan"))),
        (live, _chain(x0, ("sinh", "cosh", "tanh", "log"))),
        (ir.arr("jvp", 0), ir.mul(live, ir.arr("v", 0))),
    ]
    equations = JVPEquations(exprs, min_ops_threshold=1)
    selection = plan_auxiliary_cache(equations)
    assert dead not in selection.cached_leaves
    assert dead in selection.runtime_nodes
    assert live in selection.cached_leaves


def test_cheap_leaves_below_threshold_select_nothing():
    """Leaves cheaper than the ops threshold produce no selection."""
    x0, x1 = ir.sym("x0"), ir.sym("x1")
    aux = ir.sym("aux_a")
    exprs = [
        (aux, ir.add(x0, x1)),
        (ir.arr("jvp", 0), ir.mul(aux, ir.arr("v", 0))),
    ]
    equations = JVPEquations(exprs, min_ops_threshold=10)
    selection = plan_auxiliary_cache(equations)
    assert selection.groups == ()
    assert selection.cached_leaves == ()
    assert aux in selection.runtime_nodes


def test_cse_locals_are_prepared_but_never_cached():
    """A ``_cse`` local feeds the cache fill but is not itself cached."""
    x0, x1 = ir.sym("x0"), ir.sym("x1")
    cse0 = ir.sym("_cse0")
    aux_a = ir.sym("aux_a")
    aux_b = ir.sym("aux_b")
    exprs = [
        (aux_a, _chain(x0, ("sin", "cos", "exp"))),
        (cse0, ir.add(x0, x1)),
        (aux_b, _chain(cse0, ("sinh", "cosh", "tanh"))),
        (ir.arr("jvp", 0), ir.mul(aux_a, ir.arr("v", 0))),
        (ir.arr("jvp", 1), ir.mul(aux_b, ir.arr("v", 1))),
    ]
    equations = JVPEquations(exprs, min_ops_threshold=5)
    assert equations.ops_cost[aux_a] == equations.ops_cost[aux_b]
    selection = plan_auxiliary_cache(equations)
    assert set(selection.cached_leaves) == {aux_a, aux_b}
    assert cse0 not in selection.cached_leaves
    assert cse0 in selection.prepare_nodes


def test_all_profitable_leaves_are_cached():
    """Every auxiliary with positive savings is selected greedily."""
    x0, x1 = ir.sym("x0"), ir.sym("x1")
    aux_a = ir.sym("aux_a")
    aux_b = ir.sym("aux_b")
    aux_c = ir.sym("aux_c")
    exprs = [
        (
            aux_a,
            _chain(
                x0,
                ("sin", "cos", "exp", "tan", "sinh", "cosh"),
                extra=x1,
            ),
        ),
        (
            aux_b,
            _chain(x0, ("tanh", "log", "asin", "acos"), extra=x1),
        ),
        (aux_c, _chain(x0, ("atan", "asinh", "acosh", "atanh"))),
        (ir.arr("jvp", 0), ir.mul(aux_a, ir.arr("v", 0))),
        (ir.arr("jvp", 1), ir.mul(aux_b, ir.arr("v", 1))),
        (ir.arr("jvp", 2), ir.mul(aux_c, ir.arr("v", 2))),
    ]
    equations = JVPEquations(exprs, min_ops_threshold=5)
    assert equations.ops_cost[aux_a] == 12
    assert equations.ops_cost[aux_b] == 8
    assert equations.ops_cost[aux_c] == 7
    selection = plan_auxiliary_cache(equations)
    assert set(selection.cached_leaves) == {aux_a, aux_b, aux_c}
    assert selection.saved == 27
    # Greedy order: largest marginal saving first.
    assert [group.seed for group in selection.groups] == [
        aux_a,
        aux_b,
        aux_c,
    ]


def test_intermediate_with_live_dependent_is_not_cached_alone():
    """A node consumed by a live auxiliary cannot be cached by itself,
    but caching its consumer first makes it eligible."""
    x0 = ir.sym("x0")
    inner = ir.sym("aux_inner")
    outer = ir.sym("aux_outer")
    exprs = [
        (inner, _chain(x0, ("sin", "cos", "exp", "tan"))),
        (outer, _chain(inner, ("sinh", "cosh", "tanh", "log"))),
        (
            ir.arr("jvp", 0),
            ir.mul(outer, ir.arr("v", 0)),
        ),
        (
            ir.arr("jvp", 1),
            ir.mul(inner, ir.arr("v", 1)),
        ),
    ]
    equations = JVPEquations(exprs, min_ops_threshold=1)
    selection = plan_auxiliary_cache(equations)
    # Both are jvp-used; caching outer removes it, then inner becomes
    # cacheable too (its only remaining consumers are jvp terms).
    assert set(selection.cached_leaves) == {inner, outer}
    assert selection.saved == equations.ops_cost[inner] + (
        equations.ops_cost[outer]
    )


def test_planner_terminates_on_wide_systems():
    """Planning stays fast when many auxiliaries feed many outputs.

    The subset-enumeration planner this replaces did not terminate
    for systems of this width (issue 603).
    """
    import time

    n = 48
    exprs = []
    for i in range(n):
        aux = ir.sym(f"aux_{i}")
        exprs.append(
            (
                aux,
                _chain(
                    ir.sym(f"x{i}"),
                    ("sin", "cos", "exp", "tan"),
                ),
            )
        )
        exprs.append(
            (
                ir.arr("jvp", i),
                ir.mul(aux, ir.arr("v", i)),
            )
        )
    equations = JVPEquations(exprs, min_ops_threshold=1)
    started = time.perf_counter()
    selection = plan_auxiliary_cache(equations)
    elapsed = time.perf_counter() - started
    assert elapsed < 10.0
    assert len(selection.cached_leaves) == len(
        selection.cached_leaf_order
    )
    assert selection.saved > 0
