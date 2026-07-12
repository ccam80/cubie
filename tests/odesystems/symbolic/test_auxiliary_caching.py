"""Unit tests for the auxiliary cache combination search.

Each test builds a real :class:`JVPEquations` from small assignment
lists whose operation counts steer the search into a specific branch
of candidate collection or combination selection.
"""

import sympy as sp

from cubie.odesystems.symbolic.parsing import JVPEquations
from cubie.odesystems.symbolic.parsing.auxiliary_caching import (
    plan_auxiliary_cache,
)


def _chain(symbol, functions, extra=None):
    """Sum of ``functions`` applied to ``symbol`` plus an optional term."""
    expr = sp.Add(*[fn(symbol) for fn in functions])
    if extra is not None:
        expr = expr + extra
    return expr


def test_zero_slot_limit_selects_nothing():
    """max_cached_terms=0 disables caching and keeps all nodes runtime."""
    x0 = sp.Symbol("x0")
    aux = sp.Symbol("aux_a")
    exprs = [
        (aux, _chain(x0, (sp.sin, sp.cos, sp.exp, sp.tan))),
        (sp.Symbol("jvp[0]"), aux * sp.Symbol("v[0]")),
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
    x0 = sp.Symbol("x0")
    dead = sp.Symbol("aux_dead")
    live = sp.Symbol("aux_live")
    exprs = [
        (dead, _chain(x0, (sp.sin, sp.cos, sp.exp, sp.tan))),
        (live, _chain(x0, (sp.sinh, sp.cosh, sp.tanh, sp.log))),
        (sp.Symbol("jvp[0]"), live * sp.Symbol("v[0]")),
    ]
    equations = JVPEquations(exprs, min_ops_threshold=1)
    selection = plan_auxiliary_cache(equations)
    assert dead not in selection.cached_leaves
    assert dead in selection.runtime_nodes
    assert live in selection.cached_leaves


def test_cheap_leaves_below_threshold_select_nothing():
    """Leaves cheaper than the ops threshold produce no candidates."""
    x0, x1 = sp.symbols("x0 x1")
    aux = sp.Symbol("aux_a")
    exprs = [
        (aux, x0 + x1),
        (sp.Symbol("jvp[0]"), aux * sp.Symbol("v[0]")),
    ]
    equations = JVPEquations(exprs, min_ops_threshold=10)
    selection = plan_auxiliary_cache(equations)
    assert selection.groups == ()
    assert selection.cached_leaves == ()
    assert aux in selection.runtime_nodes


def test_equal_saved_prefers_lower_fill_cost():
    """Among equal-saved equal-size sets the cheaper fill wins."""
    x0, x1 = sp.symbols("x0 x1")
    cse0 = sp.Symbol("_cse0")
    aux_a = sp.Symbol("aux_a")
    aux_b = sp.Symbol("aux_b")
    exprs = [
        (aux_a, _chain(x0, (sp.sin, sp.cos, sp.exp))),
        (cse0, x0 + x1),
        (aux_b, _chain(cse0, (sp.sinh, sp.cosh, sp.tanh))),
        (sp.Symbol("jvp[0]"), aux_a * sp.Symbol("v[0]")),
        (sp.Symbol("jvp[1]"), aux_b * sp.Symbol("v[1]")),
    ]
    equations = JVPEquations(exprs, min_ops_threshold=5)
    assert equations.ops_cost[aux_a] == equations.ops_cost[aux_b]
    selection = plan_auxiliary_cache(equations)
    assert set(selection.cached_leaves) == {aux_a, aux_b}


def test_slightly_smaller_saving_prefers_fewer_slots():
    """A near-best saving with fewer leaves displaces the best state."""
    x0, x1 = sp.symbols("x0 x1")
    aux_a = sp.Symbol("aux_a")
    aux_b = sp.Symbol("aux_b")
    aux_c = sp.Symbol("aux_c")
    exprs = [
        (
            aux_a,
            _chain(
                x0,
                (sp.sin, sp.cos, sp.exp, sp.tan, sp.sinh, sp.cosh),
                extra=x1,
            ),
        ),
        (aux_b, _chain(x0, (sp.tanh, sp.log, sp.asin, sp.acos), extra=x1)),
        (aux_c, _chain(x0, (sp.atan, sp.asinh, sp.acosh, sp.atanh))),
        (sp.Symbol("jvp[0]"), aux_a * sp.Symbol("v[0]")),
        (sp.Symbol("jvp[1]"), aux_b * sp.Symbol("v[1]")),
        (sp.Symbol("jvp[2]"), aux_c * sp.Symbol("v[2]")),
    ]
    equations = JVPEquations(exprs, min_ops_threshold=5)
    assert equations.ops_cost[aux_a] == 12
    assert equations.ops_cost[aux_b] == 8
    assert equations.ops_cost[aux_c] == 7
    selection = plan_auxiliary_cache(equations)
    assert set(selection.cached_leaves) == {aux_a, aux_b, aux_c}
    assert selection.saved == 27
