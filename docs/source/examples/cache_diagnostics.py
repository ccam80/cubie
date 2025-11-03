"""Diagnostic helpers for auxiliary caching heuristics."""

from typing import Callable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import sympy as sp

from cubie.odesystems.symbolic.parsing.auxiliary_caching import (
    CacheSelection,
    gather_seed_diagnostics,
    plan_auxiliary_cache,
)
from cubie.odesystems.symbolic.codegen.jacobian import generate_analytical_jvp
from cubie.odesystems.symbolic.parsing.jvp_equations import JVPEquations
from cubie.odesystems.symbolic.symbolicODE import SymbolicODE


Assignments = Sequence[Tuple[sp.Symbol, sp.Expr]]
SystemFactory = Callable[[], SymbolicODE]
SystemInput = Union[
    SymbolicODE,
    SystemFactory,
    JVPEquations,
    Assignments,
]


def _clone_equations(
    equations: JVPEquations,
    max_cached: Optional[int],
    min_ops: Optional[int],
) -> JVPEquations:
    """Return ``equations`` with optional caching overrides applied.

    Parameters
    ----------
    equations
        Source Jacobian-vector product assignments.
    max_cached
        Optional override for the cache slot limit.
    min_ops
        Optional override for the minimum saved operations threshold.

    Returns
    -------
    JVPEquations
        Either the original instance or a copy with the overrides applied.
    """

    slot_limit = equations.cache_slot_limit
    min_threshold = equations.min_ops_threshold
    if max_cached is not None or min_ops is not None:
        slot_limit = slot_limit if max_cached is None else max_cached
        min_threshold = min_threshold if min_ops is None else min_ops
        return JVPEquations(
            equations.ordered_assignments,
            max_cached_terms=slot_limit,
            min_ops_threshold=min_threshold,
        )
    return equations


def _equations_from_symbolic(
    system: SymbolicODE,
    max_cached: Optional[int],
    min_ops: Optional[int],
) -> JVPEquations:
    """Return ``JVPEquations`` constructed from ``system``.

    Parameters
    ----------
    system
        Symbolic system providing equations and ordering metadata.
    max_cached
        Optional override for the cache slot limit.
    min_ops
        Optional override for the minimum saved operations threshold.

    Returns
    -------
    JVPEquations
        Jacobian-vector assignments compatible with caching diagnostics.
    """

    jvp_equations = generate_analytical_jvp(
        system.equations,
        input_order=system.indices.states.index_map,
        output_order=system.indices.dxdt.index_map,
        observables=system.indices.observable_symbols,
        cse=True,
    )
    return _clone_equations(jvp_equations, max_cached, min_ops)


def _equations_from_assignments(
    assignments: Assignments,
    max_cached: Optional[int],
    min_ops: Optional[int],
) -> JVPEquations:
    """Return ``JVPEquations`` constructed from ``assignments``.

    Parameters
    ----------
    assignments
        Ordered ``(lhs, rhs)`` pairs describing auxiliary and JVP terms.
    max_cached
        Optional override for the cache slot limit.
    min_ops
        Optional override for the minimum saved operations threshold.

    Returns
    -------
    JVPEquations
        Jacobian-vector assignments built from ``assignments``.
    """

    return JVPEquations(
        assignments,
        max_cached_terms=max_cached,
        min_ops_threshold=10 if min_ops is None else min_ops,
    )


def _resolve_equations(
    source: SystemInput,
    max_cached: Optional[int],
    min_ops: Optional[int],
) -> JVPEquations:
    """Return ``JVPEquations`` derived from ``source``.

    Parameters
    ----------
    source
        SymbolicODE instance, callable returning one, ``JVPEquations``
        instance, or explicit assignment sequence.
    max_cached
        Optional override for the cache slot limit.
    min_ops
        Optional override for the minimum saved operations threshold.

    Returns
    -------
    JVPEquations
        Jacobian-vector assignments compatible with caching diagnostics.

    Raises
    ------
    TypeError
        If ``source`` cannot be converted into ``JVPEquations``.
    ValueError
        If an assignment sequence is provided but empty.
    """

    if isinstance(source, JVPEquations):
        return _clone_equations(source, max_cached, min_ops)
    if isinstance(source, SymbolicODE):
        return _equations_from_symbolic(source, max_cached, min_ops)
    if callable(source):
        system = source()
        if not isinstance(system, SymbolicODE):
            raise TypeError(
                "System factory must return a SymbolicODE instance."
            )
        return _equations_from_symbolic(system, max_cached, min_ops)
    if isinstance(source, Sequence):
        if not source:
            raise ValueError("Assignment sequence must not be empty.")
        first = source[0]
        if not isinstance(first, tuple) or len(first) != 2:
            raise TypeError(
                "Assignments must be sequences of ``(lhs, rhs)`` tuples."
            )
        return _equations_from_assignments(source, max_cached, min_ops)
    raise TypeError(
        "Diagnostics require a SymbolicODE, JVPEquations, assignments, or "
        "SymbolicODE factory."
    )


def build_report(equations: JVPEquations) -> str:
    """Return formatted diagnostics for ``equations``.

    Parameters
    ----------
    equations
        Jacobian-vector assignments prepared for auxiliary caching.

    Returns
    -------
    str
        Human-readable diagnostics summarising caching behaviour.
    """

    diagnostics = gather_seed_diagnostics(equations)
    selection = plan_auxiliary_cache(equations)
    groups_by_leaves = {
        tuple(group.leaves): group for group in selection.groups
    }
    lines = [
        "Auxiliary caching diagnostics:",
        f"Cache slot limit: {equations.cache_slot_limit}",
        f"Minimum saved operations: {equations.min_ops_threshold}",
    ]
    for seed_diag in diagnostics:
        lines.append(
            f"Seed {seed_diag.seed} | total_ops={seed_diag.total_ops}"
            f" | closure_uses={seed_diag.closure_uses}"
        )
        if not seed_diag.reachable:
            lines.append("  (no reachable leaves)")
            continue
        lines.append(
            "  reachable leaves: "
            + ", ".join(str(sym) for sym in seed_diag.reachable)
        )
        if not seed_diag.simulations:
            lines.append("  (no viable simulations)")
            continue
        for simulation in seed_diag.simulations:
            leaves = ", ".join(str(sym) for sym in simulation.leaves)
            selected = tuple(simulation.leaves) in groups_by_leaves
            prefix = "*" if selected else "-"
            lines.append(
                f"  {prefix} leaves: [{leaves}] | saved={simulation.saved}"
                f" | fill={simulation.fill_cost}"
                f" | meets_threshold={simulation.meets_threshold}"
            )
            if selected:
                lines.append("      (selected in final plan)")
    lines.append("")
    lines.extend(_format_selection(selection, equations))
    return "\n".join(lines)


def run_diagnostics(
    source: SystemInput,
    max_cached: Optional[int] = None,
    min_ops: Optional[int] = None,
) -> str:
    """Return a diagnostic report for ``source``.

    Parameters
    ----------
    source
        SymbolicODE system, callable returning one, ``JVPEquations`` instance,
        or explicit assignment sequence.
    max_cached
        Optional override for the cache slot limit.
    min_ops
        Optional override for the minimum saved operations threshold.

    Returns
    -------
    str
        Human-readable diagnostics summarising caching behaviour.
    """

    equations = _resolve_equations(source, max_cached, min_ops)
    return build_report(equations)


def _format_selection(
    selection: CacheSelection,
    equations: JVPEquations,
) -> Tuple[str, ...]:
    """Return formatted summary lines for ``selection``.

    Parameters
    ----------
    selection
        Cache plan returned by :func:`plan_auxiliary_cache`.
    equations
        Jacobian-vector assignments that provide expression metadata.

    Returns
    -------
    tuple of str
        Formatted summary including cached and runtime expressions.
    """

    order_idx = equations.order_index
    expr_map = equations.non_jvp_exprs

    def _sorted_symbols(symbols: Sequence[sp.Symbol]) -> Tuple[sp.Symbol, ...]:
        return tuple(
            sorted(
                symbols,
                key=lambda sym: order_idx.get(sym, len(order_idx)),
            )
        )

    def _format_symbol(symbol: sp.Symbol) -> str:
        expr = expr_map.get(symbol)
        if expr is None:
            return f"    - {symbol} (expression unavailable)"
        return f"    - {symbol} = {sp.sstr(expr)}"

    cached_union: List[sp.Symbol] = []
    seen: Set[sp.Symbol] = set()
    for group in (selection.cached_leaves, selection.removal_nodes):
        for symbol in group:
            if symbol in seen:
                continue
            seen.add(symbol)
            cached_union.append(symbol)
    cached_symbols = _sorted_symbols(cached_union)
    prepare_symbols = _sorted_symbols(selection.prepare_nodes)
    runtime_symbols = _sorted_symbols(selection.runtime_nodes)
    runtime_set = set(selection.runtime_nodes)
    overlap_symbols = _sorted_symbols(
        tuple(sym for sym in prepare_symbols if sym in runtime_set)
    )

    lines = [
        "Final selection:",
        f"  saved operations: {selection.saved}",
        f"  fill cost: {selection.fill_cost}",
        "  cached leaves:",
    ]
    for leaf in selection.cached_leaf_order:
        lines.append(_format_symbol(leaf))
    lines.append("  cached variables (including auxiliary intermediates):")
    if cached_symbols:
        for symbol in cached_symbols:
            lines.append(_format_symbol(symbol))
    else:
        lines.append("    - (none)")
    lines.append("  removed runtime nodes:")
    if selection.removal_nodes:
        for node in _sorted_symbols(selection.removal_nodes):
            lines.append(_format_symbol(node))
    else:
        lines.append("    - (none)")
    lines.append("  preparation expressions:")
    if prepare_symbols:
        for node in prepare_symbols:
            lines.append(_format_symbol(node))
    else:
        lines.append("    - (none)")
    lines.append("  runtime expressions:")
    if runtime_symbols:
        for node in runtime_symbols:
            lines.append(_format_symbol(node))
    else:
        lines.append("    - (none)")
    lines.append("  expressions evaluated during both preparation and runtime:")
    if overlap_symbols:
        for node in overlap_symbols:
            lines.append(_format_symbol(node))
    else:
        lines.append("    - (none)")
    return tuple(lines)

if __name__ == "__main__":
    from cubie.odesystems.symbolic.symbolicODE import SymbolicODE, create_ODE_system

    THREE_STATE_VERY_STIFF_EQUATIONS = [
        "dx0 = -k1 * (x0 - x1) - n0 * x0**3 + d0",
        "dx1 = k1 * (x0 - x1) - k2 * (x1 - x2) - n1 * x1**3",
        "dx2 = k2 * (x1 - x2) - k3 * (x2 - c0) - n2 * x2**3",
        "r0 = x0 - x1",
        "r1 = x1 - x2",
        "r2 = x0 + x1 + x2",
    ]

    THREE_STATE_VERY_STIFF_STATES = {"x0": 0.5, "x1": 0.25, "x2": 0.1}
    THREE_STATE_VERY_STIFF_PARAMETERS = {
        "k1": 150.0,
        "k2": 900.0,
        "k3": 1200.0,
        "n0": 40.0,
        "n1": 30.0,
        "n2": 20.0,
    }
    THREE_STATE_VERY_STIFF_CONSTANTS = {"c0": 0.5}
    THREE_STATE_VERY_STIFF_DRIVERS = ["d0"]
    THREE_STATE_VERY_STIFF_OBSERVABLES = ["r0", "r1", "r2"]
    stiff_system = create_ODE_system(
        dxdt=THREE_STATE_VERY_STIFF_EQUATIONS,
        states=THREE_STATE_VERY_STIFF_STATES,
        parameters=THREE_STATE_VERY_STIFF_PARAMETERS,
        constants=THREE_STATE_VERY_STIFF_CONSTANTS,
        drivers=THREE_STATE_VERY_STIFF_DRIVERS,
        observables=THREE_STATE_VERY_STIFF_OBSERVABLES,
        precision=np.float32,
        name="three_state_very_stiff",
        strict=True,
    )

    THREE_CHAMBER_EQUATIONS = [
        "P_a = E_a * V_a",
        "P_v = E_v * V_v",
        "P_h = E_h * V_h * d1",
        "Q_i = (P_v - P_h) / R_i if P_v > P_h else 0",
        "Q_o = (P_h - P_a) / R_o if P_h > P_a else 0",
        "Q_c = (P_a - P_v) / R_c",
        "dV_h = Q_i - Q_o",
        "dV_a = Q_o - Q_c",
        "dV_v = Q_c - Q_i",
    ]

    THREE_CHAMBER_STATES = {"V_h": 1.0, "V_a": 1.0, "V_v": 1.0}
    THREE_CHAMBER_PARAMETERS = {
        "E_h": 0.52,
        "E_a": 0.0133,
        "E_v": 0.0624,
        "R_i": 0.012,
        "R_o": 1.0,
        "R_c": 1.0 / 114.0,
        "V_s3": 2.0,
    }
    THREE_CHAMBER_CONSTANTS: dict[str, float] = {}
    THREE_CHAMBER_DRIVERS = ["d1"]
    THREE_CHAMBER_OBSERVABLES = ["P_a", "P_v", "P_h", "Q_i", "Q_o", "Q_c"]

    threecm = create_ODE_system(
        dxdt=THREE_CHAMBER_EQUATIONS,
        states=THREE_CHAMBER_STATES,
        parameters=THREE_CHAMBER_PARAMETERS,
        constants=THREE_CHAMBER_CONSTANTS,
        drivers=THREE_CHAMBER_DRIVERS,
        observables=THREE_CHAMBER_OBSERVABLES,
        precision=np.float32,
        name="three_chamber_system",
        strict=True,
    )

    stiff_report = run_diagnostics(stiff_system, 6, 5)
    print(stiff_report)
    # threecm_report = run_diagnostics(threecm, 6, 5)
    # print(threecm_report)