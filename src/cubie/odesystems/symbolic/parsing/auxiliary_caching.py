"""Auxiliary caching planner for JVP solver helpers.

Identifies intermediate auxiliary expressions in Jacobian-vector
product computations that can be precomputed once per step and reused,
reducing redundant arithmetic across JVP evaluations without storing
the full Jacobian.

The planner is greedy and polynomial: starting from an empty
selection, it repeatedly adds the auxiliary whose caching yields the
largest marginal saving in runtime operations, subject to the slot
limit and plan validity, and stops when no addition improves the
plan. Every candidate evaluation is a linear pass over the dependency
graph, so planning time grows polynomially with system size
(issue #603).

Published Classes
-----------------
:class:`CacheGroup`
    Frozen attrs container describing one greedy addition: the leaf
    cached and the marginal savings it contributed.

:class:`CacheSelection`
    Frozen attrs container capturing the final cache plan: which
    leaves to cache, which nodes to remove from runtime, and the
    estimated savings.

Published Functions
-------------------
:func:`plan_auxiliary_cache`
    Analyse a :class:`~cubie.odesystems.symbolic.parsing.jvp_equations.JVPEquations`
    instance and persist the computed cache plan.

See Also
--------
:class:`~cubie.odesystems.symbolic.parsing.jvp_equations.JVPEquations`
    Owns the dependency metadata consumed by this module.
:mod:`cubie.odesystems.symbolic.codegen.linear_operators`
    Generates cached linear operator code using the cache plan.
"""

from typing import Optional, Sequence, Set, Tuple

import attrs

from cubie.odesystems.symbolic.engine import expr as ir
from cubie.odesystems.symbolic.parsing.jvp_equations import JVPEquations

# Candidates examined per greedy round are capped at this multiple of
# the slot limit (ranked by cumulative cost) so planning stays fast on
# very large systems.
_CANDIDATE_CAP_FACTOR = 8


@attrs.frozen
class CacheGroup:
    """Describe one greedy addition to the cache plan.

    Parameters
    ----------
    seed
        The auxiliary symbol cached by this addition.
    leaves
        Leaves cached so far, including this addition.
    removal
        Symbols removed from runtime evaluation after this addition.
    prepare
        Symbols evaluated when populating the cache after this
        addition.
    saved
        Marginal runtime operations removed by this addition.
    fill_cost
        Total cache-fill cost after this addition.
    """

    seed = attrs.field()
    leaves = attrs.field(converter=tuple)
    removal = attrs.field(converter=tuple)
    prepare = attrs.field(converter=tuple)
    saved = attrs.field()
    fill_cost = attrs.field()


@attrs.frozen
class CacheSelection:
    """Capture the final auxiliary cache plan.

    Parameters
    ----------
    groups
        Greedy additions in selection order.
    cached_leaves
        Auxiliary symbols whose values are cached.
    cached_leaf_order
        Cached leaves in evaluation order.
    removal_nodes
        Symbols removed from runtime evaluation.
    runtime_nodes
        Symbols that remain in runtime evaluation.
    prepare_nodes
        Symbols evaluated when populating the cache.
    saved
        Total operations saved by caching.
    fill_cost
        Operations required to populate the cache.
    """

    groups = attrs.field(converter=tuple)
    cached_leaves = attrs.field(converter=tuple)
    cached_leaf_order = attrs.field(converter=tuple)
    removal_nodes = attrs.field(converter=tuple)
    runtime_nodes = attrs.field(converter=tuple)
    prepare_nodes = attrs.field(converter=tuple)
    saved = attrs.field()
    fill_cost = attrs.field()


def _is_cse_symbol(node) -> bool:
    """Return whether ``node`` is a generated ``_cse`` local."""
    return isinstance(node, ir.Sym) and node.name.startswith("_cse")


def _simulate_cached_leaves(
    equations: JVPEquations,
    leaves: Sequence,
) -> Optional[Tuple[int, Set, Set, int]]:
    """Simulate caching the given leaves and compute savings.

    Parameters
    ----------
    equations
        JVP equations containing dependency and cost information.
    leaves
        Symbols to simulate caching.

    Returns
    -------
    tuple or None
        If caching is valid, returns ``(saved, removal, prepare,
        fill_cost)`` where saved is runtime operations saved, removal
        is symbols removed from runtime, prepare is symbols needed to
        fill the cache, and fill_cost is operations to populate the
        cache. Returns None when a removed node still has a live
        dependent (invalid plan).
    """
    dependencies = equations.dependencies
    dependents = equations.dependents
    ops_cost = equations.ops_cost
    ref_counts = dict(equations.reference_counts)
    removal = set()
    stack = list(leaves)
    while stack:
        node = stack.pop()
        if _is_cse_symbol(node):
            continue
        if node in removal:
            continue
        removal.add(node)
        for dep in dependencies.get(node, set()):
            ref_counts[dep] -= 1
            if ref_counts[dep] == 0:
                stack.append(dep)
    for node in removal:
        for child in dependents.get(node, set()):
            if child not in removal:
                return None
    prepare = set()
    stack = list(leaves)
    while stack:
        node = stack.pop()
        if node in prepare:
            continue
        prepare.add(node)
        stack.extend(dependencies.get(node, set()))
    saved = sum(ops_cost.get(node, 0) for node in removal)
    fill_cost = sum(ops_cost.get(node, 0) for node in prepare)
    return saved, removal, prepare, fill_cost


def _candidate_symbols(equations: JVPEquations) -> list:
    """Return cache candidates ranked by descending cumulative cost.

    Only auxiliaries that feed the JVP outputs qualify; ``_cse``
    locals are excluded (matching the removal simulation, which never
    removes them). The list is capped so greedy planning stays cheap
    on very large systems.
    """
    total_cost = equations.total_ops_cost
    order_idx = equations.order_index
    candidates = [
        symbol
        for symbol in equations.non_jvp_order
        if not _is_cse_symbol(symbol)
        and equations.jvp_closure_usage.get(symbol, 0) > 0
    ]
    candidates.sort(
        key=lambda symbol: (
            -total_cost.get(symbol, 0),
            order_idx.get(symbol, len(order_idx)),
        )
    )
    cap = max(1, _CANDIDATE_CAP_FACTOR * equations.cache_slot_limit)
    return candidates[:cap]


def _empty_selection(equations: JVPEquations) -> CacheSelection:
    """Return the no-caching plan."""
    return CacheSelection(
        groups=tuple(),
        cached_leaves=tuple(),
        cached_leaf_order=tuple(),
        removal_nodes=tuple(),
        runtime_nodes=tuple(equations.non_jvp_order),
        prepare_nodes=tuple(),
        saved=0,
        fill_cost=0,
    )


def plan_auxiliary_cache(equations: JVPEquations) -> CacheSelection:
    """Compute and persist the auxiliary cache plan for ``equations``.

    Greedily grows the cached-leaf set by the auxiliary with the
    largest marginal runtime saving until the slot limit is reached
    or no candidate improves the plan, then keeps the result only
    when the total saving meets ``min_ops_threshold``.

    Parameters
    ----------
    equations
        JVP equations to optimize with caching.

    Returns
    -------
    CacheSelection
        The computed cache plan, also stored in ``equations``.
    """
    slot_limit = equations.cache_slot_limit
    min_ops = equations.min_ops_threshold
    order_idx = equations.order_index

    selection = _empty_selection(equations)
    if slot_limit <= 0:
        equations.update_cache_selection(selection)
        return selection

    candidates = _candidate_symbols(equations)
    if not candidates:
        equations.update_cache_selection(selection)
        return selection

    chosen: list = []
    chosen_set: Set = set()
    groups: list = []
    current_saved = 0
    current = (0, set(), set(), 0)

    while len(chosen) < slot_limit:
        # While the plan is below the ops threshold, any positive
        # marginal helps it qualify; once it qualifies, each extra
        # cache slot must pay for itself with min_ops on its own.
        if current_saved < min_ops:
            required = 1
        else:
            required = min_ops
        best_symbol = None
        best_result = None
        best_key = None
        for symbol in candidates:
            if symbol in chosen_set or symbol in current[1]:
                # Already cached, or already removed as a dead
                # dependency of the cached set.
                continue
            result = _simulate_cached_leaves(
                equations, chosen + [symbol]
            )
            if result is None:
                continue
            marginal = result[0] - current_saved
            if marginal < required:
                continue
            key = (
                -marginal,
                result[3],
                order_idx.get(symbol, len(order_idx)),
            )
            if best_key is None or key < best_key:
                best_key = key
                best_symbol = symbol
                best_result = result
        if best_symbol is None:
            break
        chosen.append(best_symbol)
        chosen_set.add(best_symbol)
        current = best_result
        groups.append(
            CacheGroup(
                seed=best_symbol,
                leaves=tuple(chosen),
                removal=tuple(
                    sorted(current[1], key=order_idx.get)
                ),
                prepare=tuple(
                    sorted(current[2], key=order_idx.get)
                ),
                saved=current[0] - current_saved,
                fill_cost=current[3],
            )
        )
        current_saved = current[0]

    if not chosen or current_saved < min_ops:
        equations.update_cache_selection(selection)
        return selection

    saved, removal_set, prepare_set, fill_cost = current
    cached_order = tuple(sorted(chosen, key=order_idx.get))
    removal_order = tuple(sorted(removal_set, key=order_idx.get))
    prepare_order = tuple(sorted(prepare_set, key=order_idx.get))
    runtime_nodes = tuple(
        symbol
        for symbol in equations.non_jvp_order
        if symbol not in removal_set
    )
    selection = CacheSelection(
        groups=tuple(groups),
        cached_leaves=cached_order,
        cached_leaf_order=cached_order,
        removal_nodes=removal_order,
        runtime_nodes=runtime_nodes,
        prepare_nodes=prepare_order,
        saved=saved,
        fill_cost=fill_cost,
    )
    equations.update_cache_selection(selection)
    return selection
