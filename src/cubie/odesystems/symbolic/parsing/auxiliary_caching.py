"""Auxiliary caching heuristics for symbolic solver helpers.

This module was created to find an alternative to storing the whole Jacobian
matrix for a Rosenbrock method, with the best of intentions and the eager
collaboration of an AI agent. The problem is not straightforward to solve,
and seemed an unecessary optimization when there were larger problems to
fix. The bones remain here, but they are heavily AI-inflected, as I never
got into the inner workings of the problem. They may save a few ops,
here and there, in some sytems, but otherwise will just quietly do nothing."""

from itertools import combinations
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import sympy as sp

from cubie.odesystems.symbolic.parsing.jvp_equations import JVPEquations


import attrs


@attrs.frozen
class CacheGroup:
    """Describe a group of cached leaves derived from a seed symbol.

    Parameters
    ----------
    seed
        Seed symbol used when exploring dependency chains.
    leaves
        Ordered tuple of auxiliary symbols whose values are cached.
    removal
        Ordered tuple of symbols removed from runtime evaluation.
    prepare
        Ordered tuple of symbols evaluated when populating the cache.
    saved
        Estimated number of runtime operations removed by caching the group.
    fill_cost
        Estimated number of operations required to populate the cache once.
    """

    seed = attrs.field()
    leaves = attrs.field(converter=tuple)
    removal = attrs.field(converter=tuple)
    prepare = attrs.field(converter=tuple)
    saved = attrs.field()
    fill_cost = attrs.field()


@attrs.frozen
class CacheSelection:
    """Capture the final auxiliary cache plan."""

    groups = attrs.field(converter=tuple)
    cached_leaves = attrs.field(converter=tuple)
    cached_leaf_order = attrs.field(converter=tuple)
    removal_nodes = attrs.field(converter=tuple)
    runtime_nodes = attrs.field(converter=tuple)
    prepare_nodes = attrs.field(converter=tuple)
    saved = attrs.field()
    fill_cost = attrs.field()


@attrs.frozen
class SeedSimulation:
    """Capture the outcome of simulating a cached leaf combination."""

    leaves = attrs.field(converter=tuple)
    removal = attrs.field(converter=tuple)
    prepare = attrs.field(converter=tuple)
    saved = attrs.field()
    fill_cost = attrs.field()
    meets_threshold = attrs.field()


@attrs.frozen
class SeedDiagnostics:
    """Collect diagnostics for a single seed symbol exploration."""

    seed = attrs.field()
    total_ops = attrs.field()
    closure_uses = attrs.field()
    reachable = attrs.field(converter=tuple)
    simulations = attrs.field(converter=tuple)


def _reachable_leaves(
    seed: sp.Symbol,
    dependents: Mapping[sp.Symbol, Set[sp.Symbol]],
    jvp_usage: Mapping[sp.Symbol, int],
    total_cost: Mapping[sp.Symbol, int],
    min_internal_cost: int,
) -> Set[sp.Symbol]:
    """Return high-value leaves reachable from ``seed``.

    Parameters
    ----------
    seed
        Starting auxiliary symbol.
    dependents
        Reverse dependency graph for auxiliary assignments.
    jvp_usage
        Direct JVP usage counts for auxiliary symbols.
    total_cost
        Cumulative operation counts for auxiliary symbols.
    min_internal_cost
        Minimum cumulative cost for treating an internal node as a cache
        candidate.
    """

    stack = [seed]
    visited = set()
    leaves = set()
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        if (
            jvp_usage.get(node, 0) > 0
            or total_cost.get(node, 0) >= min_internal_cost
            or node == seed
        ):
            leaves.add(node)
        for child in dependents.get(node, set()):
            stack.append(child)
    return leaves


def _prepare_nodes_for_leaves(
    leaves: Iterable[sp.Symbol],
    dependencies: Mapping[sp.Symbol, Set[sp.Symbol]],
) -> Set[sp.Symbol]:
    """Return dependencies that must execute to populate ``leaves``."""

    stack = list(leaves)
    prepare = set()
    while stack:
        node = stack.pop()
        if node in prepare:
            continue
        prepare.add(node)
        stack.extend(dependencies.get(node, set()))
    return prepare


def _simulate_cached_leaves(
    equations: JVPEquations,
    leaves: Sequence[sp.Symbol],
) -> Optional[Tuple[int, Set[sp.Symbol], Set[sp.Symbol], int]]:
    """Return savings metadata for cached ``leaves``."""

    dependencies = equations.dependencies
    dependents = equations.dependents
    ops_cost = equations.ops_cost
    ref_counts = dict(equations.reference_counts)
    removal = set()
    stack = list(leaves)
    while stack:
        node = stack.pop()
        if str(node).startswith("_cse"):
            continue
        if node in removal:
            continue
        removal.add(node)
        for dep in dependencies.get(node, set()):
            if dep not in ref_counts:
                continue
            ref_counts[dep] -= 1
            if ref_counts[dep] == 0:
                stack.append(dep)
    for node in removal:
        for child in dependents.get(node, set()):
            if child not in removal:
                return None
    prepare = _prepare_nodes_for_leaves(leaves, dependencies)
    saved = sum(ops_cost.get(node, 0) for node in removal)
    fill_cost = sum(ops_cost.get(node, 0) for node in prepare)
    return saved, removal, prepare, fill_cost


def gather_seed_diagnostics(equations: JVPEquations) -> Tuple[SeedDiagnostics, ...]:
    """Return detailed simulation diagnostics for each caching seed."""

    order_idx = equations.order_index
    total_cost = equations.total_ops_cost
    slot_limit = equations.cache_slot_limit
    if slot_limit <= 0:
        return tuple()
    dependents = equations.dependents
    jvp_usage = equations.jvp_usage
    min_ops = equations.min_ops_threshold
    min_internal_cost = max(min_ops, 1)
    diagnostics = []
    seeds = sorted(
        equations.non_jvp_order,
        key=lambda sym: (
            -total_cost.get(sym, 0),
            order_idx.get(sym, len(order_idx)),
        ),
    )
    for seed in seeds:
        closure_uses = equations.jvp_closure_usage.get(seed, 0)
        if closure_uses == 0:
            continue
        reachable = _reachable_leaves(
            seed,
            dependents,
            jvp_usage,
            total_cost,
            min_internal_cost,
        )
        if not reachable:
            continue
        ordered_leaves = tuple(
            sorted(
                reachable,
                key=lambda sym: (
                    -total_cost.get(sym, 0),
                    order_idx.get(sym, len(order_idx)),
                ),
            )
        )
        simulations = []
        max_size = min(len(ordered_leaves), slot_limit)
        for size in range(1, max_size + 1):
            for subset in combinations(ordered_leaves, size):
                simulation = _simulate_cached_leaves(
                    equations,
                    subset,
                )
                if simulation is None:
                    continue
                saved, removal, prepare, fill_cost = simulation
                meets_threshold = saved >= min_ops
                simulations.append(
                    SeedSimulation(
                        leaves=tuple(subset),
                        removal=tuple(
                            sorted(removal, key=order_idx.get)
                        ),
                        prepare=tuple(
                            sorted(prepare, key=order_idx.get)
                        ),
                        saved=saved,
                        fill_cost=fill_cost,
                        meets_threshold=meets_threshold,
                    )
                )
        diagnostics.append(
            SeedDiagnostics(
                seed=seed,
                total_ops=total_cost.get(seed, 0),
                closure_uses=closure_uses,
                reachable=ordered_leaves,
                simulations=tuple(simulations),
            )
        )
    return tuple(diagnostics)


def _collect_candidates(
    equations: JVPEquations,
) -> List[CacheGroup]:
    """Return candidate cache groups explored from each seed symbol."""

    order_idx = equations.order_index
    total_cost = equations.total_ops_cost
    slot_limit = equations.cache_slot_limit
    if slot_limit <= 0:
        return []
    dependents = equations.dependents
    jvp_usage = equations.jvp_usage
    min_ops = equations.min_ops_threshold
    min_internal_cost = max(min_ops, 1)
    candidate_map = {}
    seeds = sorted(
        equations.non_jvp_order,
        key=lambda sym: (
            -total_cost.get(sym, 0),
            order_idx.get(sym, len(order_idx)),
        ),
    )
    for seed in seeds:
        if equations.jvp_closure_usage.get(seed, 0) == 0:
            continue
        reachable = _reachable_leaves(
            seed,
            dependents,
            jvp_usage,
            total_cost,
            min_internal_cost,
        )
        if not reachable:
            continue
        ordered_leaves = sorted(
            reachable,
            key=lambda sym: (
                -total_cost.get(sym, 0),
                order_idx.get(sym, len(order_idx)),
            ),
        )
        max_size = min(len(ordered_leaves), slot_limit)
        for size in range(1, max_size + 1):
            for subset in combinations(ordered_leaves, size):
                simulation = _simulate_cached_leaves(
                    equations,
                    subset,
                )
                if simulation is None:
                    continue
                saved, removal, prepare, fill_cost = simulation
                if saved < min_ops:
                    continue
                group = CacheGroup(
                    seed=seed,
                    leaves=tuple(subset),
                    removal=tuple(
                        sorted(removal, key=order_idx.get)
                    ),
                    prepare=tuple(
                        sorted(prepare, key=order_idx.get)
                    ),
                    saved=saved,
                    fill_cost=fill_cost,
                )
                key = (
                    frozenset(subset),
                    frozenset(removal),
                )
                existing = candidate_map.get(key)
                if existing is None:
                    candidate_map[key] = group
                    continue
                if saved > existing.saved:
                    candidate_map[key] = group
                    continue
                if saved == existing.saved and fill_cost < existing.fill_cost:
                    candidate_map[key] = group
    return sorted(
        candidate_map.values(),
        key=lambda group: (
            group.saved,
            -len(group.leaves),
            -group.fill_cost,
        ),
        reverse=True,
    )


def _evaluate_leaves(
    equations: JVPEquations,
    leaves_key: frozenset,
    memo: Dict[str, Dict],
) -> Optional[Tuple[int, Set[sp.Symbol], Set[sp.Symbol], int]]:
    """Return cached evaluation metadata for the provided leaves."""

    leaves_memo = memo.setdefault("leaves", {})
    removal_memo = memo.setdefault("removal", {})
    if leaves_key in leaves_memo:
        return leaves_memo[leaves_key]
    if not leaves_key:
        result = (0, set(), set(), 0)
        leaves_memo[leaves_key] = result
        return result
    simulation = _simulate_cached_leaves(
        equations,
        tuple(leaves_key),
    )
    if simulation is None:
        leaves_memo[leaves_key] = None
        return None
    saved, removal, prepare, fill_cost = simulation
    removal_key = frozenset(removal)
    existing = removal_memo.get(removal_key)
    if existing is not None:
        leaves_memo[leaves_key] = existing
        return existing
    result = (saved, removal, prepare, fill_cost)
    leaves_memo[leaves_key] = result
    removal_memo[removal_key] = result
    return result


def _search_group_combinations(
    equations: JVPEquations,
    candidates: Sequence[CacheGroup],
) -> CacheSelection:
    """Return the optimal combination of cache groups."""

    order_idx = equations.order_index
    slot_limit = equations.cache_slot_limit
    if not candidates or slot_limit <= 0:
        runtime_nodes = tuple(equations.non_jvp_order)
        return CacheSelection(
            groups=tuple(),
            cached_leaves=tuple(),
            cached_leaf_order=tuple(),
            removal_nodes=tuple(),
            runtime_nodes=runtime_nodes,
            prepare_nodes=tuple(),
            saved=0,
            fill_cost=0,
        )

    min_ops = equations.min_ops_threshold
    memo = {"leaves": {}, "removal": {}}
    best_state = None
    stack = [(0, frozenset(), tuple())]
    while stack:
        start, leaves_key, chosen = stack.pop()
        evaluation = _evaluate_leaves(
            equations,
            leaves_key,
            memo,
        )
        if evaluation is None:
            continue
        saved, removal_set, prepare_set, fill_cost = evaluation
        if leaves_key and saved >= min_ops:
            if best_state is None:
                best_state = (
                    leaves_key,
                    chosen,
                    removal_set,
                    prepare_set,
                    saved,
                    fill_cost,
                )
            else:
                best_leaves = best_state[0]
                best_saved = best_state[4]
                best_fill = best_state[5]
                if saved > best_saved:
                    improvement = saved - best_saved
                    if improvement >= min_ops or len(leaves_key) <= len(
                        best_leaves
                    ):
                        best_state = (
                            leaves_key,
                            chosen,
                            removal_set,
                            prepare_set,
                            saved,
                            fill_cost,
                        )
                elif saved == best_saved:
                    if len(leaves_key) < len(best_leaves):
                        best_state = (
                            leaves_key,
                            chosen,
                            removal_set,
                            prepare_set,
                            saved,
                            fill_cost,
                        )
                    elif len(leaves_key) == len(best_leaves) and fill_cost < best_fill:
                        best_state = (
                            leaves_key,
                            chosen,
                            removal_set,
                            prepare_set,
                            saved,
                            fill_cost,
                        )
                else:
                    deficit = best_saved - saved
                    if deficit < min_ops and len(leaves_key) < len(best_leaves):
                        best_state = (
                            leaves_key,
                            chosen,
                            removal_set,
                            prepare_set,
                            saved,
                            fill_cost,
                        )
        for idx in range(start, len(candidates)):
            group = candidates[idx]
            new_leaves = leaves_key.union(group.leaves)
            if len(new_leaves) > slot_limit:
                continue
            stack.append((idx + 1, frozenset(new_leaves), chosen + (group,)))

    if best_state is None:
        runtime_nodes = tuple(equations.non_jvp_order)
        return CacheSelection(
            groups=tuple(),
            cached_leaves=tuple(),
            cached_leaf_order=tuple(),
            removal_nodes=tuple(),
            runtime_nodes=runtime_nodes,
            prepare_nodes=tuple(),
            saved=0,
            fill_cost=0,
        )

    (
        leaves_key,
        best_groups,
        removal_set,
        prepare_set,
        saved,
        fill_cost,
    ) = best_state
    cached_order = tuple(sorted(leaves_key, key=order_idx.get))
    removal_order = tuple(sorted(removal_set, key=order_idx.get))
    prepare_order = tuple(sorted(prepare_set, key=order_idx.get))
    runtime_nodes = tuple(
        sym for sym in equations.non_jvp_order if sym not in removal_set
    )
    return CacheSelection(
        groups=best_groups,
        cached_leaves=cached_order,
        cached_leaf_order=cached_order,
        removal_nodes=removal_order,
        runtime_nodes=runtime_nodes,
        prepare_nodes=prepare_order,
        saved=saved,
        fill_cost=fill_cost,
    )


def plan_auxiliary_cache(equations: JVPEquations) -> CacheSelection:
    """Compute and persist the auxiliary cache plan for ``equations``."""

    candidates = _collect_candidates(equations)
    selection = _search_group_combinations(
        equations,
        candidates,
    )
    equations.update_cache_selection(selection)
    return selection


def select_cached_nodes(
    equations: JVPEquations,
) -> Tuple[List[sp.Symbol], Set[sp.Symbol]]:
    """Return cached leaves and runtime nodes for ``equations``."""

    selection = equations.cache_selection
    return list(selection.cached_leaf_order), set(selection.runtime_nodes)
