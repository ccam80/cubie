"""Auxiliary caching heuristics for symbolic solver helpers."""

from itertools import combinations
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import sympy as sp

from cubie.odesystems.symbolic.jvp_equations import JVPEquations


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
        Estimated number of operations saved by caching the group.
    """

    seed = attrs.field()
    leaves = attrs.field(converter=tuple)
    removal = attrs.field(converter=tuple)
    prepare = attrs.field(converter=tuple)
    saved = attrs.field()


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
def _reachable_leaves(
    seed: sp.Symbol,
    dependents: Mapping[sp.Symbol, Set[sp.Symbol]],
    jvp_usage: Mapping[sp.Symbol, int],
) -> Set[sp.Symbol]:
    """Return JVP-dependent leaves reachable from ``seed``."""

    stack = [seed]
    visited = set()
    leaves = set()
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        if jvp_usage.get(node, 0) > 0:
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
) -> Optional[Tuple[int, Set[sp.Symbol]]]:
    """Return saved operations and removed nodes for cached ``leaves``."""

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
    saved = sum(ops_cost.get(node, 0) for node in removal)
    return saved, removal


def _collect_candidates(
    equations: JVPEquations,
) -> List[CacheGroup]:
    """Return candidate cache groups explored from each seed symbol."""

    order_idx = equations.order_index
    slot_limit = equations.cache_slot_limit
    if slot_limit <= 0:
        return []
    dependents = equations.dependents
    dependencies = equations.dependencies
    jvp_usage = equations.jvp_usage
    min_ops = equations.min_ops_threshold
    candidate_map = {}
    for seed in equations.non_jvp_order:
        if equations.jvp_closure_usage.get(seed, 0) == 0:
            continue
        reachable = _reachable_leaves(seed, dependents, jvp_usage)
        if not reachable:
            continue
        ordered_leaves = sorted(reachable, key=order_idx.get)
        max_size = min(len(ordered_leaves), slot_limit)
        for size in range(1, max_size + 1):
            for subset in combinations(ordered_leaves, size):
                simulation = _simulate_cached_leaves(
                    equations,
                    subset,
                )
                if simulation is None:
                    continue
                saved, removal = simulation
                if saved < min_ops:
                    continue
                prepare_nodes = _prepare_nodes_for_leaves(subset, dependencies)
                group = CacheGroup(
                    seed=seed,
                    leaves=tuple(subset),
                    removal=tuple(
                        sorted(removal, key=order_idx.get)
                    ),
                    prepare=tuple(
                        sorted(prepare_nodes, key=order_idx.get)
                    ),
                    saved=saved,
                )
                key = frozenset(subset)
                existing = candidate_map.get(key)
                if existing is None or saved > existing.saved:
                    candidate_map[key] = group
    return sorted(
        candidate_map.values(), key=lambda group: group.saved, reverse=True
    )


def _evaluate_leaves(
    equations: JVPEquations,
    leaves_key: frozenset,
    dependencies: Mapping[sp.Symbol, Set[sp.Symbol]],
    memo: Dict[
        frozenset,
        Optional[Tuple[int, Set[sp.Symbol], Set[sp.Symbol]]],
    ],
) -> Optional[Tuple[int, Set[sp.Symbol], Set[sp.Symbol]]]:
    """Return cached evaluation metadata for the provided leaves."""

    if leaves_key in memo:
        return memo[leaves_key]
    if not leaves_key:
        result = (0, set(), set())
        memo[leaves_key] = result
        return result
    simulation = _simulate_cached_leaves(
        equations,
        tuple(leaves_key),
    )
    if simulation is None:
        memo[leaves_key] = None
        return None
    saved, removal = simulation
    prepare = _prepare_nodes_for_leaves(leaves_key, dependencies)
    result = (saved, removal, prepare)
    memo[leaves_key] = result
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
        )

    dependencies = equations.dependencies
    min_ops = equations.min_ops_threshold
    memo = {}
    best_state = None
    stack = [(0, frozenset(), tuple())]
    while stack:
        start, leaves_key, chosen = stack.pop()
        evaluation = _evaluate_leaves(
            equations,
            leaves_key,
            dependencies,
            memo,
        )
        if evaluation is None:
            continue
        saved, removal_set, prepare_set = evaluation
        if leaves_key and saved >= min_ops:
            if best_state is None:
                best_state = (
                    leaves_key,
                    chosen,
                    removal_set,
                    prepare_set,
                    saved,
                )
            else:
                best_leaves = best_state[0]
                best_saved = best_state[-1]
                if saved > best_saved:
                    improvement = saved - best_saved
                    if improvement >= min_ops or len(leaves_key) <= len(best_leaves):
                        best_state = (
                            leaves_key,
                            chosen,
                            removal_set,
                            prepare_set,
                            saved,
                        )
                elif saved == best_saved:
                    if len(leaves_key) < len(best_leaves):
                        best_state = (
                            leaves_key,
                            chosen,
                            removal_set,
                            prepare_set,
                            saved,
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
        )

    leaves_key, best_groups, removal_set, prepare_set, saved = best_state
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
