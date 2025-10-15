"""Auxiliary caching heuristics for symbolic solver helpers."""

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import sympy as sp

from cubie.odesystems.symbolic.jvp_equations import JVPEquations

def is_cse_symbol(symbol: sp.Symbol) -> bool:
    """Return True if ``symbol`` starts with _cse."""
    return str(symbol).startswith("_cse")


def simulate_removal(
    symbol: sp.Symbol,
    active_nodes: Set[sp.Symbol],
    current_ref_counts: Dict[sp.Symbol, int],
    equations: JVPEquations,
    cse_depth: Optional[int] = None,
) -> Tuple[int, Set[sp.Symbol], Set[sp.Symbol]]:
    """Estimate saved operations when removing a symbol from active nodes.

    Parameters
    ----------
    symbol
        Candidate auxiliary assignment considered for caching.
    active_nodes
        Symbols still present in the runtime evaluation order.
    current_ref_counts
        Reference count map for active symbols, including JVP usage.
    equations
        Structured view of the Jacobian-vector product assignments providing
        dependency graphs and operation counts.
    cse_depth
        Maximum number of ``"_cse"`` dependency layers traversed when
        exploring common subexpression closures. ``None`` traverses the full
        dependency graph.

    Returns
    -------
    tuple of int and two sets
        Estimated operations saved, the dependency closure removed from the
        active set, and the leaf symbols that require caching to enable the
        removal.
    """

    if symbol not in active_nodes:
        return 0, set(), set()
    dependencies = equations.dependencies
    dependents = equations.dependents
    ops_cost = equations.ops_cost
    temp_counts = current_ref_counts.copy()
    to_remove = set()
    stack = [(symbol, cse_depth, True)]
    cache_group = set()
    while stack:
        node, depth_left, cache_member = stack.pop()
        if node in to_remove or node not in active_nodes:
            continue
        to_remove.add(node)
        if (
            cache_member
            and not is_cse_symbol(node)
            and not dependents.get(node)
        ):
            cache_group.add(node)
        if is_cse_symbol(node):
            if depth_left is None or depth_left >= 0:
                for child in dependents.get(node, set()):
                    if child in active_nodes and child not in to_remove:
                        stack.append((child, depth_left, True))
        for dep in dependencies.get(node, set()):
            if dep not in temp_counts:
                continue
            temp_counts[dep] -= 1
            if dep in active_nodes and temp_counts[dep] == 0:
                if is_cse_symbol(dep) and depth_left is not None:
                    next_depth = depth_left - 1
                    if next_depth < 0:
                        continue
                elif is_cse_symbol(dep):
                    next_depth = None
                else:
                    next_depth = depth_left
                stack.append((dep, next_depth, False))
            elif is_cse_symbol(dep):
                if depth_left is None:
                    dep_depth = None
                else:
                    dep_depth = depth_left - 1
                    if dep_depth < 0:
                        continue
                for child in dependents.get(dep, set()):
                    if child in active_nodes and child not in to_remove:
                        stack.append((child, dep_depth, True))
    saved = sum(
        ops_cost.get(node, 0) for node in to_remove if node in active_nodes
    )
    return saved, to_remove, cache_group


def max_cse_depth(
    symbol: sp.Symbol,
    equations: JVPEquations,
    memo: Optional[Dict[sp.Symbol, int]] = None,
) -> int:
    """Return the maximum number of CSE layers reachable from ``symbol``.

    Parameters
    ----------
    symbol
        Auxiliary assignment symbol explored for upstream CSE depth.
    equations
        Structured view of the Jacobian-vector product assignments providing
        dependency graphs.
    memo
        Optional memoisation dictionary reused across invocations.

    Returns
    -------
    int
        Maximum number of ``"_cse"`` layers encountered along any upstream
        dependency path.
    """

    if memo is None:
        memo = {}
    if symbol in memo:
        return memo[symbol]
    dependencies = equations.dependencies
    max_depth = 0
    for dep in dependencies.get(symbol, set()):
        depth = max_cse_depth(dep, equations, memo)
        if str(dep).startswith("_cse"):
            depth += 1
        if depth > max_depth:
            max_depth = depth
    memo[symbol] = max_depth
    return max_depth


@dataclass(frozen=True)
class CacheCandidate:
    """Describe a potential group of auxiliaries to cache.

    Parameters
    ----------
    symbol
        Seed symbol used when simulating removal.
    depth
        Maximum ``"_cse"`` depth traversed while forming the group.
    saved
        Estimated operations saved if the group is cached.
    removal
        Symbols removed from the runtime evaluation order by caching the
        group.
    cache_nodes
        Leaf symbols whose values must be stored in the cache.
    """

    symbol: sp.Symbol
    depth: Optional[int]
    saved: int
    removal: Tuple[sp.Symbol, ...]
    cache_nodes: Tuple[sp.Symbol, ...]


def generate_cache_candidates(
    equations: JVPEquations,
    active_nodes: Set[sp.Symbol],
    current_ref_counts: Dict[sp.Symbol, int],
    depth_memo: Optional[Dict[sp.Symbol, int]] = None,
) -> List[CacheCandidate]:
    """Construct cache candidates by simulating removals for each symbol.

    Parameters
    ----------
    active_nodes
        Symbols still considered for runtime execution.
    current_ref_counts
        Reference count map incorporating JVP usage.
    equations
        Structured view of the Jacobian-vector product assignments providing
        dependency graphs and cost metadata.
    depth_memo
        Optional memo dict reused for :func:`max_cse_depth` lookups.

    Returns
    -------
    list of CacheCandidate
        Cache candidates ordered by descending savings.
    """

    if depth_memo is None:
        depth_memo = {}
    candidate_map = {}
    non_jvp_order = equations.non_jvp_order
    dependents = equations.dependents
    max_cached_terms = equations.cache_slot_limit
    min_ops_threshold = equations.min_ops_threshold
    for symbol in non_jvp_order:
        if symbol not in active_nodes:
            continue
        max_depth = max_cse_depth(symbol, equations, depth_memo)
        depth_options = list(range(max_depth, -1, -1)) or [0]
        for depth in depth_options:
            saved, removal, group_nodes = simulate_removal(
                symbol,
                active_nodes,
                current_ref_counts,
                equations,
                cse_depth=depth,
            )
            cache_nodes = [
                node
                for node in non_jvp_order
                if node in group_nodes
                and node in active_nodes
                and not is_cse_symbol(node)
                and not dependents.get(node)
            ]
            group_size = len(cache_nodes)
            if (
                group_size == 0
                or group_size > max_cached_terms
                or saved < min_ops_threshold
            ):
                continue
            key = frozenset(cache_nodes)
            removal_tuple = tuple(
                node for node in removal if node in active_nodes
            )
            candidate = CacheCandidate(
                symbol=symbol,
                depth=depth,
                saved=saved,
                removal=removal_tuple,
                cache_nodes=tuple(cache_nodes),
            )
            existing = candidate_map.get(key)
            if existing is None or saved > existing.saved:
                candidate_map[key] = candidate
    candidates = sorted(
        candidate_map.values(), key=lambda cand: cand.saved, reverse=True
    )
    return list(candidates)


def evaluate_candidate_combinations(
    equations: JVPEquations,
    candidates: Sequence[CacheCandidate],
) -> List[CacheCandidate]:
    """Return the candidate subset that maximises saved operations.

    Parameters
    ----------
    candidates
        Candidate cache groups generated by :func:`generate_cache_candidates`.
    equations
        Structured view of the Jacobian-vector product assignments providing
        operation cost metadata and cache size limits.

    Returns
    -------
    list of CacheCandidate
        Candidate subset whose cached leaves fit within the slot budget while
        maximising saved operations.
    """

    best_saved_total = 0
    best_selection = []

    ops_cost = equations.ops_cost
    max_cached_terms = equations.cache_slot_limit

    def dfs(
        start: int,
        cached_union: Set[sp.Symbol],
        removal_union: Set[sp.Symbol],
        saved_total: int,
        chosen: List[CacheCandidate],
    ) -> None:
        nonlocal best_saved_total, best_selection
        if saved_total > best_saved_total:
            best_saved_total = saved_total
            best_selection = list(chosen)
        for idx in range(start, len(candidates)):
            candidate = candidates[idx]
            next_cached = cached_union | set(candidate.cache_nodes)
            if len(next_cached) > max_cached_terms:
                continue
            new_nodes = set(candidate.removal) - removal_union
            increment = sum(ops_cost.get(node, 0) for node in new_nodes)
            next_saved = saved_total + increment
            next_removal = removal_union | set(candidate.removal)
            chosen.append(candidate)
            dfs(idx + 1, next_cached, next_removal, next_saved, chosen)
            chosen.pop()

    dfs(0, set(), set(), 0, [])
    return best_selection


def select_cached_nodes(
    equations: JVPEquations,
) -> Tuple[List[sp.Symbol], Set[sp.Symbol]]:
    """Select auxiliary nodes to cache based on estimated savings.

    Parameters
    ----------
    equations
        Structured view of the Jacobian-vector product assignments providing
        dependency graphs and cost metadata.

    Returns
    -------
    tuple of list and set
        Cached auxiliary nodes ordered by evaluation sequence and remaining
        runtime nodes after the cached removals are applied.
    """

    non_jvp_order = equations.non_jvp_order
    dependents = equations.dependents
    jvp_usage = equations.jvp_usage
    active_nodes = set(non_jvp_order)
    current_ref_counts = {}
    for sym in non_jvp_order:
        current_ref_counts[sym] = len(dependents[sym]) + jvp_usage.get(sym, 0)
    depth_memo = {}
    candidates = generate_cache_candidates(
        equations,
        active_nodes,
        current_ref_counts,
        depth_memo,
    )
    if not candidates:
        return [], active_nodes
    selected = evaluate_candidate_combinations(equations, candidates)
    if not selected:
        return [], active_nodes
    cached_union = set()
    for candidate in selected:
        cached_union.update(candidate.cache_nodes)
    cached_nodes = [sym for sym in non_jvp_order if sym in cached_union]
    removed_union = set()
    for candidate in selected:
        removed_union.update(candidate.removal)
    remaining = {sym for sym in non_jvp_order if sym not in removed_union}
    return cached_nodes, remaining

