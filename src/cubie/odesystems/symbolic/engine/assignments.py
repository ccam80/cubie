"""Assignment-list transforms over engine IR expressions.

Operates on ordered lists of ``(lhs, rhs)`` IR pairs — the working
representation every generator builds before printing. Provides the
IR equivalents of the SymPy-era helpers: topological sorting,
dead-assignment pruning, and common-subexpression extraction.

Published Functions
-------------------
:func:`topological_sort`
    Order assignments so dependencies precede their uses.
:func:`prune_unused`
    Drop assignments that do not feed the requested outputs.
:func:`cse_and_stack`
    Extract shared subexpressions into ``_cse<i>`` locals and return
    the combined, dependency-ordered assignment list.

Notes
-----
Because IR nodes are hash-consed, "common subexpression" means
"IR node referenced more than once" — detection is a single
reference-counting pass over the DAG, not a search.
"""

from typing import Dict, Iterable, List, Optional, Set, Tuple

from cubie.odesystems.symbolic.engine.expr import (
    Add as AddNode,
    Arr,
    Call,
    Expr,
    Mul as MulNode,
    Num,
    Sym,
    _children,
    _rebuild,
    add,
    free_atoms,
    mul,
    sym,
)

__all__ = [
    "topological_sort",
    "prune_unused",
    "cse_and_stack",
]

Assignment = Tuple[Expr, Expr]


def topological_sort(
    assignments: Iterable[Assignment],
) -> List[Assignment]:
    """Order assignments so every dependency precedes its uses.

    Parameters
    ----------
    assignments
        ``(lhs, rhs)`` pairs; each ``lhs`` is a :class:`Sym` or
        :class:`Arr` node.

    Returns
    -------
    list of tuple
        Dependency-ordered assignments. Ties keep input order first,
        then break deterministically by the dependent's sort key —
        never by hash order.

    Raises
    ------
    ValueError
        When a dependency cycle prevents ordering.
    """
    pairs = list(assignments)
    sym_map: Dict[Expr, Expr] = {lhs: rhs for lhs, rhs in pairs}
    order_index = {lhs: i for i, (lhs, _) in enumerate(pairs)}
    assignees = set(sym_map)

    incoming: Dict[Expr, int] = {}
    dependents: Dict[Expr, List[Expr]] = {}
    for lhs, rhs in pairs:
        deps = free_atoms(rhs) & assignees
        incoming[lhs] = len(deps)
        for dep in deps:
            dependents.setdefault(dep, []).append(lhs)

    ready = [lhs for lhs, _ in pairs if incoming[lhs] == 0]
    result: List[Assignment] = []
    cursor = 0
    while cursor < len(ready):
        current = ready[cursor]
        cursor += 1
        result.append((current, sym_map[current]))
        waiters = dependents.get(current)
        if not waiters:
            continue
        released = []
        for waiter in waiters:
            incoming[waiter] -= 1
            if incoming[waiter] == 0:
                released.append(waiter)
        released.sort(key=lambda node: order_index[node])
        ready.extend(released)

    if len(result) != len(pairs):
        remaining = assignees - {lhs for lhs, _ in result}
        names = sorted(str(node) for node in remaining)
        raise ValueError(
            f"Circular dependency detected. Remaining symbols: {names}"
        )
    return result


def prune_unused(
    assignments: Iterable[Assignment],
    output_name: Optional[str] = None,
    output_symbols: Optional[Iterable[Expr]] = None,
) -> List[Assignment]:
    """Drop assignments that do not feed the requested outputs.

    Parameters
    ----------
    assignments
        Topologically ordered ``(lhs, rhs)`` pairs.
    output_name
        Array name identifying outputs: every ``Arr(output_name, i)``
        left-hand side is an output. Ignored when ``output_symbols``
        is given.
    output_symbols
        Explicit output left-hand sides to retain.

    Returns
    -------
    list of tuple
        The assignments transitively required by the outputs, in
        their original relative order. Returned unchanged when no
        output matches (mirrors the SymPy-era behaviour).
    """
    pairs = list(assignments)
    if not pairs:
        return pairs
    all_lhs = {lhs for lhs, _ in pairs}
    if output_symbols is not None:
        outputs = set(output_symbols) & all_lhs
    else:
        outputs = {
            lhs
            for lhs in all_lhs
            if isinstance(lhs, Arr) and lhs.name == output_name
        }
    if not outputs:
        return pairs

    used: Set[Expr] = set(outputs)
    kept: List[Assignment] = []
    for lhs, rhs in reversed(pairs):
        if lhs in used:
            kept.append((lhs, rhs))
            used.update(free_atoms(rhs) & all_lhs)
    kept.reverse()
    return kept


def _is_extractable(node: Expr) -> bool:
    """Return whether a shared node is worth naming as a CSE local."""
    if isinstance(node, (Sym, Arr, Num)):
        return False
    if isinstance(node, Call) and not node.args:
        return False
    return True


# Args appearing in more than this many products/sums are too generic
# to seed subset matching (think ``h`` multiplying every JVP term);
# pairing them would cost O(n^2) for near-zero sharing value.
_SUBSET_PAIR_CAP = 100


def _find_partial_subsets(
    nodes: List[Expr],
    raw_build,
) -> Dict[Expr, Tuple[Expr, Tuple[Expr, ...]]]:
    """Match shared argument subsets across n-ary Add/Mul nodes.

    Flattening destroys nested sharing: ``2*e*a`` interns as
    ``Mul(2, e, a)``, which does not contain the shared ``Mul(e, a)``
    as a child. This pass finds argument subsets (size >= 2, numeric
    coefficients excluded) common to at least two nodes and assigns
    each node its largest such subset.

    Parameters
    ----------
    nodes
        Distinct Add or Mul nodes in first-appearance order.
    raw_build
        Constructor building the interned subset node from a tuple of
        two-or-more args.

    Returns
    -------
    dict
        Mapping from node to ``(subset_node, remaining_args)``.
    """
    arg_sets: Dict[Expr, frozenset] = {}
    by_arg: Dict[Expr, List[int]] = {}
    for position, node in enumerate(nodes):
        significant = frozenset(
            a for a in node.args if not isinstance(a, Num)
        )
        arg_sets[node] = significant
        for argument in significant:
            by_arg.setdefault(argument, []).append(position)

    # Candidate subsets from pairwise intersections, discovered via
    # the shared-argument index so unrelated nodes never pair up.
    best: Dict[int, frozenset] = {}
    seen_pairs: Set[Tuple[int, int]] = set()
    for argument, positions in by_arg.items():
        if len(positions) < 2 or len(positions) > _SUBSET_PAIR_CAP:
            continue
        for i, pos_a in enumerate(positions):
            set_a = arg_sets[nodes[pos_a]]
            for pos_b in positions[i + 1:]:
                pair = (pos_a, pos_b)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                common = set_a & arg_sets[nodes[pos_b]]
                if len(common) < 2:
                    continue
                for position in pair:
                    node_args = arg_sets[nodes[position]]
                    if common == node_args and len(
                        nodes[position].args
                    ) == len(common):
                        # The node IS the subset; plain counting
                        # already handles whole-node sharing.
                        continue
                    current = best.get(position)
                    if current is None or len(common) > len(current):
                        best[position] = common
                    elif current is not None and len(common) == len(
                        current
                    ) and common is not current:
                        # Deterministic tie-break by sort key.
                        chosen = min(
                            (
                                tuple(
                                    sorted(
                                        a.sort_key for a in common
                                    )
                                ),
                                common,
                            ),
                            (
                                tuple(
                                    sorted(
                                        a.sort_key for a in current
                                    )
                                ),
                                current,
                            ),
                        )[1]
                        best[position] = chosen

    adopted: Dict[Expr, Tuple[Expr, Tuple[Expr, ...]]] = {}
    for position, subset in best.items():
        node = nodes[position]
        subset_args = tuple(
            sorted(subset, key=lambda a: a.sort_key)
        )
        subset_node = raw_build(subset_args)
        if subset_node is node:
            continue
        remaining = tuple(
            a for a in node.args if a not in subset
        )
        adopted[node] = (subset_node, remaining)
    return adopted


def cse_and_stack(
    assignments: Iterable[Assignment],
    symbol: Optional[str] = None,
) -> List[Assignment]:
    """Extract shared subexpressions and return ordered assignments.

    Parameters
    ----------
    assignments
        ``(lhs, rhs)`` pairs defining the computation.
    symbol
        Prefix for generated locals. Defaults to ``"_cse"``.
        Numbering continues after any existing ``<symbol><n>``
        left-hand sides, as the SymPy-era helper did.

    Returns
    -------
    list of tuple
        Dependency-ordered assignments in which every IR node that is
        referenced more than once (and is worth naming) is assigned
        to a ``<symbol><n>`` local and reused by name.

    Notes
    -----
    Piecewise conditions are extracted like any other shared node;
    booleans are valid locals in the generated source.
    """
    if symbol is None:
        symbol = "_cse"
    pairs = list(assignments)

    start_index = 0
    prefix_found = False
    max_index = -1
    for lhs, _ in pairs:
        if isinstance(lhs, Sym) and lhs.name.startswith(symbol):
            prefix_found = True
            suffix = lhs.name[len(symbol):]
            if suffix.isdigit():
                max_index = max(max_index, int(suffix))
    if prefix_found:
        start_index = max_index + 1

    # Count references of every composite node across all RHS roots,
    # and record distinct Add/Mul nodes for subset matching.
    counts: Dict[Expr, int] = {}
    visited_roots: Set[Expr] = set()
    add_nodes: List[Expr] = []
    mul_nodes: List[Expr] = []

    def count(node: Expr) -> None:
        current = counts.get(node, 0)
        counts[node] = current + 1
        if current == 0:
            if type(node) is AddNode:
                add_nodes.append(node)
            elif type(node) is MulNode:
                mul_nodes.append(node)
            for child in _children(node):
                count(child)

    for _, rhs in pairs:
        # Each root counts once per assignment that uses it, so a
        # full RHS repeated across assignments is extracted too.
        if rhs in visited_roots:
            counts[rhs] += 1
            continue
        visited_roots.add(rhs)
        count(rhs)

    # Partial sharing: n-ary flattening hides subset reuse (e.g.
    # ``2*e*a`` vs ``e*a``); match subsets and count them as virtual
    # occurrences so they qualify for extraction.
    adopted: Dict[Expr, Tuple[Expr, Tuple[Expr, ...]]] = {}
    adopted.update(
        _find_partial_subsets(mul_nodes, lambda args: mul(*args))
    )
    adopted.update(
        _find_partial_subsets(add_nodes, lambda args: add(*args))
    )
    for node, (subset_node, _) in adopted.items():
        counts[subset_node] = (
            counts.get(subset_node, 0) + counts.get(node, 1)
        )

    shared = [
        node
        for node, n_refs in counts.items()
        if n_refs > 1 and _is_extractable(node)
    ]
    if not shared:
        return topological_sort(pairs)
    shared_set = set(shared)

    # Drop adoptions whose subset did not end up shared, so the
    # rewrite phase does not restructure products for nothing.
    adopted = {
        node: (subset_node, remaining)
        for node, (subset_node, remaining) in adopted.items()
        if subset_node in shared_set
    }

    # Assign names in first-appearance order for deterministic and
    # readable output: walk the RHS roots in order, depth-first.
    name_order: List[Expr] = []
    seen: Set[Expr] = set()

    def collect(node: Expr) -> None:
        if node in seen:
            return
        seen.add(node)
        for child in _children(node):
            collect(child)
        adoption = adopted.get(node)
        if adoption is not None and adoption[0] not in seen:
            collect(adoption[0])
        if node in shared_set:
            name_order.append(node)

    for _, rhs in pairs:
        collect(rhs)

    replacements: Dict[Expr, Expr] = {}
    cse_assignments: List[Assignment] = []
    for offset, node in enumerate(name_order):
        local = sym(f"{symbol}{start_index + offset}")
        replacements[node] = local

    def rewrite(node: Expr, memo: Dict[Expr, Expr]) -> Expr:
        cached = memo.get(node)
        if cached is not None:
            return cached
        adoption = adopted.get(node)
        if adoption is not None:
            subset_node, remaining = adoption
            local = replacements[subset_node]
            rebuilt_rest = tuple(
                _lookup(child, memo) for child in remaining
            )
            if type(node) is MulNode:
                result = mul(local, *rebuilt_rest)
            else:
                result = add(local, *rebuilt_rest)
        else:
            children = _children(node)
            if children:
                new_children = tuple(
                    _lookup(child, memo) for child in children
                )
                if new_children != children:
                    result = _rebuild(node, new_children)
                else:
                    result = node
            else:
                result = node
        memo[node] = result
        return result

    def _lookup(node: Expr, memo: Dict[Expr, Expr]) -> Expr:
        replacement = replacements.get(node)
        if replacement is not None:
            return replacement
        return rewrite(node, memo)

    memo: Dict[Expr, Expr] = {}
    for node in name_order:
        # Rewrite each extracted node's body in terms of previously
        # extracted locals (children first, so bodies nest properly).
        body = rewrite(node, memo)
        cse_assignments.append((replacements[node], body))

    rewritten_pairs: List[Assignment] = [
        (lhs, _lookup(rhs, memo)) for lhs, rhs in pairs
    ]

    return topological_sort(rewritten_pairs + cse_assignments)
