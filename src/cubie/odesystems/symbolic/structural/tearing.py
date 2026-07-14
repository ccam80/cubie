"""Tearing algorithms: Modia and Carpanzano.

Ports StateSelection.jl's ``modia_tearing.jl`` and
``carpanzano_tearing.jl`` (the current MTK default), including the
exact matching of purely integer-linear SCCs via a restricted
fraction-free Bareiss factorisation, plus the shared helpers
(``contract_variables``, ``free_equations``) and the
:class:`TearingResult` container.

Published Classes
-----------------
:class:`TearingResult`
    Matching, pre-tearing matching, and BLT-sorted variable SCCs.

:class:`ModiaTearing`, :class:`CarpanzanoTearing`
    Tearing algorithms callable on a
    :class:`~cubie.odesystems.symbolic.structural.system_structure.SystemStructure`.
"""

import os
import warnings
from typing import Callable, Dict, List, Optional, Set, Tuple

from cubie.odesystems.symbolic.structural.bipartite import (
    BipartiteGraph,
    Matching,
    UNASSIGNED,
    maximal_matching,
)
from cubie.odesystems.symbolic.structural.clil import (
    SparseMatrixCLIL,
    bareiss,
    bareiss_update_virtual_colswap_clil,
)
from cubie.odesystems.symbolic.structural.digraph import (
    DiCMOBiGraphT,
    IncrementalCycleTracker,
    find_var_sccs,
    neighborhood_in,
)
from cubie.odesystems.symbolic.structural.singularity_removal import (
    RestrictedBareissContext,
)
from cubie.odesystems.symbolic.structural.system_structure import (
    SystemStructure,
)


class OrderedSet:
    """Insertion-ordered set (dict-backed) for deterministic tearing."""

    def __init__(self, iterable=()) -> None:
        self._d = dict.fromkeys(iterable)

    def add(self, item) -> None:
        self._d[item] = None

    def discard(self, item) -> None:
        self._d.pop(item, None)

    def update(self, iterable) -> None:
        for item in iterable:
            self._d[item] = None

    def clear(self) -> None:
        self._d.clear()

    def __contains__(self, item) -> bool:
        return item in self._d

    def __iter__(self):
        return iter(self._d)

    def __len__(self) -> int:
        return len(self._d)


class TearingResult:
    """Result of a tearing algorithm.

    Parameters
    ----------
    var_eq_matching
        Differential variables are matched to
        :data:`~cubie.odesystems.symbolic.structural.bipartite.SELECTED_STATE`,
        solved variables to the equation they are solved from, and
        remaining (torn) algebraic variables to ``UNASSIGNED``.
    full_var_eq_matching
        The maximal matching prior to tearing; used to identify the
        equations of each SCC.
    var_sccs
        Variable SCCs in dependency (BLT) order.
    """

    def __init__(
        self,
        var_eq_matching: Matching,
        full_var_eq_matching: Matching,
        var_sccs: List[List[int]],
    ) -> None:
        self.var_eq_matching = var_eq_matching
        self.full_var_eq_matching = full_var_eq_matching
        self.var_sccs = var_sccs


def contract_variables(
    graph: BipartiteGraph,
    var_eq_matching: Matching,
    var_rename: List[int],
    eq_rename: List[int],
    nelim_eq: int,
    nelim_var: int,
) -> BipartiteGraph:
    """Contract eliminated variables out of the incidence graph.

    Every incidence on an eliminated variable is replaced by
    incidences on the retained variables it (transitively) depends on
    through the matching-induced digraph. ``var_rename``/``eq_rename``
    map old indices to new 0-based indices with ``-1`` marking
    eliminated entries.
    """

    dig = DiCMOBiGraphT(graph, var_eq_matching)
    var_deps = [
        [
            var_rename[v2]
            for v2 in neighborhood_in(dig, v)
            if var_rename[v2] != -1
        ]
        for v in range(graph.ndsts())
    ]

    newgraph = BipartiteGraph(
        graph.nsrcs() - nelim_eq, graph.ndsts() - nelim_var
    )
    for e in range(graph.nsrcs()):
        ne = eq_rename[e]
        if ne == -1:
            continue
        for v in graph.s_neighbors(e):
            newvar = var_rename[v]
            if newvar != -1:
                newgraph.add_edge(ne, newvar)
            else:
                for nv in var_deps[v]:
                    newgraph.add_edge(ne, nv)
    return newgraph


def free_equations(
    graph: BipartiteGraph,
    vars_scc: List[List[int]],
    var_eq_matching: Matching,
    varfilter: Callable[[int], bool],
) -> List[int]:
    """Equations not matched to any filtered variable in the SCCs."""

    ne = graph.nsrcs()
    seen_eqs = [False] * ne
    for scc in vars_scc:
        for var in scc:
            if not varfilter(var):
                continue
            ieq = var_eq_matching[var]
            if isinstance(ieq, int):
                seen_eqs[ieq] = True
    return [i for i, seen in enumerate(seen_eqs) if not seen]


def try_assign_eq(
    ict: IncrementalCycleTracker, vj: int, eq: int
) -> bool:
    """Assign ``eq`` to variable ``vj`` if it creates no cycle."""

    graph = ict.graph

    def apply(g: DiCMOBiGraphT) -> None:
        g.matching[vj] = eq
        g.ne += len(g.graph.s_neighbors(eq)) - 1

    srcs = [v for v in graph.graph.s_neighbors(eq) if v != vj]
    return ict.add_edge_checked(apply, srcs, vj)


def _try_assign_any(
    ict: IncrementalCycleTracker,
    candidates: List[int],
    v_active: Set[int],
    eq: int,
    condition: Callable[[int], bool] = lambda _v: True,
) -> bool:
    graph = ict.graph
    for vj in candidates:
        if (
            vj in v_active
            and graph.matching[vj] is UNASSIGNED
            and condition(vj)
        ):
            if try_assign_eq(ict, vj, eq):
                return True
    return False


def tear_equations(
    ict: IncrementalCycleTracker,
    solvable_adjacency: List[List[int]],
    eqs: List[int],
    v_active: Set[int],
    isder: Optional[Callable[[int], bool]],
    varpriority: Optional[Callable[[int], int]] = None,
) -> IncrementalCycleTracker:
    """Greedy Modia-style equation assignment within a block.

    First assigns equations with a single solvable candidate, then
    the rest; when ``isder`` is given, derivative candidates take
    precedence within an equation. When ``varpriority`` is given,
    candidates are tried lowest-priority first (stable), so
    high-priority variables remain tear variables.
    """

    has_der = [False]
    if isder is not None:

        def isder_tracking(v: int) -> bool:
            r = isder(v)
            has_der[0] = has_der[0] or r
            return r

    for only_single_solvable in (True, False):
        for eq in eqs:
            vs = solvable_adjacency[eq]
            if (len(vs) == 1) != only_single_solvable:
                continue
            if varpriority is not None and len(vs) > 1:
                vs = sorted(vs, key=varpriority)
            if isder is not None:
                _try_assign_any(
                    ict, vs, v_active, eq, isder_tracking
                )
                if has_der[0]:
                    has_der[0] = False
                    continue
            _try_assign_any(ict, vs, v_active, eq)
    return ict


def tear_graph_block_modia(
    var_eq_matching: Matching,
    ict: IncrementalCycleTracker,
    solvable_graph: BipartiteGraph,
    eqs: List[int],
    variables: Set[int],
    isder: Optional[Callable[[int], bool]],
    varpriority: Optional[Callable[[int], int]] = None,
) -> None:
    """Tear one SCC block and copy the assignments into the matching."""

    tear_equations(
        ict, solvable_graph.fadjlist, eqs, variables, isder, varpriority
    )
    for var in variables:
        var_eq_matching[var] = ict.graph.matching[var]


def update_full_var_eq_matching(
    graph: BipartiteGraph,
    full_var_eq_matching: Matching,
    var_eq_matching: Matching,
    scc_vars: List[int],
    remaining_eqs: List[int],
    varfilter: Callable[[int], bool],
) -> None:
    """Rebuild the full matching for an SCC after tearing.

    Copies the torn assignments and matches each still-unassigned
    variable to a remaining equation (preferring incident ones) so
    that, for a fully determined system, every valid variable is
    matched.
    """

    remaining = list(remaining_eqs)
    for var in scc_vars:
        if varfilter(var):
            eq = var_eq_matching[var]
            full_var_eq_matching[var] = eq
            if isinstance(eq, int) and eq in remaining:
                remaining.remove(eq)

    for var in scc_vars:
        if not remaining:
            break
        if not varfilter(var):
            continue
        if isinstance(full_var_eq_matching[var], int):
            continue
        found = False
        for eq in remaining:
            if graph.has_edge(eq, var):
                found = True
                full_var_eq_matching[var] = eq
                remaining.remove(eq)
                break
        if found:
            continue
        eq = remaining.pop(0)
        full_var_eq_matching[var] = eq


def tear_graph_modia(
    structure: SystemStructure,
    isder: Optional[Callable[[int], bool]] = None,
    varfilter: Callable[[int], bool] = lambda v: True,
    eqfilter: Callable[[int], bool] = lambda eq: True,
) -> Tuple[Matching, Matching, List[List[int]]]:
    """Modia tearing over the whole system, SCC by SCC."""

    graph = structure.graph
    solvable_graph = structure.solvable_graph
    priorities = structure.state_priorities
    varpriority = (
        None if priorities is None else (lambda v: priorities[v])
    )
    var_eq_matching = maximal_matching(
        graph, srcfilter=eqfilter, dstfilter=varfilter
    )
    max_eq = max(
        (x for x in var_eq_matching if isinstance(x, int)), default=-1
    )
    var_eq_matching = var_eq_matching.complete(
        max(len(var_eq_matching), max_eq + 1)
    )
    full_var_eq_matching = var_eq_matching.copy()
    var_sccs = find_var_sccs(graph, var_eq_matching)
    vargraph = DiCMOBiGraphT(graph)
    ict = IncrementalCycleTracker(vargraph)

    free_eqs = free_equations(
        graph, var_sccs, var_eq_matching, varfilter
    )
    is_overdetermined = bool(free_eqs)
    for scc_vars in var_sccs:
        ieqs = []
        filtered_vars = set()
        remaining_eqs = []
        for var in scc_vars:
            if varfilter(var):
                filtered_vars.add(var)
                matched = var_eq_matching[var]
                if matched is not UNASSIGNED:
                    ieqs.append(matched)
                    remaining_eqs.append(matched)
            var_eq_matching[var] = UNASSIGNED
        tear_graph_block_modia(
            var_eq_matching,
            ict,
            solvable_graph,
            ieqs,
            filtered_vars,
            isder,
            varpriority,
        )
        update_full_var_eq_matching(
            graph,
            full_var_eq_matching,
            var_eq_matching,
            scc_vars,
            remaining_eqs,
            varfilter,
        )
        if not is_overdetermined:
            vargraph.ne = 0
            for var in scc_vars:
                vargraph.matching[var] = UNASSIGNED
    return var_eq_matching, full_var_eq_matching, var_sccs


class ModiaTearing:
    """Modia tearing algorithm (greedy acyclic assignment)."""

    def __init__(
        self,
        isder: Optional[Callable[[int], bool]] = None,
        varfilter: Callable[[int], bool] = lambda v: True,
        eqfilter: Callable[[int], bool] = lambda eq: True,
    ) -> None:
        self.isder = isder
        self.varfilter = varfilter
        self.eqfilter = eqfilter

    def __call__(
        self, structure: SystemStructure
    ) -> Tuple[TearingResult, dict]:
        var_eq_matching, full_var_eq_matching, var_sccs = (
            tear_graph_modia(
                structure,
                self.isder,
                varfilter=self.varfilter,
                eqfilter=self.eqfilter,
            )
        )
        return (
            TearingResult(
                var_eq_matching, full_var_eq_matching, var_sccs
            ),
            {},
        )


def exact_scc_matching(
    structure: SystemStructure,
    mm: SparseMatrixCLIL,
    mm_row_of: Dict[int, int],
    var_eq_matching: Matching,
    active_vars: List[int],
    active_eqs: List[int],
    isder: Optional[Callable[[int], bool]],
    rewrites: List[Tuple[int, List[int], List[int]]],
) -> bool:
    """Try to match an integer-linear SCC exactly via Bareiss.

    Applicable when the SCC is square (n >= 2) and every equation is
    an integer-linear row in sync with the incidence graph. On full
    rank, the equations are replaced by their triangular reduced
    forms (in ``mm`` and both graphs), each is matched to its proven
    nonzero pivot, and the rewrites are recorded so the caller can
    update the symbolic equations. Rank deficiency reports the SCC as
    exactly singular and falls back to the structural heuristics.
    """

    n = len(active_eqs)
    if n < 2 or n != len(active_vars):
        return False
    graph = structure.graph
    solvable_graph = structure.solvable_graph

    rowids = []
    for e in active_eqs:
        i = mm_row_of.get(e)
        if i is None:
            return False
        cols = mm.row_cols[i]
        nbrs = graph.s_neighbors(e)
        if len(nbrs) != len(cols):
            return False
        if sorted(nbrs) != cols:
            return False
        rowids.append(i)

    nvars = graph.ndsts()
    active_mask = [False] * nvars
    tier1 = [False] * nvars
    for v in active_vars:
        active_mask[v] = True
        if isder is not None and isder(v):
            tier1[v] = True

    sub = SparseMatrixCLIL(
        mm.nparentrows,
        mm.ncols,
        [mm.nzrows[i] for i in rowids],
        [list(mm.row_cols[i]) for i in rowids],
        [list(mm.row_vals[i]) for i in rowids],
    )
    ctx = RestrictedBareissContext(tier1, active_mask, None)

    def clil_update(matrix, k, swapto, pivot, last_pivot):
        bareiss_update_virtual_colswap_clil(
            matrix, k, swapto[1], pivot, last_pivot
        )

    def synced_swap(matrix, i, j):
        matrix.swaprows(i, j)

    bareiss(
        sub,
        ctx.find_pivot,
        swapcols=None,
        swaprows=synced_swap,
        update=clil_update,
    )
    rank = len(ctx.pivots)

    if rank < n:
        warnings.warn(
            f"An SCC of {n} integer-linear equations is exactly "
            f"singular over its own variables (rank {rank}): the "
            "block cannot determine the variables it is scheduled to "
            "solve. Falling back to structural tearing; expect a "
            "singular linear system downstream. Equations: "
            f"{sorted(active_eqs)}."
        )
        return False

    if "EXACT_SCC_DEBUG" in os.environ:
        print(
            f"exact SCC match: n={n} eqs={sorted(active_eqs)}"
        )
    for k in range(n):
        e = sub.nzrows[k]
        i = mm_row_of[e]
        mm.row_cols[i] = sub.row_cols[k]
        mm.row_vals[i] = sub.row_vals[k]
        graph.set_neighbors(e, sub.row_cols[k])
        if solvable_graph is not None:
            solvable_graph.set_neighbors(e, sub.row_cols[k])
        var_eq_matching[ctx.pivots[k]] = e
        rewrites.append((e, sub.row_cols[k], sub.row_vals[k]))
    return True


def find_single_solvable_eq(
    structure: SystemStructure,
    var_eq_matching: Matching,
    active_vars: List[int],
    active_eqs: List[int],
    condition: Callable[[int], bool] = lambda _v: True,
) -> int:
    """Find an equation solvable for exactly one active variable.

    Returns the equation index (matching updated), or ``-1``.
    """

    graph = structure.graph
    solvable_graph = structure.solvable_graph
    active_var_set = (
        active_vars
        if isinstance(active_vars, set)
        else set(active_vars)
    )
    for ieq in active_eqs:
        nbors = [
            v
            for v in graph.s_neighbors(ieq)
            if v in active_var_set
        ]
        if len(nbors) != 1:
            continue
        var = nbors[0]
        if not solvable_graph.has_edge(ieq, var):
            continue
        if not condition(var):
            continue
        var_eq_matching[var] = ieq
        return ieq
    return -1


def carpanzano_tear_scc(
    alg: "CarpanzanoTearing",
    structure: SystemStructure,
    var_eq_matching: Matching,
    active_vars: OrderedSet,
    active_eqs: OrderedSet,
) -> None:
    """Tear one SCC with Carpanzano's algorithm A1 heuristics.

    Solvable variables are matched to the equations that solve them;
    variables left unmatched become tear (iteration) variables.
    Mutates ``active_vars`` and ``active_eqs``.
    """

    graph = structure.graph
    solvable_graph = structure.solvable_graph
    varpriority = structure.state_priorities
    canonrank = structure.canonical_ranks

    def tearkey(v: int) -> Tuple[int, int]:
        return (
            0 if varpriority is None else -varpriority[v],
            0 if canonrank is None else canonrank[v],
        )

    # Drop variables not solvable from any equation in this SCC.
    active_vars_list = [
        v
        for v in active_vars
        if any(
            e in active_eqs for e in solvable_graph.d_neighbors(v)
        )
    ]
    active_vars.clear()
    active_vars.update(active_vars_list)

    while active_vars:
        # Prefer equations solvable for exactly one derivative
        # candidate, then any single-candidate equation.
        if alg.isder is not None:
            single = find_single_solvable_eq(
                structure,
                var_eq_matching,
                active_vars,
                list(active_eqs),
                alg.isder,
            )
            if single >= 0:
                active_eqs.discard(single)
                solved_var = next(
                    v
                    for v in active_vars
                    if var_eq_matching[v] == single
                )
                active_vars.discard(solved_var)
                continue
        single = find_single_solvable_eq(
            structure,
            var_eq_matching,
            active_vars,
            list(active_eqs),
        )
        if single >= 0:
            active_eqs.discard(single)
            solved_var = next(
                v
                for v in active_vars
                if var_eq_matching[v] == single
            )
            active_vars.discard(solved_var)
            continue

        # Heuristic 1: equations with minimum active incidence.
        enodes_with_min_incidence = []
        min_incidence_cnt = None
        for ieq in active_eqs:
            cnt = sum(
                1
                for v in graph.s_neighbors(ieq)
                if v in active_vars
            )
            if min_incidence_cnt is not None and cnt > min_incidence_cnt:
                continue
            if cnt == min_incidence_cnt:
                enodes_with_min_incidence.append(ieq)
                continue
            min_incidence_cnt = cnt
            enodes_with_min_incidence = [ieq]

        # A variable present but not solvable in one of these
        # equations becomes algebraic (torn).
        found_algvar = False
        alg_candidate = -1
        for ieq in enodes_with_min_incidence:
            non_solvable = [
                v
                for v in graph.s_neighbors(ieq)
                if v in active_vars
                and not solvable_graph.has_edge(ieq, v)
            ]
            if not non_solvable:
                continue
            for ivar in non_solvable:
                if varpriority is None and canonrank is None:
                    alg_candidate = ivar
                    found_algvar = True
                    break
                if alg_candidate < 0 or tearkey(ivar) < tearkey(
                    alg_candidate
                ):
                    alg_candidate = ivar
            if found_algvar:
                break

        if alg_candidate >= 0:
            active_vars.discard(alg_candidate)
            found_algvar = True
        if found_algvar:
            continue

        # Heuristic 2: variable with maximum incidence, solvable in
        # the fewest equations, tie-broken by priority/rank.
        alg_var = -1
        max_incidence_cnt = None
        min_solvable_cnt = None
        best_key = None
        for ivar in active_vars:
            cnt = sum(
                1
                for e in graph.d_neighbors(ivar)
                if e in active_eqs
            )
            solvable_cnt = sum(
                1
                for e in solvable_graph.d_neighbors(ivar)
                if e in active_eqs
            )
            key = tearkey(ivar)
            if (
                alg_var < 0
                or cnt > max_incidence_cnt
                or (
                    cnt == max_incidence_cnt
                    and (
                        solvable_cnt < min_solvable_cnt
                        or (
                            solvable_cnt == min_solvable_cnt
                            and key < best_key
                        )
                    )
                )
            ):
                alg_var = ivar
                max_incidence_cnt = cnt
                min_solvable_cnt = solvable_cnt
                best_key = key
        active_vars.discard(alg_var)


class CarpanzanoTearing:
    """Carpanzano (2000) tearing with exact integer-linear SCCs.

    Parameters
    ----------
    isder
        Marks non-dummy derivative variables; equations solvable for
        exactly one such variable are preferred.
    varfilter, eqfilter
        Participation filters for the initial maximal matching.
    mm
        The integer-linear subsystem matrix; when provided, purely
        integer-linear SCCs are matched exactly through a restricted
        Bareiss factorisation.
    """

    def __init__(
        self,
        isder: Optional[Callable[[int], bool]] = None,
        varfilter: Callable[[int], bool] = lambda v: True,
        eqfilter: Callable[[int], bool] = lambda eq: True,
        mm: Optional[SparseMatrixCLIL] = None,
    ) -> None:
        self.isder = isder
        self.varfilter = varfilter
        self.eqfilter = eqfilter
        self.mm = mm

    def __call__(
        self, structure: SystemStructure
    ) -> Tuple[TearingResult, dict]:
        graph = structure.graph
        var_eq_matching = maximal_matching(
            graph, srcfilter=self.eqfilter, dstfilter=self.varfilter
        )
        max_eq = max(
            (x for x in var_eq_matching if isinstance(x, int)),
            default=-1,
        )
        var_eq_matching = var_eq_matching.complete(
            max(len(var_eq_matching), max_eq + 1)
        )
        full_var_eq_matching = var_eq_matching.copy()
        var_sccs = find_var_sccs(graph, var_eq_matching)

        mm_row_of = None
        if self.mm is not None:
            mm_row_of = {
                e: i for i, e in enumerate(self.mm.nzrows)
            }
        rewrites = []

        for scc_vars in var_sccs:
            active_vars = []
            active_eqs = []
            remaining_eqs = []
            for var in scc_vars:
                if self.varfilter(var):
                    active_vars.append(var)
                    matched = var_eq_matching[var]
                    if isinstance(matched, int):
                        active_eqs.append(matched)
                        remaining_eqs.append(matched)
                var_eq_matching[var] = UNASSIGNED
            exact = False
            if mm_row_of is not None:
                exact = exact_scc_matching(
                    structure,
                    self.mm,
                    mm_row_of,
                    var_eq_matching,
                    active_vars,
                    active_eqs,
                    self.isder,
                    rewrites,
                )
            if not exact:
                carpanzano_tear_scc(
                    self,
                    structure,
                    var_eq_matching,
                    OrderedSet(active_vars),
                    OrderedSet(active_eqs),
                )
            update_full_var_eq_matching(
                graph,
                full_var_eq_matching,
                var_eq_matching,
                scc_vars,
                remaining_eqs,
                self.varfilter,
            )

        extra = {"linear_rewrite": rewrites}
        return (
            TearingResult(
                var_eq_matching, full_var_eq_matching, var_sccs
            ),
            extra,
        )
