"""Integer-linear singularity removal via exact Bareiss elimination.

Port of StateSelection.jl's ``singularity_removal.jl``: identifies the
integer-coefficient homogeneous linear subsystem, factorises it with a
fraction-free Bareiss elimination using tiered pivot selection
(purely-linear algebraic variables first, then highest-differentiated
variables, then anything), and rewrites the equations to the reduced
triangular forms. Redundant equations reduce to empty rows and are
removed; underconstrained purely-linear variables are reported through
a pluggable hook.

Published Functions
-------------------
:func:`structural_singularity_removal`
    The pass entry point; returns the reduced integer subsystem.

:func:`get_new_mm`
    Rebuild the integer subsystem matrix after removals/aliasing.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union

from cubie.odesystems.symbolic.structural.bipartite import BipartiteGraph
from cubie.odesystems.symbolic.structural.clil import (
    SparseMatrixCLIL,
    bareiss,
    bareiss_update_virtual_colswap_clil,
)
from cubie.odesystems.symbolic.structural.diffgraph import DiffGraph
from cubie.odesystems.symbolic.structural.pantelides import (
    computed_highest_diff_variables,
)
from cubie.odesystems.symbolic.structural.system_structure import (
    StructuralState,
    SystemStructure,
)


def is_algebraic(var_to_diff: DiffGraph, v: int) -> bool:
    """Whether variable ``v`` has no derivative relations at all."""

    return (
        var_to_diff[v] is None
        and var_to_diff.diff_to_primal[v] is None
    )


def find_first_linear_variable(
    matrix: SparseMatrixCLIL,
    row_range,
    mask,
    var_priorities: Optional[List[int]] = None,
):
    """Locate a pivot among the rows in ``row_range``.

    Prefers, in order: lower-priority pivot columns, then rows with
    fewer stored entries. Returns ``((row, col), value)`` or ``None``.
    """

    eadj = matrix.row_cols
    candidate_i = -1
    candidate_v = -1
    candidate_val = 0
    candidate_nnz = 0
    for i in row_range:
        vertices = eadj[i]
        nnz = len(vertices)
        if (
            candidate_v >= 0
            and var_priorities is None
            and nnz >= candidate_nnz
        ):
            continue
        for j, v in enumerate(vertices):
            if mask is not None and not mask[v]:
                continue
            if (
                candidate_v < 0
                or (var_priorities is None and nnz < candidate_nnz)
                or (
                    var_priorities is not None
                    and (
                        var_priorities[v] < var_priorities[candidate_v]
                        or (
                            var_priorities[v]
                            == var_priorities[candidate_v]
                            and nnz < candidate_nnz
                        )
                    )
                )
            ):
                candidate_i = i
                candidate_v = v
                candidate_val = matrix.row_vals[i][j]
                candidate_nnz = nnz
    if candidate_v < 0:
        return None
    return ((candidate_i, candidate_v), candidate_val)


class _MMSortKey:
    """Row sort key: sparsity, columns, coefficients, equation index."""

    def __init__(self, mm: SparseMatrixCLIL) -> None:
        self.mm = mm

    def __call__(self, i: int):
        mm = self.mm
        return (
            len(mm.row_cols[i]),
            mm.row_cols[i],
            mm.row_vals[i],
            mm.nzrows[i],
        )


def sort_mm_rows(mm: SparseMatrixCLIL) -> SparseMatrixCLIL:
    """Sort rows for reproducible Bareiss results."""

    key = _MMSortKey(mm)
    rperm = sorted(range(len(mm.nzrows)), key=key)
    return SparseMatrixCLIL(
        mm.nparentrows,
        mm.ncols,
        [mm.nzrows[i] for i in rperm],
        [mm.row_cols[i] for i in rperm],
        [mm.row_vals[i] for i in rperm],
    )


class BareissContext:
    """Tiered pivot search for the singularity-removal factorisation.

    Tier 1 pivots among purely-linear variables, tier 2 among
    highest-differentiated variables, and the final tier is
    unrestricted; the boundary ranks are recorded.
    """

    def __init__(
        self,
        is_linear_variables: List[bool],
        is_highest_diff: List[bool],
        var_priorities: Optional[List[int]] = None,
    ) -> None:
        self.rank1 = None
        self.rank2 = None
        self.pivots = []
        self.is_linear_variables = is_linear_variables
        self.is_highest_diff = is_highest_diff
        self.var_priorities = var_priorities

    def find_pivot(self, matrix: SparseMatrixCLIL, k: int):
        nrows = len(matrix.nzrows)
        if self.rank1 is None:
            r = find_first_linear_variable(
                matrix,
                range(k, nrows),
                self.is_linear_variables,
                self.var_priorities,
            )
            if r is not None:
                self.pivots.append(r[0][1])
                return r
            self.rank1 = k
        if self.rank2 is None:
            r = find_first_linear_variable(
                matrix,
                range(k, nrows),
                self.is_highest_diff,
                self.var_priorities,
            )
            if r is not None:
                self.pivots.append(r[0][1])
                return r
            self.rank2 = k
        r = find_first_linear_variable(
            matrix, range(k, nrows), None, self.var_priorities
        )
        if r is not None:
            self.pivots.append(r[0][1])
        return r


class RestrictedBareissContext:
    """Two-tier pivot search that never falls back to any-column.

    Used for exact SCC matching during tearing: once both tiers are
    exhausted the factorisation stops, guaranteeing every pivot is an
    eligible variable.
    """

    def __init__(
        self,
        tier1: List[bool],
        tier2: List[bool],
        var_priorities: Optional[List[int]] = None,
    ) -> None:
        self.pivots = []
        self.tier1 = tier1
        self.tier2 = tier2
        self.valid_pivot_mask = [True] * len(tier1)
        self.var_priorities = var_priorities
        self.tier1_done = False

    def _masked(self, tier: List[bool]) -> List[bool]:
        return [
            t and v for t, v in zip(tier, self.valid_pivot_mask)
        ]

    def find_pivot(self, matrix: SparseMatrixCLIL, k: int):
        nrows = len(matrix.nzrows)
        if not self.tier1_done:
            r = find_first_linear_variable(
                matrix,
                range(k, nrows),
                self._masked(self.tier1),
                self.var_priorities,
            )
            if r is not None:
                self.pivots.append(r[0][1])
                self.valid_pivot_mask[r[0][1]] = False
                return r
            self.tier1_done = True
        r = find_first_linear_variable(
            matrix,
            range(k, nrows),
            self._masked(self.tier2),
            self.var_priorities,
        )
        if r is None:
            return None
        self.pivots.append(r[0][1])
        self.valid_pivot_mask[r[0][1]] = False
        return r


def _synced_swaprows(mold: Optional[SparseMatrixCLIL]):
    def swap(matrix: SparseMatrixCLIL, i: int, j: int) -> None:
        if mold is not None:
            mold.swaprows(i, j)
        matrix.swaprows(i, j)

    return swap


def _clil_update(matrix, k, swapto, pivot, last_pivot) -> None:
    bareiss_update_virtual_colswap_clil(
        matrix, k, swapto[1], pivot, last_pivot
    )


def do_bareiss(
    matrix: SparseMatrixCLIL,
    mold: Optional[SparseMatrixCLIL],
    is_linear_variables: List[bool],
    is_highest_diff: List[bool],
    var_priorities: Optional[List[int]] = None,
) -> Tuple[int, int, int, List[int]]:
    """Tiered Bareiss factorisation of ``matrix`` in place.

    Returns ``(rank1, rank2, rank3, pivots)`` where the ranks bound
    the tier-1/tier-2/total pivot counts.
    """

    ctx = BareissContext(
        is_linear_variables, is_highest_diff, var_priorities
    )
    rank3, _, _ = bareiss(
        matrix,
        ctx.find_pivot,
        swapcols=None,
        swaprows=_synced_swaprows(mold),
        update=_clil_update,
    )
    rank2 = ctx.rank2 if ctx.rank2 is not None else rank3
    rank1 = ctx.rank1 if ctx.rank1 is not None else rank2
    return (rank1, rank2, rank3, ctx.pivots)


def _uf_find(parent: List[int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _uf_union(parent: List[int], a: int, b: int) -> None:
    ra = _uf_find(parent, a)
    rb = _uf_find(parent, b)
    if ra != rb:
        parent[max(ra, rb)] = min(ra, rb)


def aag_bareiss(
    structure: SystemStructure, mm_orig: SparseMatrixCLIL
) -> Tuple[SparseMatrixCLIL, List[int], Tuple[int, int, int, List[int]]]:
    """Bareiss-factorise the integer-linear subsystem.

    Rows are partitioned into connected components (rows connected iff
    they share a column) and each component is eliminated separately,
    which is exact and immune to cross-component pivot perturbation.

    Returns the factored matrix, the purely-linear (solvable)
    variables, and the tier rank/pivot summary.
    """

    graph = structure.graph
    var_to_diff = structure.var_to_diff
    mm = mm_orig.copy()
    linear_equations_set = set(mm_orig.nzrows)

    nvars = len(var_to_diff)
    is_linear_variables = [
        is_algebraic(var_to_diff, v) for v in range(nvars)
    ]
    is_highest_diff = computed_highest_diff_variables(structure)
    for i in range(graph.nsrcs()):
        if i in linear_equations_set and all(
            is_algebraic(var_to_diff, v) for v in graph.s_neighbors(i)
        ):
            continue
        for j in graph.s_neighbors(i):
            is_linear_variables[j] = False
    solvable_variables = [
        v for v, flag in enumerate(is_linear_variables) if flag
    ]

    sp_prios = structure.state_priorities
    cranks = structure.canonical_ranks
    if cranks is None:
        var_priorities = sp_prios
    elif sp_prios is None:
        var_priorities = cranks
    else:
        big = (max(cranks) if cranks else 0) + 1
        var_priorities = [
            sp_prios[i] * big + cranks[i] for i in range(len(sp_prios))
        ]

    # Partition rows into connected components.
    nrows = len(mm.nzrows)
    parent = list(range(nrows))
    col_to_row = {}
    for i in range(nrows):
        for c in mm.row_cols[i]:
            j = col_to_row.get(c)
            if j is None:
                col_to_row[c] = i
            else:
                _uf_union(parent, i, j)
    components = {}
    for i in range(nrows):
        r = _uf_find(parent, i)
        components.setdefault(r, []).append(i)

    if len(components) <= 1:
        mm = sort_mm_rows(mm)
        bar = do_bareiss(
            mm, None, is_linear_variables, is_highest_diff,
            var_priorities,
        )
        return mm, solvable_variables, bar

    comp_rows = sorted(components.values(), key=lambda rows: rows[0])

    all_nzrows = []
    all_row_cols = []
    all_row_vals = []
    tier1_pivots = []
    tier2_pivots = []
    tier3_pivots = []
    rank1 = 0
    rank2_extra = 0
    rank3_extra = 0
    for rows in comp_rows:
        rows.sort(key=_MMSortKey(mm))
        sub = SparseMatrixCLIL(
            mm.nparentrows,
            mm.ncols,
            [mm.nzrows[i] for i in rows],
            [list(mm.row_cols[i]) for i in rows],
            [list(mm.row_vals[i]) for i in rows],
        )
        rank1_c, rank2_c, rank3_c, pivots_c = do_bareiss(
            sub, None, is_linear_variables, is_highest_diff,
            var_priorities,
        )
        all_nzrows.extend(sub.nzrows)
        all_row_cols.extend(sub.row_cols)
        all_row_vals.extend(sub.row_vals)
        tier1_pivots.extend(pivots_c[:rank1_c])
        tier2_pivots.extend(pivots_c[rank1_c:rank2_c])
        tier3_pivots.extend(pivots_c[rank2_c:rank3_c])
        rank1 += rank1_c
        rank2_extra += rank2_c - rank1_c
        rank3_extra += rank3_c - rank2_c
    pivots_merged = tier1_pivots + tier2_pivots + tier3_pivots
    rank2 = rank1 + rank2_extra
    rank3 = rank2 + rank3_extra
    mm = SparseMatrixCLIL(
        mm.nparentrows, mm.ncols, all_nzrows, all_row_cols, all_row_vals
    )
    return mm, solvable_variables, (rank1, rank2, rank3, pivots_merged)


def force_var_to_zero(
    structure: SystemStructure, ils: SparseMatrixCLIL, v: int
) -> SparseMatrixCLIL:
    """Append the equation ``v == 0`` for an underconstrained variable."""

    from cubie.odesystems.symbolic.structural.bipartite import SRC

    ils.nparentrows += 1
    ils.nzrows.append(ils.nparentrows - 1)
    ils.row_cols.append([v])
    ils.row_vals.append([1])
    structure.graph.add_vertex(SRC)
    if structure.solvable_graph is not None:
        structure.solvable_graph.add_vertex(SRC)
    structure.graph.add_edge(ils.nparentrows - 1, v)
    if structure.solvable_graph is not None:
        structure.solvable_graph.add_edge(ils.nparentrows - 1, v)
    structure.eq_to_diff.add_vertex()
    return ils


class IgnoreUnderconstrainedVariable:
    """Record underconstrained variables without altering the system."""

    def __init__(self) -> None:
        self.underconstrained = []

    def __call__(
        self,
        structure: SystemStructure,
        ils: SparseMatrixCLIL,
        v: int,
    ) -> SparseMatrixCLIL:
        self.underconstrained.append(v)
        return ils


class PivotInfo:
    """Summary of the pivots chosen during singularity removal."""

    def __init__(
        self,
        n_linear_vars: int,
        n_highest_diff_vars: int,
        pivots: List[int],
    ) -> None:
        self.n_linear_vars = n_linear_vars
        self.n_highest_diff_vars = n_highest_diff_vars
        self.pivots = pivots


def structural_singularity_removal(
    state: StructuralState,
    variable_underconstrained: Optional[Callable] = None,
    return_pivots: bool = False,
    **kwargs,
):
    """Run the integer-linear singularity removal pass.

    Factorises the integer-linear subsystem exactly, applies the
    underconstrained-variable hook to purely-linear variables that
    were not pivoted, and updates the incidence and solvable graphs
    with the reduced row contents.

    Returns the reduced :class:`SparseMatrixCLIL` (and a
    :class:`PivotInfo` when ``return_pivots`` is true).
    """

    if variable_underconstrained is None:
        variable_underconstrained = force_var_to_zero
    mm = state.linear_subsys_adjmat(**kwargs)
    if len(mm.nzrows) == 0:
        if return_pivots:
            return mm, PivotInfo(0, 0, [])
        return mm

    structure = state.structure
    ils, solvable_variables, (rank1, rank2, _rank3, pivots) = (
        aag_bareiss(structure, mm)
    )
    rk1vars = set(pivots[:rank1])
    for v in solvable_variables:
        if v in rk1vars:
            continue
        ils = variable_underconstrained(structure, ils, v)

    for ei, e in enumerate(ils.nzrows):
        structure.graph.set_neighbors(e, ils.row_cols[ei])
    if structure.solvable_graph is not None:
        for ei, e in enumerate(ils.nzrows):
            structure.solvable_graph.set_neighbors(e, ils.row_cols[ei])

    if return_pivots:
        return ils, PivotInfo(rank1, rank2, pivots)
    return ils


def _add_row_coeffs(
    row_col_i: List[int],
    row_val_i: List[int],
    old_to_new_var: List[int],
    aliases: Dict[int, Union[int, Dict[int, int]]],
    old_var: int,
    coeff: int,
) -> None:
    alias = aliases.get(old_var)
    if alias is None:
        return
    if isinstance(alias, int):
        row_col_i.append(old_to_new_var[alias])
        row_val_i.append(coeff)
        return
    for col, val in sorted(alias.items()):
        row_col_i.append(old_to_new_var[col])
        row_val_i.append(coeff * val)


def get_new_mm(
    aliases: Dict[int, Union[int, Dict[int, int]]],
    old_to_new_eq: List[int],
    old_to_new_var: List[int],
    mm: SparseMatrixCLIL,
) -> SparseMatrixCLIL:
    """Rebuild ``mm`` after removing and aliasing variables/equations.

    ``aliases`` maps a removed variable either to a single surviving
    variable (perfect alias) or to a sparse linear combination
    ``{variable: coefficient}``. Aliasing is not recursive: no alias
    may reference another removed variable.
    """

    new_row_cols = []
    new_row_vals = []
    new_nzrows = []

    for i, eq in enumerate(mm.nzrows):
        if old_to_new_eq[eq] < 0:
            continue
        row_col_i = []
        row_val_i = []
        still_valid = True
        for var, coeff in zip(mm.row_cols[i], mm.row_vals[i]):
            if old_to_new_var[var] >= 0:
                row_col_i.append(old_to_new_var[var])
                row_val_i.append(coeff)
                continue
            if var not in aliases:
                still_valid = False
                break
            _add_row_coeffs(
                row_col_i, row_val_i, old_to_new_var, aliases, var, coeff
            )
        if not still_valid:
            continue
        if any(c < 0 for c in row_col_i):
            raise ValueError(
                "an alias produced a linear combination of a removed "
                "variable; no variable may alias a removed variable"
            )
        pairs = sorted(zip(row_col_i, row_val_i))
        final_cols = []
        final_vals = []
        for col, val in pairs:
            if final_cols and col == final_cols[-1]:
                final_vals[-1] += val
                if final_vals[-1] == 0:
                    final_cols.pop()
                    final_vals.pop()
            else:
                final_cols.append(col)
                final_vals.append(val)
        new_row_cols.append(final_cols)
        new_row_vals.append(final_vals)
        new_nzrows.append(old_to_new_eq[eq])

    new_mm = SparseMatrixCLIL(
        sum(1 for e in old_to_new_eq if e >= 0),
        sum(1 for v in old_to_new_var if v >= 0),
        new_nzrows,
        new_row_cols,
        new_row_vals,
    )
    return new_mm.dropzeros()
