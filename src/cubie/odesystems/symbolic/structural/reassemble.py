"""Reassembly of the simplified system from tearing decisions.

Port of ModelingToolkitTearing's ``reassemble.jl``
(``DefaultReassembleAlgorithm``): renames dummy derivatives to
algebraic variables, lowers higher-order derivatives to first order,
solves each matched equation for its variable (producing differential
equations and observed equations), leaves torn equations as algebraic
residuals, and reorders the state into BLT form.

Published Classes
-----------------
:class:`ReassembledSystem`
    The reassembled system pieces consumed by the pipeline driver.

Published Functions
-------------------
:func:`default_reassemble`
    Run the default reassembly on a tearing result.
"""

import warnings
from typing import Dict, List, Optional, Tuple

import sympy as sp

from cubie.odesystems.symbolic.structural.bipartite import (
    Matching,
    SELECTED_STATE,
    UNASSIGNED,
)
from cubie.odesystems.symbolic.structural.clil import SparseMatrixCLIL
from cubie.odesystems.symbolic.structural.digraph import (
    DiCMOBiGraphF,
    toposort_equations,
)
from cubie.odesystems.symbolic.structural.singularity_removal import (
    get_new_mm,
)
from cubie.odesystems.symbolic.structural.symbolics import (
    fixpoint_sub,
    linear_expansion,
    lower_varname,
)
from cubie.odesystems.symbolic.structural.system_structure import (
    Equation,
    StructuralState,
)
from cubie.odesystems.symbolic.structural.tearing import (
    TearingResult,
    contract_variables,
)


def var_order(dv: int, diff_to_var) -> Tuple[int, int]:
    """Derivative order and lowest-order variable of ``dv``'s chain."""

    order = 0
    while diff_to_var[dv] is not None:
        order += 1
        dv = diff_to_var[dv]
    return order, dv


class ReassembledSystem:
    """Pieces of the reassembled system.

    Parameters
    ----------
    state
        The (reordered) structural state.
    neweqs
        Solved differential equations followed by algebraic residual
        equations, in BLT order. Differential equations have their
        derivative symbol as the LHS; residuals have LHS zero.
    solved_eqs
        Observed equations from tearing (explicitly solved
        variables), in solve order.
    observed
        Full observed list: solved equations plus alias/trivially
        torn equations, with dummy-derivative renames applied.
    unknowns
        Final unknowns (selected states then torn algebraic
        variables), in BLT order.
    var_sccs
        Variable SCCs over the final unknown indices.
    dummy_sub
        Renames applied to dummy derivatives (old derivative symbol
        to its algebraic replacement).
    extra_unknowns
        Unknowns of underdetermined systems not matched to any
        equation.
    """

    def __init__(
        self,
        state: StructuralState,
        neweqs: List[Equation],
        diff_eq_states: List[Optional[sp.Symbol]],
        solved_eqs: List[Equation],
        observed: List[Equation],
        unknowns: List[sp.Symbol],
        var_sccs: List[List[int]],
        dummy_sub: Dict[sp.Symbol, sp.Symbol],
        extra_unknowns: List[sp.Symbol],
    ) -> None:
        self.state = state
        self.neweqs = neweqs
        self.diff_eq_states = diff_eq_states
        self.solved_eqs = solved_eqs
        self.observed = observed
        self.unknowns = unknowns
        self.var_sccs = var_sccs
        self.dummy_sub = dummy_sub
        self.extra_unknowns = extra_unknowns


def substitute_derivatives_algevars(
    state: StructuralState,
    neweqs: List[Equation],
    var_eq_matching: Matching,
    dummy_sub: Dict[sp.Symbol, sp.Symbol],
) -> None:
    """Replace derivatives of non-selected variables by dummy names.

    State selection may determine that some differential variables
    are algebraic variables in disguise; their derivative symbols are
    renamed to user-visible ``x_t`` variables and the derivative edge
    is cut. After this pass, ``SelectedState`` information is no
    longer needed.
    """

    structure = state.structure
    graph = structure.graph
    var_to_diff = structure.var_to_diff
    diff_to_var = var_to_diff.invview()
    registry = state.registry

    for var in range(len(state.fullvars)):
        dv = var_to_diff[var]
        if dv is None:
            continue
        if var_eq_matching[var] is SELECTED_STATE:
            continue
        dd = state.fullvars[dv]
        base, order = registry.base_and_order(dd)
        v_t = sp.Symbol(
            lower_varname(base.name, order, registry.reserved),
            real=True,
        )
        for eq in graph.d_neighbors(dv):
            neweqs[eq] = neweqs[eq].xreplace({dd: v_t})
        dummy_sub[dd] = v_t
        state.fullvars[dv] = v_t
        del state.var2idx[dd]
        state.var2idx[v_t] = dv
        registry.rename(dd, v_t)
        # Higher orders (D(D(x)) -> D(x_t)) keep their internal
        # symbols; the registry rename above rebased their chain onto
        # the new variable.
        diff_to_var[dv] = None


def find_duplicate_dd(
    dv: int,
    solvable_graph,
    diff_to_var,
    linear_eqs: Dict[int, int],
    mm: Optional[SparseMatrixCLIL],
) -> Optional[Tuple[int, int]]:
    """Find a pre-existing ``D(x) ~ x_t`` equation for ``dv``."""

    if mm is None:
        return None
    for eq in solvable_graph.d_neighbors(dv):
        mi = linear_eqs.get(eq)
        if mi is None:
            continue
        rvs = mm.row_cols[mi]
        nzs = mm.row_vals[mi]
        if (
            len(nzs) == 2
            and abs(nzs[0]) == 1
            and nzs[0] == -nzs[1]
        ):
            v_t = rvs[1] if rvs[0] == dv else rvs[0]
            if diff_to_var[v_t] is None:
                if dv not in rvs:
                    raise AssertionError(
                        "duplicate dummy-derivative row does not "
                        "contain the derivative variable"
                    )
                return eq, v_t
    return None


def _insert_sccs(
    var_sccs: List[List[int]],
    sccs_to_insert: List[Tuple[int, List[int]]],
) -> List[List[int]]:
    """Insert singleton SCCs at the requested indices."""

    old_idx = 0
    insert_idx = 0
    new_sccs = []
    total = len(var_sccs) + len(sccs_to_insert)
    for _ in range(total):
        if (
            insert_idx < len(sccs_to_insert)
            and sccs_to_insert[insert_idx][0] == old_idx
        ):
            new_sccs.append(sccs_to_insert[insert_idx][1])
            insert_idx += 1
        else:
            new_sccs.append(list(var_sccs[old_idx]))
            old_idx += 1
    return [scc for scc in new_sccs if scc]


def generate_derivative_variables(
    state: StructuralState,
    neweqs: List[Equation],
    var_eq_matching: Matching,
    full_var_eq_matching: Matching,
    var_sccs: List[List[int]],
    mm: Optional[SparseMatrixCLIL],
) -> List[List[int]]:
    """Lower the system to first order by adding ``D(x) ~ x_t``.

    For every differentiated variable whose derivative is not solved
    from any equation, introduce the variable ``x_t`` and the
    equation ``0 ~ D(x) - x_t`` (unless an equivalent equation
    already exists), match ``D(x)`` to it, and update the SCCs.
    Returns the new SCC list.
    """

    structure = state.structure
    graph = structure.graph
    solvable_graph = structure.solvable_graph
    var_to_diff = structure.var_to_diff
    diff_to_var = var_to_diff.invview()
    registry = state.registry
    linear_eqs = {}
    if mm is not None:
        linear_eqs = {e: i for i, e in enumerate(mm.nzrows)}

    v_to_scc = [None] * graph.ndsts()
    for i, scc in enumerate(var_sccs):
        for j, v in enumerate(scc):
            v_to_scc[v] = (i, j)

    v_t_dvs = []

    for v in range(len(var_to_diff)):
        dv = var_to_diff[v]
        if dv is None:
            continue
        if isinstance(var_eq_matching[dv], int):
            continue

        dd = find_duplicate_dd(
            dv, solvable_graph, diff_to_var, linear_eqs, mm
        )
        if dd is None:
            dx = state.fullvars[dv]
            order, lv = var_order(dv, diff_to_var)
            base, order_r = registry.base_and_order(dx)
            x_t = sp.Symbol(
                lower_varname(
                    base.name, order_r, registry.reserved
                ),
                real=True,
            )
            # Add x_t to the graph.
            v_t = _add_dd_variable(state, x_t, dv)
            # Add 0 ~ D(x) - x_t to the graph.
            dummy_eq = _add_dd_equation(
                state, neweqs, Equation(sp.S.Zero, dx - x_t), dv, v_t
            )
            for e in list(graph.d_neighbors(dv)):
                graph.add_edge(e, v_t)
            var_eq_matching.push(UNASSIGNED)
            full_var_eq_matching.push(UNASSIGNED)
            dd = (dummy_eq, v_t)
        dummy_eq, v_t = dd
        var_to_diff[v_t] = var_to_diff[dv]
        old_matched_eq = full_var_eq_matching[dv]
        var_eq_matching[dv] = dummy_eq
        full_var_eq_matching[dv] = dummy_eq
        full_var_eq_matching[v_t] = old_matched_eq
        v_t_dvs.append((v_t, dv))

    sccs_to_insert = []
    idxs_to_remove = {}
    for v_t, dv in v_t_dvs:
        i, j = v_to_scc[dv]
        var_sccs[i][j] = v_t
        if v_t < len(v_to_scc) and v_to_scc[v_t] is not None:
            i2, j2 = v_to_scc[v_t]
            idxs_to_remove.setdefault(i2, []).append(j2)
        # The new singleton SCC for dv must run before the SCC of
        # its higher derivative so total_sub is populated in order.
        ddv = var_to_diff[dv]
        if (
            isinstance(ddv, int)
            and ddv < len(v_to_scc)
            and v_to_scc[ddv] is not None
        ):
            i_insert = min(i, v_to_scc[ddv][0])
        else:
            i_insert = i
        sccs_to_insert.append((i_insert, [dv]))

    def chain_height(dv: int) -> int:
        h = 0
        v = dv
        while True:
            if v >= len(var_to_diff):
                break
            v = var_to_diff[v]
            if not isinstance(v, int):
                break
            h += 1
        return h

    sccs_to_insert.sort(
        key=lambda item: (item[0], -chain_height(item[1][0]))
    )
    for i, idxs in idxs_to_remove.items():
        for j in sorted(idxs, reverse=True):
            del var_sccs[i][j]
    new_sccs = _insert_sccs(var_sccs, sccs_to_insert)

    if mm is not None:
        mm.ncols = graph.ndsts()
    return new_sccs


def _add_dd_variable(
    state: StructuralState, x_t: sp.Symbol, dv: int
) -> int:
    """Add the dummy variable ``x_t`` mirroring variable ``dv``."""

    from cubie.odesystems.symbolic.structural.bipartite import DST

    structure = state.structure
    state.fullvars.append(x_t)
    structure.state_priorities.append(structure.state_priorities[dv])
    structure.canonical_ranks.append(structure.canonical_ranks[dv])
    state.always_present.append(False)
    v_t = structure.var_to_diff.add_vertex()
    structure.graph.add_vertex(DST)
    structure.solvable_graph.add_vertex(DST)
    structure.var_to_diff[v_t] = structure.var_to_diff[dv]
    state.var2idx[x_t] = v_t
    return v_t


def _add_dd_equation(
    state: StructuralState,
    neweqs: List[Equation],
    eq: Equation,
    dv: int,
    v_t: int,
) -> int:
    """Append ``0 ~ D(x) - x_t`` and its graph vertices/edges."""

    from cubie.odesystems.symbolic.structural.bipartite import SRC

    structure = state.structure
    neweqs.append(eq)
    state.eqs.append(eq)
    state.original_eqs.append(eq)
    structure.graph.add_vertex(SRC)
    dummy_eq = len(neweqs) - 1
    structure.graph.add_edge(dummy_eq, dv)
    structure.graph.add_edge(dummy_eq, v_t)
    structure.solvable_graph.add_vertex(SRC)
    structure.solvable_graph.add_edge(dummy_eq, dv)
    structure.eq_to_diff.add_vertex()
    return dummy_eq


def get_sorted_scc(
    digraph: DiCMOBiGraphF,
    full_var_eq_matching: Matching,
    var_eq_matching: Matching,
    scc: List[int],
) -> Tuple[List[int], List[int]]:
    """Sort one SCC's variables and equations into solve order."""

    eq_var_matching = var_eq_matching.invview()
    scc_eqs = []
    scc_solved_eqs = []
    for v in scc:
        e = full_var_eq_matching[v]
        if isinstance(e, int):
            scc_eqs.append(e)
        e = var_eq_matching[v]
        if isinstance(e, int):
            scc_solved_eqs.append(e)
    sorted_solved = list(
        reversed(toposort_equations(digraph, scc_solved_eqs))
    )
    solved_set = set(scc_solved_eqs)
    scc_eqs_sorted = sorted_solved + [
        e for e in scc_eqs if e not in solved_set
    ]
    scc_vars = []
    for e in scc_eqs_sorted:
        v = eq_var_matching[e] if e < len(eq_var_matching.match) else (
            UNASSIGNED
        )
        if isinstance(v, int):
            scc_vars.append(v)
    var_set = set(scc_vars)
    scc_vars.extend(v for v in scc if v not in var_set)
    return scc_vars, scc_eqs_sorted


class EquationGenerator:
    """Accumulates generated equations and their orderings."""

    def __init__(self, state: StructuralState) -> None:
        self.state = state
        self.total_sub = {}
        self.neweqs_out = []
        # For differential equations, the state symbol whose
        # derivative the equation defines (None for algebraic rows).
        # Derived from the graph's derivative chain, which
        # find_duplicate_dd may rewire away from the registry chain.
        self.diff_eq_states = []
        self.eq_ordering = []
        self.var_ordering = []
        self.solved_eqs = []
        self.solved_vars = []

    def is_solvable(self, ieq, iv) -> bool:
        solvable_graph = self.state.structure.solvable_graph
        return (
            isinstance(ieq, int)
            and isinstance(iv, int)
            and solvable_graph.has_edge(ieq, iv)
        )

    def is_dervar(self, iv: int) -> bool:
        return self.state.structure.isdervar(iv)

    def codegen_equation(
        self, eq: Equation, ieq: int, iv, simplify: bool = False
    ) -> None:
        """Generate the output form of ``eq``.

        Solvable equations of derivative variables become
        differential equations; solvable equations of algebraic
        variables become observed equations; everything else stays as
        an algebraic residual.
        """

        state = self.state
        structure = state.structure
        graph = structure.graph
        diff_to_var = structure.var_to_diff.invview()
        total_sub = self.total_sub

        issolvable = self.is_solvable(ieq, iv)
        isdervar = issolvable and self.is_dervar(iv)
        if issolvable and isdervar:
            var = state.fullvars[iv]
            rhs = _solve_for(eq, var, simplify)
            rhs = fixpoint_sub(rhs, total_sub)
            neweq = Equation(var, rhs)
            # Any equation incident on `iv` will have it substituted:
            # rewire incidence through this equation's variables.
            for e in list(graph.d_neighbors(iv)):
                if e == ieq:
                    continue
                for v in graph.s_neighbors(ieq):
                    graph.add_edge(e, v)
                graph.rem_edge(e, iv)
            total_sub[var] = rhs
            self.neweqs_out.append(neweq)
            self.diff_eq_states.append(
                state.fullvars[diff_to_var[iv]]
            )
            self.eq_ordering.append(ieq)
            self.var_ordering.append(diff_to_var[iv])
        elif issolvable:
            var = state.fullvars[iv]
            residual = eq.lhs - eq.rhs
            a, b, islinear = linear_expansion(residual, var)
            if not islinear or a == sp.S.Zero:
                warnings.warn(
                    f"Tearing: solving {eq} for {var} is singular!"
                )
                return
            rhs = -b / a
            if simplify:
                rhs = sp.simplify(rhs)
            neweq = Equation(var, fixpoint_sub(rhs, total_sub))
            self.solved_eqs.append(neweq)
            self.solved_vars.append(iv)
        else:
            rhs = fixpoint_sub(eq.residual(), total_sub)
            self.neweqs_out.append(Equation(sp.S.Zero, rhs))
            self.diff_eq_states.append(None)
            self.eq_ordering.append(ieq)
            self.var_ordering.append(-1)


def _solve_for(eq: Equation, var: sp.Symbol, simplify: bool) -> sp.Expr:
    residual = eq.lhs - eq.rhs
    a, b, islinear = linear_expansion(residual, var)
    if not islinear or a == sp.S.Zero:
        raise ValueError(
            f"equation {eq} is not solvable for {var} despite a "
            "solvable-graph edge"
        )
    rhs = -b / a
    if simplify:
        rhs = sp.simplify(rhs)
    return rhs


def get_extra_eqs_vars(
    state: StructuralState,
    var_eq_matching: Matching,
    full_var_eq_matching: Matching,
    fully_determined: bool,
) -> Tuple[List[int], List[int]]:
    """Unmatched equations/variables of non-fully-determined systems."""

    if fully_determined:
        return [], []
    extra_eqs = []
    extra_vars = []
    full_eq_var_matching = full_var_eq_matching.invview()
    graph = state.structure.graph
    for v in range(graph.ndsts()):
        eq = full_var_eq_matching[v]
        if isinstance(eq, int):
            continue
        if var_eq_matching[v] is not UNASSIGNED:
            continue
        extra_vars.append(v)
    for eq in range(graph.nsrcs()):
        v = (
            full_eq_var_matching[eq]
            if eq < len(full_eq_var_matching.match)
            else UNASSIGNED
        )
        if isinstance(v, int):
            continue
        extra_eqs.append(eq)
    return extra_eqs, extra_vars


def generate_system_equations(
    state: StructuralState,
    neweqs: List[Equation],
    var_eq_matching: Matching,
    full_var_eq_matching: Matching,
    var_sccs: List[List[int]],
    extra_eqs_vars: Tuple[List[int], List[int]],
    simplify: bool = False,
    inline_linear_sccs: bool = False,
    analytical_linear_scc_limit: int = 2,
    allow_symbolic: bool = False,
    allow_parameter: bool = True,
) -> Tuple[
    List[Equation], List[Equation], List[int], List[int], int, int
]:
    """Solve matched equations and order the system into BLT form.

    Returns ``(neweqs, solved_eqs, eq_ordering, var_ordering,
    n_solved_eqs, n_solved_vars)``.
    """

    structure = state.structure
    graph = structure.graph
    var_to_diff = structure.var_to_diff
    diff_to_var = var_to_diff.invview()
    eq_var_matching = var_eq_matching.invview()
    extra_eqs, extra_vars = extra_eqs_vars

    gen = EquationGenerator(state)

    # Solve extra (overdetermined) equations first to respect
    # topological order.
    for eq in extra_eqs:
        var = (
            eq_var_matching[eq]
            if eq < len(eq_var_matching.match)
            else UNASSIGNED
        )
        if not isinstance(var, int):
            continue
        gen.codegen_equation(neweqs[eq], eq, var, simplify)

    def ispresent(i: int) -> bool:
        if graph.d_neighbors(i):
            return True
        dvi = var_to_diff[i]
        return dvi is not None and bool(graph.d_neighbors(dvi))

    digraph = DiCMOBiGraphF(graph, var_eq_matching)
    for i, scc in enumerate(var_sccs):
        vscc, escc = get_sorted_scc(
            digraph, full_var_eq_matching, var_eq_matching, scc
        )
        var_sccs[i] = vscc
        if len(escc) != len(vscc):
            if not escc:
                continue
            escc = [e for e in escc if e not in set(extra_eqs)]
            if not escc:
                continue
            vscc = [v for v in vscc if v not in set(extra_vars)]
            if not vscc:
                continue

        linsol = None
        if inline_linear_sccs:
            linsol = _get_linear_scc_linsol(
                state,
                escc,
                vscc,
                neweqs,
                var_eq_matching,
                gen.total_sub,
                analytical_linear_scc_limit,
                simplify,
                allow_symbolic,
                allow_parameter,
            )
        if linsol is not None:
            solutions, eqs_mask, vars_mask = linsol
            _escc = [e for e, m in zip(escc, eqs_mask) if m]
            _vscc = [v for v, m in zip(vscc, vars_mask) if m]
            for j, (ieq, ivar) in enumerate(zip(_escc, _vscc)):
                rhs = solutions[j]
                int_iv = diff_to_var[ivar]
                if int_iv is not None:
                    dx_sym = state.fullvars[ivar]
                    gen.neweqs_out.append(Equation(dx_sym, rhs))
                    gen.diff_eq_states.append(
                        state.fullvars[int_iv]
                    )
                    gen.eq_ordering.append(ieq)
                    gen.var_ordering.append(int_iv)
                    for e in list(graph.d_neighbors(ivar)):
                        if e == ieq:
                            continue
                        for vsym in rhs.free_symbols:
                            v_idx = state.var2idx.get(vsym)
                            if v_idx is not None:
                                graph.add_edge(e, v_idx)
                        graph.rem_edge(e, ivar)
                    gen.total_sub[dx_sym] = rhs
                else:
                    var_sym = fixpoint_sub(
                        state.fullvars[ivar], gen.total_sub
                    )
                    gen.solved_eqs.append(Equation(var_sym, rhs))
                    gen.solved_vars.append(ivar)
            for e, m in zip(escc, eqs_mask):
                if m:
                    continue
                var = eq_var_matching[e]
                gen.codegen_equation(neweqs[e], e, var, simplify)
        else:
            for ieq in escc:
                iv = (
                    eq_var_matching[ieq]
                    if ieq < len(eq_var_matching.match)
                    else UNASSIGNED
                )
                gen.codegen_equation(neweqs[ieq], ieq, iv, simplify)

    for eq in extra_eqs:
        var = (
            eq_var_matching[eq]
            if eq < len(eq_var_matching.match)
            else UNASSIGNED
        )
        if isinstance(var, int):
            continue
        gen.codegen_equation(neweqs[eq], eq, var, simplify)

    var_ordering = gen.var_ordering
    diff_vars = [v for v in var_ordering if v >= 0]
    diff_vars_set = set(diff_vars)
    if len(diff_vars_set) != len(diff_vars):
        raise ValueError(
            "Tearing internal error: lowering DAE into semi-implicit "
            "ODE failed!"
        )
    solved_vars_set = set(gen.solved_vars)
    extra_vars_set = set(extra_vars)

    # Fill algebraic (torn) variable slots.
    offset = 0
    for i, v in enumerate(var_ordering):
        if v >= 0:
            continue
        index = None
        for j in range(offset, graph.ndsts()):
            if (
                j not in diff_vars_set
                and j not in solved_vars_set
                and j not in extra_vars_set
                and diff_to_var[j] is None
                and ispresent(j)
            ):
                index = j
                break
        if index is None:
            break
        var_ordering[i] = index
        offset = index + 1
    var_ordering = [v for v in var_ordering if v >= 0]
    used = set(var_ordering) | solved_vars_set
    var_ordering = var_ordering + [
        v for v in range(graph.ndsts()) if v not in used
    ]
    return (
        gen.neweqs_out,
        gen.diff_eq_states,
        gen.solved_eqs,
        gen.eq_ordering,
        var_ordering,
        len(gen.solved_vars),
        len(solved_vars_set),
    )


def _get_linear_scc_linsol(
    state: StructuralState,
    alg_eqs: List[int],
    alg_vars: List[int],
    neweqs: List[Equation],
    var_eq_matching: Matching,
    total_sub: Dict,
    analytical_linear_scc_limit: int,
    simplify: bool,
    allow_symbolic: bool,
    allow_parameter: bool,
) -> Optional[Tuple[List[sp.Expr], List[bool], List[bool]]]:
    """Solve a linear algebraic SCC analytically when small enough.

    Returns ``(solutions, eqs_mask, vars_mask)`` covering the masked
    subset of the SCC, or ``None`` when the SCC is nonlinear, already
    fully torn, too large, or singular. Only the analytical (small-N)
    path is supported; larger linear SCCs remain algebraic residuals
    handled by the implicit solver.
    """

    structure = state.structure
    graph = structure.graph
    diff_to_var = structure.var_to_diff.invview()

    all_torn = True
    for iv in alg_vars:
        all_torn = all_torn and (
            isinstance(var_eq_matching[iv], int)
            and not structure.isdervar(iv)
        )
    if all_torn:
        return None

    n = len(alg_eqs)
    if n != len(alg_vars):
        return None
    variables = [state.fullvars[v] for v in alg_vars]

    b = []
    for ieq in alg_eqs:
        resid = neweqs[ieq].rhs - neweqs[ieq].lhs
        if simplify:
            resid = sp.simplify(resid)
        b.append(fixpoint_sub(resid, total_sub))

    a_matrix = [[sp.S.Zero] * n for _ in range(n)]
    for varidx, var in enumerate(variables):
        for eqidx in range(n):
            if not graph.has_edge(alg_eqs[eqidx], alg_vars[varidx]):
                continue
            p, q, islinear = linear_expansion(b[eqidx], var)
            if not islinear:
                return None
            a_matrix[eqidx][varidx] = p
            b[eqidx] = q
    b = [-resid for resid in b]

    # Eliminate rows already matched to a variable (torn rows).
    eqs_mask = [True] * n
    vars_mask = [True] * n
    eq_var_matching = var_eq_matching.invview()
    var_to_local = {v: i for i, v in enumerate(alg_vars)}
    aliases = {}
    constants = {}
    for i in range(n):
        matched = (
            eq_var_matching[alg_eqs[i]]
            if alg_eqs[i] < len(eq_var_matching.match)
            else UNASSIGNED
        )
        if not isinstance(matched, int):
            continue
        if matched not in var_to_local:
            continue
        ivar = var_to_local[matched]
        eqs_mask[i] = False
        vars_mask[ivar] = False
        var_coeff = a_matrix[i][ivar]
        if var_coeff == sp.S.Zero:
            return None
        combo = {}
        for j in range(n):
            if j == ivar:
                continue
            if a_matrix[i][j] != sp.S.Zero:
                combo[j] = -a_matrix[i][j] / var_coeff
        aliases[ivar] = combo
        constants[ivar] = b[i] / var_coeff

    # Resolve aliases that reference other eliminated variables
    # (topologically; tearing guarantees acyclicity).
    changed = True
    guard = 0
    while changed:
        changed = False
        guard += 1
        if guard > n + 1:
            return None
        for ivar, combo in list(aliases.items()):
            new_combo = {}
            cst = constants[ivar]
            dirty = False
            for jvar, coeff in combo.items():
                if jvar in aliases and jvar != ivar:
                    dirty = True
                    for kvar, kcoeff in aliases[jvar].items():
                        new_combo[kvar] = (
                            new_combo.get(kvar, sp.S.Zero)
                            + coeff * kcoeff
                        )
                    cst = cst + coeff * constants[jvar]
                else:
                    new_combo[jvar] = (
                        new_combo.get(jvar, sp.S.Zero) + coeff
                    )
            if dirty:
                changed = True
                aliases[ivar] = {
                    k: v for k, v in new_combo.items() if v != 0
                }
                constants[ivar] = cst

    # Substitute eliminated variables into the retained rows.
    kept_rows = [i for i in range(n) if eqs_mask[i]]
    kept_cols = [j for j in range(n) if vars_mask[j]]
    reduced_n = len(kept_rows)
    if reduced_n != len(kept_cols):
        return None
    if reduced_n == 0:
        return None
    if reduced_n > analytical_linear_scc_limit and reduced_n != 1:
        return None

    a_red = [
        [sp.S.Zero] * reduced_n for _ in range(reduced_n)
    ]
    b_red = []
    for ri, i in enumerate(kept_rows):
        row_b = b[i]
        for j in range(n):
            coeff = a_matrix[i][j]
            if coeff == sp.S.Zero:
                continue
            if vars_mask[j]:
                a_red[ri][kept_cols.index(j)] += coeff
            else:
                row_b = row_b - coeff * constants[j]
                for kvar, kcoeff in aliases[j].items():
                    if not vars_mask[kvar]:
                        return None
                    a_red[ri][kept_cols.index(kvar)] += coeff * kcoeff
        b_red.append(row_b)

    if reduced_n > 1 and not allow_symbolic:
        for row in a_red:
            for entry in row:
                if entry.is_number:
                    continue
                if not allow_parameter:
                    return None
                for sym in entry.free_symbols:
                    if sym in state.var2idx:
                        return None

    a_sym = sp.Matrix(a_red)
    b_sym = sp.Matrix(b_red)
    try:
        solution = a_sym.LUsolve(b_sym)
    except (ValueError, ZeroDivisionError):
        return None
    solutions = [
        fixpoint_sub(sp.simplify(sol) if simplify else sol, total_sub)
        for sol in solution
    ]
    return solutions, eqs_mask, vars_mask


def reorder_vars(
    state: StructuralState,
    var_eq_matching: Matching,
    var_sccs: List[List[int]],
    eq_ordering: List[int],
    var_ordering: List[int],
    nsolved_eq: int,
    nsolved_var: int,
) -> None:
    """Permute the state into the generated (BLT) ordering.

    Eliminated (solved) variables and equations are contracted out of
    the graphs.
    """

    structure = state.structure
    graph = structure.graph
    solvable_graph = structure.solvable_graph
    var_to_diff = structure.var_to_diff
    eq_to_diff = structure.eq_to_diff

    eqsperm = [-1] * graph.nsrcs()
    for i, v in enumerate(eq_ordering):
        eqsperm[v] = i
    varsperm = [-1] * graph.ndsts()
    for i, v in enumerate(var_ordering):
        varsperm[v] = i

    new_graph = contract_variables(
        graph, var_eq_matching, varsperm, eqsperm, nsolved_eq,
        nsolved_var,
    )
    new_solvable_graph = contract_variables(
        solvable_graph,
        var_eq_matching,
        varsperm,
        eqsperm,
        nsolved_eq,
        nsolved_var,
    )

    from cubie.odesystems.symbolic.structural.diffgraph import DiffGraph

    new_var_to_diff = DiffGraph(len(var_ordering), with_badj=True)
    for v in range(len(var_to_diff)):
        d = var_to_diff[v]
        v2 = varsperm[v]
        if v2 < 0 or d is None:
            continue
        d2 = varsperm[d]
        new_var_to_diff[v2] = d2 if d2 >= 0 else None
    new_eq_to_diff = DiffGraph(len(eq_ordering), with_badj=True)
    for e in range(len(eq_to_diff)):
        d = eq_to_diff[e]
        e2 = eqsperm[e]
        if e2 < 0 or d is None:
            continue
        d2 = eqsperm[d]
        new_eq_to_diff[e2] = d2 if d2 >= 0 else None
    new_fullvars = [state.fullvars[v] for v in var_ordering]

    for scc in var_sccs:
        scc[:] = [
            varsperm[v] for v in scc if varsperm[v] >= 0
        ]
    var_sccs[:] = [scc for scc in var_sccs if scc]

    structure.graph = new_graph.complete()
    structure.solvable_graph = new_solvable_graph.complete()
    structure.var_to_diff = new_var_to_diff
    structure.eq_to_diff = new_eq_to_diff
    structure.state_priorities = [
        structure.state_priorities[v] for v in var_ordering
    ]
    structure.canonical_ranks = [
        structure.canonical_ranks[v] for v in var_ordering
    ]
    state.always_present = [
        state.always_present[v] for v in var_ordering
    ]
    state.fullvars = new_fullvars
    state.var2idx = {v: i for i, v in enumerate(new_fullvars)}


def default_reassemble(
    state: StructuralState,
    tearing_result: TearingResult,
    mm: Optional[SparseMatrixCLIL],
    fully_determined: bool = True,
    simplify: bool = False,
    inline_linear_sccs: bool = False,
    analytical_linear_scc_limit: int = 2,
    allow_symbolic: bool = False,
    allow_parameter: bool = True,
    **_ignored,
) -> ReassembledSystem:
    """Reassemble the simplified system from a tearing result."""

    var_eq_matching = tearing_result.var_eq_matching
    full_var_eq_matching = tearing_result.full_var_eq_matching
    var_sccs = [list(s) for s in tearing_result.var_sccs]

    extra_eqs_vars = get_extra_eqs_vars(
        state, var_eq_matching, full_var_eq_matching, fully_determined
    )
    neweqs = list(state.eqs)
    dummy_sub = {}
    extra_unknowns = [
        state.fullvars[v] for v in extra_eqs_vars[1]
    ]

    substitute_derivatives_algevars(
        state, neweqs, var_eq_matching, dummy_sub
    )
    var_sccs = generate_derivative_variables(
        state,
        neweqs,
        var_eq_matching,
        full_var_eq_matching,
        var_sccs,
        mm,
    )
    (
        neweqs_out,
        diff_eq_states,
        solved_eqs,
        eq_ordering,
        var_ordering,
        nelim_eq,
        nelim_var,
    ) = generate_system_equations(
        state,
        neweqs,
        var_eq_matching,
        full_var_eq_matching,
        var_sccs,
        extra_eqs_vars,
        simplify=simplify,
        inline_linear_sccs=inline_linear_sccs,
        analytical_linear_scc_limit=analytical_linear_scc_limit,
        allow_symbolic=allow_symbolic,
        allow_parameter=allow_parameter,
    )
    reorder_vars(
        state,
        var_eq_matching,
        var_sccs,
        eq_ordering,
        var_ordering,
        nelim_eq,
        nelim_var,
    )

    # Final unknowns: variables that are not derivatives and occur in
    # the reduced system, in BLT order, plus extra unknowns.
    structure = state.structure
    graph = structure.graph
    var_to_diff = structure.var_to_diff
    diff_to_var = var_to_diff.invview()

    def ispresent(i: int) -> bool:
        if graph.d_neighbors(i):
            return True
        dvi = var_to_diff[i]
        return dvi is not None and bool(graph.d_neighbors(dvi))

    obs_sub = dict(dummy_sub)
    for eq in neweqs_out:
        if eq.lhs != sp.S.Zero:
            obs_sub[eq.lhs] = eq.rhs

    observed = list(solved_eqs)
    observed.extend(
        obs.xreplace(obs_sub) for obs in state.additional_observed
    )

    unknown_idxs = [
        i
        for i in range(len(state.fullvars))
        if diff_to_var[i] is None and ispresent(i)
    ]
    unknowns = [state.fullvars[i] for i in unknown_idxs]
    for extra in extra_unknowns:
        if extra not in unknowns:
            unknowns.append(extra)

    unknown_set = set(unknown_idxs)
    for scc in var_sccs:
        scc[:] = [v for v in scc if v in unknown_set]
    var_sccs = [scc for scc in var_sccs if scc]

    return ReassembledSystem(
        state,
        neweqs_out,
        diff_eq_states,
        solved_eqs,
        observed,
        unknowns,
        var_sccs,
        dummy_sub,
        extra_unknowns,
    )
