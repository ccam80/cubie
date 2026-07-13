"""Structural simplification pipeline driver.

Port of ModelingToolkit's ``mtkcompile!`` continuous-system pipeline:
perfect-alias elimination, trivial tearing, integer-linear alias
elimination (singularity removal), consistency checking, Pantelides
index reduction with dummy-derivative state selection, tearing, and
reassembly into an explicit (or semi-explicit mass-matrix) system.

Published Classes
-----------------
:class:`SimplifiedSystem`
    The simplification result consumed by cubie's parser/codegen.

Published Functions
-------------------
:func:`structural_simplify`
    Run the full pipeline on a
    :class:`~cubie.odesystems.symbolic.structural.system_structure.StructuralState`.
"""

import warnings
from typing import Dict, List, Optional, Tuple

import sympy as sp

from cubie.odesystems.symbolic.structural.alias_elimination import (
    alias_elimination,
    eliminate_perfect_aliases,
    trivial_tearing,
)
from cubie.odesystems.symbolic.structural.bipartite import (
    Matching,
    UNASSIGNED,
)
from cubie.odesystems.symbolic.structural.consistency import (
    check_consistency,
)
from cubie.odesystems.symbolic.structural.dummy_derivatives import (
    _tear_with_dummies,
    dummy_derivative_graph,
)
from cubie.odesystems.symbolic.structural.pantelides import pantelides
from cubie.odesystems.symbolic.structural.reassemble import (
    ReassembledSystem,
    default_reassemble,
)
from cubie.odesystems.symbolic.structural.singularity_removal import (
    get_new_mm,
)
from cubie.odesystems.symbolic.structural.symbolics import as_small_int
from cubie.odesystems.symbolic.structural.system_structure import (
    Equation,
    StructuralState,
)


class SimplifiedSystem:
    """Result of structural simplification.

    Parameters
    ----------
    states
        Final solver unknowns in BLT order: differential states
        first, then torn algebraic variables.
    differential_states
        The subset of ``states`` integrated through their solved
        derivatives.
    algebraic_states
        The torn (iteration) variables constrained by ``residuals``.
    dxdt
        Map from each differential state to its explicit derivative
        expression.
    residuals
        Algebraic residual expressions (each constrained to zero).
        Empty for fully torn systems.
    observed
        Topologically sorted ``(symbol, expression)`` assignments for
        eliminated variables.
    mass_matrix
        ``None`` when there are no residuals; otherwise the singular
        diagonal mass matrix (identity for differential states, zero
        rows for algebraic constraints).
    dummy_sub
        Renames applied to dummy derivatives.
    var_sccs
        BLT blocks over ``states`` indices.
    state
        The final structural state, for diagnostics.
    """

    def __init__(
        self,
        states: List[sp.Symbol],
        differential_states: List[sp.Symbol],
        algebraic_states: List[sp.Symbol],
        dxdt: Dict[sp.Symbol, sp.Expr],
        residuals: List[sp.Expr],
        observed: List[Tuple[sp.Symbol, sp.Expr]],
        mass_matrix: Optional[sp.Matrix],
        dummy_sub: Dict[sp.Symbol, sp.Symbol],
        var_sccs: List[List[int]],
        state: StructuralState,
    ) -> None:
        self.states = states
        self.differential_states = differential_states
        self.algebraic_states = algebraic_states
        self.dxdt = dxdt
        self.residuals = residuals
        self.observed = observed
        self.mass_matrix = mass_matrix
        self.dummy_sub = dummy_sub
        self.var_sccs = var_sccs
        self.state = state

    def __repr__(self) -> str:
        return (
            f"SimplifiedSystem({len(self.differential_states)} "
            f"differential, {len(self.algebraic_states)} algebraic, "
            f"{len(self.observed)} observed)"
        )


def _integer_jacobian(state: StructuralState):
    """Integer Jacobian closure for dummy-derivative rank checks."""

    def jac(
        eq_idxs: List[int], var_idxs: List[int]
    ) -> Optional[List[List[int]]]:
        rows = []
        for e in eq_idxs:
            rhs = state.eqs[e].rhs
            row = []
            for v in var_idxs:
                entry = sp.diff(rhs, state.fullvars[v])
                if not entry.is_number:
                    return None
                small = as_small_int(entry)
                if small is None:
                    return None
                row.append(small)
            rows.append(row)
        return rows

    return jac


def _pantelides_reassemble_state(
    state: StructuralState, var_eq_matching: Matching
) -> StructuralState:
    """Rebuild a first-analysis state after bare index reduction.

    Port of MTK's ``pantelides_reassemble``: keep, for each matched
    highest-differentiated variable, the (differentiated) equation it
    is matched to, and rebuild a fresh structural state from those
    equations.
    """

    matched_eqs = sorted(
        {
            e
            for e in var_eq_matching
            if isinstance(e, int)
        }
    )
    new_eqs = [state.eqs[e] for e in matched_eqs]
    unknowns = []
    seen = set()
    for v in state.fullvars:
        base, _ = state.registry.base_and_order(v)
        if base not in seen:
            seen.add(base)
            unknowns.append(base)
    priorities = {
        state.fullvars[i]: state.structure.state_priorities[i]
        for i in range(len(state.fullvars))
    }
    return StructuralState(
        new_eqs,
        unknowns,
        state.registry,
        state.known_symbols - {state.time_symbol},
        state.time_symbol,
        known_derivative_map=state.known_derivative_map,
        state_priorities=priorities,
        irreducibles=state.irreducibles,
    )


def _assemble_result(
    reassembled: ReassembledSystem,
) -> SimplifiedSystem:
    """Convert a reassembled system into the cubie-facing result."""

    state = reassembled.state

    dxdt = {}
    differential_states = []
    residuals = []
    for eq, diff_state in zip(
        reassembled.neweqs, reassembled.diff_eq_states
    ):
        if diff_state is None:
            residuals.append(eq.rhs)
            continue
        differential_states.append(diff_state)
        dxdt[diff_state] = eq.rhs

    diff_set = set(differential_states)
    algebraic_states = [
        s for s in reassembled.unknowns if s not in diff_set
    ]
    states = differential_states + algebraic_states

    if len(residuals) != len(algebraic_states):
        # Balanced systems always pair up; unbalanced systems (run
        # with fully_determined=False) may not.
        warnings.warn(
            f"{len(residuals)} residual equations for "
            f"{len(algebraic_states)} algebraic states; the system "
            "is not fully determined"
        )

    n = len(states)
    if residuals:
        mass = sp.zeros(n, n)
        for i in range(len(differential_states)):
            mass[i, i] = sp.S.One
    else:
        mass = None

    observed = _topsort_observed(
        [(eq.lhs, eq.rhs) for eq in reassembled.observed]
    )

    return SimplifiedSystem(
        states,
        differential_states,
        algebraic_states,
        dxdt,
        residuals,
        observed,
        mass,
        reassembled.dummy_sub,
        reassembled.var_sccs,
        state,
    )


def _topsort_observed(
    observed: List[Tuple[sp.Expr, sp.Expr]],
) -> List[Tuple[sp.Symbol, sp.Expr]]:
    """Topologically sort observed assignments by dependency."""

    from cubie.odesystems.symbolic.sym_utils import topological_sort

    pairs = []
    seen = set()
    for lhs, rhs in observed:
        if not isinstance(lhs, sp.Symbol):
            raise AssertionError(
                f"observed equation LHS {lhs} is not a symbol"
            )
        if lhs in seen:
            continue
        seen.add(lhs)
        pairs.append((lhs, rhs))
    return topological_sort(pairs)


def structural_simplify(
    state: StructuralState,
    fully_determined: bool = True,
    dummy_derivative: bool = True,
    consistency_check: bool = True,
    simplify: bool = False,
    conservative: bool = False,
    allow_symbolic: bool = False,
    allow_parameter: bool = True,
    inline_linear_sccs: bool = False,
    analytical_linear_scc_limit: int = 2,
) -> SimplifiedSystem:
    """Run the full structural simplification pipeline.

    Parameters
    ----------
    state
        The structural state to simplify (mutated).
    fully_determined
        Whether the system must have matching equation and unknown
        counts; disables the consistency check and index reduction
        when false (tearing only).
    dummy_derivative
        Use dummy-derivative state selection (the default MTK path).
        When false, bare Pantelides index reduction runs first and
        the resulting system is re-analysed.
    consistency_check
        Verify balance/nonsingularity before state selection.
    simplify
        Apply :func:`sympy.simplify` to solved right-hand sides.
    conservative
        Restrict tearing to coefficients with absolute value one.
    allow_symbolic, allow_parameter
        Solvability limits on symbolic pivots (division safety).
    inline_linear_sccs, analytical_linear_scc_limit
        Solve small linear algebraic SCCs analytically instead of
        leaving them as residuals.
    """

    solve_kwargs = {
        "allow_symbolic": allow_symbolic,
        "allow_parameter": allow_parameter,
        "conservative": conservative,
    }

    eliminate_perfect_aliases(state)
    trivial_tearing(state)
    mm = alias_elimination(state, **solve_kwargs)
    old_to_new_eq, old_to_new_var, aliases = (
        eliminate_perfect_aliases(state)
    )
    mm = get_new_mm(aliases, old_to_new_eq, old_to_new_var, mm)
    state.mm = mm
    if mm.nparentrows != state.structure.graph.nsrcs() or (
        mm.ncols != state.structure.graph.ndsts()
    ):
        raise AssertionError(
            f"Invalid mm. Got (nparentrows, ncols) = "
            f"({mm.nparentrows}, {mm.ncols}). Expected "
            f"({state.structure.graph.nsrcs()}, "
            f"{state.structure.graph.ndsts()})."
        )

    if consistency_check and fully_determined:
        check_consistency(state)

    reassemble_kwargs = {
        "fully_determined": fully_determined,
        "simplify": simplify,
        "inline_linear_sccs": inline_linear_sccs,
        "analytical_linear_scc_limit": analytical_linear_scc_limit,
        "allow_symbolic": allow_symbolic,
        "allow_parameter": allow_parameter,
    }

    if fully_determined and dummy_derivative:
        tearing_result, _extras = dummy_derivative_graph(
            state,
            _integer_jacobian(state),
            state_priority=lambda v: (
                state.structure.state_priorities[v]
            ),
            **solve_kwargs,
        )
        reassembled = default_reassemble(
            state, tearing_result, state.mm, **reassemble_kwargs
        )
    elif fully_determined:
        # Bare index reduction, then re-analyse and select states.
        if state.structure.solvable_graph is None:
            state.find_solvables(**solve_kwargs)
        state.structure.complete()
        var_eq_matching = pantelides(state, finalize=False)
        state = _pantelides_reassemble_state(state, var_eq_matching)
        mm = alias_elimination(state, **solve_kwargs)
        state.mm = mm
        tearing_result, _extras = dummy_derivative_graph(
            state,
            _integer_jacobian(state),
            state_priority=lambda v: (
                state.structure.state_priorities[v]
            ),
            **solve_kwargs,
        )
        reassembled = default_reassemble(
            state, tearing_result, state.mm, **reassemble_kwargs
        )
    else:
        if state.structure.solvable_graph is None:
            state.find_solvables(**solve_kwargs)
        state.structure.complete()
        tearing_result, _extras = _tear_with_dummies(
            state.structure, set(), None, state.mm
        )
        reassembled = default_reassemble(
            state, tearing_result, state.mm, **reassemble_kwargs
        )

    return _assemble_result(reassembled)
