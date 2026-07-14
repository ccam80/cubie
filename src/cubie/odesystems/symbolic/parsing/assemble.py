"""Assemble normalised systems into parser products.

Two backends behind the unified front end
(:mod:`~cubie.odesystems.symbolic.parsing.normalise`):
:func:`assemble_explicit` packages a system that is already in
solved explicit form directly into
:class:`~cubie.odesystems.symbolic.parsing.parser.ParsedEquations`,
and :func:`assemble_simplified` runs MTK-style structural
simplification first and maps the result back onto the same
products, together with the
:class:`~cubie.odesystems.symbolic.structural.simplify.SimplifiedSystem`
carrying the mass matrix for torn systems.
"""

from typing import Any, Callable, Dict, Iterable, List, Optional
from warnings import warn

import sympy as sp

from cubie.odesystems.symbolic.indexedbasemaps import IndexedBases
from cubie.odesystems.symbolic.parsing.normalise import (
    NormalisedSystem,
)
from cubie.odesystems.symbolic.parsing.parser import (
    EquationWarning,
    ParsedEquations,
    TIME_SYMBOL,
)
from cubie.odesystems.symbolic.structural.simplify import (
    structural_simplify,
)
from cubie.odesystems.symbolic.structural.system_structure import (
    StructuralState,
)
from cubie.odesystems.symbolic.sym_utils import hash_system_definition


def _observable_substitutions(
    definitions: List,
) -> Dict[sp.Symbol, sp.Expr]:
    """Fully expand observable definitions against each other.

    ``definitions`` is a list of ``(symbol, expression)`` pairs.
    Raises when the definitions contain a cycle.
    """

    subs = {sym: expr for sym, expr in definitions}
    for _ in range(len(subs) + 1):
        expanded = {
            sym: expr.xreplace(subs) for sym, expr in subs.items()
        }
        if expanded == subs:
            return subs
        subs = expanded
    raise AssertionError(
        "observable inlining did not converge; the observable "
        "definitions contain a cycle"
    )


def _finalise_symbols_and_products(
    equation_map,
    index_map,
    user_functions,
    user_function_derivatives,
    rename,
    extra_symbols=(),
):
    """Build ``all_symbols``, ``ParsedEquations``, and the hash."""

    all_symbols = index_map.all_symbols.copy()
    all_symbols.setdefault("t", TIME_SYMBOL)
    for sym in extra_symbols:
        all_symbols.setdefault(sym.name, sym)

    if user_functions:
        all_symbols.update(
            {name: fn for name, fn in user_functions.items()}
        )
        if user_function_derivatives:
            all_symbols.update(
                {
                    fn.__name__: fn
                    for fn in user_function_derivatives.values()
                    if callable(fn)
                }
            )
        if rename:
            all_symbols["__function_aliases__"] = {
                v: k for k, v in rename.items()
            }

    parsed_equations = ParsedEquations.from_equations(
        equation_map, index_map
    )
    fn_hash = hash_system_definition(
        parsed_equations,
        index_map.constants.default_values,
        observable_labels=index_map.observables.ref_map.keys(),
    )
    return all_symbols, parsed_equations, fn_hash


def assemble_explicit(
    normalised: NormalisedSystem,
    states: Dict[str, float],
    observables: List[str],
    parameters,
    constants,
    driver_names: List[str],
    driver_dict: Optional[Dict[str, Any]],
    user_functions: Optional[Dict[str, Callable]],
    user_function_derivatives: Optional[Dict[str, Callable]],
    state_units=None,
    parameter_units=None,
    constant_units=None,
    observable_units=None,
    driver_units=None,
):
    """Package an explicit-shaped normalised system directly.

    The system is already in solved form: every state has one
    first-order derivative equation and the remaining equations are
    output assignments. Observable definitions consumed by the
    dynamics are inlined so the generated ``dxdt`` never reads the
    observables buffer.
    """

    registry = normalised.registry
    index_map = IndexedBases.from_user_inputs(
        states,
        parameters,
        constants,
        observables,
        driver_names,
        state_units=state_units,
        parameter_units=parameter_units,
        constant_units=constant_units,
        observable_units=observable_units,
        driver_units=driver_units,
    )
    for param in normalised.new_params:
        index_map.parameters.push(param)
    if driver_dict is not None:
        index_map.drivers.set_passthrough_defaults(driver_dict)

    observable_set = set(observables)
    pairs = []
    for eq in normalised.equations:
        lhs = eq.lhs
        if registry.is_derivative(lhs):
            base, _ = registry.base_and_order(lhs)
            lhs = sp.Symbol(f"d{base.name}", real=True)
        pairs.append((lhs, eq.rhs))

    observable_defs = [
        (lhs, rhs) for lhs, rhs in pairs if lhs.name in observable_set
    ]
    obs_subs = _observable_substitutions(observable_defs)
    equation_map = [
        (lhs, rhs)
        if lhs.name in observable_set
        else (lhs, rhs.xreplace(obs_subs))
        for lhs, rhs in pairs
    ]

    aux_syms = [
        sp.Symbol(name, real=True) for name in normalised.aux_names
    ]
    all_symbols, parsed_equations, fn_hash = (
        _finalise_symbols_and_products(
            equation_map,
            index_map,
            user_functions,
            user_function_derivatives,
            normalised.rename,
            extra_symbols=aux_syms,
        )
    )
    for param in normalised.new_params:
        all_symbols[str(param)] = param
    return (
        index_map,
        all_symbols,
        normalised.funcs,
        parsed_equations,
        fn_hash,
        None,
    )


def assemble_simplified(
    normalised: NormalisedSystem,
    states: Dict[str, float],
    observables: List[str],
    parameters: Dict[str, float],
    constants: Dict[str, float],
    driver_names: List[str],
    driver_dict: Optional[Dict[str, Any]],
    known_symbol_map: Dict[str, sp.Symbol],
    user_functions: Optional[Dict[str, Callable]],
    user_function_derivatives: Optional[Dict[str, Callable]],
    state_priority: Optional[Dict[str, float]] = None,
    irreducible: Optional[Iterable[str]] = None,
    state_units=None,
    parameter_units=None,
    constant_units=None,
    observable_units=None,
    driver_units=None,
    simplify_options: Optional[Dict[str, Any]] = None,
):
    """Structurally simplify a normalised system and package it.

    Returns the standard parser products plus the
    :class:`~cubie.odesystems.symbolic.structural.simplify.SimplifiedSystem`
    (which carries the mass matrix for torn systems). Solver states
    keep their declaration order where they survive simplification;
    introduced states are appended.
    """

    parameters = dict(parameters)
    for param in normalised.new_params:
        parameters.setdefault(str(param), 0.0)
        known_symbol_map.setdefault(
            str(param), sp.Symbol(str(param), real=True)
        )

    unknown_syms = [
        sp.Symbol(name, real=True)
        for name in sorted(normalised.unknown_names)
    ]
    priorities = {}
    for name, value in (state_priority or {}).items():
        priorities[sp.Symbol(str(name), real=True)] = value
    irreducible_syms = [
        sp.Symbol(str(name), real=True)
        for name in (irreducible or [])
    ]

    structural_state = StructuralState(
        normalised.equations,
        unknown_syms,
        normalised.registry,
        set(known_symbol_map.values()),
        TIME_SYMBOL,
        state_priorities=priorities,
        irreducibles=irreducible_syms,
    )
    simplified = structural_simplify(
        structural_state, **(simplify_options or {})
    )

    # -- Order the solver states -------------------------------------
    # Declared states keep their declaration order where they
    # survive; states introduced by simplification are appended in
    # simplifier order.
    surviving = {sym.name: sym for sym in simplified.states}
    final_states = [
        surviving[name] for name in states if name in surviving
    ]
    final_states += [
        sym for sym in simplified.states if sym.name not in states
    ]

    final_state_values = {}
    default_warned = []
    for sym in final_states:
        if sym.name in states:
            final_state_values[sym.name] = states[sym.name]
        else:
            final_state_values[sym.name] = 0.0
            default_warned.append(sym.name)
    if default_warned:
        warn(
            f"States {default_warned} were introduced by structural "
            "simplification and default to initial value 0.0. Set "
            "initial values through the solver as needed.",
            EquationWarning,
        )

    observed_lhs_names = {sym.name for sym, _ in simplified.observed}
    eliminated = [
        name
        for name in states
        if name not in surviving and name not in observables
    ]
    if eliminated:
        reduced = [
            name for name in eliminated if name in observed_lhs_names
        ]
        removed = [
            name
            for name in eliminated
            if name not in observed_lhs_names
        ]
        if reduced:
            warn(
                f"States {reduced} were eliminated by structural "
                "simplification (they reduce to functions of the "
                "remaining states); their initial values are "
                "ignored. Declare them as observables to record "
                "their trajectories.",
                EquationWarning,
            )
        if removed:
            warn(
                f"States {removed} were removed entirely by "
                "structural simplification; their initial values "
                "are ignored.",
                EquationWarning,
            )

    final_state_names = set(final_state_values)
    final_observables = []
    for name in observables:
        if name in final_state_names:
            warn(
                f"Declared observable {name} was selected as a "
                "solver state; its trajectory is available as a "
                "state output.",
                EquationWarning,
            )
            continue
        if name not in observed_lhs_names:
            raise ValueError(
                f"Observable {name} was declared but has no "
                "defining equation after simplification."
            )
        final_observables.append(name)

    if state_units is not None:
        if not isinstance(state_units, dict):
            state_units = dict(zip(states, state_units))
        state_units = {
            k: v
            for k, v in state_units.items()
            if k in final_state_names
        }

    index_map = IndexedBases.from_user_inputs(
        final_state_values,
        parameters,
        constants,
        final_observables,
        driver_names,
        state_units=state_units,
        parameter_units=parameter_units,
        constant_units=constant_units,
        observable_units=observable_units,
        driver_units=driver_units,
    )
    if driver_dict is not None:
        index_map.drivers.set_passthrough_defaults(driver_dict)

    # Declared observables are pure outputs in cubie: the dxdt device
    # function excludes observable-LHS assignments and reads
    # observable symbols from a buffer that is stale during stage
    # evaluation. Inline their defining expressions into every
    # consuming equation so the dynamics never read that buffer.
    final_observable_set = set(final_observables)
    observable_sub = _observable_substitutions(
        [
            (sym, expr)
            for sym, expr in simplified.observed
            if sym.name in final_observable_set
        ]
    )

    def _inline_observables(expr: sp.Expr) -> sp.Expr:
        return expr.xreplace(observable_sub)

    diff_set = set(simplified.differential_states)
    residual_for = dict(
        zip(simplified.algebraic_states, simplified.residuals)
    )
    equation_map = []
    for sym in final_states:
        lhs = sp.Symbol(f"d{sym.name}", real=True)
        if sym in diff_set:
            equation_map.append(
                (lhs, _inline_observables(simplified.dxdt[sym]))
            )
        elif sym in residual_for:
            equation_map.append(
                (lhs, _inline_observables(residual_for[sym]))
            )
    for sym, expr in simplified.observed:
        if sym in observable_sub:
            # Declared observables keep their chained form; the
            # observables pass evaluates them in topological order.
            equation_map.append((sym, expr))
        else:
            equation_map.append((sym, _inline_observables(expr)))

    # Rebuild the mass matrix over the final state order: identity
    # rows for differential states, zero rows for the residual rows
    # paired to torn algebraic states.
    mass = None
    if simplified.mass_matrix is not None:
        n = len(final_states)
        mass = sp.zeros(n, n)
        for i, sym in enumerate(final_states):
            if sym in diff_set:
                mass[i, i] = sp.S.One
    simplified.mass_matrix = mass
    simplified.states = final_states

    observed_syms = [sym for sym, _ in simplified.observed]
    all_symbols, parsed_equations, fn_hash = (
        _finalise_symbols_and_products(
            equation_map,
            index_map,
            user_functions,
            user_function_derivatives,
            normalised.rename,
            extra_symbols=observed_syms,
        )
    )
    return (
        index_map,
        all_symbols,
        normalised.funcs,
        parsed_equations,
        fn_hash,
        simplified,
    )
