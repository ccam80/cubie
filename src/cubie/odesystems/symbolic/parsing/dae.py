"""DAE input parsing and structural simplification front end.

Accepts general differential-algebraic equations — implicit equations
(``0 = g(...)``), higher-order derivatives (nested ``d(d(x, t), t)``
or :class:`sympy.Derivative` of any order), derivative terms anywhere
in an expression, and algebraic unknowns — runs MTK-style structural
simplification (
:func:`~cubie.odesystems.symbolic.structural.simplify.structural_simplify`
), and maps the simplified system back onto the standard parser
products consumed by
:class:`~cubie.odesystems.symbolic.symbolicODE.SymbolicODE`.

Published Functions
-------------------
:func:`parse_dae_input`
    Parse, simplify, and return
    ``(index_map, all_symbols, funcs, parsed_equations, fn_hash,
    simplified)``.
"""

import re
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union
from warnings import warn

import sympy as sp
from sympy.core.function import AppliedUndef
from sympy.parsing.sympy_parser import parse_expr

from cubie.odesystems.symbolic.indexedbasemaps import IndexedBases
from cubie.odesystems.symbolic.parsing.parser import (
    DRIVER_SETTING_KEYS,
    EquationWarning,
    KNOWN_FUNCTIONS,
    PARSE_TRANSFORMS,
    TIME_SYMBOL,
    ParsedEquations,
    _build_sympy_user_functions,
    _func_call_re,
    _inline_nondevice_calls,
    _normalise_indexed_tokens,
    _rename_user_calls,
    _sanitise_input_math,
)
from cubie.odesystems.symbolic.structural.simplify import (
    SimplifiedSystem,
    structural_simplify,
)
from cubie.odesystems.symbolic.structural.symbolics import (
    DerivativeRegistry,
)
from cubie.odesystems.symbolic.structural.system_structure import (
    Equation,
    StructuralState,
)
from cubie.odesystems.symbolic.sym_utils import hash_system_definition

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_]\w*$")


def _replace_derivative_calls(
    expr: sp.Expr,
    registry: DerivativeRegistry,
    unknown_names: set,
) -> sp.Expr:
    """Replace ``d(x, t)`` calls with registry derivative symbols.

    Nested calls resolve innermost-first, producing higher-order
    derivative symbols. Raises when the differentiated quantity is
    not an unknown symbol.
    """

    def repl(call: sp.Expr) -> sp.Expr:
        args = call.args
        if len(args) != 2 or args[1] != TIME_SYMBOL:
            raise ValueError(
                f"Derivative notation must be d(<symbol>, t); got "
                f"{call}"
            )
        inner = args[0]
        if not isinstance(inner, sp.Symbol):
            raise ValueError(
                f"Cannot differentiate non-symbol expression {inner} "
                "in d() notation."
            )
        base, _ = registry.base_and_order(inner)
        if base.name not in unknown_names:
            raise ValueError(
                f"d({inner}, t) differentiates {inner}, but "
                f"{base} is not a declared state or unknown."
            )
        return registry.derivative(inner)

    def is_d_call(node: sp.Basic) -> bool:
        return (
            isinstance(node, AppliedUndef)
            and node.func.__name__ == "d"
        )

    # replace() rebuilds post-order, so inner calls resolve before
    # outer ones and nesting yields higher-order symbols.
    return expr.replace(is_d_call, repl)


def _replace_sympy_derivatives(
    expr: sp.Expr,
    registry: DerivativeRegistry,
    unknown_names: set,
) -> sp.Expr:
    """Replace :class:`sympy.Derivative` nodes with registry symbols."""

    def repl(node: sp.Derivative) -> sp.Expr:
        inner = node.expr
        if not isinstance(inner, sp.Symbol):
            raise ValueError(
                f"Cannot differentiate non-symbol expression "
                f"{inner}."
            )
        if inner.name not in unknown_names:
            raise ValueError(
                f"Derivative of {inner} found, but {inner} is not a "
                "declared state or unknown."
            )
        order = 0
        for var, count in node.variable_count:
            if var != TIME_SYMBOL and str(var) != "t":
                raise ValueError(
                    f"Only time derivatives are supported; got "
                    f"{node}."
                )
            order += int(count)
        sym = inner
        for _ in range(order):
            sym = registry.derivative(sym)
        return sym

    return expr.replace(
        lambda node: isinstance(node, sp.Derivative), repl
    )


def _process_dae_calls(
    lines: Iterable[str],
    user_functions: Optional[Dict[str, Callable]],
) -> Dict[str, Callable]:
    """Resolve callables referenced in equations, allowing ``d()``."""

    calls = set()
    user_functions = user_functions or {}
    for line in lines:
        calls |= set(_func_call_re.findall(line))
    calls.discard("d")
    funcs = {}
    for name in calls:
        if name in user_functions:
            funcs[name] = user_functions[name]
        elif name in KNOWN_FUNCTIONS:
            funcs[name] = KNOWN_FUNCTIONS[name]
        else:
            raise ValueError(
                f"Your equations contain a call to a function "
                f"{name}() that isn't part of Sympy and wasn't "
                f"provided in the user_functions dict."
            )
    return funcs


def _parse_string_equations(
    lines,
    raw_lines,
    registry,
    unknown_names,
    known_symbol_map,
    user_functions,
    user_function_derivatives,
    strict,
):
    """Parse ``lhs = rhs`` lines into structural equations.

    Returns ``(equations, funcs, new_params, aux_names)``.
    """

    funcs = _process_dae_calls(lines, user_functions)
    sanitized_lines, rename = _rename_user_calls(
        lines, user_functions or {}
    )
    parse_locals, _alias_map, _dev_map = _build_sympy_user_functions(
        user_functions or {}, rename, user_function_derivatives
    )

    local_dict = dict(known_symbol_map)
    local_dict.update(parse_locals)
    local_dict.setdefault("t", TIME_SYMBOL)
    local_dict["d"] = sp.Function("d")
    for name in unknown_names:
        local_dict.setdefault(name, sp.Symbol(name, real=True))

    equations = []
    new_params = []
    aux_names = []
    for raw_line, line in zip(raw_lines, sanitized_lines):
        lhs_str, rhs_str = [p.strip() for p in line.split("=", 1)]

        rhs_text = _sanitise_input_math(rhs_str)
        if strict:
            try:
                rhs_expr = parse_expr(
                    rhs_text,
                    transformations=PARSE_TRANSFORMS,
                    local_dict=local_dict,
                )
            except (NameError, TypeError) as exc:
                raise ValueError(
                    f"Undefined symbols in equation '{raw_line}'"
                ) from exc
        else:
            # Without transformations parse_expr auto-creates
            # symbols for undeclared names (inferred as parameters
            # below), mirroring the standard parser.
            rhs_expr = parse_expr(rhs_text, local_dict=local_dict)
        rhs_expr = _inline_nondevice_calls(
            rhs_expr, user_functions or {}, rename
        )
        rhs_expr = _replace_derivative_calls(
            rhs_expr, registry, unknown_names
        )

        if lhs_str == "0":
            lhs_expr = sp.S.Zero
        elif lhs_str.startswith("d") and "(" in lhs_str:
            lhs_expr = parse_expr(
                lhs_str,
                transformations=PARSE_TRANSFORMS,
                local_dict=local_dict,
            )
            lhs_expr = _replace_derivative_calls(
                lhs_expr, registry, unknown_names
            )
        elif _IDENTIFIER_PATTERN.match(lhs_str):
            if (
                lhs_str.startswith("d")
                and len(lhs_str) > 1
                and lhs_str[1:] in unknown_names
            ):
                lhs_expr = registry.derivative(
                    sp.Symbol(lhs_str[1:], real=True)
                )
            elif lhs_str in known_symbol_map:
                raise ValueError(
                    f"{lhs_str} is an immutable input (constant, "
                    "parameter, or driver) but is being assigned. It "
                    "must be a state, observable, or auxiliary."
                )
            else:
                lhs_expr = local_dict.get(
                    lhs_str, sp.Symbol(lhs_str, real=True)
                )
                if lhs_str not in unknown_names:
                    if strict:
                        raise ValueError(
                            f"Equation assigns undeclared symbol "
                            f"{lhs_str} (strict mode)."
                        )
                    aux_names.append(lhs_str)
                    unknown_names.add(lhs_str)
                    local_dict.setdefault(lhs_str, lhs_expr)
        else:
            raise ValueError(
                f"Unsupported left-hand side '{lhs_str}' in equation "
                f"'{raw_line}'. Expected a symbol, dX, d(x, t), or 0."
            )

        equations.append(Equation(lhs_expr, rhs_expr))

    # Infer undeclared RHS symbols as parameters (non-strict), after
    # derivative replacement so derivative symbols don't count.
    # Auto-created symbols lack the real assumption, so they are
    # coerced to the canonical real symbols used everywhere else.
    declared = set(known_symbol_map.values()) | {TIME_SYMBOL}
    declared |= {
        sp.Symbol(name, real=True) for name in unknown_names
    }
    inferred_subs = {}
    for eq in equations:
        for sym in eq.free_symbols():
            if sym in declared or sym in inferred_subs:
                continue
            if registry.is_derivative(sym):
                continue
            if strict:
                raise ValueError(
                    f"Equations reference undefined symbol {sym}."
                )
            real_sym = sp.Symbol(sym.name, real=True)
            inferred_subs[sym] = real_sym
            new_params.append(real_sym)
            declared.add(real_sym)
    if inferred_subs:
        for i in range(len(equations)):
            equations[i] = equations[i].xreplace(inferred_subs)

    return equations, funcs, new_params, aux_names


def _parse_sympy_equations(
    dxdt,
    registry,
    unknown_names,
    known_symbol_map,
    strict,
):
    """Normalise SymPy DAE input into structural equations."""

    if isinstance(dxdt, (list, tuple)):
        raw_equations = list(dxdt)
    else:
        raw_equations = [dxdt]

    canonical = {}
    for name, sym in known_symbol_map.items():
        canonical[sp.Symbol(name)] = sym
        canonical[sp.Symbol(name, real=True)] = sym
    for name in unknown_names:
        canonical[sp.Symbol(name)] = sp.Symbol(name, real=True)

    equations = []
    for i, eq in enumerate(raw_equations):
        if isinstance(eq, sp.Equality):
            lhs, rhs = eq.lhs, eq.rhs
        elif isinstance(eq, tuple) and len(eq) == 2:
            lhs, rhs = eq
        else:
            raise TypeError(
                f"Equation {i}: expected sp.Eq or a (lhs, rhs) "
                f"tuple, got {type(eq).__name__}."
            )
        lhs = sp.sympify(lhs).subs(canonical, simultaneous=True)
        rhs = sp.sympify(rhs).subs(canonical, simultaneous=True)
        lhs = _replace_sympy_derivatives(lhs, registry, unknown_names)
        rhs = _replace_sympy_derivatives(rhs, registry, unknown_names)
        equations.append(Equation(lhs, rhs))

    new_params = []
    aux_names = []
    declared = set(known_symbol_map.values()) | {TIME_SYMBOL}
    declared |= {
        sp.Symbol(name, real=True) for name in unknown_names
    }
    for eq in equations:
        lhs = eq.lhs
        if (
            isinstance(lhs, sp.Symbol)
            and lhs not in declared
            and not registry.is_derivative(lhs)
        ):
            if strict:
                raise ValueError(
                    f"Equation assigns undeclared symbol {lhs} "
                    "(strict mode)."
                )
            aux_names.append(lhs.name)
            unknown_names.add(lhs.name)
            declared.add(lhs)
        for sym in eq.free_symbols():
            if sym in declared or registry.is_derivative(sym):
                continue
            if strict:
                raise ValueError(
                    f"Equations reference undefined symbol {sym}."
                )
            new_params.append(sym)
            declared.add(sym)
    return equations, {}, new_params, aux_names


def parse_dae_input(
    dxdt: Union[str, Iterable],
    states: Optional[Union[Dict[str, float], Iterable[str]]] = None,
    observables: Optional[Iterable[str]] = None,
    parameters: Optional[Union[Dict[str, float], Iterable[str]]] = None,
    constants: Optional[Union[Dict[str, float], Iterable[str]]] = None,
    drivers: Optional[Union[Iterable[str], Dict[str, Any]]] = None,
    user_functions: Optional[Dict[str, Callable]] = None,
    user_function_derivatives: Optional[Dict[str, Callable]] = None,
    strict: bool = False,
    state_priority: Optional[Dict[str, float]] = None,
    irreducible: Optional[Iterable[str]] = None,
    state_units: Optional[Union[Dict[str, str], Iterable[str]]] = None,
    parameter_units: Optional[
        Union[Dict[str, str], Iterable[str]]
    ] = None,
    constant_units: Optional[Union[Dict[str, str], Iterable[str]]] = None,
    observable_units: Optional[
        Union[Dict[str, str], Iterable[str]]
    ] = None,
    driver_units: Optional[Union[Dict[str, str], Iterable[str]]] = None,
    **simplify_options,
) -> Tuple[
    IndexedBases,
    Dict[str, object],
    Dict[str, Callable],
    ParsedEquations,
    str,
    SimplifiedSystem,
]:
    """Parse DAE input, structurally simplify, and package the result.

    Parameters
    ----------
    dxdt
        Equations as strings or SymPy equations. In addition to the
        standard explicit forms, implicit equations (``0 = g(...)``),
        higher-order/nested derivatives, and derivative terms inside
        expressions are accepted.
    states
        All unknowns of the DAE (differential or algebraic — the
        simplifier decides which become solver states) with initial
        values.
    observables
        Output variables to record. They participate as unknowns and
        must be defined by the equations.
    parameters, constants, drivers
        As for :func:`~cubie.odesystems.symbolic.parsing.parser.parse_input`.
    state_priority
        Optional per-unknown priorities; higher-priority variables
        are preferred as solver states.
    irreducible
        Unknowns that must not be eliminated.
    **simplify_options
        Forwarded to
        :func:`~cubie.odesystems.symbolic.structural.simplify.structural_simplify`.

    Returns
    -------
    tuple
        ``(index_map, all_symbols, funcs, parsed_equations, fn_hash,
        simplified)`` — the standard parser products plus the
        :class:`~cubie.odesystems.symbolic.structural.simplify.SimplifiedSystem`
        (which carries the mass matrix for torn systems).

    Notes
    -----
    Newly introduced states (dummy derivatives such as ``x_t``)
    default to an initial value of ``0.0`` with a warning; set their
    initial values through the solver as needed. Driver symbols
    differentiate to zero during index reduction, matching MTK's
    default for time-dependent parameters.
    """

    if callable(dxdt) and not isinstance(dxdt, (str, list, tuple)):
        raise TypeError(
            "Callable dxdt input is explicit-ODE only and cannot be "
            "combined with simplify=True."
        )

    states = dict(states) if isinstance(states, dict) else {
        name: 0.0 for name in (states or [])
    }
    observables = list(observables or [])
    parameters = parameters if parameters is not None else {}
    constants = constants if constants is not None else {}
    if isinstance(parameters, dict):
        parameters = dict(parameters)
    else:
        parameters = {name: 0.0 for name in parameters}
    if isinstance(constants, dict):
        constants = dict(constants)
    else:
        constants = {name: 0.0 for name in constants}

    driver_dict = None
    if drivers is None:
        driver_names = []
    elif isinstance(drivers, dict):
        driver_dict = drivers
        driver_names = [
            key
            for key in drivers.keys()
            if key not in DRIVER_SETTING_KEYS
        ]
        if not driver_names:
            raise ValueError(
                "Driver dictionary must include at least one driver "
                "symbol."
            )
    else:
        driver_names = list(drivers)

    known_symbol_map = {}
    for name in list(parameters) + list(constants) + driver_names:
        known_symbol_map[str(name)] = sp.Symbol(str(name), real=True)

    unknown_names = {str(name) for name in states}
    unknown_names |= {str(name) for name in observables}

    reserved = (
        set(known_symbol_map)
        | unknown_names
        | {"t"}
        | set(KNOWN_FUNCTIONS)
    )
    registry = DerivativeRegistry(reserved)

    raw_lines = []
    if isinstance(dxdt, str) or (
        isinstance(dxdt, (list, tuple))
        and dxdt
        and isinstance(dxdt[0], str)
    ):
        if isinstance(dxdt, str):
            lines = [
                ln.strip()
                for ln in dxdt.strip().splitlines()
                if ln.strip()
            ]
        else:
            lines = [ln.strip() for ln in dxdt if ln.strip()]
        raw_lines = list(lines)
        lines = _normalise_indexed_tokens(lines)
        equations, funcs, new_params, aux_names = (
            _parse_string_equations(
                lines,
                raw_lines,
                registry,
                unknown_names,
                known_symbol_map,
                user_functions,
                user_function_derivatives,
                strict,
            )
        )
    else:
        equations, funcs, new_params, aux_names = (
            _parse_sympy_equations(
                dxdt,
                registry,
                unknown_names,
                known_symbol_map,
                strict,
            )
        )

    for param in new_params:
        parameters[str(param)] = 0.0
        known_symbol_map[str(param)] = sp.Symbol(
            str(param), real=True
        )

    unknown_syms = [
        sp.Symbol(name, real=True) for name in sorted(unknown_names)
    ]
    priorities = {}
    for name, value in (state_priority or {}).items():
        priorities[sp.Symbol(str(name), real=True)] = value
    irreducible_syms = [
        sp.Symbol(str(name), real=True) for name in (irreducible or [])
    ]

    state = StructuralState(
        equations,
        unknown_syms,
        registry,
        set(known_symbol_map.values()),
        TIME_SYMBOL,
        state_priorities=priorities,
        irreducibles=irreducible_syms,
    )
    simplified = structural_simplify(state, **simplify_options)

    # -- Map the simplified system back to parser products ----------
    final_state_values = {}
    default_warned = []
    for sym in simplified.states:
        name = sym.name
        if name in states:
            final_state_values[name] = states[name]
        else:
            final_state_values[name] = 0.0
            default_warned.append(name)
    if default_warned:
        warn(
            f"States {default_warned} were introduced by structural "
            "simplification and default to initial value 0.0. Set "
            "initial values through the solver if needed.",
            EquationWarning,
        )

    observed_lhs_names = {sym.name for sym, _ in simplified.observed}
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

    if isinstance(state_units, dict):
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
    observable_sub = {}
    for sym, expr in simplified.observed:
        if sym.name in set(final_observables):
            observable_sub[sym] = expr.xreplace(observable_sub)

    def _inline_observables(expr: sp.Expr) -> sp.Expr:
        for _ in range(len(observable_sub) + 1):
            new_expr = expr.xreplace(observable_sub)
            if new_expr == expr:
                return new_expr
            expr = new_expr
        return expr

    equation_map = []
    for sym in simplified.differential_states:
        lhs = sp.Symbol(f"d{sym.name}", real=True)
        equation_map.append(
            (lhs, _inline_observables(simplified.dxdt[sym]))
        )
    for sym, residual in zip(
        simplified.algebraic_states, simplified.residuals
    ):
        lhs = sp.Symbol(f"d{sym.name}", real=True)
        equation_map.append((lhs, _inline_observables(residual)))
    for sym, expr in simplified.observed:
        if sym in observable_sub:
            equation_map.append((sym, expr))
        else:
            equation_map.append((sym, _inline_observables(expr)))

    all_symbols = index_map.all_symbols.copy()
    all_symbols.setdefault("t", TIME_SYMBOL)
    for sym, _expr in simplified.observed:
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
        _, rename = _rename_user_calls(
            raw_lines, user_functions or {}
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
    return (
        index_map,
        all_symbols,
        funcs,
        parsed_equations,
        fn_hash,
        simplified,
    )
