"""Normalise user equation input into structural equations.

The single symbolic front end: string and SymPy input converge on
one representation — a list of
:class:`~cubie.odesystems.symbolic.structural.system_structure.Equation`
objects whose derivatives are
:class:`~cubie.odesystems.symbolic.structural.symbolics.DerivativeRegistry`
symbols — plus the resolved declarations. :func:`classify_system`
then decides whether the system is already in solved explicit form
(the fast assembly path) or needs structural simplification.

Left-hand sides are state-aware: ``dX`` names the derivative of
``X`` only when ``X`` is a declared unknown (otherwise ``dX`` is an
auxiliary, or an inferred state when no states were declared).
Right-hand-side ``dX`` tokens are *not* derivatives; they bind to
the ``dX`` assignment emitted for state ``X``, preserving
assignment-reference semantics. Derivatives inside expressions use
the explicit ``d(x, t)`` call (strings) or
:class:`sympy.Derivative` (SymPy input).

Published Functions
-------------------
:func:`normalise_input`
    Parse string or SymPy equations into a
    :class:`NormalisedSystem`.
:func:`classify_system`
    Return ``"explicit"`` or ``"dae"`` for a normalised system.
"""

import re
from typing import Callable, Dict, Iterable, List, Optional

import sympy as sp
from sympy.core.function import AppliedUndef
from sympy.parsing.sympy_parser import parse_expr

from cubie.odesystems.symbolic.parsing.parser import (
    KNOWN_FUNCTIONS,
    PARSE_TRANSFORMS,
    TIME_SYMBOL,
    _build_sympy_user_functions,
    _func_call_re,
    _inline_nondevice_calls,
    _normalise_indexed_tokens,
    _rename_user_calls,
    _sanitise_input_math,
)
from cubie.odesystems.symbolic.structural.symbolics import (
    DerivativeRegistry,
)
from cubie.odesystems.symbolic.structural.system_structure import (
    Equation,
)

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_]\w*$")
_NUMERIC_LITERAL_PATTERN = re.compile(
    r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$"
)
_D_CALL_LHS_PATTERN = re.compile(
    r"^d\s*\(\s*([A-Za-z_]\w*)\s*,\s*t\s*\)$"
)


class NormalisedSystem:
    """Equations and declarations in the shared front-end form.

    Parameters
    ----------
    equations
        Structural equations with registry derivative symbols.
    registry
        The derivative registry used by ``equations``.
    funcs
        Callables referenced by the equations, resolved against
        ``user_functions`` and :data:`KNOWN_FUNCTIONS`.
    unknown_names
        All unknown names (declared states, observables, and
        inferred auxiliaries).
    aux_names
        Names of auxiliaries inferred from assignments.
    new_params
        Symbols inferred as parameters from equation usage.
    inferred_states
        State names inferred from derivative assignments when no
        states were declared (non-strict input only).
    rename
        User-function rename map from the string path (empty for
        SymPy input).
    """

    def __init__(
        self,
        equations: List[Equation],
        registry: DerivativeRegistry,
        funcs: Dict[str, Callable],
        unknown_names: set,
        aux_names: List[str],
        new_params: List[sp.Symbol],
        inferred_states: List[str],
        rename: Dict[str, str],
    ) -> None:
        self.equations = equations
        self.registry = registry
        self.funcs = funcs
        self.unknown_names = unknown_names
        self.aux_names = aux_names
        self.new_params = new_params
        self.inferred_states = inferred_states
        self.rename = rename


def _process_calls(
    lines: Iterable[str],
    user_functions: Optional[Dict[str, Callable]] = None,
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


def _parse_string_equations(
    lines,
    raw_lines,
    registry,
    unknown_names,
    known_symbol_map,
    user_functions,
    user_function_derivatives,
    strict,
    state_names,
):
    """Parse ``lhs = rhs`` lines into structural equations.

    Returns ``(equations, funcs, new_params, aux_names,
    inferred_states, rename)``.
    """

    funcs = _process_calls(lines, user_functions)
    sanitized_lines, rename = _rename_user_calls(
        lines, user_functions or {}
    )
    parse_locals, _alias_map, _dev_map = _build_sympy_user_functions(
        user_functions or {}, rename, user_function_derivatives
    )

    local_dict = dict(known_symbol_map)
    local_dict.update(parse_locals)
    local_dict.setdefault("t", TIME_SYMBOL)
    # The d(x, t) notation reserves the name ``d`` as a function
    # only when it is used; otherwise ``d`` stays available as an
    # ordinary symbol.
    if any(re.search(r"\bd\s*\(", line) for line in sanitized_lines):
        local_dict["d"] = sp.Function("d")
    for name in unknown_names:
        local_dict.setdefault(name, sp.Symbol(name, real=True))
    # RHS references to dX bind to the derivative assignment of
    # state X (assignment-reference semantics, not a derivative
    # term). Declared names win over the generated symbol.
    derivative_names = set()
    for name in state_names:
        dname = f"d{name}"
        if dname not in local_dict:
            local_dict[dname] = sp.Symbol(dname, real=True)
            derivative_names.add(dname)

    equations = []
    new_params = []
    aux_names = []
    inferred_states = []

    def infer_state(name):
        unknown_names.add(name)
        inferred_states.append(name)
        local_dict.setdefault(name, sp.Symbol(name, real=True))
        dname = f"d{name}"
        if dname not in local_dict:
            local_dict[dname] = sp.Symbol(dname, real=True)
            derivative_names.add(dname)

    for raw_line, line in zip(raw_lines, sanitized_lines):
        lhs_str, rhs_str = [p.strip() for p in line.split("=", 1)]

        # The left-hand side is analysed before the right-hand side
        # so state inference and LHS-specific errors take priority
        # over undefined-symbol errors from the RHS parse.
        if _NUMERIC_LITERAL_PATTERN.match(lhs_str):
            lhs_expr = sp.Number(lhs_str)
        elif lhs_str.startswith("d") and "(" in lhs_str:
            # d(x, t) notation: explicit derivative intent, so an
            # undeclared x is inferred as a state in non-strict
            # mode.
            inner_match = _D_CALL_LHS_PATTERN.match(lhs_str)
            if (
                inner_match
                and inner_match.group(1) not in unknown_names
            ):
                name = inner_match.group(1)
                if strict:
                    raise ValueError(
                        f"Unknown state in derivative notation: "
                        f"d({name}, t). No state called {name} "
                        "found."
                    )
                infer_state(name)
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
            elif (
                lhs_str.startswith("d")
                and len(lhs_str) > 1
                and not state_names
                and not strict
                and _IDENTIFIER_PATTERN.match(lhs_str[1:])
            ):
                # No states declared: a dX assignment infers state X
                # (quickstart convenience). With declared states,
                # unknown d-prefixed names are auxiliaries instead.
                name = lhs_str[1:]
                infer_state(name)
                lhs_expr = registry.derivative(
                    sp.Symbol(name, real=True)
                )
            elif lhs_str in known_symbol_map:
                raise ValueError(
                    f"{lhs_str} is an immutable input (constant, "
                    "parameter, or driver) but is being assigned. It "
                    "must be a state, observable, or auxiliary."
                )
            else:
                candidate = local_dict.get(lhs_str)
                if isinstance(candidate, sp.Symbol):
                    lhs_expr = candidate
                else:
                    lhs_expr = sp.Symbol(lhs_str, real=True)
                if lhs_str not in unknown_names:
                    # An assignment defines the symbol, so even
                    # strict mode admits anonymous auxiliaries.
                    aux_names.append(lhs_str)
                    unknown_names.add(lhs_str)
                    local_dict.setdefault(lhs_str, lhs_expr)
        else:
            raise ValueError(
                f"Unsupported left-hand side '{lhs_str}' in equation "
                f"'{raw_line}'. Expected a symbol, dX, d(x, t), or a "
                "number (implicit equation)."
            )

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
            # below).
            rhs_expr = parse_expr(rhs_text, local_dict=local_dict)
        rhs_expr = _inline_nondevice_calls(
            rhs_expr, user_functions or {}, rename
        )
        rhs_expr = _replace_derivative_calls(
            rhs_expr, registry, unknown_names
        )

        equations.append(Equation(lhs_expr, rhs_expr))

    # Infer undeclared RHS symbols as parameters (non-strict), after
    # derivative replacement so derivative symbols don't count.
    # Exclusion is by name so a forward reference to a later ``dX``
    # assignment binds to it instead of becoming a parameter.
    # Auto-created symbols lack the real assumption, so they are
    # coerced to the canonical real symbols used everywhere else.
    declared_names = (
        set(known_symbol_map)
        | unknown_names
        | derivative_names
        | {"t"}
    )
    inferred_subs = {}
    new_param_names = set()
    for eq in equations:
        for sym in eq.free_symbols():
            if sym in inferred_subs or registry.is_derivative(sym):
                continue
            if sym.name in declared_names:
                if not sym.assumptions0.get("real"):
                    inferred_subs[sym] = sp.Symbol(
                        sym.name, real=True
                    )
                continue
            if strict:
                raise ValueError(
                    f"Equations reference undefined symbol {sym}."
                )
            real_sym = sp.Symbol(sym.name, real=True)
            inferred_subs[sym] = real_sym
            if sym.name not in new_param_names:
                new_param_names.add(sym.name)
                new_params.append(real_sym)
            declared_names.add(sym.name)
    if inferred_subs:
        for i in range(len(equations)):
            equations[i] = equations[i].xreplace(inferred_subs)

    return (
        equations,
        funcs,
        new_params,
        aux_names,
        inferred_states,
        rename,
    )


def _parse_sympy_equations(
    dxdt,
    registry,
    unknown_names,
    known_symbol_map,
    user_functions,
    user_function_derivatives,
    strict,
    state_names,
):
    """Normalise SymPy equation input into structural equations.

    Returns ``(equations, funcs, new_params, aux_names,
    inferred_states, rename)``. User-function calls appearing as
    :class:`~sympy.core.function.AppliedUndef` nodes are resolved
    against ``user_functions``: non-device callables are inlined and
    device callables are kept as symbolic calls.
    """

    if isinstance(dxdt, (list, tuple)):
        raw_equations = list(dxdt)
    else:
        raw_equations = [dxdt]

    user_functions = user_functions or {}
    funcs = {}

    canonical = {}
    for name, sym in known_symbol_map.items():
        canonical[sp.Symbol(name)] = sym
        canonical[sp.Symbol(name, real=True)] = sym
    for name in unknown_names:
        canonical[sp.Symbol(name)] = sp.Symbol(name, real=True)
    derivative_names = set()
    for name in state_names:
        dname = f"d{name}"
        if dname not in known_symbol_map:
            canonical[sp.Symbol(dname)] = sp.Symbol(dname, real=True)
            derivative_names.add(dname)

    def resolve_calls(expr: sp.Expr) -> sp.Expr:
        called = {
            node.func.__name__
            for node in expr.atoms(AppliedUndef)
        }
        called.discard("d")
        for name in called:
            if name in user_functions:
                funcs[name] = user_functions[name]
            elif name not in KNOWN_FUNCTIONS:
                raise ValueError(
                    f"Your equations contain a call to a function "
                    f"{name}() that isn't part of Sympy and wasn't "
                    f"provided in the user_functions dict."
                )
        if user_functions:
            expr = _inline_nondevice_calls(
                expr,
                user_functions,
                {name: name for name in user_functions},
            )
        return expr

    equations = []
    aux_names = []
    inferred_states = []

    def bind_lhs(lhs: sp.Expr) -> sp.Expr:
        """Apply the state-aware ``dX`` rule to a symbol LHS."""

        if not isinstance(lhs, sp.Symbol):
            return lhs
        if registry.is_derivative(lhs):
            return lhs
        name = lhs.name
        if not (name.startswith("d") and len(name) > 1):
            return lhs
        base = name[1:]
        if base in unknown_names:
            return registry.derivative(sp.Symbol(base, real=True))
        if (
            not state_names
            and not strict
            and _IDENTIFIER_PATTERN.match(base)
        ):
            unknown_names.add(base)
            inferred_states.append(base)
            return registry.derivative(sp.Symbol(base, real=True))
        return lhs

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
        lhs = bind_lhs(lhs)
        if isinstance(lhs, sp.Symbol) and lhs.name in known_symbol_map:
            raise ValueError(
                f"{lhs.name} is an immutable input (constant, "
                "parameter, or driver) but is being assigned. It "
                "must be a state, observable, or auxiliary."
            )
        lhs = resolve_calls(lhs)
        rhs = resolve_calls(rhs)
        equations.append(Equation(lhs, rhs))

    new_params = []
    declared = set(known_symbol_map.values()) | {TIME_SYMBOL}
    declared |= {
        sp.Symbol(name, real=True)
        for name in unknown_names | derivative_names
    }
    declared_names = (
        set(known_symbol_map)
        | unknown_names
        | derivative_names
        | {"t"}
    )
    for eq in equations:
        lhs = eq.lhs
        if (
            isinstance(lhs, sp.Symbol)
            and lhs.name not in declared_names
            and not registry.is_derivative(lhs)
        ):
            # An assignment defines the symbol, so even strict mode
            # admits anonymous auxiliaries.
            aux_names.append(lhs.name)
            unknown_names.add(lhs.name)
            declared_names.add(lhs.name)
    for eq in equations:
        for sym in eq.free_symbols():
            if (
                sym.name in declared_names
                or registry.is_derivative(sym)
            ):
                continue
            if strict:
                raise ValueError(
                    f"Equations reference undefined symbol {sym}."
                )
            new_params.append(sym)
            declared_names.add(sym.name)
    return equations, funcs, new_params, aux_names, inferred_states, {}


def normalise_input(
    dxdt,
    unknown_names: set,
    known_symbol_map: Dict[str, sp.Symbol],
    user_functions: Optional[Dict[str, Callable]],
    user_function_derivatives: Optional[Dict[str, Callable]],
    strict: bool,
    state_names: Iterable[str],
) -> NormalisedSystem:
    """Normalise string or SymPy equations into structural form.

    Parameters
    ----------
    dxdt
        Equation strings (newline-joined or a sequence) or SymPy
        equations (:class:`~sympy.Equality` or ``(lhs, rhs)``
        tuples).
    unknown_names
        Names of the declared unknowns (states and observables).
        Mutated: inferred auxiliaries and states are added.
    known_symbol_map
        Immutable inputs (parameters, constants, drivers) by name.
    user_functions, user_function_derivatives
        Callables referenced in the equations and their analytic
        derivative helpers.
    strict
        Reject undeclared symbols instead of inferring them.
    state_names
        Names of the declared states; controls derivative-name
        binding on right-hand sides and ``dX`` state inference.
    """

    state_names = set(state_names)
    reserved = (
        set(known_symbol_map)
        | set(unknown_names)
        | {"t"}
        | set(KNOWN_FUNCTIONS)
    )
    registry = DerivativeRegistry(reserved)

    is_string_input = isinstance(dxdt, str) or (
        isinstance(dxdt, (list, tuple))
        and bool(dxdt)
        and isinstance(dxdt[0], str)
    )
    if is_string_input:
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
        (
            equations,
            funcs,
            new_params,
            aux_names,
            inferred_states,
            rename,
        ) = _parse_string_equations(
            lines,
            raw_lines,
            registry,
            unknown_names,
            known_symbol_map,
            user_functions,
            user_function_derivatives,
            strict,
            state_names,
        )
    else:
        (
            equations,
            funcs,
            new_params,
            aux_names,
            inferred_states,
            rename,
        ) = _parse_sympy_equations(
            dxdt,
            registry,
            unknown_names,
            known_symbol_map,
            user_functions,
            user_function_derivatives,
            strict,
            state_names,
        )

    return NormalisedSystem(
        equations=equations,
        registry=registry,
        funcs=funcs,
        unknown_names=unknown_names,
        aux_names=aux_names,
        new_params=new_params,
        inferred_states=inferred_states,
        rename=rename,
    )


def classify_system(
    normalised: NormalisedSystem,
    state_names: Iterable[str],
    observable_names: Iterable[str],
) -> str:
    """Classify a normalised system as ``"explicit"`` or ``"dae"``.

    A system is explicit-shaped when it is already in solved form:
    every equation assigns either the first derivative of a declared
    state or a plain output symbol, no right-hand side contains a
    derivative, no left-hand side repeats, every declared state has
    exactly one derivative equation, and every declared observable
    is assigned. Anything else needs structural simplification.
    """

    registry = normalised.registry
    state_set = set(state_names)
    observable_set = set(observable_names)
    derived_states = set()
    assigned = set()

    for eq in normalised.equations:
        lhs = eq.lhs
        if registry.is_derivative(lhs):
            base, order = registry.base_and_order(lhs)
            if order != 1 or base.name not in state_set:
                return "dae"
            if base.name in derived_states:
                return "dae"
            derived_states.add(base.name)
        elif isinstance(lhs, sp.Symbol):
            if lhs.name in state_set:
                # Declared state assigned algebraically.
                return "dae"
            if lhs.name in assigned:
                return "dae"
            assigned.add(lhs.name)
        else:
            # Implicit equation (numeric or expression LHS).
            return "dae"
        for sym in eq.rhs.free_symbols:
            if registry.is_derivative(sym):
                return "dae"

    if derived_states != state_set:
        return "dae"
    if not observable_set <= assigned:
        return "dae"
    return "explicit"
