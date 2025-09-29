"""Parse symbolic ODE descriptions into structured SymPy objects."""

import re
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from warnings import warn

import sympy as sp
from sympy.parsing.sympy_parser import T, parse_expr
from sympy.core.function import AppliedUndef

from .indexedbasemaps import IndexedBases
from .sym_utils import hash_system_definition
from cubie._utils import is_devfunc

# Lambda notation, Auto-number, factorial notation, implicit multiplication
PARSE_TRANSORMS = (T[0][0], T[3][0], T[4][0], T[8][0])

TIME_SYMBOL = sp.Symbol("t", real=True)

KNOWN_FUNCTIONS = {
    # Basic mathematical functions
    'exp': sp.exp,
    'log': sp.log,
    'sqrt': sp.sqrt,
    'pow': sp.Pow,

    # Trigonometric functions
    'sin': sp.sin,
    'cos': sp.cos,
    'tan': sp.tan,
    'asin': sp.asin,
    'acos': sp.acos,
    'atan': sp.atan,
    'atan2': sp.atan2,

    # Hyperbolic functions
    'sinh': sp.sinh,
    'cosh': sp.cosh,
    'tanh': sp.tanh,
    'asinh': sp.asinh,
    'acosh': sp.acosh,
    'atanh': sp.atanh,

    # Special functions
    'erf': sp.erf,
    'erfc': sp.erfc,
    'gamma': sp.gamma,
    'lgamma': sp.loggamma,

    # Rounding and absolute
    'Abs': sp.Abs,
    'abs': sp.Abs,
    'floor': sp.floor,
    'ceil': sp.ceiling,
    'ceiling': sp.ceiling,

    # Min/Max
    'Min': sp.Min,
    'Max': sp.Max,
    'min': sp.Min,
    'max': sp.Max,

    # Functions that need custom handling - placeholder will not
    # work for differentiation.
    # 'log10': sp.Function('log10'),
    # 'log2': sp.Function('log2'),
    # 'log1p': sp.Function('log1p'),
    # 'hypot': sp.Function('hypot'),
    # 'expm1': sp.Function('expm1'),
    # 'copysign': sp.Function('copysign'),
    # 'fmod': sp.Function('fmod'),
    # 'modf': sp.Function('modf'),
    # 'frexp': sp.Function('frexp'),
    # 'ldexp': sp.Function('ldexp'),
    # 'remainder': sp.Function('remainder'),
    # 'fabs': sp.Abs,
    # 'isnan': sp.Function('isnan'),
    # 'isinf': sp.Function('isinf'),
    # 'isfinite': sp.Function('isfinite'),

    'Piecewise': sp.Piecewise,
    'sign': sp.sign,
}


class EquationWarning(Warning):
    """Warning raised for recoverable issues in equation definitions."""

_func_call_re = re.compile(r"\b([A-Za-z_]\w*)\s*\(")

# ---------------------------- Input cleaning ------------------------------- #
def _sanitise_input_math(expr_str: str) -> str:
    """Convert Python conditional syntax into SymPy-compatible constructs.

    Parameters
    ----------
    expr_str
        Expression string to sanitise before parsing.

    Returns
    -------
    str
        SymPy-compatible expression string.
    """
    expr_str = _replace_if(expr_str)
    return expr_str

def _replace_if(expr_str: str) -> str:
    """Recursively replace ternary conditionals with ``Piecewise`` blocks.

    Parameters
    ----------
    expr_str
        Expression string that may contain inline conditional expressions.

    Returns
    -------
    str
        Expression with ternary conditionals rewritten for SymPy parsing.
    """
    match = re.search(r"(.+?) if (.+?) else (.+)", expr_str)
    if match:
        true_str = _replace_if(match.group(1).strip())
        cond_str = _replace_if(match.group(2).strip())
        false_str = _replace_if(match.group(3).strip())
        return f"Piecewise(({true_str}, {cond_str}), ({false_str}, True))"
    return expr_str

# ---------------------------- Function handling --------------------------- #

def _rename_user_calls(
    lines: Iterable[str],
    user_functions: Optional[Dict[str, Callable]] = None,
) -> Tuple[List[str], Dict[str, str]]:
    """Rename user-defined callables to avoid collisions with SymPy names.

    Parameters
    ----------
    lines
        Raw equation strings to inspect for function calls.
    user_functions
        Mapping of user-defined names to callables referenced in the
        equations.

    Returns
    -------
    tuple
        Sanitised lines and a mapping from original names to suffixed names.
    """
    if not user_functions:
        return list(lines), {}
    rename = {name: f"{name}_" for name in user_functions.keys()}
    renamed_lines = []
    # Replace only function-call tokens: name( -> name_(
    for line in lines:
        new_line = line
        for name, underscored in rename.items():
            new_line = re.sub(rf"\b{name}\s*\(", f"{underscored}(", new_line)
        renamed_lines.append(new_line)
    return renamed_lines, rename


def _build_sympy_user_functions(
    user_functions: Optional[Dict[str, Callable]],
    rename: Dict[str, str],
    user_function_derivatives: Optional[Dict[str, Callable]] = None,
) -> Tuple[Dict[str, object], Dict[str, str], Dict[str, bool]]:
    """Create SymPy ``Function`` placeholders for user-defined callables.

    Parameters
    ----------
    user_functions
        Mapping of user-provided callable names to their implementations.
    rename
        Mapping from original user function names to temporary suffixed names
        used during parsing.
    user_function_derivatives
        Mapping from user function names to callables that evaluate analytic
        derivatives.

    Returns
    -------
    tuple
        Parsing locals, pretty-name aliases, and device-function flags.

    Notes
    -----
    Device functions or user functions with derivative helpers are wrapped in
    dynamic ``Function`` subclasses whose ``fdiff`` method yields symbolic
    derivative placeholders so that downstream printers can emit gradient
    kernels.
    """
    parse_locals = {}
    alias_map = {}
    is_device_map = {}

    for orig_name, func in (user_functions or {}).items():
        sym_name = rename.get(orig_name, orig_name)
        alias_map[sym_name] = orig_name
        dev = is_devfunc(func)
        is_device_map[sym_name] = dev
        # Resolve derivative print name (if provided)
        deriv_callable = None
        if user_function_derivatives and orig_name in user_function_derivatives:
            deriv_callable = user_function_derivatives[orig_name]
        deriv_print_name = None
        if deriv_callable is not None:
            try:
                deriv_print_name = deriv_callable.__name__
            except Exception:
                deriv_print_name = None
        should_wrap = dev or deriv_callable is not None
        if should_wrap:
            # Build a dynamic Function subclass with name sym_name and fdiff
            # that generates <deriv_print_name or d_orig>(args..., argindex-1)
            def _make_class(sym_name=sym_name, orig_name=orig_name, deriv_print_name=deriv_print_name):
                class _UserDevFunc(sp.Function):
                    nargs = None
                    @classmethod
                    def eval(cls, *args):
                        return None
                    def fdiff(self, argindex=1):
                        target_name = deriv_print_name or f"d_{orig_name}"
                        deriv_func = sp.Function(target_name)
                        return deriv_func(*self.args, sp.Integer(argindex - 1))
                _UserDevFunc.__name__ = sym_name
                return _UserDevFunc
            parse_locals[sym_name] = _make_class()
        else:
            parse_locals[sym_name] = sp.Function(sym_name)
    return parse_locals, alias_map, is_device_map


def _inline_nondevice_calls(
    expr: sp.Expr,
    user_functions: Dict[str, Callable],
    rename: Dict[str, str],
) -> sp.Expr:
    """Inline callable results for non-device user functions when possible.

    Parameters
    ----------
    expr
        Expression potentially containing calls to user-defined functions.
    user_functions
        Mapping from user-provided function names to their implementations.
    rename
        Mapping from original user function names to suffixed parser names.

    Returns
    -------
    sympy.Expr
        Expression with inlineable calls replaced by their evaluated result.
    """
    if not user_functions:
        return expr

    def _try_inline(applied):
        # applied is an AppliedUndef or similar; get its name
        name = applied.func.__name__
        # reverse-map if this is an underscored user function
        orig_name = None
        for k, v in rename.items():
            if v == name:
                orig_name = k
                break
        if orig_name is None:
            return applied
        fn = user_functions.get(orig_name)
        if fn is None or is_devfunc(fn):
            return applied
        try:
            # Try evaluate on SymPy args
            val = fn(*applied.args)
            # Ensure it's a SymPy expression
            if isinstance(val, (sp.Expr, sp.Symbol)):
                return val
            # Fall back to keeping symbolic call
            return applied
        except Exception:
            return applied

    # Replace any AppliedUndef whose name matches an underscored function
    for _, sym_name in rename.items():
        f = sp.Function(sym_name)
        expr = expr.replace(lambda e: isinstance(e, AppliedUndef) and e.func == f, _try_inline)
    return expr


def _process_calls(
    equations_input: Iterable[str],
    user_functions: Optional[Dict[str, Callable]] = None,
) -> Dict[str, Callable]:
    """Resolve callable names referenced in the user equations.

    Parameters
    ----------
    equations_input
        Equations describing the system dynamics.
    user_functions
        Mapping from user-provided function names to callables.

    Returns
    -------
    dict
        Resolved callables keyed by their names as they appear in equations.
    """
    calls = set()
    if user_functions is None:
        user_functions = {}
    for line in equations_input:
        calls |= set(_func_call_re.findall(line))
    funcs = {}
    for name in calls:
        if name in user_functions:
            funcs[name] = user_functions[name]
        elif name in KNOWN_FUNCTIONS:
            funcs[name] = KNOWN_FUNCTIONS[name]
        else:
            raise ValueError(f"Your dxdt code contains a call to a "
                             f"function {name}() that isn't part of Sympy "
                             f"and wasn't provided in the user_functions "
                             f"dict.")
    # Tests: non-listed sympy function errors
    # Tests: user function passes
    # Tests: user function overrides listed sympy function
    return funcs

def _process_parameters(
    states: Union[Dict[str, float], Iterable[str]],
    parameters: Union[Dict[str, float], Iterable[str]],
    constants: Union[Dict[str, float], Iterable[str]],
    observables: Iterable[str],
    drivers: Iterable[str],
) -> IndexedBases:
    """Convert user-specified symbols into ``IndexedBases`` structures.

    Parameters
    ----------
    states
        State symbols or mapping to initial values.
    parameters
        Parameter symbols or mapping to default values.
    constants
        Constant symbols or mapping to default values.
    observables
        Observable symbol names supplied by the user.
    drivers
        External driver symbol names.

    Returns
    -------
    IndexedBases
        Structured representation of all indexed symbol collections.
    """
    indexed_bases = IndexedBases.from_user_inputs(states,
                                                  parameters,
                                                  constants,
                                                  observables,
                                                  drivers)
    return indexed_bases


def _lhs_pass(
    lines: Sequence[str],
    indexed_bases: IndexedBases,
    strict: bool = True,
) -> Dict[str, sp.Symbol]:
    """Validate left-hand sides and infer anonymous auxiliaries.

    Parameters
    ----------
    lines
        Equations supplied by the user.
    indexed_bases
        Indexed symbol collections constructed from user inputs.
    strict
        When ``False``, unknown state derivatives are inferred automatically
        but other assignments remain anonymous auxiliaries.

    Returns
    -------
    dict
        Symbols for auxiliary observables introduced implicitly in equations.

    Notes
    -----
    Anonymous auxiliaries ease model authoring but are not persisted as
    saved observables; tracking them ensures generated SymPy code remains
    consistent with the equations.
    """
    anonymous_auxiliaries = {}
    assigned_obs = set()
    underived_states = set(indexed_bases.dxdt_names)
    state_names = set(indexed_bases.state_names)
    observable_names = set(indexed_bases.observable_names)
    param_names = set(indexed_bases.parameter_names)
    constant_names = set(indexed_bases.constant_names)
    driver_names = set(indexed_bases.driver_names)
    states = indexed_bases.states
    observables = indexed_bases.observables
    dxdt = indexed_bases.dxdt

    for line in lines:
        lhs, rhs = [p.strip() for p in line.split("=", 1)]
        if lhs.startswith("d"):
            state_name = lhs[1:]
            s_sym = sp.Symbol(state_name, real=True)
            if state_name not in state_names:
                if state_name in observable_names:
                    warn(
                        f"Your equation included d{state_name}, but "
                        f"{state_name} was listed as an observable. It has"
                        "been converted into a state.",
                        EquationWarning,
                    )
                    states.push(s_sym)
                    dxdt.push(sp.Symbol(f"d{state_name}", real=True))
                    observables.pop(s_sym)
                    state_names.add(state_name)
                    observable_names.discard(state_name)
                else:
                    if strict:
                        raise ValueError(
                            f"Unknown state derivative: {lhs}. "
                            f"No state or observable called {state_name} found."
                        )
                    else:
                        states.push(s_sym)
                        dxdt.push(sp.Symbol(f"d{state_name}", real=True))
                        state_names.add(state_name)
                        underived_states.add(f"d{state_name}")
            underived_states -= {lhs}

        elif lhs in state_names:
            raise ValueError(
                f"State {lhs} cannot be assigned directly. All "
                f"states must be defined as derivatives with d"
                f"{lhs} = [...]"
            )

        elif (
            lhs in param_names
            or lhs in constant_names
            or lhs in driver_names
        ):
            raise ValueError(
                f"{lhs} was entered as an immutable "
                f"input (constant, parameter, or driver)"
                ", but it is being assigned to. Cubie "
                "can't handle this - if it's being "
                "assigned to, it must be either a state, an "
                "observable, or undefined."
            )

        else:
            if lhs not in observable_names:
                warn(
                    f"The intermediate variable {lhs} was assigned to "
                    f"but not listed as an observable. It will be treated "
                    f"as an anonymous auxiliary and its trajectory will "
                    f"not be saved.",
                    EquationWarning,
                )
                anonymous_auxiliaries[lhs] = sp.Symbol(lhs, real=True)
            if lhs in observable_names:
                assigned_obs.add(lhs)

    missing_obs = set(indexed_bases.observable_names) - assigned_obs
    if missing_obs:
        raise ValueError(f"Observables {missing_obs} are never assigned "
                         f"to.")

    if underived_states:
        warn(
            f"States {underived_states} have no associated derivative "
            f"term. In the Cubie world, this makes it an 'observable'. "
            f"{underived_states} have been moved from states to observables.",
            EquationWarning,
        )
        for state in underived_states:
            s_sym = sp.Symbol(state, real=True)
            if state in observables:
                raise ValueError(
                    f"State {state} is already both observable and state. "
                    f"It needs to be an observable if it has no derivative"
                    f"term."
                )
            observables.push(s_sym)
            states.pop(s_sym)
            dxdt.pop(s_sym)
            observable_names.add(state)

    return anonymous_auxiliaries

def _rhs_pass(
    lines: Iterable[str],
    all_symbols: Dict[str, sp.Symbol],
    user_funcs: Optional[Dict[str, Callable]] = None,
    user_function_derivatives: Optional[Dict[str, Callable]] = None,
    strict: bool = True,
) -> Tuple[List[Tuple[sp.Symbol, sp.Expr]], Dict[str, Callable], List[sp.Symbol]]:
    """Parse right-hand sides, validating symbols and callable usage.

    Parameters
    ----------
    lines
        Equations supplied by the user.
    all_symbols
        Mapping from symbol names to SymPy symbols.
    user_funcs
        Optional mapping of user-provided callables referenced in equations.
    user_function_derivatives
        Optional mapping of user-provided derivative helpers.
    strict
        When ``False``, unknown symbols are inferred from expressions.

    Returns
    -------
    tuple
        Parsed expressions, callable mapping, and any inferred symbols.
    """
    expressions = []
    # Detect all calls as before for erroring on unknown names and for returning funcs
    funcs = _process_calls(lines, user_funcs)

    # Prepare user function environment with underscore renaming to avoid collisions
    sanitized_lines, rename = _rename_user_calls(lines, user_funcs or {})
    parse_locals, alias_map, dev_map = _build_sympy_user_functions(
        user_funcs or {}, rename, user_function_derivatives
    )

    # Expose mapping for the printer via special key in all_symbols (copied by caller)
    local_dict = all_symbols.copy()
    local_dict.update(parse_locals)
    local_dict.setdefault("t", TIME_SYMBOL)
    new_symbols = []
    for raw_line, line in zip(lines, sanitized_lines):
        lhs, rhs = [p.strip() for p in line.split("=", 1)]
        rhs_expr = _sanitise_input_math(rhs)
        if strict:
            # don't auto-add symbols
            try:
                rhs_expr = parse_expr(
                        rhs_expr,
                        transformations=PARSE_TRANSORMS,
                        local_dict=local_dict)
            except (NameError, TypeError) as e:
                # Provide the original (unsanitized) line in message
                raise ValueError(f"Undefined symbols in equation '{raw_line}'") from e
        else:
            rhs_expr = parse_expr(
                rhs_expr,
                local_dict=local_dict,
            )
            new_inputs = [
                sym for sym in rhs_expr.free_symbols if sym not in local_dict.values()
            ]
            for sym in new_inputs:
                new_symbols.append(sym)

        # Attempt to inline non-device functions that can accept SymPy args
        rhs_expr = _inline_nondevice_calls(rhs_expr, user_funcs or {}, rename)

        expressions.append([local_dict.get(lhs, all_symbols[lhs] if lhs in all_symbols else sp.Symbol(lhs, real=True)), rhs_expr])

    # Return expressions along with funcs mapping (original names)
    return expressions, funcs, new_symbols

def parse_input(
    dxdt: Union[str, Iterable[str]],
    states: Optional[Union[Dict[str, float], Iterable[str]]] = None,
    observables: Optional[Iterable[str]] = None,
    parameters: Optional[Union[Dict[str, float], Iterable[str]]] = None,
    constants: Optional[Union[Dict[str, float], Iterable[str]]] = None,
    drivers: Optional[Iterable[str]] = None,
    user_functions: Optional[Dict[str, Callable]] = None,
    user_function_derivatives: Optional[Dict[str, Callable]] = None,
    strict: bool = False,
) -> Tuple[
    IndexedBases,
    Dict[str, object],
    Dict[str, Callable],
    List[Tuple[sp.Symbol, sp.Expr]],
    str,
]:
    """Process user equations and symbol metadata into structured components.

    Parameters
    ----------
    dxdt
        System equations, either as a newline-delimited string or iterable of
        strings.
    states
        State variables provided as names or a mapping to initial values.
    observables
        Observable variable names whose trajectories should be saved.
    parameters
        Parameter names or mapping to default values.
    constants
        Constant names or mapping to default values that remain fixed across
        runs.
    drivers
        Driver variable names supplied at runtime.
    user_functions
        Mapping of callable names used in equations to their implementations.
    user_function_derivatives
        Mapping of callable names to derivative helper functions.
    strict
        When ``False``, infer missing symbol declarations from equation usage.

    Returns
    -------
    tuple
        Indexed bases, combined symbol mapping, callable mapping, parsed
        equations, and the system hash.

    Notes
    -----
    When ``strict`` is ``False``, undeclared variables inferred from equation
    usage are added automatically, except for anonymous auxiliaries that are
    retained for intermediate computation but not persisted as observables.
    """
    if states is None:
        states = {}
        if strict:
            raise ValueError(
                "No state symbols were provided - if you want to build a model "
                "from a set of equations alone, set strict=False"
            )
    if observables is None:
        observables = []
    if parameters is None:
        parameters = {}
    if constants is None:
        constants = {}
    if drivers is None:
        drivers = []

    index_map = _process_parameters(
        states=states,
        parameters=parameters,
        constants=constants,
        observables=observables,
        drivers=drivers,
    )

    if isinstance(dxdt, str):
        lines = [
            line.strip() for line in dxdt.strip().splitlines() if line.strip()
        ]
    elif isinstance(dxdt, list) or isinstance(dxdt, tuple):
        lines = [line.strip() for line in dxdt if line.strip()]
    else:
        raise ValueError("dxdt must be a string or a list/tuple of strings")

    constants = index_map.constants.default_values
    fn_hash = hash_system_definition(dxdt, constants)
    anon_aux = _lhs_pass(lines, index_map, strict=strict)
    all_symbols = index_map.all_symbols.copy()
    all_symbols.setdefault("t", TIME_SYMBOL)
    all_symbols.update(anon_aux)

    equation_map, funcs, new_params = _rhs_pass(
        lines=lines,
        all_symbols=all_symbols,
        user_funcs=user_functions,
        user_function_derivatives=user_function_derivatives,
        strict=strict,
    )

    # Expose user functions in the returned symbols dict (original names)
    # and alias mapping for the printer under a special key
    if user_functions:
        all_symbols.update({name: fn for name, fn in user_functions.items()})
        # Also expose derivative callables if provided
        if user_function_derivatives:
            all_symbols.update(
                {
                    fn.__name__: fn
                    for fn in user_function_derivatives.values()
                    if callable(fn)
                }
            )
        # Build alias map underscored -> original for the printer
        _, rename = _rename_user_calls(lines, user_functions or {})
        if rename:
            alias_map = {v: k for k, v in rename.items()}
            all_symbols['__function_aliases__'] = alias_map

    for param in new_params:
        index_map.parameters.push(param)

    return index_map, all_symbols, funcs, equation_map, fn_hash
