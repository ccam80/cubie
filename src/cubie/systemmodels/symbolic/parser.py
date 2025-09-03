"""Parsing helpers for symbolic ODE definitions."""

import re
from typing import Dict, Iterable, Optional, Tuple, Union
from warnings import warn

import sympy as sp
from sympy.parsing.sympy_parser import T, parse_expr
from sympy.core.function import AppliedUndef

from .indexedbasemaps import IndexedBases
from .sym_utils import hash_system_definition
from cubie._utils import is_devfunc

# Lambda notation, Auto-number, factorial notation, implicit multiplication
PARSE_TRANSORMS = (T[0][0], T[3][0], T[4][0], T[8][0])

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
    pass

_func_call_re = re.compile(r"\b([A-Za-z_]\w*)\s*\(")

# ---------------------------- Input cleaning ------------------------------- #
def _sanitise_input_math(expr_str: str):
    """Replace constructs that are logical in python but not in Sympy."""
    expr_str = _replace_if(expr_str)
    return expr_str

def _replace_if(expr_str: str):
    match = re.search(r"(.+?) if (.+?) else (.+)", expr_str)
    if match:
        true_str = _replace_if(match.group(1).strip())
        cond_str = _replace_if(match.group(2).strip())
        false_str = _replace_if(match.group(3).strip())
        return f"Piecewise(({true_str}, {cond_str}), ({false_str}, True))"
    return expr_str

# ---------------------------- Function handling --------------------------- #

def _rename_user_calls(lines: Iterable[str], user_functions: Dict[str, callable]):
    """Return new lines with user function names suffixed with '_' for parsing.

    Also returns a mapping of original->underscored names for later use.
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


def _build_sympy_user_functions(user_functions: Dict[str, callable], rename: Dict[str, str], user_function_derivatives: Optional[Dict[str, callable]] = None):
    """Create SymPy Function placeholders (or subclasses) for user functions.

    For device functions, create a dynamic SymPy Function subclass with fdiff
    returning d_<name>(args..., argindex).

    Returns
    -------
    parse_locals: Dict[str, Any]
        Names (underscored) to SymPy Function objects/classes for parse_expr.
    alias_map: Dict[str, str]
        Underscored name -> original printable name for the code printer.
    is_device_map: Dict[str, bool]
        Underscored name -> whether it was a device function.
    """
    parse_locals: Dict[str, object] = {}
    alias_map: Dict[str, str] = {}
    is_device_map: Dict[str, bool] = {}

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
        if dev:
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


def _inline_nondevice_calls(expr: sp.Expr,
                            user_functions: Dict[str, callable],
                            rename: Dict[str, str]):
    """Attempt to inline non-device user function calls if they can accept SymPy args.

    This replaces f_(args) with user_functions['f'](*args) when evaluation succeeds.
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


def _process_calls(equations_input: Iterable[str],
                   user_functions: Optional[Dict[str, callable]] = None):
    """ map known SymPy callables (e.g., 'exp') to Sympy functions """
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

def _process_parameters(states,
                        parameters,
                        constants,
                        observables,
                        drivers):
    """Process parameters and constants into indexed bases."""
    indexed_bases = IndexedBases.from_user_inputs(states,
                                                  parameters,
                                                  constants,
                                                  observables,
                                                  drivers)
    return indexed_bases


def _lhs_pass(
    lines,
    indexed_bases: IndexedBases,
    strict=True
    ) -> dict[str, sp.Symbol]:
    """ Process the left-hand-sides of all equations.

    Parameters
    ----------
    lines: list of str
        User-supplied list of equations that make up the dxdt function
    indexed_bases: IndexedBases
        The collection of maps from labels to indexed bases for the system
        generated by '_process_parameters'.
    strict: True
        If False, unrecognised symbols are added automatically to states,
        parameters, and observables as inferred from the equations.

    Returns
    -------
    Anonymous Auxiliaries: dict
        Auxiliary(observable) variables that aren't defined in the
        observables dictionary.

    Notes
    -----
    It is assumed that anonymous auxiliaries were included to make
    model-writing easier, and they won't be saved, but we need to keep
    track of the symbols for the Sympy math used in code generation.
    """
    anonymous_auxiliaries = {}
    assigned_obs = set()
    underived_states = set(indexed_bases.dxdt_names)
    state_names = indexed_bases.state_names
    observable_names = indexed_bases.observable_names
    param_names = indexed_bases.parameter_names
    constant_names = indexed_bases.constant_names
    driver_names = indexed_bases.driver_names
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
                else:
                    if strict:
                        raise ValueError(
                            f"Unknown state derivative: {lhs}. "
                            f"No state or observable called {state_name} found."
                        )
                    else:
                        states.push(s_sym)
                        dxdt.push(sp.Symbol(f"d{state_name}", real=True))
            underived_states -= {lhs}

        elif lhs in indexed_bases.state_names:
            raise ValueError(
                f"State {lhs} cannot be assigned directly. All "
                f"states must be defined as derivatives with d"
                f"{lhs} = [...]"
            )

        elif lhs in param_names or lhs in constant_names or lhs in driver_names:
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
                if strict:
                    warn(
                        f"The intermediate variable {lhs} was assigned to "
                        f"but not listed as an observable. It's trajectory will "
                        f"not be saved.",
                        EquationWarning,
                    )
                    anonymous_auxiliaries[lhs] = sp.Symbol(lhs, real=True)
                else:
                    observables.push(sp.Symbol(lhs, real=True))
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

    return anonymous_auxiliaries

def _rhs_pass(lines: Iterable[str],
              all_symbols: Dict[str, sp.Symbol],
              user_funcs: Optional[Dict[str, callable]] = None,
              user_function_derivatives: Optional[Dict[str, callable]] = None,
              strict=True):
    """Process expressions, checking symbols and finding callables.

    Parameters
    ----------
    lines: list of str
        User-supplied list of equations that make up the dxdt function
    all_symbols: dict
        All symbols defined in the model, including anonymous auxiliaries.
    strict: True
        If False, unrecognised symbols are added automatically to states,
        parameters, and observables as inferred from the equations.

    Returns
    -------
    tuple of tuples of (sp.Symbol, sp.Expr), dict
    tuple of (lhs, rhs) expressions, dict of callable functions

    """
    expressions = []
    # Detect all calls as before for erroring on unknown names and for returning funcs
    funcs = _process_calls(lines, user_funcs)

    # Prepare user function environment with underscore renaming to avoid collisions
    sanitized_lines, rename = _rename_user_calls(lines, user_funcs or {})
    parse_locals, alias_map, dev_map = _build_sympy_user_functions(user_funcs or {}, rename, user_function_derivatives)

    # Expose mapping for the printer via special key in all_symbols (copied by caller)
    local_dict = all_symbols.copy()
    local_dict.update(parse_locals)
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
            new_inputs = [sym for sym in rhs_expr.free_symbols if sym
            not in local_dict.values()]
            for sym in new_inputs:
                new_symbols.append(sym)

        # Attempt to inline non-device functions that can accept SymPy args
        rhs_expr = _inline_nondevice_calls(rhs_expr, user_funcs or {}, rename)

        expressions.append([local_dict.get(lhs, all_symbols[lhs] if lhs in all_symbols else sp.Symbol(lhs, real=True)), rhs_expr])

    # Return expressions along with funcs mapping (original names)
    return expressions, funcs, new_symbols

def parse_input(
        dxdt = Union[str, Iterable[str]],
        states: Optional[Union[Dict, Iterable[str]]] = None,
        observables: Optional[Iterable[str]] = None,
        parameters: Optional[Union[Dict, Iterable[str]]] = None,
        constants: Optional[Union[Dict, Iterable[str]]] = None,
        drivers: Optional[Iterable[str]] = None,
        user_functions: Optional[Dict[str, callable]] = None,
        user_function_derivatives: Optional[Dict[str, callable]] = None,
        strict=False
) -> Tuple[IndexedBases,Dict[str,sp.Symbol],Dict[str,callable],list, str]:
    """Process user input in the form of equations and symbols.

    When strict is False, this function can accept a set of equations and
    infer which variables are states and observables variables from the lhs of
    equations, then assign all other variables to "parameters". When strict
    is false, this function will check that all variables are in the correct
    category based on use, and will throw an error if it can't reconcile them.
    The only exception to this is auxiliary variables - an intermediate result
    that is calculated from inputs and used in outputs but not otherwise
    saved.

    Parameters
    ----------
    dxdt: str or iterable of str
        The equations that make up the system, either as a single string with
        newlines separating equations, or as an iterable of strings.
        Each equation must be of the form "lhs = rhs", where lhs is either
        "d<state>" for state derivatives, or the name of an observable or
        auxiliary variable.
    states: dict or iterable of str, optional
        The state variables of the system, either as a dictionary mapping
        variable names to default initial values, or as an iterable of
        variable names. If an iterable, all initial values will be set to
        0.0 by default.
    observables: iterable of str, optional
        Auxiliary variables (assigned to, but never given in terms of a
        derivative) which you might want to save or examine during the
        simulation.
    parameters: dict or iterable of str, optional
        Input parameters of the system, either as a dictionary mapping labels
        to default values, or as an iterable of variable names. If an
        iterable, all parameters will be set to 0.0 by default.
    constants: dict or iterable of str, optional
        Like parameters, but these are values you do not want to modify
        between batches. The more of your parameters are constant,
        the faster the simulation will run.
    drivers: iterable of str, optional
        Terms that represent "driving" or "forcing" variables in your equation,
        which will be expected as an input at runtime.
    user_functions: dict of str, callable, optional
        If you call a custom function inside your dxdt equations, include it in
        this dictionary. The dictionary should key the string you use to
        call the function in the equations to a callable of the function
        itself.

    Returns
    -------
    tuple of IndexedBases, dict of str, sp.Symbol, dict of str, callable,
    dict of sp.Symbol, sp.Expr, int
        - IndexedBases - the system-building indexed bases object,
        which contains all of your symbols and their values and references.
        - All symbols - a dictionary mapping all variable names to their
        Sympy Symbol instances.
        - funcs - a dictionary mapping all called functions to their callables
        - fn_hash - the unique hash which describes your equations and
        constants.

    """
    if states is None:
        states = {}
        if strict:
            raise ValueError("No state symbols were provided - if you want to"
            "build a model from a set of equations alone, set strict=False")
    if observables is None:
        observables = []
    if parameters is None:
        parameters = {}
    if constants is None:
        constants = {}
    if drivers is None:
        drivers = []

    """Parse a symbolic input of equations and string symbols."""
    index_map = _process_parameters(states=states,
                                    parameters=parameters,
                                    constants=constants,
                                    observables=observables,
                                    drivers=drivers)

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
    all_symbols.update(anon_aux)

    equation_map, funcs, new_params = _rhs_pass(lines=lines,
                                      all_symbols=all_symbols,
                                      user_funcs=user_functions,
                                      user_function_derivatives=user_function_derivatives,
                                      strict=strict)

    # Expose user functions in the returned symbols dict (original names)
    # and alias mapping for the printer under a special key
    if user_functions:
        all_symbols.update({name: fn for name, fn in user_functions.items()})
        # Also expose derivative callables if provided
        if user_function_derivatives:
            all_symbols.update({fn.__name__: fn for fn in user_function_derivatives.values() if callable(fn)})
        # Build alias map underscored -> original for the printer
        _, rename = _rename_user_calls(lines, user_functions or {})
        if rename:
            alias_map = {v: k for k, v in rename.items()}
            all_symbols['__function_aliases__'] = alias_map

    for param in new_params:
        index_map.parameters.push(param)

    return index_map, all_symbols, funcs, equation_map, fn_hash
