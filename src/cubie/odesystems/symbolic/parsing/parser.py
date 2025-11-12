"""Parse symbolic ODE descriptions into structured SymPy objects."""

import re
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from warnings import warn

import sympy as sp
from sympy.parsing.sympy_parser import T, parse_expr
from sympy.core.function import AppliedUndef

from ..indexedbasemaps import IndexedBases
from ..sym_utils import hash_system_definition
from cubie._utils import is_devfunc

# Lambda notation, Auto-number, factorial notation, implicit multiplication
PARSE_TRANSORMS = (T[0][0], T[3][0], T[4][0], T[8][0])

_INDEXED_NAME_PATTERN = re.compile(r"(?P<name>[A-Za-z_]\w*)\[(?P<index>\d+)\]")

TIME_SYMBOL = sp.Symbol("t", real=True)
DRIVER_SETTING_KEYS = {"time", "dt", "wrap", "order"}

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


@dataclass(frozen=True)
class ParsedEquations:
    """Container separating state, observable, and auxiliary assignments.

    Parameters
    ----------
    ordered
        Equations in evaluation order exactly as supplied by the parser.
    state_derivatives
        Equations whose left-hand side corresponds to ``dx/dt`` outputs.
    observables
        Equations assigning user-requested observable symbols.
    auxiliaries
        Anonymous helper assignments required by either ``dx/dt`` or the
        observables.
    state_symbols
        Symbols that identify the derivative outputs.
    observable_symbols
        Symbols designating observables.
    auxiliary_symbols
        Symbols introduced for intermediate calculations.
    """

    ordered: Tuple[Tuple[sp.Symbol, sp.Expr], ...]
    state_derivatives: Tuple[Tuple[sp.Symbol, sp.Expr], ...]
    observables: Tuple[Tuple[sp.Symbol, sp.Expr], ...]
    auxiliaries: Tuple[Tuple[sp.Symbol, sp.Expr], ...]
    _state_symbols: frozenset[sp.Symbol] = field(repr=False)
    _observable_symbols: frozenset[sp.Symbol] = field(repr=False)
    _auxiliary_symbols: frozenset[sp.Symbol] = field(repr=False)

    def __iter__(self) -> Iterable[Tuple[sp.Symbol, sp.Expr]]:
        """Iterate over all equations in the original evaluation order."""

        return iter(self.ordered)

    def __len__(self) -> int:
        """Return the number of stored equations."""

        return len(self.ordered)

    def __getitem__(self, index: int) -> Tuple[sp.Symbol, sp.Expr]:
        """Return the equation at ``index`` from the original ordering."""

        return self.ordered[index]

    def copy(self) -> Dict[sp.Symbol, sp.Expr]:
        """Return a mapping copy compatible with ``topological_sort``."""

        return {lhs: rhs for lhs, rhs in self.ordered}

    def to_equation_list(self) -> list[Tuple[sp.Symbol, sp.Expr]]:
        """Return the stored equations as a mutable list."""

        return list(self.ordered)

    @property
    def state_symbols(self) -> frozenset[sp.Symbol]:
        """Symbols representing derivative outputs."""

        return self._state_symbols

    @property
    def observable_symbols(self) -> frozenset[sp.Symbol]:
        """Symbols representing observable outputs."""

        return self._observable_symbols

    @property
    def auxiliary_symbols(self) -> frozenset[sp.Symbol]:
        """Symbols representing auxiliary assignments."""

        return self._auxiliary_symbols

    def non_observable_equations(self) -> list[Tuple[sp.Symbol, sp.Expr]]:
        """Return equations whose outputs are not observables."""

        observable_syms = self.observable_symbols
        return [eq for eq in self.ordered if eq[0] not in observable_syms]

    @property
    def dxdt_equations(self) -> Tuple[Tuple[sp.Symbol, sp.Expr], ...]:
        """Return equations required to evaluate ``dx/dt`` outputs."""

        return tuple(self.non_observable_equations())

    @property
    def observable_system(self) -> Tuple[Tuple[sp.Symbol, sp.Expr], ...]:
        """Return equations contributing to observable evaluation."""

        return self.ordered

    @classmethod
    def from_equations(
        cls,
        equations: Iterable[Tuple[sp.Symbol, sp.Expr]],
        index_map: "IndexedBases",
    ) -> "ParsedEquations":
        """Partition equations according to their assigned symbols."""

        if isinstance(equations, dict):
            items = list(equations.items())
        else:
            items = list(equations)
        ordered = tuple((lhs, rhs) for lhs, rhs in items)
        state_symbols = frozenset(index_map.dxdt.ref_map.keys())
        observable_symbols = frozenset(index_map.observables.ref_map.keys())
        state_eqs = tuple(eq for eq in ordered if eq[0] in state_symbols)
        observable_eqs = tuple(
            eq for eq in ordered if eq[0] in observable_symbols
        )
        auxiliary_eqs = tuple(
            eq
            for eq in ordered
            if eq[0] not in state_symbols and eq[0] not in observable_symbols
        )
        auxiliary_symbols = frozenset(eq[0] for eq in auxiliary_eqs)
        return cls(
            ordered=ordered,
            state_derivatives=state_eqs,
            observables=observable_eqs,
            auxiliaries=auxiliary_eqs,
            _state_symbols=state_symbols,
            _observable_symbols=observable_symbols,
            _auxiliary_symbols=auxiliary_symbols,
        )


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


def _normalise_indexed_tokens(lines: Iterable[str]) -> list[str]:
    """Collapse numeric index access into scalar-style symbol names.

    Parameters
    ----------
    lines
        Raw equation strings supplied by the user.

    Returns
    -------
    list[str]
        Lines with occurrences of ``name[index]`` rewritten as ``nameindex``
        whenever ``index`` is an integer literal.
    """

    def _replace(match: re.Match[str]) -> str:
        base = match.group("name")
        index = match.group("index")
        return f"{base}{index}"

    return [_INDEXED_NAME_PATTERN.sub(_replace, line) for line in lines]

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
    raw_lines: Optional[Sequence[str]] = None,
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
    raw_lines
        Optional representation of the original equations prior to indexed
        token normalisation. When provided, error messages reference these
        inputs.

    Returns
    -------
    tuple
        Parsed expressions, callable mapping, and any inferred symbols.
    """
    lines = list(lines)
    expressions = []
    # Detect all calls as before for erroring on unknown names and for returning funcs
    funcs = _process_calls(lines, user_funcs)

    # Prepare user function environment with underscore renaming to avoid collisions
    sanitized_lines, rename = _rename_user_calls(lines, user_funcs or {})
    if raw_lines is None:
        raw_iter: Sequence[str] = lines
    else:
        raw_iter = list(raw_lines)
    parse_locals, alias_map, dev_map = _build_sympy_user_functions(
        user_funcs or {}, rename, user_function_derivatives
    )

    # Expose mapping for the printer via special key in all_symbols (copied by caller)
    local_dict = all_symbols.copy()
    local_dict.update(parse_locals)
    local_dict.setdefault("t", TIME_SYMBOL)
    new_symbols = []
    for raw_line, line in zip(raw_iter, sanitized_lines):
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

        expressions.append(
            [
                local_dict.get(
                    lhs,
                    all_symbols[lhs]
                    if lhs in all_symbols
                    else sp.Symbol(lhs, real=True),
                ),
                rhs_expr,
            ]
        )

    declared_symbols = {
        value for value in all_symbols.values() if isinstance(value, sp.Symbol)
    }
    new_symbol_set = set(new_symbols)
    rhs_symbols = {
        symbol for _, expression in expressions for symbol in expression.free_symbols
    }
    unresolved_symbols = sorted(
        str(symbol)
        for symbol in rhs_symbols
        if symbol not in declared_symbols and symbol not in new_symbol_set
    )
    if unresolved_symbols:
        raise ValueError(
            "Equations reference undefined symbols: "
            f"{unresolved_symbols}."
        )

    return expressions, funcs, new_symbols

def parse_input(
    dxdt: Union[str, Iterable[str]],
    states: Optional[Union[Dict[str, float], Iterable[str]]] = None,
    observables: Optional[Iterable[str]] = None,
    parameters: Optional[Union[Dict[str, float], Iterable[str]]] = None,
    constants: Optional[Union[Dict[str, float], Iterable[str]]] = None,
    drivers: Optional[Union[Iterable[str], Dict[str, Any]]] = None,
    user_functions: Optional[Dict[str, Callable]] = None,
    user_function_derivatives: Optional[Dict[str, Callable]] = None,
    strict: bool = False,
) -> Tuple[
    IndexedBases,
    Dict[str, object],
    Dict[str, Callable],
    ParsedEquations,
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
        Driver variable names supplied at runtime. Accepts either an iterable
        of driver labels or a dictionary mapping driver labels to default
        values and, when using driver arrays, configuration entries such as
        ``time``, ``dt``, ``wrap``, and ``order``.
    user_functions
        Mapping of callable names used in equations to their implementations.
    user_function_derivatives
        Mapping of callable names to derivative helper functions.
    strict
        When ``False``, infer missing symbol declarations from equation usage.

    Returns
    -------
    tuple
        Indexed bases, combined symbol mapping, callable mapping, partitioned
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
    driver_dict = None
    if drivers is None:
        drivers = []
    elif isinstance(drivers, dict):
        driver_dict = drivers
        drivers = [
            key for key in drivers.keys() if key not in DRIVER_SETTING_KEYS
        ]
        if len(drivers) == 0:
            raise ValueError(
                "Driver dictionary must include at least one driver symbol."
            )

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

    raw_lines = list(lines)
    lines = _normalise_indexed_tokens(lines)

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
        raw_lines=raw_lines,
    )

    for param in new_params:
        index_map.parameters.push(param)
        all_symbols[str(param)] = param

    if driver_dict is not None:
        index_map.drivers.set_passthrough_defaults(driver_dict)

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

    parsed_equations = ParsedEquations.from_equations(equation_map, index_map)

    return index_map, all_symbols, funcs, parsed_equations, fn_hash
