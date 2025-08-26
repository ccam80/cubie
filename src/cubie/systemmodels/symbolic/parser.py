"""Parsing helpers for symbolic ODE definitions."""

import re
from warnings import warn
from typing import Iterable, Dict, Optional, Union
import sympy as sp
from numpy.typing import ArrayLike, NDArray
from base64 import b64encode



class EquationWarning(Warning):
    pass

_func_call_re = re.compile(r"\b([A-Za-z_]\w*)\s*\(")

class IndexedBaseMap:
    def __init__(self,
                 base_name: str,
                 input_symbols: Iterable[str],
                 input_defaults: Optional[Union[ArrayLike, NDArray]] = None,
                 length=0,
                 real = True):

        if length == 0:
            length = len(input_symbols)

        self.length = length
        self.base_name = base_name
        self.real = real
        self.base = sp.IndexedBase(base_name, shape=(length,), real=real)
        self.index_map = {name: index
                        for index, name in enumerate(input_symbols)}
        self.ref_map = {name: self.base[index]
                        for index, name in enumerate(input_symbols)}
        if input_defaults is None:
            input_defaults = [0.0] * length
        elif len(input_defaults) != length:
            raise ValueError("Input defaults must be the same length as the "
                             "list of symbols")
        self.default_values = dict(zip(self.ref_map.keys(),
                                       input_defaults))


    def pop(self, name):
        """Remove a symbol from this object"""
        self.ref_map.pop(name)
        self.index_map.pop(name)
        self.base = sp.IndexedBase(self.base_name,
                                   shape=(len(self.ref_map),),
                                   real=self.real)
        self.length = len(self.ref_map)

    def push(self, name):
        """Adds a symbol to this object"""
        index = self.length
        self.base = sp.IndexedBase(self.base_name,
                                   shape=(index + 1,),
                                   real=self.real)
        self.ref_map[name] = self.base[index]
        self.index_map[name] = index


class IndexedBases:
    def __init__(self,
                 states: IndexedBaseMap,
                 parameters: IndexedBaseMap,
                 constants: IndexedBaseMap,
                 observables: IndexedBaseMap,
                 drivers: IndexedBaseMap,
                 dxdt: IndexedBaseMap):
        self.states = states
        self.parameters = parameters
        self.constants = constants
        self.observables = observables
        self.drivers = drivers
        self.dxdt = dxdt
        self.all_indices = {**self.states.ref_map,
                            **self.parameters.ref_map,
                            **self.constants.ref_map,
                            **self.observables.ref_map,
                            **self.drivers.ref_map,
                            **self.dxdt.ref_map}

    @classmethod
    def from_user_inputs(cls,
                         states: Union[dict,Iterable[str]],
                         parameters: Union[dict,Iterable[str]],
                         constants: Union[dict,Iterable[str]],
                         observables: Iterable[str],
                         drivers: Iterable[str]):
        states_ = IndexedBaseMap(
                "state",
                [s for s in states])
        parameters_ = IndexedBaseMap(
                "parameters",
                [p for p in parameters])
        constants_ = IndexedBaseMap(
                "constants",
                [c for c in constants])
        observables_ = IndexedBaseMap(
                "observables",
                observables)
        drivers_ = IndexedBaseMap(
                "drivers",
                drivers)
        dxdt_ = IndexedBaseMap(
                "dxdt",
                [f"d{s}" for s in states])
        return cls(states_,
                   parameters_,
                   constants_,
                   observables_,
                   drivers_,
                   dxdt_)

    @property
    def state_names(self):
        return self.states.ref_map.keys()

    @property
    def state_values(self):
        return self.states.default_values

    @property
    def parameter_names(self):
        return self.parameters.ref_map.keys()

    @property
    def parameter_values(self):
        return self.parameters.default_values

    @property
    def constant_names(self):
        return self.constants.ref_map.keys()

    @property
    def constant_values(self):
        return self.constants.default_values

    @property
    def observable_names(self):
        return self.observables.ref_map.keys()

    @property
    def driver_names(self):
        return self.drivers.ref_map.keys()

    @property
    def dxdt_names(self) -> Iterable[str]:
        return list(self.dxdt.ref_map.keys())

    @property
    def all_symbols(self) -> dict[str, sp.Symbol]:
        return {**self.states.ref_map,
                **self.parameters.ref_map,
                **self.constants.ref_map,
                **self.observables.ref_map,
                **self.drivers.ref_map,
                **self.dxdt.ref_map}

    def __getitem__(self, item):
        """Returns a reference to the indexed base for any symbol in the map"""
        return self.all_indices[item]


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

# -------------------------- Process equations ------------------------------ #
def _process_calls(equations_input: Iterable[str],
                   user_functions: Optional[Dict[str, callable]] = None):
    """ map known SymPy callables (e.g., 'exp') to Sympy functions """
    calls = set()
    for line in equations_input:
        calls |= set(_func_call_re.findall(line))
    funcs = {}
    for name in calls:
        fn = getattr(sp, name, None)
        if callable(fn):
            funcs[name] = fn
        elif name in user_functions:
            funcs[name] = user_functions[name] #TODO: Figure out how
            # exactly this works with Sympy - really, in code we just want
            # it to be pasted in in text as-is
        else:
            raise ValueError(f"Your dxdt code contains a call to a function"
                            f" {name}() that isn't part of Sympy and wasn't "
                             f"provided in the user_functions dict.")
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
    ) -> dict[str, sp.Symbol]:
    """ Process the left-hand-sides of all equations.

    Parameters
    ----------
    lines: list of str
        User-supplied list of equations that make up the dxdt function
    indexed_bases: IndexedBases
        The collection of maps from labels to indexed bases for the system
        generated by '_process_parameters'.

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
            if state_name not in state_names:
                if state_name in observable_names:
                    warn(
                        f"Your equation included d{state_name}, but "
                        f"{state_name} was listed as an observable. It has"
                        "been converted into a state.",
                        EquationWarning,
                    )
                    states.push(state_name)
                    dxdt.push(state_name)

                    observables.pop(state_name)
                else:
                    ValueError(f"Unknown state derivative: {lhs}.")
                    f"No state or observable called {state_name} found."
            underived_states -= lhs

        elif lhs in states:
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
            if lhs not in observables:
                warn(
                    f"The intermediate variable {lhs} was assigned to "
                    f"but not listed as an observable. It's trajectory will "
                    f"not be saved.",
                    EquationWarning,
                )
                anonymous_auxiliaries[lhs] = sp.Symbol(lhs, real=True)
            assigned_obs.add(lhs)

    missing_obs = set(observable_names) - assigned_obs
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
            if state in observables:
                raise ValueError(
                    f"State {state} is already both observable and state. "
                    f"It needs to be an observable if it has no derivative"
                    f"term."
                )
            observables.push(state)
            states.pop(state)
            dxdt.pop(state)

    return anonymous_auxiliaries

def _rhs_pass(lines: Iterable[str],
              all_symbols: Dict[str, sp.Symbol],
              user_funcs: Optional[Dict[str, callable]] = None):
    """Process expressions, checking symbols and finding callables.

    Parameters
    ----------
    lines: list of str
        User-supplied list of equations that make up the dxdt function
    all_symbols: dict
        All symbols defined in the model, including anonymous auxiliaries.

    returns:


    """
    expressions = {}
    funcs = _process_calls(lines, user_funcs)
    all_symbols.update(funcs)
    symbols_and_calls = all_symbols.update(funcs)
    for line in lines:
        lhs, rhs = [p.strip() for p in line.split("=", 1)]
        try:
            rhs_expr = _sanitise_input_math(rhs)
            rhs_expr = sp.sympify(rhs_expr, locals=symbols_and_calls)
        except (NameError, TypeError):
            raise ValueError(f"Undefined symbols in equation '{line}'")
        expressions[lhs] = rhs_expr

    return expressions, funcs


def hash_dxdt(dxdt: Union[str, Iterable[str]]) -> str:
    """Generate a hash of the dxdt function

    Clean and hash the dxdt input, to compare with cached versions of the
    function to check whether a rebuild is required.

    Parameters
    ----------
    dxdt : str or iterable of str
        The string representation of the dxdt function.

    Returns
    -------
    int: hash of the dxdt function

    Notes
    -----
    Concatenates all strings in the iterable into a single string, then removes
    all whitespace characters. The result is hashed using Python's built-in
    hash algorithm
    """
    if isinstance(dxdt, (list, tuple)):
        dxdt = "".join(dxdt)

    # Remove all whitespace characters
    normalized = "".join(dxdt.split())

    # Generate hash for compact, unique representation
    return hash(normalized)

def parse_input(
        states: Union[Dict, Iterable[str]],
        observables: Iterable[str],
        parameters: Union[Dict, Iterable[str]],
        constants: Union[Dict, Iterable[str]],
        drivers: Iterable[str],
        user_functions: Optional[Dict[str, callable]] = None,
        dxdt = Union[str, Iterable[str]],
):
    """Create a :class:`SymbolicODE` from manual string input."""
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

    fn_hash = hash_dxdt(dxdt)
    anon_aux = _lhs_pass(lines, index_map)
    all_symbols = index_map.all_symbols.copy()
    all_symbols.update(anon_aux)
    equation_map, funcs = _rhs_pass(lines=lines,
                             all_symbols=all_symbols,
                             user_funcs=user_functions)
    # Funcs dead-end here, expose if they prove vital for codegen.

    return index_map, all_symbols, equation_map, fn_hash