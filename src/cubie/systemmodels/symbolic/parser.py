"""Parsing helpers for symbolic ODE definitions."""

import re
from warnings import warn

import numpy as np
import sympy as sp

from cubie.systemmodels.symbolic.symbolicODE import SymbolicODE


class EquationWarning(Warning):
    pass


def CUDA_substitutions(expr_str):
    """Replace CUDA-specific substitutions in a string."""
    expr_str = _replace_if(expr_str)
    # TODO: Math swaps - pow for *, etc.
    return expr_str


def _replace_if(expr_str):
    match = re.search(r"(.+?) if (.+?) else (.+)", expr_str)
    if match:
        true_str = _replace_if(match.group(1).strip())
        cond_str = _replace_if(match.group(2).strip())
        false_str = _replace_if(match.group(3).strip())
        return f"Piecewise(({true_str}, {cond_str}), ({false_str}, True))"
    return expr_str




def collect_symbols(containers):
    """Get a complete dict of symbols from an iterable of lists/dicts"""
    symbols = {}
    for container in containers:
        symbols.update({name: sp.symbols(name) for name in container})
    return symbols


# TODO: I think this is a good opportunity for a factory. - return SymbolicODE
#  if dxdt is a string/list of strings, and return a generic ODE if it's a
#  list of
#  sympy expressions.
def create_system(
    observables,
    parameters,
    constants,
    drivers,
    states,
    dxdt,
    precision=np.float64,
):
    """Create a :class:`SymbolicODE` from manual string input."""
    all_symbols = collect_symbols(
        [observables, parameters, constants, drivers, states]
    )

    if isinstance(dxdt, str):
        lines = [
            line.strip() for line in dxdt.strip().splitlines() if line.strip()
        ]
    else:
        lines = [line.strip() for line in dxdt if line.strip()]

    lhs_outputs = _lhs_pass(lines=lines,
                            states=states,
                            parameters=parameters,
                            constants=constants,
                            drivers=drivers,
                            observables=observables,
                            all_symbols=all_symbols
    )

    states, observables, all_symbols = lhs_outputs
    equations = _rhs_pass(lines, all_symbols)

    state_syms = {all_symbols[n]: v for n, v in states.items()}
    param_syms = {all_symbols[n]: v for n, v in parameters.items()}
    const_syms = {all_symbols[n]: v for n, v in (constants or {}).items()}
    obs_syms = [all_symbols[n] for n in observables]
    driver_syms = [all_symbols[n] for n in drivers]

    return SymbolicODE(
        states=state_syms,
        parameters=param_syms,
        constants=const_syms,
        observables=obs_syms,
        drivers=driver_syms,
        equations=equations,
        precision=precision,
    )


def _lhs_pass(
    lines,
    states,
    parameters,
    constants,
    drivers,
    observables,
    all_symbols
):
    assigned_obs = set()
    deriv_states = set()
    for line in lines:
        lhs, rhs = [p.strip() for p in line.split("=", 1)]
        if lhs.startswith("d"):
            state_name = lhs[1:]
            if state_name not in states:
                if state_name in observables:
                    warn(
                        f"Your equation included d{state_name}, but "
                        f"{state_name} was listed as an observable. It has"
                        "been converted into a state.",
                        EquationWarning,
                    )
                    states.extend(state_name)
                    deriv_states.add(state_name)
                    observables.pop(state_name, None)
                else:
                    ValueError(f"Unknown state derivative: d{state_name}.")
                    f"No state or observable called {state_name} found."
            deriv_states.add(state_name)

        elif lhs in states:
            raise ValueError(
                f"State {lhs} cannot be assigned directly. All "
                f"states must be defined as derivatives with d"
                f"{lhs} = [...]"
            )

        elif lhs in parameters or lhs in constants or lhs in drivers:
            raise ValueError(
                f"{lhs} was entered as an immutable "
                f"input (constant, parameter, or driver)"
                ", but it is being assigned to. Cubie "
                "can't handle this - if it's being "
                "assigned to, it must be either an "
                "observable."
            )

        else:
            if lhs not in observables:
                warn(
                    f"The intermediate variable {lhs} was assigned to "
                    f"but not listed as an observable. It has been added"
                    f" as an observable.",
                    EquationWarning,
                )
                observables.extend(lhs)
                all_symbols[lhs] = sp.Symbol(lhs)
            assigned_obs.add(lhs)

    missing_obs = set(observables) - assigned_obs
    if missing_obs:
        raise ValueError(f"Observables {missing_obs} are never assigned "
                         f"to.")
    missing_states = set(states) - deriv_states
    if missing_states:
        warn(
            f"States {missing_states} have no associated derivative "
            f"term. In the Cubie world, this makes it an 'observable'. "
            f"{missing_states} have been moved from states to observables.",
            EquationWarning,
        )
        for state in missing_states:
            if state in observables:
                raise ValueError(
                    f"State {state} is both observable and state"
                )
            observables.extend(state)
            states.pop(state, None)

    return states, observables, all_symbols


def _rhs_pass(lines, all_symbols):
    """Setup equations from input lines with checked lhs symbols."""
    equations = []
    for line in lines:
        lhs, rhs = [p.strip() for p in line.split("=", 1)]
        try:
            rhs_expr = CUDA_substitutions(rhs)
            rhs_expr = sp.sympify(rhs_expr, locals=all_symbols)
        except (NameError, TypeError):
            raise ValueError(f"Undefined symbols in equation '{line}'")

        if lhs.startswith("d"):
            lhs = lhs[1:]
        equations.append(sp.Eq(all_symbols[lhs], rhs_expr))

    return equations