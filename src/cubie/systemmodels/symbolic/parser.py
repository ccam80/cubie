"""Parsing helpers for symbolic ODE definitions."""

import re
import sympy as sp
import numpy as np

from cubie.systemmodels.symbolic.symbolicODE import SymbolicODE


def setup_system(
    observables,
    parameters,
    constants,
    drivers,
    states,
    dxdt,
    precision=np.float64,
):
    """Create a :class:`SymbolicODE` from manual string input."""

    def _replace_if(expr_str):
        match = re.search(r"(.+?) if (.+?) else (.+)", expr_str)
        if match:
            true_str = _replace_if(match.group(1).strip())
            cond_str = _replace_if(match.group(2).strip())
            false_str = _replace_if(match.group(3).strip())
            return (
                f"Piecewise(({true_str}, {cond_str}), ({false_str}, True))"
            )
        return expr_str

    symbol_names = (
        set(observables)
        | set(parameters)
        | set(constants)
        | set(drivers)
        | set(states)
    )
    symbols = {name: sp.symbols(name) for name in symbol_names}

    if isinstance(dxdt, str):
        lines = [l.strip() for l in dxdt.strip().splitlines() if l.strip()]
    else:
        lines = [l.strip() for l in dxdt if l.strip()]

    equations = []
    assigned_obs = set()
    deriv_states = set()
    for line in lines:
        lhs, rhs = [p.strip() for p in line.split("=", 1)]
        rhs_expr = eval(
            _replace_if(rhs), {"Piecewise": sp.Piecewise}, symbols
        )
        if lhs.startswith("d"):
            state_name = lhs[1:]
            if state_name not in states:
                raise ValueError(f"Unknown state derivative: {state_name}")
            equations.append(sp.Eq(symbols[state_name], rhs_expr))
            deriv_states.add(state_name)
        elif lhs in states:
            raise ValueError(f"State {lhs} cannot be assigned directly")
        else:
            if lhs not in observables:
                raise ValueError(f"Unknown observable: {lhs}")
            equations.append(sp.Eq(symbols[lhs], rhs_expr))
            assigned_obs.add(lhs)

    missing_obs = set(observables) - assigned_obs
    if missing_obs:
        raise ValueError(f"Observables without assignment: {missing_obs}")
    missing_states = set(states) - deriv_states
    if missing_states:
        raise ValueError(f"States without derivatives: {missing_states}")

    state_syms = {symbols[n]: v for n, v in states.items()}
    param_syms = {symbols[n]: v for n, v in parameters.items()}
    const_syms = {symbols[n]: v for n, v in (constants or {}).items()}
    obs_syms = [symbols[n] for n in observables]
    driver_syms = [symbols[n] for n in drivers]

    return SymbolicODE(
        states=state_syms,
        parameters=param_syms,
        constants=const_syms,
        observables=obs_syms,
        drivers=driver_syms,
        equations=equations,
        precision=precision,
    )
