"""Symbolic ODE system built from :mod:`sympy` expressions.

This module extends the basic symbolic prototype by adding support for a
restricted set of mathematical operators, automatic Jacobian generation and a
Numba CUDA implementation.  The implementation of placeholder math functions
and Jacobian generation borrows ideas from
`chaste-codegen <https://github.com/ModellingWebLab/chaste-codegen>`_.

The code copied from *chaste-codegen* is licensed under the MIT licence and has
been adapted for use in this project.
"""

import numpy as np
import sympy as sp
from numba import cuda, from_dtype
import re

from cubie.systemmodels.systems.GenericODE import GenericODE
from cubie.systemmodels.symbolic.math_functions import (
    subs_math_func_placeholders,
)
from cubie.systemmodels.symbolic.jacobian import get_jacobian_matrix


class SymbolicODESystem(GenericODE):
    """Create an ODE system from SymPy expressions.

    Parameters are provided as SymPy symbols.  The differential equations are
    provided as a list of ``sympy.Eq`` objects where the left hand side is a
    state or observable symbol and the right hand side is an expression
    composed of states, parameters, constants and previously defined
    observables.
    """

    def __init__(self, *, constants=None, parameters=None, states=None,
                 observables=None, equations=None, drivers=None,
                 precision=np.float64, num_drivers=None):
        if equations is None:
            raise ValueError("equations must be provided")

        # Store the original SymPy symbols for later use
        self.state_symbols, state_vals = self._prepare_container(states)
        self.parameter_symbols, param_vals = self._prepare_container(
            parameters
        )
        self.constant_symbols, const_vals = self._prepare_container(constants)
        self.observable_symbols, obs_vals = self._prepare_container(
            observables
        )
        self.driver_symbols = list(drivers) if drivers is not None else []

        self.equations = equations

        if num_drivers is None:
            num_drivers = max(1, len(self.driver_symbols))

        super().__init__(
            initial_values=state_vals,
            parameters=param_vals,
            constants=const_vals,
            observables=obs_vals,
            precision=precision,
            num_drivers=num_drivers,
        )

        self._validate_equations()

    @staticmethod
    def _prepare_container(container):
        """Convert a container of SymPy symbols to a form suitable for
        :class:`SystemValues`.

        Returns a tuple ``(symbols, sv_input)`` where ``symbols`` is a list of
        SymPy symbols and ``sv_input`` is either a list of names or a dict of
        ``name -> value``.
        """
        if container is None:
            return [], {}
        if isinstance(container, dict):
            symbols = list(container.keys())
            sv_input = {s.name: container[s] for s in symbols}
            return symbols, sv_input
        else:
            symbols = list(container)
            sv_input = [s.name for s in symbols]
            return symbols, sv_input

    def _validate_equations(self):
        allowed = set(self.state_symbols + self.parameter_symbols +
                      self.constant_symbols + self.observable_symbols +
                      self.driver_symbols)
        for eq in self.equations:
            if not isinstance(eq, sp.Eq):
                raise TypeError("equations must be sympy.Eq instances")
            if eq.lhs not in allowed:
                raise ValueError("Equation LHS must be a state or observable")
            undefined = eq.rhs.free_symbols - allowed
            if undefined:
                raise ValueError(
                    f"Equation RHS contains undefined symbols: {undefined}"
                )

    def build(self):
        """Compile the system into CUDA device functions.

        Returns the compiled derivative function.  The Jacobian device function
        is available as :attr:`jacobian_function`.
        """
        global global_constants
        global_constants = self.compile_settings.constants.values_array.astype(
            self.precision
        )

        dxdt_func = self._build_dxdt()
        self._jacobian_function = self._build_jacobian()
        return dxdt_func

    # ------------------------------------------------------------------
    # Device code generation helpers
    # ------------------------------------------------------------------

    def _build_dxdt(self):
        numba_precision = from_dtype(self.precision)

        code_lines = self._generate_code_lines()
        func_code = [
            "def sympy_dxdt(state, parameters, driver, observables, dxdt):"
        ]
        func_code.extend([f"    {line}" for line in code_lines])
        ns = {}
        exec("\n".join(func_code), {}, ns)
        python_func = ns["sympy_dxdt"]

        jitted = cuda.jit(
            (
                numba_precision[:],
                numba_precision[:],
                numba_precision[:],
                numba_precision[:],
                numba_precision[:],
            ),
            device=True,
            inline=True,
        )(python_func)
        return jitted

    def _build_jacobian(self):
        numba_precision = from_dtype(self.precision)

        code_lines = self._generate_jacobian_code_lines()
        func_code = [
            "def sympy_jacobian(state, parameters, driver, observables, J):"
        ]
        func_code.extend([f"    {line}" for line in code_lines])
        ns = {}
        exec("\n".join(func_code), {}, ns)
        python_func = ns["sympy_jacobian"]

        jitted = cuda.jit(
            (
                numba_precision[:],
                numba_precision[:],
                numba_precision[:],
                numba_precision[:],
                numba_precision[:, :],
            ),
            device=True,
            inline=True,
        )(python_func)
        return jitted

    def _generate_code_lines(self):
        state_idx = {s: i for i, s in enumerate(self.state_symbols)}
        obs_idx = {s: i for i, s in enumerate(self.observable_symbols)}
        driver_idx = {s: i for i, s in enumerate(self.driver_symbols)}
        param_idx = {s: i for i, s in enumerate(self.parameter_symbols)}
        const_idx = {s: i for i, s in enumerate(self.constant_symbols)}

        state_arr = sp.IndexedBase('state')
        param_arr = sp.IndexedBase('parameters')
        const_arr = sp.IndexedBase('global_constants')
        obs_arr = sp.IndexedBase('observables')
        driver_arr = sp.IndexedBase('driver')

        repl = {s: state_arr[i] for s, i in state_idx.items()}
        repl.update({s: param_arr[i] for s, i in param_idx.items()})
        repl.update({s: const_arr[i] for s, i in const_idx.items()})
        repl.update({s: obs_arr[i] for s, i in obs_idx.items()})
        repl.update({s: driver_arr[i] for s, i in driver_idx.items()})

        lines = []
        for eq in self.equations:
            rhs = subs_math_func_placeholders(eq.rhs).subs(repl)
            rhs_code = sp.pycode(rhs)
            if eq.lhs in state_idx:
                lines.append(f"dxdt[{state_idx[eq.lhs]}] = {rhs_code}")
            else:
                lines.append(f"observables[{obs_idx[eq.lhs]}] = {rhs_code}")
        return lines

    @property
    def jacobian_function(self):
        """Return the compiled Jacobian device function."""
        if (
            getattr(self, "_jacobian_function", None) is None
            and not self.cache_valid
        ):
            self.device_function  # triggers build
        return getattr(self, "_jacobian_function", None)

    def correct_answer_python(self, states, parameters, drivers):
        state_idx = {s: i for i, s in enumerate(self.state_symbols)}
        param_idx = {s: i for i, s in enumerate(self.parameter_symbols)}
        const_idx = {s: i for i, s in enumerate(self.constant_symbols)}
        obs_idx = {s: i for i, s in enumerate(self.observable_symbols)}
        driver_idx = {s: i for i, s in enumerate(self.driver_symbols)}

        values = {}
        for sym, i in state_idx.items():
            values[sym] = states[i]
        for sym, i in param_idx.items():
            values[sym] = parameters[i]
        for sym, i in const_idx.items():
            values[sym] = self.compile_settings.constants.values_array[i]
        for sym, i in driver_idx.items():
            values[sym] = drivers[i]

        dxdt = np.zeros(self.num_states, dtype=self.precision)
        observables = np.zeros(self.num_observables, dtype=self.precision)

        for eq in self.equations:
            rhs_val = float(subs_math_func_placeholders(eq.rhs).subs(values))
            if eq.lhs in state_idx:
                dxdt[state_idx[eq.lhs]] = rhs_val
            else:
                observables[obs_idx[eq.lhs]] = rhs_val
                values[eq.lhs] = rhs_val
        return dxdt, observables

    # ------------------------------------------------------------------
    # Jacobian helpers
    # ------------------------------------------------------------------

    def _generate_jacobian_code_lines(self):
        state_idx = {s: i for i, s in enumerate(self.state_symbols)}
        param_idx = {s: i for i, s in enumerate(self.parameter_symbols)}
        const_idx = {s: i for i, s in enumerate(self.constant_symbols)}
        obs_idx = {s: i for i, s in enumerate(self.observable_symbols)}
        driver_idx = {s: i for i, s in enumerate(self.driver_symbols)}

        state_arr = sp.IndexedBase("state")
        param_arr = sp.IndexedBase("parameters")
        const_arr = sp.IndexedBase("global_constants")
        obs_arr = sp.IndexedBase("observables")
        driver_arr = sp.IndexedBase("driver")

        repl = {s: state_arr[i] for s, i in state_idx.items()}
        repl.update({s: param_arr[i] for s, i in param_idx.items()})
        repl.update({s: const_arr[i] for s, i in const_idx.items()})
        repl.update({s: obs_arr[i] for s, i in obs_idx.items()})
        repl.update({s: driver_arr[i] for s, i in driver_idx.items()})

        deriv_eqs = [eq for eq in self.equations if eq.lhs in state_idx]
        cse_eqs, J = get_jacobian_matrix(self.state_symbols, deriv_eqs)

        lines = []
        for sym, expr in cse_eqs:
            expr = subs_math_func_placeholders(expr).subs(repl)
            lines.append(f"{sp.pycode(sym)} = {sp.pycode(expr)}")

        for i in range(J.rows):
            for j in range(J.cols):
                expr = subs_math_func_placeholders(J[i, j]).subs(repl)
                expr_code = sp.pycode(expr)
                lines.append(f"J[{i}, {j}] = {expr_code}")
        return lines

    def correct_jacobian_python(self, states, parameters, drivers):
        state_idx = {s: i for i, s in enumerate(self.state_symbols)}
        param_idx = {s: i for i, s in enumerate(self.parameter_symbols)}
        const_idx = {s: i for i, s in enumerate(self.constant_symbols)}
        driver_idx = {s: i for i, s in enumerate(self.driver_symbols)}

        values = {}
        for sym, i in state_idx.items():
            values[sym] = states[i]
        for sym, i in param_idx.items():
            values[sym] = parameters[i]
        for sym, i in const_idx.items():
            values[sym] = self.compile_settings.constants.values_array[i]
        for sym, i in driver_idx.items():
            values[sym] = drivers[i]

        deriv_eqs = [eq for eq in self.equations if eq.lhs in state_idx]
        rhs = [subs_math_func_placeholders(eq.rhs) for eq in deriv_eqs]
        J = sp.Matrix(rhs).jacobian(sp.Matrix(self.state_symbols))
        J_num = np.array(J.subs(values)).astype(self.precision)
        return J_num


def setup_system(observables, parameters, constants, drivers, states, dxdt,
                 precision=np.float64):
    """Create a :class:`SymbolicODESystem` from manual string input."""

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

    return SymbolicODESystem(
        states=state_syms,
        parameters=param_syms,
        constants=const_syms,
        observables=obs_syms,
        drivers=driver_syms,
        equations=equations,
        precision=precision,
    )