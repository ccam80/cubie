"""Symbolic ODE system built from :mod:`sympy` expressions.

This module extends the basic symbolic prototype by adding support for a
restricted set of mathematical operators, automatic Jacobian generation and a
Numba CUDA implementation.  The implementation of placeholder math functions
and Jacobian generation borrows ideas from
`chaste-codegen <https://github.com/ModellingWebLab/chaste-codegen>`_.

The code copied from *chaste-codegen* is licensed under the MIT licence and has
been adapted for use in this project.
"""
from typing import Callable, Optional

import attrs
import numpy as np
import sympy as sp
from numba import from_dtype

from cubie.systemmodels.symbolic.dxdt import generate_dxdt_function
from cubie.systemmodels.symbolic.file_generation import GeneratedFile
from cubie.systemmodels.symbolic.jacobian import (
    generate_jacobian_function,
    get_jacobian_matrix,
)
from cubie.systemmodels.symbolic.math_functions import (
    subs_math_func_placeholders,
)
from cubie.systemmodels.systems.GenericODE import GenericODE


#TODO: Consider hoisting this into GenericODE - it's useful if we need to
# add analytical derivatives to manually-defined systems.
@attrs.define()
class ODECache:
    dxdt: Optional[Callable] = None
    jacv: Optional[Callable] = None

@attrs.define(slots=False)
class SymbolIndices:
    state: dict[sp.Symbol, int] = attrs.field(factory=dict),
    observables: dict[sp.Symbol, int] = attrs.field(factory=dict),
    parameters: dict[sp.Symbol, int] = attrs.field(factory=dict),
    constants: dict[sp.Symbol, int] = attrs.field(factory=dict),
    drivers: dict[sp.Symbol, int] = attrs.field(factory=dict)

    @classmethod
    def from_symbols_dicts(cls,
                           state,
                           observables,
                           parameters,
                           constants_,
                           drivers):
        state = {s: i for i, s in enumerate(state)}
        parameters = {s: i for i, s in enumerate(parameters)}
        constants_ = {s: i for i, s in enumerate(constants_)}
        observables = {s: i for i, s in enumerate(observables)}
        drivers = {s: i for i, s in enumerate(drivers)}

        return cls(state=state,
                   observables=observables,
                   constants=constants_,
                   parameters=parameters,
                   drivers=drivers)

#TODO: Rehome this - was causing a circular import in parser
def parse_values(container):
    """Convert a list or dict of names:vals to a list of symbols and a dict.

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

class SymbolicODE(GenericODE):
    """Create an ODE system from SymPy expressions.

    Parameters are provided as SymPy symbols.  The differential equations are
    provided as a list of ``sympy.Eq`` objects where the left hand side is a
    state or observable symbol and the right hand side is an expression
    composed of states, parameters, constants and previously defined
    observables.
    """

    def __init__(self,
                 constants=None,
                 parameters=None,
                 states=None,
                 observables=None,
                 equations=None,
                 drivers=None,
                 precision=np.float64,
                 name: str =None):
        if equations is None:
            raise ValueError("equations must be provided")
        # Store the original SymPy symbols for later use
        self.state_symbols, state_vals = parse_values(states)
        self.parameter_symbols, param_vals = parse_values(
            parameters
        )
        self.constant_symbols, const_vals = parse_values(constants)
        self.observable_symbols, obs_vals = parse_values(observables)
        self.driver_symbols = list(drivers) if drivers is not None else []
        ndriv = len(self.driver_symbols)
        self.equations = equations

        if name is None:
            nstates = len(self.state_symbols)
            nparams = len(self.parameter_symbols)
            nconst = len(self.constant_symbols)
            nobs = len(self.observable_symbols)
            name = (f"SymbolicODE_{nstates}x_{nparams}p_{nconst}c_{nobs}ob_"
                    f"{ndriv}d")
        self.filename = f"{name}.py"
        self.gen_file = GeneratedFile(name)

        super().__init__(
            initial_values=state_vals,
            parameters=param_vals,
            constants=const_vals,
            observables=obs_vals,
            precision=precision,
            num_drivers=ndriv,
        )
        self._validate_equations()

        self.indices = SymbolIndices.from_symbols_dicts(
                state=self.state_symbols,
                observables=self.observable_symbols,
                parameters=self.parameter_symbols,
                constants_=self.constant_symbols,
                drivers=self.driver_symbols
        )


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

    def build(self, jacobian=False):
        """Compile the dxdt and (if requested_ jacobian device functions..
        """
        global constants
        constants = self.compile_settings.constants.values_array.astype(
            self.precision
        )

        dxdt_func = self._build_dxdt()
        jacobian_function = self._build_jacobian()

        return ODECache(dxdt = dxdt_func,
                        jacv = jacobian_function)

    # ------------------------------------------------------------------
    # Device code generation helpers
    # ------------------------------------------------------------------

    def _build_dxdt(self):
        numba_precision = from_dtype(self.precision)
        constants = self.compile_settings.constants.values_array.astype(
                self.precision)
        code_lines = self._generate_dxdt_lines()
        dxdt_factory = generate_dxdt_function(code_lines, self.gen_file)
        dxdt = dxdt_factory(constants, numba_precision)
        return dxdt

    def _build_jacobian(self):
        numba_precision = from_dtype(self.precision)
        constants = self.compile_settings.constants.values_array.astype(
            self.precision
        )
        code_lines = self._generate_jacobian_code_lines()
        jac_v_factory = generate_jacobian_function(code_lines, self.gen_file)
        jac_v = jac_v_factory(constants, numba_precision)
        return jac_v

    @property
    def array_base_map(self):
        """Returns mapping of symbol name to array[index] reference"""
        indices = self.indices

        state_arr = sp.IndexedBase("state")
        param_arr = sp.IndexedBase("parameters")
        const_arr = sp.IndexedBase("constants")
        obs_arr = sp.IndexedBase("observables")
        driver_arr = sp.IndexedBase("driver")

        arrays = {'state': state_arr,
                  'parameters': param_arr,
                  'constants': const_arr,
                  'observables': obs_arr,
                  'drivers': driver_arr}
        array_refs = {}

        for name, index_dict in indices.__dict__.items():
            array_refs.update({s: arrays[name][i] for s, i
                               in index_dict.items()})

        return array_refs

    def _generate_dxdt_lines(self):
        array_refs = self.array_base_map
        lines = []
        for eq in self.equations:
            rhs = subs_math_func_placeholders(eq.rhs).subs(array_refs)
            rhs_code = sp.pycode(rhs)
            if eq.lhs in self.state_symbols:
                lines.append(f"    dxdt[{self.indices.state[eq.lhs]}] ="
                             f" {rhs_code}")
            else:
                lines.append(f"    observables["
                             f"{self.indices.observables[eq.lhs]}] ="
                             f" {rhs_code}")
        return lines

    def correct_answer_python(self, states, parameters, drivers):
        indices = self.indices

        values = {}
        for sym, i in indices.state.items():
            values[sym] = states[i]
        for sym, i in indices.parameters.items():
            values[sym] = parameters[i]
        for sym, i in indices.constants.items():
            values[sym] = self.compile_settings.constants.values_array[i]
        for sym, i in indices.drivers.items():
            values[sym] = drivers[i]

        dxdt = np.zeros(self.num_states, dtype=self.precision)
        observables = np.zeros(self.num_observables, dtype=self.precision)

        for eq in self.equations:
            rhs_val = float(subs_math_func_placeholders(eq.rhs).subs(values))
            if eq.lhs in indices.state:
                dxdt[indices.state[eq.lhs]] = rhs_val
            else:
                observables[indices.observables[eq.lhs]] = rhs_val
                values[eq.lhs] = rhs_val
        return dxdt, observables

    # ------------------------------------------------------------------
    # Jacobian helpers
    # ------------------------------------------------------------------

    def _generate_jacobian_code_lines(self):
        array_refs = self.array_base_map
        cse_eqs, J = get_jacobian_matrix(self.state_symbols, self.equations)
        lines = []

        for sym, expr in cse_eqs:
            expr = subs_math_func_placeholders(expr).subs(array_refs)
            lines.append(f"{sp.pycode(sym)} = {sp.pycode(expr)}")

        for i in range(J.rows):
            for j in range(J.cols):
                expr = subs_math_func_placeholders(J[i, j]).subs(array_refs)
                expr_code = sp.pycode(expr)
                lines.append(f"    Jv[{i}, {j}] = {expr_code}")
        return lines

    def correct_jacobian_python(self, states, parameters, drivers):
        indices = self.indices

        values = {}
        for sym, i in indices.state.items():
            values[sym] = states[i]
        for sym, i in indices.parameters.items():
            values[sym] = parameters[i]
        for sym, i in indices.constants.items():
            values[sym] = self.compile_settings.constants.values_array[i]
        for sym, i in indices.constants.items():
            values[sym] = drivers[i]

        deriv_eqs = [eq for eq in self.equations if eq.lhs in indices.state]
        rhs = [subs_math_func_placeholders(eq.rhs) for eq in deriv_eqs]
        J = sp.Matrix(rhs).jacobian(sp.Matrix(self.state_symbols))
        J_num = np.array(J.subs(values)).astype(self.precision)
        return J_num

    @property
    def dxdt(self):
        return self.get_cached_output("dxdt")

    @property
    def jac_v(self):
        return self.get_cached_output("jac_v")
