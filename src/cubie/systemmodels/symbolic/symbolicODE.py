"""Symbolic ODE system built from :mod:`sympy` expressions.

This module extends the basic symbolic prototype by adding support for a
restricted set of mathematical operators, automatic Jacobian generation and a
Numba CUDA implementation.  The implementation of placeholder math functions
and Jacobian generation borrows ideas from
`chaste-codegen <https://github.com/ModellingWebLab/chaste-codegen>`_.

The code copied from *chaste-codegen* is licensed under the MIT licence and has
been adapted for use in this project.
"""
from typing import Callable, Optional, Iterable, Union, Dict

import attrs
import numpy as np
import sympy as sp
from numba import from_dtype, cuda
from numba_cuda.numba.cuda.simulator.dispatcher import CUDADispatcher

from cubie.systemmodels.symbolic.dxdt import generate_dxdt_function
from cubie.systemmodels.symbolic.odefile import ODEFile
from cubie.systemmodels.symbolic.jacobian import (
    generate_jacobian_function,
    generate_analytical_jac_v,
)
from cubie.systemmodels.symbolic.parser import IndexedBases, \
    parse_input, hash_dxdt
from cubie.systemmodels.systems.GenericODE import GenericODE


#TODO: Consider hoisting this into GenericODE - it's useful if we need to
# add analytical derivatives to manually-defined systems.
@attrs.define()
class ODECache:
    dxdt: Optional[Callable] = None
    jac_v: Optional[Callable] = None

def build_system(dxdt, jacobian=False):
    """Create an ODE system from SymPy expressions."""

class SymbolicODE(GenericODE):
    """Create an ODE system from SymPy expressions.

    Parameters are provided as SymPy symbols.  The differential equations are
    provided as a list of ``sympy.Eq`` objects where the left hand side is a
    state or observable symbol and the right hand side is an expression
    composed of states, parameters, constants and previously defined
    observables.
    """

    def __init__(
        self,
        equation_map: Dict[str, sp.Expr],
        all_indexed_bases: IndexedBases,
        all_symbols: Optional[dict[str, sp.Symbol]] = None,
        precision=np.float64,
        fn_hash: Optional[int] =None,
        jac_v: Optional[Callable] = None,
        name: str = None,
    ):
        if equation_map is None:
            raise ValueError("equations must be provided")
        if (isinstance(equation_map, str) or
            (isinstance(equation_map, Iterable)
             and isinstance(equation_map[0], str))):
            raise ValueError("Equation map should be a dict mapping a string "
                             "(with corresponding symbol in indices dict) to "
                             "a Sympy expression.")

        if all_symbols is None:
            all_symbols = all_indexed_bases.all_symbols
        self.all_symbols = all_symbols

        if fn_hash is None:
            dxdt_str = [f"{lhs}={str(rhs)}" for lhs, rhs
                        in equation_map.items()]
            fn_hash = hash(dxdt_str)
        if name is None:
            name = fn_hash

        self.filename = f"{name}.py"
        self.gen_file = ODEFile(name)

        ndriv = all_indexed_bases.drivers.length
        self.equations = equation_map
        self.indices = all_indexed_bases
        self.fn_hash = fn_hash
        self.user_jac_v = jac_v

        super().__init__(
            initial_values=all_indexed_bases.state_values,
            parameters=all_indexed_bases.parameter_values,
            constants=all_indexed_bases.constant_values,
            observables=all_indexed_bases.observable_names,
            precision=precision,
            num_drivers=ndriv,
        )

    @classmethod
    def create(cls,
              dxdt: Union[str, Iterable[str]],
              states: Union[dict,Iterable[str]],
              observables: Iterable[str],
              parameters: Union[dict,Iterable[str]],
              constants: Union[dict,Iterable[str]],
              drivers: Iterable[str],
              user_functions: Optional[dict[str, Callable]] = None,
              name: str = None):

        sys_components = parse_input(
                states = states,
                observables = observables,
                parameters = parameters,
                constants = constants,
                drivers = drivers,
                user_functions=user_functions,
                dxdt = dxdt,
        )
        equation_map, index_map, all_symbols, fn_hash = sys_components

        return cls(equation_map=equation_map,
                   all_indexed_bases=index_map,
                   all_symbols=all_symbols,
                   name=name,
                   fn_hash=fn_hash,
                   precision=np.float64)


    def build(self, jacobian=False):
        """Compile the dxdt and (if requested_ jacobian device functions..
        """
        dxdt_func = self._build_dxdt()
        jacobian_function = self._build_jacobian()
        return ODECache(dxdt = dxdt_func,
                        jac_v = jacobian_function)

    # ------------------------------------------------------------------
    # Device code generation helpers
    # ------------------------------------------------------------------

    def _build_dxdt(self):
        numba_precision = from_dtype(self.precision)
        constants = self.indices.constant_values.astype(self.precision)
        code_lines = self._generate_dxdt_lines()
        dxdt_factory = generate_dxdt_function(code_lines, self.gen_file)
        dxdt = dxdt_factory(constants, numba_precision)
        return dxdt

    def _build_jacobian(self):
        if self.user_jac_v:
            if isinstance(self.user_jac_v, CUDADispatcher):
                return self.user_jac_v
            if callable(self.user_jac_v):
                try:
                    return cuda.jit()(self.user_jac_v)
                except Exception as e:
                    print(f"Error compiling user-provided Jacobian: {e}")
            else:
                raise ValueError("user_jac_v must be a callable or a "
                                 "Numba Dispatcher object")
        else:

            numba_precision = from_dtype(self.precision)
            constants = self.compile_settings.constants.values_array.astype(
                self.precision)
            code_lines = generate_analytical_jac_v(self)
            jac_v_factory = generate_jacobian_function(code_lines, self.gen_file)
            jac_v = jac_v_factory(constants, numba_precision)
            return jac_v

    @property
    def dxdt(self):
        return self.get_cached_output("dxdt")

    @property
    def jac_v(self):
        return self.get_cached_output("jac_v")
