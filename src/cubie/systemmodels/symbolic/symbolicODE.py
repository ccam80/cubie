"""Symbolic ODE system built from :mod:`sympy` expressions.

This module extends the basic symbolic prototype by adding support for a
restricted set of mathematical operators, automatic Jacobian generation and a
Numba CUDA implementation.  The implementation of placeholder math functions
and Jacobian generation borrows ideas from
`chaste-codegen <https://github.com/ModellingWebLab/chaste-codegen>`_.

The code copied from *chaste-codegen* is licensed under the MIT licence and has
been adapted for use in this project.
"""

from typing import Callable, Iterable, Optional, Union

import attrs
import numpy as np
import sympy as sp
from numba import cuda, from_dtype

from cubie.systemmodels.symbolic.odefile import ODEFile
from cubie.systemmodels.symbolic.parser import IndexedBases, parse_input
from cubie.systemmodels.systems.GenericODE import GenericODE


#TODO: Consider hoisting this into GenericODE - it's useful if we need to
# add analytical derivatives to manually-defined systems.
@attrs.define()
class ODECache:
    dxdt: Optional[Callable] = None
    jvp: Optional[Callable] = None
    vjp: Optional[Callable] = None

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
        equations: Iterable[tuple[sp.Symbol, sp.Expr]],
        all_indexed_bases: IndexedBases,
        all_symbols: Optional[dict[str, sp.Symbol]] = None,
        precision=np.float64,
        fn_hash: Optional[int] =None,
        jvp: Optional[Callable] = None,
        vjp: Optional[Callable] = None,
        user_functions: Optional[dict[str, Callable]] = None,
        name: str = None,
    ):
        if all_symbols is None:
            all_symbols = all_indexed_bases.all_symbols
        self.all_symbols = all_symbols

        if fn_hash is None:
            dxdt_str = [f"{lhs}={str(rhs)}" for lhs, rhs
                        in equations]
            fn_hash = hash(dxdt_str)
        if name is None:
            name = fn_hash

        self.filename = f"{name}.py"
        self.gen_file = ODEFile(name, fn_hash)

        ndriv = all_indexed_bases.drivers.length
        self.equations = equations
        self.indices = all_indexed_bases
        self.fn_hash = fn_hash
        self.user_jvp = jvp
        self.user_vjp = vjp

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
        #TODO: Handle the update_constants functionality updating the
        # constant values in the indexedbasemap container.
        sys_components = parse_input(
                states = states,
                observables = observables,
                parameters = parameters,
                constants = constants,
                drivers = drivers,
                user_functions=user_functions,
                dxdt = dxdt,
        )
        equations, index_map, functions, all_symbols, fn_hash = sys_components
        return cls(equations=equations,
                   all_indexed_bases=index_map,
                   all_symbols=all_symbols,
                   name=name,
                   fn_hash=fn_hash,
                   user_functions = functions,
                   precision=np.float64)


    def build(self, jacobian=False):
        """Compile the dxdt and (if requested_ jacobian device functions..
        """
        dxdt_func = self._build_dxdt()
        jvp = self._build_jvp()
        vjp = self._build_vjp()
        return ODECache(dxdt = dxdt_func,
                        jvp = jvp,
                        vjp = vjp)

    # ------------------------------------------------------------------
    # Device code generation helpers
    # ------------------------------------------------------------------

    def _build_dxdt(self):
        numba_precision = from_dtype(self.precision)
        constants = self.indices.constant_values.astype(self.precision)
        dxdt_factory = self.gen_file.get_dxdt_fac(self.equations,
                                                  self.indices)
        dxdt = dxdt_factory(constants, numba_precision)
        return dxdt

    def _build_vjp(self):
        if self.user_vjp:
            if callable(self.user_vjp):
                try:
                    return cuda.jit()(self.user_vjp)
                except Exception as e:
                    print(f"Error compiling user-provided Jacobian: {e}")
            else:
                raise ValueError(
                    "user_vjp must be a callable or a Numba Dispatcher object"
                )
        else:
            numba_precision = from_dtype(self.precision)
            constants = self.indices.constant_values.values().astype(
                self.precision
            )
            vjp_factory = self.gen_file.get_vjp_fac(
                self.equations, self.indices
            )
            vjp = vjp_factory(constants, numba_precision)
            return vjp

    def _build_jvp(self):
        if self.user_jvp:
            if callable(self.user_jvp):
                try:
                    return cuda.jit()(self.user_jvp)
                except Exception as e:
                    print(f"Error compiling user-provided Jacobian: {e}")
            else:
                raise ValueError("user_jvp must be a callable or a "
                                 "Numba Dispatcher object")
        else:

            numba_precision = from_dtype(self.precision)
            constants = self.indices.constant_values.values(
                                                    ).astype(self.precision)
            jvp_factory = self.gen_file.get_jvp_fac(self.equations,
                                                      self.indices)
            jvp = jvp_factory(constants, numba_precision)
            return jvp

    @property
    def dxdt(self):
        return self.get_cached_output("dxdt")

    @property
    def jvp(self):
        return self.get_cached_output("jvp")
