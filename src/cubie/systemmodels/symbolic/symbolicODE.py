"""Symbolic ODE system built from :mod:`sympy` expressions.

This module extends the basic symbolic prototype by adding support for a
restricted set of mathematical operators, automatic Jacobian generation and a
Numba CUDA implementation.  The implementation of placeholder math functions
and Jacobian generation borrows ideas from
`chaste-codegen <https://github.com/ModellingWebLab/chaste-codegen>`_.

The code copied from *chaste-codegen* is licensed under the MIT licence and has
been adapted for use in this project.
"""

import warnings
from typing import Callable, Iterable, Optional, Set, Union

import numpy as np
import sympy as sp
from numba import cuda, from_dtype

from cubie._utils import is_devfunc
from cubie.systemmodels.symbolic.odefile import ODEFile
from cubie.systemmodels.symbolic.parser import IndexedBases, parse_input
from cubie.systemmodels.symbolic.sym_utils import hash_system_definition
from cubie.systemmodels.systems.baseODE import BaseODE, ODECache


def create_ODE_system(
    dxdt: Union[str, Iterable[str]],
    states: Optional[Union[dict, Iterable[str]]] = None,
    observables: Optional[Iterable[str]] = None,
    parameters: Optional[Union[dict, Iterable[str]]] = None,
    constants: Optional[Union[dict, Iterable[str]]] = None,
    drivers: Optional[Iterable[str]] = None,
    user_functions: Optional[dict[str, Callable]] = None,
    name: Optional[str] = None,
    strict: bool = False,
):
    """Create an ODE system from SymPy expressions."""
    SymbODE = SymbolicODE.create(dxdt=dxdt,
                                 states=states,
                                 observables=observables,
                                 parameters=parameters,
                                 constants=constants,
                                 drivers=drivers,
                                 user_functions=user_functions,
                                 name=name,
                                 strict=strict,)
    return SymbODE

class SymbolicODE(BaseODE):
    """Create an ODE system from SymPy expressions.

    Parameters are provided as SymPy symbols.  The differential equations are
    provided as a list of (lhs, rhs) tuples objects where the left hand side
    is a differential or auxiliary/observable symbol and the right hand side
    is an expression composed of states, parameters, constants and previously
    defined auxiliaries/observables.
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
        autojvp: bool = True,
        autovjp: bool = True,
        user_functions: Optional[dict[str, Callable]] = None,
        name: str = None,
    ):
        """autojvp: bool
                Automatically generate the Jacobian-vector product function on
                build
            autovjp: bool
                Automatically generate the vector-Jacobian product function on
                build
        """
        if all_symbols is None:
            all_symbols = all_indexed_bases.all_symbols
        self.all_symbols = all_symbols

        if fn_hash is None:
            dxdt_str = [f"{lhs}={str(rhs)}" for lhs, rhs
                        in equations]
            constants = all_indexed_bases.constants.default_values
            fn_hash = hash_system_definition(dxdt_str, constants)
        if name is None:
            name = fn_hash

        self.name = name
        self.gen_file = ODEFile(name, fn_hash)

        ndriv = all_indexed_bases.drivers.length
        self.equations = equations
        self.indices = all_indexed_bases
        self.fn_hash = fn_hash
        self.user_jvp = jvp
        self.user_vjp = vjp
        self.autojvp = autojvp
        self.autovjp = autovjp
        self.user_functions = user_functions

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
              states: Optional[Union[dict,Iterable[str]]] = None,
              observables: Optional[Iterable[str]] = None,
              parameters: Optional[Union[dict,Iterable[str]]] = None,
              constants: Optional[Union[dict,Iterable[str]]] = None,
              drivers: Optional[Iterable[str]] = None,
              user_functions: Optional[Optional[dict[str, Callable]]] = None,
              name: Optional[str] = None,
              strict=False):

        sys_components = parse_input(
                states = states,
                observables = observables,
                parameters = parameters,
                constants = constants,
                drivers = drivers,
                user_functions=user_functions,
                dxdt = dxdt,
                strict=strict
        )
        index_map, all_symbols, functions, equations, fn_hash = sys_components
        return cls(equations=equations,
                   all_indexed_bases=index_map,
                   all_symbols=all_symbols,
                   name=name,
                   fn_hash=fn_hash,
                   user_functions = functions,
                   precision=np.float64)


    def build(self):
        """Compile the dxdt and jvp/vjp."""
        numba_precision = from_dtype(self.precision)
        constants = self.constants.values_array
        new_hash = hash_system_definition(
            self.equations, self.indices.constants.default_values)
        if new_hash != self.fn_hash:
            self.gen_file = ODEFile(self.name, new_hash)


        if self.autojvp:
            jvp = self._build_jvp(numba_precision, constants)
        else:
            jvp = -1
        if self.autovjp:
            vjp = self._build_vjp(numba_precision, constants)
        else:
            vjp = -1
        dxdt_func = self._build_dxdt(numba_precision, constants)

        return ODECache(dxdt = dxdt_func,
                        jvp = jvp,
                        vjp = vjp)

    def _build_dxdt(self, numba_precision, constants):
        dxdt_factory = self.gen_file.get_dxdt_fac(self.equations,
                                                  self.indices)
        dxdt = dxdt_factory(constants, numba_precision)
        return dxdt

    def _build_jac_product(self,
                           numba_precision,
                           constants,
                           direction = "jvp"):
        """Compile (if not provided) the Jacobian-product device function.

        Parameters
        ----------
        numba-precision : numba type
            the desired floating-point type to compile for
        constants : numpy array
            constant values to be "baked in" to the compiled function
        direction : "jvp" or "vjp"
            Which product to produce: jacobian-vector or vector-jacobian
            product.

        Returns
        -------
        device function:
            The compiled Jacobian-product device function.

        Notes
        -----
        Does its best to use the user-provided function - if it's a device
        function, return it straight away.  If it's a jit-compilable function,
        jit-compile and return. Otherwise, fall back to generating the function
        from the provided dxdt equations.
        """
        if direction == "jvp":
            userfunc = self.user_jvp
            fac_factory = self.gen_file.get_jvp_fac
        elif direction == "vjp":
            userfunc = self.user_vjp
            fac_factory = self.gen_file.get_vjp_fac
        else:
            raise ValueError(f"Invalid direction: {direction}")

        # Use user-provided function if we can
        if userfunc:
            if callable(userfunc):
                if is_devfunc(userfunc):
                    return userfunc
                else:
                    try:
                        return cuda.jit()(userfunc)
                    except Exception as e:
                        warnings.warn(f"Error compiling user-provided "
                                      f"{direction} function: {e}, falling "
                                      f"back to automatic generation.")
            else:
                warnings.warn("user_vjp must be a cuda-compilable function "
                              "or a cuda device function, got type "
                              f"{type(self.user_vjp)}. Falling back to "
                              f"automatic generation."
                )

        # Generate one if not
        jac_fac = fac_factory(self.equations, self.indices)
        jac_product = jac_fac(constants, numba_precision)
        return jac_product

    def _build_vjp(self, numba_precision, constants):
        """Compile (if none provided) and return the VJP device function."""
        return self._build_jac_product(numba_precision, constants,
                                       direction="vjp")

    def _build_jvp(self, numba_precision, constants):
        """Compile (if none provided) and return the JVP device function."""

        return self._build_jac_product(numba_precision, constants,
                                       direction="jvp")

    def set_constants(self, updates_dict=None, silent=False, **kwargs
                      ) -> Set[str]:
        """Update the constants of the system.

        Parameters
        ----------
            updates_dict : dict of strings, floats
                A dictionary mapping constant names to their new values.
            silent : bool
                If True, suppress warnings about keys not found, default False.
            **kwargs: key-value pairs
                Additional constant updates in key=value form, overrides
                updates_dict.

        Returns
        -------
        set of str:
            All labels that were recognized (and therefore updated)

        Notes
        -----
        First silently updates the constants in the indexed base map, then
        calls the base ODE class's set constants method.
        """

        self.indices.update_constants(updates_dict, **kwargs)
        recognized = super().set_constants(updates_dict,
                                 silent=silent)
        return recognized
