"""Symbolic ODE system built from :mod:`sympy` expressions.
"""

from typing import Callable, Iterable, Optional, Set, Union

import numpy as np
import sympy as sp
from cubie.odesystems.symbolic.dxdt import generate_dxdt_fac_code
from cubie.odesystems.symbolic.odefile import ODEFile
from cubie.odesystems.symbolic.solver_helpers import (
    generate_neumann_preconditioner_code,
    generate_operator_apply_code,
    generate_residual_end_state_code,
    generate_stage_residual_code,
)
from cubie.odesystems.symbolic.parser import IndexedBases, parse_input
from cubie.odesystems.symbolic.sym_utils import hash_system_definition
from cubie.odesystems.baseODE import BaseODE, ODECache


def create_ODE_system(
    dxdt: Union[str, Iterable[str]],
    states: Optional[Union[dict, Iterable[str]]] = None,
    observables: Optional[Iterable[str]] = None,
    parameters: Optional[Union[dict, Iterable[str]]] = None,
    constants: Optional[Union[dict, Iterable[str]]] = None,
    drivers: Optional[Iterable[str]] = None,
    user_functions: Optional[dict[str, Callable]] = None,
    name: Optional[str] = None,
    precision: np.dtype = np.float32,
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
                                 precision=precision,
                                 strict=strict)
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
        fn_hash: Optional[int] = None,
        user_functions: Optional[dict[str, Callable]] = None,
        name: str = None,
    ):
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
        self.user_functions = user_functions

        super().__init__(
            initial_values=all_indexed_bases.state_values,
            parameters=all_indexed_bases.parameter_values,
            constants=all_indexed_bases.constant_values,
            observables=all_indexed_bases.observable_names,
            precision=precision,
            num_drivers=ndriv,
            name=name
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
              precision: np.dtype = np.float32,
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
                   fn_hash=int(fn_hash),
                   user_functions = functions,
                   precision=precision)


    def build(self):
        """Compile the ``dxdt`` function and populate the cache."""
        numba_precision = self.numba_precision
        constants = self.constants.values_dict
        new_hash = hash_system_definition(
            self.equations, self.indices.constants.default_values
        )
        if new_hash != self.fn_hash:
            self.gen_file = ODEFile(self.name, new_hash)
            self.fn_hash = new_hash

        code = generate_dxdt_fac_code(
            self.equations, self.indices, "dxdt_factory"
        )
        factory = self.gen_file.import_function("dxdt_factory", code)
        dxdt_func = factory(constants, numba_precision)
        self._cache = ODECache(dxdt=dxdt_func)
        self._cache_valid = True
        self._cache_valid = False
        return self._cache


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

    def get_solver_helper(self,
                          func_type: str,
                          beta: float = 1.0,
                          gamma: float = 1.0,
                          preconditioner_order: int = 1,
                          mass: Optional[Union[np.ndarray, sp.Matrix]] = None):
        """Return a generated solver helper device function.

        Solvers use a linear operator, preconditioner, and residual function.
        The operator which they apply is parameterised to be general to a
        variety of numerical integration algorithms. The operator is of the
        form (beta*M + a_ij*h*gamma*J)(v). M, beta and gamma are
        configurable in the factory; a_ij and h are set at runtime.
        Preconditioners and residual functions use a subset of these
        parameters.

        Parameters
        ----------
        func_type : str
            Identifier for the requested helper. Accepted keys are
            ``"linear_operator"``, ``"neumann_preconditioner"``,
            ``"end_residual"``, and ``"stage_residual"``.
        beta : float, optional
            Shift parameter for the linear operator. Defaults to 1.0
        gamma : float, optional
            Weight of the Jacobian term in the linear operator. Defaults
            to 1.0
        preconditioner_order : int, optional
            Polynomial order of the preconditioner. Defaults to 1. Unused in
            generating the linear operator
        mass : np.ndarray or sp.Matrix, optional
            Mass matrix for the linear operator. Defaults to Identity

        Returns
        -------
        callable
            CUDA device function corresponding to ``func_name``.

        Raises
        ------
        KeyError
            If ``func_name`` is not recognised.
        """
        try:
            return self.get_cached_output(func_type)
        except NotImplementedError:
            pass

        numba_precision = self.numba_precision
        constants = self.constants.values_dict

        if func_type == "linear_operator":
            code = generate_operator_apply_code(
                self.equations,
                self.indices,
                M=mass,
                func_name=func_type,
            )
        elif func_type == "neumann_preconditioner":
            code = generate_neumann_preconditioner_code(
                self.equations,
                self.indices,
                func_type,
            )
        elif func_type == "end_residual":
            code = generate_residual_end_state_code(
                self.equations,
                self.indices,
                M=mass,
                func_name=func_type,
            )
        elif func_type == "stage_residual":
            code = generate_stage_residual_code(
                self.equations,
                self.indices,
                M=mass,
                func_name="stage_residual",
            )
        else:
            raise NotImplementedError(
                    f"Solver helper '{func_type}' is not implemented."
            )

        factory = self.gen_file.import_function(func_type, code)
        func = factory(
                constants=constants,
                precision=numba_precision,
                beta = beta,
                gamma=gamma,
                order=preconditioner_order
        )
        setattr(self._cache, func_type, func)
        return func
