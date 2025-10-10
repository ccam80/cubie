"""Symbolic ODE system built from :mod:`sympy` expressions."""

from typing import Any, Callable, Iterable, Optional, Set, Union

import numpy as np
import sympy as sp
from cubie.integrators.array_interpolator import ArrayInterpolator
from cubie.odesystems.symbolic.dxdt import (
    generate_dxdt_fac_code,
    generate_observables_fac_code,
)
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
from cubie._utils import PrecisionDType


def create_ODE_system(
    dxdt: Union[str, Iterable[str]],
    states: Optional[Union[dict[str, float], Iterable[str]]] = None,
    observables: Optional[Iterable[str]] = None,
    parameters: Optional[Union[dict[str, float], Iterable[str]]] = None,
    constants: Optional[Union[dict[str, float], Iterable[str]]] = None,
    drivers: Optional[Union[Iterable[str], dict[str, Any]]] = None,
    user_functions: Optional[dict[str, Callable]] = None,
    name: Optional[str] = None,
    precision: PrecisionDType = np.float32,
    strict: bool = False,
) -> "SymbolicODE":
    """Create a :class:`SymbolicODE` from SymPy definitions.

    Parameters
    ----------
    dxdt
        System equations defined as either a single string or an iterable of
        equation strings in ``lhs = rhs`` form.
    states
        State labels either as an iterable or as a mapping to default initial
        values.
    observables
        Observable variable labels to expose from the generated system.
    parameters
        Parameter labels either as an iterable or as a mapping to default
        values.
    constants
        Constant labels either as an iterable or as a mapping to default
        values.
    drivers
        External driver variable labels required at runtime. Accepts either
        an iterable of driver symbol names or a dictionary mapping driver
        names to default values or driver-array samples and configuration
        entries.
    user_functions
        Custom callables referenced within ``dxdt`` expressions.
    name
        Identifier used for generated files. Defaults to the hash of the system
        definition.
    precision
        Target floating-point precision used when compiling the system.
    strict
        When ``True`` require every symbol to be explicitly categorised.

    Returns
    -------
    SymbolicODE
        Fully constructed symbolic system ready for compilation.
    """
    symbolic_ode = SymbolicODE.create(
        dxdt=dxdt,
        states=states,
        observables=observables,
        parameters=parameters,
        constants=constants,
        drivers=drivers,
        user_functions=user_functions,
        name=name,
        precision=precision,
        strict=strict,
    )
    return symbolic_ode

class SymbolicODE(BaseODE):
    """Symbolic representation of an ODE system.

    Parameters are provided as SymPy symbols and the differential equations are
    supplied as ``(lhs, rhs)`` tuples where the left-hand side is a derivative
    or observable symbol. Right-hand sides combine states, parameters,
    constants, and intermediate observables.

    Parameters
    ----------
    equations
        Ordered symbolic equations describing the system dynamics.
    all_indexed_bases
        Indexed base collections providing access to state, parameter,
        constant, and observable metadata.
    all_symbols
        Mapping from symbol names to their :class:`sympy.Symbol` instances.
    precision
        Target floating-point precision used for generated kernels.
    fn_hash
        Precomputed system hash. When omitted it is derived from the equations
        and constants.
    user_functions
        Runtime callables referenced within the symbolic expressions.
    name
        Identifier used for generated modules.
    """

    def __init__(
        self,
        equations: Iterable[tuple[sp.Symbol, sp.Expr]],
        all_indexed_bases: IndexedBases,
        all_symbols: Optional[dict[str, sp.Symbol]] = None,
        precision: PrecisionDType = np.float64,
        fn_hash: Optional[int] = None,
        user_functions: Optional[dict[str, Callable]] = None,
        name: Optional[str] = None,
    ):
        """Initialise the symbolic system instance.

        Parameters
        ----------
        equations
            Ordered symbolic equations describing the system dynamics.
        all_indexed_bases
            Indexed base collections providing access to state, parameter,
            constant, and observable metadata.
        all_symbols
            Mapping from symbol names to their :class:`sympy.Symbol` instances.
        precision
            Target floating-point precision used for generated kernels.
        fn_hash
            Precomputed system hash. When omitted it is derived from the
            equations and constants.
        user_functions
            Runtime callables referenced within the symbolic expressions.
        name
            Identifier used for generated modules.

        Returns
        -------
        None
            ``None``.
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
        self.user_functions = user_functions
        self.driver_defaults = all_indexed_bases.drivers.default_values

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
    def create(
        cls,
        dxdt: Union[str, Iterable[str]],
        states: Optional[Union[dict[str, float], Iterable[str]]] = None,
        observables: Optional[Iterable[str]] = None,
        parameters: Optional[Union[dict[str, float], Iterable[str]]] = None,
        constants: Optional[Union[dict[str, float], Iterable[str]]] = None,
        drivers: Optional[Union[Iterable[str], dict[str, Any]]] = None,
        user_functions: Optional[dict[str, Callable]] = None,
        name: Optional[str] = None,
        precision: PrecisionDType = np.float32,
        strict: bool = False,
    ) -> "SymbolicODE":
        """Parse user inputs and instantiate a :class:`SymbolicODE`.

        Parameters
        ----------
        dxdt
            System equations defined as either a single string or an iterable
            of equation strings in ``lhs = rhs`` form.
        states
            State labels either as an iterable or as a mapping to default
            initial values.
        observables
            Observable variable labels to expose from the generated system.
        parameters
            Parameter labels either as an iterable or as a mapping to default
            values.
        constants
            Constant labels either as an iterable or as a mapping to default
            values.
        drivers
            External driver variable labels required at runtime. May be an
            iterable of driver labels or a dictionary describing driver
            defaults or driver-array samples alongside configuration entries.
        user_functions
            Custom callables referenced within ``dxdt`` expressions.
        name
            Identifier used for generated files. Defaults to the hash of the
            system definition.
        precision
            Target floating-point precision used when compiling the system.
        strict
            When ``True`` require every symbol to be explicitly categorised.

        Returns
        -------
        SymbolicODE
            Fully constructed symbolic system ready for compilation.
        """

        if isinstance(drivers, dict) and (
            "time" in drivers or "dt" in drivers
        ):
            ArrayInterpolator(precision=precision, drivers_dict=drivers)

        sys_components = parse_input(
            states=states,
            observables=observables,
            parameters=parameters,
            constants=constants,
            drivers=drivers,
            user_functions=user_functions,
            dxdt=dxdt,
            strict=strict,
        )
        index_map, all_symbols, functions, equations, fn_hash = sys_components
        return cls(equations=equations,
                   all_indexed_bases=index_map,
                   all_symbols=all_symbols,
                   name=name,
                   fn_hash=int(fn_hash),
                   user_functions = functions,
                   precision=precision)


    def build(self) -> ODECache:
        """Compile the ``dxdt`` factory and refresh the cache.

        Returns
        -------
        ODECache
            Cache populated with the compiled ``dxdt`` callable.
        """
        numba_precision = self.numba_precision
        constants = self.constants.values_dict
        new_hash = hash_system_definition(
            self.equations, self.indices.constants.default_values
        )
        if new_hash != self.fn_hash:
            self.gen_file = ODEFile(self.name, new_hash)
            self.fn_hash = new_hash

        dxdt_code = generate_dxdt_fac_code(
            self.equations, self.indices, "dxdt_factory"
        )
        dxdt_factory = self.gen_file.import_function("dxdt_factory", dxdt_code)
        dxdt_func = dxdt_factory(constants, numba_precision)

        observables_code = generate_observables_fac_code(
            self.equations, self.indices, func_name="observables_factory"
        )
        observables_factory = self.gen_file.import_function(
            "observables_factory", observables_code
        )
        observables_func = observables_factory(constants, numba_precision)

        self._cache = ODECache(dxdt=dxdt_func, observables=observables_func)
        self._cache_valid = True
        self._cache_valid = False
        return self._cache


    def set_constants(
        self,
        updates_dict: Optional[dict[str, float]] = None,
        silent: bool = False,
        **kwargs: float,
    ) -> Set[str]:
        """Update constant values in-place.

        Parameters
        ----------
        updates_dict
            Mapping from constant names to replacement values.
        silent
            When ``True`` suppress warnings for unknown labels.
        **kwargs
            Additional constant overrides supplied as keyword arguments.

        Returns
        -------
        set[str]
            Labels that were recognised and updated.

        Notes
        -----
        Constants are first updated in the indexed base map before delegating
        to :meth:`BaseODE.set_constants` for cache management.
        """
        self.indices.update_constants(updates_dict, **kwargs)
        recognized = super().set_constants(updates_dict,
                                 silent=silent)
        return recognized

    def get_solver_helper(
        self,
        func_type: str,
        beta: float = 1.0,
        gamma: float = 1.0,
        preconditioner_order: int = 2,
        mass: Optional[Union[np.ndarray, sp.Matrix]] = None,
    ) -> Callable:
        """Return a generated solver helper device function.

        Solvers use a linear operator, preconditioner, and residual function.
        The operator is parameterised as
        ``(beta * M + a_ij * h * gamma * J)(v)`` where ``M`` is an optional
        mass matrix, ``J`` is the Jacobian, and ``a_ij`` and ``h`` are supplied
        at runtime. Preconditioners and residual functions use subsets of these
        parameters.

        Parameters
        ----------
        func_type
            Helper identifier. Supported values are ``"linear_operator"``,
            ``"neumann_preconditioner"``, ``"end_residual"``, and
            ``"stage_residual"``.
        beta
            Shift parameter for the linear operator.
        gamma
            Weight applied to the Jacobian term in the linear operator.
        preconditioner_order
            Polynomial order of the Neumann preconditioner.
        mass
            Mass matrix applied by the linear operator. When omitted the
            identity matrix is assumed.

        Returns
        -------
        Callable
            CUDA device function implementing the requested helper.

        Raises
        ------
        NotImplementedError
            Raised when ``func_type`` does not correspond to a supported
            helper.
        """
        try:
            return self.get_cached_output(func_type)
        except NotImplementedError:
            pass

        numba_precision = self.numba_precision
        constants = self.constants.values_dict

        factory_kwargs = {
            "constants": constants,
            "precision": numba_precision,
        }

        if func_type == "linear_operator":
            code = generate_operator_apply_code(
                self.equations,
                self.indices,
                M=mass,
                func_name=func_type,
            )
            factory_kwargs.update(
                beta=beta,
                gamma=gamma,
                order=preconditioner_order,
            )
        elif func_type == "neumann_preconditioner":
            code = generate_neumann_preconditioner_code(
                self.equations,
                self.indices,
                func_type,
            )
            factory_kwargs.update(
                beta=beta,
                gamma=gamma,
                order=preconditioner_order,
            )
        elif func_type == "end_residual":
            code = generate_residual_end_state_code(
                self.equations,
                self.indices,
                M=mass,
                func_name=func_type,
            )
            factory_kwargs.update(
                beta=beta,
                gamma=gamma,
                order=preconditioner_order,
            )
        elif func_type == "stage_residual":
            code = generate_stage_residual_code(
                self.equations,
                self.indices,
                M=mass,
                func_name="stage_residual",
            )
            factory_kwargs.update(
                beta=beta,
                gamma=gamma,
                order=preconditioner_order,
            )
        else:
            raise NotImplementedError(
                    f"Solver helper '{func_type}' is not implemented."
            )

        factory = self.gen_file.import_function(func_type, code)
        func = factory(**factory_kwargs)
        setattr(self._cache, func_type, func)
        return func
