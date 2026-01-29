"""Symbolic ODE system built from :mod:`sympy` expressions.

Published Classes
-----------------
:class:`SymbolicODE`
    Concrete :class:`~cubie.odesystems.baseODE.BaseODE` subclass that
    generates CUDA device functions from SymPy equations. Handles
    codegen caching, solver helper generation, and constant/parameter
    conversion.

    >>> from cubie.odesystems.symbolic.symbolicODE import (
    ...     create_ODE_system,
    ... )
    >>> ode = create_ODE_system(
    ...     dxdt="dx = -k * x",
    ...     states={"x": 1.0},
    ...     parameters={"k": 0.5},
    ... )
    >>> ode.num_states
    1

Published Functions
-------------------
:func:`create_ODE_system`
    Convenience wrapper around :meth:`SymbolicODE.create`.

    >>> ode = create_ODE_system("dx = -x", states={"x": 1.0})
    >>> ode.num_states
    1

See Also
--------
:class:`~cubie.odesystems.baseODE.BaseODE`
    Abstract parent providing cache management and value containers.
:class:`~cubie.odesystems.symbolic.odefile.ODEFile`
    Disk-backed cache for generated factory functions.
:mod:`cubie.odesystems.symbolic.parsing.parser`
    Parses string or SymPy equations into structured components.
:mod:`cubie.odesystems.symbolic.codegen`
    Code generation modules invoked by :meth:`SymbolicODE.get_solver_helper`.
"""

from typing import (
    Any,
    Callable,
    Iterable,
    Optional,
    Sequence,
    Set,
    Union,
)

from numpy import float32, ndarray
import sympy as sp
from cubie.integrators.array_interpolator import ArrayInterpolator
from cubie.odesystems.symbolic.codegen.dxdt import (
    generate_dxdt_fac_code,
    generate_observables_fac_code,
)
from cubie.odesystems.symbolic.codegen import (
    generate_cached_jvp_code,
    generate_cached_operator_apply_code,
    generate_neumann_preconditioner_cached_code,
    generate_neumann_preconditioner_code,
    generate_n_stage_neumann_preconditioner_code,
    generate_n_stage_linear_operator_code,
    generate_n_stage_residual_code,
    generate_operator_apply_code,
    generate_prepare_jac_code,
    generate_stage_residual_code,
)
from cubie.odesystems.symbolic.codegen.jacobian import generate_analytical_jvp
from cubie.odesystems.symbolic.odefile import ODEFile
from cubie.odesystems.symbolic.parsing import (
    IndexedBases,
    JVPEquations,
    ParsedEquations,
    parse_input,
)
from cubie.odesystems.symbolic.codegen.time_derivative import (
    generate_time_derivative_fac_code,
)
from cubie.odesystems.symbolic.sym_utils import hash_system_definition
from cubie.odesystems.baseODE import BaseODE, ODECache
from cubie._utils import PrecisionDType
from cubie.time_logger import default_timelogger

# Helper types that require beta, gamma, and order factory kwargs
_HELPERS_NEEDING_PRECONDITIONER_KWARGS = frozenset((
    "linear_operator",
    "linear_operator_cached",
    "neumann_preconditioner",
    "neumann_preconditioner_cached",
    "stage_residual",
    "n_stage_residual",
    "n_stage_linear_operator",
    "n_stage_neumann_preconditioner",
))


def create_ODE_system(
    dxdt: Union[str, Iterable[str]],
    precision: PrecisionDType = float32,
    states: Optional[Union[dict[str, float], Iterable[str]]] = None,
    observables: Optional[Iterable[str]] = None,
    parameters: Optional[Union[dict[str, float], Iterable[str]]] = None,
    constants: Optional[Union[dict[str, float], Iterable[str]]] = None,
    drivers: Optional[Union[Iterable[str], dict[str, Any]]] = None,
    user_functions: Optional[dict[str, Callable]] = None,
    name: Optional[str] = None,
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
        Parsed equations describing the system dynamics.
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
        equations: ParsedEquations,
        precision: PrecisionDType,
        all_indexed_bases: IndexedBases,
        all_symbols: Optional[dict[str, sp.Symbol]] = None,
        fn_hash: Optional[str] = None,
        user_functions: Optional[dict[str, Callable]] = None,
        name: Optional[str] = None,
    ):
        """Initialise the symbolic system instance.

        Parameters
        ----------
        equations
            Parsed equations describing the system dynamics.
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
        """
        if all_symbols is None:
            all_symbols = all_indexed_bases.all_symbols
        self.all_symbols = all_symbols

        if fn_hash is None:
            constants = all_indexed_bases.constants.default_values
            fn_hash = hash_system_definition(
                equations,
                constants,
                observable_labels=all_indexed_bases.observables.ref_map.keys(),
            )
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
        self.registered_helper_events = set()

        super().__init__(
            initial_values=all_indexed_bases.state_values,
            parameters=all_indexed_bases.parameter_values,
            constants=all_indexed_bases.constant_values,
            observables=all_indexed_bases.observable_names,
            precision=precision,
            num_drivers=ndriv,
            name=name,
        )
        self._jacobian_aux_count: Optional[int] = None
        self._jvp_exprs: Optional[JVPEquations] = None

    @classmethod
    def create(
        cls,
        dxdt: Union[str, Iterable[str]],
        precision: PrecisionDType,
        states: Optional[Union[dict[str, float], Iterable[str]]] = None,
        observables: Optional[Iterable[str]] = None,
        parameters: Optional[Union[dict[str, float], Iterable[str]]] = None,
        constants: Optional[Union[dict[str, float], Iterable[str]]] = None,
        drivers: Optional[Union[Iterable[str], dict[str, Any]]] = None,
        user_functions: Optional[dict[str, Callable]] = None,
        name: Optional[str] = None,
        strict: bool = False,
        state_units: Optional[Union[dict[str, str], Iterable[str]]] = None,
        parameter_units: Optional[Union[dict[str, str], Iterable[str]]] = None,
        constant_units: Optional[Union[dict[str, str], Iterable[str]]] = None,
        observable_units: Optional[
            Union[dict[str, str], Iterable[str]]
        ] = None,
        driver_units: Optional[Union[dict[str, str], Iterable[str]]] = None,
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
        state_units
            Optional units for states. Defaults to "dimensionless".
        parameter_units
            Optional units for parameters. Defaults to "dimensionless".
        constant_units
            Optional units for constants. Defaults to "dimensionless".
        observable_units
            Optional units for observables. Defaults to "dimensionless".
        driver_units
            Optional units for drivers. Defaults to "dimensionless".

        Returns
        -------
        SymbolicODE
            Fully constructed symbolic system ready for compilation.
        """

        if isinstance(drivers, dict) and (
            "time" in drivers or "dt" in drivers
        ):
            ArrayInterpolator(precision=precision, drivers_dict=drivers)

        # Register timing event for parsing (one-time registration)
        default_timelogger.register_event(
            "symbolic_ode_parsing",
            "codegen",
            "Codegen time for symbolic ODE parsing",
        )

        # Start timing for parsing operation
        default_timelogger.start_event("symbolic_ode_parsing")
        sys_components = parse_input(
            states=states,
            observables=observables,
            parameters=parameters,
            constants=constants,
            drivers=drivers,
            user_functions=user_functions,
            dxdt=dxdt,
            strict=strict,
            state_units=state_units,
            parameter_units=parameter_units,
            constant_units=constant_units,
            observable_units=observable_units,
            driver_units=driver_units,
        )
        index_map, all_symbols, functions, equations, fn_hash = sys_components
        symbolic_ode = cls(
            equations=equations,
            all_indexed_bases=index_map,
            all_symbols=all_symbols,
            name=name,
            fn_hash=fn_hash,
            user_functions=functions,
            precision=precision,
        )
        default_timelogger.stop_event("symbolic_ode_parsing")
        return symbolic_ode

    @property
    def jacobian_aux_count(self) -> Optional[int]:
        """Return the number of cached Jacobian auxiliary values."""

        return self._jacobian_aux_count

    @property
    def state_units(self) -> dict[str, str]:
        """Return units for state variables."""
        return self.indices.states.units

    @property
    def parameter_units(self) -> dict[str, str]:
        """Return units for parameters."""
        return self.indices.parameters.units

    @property
    def constant_units(self) -> dict[str, str]:
        """Return units for constants."""
        return self.indices.constants.units

    @property
    def observable_units(self) -> dict[str, str]:
        """Return units for observables."""
        return self.indices.observables.units

    @property
    def driver_units(self) -> dict[str, str]:
        """Return units for drivers."""
        return self.indices.drivers.units

    def _get_jvp_exprs(self) -> JVPEquations:
        """Return cached Jacobian-vector assignments."""

        if self._jvp_exprs is None:
            self._jvp_exprs = generate_analytical_jvp(
                self.equations,
                input_order=self.indices.states.index_map,
                output_order=self.indices.dxdt.index_map,
                observables=self.indices.observable_symbols,
                cse=True,
            )
        return self._jvp_exprs

    def build(self) -> ODECache:
        """Compile the ``dxdt`` factory and refresh the cache.

        Returns
        -------
        ODECache
            Cache populated with the compiled ``dxdt`` callable.
        """
        numba_precision = self.numba_precision
        constants = self.constants.values_dict
        self._jacobian_aux_count = None
        new_hash = hash_system_definition(
            self.equations,
            self.indices.constants.default_values,
            observable_labels=self.indices.observables.ref_map.keys(),
        )
        if new_hash != self.fn_hash:
            self.gen_file = ODEFile(self.name, new_hash)
            self.fn_hash = new_hash

        dxdt_code = None
        if not self.gen_file.function_is_cached("dxdt_factory"):
            dxdt_code = generate_dxdt_fac_code(
                self.equations, self.indices, "dxdt_factory"
            )
        dxdt_factory, _ = self.gen_file.import_function(
            "dxdt_factory", dxdt_code
        )
        dxdt_func = dxdt_factory(constants, numba_precision)

        obs_code = None
        if not self.gen_file.function_is_cached("observables_factory"):
            obs_code = generate_observables_fac_code(
                self.equations, self.indices,
                func_name="observables_factory",
            )
        observables_factory, _ = self.gen_file.import_function(
            "observables_factory", obs_code
        )
        evaluate_observables = observables_factory(constants, numba_precision)

        return ODECache(
            dxdt=dxdt_func,
            observables=evaluate_observables,
        )

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
        recognized = super().set_constants(updates_dict, silent=silent)
        return recognized

    def make_parameter(self, name: str) -> None:
        """Convert a constant to a swept parameter.

        The constant becomes a parameter that can be varied at runtime without
        recompilation. The current value becomes the parameter's default value.

        Parameters
        ----------
        name
            Name of the constant to convert.

        Raises
        ------
        KeyError
            If the name is not found in constants.
        """
        value = self.constants.values_dict.get(name, 0.0)
        self.indices.constant_to_parameter(name)

        self.compile_settings.constants.remove_entry(name)
        self.compile_settings.parameters.add_entry(name, value)

        self._invalidate_cache()

    def make_constant(self, name: str) -> None:
        """Convert a parameter to a compile-time constant.

        The parameter becomes a constant that is embedded into compiled kernels.
        The current value becomes the constant's value.

        Parameters
        ----------
        name
            Name of the parameter to convert.

        Raises
        ------
        KeyError
            If the name is not found in parameters.
        """
        value = self.parameters.values_dict.get(name, 0.0)
        self.indices.parameter_to_constant(name)

        self.compile_settings.parameters.remove_entry(name)
        self.compile_settings.constants.add_entry(name, value)

        self._invalidate_cache()

    def set_constant_value(self, name: str, value: float) -> None:
        """Set the value of a constant.

        Parameters
        ----------
        name
            Name of the constant.
        value
            New value for the constant.

        Raises
        ------
        KeyError
            If the name is not found in constants.
        """
        self.set_constants({name: value})

    def set_parameter_value(self, name: str, value: float) -> None:
        """Set the default value of a parameter.

        Parameters
        ----------
        name
            Name of the parameter.
        value
            New default value for the parameter.

        Raises
        ------
        KeyError
            If the name is not found in parameters.
        """
        self.parameters[name] = value
        self.indices.parameters.update_values({name: value})

    def set_initial_value(self, name: str, value: float) -> None:
        """Set the initial value of a state variable.

        Parameters
        ----------
        name
            Name of the state variable.
        value
            New initial value.

        Raises
        ------
        KeyError
            If the name is not found in states.
        """
        self.initial_values[name] = value
        self.indices.states.update_values({name: value})

    def get_constants_info(self) -> list[dict]:
        """Return information about all constants.

        Returns
        -------
        list of dict
            Each dict contains 'name', 'value', and 'unit' keys.
        """
        result = []
        for name in self.indices.constant_names:
            result.append({
                'name': name,
                'value': self.constants.values_dict.get(name, 0.0),
                'unit': self.constant_units.get(name, 'dimensionless'),
            })
        return result

    def get_parameters_info(self) -> list[dict]:
        """Return information about all parameters.

        Returns
        -------
        list of dict
            Each dict contains 'name', 'value', and 'unit' keys.
        """
        result = []
        for name in self.indices.parameter_names:
            result.append({
                'name': name,
                'value': self.parameters.values_dict.get(name, 0.0),
                'unit': self.parameter_units.get(name, 'dimensionless'),
            })
        return result

    def get_states_info(self) -> list[dict]:
        """Return information about all state variables.

        Returns
        -------
        list of dict
            Each dict contains 'name', 'value', and 'unit' keys.
        """
        result = []
        for name in self.indices.state_names:
            result.append({
                'name': name,
                'value': self.initial_values.values_dict.get(name, 0.0),
                'unit': self.state_units.get(name, 'dimensionless'),
            })
        return result

    def constants_gui(self, blocking: bool = True) -> None:
        """Launch a Qt GUI for editing constants and parameters.

        The GUI displays all constants and parameters with their values and
        units. Users can convert between constants and parameters using a
        checkbox, and edit values directly.

        Parameters
        ----------
        blocking
            If True (default), block until the dialog is closed.
            If False, return immediately with the dialog still open.

        Notes
        -----
        Requires a Qt binding (PyQt6, PyQt5, PySide6, or PySide2).

        Example
        -------
        >>> ode = load_cellml_model("model.cellml")
        >>> ode.constants_gui()  # Opens editor dialog
        """
        from cubie.gui.constants_editor import show_constants_editor
        show_constants_editor(self, blocking=blocking)

    def states_gui(self, blocking: bool = True) -> None:
        """Launch a Qt GUI for editing initial state values.

        The GUI displays all state variables with their initial values and
        units. Users can edit the initial values directly.

        Parameters
        ----------
        blocking
            If True (default), block until the dialog is closed.
            If False, return immediately with the dialog still open.

        Notes
        -----
        Requires a Qt binding (PyQt6, PyQt5, PySide6, or PySide2).

        Example
        -------
        >>> ode = load_cellml_model("model.cellml")
        >>> ode.states_gui()  # Opens editor dialog
        """
        from cubie.gui.states_editor import show_states_editor
        show_states_editor(self, blocking=blocking)

    def get_solver_helper(
        self,
        func_type: str,
        beta: float = 1.0,
        gamma: float = 1.0,
        preconditioner_order: int = 2,
        mass: Optional[Union[ndarray, sp.Matrix]] = None,
        stage_coefficients: Optional[
            Sequence[Sequence[Union[float, sp.Expr]]]
        ] = None,
        stage_nodes: Optional[Sequence[Union[float, sp.Expr]]] = None,
    ) -> Union[Callable, int]:
        """Return a generated solver helper device function.

        Parameters
        ----------
        func_type
            Helper identifier. Supported values are ``"linear_operator"``,
            ``"linear_operator_cached"``, ``"neumann_preconditioner"``,
            ``"neumann_preconditioner_cached"``, ``"stage_residual"``,
            ``"n_stage_residual"``, ``"n_stage_linear_operator"`,
            ``"n_stage_neumann_preconditioner"``, ``"prepare_jac"`,
            ``"cached_aux_count"`` and ``"calculate_cached_jvp"``.
        beta
            Shift parameter for the linear operator.
        gamma
            Weight applied to the Jacobian term in the linear operator.
        preconditioner_order
            Polynomial order of the Neumann preconditioner.
        mass
            Mass matrix applied by the linear operator. When omitted the
            identity matrix is assumed.
        stage_coefficients
            FIRK tableau coefficients used to evaluate stage states. Required
            for flattened helpers.
        stage_nodes
            FIRK stage nodes expressed as timestep fractions. The stage count
            is inferred from ``len(stage_nodes)``.

        Returns
        -------
        Callable or int
            CUDA device function implementing the requested helper or the
            cached auxiliary count for ``"cached_aux_count"``.

        Raises
        ------
        NotImplementedError
            Raised when ``func_type`` does not correspond to a supported
            helper.
        """
        solver_updates = {
            "beta": beta,
            "gamma": gamma,
            "preconditioner_order": preconditioner_order,
            "mass": mass,
        }
        self.update(solver_updates, silent=True)

        # Register timing event for this helper type if not already registered
        event_name = f"solver_helper_{func_type}"

        if event_name not in self.registered_helper_events:
            default_timelogger.register_event(
                event_name,
                "codegen",
                f"Codegen time for solver helper {func_type}",
            )
            self.registered_helper_events.add(event_name)

        try:
            func = self.get_cached_output(func_type)
            return func
        except NotImplementedError:
            pass

        # Determine factory_name for n_stage helpers (needed to check cache)
        if func_type == "n_stage_residual":
            factory_name = f"n_stage_residual_{len(stage_nodes)}"
        elif func_type == "n_stage_linear_operator":
            factory_name = f"n_stage_linear_operator_{len(stage_nodes)}"
        elif func_type == "n_stage_neumann_preconditioner":
            factory_name = f"n_stage_neumann_preconditioner_{len(stage_nodes)}"
        else:
            factory_name = func_type

        # Handle cached_aux_count specially - it doesn't generate code
        # and doesn't depend on whether other functions are cached
        if func_type == "cached_aux_count":
            default_timelogger.start_event(event_name, skipped=True)
            if self._jacobian_aux_count is None:
                self.get_solver_helper("prepare_jac")
            default_timelogger.stop_event(event_name)
            return self._jacobian_aux_count

        # Check if function is already in file cache (skipped if so)
        is_cached = self.gen_file.function_is_cached(factory_name)

        # Start timing for helper generation, marking as skipped if cached
        default_timelogger.start_event(event_name, skipped=is_cached)
        numba_precision = self.numba_precision
        constants = self.constants.values_dict

        factory_kwargs = {
            "constants": constants,
            "precision": numba_precision,
        }

        # Skip expensive code generation when function is already cached
        code = None
        if not is_cached:
            # factory_name already set above based on func_type
            if func_type == "linear_operator":
                code = generate_operator_apply_code(
                    self.equations,
                    self.indices,
                    M=mass,
                    func_name=factory_name,
                    jvp_equations=self._get_jvp_exprs(),
                )
            elif func_type == "linear_operator_cached":
                code = generate_cached_operator_apply_code(
                    self.equations,
                    self.indices,
                    M=mass,
                    func_name=factory_name,
                    jvp_equations=self._get_jvp_exprs(),
                )
            elif func_type == "prepare_jac":
                code, aux_count = generate_prepare_jac_code(
                    self.equations,
                    self.indices,
                    func_name=factory_name,
                    jvp_equations=self._get_jvp_exprs(),
                )
                self._jacobian_aux_count = aux_count
            elif func_type == "calculate_cached_jvp":
                code = generate_cached_jvp_code(
                    self.equations,
                    self.indices,
                    func_name=factory_name,
                    jvp_equations=self._get_jvp_exprs(),
                )
            elif func_type == "neumann_preconditioner":
                code = generate_neumann_preconditioner_code(
                    self.equations,
                    self.indices,
                    factory_name,
                    jvp_equations=self._get_jvp_exprs(),
                )
            elif func_type == "neumann_preconditioner_cached":
                code = generate_neumann_preconditioner_cached_code(
                    self.equations,
                    self.indices,
                    factory_name,
                    jvp_equations=self._get_jvp_exprs(),
                )
            elif func_type == "stage_residual":
                code = generate_stage_residual_code(
                    self.equations,
                    self.indices,
                    M=mass,
                    func_name=factory_name,
                )
            elif func_type == "time_derivative_rhs":
                code = generate_time_derivative_fac_code(
                    self.equations,
                    self.indices,
                    func_name=factory_name,
                )
            elif func_type == "n_stage_residual":
                code = generate_n_stage_residual_code(
                    equations=self.equations,
                    index_map=self.indices,
                    stage_coefficients=stage_coefficients,
                    stage_nodes=stage_nodes,
                    M=mass,
                    func_name=factory_name,
                )
            elif func_type == "n_stage_linear_operator":
                code = generate_n_stage_linear_operator_code(
                    equations=self.equations,
                    index_map=self.indices,
                    stage_coefficients=stage_coefficients,
                    stage_nodes=stage_nodes,
                    M=mass,
                    func_name=factory_name,
                    jvp_equations=self._get_jvp_exprs(),
                )
            elif func_type == "n_stage_neumann_preconditioner":
                code = generate_n_stage_neumann_preconditioner_code(
                    equations=self.equations,
                    index_map=self.indices,
                    stage_coefficients=stage_coefficients,
                    stage_nodes=stage_nodes,
                    func_name=factory_name,
                    jvp_equations=self._get_jvp_exprs(),
                )
            else:
                raise NotImplementedError(
                    f"Solver helper '{func_type}' is not implemented."
                )

        # Set factory_kwargs for types that need preconditioner parameters
        if func_type in _HELPERS_NEEDING_PRECONDITIONER_KWARGS:
            factory_kwargs.update(
                beta=beta,
                gamma=gamma,
                order=preconditioner_order,
            )

        factory, was_cached = self.gen_file.import_function(factory_name, code)

        # For prepare_jac, retrieve aux_count from cached factory if needed
        if func_type == "prepare_jac" and self._jacobian_aux_count is None:
            self._jacobian_aux_count = getattr(factory, 'aux_count', 0)

        func = factory(**factory_kwargs)
        setattr(self._cache, func_type, func)
        default_timelogger.stop_event(event_name)

        return func
