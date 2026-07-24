"""Base classes for defining and compiling CUDA-backed ODE systems.

Published Classes
-----------------
:class:`ODECache`
    Attrs container caching compiled CUDA device functions for an ODE
    system (``dxdt``, linear operator, preconditioner, etc.).

:class:`BaseODE`
    Abstract factory base for CUDA-backed ODE systems. Manages value
    containers, precision selection, and cache invalidation.

    >>> from numpy import float32
    >>> # Subclass and override build() to use:
    >>> class MyODE(BaseODE):
    ...     def build(self):
    ...         pass  # compile dxdt here
    >>> ode = MyODE(
    ...     precision=float32,
    ...     default_initial_values={"x": 0.0},
    ...     default_parameters={"k": 1.0},
    ... )
    >>> ode.num_states
    1

See Also
--------
:class:`~cubie.CUDAFactory.CUDAFactory`
    Parent factory class providing compilation and caching.
:class:`~cubie.odesystems.ODEData.ODEData`
    Compile settings container owned by ``BaseODE``.
:class:`~cubie.odesystems.symbolic.symbolicODE.SymbolicODE`
    Concrete subclass that generates device functions from SymPy
    expressions.
"""

from abc import abstractmethod
from functools import partial
from typing import Any, Callable, Dict, Optional, Set, Union

from attrs import define, field
from numpy import float32

from cubie._serialize import canonical_digest
from cubie.CUDAFactory import CUDAFactory, CUDADispatcherCache
from cubie._utils import PrecisionDType
from cubie.odesystems.ODEData import ODEData
from cubie.odesystems.SystemValues import SystemValues


@define
class ODECache(CUDADispatcherCache):
    """Cache compiled CUDA device and support functions for an ODE system.

    Attributes default to ``-1`` when the corresponding function is not built.
    """

    dxdt: Optional[Callable] = field()
    linear_operator: Optional[Union[Callable, int]] = field(default=-1)
    linear_operator_cached: Optional[Union[Callable, int]] = field(default=-1)
    neumann_preconditioner: Optional[Union[Callable, int]] = field(default=-1)
    neumann_preconditioner_cached: Optional[Union[Callable, int]] = field(
        default=-1
    )
    jacobi_preconditioner: Optional[Union[Callable, int]] = field(default=-1)
    jacobi_preconditioner_cached: Optional[Union[Callable, int]] = field(
        default=-1
    )
    stage_residual: Optional[Union[Callable, int]] = field(default=-1)
    n_stage_residual: Optional[Union[Callable, int]] = field(default=-1)
    n_stage_linear_operator: Optional[Union[Callable, int]] = field(default=-1)
    n_stage_neumann_preconditioner: Optional[Union[Callable, int]] = field(
        default=-1
    )
    n_stage_jacobi_preconditioner: Optional[Union[Callable, int]] = field(
        default=-1
    )
    observables: Optional[Union[Callable, int]] = field(default=-1)
    prepare_jac: Optional[Union[Callable, int]] = field(default=-1)
    calculate_cached_jvp: Optional[Union[Callable, int]] = field(default=-1)
    time_derivative_rhs: Optional[Union[Callable, int]] = field(default=-1)
    cached_aux_count: Optional[int] = field(default=-1)


class BaseODE(CUDAFactory):
    """Abstract base for CUDA-backed ordinary differential equation systems.

    Subclasses override :meth:`build` to compile a CUDA device function that
    advances the system state and, optionally, provide analytic helpers via
    :meth:`get_solver_helper`. The base class handles value management,
    precision selection, and caching through :class:`CUDAFactory`.

    Notes
    -----
    Only functions cached during :meth:`build` (typically ``dxdt``) are
    available on this base class. Solver helper functions such as the linear
    operator or preconditioner are generated only by subclasses like
    :class:`SymbolicODE`.
    """

    def __init__(
        self,
        precision: PrecisionDType = float32,
        initial_values: Optional[Dict[str, float]] = None,
        parameters: Optional[Dict[str, float]] = None,
        constants: Optional[Dict[str, float]] = None,
        observables: Optional[Dict[str, float]] = None,
        default_initial_values: Optional[Dict[str, float]] = None,
        default_parameters: Optional[Dict[str, float]] = None,
        default_constants: Optional[Dict[str, float]] = None,
        default_observable_names: Optional[Dict[str, float]] = None,
        num_drivers: int = 1,
        name: Optional[str] = None,
        mass: Any = None,
    ) -> None:
        """Initialize the ODE system.

        Parameters
        ----------
        initial_values
            Initial values for state variables.
        parameters
            Parameter values for the system.
        constants
            Constants that are not expected to change between simulations.
        observables
            Observable values to track.
        default_initial_values
            Default initial values if ``initial_values`` omits entries.
        default_parameters
            Default parameter values if ``parameters`` omits entries.
        default_constants
            Default constant values if ``constants`` omits entries.
        default_observable_names
            Default observable names if ``observables`` omits entries.
        precision
            Precision factory used for calculations. Defaults to
            :class:`numpy.float32`.
        num_drivers
            Number of driver or forcing functions. Defaults to ``1``.
        name
            Printable identifier for the system. Defaults to ``None``.
        mass
            Solver mass matrix; ``None`` implies identity. Singular
            diagonal matrices express semi-explicit DAE systems.
        """
        super().__init__()
        system_data = ODEData.from_BaseODE_initargs(
            initial_values=initial_values,
            parameters=parameters,
            constants=constants,
            observables=observables,
            default_initial_values=default_initial_values,
            default_parameters=default_parameters,
            default_constants=default_constants,
            default_observable_names=default_observable_names,
            precision=precision,
            num_drivers=num_drivers,
            mass=mass,
        )
        self.setup_compile_settings(system_data)
        self.name = name

    @property
    def mass(self) -> Any:
        """Return the system's mass matrix.

        ``None`` implies identity. The matrix is part of the system
        definition, fixed at construction: structural simplification
        supplies a singular diagonal matrix for systems with torn
        algebraic residual equations, and hand-formulated
        semi-explicit DAEs supply theirs through the ``mass``
        constructor argument. Systems with a mass matrix require an
        implicit algorithm.
        """

        return self.compile_settings.mass

    def __repr__(self) -> str:
        if self.name is None:
            name = "ODE System"
        else:
            name = self.name
        return (
            f"{name}"
            "--"
            f"\n{self.states},"
            f"\n{self.parameters},"
            f"\n{self.constants},"
            f"\n{self.observables},"
            f"\n{self.num_drivers})"
        )

    @abstractmethod
    def build(self) -> ODECache:
        """Compile the ``dxdt`` system as a CUDA device function.

        Returns
        -------
        ODECache
            Cache containing the built ``dxdt`` function. Subclasses may add
            further solver helpers to this cache as needed.

        Notes
        -----
        Bring constants into local (outer) scope before defining ``dxdt``
        because CUDA device functions cannot reference ``self``.
        """
        # return ODECache(dxdt=dxdt)

    def update(
        self,
        updates_dict: Optional[Dict[str, float]],
        silent: bool = False,
        **kwargs: float,
    ) -> Set[str]:
        """Update compile settings through the :class:`CUDAFactory` interface.

        Pass updates through the compile-settings interface, which invalidates
        caches when an update succeeds.

        Parameters
        ----------
        updates_dict
            Dictionary of updates to apply.
        silent
            Set to ``True`` to suppress warnings about missing keys.
        **kwargs
            Additional updates specified as keyword arguments.

        Returns
        -------
        set of str
            Labels that were recognized and updated.

        Notes
        -----
        Pass ``silent=True`` when performing bulk updates that may include
        values for other components to suppress warnings about missing keys.
        """

        if updates_dict is None:
            updates_dict = {}
        updates = updates_dict.copy()
        if kwargs:
            updates.update(kwargs)
        if updates == {}:
            return set()

        recognised = self.update_compile_settings(
            updates,
            silent=True,
        )
        recognised_constants = self.set_constants(
            updates,
            silent=True,
        )

        recognised |= recognised_constants

        if not silent:
            unrecognised = set(updates.keys()) - recognised
            if unrecognised:
                raise KeyError(
                    "Unrecognized parameters in update: "
                    f"{unrecognised}. These parameters were not updated.",
                )

        return recognised

    def set_constants(
        self,
        updates_dict: Optional[Dict[str, float]] = None,
        silent: bool = False,
        **kwargs: float,
    ) -> Set[str]:
        """Update constant values in the system.

        Parameters
        ----------
        updates_dict
            Mapping from constant names to their new values.
        silent
            Set to ``True`` to suppress warnings about missing keys.
        **kwargs
            Additional constant updates provided as keyword arguments. These
            override entries in ``updates_dict``.

        Returns
        -------
        set of str
            Labels that were recognized and updated.
        """
        if updates_dict is None:
            updates_dict = {}
        updates_dict = updates_dict.copy()
        if kwargs:
            updates_dict.update(kwargs)
        if updates_dict == {}:
            return set()

        const = self.compile_settings.constants
        recognised = set(updates_dict.keys()) & const.values_dict.keys()
        unrecognised = set(updates_dict.keys()) - recognised
        if recognised:
            # The held container is sealed: modify a copy and pass it
            # through the write boundary.
            new_const = const.copy()
            new_const.update_from_dict(
                {key: updates_dict[key] for key in recognised}
            )
            self.update_compile_settings(constants=new_const, silent=True)

        if not silent and unrecognised:
            raise KeyError(
                f"Unrecognized parameters in update: {unrecognised}. "
                "These parameters were not updated.",
            )

        return recognised

    @property
    def parameters(self) -> "SystemValues":
        """Parameter values configured for the system."""
        return self.compile_settings.parameters

    @property
    def states(self) -> "SystemValues":
        """Initial state values configured for the system."""
        return self.compile_settings.initial_states

    @property
    def initial_values(self) -> "SystemValues":
        """Alias for :attr:`states`."""
        return self.compile_settings.initial_states

    @property
    def observables(self) -> "SystemValues":
        """Observable definitions configured for the system."""
        return self.compile_settings.observables

    @property
    def constants(self) -> "SystemValues":
        """Constant values configured for the system."""
        return self.compile_settings.constants

    @property
    def num_states(self) -> int:
        """Number of state variables."""
        return self.compile_settings.num_states

    @property
    def num_observables(self) -> int:
        """Number of observable variables."""
        return self.compile_settings.num_observables

    @property
    def num_parameters(self) -> int:
        """Number of parameters."""
        return self.compile_settings.num_parameters

    @property
    def num_constants(self) -> int:
        """Number of constants."""
        return self.compile_settings.num_constants

    @property
    def num_drivers(self) -> int:
        """Number of driver variables."""
        return self.compile_settings.num_drivers

    @property
    def sizes(self):
        """System component sizes cached for solvers."""
        return self.compile_settings.sizes

    @property
    def evaluate_f(self):
        """Compiled ``dxdt(state, parameters, drivers, observables, out, t)`` device function."""
        return self.get_cached_output("dxdt")

    @property
    def evaluate_observables(self) -> Callable:
        """Compiled ``get_observables(state, parameters, drivers, observables, t)`` device function."""
        return self.get_cached_output("observables")

    @property
    def _constants_hash(self) -> str:
        """Hash of the current constant values."""
        const_values = tuple()
        if self.constants is not None:
            const_values = tuple(
                (name, float(value))
                for name, value in sorted(
                    self.constants.values_dict.items()
                )
            )
        return canonical_digest(const_values)

    @property
    def config_hash(self):
        """Configuration hash incorporating constant values."""
        return canonical_digest(
            ("cubie-ode-config", super().config_hash, self._constants_hash)
        )

    @property
    def settings_and_constants_hash(self) -> str:
        """Hash of this system's own compile settings and constants.

        Excludes child factories, so a factory owned by the system can
        fold this hash into its own configuration without creating a
        self-referential hash chain.
        """
        return canonical_digest(
            (
                "cubie-ode-settings",
                self.compile_settings.values_hash,
                self._constants_hash,
            )
        )

    def solver_helper_getter(
        self, cache_policy: Optional[Any] = None
    ) -> Callable:
        """Bind a consumer's cache policy as helper-request context.

        Parameters
        ----------
        cache_policy
            The consumer's cache policy, forwarded with every
            request made through the returned callable. Service
            context only — it never enters any identity.

        Returns
        -------
        Callable
            A callable with the :meth:`get_solver_helper` contract.

        Notes
        -----
        Consumers that own a cache policy (e.g. a batch solver
        kernel) each hold one getter, so no policy state is ever
        written onto a shared system.
        """
        return partial(self.get_solver_helper, cache_policy=cache_policy)

    def get_solver_helper(
        self,
        func_name: str,
        solver_beta: Optional[float] = None,
        solver_gamma: Optional[float] = None,
        preconditioner_order: Optional[int] = None,
        cache_policy: Optional[Any] = None,
    ) -> Callable:
        """Retrieve a cached solver helper function.

        Helpers that consume a mass matrix read the system's own
        :attr:`mass`; the matrix is part of the system definition,
        not an algorithm parameter.

        Parameters
        ----------
        func_name
            Identifier for the helper function.
        solver_beta
            Shift parameter for the linear operator. Ignored by this
            base implementation.
        solver_gamma
            Weight of the Jacobian term in the linear operator. Ignored
            by this base implementation.
        preconditioner_order
            Polynomial order of the preconditioner. Ignored by this
            base implementation.
        cache_policy
            The requesting consumer's cache policy, forwarded to
            diagnostic services run on its behalf. Ignored here.

        Returns
        -------
        Callable
            Cached device function corresponding to ``func_name``.

        Notes
        -----
        Returns ``NotImplementedError`` if the ``ODESystem`` lacks generated
        code for the requested helper.
        """
        return self.get_cached_output(func_name)
