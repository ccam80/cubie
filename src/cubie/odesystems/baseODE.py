"""Base classes for defining and compiling CUDA-backed ODE systems."""

from abc import abstractmethod
from typing import Any, Callable, Dict, Optional, Set, Tuple, Union

import attrs
import numpy as np
from numpy.typing import NDArray

from cubie.CUDAFactory import CUDAFactory
from cubie._utils import PrecisionDType
from cubie.odesystems.ODEData import ODEData


@attrs.define
class ODECache:
    """Cache compiled CUDA device and support functions for an ODE system.

    Attributes default to ``-1`` when the corresponding function is not built.
    """

    dxdt: Optional[Callable] = attrs.field()
    linear_operator: Optional[Union[Callable, int]] = attrs.field(default=-1)
    linear_operator_cached: Optional[Union[Callable, int]] = attrs.field(
        default=-1
    )
    neumann_preconditioner: Optional[Union[Callable, int]] = attrs.field(
        default=-1
    )
    neumann_preconditioner_cached: Optional[Union[Callable, int]] = attrs.field(
        default=-1
    )
    stage_residual: Optional[Union[Callable, int]] = attrs.field(default=-1)
    n_stage_residual: Optional[Union[Callable, int]] = attrs.field(default=-1)
    n_stage_linear_operator: Optional[Union[Callable, int]] = attrs.field(
        default=-1
    )
    n_stage_neumann_preconditioner: Optional[Union[Callable, int]] = attrs.field(
        default=-1
    )
    observables: Optional[Union[Callable, int]] = attrs.field(default=-1)
    prepare_jac: Optional[Union[Callable, int]] = attrs.field(default=-1)
    calculate_cached_jvp: Optional[Union[Callable, int]] = attrs.field(
        default=-1
    )
    time_derivative_rhs: Optional[Union[Callable, int]] = attrs.field(
        default=-1
    )
    cached_aux_count: Optional[int] = attrs.field(default=-1)


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
        initial_values: Optional[Dict[str, float]] = None,
        parameters: Optional[Dict[str, float]] = None,
        constants: Optional[Dict[str, float]] = None,
        observables: Optional[Dict[str, float]] = None,
        default_initial_values: Optional[Dict[str, float]] = None,
        default_parameters: Optional[Dict[str, float]] = None,
        default_constants: Optional[Dict[str, float]] = None,
        default_observable_names: Optional[Dict[str, float]] = None,
        precision: PrecisionDType = np.float64,
        num_drivers: int = 1,
        name: Optional[str] = None,
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
            :class:`numpy.float64`.
        num_drivers
            Number of driver or forcing functions. Defaults to ``1``.
        name
            Printable identifier for the system. Defaults to ``None``.
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
        )
        self.setup_compile_settings(system_data)
        self.name = name

    def __repr__(self) -> str:
        if self.name is None:
            name = "ODE System"
        else:
            name = self.name
        return (f"{self.name}"
                "--"
                f"\n{self.states},"
                f"\n{self.parameters},"
                f"\n{self.constants},"
                f"\n{self.observables},"
                f"\n{self.num_drivers})")


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

    def correct_answer_python(
        self,
        states: NDArray[np.floating[Any]],
        parameters: NDArray[np.floating[Any]],
        drivers: NDArray[np.floating[Any]],
    ) -> Tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        """Python reference ``dxdt`` for testing.

        Parameters
        ----------
        states
            Current state values.
        parameters
            Parameter values.
        drivers
            Driver or forcing values.

        Returns
        -------
        tuple of numpy.ndarray
            Tuple containing the state derivatives and observable outputs.
        """
        return np.asarray([0]), np.asarray([0])

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
        recognised = const.update_from_dict(updates_dict, silent=True)
        unrecognised = set(updates_dict.keys()) - recognised
        self.update_compile_settings(constants=const, silent=True)

        if not silent and unrecognised:
            raise KeyError(
                f"Unrecognized parameters in update: {unrecognised}. "
                "These parameters were not updated.",
            )

        return recognised


    @property
    def parameters(self):
        """Parameter values configured for the system."""
        return self.compile_settings.parameters

    @property
    def states(self):
        """Initial state values configured for the system."""
        return self.compile_settings.initial_states

    @property
    def initial_values(self):
        """Alias for :attr:`states`."""
        return self.compile_settings.initial_states

    @property
    def observables(self):
        """Observable definitions configured for the system."""
        return self.compile_settings.observables

    @property
    def constants(self):
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
    def precision(self):
        """Precision factory configured for the system."""
        return self.compile_settings.precision

    @property
    def numba_precision(self):
        """Numba representation of the configured precision."""
        return self.compile_settings.numba_precision

    @property
    def simsafe_precision(self):
        """Precision promoted for CUDA simulator compatibility."""
        return self.compile_settings.simsafe_precision

    @property
    def dxdt_function(self):
        """Compiled CUDA device function for ``dxdt``."""
        return self.get_cached_output("dxdt")

    @property
    def observables_function(self) -> Callable:
        """Return the compiled observables device function.

        Returns
        -------
        Callable
            CUDA device function that computes observables without updating
            the derivative buffer.
        """
        return self.get_cached_output("observables")

    def get_solver_helper(self,
                          func_name: str,
                          beta: float = 1.0,
                          gamma: float = 1.0,
                          mass: Any = 1.0,
                          preconditioner_order: int = 0) -> Callable:
        """Retrieve a cached solver helper function.

        Parameters
        ----------
        func_name
            Identifier for the helper function.
        beta
            Shift parameter for the linear operator. Defaults to ``1.0``.
        gamma
            Weight of the Jacobian term in the linear operator. Defaults to
            ``1.0``.
        preconditioner_order
            Polynomial order of the preconditioner. Defaults to ``0``. Unused
            when generating the linear operator.
        mass
            Mass matrix used by the linear operator. Defaults to identity.

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
