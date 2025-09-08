from abc import abstractmethod
from typing import Callable, Dict, Optional, Tuple, Union

import attrs
import numpy as np
from numpy.typing import NDArray

from cubie.CUDAFactory import CUDAFactory
from cubie.odesystems.ODEData import ODEData


@attrs.define
class ODECache:
    """Cache for compiled CUDA device and support functions.

    Attributes default to ``-1`` when the corresponding function is not built.
    """

    dxdt: Optional[Callable] = attrs.field()
    linear_operator: Optional[Union[Callable, int]] = attrs.field(default=-1)
    neumann_preconditioner: Optional[Union[Callable, int]] = attrs.field(
        default=-1
    )


class BaseODE(CUDAFactory):
    """Abstract base class for ODE systems.

    This class is designed to be subclassed for specific systems so that the
    shared machinery used to interface with CUDA can be reused. When subclassing,
    you should overload the build() and correct_answer_python() methods to provide
    the specific ODE system you want to simulate.

    Notes
    -----
    Only functions cached during :meth:`build` (typically ``dxdt``) are
    available on this base class. Solver helper functions such as the linear
    operator or preconditioner are generated only by subclasses like
    :class:`SymbolicODE`.

    Parameters
    ----------
    initial_values : dict, optional
        Initial values for state variables.
    parameters : dict, optional
        Parameter values for the system.
    constants : dict, optional
        Constants that are not expected to change between simulations.
    observables : dict, optional
        Observable values to track.
    default_initial_values : dict, optional
        Default initial values if not provided in initial_values.
    default_parameters : dict, optional
        Default parameter values if not provided in parameters.
    default_constants : dict, optional
        Default constant values if not provided in constants.
    default_observable_names : dict, optional
        Default observable names if not provided in observables.
    precision : numpy.dtype, optional
        Precision to use for calculations, by default np.float64.
    num_drivers : int, optional
        Number of driver/forcing functions, by default 1.
    **kwargs : dict
        Additional arguments.

    Notes
    -----
    If you do implement a correct_answer_python() method, then you can subclass
    the SystemTester class in tests/odesystems/SystemTester.py and overload
    system_class with your ODE class name. The generate_system_tests function
    can then generate a set of floating-point and missing-input tests to see if
    your system behaves as expected.

    Most systems will contain a default set of initial values, parameters,
    constants, and observables. This parent class does not contain them, but
    instead can be instantiated with a set of values of any size, for testing
    purposes. The default values provide a way to both set a default state and
    to provide the set of modifiable entries. This means that a user can't add
    in a state or parameter when solving the system that ends up having no
    effect on the system.
    """

    def __init__(
        self,
        initial_values=None,
        parameters=None,
        constants=None,
        observables=None,
        default_initial_values=None,
        default_parameters=None,
        default_constants=None,
        default_observable_names=None,
        precision=np.float64,
        num_drivers=1,
        name=None
    ):
        """Initialize the ODE system.

        Parameters
        ----------
        initial_values : dict, optional
            Initial values for state variables.
        parameters : dict, optional
            Parameter values for the system.
        constants : dict, optional
            Constants that are not expected to change between simulations.
        observables : Sequence[str], optional
            Observable values to track.
        default_initial_values : dict, optional
            Default initial values if not provided in initial_values.
        default_parameters : dict, optional
            Default parameter values if not provided in parameters.
        default_constants : dict, optional
            Default constant values if not provided in constants.
        default_observable_names : Sequence[str], optional
            Default observable names if not provided in observables.
        precision : numpy.dtype, optional
            Precision to use for calculations, by default np.float64.
        num_drivers : int, optional
            Number of driver/forcing functions, by default 1.
        name: str, optional
            String name for the system, for printing, default None.
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

    def __repr__(self):
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
        """Compile the dxdt system as a CUDA device function.

        Returns
        -------
        ODECache
            Cache containing the built ``dxdt`` function. Subclasses may add
            further solver helpers to this cache as needed.

        Notes
        -----
        Bring constants into local (outer) scope before you define the dxdt
        function, as the CUDA device function can't handle a reference to self.
        """
        # return ODECache(dxdt=dxdt)

    def correct_answer_python(
        self, states, parameters, drivers
    ) -> Tuple[NDArray, NDArray]:
        """Python version of the dxdt function for testing.

        This function is used in testing. Override this with a simpler, Python
        version of the dxdt function if you want to use the SystemTester
        class to check your function is working as expected.

        Parameters
        ----------
        states : numpy.ndarray
            Current state values.
        parameters : numpy.ndarray
            Parameter values.
        drivers : numpy.ndarray
            Driver/forcing values.

        Returns
        -------
        tuple of numpy.ndarray
            A tuple containing (dxdt, observables) arrays.
        """
        return np.asarray([0]), np.asarray([0])

    def update(self, updates_dict, silent=False, **kwargs):
        """Update compile settings through the CUDAFactory interface.

        Pass updates to compile settings through the CUDAFactory interface,
        which will invalidate cache if an update is successful.

        Parameters
        ----------
        updates_dict : dict
            Dictionary of updates to apply.
        silent : bool, optional
            If True, suppress warnings about keys not found, by default False.
        **kwargs : dict
            Additional update parameters.

        Notes
        -----
        Pass silent=True if doing a bulk update with other component's params
        to suppress warnings about keys not found.
        """
        return self.set_constants(updates_dict, silent=silent, **kwargs)

    def set_constants(
        self,
        updates_dict: Dict[str, float] = None,
        silent: bool = False,
        **kwargs,
    ):
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
        """
        if updates_dict is None:
            updates_dict = {}
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
        """Get the parameters of the system.

        Returns
        -------
        SystemValues
            The parameters of the system.
        """
        return self.compile_settings.parameters

    @property
    def states(self):
        """Get the states of the system.

        Returns
        -------
        SystemValues
            The initial values of the system.
        """
        return self.compile_settings.initial_states

    @property
    def initial_values(self):
        """Get the initial values of the system. Alias for BaseODE.states

        Returns
        -------
        SystemValues
            The initial values of the system.
        """
        return self.compile_settings.initial_states

    @property
    def observables(self):
        """Get the observables of the system.

        Returns
        -------
        SystemValues
            The observables of the system.
        """
        return self.compile_settings.observables

    @property
    def constants(self):
        """Get the constants of the system.

        Returns
        -------
        SystemValues
            The constants of the system.
        """
        return self.compile_settings.constants

    @property
    def num_states(self):
        """Get the number of state variables.

        Returns
        -------
        int
            Number of state variables.
        """
        return self.compile_settings.num_states

    @property
    def num_observables(self):
        """Get the number of observable variables.

        Returns
        -------
        int
            Number of observable variables.
        """
        return self.compile_settings.num_observables

    @property
    def num_parameters(self):
        """Get the number of parameters.

        Returns
        -------
        int
            Number of parameters.
        """
        return self.compile_settings.num_parameters

    @property
    def num_constants(self):
        """Get the number of constants.

        Returns
        -------
        int
            Number of constants.
        """
        return self.compile_settings.num_constants

    @property
    def num_drivers(self):
        """Get the number of driver variables.

        Returns
        -------
        int
            Number of driver variables.
        """
        return self.compile_settings.num_drivers

    @property
    def sizes(self):
        """Get system sizes.

        Returns
        -------
        SystemSizes
            Dictionary of sizes (number of states, parameters, observables,
            constants, drivers) for the system.
        """
        return self.compile_settings.sizes

    @property
    def precision(self):
        """Get the precision of the system.

        Returns
        -------
        numpy.dtype
            The precision of the system (numba type, float32 or float64).
        """
        return self.compile_settings.precision

    @property
    def dxdt_function(self):
        """Get the compiled device function.

        Returns
        -------
        function
            The compiled CUDA device function.
        """
        return self.get_cached_output("dxdt")

    @property
    def operator_function(self):
        """Return the compiled linear-operator device function."""
        return self.get_solver_helper("operator")

    @property
    def neumann_preconditioner_function(self):
        """Return the compiled Neumann preconditioner device function."""
        return self.get_solver_helper("neumann")
    def get_solver_helper(self, func_name: str):
        """Retrieve a cached solver helper function.

        Parameters
        ----------
        func_name : str
            Identifier for the helper function.

        Returns
        -------
        Callable
            The cached device function corresponding to ``func_name``.
        """
        return self.get_cached_output(func_name)
