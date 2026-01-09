"""Containers describing ODE system metadata used by factories."""

from typing import Optional, Dict, Any

from attrs import define, field
from attrs.validators import (
    instance_of as attrsval_instance_of,
    optional as attrsval_optional,
)

from numba import from_dtype as numba_from_dtype

from cubie.CUDAFactory import CUDAFactoryConfig
from cubie._utils import (
    PrecisionDType,
)
from cubie.odesystems.SystemValues import SystemValues


@define
class SystemSizes:
    """Store counts for each component category in an ODE system.

    Parameters
    ----------
    states
        Number of state variables in the system.
    observables
        Number of observable variables in the system.
    parameters
        Number of parameters in the system.
    constants
        Number of constants in the system.
    drivers
        Number of driver variables in the system.

    Notes
    -----
    This data class is passed to CUDA kernels so they can size device buffers
    and shared-memory structures correctly.
    """

    states: int = field(validator=attrsval_instance_of(int))
    observables: int = field(validator=attrsval_instance_of(int))
    parameters: int = field(validator=attrsval_instance_of(int))
    constants: int = field(validator=attrsval_instance_of(int))
    drivers: int = field(validator=attrsval_instance_of(int))


@define
class ODEData(CUDAFactoryConfig):
    """Bundle numerical values and metadata for an ODE system.

    Parameters
    ----------
    constants
        System constants that do not change during simulation.
    parameters
        Tunable system parameters that may vary between simulations.
    initial_states
        Initial state values for the ODE system.
    observables
        Observable variables to track during simulation.
    precision
        Precision factory used for numerical calculations. Defaults to
        :class:`numpy.float32`.
    num_drivers
        Number of driver or forcing functions. Defaults to ``1``.

    Returns
    -------
    ODEData
        Instance containing all values and derived sizes needed for CUDA
        compilation.
    """

    constants: Optional[SystemValues] = field(
        validator=attrsval_optional(
            attrsval_instance_of(
                SystemValues,
            ),
        ),
    )
    parameters: Optional[SystemValues] = field(
        validator=attrsval_optional(
            attrsval_instance_of(
                SystemValues,
            ),
        ),
    )
    initial_states: SystemValues = field(
        validator=attrsval_optional(
            attrsval_instance_of(
                SystemValues,
            ),
        ),
    )
    observables: SystemValues = field(
        validator=attrsval_optional(
            attrsval_instance_of(
                SystemValues,
            ),
        ),
    )
    num_drivers: int = field(validator=attrsval_instance_of(int), default=1)
    _mass: Any = field(default=None, eq=False)

    def update_precisions(self, updates_dict):
        """Update precision of all values in the ODEData instance."""
        if "precision" in updates_dict:
            precision = updates_dict["precision"]
            self.parameters.precision = precision
            self.constants.precision = precision
            self.initial_states.precision = precision
            self.observables.precision = precision

    @property
    def num_states(self) -> int:
        """Number of state variables."""
        return self.initial_states.n

    @property
    def num_observables(self) -> int:
        """Number of observable variables."""
        return self.observables.n

    @property
    def num_parameters(self) -> int:
        """Number of parameters."""
        return self.parameters.n

    @property
    def num_constants(self) -> int:
        """Number of constants."""
        return self.constants.n

    @property
    def sizes(self) -> SystemSizes:
        """System component sizes grouped for CUDA kernels."""
        return SystemSizes(
            states=self.num_states,
            observables=self.num_observables,
            parameters=self.num_parameters,
            constants=self.num_constants,
            drivers=self.num_drivers,
        )

    @property
    def beta(self) -> float:
        """Return the cached solver shift parameter."""
        return self.precision(self._beta)

    @property
    def gamma(self) -> float:
        """Return the cached solver Jacobian weight."""
        return self.precision(self._gamma)

    @property
    def mass(self) -> Any:
        """Return the cached solver mass matrix."""
        return self._mass

    @classmethod
    def from_BaseODE_initargs(
        cls,
        precision: PrecisionDType,
        initial_values: Optional[Dict[str, float]] = None,
        parameters: Optional[Dict[str, float]] = None,
        constants: Optional[Dict[str, float]] = None,
        observables: Optional[Dict[str, float]] = None,
        default_initial_values: Optional[Dict[str, float]] = None,
        default_parameters: Optional[Dict[str, float]] = None,
        default_constants: Optional[Dict[str, float]] = None,
        default_observable_names: Optional[Dict[str, float]] = None,
        num_drivers: int = 1,
    ) -> "ODEData":
        """Create :class:`ODEData` from ``BaseODE`` initialization arguments.

        Parameters
        ----------
        initial_values
            Initial values for state variables.
        parameters
            Parameter values for the system.
        constants
            Constants that are not expected to change during simulation.
        observables
            Auxiliary variables to track during simulation.
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

        Returns
        -------
        ODEData
            Initialized data container suitable for CUDA compilation.
        """
        init_values = SystemValues(
            initial_values, precision, default_initial_values, name="States"
        )
        parameters = SystemValues(
            parameters,
            precision,
            default_parameters,
            name="Parameters",
        )
        observables = SystemValues(
            observables,
            precision,
            default_observable_names,
            name="Observables",
        )
        constants = SystemValues(
            constants,
            precision,
            default_constants,
            name="Constants",
        )

        return cls(
            constants=constants,
            parameters=parameters,
            initial_states=init_values,
            observables=observables,
            precision=precision,
            num_drivers=num_drivers,
        )
