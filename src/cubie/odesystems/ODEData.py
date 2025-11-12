"""Containers describing ODE system metadata used by factories."""

from typing import Optional, Dict, Any

import attrs
import numpy as np
from numpy import float32

import numba

from cubie.cuda_simsafe import from_dtype as simsafe_dtype
from cubie._utils import (
    PrecisionDType,
    getype_validator,
    inrangetype_validator,
    precision_converter,
    precision_validator,
)
from cubie.odesystems.SystemValues import SystemValues


@attrs.define
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

    states: int = attrs.field(validator=attrs.validators.instance_of(int))
    observables: int = attrs.field(validator=attrs.validators.instance_of(int))
    parameters: int = attrs.field(validator=attrs.validators.instance_of(int))
    constants: int = attrs.field(validator=attrs.validators.instance_of(int))
    drivers: int = attrs.field(validator=attrs.validators.instance_of(int))


@attrs.define
class ODEData:
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

    constants: Optional[SystemValues] = attrs.field(
        validator=attrs.validators.optional(
            attrs.validators.instance_of(
                SystemValues,
            ),
        ),
    )
    parameters: Optional[SystemValues] = attrs.field(
        validator=attrs.validators.optional(
            attrs.validators.instance_of(
                SystemValues,
            ),
        ),
    )
    initial_states: SystemValues = attrs.field(
        validator=attrs.validators.optional(
            attrs.validators.instance_of(
                SystemValues,
            ),
        ),
    )
    observables: SystemValues = attrs.field(
        validator=attrs.validators.optional(
            attrs.validators.instance_of(
                SystemValues,
            ),
        ),
    )
    precision: PrecisionDType = attrs.field(
        converter=precision_converter,
        validator=precision_validator,
        default=float32
    )
    num_drivers: int = attrs.field(
        validator=attrs.validators.instance_of(int), default=1
    )
    _beta: float = attrs.field(
        default=1.0,
        validator=inrangetype_validator(float, -32678, 32677),
    )
    _gamma: float = attrs.field(
        default=1.0,
        validator=inrangetype_validator(float, -32678, 32677),
    )
    preconditioner_order: int = attrs.field(
        default=0,
        validator=getype_validator(int, 0),
    )
    _mass: Any = attrs.field(default=None, eq=False)

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
    def numba_precision(self) -> type:
        """Numba representation of the configured precision."""
        return numba.from_dtype(self.precision)

    @property
    def simsafe_precision(self) -> type:
        """Precision promoted for CUDA simulator compatibility."""
        return simsafe_dtype(self.precision)

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
