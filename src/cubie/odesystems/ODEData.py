from typing import Optional

import attrs
import numpy as np
from numpy import float32

from cubie.odesystems.SystemValues import SystemValues


@attrs.define
class SystemSizes:
    """Data structure to hold the sizes of various components of a system.

    Parameters
    ----------
    states : int
        Number of state variables in the system.
    observables : int
        Number of observable variables in the system.
    parameters : int
        Number of parameters in the system.
    constants : int
        Number of constants in the system.
    drivers : int
        Number of driver variables in the system.

    Notes
    -----
    This is used to pass size information to the ODE solver kernel.
    """

    states: int = attrs.field(validator=attrs.validators.instance_of(int))
    observables: int = attrs.field(validator=attrs.validators.instance_of(int))
    parameters: int = attrs.field(validator=attrs.validators.instance_of(int))
    constants: int = attrs.field(validator=attrs.validators.instance_of(int))
    drivers: int = attrs.field(validator=attrs.validators.instance_of(int))


@attrs.define
class ODEData:
    """Data structure to hold ODE system parameters and initial states.

    Parameters
    ----------
    constants : SystemValues, optional
        System constants that do not change during simulation.
    parameters : SystemValues, optional
        System parameters that can change during simulation.
    initial_states : SystemValues
        Initial state values for the ODE system.
    observables : SystemValues
        Observable variables to track during simulation.
    precision : type, optional
        Data type for numerical calculations, by default float32.
    num_drivers : int, optional
        Number of driver/forcing functions, by default 1.
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
    precision: type = attrs.field(
        validator=attrs.validators.instance_of(type), default=float32
    )
    num_drivers: int = attrs.field(
        validator=attrs.validators.instance_of(int), default=1
    )

    @property
    def num_states(self):
        """Get the number of state variables.

        Returns
        -------
        int
            Number of state variables.
        """
        return self.initial_states.n

    @property
    def num_observables(self):
        """Get the number of observable variables.

        Returns
        -------
        int
            Number of observable variables.
        """
        return self.observables.n

    @property
    def num_parameters(self):
        """Get the number of parameters.

        Returns
        -------
        int
            Number of parameters.
        """
        return self.parameters.n

    @property
    def num_constants(self):
        """Get the number of constants.

        Returns
        -------
        int
            Number of constants.
        """
        return self.constants.n

    @property
    def sizes(self):
        """Get system sizes.

        Returns
        -------
        SystemSizes
            Object containing sizes for all system components.
        """
        return SystemSizes(
            states=self.num_states,
            observables=self.num_observables,
            parameters=self.num_parameters,
            constants=self.num_constants,
            drivers=self.num_drivers,
        )

    @classmethod
    def from_BaseODE_initargs(
        cls,
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
    ):
        """Create ODEData from BaseODE initialization arguments.

        Parameters
        ----------
        initial_values : dict, optional
            Initial values for state variables.
        parameters : dict, optional
            Parameter values for the system.
        constants : dict, optional
            Constants that are not expected to change during simulation.
        observables : dict, optional
            Auxiliary variables to track during simulation.
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

        Returns
        -------
        ODEData
            Initialized ODEData object.
        """
        init_values = SystemValues(
            initial_values, precision, default_initial_values, name="States"
        )
        parameters = SystemValues(parameters, precision, default_parameters,
                                  name="Parameters")
        observables = SystemValues(
            observables, precision, default_observable_names, name="Observables"
        )
        constants = SystemValues(constants, precision, default_constants,
                                 name="Constants")

        return cls(
            constants=constants,
            parameters=parameters,
            initial_states=init_values,
            observables=observables,
            precision=precision,
            num_drivers=num_drivers,
        )
