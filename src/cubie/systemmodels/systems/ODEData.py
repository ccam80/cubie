from typing import Optional

import attrs
import numpy as np
from numpy import float32

from cubie.systemmodels.SystemValues import SystemValues


@attrs.define
class SystemSizes:
    """
    Data structure to hold the sizes of various components of a system.
    This is used to pass size information to the ODE solver kernel.
    """
    states: int = attrs.field(validator=attrs.validators.instance_of(int))
    observables: int = attrs.field(validator=attrs.validators.instance_of(int))
    parameters: int = attrs.field(validator=attrs.validators.instance_of(int))
    constants: int = attrs.field(validator=attrs.validators.instance_of(int))
    drivers: int = attrs.field(validator=attrs.validators.instance_of(int))


@attrs.define
class ODEData:
    """
    Data structure to hold ODE system parameters, initial states,
    and forcing vectors.
    This is used to pass data to the ODE solver kernel.
    """
    constants: Optional[SystemValues] = attrs.field(
            validator=attrs.validators.optional(
                    attrs.validators.instance_of(SystemValues, ), ), )
    parameters: Optional[SystemValues] = attrs.field(
            validator=attrs.validators.optional(
                    attrs.validators.instance_of(SystemValues, ), ), )
    initial_states: SystemValues = attrs.field(
            validator=attrs.validators.optional(
                    attrs.validators.instance_of(SystemValues, ), ), )
    observables: SystemValues = attrs.field(
            validator=attrs.validators.optional(
                    attrs.validators.instance_of(SystemValues, ), ), )
    precision: type = attrs.field(validator=attrs.validators.instance_of(type),
                                  default=float32)
    num_drivers: int = attrs.field(validator=attrs.validators.instance_of(int),
                                   default=1)

    @property
    def num_states(self):
        return self.initial_states.n

    @property
    def num_observables(self):
        return self.observables.n

    @property
    def num_parameters(self):
        return self.parameters.n

    @property
    def num_constants(self):
        return self.constants.n

    @property
    def sizes(self):
        """Returns a dictionary of sizes for the ODE data."""
        return SystemSizes(states=self.num_states,
                           observables=self.num_observables,
                           parameters=self.num_parameters,
                           constants=self.num_constants,
                           drivers=self.num_drivers, )

    @classmethod
    def from_genericODE_initargs(cls, initial_values=None, parameters=None,
                                 # parameters that can change during simulation
                                 constants=None,
                                 # Parameters that are not expected to change during simulation
                                 observables=None,
                                 # Auxiliary variables you might want to track during simulation
                                 default_initial_values=None,
                                 default_parameters=None,
                                 default_constants=None,
                                 default_observable_names=None,
                                 precision=np.float64, num_drivers=1, ):
        init_values = SystemValues(initial_values, precision,
                                   default_initial_values)
        parameters = SystemValues(parameters, precision, default_parameters)
        observables = SystemValues(observables, precision,
                                   default_observable_names)
        constants = SystemValues(constants, precision, default_constants)

        return cls(constants=constants, parameters=parameters,
                   initial_states=init_values, observables=observables,
                   precision=precision, num_drivers=num_drivers, )
