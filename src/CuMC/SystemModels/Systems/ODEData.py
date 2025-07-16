import attrs
import numpy as np
from numba.types import Float, float32
from typing import Optional
from CuMC.SystemModels.SystemValues import SystemValues
from numba import from_dtype

@attrs.define
class ODEData:
    """
    Data structure to hold ODE system parameters, initial states, and forcing vectors.
    This is used to pass data to the ODE solver kernel.
    """
    constants: Optional[SystemValues] = attrs.field(validator=attrs.validators.optional(attrs.validators.instance_of(
            SystemValues)))
    parameters: Optional[SystemValues] = attrs.field(validator=attrs.validators.optional(attrs.validators.instance_of(
            SystemValues)))
    initial_states: SystemValues = attrs.field(validator=attrs.validators.optional(attrs.validators.instance_of(
            SystemValues)))
    observables: SystemValues = attrs.field(validator=attrs.validators.optional(attrs.validators.instance_of(
            SystemValues)))
    precision: Float = attrs.field(validator=attrs.validators.instance_of(Float), default=float32)
    num_drivers: int = attrs.field(validator=attrs.validators.instance_of(int), default=1)
    # def __attrs_post_init__(self):

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
        return {
            'n_states': self.num_states,
            'n_observables': self.num_observables,
            'n_parameters': self.num_parameters,
            'n_constants': self.num_constants,
            'n_drivers': self.num_drivers
        }

    @classmethod
    def from_genericODE_initargs(cls,
                                 initial_values=None,
                                 parameters=None,  # parameters that can change during simulation
                                 constants=None,  # Parameters that are not expected to change during simulation
                                 observables=None,  # Auxiliary variables you might want to track during simulation
                                 default_initial_values=None,
                                 default_parameters=None,
                                 default_constants=None,
                                 default_observable_names=None,
                                 precision=np.float64,
                                 num_drivers=1,
                                 ):
        init_values = SystemValues(initial_values, precision, default_initial_values)
        parameters = SystemValues(parameters, precision, default_parameters)
        observables = SystemValues(observables, precision, default_observable_names)
        constants =  SystemValues(constants, precision, default_constants)

        return cls(constants=constants,
                   parameters=parameters,
                   initial_states=init_values,
                   observables=observables,
                   precision=from_dtype(precision),
                   num_drivers=num_drivers,
                   )

