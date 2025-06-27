# -*- coding: utf-8 -*-
"""
Created on Wed May 28 10:36:56 2025

@author: cca79
"""


from numba import cuda, from_dtype, float32, float64
from CuMC.SystemModels.Systems.GenericODE import GenericODE
import numpy as np





class Decays(GenericODE):
    """ Give it a list of coefficients, and it will create a system in which each state variable decays exponentially at
    a rate proportional to its position. Observables are the same as state variables * parameters (coefficients) + index.

    i.e. if coefficients = [c1, c2, c3], then the system will have three state variables x0, x1, x2,
    and:

    dx[0] = -x[0]/1,
    dx[1] = x[1]/2,
    dx[2] = x[2]/3

    obs[0] = dx[0]*c1 + 1 + step_count,
    obs[1] = dx[1]*c2 + 2 + step_count,
    obs[2] = dx[2]*c3 + 3 + step_count.


    Really just exists for testing.
    """
    def __init__(self,
                 precision=np.float64,
                 **kwargs):
        #let the user specify if we need a unified template for testing - so it looks the same as a real system.
        if "instantiation" in kwargs:
            instantiation = kwargs["instantiation"]
            if instantiation == "unified":
                initial_values=kwargs["initial_values"]
                parameters=kwargs["parameters"]
                constants=kwargs["constants"]
                observables= kwargs["observables"]
                n_drivers=kwargs["num_drivers"]
        elif "coefficients" in kwargs:

            coefficients = kwargs["coefficients"]

            nterms = len(coefficients)
            observables = [f'x{i}' for i in range(nterms)]
            initial_values = {f'x{i}': 1.0 for i in range(nterms)}
            parameters = {f'p{i}': coefficients[i] for i in range(nterms)}
            constants = {f'c{i}': i for i in range(nterms)}
            n_drivers = 1 #use time as the driver
        else:
            raise ValueError("No coefficients or unified instantiation arguments provided for Decays system.")

        super().__init__(initial_values=initial_values,
                        parameters=parameters,
                        constants=constants,
                        observables=observables,
                        precision=precision,
                        num_drivers=n_drivers)


    def build(self):
        # Hoist fixed parameters to global namespace
        global global_constants
        global_constants = self.constants.values_array
        n_terms = self.num_states

        @cuda.jit((self.precision[:],
                   self.precision[:],
                   self.precision[:],
                   self.precision[:],
                   self.precision[:]),
                  device=True,
                  inline=True)
        def dxdt(state,
                 parameters,
                 driver,
                 observables,
                 dxdt
                 ):
            """
               dx[i] =
               observables[i] = state[i] * parameters[i]
            """
            for i in range(n_terms):
                dxdt[i] = state[i] / (i+1)
                observables[i] = dxdt[i] * parameters[i] + global_constants[i] + driver[0]


        self.dxdtfunc = dxdt

    def correct_answer_python(self, states, parameters, drivers):
        """ Python testing function - do it in python and compare results."""
        numpy_precision = np.float64 if self.precision == float64 else np.float32

        indices = np.arange(len(states))

        dxdt = states / (indices + 1)
        observables = dxdt * parameters + indices + drivers[0]

        return dxdt, observables