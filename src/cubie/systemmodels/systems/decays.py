# -*- coding: utf-8 -*-
"""
Created on Wed May 28 10:36:56 2025

@author: cca79
"""

import numpy as np
from numba import cuda, from_dtype

from cubie.systemmodels.systems.GenericODE import GenericODE


class Decays(GenericODE):
    """ Give it a list of coefficients, and it will create a system in which
    each state variable decays exponentially at
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

    def __init__(self, precision=np.float64, **kwargs, ):

        coefficients = kwargs["coefficients"]

        nterms = len(coefficients)
        observables = [f'o{i}' for i in range(nterms)]
        initial_values = {f'x{i}': 1.0 for i in range(nterms)}
        parameters = {f'p{i}': coefficients[i] for i in range(nterms)}
        constants = {f'c{i}': i for i in range(nterms)}
        n_drivers = 1  # use time as the driver

        super().__init__(initial_values=initial_values, parameters=parameters,
                         constants=constants, observables=observables,
                         precision=precision, num_drivers=n_drivers, )

    def build(self):
        # Hoist fixed parameters to global namespace
        global global_constants
        global_constants = self.compile_settings.constants.values_array
        n_terms = self.sizes.states
        numba_precision = from_dtype(self.precision)

        @cuda.jit((numba_precision[:], numba_precision[:], numba_precision[:],
                   numba_precision[:], numba_precision[:]), device=True,
                  inline=True, )
        def dxdtfunc(state, parameters, driver, observables, dxdt, ):
            """
               dx[i] = state[i] / (i+1)
               observables[i] = state[i] * parameters[i] + constants[i] + driver[0]
            """
            for i in range(n_terms):
                dxdt[i] = -state[i] / (i + 1)
                observables[i] = dxdt[i] * parameters[i] + global_constants[
                    i] + driver[0]

        return dxdtfunc

    def correct_answer_python(self, states, parameters, drivers):
        """ Python testing function - do it in python and compare results."""

        indices = np.arange(len(states))
        observables = np.zeros(self.sizes.observables)
        dxdt = -states / (indices + 1)

        for i in range(self.sizes.observables):
            observables[i] = (dxdt[i % self.sizes.states] * parameters[
                i % self.sizes.parameters] + drivers[0] + global_constants[i])

        return dxdt, observables
