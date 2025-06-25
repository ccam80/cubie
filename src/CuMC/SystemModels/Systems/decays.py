# -*- coding: utf-8 -*-
"""
Created on Wed May 28 10:36:56 2025

@author: cca79
"""
if __name__ == "__main__":
    import os
    os.environ["NUMBA_ENABLE_CUDASIM"] = "0"
    os.environ["NUMBA_CUDA_DEBUGINFO"] = "0"
    os.environ["NUMBA_OPT"] = "1"

import numba as nb
from numba import cuda, from_dtype
from CuMC.SystemModels.genericODE import genericODE
import numpy as np





class Decays(genericODE):
    """ Give it a list of coefficients, and it will create a system in which each state variable decays exponentially at
    a rate proportional to its position. Observables are the same as state variables * parameters (coefficients).

    i.e. if coefficients = [1, 2, 3], then the system will have three state variables x0, x1, x2,
    and dx[0] = -x[0]/1, dx[1] = x[1]/2, dx[2] = x[2]/3. obs[0] = x[0]*1, obs[1] = x[1]*2, obs[2] = x[2]*3.

    observables are the same as state variables * parameters (coefficients)

    Really just exists for testing.
    """
    def __init__(self,
                 coefficients = [1],
                 precision=np.float64):

        nterms = len(coefficients)
        observables = [f'x{i}' for i in range(nterms)]
        initial_values = {f'x{i}': 1.0 for i in range(nterms)}
        parameters = {f'c{i}': coefficients[i] for i in range(nterms)}
        constants = {}
        n_drivers = 0 #use time as the driver

        super().__init__(initial_values=initial_values,
                        parameters=parameters,
                        constants=constants,
                        observables=observables,
                        default_initial_values=initial_values,
                        default_parameters=parameters,
                        default_constants=constants,
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
                observables[i] = state[i] * parameters[i]


        self.dxdtfunc = dxdt

#******************************* TEST CODE ******************************** #
if __name__ == '__main__':
    precision = np.float32

    sys = decays(precision=precision, coefficients = [2, 2, 2])
    sys.build()
    dxdtfunc = sys.dxdtfunc
    numba_precision = from_dtype(precision)

    nstates = sys.num_states
    npar = sys.num_parameters
    nobs = sys.num_observables

    @cuda.jit()
    def dummykernel(outarray,
                    d_inits,
                    parameters,
                    driver):

        l_dxdt = cuda.local.array(shape=(nstates), dtype=numba_precision)
        l_states = cuda.local.array(shape=(nstates), dtype=numba_precision)
        l_parameters = cuda.local.array(shape=(npar), dtype=numba_precision)
        l_observables = cuda.local.array(shape=(nobs), dtype=numba_precision)
        l_driver = cuda.local.array(shape=(1), dtype=numba_precision)


        for i in range(npar):
            l_parameters[i] = parameters[i]
            # print(l_parameters[i])
        for i in range(nstates):
            l_states[i] = d_inits[i]


        l_driver[0] = driver[0]
        l_dxdt[:] = precision(0.0)

        dxdtfunc(l_states,
             l_parameters,
             l_driver,
             l_observables,
             l_dxdt
             )

        for i in range(nstates):
            outarray[i] = l_dxdt[i]



    outtest = np.zeros(sys.num_states, dtype=precision)
    out = cuda.to_device(outtest)
    params = cuda.to_device(sys.parameters.values_array)
    inits = cuda.to_device(sys.init_values.values_array)

    driver=[precision(1.0)]
    driver = cuda.to_device(driver)

    print("Testing to see if your dxdt function compiles using CUDA...")
    dummykernel[1,1](out,
                     inits,
                     params,
                     driver
                     )
    cuda.synchronize()
    out.copy_to_host(outtest)
    print(outtest)
