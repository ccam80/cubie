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
from numba import float64, float32

from CuMC.SystemModels.genericODE import genericODE

import numpy as np


default_parameters = {'E_h': 0.52,
                     'E_a': 0.0133,
                     'E_v': 0.0624,
                     'R_i': 0.012,
                     'R_o': 1.0,
                     'R_c': 1/114,
                     'V_s3': 2.0}

default_initial_values = {'V_h': 1.0,
                          'V_a': 1.0,
                          'V_v': 1.0}

default_observables =['P_a','P_v','P_h','Q_i','Q_o','Q_c']   # Flow in circulation

default_constants = {}




class ThreeChamberModel(genericODE):
    """ Three chamber model as laid out in [Pironet's thesis reference].

    """
    def __init__(self,
                 initial_values = None,
                 parameters = None,
                 observables = default_observables,
                 constants = None,
                 precision=np.float64,
                 default_initial_values = default_initial_values,
                 default_parameters = default_parameters,
                 default_constants = default_constants,
                 **kwargs):
        super().__init__(initial_values=initial_values,
                        parameters=parameters, #parameters that can change during simulation
                        constants=constants, #Parameters that are not expected to change during simulation
                        observables=observables, #Auxiliary variables you might want to track during simulation
                        default_initial_values=default_initial_values,
                        default_parameters=default_parameters,
                        default_constants=default_constants,
                        precision=precision)


    def build(self):
        # Hoist fixed parameters to global namespace
        global global_constants
        global_constants = self.constants

        @cuda.jit((self.precision[:],
                   self.precision[:],
                   self.precision[:],
                   self.precision[:],
                   self.precision[:]),
                  device=True,
                  inline=True)
        def three_chamber_model_dV(state,
                                 parameters,
                                 driver,
                                 observables,
                                 dxdt
                                 ):
            """

            State (CUDA device array - local or shared for speed):
                0: V_h: Volume in heart - dV_h/dt = Q_i - Q_o
                1: V_a: Volume in arteries - dV_a/dt = Q_o - Q_c
                2: V_v: Volume in vains - dV_v/dt = Q_c - Q_i

            Parameters (CUDA device array - local or shared for speed):

                0: E_h: Elastance of Heart  (e(t) multiplier)
                1: E_a: Elastance of Arteries
                2: E_v: Elastance of Ventricles
                3: R_i: Resistance of input (mitral) valve
                4: R_o: Resistance of output (atrial) valve
                5: R_c: Resistance of circulation (arteries -> veins)
                6: SBV: The total stressed blood volume - the volume in the three chambers,
                        not pooled in the body

            Driver/forcing (CUDA device array - local or shared for speed):

                e(t):  current value of driver function

            dxdt (CUDA device array - local or shared for speed):

                Input values not used!
                0: dV_h: increment in V_h
                1: dV_a: increment in V_a
                2: dV_v: increment in V_v

            Observables (CUDA device array - local or shared for speed):

                Input values not used!
                0: P_a: Pressure in arteries -  E_a * V_a
                1: P_v: Pressure in veins = E_v * V_v
                2: P_h: Pressure in "heart" = e(t) * E_h * V_h where e(t) is the time-varying elastance driver function
                3: Q_i: Flow through "input valve" (Mitral) = (P_v - P_h) / R_i
                4: Q_o: Flow through "output valve" (Aortic) = (P_h - P_a) / R_o
                5: Q_c: Flow in circulation = (P_a - P_v) / R_c

            returns:
                None, modifications are made to the dxdt and observables arrays in-place to avoid allocating

           """
            # Extract parameters from input arrays - purely for readability
            E_h = parameters[0]
            E_a = parameters[1]
            E_v = parameters[2]
            R_i = parameters[3]
            R_o = parameters[4]
            R_c = parameters[5]
            # SBV = parameters[6]

            V_h = state[0]
            V_a = state[1]
            V_v = state[2]

            # Calculate auxiliary (observable) values
            P_a = E_a * V_a
            P_v = E_v * V_v
            P_h = E_h * V_h * driver[0]
            Q_i = ((P_v - P_h) / R_i) if (P_v > P_h) else 0
            Q_o = ((P_h - P_a) / R_o) if (P_h > P_a) else 0
            Q_c = (P_a - P_v) / R_c

            # Calculate gradient
            dV_h = Q_i - Q_o
            dV_a = Q_o - Q_c
            dV_v = Q_c - Q_i

            # Package values up into output arrays, overwriting for speed.
            # TODO: Optimisation target. some of these values will go unused, can reduce memory operations by only saving a requested subset.
            observables[0] = P_a
            observables[1] = P_v
            observables[2] = P_h
            observables[3] = Q_i
            observables[4] = Q_o
            observables[5] = Q_c

            dxdt[0] = dV_h
            dxdt[1] = dV_a
            dxdt[2] = dV_v


        self.dxdtfunc = three_chamber_model_dV

#******************************* TEST CODE ******************************** #
if __name__ == '__main__':
    precision = np.float32

    sys = ThreeChamberModel(precision=precision)
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


        # x = cuda.threadIdx.x
        # bx = cuda.blockIdx.x
        # if x == 0 and bx == 0:
        #     from pdb import set_trace;
        #     set_trace()
        # print(l_parameters)
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
