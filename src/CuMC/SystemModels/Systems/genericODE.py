# -*- coding: utf-8 -*-
"""
Created on Wed May 28 10:36:56 2025

@author: cca79
"""


import numba as nb
from numba import cuda, from_dtype

from CuMC.SystemModels.SystemValues import SystemValues

import numpy as np
from cupy import asarray



default_parameters = {'E_h': 0.52,
                    'E_a' : 0.0133,
                    'E_v' : 0.0624,
                    'R_i' : 0.012,
                    'R_o' : 1.0,
                    'R_c' : 1/114,
                    'V_s3' : 2.0}

default_initial_values = {'Arterial pressure': 0.0,
                        'Arterial SBV': 1.0,
                        'Venous pressure': 2.0,
                        'Venous SBV': 3.0,
                        'Cardiac Pressure': 4.0,
                        'Cardiac SBV': 5.0,
                        'Circulatory Flow': 6.0,
                        'Cardiac input flow': 7.0,
                        'Cardiac output flow': 8.0}

default_observables = {'P_a': 0.0,  # Pressure in arteries
                      'P_v': 0.0,   # Pressure in veins
                      'P_h': 0.0,   # Pressure in heart
                      'Q_i': 0.0,   # Flow through input valve
                      'Q_o': 0.0,   # Flow through output valve
                      'Q_c': 0.0}   # Flow in circulation

default_constants = {}



class genericODE:

    
    """
    """
    def __init__(self,
                 initial_values,
                 parameters,
                 observables=None,
                 constants=None,
                 mutable_parameters=None,
                 precision=np.float64,
                 **kwargs):
        """Initialize the ODE system with initial values, parameters, and observables.

        Args:
            initial_values (dict): Initial values for state variables
            parameters (dict): Parameter values for the system
            observables (dict): Observable values to track
            mutable_parameters (list, optional): List of parameter names that can change during simulation
            precision (numpy.dtype, optional): Precision to use for calculations
            **kwargs: Additional arguments
        """

        self.init_values = SystemValues(initial_values, default_initial_values, precision=precision)
        self.parameters = SystemValues(parameters, default_parameters, precision=precision)
        self.observables = SystemValues(observables, default_observables, precision=precision)
        self.constants = SystemValues(constants, default_constants, precision=precision)
        self.mutable_parameters = mutable_parameters if mutable_parameters is not None else []
        
        self.precision = precision


    def build(self):
        """Build the ODE system by setting up the dxdt function and related attributes."""
        num_states = self.init_values.n
        precision = from_dtype(self.precision)
        self.noise_sigmas = np.zeros(num_states, dtype=precision)

        # Hoist fixed parameters to global namespace
        global global_constants
        global_constants = self.constants

        @cuda.jit((precision[:],
                   precision[:],
                   precision,
                   precision[:],
                   precision[:]),
                  device=True,
                  inline=True)
        def three_chamber_model_dv(state,
                                   parameters,
                                   constants,
                                   driver,
                                   observables,
                                   dxdt
                                   ):
            """
            Driver/forcing (numeric):

                e(t):  current value of driver function

            State (np.array or array-like):

                0: V_h: Volume in heart - dV_h/dt = Q_i - Q_o
                1: V_a: Volume in arteries - dV_a/dt = Q_o - Q_c
                2: V_v: Volume in vains - dV_v/dt = Q_c - Q_i

            Parameters (np.array or array-like):

                0: E_h: Elastance of Heart  (e(t) multiplier)
                1: E_a: Elastance of Arteries
                2: E_v: Elastance of Ventricles
                3: R_i: Resistance of input (mitral) valve
                4: R_o: Resistance of output (atrial) valve
                5: R_c: Resistance of circulation (arteries -> veins)
                6: SBV: The total stressed blood volume - the volume in the three chambers,
                        not pooled in the body


            dxdt (np.array or array-like):

                Input values not used!
                0: dV_h: increment in V_h
                1: dV_a: increment in V_a
                2: dV_v: increment in V_v

            Observables (np.array or array-like):

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

            #Extract parameters from the parameters array
            # Note: There is some balance to be struck here between flexibility and performance. In most kernels, many
            # parameters will be fixed, so we can hoist them to the global namespace for speed. However, it would be 
            # too onerous to modify the code in this function manually for every different combination of fixed/free parameters.
            # Junie: please provide a solution to this problem ideally shifting all but a given list of parameters to 
            # the global namespace when the system is built.
            # For now, we assume that all parameters are mutable and passed in the parameters array.
            E_h = parameters[0]
            E_a = parameters[1]
            E_v = parameters[2]
            R_i = parameters[3]
            R_o = parameters[4]
            R_c = parameters[5]
            SBV = parameters[6]

            V_h = state[0]
            V_a = state[1]
            V_v = state[2]

            # Calculate pressures
            P_a = E_a * V_a
            P_v = E_v * V_v
            P_h = E_h * V_h * driver

            # Calculate flows
            Q_i = ((P_v - P_h) / R_i) if (P_v > P_h) else 0
            Q_o = ((P_h - P_a) / R_o) if (P_h > P_a) else 0
            Q_c = (P_a - P_v) / R_c  # Fixed the division operator

            # Calculate gradient
            dV_h = Q_i - Q_o;
            dV_a = Q_o - Q_c;
            dV_v = Q_c - Q_i;

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


        self.dxdtfunc = three_chamber_model_dv
        
        #Clean up globals
        del global_constants
    


    def update_constants(self, updates_dict=None, **kwargs):
        """Update parameter values.

        Args:
            updates_dict (dict, optional): Dictionary of parameter updates
            **kwargs: Additional parameter updates
        """
        if updates_dict is None:
            updates_dict = {}

        combined_updates = {**updates_dict, **kwargs}

        # Note: If the same value occurs in the dict and
        # keyword args, the kwargs one will win.
        for key, value in combined_updates.items():
            # Update the parameter in the parameters SystemValues object
            self.parameters.set_parameter(key, self.precision(value))

            # If this is a fixed parameter, update the global variable too
            if key not in self.mutable_parameters:
                globals()[key] = self.precision(value)

    def set_noise_sigmas(self, noise_vector):
        self.noise_sigmas = np.asarray(noise_vector, dtype=self.precision)

    def get_noise_sigmas(self):
        return self.noise_sigmas.copy()

#******************************* TEST CODE ******************************** #
# if __name__ == '__main__':


    # sys = diffeq_system()
    # dxdt = sys.dxdtfunc

    # @cuda.jit()
    # def testkernel(out):
    #     # precision = np.float32
    #     # numba_precision = float32
    #     l_dxdt = cuda.local.array(shape=NUM_STATES, dtype=numba_precision)
    #     l_states = cuda.local.array(shape=NUM_STATES, dtype=numba_precision)
    #     l_constants = cuda.local.array(shape=NUM_CONSTANTS, dtype=numba_precision)
    #     l_states[:] = precision(1.0)
    #     l_constants[:] = precision(1.0)

    #     t = precision(1.0)
    #     dxdt(l_dxdt,
    #         l_states,
    #         l_constants,
    #         t)

    #     out = l_dxdt


    #     NUM_STATES = 5
    #     NUM_CONSTANTS = 14
    #     outtest = np.zeros(NUM_STATES, dtype=np.float4)
    #     out = cuda.to_device(outtest)
    #     print("Testing to see if your dxdt function compiles using CUDA...")
    #     testkernel[1,1](out)
    #     cuda.synchronize()
    #     out.copy_to_host(outtest)
    #     print(outtest)
