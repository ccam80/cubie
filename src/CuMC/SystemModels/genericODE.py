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






class genericODE:


    """
    """
    def __init__(self,
                 initial_values=None,
                 parameters=None, #parameters that can change during simulation
                 constants=None, #Parameters that are not expected to change during simulation
                 observables=None, #Auxiliary variables you might want to track during simulation
                 default_initial_values=None,
                 default_parameters=None,
                 default_constants=None,
                 precision=np.float64,
                 num_drivers=1,
                 **kwargs):
        """Initialize the ODE system with initial values, parameters, and observables.

        Args:
            initial_values (dict, optional): Initial values for state variables. Default is None.
            parameters (dict, optional): Parameter values for the system. Default is None.
            constants (dict, optional): Constants that are not expected to change during simulation. Default is None.
            observables (dict, optional): Observable values to track. Default is None.
            default_initial_values (dict, optional): Default initial values if not provided in initial_values. Default is None.
            default_parameters (dict, optional): Default parameter values if not provided in parameters. Default is None.
            default_constants (dict, optional): Default constant values if not provided in constants. Default is None.
            precision (numpy.dtype, optional): Precision to use for calculations. Default is np.float64.
            **kwargs: Additional arguments
        """

        self.init_values = SystemValues(initial_values, precision, default_initial_values)
        self.parameters = SystemValues(parameters, precision, default_parameters)
        self.observables = SystemValues(observables, precision)
        self.constants = SystemValues(constants, precision, default_constants)
        self.precision = from_dtype(precision)

        self.num_states = self.init_values.n
        self.num_parameters = self.parameters.n
        self.num_observables = self.observables.n
        self.num_drivers = num_drivers

    def build(self):
        """Compile the dxdt system as a CUDA device function."""
        # Hoist fixed parameters to global namespace
        global global_constants
        global_constants = self.constants

        #TODO: overwrite with a simpler example model.
        @cuda.jit(
            (self.precision[:],
                   self.precision[:],
                   self.precision[:],
                   self.precision[:],
                   self.precision[:]),
                  device=True,
                  inline=True)
        def your_model_name_dxdt(state,
                                 parameters,
                                 driver,
                                 observables,
                                 dxdt
                                 ):
            """
            State (CUDA device array - local or shared for speed):
                The state of the system at time (t-1), used to calculate the diffentials.

            Parameters (CUDA device array - local or shared for speed):
                An array of parameters that can change during the simulation (i.e. between runs).
                Store these locally or in shared memory, as each thread will need it's own set.

            Driver/forcing (CUDA device array - local or shared for speed):
                A numeric value or array of any external forcing terms applied to the system.

            dxdt (CUDA device array - local or shared for speed):
                The output array to be written to in-place

            Observables (CUDA device array - local or shared for speed):
                An output array to store any auxiliary variables you might want to track during simulation.

            returns:
                None, modifications are made to the dxdt and observables arrays in-place to avoid allocating

           """

            #Extract parameters from the parameters array
            # Note: There is some balance to be struck here between flexibility and performance. In most kernels, many
            # parameters will be fixed, so we can hoist them to the global namespace for speed. However, it would be 
            # too onerous to modify the code in this function manually for every different combination of fixed/free parameters.
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


        self.dxdtfunc = your_model_name_dxdt

        #Clean up globals
        del global_constants




    def get_parameters(self, keys):
        """Get a parameter value.

        Args:
            key key (str, list(str), int, slice): The parameter key(s) to retrieve

        Returns:
            The parameter value(s)
        """
        return self.parameters[keys]

    def set_parameters(self, keys, values):
        """Set a parameter value.

        Args:
            key (str, list(str), int, slice): The parameter key(s) to set
            value: The value to set
        """
        self.parameters[keys] = values

    def get_initial_values(self, keys):
        """Get an initial value.

        Args:
            key (str, list(str), int, slice): The initial value key(s) to retrieve

        Returns:
            The initial value(s)
        """
        return self.init_values[keys]

    def set_initial_values(self, keys, values):
        """Set an initial value.

        Args:
            key (str): The initial value key to set
            value: The value to set
        """
        self.init_values[keys] = (values)


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
