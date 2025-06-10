# -*- coding: utf-8 -*-
"""
Created on Wed May 28 10:36:56 2025

@author: cca79
"""


import numba as nb
from numba import cuda
from numba import float64, float32

import numpy as np

#Grab the system precision from an environment variable. This is onerous, but
#allows the device function to be compiled at import time using the correct value.
import os
precision = os.environ.get("cuda_precision")

if precision == "float32":
    precision = float32
elif precision == "float64":
    precision = float64
elif precision is None:
    precision = float64


defaults = {'E_h': 0.52,
                'E_a' : 0.0133,
                'E_v' : 0.0624,
                'R_i' : 0.012,
                'R_o' : 1.0,
                'R_c' : 1/114,
                'V_s3' : 2.0}

state_labels = {'Arterial pressure': 0,
                'Arterial SBV': 1,
                'Venous pressure': 2,
                'Venous SBV': 3,
                'Cardiac Pressure': 4,
                'Cardiac SBV': 5,
                'Circulatory Flow': 6,
                'Cardiac input flow': 7,
                'Cardiac output flow': 8}

""" state array looks like:
    [P_a,  #0
     V_sa, #1
     P_v,  #2
     V_sv, #3
     P_h,  #4
     V_sh, #5
     Q_c,  #6
     W_i,  #7
     Q_o   #8] """




class diffeq_system:
    """ This class should contain all system definitions. The constants management
    scheme can be a little tricky, because the GPU stuff can't handle dictionaries.
    The constants_array will be passed to your dxdt function - you can use the indices
    given in self.constant_indices to map them out while you set up your dxdt function.

    > test_system = diffeq_system()
    > print(diffeq_system.constant_indices)

    - Place all of your system constants and their labelsin the constants_dict.
    - Update self.num_states to match the number of state variables/ODEs you
    need to solve.
    - Feel free to define any helper functions inside the __init__ function.
    These must have the cuda.jit decorator with a signature (return(arg)), like you can
    see in the example functions.
    You can call these in the dxdt function.
    - update noise_sigmas with the std dev of gaussian noise in any state if
    you're doing a "noisy" run.

    Many numpy (and other) functions won't work inside the dxdt or CUDA device
    functions. Try using the Cupy function instead if you get an error.

    """
    def __init__(self,
                 num_states =3,
                 num_algebraics=5,
                 precision=np.float64,
                 state_labels = state_labels,
                 **kwargs):
        """Set system constant values then function as a factory function to
        build CUDA device functions for use in the ODE solver kernel. No
        arguments, no returns it's all just bad coding practice in here.

        Everything except for the constants_array and constant_indices generators
        and dxdt assignment at the end is an example, you will need to overwrite"""

        self.num_states = num_states
        self.precision = precision
        self.numba_precision = from_dtype(precision)
        # self.noise_sigmas = np.zeros(self.num_states, dtype=precision)

        self.constants_dict  = system_constants(kwargs)
        self.constants_array = asarray([constant for (label, constant) in self.constants_dict.items()], dtype=precision)
        self.constant_indices = {label: index for index, (label, constant) in enumerate(self.constants_dict.items())}

        self.state_labels = state_labels



        @cuda.jit((self.numba_precision[:],
                   self.numba_precision[:],
                   self.numba_precision[:],
                   self.numba_precision[:],
                   self.numba_precision),
                  device=True,
                  inline=True,)
        def dxdtfunc(outarray,
                     state,
                     constants,
                     algebraics,
                     t):
            """ Put your dxdt calculations in here, including any reference signal
            or other math. Ugly is good here, avoid creating local variables and
            partial calculations - a long string of multiplies and adds, referring to
            the same array, might help the compiler make it fast. Avoid low powers,
            use consecutive multiplications instead.

            For a list of supported math functions you can include, see
            :https://numba.readthedocs.io/en/stable/cuda/cudapysupported.html"""

        	"""" state array looks like:
                [P_a,  #0
                 V_sa, #1
                 P_v,  #2
                 V_sv, #3
                 P_h,  #4
                 V_sh, #5
                 Q_c,  #6
                 Q_i,  #7
                 Q_o   #8] """

            """Constants array looks like:
                ['E_h':  0,
                  'E_a' : 1,
                  'E_v' : 2,
                  'R_i' : 3,
                  'R_o' : 4,
                  'R_c' : 5,
                  'V_s3' : 6] """
            ### Algebraic
            outarray[0] = constants[1] * states[1]
            outarray[2] = constants[2] * states[3]
            outarray[4] =

            #Consider inverting resistances to get conductances for this step if this whole thing proves worthwhile
            outarray[7] = max((states[2] - states[4])  / constants[3], 0.0)
            outarray[8] = max((states[4] - states[0])  / constants[4], 0.0)
            outarray[6] = (states[0] - states[2]) / constants[4]


            #Separate algebraic and derivate parts of this array.


        self.dxdtfunc = dxdtfunc
        self.clipfunc = clamp


    def update_constants(self, updates_dict=None, **kwargs):
        if updates_dict is None:
            updates_dict = {}

        combined_updates = {**updates_dict, **kwargs}

        # Note: If the same value occurs in the dict and
        # keyword args, the kwargs one will win.
        for key, item in combined_updates.items():
            self.constants_dict.set_parameter(key, self.precision(item))

        self.constants_array = asarray([constant for (label, constant) in self.constants_dict.items()], dtype=self.precision)

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
@cuda.jit((precision[:],
           precision[:],
           precision,
           precision[:],
           precision[:]),
          device=True,
          inline=True)
def three_chamber_model_dV(state,
                           parameters,
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


    #Calculate auxiliary (observable) values
    P_a = E_a * V_a
    P_v = E_v * V_v
    P_h = E_h * V_h * driver;
    Q_i = ((P_v - P_h) / R_i) if (P_v > P_h) else 0
    Q_o = ((P_h - P_a) / R_o) if (P_h > P_a) else 0
    Q_c = P_a - P_v / R_c;

    #Calculate gradient
    dV_h = Q_i - Q_o;
    dV_a = Q_o - Q_c;
    dV_v = Q_c - Q_i;

    #Package values up into output arrays, overwriting for speed.
    #TODO: Optimisation target. some of these values will go unused, can reduce memory operations by only saving a requested subset.
    observables[0] = P_a
    observables[1] = P_v
    observables[2] = P_h
    observables[3] = Q_i
    observables[4] = Q_o
    observables[5] = Q_c

    dxdt[0] = dV_h
    dxdt[1] = dV_a
    dxdt[2] = dV_v