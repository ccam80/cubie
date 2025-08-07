# -*- coding: utf-8 -*-
"""
Created on Wed May 28 10:36:56 2025

@author: cca79
"""
import numpy as np
from numba import cuda, from_dtype

from cubie.systemmodels.systems.GenericODE import GenericODE

default_parameters = {'E_h': 0.52, 'E_a': 0.0133, 'E_v': 0.0624, 'R_i': 0.012,
                      'R_o': 1.0, 'R_c': 1 / 114, 'V_s3': 2.0}

default_initial_values = {'V_h': 1.0, 'V_a': 1.0, 'V_v': 1.0}

default_observable_names = ['P_a', 'P_v', 'P_h', 'Q_i', 'Q_o',
                            'Q_c']  # Flow in circulation

default_constants = {}


# noinspection PyPep8Naming
class ThreeChamberModel(GenericODE):
    """ Three chamber model as laid out in [Pironet's thesis reference].

    """

    def __init__(self, initial_values=None, parameters=None, constants=None,
                 observables=None, precision=np.float64,
                 default_initial_values=default_initial_values,
                 default_parameters=default_parameters,
                 default_constants=default_constants,
                 default_observable_names=default_observable_names,
                 num_drivers=1,
                 # Error: This probably shouldn't be an instantiation
                 # parameter, but rather a property of the system.
                 **kwargs, ):
        super().__init__(initial_values=initial_values, parameters=parameters,
                         constants=constants, observables=observables,
                         default_initial_values=default_initial_values,
                         default_parameters=default_parameters,
                         default_constants=default_constants,
                         default_observable_names=default_observable_names,
                         precision=precision, num_drivers=num_drivers, )

    def build(self):
        # Hoist fixed parameters to global namespace
        global global_constants
        global_constants = self.compile_settings.constants.values_array.astype(
                self.precision)

        numba_precision = from_dtype(self.precision)

        @cuda.jit((numba_precision[:], numba_precision[:], numba_precision[:],
                   numba_precision[:], numba_precision[:]), device=True,
                  inline=True, )
        def three_chamber_model_dv(state, parameters, driver, observables,
                                   dxdt, ):
            """

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
            observables[0] = P_a
            observables[1] = P_v
            observables[2] = P_h
            observables[3] = Q_i
            observables[4] = Q_o
            observables[5] = Q_c

            dxdt[0] = dV_h
            dxdt[1] = dV_a
            dxdt[2] = dV_v

        return three_chamber_model_dv

    def correct_answer_python(self, states, parameters, drivers):
        """ More-direct port of Nic Davey's MATLAB implementation.         """

        E_h, E_a, E_v, R_i, R_o, R_c, _ = parameters
        V_h, V_a, V_v = states
        driver = drivers[0]

        P_v = E_v * V_v
        P_h = E_h * V_h * driver
        P_a = E_a * V_a

        if P_v > P_h:
            Q_i = (P_v - P_h) / R_i
        else:
            Q_i = 0

        if P_h > P_a:
            Q_o = (P_h - P_a) / R_o
        else:
            Q_o = 0

        Q_c = (P_a - P_v) / R_c

        dxdt = np.asarray([Q_i - Q_o, Q_o - Q_c, Q_c - Q_i],
                          dtype=self.precision)
        observables = np.asarray([P_a, P_v, P_h, Q_i, Q_o, Q_c],
                                 dtype=self.precision)

        return dxdt, observables
