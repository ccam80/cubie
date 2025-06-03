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