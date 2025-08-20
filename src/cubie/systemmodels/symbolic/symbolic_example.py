"""Symbolic example of the ThreeCM class"""

observables = ["P_a", "P_v", "P_h", "Q_i", "Q_o", "Q_c"]
parameters = {
    "E_h": 0.52,
    "E_a": 0.0133,
    "E_v": 0.0624,
    "R_i": 0.012,
    "R_o": 1.0,
    "R_c": 1 / 114,
    "V_s3": 2.0,
}
constants = {'R_o': 0.012}

driver = ['driver']
states = {"V_h": 1.0,
          "V_a": 1.0,
          "V_v": 1.0}

#Execute this partial implementation if it allows us to get the symbols into
# local scope for coding the dxdt functions.
incomplete_system = symbolic.setup_system(observables, parameters, constants,
                                    driver, states)
#make symbols available in local scope maybe?
dxdt = """P_a = E_a * V_a
          P_v = E_v * V_v
          P_h = E_h * V_h * driver[0]
          Q_i = ((P_v - P_h) / R_i) if (P_v > P_h) else 0
          Q_o = ((P_h - P_a) / R_o) if (P_h > P_a) else 0
          Q_c = (P_a - P_v) / R_c
    
          dV_h = Q_i - Q_o
          dV_a = Q_o - Q_c
          dV_v = Q_c - Q_i
"""
#OR
dxdt = ["P_a = E_a * V_a",
        "P_v = E_v * V_v",
        "P_h = E_h * V_h * driver",
        "Q_i = ((P_v - P_h) / R_i) if (P_v > P_h) else 0",
        "Q_o = ((P_h - P_a) / R_o) if (P_h > P_a) else 0",
        "Q_c = (P_a - P_v) / R_c",
        "dV_h = Q_i - Q_o",
        "dV_a = Q_o - Q_c",
        "dV_v = Q_c - Q_i",]

# OR, if we can make symbols available (preferred option)
dxdt = [P_a = E_a * V_a,
        P_v = E_v * V_v,
        P_h = E_h * V_h * driver,
        Q_i = ((P_v - P_h) / R_i) if (P_v > P_h) else 0,
        Q_o = ((P_h - P_a) / R_o) if (P_h > P_a) else 0,
        Q_c = (P_a - P_v) / R_c,
        dV_h = Q_i - Q_o,
        dV_a = Q_o - Q_c,
        dV_v = Q_c - Q_i,]

#We check that all observables are assigned to, that no states are assigned
# to, and that all states have a related dstate assignment. If a state is
# asssigned to directly and there is no dstate assignment, a warning is
# raised, and the state is removed from states and added to observables.

#Sympy generates the following code:

from cubie.batch_solving.math_functions import [used_functions]
from numba import cuda

#Instantiation of systemvalues etc as per GenericODE

@cuda.jit(
            (
                numba_precision[:],
                numba_precision[:],
                numba_precision[:],
                numba_precision[:],
                numba_precision[:],
            ),
            device=True,
            inline=True,
        )
        def dxdt(
            state,
            parameters,
            driver,
            observables,
            dxdt,
        ):
    # replace the symbols in the dxdt functions with array references. The
    # arrays will be state, parameters, observables, driver, and the indices
    # will correspond to the order in which the symbols were declared in the
    # dicts/lists at the top.
    # any variable with a preceding d will be part of the dxdt array, with an
    # index corresponding to the state index of the symbol following the d

    # Operators have been replaced with numba equivalents, e.g. if/else
    # statements have been swapped for selps.

            observables[0] = parameters[1] * state[1]
            observables[1] = parameters[2] * state[2]
            observables[2] = parameters[0] * state[0] * driver[0]
            observables[3] = cuda.selp(
                    observables[1] > observables[2],
                    (observables[1] - observables[2]) / parameters[3],
                    0)
            observables[4] = cuda.selp(
                (observables[2] > observables[0]),
                ((observables[2] - observables[0]) / parameters[4]),
                0
            )
            observables[5] = (observables[0] - observables[1]) / parameters[5]
            dxdt[0] = observables[3] - observables[4]
            dxdt[1] = observables[4] - observables[5]
            dxdt[2] = observables[5] - observables[3]


@cuda.jit(
            (
                numba_precision[:],
                numba_precision[:],
                numba_precision[:],
                numba_precision[:],
                numba_precision[:],
            ),
            device=True,
            inline=True,
        )
        def JvP(
            state,
            parameters,
            driver,
            observables,
            dxdt,
        ):
    # jacobian of dxdt wrt state + parameters, evaluated analytically with
    # operators swapped for save operators

