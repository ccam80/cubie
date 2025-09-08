# -*- coding: utf-8 -*-
"""Three chamber cardiovascular model.

This module implements a three chamber cardiovascular model based on
Antoine Pironet's thesis , suitable for CUDA execution.
"""

import numpy as np
from numba import cuda, from_dtype

from cubie.odesystems.baseODE import BaseODE, ODECache

default_parameters = {
    "E_h": 0.52,
    "E_a": 0.0133,
    "E_v": 0.0624,
    "R_i": 0.012,
    "R_o": 1.0,
    "R_c": 1 / 114,
    "V_s3": 2.0,
}

default_initial_values = {"V_h": 1.0, "V_a": 1.0, "V_v": 1.0}

default_observable_names = [
    "P_a",
    "P_v",
    "P_h",
    "Q_i",
    "Q_o",
    "Q_c",
]  # Flow in circulation

default_constants = {}


# noinspection PyPep8Naming
class ThreeChamberModel(BaseODE):
    """Three chamber cardiovascular model.

    A cardiovascular model with three chambers (heart, arteries, veins) as
    described in Antoine Pironet's thesis[1]_.

    Parameters
    ----------
    initial_values : dict, optional
        Initial values for state variables (V_h, V_a, V_v).
    parameters : dict, optional
        Parameter values for the system (E_h, E_a, E_v, R_i, R_o, R_c, V_s3).
    constants : dict, optional
        Constants that are not expected to change between simulations.
    observables : dict, optional
        Observable values to track (P_a, P_v, P_h, Q_i, Q_o, Q_c).
    precision : numpy.dtype, optional
        Precision to use for calculations, by default np.float64.
    default_initial_values : dict, optional
        Default initial values if not provided in initial_values.
    default_parameters : dict, optional
        Default parameter values if not provided in parameters.
    default_constants : dict, optional
        Default constant values if not provided in constants.
    default_observable_names : list, optional
        Default observable names if not provided in observables.
    num_drivers : int, optional
        Number of driver/forcing functions, by default 1.
    **kwargs : dict
        Additional arguments.

    Notes
    -----
    State variables:
    - V_h: Volume in heart
    - V_a: Volume in arteries
    - V_v: Volume in veins

    Parameters:
    - E_h: Elastance of Heart (e(t) multiplier)
    - E_a: Elastance of Arteries
    - E_v: Elastance of Ventricles
    - R_i: Resistance of input (mitral) valve
    - R_o: Resistance of output (atrial) valve
    - R_c: Resistance of circulation (arteries -> veins)
    - V_s3: Total stressed blood volume

    Observables:
    - P_a: Pressure in arteries
    - P_v: Pressure in veins
    - P_h: Pressure in heart
    - Q_i: Flow through input valve (Mitral)
    - Q_o: Flow through output valve (Aortic)
    - Q_c: Flow in circulation

    References
    ----------
    [1] A. Pironet. "Model-Based Prediction of the Response to Vascular
     Therapy." Unpublished doctoral thesis, ULiège - Université de Liège, 2016.
    https://hdl.handle.net/2268/194747
    """

    def __init__(
        self,
        initial_values=None,
        parameters=None,
        constants=None,
        observables=None,
        precision=np.float64,
        default_initial_values=default_initial_values,
        default_parameters=default_parameters,
        default_constants=default_constants,
        default_observable_names=default_observable_names,
        num_drivers=1,
        **kwargs,
    ):
        """Initialize the three chamber model.

        Parameters
        ----------
        initial_values : dict, optional
            Initial values for state variables.
        parameters : dict, optional
            Parameter values for the system.
        constants : dict, optional
            Constants that are not expected to change between simulations.
        observables : dict, optional
            Observable values to track.
        precision : numpy.dtype, optional
            Precision to use for calculations, by default np.float64.
        default_initial_values : dict, optional
            Default initial values if not provided in initial_values.
        default_parameters : dict, optional
            Default parameter values if not provided in parameters.
        default_constants : dict, optional
            Default constant values if not provided in constants.
        default_observable_names : list, optional
            Default observable names if not provided in observables.
        num_drivers : int, optional
            Number of driver/forcing functions, by default 1.
        **kwargs : dict
            Additional arguments.

        Notes
        -----
        num_drivers probably shouldn't be an instantiation parameter, but
        rather a property of the system.
        """
        super().__init__(
            initial_values=initial_values,
            parameters=parameters,
            constants=constants,
            observables=observables,
            default_initial_values=default_initial_values,
            default_parameters=default_parameters,
            default_constants=default_constants,
            default_observable_names=default_observable_names,
            precision=precision,
            num_drivers=num_drivers,
        )

    def build(self):
        """Build the CUDA device function for the three chamber model.

        Returns
        -------
        function
            Compiled CUDA device function implementing the three chamber
            cardiovascular dynamics.
        """
        # Hoist fixed parameters to global namespace
        global global_constants
        global_constants = self.compile_settings.constants.values_array.astype(
            self.precision
        )

        numba_precision = from_dtype(self.precision)

        # no cover: start
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
        def three_chamber_model_dv(
            state,
            parameters,
            driver,
            observables,
            out,
        ):  # pragma: no cover
            """Three chamber model dynamics implementation.

            Parameters
            ----------
            state : numpy.ndarray
                State vector [V_h, V_a, V_v] where:
                - V_h: Volume in heart
                - V_a: Volume in arteries
                - V_v: Volume in veins
            parameters : numpy.ndarray
                Parameter vector [E_h, E_a, E_v, R_i, R_o, R_c, V_s3] where:
                - E_h: Elastance of Heart (e(t) multiplier)
                - E_a: Elastance of Arteries
                - E_v: Elastance of Ventricles
                - R_i: Resistance of input (mitral) valve
                - R_o: Resistance of output (atrial) valve
                - R_c: Resistance of circulation (arteries -> veins)
                - V_s3: Total stressed blood volume
            driver : numpy.ndarray
                Driver/forcing array containing e(t) - current value of
                driver function.
            observables : numpy.ndarray
                Output array for observables [P_a, P_v, P_h, Q_i, Q_o, Q_c]:
                - P_a: Pressure in arteries
                - P_v: Pressure in veins
                - P_h: Pressure in heart
                - Q_i: Flow through input valve (Mitral)
                - Q_o: Flow through output valve (Aortic)
                - Q_c: Flow in circulation
            dxdt : numpy.ndarray
                Output array for state derivatives [dV_h, dV_a, dV_v]:
                - dV_h/dt = Q_i - Q_o
                - dV_a/dt = Q_o - Q_c
                - dV_v/dt = Q_c - Q_i

            Notes
            -----
            Modifications are made to the dxdt and observables arrays in-place
            to avoid allocating.
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

            out[0] = dV_h
            out[1] = dV_a
            out[2] = dV_v

        return ODECache(dxdt=three_chamber_model_dv)
        # no cover: stop

    def correct_answer_python(self, states, parameters, drivers):
        """Python version of the three chamber model for testing.

        More-direct port of Nic Davey's MATLAB implementation.

        Parameters
        ----------
        states : numpy.ndarray
            Current state values [V_h, V_a, V_v].
        parameters : numpy.ndarray
            Parameter values [E_h, E_a, E_v, R_i, R_o, R_c, V_s3].
        drivers : numpy.ndarray
            Driver/forcing values [e(t)].

        Returns
        -------
        tuple of numpy.ndarray
            A tuple containing (dxdt, observables) arrays.
        """
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

        dxdt = np.asarray(
            [Q_i - Q_o, Q_o - Q_c, Q_c - Q_i], dtype=self.precision
        )
        observables = np.asarray(
            [P_a, P_v, P_h, Q_i, Q_o, Q_c], dtype=self.precision
        )

        return dxdt, observables
