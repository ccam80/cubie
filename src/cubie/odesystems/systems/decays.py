# -*- coding: utf-8 -*-
"""Exponential decay ODE system for testing.

This module provides a simple ODE system where each state variable decays
exponentially at a rate proportional to its position.
"""

import numpy as np
from numba import cuda, from_dtype

from cubie.odesystems.baseODE import BaseODE, ODECache


class Decays(BaseODE):
    """Exponential decay system for testing purposes.

    A system where each state variable decays exponentially at a rate
    proportional to its position. Observables are the same as state variables
    * parameters (coefficients) + index.

    Parameters
    ----------
    precision : numpy.dtype, optional
        Data type for calculations, by default np.float64.
    **kwargs : dict
        Must contain 'coefficients' key with list of decay coefficients.

    Notes
    -----
    If coefficients = [c1, c2, c3], then the system will have three state
    variables x0, x1, x2, and:

    dx[0] = -x[0]/1,
    dx[1] = x[1]/2,
    dx[2] = x[2]/3

    obs[0] = dx[0]*c1 + 1 + step_count,
    obs[1] = dx[1]*c2 + 2 + step_count,
    obs[2] = dx[2]*c3 + 3 + step_count.

    Really just exists for testing.
    """

    def __init__(
        self,
        precision=np.float64,
        **kwargs,
    ):
        """Initialize the decay system.

        Parameters
        ----------
        precision : numpy.dtype, optional
            Data type for calculations, by default np.float64.
        **kwargs : dict
            Must contain 'coefficients' key with list of decay coefficients.
        """
        coefficients = kwargs["coefficients"]

        nterms = len(coefficients)
        observables = [f"o{i}" for i in range(nterms)]
        initial_values = {f"x{i}": 1.0 for i in range(nterms)}
        parameters = {f"p{i}": coefficients[i] for i in range(nterms)}
        constants = {f"c{i}": i for i in range(nterms)}
        n_drivers = 1  # use time as the driver

        super().__init__(
            initial_values=initial_values,
            parameters=parameters,
            constants=constants,
            observables=observables,
            precision=precision,
            num_drivers=n_drivers,
        )

    def build(self):
        """Build the CUDA device function for the decay system.

        Returns
        -------
        function
            Compiled CUDA device function implementing the decay dynamics.
        """
        # Hoist fixed parameters to global namespace
        global global_constants
        global_constants = self.compile_settings.constants.values_array
        n_terms = self.sizes.states
        numba_precision = from_dtype(self.precision)

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
        def dxdtfunc(
            state,
            parameters,
            driver,
            observables,
            dxdt,
        ):
            """Decay dynamics implementation.

            Parameters
            ----------
            state : numpy.ndarray
                Current state values.
            parameters : numpy.ndarray
                Parameter values (coefficients).
            driver : numpy.ndarray
                Driver/forcing values (time).
            observables : numpy.ndarray
                Output array for observable values.
            dxdt : numpy.ndarray
                Output array for state derivatives.

            Notes
            -----
            dx[i] = state[i] / (i+1)
            observables[i] = state[i] * parameters[i] + constants[i] + driver[0]
            """
            for i in range(n_terms):
                dxdt[i] = -state[i] / (i + 1)
                observables[i] = (
                    dxdt[i] * parameters[i] + global_constants[i] + driver[0]
                )
        return ODECache(dxdt=dxdtfunc)

    def correct_answer_python(self, states, parameters, drivers):
        """Python testing function.

        Do it in python and compare results with CUDA implementation.

        Parameters
        ----------
        states : numpy.ndarray
            Current state values.
        parameters : numpy.ndarray
            Parameter values.
        drivers : numpy.ndarray
            Driver values.

        Returns
        -------
        tuple of numpy.ndarray
            A tuple containing (dxdt, observables) arrays.
        """
        indices = np.arange(len(states))
        observables = np.zeros(self.sizes.observables)
        dxdt = -states / (indices + 1)

        for i in range(self.sizes.observables):
            observables[i] = (
                dxdt[i % self.sizes.states]
                * parameters[i % self.sizes.parameters]
                + drivers[0]
                + global_constants[i]
            )

        return dxdt, observables
