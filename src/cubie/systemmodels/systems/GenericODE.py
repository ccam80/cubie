# -*- coding: utf-8 -*-
"""
Created on Wed May 28 10:36:56 2025

@author: cca79
"""

import numpy as np
from numba import cuda, from_dtype

from cubie.CUDAFactory import CUDAFactory
from cubie.systemmodels.SystemValues import SystemValues
from cubie.systemmodels.systems.ODEData import ODEData


class GenericODE(CUDAFactory):
    """
    Template class for a system of ODEs. This class is designed to be
    subclassed for specific systems so that the
    "shared machinery" used to interface with CUDA can be reused. When subclassing, you should overload the build() and
    correct_answer_python() (if you want to implement testing) methods to provide the specific ODE system you want to
    simulate.

    If you do implement a correct_answer_python() method, then you can subclass the SystemTester class in
    tests/systemmodels/SystemTester.py and overload system_class with your ODE class name. The generate_system_tests
    function (see test_threeCM.py for an example) can then generate a set of floating-point and missing-input tests
    to see if your system behaves as expected.

    Most systems will contain a default set of initial values, parameters, constants, and observables (check the example in
    threeCM.py). This parent class does not contain them, but instead can be instantiated with a set of values of any
    size, for testing purposes. The default values provide a way to both set a default state and to provide the set
    of modifiable entries. This means that a user can't add in a state or parameter when "solving" the system that
    ends up
    having no effect on the system.
    """

    def __init__(self, initial_values=None, parameters=None,
                 # parameters that can change during simulation
                 constants=None,
                 # Parameters that are not expected to change during simulation
                 observables=None,
                 # Auxiliary variables you might want to track during simulation
                 default_initial_values=None, default_parameters=None,
                 default_constants=None, default_observable_names=None,
                 precision=np.float64, num_drivers=1, **kwargs, ):
        """Initialize the ODE system with initial values, parameters, and observables.

        Args:
            initial_values (dict, optional): Initial values for state variables. Default is None.
            parameters (dict, optional): Parameter values for the system. Default is None.
            constants (dict, optional): Constants that are not expected to change between simulations. Default is None.
            observables (dict, optional): Observable values to track. Default is None.
            default_initial_values (dict, optional): Default initial values if not provided in initial_values. Default is None.
            default_parameters (dict, optional): Default parameter values if not provided in parameters. Default is None.
            default_constants (dict, optional): Default constant values if not provided in constants. Default is None.
            precision (numpy.dtype, optional): Precision to use for calculations. Default is np.float64.
            **kwargs: Additional arguments
        """
        super().__init__()
        system_data = ODEData.from_genericODE_initargs(
                initial_values=initial_values, parameters=parameters,
                constants=constants, observables=observables,
                default_initial_values=default_initial_values,
                default_parameters=default_parameters,
                default_constants=default_constants,
                default_observable_names=default_observable_names,
                precision=precision, num_drivers=num_drivers, )
        self.setup_compile_settings(system_data)

    @property
    def parameters(self):
        """Get the parameters of the system.

        Returns:
            SystemValues: The parameters of the system.
        """
        return self.compile_settings.parameters

    @property
    def initial_values(self):
        """Get the initial values of the system.

        Returns:
            SystemValues: The initial values of the system.
        """
        return self.compile_settings.initial_states

    @property
    def observables(self):
        """Get the observables of the system.

        Returns:
            SystemValues: The observables of the system.
        """
        return self.compile_settings.observables

    @property
    def contants(self):
        """Get the constants of the system.

        Returns:
            SystemValues: The constants of the system.
        """
        return self.compile_settings.constants

    @property
    def num_states(self):
        return self.compile_settings.num_states

    @property
    def num_observables(self):
        return self.compile_settings.num_observables

    @property
    def num_parameters(self):
        return self.compile_settings.num_parameters

    @property
    def num_constants(self):
        return self.compile_settings.num_constants

    @property
    def num_drivers(self):
        return self.compile_settings.num_drivers

    @property
    def sizes(self):
        """Returns a dict of sizes (number of states, parameters, observables, constants, drivers) for the system."""
        return self.compile_settings.sizes

    @property
    def precision(self):
        """Get the precision of the system.

        Returns:
            The precision of the system (numba type, float32 or float64)
        """
        return self.compile_settings.precision

    @property
    def dxdt_function(self):
        return self.device_function

    def build(self):
        """Compile the dxdt system as a CUDA device function."""
        # Optimisation: Check whether this is being compiled-in already - we haven't declared anything global in the
        #  rest of the project, but can go back to declaring constants as global if the compiler is missing it. The
        #  only reason this might happen is if the system sees a numpy array and presumes that the user wants to pass
        #  in and copy back out, but that should only happen if the array is an argument.

        # Get dxdt contents into local scope, as the CUDA device function can't handle a reference to self
        constants = self.compile_settings.constants.values_array
        sizes = self.sizes
        n_params = sizes.parameters
        n_states = sizes.states
        n_obs = sizes.observables
        n_constants = sizes.constants
        n_drivers = sizes.drivers
        numba_precision = from_dtype(self.precision)

        @cuda.jit((numba_precision[:], numba_precision[:], numba_precision[:],
                   numba_precision[:], numba_precision[:]), device=True,
                inline=True, )
        def dummy_model_dxdt(state, parameters, drivers, observables, dxdt, ):
            """
            State (CUDA device array - local or shared for speed):
                The state of the system at time (t-1), used to calculate the differentials.

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
            # This placeholder model just overwrites the dxdt and observables arrays with the values in
            # state + parameters/constants+drivers.
            for i in range(n_states):
                if n_params > 0:
                    dxdt[i] = state[i] + parameters[i % n_params]
                else:
                    dxdt[i] = state[i]
            for i in range(n_obs):
                if n_constants > 0:
                    constant = constants[i % n_constants]
                else:
                    constant = numba_precision(0.0)
                if n_drivers > 0:
                    driver = drivers[i % n_drivers]
                else:
                    driver = numba_precision(0.0)

                observables[i] = constant + driver

        return dummy_model_dxdt

    def correct_answer_python(self, states, parameters, drivers):
        """This function is used in testing. Overload this with a simpler, Python version of the dxdt function.
        This will be run in a python test to compare the output of your CUDA function with this known, correct answer."""
        sizes = self.sizes

        n_parameters = sizes.parameters
        n_drivers = sizes.drivers
        n_constants = sizes.constants
        n_states = sizes.states
        n_observables = sizes.observables

        dxdt = np.zeros(n_states, dtype=self.precision)
        observables = np.zeros(n_observables, dtype=self.precision)

        if n_parameters <= 0:
            parameters = np.zeros(n_states, dtype=self.precision)
            n_parameters = n_states

        if n_drivers <= 0:
            drivers = np.zeros(n_observables, dtype=self.precision)
            n_drivers = n_observables

        if n_constants <= 0:
            _constants = np.zeros(n_observables, dtype=self.precision)
            n_constants = n_observables
        else:
            _constants = self.compile_settings.constants.values_array

        for i, state in enumerate(states):
            dxdt[i] = state + parameters[i % n_parameters]
        for i in range(len(observables)):
            observables[i] = drivers[i % n_drivers] + _constants[
                i % n_constants]

        return dxdt, observables

    def update(self, updates_dict, silent=False, **kwargs):
        """
        Pass updates to compile settings through the CUDAFactory interface, which will invalidate cache if an update
        is successful. Pass silent=True if doing a bulk update with other component's params to suppress warnings
        about keys not found.

        Args:
            silent (bool): If True, suppress warnings about unrecognized parameters
            **kwargs: Parameter updates to apply

        Returns:
            list: unrecognized_params
        """
        return self.set_constants(updates_dict, silent=silent, **kwargs)

    def set_constants(self, updates_dict=None, silent=False, **kwargs):
        """Update the constants of the system. Does not relabel parameters to constants, just updates values already
        compiled as constants and forces a rebuild with new compile-time constants.

        Args:
            updates_dict (dict): A dictionary of constant names and their new values.
        """
        if updates_dict is None:
            updates_dict = {}
        if kwargs:
            updates_dict.update(kwargs)
        if updates_dict == {}:
            return []

        const = self.compile_settings.constants
        recognised = const.update_from_dict(updates_dict, silent=True)
        unrecognised = set(updates_dict.keys()) - recognised
        self.update_compile_settings(constants=const, silent=True)

        if not silent and unrecognised:
            raise KeyError(
                    f"Unrecognized parameters in update: {unrecognised}. "
                    "These parameters were not updated.", )

        return recognised
