# -*- coding: utf-8 -*-
"""
Created on Wed May 28 10:36:56 2025

@author: cca79
"""


from numba import cuda, from_dtype, float32, float64

from CuMC.SystemModels.SystemValues import SystemValues

import numpy as np



#TODO: Rethink this default values idea - the possible values should definitely be defined in this file, but how?
# default_initial_values = {'x0': 1.0}
# default_parameters = {'p0': 2.0}
# default_constants = {'c0': 0.0}
# default_observables = {'o0'}


class GenericODE:
    """
    Template class for a system of ODEs. This class is designed to be subclassed for specific systems, so that the
    "shared machinery" used to interface with CUDA can be reused. When subclassing, you should overload the build() and
    correct_answer_python() (if you want to implement testing) methods to provide the specific ODE system you want to
    simulate.

    If you do implement a correct_answer_python() method, then you can subclass the SystemTester class in
    tests/SystemModels/SystemTester.py and overload system_class with your ODE class name. The generate_system_tests
    function (see test_threeCM.py for an example) can then generate a set of floating-point and missing-input tests
    to see if your system behaves as expected.
    """


    def __init__(self,
                 initial_values=None,
                 parameters=None, #parameters that can change during simulation
                 constants=None, #Parameters that are not expected to change during simulation
                 observables=None, #Auxiliary variables you might want to track during simulation
                 default_initial_values=None,
                 default_parameters=None,
                 default_constants=None,
                 default_observable_names=None,
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
        self.observables = SystemValues(observables, precision, default_observable_names)
        self.constants = SystemValues(constants, precision, default_constants)

        if precision in [np.float64, np.float32]:
            self.precision = from_dtype(precision)
        elif precision in [float32, float64]:
            self.precision = precision
        else:
            raise ValueError("Precision must be a numpy or numba dtype (float32/float64).")

        self.dxdtfunc = None

        self.num_states = self.init_values.n
        self.num_parameters = self.parameters.n
        self.num_observables = self.observables.n
        self.num_drivers = num_drivers
        self.num_constants = self.constants.n

        self.needs_compilation = True  #Set to false when system is updated, so that an out-of-date system isn't used

    def build(self):
        """Compile the dxdt system as a CUDA device function."""
        # Hoist fixed parameters to global namespace
        if not self.needs_compilation:
            return self.dxdtfunc
        else:
            # get loop-length parameters as local variables so they are treated as compile-time constants - the compiler
            # can't handle any references to self.
            global constants
            constants = self.constants.values_array
            n_params = self.num_parameters
            n_states = self.num_states
            n_obs = self.num_observables
            n_constants = self.num_constants
            n_drivers = self.num_drivers
            precision = self.precision

            @cuda.jit(
                (self.precision[:],
                       self.precision[:],
                       self.precision[:],
                       self.precision[:],
                       self.precision[:]),
                         device=True,
                        inline=True)
            def dummy_model_dxdt(state,
                                 parameters,
                                 drivers,
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

                # the dummy model just overwrites the dxdt and observables arrays with the values in
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
                        constant = precision(0.0)
                    if n_drivers > 0:
                        driver = drivers[i % n_drivers]
                    else:
                        driver = precision(0.0)

                    observables[i] = constant + driver

            self.dxdtfunc = dummy_model_dxdt

            self.needs_compilation = False
            # del constants

    def correct_answer_python(self, states, parameters, drivers):
        """This function is used in testing. Overload this with a simpler, Python version of the dxdt function.
        This will be run in a python test to compare the output of your CUDA function with this known, correct answer."""
        numpy_precision = np.float64 if self.precision==float64 else np.float32
        dxdt = np.zeros(self.num_states, dtype=numpy_precision)
        observables = np.zeros(self.num_observables, dtype=numpy_precision)

        n_parameters = self.num_parameters
        n_drivers = self.num_drivers
        n_constants = self.num_constants
        n_states = self.num_states
        n_observables = self.num_observables

        if n_parameters <= 0:
            parameters = np.zeros(n_states, dtype=numpy_precision)
            n_parameters = n_states

        if n_drivers <= 0:
            drivers = np.zeros(n_observables, dtype=numpy_precision)
            n_drivers = n_observables

        if n_constants <= 0:
            constants = np.zeros(n_observables, dtype=numpy_precision)
            n_constants = n_observables
        else:
            constants= self.constants.values_array

        for i, state in enumerate(states):
            dxdt[i] = state + parameters[i % n_parameters]
        for i in range(len(observables)):
            observables[i] = drivers[i % n_drivers] + constants[i%n_constants]

        return dxdt, observables


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
