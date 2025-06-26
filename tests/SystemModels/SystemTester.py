import pytest
import numpy as np
from numba import cuda
from numpy.testing import assert_allclose

class SystemTester:
    """Class to test common functionality of system models. Subclass this to add your model as self.system, and overload
    the "correct_answer" fixture to provide the expected output for a given input."""

    @pytest.fixture(scope="class")
    def system_class(self):
        return None

    def instantiate_system(self,
                       system_class,
                       precision,
                       states_list,
                       params_list,
                       obs_list,
                       constants_dict,
                       num_drivers):
        self.system_instance = system_class(states_list,
                                            params_list,
                                            constants_dict,
                                            obs_list,
                                            num_drivers=num_drivers,
                                            precision=precision)

    def build_system(self):
        """Build or compile the system if needed."""
        self.system_instance.build()

    def build_test_kernel(self):
        """Builds a CUDA kernel used to check output correctness."""
        precision = self.system_instance.precision
        dxdt_func = self.system_instance.dxdtfunc

        n_states = self.system_instance.num_states
        n_par = self.system_instance.num_parameters
        n_obs = self.system_instance.num_observables
        n_drivers = self.system_instance.num_drivers

        @cuda.jit()
        def test_kernel(outarray, obs_array, input_arr, param_arr, drivers_arr):
            l_dxdt = cuda.local.array(shape=n_states, dtype=precision)
            l_states = cuda.local.array(shape=n_states, dtype=precision)
            l_parameters = cuda.local.array(shape=n_par, dtype=precision)
            l_observables = cuda.local.array(shape=n_obs, dtype=precision)
            l_driver = cuda.local.array(shape=n_drivers, dtype=precision)

            for i in range(n_par):
                l_parameters[i] = param_arr[i]
            for i in range(n_states):
                l_states[i] = input_arr[i]
            for i in range(n_drivers):
                l_driver[i] = drivers_arr[i]

            for i in range(n_states):
                l_dxdt[i] = precision(0.0)

            dxdt_func(l_states, l_parameters, l_driver, l_observables, l_dxdt)

            for i in range(n_states):
                outarray[i] = l_dxdt[i]
            for i in range(n_obs):
                obs_array[i] = l_observables[i]

        self.test_kernel = test_kernel

    def test_instantiation(self, system_class, instantiate_settings, input_data, test_name):
        """Checks if system instantiates without errors for valid sets."""
        precision, s_list, p_list, o_list, const_dict, n_drivers = instantiate_settings
        self.instantiate_system(system_class, precision, s_list, p_list, o_list, const_dict, n_drivers)
        assert isinstance(self.system_instance, system_class), \
            "System did not instantiate as expected."

    def test_compilation(self, system_class, instantiate_settings, input_data, test_name):
        """Checks if the system builds or compiles."""
        precision, s_list, p_list, o_list, const_dict, n_drivers = instantiate_settings
        self.instantiate_system(system_class, precision, s_list, p_list, o_list, const_dict, n_drivers)
        self.build_system()
        assert self.system_instance.dxdtfunc is not None, "dxdt function missing after build."

    def test_correct_output(self, system_class, instantiate_settings, input_data, test_name):
        """Checks if the output matches expected values."""
        precision, s_list, p_list, o_list, const_dict, n_drivers = instantiate_settings
        self.instantiate_system(system_class, precision, s_list, p_list, o_list, const_dict, n_drivers)
        self.build_system()
        self.build_test_kernel()

        dx = np.zeros(self.system_instance.num_states, dtype=precision)
        observables = np.zeros(self.system_instance.num_observables, dtype=precision)

        self.test_kernel[1, 1](dx, observables, input_data[0], input_data[1], input_data[2])
        expected_dx, expected_obs = self.correct_answer(instantiate_settings, input_data)
        assert_allclose(dx, expected_dx, err_msg="dx mismatch")
        assert_allclose(observables, expected_obs, err_msg="observables mismatch")

    def correct_answer(self, instantiate_settings, input_data):
        """By default returns zeros; subclasses can override."""
        n_states = self.system_instance.num_states
        n_obs = self.system_instance.num_observables
        return np.zeros(n_states), np.zeros(n_obs)

