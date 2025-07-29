import pytest
import numpy as np
from tests.SystemModels.Systems.SystemTester import SystemTester
from numpy.testing import assert_allclose
from CuMC.SystemModels.Systems.decays import Decays

from tests._utils import generate_test_array


# One-off modifications of _utils for this unique system:

def random_system_values(coeffs, precision=np.float64, randscale=1e6):
    n = len(coeffs)

    inits = generate_test_array(precision, n, style='random', scale=randscale)
    params = generate_test_array(precision, n, style='random', scale=randscale)
    drivers = generate_test_array(precision, 1, style='random', scale=randscale)

    return inits, params, drivers


def create_random_test_set(coeffs, precision=np.float64, randscale=1e6):
    coeffs = [precision(c) for c in range(coeffs)]
    inits, params, drivers = random_system_values(coeffs, precision, randscale)
    input_data = (inits, params, drivers)
    instantiation_parameters = (precision, coeffs)
    return (instantiation_parameters, input_data,
            f"Random test set with {len(params)} coefficients of scale {randscale} of type {precision}")


def generate_decays_tests(param_lengths=[1, 100], log10_scalerange=(-6, 6), range_step=6):
    test_cases = []
    test_cases += [create_random_test_set(10, np.float32, 10.0 ** -6)]
    test_cases += [create_random_test_set(10, np.float32, 10.0 ** 0)]
    test_cases += [create_random_test_set(10, np.float32, 10.0 ** 6)]
    test_cases += [create_random_test_set(100, np.float32, (-6,6))]
    test_cases += [create_random_test_set(100, np.float64, (-6,6))]

    return test_cases

testsets = generate_decays_tests()
@pytest.mark.parametrize("instantiate_settings, input_data, test_name",
                         testsets,
                         ids=[testset[2] for testset in testsets])
class TestDecays(SystemTester):
    """Decays is a special tester function with minimal instantiation parameters, so requires overloading a lot of
    SystemTesters methods.."""

    @pytest.fixture(scope="class", autouse=True)
    def system_class(self):
        return Decays

    def instantiate_system(self, system_class, coefficients, precision=np.float32):
        """Instantiate the decays system with default parameters."""
        self.system_instance = system_class(precision=precision, coefficients=coefficients)
        return self.system_instance

    def test_instantiation(self, system_class, instantiate_settings, input_data, test_name):
        """Checks if system instantiates without errors for valid sets."""
        precision, coefficients = instantiate_settings
        self.instantiate_system(system_class, coefficients, precision)
        assert isinstance(self.system_instance, system_class), \
            "System did not instantiate as expected."

    def test_compilation(self, system_class, instantiate_settings, input_data, test_name):
        """Checks if the system builds or compiles."""
        precision, coefficients = instantiate_settings
        self.instantiate_system(system_class, coefficients, precision)
        dxdt_function = self.build_system()
        assert dxdt_function is not None, "dxdt function missing after build."

    @pytest.mark.nocudasim
    def test_correct_output(self, system_class, instantiate_settings, input_data, test_name):
        """Checks if the output matches expected values."""
        precision, coefficients = instantiate_settings
        self.instantiate_system(system_class, coefficients, precision)
        self.build_test_kernel()

        dx = np.zeros(self.system_instance.sizes.states, dtype=precision)
        observables = np.zeros(self.system_instance.sizes.observables, dtype=precision)

        self.test_kernel[1, 1](dx, observables, input_data[0], input_data[1], input_data[2])
        expected_dx, expected_obs = self.system_instance.correct_answer_python(*input_data)
        if precision == np.float32:
            rtol = 5e-5 # float32 will underperform in fixed-precision land, and on big systems this error will stack
        else:
            rtol = 1e-12
        assert_allclose(dx, expected_dx, rtol=rtol, err_msg="dx mismatch")
        assert_allclose(observables, expected_obs, rtol=rtol, err_msg="observables mismatch")

    @pytest.mark.nocudasim
    def test_constants_edit(self, system_class, instantiate_settings, input_data, test_name):
        """ Checks if constant edits are successfully compiled into the system. """
        precision, coefficients = instantiate_settings

        self.instantiate_system(system_class, coefficients, precision)
        self.build_test_kernel()

        if self.system_instance.sizes.constants == 0:
            pytest.skip("No constants to edit in this system.")

        dx = np.zeros(self.system_instance.sizes.states, dtype=precision)
        observables = np.zeros(self.system_instance.sizes.observables, dtype=precision)

        self.test_kernel[1, 1](dx, observables, input_data[0], input_data[1], input_data[2])

        if precision == np.float32:
            rtol = 1e-5  # float32 will underperform in fixed-precision land, and on big systems this error will stack
        else:
            rtol = 1e-12

        expected_dx, expected_obs = self.system_instance.correct_answer_python(*input_data)

        assert_allclose(dx, expected_dx, rtol=rtol, err_msg="initial dx mismatch")
        assert_allclose(observables, expected_obs, rtol=rtol, err_msg="initial observables mismatch")

        for key, value in self.system_instance.compile_settings.constants.values_dict.items():
            self.system_instance.set_constants({key: value * 10.0})

        expected_dx, expected_obs = self.system_instance.correct_answer_python(*input_data)
        with pytest.raises(AssertionError):
            assert_allclose(observables, expected_dx, rtol=rtol, err_msg="pre-rebuild mismatch - expected")

        self.build_test_kernel()
        expected_dx, expected_obs = self.system_instance.correct_answer_python(*input_data)
        self.test_kernel[1, 1](dx, observables, input_data[0], input_data[1], input_data[2])

        assert_allclose(dx, expected_dx, rtol=rtol, err_msg="post-edit dx mismatch")
        assert_allclose(observables, expected_obs, rtol=rtol, err_msg="post-edit observables mismatch")
