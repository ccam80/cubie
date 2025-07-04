import pytest
import numpy as np
from tests.SystemModels.SystemTester import SystemTester
from numpy.testing import assert_allclose
from tests.SystemModels._utils import generate_system_tests
from CuMC.SystemModels.Systems.decays import Decays

from tests._utils import generate_test_array
# from tests.SystemModels._utils import random_system_values


# One-off modifications of _utils for this unique system:

def random_system_values(coeffs, precision=np.float64, randscale=1e6):
    n = len(coeffs)

    inits = generate_test_array(precision, n, style='random', scale=randscale)
    params = generate_test_array(precision, n, style='random', scale=randscale)
    drivers = generate_test_array(precision, 1, style='random', scale=randscale)

    return inits, params, drivers


def create_random_test_set(coeffs, precision=np.float64, randscale=1e6):
    inits, params, drivers = random_system_values(coeffs, precision, randscale)

    input_data = (inits, params, drivers)
    instantiation_parameters = (precision, coeffs)
    return (instantiation_parameters, input_data,
            f"Random test set with {len(params)} coefficients of scale {randscale} of type {precision}")


def generate_decays_tests(param_lengths=[1, 10, 100], log10_scalerange=(-6, 6), tests_per_category=3):
    test_cases = []
    coeffses = [list(range(n)) for n in param_lengths]
    samescales = np.arange(log10_scalerange[0], log10_scalerange[1] + 1, tests_per_category)

    precisions = (np.float32, np.float64)
    for precision in (precisions):
        for coeffs in coeffses:
            coeffs = [precision(c) for c in coeffs]
            #single-scale random tests
            test_cases += [create_random_test_set(coeffs, precision, 10.0 ** scale) for scale in samescales]
            # mixed-scale random tests
            test_cases += [create_random_test_set(coeffs, precision, log10_scalerange) for scale in
                           range(tests_per_category)]

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
        self.build_system()
        assert self.system_instance.dxdtfunc is not None, "dxdt function missing after build."

    def test_correct_output(self, system_class, instantiate_settings, input_data, test_name):
        """Checks if the output matches expected values."""
        precision, coefficients = instantiate_settings
        self.instantiate_system(system_class, coefficients, precision)
        self.build_system()
        self.build_test_kernel()

        dx = np.zeros(self.system_instance.num_states, dtype=precision)
        observables = np.zeros(self.system_instance.num_observables, dtype=precision)

        self.test_kernel[1, 1](dx, observables, input_data[0], input_data[1], input_data[2])
        expected_dx, expected_obs = self.system_instance.correct_answer_python(*input_data)
        if precision == np.float32:
            rtol = 1e-5 #float32 will underperform in fixed-precision land, and on big systems this error will stack
        else:
            rtol = 1e-12
        assert_allclose(dx, expected_dx, rtol=rtol, err_msg="dx mismatch")
        assert_allclose(observables, expected_obs, rtol=rtol, err_msg="observables mismatch")







