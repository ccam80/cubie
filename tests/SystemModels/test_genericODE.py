import numpy as np


#TODO: sort these tests into a better structure, with separate tests for each situation. Paramaterise.
def test_genericODE_getters_setters():
    """Test the getter and setter methods for parameters and initial values in genericODE."""
    from CuMC.SystemModels.genericODE import genericODE

    # Create a genericODE instance with some initial values and parameters
    initial_values = {"V_h": 1.0, "V_a": 2.0, "V_v": 3.0}
    parameters = {"E_h": 4.0, "E_a": 5.0, "E_v": 6.0, "R_i": 7.0, "R_o": 8.0, "R_c": 9.0, "SBV": 10.0}
    precision = np.float32
    ode = genericODE(initial_values=initial_values, parameters=parameters, precision=precision)

    # Test get_parameter
    assert ode.get_parameters("E_h") == 4.0
    assert ode.get_parameters("E_a") == 5.0
    assert np.array_equal(ode.get_parameters(["E_v", "R_i"]), np.asarray([6.0, 7.0], dtype=precision))

    # Test set_parameter
    ode.set_parameters("E_h", 11.0)
    assert ode.get_parameters("E_h") == 11.0

    # Test get_initial_value
    assert ode.get_initial_values("V_h") == 1.0
    assert ode.get_initial_values("V_a") == 2.0
    assert np.array_equal(ode.get_initial_values(["V_v", "V_h"]), np.asarray([3.0, 1.0], dtype=precision))

    # Test set_initial_value
    ode.set_initial_values("V_h", 12.0)
    assert ode.get_initial_values("V_h") == 12.0
