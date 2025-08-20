import pytest
from numba import cuda
from numpy.testing import assert_allclose
from cubie.systemmodels.symbolic import setup_system


def manual_settings_defaults():
    return {
        "observables": ["obs1", "obs2"],
        "parameters": {"k1": 0.32,
                       'k2':0.91},
        "constants": {'c1': 2.1,
                      'c2': 1.8},
        "drivers": {"d1": 0.9,
                    'd2': 0.8},
        "states": {"x1": 0.5,
                   'x2': 2.0},
        "dxdt": ["obs1 = k1 * x1 * d2 + d1 * c1",
                 "obs2 = c2 * c2 * k2 + x1",
                 "dx1 = obs1 + c2",
                 "dx2 = c1"],
    }


@pytest.fixture(scope="function")
def manual_settings_override(request):
    return request.param if hasattr(request, "param") else {}


@pytest.fixture(scope="function")
def manual_settings(manual_settings_override):
    settings = manual_settings_defaults()
    settings.update(manual_settings_override)
    return settings

@pytest.fixture(scope="function")
def symbolic_system(manual_settings, precision):
    symbolic_system = setup_system(**manual_settings)
    return symbolic_system

@pytest.fixture(scope="function")
def input_values(manual_settings, precision):
    init_vals = [value for value in manual_settings["states"].values()]
    param_vals = [value for value in manual_settings["parameters"].values()]
    driver_vals = [value for value in manual_settings["drivers"].values()]
    constants = [
        precision(value) for value in manual_settings["constants"].values()
    ]
    return init_vals, param_vals, driver_vals, constants

@pytest.fixture(scope="function")
def device_arrays(symbolic_system, manual_settings, precision):
    sizes = symbolic_system.sizes
    state = cuda.device_array(sizes.states, dtype=precision)
    obs = cuda.device_array(sizes.observables, dtype=precision)
    dxdt = cuda.device_array(sizes.states, dtype=precision)
    parameters = cuda.device_array(sizes.parameters, dtype=precision)
    drivers = cuda.device_array(sizes.drivers, dtype=precision)
    init_vals = [value for value in manual_settings["states"].values()]
    param_vals = [value for value in manual_settings["parameters"].values()]
    driver_vals = [value for value in manual_settings["drivers"].values()]
    constants = [precision(value) for value in manual_settings[
        "constants"].values()]

    for i in range(sizes.states):
        state[i] = init_vals[i]
    for i in range(sizes.parameters):
        parameters[i] = param_vals[i]
    for i in range(sizes.drivers):
        drivers[i] = driver_vals[i]

    dxdt[:] = precision(0.0)
    obs[:] = precision(0.0)

    return {"state": state,
            "observables": obs,
            "parameters": parameters,
            "drivers": drivers,
            "dxdt": dxdt,
            "constants": constants}

@pytest.fixture(scope="function")
def correct_answer(symbolic_system, input_values):
    inits, params, drivers, _ = input_values
    dxdt, obs = symbolic_system.correct_answer_python(inits, params, drivers)
    return dxdt, obs

# def test_setup_system(symbolic_system, precision, correct_answer):
#     symbolic_system.build()
#
#     dxdt, obs = correct_answer
#     assert dxdt[0] == pytest.approx(6.0)
#     assert obs[0] == pytest.approx(6.0)

@pytest.fixture(scope="function")
def test_kernel(symbolic_system):
    func = symbolic_system.device_function
    @cuda.jit()
    def test_kernel(state,
                    parameters,
                    drivers,
                    observables,
                    dxdt):
        func(state, parameters, drivers, observables, dxdt)
    return test_kernel

@pytest.fixture(scope="function")
def run_dxdt(test_kernel, device_arrays):
    test_kernel[1,1](device_arrays["state"],
                device_arrays["parameters"],
                device_arrays["drivers"],
                device_arrays["observables"],
                device_arrays["dxdt"])
    observables = device_arrays["observables"].copy_to_host()
    dxdt = device_arrays["dxdt"].copy_to_host()
    return dxdt, observables

def test_dxdt_output(
    run_dxdt, manual_settings, symbolic_system, correct_answer
):
    dxdt_expected, obs_expected = correct_answer
    dxdt_actual, obs_actual = run_dxdt
    print(dxdt_expected)
    assert_allclose(dxdt_expected, dxdt_actual, rtol=1e-6, atol=1e-6)
    assert_allclose(obs_expected, obs_actual, rtol=1e-6, atol=1e-6)

def test_jacobian_outpu(
        symbolic_system, input_values, device_arrays, precision, run_dxdt
):
    jac = symbolic_system._jacobian