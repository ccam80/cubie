import pytest
from numba import cuda
from numpy.testing import assert_allclose
from cubie.systemmodels.symbolic.parser import create_system
import numpy as np

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
                 "obs2 = c2 * c2 * k2 + x1 + x2 ** 2 + obs1",
                 "dx1 = obs1 + c2",
                 "dx2 = c1 + obs1 + obs2"],
    }

@pytest.fixture(scope="function")
def sanity_check(manual_settings, precision):
    k1 = manual_settings["parameters"]["k1"]
    k2 = manual_settings["parameters"]["k2"]
    c1 = manual_settings["constants"]["c1"]
    c2 = manual_settings["constants"]["c2"]
    d1 = manual_settings["drivers"]["d1"]
    d2 = manual_settings["drivers"]["d2"]
    x1 = manual_settings["states"]["x1"]
    x2 = manual_settings["states"]["x2"]

    obs1 = k1 * x1 * d2 + d1 * c1
    obs2 = c2 * c2 * k2 + x1 + x2 ** 2 + obs1
    dx1 = obs1 + c2
    dx2 = c1 + obs1 + obs2

    return (np.asarray([dx1,dx2], dtype=precision),
            np.asarray([obs1, obs2], dtype=precision))

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
    symbolic_system = create_system(**manual_settings)
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


@pytest.fixture(scope="function")
def test_kernel(symbolic_system):
    func = symbolic_system.dxdt
    @cuda.jit()
    def test_kernel(state,
                    parameters,
                    drivers,
                    observables,
                    dxdt):
        func(state, parameters, drivers, observables, dxdt)
    return test_kernel

@pytest.fixture(scope="function")
def v(symbolic_system, input_values):
    test = np.zeros(symbolic_system.sizes.states, dtype=symbolic_system.precision)
    test[:] = 1.0  # Small perturbation for finite difference approximation
    return cuda.to_device(test)

@pytest.fixture(scope="function")
def jacv_kernel(symbolic_system, input_values, v):
    jacv = symbolic_system.jac_v
    @cuda.jit()
    def test_kernel(state,
                    parameters,
                    drivers,
                    v,
                    Jv):
        jacv(state, parameters, drivers, v, Jv)
    return test_kernel

@pytest.fixture(scope="function")
def run_jac(symbolic_system, device_arrays, v, jacv_kernel):
    Jv = cuda.device_array(symbolic_system.sizes.states, dtype=symbolic_system.precision)
    jacv_kernel[1,1](device_arrays["state"],
                     device_arrays["parameters"],
                     device_arrays["drivers"],
                     v,
                     Jv)
    return Jv.copy_to_host()

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
    run_dxdt, manual_settings, symbolic_system, correct_answer, sanity_check
):
    dxdt_expected, obs_expected = correct_answer
    dxdt_actual, obs_actual = run_dxdt
    dxdt_manual, obs_manual = sanity_check
    print(dxdt_expected)

    assert_allclose(dxdt_manual, dxdt_expected, rtol=1e-6, atol=1e-6)
    assert_allclose(obs_manual, obs_expected, rtol=1e-6, atol=1e-6)
    assert_allclose(dxdt_expected, dxdt_actual, rtol=1e-6, atol=1e-6)
    assert_allclose(obs_expected, obs_actual, rtol=1e-6, atol=1e-6)

@pytest.mark.parametrize("precision_override", [np.float32, np.float64],
                         indirect=True)
def test_jac_v_output(
        run_dxdt, run_jac, test_kernel, manual_settings, device_arrays,
        symbolic_system, v, correct_answer, precision):

    dxdt_actual, _ = run_dxdt

    dxdt_new = cuda.device_array(symbolic_system.sizes.states, dtype=precision)
    state = device_arrays['state']
    current_state = state.copy_to_host()
    Jv = run_jac
    v_host = v.copy_to_host()
    current_state = current_state + v_host
    state = cuda.to_device(current_state)

    test_kernel[1, 1](
        state,
        device_arrays["parameters"],
        device_arrays["drivers"],
        device_arrays["observables"],
        dxdt_new,
    )

    dxdt_fwd_host = dxdt_new.copy_to_host()
    current_state = current_state - 2 * v_host
    state = cuda.to_device(current_state)
    test_kernel[1, 1](
        state,
        device_arrays["parameters"],
        device_arrays["drivers"],
        device_arrays["observables"],
        dxdt_new,
    )

    dxdt_back_host = dxdt_new.copy_to_host()
    fwd_diff = dxdt_fwd_host - dxdt_actual
    back_diff = dxdt_actual - dxdt_back_host
    Jv = Jv
    dxdt_central_diff = (dxdt_fwd_host - dxdt_back_host) / 2
    error = Jv - dxdt_central_diff
    print(error)
    assert_allclose(Jv, dxdt_central_diff, rtol=1e-5, atol=1e-5)
