import numpy as np
import pytest
from numba import cuda, from_dtype
from numpy.testing import assert_allclose

from tests.integrators.cpu_reference import get_ref_step_function

STATUS_MASK = 0xFFFF


def _device_step(
    step_object,
    solver_settings,
    cpu_system,
    system,
    precision,
    initial_state,
    cpu_driver_evaluator,
    driver_array,
):
    if driver_array is None:
        pytest.skip("driver_array fixture returned None")

    step_function = step_object.step_function
    step_size = solver_settings["dt_min"]
    n_states = system.sizes.states
    params = system.parameters.values_array.astype(precision)
    state = np.asarray(initial_state, dtype=precision)
    drivers = cpu_driver_evaluator.evaluate(precision(0.0))
    driver_coefficients = np.array(
        driver_array.coefficients, dtype=precision, copy=True
    )
    observables = cpu_system.observables(state, params, drivers, precision(
            0.0))
    proposed_state = np.zeros_like(state)
    work_len = max(step_object.local_scratch_required, n_states)
    work_buffer = np.zeros(work_len, dtype=precision)
    error = np.zeros(n_states, dtype=precision)
    status = np.zeros(1, dtype=np.int32)

    shared_elems = step_object.shared_memory_required
    shared_bytes = np.dtype(precision).itemsize * shared_elems
    persistent_len = max(1, step_object.persistent_local_required)
    numba_precision = from_dtype(precision)
    dt_value = precision(step_size)

    d_state = cuda.to_device(state)
    d_proposed = cuda.to_device(proposed_state)
    d_work = cuda.to_device(work_buffer)
    d_params = cuda.to_device(params)
    d_driver_coeffs = cuda.to_device(driver_coefficients)
    d_drivers = cuda.to_device(drivers)
    proposed_drivers = np.zeros_like(drivers)

    d_observables = cuda.to_device(observables)
    d_proposed_observables = cuda.to_device(observables)
    d_proposed_drivers = cuda.to_device(proposed_drivers)
    d_error = cuda.to_device(error)
    d_status = cuda.to_device(status)

    threads = step_object.threads_per_step

    @cuda.jit
    def kernel(
        state_vec,
        proposed_vec,
        work_vec,
        params_vec,
        driver_coeffs_vec,
        drivers_vec,
        proposed_drivers_vec,
        observables_vec,
        proposed_observables_vec,
        error_vec,
        status_vec,
        dt_scalar,
        time_scalar,
    ):
        idx = cuda.grid(1)
        if idx > 0:
            return
        shared = cuda.shared.array(shared_elems, dtype=numba_precision)
        persistent = cuda.local.array(persistent_len, dtype=numba_precision)
        result = step_function(
            state_vec,
            proposed_vec,
            work_vec,
            params_vec,
            driver_coeffs_vec,
            drivers_vec,
            proposed_drivers_vec,
            observables_vec,
            proposed_observables_vec,
            error_vec,
            dt_scalar,
            time_scalar,
            shared,
            persistent,
        )
        status_vec[0] = result

    kernel[1, threads, 0, shared_bytes](
        d_state,
        d_proposed,
        d_work,
        d_params,
        d_driver_coeffs,
        d_drivers,
        d_proposed_drivers,
        d_observables,
        d_proposed_observables,
        d_error,
        d_status,
        dt_value,
        numba_precision(0.0),
    )
    cuda.synchronize()

    status_value = int(d_status.copy_to_host()[0])
    return {
        "state": d_proposed.copy_to_host(),
        "observables": d_proposed_observables.copy_to_host(),
        "error": d_error.copy_to_host(),
        "status": status_value & STATUS_MASK,
        "niters": (status_value >> 16) & STATUS_MASK,
    }


def _cpu_step(
    solver_settings,
    cpu_system,
    implicit_step_settings,
    initial_state,
    driver_array,
    cpu_driver_evaluator,
):
    step_function = get_ref_step_function("crank_nicolson")
    dt_value = solver_settings["dt_min"]
    state = np.asarray(initial_state, dtype=cpu_system.precision)
    params = cpu_system.system.parameters.values_array.astype(
        cpu_system.precision
    )
    coeffs = np.array(
        driver_array.coefficients,
        dtype=cpu_system.precision,
        copy=True,
    )
    driver_eval = cpu_driver_evaluator.with_coefficients(coeffs)
    result = step_function(
        cpu_system,
        driver_eval,
        state=state,
        params=params,
        dt=dt_value,
        tol=implicit_step_settings["nonlinear_tolerance"],
        time=0.0,
    )
    return {
        "state": result.state.astype(cpu_system.precision, copy=True),
        "observables": result.observables.astype(
            cpu_system.precision, copy=True
        ),
        "error": result.error.astype(cpu_system.precision, copy=True),
        "status": result.status & STATUS_MASK,
        "niters": (result.status >> 16) & STATUS_MASK,
    }


@pytest.mark.parametrize(
    "system_override",
    ["three_chamber"],
    indirect=True,
)
@pytest.mark.parametrize(
    "solver_settings_override",
    [
        {
            "algorithm": "crank_nicolson",
            "step_controller": "pid",
            "dt_min": 1e-4,
            "dt_max": 1e-4,
            "atol": 1e-6,
            "rtol": 1e-6,
        }
    ],
    indirect=True,
)
def test_crank_nicolson_step_matches_cpu(
    step_object,
    solver_settings,
    system,
    precision,
    initial_state,
    cpu_driver_evaluator,
    driver_array,
    cpu_system,
    implicit_step_settings,
    tolerance,
):
    device = _device_step(
        step_object,
        solver_settings,
        cpu_system,
        system,
        precision,
        initial_state,
        cpu_driver_evaluator,
        driver_array,
    )
    cpu = _cpu_step(
        solver_settings,
        cpu_system,
        implicit_step_settings,
        initial_state,
        driver_array,
        cpu_driver_evaluator,
    )

    assert device["status"] == cpu["status"]
    assert device["niters"] == cpu["niters"]
    assert_allclose(
        device["state"],
        cpu["state"],
        rtol=tolerance.rel_tight,
        atol=tolerance.abs_tight,
    ), f"state mismatch: \ndevice={device['state']}\,cpu={cpu['state']}"
    assert_allclose(
        device["observables"],
        cpu["observables"],
        rtol=tolerance.rel_tight,
        atol=tolerance.abs_tight,
    ), f"observables mismatch: \ndevice={device['observables']}\,cpu={cpu['observables']}"
    assert_allclose(
        device["error"],
        cpu["error"],
        rtol=tolerance.rel_tight,
        atol=tolerance.abs_tight,
    ), f"error mismatch: \ndevice={device['error']}\,cpu={cpu['error']}"
