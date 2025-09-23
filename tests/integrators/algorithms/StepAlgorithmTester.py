"""Reusable tester for single-step integration algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from numba import cuda, from_dtype
from numpy.testing import assert_allclose

from tests.integrators.cpu_reference import CPUODESystem, get_ref_step_fn
from _utils import assert_integration_outputs, \
    _driver_sequence

Array = np.ndarray
STATUS_MASK = 0xFFFF

@dataclass
class StepResult:
    """Container holding the outputs of a single step execution."""

    state: Array
    observables: Array
    error: Array
    status: int
    niters: int

def generate_step_props(n_states: int) -> dict[str, dict[str, Any]]:
    """Generate expected properties for each algorithm given n_states."""
    return {
        "euler": {
            "threads_per_step": 1,
            "persistent_local_required": 0,
            "is_multistage": False,
            "is_implicit": False,
            "is_adaptive": False,
            "order": 1,
            "shared_memory_required": 0,
            "local_scratch_required": 0,
        },
        "backwards_euler": {
            "threads_per_step": 1,
            "persistent_local_required": 0,
            "is_multistage": False,
            "is_implicit": True,
            "is_adaptive": False,
            "order": 1,
            "shared_memory_required": 0,
            "local_scratch_required": 4 * n_states,
        },
        "backwards_euler_pc": {
            "threads_per_step": 1,
            "persistent_local_required": 0,
            "is_multistage": False,
            "is_implicit": True,
            "is_adaptive": False,
            "order": 1,
            "shared_memory_required": 0,
            "local_scratch_required": 4 * n_states,
        },
        "crank_nicolson": {
            "threads_per_step": 1,
            "persistent_local_required": 0,
            "is_multistage": False,
            "is_implicit": True,
            "is_adaptive": True,
            "order": 2,
            "shared_memory_required": 0,
            "local_scratch_required": 5 * n_states,
        },
    }

@pytest.fixture(scope="function")
def expected_step_properties(system) -> dict[str, Any]:
    """Generate expected properties for each algorithm given n_states."""
    return generate_step_props(n_states=system.sizes.states)

@pytest.fixture(scope="function")
def step_inputs(system, precision, initial_state, solver_settings) -> dict[
    str,
Array]:
    """State, parameters, and drivers for a single step execution."""
    if system.num_drivers > 0:
        drivers = _driver_sequence(
            samples=2,
            total_time=float(solver_settings['dt_min']),
            n_drivers=system.num_drivers,
            precision=precision,
        )
        drivers_now = np.asarray(drivers[0], dtype=precision)
        drivers_next = np.asarray(drivers[1], dtype=precision)
    else:
        drivers_now = np.zeros(1, dtype=precision)
        drivers_next = np.zeros(1, dtype=precision)
    return {
        "state": initial_state,
        "parameters": system.parameters.values_array.astype(precision),
        "drivers_now": drivers_now,
        "drivers_next": drivers_next,
    }


@pytest.fixture(scope="function")
def device_step_results(
    step_object,
    solver_settings,
    precision,
    step_inputs,
    system,
) -> StepResult:
    """Execute the CUDA step and collect host-side outputs."""

    step_fn = step_object.step_function
    step_size = solver_settings['dt_min']
    n_states = system.sizes.states
    params = step_inputs["parameters"]
    state = step_inputs["state"]
    driver_key = "drivers_next" if step_object.is_implicit else "drivers_now"
    drivers = step_inputs[driver_key]
    observables = np.zeros(system.sizes.observables, dtype=precision)
    proposed_state = np.zeros_like(state)
    work_buffer = np.zeros(n_states, dtype=precision)
    error = np.zeros(n_states, dtype=precision)
    status = np.full(1, 0, dtype=np.int32)

    shared_elems = step_object.shared_memory_required
    shared_bytes = precision(0).itemsize * shared_elems
    persistent_len = max(1, step_object.persistent_local_required)
    numba_precision = from_dtype(precision)
    dt_value = precision(step_size)

    d_state = cuda.to_device(state)
    d_proposed = cuda.to_device(proposed_state)
    d_work = cuda.to_device(work_buffer)
    d_params = cuda.to_device(params)
    d_drivers = cuda.to_device(drivers)
    d_observables = cuda.to_device(observables)
    d_error = cuda.to_device(error)
    d_status = cuda.to_device(status)

    @cuda.jit
    def kernel(
        state_vec,
        proposed_vec,
        work_vec,
        params_vec,
        drivers_vec,
        observables_vec,
        error_vec,
        status_vec,
        dt_scalar,
    ) -> None:
        idx = cuda.grid(1)
        if idx > 0:
            return
        shared = cuda.shared.array(0, dtype=numba_precision)
        persistent = cuda.local.array(persistent_len, dtype=numba_precision)
        result = step_fn(
            state_vec,
            proposed_vec,
            work_vec,
            params_vec,
            drivers_vec,
            observables_vec,
            error_vec,
            dt_scalar,
            shared,
            persistent,
        )
        status_vec[0] = result

    kernel[1, 1, 0, shared_bytes](
        d_state,
        d_proposed,
        d_work,
        d_params,
        d_drivers,
        d_observables,
        d_error,
        d_status,
        dt_value,
    )
    cuda.synchronize()
    
    status_value = int(d_status.copy_to_host()[0])
    return StepResult(
        state=d_proposed.copy_to_host(),
        observables=d_observables.copy_to_host(),
        error=d_error.copy_to_host(),
        status=status_value & STATUS_MASK,
        niters=(status_value >> 16) & STATUS_MASK,
    )


@pytest.fixture(scope="function")
def cpu_step_results(
    solver_settings,
    cpu_system: CPUODESystem,
    step_inputs,
    implicit_step_settings
) -> StepResult:
    """Execute the CPU reference stepper."""

    step_fn = get_ref_step_fn(solver_settings["algorithm"])
    dt = solver_settings["dt_min"]
    state = np.asarray(step_inputs["state"], dtype=cpu_system.precision)
    params = np.asarray(step_inputs["parameters"], dtype=cpu_system.precision)
    drivers_now = np.asarray(step_inputs["drivers_now"],
                             dtype=cpu_system.precision
    )
    drivers_next = np.asarray(
        step_inputs["drivers_next"], dtype=cpu_system.precision
    )

    result = step_fn(
        cpu_system,
        state=state,
        params=params,
        drivers_now=drivers_now,
        drivers_next=drivers_next,
        dt=dt,
        tol=implicit_step_settings['nonlinear_tolerance']
        )

    return StepResult(
        state=result.state.astype(cpu_system.precision, copy=True),
        observables=result.observables.astype(cpu_system.precision, copy=True),
        error=result.error.astype(cpu_system.precision, copy=True),
        status=int(result.status),
        niters=result.niters,
    )


#All-in-one step test to share fixture setup at the expense of readability
@pytest.mark.parametrize(
    "solver_settings_override",
    [
        {"algorithm": "euler", 'step_controller': 'fixed'},
        {"algorithm": "backwards_euler"},
        {"algorithm": "backwards_euler_pc"},
        {"algorithm": "crank_nicolson", 'step_controller': 'pid', 'atol':
            1e-4, 'rtol': 1e-4, 'dt_min': 0.001},
    ],
    ids=["euler", "backwards_euler", "backwards_euler_pc", "crank_nicolson"],
    indirect=True,
)
def test_algorithm(
       step_object,
       solver_settings,
       system,
       precision,
       implicit_step_settings,
       expected_step_properties,
       cpu_step_results,
       device_step_results,
       cpu_loop_outputs,
       device_loop_outputs,
       output_functions,
       ) -> None:
    """Ensure the step function is compiled and callable."""
    # Test that it builds
    assert callable(step_object.step_function), "step_fn_builds"

    # test getters
    properties = expected_step_properties[solver_settings["algorithm"]]
    assert step_object.is_implicit is properties["is_implicit"], \
        "is_implicit getter"
    assert step_object.is_adaptive is properties["is_adaptive"], \
        "is_adaptive getter"
    assert step_object.is_multistage is properties["is_multistage"],\
        "is_multistage getter"
    assert step_object.order == properties["order"],\
        "order getter"
    assert (
        step_object.shared_memory_required
        == properties["shared_memory_required"]
    ), "shared_memory_required getter"
    assert step_object.local_scratch_required \
        == properties["local_scratch_required"],\
        "local_scratch_required getter"

    assert step_object.persistent_local_required \
        == properties["persistent_local_required"], \
        "persistent_local_required getter"

    assert (
        step_object.threads_per_step == properties["threads_per_step"]
    ), "threads_per_step getter"

    config = step_object.compile_settings
    assert config.n == system.sizes.states, "compile_settings.n getter"
    assert config.precision == precision, "compile_settings.precision getter"

    if properties["is_implicit"]:
        assert step_object.nonlinear_solver_function is not None
        # assert config.get_solver_helper_fn is system.get_solver_helper
        matrix = config.M
        assert matrix.shape == (system.sizes.states, system.sizes.states)
        assert config.preconditioner_order == implicit_step_settings[
                "preconditioner_order"], "preconditioner order set"
        assert config.max_linear_iters == implicit_step_settings[
    "max_linear_iters"], "max_linear_iters set"
        assert config.linear_correction_type == implicit_step_settings[
                "correction_type"], "linear_correction_type set"
        assert config.max_newton_iters == implicit_step_settings[
            "max_newton_iters"], "max_newton_iters set"
        assert config.newton_max_backtracks == implicit_step_settings[
                "newton_max_backtracks"], "newton_max_backtracks set"
        assert config.linsolve_tolerance == pytest.approx(
            implicit_step_settings["linear_tolerance"]
        ), "linsolve_tolerance set"
        assert config.nonlinear_tolerance == pytest.approx(
            implicit_step_settings["nonlinear_tolerance"]
        ), "nonlinear_tolerance set"
        assert config.newton_damping == pytest.approx(
            implicit_step_settings["newton_damping"]
        ), "newton_damping set"
        assert callable(system.get_solver_helper)
    else:
        assert step_object.nonlinear_solver_function is None
        assert step_object.dt == pytest.approx(solver_settings["dt_min"])

    if step_object.is_implicit:
        updates = {
            "max_newton_iters": int(
                max(1, implicit_step_settings["max_newton_iters"] // 2)
            ),
            "linsolve_tolerance":
            implicit_step_settings["linear_tolerance"] * 0.5,
            "nonlinear_tolerance":
            implicit_step_settings["nonlinear_tolerance"] * 0.5,
            "newton_damping":
            implicit_step_settings["newton_damping"] * 0.9,
            "preconditioner_order":
            implicit_step_settings["preconditioner_order"] + 1,
        }
        recognised = step_object.update(updates)
        assert set(updates).issubset(recognised), "updates recognised"
        config = step_object.compile_settings
        assert config.max_newton_iters == updates["max_newton_iters"], \
            "max_newton_iters update"
        assert config.preconditioner_order == updates[
            "preconditioner_order"
        ], "preconditioner_order update"
        assert config.linsolve_tolerance == pytest.approx(
            updates["linsolve_tolerance"]
        ), "linsolve_tolerance update"
        assert config.nonlinear_tolerance == pytest.approx(
            updates["nonlinear_tolerance"]
        ), "nonlinear_tolerance update"
        assert config.newton_damping == pytest.approx(
            updates["newton_damping"]
        ), "newton_damping update"
    else:
        new_dt = float(solver_settings["dt_min"]) * 0.5
        recognised = step_object.update({"dt": new_dt})
        assert "dt" in recognised, "dt recognised"
        assert step_object.dt == pytest.approx(new_dt), "dt update"


    tolerances = {"rtol": 1e-4, "atol": 1e-4}
    assert device_step_results.status == cpu_step_results.status, \
        "status matches"
    assert device_step_results.niters == cpu_step_results.niters, \
        "niters match"
    assert_allclose(
        device_step_results.state,
        cpu_step_results.state,
        rtol=tolerances["rtol"],
        atol=tolerances["atol"],
    ), "state matches"
    assert_allclose(
        device_step_results.observables,
        cpu_step_results.observables,
        rtol=tolerances["rtol"],
        atol=tolerances["atol"],
    ), "observables matches"
    if step_object.is_adaptive:
        assert_allclose(
            device_step_results.error,
            cpu_step_results.error,
            rtol=tolerances["rtol"],
            atol=tolerances["atol"],
        ), "error matches"

    # Run a short loop to ensure step works in that context
    assert device_loop_outputs.status == 0
    assert_integration_outputs(
        cpu_loop_outputs,
        device_loop_outputs,
        output_functions,
        rtol=tolerances["rtol"],
        atol=tolerances["atol"],
    )

