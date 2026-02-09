"""Numerical correctness tests for single-step integration algorithms.

Compares CUDA device step results against CPU reference implementations
for all algorithm families across two consecutive integration steps.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from cubie.integrators.algorithms import get_algorithm_step
from tests.integrators.cpu_reference import CPUODESystem
from tests._utils import (
    MID_RUN_PARAMS,
    STATUS_MASK,
    AlgorithmStepResult,
    DualStepResult,
    merge_dicts,
    merge_param,
    ALGORITHM_PARAM_SETS,
    _execute_step_twice,
    _execute_cpu_step_twice,
)

Array = np.ndarray


# Merged cases for constant_deriv system tests
STEP_CASES_CONSTANT_DERIV = [
    merge_param(
        merge_dicts(MID_RUN_PARAMS, {"system_type": "constant_deriv"}),
        case,
    )
    for case in ALGORITHM_PARAM_SETS
]


# ── Two-step device-vs-CPU comparison ─────────────────────── #

@pytest.mark.parametrize(
    "solver_settings_override",
    ALGORITHM_PARAM_SETS,
    indirect=True,
)
def test_two_steps(
    solver_settings,
    step_object,
    precision,
    algorithm_step_inputs,
    system,
    driver_array,
    cpu_system,
    cpu_driver_evaluator,
):
    """Ensure shared-cache reuse yields consistent results
    across devices."""

    gpu_result = _execute_step_twice(
        step_object=step_object,
        solver_settings=solver_settings,
        precision=precision,
        step_inputs=algorithm_step_inputs,
        system=system,
        driver_array=driver_array,
    )

    cpu_result = _execute_cpu_step_twice(
        solver_settings=solver_settings,
        step_inputs=algorithm_step_inputs,
        cpu_system=cpu_system,
        cpu_driver_evaluator=cpu_driver_evaluator,
        step_object=step_object,
    )

    assert all(status == 0 for status in gpu_result.statuses)
    assert all(status == 0 for status in cpu_result.statuses)
    assert gpu_result.statuses == cpu_result.statuses

    tol = {"rtol": 5e-7, "atol": 1e-7}

    assert_allclose(
        gpu_result.first_state,
        cpu_result.first_state,
        **tol,
    )
    assert_allclose(
        gpu_result.second_state,
        cpu_result.second_state,
        **tol,
    )
    assert_allclose(
        gpu_result.first_error,
        cpu_result.first_error,
        **tol,
    )
    assert_allclose(
        gpu_result.second_error,
        cpu_result.second_error,
        **tol,
    )
    assert_allclose(
        gpu_result.first_observables,
        cpu_result.first_observables,
        **tol,
    )
    assert_allclose(
        gpu_result.second_observables,
        cpu_result.second_observables,
        **tol,
    )

    delta = np.abs(
        gpu_result.second_state - gpu_result.first_state
    )
    assert np.any(delta > precision(1e-10))


# ── All algorithms match Euler for constant-derivative system ─ #

@pytest.mark.parametrize(
    "solver_settings_override",
    STEP_CASES_CONSTANT_DERIV,
    indirect=True,
)
def test_against_euler(
    solver_settings,
    step_object,
    precision,
    algorithm_step_inputs,
    system,
    driver_array,
    cpu_system,
    cpu_driver_evaluator,
    tolerance,
):
    """All algorithms match Euler for constant-derivative system.

    For a system with constant derivatives (dx/dt = constant), all
    numerical integration algorithms should produce identical results
    to the Euler method, since higher-order Taylor terms vanish.
    """

    device_result = _execute_step_twice(
        step_object=step_object,
        solver_settings=solver_settings,
        precision=precision,
        step_inputs=algorithm_step_inputs,
        system=system,
        driver_array=driver_array,
    )

    euler_settings = solver_settings.copy()
    euler_settings["algorithm"] = "euler"

    euler_algorithm_settings = {
        "algorithm": "euler",
        "n": system.sizes.states,
        "dt": euler_settings["dt"],
        "evaluate_f": system.evaluate_f,
        "evaluate_observables": system.evaluate_observables,
    }
    euler_step_obj = get_algorithm_step(
        precision, euler_algorithm_settings
    )

    euler_result = _execute_cpu_step_twice(
        solver_settings=euler_settings,
        step_inputs=algorithm_step_inputs,
        cpu_system=cpu_system,
        cpu_driver_evaluator=cpu_driver_evaluator,
        step_object=euler_step_obj,
    )

    assert all(status == 0 for status in device_result.statuses)
    assert all(status == 0 for status in euler_result.statuses)

    tol = {
        "rtol": tolerance.rel_tight * 3,
        "atol": tolerance.abs_tight * 3,
    }

    assert_allclose(
        device_result.first_state,
        euler_result.first_state,
        **tol,
    )
    assert_allclose(
        device_result.second_state,
        euler_result.second_state,
        **tol,
    )
    assert_allclose(
        device_result.first_observables,
        euler_result.first_observables,
        **tol,
    )
    assert_allclose(
        device_result.second_observables,
        euler_result.second_observables,
        **tol,
    )
