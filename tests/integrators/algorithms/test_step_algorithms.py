"""Reusable tester for single-step integration algorithms."""

from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
import pytest
from numba import cuda, from_dtype
from numpy.testing import assert_allclose

from cubie.integrators.algorithms import get_algorithm_step
from cubie.integrators.algorithms.backwards_euler import BackwardsEulerStep
from cubie.integrators.algorithms.backwards_euler_predict_correct import (
    BackwardsEulerPCStep,
)
from cubie.integrators.algorithms.crank_nicolson import CrankNicolsonStep
from cubie.integrators.algorithms.explicit_euler import ExplicitEulerStep
from cubie.integrators.algorithms.generic_dirk import DIRKStep
from cubie.integrators.algorithms.generic_dirk_tableaus import (
    DEFAULT_DIRK_TABLEAU,
    DIRK_TABLEAU_REGISTRY,
)
from cubie.integrators.algorithms.generic_erk import ERKStep
from cubie.integrators.algorithms.generic_erk_tableaus import (
    DEFAULT_ERK_TABLEAU,
    ERK_TABLEAU_REGISTRY,
)
from cubie.integrators.algorithms.generic_rosenbrock_w import (
    GenericRosenbrockWStep,
)
from cubie.integrators.algorithms.generic_rosenbrockw_tableaus import (
    DEFAULT_ROSENBROCK_TABLEAU,
    ROSENBROCK_TABLEAUS,
)
from tests.integrators.cpu_reference import (
    CPUODESystem,
    get_ref_step_factory,
    get_ref_stepper,
)
from tests.integrators.cpu_reference.algorithms import (
    CPUDIRKStep,
    CPUERKStep,
    CPURosenbrockWStep,
)

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


def _expected_order(step_object: Any, tableau: Any) -> int:
    """Return the theoretical order of accuracy for ``step_object``."""

    if tableau is not None:
        return tableau.order
    if isinstance(
        step_object,
        (
            ExplicitEulerStep,
            BackwardsEulerStep,
            BackwardsEulerPCStep,
        ),
    ):
        return 1
    if isinstance(step_object, CrankNicolsonStep):
        return 2
    raise NotImplementedError(
        f"Order expectation missing for {type(step_object).__name__}."
    )


def _expected_memory_requirements(
    step_object: Any,
    tableau: Any,
    n_states: int,
    extra_shared: int,
) -> tuple[int, int]:
    """Return the expected shared and local scratch requirements."""

    if isinstance(step_object, ERKStep):
        stage_count = tableau.stage_count
        accumulator_span = max(stage_count - 1, 0) * n_states
        return accumulator_span, n_states
    if isinstance(step_object, DIRKStep):
        stage_count = tableau.stage_count
        accumulator_span = max(stage_count - 1, 0) * n_states
        shared = accumulator_span + 2 * n_states + extra_shared
        local = 2 * n_states
        return shared, local
    if isinstance(step_object, GenericRosenbrockWStep):
        stage_count = tableau.stage_count
        accumulator_span = max(stage_count - 1, 0) * n_states
        shared = 2 * accumulator_span + extra_shared
        local = 4 * n_states
        return shared, local
    if isinstance(
        step_object,
        (BackwardsEulerStep, BackwardsEulerPCStep, CrankNicolsonStep),
    ):
        shared = 2 * n_states + extra_shared
        return shared, 0
    if isinstance(step_object, ExplicitEulerStep):
        return 0, 0
    raise NotImplementedError(
        "Memory expectation missing for "
        f"{type(step_object).__name__}."
    )


ALIAS_CASES = [
    pytest.param(
        "erk",
        ERKStep,
        DEFAULT_ERK_TABLEAU,
        CPUERKStep,
        id="erk",
    ),
    pytest.param(
        "dirk",
        DIRKStep,
        DEFAULT_DIRK_TABLEAU,
        CPUDIRKStep,
        id="dirk",
    ),
    pytest.param(
        "rosenbrock",
        GenericRosenbrockWStep,
        DEFAULT_ROSENBROCK_TABLEAU,
        CPURosenbrockWStep,
        id="rosenbrock",
    ),
    pytest.param(
        "dormand-prince-54",
        ERKStep,
        ERK_TABLEAU_REGISTRY["dormand-prince-54"],
        CPUERKStep,
        marks=pytest.mark.specific_algos,
        id="erk-dormand-prince-54",
    ),
    pytest.param(
        "dopri54",
        ERKStep,
        DEFAULT_ERK_TABLEAU,
        CPUERKStep,
        marks=pytest.mark.specific_algos,
        id="erk-dopri54",
    ),
    pytest.param(
        "cash-karp-54",
        ERKStep,
        ERK_TABLEAU_REGISTRY["cash-karp-54"],
        CPUERKStep,
        marks=pytest.mark.specific_algos,
        id="erk-cash-karp-54",
    ),
    pytest.param(
        "fehlberg-45",
        ERKStep,
        ERK_TABLEAU_REGISTRY["fehlberg-45"],
        CPUERKStep,
        marks=pytest.mark.specific_algos,
        id="erk-fehlberg-45",
    ),
    pytest.param(
        "bogacki-shampine-32",
        ERKStep,
        ERK_TABLEAU_REGISTRY["bogacki-shampine-32"],
        CPUERKStep,
        marks=pytest.mark.specific_algos,
        id="erk-bogacki-shampine-32",
    ),
    pytest.param(
        "heun-21",
        ERKStep,
        ERK_TABLEAU_REGISTRY["heun-21"],
        CPUERKStep,
        marks=pytest.mark.specific_algos,
        id="erk-heun-21",
    ),
    pytest.param(
        "ralston-33",
        ERKStep,
        ERK_TABLEAU_REGISTRY["ralston-33"],
        CPUERKStep,
        marks=pytest.mark.specific_algos,
        id="erk-ralston-33",
    ),
    pytest.param(
        "classical-rk4",
        ERKStep,
        ERK_TABLEAU_REGISTRY["classical-rk4"],
        CPUERKStep,
        marks=pytest.mark.specific_algos,
        id="erk-classical-rk4",
    ),
    pytest.param(
        "implicit_midpoint",
        DIRKStep,
        DIRK_TABLEAU_REGISTRY["implicit_midpoint"],
        CPUDIRKStep,
        marks=pytest.mark.specific_algos,
        id="dirk-implicit-midpoint",
    ),
    pytest.param(
        "trapezoidal_dirk",
        DIRKStep,
        DIRK_TABLEAU_REGISTRY["trapezoidal_dirk"],
        CPUDIRKStep,
        marks=pytest.mark.specific_algos,
        id="dirk-trapezoidal",
    ),
    pytest.param(
        "sdirk_2_2",
        DIRKStep,
        DIRK_TABLEAU_REGISTRY["sdirk_2_2"],
        CPUDIRKStep,
        marks=pytest.mark.specific_algos,
        id="dirk-sdirk-2-2",
    ),
    pytest.param(
        "lobatto_iiic_3",
        DIRKStep,
        DIRK_TABLEAU_REGISTRY["lobatto_iiic_3"],
        CPUDIRKStep,
        marks=pytest.mark.specific_algos,
        id="dirk-lobatto-iiic-3",
    ),
    pytest.param(
        "ros3p",
        GenericRosenbrockWStep,
        ROSENBROCK_TABLEAUS["ros3p"],
        CPURosenbrockWStep,
        marks=pytest.mark.specific_algos,
        id="rosenbrock-ros3p",
    ),
    pytest.param(
        "rosenbrock_w6s4os",
        GenericRosenbrockWStep,
        ROSENBROCK_TABLEAUS["rosenbrock_w6s4os"],
        CPURosenbrockWStep,
        marks=pytest.mark.specific_algos,
        id="rosenbrock-w6s4os",
    ),
]

STEP_CASES = [
    pytest.param(
        {
            "algorithm": "euler",
            "step_controller": "fixed",
            "dt": 0.0025,
            "dt_min": 0.0025,
            "dt_save": 0.2,
            "output_types": ["state"],
            "saved_state_indices": [0, 1, 2],
            "newton_tolerance": 5e-6,
            "krylov_tolerance": 1e-6,
        },
        {},
        id="euler",
    ),
    pytest.param(
        {
            "algorithm": "backwards_euler",
            "step_controller": "fixed",
            "dt": 0.0025,
            "dt_save": 0.2,
            "output_types": ["state"],
            "saved_state_indices": [0, 1, 2],
            "newton_tolerance": 5e-6,
            "krylov_tolerance": 1e-6,
        },
        {},
        id="backwards_euler",
    ),
    pytest.param(
        {
            "algorithm": "backwards_euler_pc",
            "step_controller": "fixed",
            "dt": 0.0025,
            "dt_save": 0.2,
            "output_types": ["state"],
            "saved_state_indices": [0, 1, 2],
            "newton_tolerance": 5e-6,
            "krylov_tolerance": 1e-6,
        },
        {},
        id="backwards_euler_pc",
    ),
    pytest.param(
        {
            "algorithm": "crank_nicolson",
            "step_controller": "pid",
            "atol": 1e-6,
            "rtol": 1e-6,
            "dt_min": 1e-6,
            "dt": 0.0025,
            "dt_save": 0.2,
            "output_types": ["state"],
            "saved_state_indices": [0, 1, 2],
            "newton_tolerance": 5e-6,
            "krylov_tolerance": 1e-6,
        },
        {},
        id="crank_nicolson_pid",
    ),
    pytest.param(
        {
            "algorithm": "crank_nicolson",
            "step_controller": "pi",
            "atol": 1e-6,
            "rtol": 1e-6,
            "dt_min": 1e-6,
            "dt": 0.0025,
            "dt_save": 0.2,
            "output_types": ["state"],
            "saved_state_indices": [0, 1, 2],
            "newton_tolerance": 5e-6,
            "krylov_tolerance": 1e-6,
        },
        {},
        id="crank_nicolson_pi",
    ),
    pytest.param(
        {
            "algorithm": "crank_nicolson",
            "step_controller": "i",
            "atol": 1e-5,
            "rtol": 1e-5,
            "dt_min": 1e-6,
            "dt": 0.0025,
            "dt_save": 0.2,
            "output_types": ["state"],
            "saved_state_indices": [0, 1, 2],
        },
        {},
        id="crank_nicolson_i",
    ),
    pytest.param(
        {
            "algorithm": "crank_nicolson",
            "step_controller": "gustafsson",
            "atol": 1e-5,
            "rtol": 1e-5,
            "dt_min": 1e-6,
            "dt": 0.0025,
            "dt_save": 0.2,
            "output_types": ["state"],
            "saved_state_indices": [0, 1, 2],
        },
        {},
        id="crank_nicolson_gustafsson",
    ),
    pytest.param(
        {
            "algorithm": "rosenbrock",
            "step_controller": "pi",
            "atol": 1e-6,
            "rtol": 1e-6,
            "dt_min": 1e-6,
            "dt": 0.0025,
            "dt_save": 0.2,
            "output_types": ["state"],
            "saved_state_indices": [0, 1, 2],
            "krylov_tolerance": 1e-6,
        },
        {},
        id="rosenbrock",
    ),
    pytest.param(
        {
            "algorithm": "erk",
            "step_controller": "pi",
            "atol": 1e-6,
            "rtol": 1e-6,
            "dt_min": 0.0025,
            "dt": 0.0025,
            "dt_save": 0.2,
            "output_types": ["state"],
            "saved_state_indices": [0, 1, 2],
        },
        {},
        id="erk",
    ),
    pytest.param(
        {
            "algorithm": "dirk",
            "step_controller": "pi",
            "atol": 1e-6,
            "rtol": 1e-6,
            "dt_min": 1e-6,
            "dt": 0.0025,
            "dt_save": 0.2,
            "output_types": ["state"],
            "saved_state_indices": [0, 1, 2],
            "newton_tolerance": 5e-6,
            "krylov_tolerance": 1e-6,
        },
        {},
        id="dirk",
    ),
    pytest.param(
        {
            "algorithm": "ros3p",
            "step_controller": "pi",
            "atol": 1e-6,
            "rtol": 1e-6,
            "dt_min": 1e-6,
            "dt": 0.0025,
            "dt_save": 0.2,
            "output_types": ["state"],
            "saved_state_indices": [0, 1, 2],
            "krylov_tolerance": 1e-6,
        },
        {},
        id="rosenbrock-ros3p",
        marks=pytest.mark.specific_algos,
    ),
    pytest.param(
        {
            "algorithm": "rosenbrock_w6s4os",
            "step_controller": "pi",
            "atol": 1e-6,
            "rtol": 1e-6,
            "dt_min": 1e-6,
            "dt": 0.0025,
            "dt_save": 0.2,
            "output_types": ["state"],
            "saved_state_indices": [0, 1, 2],
            "krylov_tolerance": 1e-6,
        },
        {},
        id="rosenbrock-w6s4os",
        marks=pytest.mark.specific_algos,
    ),
    pytest.param(
        {
            "algorithm": "dormand-prince-54",
            "step_controller": "pi",
            "atol": 1e-6,
            "rtol": 1e-6,
            "dt_min": 0.0025,
            "dt": 0.0025,
            "dt_save": 0.2,
            "output_types": ["state"],
            "saved_state_indices": [0, 1, 2],
        },
        {},
        id="erk-dormand-prince-54",
        marks=pytest.mark.specific_algos,
    ),
    pytest.param(
        {
            "algorithm": "dopri54",
            "step_controller": "pi",
            "atol": 1e-6,
            "rtol": 1e-6,
            "dt_min": 0.0025,
            "dt": 0.0025,
            "dt_save": 0.2,
            "output_types": ["state"],
            "saved_state_indices": [0, 1, 2],
        },
        {},
        id="erk-dopri54",
        marks=pytest.mark.specific_algos,
    ),
    pytest.param(
        {
            "algorithm": "cash-karp-54",
            "step_controller": "pi",
            "atol": 1e-6,
            "rtol": 1e-6,
            "dt_min": 0.0025,
            "dt": 0.0025,
            "dt_save": 0.2,
            "output_types": ["state"],
            "saved_state_indices": [0, 1, 2],
        },
        {},
        id="erk-cash-karp-54",
        marks=pytest.mark.specific_algos,
    ),
    pytest.param(
        {
            "algorithm": "fehlberg-45",
            "step_controller": "pi",
            "atol": 1e-6,
            "rtol": 1e-6,
            "dt_min": 0.0025,
            "dt": 0.0025,
            "dt_save": 0.2,
            "output_types": ["state"],
            "saved_state_indices": [0, 1, 2],
        },
        {},
        id="erk-fehlberg-45",
        marks=pytest.mark.specific_algos,
    ),
    pytest.param(
        {
            "algorithm": "bogacki-shampine-32",
            "step_controller": "pi",
            "atol": 1e-6,
            "rtol": 1e-6,
            "dt_min": 0.0025,
            "dt": 0.0025,
            "dt_save": 0.2,
            "output_types": ["state"],
            "saved_state_indices": [0, 1, 2],
        },
        {},
        id="erk-bogacki-shampine-32",
        marks=pytest.mark.specific_algos,
    ),
    pytest.param(
        {
            "algorithm": "heun-21",
            "step_controller": "fixed",
            "dt_min": 0.0025,
            "dt": 0.0025,
            "dt_save": 0.2,
            "output_types": ["state"],
            "saved_state_indices": [0, 1, 2],
        },
        {},
        id="erk-heun-21",
        marks=pytest.mark.specific_algos,
    ),
    pytest.param(
        {
            "algorithm": "ralston-33",
            "step_controller": "fixed",
            "dt_min": 0.0025,
            "dt": 0.0025,
            "dt_save": 0.2,
            "output_types": ["state"],
            "saved_state_indices": [0, 1, 2],
        },
        {},
        id="erk-ralston-33",
        marks=pytest.mark.specific_algos,
    ),
    pytest.param(
        {
            "algorithm": "classical-rk4",
            "step_controller": "fixed",
            "dt_min": 0.0025,
            "dt": 0.0025,
            "dt_save": 0.2,
            "output_types": ["state"],
            "saved_state_indices": [0, 1, 2],
        },
        {},
        id="erk-classical-rk4",
        marks=pytest.mark.specific_algos,
    ),
    pytest.param(
        {
            "algorithm": "implicit_midpoint",
            "step_controller": "fixed",
            "dt": 0.0025,
            "dt_save": 0.2,
            "output_types": ["state"],
            "saved_state_indices": [0, 1, 2],
            "newton_tolerance": 5e-6,
            "krylov_tolerance": 1e-6,
        },
        {},
        id="dirk-implicit-midpoint",
        marks=pytest.mark.specific_algos,
    ),
    pytest.param(
        {
            "algorithm": "trapezoidal_dirk",
            "step_controller": "fixed",
            "dt": 0.0025,
            "dt_save": 0.2,
            "output_types": ["state"],
            "saved_state_indices": [0, 1, 2],
            "newton_tolerance": 5e-6,
            "krylov_tolerance": 1e-6,
        },
        {},
        id="dirk-trapezoidal",
        marks=pytest.mark.specific_algos,
    ),
    pytest.param(
        {
            "algorithm": "sdirk_2_2",
            "step_controller": "pi",
            "atol": 1e-6,
            "rtol": 1e-6,
            "dt_min": 1e-6,
            "dt": 0.0025,
            "dt_save": 0.2,
            "output_types": ["state"],
            "saved_state_indices": [0, 1, 2],
            "newton_tolerance": 5e-6,
            "krylov_tolerance": 1e-6,
        },
        {},
        id="dirk-sdirk-2-2",
        marks=pytest.mark.specific_algos,
    ),
    pytest.param(
        {
            "algorithm": "lobatto_iiic_3",
            "step_controller": "fixed",
            "dt": 0.0025,
            "dt_save": 0.2,
            "output_types": ["state"],
            "saved_state_indices": [0, 1, 2],
            "newton_tolerance": 5e-6,
            "krylov_tolerance": 1e-6,
        },
        {},
        id="dirk-lobatto-iiic-3",
        marks=pytest.mark.specific_algos,
    ),
]

@pytest.mark.parametrize(
    "alias_key, expected_step_type, expected_tableau, expected_cpu_step",
    ALIAS_CASES,
)
def test_algorithm_factory_resolves_tableau_alias(
    alias_key,
    expected_step_type,
    expected_tableau,
    expected_cpu_step,
):
    """Algorithm factory should inject the tableau advertised for aliases."""

    step = get_algorithm_step(
        np.float64,
        settings={"algorithm": alias_key, "n": 2, "dt": 1e-3},
        warn_on_unused=False,
    )
    assert isinstance(step, expected_step_type)
    tableau_value = getattr(step, "tableau", None)
    if tableau_value is None:
        tableau_value = step.compile_settings.tableau
    assert tableau_value is expected_tableau


@pytest.mark.parametrize(
    "alias_key, expected_step_type, expected_tableau, expected_cpu_step",
    ALIAS_CASES,
)
def test_cpu_reference_resolves_tableau_alias(
    alias_key,
    expected_step_type,
    expected_tableau,
    expected_cpu_step,
    cpu_system: CPUODESystem,
    cpu_driver_evaluator,
):
    """CPU reference helpers should resolve alias tableaus consistently."""
    factory = get_ref_step_factory(alias_key)
    bound_step = factory(
        cpu_system,
        cpu_driver_evaluator,
        newton_tol=1e-10,
        newton_max_iters=25,
        linear_tol=1e-10,
        linear_max_iters=cpu_system.system.sizes.states,
    )
    default_tableaus = {
        CPUERKStep: DEFAULT_ERK_TABLEAU,
        CPUDIRKStep: DEFAULT_DIRK_TABLEAU,
        CPURosenbrockWStep: DEFAULT_ROSENBROCK_TABLEAU,
    }
    default_tableau = default_tableaus[expected_cpu_step]
    if isinstance(bound_step, partial):
        assert isinstance(bound_step.func, expected_cpu_step)
        assert bound_step.keywords.get("tableau") is expected_tableau
    else:
        assert isinstance(bound_step, expected_cpu_step)
        if expected_tableau is not default_tableau:
            raise AssertionError("Resolved tableau was not bound to stepper.")


def generate_step_props(n_states: int) -> dict[str, dict[str, Any]]:
    """Generate expected properties for each algorithm given n_states."""
    return {
        "euler": {
            "threads_per_step": 1,
            "persistent_local_required": 0,
            "is_multistage": False,
            "is_implicit": False,
            "is_adaptive": False,
        },
        "backwards_euler": {
            "threads_per_step": 1,
            "persistent_local_required": 0,
            "is_multistage": False,
            "is_implicit": True,
            "is_adaptive": False,
        },
        "backwards_euler_pc": {
            "threads_per_step": 1,
            "persistent_local_required": 0,
            "is_multistage": False,
            "is_implicit": True,
            "is_adaptive": False,
        },
        "crank_nicolson": {
            "threads_per_step": 1,
            "persistent_local_required": 0,
            "is_multistage": False,
            "is_implicit": True,
            "is_adaptive": True,
        },
        "rosenbrock": {
            "threads_per_step": 1,
            "persistent_local_required": 0,
            "is_multistage": True,
            "is_implicit": True,
            "is_adaptive": True,
        },
        "erk": {
            "threads_per_step": 1,
            "persistent_local_required": 0,
            "is_multistage": DEFAULT_ERK_TABLEAU.stage_count > 1,
            "is_implicit": False,
            "is_adaptive": DEFAULT_ERK_TABLEAU.has_error_estimate,
        },
        "dirk": {
            "threads_per_step": 1,
            "persistent_local_required": 0,
            "is_multistage": DEFAULT_DIRK_TABLEAU.stage_count > 1,
            "is_implicit": True,
            "is_adaptive": DEFAULT_DIRK_TABLEAU.has_error_estimate,
        },
    }

@pytest.fixture(scope="session")
def expected_step_properties(system) -> dict[str, Any]:
    """Generate expected properties for each algorithm given n_states."""
    return generate_step_props(n_states=system.sizes.states)

@pytest.fixture(scope="session")
def step_inputs(
    system,
    precision,
    initial_state,
    solver_settings,
    cpu_driver_evaluator,
) -> dict[str, Array]:
    """State, parameters, and drivers for a single step execution."""
    width = system.num_drivers
    driver_coefficients = np.array(
        cpu_driver_evaluator.coefficients, dtype=precision, copy=True
    )
    return {
        "state": initial_state,
        "parameters": system.parameters.values_array.astype(precision),
        "drivers": np.zeros(width, dtype=precision),
        "driver_coefficients": driver_coefficients,
    }


@pytest.fixture(scope="session")
def device_step_results(
    step_object,
    solver_settings,
    precision,
    step_inputs,
    cpu_driver_evaluator,
    system,
    driver_array,
) -> StepResult:
    """Execute the CUDA step and collect host-side outputs."""

    step_function = step_object.step_function
    step_size = solver_settings['dt']
    n_states = system.sizes.states
    params = step_inputs["parameters"]
    state = step_inputs["state"]
    driver_coefficients = step_inputs["driver_coefficients"]
    drivers = np.zeros(system.sizes.drivers, dtype=precision)
    observables = np.zeros(system.sizes.observables, dtype=precision)
    proposed_state = np.zeros_like(state)
    error = np.zeros(n_states, dtype=precision)
    status = np.full(1, 0, dtype=np.int32)

    shared_elems = step_object.shared_memory_required
    shared_bytes = precision(0).itemsize * shared_elems
    persistent_len = max(1, step_object.persistent_local_required)
    numba_precision = from_dtype(precision)
    dt_value = precision(step_size)

    d_state = cuda.to_device(state)
    d_proposed = cuda.to_device(proposed_state)
    d_params = cuda.to_device(params)
    d_drivers = cuda.to_device(drivers)
    d_driver_coeffs = cuda.to_device(driver_coefficients)
    proposed_drivers = np.zeros_like(drivers)

    d_observables = cuda.to_device(observables)
    d_proposed_observables = cuda.to_device(observables)
    d_proposed_drivers = cuda.to_device(proposed_drivers)
    d_error = cuda.to_device(error)
    d_status = cuda.to_device(status)

    driver_function = driver_array.evaluation_function
    observables_function = system.observables_function

    @cuda.jit
    def kernel(
        state_vec,
        proposed_vec,
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
    ) -> None:
        idx = cuda.grid(1)
        if idx > 0:
            return
        shared = cuda.shared.array(0, dtype=numba_precision)
        persistent = cuda.local.array(persistent_len, dtype=numba_precision)
        driver_function(precision(0.0), driver_coefficients, drivers_vec)
        observables_function(state, params_vec, drivers_vec, observables_vec,
                             precision(0.0))
        shared[:] = precision(0.0)
        result = step_function(
            state_vec,
            proposed_vec,
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

    kernel[1, 1, 0, shared_bytes](
        d_state,
        d_proposed,
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
    return StepResult(
        state=d_proposed.copy_to_host(),
        observables=d_proposed_observables.copy_to_host(),
        error=d_error.copy_to_host(),
        status=status_value & STATUS_MASK,
        niters=(status_value >> 16) & STATUS_MASK,
    )


@pytest.fixture(scope="session")
def cpu_step_results(
    solver_settings,
    cpu_system: CPUODESystem,
    step_inputs,
    cpu_driver_evaluator,
    step_object,
) -> StepResult:
    """Execute the CPU reference stepper."""

    tableau = getattr(step_object, "tableau", None)
    dt = solver_settings["dt"]
    state = np.asarray(step_inputs["state"], dtype=cpu_system.precision)
    params = np.asarray(step_inputs["parameters"], dtype=cpu_system.precision)
    if cpu_system.system.num_drivers > 0:
        driver_evaluator = cpu_driver_evaluator.with_coefficients(
            step_inputs["driver_coefficients"]
        )
    else:
        driver_evaluator = cpu_driver_evaluator

    stepper = get_ref_stepper(
        cpu_system,
        driver_evaluator,
        solver_settings["algorithm"],
        newton_tol=solver_settings["newton_tolerance"],
        newton_max_iters=solver_settings["max_newton_iters"],
        linear_tol=solver_settings["krylov_tolerance"],
        linear_max_iters=solver_settings["max_linear_iters"],
        tableau=tableau,
    )

    result = stepper(
        state=state,
        params=params,
        dt=dt,
        time=0.0,
    )

    return StepResult(
        state=result.state.astype(cpu_system.precision, copy=True),
        observables=result.observables.astype(cpu_system.precision, copy=True),
        error=result.error.astype(cpu_system.precision, copy=True),
        status=result.status & STATUS_MASK,
        niters=(result.status >> 16) & STATUS_MASK,
    )


#All-in-one step test to share fixture setup at the expense of readability
@pytest.mark.parametrize(
    "solver_settings_override, system_override",
    STEP_CASES,
    indirect=True,
)

def test_algorithm(
       step_object_mutable,
       solver_settings,
       system,
       precision,
       expected_step_properties,
       cpu_step_results,
       device_step_results,
       output_functions,
       tolerance,
       ) -> None:
    """Ensure the step function is compiled and callable."""
    step_object = step_object_mutable
    # Test that it builds
    assert callable(step_object.step_function), "step_function_builds"

    # test getters
    algorithm = solver_settings["algorithm"]
    properties = expected_step_properties.get(algorithm)
    if properties is not None:
        assert step_object.is_implicit is properties["is_implicit"], \
            "is_implicit getter"
        assert step_object.is_adaptive is properties["is_adaptive"], \
            "is_adaptive getter"
        assert step_object.is_multistage is properties["is_multistage"],\
            "is_multistage getter"
        assert step_object.persistent_local_required \
            == properties["persistent_local_required"], \
            "persistent_local_required getter"
        assert (
            step_object.threads_per_step == properties["threads_per_step"]
        ), "threads_per_step getter"
    config = step_object.compile_settings
    assert config.n == system.sizes.states, "compile_settings.n getter"
    assert config.precision == precision, "compile_settings.precision getter"

    tableau = getattr(step_object, "tableau", None)
    if tableau is not None and tableau.b_hat is not None:
        expected_error = tuple(
            b_value - b_hat_value
            for b_value, b_hat_value in zip(tableau.b, tableau.b_hat)
        )
        assert tableau.d == pytest.approx(expected_error), "embedded weights"

    expected_order = _expected_order(step_object, tableau)
    assert step_object.order == expected_order, "order getter"

    extra_shared = system._jacobian_aux_count or 0
    expected_shared, expected_local = _expected_memory_requirements(
        step_object,
        tableau,
        system.sizes.states,
        extra_shared,
    )
    assert (
        step_object.shared_memory_required == expected_shared
    ), "shared_memory_required getter"
    assert (
        step_object.local_scratch_required == expected_local
    ), "local_scratch_required getter"

    if properties is not None and properties["is_implicit"]:
        if algorithm == "rosenbrock":
            assert config.max_linear_iters == solver_settings[
                "max_linear_iters"
            ], "max_linear_iters set"
            assert config.linear_correction_type == solver_settings[
                "correction_type"
            ], "linear_correction_type set"
            assert config.krylov_tolerance == pytest.approx(
                solver_settings["krylov_tolerance"],
                rel=tolerance.rel_tight,
                abs=tolerance.abs_tight,
            ), "krylov_tolerance set"
        else:
            matrix = config.M
            assert matrix.shape == (system.sizes.states, system.sizes.states)
            assert config.preconditioner_order == solver_settings[
                "preconditioner_order"
            ], "preconditioner order set"
            assert config.max_linear_iters == solver_settings[
                "max_linear_iters"
            ], "max_linear_iters set"
            assert config.linear_correction_type == solver_settings[
                "correction_type"
            ], "linear_correction_type set"
            assert config.max_newton_iters == solver_settings[
                "max_newton_iters"
            ], "max_newton_iters set"
            assert config.newton_max_backtracks == solver_settings[
                "newton_max_backtracks"
            ], "newton_max_backtracks set"
            assert config.krylov_tolerance == pytest.approx(
                solver_settings["krylov_tolerance"],
                rel=tolerance.rel_tight,
                abs=tolerance.abs_tight,
            ), "krylov_tolerance set"
            assert config.newton_tolerance == pytest.approx(
                solver_settings["newton_tolerance"],
                rel=tolerance.rel_tight,
                abs=tolerance.abs_tight,
            ), "newton_tolerance set"
            assert config.newton_damping == pytest.approx(
                solver_settings["newton_damping"],
                rel=tolerance.rel_tight,
                abs=tolerance.abs_tight,
            ), "newton_damping set"
        assert callable(system.get_solver_helper)
    elif properties is not None:
        assert step_object.dt == pytest.approx(
            solver_settings["dt_min"],
            rel=tolerance.rel_tight,
            abs=tolerance.abs_tight,
        )

    if step_object.is_implicit:
        if algorithm == "rosenbrock":
            updates = {
                "max_linear_iters": max(
                    1, solver_settings["max_linear_iters"] // 2
                ),
                "krylov_tolerance": solver_settings["krylov_tolerance"] * 0.5,
                "linear_correction_type": "steepest_descent",
            }
            recognised = step_object.update(updates)
            assert set(updates).issubset(recognised), "updates recognised"
            config = step_object.compile_settings
            assert config.max_linear_iters == updates["max_linear_iters"], \
                "max_linear_iters update"
            assert config.linear_correction_type == updates[
                "linear_correction_type"
            ], "linear_correction_type update"
            assert config.krylov_tolerance == pytest.approx(
                updates["krylov_tolerance"],
                rel=tolerance.rel_tight,
                abs=tolerance.abs_tight,
            ), "krylov_tolerance update"
        else:
            updates = {
                "max_newton_iters": int(
                    max(1, solver_settings["max_newton_iters"] // 2)
                ),
                "krylov_tolerance":
                solver_settings["krylov_tolerance"] * 0.5,
                "newton_tolerance":
                solver_settings["newton_tolerance"] * 0.5,
                "newton_damping":
                solver_settings["newton_damping"] * 0.9,
                "preconditioner_order":
                solver_settings["preconditioner_order"] + 1,
            }
            recognised = step_object.update(updates)
            assert set(updates).issubset(recognised), "updates recognised"
            config = step_object.compile_settings
            assert config.max_newton_iters == updates["max_newton_iters"], \
                "max_newton_iters update"
            assert config.preconditioner_order == updates[
                "preconditioner_order"
            ], "preconditioner_order update"
            assert config.krylov_tolerance == pytest.approx(
                updates["krylov_tolerance"],
                rel=tolerance.rel_tight,
                abs=tolerance.abs_tight,
            ), "krylov_tolerance update"
            assert config.newton_tolerance == pytest.approx(
                updates["newton_tolerance"],
                rel=tolerance.rel_tight,
                abs=tolerance.abs_tight,
            ), "newton_tolerance update"
            assert config.newton_damping == pytest.approx(
                updates["newton_damping"],
                rel=tolerance.rel_tight,
                abs=tolerance.abs_tight,
            ), "newton_damping update"
    else:
        new_dt = float(solver_settings["dt_min"]) * 0.5
        recognised = step_object.update({"dt": new_dt})
        assert "dt" in recognised, "dt recognised"
        assert step_object.dt == pytest.approx(
            new_dt,
            rel=tolerance.rel_tight,
            abs=tolerance.abs_tight,
        ), "dt update"

    # Test equality for a single step
    tolerances = {
        "rtol": tolerance.rel_tight,
        "atol": tolerance.abs_tight,
    }
    assert device_step_results.status == cpu_step_results.status, \
        "status matches"

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


    assert device_step_results.niters == cpu_step_results.niters, \
        "niters match"