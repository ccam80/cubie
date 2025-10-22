"""Reusable tester for single-step integration algorithms."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from numba import cuda, from_dtype, int16
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


@dataclass
class DualStepResult:
    """Container recording back-to-back step executions."""

    first_state: Array
    second_state: Array
    first_observables: Array
    second_observables: Array
    first_error: Array
    second_error: Array
    statuses: tuple[int, int]


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
            "newton_tolerance": 1e-7,
            "krylov_tolerance": 1e-7,
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
            "newton_tolerance": 1e-7,
            "krylov_tolerance": 1e-7,
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
            "newton_tolerance": 1e-7,
            "krylov_tolerance": 1e-7,
        },
        {},
        id="backwards_euler_pc",
    ),
    pytest.param(
        {
            "algorithm": "crank_nicolson",
            "atol": 1e-6,
            "rtol": 1e-6,
            "dt_min": 1e-6,
            "dt": 0.0025,
            "dt_save": 0.2,
            "output_types": ["state"],
            "saved_state_indices": [0, 1, 2],
            "newton_tolerance": 1e-7,
            "krylov_tolerance": 1e-7,
        },
        {},
        id="crank_nicolson",
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
            "newton_tolerance": 1e-7,
            "krylov_tolerance": 1e-7,
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
            "newton_tolerance": 1e-7,
            "krylov_tolerance": 1e-7,
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
            "newton_tolerance": 1e-7,
            "krylov_tolerance": 1e-7,
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
            "newton_tolerance": 1e-7,
            "krylov_tolerance": 1e-7,
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
            "newton_tolerance": 1e-7,
            "krylov_tolerance": 1e-7,
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
            "newton_tolerance": 1e-7,
            "krylov_tolerance": 1e-7,
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
            "newton_tolerance": 1e-7,
            "krylov_tolerance": 1e-7,
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
            "newton_tolerance": 1e-7,
            "krylov_tolerance": 1e-7,
        },
        {},
        id="dirk-lobatto-iiic-3",
        marks=pytest.mark.specific_algos,
    ),
]


def _cache_case(**kwargs: Any) -> dict[str, Any]:
    """Return overrides for cache reuse scenarios."""

    base = {
        "dt": 0.0025,
        "dt_min": 0.0025,
        "dt_save": 0.2,
        "output_types": ["state"],
        "saved_state_indices": [0, 1, 2],
    }
    base.update(kwargs)
    return base


CACHE_REUSE_CASES = [
    pytest.param(
        _cache_case(algorithm="heun-21", step_controller="fixed"),
        id="erk-heun-21-cache",
        marks=pytest.mark.specific_algos,
    ),
    pytest.param(
        _cache_case(algorithm="ralston-33", step_controller="fixed"),
        id="erk-ralston-33-cache",
        marks=pytest.mark.specific_algos,
    ),
    pytest.param(
        _cache_case(
            algorithm="trapezoidal_dirk",
            step_controller="fixed",
            newton_tolerance=1e-7,
            krylov_tolerance=1e-7,
        ),
        id="dirk-trapezoidal-cache",
        marks=pytest.mark.specific_algos,
    ),
    pytest.param(
        _cache_case(
            algorithm="sdirk_2_2",
            step_controller="pi",
            atol=1e-6,
            rtol=1e-6,
            dt_min=1e-6,
            newton_tolerance=1e-7,
            krylov_tolerance=1e-7,
        ),
        id="dirk-sdirk-2-2-cache",
        marks=pytest.mark.specific_algos,
    ),
    pytest.param(
        _cache_case(
            algorithm="ros3p",
            step_controller="pi",
            atol=1e-6,
            rtol=1e-6,
            dt_min=1e-6,
            newton_tolerance=1e-7,
            krylov_tolerance=1e-7,
        ),
        id="rosenbrock-ros3p-cache",
        marks=pytest.mark.specific_algos,
    ),
    pytest.param(
        _cache_case(
            algorithm="rosenbrock_w6s4os",
            step_controller="pi",
            atol=1e-6,
            rtol=1e-6,
            dt_min=1e-6,
            newton_tolerance=1e-7,
            krylov_tolerance=1e-7,
        ),
        id="rosenbrock-w6s4os-cache",
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
        linear_correction_type="minimal_residual",
        preconditioner_order=2,
    )

    assert isinstance(bound_step, expected_cpu_step)
    assert bound_step.tableau is expected_tableau

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
        first_step_flag = int16(1)
        accepted_flag = int16(1)
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
            first_step_flag,
            accepted_flag,
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


def _execute_step_twice(
    step_object,
    solver_settings,
    precision,
    step_inputs,
    system,
    driver_array,
) -> DualStepResult:
    """Run the compiled step twice without clearing shared memory."""

    shared_elems = step_object.shared_memory_required
    if shared_elems == 0:
        pytest.skip("Algorithm does not expose a shared cache to reuse.")

    step_function = step_object.step_function
    driver_function = driver_array.evaluation_function
    observables_function = system.observables_function

    params = step_inputs["parameters"]
    state = np.asarray(step_inputs["state"], dtype=precision)
    driver_coefficients = step_inputs["driver_coefficients"]

    n_states = system.sizes.states
    n_drivers = system.sizes.drivers
    n_observables = system.sizes.observables

    proposed_state_first = np.zeros_like(state)
    proposed_state_second = np.zeros_like(state)

    error_first = np.zeros(n_states, dtype=precision)
    error_second = np.zeros(n_states, dtype=precision)

    drivers_current = np.zeros(n_drivers, dtype=precision)
    proposed_drivers_first = np.zeros(n_drivers, dtype=precision)
    proposed_drivers_second = np.zeros(n_drivers, dtype=precision)

    observables_current = np.zeros(n_observables, dtype=precision)
    proposed_observables_first = np.zeros(n_observables, dtype=precision)
    proposed_observables_second = np.zeros(n_observables, dtype=precision)

    status = np.zeros(2, dtype=np.int32)

    shared_bytes = precision(0).itemsize * shared_elems
    persistent_len = max(1, step_object.persistent_local_required)
    numba_precision = from_dtype(precision)
    dt_value = precision(solver_settings["dt"])

    d_state = cuda.to_device(state)
    d_params = cuda.to_device(params)
    d_driver_coeffs = cuda.to_device(driver_coefficients)

    d_proposed_first = cuda.to_device(proposed_state_first)
    d_proposed_second = cuda.to_device(proposed_state_second)

    d_drivers_current = cuda.to_device(drivers_current)
    d_proposed_drivers_first = cuda.to_device(proposed_drivers_first)
    d_proposed_drivers_second = cuda.to_device(proposed_drivers_second)

    d_observables_current = cuda.to_device(observables_current)
    d_proposed_observables_first = cuda.to_device(proposed_observables_first)
    d_proposed_observables_second = cuda.to_device(proposed_observables_second)

    d_error_first = cuda.to_device(error_first)
    d_error_second = cuda.to_device(error_second)

    d_status = cuda.to_device(status)

    state_len = int(n_states)
    driver_len = int(n_drivers)
    observable_len = int(n_observables)

    @cuda.jit
    def kernel(
        state_vec,
        params_vec,
        driver_coeffs_vec,
        drivers_current_vec,
        proposed_drivers_vec_first,
        proposed_drivers_vec_second,
        observables_current_vec,
        proposed_observables_vec_first,
        proposed_observables_vec_second,
        proposed_vec_first,
        proposed_vec_second,
        error_vec_first,
        error_vec_second,
        status_vec,
        dt_scalar,
    ) -> None:
        idx = cuda.grid(1)
        if idx > 0:
            return
        shared = cuda.shared.array(0, dtype=numba_precision)
        persistent = cuda.local.array(persistent_len, dtype=numba_precision)
        zero = numba_precision(0.0)

        for cache_idx in range(shared.shape[0]):
            shared[cache_idx] = zero

        driver_function(zero, driver_coefficients, drivers_current_vec)
        observables_function(
            state_vec,
            params_vec,
            drivers_current_vec,
            observables_current_vec,
            zero,
        )

        first_status = step_function(
            state_vec,
            proposed_vec_first,
            params_vec,
            driver_coeffs_vec,
            drivers_current_vec,
            proposed_drivers_vec_first,
            observables_current_vec,
            proposed_observables_vec_first,
            error_vec_first,
            dt_scalar,
            zero,
            int16(1),
            int16(1),
            shared,
            persistent,
        )
        status_vec[0] = first_status

        for elem in range(state_len):
            state_vec[elem] = proposed_vec_first[elem]
        for drv_idx in range(driver_len):
            drivers_current_vec[drv_idx] = proposed_drivers_vec_first[drv_idx]
        for obs_idx in range(observable_len):
            observables_current_vec[obs_idx] = proposed_observables_vec_first[
                obs_idx
            ]

        second_status = step_function(
            state_vec,
            proposed_vec_second,
            params_vec,
            driver_coeffs_vec,
            drivers_current_vec,
            proposed_drivers_vec_second,
            observables_current_vec,
            proposed_observables_vec_second,
            error_vec_second,
            dt_scalar,
            dt_scalar,
            int16(0),
            int16(1),
            shared,
            persistent,
        )
        status_vec[1] = second_status

    kernel[1, 1, 0, shared_bytes](
        d_state,
        d_params,
        d_driver_coeffs,
        d_drivers_current,
        d_proposed_drivers_first,
        d_proposed_drivers_second,
        d_observables_current,
        d_proposed_observables_first,
        d_proposed_observables_second,
        d_proposed_first,
        d_proposed_second,
        d_error_first,
        d_error_second,
        d_status,
        dt_value,
    )
    cuda.synchronize()

    status_host = d_status.copy_to_host()

    first_state = d_proposed_first.copy_to_host()
    second_state = d_proposed_second.copy_to_host()

    first_observables = d_proposed_observables_first.copy_to_host()
    second_observables = d_proposed_observables_second.copy_to_host()

    first_error = d_error_first.copy_to_host()
    second_error = d_error_second.copy_to_host()

    statuses = (
        int(status_host[0]) & STATUS_MASK,
        int(status_host[1]) & STATUS_MASK,
    )

    return DualStepResult(
        first_state=first_state,
        second_state=second_state,
        first_observables=first_observables,
        second_observables=second_observables,
        first_error=first_error,
        second_error=second_error,
        statuses=statuses,
    )


def _execute_cpu_step_twice(
    solver_settings,
    step_inputs,
    cpu_system: CPUODESystem,
    cpu_driver_evaluator,
    step_object,
) -> DualStepResult:
    """Run the CPU reference step twice with shared cache reuse enabled."""

    tableau = getattr(step_object, "tableau", None)
    dt = solver_settings["dt"]
    precision = cpu_system.precision

    state = np.asarray(step_inputs["state"], dtype=precision)
    params = np.asarray(step_inputs["parameters"], dtype=precision)

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
        linear_correction_type=solver_settings["correction_type"],
        preconditioner_order=solver_settings["preconditioner_order"],
        tableau=tableau,
        newton_damping=solver_settings["newton_damping"],
        newton_max_backtracks=solver_settings["newton_max_backtracks"],
    )

    first_result = stepper(
        state=state,
        params=params,
        dt=dt,
        time=0.0,
    )

    second_result = stepper(
        state=first_result.state.astype(precision, copy=True),
        params=params,
        dt=dt,
        time=dt,
    )

    return DualStepResult(
        first_state=first_result.state.astype(precision, copy=True),
        second_state=second_result.state.astype(precision, copy=True),
        first_observables=first_result.observables.astype(precision, copy=True),
        second_observables=second_result.observables.astype(
            precision, copy=True
        ),
        first_error=first_result.error.astype(precision, copy=True),
        second_error=second_result.error.astype(precision, copy=True),
        statuses=(
            first_result.status & STATUS_MASK,
            second_result.status & STATUS_MASK,
        ),
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
        linear_correction_type=solver_settings["correction_type"],
        preconditioner_order=solver_settings["preconditioner_order"],
        tableau=tableau,
        newton_damping=solver_settings["newton_damping"],
        newton_max_backtracks=solver_settings["newton_max_backtracks"],
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


# Cached reuse validation
@pytest.mark.parametrize(
    "solver_settings_override",
    CACHE_REUSE_CASES,
    indirect=["solver_settings_override"],
)
def test_stage_cache_reuse(
    solver_settings,
    step_object,
    precision,
    step_inputs,
    system,
    driver_array,
    cpu_system,
    cpu_driver_evaluator,
):
    """Ensure shared-cache reuse yields consistent results across devices."""

    gpu_result = _execute_step_twice(
        step_object=step_object,
        solver_settings=solver_settings,
        precision=precision,
        step_inputs=step_inputs,
        system=system,
        driver_array=driver_array,
    )

    cpu_result = _execute_cpu_step_twice(
        solver_settings=solver_settings,
        step_inputs=step_inputs,
        cpu_system=cpu_system,
        cpu_driver_evaluator=cpu_driver_evaluator,
        step_object=step_object,
    )

    assert all(status == 0 for status in gpu_result.statuses)
    assert all(status == 0 for status in cpu_result.statuses)
    assert gpu_result.statuses == cpu_result.statuses

    tol = {"rtol": 1e-7, "atol": 1e-7}

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

    delta = np.abs(gpu_result.second_state - gpu_result.first_state)
    assert np.any(delta > precision(1e-10))


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
    algorithm = solver_settings[("algorithm")]
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

    # This is not going to work; the CPU solvers perform better and tend to
    # converge more rapidly.
    # assert device_step_results.niters == cpu_step_results.niters, \
    #     "niters match"


def test_rosenbrock_solver_helper_cache_refresh(system) -> None:
    """Rosenbrock helpers rebuild when tableau parameters change."""

    ros3p = ROSENBROCK_TABLEAUS["ros3p"]
    w6s4os = ROSENBROCK_TABLEAUS["rosenbrock_w6s4os"]
    mass_matrix = np.eye(system.sizes.states, dtype=system.precision)

    first = system.get_solver_helper(
        "linear_operator",
        beta=1.0,
        gamma=ros3p.gamma,
        preconditioner_order=2,
        mass=mass_matrix,
    )
    repeat = system.get_solver_helper(
        "linear_operator",
        beta=1.0,
        gamma=ros3p.gamma,
        preconditioner_order=2,
        mass=mass_matrix,
    )
    assert repeat is first

    second = system.get_solver_helper(
        "linear_operator",
        beta=1.0,
        gamma=w6s4os.gamma,
        preconditioner_order=2,
        mass=mass_matrix,
    )
    assert second is not first

    gamma_cached = system.compile_settings.gamma
    assert gamma_cached == pytest.approx(w6s4os.gamma)
    assert np.array_equal(system.compile_settings.mass, mass_matrix)
