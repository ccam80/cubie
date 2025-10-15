"""Reusable tester for single-step integration algorithms."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from numba import cuda, from_dtype
from numpy.testing import assert_allclose

from cubie.integrators.algorithms import get_algorithm_step
from cubie.integrators.algorithms.generic_dirk import DIRKStep
from cubie.integrators.algorithms.generic_dirk_tableaus import (
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
    ROSENBROCK_TABLEAUS,
)
from tests.integrators.cpu_reference import (
    CPUODESystem,
    dirk_step,
    erk_step,
    get_ref_step_function,
    rosenbrock_step,
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


ALIAS_CASES = [
    pytest.param(
        "dormand-prince-54",
        ERKStep,
        ERK_TABLEAU_REGISTRY["dormand-prince-54"],
        erk_step,
        id="erk-dormand-prince-54",
    ),
    pytest.param(
        "dopri54",
        ERKStep,
        DEFAULT_ERK_TABLEAU,
        erk_step,
        marks=pytest.mark.specific_algos,
        id="erk-dopri54",
    ),
    pytest.param(
        "cash-karp-54",
        ERKStep,
        ERK_TABLEAU_REGISTRY["cash-karp-54"],
        erk_step,
        marks=pytest.mark.specific_algos,
        id="erk-cash-karp-54",
    ),
    pytest.param(
        "sdirk_2_2",
        DIRKStep,
        DIRK_TABLEAU_REGISTRY["sdirk_2_2"],
        dirk_step,
        id="dirk-sdirk-2-2",
    ),
    pytest.param(
        "lobatto_iiic_3",
        DIRKStep,
        DIRK_TABLEAU_REGISTRY["lobatto_iiic_3"],
        dirk_step,
        marks=pytest.mark.specific_algos,
        id="dirk-lobatto-iiic-3",
    ),
    pytest.param(
        "ros3p",
        GenericRosenbrockWStep,
        ROSENBROCK_TABLEAUS["ros3p"],
        rosenbrock_step,
        id="rosenbrock-ros3p",
    ),
]

@pytest.mark.parametrize(
    "alias_key, expected_step_type, expected_tableau, cpu_step",
    ALIAS_CASES,
)
def test_algorithm_factory_resolves_tableau_alias(
    alias_key,
    expected_step_type,
    expected_tableau,
    cpu_step,
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
    "alias_key, expected_step_type, expected_tableau, cpu_step",
    ALIAS_CASES,
)
def test_cpu_reference_resolves_tableau_alias(
    alias_key,
    expected_step_type,
    expected_tableau,
    cpu_step,
):
    """CPU reference helpers should resolve alias tableaus consistently."""

    stepper = get_ref_step_function(alias_key)
    assert stepper.func is cpu_step
    assert stepper.keywords["tableau"] is expected_tableau


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
            "shared_memory_required": 2 * n_states,
            "local_scratch_required": 0,
        },
        "backwards_euler_pc": {
            "threads_per_step": 1,
            "persistent_local_required": 0,
            "is_multistage": False,
            "is_implicit": True,
            "is_adaptive": False,
            "order": 1,
            "shared_memory_required": 2 * n_states,
            "local_scratch_required": 0,
        },
        "crank_nicolson": {
            "threads_per_step": 1,
            "persistent_local_required": 0,
            "is_multistage": False,
            "is_implicit": True,
            "is_adaptive": True,
            "order": 2,
            "shared_memory_required": 2 * n_states,
            "local_scratch_required": 0,
        },
        "rosenbrock": {
            "threads_per_step": 1,
            "persistent_local_required": 0,
            "is_multistage": True,
            "is_implicit": True,
            "is_adaptive": True,
            "order": 4,
            "shared_memory_required": 10 * n_states,
            "local_scratch_required": 4 * n_states,
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
    step_function = get_ref_step_function(
        solver_settings["algorithm"], tableau=tableau
    )
    dt = solver_settings["dt"]
    state = np.asarray(step_inputs["state"], dtype=cpu_system.precision)
    params = np.asarray(step_inputs["parameters"], dtype=cpu_system.precision)
    if cpu_system.system.num_drivers > 0:
        driver_evaluator = cpu_driver_evaluator.with_coefficients(
            step_inputs["driver_coefficients"]
        )
    else:
        driver_evaluator = cpu_driver_evaluator

    result = step_function(
        cpu_system,
        driver_evaluator,
        state=state,
        params=params,
        dt=dt,
        tol=solver_settings['newton_tolerance'],
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
    [
        [{"algorithm": "euler", 'step_controller': 'fixed'}, {}],
        [{"algorithm": "backwards_euler"}, {}],
        [{"algorithm": "backwards_euler_pc", "dt_min": 0.0025}, {}],
        [{"algorithm": "crank_nicolson", 'step_controller': 'pid', 'atol':
            1e-6, 'rtol': 1e-6, 'dt_min': 1e-6}, {}],
        [{"algorithm": "rosenbrock", 'step_controller': 'pi',
         'krylov_tolerance': 1e-7}, {}],
    ],
    ids=[
        "euler",
        "backwards_euler",
        "backwards_euler_pc",
        "crank_nicolson",
        "rosenbrock",
    ],
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
    properties = expected_step_properties[algorithm]
    assert step_object.is_implicit is properties["is_implicit"], \
        "is_implicit getter"
    assert step_object.is_adaptive is properties["is_adaptive"], \
        "is_adaptive getter"
    assert step_object.is_multistage is properties["is_multistage"],\
        "is_multistage getter"
    # assert step_object.order == properties["order"],\
    #     "order getter"
    # TODO: Update to fetch tableau order
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

    tableau = getattr(step_object, "tableau", None)
    if tableau is not None and tableau.b_hat is not None:
        expected_error = tuple(
            b_value - b_hat_value
            for b_value, b_hat_value in zip(tableau.b, tableau.b_hat)
        )
        assert tableau.d == pytest.approx(expected_error), "embedded weights"

    if properties["is_implicit"]:
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
    else:
        assert step_object.dt == pytest.approx(
            solver_settings["dt_min"],
            rel=tolerance.rel_tight,
            abs=tolerance.abs_tight,
        )

    if step_object.is_implicit:
        if algorithm == "rosenbrock":
            updates = {
                "max_linear_iters": max(1, solver_settings["max_linear_iters"] // 2),
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

    if system._jacobian_aux_count is not None:
        extra_shared = system._jacobian_aux_count
    else:
        extra_shared = 0
    # assert (
    #     step_object.shared_memory_required
    #     == (properties["shared_memory_required"] + extra_shared)
    # ), "shared_memory_required getter" #TODO: this needs to be tableau based

    # Test equality for a single step
    tolerances = {
        "rtol": tolerance.rel_tight,
        "atol": tolerance.abs_tight,
    }
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


