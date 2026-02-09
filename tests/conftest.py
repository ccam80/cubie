from pathlib import Path
from types import SimpleNamespace
import os
import tempfile
import shutil

import numpy as np
import pytest
from pytest import MonkeyPatch

from tests._utils import (
    _build_solver_instance,
    _build_cpu_step_controller,
    _get_algorithm_order,
    _get_algorithm_tableau,
    _build_enhanced_algorithm_settings,
    _get_evaluate_driver_at_t,
    _get_driver_del_t,
    _run_device_step,
    _run_device_algorithm_step,
    _execute_cpu_step_twice as _cpu_step_twice,
    StepResult,
    AlgorithmStepResult,
    STATUS_MASK,
)
import attrs

from cubie.batchsolving.BatchInputHandler import BatchInputHandler
from cubie.batchsolving.SystemInterface import SystemInterface
from cubie.buffer_registry import buffer_registry
from cubie.integrators.SingleIntegratorRun import SingleIntegratorRun
from cubie._utils import merge_kwargs_into_settings
from cubie.integrators.step_control import get_controller
from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel
from cubie.integrators.algorithms.base_algorithm_step import (
    ALL_ALGORITHM_STEP_PARAMETERS,
)
from cubie.integrators.loops.ode_loop import ALL_LOOP_SETTINGS

from cubie.integrators.step_control.base_step_controller import (
    ALL_STEP_CONTROLLER_PARAMETERS,
)
from cubie.integrators.array_interpolator import ArrayInterpolator
from cubie.memory import default_memmgr
from cubie.memory.mem_manager import (
    ALL_MEMORY_MANAGER_PARAMETERS,
    MemoryManager,
)
from cubie.outputhandling.output_functions import (
    OutputFunctions,
    ALL_OUTPUT_FUNCTION_PARAMETERS,
)
from tests.integrators.cpu_reference import (
    CPUODESystem,
    DriverEvaluator,
    run_reference_loop,
    get_ref_stepper,
)

from tests._utils import _driver_sequence, run_device_loop
from numpy.typing import NDArray
Array = NDArray[np.floating]
from tests.system_fixtures import (
    build_large_nonlinear_system,
    build_three_chamber_system,
    build_three_state_constant_deriv_system,
    build_three_state_linear_system,
    build_three_state_nonlinear_system,
    build_three_state_very_stiff_system,
)

enable_tempdir = "1"
os.environ["CUBIE_GENERATED_DIR_REDIRECT"] = enable_tempdir
np.set_printoptions(linewidth=120, threshold=np.inf, precision=12)


# --------------------------------------------------------------------------- #
#                           Test ordering hook                                #
# --------------------------------------------------------------------------- #
@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    # move tests which close the CUDA context to the very end, so that the
    # streams used in session-scoped fixtures don't disappear on them mid-run
    final_basenames = {"test_cupyemm.py", "test_memmgmt.py"}
    # iterate over a shallow copy to allow safe in-place removals/appends
    for item in items[:]:
        if item.fspath.basename in final_basenames:
            items.remove(item)
            items.append(item)
    pass


# --------------------------------------------------------------------------- #
#                            Codegen Redirect                                 #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session", autouse=True)
def codegen_dir():
    """Redirect code generation to a temporary directory for the whole session.

    Use tempfile.mkdtemp instead of pytest's tmp path so the directory isn't
    removed automatically between parameterized test cases. Remove the
    directory at session teardown.

    Toggle: set environment variable `CUBIE_GENERATED_DIR_REDIRECT` to `0` to
    disable the temporary redirect and keep the original
    `odefile.GENERATED_DIR`.
    """
    import os
    from cubie.odesystems.symbolic import odefile

    original_dir = getattr(odefile, "GENERATED_DIR", None)
    redirect_enabled = int(os.environ.get("CUBIE_GENERATED_DIR_REDIRECT", "1"))

    if not redirect_enabled:
        # Don't change odefile.GENERATED_DIR; yield the original value (or None).
        yield Path(original_dir) if original_dir is not None else None
        return

    gen_dir = Path(tempfile.mkdtemp(prefix="cubie_generated_"))
    mp = MonkeyPatch()
    mp.setattr(odefile, "GENERATED_DIR", gen_dir, raising=True)
    try:
        yield gen_dir
    finally:
        # restore original attribute and remove temporary dir. Wrap in
        # try/except in case multiple workers attempt to delete the same
        # directory when running tests in parallel.
        try:
            mp.undo()
            shutil.rmtree(gen_dir, ignore_errors=True)
        except PermissionError:
            pass


# ========================================
# SETTINGS DICTS (override -> fixture -> override -> fixture)
# ========================================


@pytest.fixture(scope="session")
def precision(solver_settings_override, solver_settings_override2):
    """Return precision from overrides, defaulting to float32.

    Precedence: override2 (class-level) is checked first, then override
    (method-level). This allows class-level precision to be overridden
    by individual test methods.

    Usage:
    @pytest.mark.parametrize("solver_settings_override",
        [{"precision": np.float64}], indirect=True)
    def test_something(precision):
        # precision will be np.float64 here
    """
    # Check override2 first (class-level), then override (method-level)
    for override in [solver_settings_override2, solver_settings_override]:
        if override and "precision" in override:
            return override["precision"]
    return np.float32


@pytest.fixture(scope="session")
def tolerance_override(request):
    if hasattr(request, "param"):
        return request.param
    return None


@pytest.fixture(scope="session")
def tolerance(tolerance_override, precision):
    if tolerance_override is not None:
        return tolerance_override

    if precision == np.float32:
        return SimpleNamespace(
            abs_loose=1e-5,
            abs_tight=1e-7,
            rel_loose=1e-5,
            rel_tight=1e-7,
        )

    if precision == np.float64:
        return SimpleNamespace(
            abs_loose=1e-9,
            abs_tight=1e-12,
            rel_loose=1e-9,
            rel_tight=1e-12,
        )

    raise ValueError("Unsupported precision for tolerance fixture")


@pytest.fixture(scope="session")
def system(
    request, solver_settings_override, solver_settings_override2, precision
):
    """Return the appropriate symbolic system, defaulting to nonlinear.

    Usage:
    @pytest.mark.parametrize("solver_settings_override",
        [{"system_type": "three_chamber"}], indirect=True)
    def test_something(system):
        # system will be the cardiovascular symbolic model here
    """
    model_type = "nonlinear"
    for override in [solver_settings_override2, solver_settings_override]:
        if override and "system_type" in override:
            model_type = override["system_type"]
            break

    if model_type == "linear":
        return build_three_state_linear_system(precision)
    if model_type == "nonlinear":
        return build_three_state_nonlinear_system(precision)
    if model_type in ["three_chamber", "threecm"]:
        return build_three_chamber_system(precision)
    if model_type == "stiff":
        return build_three_state_very_stiff_system(precision)
    if model_type == "large":
        return build_large_nonlinear_system(precision)
    if model_type == "constant_deriv":
        return build_three_state_constant_deriv_system(precision)
    if isinstance(model_type, object):
        return model_type

    raise ValueError(f"Unknown model type: {model_type}")


@pytest.fixture(scope="session")
def thread_mem_manager():
    """Instantiate a memory manager instance in each thread"""
    return MemoryManager()


@pytest.fixture(scope="session")
def solver_settings_override(request):
    """Override for solver settings, if provided."""
    return request.param if hasattr(request, "param") else {}


@pytest.fixture(scope="session")
def solver_settings_override2(request):
    """Override for solver settings, if provided. A second one, so that we
    can do a class-level and function-level override without conflicts."""
    return request.param if hasattr(request, "param") else {}


@pytest.fixture(scope="session")
def solver_settings(
    solver_settings_override, solver_settings_override2, system, precision
):
    """Create LoopStepConfig with default solver configuration."""
    # Clear the buffer_registry singleton when we set up a new system
    buffer_registry.reset()
    defaults = {
        "algorithm": "euler",
        "system_type": "nonlinear",
        "duration": np.float64(0.2),
        "warmup": np.float64(0.0),
        "t0": np.float64(0.0),
        "dt": precision(0.01),
        "dt_min": precision(1e-7),
        "dt_max": precision(1.0),
        "save_every": precision(0.02),
        "summarise_every": precision(0.04),
        "sample_summaries_every": precision(0.02),
        "atol": precision(1e-6),
        "rtol": precision(1e-6),
        "saved_state_indices": [0, 1],
        "saved_observable_indices": [0, 1],
        "summarised_state_indices": [0, 1],
        "summarised_observable_indices": [0, 1],
        "output_types": ["state", "time", "observables", "mean"],
        "blocksize": 32,
        "stream": 0,
        "profileCUDA": False,
        "memory_manager": default_memmgr,
        "stream_group": "test_group",
        "mem_proportion": None,
        "step_controller": "fixed",
        "precision": precision,
        "driverspline_order": 3,
        "driverspline_wrap": False,
        "driverspline_boundary_condition": "clamped",
        "krylov_atol": precision(1e-7),
        "krylov_rtol": precision(1e-7),
        "linear_correction_type": "minimal_residual",
        "newton_atol": precision(1e-7),
        "newton_rtol": precision(1e-7),
        "preconditioner_order": 2,
        "krylov_max_iters": 50,
        "newton_max_iters": 50,
        "newton_damping": precision(0.85),
        "newton_max_backtracks": 25,
        "min_gain": precision(0.1),
        "max_gain": precision(5.0),
        "safety": precision(0.9),
        "n": system.sizes.states,
        "kp": precision(0.7),
        "ki": precision(-0.4),
        "kd": precision(0.0),
        "deadband_min": precision(0.95),
        "deadband_max": precision(1.05),
    }

    float_keys = {
        "duration",
        "warmup",
        "dt",
        "dt_min",
        "dt_max",
        "save_every",
        "summarise_every",
        "sample_summaries_every",
        "atol",
        "rtol",
        "krylov_atol",
        "krylov_rtol",
        "newton_atol",
        "newton_rtol",
        "newton_damping",
        "kp",
        "ki",
        "kd",
        "deadband_min",
        "deadband_max",
    }
    for override in [solver_settings_override, solver_settings_override2]:
        if override:
            # Update defaults with any overrides provided
            for key, value in override.items():
                if key in float_keys:
                    # Handle None values for optional float parameters
                    if value is None:
                        defaults[key] = None
                    else:
                        defaults[key] = precision(value)
                else:
                    defaults[key] = value

    # Add derived metadata
    defaults["algorithm_order"] = _get_algorithm_order(defaults["algorithm"])
    defaults["n_states"] = system.sizes.states
    defaults["n_parameters"] = system.sizes.parameters
    defaults["n_drivers"] = system.sizes.drivers
    defaults["n_observables"] = system.sizes.observables

    return defaults


@pytest.fixture(scope="session")
def driver_settings_override(request):
    """Optional override for driver array configuration."""

    return request.param if hasattr(request, "param") else None


@pytest.fixture(scope="session")
def driver_settings(
    driver_settings_override,
    solver_settings,
    system,
    precision,
):
    """Return default driver samples mapped to system driver symbols."""

    if system.num_drivers == 0:
        return None

    if solver_settings["save_every"] is None:
        dt_sample = solver_settings["duration"] / 10.0
    else:
        dt_sample = precision(solver_settings["save_every"]) / 2.0
    total_span = precision(solver_settings["duration"])
    t0 = precision(solver_settings["warmup"])

    order = int(solver_settings["driverspline_order"])

    samples = int(np.ceil(total_span / dt_sample)) + 1
    samples = max(samples, order + 1)
    total_time = precision(dt_sample) * max(samples - 1, 1)

    driver_matrix = _driver_sequence(
        samples=samples,
        total_time=total_time,
        n_drivers=system.num_drivers,
        precision=precision,
    )
    driver_names = list(system.indices.driver_names)
    drivers_dict = {
        name: np.array(driver_matrix[:, idx], dtype=precision, copy=True)
        for idx, name in enumerate(driver_names)
    }
    drivers_dict["dt"] = precision(dt_sample)
    drivers_dict["wrap"] = solver_settings["driverspline_wrap"]
    drivers_dict["order"] = order
    drivers_dict["boundary_condition"] = solver_settings[
        "driverspline_boundary_condition"
    ]
    drivers_dict["t0"] = t0

    if driver_settings_override:
        for key, value in driver_settings_override.items():
            drivers_dict[key] = value

    return drivers_dict


@pytest.fixture(scope="session")
def driver_array(
    driver_settings,
    solver_settings,
    precision,
):
    """Instantiate :class:`ArrayInterpolator` for the configured system."""

    if driver_settings is None:
        return None

    return ArrayInterpolator(
        precision=precision,
        input_dict=driver_settings,
    )


@pytest.fixture(scope="session")
def cpu_driver_evaluator(
    driver_settings,
    driver_array,
    solver_settings,
    precision,
    system,
) -> DriverEvaluator:
    """Return a CPU evaluator configured from the driver fixtures."""

    width = system.num_drivers
    order = int(solver_settings["driverspline_order"])
    if driver_settings is None or width == 0 or driver_array is None:
        coeffs = np.zeros((1, width, order + 1), dtype=precision)
        dt_value = precision(solver_settings["save_every"]) / 2.0
        t0_value = 0.0
        wrap_value = bool(solver_settings["driverspline_wrap"])
    else:
        coeffs = np.array(
            driver_array.coefficients,
            dtype=precision,
            copy=True,
        )
        dt_value = precision(driver_array.dt)
        t0_value = precision(driver_array.t0)
        wrap_value = bool(driver_array.wrap)

    return DriverEvaluator(
        coefficients=coeffs,
        dt=dt_value,
        t0=t0_value,
        wrap=wrap_value,
        precision=precision,
        boundary_condition=(
            None if driver_array is None else driver_array.boundary_condition
        ),
    )


@pytest.fixture(scope="session")
def algorithm_settings(solver_settings):
    """Filter algorithm configuration from solver_settings dict.

    Note: Functions (evaluate_f, evaluate_observables,
    get_solver_helper_fn, evaluate_driver_at_t, driver_del_t) are NOT
    included in settings. These are passed directly when building
    step objects, not stored in settings dict.
    """
    settings, _ = merge_kwargs_into_settings(
        kwargs=solver_settings,
        valid_keys=ALL_ALGORITHM_STEP_PARAMETERS,
    )
    # n_drivers comes from solver_settings (added in Task Group 1)
    # Functions are NOT part of algorithm_settings
    return settings


@pytest.fixture(scope="session")
def loop_settings(solver_settings):
    settings, _ = merge_kwargs_into_settings(
        kwargs=solver_settings,
        valid_keys=ALL_LOOP_SETTINGS,
    )
    return settings


@pytest.fixture(scope="session")
def step_controller_settings(solver_settings):
    """Base configuration used to instantiate loop step controllers.

    algorithm_order comes from solver_settings which was enriched with
    this metadata during fixture setup.
    """
    settings, _ = merge_kwargs_into_settings(
        kwargs=solver_settings,
        valid_keys=ALL_STEP_CONTROLLER_PARAMETERS,
    )
    settings.update(algorithm_order=solver_settings["algorithm_order"])
    return settings


# ========================================
# OBJECT FIXTURES
# ========================================


@pytest.fixture(scope="session")
def output_settings(solver_settings):
    settings, _ = merge_kwargs_into_settings(
        kwargs=solver_settings,
        valid_keys=ALL_OUTPUT_FUNCTION_PARAMETERS,
    )
    return settings


@pytest.fixture(scope="session")
def memory_settings(solver_settings):
    settings, _ = merge_kwargs_into_settings(
        kwargs=solver_settings,
        valid_keys=ALL_MEMORY_MANAGER_PARAMETERS,
    )
    return settings


@pytest.fixture(scope="session")
def output_functions(output_settings, system, precision):
    output_settings.pop("precision", None)
    outputfunctions = OutputFunctions(
        system.sizes.states,
        system.sizes.parameters,
        precision=precision,
        **output_settings,
    )
    return outputfunctions


@pytest.fixture(scope="function")
def output_functions_mutable(output_settings, system, precision):
    """Return a fresh ``OutputFunctions`` for mutation-prone tests."""
    settings = output_settings.copy()
    settings.pop("precision", None)
    return OutputFunctions(
        system.sizes.states,
        system.sizes.observables,
        precision=precision,
        **settings,
    )


@pytest.fixture(scope="session")
def solverkernel(
    solver_settings,
    system,
    driver_array,
    step_controller_settings,
    algorithm_settings,
    output_settings,
    memory_settings,
    loop_settings,
):
    """Top-level composite fixture for BatchSolverKernel.

    Exception to single-fixture rule: Requests both system and driver_array
    as these are the two fundamental base CUDAFactory fixtures. All other
    dependencies are settings fixtures.
    """
    evaluate_driver_at_t = _get_evaluate_driver_at_t(driver_array)
    driver_del_t = _get_driver_del_t(driver_array)
    # Add system functions to algorithm_settings for BatchSolverKernel
    enhanced_algorithm_settings = _build_enhanced_algorithm_settings(
        algorithm_settings, system, driver_array
    )
    return BatchSolverKernel(
        system,
        evaluate_driver_at_t=evaluate_driver_at_t,
        driver_del_t=driver_del_t,
        profileCUDA=solver_settings["profileCUDA"],
        step_control_settings=step_controller_settings,
        algorithm_settings=enhanced_algorithm_settings,
        output_settings=output_settings,
        memory_settings=memory_settings,
        loop_settings=loop_settings,
    )


@pytest.fixture(scope="function")
def solverkernel_mutable(
    solver_settings,
    system,
    driver_array,
    step_controller_settings,
    algorithm_settings,
    output_settings,
    memory_settings,
    loop_settings,
):
    """Function-scoped composite fixture for BatchSolverKernel.

    Exception to single-fixture rule: Requests both system and driver_array
    as these are the two fundamental base CUDAFactory fixtures. All other
    dependencies are settings fixtures.
    """
    evaluate_driver_at_t = _get_evaluate_driver_at_t(driver_array)
    driver_del_t = _get_driver_del_t(driver_array)
    # Add system functions to algorithm_settings for BatchSolverKernel
    enhanced_algorithm_settings = _build_enhanced_algorithm_settings(
        algorithm_settings, system, driver_array
    )
    return BatchSolverKernel(
        system,
        evaluate_driver_at_t=evaluate_driver_at_t,
        driver_del_t=driver_del_t,
        profileCUDA=solver_settings["profileCUDA"],
        step_control_settings=step_controller_settings,
        algorithm_settings=enhanced_algorithm_settings,
        output_settings=output_settings,
        memory_settings=memory_settings,
        loop_settings=loop_settings,
    )


@pytest.fixture(scope="session")
def solver(system, solver_settings, driver_array, thread_mem_manager):
    return _build_solver_instance(
        system=system,
        solver_settings=solver_settings,
        driver_array=driver_array,
        memory_manager=thread_mem_manager,
    )


@pytest.fixture(scope="function")
def solver_mutable(
    system,
    solver_settings,
    driver_array,
    thread_mem_manager,
):
    return _build_solver_instance(
        system=system,
        solver_settings=solver_settings,
        driver_array=driver_array,
        memory_manager=thread_mem_manager,
    )


@pytest.fixture(scope="session")
def step_controller(precision, step_controller_settings):
    """Instantiate the requested step controller for loop execution."""
    controller = get_controller(precision, step_controller_settings)
    return controller


@pytest.fixture(scope="function")
def step_controller_mutable(precision, step_controller_settings):
    """Return a fresh step controller for mutation-focused tests."""

    return get_controller(precision, step_controller_settings)


@pytest.fixture(scope="session")
def loop(single_integrator_run):
    """Return the IVPLoop from single_integrator_run.

    SingleIntegratorRun builds all components internally, including the
    loop. Access the cached loop instance rather than rebuilding.
    """
    return single_integrator_run._loop


@pytest.fixture(scope="function")
def loop_mutable(single_integrator_run_mutable):
    """Return the IVPLoop from mutable single_integrator_run.

    Function-scoped variant for mutation-focused tests.
    """
    return single_integrator_run_mutable._loop


@pytest.fixture(scope="session")
def single_integrator_run(
    system,
    solver_settings,
    driver_array,
    step_controller_settings,
    algorithm_settings,
    output_settings,
    loop_settings,
):
    """Top-level composite fixture for SingleIntegratorRun.

    Exception to single-fixture rule: Requests both system and driver_array
    as these are the two fundamental base CUDAFactory fixtures. All other
    dependencies are settings fixtures.
    """
    evaluate_driver_at_t = _get_evaluate_driver_at_t(driver_array)
    driver_del_t = _get_driver_del_t(driver_array)
    # Add system functions to algorithm_settings for SingleIntegratorRun
    enhanced_algorithm_settings = _build_enhanced_algorithm_settings(
        algorithm_settings, system, driver_array
    )
    return SingleIntegratorRun(
        system=system,
        evaluate_driver_at_t=evaluate_driver_at_t,
        driver_del_t=driver_del_t,
        step_control_settings=step_controller_settings,
        algorithm_settings=enhanced_algorithm_settings,
        output_settings=output_settings,
        loop_settings=loop_settings,
    )


@pytest.fixture(scope="function")
def single_integrator_run_mutable(
    system,
    solver_settings,
    driver_array,
    step_controller_settings,
    algorithm_settings,
    output_settings,
    loop_settings,
):
    """Function-scoped composite fixture for SingleIntegratorRun.

    Exception to single-fixture rule: Requests both system and driver_array
    as these are the two fundamental base CUDAFactory fixtures. All other
    dependencies are settings fixtures.
    """
    evaluate_driver_at_t = _get_evaluate_driver_at_t(driver_array)
    driver_del_t = _get_driver_del_t(driver_array)
    # Add system functions to algorithm_settings for SingleIntegratorRun
    enhanced_algorithm_settings = _build_enhanced_algorithm_settings(
        algorithm_settings, system, driver_array
    )
    return SingleIntegratorRun(
        system=system,
        loop_settings=loop_settings,
        evaluate_driver_at_t=evaluate_driver_at_t,
        driver_del_t=driver_del_t,
        step_control_settings=step_controller_settings,
        algorithm_settings=enhanced_algorithm_settings,
        output_settings=output_settings,
    )


@pytest.fixture(scope="session")
def cpu_system(system):
    """Return a CPU-based system."""
    return CPUODESystem(system)


@pytest.fixture(scope="session")
def step_object(single_integrator_run):
    """Return the step object from single_integrator_run.

    This avoids double-building by extracting the already-built step object
    from the composite single_integrator_run fixture.
    """
    return single_integrator_run._algo_step


@pytest.fixture(scope="function")
def step_object_mutable(single_integrator_run_mutable):
    """Return the mutable step object from single_integrator_run_mutable.

    Function-scoped variant for mutation-focused tests.
    """
    return single_integrator_run_mutable._algo_step


@pytest.fixture(scope="session")
def cpu_step_controller(precision, step_controller_settings):
    """Instantiate the requested step controller for loop execution."""

    return _build_cpu_step_controller(
        precision=precision,
        step_controller_settings=step_controller_settings,
    )


# ========================================
# DEVICE UNIT TEST FIXTURES: STEP CONTROLLERS
# ========================================


@pytest.fixture(scope='function')
def step_setup(request, precision, system):
    """Parametrizable inputs for a single controller invocation.

    Use with ``indirect=True`` to vary error, dt, local_mem, etc.
    Defaults: dt0=0.05, error=0.01*ones, state=ones,
    state_prev=ones, local_mem=zeros(2).
    """
    n = system.sizes.states
    setup_dict = {
        'dt0': 0.05,
        'error': np.asarray(
            [0.01] * system.sizes.states, dtype=precision
        ),
        'state': np.ones(n, dtype=precision),
        'state_prev': np.ones(n, dtype=precision),
        'local_mem': np.zeros(2, dtype=precision),
        'niters': 1,
    }
    if hasattr(request, 'param'):
        for key, value in request.param.items():
            if key in setup_dict:
                setup_dict[key] = value
    return setup_dict


@pytest.fixture(scope='function')
def device_step_results(step_controller, precision, step_setup):
    """Run device controller once, return StepResult."""
    return _run_device_step(
        step_controller.device_function,
        precision,
        step_setup['dt0'],
        step_setup['error'],
        state=step_setup['state'],
        state_prev=step_setup['state_prev'],
        local_mem=step_setup['local_mem'],
        niters=step_setup['niters'],
    )


@pytest.fixture(scope='function')
def cpu_step_results(cpu_step_controller, precision, step_setup):
    """Run CPU reference controller once, return StepResult."""
    controller = cpu_step_controller
    kind = controller.kind.lower()
    controller.dt = step_setup['dt0']
    state = np.asarray(step_setup['state'], dtype=precision)
    state_prev = np.asarray(
        step_setup['state_prev'], dtype=precision
    )
    error_vec = np.asarray(step_setup['error'], dtype=precision)
    provided_local = np.asarray(
        step_setup['local_mem'], dtype=precision
    )

    if kind == 'pi':
        controller._prev_nrm2 = float(provided_local[0])
    elif kind == 'pid':
        controller._prev_nrm2 = float(provided_local[0])
        controller._prev_prev_nrm2 = float(provided_local[1])
    elif kind == 'gustafsson':
        controller._prev_dt = float(provided_local[0])
        controller._prev_nrm2 = float(provided_local[1])

    accept = controller.propose_dt(
        prev_state=state_prev,
        new_state=state,
        error_vector=error_vec,
        niters=step_setup['niters'],
    )
    errornorm = controller.error_norm(state_prev, state, error_vec)

    if kind == 'i':
        out_local = np.zeros(0, dtype=precision)
    elif kind == 'pi':
        out_local = np.array([errornorm], dtype=precision)
    elif kind == 'pid':
        out_local = np.array([
            controller._prev_nrm2,
            controller._prev_prev_nrm2,
        ], dtype=precision)
    elif kind == 'gustafsson':
        out_local = np.array([
            controller._prev_dt,
            errornorm,
        ], dtype=precision)
    else:
        out_local = np.zeros(0, dtype=precision)

    return StepResult(controller.dt, int(accept), out_local)


# ========================================
# DEVICE UNIT TEST FIXTURES: ALGORITHM STEPS
# ========================================


@pytest.fixture(scope="session")
def algorithm_step_inputs(
    system,
    precision,
    initial_state,
    solver_settings,
    cpu_driver_evaluator,
) -> dict:
    """State, parameters, and drivers for a single step execution."""
    width = system.num_drivers
    driver_coefficients = np.array(
        cpu_driver_evaluator.coefficients,
        dtype=precision,
        copy=True,
    )
    return {
        "state": initial_state,
        "parameters": system.parameters.values_array.astype(
            precision
        ),
        "drivers": np.zeros(width, dtype=precision),
        "driver_coefficients": driver_coefficients,
    }


@pytest.fixture(scope="session")
def device_algorithm_step_results(
    step_object,
    solver_settings,
    precision,
    algorithm_step_inputs,
    system,
    driver_array,
) -> AlgorithmStepResult:
    """Execute the CUDA step and collect host-side outputs."""
    return _run_device_algorithm_step(
        step_object=step_object,
        solver_settings=solver_settings,
        precision=precision,
        step_inputs=algorithm_step_inputs,
        system=system,
        driver_array=driver_array,
    )


@pytest.fixture(scope="session")
def cpu_algorithm_step_results(
    solver_settings,
    cpu_system,
    algorithm_step_inputs,
    cpu_driver_evaluator,
    step_object,
) -> AlgorithmStepResult:
    """Execute the CPU reference stepper."""

    tableau = getattr(step_object, "tableau", None)
    dt = solver_settings["dt"]
    state = np.asarray(
        algorithm_step_inputs["state"],
        dtype=cpu_system.precision,
    )
    params = np.asarray(
        algorithm_step_inputs["parameters"],
        dtype=cpu_system.precision,
    )
    if cpu_system.system.num_drivers > 0:
        driver_evaluator = (
            cpu_driver_evaluator.with_coefficients(
                algorithm_step_inputs["driver_coefficients"]
            )
        )
    else:
        driver_evaluator = cpu_driver_evaluator

    stepper = get_ref_stepper(
        cpu_system,
        driver_evaluator,
        solver_settings["algorithm"],
        newton_tol=solver_settings["newton_atol"],
        newton_max_iters=solver_settings["newton_max_iters"],
        linear_tol=solver_settings["krylov_atol"],
        linear_max_iters=solver_settings["krylov_max_iters"],
        linear_correction_type=solver_settings[
            "linear_correction_type"
        ],
        preconditioner_order=solver_settings[
            "preconditioner_order"
        ],
        tableau=tableau,
        newton_damping=solver_settings["newton_damping"],
        newton_max_backtracks=solver_settings[
            "newton_max_backtracks"
        ],
    )

    result = stepper.step(
        state=state,
        params=params,
        dt=dt,
        time=0.0,
    )

    return AlgorithmStepResult(
        state=result.state.astype(
            cpu_system.precision, copy=True
        ),
        observables=result.observables.astype(
            cpu_system.precision, copy=True
        ),
        error=result.error.astype(
            cpu_system.precision, copy=True
        ),
        status=result.status & STATUS_MASK,
        n_iters=(result.status >> 16) & STATUS_MASK,
    )


# ========================================
# INPUT FIXTURES
# ========================================


@pytest.fixture(scope="session")
def initial_state(system, precision, request):
    """Return a copy of the system's initial state vector."""
    if hasattr(request, "param"):
        try:
            request_inits = np.asarray(
                request.param,
                dtype=precision,
            )
            if (
                request_inits.ndim != 1
                or request_inits.shape[0] != system.sizes.states
            ):
                raise ValueError(
                    "initial state override has incorrect shape",
                )
        except TypeError as error:
            raise TypeError(
                "initial state override could not be coerced into numpy array",
            ) from error
        return request_inits
    return system.initial_values.values_array.astype(precision, copy=True)


# ========================================
# COMPUTED OUTPUT FIXTURES
# ========================================
@pytest.fixture(scope="session")
def cpu_loop_runner(
    system,
    cpu_system,
    precision,
    solver_settings,
    step_controller_settings,
    output_functions,
    cpu_driver_evaluator,
):
    """Return a callable for generating CPU reference loop outputs."""

    def _run_loop(
        *,
        initial_values=None,
        parameters=None,
        driver_coefficients=None,
    ):
        initial_vec = (
            np.array(initial_values, dtype=precision, copy=True)
            if initial_values is not None
            else np.array(
                system.initial_values.values_array,
                dtype=precision,
                copy=True,
            )
        )
        parameter_vec = (
            np.array(parameters, dtype=precision, copy=True)
            if parameters is not None
            else np.array(
                system.parameters.values_array,
                dtype=precision,
                copy=True,
            )
        )

        inputs = {
            "initial_values": initial_vec,
            "parameters": parameter_vec,
        }
        if driver_coefficients is not None:
            inputs["driver_coefficients"] = np.array(
                driver_coefficients, dtype=precision, copy=True
            )

        controller = _build_cpu_step_controller(
            precision=precision,
            step_controller_settings=step_controller_settings,
        )
        tableau = _get_algorithm_tableau(solver_settings["algorithm"])
        return run_reference_loop(
            evaluator=cpu_system,
            inputs=inputs,
            driver_evaluator=cpu_driver_evaluator,
            solver_settings=solver_settings,
            output_functions=output_functions,
            controller=controller,
            tableau=tableau,
        )

    return _run_loop


@pytest.fixture(scope="session")
def cpu_loop_outputs(
    system,
    cpu_system,
    precision,
    initial_state,
    solver_settings,
    step_controller_settings,
    output_functions,
    cpu_driver_evaluator,
    driver_array,
    single_integrator_run,
) -> dict[str, Array]:
    """Execute the CPU reference loop with the provided configuration."""
    inputs = {
        "initial_values": initial_state.copy(),
        "parameters": system.parameters.values_array.copy(),
    }
    coefficients = (
        driver_array.coefficients if driver_array is not None else None
    )
    inputs["driver_coefficients"] = coefficients

    controller = _build_cpu_step_controller(
        precision=precision,
        step_controller_settings=step_controller_settings,
    )
    # Extract step_object from single_integrator_run
    step_object = single_integrator_run._algo_step
    tableau = getattr(step_object, "tableau", None)
    return run_reference_loop(
        evaluator=cpu_system,
        inputs=inputs,
        driver_evaluator=cpu_driver_evaluator,
        solver_settings=solver_settings,
        output_functions=output_functions,
        controller=controller,
        tableau=tableau,
    )


@pytest.fixture(scope="session")
def device_loop_outputs(
    system,
    single_integrator_run,
    initial_state,
    solver_settings,
    driver_array,
):
    """Execute the device loop with the provided configuration."""
    return run_device_loop(
        singleintegratorrun=single_integrator_run,
        system=system,
        initial_state=initial_state,
        solver_config=solver_settings,
        driver_array=driver_array,
    )


# ========================================
# BATCH INPUT FIXTURES
# ========================================
@pytest.fixture(scope="session")
def system_interface(system) -> SystemInterface:
    """Return a SystemInterface wrapping the configured system."""
    return SystemInterface.from_system(system)


@pytest.fixture(scope="function")
def system_interface_mutable(system) -> SystemInterface:
    """Return a fresh SystemInterface for mutation tests."""
    return SystemInterface.from_system(system)


@pytest.fixture(scope="session")
def input_handler(system) -> BatchInputHandler:
    """Return a batch input handler for the configured system."""
    return BatchInputHandler.from_system(system)


@pytest.fixture(scope="session")
def batch_settings_override(request) -> dict:
    """Override values for batch grid settings when parametrised."""
    return request.param if hasattr(request, "param") else {}


@pytest.fixture(scope="session")
def batch_settings(batch_settings_override) -> dict:
    """Return default batch grid settings merged with overrides."""
    defaults = {
        "num_state_vals_0": 2,
        "num_state_vals_1": 0,
        "num_param_vals_0": 2,
        "num_param_vals_1": 0,
        "kind": "combinatorial",
    }
    defaults.update(
        {k: v for k, v in batch_settings_override.items() if k in defaults}
    )
    return defaults


@pytest.fixture(scope="session")
def batch_request(system, batch_settings, precision) -> dict[str, Array]:
    """Build a request dictionary describing the batch sweep."""
    state_names = list(system.initial_values.names)
    param_names = list(system.parameters.names)
    return {
        state_names[0]: np.concatenate([
            np.linspace(
                0.1, 1.0,
                batch_settings["num_state_vals_0"],
                dtype=precision,
            ),
            [system.initial_values.values_dict[state_names[0]]],
        ]),
        state_names[1]: np.concatenate([
            np.linspace(
                0.1, 1.0,
                batch_settings["num_state_vals_1"],
                dtype=precision,
            ),
            [system.initial_values.values_dict[state_names[1]]],
        ]),
        param_names[0]: np.concatenate([
            np.linspace(
                0.1, 1.0,
                batch_settings["num_param_vals_0"],
                dtype=precision,
            ),
            [system.parameters.values_dict[param_names[0]]],
        ]),
        param_names[1]: np.concatenate([
            np.linspace(
                0.1, 1.0,
                batch_settings["num_param_vals_1"],
                dtype=precision,
            ),
            [system.parameters.values_dict[param_names[1]]],
        ]),
    }


@pytest.fixture(scope="session")
def batch_input_arrays(
    batch_request,
    batch_settings,
    input_handler,
    system,
) -> tuple[Array, Array]:
    """Return the initial state and parameter arrays for the batch run."""
    state_names = set(system.initial_values.names)
    param_names = set(system.parameters.names)

    states_dict = {
        k: v for k, v in batch_request.items() if k in state_names
    }
    params_dict = {
        k: v for k, v in batch_request.items() if k in param_names
    }

    return input_handler(
        states=states_dict,
        params=params_dict,
        kind=batch_settings["kind"],
    )


@attrs.define
class BatchResult:
    """Container for CPU reference outputs for a single batch run."""

    state: Array
    observables: Array
    state_summaries: Array
    observable_summaries: Array
    status: int


@pytest.fixture(scope="session")
def cpu_batch_results(
    batch_input_arrays,
    cpu_loop_runner,
    system,
    solver_settings,
    precision,
    driver_array,
) -> BatchResult:
    """Compute CPU reference outputs for each run in the batch."""
    initial_sets, parameter_sets = batch_input_arrays
    results: list[BatchResult] = []
    coefficients = (
        driver_array.coefficients if driver_array is not None else None
    )
    n_runs = initial_sets.shape[1]
    for idx in range(n_runs):
        loop_result = cpu_loop_runner(
            initial_values=initial_sets[:, idx],
            parameters=parameter_sets[:, idx],
            driver_coefficients=coefficients,
        )
        results.append(
            BatchResult(
                state=loop_result["state"],
                observables=loop_result["observables"],
                state_summaries=loop_result["state_summaries"],
                observable_summaries=loop_result["observable_summaries"],
                status=int(loop_result["status"]),
            )
        )

    return BatchResult(
        state=np.stack([r.state for r in results], axis=2),
        observables=np.stack([r.observables for r in results], axis=2),
        state_summaries=np.stack(
            [r.state_summaries for r in results], axis=2,
        ),
        observable_summaries=np.stack(
            [r.observable_summaries for r in results], axis=2,
        ),
        status=0 if all(r.status == 0 for r in results) else 1,
    )
