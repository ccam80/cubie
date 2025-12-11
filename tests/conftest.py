from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional
import os

import numpy as np
import pytest
from pytest import MonkeyPatch

from tests._utils import _build_enhanced_algorithm_settings
from cubie import SymbolicODE
from cubie.integrators.SingleIntegratorRun import SingleIntegratorRun
from cubie._utils import merge_kwargs_into_settings
from cubie.integrators.step_control import get_controller
from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel
from cubie.batchsolving.solver import Solver
from cubie.integrators.algorithms.base_algorithm_step import \
    ALL_ALGORITHM_STEP_PARAMETERS
from cubie.integrators.loops.ode_loop import (
    ALL_LOOP_SETTINGS,
    LoopBufferSettings,
)

from cubie.integrators.step_control.base_step_controller import (
    ALL_STEP_CONTROLLER_PARAMETERS,
)
from cubie.integrators.array_interpolator import ArrayInterpolator
from cubie.memory import default_memmgr
from cubie.memory.mem_manager import ALL_MEMORY_MANAGER_PARAMETERS
from cubie.outputhandling.output_functions import (
    OutputFunctions,
    ALL_OUTPUT_FUNCTION_PARAMETERS,
)
from tests.integrators.cpu_reference import (
    CPUAdaptiveController,
    CPUODESystem,
    DriverEvaluator,
    run_reference_loop,
)
from tests._utils import _driver_sequence, run_device_loop
from tests.integrators.loops.test_ode_loop import Array
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
    import tempfile
    import shutil
    import os
    from cubie.odesystems.symbolic import odefile

    original_dir = getattr(odefile, "GENERATED_DIR", None)
    redirect_enabled = int(os.environ.get("CUBIE_GENERATED_DIR_REDIRECT",
                                    "1"))

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
        # restore original attribute and remove temporary dir
        mp.undo()
        shutil.rmtree(gen_dir, ignore_errors=True)

# ========================================
# HELPER BUILDERS
# ========================================


def _get_driver_function(
    driver_array: Optional[ArrayInterpolator],
) -> Optional[Callable[..., Any]]:
    """Return the evaluation callable for ``driver_array`` if it exists."""

    if driver_array is None:
        return None
    return driver_array.evaluation_function


def _get_driver_del_t(
    driver_array: Optional[ArrayInterpolator],
) -> Optional[Callable[..., Any]]:
    """Return the time-derivative evaluation callable for ``driver_array``."""

    if driver_array is None:
        return None
    return driver_array.driver_del_t


def _build_solver_instance(
    system: SymbolicODE,
    solver_settings: Dict[str, Any],
    driver_array: Optional[ArrayInterpolator],
) -> Solver:
    """Instantiate :class:`Solver` configured with ``solver_settings``."""

    solver = Solver(system, **solver_settings)
    driver_function = _get_driver_function(driver_array)
    solver.update({"driver_function": driver_function})
    return solver


def build_single_integrator_run(
    system: SymbolicODE,
    solver_settings: Dict[str, Any],
    algorithm_settings: Dict[str, Any],
    step_controller_settings: Dict[str, Any],
    output_settings: Dict[str, Any],
    loop_settings: Dict[str, Any],
    driver_array: Optional[ArrayInterpolator] = None,
) -> SingleIntegratorRun:
    """Build a SingleIntegratorRun for tests.
    
    This helper creates a fully configured SingleIntegratorRun instance,
    injecting system functions into algorithm_settings before construction.
    """
    driver_function = _get_driver_function(driver_array)
    driver_del_t = _get_driver_del_t(driver_array)
    
    # Enhance algorithm_settings with system functions
    enhanced_algorithm_settings = _build_enhanced_algorithm_settings(
        algorithm_settings, system, driver_array
    )
    
    return SingleIntegratorRun(
        system=system,
        driver_function=driver_function,
        driver_del_t=driver_del_t,
        algorithm_settings=enhanced_algorithm_settings,
        step_control_settings=step_controller_settings,
        output_settings=output_settings,
        loop_settings=loop_settings,
    )


def _build_cpu_step_controller(
    precision: np.dtype,
    step_controller_settings: Dict[str, Any],
) -> CPUAdaptiveController:
    """Return a CPU adaptive controller initialised from the settings."""

    kind = step_controller_settings["step_controller"].lower()
    controller = CPUAdaptiveController(
        kind=kind,
        dt=step_controller_settings["dt"],
        dt_min=step_controller_settings["dt_min"],
        dt_max=step_controller_settings["dt_max"],
        atol=step_controller_settings["atol"],
        rtol=step_controller_settings["rtol"],
        order=step_controller_settings["algorithm_order"],
        min_gain=step_controller_settings["min_gain"],
        max_gain=step_controller_settings["max_gain"],
        precision=precision,
        deadband_min=step_controller_settings["deadband_min"],
        deadband_max=step_controller_settings["deadband_max"],
        max_newton_iters=step_controller_settings["max_newton_iters"],
    )
    if kind == "pi":
        controller.kp = step_controller_settings["kp"]
        controller.ki = step_controller_settings["ki"]
    elif kind == "pid":
        controller.kp = step_controller_settings["kp"]
        controller.ki = step_controller_settings["ki"]
        controller.kd = step_controller_settings["kd"]
    return controller


def _get_algorithm_order(algorithm_name_or_tableau):
    """Get algorithm order without building step object.
    
    Parameters
    ----------
    algorithm_name_or_tableau : str or ButcherTableau
        Algorithm identifier or tableau instance.
    
    Returns
    -------
    int
        Algorithm order.
    """
    from cubie.integrators.algorithms import (
        resolve_alias, resolve_supplied_tableau
    )
    from cubie.integrators.algorithms.generic_rosenbrock_w import (
        GenericRosenbrockWStep,
        DEFAULT_ROSENBROCK_TABLEAU,
    )
    
    if isinstance(algorithm_name_or_tableau, str):
        algorithm_type, tableau = resolve_alias(algorithm_name_or_tableau)
    else:
        algorithm_type, tableau = resolve_supplied_tableau(
            algorithm_name_or_tableau
        )
    
    # For rosenbrock without explicit tableau, use default
    if (algorithm_type is GenericRosenbrockWStep and tableau is None):
        tableau = DEFAULT_ROSENBROCK_TABLEAU
    
    # Extract order from tableau if available
    if tableau is not None and hasattr(tableau, 'order'):
        return tableau.order
    
    # Default orders for algorithms without tableaus
    defaults = {
        'euler': 1,
        'backwards_euler': 1,
        'backwards_euler_pc': 1,
        'crank_nicolson': 2,
    }
    
    if isinstance(algorithm_name_or_tableau, str):
        algorithm_name = algorithm_name_or_tableau.lower()
        return defaults.get(algorithm_name, 1)
    
    return 1


def _get_algorithm_is_adaptive(algorithm_name_or_tableau):
    """Determine if algorithm is adaptive without building step object.
    
    Parameters
    ----------
    algorithm_name_or_tableau : str or ButcherTableau
        Algorithm identifier or tableau instance.
    
    Returns
    -------
    bool
        True if algorithm has embedded error estimate (adaptive).
    """
    from cubie.integrators.algorithms import (
        resolve_alias, resolve_supplied_tableau
    )
    from cubie.integrators.algorithms.generic_rosenbrock_w import (
        GenericRosenbrockWStep,
        DEFAULT_ROSENBROCK_TABLEAU,
    )
    
    if isinstance(algorithm_name_or_tableau, str):
        algorithm_type, tableau = resolve_alias(algorithm_name_or_tableau)
    else:
        algorithm_type, tableau = resolve_supplied_tableau(
            algorithm_name_or_tableau
        )
    
    # For rosenbrock without explicit tableau, use default
    if (algorithm_type is GenericRosenbrockWStep and tableau is None):
        tableau = DEFAULT_ROSENBROCK_TABLEAU
    
    # Check if tableau has error estimate
    if tableau is not None and hasattr(tableau, 'has_error_estimate'):
        return tableau.has_error_estimate
    
    # Non-adaptive algorithms by default
    return False


def _get_algorithm_tableau(algorithm_name_or_tableau):
    """Get tableau for an algorithm without building step object.
    
    Parameters
    ----------
    algorithm_name_or_tableau : str or ButcherTableau
        Algorithm identifier or tableau instance.
    
    Returns
    -------
    tableau or None
        The tableau if available, None otherwise.
    """
    from cubie.integrators.algorithms import (
        resolve_alias, resolve_supplied_tableau
    )
    from cubie.integrators.algorithms.generic_rosenbrock_w import (
        GenericRosenbrockWStep,
        DEFAULT_ROSENBROCK_TABLEAU,
    )
    
    if isinstance(algorithm_name_or_tableau, str):
        algorithm_type, tableau = resolve_alias(algorithm_name_or_tableau)
    else:
        algorithm_type, tableau = resolve_supplied_tableau(
            algorithm_name_or_tableau
        )
    
    # For rosenbrock without explicit tableau, use default
    if (algorithm_type is GenericRosenbrockWStep and tableau is None):
        tableau = DEFAULT_ROSENBROCK_TABLEAU
    
    return tableau


# ========================================
# SETTINGS DICTS (override -> fixture -> override -> fixture)
# ========================================

@pytest.fixture(scope="session")
def precision(solver_settings_override, solver_settings_override2):
    """Return precision from overrides, defaulting to float32.

    Usage:
    @pytest.mark.parametrize("solver_settings_override",
        [{"precision": np.float64}], indirect=True)
    def test_something(precision):
        # precision will be np.float64 here
    """
    for override in [solver_settings_override2, solver_settings_override]:
        if override and 'precision' in override:
            return override['precision']
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
def system(request, solver_settings_override, solver_settings_override2,
           precision):
    """Return the appropriate symbolic system, defaulting to nonlinear.

    Usage:
    @pytest.mark.parametrize("solver_settings_override",
        [{"system_type": "three_chamber"}], indirect=True)
    def test_something(system):
        # system will be the cardiovascular symbolic model here
    """
    model_type = 'nonlinear'
    for override in [solver_settings_override2, solver_settings_override]:
        if override and 'system_type' in override:
            model_type = override['system_type']
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
def solver_settings_override(request):
    """Override for solver settings, if provided."""
    return request.param if hasattr(request, "param") else {}

@pytest.fixture(scope="session")
def solver_settings_override2(request):
    """Override for solver settings, if provided. A second one, so that we
    can do a class-level and function-level override without conflicts."""
    return request.param if hasattr(request, "param") else {}

@pytest.fixture(scope="session")
def solver_settings(solver_settings_override, solver_settings_override2,
    system, precision):
    """Create LoopStepConfig with default solver configuration."""
    defaults = {
        "algorithm": "euler",
        "system_type": "nonlinear",
        "duration": np.float64(0.2),
        "warmup": np.float64(0.0),
        "t0": np.float64(0.0),
        "dt": precision(0.01),
        "dt_min": precision(1e-7),
        "dt_max": precision(1.0),
        "dt_save": precision(0.1),
        "dt_summarise": precision(0.2),
        "atol": precision(1e-6),
        "rtol": precision(1e-6),
        "saved_state_indices": [0, 1],
        "saved_observable_indices": [0, 1],
        "summarised_state_indices": [0, 1],
        "summarised_observable_indices": [0, 1],
        "output_types": ["state"],
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
        "krylov_tolerance": precision(1e-6),
        "correction_type": "minimal_residual",
        "newton_tolerance": precision(1e-6),
        "preconditioner_order": 2,
        "max_linear_iters": 20,
        "max_newton_iters": 20,
        "newton_damping": precision(0.85),
        "newton_max_backtracks": 25,
        "min_gain": precision(0.1),
        "max_gain": precision(5.0),
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
        "dt_save",
        "dt_summarise",
        "atol",
        "rtol",
        "krylov_tolerance",
        "newton_tolerance",
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
                    defaults[key] = precision(value)
                else:
                    defaults[key] = value

    # Add derived metadata
    defaults['algorithm_order'] = _get_algorithm_order(defaults['algorithm'])
    defaults['n_states'] = system.sizes.states
    defaults['n_parameters'] = system.sizes.parameters
    defaults['n_drivers'] = system.sizes.drivers
    defaults['n_observables'] = system.sizes.observables

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

    dt_sample = precision(solver_settings["dt_save"]) / 2.0
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
        dt_value = precision(solver_settings["dt_save"]) / 2.0
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
    
    Note: Functions (dxdt_function, observables_function, 
    get_solver_helper_fn, driver_function, driver_del_t) are NOT 
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
    settings.update(algorithm_order=solver_settings['algorithm_order'])
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
def output_functions(output_settings, system):
    outputfunctions = OutputFunctions(
        system.sizes.states,
        system.sizes.parameters,
        **output_settings,
    )
    return outputfunctions


@pytest.fixture(scope="function")
def output_functions_mutable(output_settings, system):
    """Return a fresh ``OutputFunctions`` for mutation-prone tests."""

    return OutputFunctions(
        system.sizes.states,
        system.sizes.parameters,
        **output_settings,
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
    driver_function = _get_driver_function(driver_array)
    driver_del_t = _get_driver_del_t(driver_array)
    # Add system functions to algorithm_settings for BatchSolverKernel
    enhanced_algorithm_settings = _build_enhanced_algorithm_settings(
        algorithm_settings, system, driver_array
    )
    return BatchSolverKernel(
        system,
        driver_function=driver_function,
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
    driver_function = _get_driver_function(driver_array)
    driver_del_t = _get_driver_del_t(driver_array)
    # Add system functions to algorithm_settings for BatchSolverKernel
    enhanced_algorithm_settings = _build_enhanced_algorithm_settings(
        algorithm_settings, system, driver_array
    )
    return BatchSolverKernel(
        system,
        driver_function=driver_function,
        driver_del_t=driver_del_t,
        profileCUDA=solver_settings["profileCUDA"],
        step_control_settings=step_controller_settings,
        algorithm_settings=enhanced_algorithm_settings,
        output_settings=output_settings,
        memory_settings=memory_settings,
        loop_settings=loop_settings
    )


@pytest.fixture(scope="session")
def solver(
    system,
    solver_settings,
    driver_array,
):
    return _build_solver_instance(
        system=system,
        solver_settings=solver_settings,
        driver_array=driver_array,
    )


@pytest.fixture(scope="function")
def solver_mutable(
    system,
    solver_settings,
    driver_array,
):
    return _build_solver_instance(
        system=system,
        solver_settings=solver_settings,
        driver_array=driver_array,
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
    loop_settings
):
    """Top-level composite fixture for SingleIntegratorRun.
    
    Exception to single-fixture rule: Requests both system and driver_array
    as these are the two fundamental base CUDAFactory fixtures. All other
    dependencies are settings fixtures.
    """
    driver_function = _get_driver_function(driver_array)
    driver_del_t = _get_driver_del_t(driver_array)
    # Add system functions to algorithm_settings for SingleIntegratorRun
    enhanced_algorithm_settings = _build_enhanced_algorithm_settings(
        algorithm_settings, system, driver_array
    )
    return SingleIntegratorRun(
        system=system,
        driver_function=driver_function,
        driver_del_t=driver_del_t,
        step_control_settings=step_controller_settings,
        algorithm_settings=enhanced_algorithm_settings,
        output_settings=output_settings,
        loop_settings=loop_settings
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
    driver_function = _get_driver_function(driver_array)
    driver_del_t = _get_driver_del_t(driver_array)
    # Add system functions to algorithm_settings for SingleIntegratorRun
    enhanced_algorithm_settings = _build_enhanced_algorithm_settings(
        algorithm_settings, system, driver_array
    )
    return SingleIntegratorRun(
        system=system,
        loop_settings=loop_settings,
        driver_function=driver_function,
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


@pytest.fixture(scope="function")
def cpu_step_controller(precision, step_controller_settings):
    """Instantiate the requested step controller for loop execution."""

    return _build_cpu_step_controller(
        precision=precision,
        step_controller_settings=step_controller_settings,
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


@pytest.fixture(scope="session")
def buffer_settings(solver_settings, output_functions):
    """Buffer settings derived from system sizes and output configuration.
    
    Uses solver_settings metadata and algorithm tableau to determine
    buffer requirements without building step objects.
    """
    is_adaptive = _get_algorithm_is_adaptive(solver_settings['algorithm'])
    n_error = solver_settings['n_states'] if is_adaptive else 0
    return LoopBufferSettings(
        n_states=solver_settings['n_states'],
        n_parameters=solver_settings['n_parameters'],
        n_drivers=solver_settings['n_drivers'],
        n_observables=solver_settings['n_observables'],
        state_summary_buffer_height=output_functions.state_summaries_buffer_height,
        observable_summary_buffer_height=output_functions.observable_summaries_buffer_height,
        n_error=n_error,
        n_counters=0,
    )


@pytest.fixture(scope="function")
def buffer_settings_mutable(solver_settings, output_functions_mutable):
    """Function-scoped buffer settings from mutable output functions.
    
    Uses solver_settings metadata and algorithm tableau to determine
    buffer requirements without building step objects.
    """
    is_adaptive = _get_algorithm_is_adaptive(solver_settings['algorithm'])
    n_error = solver_settings['n_states'] if is_adaptive else 0
    return LoopBufferSettings(
        n_states=solver_settings['n_states'],
        n_parameters=solver_settings['n_parameters'],
        n_drivers=solver_settings['n_drivers'],
        n_observables=solver_settings['n_observables'],
        state_summary_buffer_height=output_functions_mutable.state_summaries_buffer_height,
        observable_summary_buffer_height=output_functions_mutable.observable_summaries_buffer_height,
        n_error=n_error,
        n_counters=0,
    )


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
        tableau = _get_algorithm_tableau(solver_settings['algorithm'])
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
        'initial_values': initial_state.copy(),
        'parameters': system.parameters.values_array.copy(),
    }
    coefficients = (
        driver_array.coefficients if driver_array is not None else None
    )
    inputs['driver_coefficients'] = coefficients

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
    return  run_device_loop(
        singleintegratorrun=single_integrator_run,
        system=system,
        initial_state=initial_state,
        solver_config=solver_settings,
        driver_array=driver_array,
    )
