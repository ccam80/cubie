from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pytest import MonkeyPatch

from cubie import SingleIntegratorRun
from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel
from cubie.batchsolving.solver import Solver
from cubie.integrators.algorithms import get_algorithm_step
from cubie.integrators.loops.ode_loop import IVPLoop
from cubie.integrators.loops.ode_loop_config import LoopSharedIndices, \
    LoopLocalIndices
from cubie.integrators.step_control.adaptive_I_controller import AdaptiveIController
from cubie.integrators.step_control.adaptive_PID_controller import (
    AdaptivePIDController,
)
from cubie.integrators.step_control.adaptive_PI_controller import AdaptivePIController
from cubie.integrators.step_control.fixed_step_controller import FixedStepController
from cubie.integrators.step_control.gustafsson_controller import GustafssonController
from cubie.integrators.driver_array import DriverArray
from cubie.memory import default_memmgr
from cubie.outputhandling.output_functions import OutputFunctions
from cubie.outputhandling.output_sizes import LoopBufferSizes
from tests.integrators.cpu_reference import (CPUODESystem,
                                              run_reference_loop, \
    CPUAdaptiveController, DriverEvaluator)
from tests._utils import run_device_loop, _driver_sequence
from tests.integrators.loops.test_ode_loop import Array
from tests.system_fixtures import (
    build_large_nonlinear_system,
    build_three_chamber_system,
    build_three_state_linear_system,
    build_three_state_nonlinear_system,
    build_three_state_very_stiff_system,
)


# ========================================
# SETTINGS DICTS (override -> fixture -> override -> fixture)
# ========================================

@pytest.fixture(scope="function")
def precision_override(request):
    if hasattr(request, "param"):
        if request.param is np.float64:
            return np.float64


@pytest.fixture(scope="function")
def precision(precision_override, system_override):
    """
    Run tests with float32 by default, upgrade to float64 for stiff problems.

    Usage:
    @pytest.mark.parametrize("precision_override", [np.float64], indirect=True)
    def test_something(precision):
        # precision will be np.float64 here
    """
    if precision_override is not None:
        return precision_override
    return np.float32


@pytest.fixture(scope="function")
def system_override(request):
    """Override for system model type, if provided."""
    if hasattr(request, "param"):
        if request.param:
            return request.param
    return "nonlinear"


@pytest.fixture(scope="function")
def system(request, system_override, precision):
    """
    Return the appropriate symbolic system, defaulting to ``linear``.

    Usage:
    @pytest.mark.parametrize("system_override", ["three_chamber"], indirect=True)
    def test_something(system):
        # system will be the cardiovascular symbolic model here
    """
    model_type = system_override

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
    if isinstance(model_type, object):
        return model_type

    raise ValueError(f"Unknown model type: {model_type}")


@pytest.fixture(scope="function")
def solver_settings_override(request):
    """Override for solver settings, if provided."""
    return request.param if hasattr(request, "param") else {}


@pytest.fixture(scope="function")
def solver_settings(solver_settings_override, precision):
    """Create LoopStepConfig with default solver configuration."""
    defaults = {
        "algorithm": "euler",
        "duration": precision(1.0),
        "warmup": precision(0.0),
        "dt_min": precision(0.01),
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
    }

    if solver_settings_override:
        # Update defaults with any overrides provided
        float_keys = {
            "duration",
            "warmup",
            "dt_min",
            "dt_max",
            "dt_save",
            "dt_summarise",
            "atol",
            "rtol",
        }
        for key, value in solver_settings_override.items():
            if key in float_keys:
                defaults[key] = precision(value)
            else:
                defaults[key] = value

    return defaults


@pytest.fixture(scope="function")
def driver_settings_override(request):
    """Optional override for driver array configuration."""

    return request.param if hasattr(request, "param") else None


@pytest.fixture(scope="function")
def driver_settings(
    driver_settings_override,
    solver_settings,
    system,
    precision,
):
    """Return default driver samples mapped to system driver symbols."""

    if system.num_drivers == 0:
        return None

    dt_sample = float(solver_settings["dt_save"]) / 2.0
    total_span = float(solver_settings["duration"] + solver_settings["warmup"])
    order = int(solver_settings["driverspline_order"])

    samples = int(np.ceil(total_span / dt_sample)) + 1
    samples = max(samples, order + 1)
    total_time = float(dt_sample) * max(samples - 1, 1)

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
    drivers_dict["dt"] = float(dt_sample)
    drivers_dict["wrap"] = bool(solver_settings["driverspline_wrap"])

    if driver_settings_override:
        for key, value in driver_settings_override.items():
            if key == "time":
                drivers_dict.pop("dt", None)
                drivers_dict[key] = np.array(value, dtype=precision, copy=True)
            elif key == "dt":
                drivers_dict.pop("time", None)
                drivers_dict[key] = float(value)
            elif key == "wrap":
                drivers_dict[key] = bool(value)
            else:
                drivers_dict[key] = np.array(value, dtype=precision, copy=True)

    return drivers_dict


@pytest.fixture(scope="function")
def driver_array(
    driver_settings,
    solver_settings,
    precision,
):
    """Instantiate :class:`DriverArray` for the configured system."""

    if driver_settings is None:
        return None

    return DriverArray(
        precision=precision,
        drivers_dict=driver_settings,
        order=int(solver_settings["driverspline_order"]),
        wrap=bool(solver_settings["driverspline_wrap"]),
    )


@pytest.fixture(scope="function")
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
        dt_value = float(solver_settings["dt_save"]) / 2.0
        t0_value = 0.0
        wrap_value = bool(solver_settings["driverspline_wrap"])
    else:
        coeffs = np.array(driver_array.coefficients, dtype=precision, copy=True)
        dt_value = float(driver_array.dt)
        t0_value = float(driver_array.t0)
        wrap_value = bool(driver_array.wrap)

    return DriverEvaluator(
        coefficients=coeffs,
        dt=dt_value,
        t0=t0_value,
        wrap=wrap_value,
        precision=precision,
    )


@pytest.fixture(scope="function")
def implicit_step_settings_override(request):
    """Override values for implicit solver helper settings."""

    return request.param if hasattr(request, "param") else {}


@pytest.fixture(scope="function")
def implicit_step_settings(solver_settings, implicit_step_settings_override):
    """Default tolerances and iteration limits for implicit solves."""

    defaults = {
        "atol": solver_settings['atol'],
        "rtol": solver_settings['rtol'],
        "linear_tolerance": 1e-6,
        "correction_type": 'minimal_residual',
        "nonlinear_tolerance": 1e-4,
        'preconditioner_order': 2,
        "max_linear_iters": 500,
        "max_newton_iters": 500,
        "newton_damping": 0.85,
        "newton_max_backtracks": 25
    }
    defaults.update(implicit_step_settings_override)
    return defaults


@pytest.fixture(scope="function")
def step_controller_settings_override(request):
    """Override dictionary for the step controller configuration."""

    return request.param if hasattr(request, "param") else {}


@pytest.fixture(scope="function")
def step_controller_settings(
    solver_settings, system, step_controller_settings_override
):
    """Base configuration used to instantiate loop step controllers."""
    precision = solver_settings["precision"]

    defaults = {
        "kind": solver_settings["step_controller"].lower(),
        "dt": precision(solver_settings["dt_min"]),
        "dt_min": precision(solver_settings["dt_min"]),
        "dt_max": precision(solver_settings["dt_max"]),
        "atol": precision(solver_settings["atol"]),
        "rtol": precision(solver_settings["rtol"]),
        "order": 1,
        "min_gain": precision(0.2),
        "max_gain": precision(5.0),
        "n": system.sizes.states,
        "kp": precision(1/18),
        "ki": precision(1/9),
        "kd": precision(1/18),
        "deadband_min": precision(1.0),
        "deadband_max": precision(1.2),
    }
    overrides = {**step_controller_settings_override}
    float_keys = {
        "dt",
        "dt_min",
        "dt_max",
        "atol",
        "rtol",
        "kp",
        "ki",
        "kd",
        "deadband_min",
        "deadband_max",
    }
    for key, value in overrides.items():
        if key in float_keys:
            defaults[key] = precision(value)
        else:
            defaults[key] = value
    return defaults


# ========================================
# OBJECT FIXTURES
# ========================================

@pytest.fixture(scope="module", autouse=True)
def codegen_dir():
    """Redirect code generation to a temporary directory for the whole session.

    Use tempfile.mkdtemp instead of pytest's tmp path so the directory isn't
    removed automatically between parameterized test cases. Remove the
    directory at session teardown.
    """
    import tempfile
    import shutil
    from cubie.odesystems.symbolic import odefile

    gen_dir = Path(tempfile.mkdtemp(prefix="cubie_generated_"))
    mp = MonkeyPatch()
    mp.setattr(odefile, "GENERATED_DIR", gen_dir, raising=True)
    try:
        yield gen_dir
    finally:
        # restore original attribute and remove temporary dir
        mp.undo()
        shutil.rmtree(gen_dir, ignore_errors=True)


@pytest.fixture(scope="function")
def output_functions(solver_settings, system):
    # Merge the default config with any overrides

    outputfunctions = OutputFunctions(
        system.sizes.states,
        system.sizes.parameters,
        solver_settings["output_types"],
        solver_settings["saved_state_indices"],
        solver_settings["saved_observable_indices"],
        solver_settings["summarised_state_indices"],
        solver_settings["summarised_observable_indices"],
    )
    return outputfunctions


@pytest.fixture(scope="function")
def solverkernel(solver_settings, system):
    return BatchSolverKernel(
        system,
        algorithm=solver_settings["algorithm"],
        duration=solver_settings["duration"],
        warmup=solver_settings["warmup"],
        dt_min=solver_settings["dt_min"],
        dt_max=solver_settings["dt_max"],
        dt_save=solver_settings["dt_save"],
        dt_summarise=solver_settings["dt_summarise"],
        atol=solver_settings["atol"],
        rtol=solver_settings["rtol"],
        saved_state_indices=solver_settings["saved_state_indices"],
        saved_observable_indices=solver_settings["saved_observable_indices"],
        output_types=solver_settings["output_types"],
        profileCUDA=solver_settings["profileCUDA"],
        memory_manager=solver_settings["memory_manager"],
        stream_group=solver_settings["stream_group"],
        mem_proportion=solver_settings["mem_proportion"],
    )


@pytest.fixture(scope="function")
def solver(system, solver_settings):
    return Solver(
        system,
        algorithm=solver_settings["algorithm"],
        duration=solver_settings["duration"],
        warmup=solver_settings["warmup"],
        dt_min=solver_settings["dt_min"],
        dt_max=solver_settings["dt_max"],
        dt_save=solver_settings["dt_save"],
        dt_summarise=solver_settings["dt_summarise"],
        atol=solver_settings["atol"],
        rtol=solver_settings["rtol"],
        saved_states=solver_settings["saved_state_indices"],
        saved_observables=solver_settings["saved_observable_indices"],
        output_types=solver_settings["output_types"],
        profileCUDA=solver_settings["profileCUDA"],
        memory_manager=solver_settings["memory_manager"],
        stream_group=solver_settings["stream_group"],
        mem_proportion=solver_settings["mem_proportion"],
    )


@pytest.fixture(scope="function")
def step_controller(precision, step_controller_settings):
    """Instantiate the requested step controller for loop execution."""

    kind = step_controller_settings["kind"].lower()
    settings = step_controller_settings
    if kind == "fixed":
        return FixedStepController(precision, step_controller_settings["dt"])
    elif kind == "i":
        controller = AdaptiveIController(
                precision=precision,
                dt_min=float(settings["dt_min"]),
                dt_max=float(settings["dt_max"]),
                atol=settings["atol"],
                rtol=settings["rtol"],
                algorithm_order=int(settings.get("order", 1)),
                n=int(settings["n"]),
                deadband_min=float(settings["deadband_min"]),
                deadband_max=float(settings["deadband_max"]),
        )
    elif kind == "pi":
        controller = AdaptivePIController(
                precision=precision,
                dt_min=float(settings["dt_min"]),
                dt_max=float(settings["dt_max"]),
                atol=settings["atol"],
                rtol=settings["rtol"],
                algorithm_order=int(settings.get("order", 1)),
                n=int(settings["n"]),
                kp=float(settings["kp"]),
                ki=float(settings["ki"]),
                deadband_min=float(settings["deadband_min"]),
                deadband_max=float(settings["deadband_max"]),
        )
    elif kind == "pid":
        controller = AdaptivePIDController(
                precision=precision,
                dt_min=float(settings["dt_min"]),
                dt_max=float(settings["dt_max"]),
                atol=settings["atol"],
                rtol=settings["rtol"],
                algorithm_order=int(settings.get("order", 1)),
                n=int(settings["n"]),
                kp=float(settings["kp"]),
                ki=float(settings["ki"]),
                kd=float(settings["kd"]),
                deadband_min=float(settings["deadband_min"]),
                deadband_max=float(settings["deadband_max"]),
        )
    elif kind == "gustafsson":
        controller = GustafssonController(
                precision=precision,
                dt_min=float(settings["dt_min"]),
                dt_max=float(settings["dt_max"]),
                atol=settings["atol"],
                rtol=settings["rtol"],
                algorithm_order=int(settings.get("order", 1)),
                n=int(settings["n"]),
                deadband_min=float(settings["deadband_min"]),
                deadband_max=float(settings["deadband_max"]),
        )
    else:
        raise ValueError(f"Unknown adaptive controller kind '{kind}'.")

    controller.update(updates_dict=settings, silent=True)
    return controller


@pytest.fixture(scope="function")
def loop(
    precision,
    system,
    step_object,
    loop_buffer_sizes,
    output_functions,
    step_controller,
    solver_settings,
    driver_array,
):
    """Construct the :class:`IVPLoop` instance used in loop tests."""
    shared_indices = LoopSharedIndices.from_sizes(
            n_states=loop_buffer_sizes.state,
            n_observables=loop_buffer_sizes.observables,
            n_parameters=loop_buffer_sizes.parameters,
            n_drivers=loop_buffer_sizes.drivers,
            state_summaries_buffer_height=loop_buffer_sizes.state_summaries,
            observable_summaries_buffer_height=loop_buffer_sizes.observable_summaries
    )
    local_indices = LoopLocalIndices.from_sizes(
            loop_buffer_sizes.state,
            step_controller.local_memory_elements,
            step_object.persistent_local_required,
    )

    driver_fn = driver_array.driver_function if driver_array is not None else None

    return IVPLoop(precision=precision, shared_indices=shared_indices,
                   local_indices=local_indices,
                   compile_flags=output_functions.compile_flags,
                   save_state_func=output_functions.save_state_func,
                   update_summaries_func=output_functions.update_summaries_func,
                   save_summaries_func=output_functions.save_summary_metrics_func,
                   step_controller_fn=step_controller.device_function,
                   step_fn=step_object.step_function,
                   driver_fn=driver_fn,
                   observables_fn=system.observables_function,
                   dt_save=solver_settings["dt_save"],
                   dt_summarise=solver_settings["dt_summarise"],
                   dt0=step_controller.dt0, dt_min=step_controller.dt_min,
                   dt_max=step_controller.dt_max,
                   is_adaptive=step_controller.is_adaptive)

@pytest.fixture(scope="function")
def single_integrator_run(system, solver_settings):
    """Instantiate :class:`SingleIntegratorRun` with test fixtures."""

    return SingleIntegratorRun(
        system=system,
        algorithm=solver_settings["algorithm"],
        dt_min=solver_settings["dt_min"],
        dt_max=solver_settings["dt_max"],
        fixed_step_size=solver_settings["dt_min"],
        dt_save=solver_settings["dt_save"],
        dt_summarise=solver_settings["dt_summarise"],
        atol=solver_settings["atol"],
        rtol=solver_settings["rtol"],
        saved_state_indices=solver_settings["saved_state_indices"],
        saved_observable_indices=solver_settings["saved_observable_indices"],
        summarised_state_indices=solver_settings["summarised_state_indices"],
        summarised_observable_indices=solver_settings[
            "summarised_observable_indices"
        ],
        output_types=solver_settings["output_types"],
        step_controller_kind=solver_settings["step_controller"],
    )

@pytest.fixture(scope='function')
def cpu_system(system):
    """Return a CPU-based system."""
    return CPUODESystem(system)


@pytest.fixture
def step_object(
    solver_settings,
    implicit_step_settings,
    precision,
    system,
    driver_array,
):
    """Return a step object for the given solver settings."""

    driver_fn = driver_array.driver_function if driver_array is not None else None
    if solver_settings["algorithm"].lower() == 'euler':
        solver_kwargs = {
            'dt': solver_settings["dt_min"],
            'precision': precision,
            'n': system.sizes.states,
            'dxdt_function': system.dxdt_function,
            'observables_function': system.observables_function,
            'driver_function': driver_fn,
        }
    else:
        solver_kwargs = {
            "precision": precision,
            "n": system.sizes.states,
            'dxdt_function': system.dxdt_function,
            'observables_function': system.observables_function,
            'get_solver_helper_fn': system.get_solver_helper,
            'preconditioner_order': implicit_step_settings[
                "preconditioner_order"
            ],
            'linsolve_tolerance': implicit_step_settings["linear_tolerance"],
            'max_linear_iters': implicit_step_settings["max_linear_iters"],
            'linear_correction_type': implicit_step_settings[
                "correction_type"
            ],
            'nonlinear_tolerance': implicit_step_settings["nonlinear_tolerance"],
            'max_newton_iters': implicit_step_settings["max_newton_iters"],
            'newton_damping': implicit_step_settings["newton_damping"],
            'newton_max_backtracks': implicit_step_settings[
                "newton_max_backtracks"
            ],
            'driver_function': driver_fn,
        }
    return get_algorithm_step(solver_settings["algorithm"].lower(), **solver_kwargs)


@pytest.fixture(scope="function")
def cpu_step_controller(precision, step_controller_settings):
    """Instantiate the requested step controller for loop execution."""
    kind = step_controller_settings["kind"].lower()

    controller = CPUAdaptiveController(
        kind=step_controller_settings["kind"].lower(),
        dt_min=step_controller_settings["dt_min"],
        dt_max=step_controller_settings["dt_max"],
        atol=step_controller_settings["atol"],
        rtol=step_controller_settings["rtol"],
        order=step_controller_settings["order"],
        precision=precision,
        deadband_min=step_controller_settings["deadband_min"],
        deadband_max=step_controller_settings["deadband_max"],
    )
    if kind == 'pi':
        controller.kp = step_controller_settings["kp"]
        controller.ki = step_controller_settings["ki"]
    elif kind == 'pid':
        controller.kp = step_controller_settings["kp"]
        controller.ki = step_controller_settings["ki"]
        controller.kd = step_controller_settings["kd"]

    return controller


# ========================================
# INPUT FIXTURES
# ========================================

@pytest.fixture(scope="function")
def initial_state(system, precision, request):
    """Return a copy of the system's initial state vector."""
    if hasattr(request, "param"):
        try:
            request_inits = np.asarray(request.param, dtype=precision)
            if request_inits.ndim != 1 or request_inits.shape[0] != system.sizes.states:
                raise ValueError("initial state override has incorrect shape")
        except:
            raise TypeError("initial state override could not be coerced into numpy array")
        return request_inits
    return system.initial_values.values_array.astype(precision, copy=True)


@pytest.fixture(scope="function")
def loop_buffer_sizes(system, output_functions):
    """Loop buffer sizes derived from the system and output configuration."""

    return LoopBufferSizes.from_system_and_output_fns(system, output_functions)


# ========================================
# COMPUTED OUTPUT FIXTURES
# ========================================
@pytest.fixture(scope="function")
def cpu_loop_runner(
    system,
    cpu_system,
    precision,
    cpu_step_controller,
    solver_settings,
    step_controller_settings,
    implicit_step_settings,
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
            else system.initial_values.values_array.astype(precision, copy=True)
        )
        parameter_vec = (
            np.array(parameters, dtype=precision, copy=True)
            if parameters is not None
            else system.parameters.values_array.astype(precision, copy=True)
        )

        inputs = {
            "initial_values": initial_vec,
            "parameters": parameter_vec,
        }
        if driver_coefficients is not None:
            inputs["driver_coefficients"] = np.array(
                driver_coefficients, dtype=precision, copy=True
            )

        return run_reference_loop(
            evaluator=cpu_system,
            inputs=inputs,
            driver_evaluator=cpu_driver_evaluator,
            solver_settings=solver_settings,
            implicit_step_settings=implicit_step_settings,
            controller=cpu_step_controller,
            output_functions=output_functions,
            step_controller_settings=step_controller_settings,
        )

    return _run_loop

@pytest.fixture(scope="function")
def cpu_loop_outputs(
    system,
    cpu_system,
    precision,
    cpu_step_controller,
    implicit_step_settings,
    initial_state,
    solver_settings,
    step_controller_settings,
    output_functions,
    cpu_driver_evaluator,
) -> dict[str, Array]:
    """Execute the CPU reference loop with the provided configuration."""
    inputs = {
        'initial_values': initial_state.copy(),
        'parameters': system.parameters.values_array.copy(),
    }

    return run_reference_loop(
        evaluator=cpu_system,
        inputs=inputs,
        driver_evaluator=cpu_driver_evaluator,
        solver_settings=solver_settings,
        implicit_step_settings=implicit_step_settings,
        controller=cpu_step_controller,
        output_functions=output_functions,
        step_controller_settings=step_controller_settings,
    )


@pytest.fixture(scope="function")
def device_loop_outputs(
    loop,
    system,
    single_integrator_run,
    initial_state,
    solver_settings,
    step_controller_settings,
    output_functions,
    cpu_system,
    driver_array,
):
    """Execute the device loop with the provided configuration."""
    return  run_device_loop(
        loop=loop,
        system=system,
        initial_state=initial_state,
        output_functions=output_functions,
        solver_config=solver_settings,
        localmem_required=single_integrator_run.local_memory_elements,
        driver_array=driver_array,
    )
