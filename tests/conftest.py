from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pytest import MonkeyPatch

from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel
from cubie.batchsolving.solver import Solver
from cubie.integrators.algorithms_ import get_algorithm_step
from cubie.integrators.loops.ode_loop import IVPLoop
from cubie.integrators.step_control.adaptive_I_controller import AdaptiveIController
from cubie.integrators.step_control.adaptive_PID_controller import (
    AdaptivePIDController,
)
from cubie.integrators.step_control.adaptive_PI_controller import AdaptivePIController
from cubie.integrators.step_control.fixed_step_controller import FixedStepController
from cubie.integrators.step_control.gustafsson_controller import GustafssonController
from cubie.memory import default_memmgr
from cubie.outputhandling.output_functions import OutputFunctions
from cubie.outputhandling.output_sizes import LoopBufferSizes
from tests.integrators.cpu_reference import (CPUODESystem,
                                              run_reference_loop, \
    CPUAdaptiveController)
from tests._utils import run_device_loop, _driver_sequence
from tests.integrators.loops.test_ode_loop import Array
from tests.system_fixtures import (
    build_large_nonlinear_system,
    build_three_chamber_system,
    build_three_state_linear_system,
    build_three_state_nonlinear_system,
    build_three_state_very_stiff_system,
)


@pytest.fixture(scope="function", autouse=True)
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
    return "linear"


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

    raise ValueError(f"Unknown model type: {model_type}")


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
def loop_compile_settings_overrides(request):
    """Parametrize this fixture indirectly to change compile settings, no need to request this fixture directly
    unless you're testing that it worked."""
    return request.param if hasattr(request, "param") else {}

@pytest.fixture(scope="function")
def solver_settings_override(request):
    """Override for solver settings, if provided."""
    return request.param if hasattr(request, "param") else {}

@pytest.fixture(scope="function")
def solver_settings(
    loop_compile_settings, solver_settings_override, precision
):
    """Create LoopStepConfig from loop_compile_settings."""
    defaults = {
        "algorithm": "euler",
        "duration": 1.0,
        "warmup": 0.0,
        "dt_min": loop_compile_settings["dt_min"],
        "dt_max": loop_compile_settings["dt_max"],
        "dt_save": loop_compile_settings["dt_save"],
        "dt_summarise": loop_compile_settings["dt_summarise"],
        "atol": loop_compile_settings["atol"],
        "rtol": loop_compile_settings["rtol"],
        "saved_state_indices": loop_compile_settings["saved_state_indices"],
        "saved_observable_indices": loop_compile_settings[
            "saved_observable_indices"
        ],
        "summarised_state_indices": loop_compile_settings[
            "summarised_state_indices"],
        "summarised_observable_indices": loop_compile_settings[
            "summarised_observable_indices"],
        "output_types": loop_compile_settings["output_functions"],
        "blocksize": 32,
        "stream": 0,
        "profileCUDA": False,
        "memory_manager": default_memmgr,
        "stream_group": "test_group",
        "mem_proportion": None,
        "step_controller": "fixed",
    }

    if solver_settings_override:
        # Update defaults with any overrides provided
        for key, value in solver_settings_override.items():
            defaults[key] = value

    return defaults


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
        "nonlinear_tolerance": 1e-6,
        'preconditioner_order': 1,
        "max_linear_iters": 100,
        "max_newton_iters": 100,
        "newton_damping": 0.5,
        "newton_max_backtracks": 25
    }
    defaults.update(implicit_step_settings_override)
    return defaults


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
def loop_compile_settings(request, system, loop_compile_settings_overrides):
    """
    Create a dictionary of compile settings for the loop function.
    This is the fixture your test should use - if you want to change the compile settings, indirectly parametrize the
    compile_settings_overrides fixture.
    """
    loop_compile_settings_dict = {
        "dt_min": 0.01,
        "dt_max": 1.0,
        "dt_save": 0.1,
        "dt_summarise": 0.2,
        "atol": 1e-6,
        "rtol": 1e-6,
        "saved_state_indices": [0, 1],
        "saved_observable_indices": [0, 1],
        "summarised_state_indices": [0, 1],
        "summarised_observable_indices": [0, 1],
        "output_functions": ["state"],
    }
    loop_compile_settings_dict.update(loop_compile_settings_overrides)
    return loop_compile_settings_dict


@pytest.fixture(scope="function")
def initial_state(system, precision):
    """Return a copy of the system's initial state vector."""

    return system.initial_values.values_array.astype(precision, copy=True)


@pytest.fixture(scope="function")
def loop_buffer_sizes(system, output_functions):
    """Loop buffer sizes derived from the system and output configuration."""

    return LoopBufferSizes.from_system_and_output_fns(system, output_functions)


@pytest.fixture(scope="function")
def step_controller_settings_override(request):
    """Override dictionary for the step controller configuration."""

    return request.param if hasattr(request, "param") else {}


@pytest.fixture(scope="function")
def step_controller_settings(
    solver_settings, system, step_controller_settings_override
):
    """Base configuration used to instantiate loop step controllers."""

    defaults = {
        "kind": solver_settings["step_controller"].lower(),
        "dt": solver_settings["dt_min"],
        "dt_min": solver_settings["dt_min"],
        "dt_max": solver_settings["dt_max"],
        "atol": solver_settings["atol"],
        "rtol": solver_settings["rtol"],
        "order": 1,
        "n": system.sizes.states,
        "kp": 0.6,
        "ki": 0.4,
        "kd": 0.1,
    }
    overrides = {**step_controller_settings_override}
    defaults.update(overrides)
    return defaults



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
        )
    else:
        raise ValueError(f"Unknown adaptive controller kind '{kind}'.")

    controller.update(updates_dict=settings, silent=True)
    return controller


@pytest.fixture(scope="function")
def loop(
    precision,
    step_object,
    loop_buffer_sizes,
    output_functions,
    step_controller,
    loop_compile_settings,
):
    """Construct the :class:`IVPLoop` instance used in loop tests."""

    return IVPLoop(
        precision=precision,
        dt_save=loop_compile_settings["dt_save"],
        dt_summarise=loop_compile_settings["dt_summarise"],
        step_controller=step_controller,
        step_object=step_object,
        buffer_sizes=loop_buffer_sizes,
        compile_flags=output_functions.compile_flags,
        save_state_func=output_functions.save_state_func,
        update_summaries_func=output_functions.update_summaries_func,
        save_summaries_func=output_functions.save_summary_metrics_func,
    )


@pytest.fixture(scope='function')
def cpu_system(system):
    """Return a CPU-based system."""
    return CPUODESystem(system)


@pytest.fixture
def step_object(solver_settings, implicit_step_settings, precision, system):
    """Return a step object for the given solver settings."""
    if solver_settings["algorithm"].lower() == 'euler':
        solver_kwargs = {
                'dt':solver_settings["dt_min"],
                'precision':precision,
                'n':system.sizes.states,
                'dxdt_function':system.dxdt_function
        }
    else:
        solver_kwargs = {
            "precision": precision,
            "n": system.sizes.states,
            'dxdt_function':system.dxdt_function,
            'get_solver_helper_fn':system.get_solver_helper,
            'preconditioner_order':implicit_step_settings[
            "preconditioner_order"],
            'linsolve_tolerance':implicit_step_settings["linear_tolerance"],
            'max_linear_iters':implicit_step_settings["max_linear_iters"],
            'linear_correction_type':implicit_step_settings[
            "correction_type"],
            'nonlinear_tolerance':implicit_step_settings["nonlinear_tolerance"],
            'max_newton_iters':implicit_step_settings["max_newton_iters"],
            'newton_damping':implicit_step_settings["newton_damping"],
            'newton_max_backtracks':implicit_step_settings[
                "newton_max_backtracks"],
        }
    return get_algorithm_step(solver_settings["algorithm"].lower(),
                              **solver_kwargs)


@pytest.fixture(scope="function")
def cpu_step_controller(precision, step_controller_settings):
    """Instantiate the requested step controller for loop execution."""
    kind = step_controller_settings["kind"].lower()

    controller = CPUAdaptiveController(
        kind=step_controller_settings["kind"].lower(),
        dt_min=step_controller_settings["dt"],
        dt_max=step_controller_settings["dt_max"],
        atol=step_controller_settings["atol"],
        rtol=step_controller_settings["rtol"],
        order=step_controller_settings["order"],
        precision=precision,
    )
    if kind == 'pi':
        controller.kp = step_controller_settings["kp"]
        controller.ki = step_controller_settings["ki"]
    elif kind == 'pid':
        controller.kp = step_controller_settings["kp"]
        controller.ki = step_controller_settings["ki"]
        controller.kd = step_controller_settings["kd"]

    return controller

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
) -> dict[str, Array]:
    """Execute the CPU reference loop with the provided configuration."""
    drivers = _driver_sequence(
            samples=int(np.ceil(
                    solver_settings["duration"] / solver_settings["dt_save"])),
            total_time=solver_settings["duration"],
            n_drivers=system.num_drivers,
            precision=precision)

    inputs = {'initial_values': initial_state.copy(),
              'parameters': system.parameters.values_array.copy(),
              'drivers': drivers}

    return run_reference_loop(
        evaluator=cpu_system,
        inputs=inputs,
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
    initial_state,
    solver_settings,
    step_controller_settings,
    output_functions,
    cpu_system,
):
    """Execute the device loop with the provided configuration."""
    return  run_device_loop(
        loop=loop,
        system=system,
        initial_state=initial_state,
        output_functions=output_functions,
        solver_config=solver_settings,
    )


