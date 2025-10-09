import numpy as np
import pytest

from cubie import merge_kwargs_into_settings
from cubie.integrators.algorithms import get_algorithm_step
from cubie.integrators.algorithms.base_algorithm_step import (
    ALL_ALGORITHM_STEP_PARAMETERS,
)
from cubie.integrators.array_interpolator import ArrayInterpolator
from cubie.integrators.loops.ode_loop import IVPLoop
from cubie.integrators.loops.ode_loop_config import (
    LoopLocalIndices,
    LoopSharedIndices,
)
from cubie.integrators.step_control import get_controller
from cubie.integrators.step_control.base_step_controller import (
    ALL_STEP_CONTROLLER_PARAMETERS,
)
from cubie.outputhandling.output_functions import OutputFunctions
from cubie.outputhandling.output_sizes import LoopBufferSizes
from cubie.odesystems.symbolic.symbolicODE import create_ODE_system
from tests._utils import assert_integration_outputs, run_device_loop


@pytest.fixture(scope="function")
def time_driver_solver_settings(precision):
    settings = {
        "algorithm": "euler",
        "duration": precision(1.0),
        "warmup": precision(0.0),
        "dt_min": precision(0.05),
        "dt_max": precision(0.05),
        "dt_save": precision(0.05),
        "dt_summarise": precision(0.1),
        "atol": precision(1e-6),
        "rtol": precision(1e-6),
        "saved_state_indices": [0],
        "saved_observable_indices": [0],
        "summarised_state_indices": [0],
        "summarised_observable_indices": [0],
        "output_types": ["state", "observables", "time"],
        "blocksize": 32,
        "stream": 0,
        "profileCUDA": False,
        "memory_manager": None,
        "stream_group": "time_driver",
        "mem_proportion": None,
        "step_controller": "fixed",
        "precision": precision,
        "driverspline_order": 3,
        "driverspline_wrap": False,
    }
    return settings


@pytest.fixture(scope="function")
def time_driver_systems(precision):
    sinusoid_equations = [
        "dx = -x + sin(t)",
        "obs = x",
    ]
    interpolated_equations = [
        "dx = -x + drive",
        "obs = x",
    ]
    function_system = create_ODE_system(
        dxdt=sinusoid_equations,
        states={"x": 0.5},
        observables=["obs"],
        precision=precision,
        strict=True,
        name="time_function_driver",
    )
    interpolated_system = create_ODE_system(
        dxdt=interpolated_equations,
        states={"x": 0.5},
        observables=["obs"],
        drivers=["drive"],
        precision=precision,
        strict=True,
        name="time_array_driver",
    )
    return function_system, interpolated_system


@pytest.fixture(scope="function")
def sinusoid_driver_array(precision, time_driver_solver_settings):
    duration = float(time_driver_solver_settings["duration"])
    sample_dt = float(time_driver_solver_settings["dt_save"])
    num_samples = int(np.round(duration / sample_dt)) + 1
    times = np.linspace(0.0, duration, num_samples, dtype=precision)
    values = np.sin(times.astype(np.float64)).astype(precision)
    input_dict = {
        "drive": values,
        "time": times,
        "order": int(time_driver_solver_settings["driverspline_order"]),
        "wrap": bool(time_driver_solver_settings["driverspline_wrap"]),
    }
    driver_array = ArrayInterpolator(
        precision=precision,
        input_dict=input_dict,
    )
    return driver_array

def _build_loop(
    system,
    solver_settings,
    output_functions,
    precision,
    driver_array=None,
):
    loop_buffer_sizes = LoopBufferSizes.from_system_and_output_fns(
        system, output_functions
    )
    driver_function = (
        driver_array.evaluation_function if driver_array is not None else None
    )
    algorithm_settings, _ = merge_kwargs_into_settings(kwargs=solver_settings,
                                                       valid_keys=ALL_ALGORITHM_STEP_PARAMETERS)
    algorithm_settings["algorithm"] = solver_settings["algorithm"]
    algorithm_settings["dt"] = float(solver_settings["dt_min"])
    algorithm_settings["n"] = system.sizes.states
    algorithm_settings["dxdt_function"] = system.dxdt_function
    algorithm_settings["observables_function"] = system.observables_function
    algorithm_settings["get_solver_helper_fn"] = (
        system.get_solver_helper
    )
    if driver_function is not None:
        algorithm_settings["driver_function"] = driver_function

    step_object = get_algorithm_step(
        precision=precision,
        settings=algorithm_settings,
    )

    controller_settings, _ = merge_kwargs_into_settings(kwargs=solver_settings,
                                                        valid_keys=ALL_STEP_CONTROLLER_PARAMETERS)
    controller_settings["step_controller"] = solver_settings[
        "step_controller"
    ]
    controller_settings.setdefault("dt", float(solver_settings["dt_min"]))
    controller_settings["n"] = system.sizes.states

    step_controller = get_controller(
        precision=precision,
        settings=controller_settings,
    )
    shared_indices = LoopSharedIndices.from_sizes(
        n_states=loop_buffer_sizes.state,
        n_observables=loop_buffer_sizes.observables,
        n_parameters=loop_buffer_sizes.parameters,
        n_drivers=loop_buffer_sizes.drivers,
        state_summaries_buffer_height=(
            loop_buffer_sizes.state_summaries
        ),
        observable_summaries_buffer_height=(
            loop_buffer_sizes.observable_summaries
        ),
        n_error=(
            loop_buffer_sizes.state if step_object.is_adaptive else 0
        ),
    )
    local_indices = LoopLocalIndices.from_sizes(
        controller_len=step_controller.local_memory_elements,
        algorithm_len=step_object.persistent_local_required,
    )
    loop = IVPLoop(
        precision=precision,
        shared_indices=shared_indices,
        local_indices=local_indices,
        compile_flags=output_functions.compile_flags,
        save_state_func=output_functions.save_state_func,
        update_summaries_func=output_functions.update_summaries_func,
        save_summaries_func=output_functions.save_summary_metrics_func,
        step_controller_fn=step_controller.device_function,
        step_function=step_object.step_function,
        driver_function=driver_function,
        observables_fn=system.observables_function,
        dt_save=float(solver_settings["dt_save"]),
        dt_summarise=float(solver_settings["dt_summarise"]),
        dt0=step_controller.dt0,
        dt_min=step_controller.dt_min,
        dt_max=step_controller.dt_max,
        is_adaptive=step_controller.is_adaptive,
    )
    return loop


def test_time_driver_array_matches_function(
    precision,
    time_driver_systems,
    time_driver_solver_settings,
    sinusoid_driver_array,
    single_integrator_run
):
    function_system, interpolated_system = time_driver_systems
    solver_settings = time_driver_solver_settings
    driver_array = sinusoid_driver_array

    output_functions_function = OutputFunctions(
        function_system.sizes.states,
        function_system.sizes.observables,
        solver_settings["output_types"],
        solver_settings["saved_state_indices"],
        solver_settings["saved_observable_indices"],
        solver_settings["summarised_state_indices"],
        solver_settings["summarised_observable_indices"],
    )
    output_functions_driver = OutputFunctions(
        interpolated_system.sizes.states,
        interpolated_system.sizes.observables,
        solver_settings["output_types"],
        solver_settings["saved_state_indices"],
        solver_settings["saved_observable_indices"],
        solver_settings["summarised_state_indices"],
        solver_settings["summarised_observable_indices"],
    )

    loop_function = _build_loop(
        function_system,
        solver_settings,
        output_functions_function,
        precision,
    )
    loop_driver = _build_loop(
        interpolated_system,
        solver_settings,
        output_functions_driver,
        precision,
        driver_array=driver_array,
    )

    reference_result = run_device_loop(
        loop=loop_function,
        system=function_system,
        initial_state=function_system.initial_values.values_array.astype(
            precision, copy=True
        ),
        localmem_required=single_integrator_run.local_memory_elements,
        sharedmem_required=single_integrator_run.shared_memory_bytes,
        output_functions=output_functions_function,
        solver_config=solver_settings,
    )
    driver_result = run_device_loop(
        loop=loop_driver,
        system=interpolated_system,
        initial_state=interpolated_system.initial_values.values_array.astype(
            precision, copy=True
        ),
        localmem_required=single_integrator_run.local_memory_elements,
        sharedmem_required=single_integrator_run.shared_memory_bytes,
        output_functions=output_functions_driver,
        solver_config=solver_settings,
        driver_array=driver_array,
    )

    assert_integration_outputs(
        reference=reference_result,
        device=driver_result,
        output_functions=output_functions_function,
        rtol=1e-5,
        atol=1e-5,
    )
