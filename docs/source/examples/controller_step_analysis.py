"""Visualise CPU reference controller step-size behaviour across systems.

This module recreates the CPU loop used in the integration tests so we can
inspect how the adaptive step-size controllers behave for a selection of ODE
systems.  The script mirrors the fixtures from :mod:`tests.conftest` closely,
particularly the solver configuration dictionaries and the driver generation
helpers.  Running the module will produce one figure per controller type where
accepted steps are plotted in blue and rejected steps in red for each system.

The implementation intentionally favours clarity over performance so that the
behaviour matches the reference implementation used in the tests.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from cubie.integrators.array_interpolator import ArrayInterpolator
from cubie.memory import default_memmgr
from cubie.outputhandling.output_functions import OutputFunctions
from tests._utils import _driver_sequence
from tests.integrators.cpu_reference import (
    Array,
    CPUAdaptiveController,
    CPUODESystem,
    DriverEvaluator,
    STATUS_MASK,
    _collect_saved_outputs,
    get_ref_stepper,
)
from tests.system_fixtures import (
    build_large_nonlinear_system,
    build_three_chamber_system,
    build_three_state_nonlinear_system,
    build_three_state_very_stiff_system,
)

ArrayFloat = NDArray[np.floating[Any]]


@dataclass
class StepRecord:
    """Time-stamped record of a proposed step.

    Parameters
    ----------
    time : float
        Simulation time when the step was proposed.
    step_size : float
        Proposed step size in seconds.
    accepted : bool
        Whether the controller accepted the step.
    run_label : str
        Identifier describing which Monte-Carlo run produced the record.
    attempt_index : int
        Monotonic counter of how many proposals have been made.
    driver_values : tuple(float, ...)
        Driver sample(s) at proposal time. Empty when system has no drivers.
    """

    time: float
    step_size: float
    accepted: bool
    run_label: str
    attempt_index: int
    driver_values: Tuple[float, ...] = ()


SYSTEM_BUILDERS: Dict[str, Any] = {
    "large": build_large_nonlinear_system,
    "threecm": build_three_chamber_system,
    "nonlinear": build_three_state_nonlinear_system,
    "stiff": build_three_state_very_stiff_system,
}
SYSTEM_LABELS: Dict[str, str] = {
    "large": "Large",
    "threecm": "Three-chamber",
    "nonlinear": "Nonlinear",
    "stiff": "Stiff",
}
SYSTEM_ORDER: Tuple[str, ...] = tuple(SYSTEM_BUILDERS.keys())


FEHLBERG_DEFAULT_OVERRIDES: Dict[str, float] = {
    "dt": 0.001953125,
    "dt_min": 1e-7,
    "newton_tolerance": 1e-7,
    "krylov_tolerance": 1e-7,
    "atol": 1e-5,
    "rtol": 1e-6,
}


def build_solver_settings(precision: type[np.floating[Any]]) -> Dict[str, Any]:
    """Return the Fehlberg-45 configuration used in the CPU tests.

    Parameters
    ----------
    precision
        Floating-point dtype used for the integration.

    Returns
    -------
    dict
        Default solver configuration mirroring ``tests.conftest`` with the
        Fehlberg overrides from :mod:`tests.integrators.loops.test_ode_loop`.
    """

    defaults: Dict[str, Any] = {
        "algorithm": "sdirk_2_2",
        "duration": precision(1.0),
        "warmup": precision(0.0),
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
        "step_controller": "pi",
        "precision": precision,
        "driverspline_order": 3,
        "driverspline_wrap": False,
        "krylov_tolerance": precision(1e-6),
        "correction_type": "minimal_residual",
        "newton_tolerance": precision(1e-6),
        "preconditioner_order": 2,
        "max_linear_iters": 5,
        "max_newton_iters": 5,
        "newton_damping": precision(0.85),
        "newton_max_backtracks": 25,
        "min_gain": precision(0.2),
        "max_gain": precision(2.0),
        "kp": precision(1 / 18),
        "ki": precision(1 / 9),
        "kd": precision(1 / 18),
        "deadband_min": precision(1.0),
        "deadband_max": precision(1.2),
    }
    #
    # for key, value in FEHLBERG_DEFAULT_OVERRIDES.items():
    #     defaults[key] = precision(value)

    return defaults


def build_implicit_step_settings(
    solver_settings: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return the implicit solver defaults from the fixtures.

    Parameters
    ----------
    solver_settings
        Mapping describing the solver configuration.

    Returns
    -------
    dict
        Dictionary of implicit solver helper settings.
    """

    return {
        "atol": solver_settings["atol"],
        "rtol": solver_settings["rtol"],
        "krylov_tolerance": solver_settings["krylov_tolerance"],
        "correction_type": solver_settings["correction_type"],
        "newton_tolerance": solver_settings["newton_tolerance"],
        "preconditioner_order": solver_settings["preconditioner_order"],
        "max_linear_iters": solver_settings["max_linear_iters"],
        "max_newton_iters": solver_settings["max_newton_iters"],
        "newton_damping": solver_settings["newton_damping"],
        "newton_max_backtracks": solver_settings["newton_max_backtracks"],
        "tableau": solver_settings.get("tableau", "sdirk_2_2"),
    }


def build_step_controller_settings(
    solver_settings: Mapping[str, Any],
    system: Any,
    overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Return the adaptive controller settings, applying overrides.

    Parameters
    ----------
    solver_settings
        Base solver configuration.
    system
        Symbolic system instance providing shape information.
    overrides
        Optional mapping of override values.

    Returns
    -------
    dict
        Controller settings compatible with ``CPUAdaptiveController``.
    """

    precision = solver_settings["precision"]
    base_settings: Dict[str, Any] = {
        "kind": solver_settings["step_controller"].lower(),
        "dt": precision(solver_settings["dt"]),
        "dt_min": precision(solver_settings["dt_min"]),
        "dt_max": precision(solver_settings["dt_max"]),
        "atol": precision(solver_settings["atol"]),
        "rtol": precision(solver_settings["rtol"]),
        "order": 5,
        "min_gain": precision(solver_settings["min_gain"]),
        "max_gain": precision(solver_settings["max_gain"]),
        "kp": precision(solver_settings["kp"]),
        "ki": precision(solver_settings["ki"]),
        "kd": precision(solver_settings["kd"]),
        "deadband_min": precision(solver_settings["deadband_min"]),
        "deadband_max": precision(solver_settings["deadband_max"]),
        "max_newton_iters": int(solver_settings["max_newton_iters"]),
    }

    if overrides:
        float_keys = {
            "dt",
            "dt_min",
            "dt_max",
            "atol",
            "rtol",
            "min_gain",
            "max_gain",
            "kp",
            "ki",
            "kd",
            "deadband_min",
            "deadband_max",
        }
        for key, value in overrides.items():
            if key in float_keys:
                base_settings[key] = precision(value)
            else:
                base_settings[key] = value

    return base_settings


def build_driver_settings(
    solver_settings: Mapping[str, Any],
    system: Any,
    precision: type[np.floating[Any]],
) -> Optional[Dict[str, Any]]:
    """Create the driver settings dictionary used by the fixtures.

    Parameters
    ----------
    solver_settings
        Mapping describing solver behaviour.
    system
        Symbolic system providing driver metadata.
    precision
        Floating-point dtype for generated arrays.

    Returns
    -------
    dict or None
        Dictionary consumed by :class:`ArrayInterpolator`, or ``None`` when the
        system has no drivers.
    """

    if system.num_drivers == 0:
        return None

    dt_sample = precision(solver_settings["dt_save"]) / 2.0
    total_span = precision(solver_settings["duration"])
    t0 = precision(solver_settings["warmup"])
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
    drivers_dict: Dict[str, ArrayFloat] = {
        name: np.array(driver_matrix[:, idx], dtype=precision, copy=True)
        for idx, name in enumerate(driver_names)
    }
    drivers_dict["dt"] = precision(dt_sample)
    drivers_dict["wrap"] = bool(solver_settings["driverspline_wrap"])
    drivers_dict["order"] = order
    drivers_dict["t0"] = t0

    return drivers_dict


def create_output_functions(
    system: Any, solver_settings: Mapping[str, Any]
) -> OutputFunctions:
    """Build :class:`OutputFunctions` mirroring the fixture helper.

    Parameters
    ----------
    system
        Symbolic system supplying size information.
    solver_settings
        Solver configuration mapping.

    Returns
    -------
    OutputFunctions
        Configured output handler instance.
    """

    return OutputFunctions(
        system.sizes.states,
        system.sizes.parameters,
        solver_settings["output_types"],
        solver_settings["saved_state_indices"],
        solver_settings["saved_observable_indices"],
        solver_settings["summarised_state_indices"],
        solver_settings["summarised_observable_indices"],
    )


def create_controller(
    settings: Mapping[str, Any], precision: type[np.floating[Any]]
) -> CPUAdaptiveController:
    """Instantiate a CPU controller matching the GPU configuration.

    Parameters
    ----------
    settings
        Controller configuration mapping.
    precision
        Floating-point dtype for the controller internals.

    Returns
    -------
    CPUAdaptiveController
        Configured controller instance.
    """

    controller = CPUAdaptiveController(
        kind=settings["kind"],
        dt=settings["dt"],
        dt_min=settings["dt_min"],
        dt_max=settings["dt_max"],
        atol=settings["atol"],
        rtol=settings["rtol"],
        order=settings["order"],
        precision=precision,
        min_gain=settings["min_gain"],
        max_gain=settings["max_gain"],
        deadband_min=settings["deadband_min"],
        deadband_max=settings["deadband_max"],
    )
    kind = settings["kind"].lower()
    if kind == "pi":
        controller.kp = settings["kp"]
        controller.ki = settings["ki"]
    elif kind == "pid":
        controller.kp = settings["kp"]
        controller.ki = settings["ki"]
        controller.kd = settings["kd"]
    return controller


def create_driver_array(
    driver_settings: Optional[Mapping[str, Any]],
    precision: type[np.floating[Any]],
) -> Optional[ArrayInterpolator]:
    """Instantiate :class:`ArrayInterpolator` when drivers are present.

    Parameters
    ----------
    driver_settings
        Mapping produced by :func:`build_driver_settings`.
    precision
        Floating-point dtype for interpolation coefficients.

    Returns
    -------
    ArrayInterpolator or None
        Interpolator matching the fixture behaviour.
    """

    if driver_settings is None:
        return None
    return ArrayInterpolator(precision=precision, input_dict=driver_settings)


def create_driver_evaluator(
    solver_settings: Mapping[str, Any],
    system: Any,
    precision: type[np.floating[Any]],
    driver_array: Optional[ArrayInterpolator],
) -> DriverEvaluator:
    """Create a driver evaluator based on the configured splines.

    Parameters
    ----------
    solver_settings
        Mapping describing solver configuration.
    system
        Symbolic system providing driver metadata.
    precision
        Floating-point dtype for generated arrays.
    driver_array
        Optional interpolator used to compute spline coefficients.

    Returns
    -------
    DriverEvaluator
        CPU-side evaluator matching the fixtures.
    """

    width = system.num_drivers
    order = int(solver_settings["driverspline_order"])
    if driver_array is None or width == 0:
        coeffs = np.zeros((1, width, order + 1), dtype=precision)
        dt_value = precision(solver_settings["dt_save"]) / 2.0
        t0_value = precision(0.0)
        wrap_value = bool(solver_settings["driverspline_wrap"])
        boundary = None
    else:
        coeffs = np.array(
            driver_array.coefficients, dtype=precision, copy=True
        )
        dt_value = precision(driver_array.dt)
        t0_value = precision(driver_array.t0)
        wrap_value = bool(driver_array.wrap)
        boundary = driver_array.boundary_condition

    return DriverEvaluator(
        coefficients=coeffs,
        dt=dt_value,
        t0=t0_value,
        wrap=wrap_value,
        precision=precision,
        boundary_condition=boundary,
    )



def generate_variants(
    system: Any,
    precision: type[np.floating[Any]],
    *,
    seed: int,
    count: int = 1,
) -> List[Tuple[ArrayFloat, ArrayFloat]]:
    """Return ``count`` diverse (state, parameter) pairs for the system.

    Parameters
    ----------
    system
        Symbolic system providing baseline values.
    precision
        Floating-point dtype for generated samples.
    seed
        Seed used to initialise the random generator.
    count
        Number of variants to generate.

    Returns
    -------
    list of tuple(ndarray, ndarray)
        Randomised state and parameter pairs.
    """

    rng = np.random.default_rng(seed)
    base_state = system.initial_values.values_array.astype(
        precision, copy=True
    )
    base_params = system.parameters.values_array.astype(precision, copy=True)
    sign_state = np.where(base_state == 0, 1.0, np.sign(base_state))
    sign_params = np.where(base_params == 0, 1.0, np.sign(base_params))
    variations: List[Tuple[ArrayFloat, ArrayFloat]] = []

    for _ in range(count):
        state_scale = rng.uniform(0.2, 1.8, size=base_state.shape)
        param_scale = rng.uniform(0.3, 2.0, size=base_params.shape)
        state_offset = rng.uniform(-0.6, 0.6, size=base_state.shape)
        param_offset = rng.uniform(-0.2, 0.2, size=base_params.shape)
        varied_state = sign_state * (
            (np.abs(base_state) + 0.5) * state_scale + state_offset
        )
        varied_params = sign_params * (
            (np.abs(base_params) + 0.1) * param_scale + param_offset
        )
        varied_state = np.clip(varied_state, -5.0, 5.0)
        varied_params = np.clip(varied_params, -1, 1)
        variations.append(
            (
                varied_state.astype(precision, copy=False),
                varied_params.astype(precision, copy=False),
            )
        )

    return variations


def generate_inputs(
    solver_settings: Mapping[str, Any],
    system: Any,
    precision: type[np.floating[Any]],
    initial_state: ArrayFloat,
    parameters: ArrayFloat,
    driver_array: Optional[ArrayInterpolator],
) -> Dict[str, ArrayFloat]:
    """Create the inputs dictionary consumed by the reference loop.

    Parameters
    ----------
    solver_settings
        Mapping describing solver behaviour.
    system
        Symbolic system providing driver metadata.
    precision
        Floating-point dtype for generated arrays.
    initial_state
        Initial condition for the run.
    parameters
        Parameter vector for the run.
    driver_array
        Optional driver interpolator providing spline coefficients.

    Returns
    -------
    dict
        Inputs compatible with :func:`run_reference_loop`.
    """

    inputs: Dict[str, ArrayFloat] = {
        "initial_values": np.asarray(initial_state, dtype=precision).copy(),
        "parameters": np.asarray(parameters, dtype=precision).copy(),
    }
    if driver_array is not None:
        inputs["driver_coefficients"] = np.asarray(
            driver_array.coefficients, dtype=precision
        ).copy()
    else:
        width = system.num_drivers
        order = int(solver_settings["driverspline_order"])
        inputs["driver_coefficients"] = np.zeros(
            (1, width, order + 1), dtype=precision
        )
    return inputs


def run_reference_loop_with_history(
    evaluator: CPUODESystem,
    inputs: Mapping[str, Array],
    solver_settings: Mapping[str, Any],
    implicit_step_settings: Mapping[str, Any],
    controller: CPUAdaptiveController,
    driver_evaluator: DriverEvaluator,
    output_functions: OutputFunctions,
    *,
    run_label: str,
) -> Tuple[Dict[str, Array], List[StepRecord]]:
    """Execute a CPU loop while recording proposed step sizes.

    Parameters
    ----------
    evaluator
        CPU representation of the symbolic system.
    inputs
        Dictionary of initial values, parameters, and driver coefficients.
    solver_settings
        Mapping describing solver behaviour.
    implicit_step_settings
        Helper settings used by implicit algorithms.
    controller
        Adaptive or fixed step controller instance.
    driver_evaluator
        Callable returning driver samples for requested times.
    output_functions
        Output configuration mirroring the GPU loop.
    run_label
        Identifier describing the Monte-Carlo variant.

    Returns
    -------
    tuple
        Tuple containing loop outputs and the recorded step history.
    """

    precision = evaluator.precision
    initial_state = inputs["initial_values"].astype(precision, copy=True)
    params = inputs["parameters"].astype(precision, copy=True)
    driver_coefficients = inputs.get("driver_coefficients")
    if driver_coefficients is not None:
        driver_fn = driver_evaluator.with_coefficients(
            np.asarray(driver_coefficients, dtype=precision)
        )
    else:
        driver_fn = driver_evaluator

    duration = precision(solver_settings["duration"])
    warmup = precision(solver_settings["warmup"])
    dt_save = precision(solver_settings["dt_save"])
    status_flags = 0
    zero = precision(0.0)
    step_records: List[StepRecord] = []

    tableau = implicit_step_settings.get("tableau")
    stepper = get_ref_stepper(
        evaluator,
        driver_fn,
        solver_settings["algorithm"],
        newton_tol=implicit_step_settings["newton_tolerance"],
        newton_max_iters=implicit_step_settings["max_newton_iters"],
        linear_tol=implicit_step_settings["krylov_tolerance"],
        linear_max_iters=implicit_step_settings["max_linear_iters"],
        linear_correction_type=implicit_step_settings["correction_type"],
        preconditioner_order=implicit_step_settings["preconditioner_order"],
        tableau=tableau,
        newton_damping=implicit_step_settings.get("newton_damping"),
        newton_max_backtracks=implicit_step_settings.get(
            "newton_max_backtracks"
        ),
    )

    saved_state_indices = np.asarray(
        solver_settings["saved_state_indices"], dtype=np.int32
    )


    save_time = output_functions.save_time
    max_save_samples = int(np.ceil(duration / dt_save))

    state = initial_state.copy()
    state_history = [state.copy()]
    observable_history: List[Array] = []
    time_history: List[float] = []
    t = precision(0.0)
    controller.dt = controller.dt0
    drivers_initial = driver_fn(precision(t))
    observables = evaluator.observables(
        state,
        params,
        drivers_initial,
        t,
    )

    if warmup > zero:
        next_save_time = warmup
    else:
        next_save_time = dt_save
        observable_history.append(observables.copy())
        time_history.append(float(t))

    end_time = precision(warmup + duration)
    equality_breaker = (
        precision(1e-7) if precision is np.float32 else precision(1e-14)
    )
    fixed_steps_per_save = int(np.ceil(dt_save / controller.dt))
    fixed_step_count = 0
    attempt_index = 0

    while t < end_time - equality_breaker:
        dt = precision(min(controller.dt, end_time - t))
        do_save = False
        if controller.is_adaptive:
            if t + dt + equality_breaker >= next_save_time:
                dt = precision(next_save_time - t)
                do_save = True
        else:
            if (fixed_step_count + 1) % fixed_steps_per_save == 0:
                do_save = True
                fixed_step_count = 0
            else:
                fixed_step_count += 1

        try:
            sampled = driver_fn(precision(t))
            sampled_tuple: Tuple[float, ...] = tuple(
                np.asarray(sampled, dtype=float).tolist()
            )
        except Exception:
            sampled_tuple = ()

        result = stepper.step(
            state=state,
            params=params,
            dt=dt,
            time=precision(t),
        )

        step_status = int(result.status)
        status_flags |= step_status & STATUS_MASK
        accept = controller.propose_dt(
            error_vector=result.error,
            prev_state=state,
            new_state=result.state,
            niters=result.niters,
        )
        step_records.append(
            StepRecord(
                time=float(t),
                step_size=float(dt),
                accepted=accept,
                run_label=run_label,
                attempt_index=attempt_index,
                driver_values=sampled_tuple,
            )
        )
        attempt_index += 1
        if not accept:
            continue

        state = result.state.copy()
        observables = result.observables.copy()
        t += precision(dt)

        if do_save:
            if len(state_history) < max_save_samples:
                state_history.append(result.state.copy())
                observable_history.append(result.observables.copy())
                time_history.append(float(next_save_time - warmup))
            next_save_time += dt_save

    state_output = _collect_saved_outputs(
        state_history, max_save_samples, saved_state_indices, precision
    )
    if save_time and time_history:
        state_output = np.column_stack(
            (state_output, np.asarray(time_history, dtype=precision))
        )
    final_status = status_flags & STATUS_MASK
    outputs = {
        "state": state_output,
        "status": final_status,
    }
    return outputs, step_records


def plot_step_histories(
    controller: str,
    histories: Mapping[str, Sequence[StepRecord]],
) -> None:
    """Render the controller-specific figure showing accepted and rejected steps.

    Parameters
    ----------
    controller
        Controller identifier (``"i"``, ``"pi"``, ``"pid"``, or
        ``"gustafsson"``).
    histories
        Mapping from system name to recorded step histories.
    """

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False, sharey=False)
    axes = axes.ravel()
    controller_label = controller.upper()

    for axis, system_name in zip(axes, SYSTEM_ORDER):
        records = histories.get(system_name, [])
        accepted = [record for record in records if record.accepted]
        rejected = [record for record in records if not record.accepted]

        if accepted:
            axis.scatter(
                [record.time for record in accepted],
                [record.step_size for record in accepted],
                color="tab:blue",
                alpha=0.7,
                marker="o",
                s=18,
                label="accepted",
            )
        if rejected:
            axis.scatter(
                [record.time for record in rejected],
                [record.step_size for record in rejected],
                color="tab:red",
                alpha=0.7,
                marker="x",
                s=18,
                label="rejected",
            )

        axis.set_title(SYSTEM_LABELS.get(system_name, system_name))
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Step size (s)")
        axis.grid(True, linestyle=":", linewidth=0.6)
        if accepted and rejected:
            axis.legend(loc="best")

    fig.suptitle(
        f"Step-size history for {controller_label} controller", fontsize=14
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.show()


def run_analysis() -> Dict[str, Dict[str, List[StepRecord]]]:
    """Execute integrations for every controller and system combination."""

    all_histories: Dict[str, Dict[str, List[StepRecord]]] = {}
    for controller in ("i", "pi", "pid", "gustafsson"):
        controller_histories: Dict[str, List[StepRecord]] = {}

        for system_name, builder in SYSTEM_BUILDERS.items():
            print(
                "Running analysis for "
                f"{controller} controller on {system_name}"
            )
            precision = (
                np.float64 if system_name == "stiff" else np.float32
            )
            system = builder(precision)
            solver_settings = build_solver_settings(precision)
            solver_settings = {
                **solver_settings,
                "step_controller": controller,
            }
            step_settings = build_step_controller_settings(
                solver_settings, system
            )
            implicit_settings = build_implicit_step_settings(solver_settings)
            output_functions = create_output_functions(system, solver_settings)
            cpu_system = CPUODESystem(system)

            driver_settings = build_driver_settings(
                solver_settings, system, precision
            )
            driver_array = create_driver_array(driver_settings, precision)
            driver_evaluator = create_driver_evaluator(
                solver_settings,
                system,
                precision,
                driver_array,
            )

            histories: List[StepRecord] = []
            variants = generate_variants(
                system,
                precision,
                seed=hash((controller, system_name)) & 0xFFFF,
            )
            for variant_idx, variant in enumerate(variants):
                print(
                    f"  Running variant {variant_idx + 1}/{len(variants)}"
                )
                initial_state, parameters = variant
                label = f"run{variant_idx + 1}"
                inputs = generate_inputs(
                    solver_settings=solver_settings,
                    system=system,
                    precision=precision,
                    initial_state=initial_state,
                    parameters=parameters,
                    driver_array=driver_array,
                )
                controller_instance = create_controller(
                    step_settings, precision
                )
                _, step_records = run_reference_loop_with_history(
                    evaluator=cpu_system,
                    inputs=inputs,
                    solver_settings=solver_settings,
                    implicit_step_settings=implicit_settings,
                    controller=controller_instance,
                    driver_evaluator=driver_evaluator,
                    output_functions=output_functions,
                    run_label=label,
                )
                histories.extend(step_records)

            controller_histories[system_name] = histories

        all_histories[controller] = controller_histories
        plot_step_histories(controller, controller_histories)



def main() -> Dict[str, Dict[str, List[StepRecord]]]:
    """Run the analysis pipeline and return the recorded histories."""
    return run_analysis()


if __name__ == "__main__":
    main()

