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
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

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
    get_ref_step_function,
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


def build_solver_settings(precision: type[np.floating[Any]]) -> Dict[str, Any]:
    """Return the default solver configuration used by the tests.

    Parameters
    ----------
    precision
        Floating-point dtype used for the integration.

    Returns
    -------
    dict
        Default solver configuration mirroring ``tests.conftest``.
    """

    return {
        "algorithm": "crank_nicolson",
        "duration": precision(1.0),
        "warmup": precision(0.0),
        "dt_min": precision(1e-6),
        "dt_max": precision(1.0),
        "dt_save": precision(0.1),
        "dt_summarise": precision(0.2),
        "atol": precision(1e-5),
        "rtol": precision(1e-5),
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
        "driverspline_wrap": True,
    }


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
        "linear_tolerance": 1e-5,
        "correction_type": "minimal_residual",
        "nonlinear_tolerance": 1e-5,
        "preconditioner_order": 2,
        "max_linear_iters": 500,
        "max_newton_iters": 500,
        "newton_damping": 0.85,
        "newton_max_backtracks": 25,
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
        "dt": precision(solver_settings["dt_min"]),
        "dt_min": precision(solver_settings["dt_min"]),
        "dt_max": precision(solver_settings["dt_max"]),
        "atol": precision(solver_settings["atol"]),
        "rtol": precision(solver_settings["rtol"]),
        "order": 2,
        "min_gain": precision(0.5),
        "max_gain": precision(2.0),
        "n": system.sizes.states,
        "kp": precision(1 / 18),
        "ki": precision(1 / 9),
        "kd": precision(1 / 18),
        "deadband_min": precision(1.0),
        "deadband_max": precision(1.2),
    }

    if overrides:
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
    t0 = float(solver_settings["warmup"])
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
    drivers_dict["wrap"] = solver_settings["driverspline_wrap"]
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
    controller.dt = settings["dt"]
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
    aryint = ArrayInterpolator(precision=precision, input_dict=driver_settings)
    # aryint.plot_interpolated(np.linspace(0, 1, 1000))
    return aryint


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
        dt_value = float(solver_settings["dt_save"]) / 2.0
        t0_value = float(solver_settings["warmup"])
        wrap_value = bool(solver_settings["driverspline_wrap"])
        boundary = "not-a-knot"
    else:
        coeffs = np.array(
            driver_array.coefficients, dtype=precision, copy=True
        )
        dt_value = float(driver_array.dt)
        t0_value = float(driver_array.t0)
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

    step_function = get_ref_step_function(solver_settings["algorithm"])

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
    max_iters = implicit_step_settings["max_newton_iters"]
    timesave=0
    equality_breaker = (
        precision(1e-7) if precision is np.float32 else precision(1e-14)
    )
    attempt_index = 0

    while t < end_time - equality_breaker:
        dt = precision(min(controller.dt, end_time - t))
        do_save = False
        if t + dt + equality_breaker >= next_save_time:
            dt = precision(next_save_time - t)
            do_save = True

        # Sample driver values at proposal time (before performing the step)
        try:
            sampled = driver_fn(precision(t))
            sampled_tuple: Tuple[float, ...] = tuple(
                np.asarray(sampled, dtype=float).tolist()
            )
        except Exception:
            sampled_tuple = ()

        timenow = perf_counter()

        result = step_function(
            evaluator,
            driver_fn,
            state=state,
            params=params,
            dt=dt,
            tol=implicit_step_settings["nonlinear_tolerance"],
            max_iters=max_iters,
            time=precision(t),
        )

        step_status = int(result.status)
        status_flags |= step_status & STATUS_MASK
        try:
            accept = controller.propose_dt(
                error_vector=result.error,
                prev_state=state,
                new_state=result.state,
                niters=result.niters,
            )
            if timenow - timesave > 5.0:
                print(f"Going slow. t={t}, last step: {dt:.6f} s, accepted="
                      f"{accept}")
                timesave=timenow
        except ValueError:
            print(f"Failed to propose step size at t={t:.2f}, step={dt:.6f}")
            break
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
        t += precision(dt)

        if do_save:
            print(f"Saving sample at t={t:.2f}, step={dt:.6f}")
            timesave = perf_counter()
            if len(state_history) < max_save_samples:
                state_history.append(result.state.copy())
                observable_history.append(result.observables.copy())
                time_history.append(float(next_save_time - warmup))
            next_save_time += dt_save

    state_output = _collect_saved_outputs(
        state_history, saved_state_indices, precision
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
    output_dir: Path,
) -> Path:
    """Create the controller-specific figure and return the saved path.

    Parameters
    ----------
    controller
        Controller identifier (``"i"``, ``"pi"``, ``"pid"``, or
        ``"gustafsson"``).
    histories
        Mapping from system name to recorded step histories.
    output_dir
        Directory in which the figure should be saved.

    Returns
    -------
    pathlib.Path
        Path to the saved figure file.
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

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"controller_step_sizes_{controller.lower()}.png"
    )
    fig.savefig(output_path, dpi=200)
    return output_path


def run_analysis(
    output_dir: Path,
    save_figs: bool = True,
) -> Tuple[List[Path], Dict[str, Dict[str, List[StepRecord]]]]:
    """Execute integrations for every controller and system combination.

    Parameters
    ----------
    output_dir
        Directory in which plots should be saved.
    save_figs
        Whether to render and save matplotlib figures.

    Returns
    -------
    tuple
        Saved figure paths and the recorded step histories.
    """

    saved_paths: List[Path] = []
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
        if save_figs:
            output_path = plot_step_histories(
                controller, controller_histories, output_dir
            )
            saved_paths.append(output_path)

    return saved_paths, all_histories


def plot_driver_histories(
    all_histories: Mapping[str, Mapping[str, Sequence[StepRecord]]],
    *,
    driver_index: int = 0,
) -> List[plt.Figure]:
    """Plot a selected driver value against time for each system.

    Parameters
    ----------
    all_histories
        Mapping of controller -> system -> list of StepRecord.
    driver_index
        Which driver to plot (0-based). Only records with sufficient
        driver_values length are used.

    Returns
    -------
    list[matplotlib.figure.Figure]
        Created figures, one per system that has drivers.
    """
    figs: List[plt.Figure] = []
    for system_name in SYSTEM_ORDER:
        # Collect all records for this system across controllers
        system_records: List[StepRecord] = []
        for controller_hist in all_histories.values():
            system_records.extend(controller_hist.get(system_name, []))
        # Determine if this system has any driver records
        has_driver = any(
            len(rec.driver_values) > driver_index for rec in system_records
        )
        if not has_driver:
            continue
        fig, ax = plt.subplots(figsize=(10, 4))
        for controller, histories in all_histories.items():
            records = [
                r
                for r in histories.get(system_name, [])
                if len(r.driver_values) > driver_index
            ]
            if not records:
                continue
            times = [r.time for r in records]
            dvals = [r.driver_values[driver_index] for r in records]
            ax.scatter(
                times,
                dvals,
                s=14,
                alpha=0.6,
                label=controller,
            )
        ax.set_title(
            f"Driver {driver_index} values vs time for {system_name} system"
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Driver value")
        ax.grid(True, linestyle=":", linewidth=0.6)
        ax.legend(loc="best")
        figs.append(fig)
    return figs


def main(
    save_figs: bool = True, output_dir: Union[str, Path] = "docs/source/_static"
) -> Tuple[List[Path], Dict[str, Dict[str, List[StepRecord]]]]:
    """Run the analysis pipeline and return the generated figure paths.

    Parameters
    ----------
    save_figs
        Whether to render and save matplotlib figures.
    output_dir
        Directory used for saved figures.

    Returns
    -------
    tuple
        Tuple of saved figure paths and recorded histories.
    """

    output_path = Path(output_dir)
    return run_analysis(output_path, save_figs)


if __name__ == "__main__":
    generated, all_histories = main(save_figs=False)
    print("Generated figures:")
    for path in generated:
        print(f"  - {path}")

    # Step-size history composite figures (per system, controllers as subplots)
    allfigs: List[plt.Figure] = []
    allaxes: List[NDArray[np.object_]] = []
    for j, system_name in enumerate(SYSTEM_ORDER):
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(12, 8),
            sharex=False,
            sharey=False,
            layout="constrained",
        )
        axes = axes.ravel()
        fig.suptitle(f"{system_name} system")
        allfigs.append(fig)
        allaxes.append(axes)
        xtype = 'index'
        for i, (controller, histories) in enumerate(all_histories.items()):
            records = histories.get(system_name, [])
            axis = allaxes[j][i]
            axis.set_yscale("log")
            accepted = [record for record in records if record.accepted]
            rejected = [record for record in records if not record.accepted]
            if accepted:
                if xtype == 'index':
                    xdata = [record.attempt_index for record in accepted]
                else:
                    xdata = [record.time for record in accepted]
                axis.scatter(
                    xdata,
                    [record.step_size for record in accepted],
                    color="tab:blue",
                    alpha=0.7,
                    marker="o",
                    s=18,
                )
            if rejected:
                if xtype == 'index':
                    xdata = [record.attempt_index for record in rejected]
                else:
                    xdata = [record.time for record in rejected]
                axis.scatter(
                    xdata,
                    [record.step_size for record in rejected],
                    color="tab:red",
                    alpha=0.7,
                    marker="x",
                    s=18,
                )
            axis.set_title(controller)
            axis.set_xlabel("Time (s)")
            axis.set_ylabel("Step size (s)")
            axis.set_xlim(auto=True)
            axis.set_ylim(auto=True)
            axis.grid(True, linestyle=":", linewidth=0.6)
    # driver_figs = plot_driver_histories(all_histories, driver_index=0)
    plt.show()

