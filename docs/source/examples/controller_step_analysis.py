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

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple
from time import perf_counter
#
# ROOT_DIR = Path(__file__).resolve().parents[3]
# if str(ROOT_DIR) not in sys.path:
#     sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from cubie.outputhandling.output_functions import OutputFunctions
from cubie.memory import default_memmgr
from tests._utils import _driver_sequence
from tests.integrators.cpu_reference import (
    Array,
    CPUAdaptiveController,
    CPUODESystem,
    DriverSampler,
    STATUS_MASK,
    get_ref_step_fn,
    _collect_saved_outputs,
)
from tests.system_fixtures import (
    build_three_chamber_system,
    build_large_nonlinear_system,
    build_three_state_nonlinear_system,
    build_three_state_very_stiff_system,
)


ArrayFloat = NDArray[np.floating[Any]]


@dataclass
class StepRecord:
    """Time-stamped record of a proposed step."""

    time: float
    step_size: float
    accepted: bool
    run_label: str


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
    """Return the default solver configuration used by the tests."""

    return {
        "algorithm": "crank_nicolson",
        "duration": precision(1.0),
        "warmup": precision(0.0),
        "dt_min": precision(1e-6),
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
    }


def build_implicit_step_settings(
    solver_settings: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return the implicit solver defaults from the fixtures."""

    return {
        "atol": solver_settings["atol"],
        "rtol": solver_settings["rtol"],
        "linear_tolerance": 1e-6,
        "correction_type": "minimal_residual",
        "nonlinear_tolerance": 1e-4,
        "preconditioner_order": 2,
        "max_linear_iters": 500,
        "max_newton_iters": 500,
        "newton_damping": 0.85,
        "newton_max_backtracks": 25,
    }


def build_step_controller_settings(
    solver_settings: Mapping[str, Any],
    system: Any,
    overrides: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Return the adaptive controller settings, applying overrides if any."""

    precision = solver_settings["precision"]
    base_settings: Dict[str, Any] = {
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
        "kp": precision(1 / 18),
        "ki": precision(1 / 9),
        "kd": precision(1 / 18),
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
        }
        for key, value in overrides.items():
            if key in float_keys:
                base_settings[key] = precision(value)
            else:
                base_settings[key] = value

    return base_settings


def create_output_functions(
    system: Any, solver_settings: Mapping[str, Any]
) -> OutputFunctions:
    """Build :class:`OutputFunctions` like the ``output_functions`` fixture."""

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
    """Instantiate a CPU controller matching the GPU configuration."""

    controller = CPUAdaptiveController(
        kind=settings["kind"],
        dt_min=settings["dt_min"],
        dt_max=settings["dt_max"],
        atol=settings["atol"],
        rtol=settings["rtol"],
        order=settings["order"],
        precision=precision,
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


def generate_variants(
    system: Any,
    precision: type[np.floating[Any]],
    *,
    seed: int,
    count: int = 1,
) -> List[Tuple[ArrayFloat, ArrayFloat]]:
    """Return ``count`` diverse (state, parameter) pairs for the system."""

    rng = np.random.default_rng(seed)
    base_state = system.initial_values.values_array.astype(
        precision, copy=True
    )
    base_params = system.parameters.values_array.astype(precision, copy=True)
    sign_state = np.where(base_state == 0, 1.0, np.sign(base_state))
    sign_params = np.where(base_params == 0, 1.0, np.sign(base_params))
    variations: List[Tuple[ArrayFloat, ArrayFloat]] = []

    for idx in range(count):
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
) -> Dict[str, ArrayFloat]:
    """Create the inputs dictionary consumed by :func:`run_reference_loop`."""

    duration = float(solver_settings["duration"])
    dt_save = float(max(solver_settings["dt_save"], precision(1e-12)))
    samples = max(int(np.ceil(duration / dt_save)), 1)
    drivers = _driver_sequence(
        samples=samples,
        total_time=duration,
        n_drivers=system.num_drivers,
        precision=precision,
    )

    return {
        "initial_values": np.asarray(initial_state, dtype=precision).copy(),
        "parameters": np.asarray(parameters, dtype=precision).copy(),
        "drivers": np.asarray(drivers, dtype=precision).copy(),
    }


def run_reference_loop_with_history(
    evaluator: CPUODESystem,
    inputs: Mapping[str, Array],
    solver_settings: Mapping[str, Any],
    implicit_step_settings: Mapping[str, Any],
    controller: CPUAdaptiveController,
    output_functions: OutputFunctions,
    *,
    run_label: str,
) -> Tuple[Dict[str, Array], List[StepRecord]]:
    """Execute a CPU loop while recording proposed step sizes."""

    precision = evaluator.precision
    initial_state = inputs["initial_values"].astype(precision, copy=True)
    params = inputs["parameters"].astype(precision, copy=True)
    forcing_vectors = inputs["drivers"].astype(precision, copy=True)
    duration = precision(solver_settings["duration"])
    warmup = precision(solver_settings["warmup"])
    dt_save = precision(solver_settings["dt_save"])
    dt_summarise = precision(solver_settings["dt_summarise"])
    status_flags = 0
    zero = precision(0.0)
    step_records: List[StepRecord] = []

    step_fn = get_ref_step_fn(solver_settings["algorithm"])
    sampler = DriverSampler(forcing_vectors, dt_save, precision)

    saved_state_indices = np.asarray(
        solver_settings["saved_state_indices"], dtype=np.int32
    )
    saved_observable_indices = np.asarray(
        solver_settings["saved_observable_indices"], dtype=np.int32
    )
    summarised_state_indices = np.asarray(
        solver_settings["summarised_state_indices"], dtype=np.int32
    )
    summarised_observable_indices = np.asarray(
        solver_settings["summarised_observable_indices"], dtype=np.int32
    )

    save_time = output_functions.save_time
    max_save_samples = int(np.ceil(duration / dt_save))

    state = initial_state.copy()
    state_history = [state.copy()]
    observable_history: List[Array] = []
    time_history: List[float] = []
    t = precision(0.0)

    if warmup > zero:
        next_save_time = warmup
        save_index = 0
    else:
        next_save_time = dt_save
        save_index = 1
        state_history = [state.copy()]
        observables = evaluator.observables(
                state,
                params,
                sampler.sample(0),
                t,
        )
        observable_history.append(observables.copy())
        time_history.append(float(t))

    end_time = precision(warmup + duration)
    max_iters = implicit_step_settings["max_newton_iters"]
    timesave = 0
    timenow = 0
    while t < end_time - precision(1e-12):
        dt = precision(min(controller.dt, end_time - t))
        do_save = False
        if t + dt + precision(1e-10) >= next_save_time:
            dt = precision(next_save_time - t)
            do_save = True

        driver_sample = sampler.sample(save_index)
        drivers_now = driver_sample
        drivers_next = driver_sample
        timenow = perf_counter()

        attempt_dt = float(dt)
        result = step_fn(
            evaluator,
            state=state,
            params=params,
            drivers_now=drivers_now,
            drivers_next=drivers_next,
            dt=dt,
            tol=implicit_step_settings["nonlinear_tolerance"],
            max_iters=max_iters,
            time=float(t),
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
            print(f"Failed to propose step size at t={t:.2f}, step={attempt_dt:.6f}")
            break
        step_records.append(
            StepRecord(
                time=float(t),
                step_size=attempt_dt,
                accepted=accept,
                run_label=run_label,
            )
        )
        if not accept:
            continue

        state = result.state.copy()
        t += precision(dt)

        if do_save:
            print(f"Saving sample at t={t:.2f}, step={attempt_dt:.6f}")
            timesave = perf_counter()
            if len(state_history) < max_save_samples:
                state_history.append(result.state.copy())
                observable_history.append(result.observables.copy())
                time_history.append(float(next_save_time - warmup))
            next_save_time += dt_save
            save_index += 1

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
    """Create the controller-specific figure and return the saved path."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False, sharey=False)
    axes = axes.ravel()
    controller_label = controller.upper()

    for axis, system_name in zip(axes, SYSTEM_ORDER):
        records = histories.get(system_name, [])
        for record in records:
            marker = "o" if record.accepted else "x"
            axis.scatter(
                record.time,
                record.step_size,
                color="tab:blue" if record.accepted else "tab:red",
                alpha=0.7,
                marker=marker,
                s=18,
            )

        axis.set_title(SYSTEM_LABELS.get(system_name, system_name))
        axis.set_xlabel("Time")
        axis.set_ylabel("Step size")
        axis.grid(True, linestyle=":", linewidth=0.6)

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


def run_analysis(output_dir: Path, save_figs=True) -> List[Path]:
    """Execute the integrations for every controller/system combination."""

    saved_paths: List[Path] = []
    all_histories = {}
    for controller in ("i", "pi", "pid", "gustafsson"):
        controller_histories: Dict[str, List[StepRecord]] = {}

        for system_name, builder in SYSTEM_BUILDERS.items():
            print(f"Running analysis for {controller} controller on {system_name}")
            precision = np.float64 if system_name == "stiff" else np.float32
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
            histories: List[StepRecord] = []

            variants = generate_variants(
                system,
                precision,
                seed=hash((controller, system_name)) & 0xFFFF,
            )
            for variant_idx, variant in enumerate(variants):
                print(f"  Running variant {variant_idx+1}/{len(variants)}")
                initial_state, parameters = variant
                label = f"run{variant_idx+1}"
                inputs = generate_inputs(
                    solver_settings=solver_settings,
                    system=system,
                    precision=precision,
                    initial_state=initial_state,
                    parameters=parameters,
                )
                controller_instance = create_controller(
                    step_settings, precision
                )
                outputs, step_records = run_reference_loop_with_history(
                    evaluator=cpu_system,
                    inputs=inputs,
                    solver_settings=solver_settings,
                    implicit_step_settings=implicit_settings,
                    controller=controller_instance,
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


def main(
    save_figs=True, output_dir: str | Path = "docs/source/_static"
) -> (List)[Path]:
    """Run the analysis pipeline and return the generated figure paths."""

    output_path = Path(output_dir)
    return run_analysis(output_path, save_figs)


if __name__ == "__main__":
    generated, all_histories = main(save_figs=False)
    for controller, histories in all_histories.items():

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False, sharey=False)
        axes = axes.ravel()
        controller_label = controller.upper()

        for axis, system_name in zip(axes, SYSTEM_ORDER):
            axis.set_yscale("log")
            records = histories.get(system_name, [])
            for record in records:
                marker = "o" if record.accepted else "x"
                axis.scatter(
                    record.time,
                    record.step_size,
                    color="tab:blue" if record.accepted else "tab:red",
                    alpha=0.7,
                    marker=marker,
                    s=18,
                )

            axis.set_title(SYSTEM_LABELS.get(system_name, system_name))
            axis.set_xlabel("Time")
            axis.set_ylabel("Step size")

            axis.grid(True, linestyle=":", linewidth=0.6)

        fig.suptitle(
                f"Step-size history for {controller_label} controller", fontsize=14
        )
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        plt.show()
