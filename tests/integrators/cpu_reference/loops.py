"""CPU reference loop implementations for integrator tests."""

from typing import Mapping, Optional, Sequence, Union

import numpy as np

from cubie.integrators.algorithms.base_algorithm_step import ButcherTableau

from .algorithms import get_ref_stepper
from .cpu_ode_system import CPUODESystem
from .cpu_utils import Array, DriverEvaluator, STATUS_MASK, _ensure_array
from .step_controllers import CPUAdaptiveController

from tests._utils import calculate_expected_summaries


def _collect_saved_outputs(
    save_history: list[Array],
    output_length: int,
    indices: Sequence[int],
    dtype: np.dtype,
) -> Array:
    """Return the saved samples at ``indices`` as a dense array.

    Parameters
    ----------
    save_history
        Sequence of saved state or observable samples.
    indices
        Column indices to extract from each saved sample.
    dtype
        Target dtype for the dense output array.

    Returns
    -------
    Array
        Dense array containing the selected samples.
    """

    width = len(indices)
    if width == 0:
        return np.zeros((output_length, 0), dtype=dtype)
    data = np.zeros((output_length, width), dtype=dtype)
    for row, sample in enumerate(save_history):
        data[row, :width] = sample[indices]
    return data


def run_reference_loop(
    evaluator: CPUODESystem,
    inputs: Mapping[str, Array],
    driver_evaluator: DriverEvaluator,
    solver_settings,
    output_functions,
    controller: CPUAdaptiveController,
    *,
    tableau: Optional[Union[str, ButcherTableau]] = None,
) -> Mapping[str, Array]:
    """Execute a CPU loop mirroring :class:`IVPLoop` behaviour.

    Parameters
    ----------
    evaluator
        CPU-side ODE evaluator used for state updates.
    inputs
        Mapping providing initial conditions and parameters.
    driver_evaluator
        Callable returning driver samples for a requested time.
    solver_settings
        Settings dictionary describing the integration problem.
    output_functions
        Output configuration mirroring the device loop behaviour.
    controller
        Adaptive or fixed step controller used during the run.
    tableau
        Optional tableau override supplied as a name or instance.

    Returns
    -------
    Mapping[str, Array]
        Dictionary containing state, observable, summary, and status arrays.
    """

    precision = evaluator.precision

    initial_state = inputs["initial_values"].astype(precision, copy=True)
    params = inputs["parameters"].astype(precision, copy=True)
    duration = np.float64(solver_settings["duration"])
    warmup = np.float64(solver_settings["warmup"])
    dt_save = precision(solver_settings["dt_save"])
    dt_summarise = precision(solver_settings["dt_summarise"])

    stepper = get_ref_stepper(
        evaluator,
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

    saved_state_indices = _ensure_array(
        solver_settings["saved_state_indices"], np.int32
    )
    saved_observable_indices = _ensure_array(
        solver_settings["saved_observable_indices"], np.int32
    )
    summarised_state_indices = _ensure_array(
        solver_settings["summarised_state_indices"], np.int32
    )
    summarised_observable_indices = _ensure_array(
        solver_settings["summarised_observable_indices"], np.int32
    )

    save_time = output_functions.save_time
    max_save_samples = int(np.ceil(duration / dt_save))

    state = initial_state.copy()
    state_history = []
    observable_history = []
    time_history = []
    t = np.float64(0.0)
    drivers_initial = driver_evaluator(precision(t))
    observables = evaluator.observables(
        state,
        params,
        drivers_initial,
        precision(t),
    )

    if warmup > np.float64(0.0):
        next_save_time = np.float64(warmup)
    else:
        next_save_time = np.float64(warmup + dt_save)
        state_history = [state.copy()]
        observable_history.append(observables.copy())
        time_history = [t]

    end_time = np.float64(warmup + duration)
    fixed_steps_per_save = int(np.ceil(dt_save / controller.dt))
    fixed_step_count = 0
    equality_breaker = (
        precision(1e-7)
        if precision is np.float32
        else precision(1e-14)
    )
    status_flags = 0
    save_idx = 0

    while save_idx < max_save_samples:
        dt = controller.dt
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
        if not accept:
            continue

        state = result.state.copy()
        observables = result.observables.copy()
        t = t + precision(dt)

        if do_save:
            if len(state_history) < max_save_samples:
                state_history.append(result.state.copy())
                observable_history.append(result.observables.copy())
                time_history.append(precision(t - warmup))
            next_save_time = next_save_time + dt_save
            save_idx += 1

    state_output = _collect_saved_outputs(
        state_history,
        max_save_samples,
        saved_state_indices,
        precision,
    )

    observables_output = _collect_saved_outputs(
        observable_history,
        max_save_samples,
        saved_observable_indices,
        precision,
    )
    if save_time:
        state_output = np.column_stack((state_output, np.asarray(time_history)))

    summarise_every = int(np.ceil(dt_summarise / dt_save))

    state_summary, observable_summary = calculate_expected_summaries(
        state_output,
        observables_output,
        summarised_state_indices,
        summarised_observable_indices,
        summarise_every,
        output_functions.compile_settings.output_types,
        output_functions.summaries_output_height_per_var,
        precision,
        dt_save=dt_save,
    )
    final_status = status_flags & STATUS_MASK

    return {
        "state": state_output,
        "observables": observables_output,
        "state_summaries": state_summary,
        "observable_summaries": observable_summary,
        "status": final_status,
    }


__all__ = ["run_reference_loop", "_collect_saved_outputs"]
