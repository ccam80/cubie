"""Utility helpers for testing :mod:`cubie.integrators.loops.ode_loop`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from numba import cuda, from_dtype
from numpy.typing import NDArray

from cubie.integrators.loops.ode_loop import IVPLoop
from cubie.outputhandling.output_functions import OutputFunctions
from cubie.outputhandling.output_sizes import OutputArrayHeights
from cubie.odesystems.baseODE import BaseODE
from tests.integrators.cpu_reference import run_reference_loop


Array = NDArray[np.floating]


@dataclass
class LoopRunResult:
    """Container holding the outputs produced by a single loop execution."""

    state: Array
    observables: Array
    state_summaries: Array
    observable_summaries: Array
    status: int

    def trimmed_to(self, reference: Mapping[str, Array]) -> "LoopRunResult":
        """Return a copy sliced to the shape of the reference arrays."""

        def _trim(array: Array, target: Array) -> Array:
            rows = target.shape[0]
            cols = target.shape[1] if target.ndim > 1 else array.shape[1]
            return array[:rows, :cols]

        return LoopRunResult(
            state=_trim(self.state, reference["state"]),
            observables=_trim(self.observables, reference["observables"]),
            state_summaries=_trim(
                self.state_summaries, reference["state_summaries"]
            ),
            observable_summaries=_trim(
                self.observable_summaries,
                reference["observable_summaries"],
            ),
            status=self.status,
        )


def build_solver_config(
    solver_settings: Mapping[str, float],
) -> dict[str, float]:
    """Normalise solver settings for use with the CPU reference helpers."""

    keys = [
        "dt_min",
        "dt_max",
        "dt_save",
        "dt_summarise",
        "warmup",
        "duration",
        "atol",
        "rtol",
    ]
    config = {
        name: float(solver_settings[name])
        for name in keys
    }
    return config


def _driver_sequence(
    *,
    samples: int,
    total_time: float,
    n_drivers: int,
    precision: np.dtype,
) -> Array:
    """Generate a deterministic driver sequence for device testing."""

    width = max(n_drivers, 1)
    drivers = np.zeros((samples, width), dtype=precision)
    if n_drivers > 0 and total_time > 0.0:
        times = np.linspace(0.0, total_time, samples, dtype=float)
        for idx in range(n_drivers):
            drivers[:, idx] = precision(
                0.5
                * (
                    1.0
                    + np.sin(2 * np.pi * (idx + 1) * times / total_time)
                )
            )
    return drivers


def build_reference_inputs(
    system: BaseODE,
    initial_state: Array,
    solver_config: Mapping[str, float],
) -> dict[str, Array]:
    """Construct the inputs expected by :func:`run_reference_loop`."""

    precision = system.precision
    dt_min = solver_config["dt_min"]
    warmup = solver_config.get("warmup", 0.0)
    duration = solver_config["duration"]
    total_time = max(warmup + duration, dt_min)
    samples = max(int(np.ceil(total_time / dt_min)), 1)
    n_drivers = system.num_drivers
    forcing = np.zeros((n_drivers, samples), dtype=precision)
    if n_drivers > 0 and total_time > 0.0:
        times = np.linspace(0.0, total_time, samples, dtype=float)
        for idx in range(n_drivers):
            forcing[idx, :] = precision(
                0.5
                * (
                    1.0
                    + np.sin(2 * np.pi * (idx + 1) * times / total_time)
                )
            )

    return {
        "initial_values": np.array(initial_state, dtype=precision, copy=True),
        "parameters": np.array(
            system.parameters.values_array, dtype=precision, copy=True
        ),
        "forcing_vectors": forcing,
    }


def build_reference_controller_settings(
    controller_settings: Mapping[str, float | int | Sequence[float]],
    solver_config: Mapping[str, float],
) -> dict[str, float | int | str]:
    """Translate GPU controller settings to the CPU helper format."""

    dt_initial = controller_settings.get(
        "dt", controller_settings.get("dt_min", solver_config["dt_min"])
    )
    dt_max = controller_settings.get("dt_max", solver_config["dt_max"])
    return {
        "kind": str(controller_settings.get("kind", "fixed")),
        "dt": float(dt_initial),
        "dt_max": float(dt_max),
        "order": int(controller_settings.get("order", 1)),
    }


def cpu_reference_outputs(
    *,
    system: BaseODE,
    initial_state: Array,
    solver_config: Mapping[str, float],
    loop_compile_settings: Mapping[str, float | Sequence[int] | Sequence[str]],
    output_functions: OutputFunctions,
    stepper: str,
    controller_settings: Mapping[str, float | int | str],
) -> dict[str, Array]:
    """Execute the CPU reference loop with the provided configuration."""

    inputs = build_reference_inputs(system, initial_state, solver_config)
    return run_reference_loop(
        system=system,
        inputs=inputs,
        solver_settings=solver_config,
        loop_compile_settings=loop_compile_settings,
        output_functions=output_functions,
        stepper=stepper,
        step_controller_settings=controller_settings,
    )


def run_device_loop(
    *,
    loop: IVPLoop,
    system: BaseODE,
    initial_state: Array,
    output_functions: OutputFunctions,
    solver_config: Mapping[str, float],
) -> LoopRunResult:
    """Execute ``loop`` on the CUDA simulator and return host-side outputs."""

    precision = loop.precision
    dt_save = float(loop.dt_save)
    warmup = float(solver_config.get("warmup", 0.0))
    duration = float(solver_config["duration"])
    total_time = max(warmup + duration, dt_save)
    save_samples = max(int(np.ceil(total_time / dt_save)) + 1, 2)

    heights = OutputArrayHeights.from_output_fns(output_functions)
    state_width = max(heights.state, 1)
    observable_width = max(heights.observables, 1)
    state_summary_width = max(heights.state_summaries, 1)
    observable_summary_width = max(heights.observable_summaries, 1)

    state_output = np.zeros((save_samples, state_width), dtype=precision)
    observables_output = np.zeros(
        (save_samples, observable_width), dtype=precision
    )

    summarise_dt = max(float(loop.dt_summarise), 1e-12)
    summary_samples = max(int(np.ceil(duration / summarise_dt)) + 1, 1)

    state_summary_output = np.zeros(
        (summary_samples, state_summary_width), dtype=precision
    )
    observable_summary_output = np.zeros(
        (summary_samples, observable_summary_width), dtype=precision
    )

    params = np.array(
        system.parameters.values_array,
        dtype=precision,
        copy=True,
    )
    drivers = _driver_sequence(
        samples=save_samples,
        total_time=total_time,
        n_drivers=system.num_drivers,
        precision=precision,
    )

    init_state = np.array(initial_state, dtype=precision, copy=True)
    status = np.zeros(1, dtype=np.int32)

    d_init = cuda.to_device(init_state)
    d_params = cuda.to_device(params)
    d_drivers = cuda.to_device(drivers)
    d_state_out = cuda.to_device(state_output)
    d_obs_out = cuda.to_device(observables_output)
    d_state_sum = cuda.to_device(state_summary_output)
    d_obs_sum = cuda.to_device(observable_summary_output)
    d_status = cuda.to_device(status)

    base_shared = loop.buffer_indices.local_end
    algo_shared = getattr(loop.algorithm, "shared_memory_required", 0)
    controller_shared = getattr(
        loop.step_controller,
        "shared_memory_required",
        0,
    )
    shared_elements = base_shared + algo_shared + controller_shared
    shared_bytes = np.dtype(precision).itemsize * shared_elements

    local_req = max(1, loop.local_memory_elements)

    loop_fn = loop.device_function
    numba_precision = from_dtype(precision)

    @cuda.jit
    def kernel(
        init_vec,
        params_vec,
        drivers_vec,
        state_out_arr,
        obs_out_arr,
        state_sum_arr,
        obs_sum_arr,
        status_arr,
    ):
        idx = cuda.grid(1)
        if idx > 0:
            return

        shared = cuda.shared.array(0, dtype=numba_precision)
        local = cuda.local.array(local_req, dtype=numba_precision)
        status_arr[0] = loop_fn(
            init_vec,
            params_vec,
            drivers_vec,
            shared,
            local,
            state_out_arr,
            obs_out_arr,
            state_sum_arr,
            obs_sum_arr,
            precision(duration),
            precision(warmup),
            precision(0.0),
        )

    kernel[1, 1, 0, shared_bytes](
        d_init,
        d_params,
        d_drivers,
        d_state_out,
        d_obs_out,
        d_state_sum,
        d_obs_sum,
        d_status,
    )
    cuda.synchronize()

    state_host = d_state_out.copy_to_host()
    observables_host = d_obs_out.copy_to_host()
    state_summary_host = d_state_sum.copy_to_host()
    observable_summary_host = d_obs_sum.copy_to_host()
    status_value = int(d_status.copy_to_host()[0])

    return LoopRunResult(
        state=state_host,
        observables=observables_host,
        state_summaries=state_summary_host,
        observable_summaries=observable_summary_host,
        status=status_value,
    )


def extract_state_and_time(
    state_output: Array, output_functions: OutputFunctions
) -> tuple[Array, Array | None]:
    """Split state output into state variables and optional time column."""

    n_state_columns = output_functions.n_saved_states
    if not output_functions.save_time:
        return state_output[:, :n_state_columns], None
    state_values = state_output[:, :n_state_columns]
    time_values = state_output[:, n_state_columns : n_state_columns + 1]
    return state_values, time_values

