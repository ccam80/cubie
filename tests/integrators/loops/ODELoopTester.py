"""Utility helpers for testing :mod:`cubie.integrators.loops.ode_loop`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Callable

import numpy as np
from numba import cuda, from_dtype
from numpy.testing import assert_allclose
from numpy.typing import NDArray
import pytest

from cubie.integrators.loops.ode_loop import IVPLoop
from cubie.outputhandling.output_functions import OutputFunctions
from cubie.outputhandling.output_sizes import OutputArrayHeights
from cubie.odesystems.baseODE import BaseODE

Array = NDArray[np.floating]


@dataclass
class LoopRunResult:
    """Container holding the outputs produced by a single loop execution."""

    state: Array
    observables: Array
    state_summaries: Array
    observable_summaries: Array
    status: int

def _driver_sequence(
    *,
    samples: int,
    total_time: float,
    n_drivers: int,
    precision,
) -> Array:
    """Drive system with a sine wave."""

    width = max(n_drivers, 1)
    drivers = np.zeros((samples, width), dtype=precision)
    if n_drivers > 0 and total_time > 0.0:
        times = np.linspace(0.0, total_time, samples, dtype=float)
        for idx in range(n_drivers):
            drivers[:, idx] = precision(
                1.0 + np.sin(2 * np.pi * (idx + 1) * times / total_time))
    return drivers


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
    dt_save = loop.dt_save
    warmup = solver_config['warmup']
    duration = solver_config["duration"]
    total_time = warmup + duration
    save_samples = int(np.ceil(precision(total_time) / precision(dt_save)))

    heights = OutputArrayHeights.from_output_fns(output_functions)

    state_width = max(heights.state, 1)
    observable_width = max(heights.observables, 1)
    state_summary_width = max(heights.state_summaries, 1)
    observable_summary_width = max(heights.observable_summaries, 1)

    state_output = np.zeros((save_samples, state_width), dtype=precision)
    observables_output = np.zeros(
        (save_samples, observable_width), dtype=precision
    )

    summarise_dt = loop.dt_summarise
    summary_samples = int(np.ceil(duration / summarise_dt))

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

    shared_elements = loop.shared_memory_elements
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
    state_output: Array,
    output_functions: OutputFunctions
) -> tuple[Array, Optional[Array]]:
    """Split state output into state variables and optional time column."""

    if not output_functions.save_time:
        return state_output, None
    n_state_columns = output_functions.n_saved_states
    state_values = state_output[:, :n_state_columns]
    time_values = state_output[:, n_state_columns : n_state_columns + 1]
    return state_values, time_values

def assert_integration_outputs(
    reference,
    device,
    output_functions,
    rtol: float,
    atol: float,
) -> None:
    """Compare state, summary, and time outputs between CPU and device."""

    flags = output_functions.compile_flags

    state_ref, time_ref = extract_state_and_time(
        reference["state"], output_functions
    )
    state_dev, time_dev = extract_state_and_time(
        device.state,
        output_functions,
    )
    observables_ref = reference["observables"]
    observables_dev = device.observables

    if flags.save_state:
        assert_allclose(state_dev,
                        state_ref,
                        rtol=rtol,
                        atol=atol,
                        err_msg="state")

    if output_functions.save_time:
        assert_allclose(
                time_dev,
                time_ref,
                rtol=rtol,
                atol=atol,
                err_msg="time")

    if flags.save_observables:
        assert_allclose(
                observables_dev,
                observables_ref,
                rtol=rtol,
                atol=atol,
                err_msg="observables")

    if flags.summarise_observables:
        assert_allclose(
            device.state_summaries,
            reference["state_summaries"],
            rtol=rtol,
            atol=atol,
            err_msg="observables",
        )

    if flags.summarise_observables:
        assert_allclose(
            device.observable_summaries,
            reference["observable_summaries"],
            rtol=rtol,
            atol=atol,
            err_msg="observables",
        )

def test_build(loop):
    assert isinstance(loop.device_function, Callable)

def test_getters(loop, step_controller, precision,
                 output_functions,
                 loop_buffer_sizes,
                 solver_settings,
                 step_object,):
    assert loop.is_adaptive == step_controller.is_adaptive
    assert loop.precision == precision
    assert loop.dt0 == step_controller.dt0
    assert loop.dt_min == step_controller.dt_min
    assert loop.dt_max == step_controller.dt_max
    assert loop.dt_save == precision(solver_settings['dt_save'])
    assert loop.dt_summarise == precision(solver_settings['dt_summarise'])
    assert (loop.local_memory_elements ==
            step_object.persistent_local_required +
            step_controller.local_memory_elements +
            loop_buffer_sizes.state + 3)
    assert loop.shared_memory_elements == (
            step_object.shared_memory_required +
            loop.buffer_indices.local_end)
    assert loop.buffer_indices is not None

@pytest.mark.parametrize('solver_settings_override, '
                         'step_controller_settings_override',
                         [({'algorithm':'euler'}, {'kind': 'fixed'}),
                           ({'algorithm':'crank_nicolson'}, {'kind': 'PID'})],
                         indirect=True, )
def test_update(step_controller, step_object, solver_settings, loop):
    if solver_settings['algorithm'].lower() == 'euler':
        updates = {'dt': 0.0001}
        loop.update(updates)
        assert step_controller.dt_min == pytest.approx(0.0001, rel=1e-6, abs=1e-6)
        assert step_object.dt == pytest.approx(0.0001, rel=1e-6, abs=1e-6)
    else:
        updates = {'dt_min': 0.0001,
                   'atol': 1e-12,
                   'ki': 2.0,
                   'max_newton_iters': 512}
        loop.update(updates)
        assert step_controller.dt_min == pytest.approx(0.0001, rel=1e-6, abs=1e-6)
        assert step_controller.atol == pytest.approx(1e-12, rel=1e-6, abs=1e-6)
        assert step_object.compile_settings.max_newton_iters == 512
        assert loop.dt_min == pytest.approx(0.0001, rel=1e-6, abs=1e-6)

def test_outputs_match_reference(
    loop,
    system,
    initial_state,
    solver_settings,
    cpu_reference_outputs,
    output_functions,
) -> None:
    device_output = run_device_loop(
        loop=loop,
        system=system,
        initial_state=initial_state,
        output_functions=output_functions,
        solver_config=solver_settings,
    )

    assert device_output.status == 0
    assert_integration_outputs(
            cpu_reference_outputs,
            device_output,
            output_functions,
            rtol=1e-5,
            atol=1e-6,
    )
