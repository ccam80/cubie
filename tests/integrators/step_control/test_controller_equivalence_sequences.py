"""Regression tests for controller parity between CPU and device paths."""

import numpy as np
import pytest
from numba import cuda

from cubie.integrators.step_control.adaptive_PI_controller import (
    AdaptivePIController,
)
from cubie.integrators.step_control.adaptive_PID_controller import (
    AdaptivePIDController,
)
from cubie.integrators.step_control.gustafsson_controller import (
    GustafssonController,
)


class ControllerTrace:
    """Capture outputs from sequential controller executions."""

    def __init__(self):
        self.accepted = []
        self.dt = []
        self.local_memory = []


class StepResult:
    """Lightweight return container mirroring GPU kernel outputs."""

    def __init__(self, dt, accepted, local_mem):
        self.dt = dt
        self.accepted = accepted
        self.local_mem = local_mem


def _run_device_step(
    device_func,
    precision,
    dt0,
    error,
    *,
    local_mem=None,
    state=None,
    state_prev=None,
    niters=1,
):
    """Execute a controller device function once."""

    err = np.asarray(error, dtype=precision)
    state_arr = (
        np.asarray(state, dtype=precision)
        if state is not None
        else np.zeros_like(err)
    )
    state_prev_arr = (
        np.asarray(state_prev, dtype=precision)
        if state_prev is not None
        else np.zeros_like(err)
    )

    dt = np.asarray([dt0], dtype=precision)
    accept = np.zeros(1, dtype=np.int32)
    if local_mem is not None:
        temp = np.asarray(local_mem, dtype=precision)
    else:
        temp = np.empty(0, dtype=precision)
    niters_val = np.int32(niters)

    @cuda.jit
    def kernel(
        dt_val,
        state_val,
        state_prev_val,
        err_val,
        niters_val,
        accept_val,
        temp_val,
    ):
        device_func(
            dt_val,
            state_val,
            state_prev_val,
            err_val,
            niters_val,
            accept_val,
            temp_val,
        )

    kernel[1, 1](dt, state_arr, state_prev_arr, err, niters_val, accept, temp)
    return StepResult(precision(dt[0]), int(accept[0]), temp.copy())


def _sequence_inputs(
    *,
    n_states,
    precision,
    base_state,
    low_error,
    high_error,
    n_steps,
):
    """Generate deterministic state and error sequences for tests."""

    states_prev = []
    states_new = []
    errors = []
    niters = []
    for idx in range(n_steps):
        offset = precision(0.05 * idx)
        delta = precision(0.01 + 0.002 * idx)
        state_prev = base_state + precision(offset)
        state_new = state_prev + precision(delta)
        if idx % 2 == 0:
            err_value = precision(low_error)
        else:
            err_value = precision(high_error)
        error_vec = np.full(n_states, err_value, dtype=precision)
        states_prev.append(state_prev)
        states_new.append(state_new)
        errors.append(error_vec)
        niters.append(1 + (idx % 3))
    return states_prev, states_new, errors, niters


@pytest.mark.parametrize(
    "solver_settings_override",
    [
        {"step_controller": "i"},
        {"step_controller": "pi"},
        {"step_controller": "pid"},
        {"step_controller": "gustafsson"},
    ],
    ids=("i", "pi", "pid", "gustafsson"),
    indirect=True,
)
@pytest.mark.parametrize(
    "step_controller_settings_override",
    [
        {
            "dt_min": 1e-4,
            "dt_max": 0.2,
            "kp": 1 / 12,
            "ki": 1 / 8,
            "kd": 1 / 10,
            "deadband_min": 0.95,
            "deadband_max": 1.05,
            "atol": 1e-3,
            "rtol": 0.0,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("system_override", ["three_chamber"], indirect=True)
class TestControllerEquivalence:
    """Step controller regression tests for CPU and device implementations."""

    def test_sequential_acceptance_matches(
        self,
        step_controller,
        cpu_step_controller,
        precision,
        system,
    ):
        """Check that dt updates and acceptance match through rejections."""

        dtype = precision
        n_states = system.sizes.states
        local_mem = np.zeros(
            step_controller.local_memory_elements,
            dtype=dtype,
        )
        base_state = system.initial_values.values_array.astype(dtype)
        low_error = dtype(1e-5)
        high_error = dtype(5.0)
        sequence = _sequence_inputs(
            n_states=n_states,
            precision=dtype,
            base_state=base_state,
            low_error=low_error,
            high_error=high_error,
            n_steps=6,
        )
        states_prev, states_new, errors, niters = sequence
        gpu_trace = ControllerTrace()
        cpu_trace = ControllerTrace()

        current_dt_device = dtype(step_controller.dt0)
        current_dt_cpu = dtype(step_controller.dt0)
        cpu_step_controller.dt = dtype(current_dt_cpu)

        for idx, (prev_state, new_state, err_vec, niter) in enumerate(
            zip(states_prev, states_new, errors, niters)
        ):
            cpu_step_controller.dt = dtype(current_dt_cpu)
            accept_cpu = cpu_step_controller.propose_dt(
                error_vector=err_vec,
                prev_state=prev_state,
                new_state=new_state,
                niters=niter,
            )
            result_cpu = StepResult(
                precision(cpu_step_controller.dt),
                int(accept_cpu),
                np.array([], dtype=dtype),
            )
            cpu_trace.dt.append(result_cpu.dt)
            cpu_trace.accepted.append(result_cpu.accepted)
            if step_controller.local_memory_elements:
                if isinstance(step_controller, AdaptivePIController):
                    mem_cpu = np.array(
                        [cpu_step_controller._prev_nrm2],
                        dtype=dtype,
                    )
                elif isinstance(step_controller, AdaptivePIDController):
                    mem_cpu = np.array(
                        [
                            cpu_step_controller._prev_nrm2,
                            cpu_step_controller._prev_prev_nrm2,
                        ],
                        dtype=dtype,
                    )
                elif isinstance(step_controller, GustafssonController):
                    prev_norm = max(
                        cpu_step_controller._prev_nrm2,
                        dtype(1e-4),
                    )
                    mem_cpu = np.array(
                        [
                            cpu_step_controller.dt,
                            prev_norm,
                        ],
                        dtype=dtype,
                    )
                else:
                    mem_cpu = np.zeros(0, dtype=dtype)
            else:
                mem_cpu = np.zeros(0, dtype=dtype)
            cpu_trace.local_memory.append(mem_cpu)

            device_result = _run_device_step(
                step_controller.device_function,
                dtype,
                dtype(current_dt_device),
                err_vec,
                state=new_state,
                state_prev=prev_state,
                local_mem=local_mem,
                niters=niter,
            )
            gpu_trace.dt.append(device_result.dt)
            gpu_trace.accepted.append(device_result.accepted)
            gpu_trace.local_memory.append(device_result.local_mem.copy())
            if step_controller.local_memory_elements:
                local_mem = device_result.local_mem.copy()
            current_dt_device = device_result.dt
            current_dt_cpu = result_cpu.dt

        assert gpu_trace.accepted == cpu_trace.accepted
        assert np.allclose(
            np.array(gpu_trace.dt),
            np.array(cpu_trace.dt),
            rtol=1e-7,
            atol=1e-7,
        )
        if step_controller.local_memory_elements:
            for gpu_mem, cpu_mem in zip(
                gpu_trace.local_memory,
                cpu_trace.local_memory,
            ):
                np.testing.assert_allclose(
                    gpu_mem[: cpu_mem.size],
                    cpu_mem,
                    rtol=1e-7,
                    atol=1e-7,
                )

    def test_rejection_retains_previous_state(
        self,
        step_controller,
        cpu_step_controller,
        precision,
        system,
    ):
        """Ensure both controllers agree on rejection bookkeeping."""

        dtype = precision
        n_states = system.sizes.states
        prev_state = system.initial_values.values_array.astype(dtype)
        new_state = prev_state + dtype(0.02)
        error_vec = np.full(n_states, dtype(10.0), dtype=dtype)
        local_mem = np.zeros(
            step_controller.local_memory_elements,
            dtype=dtype,
        )

        cpu_step_controller.dt = dtype(step_controller.dt0)
        accept_cpu = cpu_step_controller.propose_dt(
            error_vector=error_vec,
            prev_state=prev_state,
            new_state=new_state,
            niters=3,
        )
        device_result = _run_device_step(
            step_controller.device_function,
            dtype,
            dtype(step_controller.dt0),
            error_vec,
            state=new_state,
            state_prev=prev_state,
            local_mem=local_mem,
            niters=3,
        )

        assert int(accept_cpu) == device_result.accepted
        assert device_result.dt == pytest.approx(
            dtype(cpu_step_controller.dt),
            rel=1e-7,
            abs=1e-7,
        )
