"""Tests for cubie.integrators.step_control.adaptive_PID_controller."""

from __future__ import annotations

import numpy as np
import pytest

from cubie.integrators.step_control.adaptive_PID_controller import (
    PIDStepControlConfig,
)
from tests._utils import _run_device_step


# ── PIDStepControlConfig ────────────────────────────────────── #


def test_kd_defaults_to_zero():
    """PIDStepControlConfig._kd defaults to 0.0."""
    config = PIDStepControlConfig(precision=np.float64)
    assert config._kd == 0.0


def test_kd_validated_as_float():
    """PIDStepControlConfig._kd validated as float."""
    config = PIDStepControlConfig(precision=np.float64, kd=0.05)
    assert config._kd == 0.05
    config2 = PIDStepControlConfig(precision=np.float64, kd=1.5)
    assert config2._kd == 1.5


def test_kd_property_returns_precision_cast():
    """PIDStepControlConfig.kd returns precision(self._kd)."""
    config = PIDStepControlConfig(precision=np.float32, kd=0.05)
    expected = np.float32(0.05)
    assert config.kd == expected
    assert type(config.kd) == type(expected)


# ── AdaptivePIDController ──────────────────────────────────── #


@pytest.mark.parametrize("solver_settings_override", [
    {"step_controller": "pid", "kd": 0.05},
], indirect=True)
def test_init_creates_pid_config(step_controller, precision):
    """Controller compile_settings is a PIDStepControlConfig."""
    assert isinstance(step_controller.compile_settings, PIDStepControlConfig)
    assert step_controller.compile_settings.kd == precision(0.05)


@pytest.mark.parametrize("solver_settings_override", [
    {"step_controller": "pid", "kp": 0.7, "ki": -0.4, "kd": 0.05},
], indirect=True)
@pytest.mark.parametrize(
    "prop, config_attr",
    [("kp", "kp"), ("ki", "ki"), ("kd", "kd")],
    ids=["kp", "ki", "kd"],
)
def test_gain_properties_forward_to_compile_settings(
    step_controller, prop, config_attr,
):
    """Gain properties (kp, ki, kd) forward to compile_settings."""
    assert getattr(step_controller, prop) == getattr(
        step_controller.compile_settings, config_attr,
    )


@pytest.mark.parametrize("solver_settings_override", [
    {"step_controller": "pid"},
], indirect=True)
def test_local_memory_elements_returns_two(step_controller):
    """PID controller requires 2 persistent local memory slots."""
    assert step_controller.local_memory_elements == 2


@pytest.mark.parametrize("solver_settings_override", [
    {"step_controller": "pid", "kp": 0.7, "ki": -0.4, "kd": 0.05},
], indirect=True)
def test_settings_dict_extends_super_with_gains(step_controller):
    """settings_dict extends parent with kp, ki, kd."""
    d = step_controller.settings_dict
    assert d["kp"] == step_controller.kp
    assert d["ki"] == step_controller.ki
    assert d["kd"] == step_controller.kd
    assert "n" in d
    assert "dt_min" in d


# ── Device unit tests ──────────────────────────────────────── #
# Items 123-131: exponent computation, buffer access, fallback
# logic, PID gain, clamping, deadband, buffer updates, return codes.


@pytest.mark.parametrize("solver_settings_override", [
    pytest.param(
        {"step_controller": "pid", "atol": 1e-3, "rtol": 0.0},
        id="atol_only",
    ),
    pytest.param(
        {"step_controller": "pid", "atol": 1e-4, "rtol": 1e-3},
        id="atol_plus_rtol",
    ),
], indirect=True)
@pytest.mark.parametrize("step_setup", [
    pytest.param(
        {"dt0": 0.005, "error": np.full(3, 5e-4)},
        id="accept_fresh",
    ),
    pytest.param(
        {"dt0": 0.005, "error": np.full(3, 5e-3)},
        id="reject_fresh",
    ),
    pytest.param(
        {"dt0": 0.005, "error": np.full(3, 5e-4),
         "local_mem": np.array([0.005, 0.8])},
        id="accept_with_history",
    ),
    pytest.param(
        {"dt0": 0.005, "error": np.full(3, 5e-3),
         "local_mem": np.array([0.005, 0.8])},
        id="reject_with_history",
    ),
    pytest.param(
        {"dt0": 0.005, "error": np.full(3, 1e-10)},
        id="gain_clamped_at_max",
    ),
    pytest.param(
        {"dt0": 0.005, "error": np.full(3, 1e2)},
        id="gain_clamped_at_min",
    ),
], indirect=True)
def test_device_output_matches_cpu(
    device_step_results, cpu_step_results, step_controller,
    precision,
):
    """Device PID-controller output matches CPU reference."""
    assert device_step_results.dt == pytest.approx(
        cpu_step_results.dt, rel=1e-6,
    )
    assert device_step_results.accepted == cpu_step_results.accepted
    valid = step_controller.local_memory_elements
    if valid:
        np.testing.assert_allclose(
            device_step_results.local_mem[:valid],
            cpu_step_results.local_mem[:valid],
            rtol=1e-6,
        )


@pytest.mark.parametrize("solver_settings_override", [
    {"step_controller": "pid", "atol": 1e-3, "rtol": 0.0},
], indirect=True)
@pytest.mark.parametrize("step_setup", [
    {"dt0": 0.005, "error": np.full(3, 5e-4)},
], indirect=True)
def test_return_code_normal(device_step_results):
    """Return code 0 under normal conditions."""
    assert device_step_results.return_code == 0


@pytest.mark.parametrize("solver_settings_override", [
    {
        "step_controller": "pid", "atol": 1e-3, "rtol": 0.0,
        "dt_min": 0.1, "dt_max": 1.0, "dt": 0.5,
    },
], indirect=True)
@pytest.mark.parametrize("step_setup", [
    {"dt0": 0.005, "error": np.full(3, 5e-3)},
], indirect=True)
def test_return_code_at_dt_min(device_step_results):
    """Return code 8 when proposed dt falls to dt_min."""
    assert device_step_results.return_code == 8


@pytest.mark.parametrize("solver_settings_override", [
    {
        "step_controller": "pid", "atol": 1e-3, "rtol": 0.0,
        "deadband_min": 0.8, "deadband_max": 1.2,
    },
], indirect=True)
@pytest.mark.parametrize("step_setup", [
    {"dt0": 0.005, "error": np.full(3, 7e-4)},
], indirect=True)
def test_deadband_matches_cpu(
    device_step_results, cpu_step_results, precision,
):
    """Deadband gain clamping matches CPU reference."""
    assert device_step_results.dt == pytest.approx(
        cpu_step_results.dt, rel=1e-6,
    )
    assert device_step_results.accepted == cpu_step_results.accepted


# ── Multi-step sequence ────────────────────────────────────── #


@pytest.mark.parametrize("solver_settings_override", [
    {"step_controller": "pid", "atol": 1e-3, "rtol": 0.0},
], indirect=True)
def test_multi_step_sequence_matches_cpu(
    step_controller, cpu_step_controller, precision, system,
):
    """PID dt and accept/reject agree over a 5-step sequence."""
    n = system.sizes.states
    local_mem = np.zeros(
        step_controller.local_memory_elements, dtype=precision,
    )
    base = system.initial_values.values_array.astype(precision)

    errors = [5e-4, 1.4e-3, 6e-4, 1.6e-3, 5e-4]
    deltas = [2e-2, 1.5e-2, 1e-2, 2.5e-2, 1.5e-2]

    dt_dev = precision(step_controller.dt)
    dt_cpu = precision(step_controller.dt)
    state = base.copy()

    for i, (err_val, delta_val) in enumerate(zip(errors, deltas)):
        prev = state.copy()
        state_new = prev + precision(delta_val)
        err = np.full(n, precision(err_val), dtype=precision)

        cpu_step_controller.dt = dt_cpu
        accept_cpu = cpu_step_controller.propose_dt(
            prev_state=prev, new_state=state_new,
            error_vector=err, niters=1,
        )
        dt_cpu = precision(cpu_step_controller.dt)

        result = _run_device_step(
            step_controller.device_function, precision, dt_dev,
            err, state=state_new, state_prev=prev,
            local_mem=local_mem, niters=1,
        )
        local_mem = result.local_mem.copy()
        dt_dev = result.dt

        assert result.accepted == int(accept_cpu), (
            f"step {i}: accept mismatch"
        )
        assert dt_dev == pytest.approx(dt_cpu, rel=1e-6), (
            f"step {i}: dt mismatch"
        )
        if accept_cpu:
            state = state_new
