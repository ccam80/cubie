"""Tests for cubie.integrators.step_control.adaptive_I_controller."""

from __future__ import annotations

import numpy as np
import pytest

from tests._utils import _run_device_step


# ── Initialization ──────────────────────────────────────────── #


@pytest.mark.parametrize("solver_settings_override", [
    {"step_controller": "i"},
], indirect=True)
def test_init_creates_config(
    step_controller, step_controller_settings,
):
    """Controller compile_settings reflect step_controller_settings."""
    cs = step_controller.compile_settings
    assert cs.algorithm_order == step_controller_settings["algorithm_order"]
    assert cs.dt_min == step_controller_settings["dt_min"]
    assert cs.dt_max == step_controller_settings["dt_max"]
    assert cs.n == step_controller_settings["n"]


# ── Properties ──────────────────────────────────────────────── #


@pytest.mark.parametrize("solver_settings_override", [
    {"step_controller": "i"},
], indirect=True)
def test_local_memory_elements_returns_zero(step_controller):
    """I controller requires no persistent local memory."""
    assert step_controller.local_memory_elements == 0


# ── Device unit tests ──────────────────────────────────────── #
# Items 82-92: norm, gain, accept/reject, clamping, deadband,
# dt update, return codes.


@pytest.mark.parametrize("solver_settings_override", [
    pytest.param(
        {"step_controller": "i", "atol": 1e-3, "rtol": 0.0},
        id="atol_only",
    ),
    pytest.param(
        {"step_controller": "i", "atol": 1e-4, "rtol": 1e-3},
        id="atol_plus_rtol",
    ),
], indirect=True)
@pytest.mark.parametrize("step_setup", [
    pytest.param(
        {"dt0": 0.005, "error": np.full(3, 5e-4)},
        id="accept_low_err",
    ),
    pytest.param(
        {"dt0": 0.005, "error": np.full(3, 5e-3)},
        id="reject_high_err",
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
    device_step_results, cpu_step_results, precision,
):
    """Device I-controller dt and accept/reject match CPU."""
    assert device_step_results.dt == pytest.approx(
        cpu_step_results.dt, rel=1e-6,
    )
    assert device_step_results.accepted == cpu_step_results.accepted


@pytest.mark.parametrize("solver_settings_override", [
    {"step_controller": "i", "atol": 1e-3, "rtol": 0.0},
], indirect=True)
@pytest.mark.parametrize("step_setup", [
    pytest.param(
        {"dt0": 0.005, "error": np.full(3, 5e-4)},
        id="normal",
    ),
], indirect=True)
def test_return_code_normal(device_step_results):
    """Return code 0 under normal conditions."""
    assert device_step_results.return_code == 0


@pytest.mark.parametrize("solver_settings_override", [
    {
        "step_controller": "i", "atol": 1e-3, "rtol": 0.0,
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
        "step_controller": "i", "atol": 1e-3, "rtol": 0.0,
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
    {"step_controller": "i", "atol": 1e-3, "rtol": 0.0},
], indirect=True)
def test_multi_step_sequence_matches_cpu(
    step_controller, cpu_step_controller, precision, system,
):
    """I-controller dt and accept/reject agree over a 5-step sequence."""
    n = system.sizes.states
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
            niters=1,
        )
        dt_dev = result.dt

        assert result.accepted == int(accept_cpu), (
            f"step {i}: accept mismatch"
        )
        assert dt_dev == pytest.approx(dt_cpu, rel=1e-6), (
            f"step {i}: dt mismatch"
        )
        if accept_cpu:
            state = state_new
