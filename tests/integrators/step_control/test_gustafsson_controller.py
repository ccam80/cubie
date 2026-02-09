"""Tests for cubie.integrators.step_control.gustafsson_controller."""

from __future__ import annotations

import numpy as np
import pytest

from cubie.integrators.step_control.gustafsson_controller import (
    GustafssonStepControlConfig,
)
from tests._utils import _run_device_step


# ── GustafssonStepControlConfig construction ──────────────────────────────── #


def test_config_gamma_default():
    """gamma defaults to 0.9."""
    cfg = GustafssonStepControlConfig(precision=np.float64)
    assert cfg._gamma == pytest.approx(0.9)


def test_config_newton_max_iters_default():
    """newton_max_iters defaults to 20."""
    cfg = GustafssonStepControlConfig(precision=np.float64)
    assert cfg._newton_max_iters == 20


def test_config_gamma_property_wrapped():
    """gamma property returns precision(self._gamma)."""
    cfg = GustafssonStepControlConfig(precision=np.float32, gamma=0.85)
    result = cfg.gamma
    expected = np.float32(0.85)
    assert result == expected
    assert isinstance(result, np.float32)


def test_config_newton_max_iters_property_converts_to_int():
    """newton_max_iters property returns int(self._newton_max_iters)."""
    cfg = GustafssonStepControlConfig(
        precision=np.float64, newton_max_iters=15,
    )
    result = cfg.newton_max_iters
    assert result == 15
    assert isinstance(result, int)


def test_config_settings_dict_includes_gamma_and_newton_max_iters():
    """settings_dict extends super with gamma and newton_max_iters."""
    cfg = GustafssonStepControlConfig(
        precision=np.float64, gamma=0.8, newton_max_iters=10,
    )
    result = cfg.settings_dict
    assert "gamma" in result
    assert "newton_max_iters" in result
    assert result["gamma"] == pytest.approx(0.8)
    assert result["newton_max_iters"] == 10


# ── GustafssonController properties ───────────────────────────────────────── #


@pytest.mark.parametrize("solver_settings_override", [
    {
        "step_controller": "gustafsson",
        "algorithm": "crank_nicolson",
    },
], indirect=True)
@pytest.mark.parametrize(
    "prop, config_attr",
    [("gamma", "gamma"), ("newton_max_iters", "newton_max_iters")],
    ids=["gamma", "newton_max_iters"],
)
def test_properties_forward_to_compile_settings(
    step_controller, prop, config_attr,
):
    """gamma and newton_max_iters forward to compile_settings."""
    assert getattr(step_controller, prop) == getattr(
        step_controller.compile_settings, config_attr,
    )


@pytest.mark.parametrize("solver_settings_override", [
    {
        "step_controller": "gustafsson",
        "algorithm": "crank_nicolson",
    },
], indirect=True)
def test_local_memory_elements(step_controller):
    """Gustafsson controller requires 2 persistent local memory slots."""
    assert step_controller.local_memory_elements == 2


# ── Device unit tests ──────────────────────────────────────── #
# Items 141-153: exponent, gain_numerator, buffer allocation,
# buffer reads with floor, norm floor (1e-12), accept/reject,
# basic vs Gustafsson gain, fallback, clamping, deadband,
# buffer updates, return codes.


@pytest.mark.parametrize("solver_settings_override", [
    pytest.param(
        {
            "step_controller": "gustafsson", "atol": 1e-3,
            "rtol": 0.0, "algorithm": "crank_nicolson",
        },
        id="atol_only",
    ),
    pytest.param(
        {
            "step_controller": "gustafsson", "atol": 1e-4,
            "rtol": 1e-3, "algorithm": "crank_nicolson",
        },
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
         "local_mem": np.array([0.004, 0.3])},
        id="accept_with_dt_prev",
    ),
    pytest.param(
        {"dt0": 0.005, "error": np.full(3, 5e-3),
         "local_mem": np.array([0.004, 0.3])},
        id="reject_with_dt_prev",
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
    """Device Gustafsson controller output matches CPU reference."""
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
    {
        "step_controller": "gustafsson", "atol": 1e-3,
        "rtol": 0.0, "algorithm": "crank_nicolson",
    },
], indirect=True)
@pytest.mark.parametrize("step_setup", [
    pytest.param(
        {"dt0": 0.005, "error": np.full(3, 5e-4), "niters": 1},
        id="niters_1",
    ),
    pytest.param(
        {"dt0": 0.005, "error": np.full(3, 5e-4), "niters": 3},
        id="niters_3",
    ),
    pytest.param(
        {"dt0": 0.005, "error": np.full(3, 5e-4), "niters": 10},
        id="niters_10",
    ),
], indirect=True)
def test_niters_modulates_gain(
    device_step_results, cpu_step_results, step_controller,
    precision,
):
    """Varying niters changes Gustafsson gain; device matches CPU."""
    assert device_step_results.dt == pytest.approx(
        cpu_step_results.dt, rel=1e-6,
    )
    assert device_step_results.accepted == cpu_step_results.accepted


@pytest.mark.parametrize("solver_settings_override", [
    {
        "step_controller": "gustafsson", "atol": 1e-3,
        "rtol": 0.0, "algorithm": "crank_nicolson",
    },
], indirect=True)
@pytest.mark.parametrize("step_setup", [
    {"dt0": 0.005, "error": np.full(3, 5e-4)},
], indirect=True)
def test_return_code_normal(device_step_results):
    """Return code 0 under normal conditions."""
    assert device_step_results.return_code == 0


@pytest.mark.parametrize("solver_settings_override", [
    {
        "step_controller": "gustafsson", "atol": 1e-3,
        "rtol": 0.0, "dt_min": 0.1, "dt_max": 1.0, "dt": 0.5,
        "algorithm": "crank_nicolson",
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
        "step_controller": "gustafsson", "atol": 1e-3,
        "rtol": 0.0, "deadband_min": 0.8, "deadband_max": 1.2,
        "algorithm": "crank_nicolson",
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
    {
        "step_controller": "gustafsson", "atol": 1e-3,
        "rtol": 0.0, "algorithm": "crank_nicolson",
    },
], indirect=True)
def test_multi_step_sequence_with_niters_variation(
    step_controller, cpu_step_controller, precision, system,
):
    """Gustafsson dt and accept/reject agree over a 5-step sequence.

    Niters varies per step to exercise the gain_numerator / (niters +
    2 * newton_max_iters) formula across different iteration counts.
    """
    n = system.sizes.states
    local_mem = np.zeros(
        step_controller.local_memory_elements, dtype=precision,
    )
    base = system.initial_values.values_array.astype(precision)

    errors = [5e-4, 1.4e-3, 6e-4, 1.6e-3, 5e-4]
    deltas = [2e-2, 1.5e-2, 1e-2, 2.5e-2, 1.5e-2]
    niters_seq = [1, 3, 2, 5, 1]

    dt_dev = precision(step_controller.dt)
    dt_cpu = precision(step_controller.dt)
    state = base.copy()

    for i, (err_val, delta_val, nit) in enumerate(
        zip(errors, deltas, niters_seq)
    ):
        prev = state.copy()
        state_new = prev + precision(delta_val)
        err = np.full(n, precision(err_val), dtype=precision)

        cpu_step_controller.dt = dt_cpu
        accept_cpu = cpu_step_controller.propose_dt(
            prev_state=prev, new_state=state_new,
            error_vector=err, niters=nit,
        )
        dt_cpu = precision(cpu_step_controller.dt)

        result = _run_device_step(
            step_controller.device_function, precision, dt_dev,
            err, state=state_new, state_prev=prev,
            local_mem=local_mem, niters=nit,
        )
        local_mem = result.local_mem.copy()
        dt_dev = result.dt

        assert result.accepted == int(accept_cpu), (
            f"step {i}: accept mismatch (niters={nit})"
        )
        assert dt_dev == pytest.approx(dt_cpu, rel=1e-6), (
            f"step {i}: dt mismatch (niters={nit})"
        )
        if accept_cpu:
            state = state_new
