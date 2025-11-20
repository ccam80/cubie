"""Tests for float64 time accumulation fix."""
import numpy as np
import pytest
from cubie import solve_ivp
from tests.system_fixtures import three_state_linear


@pytest.mark.parametrize("precision", [np.float32])
def test_float32_small_timestep_accumulation(three_state_linear, precision):
    """Verify time accumulates correctly with float32 precision and small dt."""
    system = three_state_linear
    system.precision = precision
    
    # Use very small dt_min that would fail with float32 time accumulation
    result = solve_ivp(
        system,
        duration=1.0,
        t0=0.0,
        algorithm="explicit_euler",
        dt=1e-8,
        precision=precision,
    )
    
    # Verify integration completed
    assert result.status_codes[0] == 0
    assert result.states.shape[0] > 0
    
    # Verify time increased throughout integration
    if result.times is not None:
        time_diffs = np.diff(result.times)
        assert np.all(time_diffs > 0), "Time should increase monotonically"


@pytest.mark.parametrize("precision", [np.float32, np.float64])
def test_long_integration_with_small_steps(three_state_linear, precision):
    """Verify long integrations with small steps complete correctly."""
    system = three_state_linear
    system.precision = precision
    
    # Long duration with small dt_min
    result = solve_ivp(
        system,
        duration=10.0,
        t0=1e10,  # Large t0 to stress test precision
        algorithm="explicit_euler",
        dt=1e-6,
        precision=precision,
    )
    
    # Verify integration completed
    assert result.status_codes[0] == 0
    assert result.states.shape[0] > 0


@pytest.mark.parametrize("precision", [np.float32])
def test_adaptive_controller_with_float32(three_state_linear, precision):
    """Verify adaptive controllers work with float32 and small dt_min."""
    system = three_state_linear
    system.precision = precision
    
    result = solve_ivp(
        system,
        duration=1.0,
        t0=0.0,
        algorithm="explicit_euler",
        step_controller="adaptive_PI",
        dt0=1e-3,
        dt_min=1e-9,  # Very small minimum step
        dt_max=1e-2,
        precision=precision,
    )
    
    # Verify integration completed without hanging
    assert result.status_codes[0] == 0
    assert result.states.shape[0] > 0
