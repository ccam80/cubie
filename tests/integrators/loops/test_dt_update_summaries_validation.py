"""Tests for dt_update_summaries parameter validation."""

import pytest
import numpy as np
from cubie.integrators.loops.ode_loop_config import ODELoopConfig
from cubie.outputhandling.output_config import OutputCompileFlags


def test_dt_update_summaries_default_to_dt_save():
    """Test that dt_update_summaries defaults to dt_save."""
    config = ODELoopConfig(
        n_states=3,
        n_parameters=0,
        n_drivers=0,
        n_observables=0,
        n_error=0,
        n_counters=0,
        state_summaries_buffer_height=0,
        observable_summaries_buffer_height=0,
        dt_save=0.1,
        dt_summarise=1.0,
    )
    
    assert config.dt_update_summaries == pytest.approx(0.1)
    assert config._dt_update_summaries == pytest.approx(0.1)


def test_dt_update_summaries_explicit_value():
    """Test that explicit dt_update_summaries is used."""
    config = ODELoopConfig(
        n_states=3,
        n_parameters=0,
        n_drivers=0,
        n_observables=0,
        n_error=0,
        n_counters=0,
        state_summaries_buffer_height=0,
        observable_summaries_buffer_height=0,
        dt_save=0.1,
        dt_summarise=1.0,
        dt_update_summaries=0.2,
    )
    
    assert config.dt_update_summaries == pytest.approx(0.2)


def test_dt_update_summaries_must_divide_dt_summarise():
    """Test that dt_update_summaries must divide dt_summarise."""
    with pytest.raises(ValueError, match="must be an integer divisor"):
        ODELoopConfig(
            n_states=3,
            n_parameters=0,
            n_drivers=0,
            n_observables=0,
            n_error=0,
            n_counters=0,
            state_summaries_buffer_height=0,
            observable_summaries_buffer_height=0,
            dt_save=0.1,
            dt_summarise=1.0,
            dt_update_summaries=0.3,
        )


def test_dt_update_summaries_positive():
    """Test that dt_update_summaries must be positive."""
    with pytest.raises((ValueError, TypeError)):
        ODELoopConfig(
            n_states=3,
            n_parameters=0,
            n_drivers=0,
            n_observables=0,
            n_error=0,
            n_counters=0,
            state_summaries_buffer_height=0,
            observable_summaries_buffer_height=0,
            dt_save=0.1,
            dt_summarise=1.0,
            dt_update_summaries=-0.1,
        )


@pytest.mark.parametrize("dt_update", [0.1, 0.2, 0.25, 0.5, 1.0])
def test_valid_dt_update_summaries_values(dt_update):
    """Test various valid dt_update_summaries values."""
    config = ODELoopConfig(
        n_states=3,
        n_parameters=0,
        n_drivers=0,
        n_observables=0,
        n_error=0,
        n_counters=0,
        state_summaries_buffer_height=0,
        observable_summaries_buffer_height=0,
        dt_save=0.1,
        dt_summarise=1.0,
        dt_update_summaries=dt_update,
    )
    
    assert config.dt_update_summaries == pytest.approx(dt_update)
    expected_updates = int(1.0 / dt_update)
    assert config.updates_per_summary == expected_updates
