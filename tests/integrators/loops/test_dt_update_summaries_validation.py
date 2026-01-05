"""Tests for timing parameter validation and None-handling logic."""

import pytest
import numpy as np
from cubie.integrators.loops.ode_loop_config import ODELoopConfig


def test_all_none_uses_defaults():
    """Test that all None sets save_last and summarise_last flags."""
    config = ODELoopConfig(
        n_states=3,
        n_parameters=0,
        n_drivers=0,
        n_observables=0,
        n_error=0,
        n_counters=0,
        state_summaries_buffer_height=0,
        observable_summaries_buffer_height=0,
    )
    
    # Sentinel values still set for loop timing calculations
    assert config.save_every == pytest.approx(0.1)
    assert config.summarise_every == pytest.approx(1.0)
    assert config.sample_summaries_every == pytest.approx(0.1)
    # But flags indicate end-of-run-only behavior
    assert config.save_last is True
    assert config.summarise_last is True


def test_only_save_every_specified():
    """Test that specifying only save_every infers other values."""
    config = ODELoopConfig(
        n_states=3,
        n_parameters=0,
        n_drivers=0,
        n_observables=0,
        n_error=0,
        n_counters=0,
        state_summaries_buffer_height=0,
        observable_summaries_buffer_height=0,
        save_every=0.2,
    )
    
    assert config.save_every == pytest.approx(0.2)
    assert config.summarise_every == pytest.approx(2.0)
    assert config.sample_summaries_every == pytest.approx(0.2)


def test_only_summarise_every_specified():
    """Test that specifying only summarise_every infers other values."""
    config = ODELoopConfig(
        n_states=3,
        n_parameters=0,
        n_drivers=0,
        n_observables=0,
        n_error=0,
        n_counters=0,
        state_summaries_buffer_height=0,
        observable_summaries_buffer_height=0,
        summarise_every=2.0,
    )
    
    assert config.save_every == pytest.approx(0.2)
    assert config.summarise_every == pytest.approx(2.0)
    assert config.sample_summaries_every == pytest.approx(0.2)


def test_save_and_summarise_specified():
    """Test that specifying save and summarise infers sample."""
    config = ODELoopConfig(
        n_states=3,
        n_parameters=0,
        n_drivers=0,
        n_observables=0,
        n_error=0,
        n_counters=0,
        state_summaries_buffer_height=0,
        observable_summaries_buffer_height=0,
        save_every=0.1,
        summarise_every=1.0,
    )
    
    assert config.save_every == pytest.approx(0.1)
    assert config.summarise_every == pytest.approx(1.0)
    assert config.sample_summaries_every == pytest.approx(0.1)


def test_all_three_specified():
    """Test that all three values can be explicitly set."""
    config = ODELoopConfig(
        n_states=3,
        n_parameters=0,
        n_drivers=0,
        n_observables=0,
        n_error=0,
        n_counters=0,
        state_summaries_buffer_height=0,
        observable_summaries_buffer_height=0,
        save_every=0.1,
        summarise_every=1.0,
        sample_summaries_every=0.2,
    )
    
    assert config.save_every == pytest.approx(0.1)
    assert config.summarise_every == pytest.approx(1.0)
    assert config.sample_summaries_every == pytest.approx(0.2)


def test_sample_must_divide_summarise():
    """Test that sample_summaries_every must divide summarise_every."""
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
            save_every=0.1,
            summarise_every=1.0,
            sample_summaries_every=0.3,
        )


def test_float32_tolerance():
    """Test that float32 uses appropriate tolerance for validation."""
    config = ODELoopConfig(
        n_states=3,
        n_parameters=0,
        n_drivers=0,
        n_observables=0,
        n_error=0,
        n_counters=0,
        state_summaries_buffer_height=0,
        observable_summaries_buffer_height=0,
        precision=np.float32,
        save_every=0.1,
        summarise_every=1.0,
        sample_summaries_every=0.1,
    )
    
    assert config.save_every == pytest.approx(0.1)
    assert config.samples_per_summary == 10



@pytest.mark.parametrize("sample_every", [0.1, 0.2, 0.25, 0.5, 1.0])
def test_valid_sample_summaries_every_values(sample_every):
    """Test various valid sample_summaries_every values."""
    config = ODELoopConfig(
        n_states=3,
        n_parameters=0,
        n_drivers=0,
        n_observables=0,
        n_error=0,
        n_counters=0,
        state_summaries_buffer_height=0,
        observable_summaries_buffer_height=0,
        save_every=0.1,
        summarise_every=1.0,
        sample_summaries_every=sample_every,
    )
    
    assert config.sample_summaries_every == pytest.approx(sample_every)
    expected_updates = int(1.0 / sample_every)
    assert config.samples_per_summary == expected_updates


def test_all_none_sets_save_last_flag():
    """Test that all None timing params sets save_last=True."""
    config = ODELoopConfig(
        n_states=3,
        n_parameters=0,
        n_drivers=0,
        n_observables=0,
        n_error=0,
        n_counters=0,
        state_summaries_buffer_height=0,
        observable_summaries_buffer_height=0,
    )
    assert config.save_last is True


def test_all_none_sets_summarise_last_flag():
    """Test that all None timing params sets summarise_last=True."""
    config = ODELoopConfig(
        n_states=3,
        n_parameters=0,
        n_drivers=0,
        n_observables=0,
        n_error=0,
        n_counters=0,
        state_summaries_buffer_height=0,
        observable_summaries_buffer_height=0,
    )
    assert config.summarise_last is True


def test_only_save_every_sets_summarise_last():
    """Test that specifying only save_every sets summarise_last=True."""
    config = ODELoopConfig(
        n_states=3,
        n_parameters=0,
        n_drivers=0,
        n_observables=0,
        n_error=0,
        n_counters=0,
        state_summaries_buffer_height=0,
        observable_summaries_buffer_height=0,
        save_every=0.2,
    )
    assert config.summarise_last is True
    assert config.save_last is False


def test_explicit_values_dont_set_flags():
    """Test that explicit timing values keep flags False."""
    config = ODELoopConfig(
        n_states=3,
        n_parameters=0,
        n_drivers=0,
        n_observables=0,
        n_error=0,
        n_counters=0,
        state_summaries_buffer_height=0,
        observable_summaries_buffer_height=0,
        save_every=0.1,
        summarise_every=1.0,
        sample_summaries_every=0.1,
    )
    assert config.save_last is False
    assert config.summarise_last is False
