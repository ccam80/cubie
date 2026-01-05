"""Tests for timing parameter validation and None-handling logic."""

import pytest
import numpy as np
from cubie.integrators.loops.ode_loop_config import ODELoopConfig
from cubie.outputhandling.output_config import OutputCompileFlags


def test_all_none_uses_defaults():
    """Test that all None uses default values."""
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
    
    assert config.save_every == pytest.approx(0.1)
    assert config.summarise_every == pytest.approx(1.0)
    assert config.sample_summaries_every == pytest.approx(0.1)


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
    assert config.updates_per_summary == 10


def test_backward_compat_dt_save():
    """Test backward compatibility with dt_save parameter."""
    with pytest.warns(DeprecationWarning, match="dt_save.*deprecated"):
        config = ODELoopConfig(
            n_states=3,
            n_parameters=0,
            n_drivers=0,
            n_observables=0,
            n_error=0,
            n_counters=0,
            state_summaries_buffer_height=0,
            observable_summaries_buffer_height=0,
            dt_save=0.2,
        )
    
    assert config.save_every == pytest.approx(0.2)
    assert config.dt_save == pytest.approx(0.2)


def test_backward_compat_dt_summarise():
    """Test backward compatibility with dt_summarise parameter."""
    with pytest.warns(DeprecationWarning, match="dt_summarise.*deprecated"):
        config = ODELoopConfig(
            n_states=3,
            n_parameters=0,
            n_drivers=0,
            n_observables=0,
            n_error=0,
            n_counters=0,
            state_summaries_buffer_height=0,
            observable_summaries_buffer_height=0,
            dt_summarise=2.0,
        )
    
    assert config.summarise_every == pytest.approx(2.0)
    assert config.dt_summarise == pytest.approx(2.0)


def test_backward_compat_dt_update_summaries():
    """Test backward compatibility with dt_update_summaries parameter."""
    with pytest.warns(DeprecationWarning, match="dt_update_summaries.*deprecated"):
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
            dt_update_summaries=0.2,
        )
    
    assert config.sample_summaries_every == pytest.approx(0.2)
    assert config.dt_update_summaries == pytest.approx(0.2)


def test_cannot_specify_both_dt_save_and_save_every():
    """Test that using both old and new names raises an error."""
    with pytest.raises(ValueError, match="Cannot specify both"):
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
            save_every=0.1,
        )


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
    assert config.updates_per_summary == expected_updates


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


def test_deprecated_params_removed():
    """Test that dt_save, dt_summarise, dt_update_summaries are not valid."""
    with pytest.raises(TypeError, match="unexpected keyword argument"):
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
        )
