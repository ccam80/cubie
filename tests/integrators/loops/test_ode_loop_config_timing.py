"""Tests for ODELoopConfig timing parameter handling.

These tests verify the duration field, duration property, and the enhanced
samples_per_summary property that provides fallback behavior when
summarise_last is True and duration is available.
"""
import numpy as np
import pytest

from cubie.integrators.loops.ode_loop_config import ODELoopConfig


class TestDurationProperty:
    """Tests for the duration property on ODELoopConfig."""

    def test_duration_property_returns_precision_cast_value(self):
        """Verify duration property returns precision-cast value."""
        config = ODELoopConfig(
            save_every=0.1,
            summarise_every=1.0,
            sample_summaries_every=0.1,
            duration=10.5,
            precision=np.float32,
        )
        result = config.duration
        assert result == np.float32(10.5)
        assert isinstance(result, np.float32)

    def test_duration_property_returns_none_when_not_set(self):
        """Verify duration property returns None when not configured."""
        config = ODELoopConfig(
            save_every=0.1,
            summarise_every=1.0,
            sample_summaries_every=0.1,
            precision=np.float32,
        )
        assert config.duration is None


class TestSamplesPerSummaryWithDuration:
    """Tests for samples_per_summary fallback with duration."""

    def test_samples_per_summary_with_summarise_last_and_duration(self):
        """Verify samples_per_summary returns duration/100 when
        summarise_last=True and duration is set.
        """
        # Create config where all timing params are None to trigger
        # save_last=True and summarise_last=True
        config = ODELoopConfig(
            save_every=None,
            summarise_every=None,
            sample_summaries_every=None,
            duration=500.0,
            precision=np.float32,
        )
        # With duration=500.0, samples_per_summary should be
        # max(1, int(500.0 / 100)) = 5
        assert config.summarise_last is True
        assert config.samples_per_summary == 5

    def test_samples_per_summary_minimum_is_one(self):
        """Verify samples_per_summary returns at least 1 for short durations."""
        config = ODELoopConfig(
            save_every=None,
            summarise_every=None,
            sample_summaries_every=None,
            duration=50.0,  # 50/100 = 0.5, should round to at least 1
            precision=np.float32,
        )
        assert config.summarise_last is True
        assert config.samples_per_summary == 1

    def test_samples_per_summary_without_duration_returns_one(self):
        """Verify samples_per_summary returns 1 as safe default when
        summarise_last=True but duration is not set.
        """
        config = ODELoopConfig(
            save_every=None,
            summarise_every=None,
            sample_summaries_every=None,
            precision=np.float32,
        )
        assert config.summarise_last is True
        assert config._duration is None
        assert config.samples_per_summary == 1

    def test_samples_per_summary_with_periodic_summarise_every(self):
        """Verify samples_per_summary uses periodic calculation when
        summarise_every is set.
        """
        config = ODELoopConfig(
            save_every=0.1,
            summarise_every=1.0,
            sample_summaries_every=0.1,
            duration=10.0,
            precision=np.float32,
        )
        # With summarise_every=1.0 and sample_summaries_every=0.1,
        # samples_per_summary should be round(1.0 / 0.1) = 10
        assert config.samples_per_summary == 10


class TestDurationValidation:
    """Tests for duration validation."""

    def test_duration_rejects_negative_value(self):
        """Verify duration rejects negative values."""
        with pytest.raises((ValueError, TypeError)):
            ODELoopConfig(
                save_every=0.1,
                summarise_every=1.0,
                sample_summaries_every=0.1,
                duration=-1.0,
                precision=np.float32,
            )

    def test_duration_rejects_zero_value(self):
        """Verify duration rejects zero value."""
        with pytest.raises((ValueError, TypeError)):
            ODELoopConfig(
                save_every=0.1,
                summarise_every=1.0,
                sample_summaries_every=0.1,
                duration=0.0,
                precision=np.float32,
            )

    def test_duration_accepts_positive_value(self):
        """Verify duration accepts positive values."""
        config = ODELoopConfig(
            save_every=0.1,
            summarise_every=1.0,
            sample_summaries_every=0.1,
            duration=100.0,
            precision=np.float32,
        )
        assert config._duration == 100.0


class TestTimingResetMechanism:
    """Tests for timing parameter reset mechanism."""

    def test_reset_timing_inference_clears_flags(self):
        """Verify reset_timing_inference resets save_last and summarise_last
        before re-inferring.
        """
        # Create config with all timing params None (sets flags to True)
        config = ODELoopConfig(
            save_every=None,
            summarise_every=None,
            sample_summaries_every=None,
            precision=np.float32,
        )
        assert config.save_last is True
        assert config.summarise_last is True

        # Now set timing params to actual values
        config._save_every = 0.1
        config._summarise_every = 1.0
        config._sample_summaries_every = 0.1

        # Call reset_timing_inference - should reset flags and re-infer
        config.reset_timing_inference()

        # Flags should be False because timing params are now set
        assert config.save_last is False
        assert config.summarise_last is False

    def test_timing_parameter_reset_on_none(self):
        """Verify that setting timing params to None after a previous value
        recalculates derived values.
        """
        # Create config with explicit values
        config = ODELoopConfig(
            save_every=0.1,
            summarise_every=1.0,
            sample_summaries_every=0.1,
            precision=np.float32,
        )
        assert config.save_last is False
        assert config.summarise_last is False

        # Reset all timing params to None
        config._save_every = None
        config._summarise_every = None
        config._sample_summaries_every = None

        # Call reset to re-infer
        config.reset_timing_inference()

        # Should now be in save_last/summarise_last mode
        assert config.save_last is True
        assert config.summarise_last is True

    def test_reset_handles_partial_none_values(self):
        """Verify reset handles cases where only some params are None."""
        # Start with all None
        config = ODELoopConfig(
            save_every=None,
            summarise_every=None,
            sample_summaries_every=None,
            precision=np.float32,
        )
        assert config.summarise_last is True

        # Set save_every only
        config._save_every = 0.5
        config._summarise_every = None
        config._sample_summaries_every = None
        config.reset_timing_inference()

        # With only save_every set, summarise_last should be True
        # and summarise_every/sample_summaries_every should be inferred
        assert config.save_last is False
        assert config.summarise_last is True
        assert config._summarise_every == 5.0  # 10 * save_every
        assert config._sample_summaries_every == 0.5  # same as save_every


class TestIVPLoopTimingReset:
    """Tests for IVPLoop.update timing reset behavior."""

    def test_update_with_none_timing_triggers_reset(self, loop_mutable):
        """Verify IVPLoop.update with timing param=None triggers
        reset_timing_inference.
        """
        loop = loop_mutable
        # Get initial state - loop has explicit timing values
        initial_save_every = loop.compile_settings._save_every
        initial_save_last = loop.compile_settings.save_last

        # Verify we start with explicit values (not None)
        assert initial_save_every is not None
        assert initial_save_last is False

        # Update with all timing params set to None
        loop.update({
            'save_every': None,
            'summarise_every': None,
            'sample_summaries_every': None,
        })

        # After reset, should be in save_last/summarise_last mode
        assert loop.compile_settings.save_last is True
        assert loop.compile_settings.summarise_last is True

    def test_update_without_none_timing_preserves_flags(self, loop_mutable):
        """Verify IVPLoop.update without None timing params does not
        trigger reset_timing_inference.
        """
        loop = loop_mutable
        # Get initial state
        initial_save_last = loop.compile_settings.save_last
        initial_summarise_last = loop.compile_settings.summarise_last

        # Update with non-timing parameter (no None timing params)
        loop.update({'dt0': 0.001})

        # Flags should be preserved
        assert loop.compile_settings.save_last == initial_save_last
        assert loop.compile_settings.summarise_last == initial_summarise_last

    def test_update_with_explicit_timing_value_no_reset(self, loop_mutable):
        """Verify IVPLoop.update with explicit timing value does not
        trigger reset when key is present but value is not None.
        """
        loop = loop_mutable
        initial_save_last = loop.compile_settings.save_last

        # Update save_every with explicit value (not None)
        loop.update({'save_every': 0.5})

        # Should not trigger reset, flags unchanged
        assert loop.compile_settings.save_last == initial_save_last
        assert loop.compile_settings._save_every == 0.5
