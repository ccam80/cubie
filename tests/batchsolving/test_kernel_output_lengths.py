"""Tests for BatchSolverKernel output length properties with None-safe handling.

These tests verify that output_length, summaries_length, and warmup_length
properties correctly handle save_every=None (save_last mode) and
summarise_every=None (summarise_last mode).
"""
import numpy as np
import pytest


class TestOutputLengthNoneSafe:
    """Tests for output_length property with None-safe handling."""

    @pytest.mark.parametrize(
        "solver_settings_override",
        [{"save_every": None, "summarise_every": None}],
        indirect=True,
    )
    def test_output_length_with_save_every_none(self, solverkernel_mutable):
        """Verify output_length returns 2 when save_every is None."""
        kernel = solverkernel_mutable
        # When save_every is None, output_length should be 2 (initial + final)
        assert kernel.output_length == 2

    @pytest.mark.parametrize(
        "solver_settings_override",
        [{"save_every": 0.05, "duration": 0.2}],
        indirect=True,
    )
    def test_output_length_with_periodic_save(
        self, solverkernel_mutable, solver_settings
    ):
        """Verify output_length calculation with explicit save_every."""
        kernel = solverkernel_mutable
        # Set duration on kernel before accessing output_length
        kernel.duration = solver_settings["duration"]
        # With save_every=0.05 and duration=0.2:
        # floor(0.2 / 0.05) + 1 = 4 + 1 = 5
        assert kernel.output_length == 5


class TestSummariesLengthNoneSafe:
    """Tests for summaries_length property with None-safe handling."""

    @pytest.mark.parametrize(
        "solver_settings_override",
        [{"save_every": None, "summarise_every": None}],
        indirect=True,
    )
    def test_summaries_length_with_summarise_every_none(
        self, solverkernel_mutable
    ):
        """Verify summaries_length returns 2 when summarise_every is None."""
        kernel = solverkernel_mutable
        # When summarise_every is None, summaries_length should be 2
        assert kernel.summaries_length == 2

    @pytest.mark.parametrize(
        "solver_settings_override",
        [{"summarise_every": 0.05, "duration": 0.2}],
        indirect=True,
    )
    def test_summaries_length_with_periodic_summarise(
        self, solverkernel_mutable, solver_settings
    ):
        """Verify summaries_length calculation with explicit summarise_every."""
        kernel = solverkernel_mutable
        # Set duration on kernel before accessing summaries_length
        kernel.duration = solver_settings["duration"]
        # With summarise_every=0.05 and duration=0.2:
        # floor(0.2 / 0.05) = 4
        assert kernel.summaries_length == 4


class TestWarmupLengthNoneSafe:
    """Tests for warmup_length property with None-safe handling."""

    @pytest.mark.parametrize(
        "solver_settings_override",
        [{"save_every": None, "summarise_every": None, "warmup": 0.1}],
        indirect=True,
    )
    def test_warmup_length_with_save_every_none(self, solverkernel_mutable):
        """Verify warmup_length returns 0 when save_every is None."""
        kernel = solverkernel_mutable
        # When save_every is None, warmup_length should be 0
        assert kernel.warmup_length == 0


class TestTimingPropertyReturnTypes:
    """Tests for timing property return types."""

    @pytest.mark.parametrize(
        "solver_settings_override",
        [{"save_every": None, "summarise_every": None}],
        indirect=True,
    )
    def test_save_every_returns_none(self, solverkernel_mutable):
        """Verify save_every returns None when not configured."""
        kernel = solverkernel_mutable
        assert kernel.save_every is None

    @pytest.mark.parametrize(
        "solver_settings_override",
        [{"save_every": None, "summarise_every": None}],
        indirect=True,
    )
    def test_summarise_every_returns_none(self, solverkernel_mutable):
        """Verify summarise_every returns None when not configured."""
        kernel = solverkernel_mutable
        assert kernel.summarise_every is None

    @pytest.mark.parametrize(
        "solver_settings_override",
        [{"save_every": 0.05, "summarise_every": 0.05}],
        indirect=True,
    )
    def test_save_every_returns_float_when_set(self, solverkernel_mutable):
        """Verify save_every returns float when configured."""
        kernel = solverkernel_mutable
        assert kernel.save_every is not None
        assert isinstance(kernel.save_every, (float, np.floating))

    @pytest.mark.parametrize(
        "solver_settings_override",
        [{"save_every": 0.05, "summarise_every": 0.05}],
        indirect=True,
    )
    def test_summarise_every_returns_float_when_set(self, solverkernel_mutable):
        """Verify summarise_every returns float when configured."""
        kernel = solverkernel_mutable
        assert kernel.summarise_every is not None
        assert isinstance(kernel.summarise_every, (float, np.floating))
