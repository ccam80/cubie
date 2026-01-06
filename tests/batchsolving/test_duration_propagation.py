"""Tests for duration propagation through the update chain.

These tests verify that duration set in BatchSolverKernel.run reaches
ODELoopConfig._duration, enabling samples_per_summary calculation when
summarise_last=True.
"""
import numpy as np
import pytest


class TestDurationPropagation:
    """Tests for duration propagation to loop config."""

    @pytest.mark.parametrize(
        "solver_settings_override",
        [{"save_every": None, "summarise_every": None, "duration": 1.0}],
        indirect=True,
    )
    def test_duration_propagates_to_loop_config(
        self, solverkernel_mutable, solver_settings
    ):
        """Verify that duration set in kernel.run reaches ODELoopConfig."""
        kernel = solverkernel_mutable
        duration = solver_settings["duration"]

        # Access the loop config before run
        loop_config = kernel.single_integrator._loop.compile_settings

        # Initially duration may be None
        initial_duration = loop_config._duration

        # Propagate duration through update
        kernel.single_integrator.update({"duration": duration}, silent=True)

        # Now loop config should have duration set
        assert loop_config._duration is not None
        assert loop_config._duration == duration

    @pytest.mark.parametrize(
        "solver_settings_override",
        [{"save_every": None, "summarise_every": None, "duration": 10.0}],
        indirect=True,
    )
    def test_samples_per_summary_uses_propagated_duration(
        self, solverkernel_mutable, solver_settings
    ):
        """Verify samples_per_summary calculates correctly with duration."""
        kernel = solverkernel_mutable
        duration = solver_settings["duration"]

        # Propagate duration to loop config
        kernel.single_integrator.update({"duration": duration}, silent=True)

        loop_config = kernel.single_integrator._loop.compile_settings

        # When summarise_last=True and summarise_every=None,
        # samples_per_summary should be duration/100
        if loop_config.summarise_last and loop_config._summarise_every is None:
            expected = max(1, int(duration / 100))
            assert loop_config.samples_per_summary == expected


class TestDurationUpdateChain:
    """Tests for duration flowing through the complete update chain."""

    @pytest.mark.parametrize(
        "solver_settings_override",
        [{"save_every": 0.05, "summarise_every": 0.05, "duration": 0.5}],
        indirect=True,
    )
    def test_duration_update_recognized(
        self, solverkernel_mutable, solver_settings
    ):
        """Verify duration is recognized as a valid update parameter."""
        kernel = solverkernel_mutable
        duration = solver_settings["duration"]

        # Update duration through single_integrator
        recognized = kernel.single_integrator.update(
            {"duration": duration}, silent=True
        )

        # Duration should be recognized
        assert "duration" in recognized

    @pytest.mark.parametrize(
        "solver_settings_override",
        [{"save_every": None, "summarise_every": None, "duration": 2.0}],
        indirect=True,
    )
    def test_duration_property_returns_value(
        self, solverkernel_mutable, solver_settings
    ):
        """Verify duration property on loop config returns the value."""
        kernel = solverkernel_mutable
        duration = solver_settings["duration"]

        # Propagate duration
        kernel.single_integrator.update({"duration": duration}, silent=True)

        loop_config = kernel.single_integrator._loop.compile_settings

        # The duration property should return precision-cast value
        assert loop_config.duration is not None
        # Allow for precision casting differences
        assert np.isclose(loop_config.duration, duration)
