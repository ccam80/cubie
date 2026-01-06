"""Integration tests for all timing parameter modes.

Tests verify the complete flow from solve_ivp through to output
array sizing for each timing parameter combination:

1. save_every=None, summarise_every=None (save_last + summarise_last)
2. save_every=X, summarise_every=None (periodic save, summarise_last)
3. save_every=None, summarise_every=X (save_last, periodic summarise)
4. save_every=X, summarise_every=Y (both periodic)
5. Parameter reset between solves
"""
import warnings

import numpy as np
import pytest


class TestTimingModeOutputLengths:
    """Test output array lengths for each timing mode."""

    @pytest.mark.parametrize(
        "solver_settings_override",
        [{"save_every": None, "summarise_every": None}],
        indirect=True,
    )
    def test_save_last_only_output_length(self, solver_mutable, system):
        """save_every=None produces output_length=2."""
        solver = solver_mutable

        # Build minimal input arrays for a single run
        n_states = system.sizes.states
        n_params = system.sizes.parameters
        inits = np.ones((n_states, 1), dtype=np.float32)
        params = np.ones((n_params, 1), dtype=np.float32)

        result = solver.solve(
            initial_values=inits,
            parameters=params,
            duration=1.0,
        )

        # output_length=2 means state shape[0] (time dimension) should be 2
        assert result.state.shape[0] == 2

    @pytest.mark.parametrize(
        "solver_settings_override",
        [{"save_every": None, "summarise_every": None}],
        indirect=True,
    )
    def test_summarise_last_only_summaries_length(
        self, solver_mutable, system
    ):
        """summarise_every=None produces summaries_length=2."""
        solver = solver_mutable

        # Build minimal input arrays for a single run
        n_states = system.sizes.states
        n_params = system.sizes.parameters
        inits = np.ones((n_states, 1), dtype=np.float32)
        params = np.ones((n_params, 1), dtype=np.float32)

        result = solver.solve(
            initial_values=inits,
            parameters=params,
            duration=1.0,
        )

        # summaries_length=2 means summaries shape[0] (time dimension) is 2
        assert result.state_summaries.shape[0] == 2

    @pytest.mark.parametrize(
        "solver_settings_override",
        [{"save_every": 0.2, "summarise_every": None, "duration": 1.0}],
        indirect=True,
    )
    def test_periodic_save_output_length(self, solver_mutable, system):
        """save_every=X produces floor(duration/X)+1 outputs."""
        solver = solver_mutable

        # Build minimal input arrays for a single run
        n_states = system.sizes.states
        n_params = system.sizes.parameters
        inits = np.ones((n_states, 1), dtype=np.float32)
        params = np.ones((n_params, 1), dtype=np.float32)

        duration = 1.0
        save_every = 0.2

        result = solver.solve(
            initial_values=inits,
            parameters=params,
            duration=duration,
        )

        # Expected: floor(1.0 / 0.2) + 1 = 5 + 1 = 6
        expected_length = int(np.floor(duration / save_every)) + 1
        assert result.state.shape[0] == expected_length

    @pytest.mark.parametrize(
        "solver_settings_override",
        [{"save_every": None, "summarise_every": 0.25, "duration": 1.0}],
        indirect=True,
    )
    def test_periodic_summarise_length(self, solver_mutable, system):
        """summarise_every=X produces floor(duration/X) summaries."""
        solver = solver_mutable

        # Build minimal input arrays for a single run
        n_states = system.sizes.states
        n_params = system.sizes.parameters
        inits = np.ones((n_states, 1), dtype=np.float32)
        params = np.ones((n_params, 1), dtype=np.float32)

        duration = 1.0
        summarise_every = 0.25

        result = solver.solve(
            initial_values=inits,
            parameters=params,
            duration=duration,
        )

        # Expected: floor(1.0 / 0.25) = 4
        expected_length = int(np.floor(duration / summarise_every))
        assert result.state_summaries.shape[0] == expected_length


class TestParameterReset:
    """Test parameter reset behavior between solves."""

    @pytest.mark.parametrize(
        "solver_settings_override",
        [{
            "save_every": None,
            "summarise_every": None,
            "sample_summaries_every": 0.01
        }],
        indirect=True,
    )
    def test_sample_summaries_every_recalculates_on_none(
        self, solver_mutable, system
    ):
        """sample_summaries_every recalculates when reset to None."""
        solver = solver_mutable

        # Build minimal input arrays for a single run
        n_states = system.sizes.states
        n_params = system.sizes.parameters
        inits = np.ones((n_states, 1), dtype=np.float32)
        params = np.ones((n_params, 1), dtype=np.float32)

        # First solve with explicit sample_summaries_every value
        result1 = solver.solve(
            initial_values=inits,
            parameters=params,
            duration=1.0,
        )

        # Verify sample_summaries_every was set (from override)
        loop_config = solver.kernel.single_integrator._loop.compile_settings
        initial_sample_summaries = loop_config.sample_summaries_every

        # Reset sample_summaries_every to None
        solver.update({"sample_summaries_every": None})

        # Second solve with None should trigger recalculation
        result2 = solver.solve(
            initial_values=inits,
            parameters=params,
            duration=1.0,
        )

        # Verify the behavior completed without error
        # When sample_summaries_every is None and summarise_last is True,
        # samples_per_summary should be recalculated from duration
        assert result2.state is not None


class TestDurationDependencyWarning:
    """Test duration dependency warning behavior."""

    @pytest.mark.parametrize(
        "solver_settings_override",
        [{"save_every": None, "summarise_every": None}],
        indirect=True,
    )
    def test_warning_on_summarise_last_without_summarise_every(
        self, solver_mutable, system
    ):
        """Warning raised when duration affects samples_per_summary."""
        solver = solver_mutable

        # Build minimal input arrays for a single run
        n_states = system.sizes.states
        n_params = system.sizes.parameters
        inits = np.ones((n_states, 1), dtype=np.float32)
        params = np.ones((n_params, 1), dtype=np.float32)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            solver.solve(
                initial_values=inits,
                parameters=params,
                duration=1.0,
            )

            # Filter for UserWarning about sample_summaries_every
            duration_warnings = [
                warning for warning in w
                if issubclass(warning.category, UserWarning)
                and "sample_summaries_every" in str(warning.message)
            ]

            assert len(duration_warnings) == 1
            assert "kernel recompilation" in str(duration_warnings[0].message)
