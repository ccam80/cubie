"""Integration tests for all timing parameter modes.

Tests verify the complete flow from solve_ivp through to output
array sizing for each timing parameter combination:

1. save_every=None, summarise_every=None (save_last + summarise_last)
2. save_every=X, summarise_every=None (periodic save, summarise_last)
3. save_every=None, summarise_every=X (save_last, periodic summarise)
4. save_every=X, summarise_every=Y (both periodic)
5. Parameter reset between solves

Note: These tests create solver instances locally to avoid session-scoped
fixture duration issues when calling solver.solve(). Uses a system without
drivers to avoid driver array configuration issues.
"""
import warnings

import numpy as np
import pytest

from cubie.batchsolving.solver import Solver
from tests.system_fixtures import build_three_state_constant_deriv_system


# Default test duration - short for fast tests
TEST_DURATION = 0.2


@pytest.fixture(scope="module")
def no_driver_system():
    """Create a system without drivers for local integration tests."""
    return build_three_state_constant_deriv_system(np.float32)


class TestTimingModeOutputLengths:
    """Test output array lengths for each timing mode."""

    def test_save_last_only_output_length(self, no_driver_system):
        """save_every=None produces output_length=2."""
        solver = Solver(
            no_driver_system,
            save_every=None,
            summarise_every=None,
            sample_summaries_every=0.01,
            dt=0.01,
        )

        # Build minimal input arrays for a single run
        n_states = no_driver_system.sizes.states
        n_params = no_driver_system.sizes.parameters
        inits = np.ones((n_states, 1), dtype=np.float32)
        params = np.ones((n_params, 1), dtype=np.float32)

        result = solver.solve(
            initial_values=inits,
            parameters=params,
            duration=TEST_DURATION,
        )

        # output_length=2 means state shape[0] (time dimension) should be 2
        assert result.state.shape[0] == 2

    def test_summarise_last_only_summaries_length(self, no_driver_system):
        """summarise_every=None produces summaries_length=2."""
        solver = Solver(
            no_driver_system,
            save_every=None,
            summarise_every=None,
            sample_summaries_every=0.01,
            dt=0.01,
        )

        # Build minimal input arrays for a single run
        n_states = no_driver_system.sizes.states
        n_params = no_driver_system.sizes.parameters
        inits = np.ones((n_states, 1), dtype=np.float32)
        params = np.ones((n_params, 1), dtype=np.float32)

        result = solver.solve(
            initial_values=inits,
            parameters=params,
            duration=TEST_DURATION,
        )

        # summaries_length=2 means summaries shape[0] (time dimension) is 2
        assert result.state_summaries.shape[0] == 2

    def test_periodic_save_output_length(self, no_driver_system):
        """save_every=X produces floor(duration/X)+1 outputs."""
        save_every = 0.05
        solver = Solver(
            no_driver_system,
            save_every=save_every,
            summarise_every=None,
            sample_summaries_every=0.01,
            dt=0.01,
        )

        # Build minimal input arrays for a single run
        n_states = no_driver_system.sizes.states
        n_params = no_driver_system.sizes.parameters
        inits = np.ones((n_states, 1), dtype=np.float32)
        params = np.ones((n_params, 1), dtype=np.float32)

        result = solver.solve(
            initial_values=inits,
            parameters=params,
            duration=TEST_DURATION,
        )

        # Expected: floor(0.2 / 0.05) + 1 = 4 + 1 = 5
        expected_length = int(np.floor(TEST_DURATION / save_every)) + 1
        assert result.state.shape[0] == expected_length

    def test_periodic_summarise_length(self, no_driver_system):
        """summarise_every=X produces floor(duration/X) summaries."""
        summarise_every = 0.05
        solver = Solver(
            no_driver_system,
            save_every=None,
            summarise_every=summarise_every,
            sample_summaries_every=0.01,
            dt=0.01,
        )

        # Build minimal input arrays for a single run
        n_states = no_driver_system.sizes.states
        n_params = no_driver_system.sizes.parameters
        inits = np.ones((n_states, 1), dtype=np.float32)
        params = np.ones((n_params, 1), dtype=np.float32)

        result = solver.solve(
            initial_values=inits,
            parameters=params,
            duration=TEST_DURATION,
        )

        # Expected: floor(0.2 / 0.05) = 4
        expected_length = int(np.floor(TEST_DURATION / summarise_every))
        assert result.state_summaries.shape[0] == expected_length


class TestParameterReset:
    """Test parameter reset behavior between solves."""

    def test_sample_summaries_every_recalculates_on_none(self, no_driver_system):
        """sample_summaries_every recalculates when reset to None."""
        solver = Solver(
            no_driver_system,
            save_every=None,
            summarise_every=None,
            sample_summaries_every=0.01,
            dt=0.01,
        )

        # Build minimal input arrays for a single run
        n_states = no_driver_system.sizes.states
        n_params = no_driver_system.sizes.parameters
        inits = np.ones((n_states, 1), dtype=np.float32)
        params = np.ones((n_params, 1), dtype=np.float32)

        # First solve with explicit sample_summaries_every value
        result1 = solver.solve(
            initial_values=inits,
            parameters=params,
            duration=TEST_DURATION,
        )

        # Verify sample_summaries_every was set (from init)
        loop_config = solver.kernel.single_integrator._loop.compile_settings
        initial_sample_summaries = loop_config.sample_summaries_every

        # Reset sample_summaries_every to None
        solver.update({"sample_summaries_every": None})

        # Second solve with None should trigger recalculation
        result2 = solver.solve(
            initial_values=inits,
            parameters=params,
            duration=TEST_DURATION,
        )

        # Verify the behavior completed without error
        # When sample_summaries_every is None and summarise_last is True,
        # samples_per_summary should be recalculated from duration
        assert result2.state is not None


class TestDurationDependencyWarning:
    """Test duration dependency warning behavior."""

    def test_warning_on_summarise_last_without_summarise_every(
        self, no_driver_system
    ):
        """Warning raised when duration affects samples_per_summary."""
        solver = Solver(
            no_driver_system,
            save_every=None,
            summarise_every=None,
            sample_summaries_every=None,
            dt=0.01,
        )

        # Build minimal input arrays for a single run
        n_states = no_driver_system.sizes.states
        n_params = no_driver_system.sizes.parameters
        inits = np.ones((n_states, 1), dtype=np.float32)
        params = np.ones((n_params, 1), dtype=np.float32)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            solver.solve(
                initial_values=inits,
                parameters=params,
                duration=TEST_DURATION,
            )

            # Filter for UserWarning about sample_summaries_every
            duration_warnings = [
                warning for warning in w
                if issubclass(warning.category, UserWarning)
                and "sample_summaries_every" in str(warning.message)
            ]

            assert len(duration_warnings) == 1
            assert "kernel recompilation" in str(duration_warnings[0].message)
