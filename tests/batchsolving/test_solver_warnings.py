"""Tests for Solver warning behavior.

These tests verify that Solver.solve emits appropriate warnings when
duration-dependent timing parameters are used without explicit
summarise_every values.

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


class TestDurationDependencyWarning:
    """Tests for duration dependency warning in Solver.solve."""

    def test_duration_dependency_warning_emitted(self, no_driver_system):
        """Verify warning is raised when summarise_last=True, no summarise_every."""
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

    def test_no_warning_with_explicit_summarise_every(self, no_driver_system):
        """Verify no warning when summarise_every is explicitly set."""
        solver = Solver(
            no_driver_system,
            save_every=None,
            summarise_every=0.05,
            sample_summaries_every=0.01,
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

            assert len(duration_warnings) == 0

    def test_no_warning_with_summarise_last_false(self, no_driver_system):
        """Verify no warning when summarise_last=False (periodic mode)."""
        solver = Solver(
            no_driver_system,
            save_every=0.05,
            summarise_every=0.05,
            sample_summaries_every=0.01,
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

            assert len(duration_warnings) == 0
