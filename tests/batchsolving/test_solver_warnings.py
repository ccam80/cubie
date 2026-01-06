"""Tests for Solver warning behavior.

These tests verify that Solver.solve emits appropriate warnings when
duration-dependent timing parameters are used without explicit
summarise_every values.
"""
import warnings

import numpy as np
import pytest


class TestDurationDependencyWarning:
    """Tests for duration dependency warning in Solver.solve."""

    @pytest.mark.parametrize(
        "solver_settings_override",
        [{"save_every": None, "summarise_every": None}],
        indirect=True,
    )
    def test_duration_dependency_warning_emitted(
        self, solver_mutable, system
    ):
        """Verify warning is raised when summarise_last=True, no summarise_every."""
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

    @pytest.mark.parametrize(
        "solver_settings_override",
        [{"save_every": None, "summarise_every": 0.1}],
        indirect=True,
    )
    def test_no_warning_with_explicit_summarise_every(
        self, solver_mutable, system
    ):
        """Verify no warning when summarise_every is explicitly set."""
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

            assert len(duration_warnings) == 0

    @pytest.mark.parametrize(
        "solver_settings_override",
        [{"save_every": 0.05, "summarise_every": 0.05}],
        indirect=True,
    )
    def test_no_warning_with_summarise_last_false(
        self, solver_mutable, system
    ):
        """Verify no warning when summarise_last=False (periodic mode)."""
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

            assert len(duration_warnings) == 0
