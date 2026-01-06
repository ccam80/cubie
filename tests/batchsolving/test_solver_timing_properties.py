"""Tests for Solver timing property type annotations and None-safety.

These tests verify that Solver.save_every, Solver.summarise_every, and
Solver.sample_summaries_every return Optional[float] and that SolveSpec
correctly handles None timing values.
"""
import numpy as np
import pytest


class TestSolverTimingPropertyReturnTypes:
    """Tests for Solver timing property return types."""

    @pytest.mark.parametrize(
        "solver_settings_override",
        [{"save_every": None, "summarise_every": None}],
        indirect=True,
    )
    def test_solver_save_every_returns_none_in_save_last_mode(
        self, solver_mutable
    ):
        """Verify Solver.save_every returns None when save_every is not set."""
        assert solver_mutable.save_every is None

    @pytest.mark.parametrize(
        "solver_settings_override",
        [{"save_every": None, "summarise_every": None}],
        indirect=True,
    )
    def test_solver_summarise_every_returns_none_in_summarise_last_mode(
        self, solver_mutable
    ):
        """Verify Solver.summarise_every returns None when not set."""
        assert solver_mutable.summarise_every is None

    @pytest.mark.parametrize(
        "solver_settings_override",
        [{"save_every": 0.05, "summarise_every": 0.05}],
        indirect=True,
    )
    def test_solver_save_every_returns_float_when_configured(
        self, solver_mutable
    ):
        """Verify Solver.save_every returns float when configured."""
        assert solver_mutable.save_every is not None
        assert isinstance(solver_mutable.save_every, (float, np.floating))

    @pytest.mark.parametrize(
        "solver_settings_override",
        [{"save_every": 0.05, "summarise_every": 0.05}],
        indirect=True,
    )
    def test_solver_summarise_every_returns_float_when_configured(
        self, solver_mutable
    ):
        """Verify Solver.summarise_every returns float when configured."""
        assert solver_mutable.summarise_every is not None
        assert isinstance(solver_mutable.summarise_every, (float, np.floating))


class TestSolveSpecNoneTimingValues:
    """Tests for SolveSpec handling of None timing values."""

    def test_solve_info_handles_none_timing_values(self):
        """Verify SolveSpec can be created with None timing values."""
        from cubie.batchsolving.solveresult import SolveSpec

        spec = SolveSpec(
            dt=0.01,
            dt_min=1e-7,
            dt_max=1.0,
            save_every=None,
            summarise_every=None,
            sample_summaries_every=None,
            atol=1e-6,
            rtol=1e-6,
            duration=0.2,
            warmup=0.0,
            t0=0.0,
            algorithm="euler",
            saved_states=None,
            saved_observables=None,
            summarised_states=None,
            summarised_observables=None,
            output_types=None,
            precision=np.float32,
        )
        assert spec.save_every is None
        assert spec.summarise_every is None
        assert spec.sample_summaries_every is None

    def test_solve_spec_accepts_float_timing_values(self):
        """Verify SolveSpec accepts float timing values."""
        from cubie.batchsolving.solveresult import SolveSpec

        spec = SolveSpec(
            dt=0.01,
            dt_min=1e-7,
            dt_max=1.0,
            save_every=0.05,
            summarise_every=0.05,
            sample_summaries_every=0.01,
            atol=1e-6,
            rtol=1e-6,
            duration=0.2,
            warmup=0.0,
            t0=0.0,
            algorithm="euler",
            saved_states=None,
            saved_observables=None,
            summarised_states=None,
            summarised_observables=None,
            output_types=None,
            precision=np.float32,
        )
        assert spec.save_every == 0.05
        assert spec.summarise_every == 0.05
        assert spec.sample_summaries_every == 0.01
