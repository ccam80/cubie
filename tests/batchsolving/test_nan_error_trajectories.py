"""Tests for nan_error_trajectories feature in solver and result handling.

This test module uses a session-scoped fixture to run a single batch solve,
then creates multiple SolveResult objects with different nan_error_trajectories
settings to test the NaN processing functionality efficiently.
"""

import numpy as np
import pytest
from cubie import solve_ivp, Solver
from cubie.batchsolving.solveresult import SolveResult


@pytest.fixture(scope="session")
def three_state_linear_session(precision):
    """Create a simple linear system for testing (session-scoped)."""
    from tests.system_fixtures import build_three_state_linear_system
    return build_three_state_linear_system(precision)


@pytest.fixture(scope="session")
def solved_batch_solver(three_state_linear_session, precision):
    """Run a single batch solve and return the solver with computed arrays.
    
    This session-scoped fixture runs once and is reused by all tests,
    significantly reducing test runtime by avoiding repeated compilation
    and batch solves.
    """
    system = three_state_linear_session
    solver = Solver(system, dt_save=0.01)
    
    # Run a single batch solve with multiple runs
    solver.solve(
        initial_values={'x0': [1.0, 2.0, 3.0], 'x1': [0.0, 0.0, 0.0], 
                       'x2': [0.0, 0.0, 0.0]},
        parameters={'p0': [0.1, 0.2, 0.3], 'p1': [0.1, 0.2, 0.3], 
                   'p2': [0.1, 0.2, 0.3]},
        duration=0.1,
    )
    return solver


class TestNaNProcessingWithMockedErrors:
    """Test NaN processing by manually modifying status codes.
    
    These tests use the session-scoped solver fixture and manually set
    status codes to test error handling without running expensive failing
    solves.
    """
    
    def test_nan_processing_with_simulated_errors(self, solved_batch_solver):
        """Verify NaN processing works by manually setting error codes."""
        # Manually inject error into status codes
        original_codes = solved_batch_solver.status_codes.copy()
        solved_batch_solver.kernel.output_arrays.host.status_codes.array[1] = 1
        
        result = SolveResult.from_solver(
            solved_batch_solver,
            nan_error_trajectories=True
        )
        
        # Verify run 1 is all NaN
        run_slice = (slice(None), slice(None), 1)
        assert np.all(np.isnan(result.time_domain_array[run_slice]))
        if result.summaries_array.size > 0:
            assert np.all(np.isnan(result.summaries_array[run_slice]))
        
        # Verify runs 0 and 2 are NOT NaN
        assert not np.all(np.isnan(result.time_domain_array[..., 0]))
        assert not np.all(np.isnan(result.time_domain_array[..., 2]))
        
        # Restore original status codes
        solved_batch_solver.kernel.output_arrays.host.status_codes.array[:] = original_codes
    
    def test_nan_disabled_preserves_error_data(self, solved_batch_solver):
        """Verify nan_error_trajectories=False preserves data even with errors."""
        # Manually inject error
        original_codes = solved_batch_solver.status_codes.copy()
        solved_batch_solver.kernel.output_arrays.host.status_codes.array[1] = 1
        
        result = SolveResult.from_solver(
            solved_batch_solver,
            nan_error_trajectories=False
        )
        
        # Even with error code, data should NOT be NaN
        assert not np.all(np.isnan(result.time_domain_array[..., 1]))
        
        # Restore original status codes
        solved_batch_solver.kernel.output_arrays.host.status_codes.array[:] = original_codes
    
    def test_successful_runs_unchanged_with_nan_enabled(self, solved_batch_solver):
        """Verify successful runs are not modified when NaN processing enabled."""
        result = SolveResult.from_solver(
            solved_batch_solver,
            nan_error_trajectories=True
        )
        
        # All runs should have status code 0 (success)
        assert np.all(result.status_codes == 0)
        
        # No data should be NaN
        assert not np.any(np.isnan(result.time_domain_array))
        if result.summaries_array.size > 0:
            assert not np.any(np.isnan(result.summaries_array))
    
    def test_multiple_errors_all_set_to_nan(self, solved_batch_solver):
        """Verify multiple failed runs all get NaN'd."""
        # Inject multiple errors
        original_codes = solved_batch_solver.status_codes.copy()
        solved_batch_solver.kernel.output_arrays.host.status_codes.array[0] = 2
        solved_batch_solver.kernel.output_arrays.host.status_codes.array[2] = 3
        
        result = SolveResult.from_solver(
            solved_batch_solver,
            nan_error_trajectories=True
        )
        
        # Runs 0 and 2 should be all NaN
        assert np.all(np.isnan(result.time_domain_array[..., 0]))
        assert np.all(np.isnan(result.time_domain_array[..., 2]))
        
        # Run 1 should NOT be NaN
        assert not np.all(np.isnan(result.time_domain_array[..., 1]))
        
        # Restore original status codes
        solved_batch_solver.kernel.output_arrays.host.status_codes.array[:] = original_codes


class TestParameterPropagation:
    """Test that nan_error_trajectories parameter propagates correctly.
    
    These tests run minimal batch solves to verify API parameter passing.
    """
    
    def test_parameter_propagation_solve_ivp(self, three_state_linear_session):
        """Verify nan_error_trajectories propagates from solve_ivp (minimal test)."""
        system = three_state_linear_session
        
        # Minimal solve to test parameter passing
        result = solve_ivp(
            system,
            y0={'x0': [1.0], 'x1': [0.0], 'x2': [0.0]},
            parameters={'p0': [0.1], 'p1': [0.1], 'p2': [0.1]},
            duration=0.05,  # Shorter duration for speed
            dt_save=0.01,
            nan_error_trajectories=True,
        )
        
        # Just verify it ran and has status codes
        assert result.status_codes is not None
    
    def test_parameter_propagation_solver_solve(self, three_state_linear_session):
        """Verify nan_error_trajectories propagates through Solver.solve."""
        system = three_state_linear_session
        solver = Solver(system, dt_save=0.01)
        
        # Minimal solve
        result = solver.solve(
            initial_values={'x0': [1.0], 'x1': [0.0], 'x2': [0.0]},
            parameters={'p0': [0.1], 'p1': [0.1], 'p2': [0.1]},
            duration=0.05,
            nan_error_trajectories=False,
        )
        
        assert result.status_codes is not None


class TestResultTypes:
    """Test behavior with different result types."""
    
    def test_raw_results_exclude_status_codes(self, three_state_linear_session):
        """Verify raw results don't include status codes."""
        system = three_state_linear_session
        
        result = solve_ivp(
            system,
            y0={'x0': [1.0], 'x1': [0.0], 'x2': [0.0]},
            parameters={'p0': [0.1], 'p1': [0.1], 'p2': [0.1]},
            duration=0.05,
            dt_save=0.01,
            results_type='raw',
        )
        
        assert isinstance(result, dict)
        assert 'status_codes' not in result
    
    @pytest.mark.parametrize("results_type", ["full", "numpy", "numpy_per_summary"])
    def test_non_raw_results_include_processing(
        self, 
        solved_batch_solver,
        results_type
    ):
        """Verify non-raw result types include NaN processing."""
        result = SolveResult.from_solver(
            solved_batch_solver,
            results_type=results_type,
            nan_error_trajectories=True
        )
        
        if results_type == "full":
            assert hasattr(result, 'status_codes')
            assert result.status_codes is not None
        else:
            # numpy types return dicts
            assert isinstance(result, dict)


class TestEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_empty_summaries_handled(self, three_state_linear_session):
        """Verify NaN processing handles empty summaries gracefully."""
        system = three_state_linear_session
        
        result = solve_ivp(
            system,
            y0={'x0': [1.0], 'x1': [0.0], 'x2': [0.0]},
            parameters={'p0': [0.1], 'p1': [0.1], 'p2': [0.1]},
            duration=0.05,
            dt_save=0.01,
            output_types=["state", "observables", "time"],
            nan_error_trajectories=True,
        )
        
        # Should not crash with empty summaries
        assert result.summaries_array.size == 0
        assert result.status_codes is not None
    
    def test_stride_order_compatibility(self, solved_batch_solver):
        """Verify NaN processing works with stride orders."""
        result = SolveResult.from_solver(
            solved_batch_solver,
            nan_error_trajectories=True
        )
        
        # Verify stride order is captured
        assert result._stride_order is not None
        assert len(result._stride_order) == 3
        assert "run" in result._stride_order
        
        # Verify shapes are consistent
        assert result.time_domain_array.ndim == 3
        assert result.status_codes.shape[0] == 3
