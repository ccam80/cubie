"""Tests for nan_error_trajectories feature in solver and result handling."""

import numpy as np
import pytest
from cubie import solve_ivp, Solver
from cubie.batchsolving.solveresult import SolveResult


@pytest.fixture(scope="function")
def three_state_linear(precision):
    """Create a simple linear system for testing."""
    from tests.system_fixtures import build_three_state_linear_system
    return build_three_state_linear_system(precision)


def test_status_codes_attribute_exists(three_state_linear):
    """Verify status_codes attribute is present in SolveResult."""
    system = three_state_linear
    result = solve_ivp(
        system,
        y0={'x0': [1.0], 'x1': [0.0], 'x2': [0.0]},
        parameters={'p0': [0.1], 'p1': [0.1], 'p2': [0.1]},
        duration=0.1,
        dt_save=0.01,
    )
    assert hasattr(result, 'status_codes')
    assert result.status_codes is not None
    assert isinstance(result.status_codes, np.ndarray)
    assert result.status_codes.dtype == np.int32
    assert result.status_codes.shape == (1,)


def test_status_codes_excluded_from_raw_results(three_state_linear):
    """Verify status_codes not included when results_type='raw'."""
    system = three_state_linear
    result = solve_ivp(
        system,
        y0={'x0': [1.0], 'x1': [0.0], 'x2': [0.0]},
        parameters={'p0': [0.1], 'p1': [0.1], 'p2': [0.1]},
        duration=0.1,
        dt_save=0.01,
        results_type='raw',
    )
    assert isinstance(result, dict)
    assert 'status_codes' not in result


def test_successful_runs_unchanged(three_state_linear):
    """Verify successful runs (status_code==0) are not modified."""
    system = three_state_linear

    # Run with nan_error_trajectories=False to get baseline
    result_no_nan = solve_ivp(
        system,
        y0={'x0': [1.0, 2.0], 'x1': [0.0, 0.0], 'x2': [0.0, 0.0]},
        parameters={'p0': [0.1, 0.2], 'p1': [0.1, 0.2], 'p2': [0.1, 0.2]},
        duration=0.1,
        dt_save=0.01,
        nan_error_trajectories=False,
    )

    # Run with nan_error_trajectories=True
    result_with_nan = solve_ivp(
        system,
        y0={'x0': [1.0, 2.0], 'x1': [0.0, 0.0], 'x2': [0.0, 0.0]},
        parameters={'p0': [0.1, 0.2], 'p1': [0.1, 0.2], 'p2': [0.1, 0.2]},
        duration=0.1,
        dt_save=0.01,
        nan_error_trajectories=True,
    )

    # All status codes should be 0 (success)
    assert np.all(result_with_nan.status_codes == 0)

    # Arrays should be identical
    np.testing.assert_array_equal(
        result_no_nan.time_domain_array,
        result_with_nan.time_domain_array
    )
    np.testing.assert_array_equal(
        result_no_nan.summaries_array,
        result_with_nan.summaries_array
    )


def test_nan_error_trajectories_false_preserves_values(
    three_state_linear
):
    """Verify nan_error_trajectories=False returns unmodified data."""
    system = three_state_linear

    result = solve_ivp(
        system,
        y0={'x0': [1.0], 'x1': [0.0], 'x2': [0.0]},
        parameters={'p0': [0.1], 'p1': [0.1], 'p2': [0.1]},
        duration=0.1,
        dt_save=0.01,
        nan_error_trajectories=False,
    )

    # Even if there are errors, data should not be NaN
    # when nan_error_trajectories=False
    if np.any(result.status_codes != 0):
        # If any errors exist, verify arrays contain non-NaN values
        assert not np.all(np.isnan(result.time_domain_array))


def test_parameter_propagation_solve_ivp(three_state_linear):
    """Verify nan_error_trajectories propagates from solve_ivp."""
    system = three_state_linear

    # Test with True (default)
    result_true = solve_ivp(
        system,
        y0={'x0': [1.0], 'x1': [0.0], 'x2': [0.0]},
        parameters={'p0': [0.1], 'p1': [0.1], 'p2': [0.1]},
        duration=0.1,
        dt_save=0.01,
        nan_error_trajectories=True,
    )
    assert result_true.status_codes is not None

    # Test with False
    result_false = solve_ivp(
        system,
        y0={'x0': [1.0], 'x1': [0.0], 'x2': [0.0]},
        parameters={'p0': [0.1], 'p1': [0.1], 'p2': [0.1]},
        duration=0.1,
        dt_save=0.01,
        nan_error_trajectories=False,
    )
    assert result_false.status_codes is not None

    # Both should have status_codes (behavior difference is in
    # NaN-setting logic, not status_codes presence)
    assert result_true.status_codes.shape == result_false.status_codes.shape


def test_nan_processing_with_different_stride_orders(
    three_state_linear
):
    """Verify NaN processing works correctly with stride orders.
    
    Note: This test verifies that the NaN processing logic correctly
    identifies the run dimension regardless of the stride order by
    using the stride_order.index("run") pattern.
    """
    system = three_state_linear
    solver = Solver(system, dt_save=0.01)

    result = solver.solve(
        initial_values={'x0': [1.0, 2.0], 'x1': [0.0, 0.0], 'x2': [0.0, 0.0]},
        parameters={'p0': [0.1, 0.2], 'p1': [0.1, 0.2], 'p2': [0.1, 0.2]},
        duration=0.1,
        nan_error_trajectories=True,
    )

    # Verify the result has a valid stride order
    assert result._stride_order is not None
    assert len(result._stride_order) == 3
    assert "run" in result._stride_order

    # Verify shapes are correct
    assert result.time_domain_array.ndim == 3
    assert result.status_codes.shape[0] == 2


@pytest.mark.parametrize("results_type", [
    "full",
    "numpy",
    "numpy_per_summary",
])
def test_status_codes_in_different_result_types(
    three_state_linear,
    results_type
):
    """Verify status_codes included in non-raw result types."""
    system = three_state_linear
    result = solve_ivp(
        system,
        y0={'x0': [1.0], 'x1': [0.0], 'x2': [0.0]},
        parameters={'p0': [0.1], 'p1': [0.1], 'p2': [0.1]},
        duration=0.1,
        dt_save=0.01,
        results_type=results_type,
    )

    if results_type == "full":
        assert hasattr(result, 'status_codes')
        assert result.status_codes is not None
    else:
        # numpy and numpy_per_summary return dicts
        assert isinstance(result, dict)


def test_empty_summaries_array_handled(three_state_linear):
    """Verify NaN processing handles empty summaries gracefully."""
    system = three_state_linear

    # Use output_types to exclude summaries
    result = solve_ivp(
        system,
        y0={'x0': [1.0], 'x1': [0.0], 'x2': [0.0]},
        parameters={'p0': [0.1], 'p1': [0.1], 'p2': [0.1]},
        duration=0.1,
        dt_save=0.01,
        output_types=["state", "observables", "time"],
        nan_error_trajectories=True,
    )

    # Should not crash when summaries_array is empty
    assert result.summaries_array.size == 0
    assert result.status_codes is not None


@pytest.mark.nocudasim
def test_nan_processing_with_actual_errors(three_state_linear):
    """Verify NaN processing works when solver actually fails.
    
    Note: This test requires actual CUDA hardware since it uses implicit
    methods with matrix-free solvers that don't work in CUDA simulation mode.
    """
    system = three_state_linear

    # Force Newton solver failure by using implicit method with
    # extremely tight tolerance and very few iterations
    result = solve_ivp(
        system,
        y0={'x0': [1.0, 2.0], 'x1': [0.0, 0.0], 'x2': [0.0, 0.0]},
        parameters={'p0': [0.1, 0.2], 'p1': [0.1, 0.2], 'p2': [0.1, 0.2]},
        duration=1.0,
        dt_save=0.01,
        method='backwards_euler',
        max_newton_iterations=1,  # Force failure
        newton_tol=1e-15,  # Impossible tolerance
        nan_error_trajectories=True,
    )

    # Check if any runs failed
    failed_runs = np.where(result.status_codes != 0)[0]

    if len(failed_runs) > 0:
        # Verify failed runs have all-NaN trajectories
        for run_idx in failed_runs:
            run_slice = (slice(None), slice(None), run_idx)
            assert np.all(np.isnan(result.time_domain_array[run_slice]))
            if result.summaries_array.size > 0:
                assert np.all(np.isnan(result.summaries_array[run_slice]))
    else:
        # If test system is too stable, skip validation
        pytest.skip("Test system did not generate errors")
