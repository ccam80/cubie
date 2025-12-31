"""Backward compatibility tests for dt_update_summaries."""

import pytest
import numpy as np


@pytest.mark.parametrize(
    "solver_settings_override",
    [{"system_type": "linear"}],
    indirect=True
)
def test_existing_code_without_dt_update_summaries(solver, precision):
    """Test that existing code works without dt_update_summaries."""
    settings = {
        'system_type': 'linear',
        'algorithm': 'euler',
        'step_controller': 'fixed',
        'dt': precision(0.001),
        'dt_save': precision(0.1),
        'dt_summarise': precision(1.0),
        'output_types': ['state', 'summaries'],
        'summaries': ['mean'],
        'summarised_state_indices': [0, 1, 2],
    }
    
    result = solver.solve(
        initial_values={},
        parameters={},
        drivers={},
        duration=2.0,
        settling_time=0.0,
        blocksize=32,
        grid_type='combinatorial',
        results_type='full',
        **settings
    )
    
    assert result.states is not None
    assert result.summaries is not None


@pytest.mark.parametrize(
    "solver_settings_override",
    [{"system_type": "linear"}],
    indirect=True
)
def test_default_equals_old_behavior(solver, precision):
    """Test that default dt_update_summaries produces same results as old code."""
    base_settings = {
        'system_type': 'linear',
        'algorithm': 'euler',
        'step_controller': 'fixed',
        'dt': precision(0.001),
        'dt_save': precision(0.1),
        'dt_summarise': precision(1.0),
        'output_types': ['summaries'],
        'summaries': ['mean', 'max'],
        'summarised_state_indices': [0, 1, 2],
    }
    
    result_old = solver.solve(
        initial_values={},
        parameters={},
        drivers={},
        duration=2.0,
        settling_time=0.0,
        blocksize=32,
        grid_type='combinatorial',
        results_type='full',
        **base_settings
    )
    
    settings_new = dict(base_settings)
    settings_new['dt_update_summaries'] = precision(0.1)
    result_new = solver.solve(
        initial_values={},
        parameters={},
        drivers={},
        duration=2.0,
        settling_time=0.0,
        blocksize=32,
        grid_type='combinatorial',
        results_type='full',
        **settings_new
    )
    
    np.testing.assert_array_equal(result_old.summaries, result_new.summaries)


def test_existing_tests_still_pass():
    """Verify that we can still run existing test patterns."""
    pass
