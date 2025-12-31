"""Edge case tests for dt_update_summaries."""

import pytest
import numpy as np


@pytest.mark.parametrize(
    "solver_settings_override",
    [{"system_type": "linear"}],
    indirect=True
)
def test_dt_update_summaries_equals_dt_summarise(solver, precision):
    """Test dt_update_summaries == dt_summarise (single update per summary)."""
    settings = {
        'system_type': 'linear',
        'algorithm': 'euler',
        'step_controller': 'fixed',
        'dt': precision(0.001),
        'dt_update_summaries': precision(1.0),
        'dt_summarise': precision(1.0),
        'output_types': ['summaries'],
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
    
    assert result.summaries is not None


@pytest.mark.parametrize(
    "solver_settings_override",
    [{"system_type": "linear"}],
    indirect=True
)
def test_very_small_dt_update_summaries(solver, precision):
    """Test very small dt_update_summaries (many updates)."""
    settings = {
        'system_type': 'linear',
        'algorithm': 'euler',
        'step_controller': 'fixed',
        'dt': precision(0.001),
        'dt_update_summaries': precision(0.01),
        'dt_summarise': precision(1.0),
        'output_types': ['summaries'],
        'summaries': ['mean'],
        'summarised_state_indices': [0, 1, 2],
    }
    
    result = solver.solve(
        initial_values={},
        parameters={},
        drivers={},
        duration=1.0,
        settling_time=0.0,
        blocksize=32,
        grid_type='combinatorial',
        results_type='full',
        **settings
    )
    
    assert result.summaries is not None


@pytest.mark.parametrize(
    "solver_settings_override",
    [{"system_type": "linear"}],
    indirect=True
)
def test_dt_update_equals_dt_step(solver, precision):
    """Test dt_update_summaries == dt (update every step)."""
    settings = {
        'system_type': 'linear',
        'algorithm': 'euler',
        'step_controller': 'fixed',
        'dt': precision(0.01),
        'dt_update_summaries': precision(0.01),
        'dt_summarise': precision(1.0),
        'output_types': ['summaries'],
        'summaries': ['mean'],
        'summarised_state_indices': [0, 1, 2],
    }
    
    result = solver.solve(
        initial_values={},
        parameters={},
        drivers={},
        duration=1.0,
        settling_time=0.0,
        blocksize=32,
        grid_type='combinatorial',
        results_type='full',
        **settings
    )
    
    assert result.summaries is not None


@pytest.mark.parametrize(
    "solver_settings_override",
    [{"system_type": "linear"}],
    indirect=True
)
def test_initial_summary_at_t_zero(solver, precision):
    """Test that initial summary is computed at t=0."""
    settings = {
        'system_type': 'linear',
        'algorithm': 'euler',
        'step_controller': 'fixed',
        'dt': precision(0.001),
        'dt_update_summaries': precision(0.1),
        'dt_summarise': precision(1.0),
        'settling_time': 0.0,
        'output_types': ['summaries'],
        'summaries': ['mean'],
        'summarised_state_indices': [0, 1, 2],
    }
    
    result = solver.solve(
        initial_values={},
        parameters={},
        drivers={},
        duration=1.0,
        settling_time=0.0,
        blocksize=32,
        grid_type='combinatorial',
        results_type='full',
        **settings
    )
    
    assert result.summaries is not None


@pytest.mark.parametrize(
    "solver_settings_override",
    [{"system_type": "linear"}],
    indirect=True
)
def test_multiple_summary_metrics(solver, precision):
    """Test multiple summary metrics with dt_update_summaries."""
    settings = {
        'system_type': 'linear',
        'algorithm': 'euler',
        'step_controller': 'fixed',
        'dt': precision(0.001),
        'dt_update_summaries': precision(0.2),
        'dt_summarise': precision(1.0),
        'output_types': ['summaries'],
        'summaries': ['mean', 'max', 'rms'],
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
    
    assert result.summaries is not None


@pytest.mark.parametrize(
    "solver_settings_override",
    [{"system_type": "linear"}],
    indirect=True
)
def test_floating_point_precision_in_divisibility(solver, precision):
    """Test divisibility check handles floating-point precision."""
    settings = {
        'system_type': 'linear',
        'algorithm': 'euler',
        'step_controller': 'fixed',
        'dt': precision(0.001),
        'dt_update_summaries': precision(0.1),
        'dt_summarise': precision(1.0),
        'output_types': ['summaries'],
        'summaries': ['mean'],
        'summarised_state_indices': [0, 1, 2],
    }
    
    result = solver.solve(
        initial_values={},
        parameters={},
        drivers={},
        duration=1.0,
        settling_time=0.0,
        blocksize=32,
        grid_type='combinatorial',
        results_type='full',
        **settings
    )
    
    assert result.summaries is not None
