"""Functional tests for dt_update_summaries parameter."""

import pytest
import numpy as np


@pytest.fixture
def base_settings(precision):
    """Base settings for dt_update_summaries tests."""
    return {
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


@pytest.mark.parametrize(
    "solver_settings_override",
    [{"system_type": "linear"}],
    indirect=True
)
def test_dt_update_summaries_equals_dt_save(solver, base_settings):
    """Test that default behavior (dt_update == dt_save) works."""
    settings = dict(base_settings)
    settings['dt_update_summaries'] = settings['dt_save']
    
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
    assert result.summaries.shape[0] == 2


@pytest.mark.parametrize(
    "solver_settings_override",
    [{"system_type": "linear"}],
    indirect=True
)
def test_dt_update_summaries_less_than_dt_save(solver, base_settings, precision):
    """Test dt_update_summaries < dt_save (more updates)."""
    settings = dict(base_settings)
    settings['dt_update_summaries'] = precision(0.05)
    
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
    assert result.summaries.shape[0] == 2


@pytest.mark.parametrize(
    "solver_settings_override",
    [{"system_type": "linear"}],
    indirect=True
)
def test_dt_update_summaries_greater_than_dt_save(solver, precision):
    """Test dt_update_summaries > dt_save (fewer updates)."""
    settings = {
        'system_type': 'linear',
        'algorithm': 'euler',
        'step_controller': 'fixed',
        'dt': precision(0.001),
        'dt_save': precision(0.1),
        'dt_update_summaries': precision(0.5),
        'dt_summarise': precision(2.0),
        'output_types': ['summaries'],
        'summaries': ['mean'],
        'summarised_state_indices': [0, 1, 2],
    }
    
    result = solver.solve(
        initial_values={},
        parameters={},
        drivers={},
        duration=4.0,
        settling_time=0.0,
        blocksize=32,
        grid_type='combinatorial',
        results_type='full',
        **settings
    )
    
    assert result.summaries is not None
    assert result.summaries.shape[0] == 2


@pytest.mark.parametrize(
    "solver_settings_override",
    [{"system_type": "linear"}],
    indirect=True
)
@pytest.mark.parametrize(
    "dt_update,expected_updates",
    [(0.1, 10), (0.2, 5), (0.25, 4), (0.5, 2), (1.0, 1)]
)
def test_various_update_frequencies(
    solver, base_settings, precision, dt_update, expected_updates
):
    """Test various dt_update_summaries frequencies."""
    settings = dict(base_settings)
    settings['dt_update_summaries'] = precision(dt_update)
    settings['dt_summarise'] = precision(1.0)
    
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
    [{"system_type": "linear", "step_controller": "pi"}],
    indirect=True
)
def test_adaptive_step_with_dt_update_summaries(solver, precision):
    """Test dt_update_summaries with adaptive stepping."""
    settings = {
        'system_type': 'linear',
        'algorithm': 'crank_nicolson',
        'step_controller': 'pi',
        'dt': precision(0.01),
        'dt_min': precision(1e-6),
        'dt_max': precision(0.5),
        'dt_save': precision(0.1),
        'dt_update_summaries': precision(0.05),
        'dt_summarise': precision(1.0),
        'output_types': ['summaries'],
        'summaries': ['mean'],
        'summarised_state_indices': [0, 1, 2],
        'atol': precision(1e-6),
        'rtol': precision(1e-5),
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
def test_settling_time_with_dt_update_summaries(solver, precision):
    """Test dt_update_summaries with settling_time > 0."""
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
        duration=2.0,
        settling_time=0.5,
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
def test_summary_only_output_mode(solver, precision):
    """Test summary-only mode with no state saves."""
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
        duration=2.0,
        settling_time=0.0,
        blocksize=32,
        grid_type='combinatorial',
        results_type='full',
        **settings
    )
    
    assert result.states is None
    assert result.summaries is not None
    assert result.summaries.shape[0] == 2
