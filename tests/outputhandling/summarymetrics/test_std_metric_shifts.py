"""
Tests for shift value persistence across multiple summary periods.

This module validates that the std, mean_std, mean_std_rms, and std_rms
metrics properly maintain their shift values across multiple summary windows
to preserve numerical stability.
"""
import numpy as np

from cubie.outputhandling import summary_metrics


def test_std_shift_persistence_across_periods():
    """Test that Std metric maintains shift value across summary periods."""
    metric = summary_metrics._metric_objects['std']
    metric.update_compile_settings(precision=np.float64)
    func_cache = metric.build()
    update_fn = func_cache.update
    save_fn = func_cache.save
    
    buffer = np.zeros(3, dtype=np.float64)
    output = np.zeros(1, dtype=np.float64)
    summarise_every = 5
    
    # First summary period
    values_period1 = [100.0, 101.0, 102.0, 103.0, 104.0]
    for idx, value in enumerate(values_period1):
        update_fn(value, buffer, idx, 0)
    
    # Save should set buffer[0] to mean of period1
    save_fn(buffer, output, summarise_every, 0)
    
    expected_mean_period1 = np.mean(values_period1)
    assert np.isclose(buffer[0], expected_mean_period1), (
        f"Expected buffer[0]={expected_mean_period1}, got {buffer[0]}"
    )
    assert buffer[1] == 0.0
    assert buffer[2] == 0.0
    
    # Second summary period - simulate save_idx continuing
    values_period2 = [105.0, 106.0, 107.0, 108.0, 109.0]
    for idx, value in enumerate(values_period2):
        current_idx = idx + summarise_every
        update_fn(value, buffer, current_idx, 0)
    
    # Verify shift is being used in second period
    assert buffer[0] == expected_mean_period1
    
    # Save second period
    save_fn(buffer, output, summarise_every, 0)
    
    expected_mean_period2 = np.mean(values_period2)
    assert np.isclose(buffer[0], expected_mean_period2), (
        f"Expected buffer[0]={expected_mean_period2}, got {buffer[0]}"
    )


def test_mean_std_shift_persistence_across_periods():
    """Test that MeanStd metric maintains shift value across summary periods."""
    metric = summary_metrics._metric_objects['mean_std']
    metric.update_compile_settings(precision=np.float64)
    func_cache = metric.build()
    update_fn = func_cache.update
    save_fn = func_cache.save
    
    buffer = np.zeros(3, dtype=np.float64)
    output = np.zeros(2, dtype=np.float64)
    summarise_every = 5
    
    # First summary period
    values_period1 = [100.0, 101.0, 102.0, 103.0, 104.0]
    for idx, value in enumerate(values_period1):
        update_fn(value, buffer, idx, 0)
    
    save_fn(buffer, output, summarise_every, 0)
    
    expected_mean_period1 = np.mean(values_period1)
    assert np.isclose(buffer[0], expected_mean_period1)
    
    # Second summary period
    values_period2 = [105.0, 106.0, 107.0, 108.0, 109.0]
    for idx, value in enumerate(values_period2):
        current_idx = idx + summarise_every
        update_fn(value, buffer, current_idx, 0)
    
    assert buffer[0] == expected_mean_period1
    
    save_fn(buffer, output, summarise_every, 0)
    
    expected_mean_period2 = np.mean(values_period2)
    assert np.isclose(buffer[0], expected_mean_period2)


def test_mean_std_rms_shift_persistence_across_periods():
    """Test that MeanStdRms metric maintains shift value across summary periods."""
    metric = summary_metrics._metric_objects['mean_std_rms']
    metric.update_compile_settings(precision=np.float64)
    func_cache = metric.build()
    update_fn = func_cache.update
    save_fn = func_cache.save
    
    buffer = np.zeros(3, dtype=np.float64)
    output = np.zeros(3, dtype=np.float64)
    summarise_every = 5
    
    # First summary period
    values_period1 = [100.0, 101.0, 102.0, 103.0, 104.0]
    for idx, value in enumerate(values_period1):
        update_fn(value, buffer, idx, 0)
    
    save_fn(buffer, output, summarise_every, 0)
    
    expected_mean_period1 = np.mean(values_period1)
    assert np.isclose(buffer[0], expected_mean_period1)
    
    # Second summary period
    values_period2 = [105.0, 106.0, 107.0, 108.0, 109.0]
    for idx, value in enumerate(values_period2):
        current_idx = idx + summarise_every
        update_fn(value, buffer, current_idx, 0)
    
    assert buffer[0] == expected_mean_period1
    
    save_fn(buffer, output, summarise_every, 0)
    
    expected_mean_period2 = np.mean(values_period2)
    assert np.isclose(buffer[0], expected_mean_period2)


def test_std_rms_shift_persistence_across_periods():
    """Test that StdRms metric maintains shift value across summary periods."""
    metric = summary_metrics._metric_objects['std_rms']
    metric.update_compile_settings(precision=np.float64)
    func_cache = metric.build()
    update_fn = func_cache.update
    save_fn = func_cache.save
    
    buffer = np.zeros(3, dtype=np.float64)
    output = np.zeros(2, dtype=np.float64)
    summarise_every = 5
    
    # First summary period
    values_period1 = [100.0, 101.0, 102.0, 103.0, 104.0]
    for idx, value in enumerate(values_period1):
        update_fn(value, buffer, idx, 0)
    
    save_fn(buffer, output, summarise_every, 0)
    
    expected_mean_period1 = np.mean(values_period1)
    assert np.isclose(buffer[0], expected_mean_period1)
    
    # Second summary period
    values_period2 = [105.0, 106.0, 107.0, 108.0, 109.0]
    for idx, value in enumerate(values_period2):
        current_idx = idx + summarise_every
        update_fn(value, buffer, current_idx, 0)
    
    assert buffer[0] == expected_mean_period1
    
    save_fn(buffer, output, summarise_every, 0)
    
    expected_mean_period2 = np.mean(values_period2)
    assert np.isclose(buffer[0], expected_mean_period2)


def test_std_numerical_stability_with_shift():
    """Test that using shift improves numerical stability for std calculation."""
    metric = summary_metrics._metric_objects['std']
    metric.update_compile_settings(precision=np.float64)
    func_cache = metric.build()
    update_fn = func_cache.update
    save_fn = func_cache.save
    
    buffer = np.zeros(3, dtype=np.float64)
    output = np.zeros(1, dtype=np.float64)
    summarise_every = 5
    
    # Use large values to test numerical stability
    values = [1e10, 1e10 + 1, 1e10 + 2, 1e10 + 3, 1e10 + 4]
    
    for idx, value in enumerate(values):
        update_fn(value, buffer, idx, 0)
    
    save_fn(buffer, output, summarise_every, 0)
    
    # Calculate expected std
    expected_std = np.std(values, ddof=0)
    
    # With proper shifting, the result should be accurate
    assert np.isclose(output[0], expected_std, rtol=1e-10), (
        f"Expected std={expected_std}, got {output[0]}"
    )
    
    # Verify shift was set to mean
    expected_mean = np.mean(values)
    assert np.isclose(buffer[0], expected_mean)

