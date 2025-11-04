"""
Tests for new summary metrics added in issue #141.

Tests validate that std, min, max_magnitude, extrema, negative_peaks, and
mean_std_rms metrics are properly registered and have correct buffer sizes.
"""

import pytest
from cubie.outputhandling import summary_metrics


def test_new_metrics_registered():
    """Test that all new metrics from issue #141 are registered."""
    expected_metrics = [
        "std",
        "min",
        "max_magnitude",
        "extrema",
        "negative_peaks",
        "mean_std_rms",
    ]
    
    for metric in expected_metrics:
        assert metric in summary_metrics.implemented_metrics, \
            f"Metric '{metric}' should be registered"


def test_std_buffer_and_output_sizes():
    """Test std metric has correct buffer and output sizes."""
    requested = ["std"]
    
    buffer_sizes = summary_metrics.buffer_sizes(requested)
    output_sizes = summary_metrics.output_sizes(requested)
    
    assert buffer_sizes == (2,), "std should use 2 buffer slots (sum, sum_sq)"
    assert output_sizes == (1,), "std should output 1 value"


def test_min_buffer_and_output_sizes():
    """Test min metric has correct buffer and output sizes."""
    requested = ["min"]
    
    buffer_sizes = summary_metrics.buffer_sizes(requested)
    output_sizes = summary_metrics.output_sizes(requested)
    
    assert buffer_sizes == (1,), "min should use 1 buffer slot"
    assert output_sizes == (1,), "min should output 1 value"


def test_max_magnitude_buffer_and_output_sizes():
    """Test max_magnitude metric has correct buffer and output sizes."""
    requested = ["max_magnitude"]
    
    buffer_sizes = summary_metrics.buffer_sizes(requested)
    output_sizes = summary_metrics.output_sizes(requested)
    
    assert buffer_sizes == (1,), "max_magnitude should use 1 buffer slot"
    assert output_sizes == (1,), "max_magnitude should output 1 value"


def test_extrema_buffer_and_output_sizes():
    """Test extrema metric has correct buffer and output sizes."""
    requested = ["extrema"]
    
    buffer_sizes = summary_metrics.buffer_sizes(requested)
    output_sizes = summary_metrics.output_sizes(requested)
    
    assert buffer_sizes == (2,), "extrema should use 2 buffer slots (max, min)"
    assert output_sizes == (2,), "extrema should output 2 values (max, min)"


def test_negative_peaks_buffer_and_output_sizes():
    """Test negative_peaks metric with parameter."""
    requested = ["negative_peaks[5]"]
    
    buffer_sizes = summary_metrics.buffer_sizes(requested)
    output_sizes = summary_metrics.output_sizes(requested)
    
    assert buffer_sizes == (8,), "negative_peaks[5] should use 3+5=8 buffer slots"
    assert output_sizes == (5,), "negative_peaks[5] should output 5 values"


def test_mean_std_rms_buffer_and_output_sizes():
    """Test mean_std_rms composite metric."""
    requested = ["mean_std_rms"]
    
    buffer_sizes = summary_metrics.buffer_sizes(requested)
    output_sizes = summary_metrics.output_sizes(requested)
    
    assert buffer_sizes == (2,), \
        "mean_std_rms should use 2 buffer slots (shared sum, sum_sq)"
    assert output_sizes == (3,), \
        "mean_std_rms should output 3 values (mean, std, rms)"


def test_mean_std_rms_vs_individual_buffer_efficiency():
    """Test that mean_std_rms uses less buffer space than individual metrics."""
    composite = ["mean_std_rms"]
    individual = ["mean", "std", "rms"]
    
    composite_buffer = summary_metrics.summaries_buffer_height(composite)
    individual_buffer = summary_metrics.summaries_buffer_height(individual)
    
    assert composite_buffer < individual_buffer, \
        "mean_std_rms should use less buffer space than mean+std+rms separately"
    assert composite_buffer == 2, "mean_std_rms should use 2 buffer slots"
    assert individual_buffer == 4, "mean+std+rms separately use 1+2+1=4 slots"


def test_extrema_vs_max_min_buffer_efficiency():
    """Test that extrema uses same buffer space as max+min combined."""
    composite = ["extrema"]
    individual = ["max", "min"]
    
    composite_buffer = summary_metrics.summaries_buffer_height(composite)
    individual_buffer = summary_metrics.summaries_buffer_height(individual)
    
    assert composite_buffer == individual_buffer, \
        "extrema should use same buffer as max+min (2 slots each)"
    assert composite_buffer == 2, "extrema should use 2 buffer slots"


def test_new_metrics_can_be_mixed_with_existing():
    """Test that new metrics work alongside existing ones."""
    requested = ["mean", "std", "max", "min", "extrema"]
    
    buffer_sizes = summary_metrics.buffer_sizes(requested)
    output_sizes = summary_metrics.output_sizes(requested)
    
    assert len(buffer_sizes) == 5
    assert len(output_sizes) == 5
    assert buffer_sizes == (1, 2, 1, 1, 2), \
        "Buffer sizes: mean=1, std=2, max=1, min=1, extrema=2"
    assert output_sizes == (1, 1, 1, 1, 2), \
        "Output sizes: mean=1, std=1, max=1, min=1, extrema=2"
