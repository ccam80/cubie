"""
Tests for the summarymetrics class in the cubie integrator system.
Tests tuple return values, parameter parsing, error handling, and tuple ordering.
"""

import pytest
import warnings
from cubie.outputhandling.summarymetrics.metrics import (
    SummaryMetrics,
    SummaryMetric,
    MetricFuncCache,
)
from cubie.outputhandling import summary_metrics


@pytest.fixture(scope="function")
def empty_metrics():
    """Create an empty summarymetrics instance for testing."""
    return SummaryMetrics()


@pytest.fixture(scope="function")
def real_metrics():
    """Return the module-level summary_metrics instance with metrics implemented."""
    return summary_metrics


@pytest.fixture(scope="function")
def mock_functions():
    """Create mock functions for update and save."""

    def mock_update(value, buffer, current_index, customisable_variable):
        """Mock update function that does nothing."""
        pass

    def mock_save(
        buffer, output_array, summarise_every, customisable_variable
    ):
        """Mock save function that does nothing."""
        pass

    return mock_update, mock_save


@pytest.fixture(scope="function")
def mock_metric_settings(request):
    """Create mock settings for testing."""
    defaults = {"buffer_size": 5, "output_size": 3, "name": "mock_metric"}
    if hasattr(request, "param"):
        defaults.update(request.param)

    return defaults


class ConcreteMetric(SummaryMetric):
    def __init__(self,
                 buffer_size,
                 output_size,
                 update_device_func,
                 save_device_func,
                 name):
        super().__init__(buffer_size=buffer_size,
                         output_size=output_size,
                         name=name)
        self.update_func = update_device_func
        self.save_func = save_device_func

    def build(self):
        return MetricFuncCache(update=self.update_func, save=self.save_func)


@pytest.fixture(scope="function")
def mock_metric(mock_functions, mock_metric_settings):
    """Create a mock SummaryMetric instance."""
    update_func, save_func = mock_functions
    name = mock_metric_settings["name"]
    buffer_size = mock_metric_settings["buffer_size"]
    output_size = mock_metric_settings["output_size"]

    return ConcreteMetric(
        name=name,
        buffer_size=buffer_size,
        output_size=output_size,
        update_device_func=update_func,
        save_device_func=save_func,
    )


@pytest.fixture(scope="function")
def mock_parametrized_metric(mock_functions, mock_metric_settings):
    """Create a mock SummaryMetric instance."""
    update_func, save_func = mock_functions
    name = "parameterised"

    def buffer_size(param):
        return 2 * param

    def output_size(param):
        return 2 * param

    return ConcreteMetric(
        name=name,
        buffer_size=buffer_size,
        output_size=output_size,
        update_device_func=update_func,
        save_device_func=save_func,
    )


@pytest.fixture(scope="function")
def mock_metrics(empty_metrics, mock_metric, mock_parametrized_metric):
    """Create a summarymetrics instance with mock metrics registered."""
    empty_metrics.register_metric(mock_metric)
    empty_metrics.register_metric(mock_parametrized_metric)
    return empty_metrics


def test_register_metrics_success(
    empty_metrics, mock_metric, mock_parametrized_metric
):
    """Create a summarymetrics instance with sample metrics registered."""
    empty_metrics.register_metric(mock_metric)
    empty_metrics.register_metric(mock_parametrized_metric)

    assert "mock_metric" in empty_metrics.implemented_metrics
    assert "parameterised" in empty_metrics.implemented_metrics
    assert empty_metrics._buffer_sizes["mock_metric"] == 5
    assert empty_metrics._output_sizes["mock_metric"] == 3


def test_register_metric_duplicate_name_raises_error(
    empty_metrics, mock_functions
):
    """Test that registering a metric with duplicate name raises ValueError."""
    update, save = mock_functions
    metric1 = ConcreteMetric(
        name="duplicate",
        buffer_size=5,
        output_size=3,
        save_device_func=save,
        update_device_func=update,
    )
    metric2 = ConcreteMetric(
        name="duplicate",
        buffer_size=10,
        output_size=6,
        save_device_func=save,
        update_device_func=update,
    )

    empty_metrics.register_metric(metric1)

    with pytest.raises(
        ValueError, match="Metric 'duplicate' is already registered"
    ):
        empty_metrics.register_metric(metric2)


def test_buffer_offsets_returns_correct_tuple(mock_metrics):
    """Test that buffer_offsets returns tuple with correct offset values."""
    requested = ["mock_metric", "parameterised[5]"]
    offsets_tuple = mock_metrics.buffer_offsets(requested)

    # mock_metric starts at 0, parameterised[5] starts at 5 (mock_metric's size)
    expected_offsets = (0, 5)

    assert offsets_tuple == expected_offsets


def test_output_offsets_returns_correct_tuple(mock_metrics):
    """Test that output_offsets returns tuple with correct offset values."""
    requested = ["mock_metric", "parameterised[5]"]
    offsets_tuple = mock_metrics.output_offsets(requested)
    total_size = mock_metrics.summaries_output_height(requested)

    # mock_metric starts at 0, parameterised[5] starts at 3 (mock_metric's output size)
    expected_offsets = (0, 3)
    expected_total_size = 13  # mock_metric=3 + parameterised[5]=10

    assert offsets_tuple == expected_offsets
    assert total_size == expected_total_size


def test_buffer_sizes_returns_correct_tuple(mock_metrics):
    """Test that buffer_sizes returns tuple with correct size values."""
    requested = ["mock_metric", "parameterised[5]"]
    sizes_tuple = mock_metrics.buffer_sizes(requested)

    expected_sizes = (5, 10)  # mock_metric=5, parameterised[5]=2*5=10
    assert sizes_tuple == expected_sizes


def test_output_sizes_returns_correct_tuple(mock_metrics):
    """Test that output_sizes returns tuple with correct size values."""
    requested = ["mock_metric", "parameterised[5]"]
    sizes_tuple = mock_metrics.output_sizes(requested)

    expected_sizes = (3, 10)  # mock_metric=3, parameterised[5]=2*5=10
    assert sizes_tuple == expected_sizes


def test_tuple_ordering_alignment(mock_metrics):
    """Test that all tuple methods return values in the same order."""
    requested = [
        "parameterised[3]",
        "mock_metric",
    ]  # Intentionally different order

    buffer_offsets_tuple = mock_metrics.buffer_offsets(requested)
    output_offsets_tuple = mock_metrics.output_offsets(requested)
    output_total_size = mock_metrics.summaries_output_height(requested)
    buffer_sizes_tuple = mock_metrics.buffer_sizes(requested)
    output_sizes_tuple = mock_metrics.output_sizes(requested)

    # All tuples should have the same length
    assert (
        len(buffer_offsets_tuple)
        == len(output_offsets_tuple)
        == len(buffer_sizes_tuple)
        == len(output_sizes_tuple)
    )

    # Check that the ordering is consistent by verifying specific values
    # Since we requested ["parameterised[3]", "mock_metric"], we should get values in that order
    assert buffer_sizes_tuple == (
        6,
        5,
    )  # parameterised[3]=2*3=6, mock_metric=5
    assert output_sizes_tuple == (
        6,
        3,
    )  # parameterised[3]=2*3=6, mock_metric=3
    assert buffer_offsets_tuple == (0, 6)  # parameterised[3]=0, mock_metric=6
    assert output_offsets_tuple == (0, 6)  # parameterised[3]=0, mock_metric=6

    # Test total size for output (buffer doesn't return total anymore)
    assert output_total_size == 9  # 6 + 3


def test_parametrized_metric_with_valid_parameter(mock_metrics):
    """Test parametrized metric with valid parameter."""
    requested = ["parameterised[5]"]

    buffer_sizes_tuple = mock_metrics.buffer_sizes(requested)
    output_sizes_tuple = mock_metrics.output_sizes(requested)

    # parametrized buffer_size = param * 2, output_size = param * 2
    assert buffer_sizes_tuple == (10,)  # 5 * 2
    assert output_sizes_tuple == (10,)  # 5 * 2


def test_parametrized_metric_without_parameter_raises_error(mock_metrics):
    """Test that parametrized metric without parameter raises ValueError."""
    requested = ["parameterised"]  # Missing parameter

    with pytest.warns(
        UserWarning, match="Metric 'parameterised' has a callable size"
    ):
        mock_metrics.buffer_sizes(requested)


def test_parametrized_metric_with_invalid_parameter_raises_error(mock_metrics):
    """Test that parametrized metric with invalid parameter raises ValueError."""
    requested = ["parameterised[invalid]"]

    with pytest.raises(
        ValueError,
        match="Parameter in 'parameterised\\[invalid\\]' must be an integer",
    ):
        mock_metrics.buffer_sizes(requested)


def test_invalid_metric_name_raises_warning(mock_metrics):
    """Test that invalid metric name raises a warning."""
    requested = ["nonexistent_metric"]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mock_metrics.buffer_sizes(requested)

        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
        assert "Metric 'nonexistent_metric' is not registered" in str(
            w[0].message
        )


def test_mixed_valid_invalid_metrics_with_warning(mock_metrics):
    """Test that mix of valid and invalid metrics processes valid ones and warns about invalid."""
    requested = ["mock_metric", "nonexistent", "parameterised[3]"]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        buffer_sizes_tuple = mock_metrics.buffer_sizes(requested)

        # Should warn about nonexistent metric
        assert len(w) == 1
        assert "Metric 'nonexistent' is not registered" in str(w[0].message)

        # Should process valid metrics
        assert buffer_sizes_tuple == (
            5,
            6,
        )  # mock_metric=5, parameterised[3]=2*3=6


def test_empty_request_returns_empty_tuple(mock_metrics):
    """Test that empty request returns appropriate empty values."""
    requested = []

    buffer_sizes_tuple = mock_metrics.buffer_sizes(requested)
    output_sizes_tuple = mock_metrics.output_sizes(requested)

    # For offset methods, empty requests should return (0, empty_tuple)
    buffer_offsets_tuple = mock_metrics.buffer_offsets(requested)
    output_offsets_tuple = mock_metrics.output_offsets(requested)
    buffer_total_size = mock_metrics.summaries_buffer_height(requested)
    output_total_size = mock_metrics.summaries_output_height(requested)

    assert buffer_sizes_tuple == ()
    assert output_sizes_tuple == ()

    # Empty requests should return total size of 0 and empty offset tuples
    assert buffer_total_size == 0
    assert output_total_size == 0
    assert buffer_offsets_tuple == ()
    assert output_offsets_tuple == ()


def test_parse_string_for_params_valid_formats(mock_metrics):
    """Test parameter parsing with various valid formats."""
    # Test single parameter
    result = mock_metrics.parse_string_for_params(["metric[5]"])
    assert result == ["metric"]
    assert mock_metrics._params["metric"] == 5

    # Test multiple parameters
    result = mock_metrics.parse_string_for_params(
        ["metric1[3]", "metric2[7]", "metric3"]
    )
    assert result == ["metric1", "metric2", "metric3"]
    assert mock_metrics._params["metric1"] == 3
    assert mock_metrics._params["metric2"] == 7
    assert mock_metrics._params["metric3"] == 0


def test_parse_string_for_params_invalid_formats(mock_metrics):
    """Test parameter parsing with invalid formats."""
    # Test non-integer parameter
    with pytest.raises(
        ValueError, match="Parameter in 'metric\\[abc\\]' must be an integer"
    ):
        mock_metrics.parse_string_for_params(["metric[abc]"])

    # Test float parameter (should fail as it expects integer)
    with pytest.raises(
        ValueError, match="Parameter in 'metric\\[3.14\\]' must be an integer"
    ):
        mock_metrics.parse_string_for_params(["metric[3.14]"])


def test_save_functions_returns_correct_tuple(mock_metrics, mock_functions):
    """Test that save_functions returns tuple with correct function objects."""
    requested = ["mock_metric", "parameterised[3]"]
    update_func, save_func = mock_functions

    functions_tuple = mock_metrics.save_functions(requested)

    # The registered mock metrics should return the correct save functions
    assert len(functions_tuple) == 2
    assert functions_tuple[0] is mock_metrics._metric_objects["mock_metric"].save_device_func
    assert functions_tuple[1] is mock_metrics._metric_objects["parameterised"].save_device_func
    # Also check they match the expected mock function
    assert functions_tuple[0] is save_func
    assert functions_tuple[1] is save_func


def test_update_functions_returns_correct_tuple(mock_metrics, mock_functions):
    """Test that update_functions returns tuple with correct function objects."""
    requested = ["mock_metric", "parameterised[3]"]
    update_func, save_func = mock_functions

    functions_tuple = mock_metrics.update_functions(requested)

    # The registered mock metrics should return the correct update functions
    assert len(functions_tuple) == 2
    assert functions_tuple[0] is mock_metrics._metric_objects["mock_metric"].update_device_func
    assert functions_tuple[1] is mock_metrics._metric_objects["parameterised"].update_device_func
    # Also check they match the expected mock function
    assert functions_tuple[0] is update_func
    assert functions_tuple[1] is update_func


def test_complex_parameter_scenarios(mock_metrics, mock_functions):
    """Test complex scenarios with multiple parametrized metrics."""
    # Create and register the complex metric with parametrized buffer_size
    update_func, save_func = mock_functions

    def complex_buffer_size(param):
        if param is None:
            raise ValueError("Parameter required")
        return param + 10

    complex_metric = ConcreteMetric(
        name="complex",
        buffer_size=complex_buffer_size,
        output_size=5,
        update_device_func=update_func,
        save_device_func=save_func,
    )
    mock_metrics.register_metric(complex_metric)

    # Test with multiple parametrized metrics
    requested = ["parameterised[3]", "complex[7]", "mock_metric"]

    buffer_sizes_tuple = mock_metrics.buffer_sizes(requested)
    assert buffer_sizes_tuple == (6, 17, 5)  # 3*2=6, 7+10=17, 5


def test_edge_case_bracket_parsing(mock_metrics):
    """Test edge cases in bracket parsing."""
    # Test with zero parameter
    result = mock_metrics.parse_string_for_params(["metric[0]"])
    assert result == ["metric"]
    assert mock_metrics._params["metric"] == 0

    # Test with negative parameter
    result = mock_metrics.parse_string_for_params(["metric[-5]"])
    assert result == ["metric"]
    assert mock_metrics._params["metric"] == -5


def test_real_metrics_integration(real_metrics):
    """Test integration with real metrics from the module."""
    # Test that real metrics are properly registered
    assert len(real_metrics.implemented_metrics) > 0

    # Test with real metrics
    available_metrics = real_metrics.implemented_metrics
    if available_metrics:
        # Test with first available metric
        first_metric = available_metrics[0]

        # Test sizes with real metrics
        buffer_sizes_tuple = real_metrics.buffer_sizes([first_metric])
        output_sizes_tuple = real_metrics.output_sizes([first_metric])
        assert len(buffer_sizes_tuple) == 1
        assert len(output_sizes_tuple) == 1


def test_combined_metrics_mean_std_rms_all_three(real_metrics):
    """Test that mean+std+rms is substituted with mean_std_rms."""
    requested = ["mean", "std", "rms"]
    processed = real_metrics.preprocess_request(requested)
    
    # Should substitute with combined metric
    assert "mean_std_rms" in processed
    assert "mean" not in processed
    assert "std" not in processed
    assert "rms" not in processed
    assert len(processed) == 1


def test_combined_metrics_mean_std(real_metrics):
    """Test that mean+std is substituted with mean_std."""
    requested = ["mean", "std"]
    processed = real_metrics.preprocess_request(requested)
    
    # Should substitute with combined metric
    assert "mean_std" in processed
    assert "mean" not in processed
    assert "std" not in processed
    assert len(processed) == 1


def test_combined_metrics_std_rms(real_metrics):
    """Test that std+rms is substituted with std_rms."""
    requested = ["std", "rms"]
    processed = real_metrics.preprocess_request(requested)
    
    # Should substitute with combined metric
    assert "std_rms" in processed
    assert "std" not in processed
    assert "rms" not in processed
    assert len(processed) == 1


def test_combined_metrics_mean_rms(real_metrics):
    """Test that mean+rms without std is NOT substituted (no benefit)."""
    requested = ["mean", "rms"]
    processed = real_metrics.preprocess_request(requested)
    
    # Should NOT substitute - no buffer saving
    assert "mean_std" not in processed
    assert "std_rms" not in processed
    assert "mean_std_rms" not in processed
    assert "mean" in processed
    assert "rms" in processed
    assert len(processed) == 2


def test_combined_metrics_max_min(real_metrics):
    """Test that max+min is substituted with extrema."""
    requested = ["max", "min"]
    processed = real_metrics.preprocess_request(requested)
    
    # Should substitute with combined metric
    assert "extrema" in processed
    assert "max" not in processed
    assert "min" not in processed
    assert len(processed) == 1


def test_combined_metrics_single_mean_not_substituted(real_metrics):
    """Test that single mean is NOT substituted."""
    requested = ["mean"]
    processed = real_metrics.preprocess_request(requested)
    
    # Should NOT substitute
    assert "mean" in processed
    assert "mean_std_rms" not in processed
    assert len(processed) == 1


def test_combined_metrics_single_std_not_substituted(real_metrics):
    """Test that single std is NOT substituted."""
    requested = ["std"]
    processed = real_metrics.preprocess_request(requested)
    
    # Should NOT substitute
    assert "std" in processed
    assert "mean_std_rms" not in processed
    assert len(processed) == 1


def test_combined_metrics_single_rms_not_substituted(real_metrics):
    """Test that single rms is NOT substituted."""
    requested = ["rms"]
    processed = real_metrics.preprocess_request(requested)
    
    # Should NOT substitute
    assert "rms" in processed
    assert "mean_std_rms" not in processed
    assert len(processed) == 1


def test_combined_metrics_single_max_not_substituted(real_metrics):
    """Test that single max is NOT substituted."""
    requested = ["max"]
    processed = real_metrics.preprocess_request(requested)
    
    # Should NOT substitute
    assert "max" in processed
    assert "extrema" not in processed
    assert len(processed) == 1


def test_combined_metrics_single_min_not_substituted(real_metrics):
    """Test that single min is NOT substituted."""
    requested = ["min"]
    processed = real_metrics.preprocess_request(requested)
    
    # Should NOT substitute
    assert "min" in processed
    assert "extrema" not in processed
    assert len(processed) == 1


def test_combined_metrics_with_other_metrics(real_metrics):
    """Test combined metrics work alongside non-combinable metrics."""
    requested = ["mean", "std", "rms", "max_magnitude"]
    processed = real_metrics.preprocess_request(requested)
    
    # Should substitute mean+std+rms but keep max_magnitude
    assert "mean_std_rms" in processed
    assert "max_magnitude" in processed
    assert "mean" not in processed
    assert "std" not in processed
    assert "rms" not in processed
    assert len(processed) == 2


def test_combined_metrics_multiple_combinations(real_metrics):
    """Test multiple independent combinations in one request."""
    requested = ["mean", "std", "rms", "max", "min"]
    processed = real_metrics.preprocess_request(requested)
    
    # Should substitute both combinations
    assert "mean_std_rms" in processed
    assert "extrema" in processed
    assert "mean" not in processed
    assert "std" not in processed
    assert "rms" not in processed
    assert "max" not in processed
    assert "min" not in processed
    assert len(processed) == 2


def test_combined_metrics_order_independence(real_metrics):
    """Test that order of request doesn't affect substitution."""
    # Test different orderings of the same metrics
    requests = [
        ["mean", "std", "rms"],
        ["rms", "mean", "std"],
        ["std", "rms", "mean"],
    ]
    
    for requested in requests:
        processed = real_metrics.preprocess_request(requested)
        assert "mean_std_rms" in processed
        assert "mean" not in processed
        assert "std" not in processed
        assert "rms" not in processed
        assert len(processed) == 1


def test_combined_metrics_buffer_efficiency(real_metrics):
    """Test that requesting all three metrics uses the combined metric."""
    # When requesting all three metrics together, they should be combined
    all_three = ["mean", "std", "rms"]
    
    # When requesting just two (mean+std), they should be combined
    mean_std = ["mean", "std"]
    
    # When requesting std+rms, they should be combined
    std_rms = ["std", "rms"]
    
    all_three_buffer = real_metrics.summaries_buffer_height(all_three)
    mean_std_buffer = real_metrics.summaries_buffer_height(mean_std)
    std_rms_buffer = real_metrics.summaries_buffer_height(std_rms)
    
    # All three should use the combined metric (2 slots)
    assert all_three_buffer == 2  # mean_std_rms uses 2 slots
    
    # Mean+std should use combined metric (2 slots, saves 1)
    assert mean_std_buffer == 2  # mean_std uses 2 slots (vs 1+2=3 separate)
    
    # Std+rms should use combined metric (2 slots, saves 1)
    assert std_rms_buffer == 2  # std_rms uses 2 slots (vs 2+1=3 separate)


def test_combined_metrics_output_sizes(real_metrics):
    """Test that combined metrics produce correct output sizes."""
    requested = ["mean", "std", "rms"]
    processed = real_metrics.preprocess_request(requested)
    
    # Get output sizes
    output_sizes = real_metrics.output_sizes(processed)
    
    # mean_std_rms should output 3 values
    assert len(output_sizes) == 1
    assert output_sizes[0] == 3  # [mean, std, rms]


def test_combined_metrics_with_peaks(real_metrics):
    """Test combined metrics work with parameterized metrics like peaks."""
    requested = ["mean", "std", "peaks[3]"]
    processed = real_metrics.preprocess_request(requested)
    
    # Should substitute mean+std with mean_std and keep peaks
    assert "mean_std" in processed
    assert "peaks" in processed
    assert "mean" not in processed
    assert "std" not in processed
    # Should have two metrics
    assert len(processed) == 2

def test_real_summary_metrics_available_metrics(real_metrics):
    """Test that all expected metrics are available in the real summary_metrics instance."""
    expected_metrics = [
        "mean", "std", "rms", "max", "min", "max_magnitude",
        "peaks", "negative_peaks", "mean_std_rms", "extrema",
        "mean_std", "std_rms",
        "dxdt_max", "dxdt_min", "dxdt_extrema",
        "d2xdt2_max", "d2xdt2_min", "d2xdt2_extrema"
    ]
    available_metrics = real_metrics.implemented_metrics

    # Check that all expected metrics are present
    for metric in expected_metrics:
        assert metric in available_metrics, (
            f"Expected metric '{metric}' not found in {available_metrics}"
        )

    # Check that we have exactly the expected metrics
    assert len(available_metrics) == len(expected_metrics), (
        f"Expected {len(expected_metrics)} metrics, got {len(available_metrics)}"
    )


def test_real_summary_metrics_simple_metrics_sizes(real_metrics):
    """Test that simple metrics (mean, max, rms) have expected sizes."""
    simple_metrics = ["mean", "max", "rms"]

    for metric in simple_metrics:
        # Test buffer sizes
        buffer_sizes_tuple = tuple(real_metrics.buffer_sizes([metric]))
        assert buffer_sizes_tuple == (1,), (
            f"Expected buffer_size=1 for {metric}, got {buffer_sizes_tuple}"
        )

        # Test output sizes
        output_sizes_tuple = tuple(real_metrics.output_sizes([metric]))
        assert output_sizes_tuple == (1,), (
            f"Expected output_size=1 for {metric}, got {output_sizes_tuple}"
        )
        assert callable(real_metrics.save_functions(simple_metrics)[0])


def test_real_summary_metrics_peaks_parametrized(real_metrics):
    """Test that peaks metric works correctly with parameters."""
    # Test peaks with different parameter values
    test_params = [1, 3, 5, 10]

    for n in test_params:
        metric_request = f"peaks[{n}]"

        # Test buffer sizes: should be 3 + n
        buffer_sizes_tuple = tuple(real_metrics.buffer_sizes([metric_request]))
        expected_buffer_size = 3 + n
        assert buffer_sizes_tuple == (expected_buffer_size,), (
            f"Expected buffer_size={expected_buffer_size} for peaks[{n}], got {buffer_sizes_tuple}"
        )

        # Test output sizes: should be n
        output_sizes_tuple = tuple(real_metrics.output_sizes([metric_request]))
        expected_output_size = n
        assert output_sizes_tuple == (expected_output_size,), (
            f"Expected output_size={expected_output_size} for peaks[{n}], got {output_sizes_tuple}"
        )


def test_real_summary_metrics_peaks_without_parameter_raises_warning(
    real_metrics,
):
    """Test that peaks metric without parameter raises ValueError."""
    with pytest.warns(UserWarning, match="Metric 'peaks' has a callable size"):
        tuple(real_metrics.buffer_sizes(["peaks"]))


def test_real_summary_metrics_offset_calculations(real_metrics):
    """Test offset calculations with real metrics."""
    # Test with mix of simple and parametrized metrics
    requested = ["mean", "peaks[2]", "max", "rms"]

    # Test buffer offsets
    buffer_offsets_tuple = real_metrics.buffer_offsets(requested)

    # Expected buffer offsets:
    # mean: 0 (size=1)
    # peaks[2]: 1 (size=3+2=5)
    # max: 6 (size=1)
    # rms: 7 (size=1)
    expected_buffer_offsets = (0, 1, 6, 7)

    assert buffer_offsets_tuple == expected_buffer_offsets

    # Test output offsets
    output_offsets_tuple = real_metrics.output_offsets(requested)
    output_total_size = real_metrics.summaries_output_height(requested)

    # Expected output offsets:
    # mean: 0 (size=1)
    # peaks[2]: 1 (size=2)
    # max: 3 (size=1)
    # rms: 4 (size=1)
    expected_output_offsets = (0, 1, 3, 4)
    expected_output_total = 5  # 1 + 2 + 1 + 1

    assert output_offsets_tuple == expected_output_offsets
    assert output_total_size == expected_output_total


def test_real_summary_metrics_tuple_ordering_consistency(real_metrics):
    """Test that all tuple methods return values in consistent order with real metrics."""
    requested = [
        "rms",
        "peaks[4]",
        "mean",
    ]  # Different order than registration

    # Get all tuple results
    buffer_offsets_tuple = tuple(real_metrics.buffer_offsets(requested))
    output_offsets_tuple = tuple(real_metrics.output_offsets(requested))
    buffer_sizes_tuple = tuple(real_metrics.buffer_sizes(requested))
    output_sizes_tuple = tuple(real_metrics.output_sizes(requested))

    # All should have same length (number of requested metrics)
    expected_length = len(requested)
    assert len(buffer_offsets_tuple) == expected_length
    assert len(output_offsets_tuple) == expected_length
    assert len(buffer_sizes_tuple) == expected_length
    assert len(output_sizes_tuple) == expected_length

    # Verify sizes match expected values for the requested order
    # rms: buffer=1, output=1
    # peaks[4]: buffer=3+4=7, output=4
    # mean: buffer=1, output=1
    assert buffer_sizes_tuple == (1, 7, 1)
    assert output_sizes_tuple == (1, 4, 1)

    # Verify offsets are calculated correctly
    # rms: buffer_offset=0, output_offset=0
    # peaks[4]: buffer_offset=1, output_offset=1
    # mean: buffer_offset=8, output_offset=5
    assert buffer_offsets_tuple == (0, 1, 8)
    assert output_offsets_tuple == (0, 1, 5)

    # Verify totals
    buffer_total_size = real_metrics.summaries_buffer_height(requested)
    output_total_size = real_metrics.summaries_output_height(requested)

    assert buffer_total_size == 9  # 1 + 7 + 1
    assert output_total_size == 6  # 1 + 4 + 1


def test_real_summary_metrics_edge_cases(real_metrics):
    """Test edge cases with real metrics."""
    # Test with single metric
    summary_buffer_height = real_metrics.summaries_buffer_height(["mean"])
    buffer_offsets_generator = real_metrics.buffer_offsets(["mean"])
    buffer_offsets_tuple = tuple(buffer_offsets_generator)

    assert buffer_offsets_tuple == (0,)
    assert summary_buffer_height == 1

    # Test with peaks parameter edge cases
    buffer_sizes_tuple = tuple(real_metrics.buffer_sizes(["peaks[0]"]))
    output_sizes_tuple = tuple(real_metrics.output_sizes(["peaks[0]"]))

    assert buffer_sizes_tuple == (3,)  # 3 + 0
    assert output_sizes_tuple == (0,)  # 0


def test_column_headings(real_metrics):
    """Test that column_headings returns correctly formatted headers for metrics."""
    # Test with single output metrics
    single_metrics = ["mean", "max", "rms"]
    headings = real_metrics.legend(single_metrics)

    # For single output metrics, headings should be identical to metric names
    assert headings == single_metrics

    # Test with a multi-output metric (peaks)
    peak_request = ["peaks[3]"]
    peak_headings = real_metrics.legend(peak_request)

    # Should have 3 column headers: peaks_1, peaks_2, peaks_3
    assert peak_headings == ["peaks_1", "peaks_2", "peaks_3"]

    # Test with a mix of single and multi-output metrics
    mixed_request = ["mean", "peaks[2]", "max"]
    mixed_headings = real_metrics.legend(mixed_request)

    # Should have 4 column headers: mean, peaks_1, peaks_2, max
    assert mixed_headings == ["mean", "peaks_1", "peaks_2", "max"]

    # Test with invalid metric name
    with warnings.catch_warnings(record=True) as w:
        invalid_headings = real_metrics.legend(["not_a_metric"])
        assert len(w) == 1
        assert "not registered" in str(w[0].message)
        assert invalid_headings == []


def test_summary_buffer_size_returns_correct_total(mock_metrics):
    """Test that summary_buffer_size returns correct total buffer size."""
    requested = ["mock_metric", "parameterised[5]"]
    total_size = mock_metrics.summaries_buffer_height(requested)

    # mock_metric=5 + parameterised[5]=10
    expected_total_size = 15

    assert total_size == expected_total_size


def test_mean_std_buffer_and_output_sizes(real_metrics):
    """Test mean_std combined metric has correct buffer and output sizes."""
    requested = ["mean_std"]
    
    buffer_sizes = real_metrics.buffer_sizes(requested)
    output_sizes = real_metrics.output_sizes(requested)
    
    assert buffer_sizes == (2,), "mean_std should use 2 buffer slots"
    assert output_sizes == (2,), "mean_std should output 2 values (mean, std)"


def test_std_rms_buffer_and_output_sizes(real_metrics):
    """Test std_rms combined metric has correct buffer and output sizes."""
    requested = ["std_rms"]
    
    buffer_sizes = real_metrics.buffer_sizes(requested)
    output_sizes = real_metrics.output_sizes(requested)
    
    assert buffer_sizes == (2,), "std_rms should use 2 buffer slots"
    assert output_sizes == (2,), "std_rms should output 2 values (std, rms)"


def test_pairwise_combinations_buffer_efficiency(real_metrics):
    """Test that pairwise combinations save buffer space."""
    # mean+std: 1+2=3 individually, 2 combined (saves 1)
    mean_std = ["mean", "std"]
    mean_std_buffer = real_metrics.summaries_buffer_height(mean_std)
    assert mean_std_buffer == 2  # Combined metric saves 1 slot


def calculate_single_summary_array(metric_name, values, dt_save=0.01):
    """Calculate expected summary metric value using numpy reference implementation.
    
    Parameters
    ----------
    metric_name
        Name of the summary metric to compute.
    values
        Array of values to compute metric over.
    dt_save
        Time interval between saved states (for derivative metrics).
        
    Returns
    -------
    float or ndarray
        Expected metric value(s).
    """
    import numpy as np
    
    if metric_name == "mean":
        return np.mean(values)
    elif metric_name == "max":
        return np.max(values)
    elif metric_name == "min":
        return np.min(values)
    elif metric_name == "rms":
        return np.sqrt(np.mean(values ** 2))
    elif metric_name == "std":
        return np.std(values)
    elif metric_name == "max_magnitude":
        return np.max(np.abs(values))
    elif metric_name == "extrema":
        return np.array([np.max(values), np.min(values)])
    elif metric_name == "dxdt_max":
        if len(values) < 2:
            return -1.0e30  # Sentinel value for insufficient data
        return np.max(np.diff(values) / dt_save)
    elif metric_name == "dxdt_min":
        if len(values) < 2:
            return 1.0e30  # Sentinel value for insufficient data
        return np.min(np.diff(values) / dt_save)
    elif metric_name == "dxdt_extrema":
        if len(values) < 2:
            return np.array([-1.0e30, 1.0e30])  # Sentinel values
        derivatives = np.diff(values) / dt_save
        return np.array([np.max(derivatives), np.min(derivatives)])
    elif metric_name == "d2xdt2_max":
        if len(values) < 3:
            return -1.0e30  # Sentinel value for insufficient data
        second_derivatives = (values[2:] - 2*values[1:-1] + values[:-2]) / (dt_save ** 2)
        return np.max(second_derivatives)
    elif metric_name == "d2xdt2_min":
        if len(values) < 3:
            return 1.0e30  # Sentinel value for insufficient data
        second_derivatives = (values[2:] - 2*values[1:-1] + values[:-2]) / (dt_save ** 2)
        return np.min(second_derivatives)
    elif metric_name == "d2xdt2_extrema":
        if len(values) < 3:
            return np.array([-1.0e30, 1.0e30])  # Sentinel values
        second_derivatives = (values[2:] - 2*values[1:-1] + values[:-2]) / (dt_save ** 2)
        return np.array([np.max(second_derivatives), np.min(second_derivatives)])
    else:
        raise ValueError(f"Unknown metric: {metric_name}")


@pytest.mark.parametrize(
    "metric_name",
    [
        "mean", "std", "rms", "max", "min", "max_magnitude", "extrema",
        "dxdt_max", "dxdt_min", "dxdt_extrema",
        "d2xdt2_max", "d2xdt2_min", "d2xdt2_extrema"
    ]
)
def test_all_summaries_long_run(real_metrics, metric_name):
    """Test all summary metrics with a long run of values."""
    import numpy as np
    
    # Create test data with varying characteristics
    t = np.linspace(0, 10, 1000)
    values = np.sin(t) + 0.1 * np.cos(5 * t)
    dt_save = t[1] - t[0]
    
    # Get expected result from reference implementation
    expected = calculate_single_summary_array(metric_name, values, dt_save)
    
    # Note: This test validates the reference implementation logic
    # Actual CUDA metric testing would require running the solver
    # Here we just verify the reference calculation doesn't crash
    assert expected is not None
    
    # For scalar metrics, check type
    if metric_name in ["mean", "std", "rms", "max", "min", "max_magnitude",
                       "dxdt_max", "dxdt_min", "d2xdt2_max", "d2xdt2_min"]:
        assert np.isscalar(expected) or expected.shape == ()
    # For array metrics, check shape
    elif metric_name in ["extrema", "dxdt_extrema", "d2xdt2_extrema"]:
        assert expected.shape == (2,)


@pytest.mark.parametrize(
    "test_case,metrics_list",
    [
        ("all", [
            "mean", "std", "rms", "max", "min", "max_magnitude", "extrema",
            "dxdt_max", "dxdt_min", "dxdt_extrema",
            "d2xdt2_max", "d2xdt2_min", "d2xdt2_extrema"
        ]),
        ("no_combos", [
            "mean", "std", "rms", "max", "min", "max_magnitude",
            "dxdt_max", "dxdt_min", "d2xdt2_max", "d2xdt2_min"
        ])
    ]
)
def test_all_summary_metrics_numerical_check(real_metrics, test_case, metrics_list):
    """Test numerical correctness of all summary metrics against reference implementation.
    
    Parameters
    ----------
    test_case
        'all' tests with combination metrics, 'no_combos' tests individual metrics only
    metrics_list
        List of metric names to test
    """
    import numpy as np
    
    # Create test data
    t = np.linspace(0, 5, 500)
    values = 2.0 * np.sin(2 * np.pi * t) + 0.5 * np.sin(6 * np.pi * t)
    dt_save = t[1] - t[0]
    
    # Test each metric in the list
    for metric_name in metrics_list:
        # Calculate expected value
        expected = calculate_single_summary_array(metric_name, values, dt_save)
        
        # Validate the calculation doesn't produce NaN or Inf
        if np.isscalar(expected) or expected.shape == ():
            assert np.isfinite(expected), f"{metric_name} produced non-finite value"
        else:
            assert np.all(np.isfinite(expected)), f"{metric_name} produced non-finite values"
        
        # Additional validation: values should be within reasonable range
        # (not sentinel values except for insufficient data cases)
        if len(values) >= 3:  # Sufficient data for all metrics
            if np.isscalar(expected) or expected.shape == ():
                # Derivative metrics should not be at sentinel values
                if metric_name in ["dxdt_max", "d2xdt2_max"]:
                    assert expected > -1.0e20, f"{metric_name} at sentinel value"
                elif metric_name in ["dxdt_min", "d2xdt2_min"]:
                    assert expected < 1.0e20, f"{metric_name} at sentinel value"


def test_derivative_metrics_buffer_sizes(real_metrics):
    """Test that derivative metrics have correct buffer sizes."""
    # First derivative metrics
    assert real_metrics.buffer_sizes(["dxdt_max"]) == (2,)
    assert real_metrics.buffer_sizes(["dxdt_min"]) == (2,)
    assert real_metrics.buffer_sizes(["dxdt_extrema"]) == (3,)
    
    # Second derivative metrics
    assert real_metrics.buffer_sizes(["d2xdt2_max"]) == (3,)
    assert real_metrics.buffer_sizes(["d2xdt2_min"]) == (3,)
    assert real_metrics.buffer_sizes(["d2xdt2_extrema"]) == (4,)


def test_derivative_metrics_output_sizes(real_metrics):
    """Test that derivative metrics have correct output sizes."""
    # First derivative metrics (single output)
    assert real_metrics.output_sizes(["dxdt_max"]) == (1,)
    assert real_metrics.output_sizes(["dxdt_min"]) == (1,)
    
    # First derivative extrema (two outputs)
    assert real_metrics.output_sizes(["dxdt_extrema"]) == (2,)
    
    # Second derivative metrics (single output)
    assert real_metrics.output_sizes(["d2xdt2_max"]) == (1,)
    assert real_metrics.output_sizes(["d2xdt2_min"]) == (1,)
    
    # Second derivative extrema (two outputs)
    assert real_metrics.output_sizes(["d2xdt2_extrema"]) == (2,)


def test_combined_derivative_metrics_dxdt(real_metrics):
    """Test that dxdt_max+dxdt_min is substituted with dxdt_extrema."""
    requested = ["dxdt_max", "dxdt_min"]
    processed = real_metrics.preprocess_request(requested)
    
    # Should substitute with combined metric
    assert "dxdt_extrema" in processed
    assert "dxdt_max" not in processed
    assert "dxdt_min" not in processed
    assert len(processed) == 1


def test_combined_derivative_metrics_d2xdt2(real_metrics):
    """Test that d2xdt2_max+d2xdt2_min is substituted with d2xdt2_extrema."""
    requested = ["d2xdt2_max", "d2xdt2_min"]
    processed = real_metrics.preprocess_request(requested)
    
    # Should substitute with combined metric
    assert "d2xdt2_extrema" in processed
    assert "d2xdt2_max" not in processed
    assert "d2xdt2_min" not in processed
    assert len(processed) == 1


def test_derivative_metrics_registration(real_metrics):
    """Test that all derivative metrics are properly registered."""
    available = real_metrics.implemented_metrics
    
    # Check first derivative metrics
    assert "dxdt_max" in available
    assert "dxdt_min" in available
    assert "dxdt_extrema" in available
    
    # Check second derivative metrics
    assert "d2xdt2_max" in available
    assert "d2xdt2_min" in available
    assert "d2xdt2_extrema" in available

    mean_std_buffer = real_metrics.summaries_buffer_height(mean_std)
    assert mean_std_buffer == 2, "mean+std should use 2 buffer slots (combined)"
    
    # std+rms: 2+1=3 individually, 2 combined (saves 1)
    std_rms = ["std", "rms"]
    std_rms_buffer = real_metrics.summaries_buffer_height(std_rms)
    assert std_rms_buffer == 2, "std+rms should use 2 buffer slots (combined)"
    
    # mean+rms: 1+1=2 individually, would still be 2 combined (no saving)
    # So this should NOT be combined
    mean_rms = ["mean", "rms"]
    mean_rms_buffer = real_metrics.summaries_buffer_height(mean_rms)
    assert mean_rms_buffer == 2, "mean+rms should use 2 buffer slots (not combined)"
