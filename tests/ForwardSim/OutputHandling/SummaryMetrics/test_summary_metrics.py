"""
Tests for the SummaryMetrics class in the CuMC integrator system.
Tests tuple return values, parameter parsing, error handling, and tuple ordering.
"""

import pytest
import warnings
from CuMC.ForwardSim.OutputHandling.SummaryMetrics.metrics import SummaryMetrics, SummaryMetric
from CuMC.ForwardSim.OutputHandling import summary_metrics


@pytest.fixture(scope="function")
def empty_metrics():
    """Create an empty SummaryMetrics instance for testing."""
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
    def mock_save(buffer, output_array, summarise_every, customisable_variable):
        """Mock save function that does nothing."""
        pass
    return mock_update, mock_save


@pytest.fixture(scope="function")
def mock_metric_settings(request):
    """Create mock settings for testing."""
    defaults = {'buffer_size': 5,
                'output_size': 3,
                'name': 'mock_metric'}
    if hasattr(request, 'param'):
        defaults.update(request.param)

    return defaults


@pytest.fixture(scope="function")
def mock_metric(mock_functions, mock_metric_settings):
    """Create a mock SummaryMetric instance."""
    update_func, save_func = mock_functions
    name = mock_metric_settings['name']
    buffer_size = mock_metric_settings['buffer_size']
    output_size = mock_metric_settings['output_size']

    return SummaryMetric(
        name=name,
        buffer_size=buffer_size,
        output_size=output_size,
        update_device_func=update_func,
        save_device_func=save_func
    )


@pytest.fixture(scope="function")
def mock_parametrized_metric(mock_functions, mock_metric_settings):
    """Create a mock SummaryMetric instance."""
    update_func, save_func = mock_functions
    name = 'parameterised'
    def buffer_size(param):
        return 2*param

    def output_size(param):
        return 2 * param

    return SummaryMetric(
            name=name,
            buffer_size=buffer_size,
            output_size=output_size,
            update_device_func=update_func,
            save_device_func=save_func
            )


@pytest.fixture(scope="function")
def mock_metrics(empty_metrics, mock_metric, mock_parametrized_metric):
    """Create a SummaryMetrics instance with mock metrics registered."""
    empty_metrics.register_metric(mock_metric)
    empty_metrics.register_metric(mock_parametrized_metric)
    return empty_metrics


def test_register_metrics_success(empty_metrics, mock_metric, mock_parametrized_metric):
    """Create a SummaryMetrics instance with sample metrics registered."""
    empty_metrics.register_metric(mock_metric)
    empty_metrics.register_metric(mock_parametrized_metric)

    assert "mock_metric" in empty_metrics.implemented_metrics
    assert "parameterised" in empty_metrics.implemented_metrics
    assert empty_metrics._buffer_sizes["mock_metric"] == 5
    assert empty_metrics._output_sizes["mock_metric"] == 3


def test_register_metric_duplicate_name_raises_error(empty_metrics, mock_functions):
    """Test that registering a metric with duplicate name raises ValueError."""
    update, save = mock_functions
    metric1 = SummaryMetric(name="duplicate", buffer_size=5, output_size=3, save_device_func=save, update_device_func=update)
    metric2 = SummaryMetric(name="duplicate", buffer_size=10, output_size=6, save_device_func=save, update_device_func=update)

    empty_metrics.register_metric(metric1)

    with pytest.raises(ValueError, match="Metric 'duplicate' is already registered"):
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
    requested = ["parameterised[3]", "mock_metric"]  # Intentionally different order

    buffer_offsets_tuple = mock_metrics.buffer_offsets(requested)
    output_offsets_tuple = mock_metrics.output_offsets(requested)
    output_total_size = mock_metrics.summaries_output_height(requested)
    buffer_sizes_tuple = mock_metrics.buffer_sizes(requested)
    output_sizes_tuple = mock_metrics.output_sizes(requested)

    # All tuples should have the same length
    assert len(buffer_offsets_tuple) == len(output_offsets_tuple) == len(buffer_sizes_tuple) == len(output_sizes_tuple)

    # Check that the ordering is consistent by verifying specific values
    # Since we requested ["parameterised[3]", "mock_metric"], we should get values in that order
    assert buffer_sizes_tuple == (6, 5)  # parameterised[3]=2*3=6, mock_metric=5
    assert output_sizes_tuple == (6, 3)  # parameterised[3]=2*3=6, mock_metric=3
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

    with pytest.warns(UserWarning, match="Metric 'parameterised' has a callable size"):
        mock_metrics.buffer_sizes(requested)


def test_parametrized_metric_with_invalid_parameter_raises_error(mock_metrics):
    """Test that parametrized metric with invalid parameter raises ValueError."""
    requested = ["parameterised[invalid]"]

    with pytest.raises(ValueError, match="Parameter in 'parameterised\\[invalid\\]' must be an integer"):
        mock_metrics.buffer_sizes(requested)


def test_invalid_metric_name_raises_warning(mock_metrics):
    """Test that invalid metric name raises a warning."""
    requested = ["nonexistent_metric"]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mock_metrics.buffer_sizes(requested)

        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
        assert "Metric 'nonexistent_metric' is not registered" in str(w[0].message)


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
        assert buffer_sizes_tuple == (5, 6)  # mock_metric=5, parameterised[3]=2*3=6


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
    result = mock_metrics.parse_string_for_params(["metric1[3]", "metric2[7]", "metric3"])
    assert result == ["metric1", "metric2", "metric3"]
    assert mock_metrics._params["metric1"] == 3
    assert mock_metrics._params["metric2"] == 7
    assert mock_metrics._params["metric3"] == 0


def test_parse_string_for_params_invalid_formats(mock_metrics):
    """Test parameter parsing with invalid formats."""
    # Test non-integer parameter
    with pytest.raises(ValueError, match="Parameter in 'metric\\[abc\\]' must be an integer"):
        mock_metrics.parse_string_for_params(["metric[abc]"])

    # Test float parameter (should fail as it expects integer)
    with pytest.raises(ValueError, match="Parameter in 'metric\\[3.14\\]' must be an integer"):
        mock_metrics.parse_string_for_params(["metric[3.14]"])


def test_save_functions_returns_correct_tuple(mock_metrics, mock_functions):
    """Test that save_functions returns tuple with correct function objects."""
    requested = ["mock_metric", "parameterised[3]"]
    update_func, save_func = mock_functions

    # Set up the save functions dictionary
    mock_metrics._save_functions["mock_metric"] = save_func
    mock_metrics._save_functions["parameterised"] = save_func

    functions_tuple = mock_metrics.save_functions(requested)

    assert len(functions_tuple) == 2
    assert functions_tuple[0] == save_func
    assert functions_tuple[1] == save_func


def test_update_functions_returns_correct_tuple(mock_metrics, mock_functions):
    """Test that update_functions returns tuple with correct function objects."""
    requested = ["mock_metric", "parameterised[3]"]
    update_func, save_func = mock_functions

    # Set up the update functions dictionary
    mock_metrics._update_functions["mock_metric"] = update_func
    mock_metrics._update_functions["parameterised"] = update_func

    functions_tuple = mock_metrics.update_functions(requested)

    assert len(functions_tuple) == 2
    assert functions_tuple[0] == update_func
    assert functions_tuple[1] == update_func


def test_complex_parameter_scenarios(mock_metrics, mock_functions):
    """Test complex scenarios with multiple parametrized metrics."""
    # Create and register the complex metric with parametrized buffer_size
    update_func, save_func = mock_functions

    def complex_buffer_size(param):
        if param is None:
            raise ValueError("Parameter required")
        return param + 10

    complex_metric = SummaryMetric(
        name="complex",
        buffer_size=complex_buffer_size,
        output_size=5,
        update_device_func=update_func,
        save_device_func=save_func
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


def test_real_summary_metrics_available_metrics(real_metrics):
    """Test that all expected metrics are available in the real summary_metrics instance."""
    expected_metrics = ["mean", "max", "rms", "peaks"]
    available_metrics = real_metrics.implemented_metrics

    # Check that all expected metrics are present
    for metric in expected_metrics:
        assert metric in available_metrics, f"Expected metric '{metric}' not found in {available_metrics}"

    # Check that we have exactly the expected metrics
    assert len(available_metrics) == len(expected_metrics), f"Expected {len(expected_metrics)} metrics, got {len(available_metrics)}"


def test_real_summary_metrics_simple_metrics_sizes(real_metrics):
    """Test that simple metrics (mean, max, rms) have expected sizes."""
    simple_metrics = ["mean", "max", "rms"]

    for metric in simple_metrics:
        # Test buffer sizes
        buffer_sizes_tuple = tuple(real_metrics.buffer_sizes([metric]))
        assert buffer_sizes_tuple == (1,), f"Expected buffer_size=1 for {metric}, got {buffer_sizes_tuple}"

        # Test output sizes
        output_sizes_tuple = tuple(real_metrics.output_sizes([metric]))
        assert output_sizes_tuple == (1,), f"Expected output_size=1 for {metric}, got {output_sizes_tuple}"
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
        assert buffer_sizes_tuple == (expected_buffer_size,), f"Expected buffer_size={expected_buffer_size} for peaks[{n}], got {buffer_sizes_tuple}"

        # Test output sizes: should be n
        output_sizes_tuple = tuple(real_metrics.output_sizes([metric_request]))
        expected_output_size = n
        assert output_sizes_tuple == (expected_output_size,), f"Expected output_size={expected_output_size} for peaks[{n}], got {output_sizes_tuple}"


def test_real_summary_metrics_peaks_without_parameter_raises_warning(real_metrics):
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
    requested = ["rms", "peaks[4]", "mean"]  # Different order than registration

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
    headings = real_metrics.column_headings(single_metrics)

    # For single output metrics, headings should be identical to metric names
    assert headings == single_metrics

    # Test with a multi-output metric (peaks)
    peak_request = ["peaks[3]"]
    peak_headings = real_metrics.column_headings(peak_request)

    # Should have 3 column headers: peaks_1, peaks_2, peaks_3
    assert peak_headings == ["peaks_1", "peaks_2", "peaks_3"]

    # Test with a mix of single and multi-output metrics
    mixed_request = ["mean", "peaks[2]", "max"]
    mixed_headings = real_metrics.column_headings(mixed_request)

    # Should have 4 column headers: mean, peaks_1, peaks_2, max
    assert mixed_headings == ["mean", "peaks_1", "peaks_2", "max"]

    # Test with invalid metric name
    with warnings.catch_warnings(record=True) as w:
        invalid_headings = real_metrics.column_headings(["not_a_metric"])
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
