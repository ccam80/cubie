import pytest
from CuMC.ForwardSim.integrators._utils import check_requested_timing_possible, convert_times_to_fixed_steps
from numpy.testing import assert_allclose

@pytest.fixture(scope='function')
def timing_parameters(request):
    """Fixture to provide timing parameters for tests, with optional overrides."""
    updates = request.param if hasattr(request, 'param') else {}

    timing_params = {'dt_min': 0.001,
                     'dt_max': 0.01,
                     'dt_save': 0.01,
                     'dt_summarise': 0.1}
    timing_params.update(updates)

    return timing_params

@pytest.mark.parametrize("timing_parameters, expected_error, expected_warning",
                         [
                             ({"dt_max": 0.0005}, (ValueError, "dt_max must be greater than or equal to dt_min"), None),
                             ({"dt_save": 0.0005}, (ValueError, "dt_save must be greater than or equal to"), None),
                             ({"dt_summarise": 0.005},
                              (ValueError, "dt_summarise must be greater than or equal to dt_save"),
                              None),
                             ({"dt_max": 0.02}, None, (UserWarning, "dt_max .* is greater than dt_save")),
                             ({}, None, None),  # Default parameters - no errors or warnings
                             ({"dt_min": 0.005, "dt_max": 0.01, "dt_save": 0.01, "dt_summarise": 0.1}, None, None),
                         ],
                         ids=[
                             "dt_max_less_than_dt_min",
                             "dt_save_less_than_dt_min",
                             "dt_summarise_less_than_dt_save",
                             "dt_max_greater_than_dt_save",
                             "default_parameters",
                             "valid_custom_parameters"
                         ],
                         indirect=True)
def test_timing_check(timing_parameters, expected_error, expected_warning):
    """Test to ensure that timing parameters are correctly set."""
    dt_min = timing_parameters['dt_min']
    dt_max = timing_parameters['dt_max']
    dt_save = timing_parameters['dt_save']
    dt_summarise = timing_parameters['dt_summarise']

    if expected_error is not None:
        with pytest.raises(expected_error[0], match=expected_error[1]):
            check_requested_timing_possible(dt_min, dt_max, dt_save, dt_summarise)
    elif expected_warning is not None:
        with pytest.warns(expected_warning[0], match=expected_warning[1]):
            check_requested_timing_possible(dt_min, dt_max, dt_save, dt_summarise)
    else:
        check_requested_timing_possible(dt_min, dt_max, dt_save, dt_summarise)


@pytest.mark.parametrize("timing_parameters, expected_answer, test_name",
                         [
                             # Exact conversions
                             ({}, (10, 10, 0.01, 0.1), "exact_conversion"),
                             # Inexact conversions that need rounding
                             ({"dt_min": 0.003}, (3, 10, 0.009, 0.09), "inexact_conversion"),
                             # Edge cases
                             ({"dt_min": 0.01, "dt_summarise": 0.01}, (1, 1, 0.01, 0.01), "minimum_steps"),
                             ({"dt_min": 0.0001, "dt_save": 0.1, "dt_summarise": 1.0}, (1000, 10, 0.1, 1.0),
                              "large_step_counts"),
                             # Different ratios between save and summarize
                             ({"dt_summarise": 0.05}, (10, 5, 0.01, 0.05), "smaller_summarize_ratio"),
                             ({"dt_save": 0.005}, (5, 20, 0.005, 0.1), "larger_summarize_ratio"),
                         ],
                         ids=lambda x: x if isinstance(x, str) else "",
                         indirect=["timing_parameters"])
def test_convert_times_to_fixed_steps(timing_parameters, expected_answer, test_name):
    """Test that the timing parameters can be converted to fixed steps."""
    internal_step_size = timing_parameters['dt_min']
    dt_save = timing_parameters['dt_save']
    dt_summarise = timing_parameters['dt_summarise']

    result_tuple = convert_times_to_fixed_steps(internal_step_size, dt_save, dt_summarise)
    assert_allclose(result_tuple, expected_answer,
                    err_msg="Expected {}, got {}".format(expected_answer, result_tuple))
