import pytest
from cubie.integrators.IntegratorRunSettings import IntegratorRunSettings


@pytest.mark.parametrize(
    "dt_min, dt_save, dt_summarise, should_warn",
    [
        (0.01, 0.1, 0.5, False),  # Valid configuration
        (0.01, 0.055, 0.11, True),  # dt_save not multiple of dt_min
        (0.01, 0.1, 0.35, True),  # dt_summarise not multiple of dt_save
        (0.01, 0.055, 0.152, True),  # Both not multiples
    ],
    ids=["valid_timing", "save_warning", "summarise_warning", "both_warnings"],
)
def test_timing_validation(dt_min, dt_save, dt_summarise, should_warn,
                           precision):
    """Test timing validation in fixed_steps property."""

    if should_warn:
        with pytest.warns(UserWarning):
            run_settings = IntegratorRunSettings(
                dt_min=dt_min, dt_save=dt_save, dt_summarise=dt_summarise
            )
    else:
        run_settings = IntegratorRunSettings(
            dt_min=dt_min, dt_save=dt_save, dt_summarise=dt_summarise
        )

    assert precision(run_settings.dt_min) == precision(dt_min)
    assert (
        precision(run_settings.dt_save)
        == precision(int(dt_save / dt_min) * dt_min)
    )
    assert (
        precision(run_settings.dt_summarise)
        == precision(int(dt_summarise / run_settings.dt_save) *
        run_settings.dt_save)
    )


@pytest.mark.parametrize(
    "dt_max, dt_min, dt_save, dt_summarise",
    [
        (0.01, 0.1, 0.05, 0.1),  # dt_max < dt_min
        (0.1, 0.01, 0.005, 0.1),  # dt_save < dt_min
        (0.1, 0.01, 0.1, 0.05),  # dt_summarise < dt_save
    ],
    ids=[
        "max_less_than_min",
        "save_less_than_min",
        "summarise_less_than_save",
    ],
)
def test_timing_validation_errors(dt_max, dt_min, dt_save, dt_summarise):
    """Test that invalid timing configurations raise errors."""
    with pytest.raises(ValueError):
        errored_settings = run_settings = IntegratorRunSettings(
            dt_min=dt_min,
            dt_max=dt_max,
            dt_save=dt_save,
            dt_summarise=dt_summarise,
        )
