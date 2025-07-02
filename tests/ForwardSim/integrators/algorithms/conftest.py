import pytest
import numpy as np


def run_settings_dict(**kwargs):
    """Default settings for simulation runtime configuration."""
    settings = {
        'duration': 1.0,
        'warmup': 0.5,
    }
    settings.update(kwargs)
    return settings


@pytest.fixture(scope='function')
def run_settings_override(request):
    """Override for run settings, if provided."""
    return request.param if hasattr(request, 'param') else {}


@pytest.fixture(scope='function')
def run_settings(run_settings_override):
    """
    Create runtime settings with defaults and potential overrides.

    Usage:
    @pytest.mark.parametrize("run_settings_override",
                           [{'duration': 20.0, 'warmup': 10.0}],
                           indirect=True)
    def test_something(run_settings):
        # run_settings will have duration=20.0 and warmup=10.0
    """
    return run_settings_dict(**run_settings_override)


def inputs_dict(system, precision, **kwargs):
    """Default input configuration for a system with the ability to override."""
    # Create a default driver pattern (zeros with occasional 1.0 pulses)
    sample_count = 100
    drivers = np.zeros((system.num_drivers, sample_count), dtype=precision)
    if system.num_drivers > 0:
        drivers[:, 0::25] = system.precision(1.0)  # Set every 25th sample to 1.0

    inputs_config = {
        'initial_values': system.init_values.values_array.copy(),
        'parameters': system.parameters.values_array.copy(),
        'forcing_vectors': drivers,
    }
    inputs_config.update(kwargs)
    return inputs_config


@pytest.fixture(scope='function')
def inputs_override(request):
    """Override for inputs, if provided."""
    return request.param if hasattr(request, 'param') else {}


@pytest.fixture(scope='function')
def inputs(system, precision, inputs_override):
    """
    Create inputs for a system with defaults and potential overrides.

    Usage:
    @pytest.mark.parametrize("inputs_override",
                           [{'parameters': custom_params}],
                           indirect=True)
    def test_something(inputs):
        # inputs will have custom parameters but default initial values and drivers
    """
    return inputs_dict(system, precision, **inputs_override)