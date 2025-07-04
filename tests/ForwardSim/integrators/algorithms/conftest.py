import pytest
import numpy as np
from warnings import warn

def run_settings_dict(**kwargs):
    """Default settings for simulation runtime configuration."""
    settings = {
        'duration': 1.0,
        'warmup': 0.0,
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

    # Create a default driver pattern (100 zeros with occasional 1.0 pulses) if not overridden
    sample_count = 100
    drivers = np.zeros((system.num_drivers, sample_count), dtype=precision)

    if system.num_drivers > 0:
        drivers[:, 0::25] = system.precision(1.0)  # Set every 25th sample to 1.0

    default_initial_values = system.init_values
    default_parameters = system.parameters
    system.init_values.values_array.copy()
    # When testing input overrides from the user, we need to make sure that they
    # fit the system. This is very intense for test parametrization handling, and
    # is really just doing the job of a higher-level system component, but it's included
    # for convenience in testing loop functions with multiple systems.
    # TODO: Replicate this logic into the ODEIntegrator API, so that it's only used
    #  when testing lower-level (loop and down) logic).
    if 'initial_values' in kwargs:
        initial_values_edits = kwargs.pop('initial_values')
        if isinstance(initial_values_edits, np.ndarray):
            # Ensure a 1D array is provided
            assert initial_values_edits.ndim == 1, "Initial values must be a 1D array."
            required_length = system.init_values.n
            provided_length = len(initial_values_edits)
            new_initial = system.init_values.values_array.copy()
            if provided_length < required_length:
                new_initial[:provided_length] = initial_values_edits
            else:
                new_initial[:] = initial_values_edits[:required_length]
                if provided_length > required_length:
                    warn("Redundant initial values provided; extra values are discarded.", UserWarning)
            kwargs['initial_values'] = new_initial

    if 'parameters' in kwargs:
        parameters_edits = kwargs.pop('parameters')
        if isinstance(parameters_edits, np.ndarray):
            # Ensure a 1D array is provided
            assert parameters_edits.ndim == 1, "Parameters must be a 1D array."
            required_length = system.parameters.n
            provided_length = len(parameters_edits)
            new_params = system.parameters.values_array.copy()
            if provided_length < required_length:
                new_params[:provided_length] = parameters_edits
            else:
                new_params[:] = parameters_edits[:required_length]
                if provided_length > required_length:
                    warn("Redundant parameters provided; extra values are discarded.", UserWarning)
            kwargs['parameters'] = new_params

    if 'forcing_vectors' in kwargs:
        forcing_override = kwargs['forcing_vectors']
        if isinstance(forcing_override, np.ndarray):
            assert forcing_override.shape[
                       0] == system.num_drivers, "forcing_vectors override must have system.num_drivers rows."

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