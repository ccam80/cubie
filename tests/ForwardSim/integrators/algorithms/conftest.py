import pytest
import numpy as np
from warnings import warn
from CuMC.ForwardSim.integrators.IntegratorRunSettings import IntegratorRunSettings

@pytest.fixture(scope='function')
def run_settings_override(request):
    """Override for run settings, if provided."""
    return request.param if hasattr(request, 'param') else {}


@pytest.fixture(scope='function')
def run_settings(loop_compile_settings, run_settings_override):
    """Create LoopStepConfig from loop_compile_settings."""
    defaults = IntegratorRunSettings(
            dt_min=loop_compile_settings['dt_min'],
            dt_max=loop_compile_settings['dt_max'],
            dt_save=loop_compile_settings['dt_save'],
            dt_summarise=loop_compile_settings['dt_summarise'],
            atol=loop_compile_settings['atol'],
            rtol=loop_compile_settings['rtol'],
            duration= 1.0,
            warmup=0.0,
            )

    if run_settings_override:
        # Update defaults with any overrides provided
        for key, value in run_settings_override.items():
            if hasattr(defaults, key):
                setattr(defaults, key, value)
            else:
                warn(f"Unknown run setting '{key}' provided; ignoring.", UserWarning)

    return defaults

def inputs_dict(system, precision, **kwargs):
    """Default input configuration for a system with the ability to override."""

    # Create a default driver pattern (100 zeros with occasional 1.0 pulses) if not overridden
    sample_count = 100
    drivers = np.zeros((system.sizes.drivers, sample_count), dtype=precision)
    updated_arrays = {}

    inputs_default = {
        'initial_values': system.initial_values.values_array.copy(),
        'parameters': system.parameters.values_array.copy(),
        'forcing_vectors': drivers,
    }

    if system.sizes.drivers > 0:
        drivers[:, 0::25] = system.precision(1.0)  # Set every 25th sample to 1.0

    if 'initial_values' in kwargs:
        initial_values_edits = kwargs.pop('initial_values')
        if isinstance(initial_values_edits, np.ndarray):
            # Ensure a 1D array is provided
            assert initial_values_edits.ndim == 1, "Initial values must be a 1D array."
            required_length = system.initial_values.n
            provided_length = len(initial_values_edits)
            new_initial = system.initial_values.values_array.copy()
            if provided_length < required_length:
                new_initial[:provided_length] = initial_values_edits
            else:
                new_initial[:] = initial_values_edits[:required_length]
                if provided_length > required_length:
                    warn("Redundant initial values provided; extra values are discarded.", UserWarning)
            updated_arrays['initial_values'] = new_initial.astype(precision)

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
            updated_arrays['parameters'] = new_params.astype(precision)

    if 'forcing_vectors' in kwargs:
        forcing_override = kwargs['forcing_vectors']
        if isinstance(forcing_override, np.ndarray):
            assert forcing_override.shape[
                       0] == system.sizes.drivers, "forcing_vectors override must have system.num_drivers rows."
            drivers = forcing_override.astype(precision)
            updated_arrays['forcing_vectors'] = drivers


    inputs_default.update(kwargs)
    return inputs_default


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