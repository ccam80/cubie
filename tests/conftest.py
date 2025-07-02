import pytest
import numpy as np
from CuMC.SystemModels.Systems.decays import Decays
from CuMC.SystemModels.Systems.threeCM import ThreeChamberModel
from CuMC.ForwardSim.integrators.output_functions import build_output_functions


"""Fixtures for instantiating lower-level components with default values that can be overriden through
indirect parametrization of the "override" fixture."""

@pytest.fixture(scope="function")
def precision_override(request):
    return request.param if hasattr(request, 'param') else None


@pytest.fixture(scope="function")
def precision(precision_override):
    """
    Run tests with float32 by default, or override with float64.

    Usage:
    @pytest.mark.parametrize("precision_override", [np.float64], indirect=True)
    def test_something(precision):
        # precision will be np.float64 here
    """
    return precision_override if precision_override==np.float64 else np.float32


@pytest.fixture(scope="function")
def threecm_model(precision):
    from CuMC.SystemModels.Systems.threeCM import ThreeChamberModel
    threeCM = ThreeChamberModel(precision=precision)
    threeCM.build()
    return threeCM


@pytest.fixture(scope="function")
def decays_123_model(precision):
    from CuMC.SystemModels.Systems.decays import Decays
    decays3 = Decays(coefficients=[precision(1.0), precision(2.0), precision(3.0)])
    decays3.build()
    return decays3


@pytest.fixture(scope="function")
def decays_1_100_model(precision):
    from CuMC.SystemModels.Systems.decays import Decays
    decays100 = Decays(coefficients=np.arange(1, 101, dtype=precision))
    decays100.build()
    return decays100

def genericODE_settings(**kwargs):
    generic_ode_settings = {'constants': {'c0': 0.0,
                                          'c1': 2.0,
                                          'c2': 3.0},
                            'initial_values': {'x0': 1.0,
                                               'x1': 0.0},
                            'parameters': {'p0': 2.0,
                                           'p1': 0.5,
                                           'p2': 5.5},
                            'observables': {'o0': 4.2,
                                            'o1': 1.8}}
    generic_ode_settings.update(kwargs)

@pytest.fixture(scope="function")
def genericODE_model_override(request):
    if hasattr(request, 'param'):
        return request.param
    return None

@pytest.fixture(scope="function")
def genericODE_model(precision, genericODE_settings):
    from CuMC.SystemModels.Systems.GenericODE import GenericODE
    generic = GenericODE(**genericODE_settings)
    generic.build()
    return generic

@pytest.fixture(scope="function")
def system_override(request):
    """Override for system model type, if provided."""
    return request.param if hasattr(request, 'param') else None

@pytest.fixture(scope="function")
def system(request, system_override):
    """
    Return the appropriate system model, defaulting to Decays123.

    Usage:
    @pytest.mark.parametrize("system_override", ["ThreeChamber"], indirect=True)
    def test_something(system):
        # system will be the ThreeChamber model here
    """
    # Use the override if provided, otherwise default to Decays123
    model_type = system_override if system_override is not None else "Decays123"

    # Initialize the appropriate model fixture based on the parameter
    if model_type == "ThreeChamber":
        model = request.getfixturevalue("threecm_model")
    elif model_type == "Decays123":
        model = request.getfixturevalue("decays_123_model")
    elif model_type == "Decays1_100":
        model = request.getfixturevalue("decays_1_100_model")
    elif model_type == "genericODE":
        model = request.getfixturevalue("genericODE_model")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.build()
    return model

@pytest.fixture(scope='function')
def output_functions(loop_compile_settings):
    # Merge the default config with any overrides

    outputfunctions = build_output_functions(
        loop_compile_settings['output_functions'],
        loop_compile_settings['saved_states'],
        loop_compile_settings['saved_observables'],
        loop_compile_settings['n_peaks']
    )
    return outputfunctions


def update_loop_compile_settings(**kwargs):
    """The standard set of compile arguments, some of which aren't used by certain algorithms (like dtmax for a fixed step)."""
    loop_compile_settings_dict = {'dt_min': 0.001,
                                  'dt_max': 0.01,
                                  'dt_save': 0.01,
                                  'dt_summarise': 0.1,
                                  'atol': 1e-6,
                                  'rtol': 1e-3,
                                  'saved_states': [0, 1],
                                  'saved_observables': [1, 2],
                                  'output_functions': ["state"],
                                  'n_peaks': 0}
    loop_compile_settings_dict.update(kwargs)
    return loop_compile_settings_dict

@pytest.fixture(scope='function')
def loop_compile_settings_overrides(request):
    """ Parametrize this fixture indirectly to change compile settings, no need to request this fixture directly
    unless you're testing that it worked."""
    return request.param if hasattr(request, 'param') else {}

@pytest.fixture(scope='function')
def loop_compile_settings(request, loop_compile_settings_overrides):
    """
    Create a dictionary of compile settings for the loop function.
    This is the fixture your test should use - if you want to change the compile settings, indirectly parametrize the
    compile_settings_overrides fixture.
    """
    return update_loop_compile_settings(**loop_compile_settings_overrides)


def calculate_expected_summaries(state_input, observables_input,
                                 loop_compile_settings,
                                 output_functions,
                                 precision):
    """Helper function to calculate expected summary values from a given pair of state and observable arrays.
    Takes extra arguments from loop_compile_settings and output_functions - pass the fixture outputs straight
    to this function.

    Arguments:
    - state_input: 2D array of shape (n_states, n_samples) with the state data to summarise
    - observables_input: 2D array of shape (n_observables, n_samples) with the observable data to summarise
    - loop_compile_settings: Dictionary with settings for the loop function, including dt_summarise, dt_save,
        output_functions, saved_states, saved_observables, n_peaks
    - output_functions: Output functions object with summary_output_length method
    - precision: Numpy dtype to use for the output arrays (e.g. np.float32 or np.float64)

    Returns:
    - expected_state_summaries: 2D array of shape (n_saved_states * summary_size_per_state, n_samples)
    - expected_obs_summaries: 2D array of shape (n_saved_observables * summary_size_per_state, n_samples)
        """
    num_samples = state_input.shape[1]

    # #Check for a list of saved state/obs but that recording not requested
    # save_state_bool = "state" in loop_compile_settings['output_functions']
    # save_observables_bool = "observables" in loop_compile_settings['output_functions']
    # saved_states = np.asarray(loop_compile_settings['saved_states']) if save_state_bool else np.asarray([])
    # saved_observables = np.asarray(loop_compile_settings['saved_observables']) if save_observables_bool else np.asarray([])
    #
    # n_saved_states = len(saved_states)
    # n_saved_observables = len(saved_observables)

    summarise_every = int(loop_compile_settings['dt_summarise'] / loop_compile_settings['dt_save'])
    summary_samples = int(num_samples / summarise_every)
    summary_size_per_state = output_functions.summary_output_length

    state_summaries_height = summary_size_per_state * state_input.shape[0]
    obs_summaries_height = summary_size_per_state * observables_input.shape[0]

    expected_state_summaries = np.zeros((state_summaries_height, summary_samples), dtype=precision)
    expected_obs_summaries = np.zeros((obs_summaries_height, summary_samples), dtype=precision)

    for (_input_array, _output_array) in ((state_input, expected_state_summaries),
                                          (observables_input, expected_obs_summaries)):
        calculate_single_summary_array(_input_array,
                                       summarise_every,
                                       summary_size_per_state,
                                       loop_compile_settings['output_functions'],
                                       loop_compile_settings['n_peaks'],
                                       _output_array)

    return expected_state_summaries, expected_obs_summaries

def calculate_single_summary_array(input_array,
                                   summarise_every,
                                   summary_size_per_state,
                                   output_functions_list,
                                   n_peaks,
                                   output_array):
    """ Summarise states in input array in the same way that the device functions do.

    Arguments:
    - input_array: 2D array of shape (n_items, n_samples) with the input data to summarise
    - summarise_every: Number of samples to summarise over
    - summary_size_per_state: Number of summary values per state (e.g. 1 for mean, 1 + n_peaks for mean and peaks)
    - output_functions_list: List of output function names to apply (e.g. ["mean", "peaks", "max", "rms"])
    - n_peaks: Number of peaks to find in the "peaks" output function
    - output_array: 2D array to store the summarised output, shape (n_items * summary_size_per_state, n_samples)

    Returns:
    - None, but output_array is filled with the summarised values.

        """
    # Sort outputs list to match the order in build_output_functions
    types_order = ["mean", "peaks", "max", "rms"]
    sorted_outputs = sorted(output_functions_list,
                            key=lambda x: types_order.index(x) if x in types_order else len(types_order))
    summary_samples = int(input_array.shape[1] / summarise_every)
    n_items = input_array.shape[0]

    # Manual cycling through possible summaries to match the approach used when building the device functions
    for j in range(n_items):
        for i in range(summary_samples):
            summary_index = 0
            start_index = i * summarise_every
            end_index = (i + 1) * summarise_every
            for output_type in sorted_outputs:
                if output_type == 'mean':
                    output_array[j * summary_size_per_state + summary_index, i] = np.mean(
                        input_array[j, start_index: end_index], axis=0)
                    summary_index += 1

                if output_type == 'peaks':
                    # Use the last two samples, like the live version does
                    start_index = i * summarise_every - 2 if i > 0 else 0
                    maxima = local_maxima(
                        input_array[j, start_index: end_index])[:n_peaks] + start_index
                    output_start_index = j * summary_size_per_state + summary_index
                    output_array[output_start_index: output_start_index + maxima.size, i] = maxima
                    summary_index += n_peaks

                if output_type == 'max':
                    _max = np.max(input_array[j, start_index: end_index], axis=0)
                    output_array[j * summary_size_per_state + summary_index, i] = _max
                    summary_index += 1

                if output_type == 'rms':
                    rms = np.sqrt(np.mean(input_array[j, start_index: end_index] ** 2, axis=0))
                    output_array[j * summary_size_per_state + summary_index, i] = rms
                    summary_index += 1

def local_maxima(signal: np.ndarray) -> np.ndarray:
    return np.flatnonzero((signal[1:-1] > signal[:-2]) & (signal[1:-1] > signal[2:])) + 1