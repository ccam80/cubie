import pytest
import numpy as np
from numpy.testing import assert_allclose
from numba import cuda, from_dtype

from CuMC.ForwardSim.OutputHandling.output_functions import OutputFunctions
from tests._utils import generate_test_array, calculate_expected_summaries

@pytest.fixture(scope='function')
def output_test_settings(output_test_settings_overrides, precision):
    """Parameters for instantiating and testing the outputfunctions class, both compile settings and run settings
    per test. Outputfunctions should not care about the higher-level modules, so we duplicate information that is
    housed elsewhere in conftest.py for use in tests of system integration (pun)."""
    output_test_settings_dict = {'num_samples':            100,
                                 'num_summaries':          10,
                                 'num_states':             10,
                                 'num_observables':        10,
                                 'saved_states':           [0, 1],
                                 'saved_observables':      [0, 1],
                                 'summarised_states':      None,
                                 'summarised_observables': None,
                                 'random_scale':           1.0,
                                 'output_types':           ["state"],
                                 'precision':              precision,
                                 'test_shared_mem':        True,
                                 }
    output_test_settings_dict.update(**output_test_settings_overrides)
    return output_test_settings_dict

@pytest.fixture(scope='function')
def output_test_settings_overrides(request):
    """ Parametrize this fixture indirectly to change test settings, no need to request this fixture directly
    unless you're testing that it worked."""
    return request.param if hasattr(request, 'param') else {}

@pytest.fixture(scope='function')
def output_functions(output_test_settings):
    """OutputHandling object under test"""
    return OutputFunctions(output_test_settings['num_states'],
                           output_test_settings['num_observables'],
                           output_test_settings['output_types'],
                           output_test_settings['saved_states'],
                           output_test_settings['saved_observables'],
                           )

@pytest.mark.parametrize("output_test_settings_overrides",
                         [{'output_types': ["state", "observables"]},
                          {'output_types': ["time"]}])
def test_save_time(output_functions, output_test_settings):
    """Test that the save_time setting is correctly set in the OutputHandling object."""
    assert output_functions.save_time == ("time" in output_test_settings['output_types'])


@pytest.mark.parametrize("output_test_settings_overrides, fails",
                         [({'output_types': ["state", "observables"]}, False),
                          ({'output_types': ["state", "observables", "max", "rms", "peaks[3]"]}, False),
                          ({'saved_states': [0], 'saved_observables': [0]}, False),
                          ({'saved_states':[20]}, True),
                          ({'saved_states': [], 'saved_observables': []}, False),
                          ({'output_types': []}, True)],
                         ids=["no_summaries", "all_summaries", "single_saved", "saved_index_out_of_bounds",
                              "saved_empty", "no_output_types"],
                         indirect=["output_test_settings_overrides"]
                         )
def test_output_functions_build(output_test_settings, fails):
    """Test happy path and failure cases for instantiating and building outputfunctions Builds a new object instead
    of using fixtures to capture errors in instantiation."""
    if fails:
        with pytest.raises(ValueError):
            OutputFunctions(output_test_settings['num_states'],
                            output_test_settings['num_observables'],
                            output_test_settings['output_types'],
                            output_test_settings['saved_states'],
                            output_test_settings['saved_observables'],
                            )

    else:
        output_functions = OutputFunctions(output_test_settings['num_states'],
                                           output_test_settings['num_observables'],
                                           output_test_settings['output_types'],
                                           output_test_settings['saved_states'],
                                           output_test_settings['saved_observables'],
                                           )
        save_state = output_functions.save_state_func
        update_summaries = output_functions.update_summaries_func
        save_summaries = output_functions.save_summary_metrics_func
        memory_required = output_functions.memory_per_summarised_variable
        buffer_memory = memory_required['buffer']
        output_memory = memory_required['output']

        # Now use these functions in your test
        assert callable(save_state)
        assert callable(update_summaries)
        assert callable(save_summaries)
        if len(output_functions.summary_types) > 0:
            assert output_memory > 0
            assert buffer_memory > 0

@pytest.fixture(scope='function')
def input_arrays(output_test_settings):
    """Random input state and observable arrays for tests."""
    num_states = output_test_settings['num_states']
    num_observables = output_test_settings['num_observables']
    num_samples = output_test_settings['num_samples']
    precision = output_test_settings['precision']
    scale = output_test_settings['random_scale']

    states = generate_test_array(precision, (num_samples, num_states), 'random', scale)
    observables = generate_test_array(precision, (num_samples, num_observables), 'random', scale)

    return states, observables

@pytest.fixture(scope='function')
def empty_output_arrays(output_test_settings, output_functions):
    """Empty output arrays for testing."""

    n_saved_states = output_functions.n_saved_states
    n_saved_observables = output_functions.n_saved_observables
    n_summarised_states = output_functions.n_summarised_states
    n_summarised_observables = output_functions.n_summarised_observables
    num_samples = output_test_settings['num_samples']
    num_summaries = output_test_settings['num_summaries']
    summary_height_per_variable = output_functions.memory_per_summarised_variable['output']
    state_summary_height = summary_height_per_variable * n_summarised_states
    observable_summary_height = summary_height_per_variable * n_summarised_observables

    precision = output_test_settings['precision']
    save_time = 'time' in output_test_settings['output_types']

    if save_time:
        n_saved_states += 1

    state_out = np.zeros((num_samples, n_saved_states), dtype=precision)
    observable_out = np.zeros((num_samples, n_saved_observables), dtype=precision)
    state_summary = np.zeros((num_summaries, state_summary_height), dtype=precision)
    observable_summary = np.zeros((num_summaries, observable_summary_height), dtype=precision)

    return state_out, observable_out, state_summary, observable_summary

@pytest.fixture(scope='function')
def expected_outputs(output_test_settings, input_arrays, precision):
    """Selected portions of input arrays - should match what the test kernel does."""
    num_samples = output_test_settings['num_samples']
    state_in, observables_in = input_arrays
    state_out = state_in[:, output_test_settings['saved_states']]
    observable_out = observables_in[:, output_test_settings['saved_observables']]
    save_time = 'time' in output_test_settings['output_types']

    if save_time:
        time_output = np.arange(num_samples, dtype=precision)
        time_output = time_output.reshape((num_samples, 1))
        state_out = np.concatenate((state_out, time_output), axis=1)

    return state_out, observable_out

@pytest.fixture(scope='function')
def expected_summaries(output_test_settings, empty_output_arrays,
                       expected_outputs, output_functions, precision):
    """Default expected summaries for the output functions."""
    state_output, observables_output = expected_outputs
    if output_functions.save_time:
        state_output = state_output[:, :-1]
    summarise_every = (output_test_settings['num_samples'] //
                       output_test_settings['num_summaries'])
    output_types = output_test_settings['output_types']
    summary_height_per_variable = output_functions.memory_per_summarised_variable['output']

    state_summaries, observable_summaries = calculate_expected_summaries(state_output,
                                 observables_output,
                                 summarise_every,
                                 output_types,
                                 summary_height_per_variable,
                                 precision)

    return state_summaries, observable_summaries


@pytest.fixture(scope='function')
def output_functions_test_kernel(precision, output_test_settings, output_functions):
    """Kernel that writes input to local state, then uses output functions to save every sample and summarise every
    summarise_every samples."""
    summarise_every = output_test_settings['num_samples'] // output_test_settings['num_summaries']
    test_shared_mem = output_test_settings['test_shared_mem']

    save_state_func = output_functions.save_state_func
    update_summary_metrics_func = output_functions.update_summaries_func
    save_summary_metrics_func = output_functions.save_summary_metrics_func

    num_states = output_test_settings['num_states']

    num_observables = output_test_settings['num_observables']

    n_summarised_states = output_functions.n_summarised_states
    n_summarised_observables = output_functions.n_summarised_observables

    memory_requirements = output_functions.memory_per_summarised_variable
    shared_memory_requirements = memory_requirements['buffer']
    state_summary_buffer_length = shared_memory_requirements * n_summarised_states
    obs_summary_buffer_length = shared_memory_requirements * n_summarised_observables

    if test_shared_mem is False:
        num_states = 1 if num_states == 0 else num_states
        num_observables = 1 if num_observables == 0 else num_observables
        state_summary_buffer_length = 1 if state_summary_buffer_length == 0 else state_summary_buffer_length
        obs_summary_buffer_length = 1 if obs_summary_buffer_length == 0 else obs_summary_buffer_length

    numba_precision = from_dtype(precision)

    @cuda.jit()
    def _output_functions_test_kernel(_state_input, _observable_input,
                                      _state_output, _observable_output,
                                      _state_summaries_output, _observable_summaries_output,
                                      ):
        """Test kernel for output functions."""

        tx = cuda.threadIdx.x
        bx = cuda.blockIdx.x

        #single-threaded test, slow as you like
        if tx != 0 or bx != 0:
            return

        if test_shared_mem:
            # Shared memory arrays for current state, current observables, and running summaries
            shared = cuda.shared.array(0, dtype=numba_precision)

            observables_start_idx = num_states
            state_summaries_start_idx = observables_start_idx + num_observables
            obs_summaries_start_idx = state_summaries_start_idx + state_summary_buffer_length
            obs_summaries_end_idx = obs_summaries_start_idx + obs_summary_buffer_length

            current_state = shared[:observables_start_idx]
            current_observable = shared[observables_start_idx:state_summaries_start_idx]
            state_summaries = shared[state_summaries_start_idx:obs_summaries_start_idx]
            observable_summaries = shared[obs_summaries_start_idx:obs_summaries_end_idx]

        else:
            current_state = cuda.local.array(num_states, dtype=numba_precision)
            current_observable = cuda.local.array(num_observables, dtype=numba_precision)
            state_summaries = cuda.local.array(state_summary_buffer_length, dtype=numba_precision)
            observable_summaries = cuda.local.array(obs_summary_buffer_length, dtype=numba_precision)

        current_state[:] = 0.0
        current_observable[:] = 0.0
        state_summaries[:] = 0.0
        observable_summaries[:] = 0.0

        for i in range(_state_input.shape[0]):
            for j in range(num_states):
                current_state[j] = _state_input[i, j]
            for j in range(num_observables):
                current_observable[j] = _observable_input[i, j]

            # Call the output functions
            save_state_func(
                    current_state,
                    current_observable,
                    _state_output[i, :],
                    _observable_output[i, :],
                    i,  # time is just loop index here
                    )

            update_summary_metrics_func(
                    current_state,
                    current_observable,
                    state_summaries,
                    observable_summaries,
                    i,
                    )

            # Save summary metrics every summarise_every samples
            if (i + 1) % summarise_every == 0:
                sample_index = int(i / summarise_every)
                save_summary_metrics_func(
                        state_summaries,
                        observable_summaries,
                        _state_summaries_output[sample_index,:],
                        _observable_summaries_output[sample_index,:],
                        summarise_every,
                        )

    return _output_functions_test_kernel


@pytest.fixture(scope='function')
def compare_input_output(output_functions_test_kernel,
                         output_functions,
                         output_test_settings,
                         input_arrays,
                         empty_output_arrays,
                         expected_outputs,
                         expected_summaries,
                         precision):
    """Test that output functions correctly save state and observable values."""

    state_input, observable_input = input_arrays
    state_output, observable_output, state_summaries_output, observable_summaries_output = empty_output_arrays


    n_summarised_states = output_functions.n_summarised_states
    n_summarised_obeservables = output_functions.n_summarised_observables

    n_states = output_test_settings['num_states']
    n_observables = output_test_settings['num_observables']

    # To the CUDA device
    d_state_input = cuda.to_device(state_input)
    d_observable_input = cuda.to_device(observable_input)
    d_state_output = cuda.to_device(state_output)
    d_observable_output = cuda.to_device(observable_output)
    d_state_summaries_output = cuda.to_device(state_summaries_output)
    d_observable_summaries_output = cuda.to_device(observable_summaries_output)

    kernel_shared_memory = n_states + n_observables  # Hard-coded from test kernel code
    summary_buffer_size = output_functions.memory_per_summarised_variable['buffer']
    summary_shared_memory = (n_summarised_states + n_summarised_obeservables) * summary_buffer_size
    dynamic_shared_memory = (kernel_shared_memory + summary_shared_memory) * precision().itemsize

    output_functions_test_kernel[1, 1, 0, dynamic_shared_memory](
            d_state_input,
            d_observable_input,
            d_state_output,
            d_observable_output,
            d_state_summaries_output,
            d_observable_summaries_output,
            )

    # Synchronize and copy results back
    cuda.synchronize()

    state_output = d_state_output.copy_to_host()
    observable_output = d_observable_output.copy_to_host()
    state_summaries_output = d_state_summaries_output.copy_to_host()
    observable_summaries_output = d_observable_summaries_output.copy_to_host()

    expected_state_output, expected_observable_output = expected_outputs
    expected_state_summaries, expected_observable_summaries = expected_summaries

    if output_functions.compile_settings.save_state:
        assert_allclose(state_output, expected_state_output, atol=1e-12, rtol=1e-12,
                        err_msg="State &| time values were not saved correctly",
                        )

    if output_functions.compile_settings.save_observables:
        assert_allclose(observable_output, expected_observable_output, atol=1e-12,
                        rtol=1e-12,
                        err_msg="Observable values were not saved correctly",
                        )

    #Adjust tolerances for precision and calculation type - rms is rough.
    rms_on = "rms" in output_test_settings['output_types']
    if precision == np.float32:
        atol = 1e-05 if not rms_on else 1e-3
        rtol = 5e-05 if not rms_on else 1e-3
    elif precision == np.float64:
        atol = 1e-12 if not rms_on else 1e-9
        rtol = 1e-12 if not rms_on else 1e-9

    if output_functions.compile_settings.summarise_states:
        assert_allclose(expected_state_summaries, state_summaries_output, atol=atol, rtol=rtol,
                        err_msg=f"State summaries didn't match expected values. Shapes: expected"
                                f"[{expected_state_summaries.shape}, actual[{state_summaries_output.shape}]",
                        )
    if output_functions.compile_settings.summarise_observables:
        assert_allclose(expected_observable_summaries, observable_summaries_output, atol=atol, rtol=rtol,
                        err_msg=f"Observable summaries didn't match expected values. Shapes: expected[{expected_observable_summaries.shape}, actual[{observable_summaries_output.shape}]",
                        )


@pytest.mark.parametrize("precision_override", [np.float32, np.float64], ids=["float32", "float64"], indirect=True)
@pytest.mark.parametrize("output_test_settings_overrides", [
    {'output_types': ["state", "observables", "mean", "max", "rms", "peaks[3]"], 'num_samples': 1000},
    {'output_types': ["state", "observables", "mean", "max", "rms", "peaks[3]"], 'num_samples': 50},
], ids=["large_dataset", "small_dataset"], indirect=True)
def test_precision_with_large_datasets(compare_input_output):
    """Test precision differences (float32 vs float64) with large datasets."""
    pass

@pytest.mark.parametrize("precision_override", [np.float32], ids=["float32"], indirect=True)
@pytest.mark.parametrize("output_test_settings_overrides", [
    {'output_types': ["state", "observables", "mean", "max", "peaks[3]"], 'test_shared_mem': False},
    {'output_types': ["state", "observables", "mean", "max", "peaks[3]"], 'test_shared_mem': True},
], ids=["local_mem", "shared_mem"], indirect=True)
def test_memory_types(compare_input_output):
    """Test shared vs local memory with complex output configurations."""
    pass

@pytest.mark.parametrize("precision_override", [np.float32], ids=["float32"], indirect=True)
@pytest.mark.parametrize("output_test_settings_overrides", [
    {'random_scale': 1e-6},
    {'random_scale': 1e6},
    {'random_scale': [-12, 12]}
], ids=["tiny_values", "large_values", "wide_range"], indirect=True)
def test_input_value_ranges(compare_input_output):
    """Test different input scales and simulation lengths."""
    pass

@pytest.mark.parametrize("precision_override", [np.float32, np.float64], ids=["float32", "float64"], indirect=True)
@pytest.mark.parametrize("output_test_settings_overrides", [
    {'output_types': ["time"]},
    {'output_types': ["state", "observables", "time"]},
    {'output_types': ["state", "observables", "time", 'mean']}
], ids=["only_time", "time_with_state_obs","time_with_state_obs_mean"], indirect=True)
def test_time_output(compare_input_output):
    """Test time output specifically."""
    pass

@pytest.mark.parametrize("precision_override", [np.float32], ids=["float32"], indirect=True)
@pytest.mark.parametrize("output_test_settings_overrides", [
    {'output_types': ["state", "observables"]},
    {'output_types': ["observables"]},
    {'output_types': ["state"]},
], ids=["state_obs", "obs_only", "state_only"], indirect=True)
def test_basic_output_configurations(compare_input_output):
    """Test basic state and observable output configurations."""
    pass

@pytest.mark.parametrize("precision_override", [np.float32], ids=["float32"], indirect=True)
@pytest.mark.parametrize("output_test_settings_overrides", [
    {'output_types': ["mean", "max", "rms"]},
    {'output_types': ["peaks[10]"]},
    {'output_types': ["state", "observables", "time", "mean", "max", "rms", "peaks[5]"]},
    {'output_types': ["state", "mean", "max"]},
    {'output_types': ["observables", "mean", "max"]},
], ids=["basic_summaries", "peaks_only", "full_summary", "state_mean_max", "obs_mean_max"], indirect=True)
def test_summary_metric_configurations(compare_input_output):
    """Test various summary metrics configurations."""
    pass

@pytest.mark.parametrize("precision_override", [np.float32], ids=["float32"], indirect=True)
@pytest.mark.parametrize("output_test_settings_overrides", [
    {'saved_states': [0], 'saved_observables': [0]},
    {'num_states': 11, 'num_observables': 6, 'saved_states': list(range(10)), 'saved_observables': list(range(5))},
    {'num_states': 51, 'num_observables': 21,'saved_states': list(range(50)), 'saved_observables': list(range(20))},
], ids=["1_1", "10_5", "50_20"], indirect=True)
def test_small_to_medium_system_sizes(compare_input_output):
    """Test various small to medium system sizes."""
    pass

@pytest.mark.parametrize("precision_override", [np.float32, np.float64], ids=["float32", "float64"], indirect=True)
@pytest.mark.parametrize("output_test_settings_overrides", [
    {'num_states': 101, 'num_observables': 100, 'saved_states': list(range(100)), 'saved_observables': list(range(
            100))},
], ids=["100_100"], indirect=True)
def test_large_system_sizes_precision(compare_input_output):
    """Test large system size with different precision."""
    pass

@pytest.mark.parametrize("precision_override", [np.float32], ids=["float32"], indirect=True)
@pytest.mark.parametrize("output_test_settings_overrides", [
    {'num_summaries': 5},
    {'num_summaries': 20},
    {'num_summaries': 50},
], ids=["few_summaries", "many_summaries", "lots_summaries"], indirect=True)
def test_summary_frequency_variations(compare_input_output):
    """Test different summary frequencies."""
    pass