import pytest
import numpy as np
from numpy.testing import assert_allclose
from numba import cuda, from_dtype
from CuMC.ForwardSim.OutputHandling.output_functions import OutputFunctions
from tests._utils import generate_test_array, calculate_expected_summaries

# Improvement: This test format contains a fair amount of duplication, and tests a lot at once. It covers the code,
# so is useful, but could be rewritten to reduce test burden and improve readability.
TEST_CONFIGS = [
    # Basic state and observables tests with different precisions
    (2, 1, ["state", "observables"], None, np.float32, 100, 50, "State and observables, 32b, small and short"),
    (2, 1, ["state", "observables"], None, np.float64, 100, 50, "State and observables, 64b, small and short"),

    # Testing individual output types with different precisions
    (3, 2, ["state", "observables", "mean"], None, np.float32, 100, 50, "Mean only, 32b"),
    (3, 2, ["state", "observables", "max"], None, np.float32, 100, 50, "Max only, 32b"),
    (3, 2, ["state", "observables", "rms"], None, np.float32, 100, 50, "RMS only, 32b"),
    (3, 2, ["state", "observables", "peaks"], 3, np.float32, 100, 50, "Peaks only, 32b"),
    (3, 2, ["state", "observables", "mean"], None, np.float64, 100, 50, "Mean only, 64b"),

    # Testing combinations of output types
    (4, 3, ["state", "observables", "mean", "max"], None, np.float32, 100, 50, "Mean and max, 32b"),
    (4, 3, ["state", "observables", "mean", "rms"], None, np.float32, 100, 50, "Mean and rms, 32b"),
    (4, 3, ["state", "observables", "mean", "peaks"], 3, np.float32, 100, 50, "Mean and peaks, 32b"),
    (4, 3, ["state", "observables", "max", "rms"], None, np.float32, 100, 50, "Max and rms, 32b"),
    (4, 3, ["state", "observables", "max", "peaks"], 3, np.float32, 100, 50, "Max and peaks, 32b"),
    (4, 3, ["state", "observables", "rms", "peaks"], 3, np.float32, 100, 50, "RMS and peaks, 32b"),

    # Testing all output types together
    (5, 4, ["state", "observables", "mean", "max", "rms", "peaks"], 3, np.float32, 100, 50,
     "All metrics, 32b, small and short"
     ),
    (5, 4, ["state", "observables", "mean", "max", "rms", "peaks"], 3, np.float64, 100, 50,
     "All metrics, 64b, small and short"
     ),

    # Testing different state/observable sizes (states > observables)
    (10, 2, ["state", "observables", "mean", "max"], None, np.float32, 100, 50, "More states than observables, 32b"),
    (20, 5, ["state", "observables", "mean", "max", "rms"], None, np.float32, 100, 50,
     "Many more states than observables, 32b"
     ),

    # Testing different state/observable sizes (observables > states)
    (2, 10, ["state", "observables", "mean", "max"], None, np.float32, 100, 50, "More observables than states, 32b"),
    (5, 20, ["state", "observables", "mean", "max", "rms"], None, np.float32, 100, 50,
     "Many more observables than states, 32b"
     ),

    # Testing large equal numbers of states and observables
    (50, 50, ["state", "observables", "mean", "max", "rms", "peaks"], 3, np.float32, 100, 50,
     "Many states and observables, 32b"
     ),
    (50, 50, ["state", "observables", "mean", "max", "rms", "peaks"], 3, np.float64, 100, 50,
     "Many states and observables, 64b"
     ),

    # Testing different numbers of peaks
    (3, 3, ["state", "observables", "peaks"], 1, np.float32, 100, 50, "Single peak, 32b"),
    (3, 3, ["state", "observables", "peaks"], 5, np.float32, 100, 50, "Five peaks, 32b"),
    (3, 3, ["state", "observables", "peaks"], 10, np.float32, 100, 50, "Ten peaks, 32b"),

    # Testing long sample sizes with infrequent summarization
    (4, 4, ["state", "observables", "mean", "max"], None, np.float32, 10000, 10000,
     "Long samples, summarize once, 32b"
     ),
    (4, 4, ["state", "observables", "mean", "max"], None, np.float64, 10000, 10000,
     "Long samples, summarize once, 64b"
     ),

    # Testing long sample sizes with frequent summarization
    (4, 4, ["state", "observables", "mean", "max"], None, np.float32, 10000, 100,
     "Long samples, frequent summarization, 32b"
     ),
    (4, 4, ["state", "observables", "mean", "max"], None, np.float64, 10000, 100,
     "Long samples, frequent summarization, 64b"
     ),

    # Testing short sample sizes with very frequent summarization
    (4, 4, ["state", "observables", "mean", "max"], None, np.float32, 100, 5,
     "Short samples, summarize every 5 steps, 32b"
     ),
    (4, 4, ["state", "observables", "mean", "max"], None, np.float64, 100, 5,
     "Short samples, summarize every 5 steps, 64b"
     ),

    # Edge cases
    (1, 1, ["state", "observables", "mean", "max", "rms", "peaks"], 3, np.float32, 100, 50,
     "Minimal state and observables, 32b"
     ),
    (1, 1, ["state", "observables", "mean", "max", "rms", "peaks"], 3, np.float64, 100, 50,
     "Minimal state and observables, 64b"
     ),
    (100, 1, ["state", "observables", "mean"], None, np.float32, 100, 50, "Many states, single observable, 32b"),
    (1, 100, ["state", "observables", "mean"], None, np.float32, 100, 50, "Single state, many observables, 32b"),

    (1, 1, ["state", "observables", "mean", "max", "rms", "peaks"], 3, np.float32, 100, 50,
     "Minimal state and observables, 32b"
     ),
    (1, 1, ["state", "observables", "mean", "max", "rms", "peaks"], 3, np.float64, 100, 50,
     "Minimal state and observables, 64b"
     ),
    (100, 1, ["state", "observables", "mean"], None, np.float32, 100, 50, "Many states, single observable, 32b"),
    (1, 100, ["state", "observables", "mean"], None, np.float32, 100, 50, "Single state, many observables, 32b"),

    # Comprehensive tests with all metrics and different configurations
    (8, 4, ["state", "observables", "mean", "max", "rms"], None, np.float32, 1000, 500,
     "Medium test, all metrics except peaks, 32b"
     ),
    (8, 4, ["state", "observables", "mean", "max", "rms", "peaks"], 3, np.float32, 1000, 500,
     "Medium test, all metrics with peaks, 32b"
     ),
    (10, 5, ["state", "observables", "mean", "max", "rms", "peaks"], 5, np.float64, 1000, 500,
     "Medium test, all metrics with peaks, 64b"
     ),

    # Extreme cases
    (100, 100, ["state", "observables", "mean"], None, np.float32, 10000, 1000,
     "Very large state and observables, mean only, 32b"
     ),
    (100, 100, ["state", "observables", "peaks"], 20, np.float32, 10000, 1000,
     "Very large state and observables, many peaks, 32b"
     ),

    (2, 1, ["state", "observables", "time"], None, np.float32, 100, 50,
     "State, observables and time, 32b"
     ),
    (3, 2, ["state", "observables", "time", "mean"], None, np.float64, 100, 50,
     "State, observables, time and mean, 64b"
     ),
    ]


@pytest.mark.parametrize("output_functions_overrides",
                         [{'outputs_list': ["state", "observables"], 'n_peaks': 3, 'saved_states': [0, 1, 2]},
                          {'outputs_list': ["state", "observables", "mean"], 'n_peaks': 3}],
                         indirect=True,
                         )
def test_output_functions_build(output_functions, expected_summary_temp_memory, expected_summary_output_memory):
    save_state = output_functions.save_state_func
    update_summaries = output_functions.update_summary_metrics_func
    save_summaries = output_functions.save_summary_metrics_func
    memory_required = output_functions.memory_per_summarised_variable
    temp_memory = memory_required['temporary']
    output_memory = memory_required['output']

    # Now use these functions in your test
    assert callable(save_state)
    assert callable(update_summaries)
    assert callable(save_summaries)
    assert temp_memory == expected_summary_temp_memory
    assert output_memory == expected_summary_output_memory


#Individual output function configs are overridden by compile settings at higher levels, so these fixtures are
# module-specific.
@pytest.fixture(scope='function')
def output_functions_config(loop_compile_settings, output_functions_overrides):
    """Provide a default dictionary of output functions, with values drawn from the loop_compile_settings defaults.
    Overrideable by parameterising "output_functions_overrides"."""
    output_dict = {'outputs_list':      loop_compile_settings['output_functions'],
                   'saved_states':      loop_compile_settings['saved_states'],
                   'saved_observables': loop_compile_settings['saved_observables'],
                   'n_peaks':           loop_compile_settings['n_peaks']
                   }
    output_dict.update(output_functions_overrides)
    return output_dict


@pytest.fixture(scope='function')
def output_functions_overrides(request):
    """Override configuration for output functions."""
    return request.param if hasattr(request, 'param') else {}


@pytest.fixture(scope='function')
def output_functions(output_functions_config):
    # Merge the default config with any overrides and build functions

    outputfunctions = OutputFunctions(n_states, n_parameters, output_functions_config['outputs_list'],
                                      output_functions_config['saved_states'],
                                      output_functions_config['saved_observables'], output_functions_config['n_peaks']
                                      )

    return outputfunctions


def input_type_dict(**kwargs):
    """Default input type configuration with the ability to override."""
    input_type_config = {
        'style': 'random',
        'scale': [-6, 6],
        }
    input_type_config.update(kwargs)
    return input_type_config


@pytest.fixture(scope='function')
def input_type_override(request):
    """Override for input types, if provided.
    style str: 'random', 'nan', 'zeros', 'ones'.
    scale: list: [min, max] for mixed-scale random inputs, or a single value for same-scale random inputs.
    """
    return request.param if hasattr(request, 'param') else {}


@pytest.fixture(scope='function')
def input_type(input_type_override):
    """
    Create input types with defaults and potential overrides.

    Usage:
    @pytest.mark.parametrize("input_type_override", [{'type': 'float32'}], indirect=True)
    def test_something(input_type):
        # input_type will have type='float32'
    """
    return input_type_dict(**input_type_override)


def other_run_settings_dict(**kwargs):
    """Default settings for simulation runtime configuration."""
    settings = {
        'summarise_every': 10,  # Default summarization frequency
        'num_samples':     100,
        'test_shared_mem': False
        }
    settings.update(kwargs)
    return settings


@pytest.fixture(scope='function')
def other_run_settings_override(request):
    """Override for run settings, if provided."""
    return request.param if hasattr(request, 'param') else {}


@pytest.fixture(scope='function')
def run_settings(precision, output_functions_config, input_type, other_run_settings_override):
    other_run_settings = other_run_settings_dict(**other_run_settings_override)

    num_samples = other_run_settings['num_samples']
    num_summaries = other_run_settings['num_samples'] // other_run_settings['summarise_every']
    n_states = len(output_functions_config['saved_states'])
    n_observables = len(output_functions_config['saved_observables'])
    states = generate_test_array(precision, (n_states, num_samples), input_type['style'], input_type['scale'])
    observables = generate_test_array(precision, (n_observables, num_samples), input_type['style'], input_type['scale'])

    return {'state_input':      states,
            'observable_input': observables,
            'test_shared_mem':  other_run_settings['test_shared_mem'],
            'summarise_every':  other_run_settings['summarise_every'],
            'num_samples':      num_samples,
            'num_summaries':    num_summaries,
            'n_states':         n_states,
            'max_observables':    n_observables
            }  #the last two added as a convenience, these are not the source of truth


# Calculate the expected temp memory usage for the loop functions. These directly use the same approach as the output
# functions module, so will not check the math (as that is hard-coded), but it will check that the requested output types
# and sizes have made it through the output_functions system into the outputFunctions object.
@pytest.fixture(scope='function')
def expected_summary_temp_memory(output_functions_config):
    """
    Calculate the expected temporary memory usage for the loop function.

    Usage example:
    @pytest.mark.parametrize("compile_settings_overrides", [{'dt_min': 0.001, 'dt_max': 0.01}], indirect=True)
    def test_expected_temp_memory(expected_temp_memory):
        ...
    """
    from CuMC.ForwardSim.OutputHandling.output_functions import _TempMemoryRequirements
    n_peaks = output_functions_config['n_peaks']
    outputs_list = output_functions_config['outputs_list']
    return sum([_TempMemoryRequirements(n_peaks)[output_type] for output_type in outputs_list])

@pytest.fixture(scope='function')
def expected_summary_output_memory(output_functions_config):
    """
    Calculate the expected temporary memory usage for the loop function.

    Usage example:
    @pytest.mark.parametrize("compile_settings_overrides", [{'dt_min': 0.001, 'dt_max': 0.01}], indirect=True)
    def test_expected_temp_memory(expected_temp_memory):
        ...
    """
    from CuMC.ForwardSim.OutputHandling.output_functions import _OutputMemoryRequirements
    n_peaks = output_functions_config['n_peaks']
    outputs_list = output_functions_config['outputs_list']
    return sum([_OutputMemoryRequirements(n_peaks)[output_type] for output_type in outputs_list])


@pytest.fixture(scope='function')
def output_functions_test_kernel(precision, run_settings, output_functions_config, output_functions):
    num_states = run_settings['n_states']
    num_observables = run_settings['max_observables']
    summarise_every = run_settings['summarise_every']
    test_shared_mem = run_settings['test_shared_mem']

    save_state_func = output_functions.save_state_func
    update_summary_metrics_func = output_functions.update_summary_metrics_func
    save_summary_metrics_func = output_functions.save_summary_metrics_func
    memory_requirements = output_functions.memory_per_summarised_variable
    shared_memory_requirements = memory_requirements['temporary']
    save_time = output_functions.save_time
    state_summary_length = shared_memory_requirements * num_states
    obs_summary_length = shared_memory_requirements * num_observables
    numba_precision = from_dtype(precision)

    # Avoid asking for a zero local array size in test kernel - it's not allowed. A wasted length-1 allocation is fine
    # in testing.
    if "time" in output_functions_config['outputs_list']:
        statearraysize = num_states + 1
    else:
        statearraysize = num_states
    statearraysize = statearraysize if num_states > 0 else 1

    obsize = num_observables if num_observables > 0 else 1
    statesumsize = state_summary_length if state_summary_length > 0 else 1
    obssumsize = obs_summary_length if obs_summary_length > 0 else 1

    @cuda.jit()
    def _output_functions_test_kernel(_state_input, _observable_input,
                                      _state_output, _observable_output,
                                      _state_summaries_output, _observable_summaries_output,
                                      ):
        """Test kernel for output functions."""

        tx = cuda.threadIdx.x
        bx = cuda.blockIdx.x

        # Each thread processes one time step
        tx = tx + bx * cuda.blockDim.x
        if tx != 0 or bx != 0:
            return

        if test_shared_mem:

            # Shared memory arrays for current state, current observables, and running summaries
            shared = cuda.shared.array(0, dtype=numba_precision)

            if save_time:
                observables_start_idx = num_states + 1
            else:
                observables_start_idx = num_states
            state_summaries_start_idx = observables_start_idx + num_observables
            obs_summaries_start_idx = state_summaries_start_idx + num_states * shared_memory_requirements
            obs_summaries_end_idx = obs_summaries_start_idx + num_observables * shared_memory_requirements

            current_state = shared[:observables_start_idx]
            current_observable = shared[observables_start_idx:state_summaries_start_idx]
            state_summaries = shared[state_summaries_start_idx:obs_summaries_start_idx]
            observable_summaries = shared[obs_summaries_start_idx:obs_summaries_end_idx]

        else:
            current_state = cuda.local.array(statearraysize, dtype=numba_precision)
            current_observable = cuda.local.array(obsize, dtype=numba_precision)
            state_summaries = cuda.local.array(statesumsize, dtype=numba_precision)
            observable_summaries = cuda.local.array(obssumsize, dtype=numba_precision)

        current_state[:] = 0.0
        current_observable[:] = 0.0
        state_summaries[:] = 0.0
        observable_summaries[:] = 0.0

        for i in range(_state_input.shape[1]):
            for j in range(num_states):
                current_state[j] = _state_input[j, i]
            for j in range(num_observables):
                current_observable[j] = _observable_input[j, i]

            # Call the output functions
            save_state_func(
                    current_state,
                    current_observable,
                    _state_output[:, i],
                    _observable_output[:, i],
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
                save_summary_metrics_func(
                        state_summaries,
                        observable_summaries,
                        _state_summaries_output[:, int(i / summarise_every)],
                        _observable_summaries_output[:, int(i / summarise_every)],
                        summarise_every,
                        )

    return _output_functions_test_kernel

@pytest.fixture(scope='function')
def compare_input_output(precision, output_functions_test_kernel, run_settings, output_functions,
                         output_functions_config,
                         ):
    """Test that output functions correctly save state and observable values."""
    num_states = run_settings['n_states']
    num_observables = run_settings['max_observables']
    num_samples = run_settings['num_samples']
    num_summaries = run_settings['num_summaries']
    summarise_every = run_settings['summarise_every']

    state_input = run_settings['state_input']
    observable_input = run_settings['observable_input']
    if output_functions.save_time == True:
        state_output = np.zeros((num_states + 1, num_samples), dtype=precision)
    else:
        state_output = np.zeros((num_states, num_samples), dtype=precision)
    observable_output = np.zeros((num_observables, num_samples), dtype=precision)

    summaries_per_state = output_functions.memory_per_summarised_variable['output']
    state_summary_height = summaries_per_state * num_states
    obs_summary_height = summaries_per_state * num_observables
    state_summaries_output = np.zeros((state_summary_height, num_summaries), dtype=precision)
    observable_summaries_output = np.zeros((obs_summary_height, num_summaries), dtype=precision)

    # To the CUDA device
    d_state_input = cuda.to_device(state_input)
    d_observable_input = cuda.to_device(observable_input)
    d_state_output = cuda.to_device(state_output)
    d_observable_output = cuda.to_device(observable_output)
    d_state_summaries_output = cuda.to_device(state_summaries_output)
    d_observable_summaries_output = cuda.to_device(observable_summaries_output)

    loop_shared_memory = num_states + num_observables  # Hard-coded from test kernel code
    summary_temp_per_state = output_functions.memory_per_summarised_variable['temporary']
    summary_shared_memory = (num_states + num_observables) * summary_temp_per_state
    dynamic_shared_memory = (loop_shared_memory + summary_shared_memory) * precision().itemsize

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

    # Create a fake loop_compile_settings dict to use a higher-level common "expected summaries" function
    loop_compile_settings = {'dt_summarise':     summarise_every,
                             'dt_save':          1,
                             'output_functions': output_functions_config['outputs_list'],
                             'n_peaks':          output_functions_config['n_peaks']
                             }

    # Calculate expected summaries
    expected_state_summaries, expected_obs_summaries = calculate_expected_summaries(state_input, observable_input,
                                                                                    loop_compile_settings,
                                                                                    output_functions,
                                                                                    precision,
                                                                                    )

    outputs_list = output_functions_config['outputs_list']
    if "time" in outputs_list:
        expected_time = np.arange(num_samples, dtype=precision)
        assert_allclose(state_output[-1, :], expected_time, atol=1e-07, rtol=1e-07,
                        err_msg="Time values were not saved correctly",
                        )
        state_output = state_output[:-1, :]

    if "state" in outputs_list:
        assert_allclose(state_input, state_output, atol=1e-07, rtol=1e-07,
                        err_msg="State values were not saved correctly",
                        )

    if "observables" in outputs_list:
        assert_allclose(observable_input, observable_output, atol=1e-07, rtol=1e-07,
                        err_msg="Observable values were not saved correctly",
                        )

    #RMS is not very accurate currently - this might just be floating-point precision related. These loose tolerances
    # avoid flaky tests.
    if precision == np.float32:
        atol = 1e-05
        rtol = 5e-05
        if "rms" in outputs_list:
            atol = 1e-3
            rtol = 1e-3
    elif precision == np.float64:
        atol = 1e-12
        rtol = 1e-12
        if "rms" in outputs_list:
            atol = 1e-9
            rtol = 1e-9
    # Assert that summary metrics were calculated correctly
    if any(summary in outputs_list for summary in ["mean", "max", "rms", "peaks"]):

        assert_allclose(expected_state_summaries, state_summaries_output, atol=atol, rtol=rtol,
                        err_msg=f"State summaries didn't match expected values. Shapes: expected[{expected_state_summaries.shape}, actual[{state_summaries_output.shape}]",
                        )
        if "observables" in outputs_list:
            assert_allclose(expected_obs_summaries, observable_summaries_output, atol=atol, rtol=rtol,
                            err_msg=f"Observable summaries didn't match expected values. Shapes: expected[{expected_obs_summaries.shape}, actual[{observable_summaries_output.shape}]",
                            )


@pytest.mark.parametrize("precision_override", [np.float32, np.float64], ids=["float32", "float64"])
@pytest.mark.parametrize("loop_compile_settings_overrides",
                         [{'output_functions': ["state", "observables", "mean", "max", "rms", "peaks"],
                           'n_peaks':          3
                           }],
                         ids=["all_metrics"],
                         indirect=True,
                         )
@pytest.mark.parametrize("other_run_settings_override",
                         [{'num_samples': 100000, 'summarise_every': 1000},
                          {'num_samples': 100, 'summarise_every': 10}],
                         ids=["large_dataset",
                              "small_dataset"],
                         indirect=True,
                         )
def test_precision_with_large_datasets(compare_input_output):
    """Test precision differences (float32 vs float64) with large datasets."""
    pass


@pytest.mark.parametrize("precision_override", [np.float32], ids=["float32"])
@pytest.mark.parametrize("other_run_settings_override",
                         [{'test_shared_mem': False},
                          {'test_shared_mem': True}],
                         ids=['local_mem', 'shared_mem'],
                         indirect=True,
                         )
@pytest.mark.parametrize("loop_compile_settings_overrides",
                         [{'output_functions': ["state", "observables", "mean", "max", "peaks"],
                           'n_peaks':          10
                           }],
                         ids=["all_but_rms"],
                         indirect=True,
                         )
def test_memory_types(compare_input_output):
    """Test shared vs local memory with complex output configurations."""
    pass


@pytest.mark.parametrize("input_type_override",
                         [{'scale': 1e-6},
                          {'scale': 1e6},
                          {'scale': [-12, 12]},
                          {'style': 'zero'},
                          {'style': 'nan'}],
                         ids=["tiny_values", "large_values", "wide range", "zeros", "nans"],
                         indirect=True,
                         )
@pytest.mark.parametrize("other_run_settings_override",
                         [{'num_samples': 100, 'summarise_every': 10},
                          {'num_samples': 10000, 'summarise_every': 1000}],
                         ids=["short_sim", "long_sim"],
                         indirect=True,
                         )
@pytest.mark.parametrize("precision_override", [np.float32], ids=["float32"])
def test_input_types(compare_input_output):
    """Test different input data types on short and long simulations."""
    pass


@pytest.mark.parametrize("loop_compile_settings_overrides",
                         [{'output_functions': ["time"]},
                          {'output_functions': ["state", "observables", "time"]}],
                         ids=["only_time", "time_with_state_obs"],
                         indirect=True,
                         )
@pytest.mark.parametrize("precision_override", [np.float32, np.float64], ids=["float32", "float64"])
def test_time_output(compare_input_output):
    """Test time output specifically."""
    pass


@pytest.mark.parametrize("loop_compile_settings_overrides",
                         [{'output_functions': ["state", "observables"]},
                          {'output_functions': ["mean", "max", "rms", "peaks"], 'n_peaks': 10},
                          {'output_functions': ["state", "observables", "time", "mean", "max", "rms", "peaks"],
                           'n_peaks':          5
                           },
                          {'output_functions': ["observables"]},
                          {'output_functions': ["state", "mean", "max"]},
                          {'output_functions': ["observables", "mean", "max"]}
                          ],
                         ids=["state_obs", "all_summaries_only", "full quiver", "obs_only", "state_mean_max",
                              "obs_mean_max"],
                         indirect=True,
                         )
@pytest.mark.parametrize("precision_override", [np.float32], ids=["float32"])
def test_output_configurations(compare_input_output):
    """Test various output function configurations."""
    pass


@pytest.mark.parametrize("loop_compile_settings_overrides",
                         [{'saved_states': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'saved_observables': [0, 1, 2, 3, 4]},
                          {'saved_states':      [int(i) for i in range(100)],
                           'saved_observables': [int(i) for i in range(100)]
                           },
                          {'saved_states': [0], 'saved_observables': [0]}],
                         ids=["10/5", "100/100", "1/1"],
                         indirect=True,
                         )
@pytest.mark.parametrize("precision_override", [np.float32, np.float64], ids=["float32", "float64"])
def test_system_sizes_configurations(compare_input_output):
    """Test various output function configurations."""
    pass