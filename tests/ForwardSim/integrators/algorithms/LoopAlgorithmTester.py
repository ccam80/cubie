#
# import os
# os.environ["NUMBA_ENABLE_CUDASIM"] = "1"
# os.environ["NUMBA_CUDA_DEBUGINFO"] = "1"
# os.environ["NUMBA_OPT"] = "0"

import pytest
import numpy as np
from numba import cuda, from_dtype
from numpy.testing import assert_allclose

from CuMC.ForwardSim.integrators.algorithms.genericIntegratorAlgorithm import GenericIntegratorAlgorithm
from conftest import calculate_expected_summaries


class TestLoopAlgorithm:
    """Base class for testing loop algorithms with different systems and configurations. Doesn't use the
    attribute or "self" system, but is set up as a class for easy inheritance and parameterization of tests.

    All parameters are set using fixtures in tests/conftest.py (general for all testers) and
    tests/integrators/algorithms/conftest.py (for things like inputs, that will change with larger arrays of runs in
    higher-level modules, which can be overridden by parametrizing (for example) precision_override,
    loop_compile_settings_overrides, inputs_override, etc. - check the fixtures in conftests.py for a full list and their
    contents"""

    @pytest.fixture(scope="module")
    def algorithm_class(self):
        """OVERRIDE THIS FIXTURE in subclasses to provide the algorithm class to test. """
        return GenericIntegratorAlgorithm

    @pytest.fixture(scope='function')
    def expected_answer(self, loop_compile_settings, run_settings, inputs, precision):
        """OVERRIDE THIS FIXTURE with a python version of what your integrator loop should return - a scipy integrator
        or homebrew CPU loop should be able to provide an answer that matches the output of a given loop to within
        floating-point precision."""
        save_state = "state" in loop_compile_settings['output_functions']
        save_observables = "observables" in loop_compile_settings['output_functions']
        n_states_total = inputs['initial_values'].shape[0]
        n_samples = int(run_settings['duration'] / loop_compile_settings['dt_save'])

        if save_state:
            saved_states = np.asarray(loop_compile_settings['saved_states'])
            n_saved_states = len(saved_states)
            expected_state_output = np.zeros((n_saved_states, n_samples), dtype=precision)
            expected_state_output[:, :] = inputs['initial_values'][saved_states][:, np.newaxis]
        else:
            expected_state_output = np.zeros((0, n_samples), dtype=precision)

        if save_observables:

            saved_observables = np.asarray(loop_compile_settings['saved_observables'])
            n_saved_observables = len(saved_observables)
            expected_observables = np.zeros((n_saved_observables, n_samples), dtype=precision)
            expected_observables[:, :] = inputs['initial_values'][
                (saved_observables % n_states_total), np.newaxis]
        else:
            expected_observables = np.zeros((0, n_samples), dtype=precision)

        return expected_state_output, expected_observables


    @pytest.fixture(scope='function')
    def loop_under_test(self, request, precision, algorithm_class, output_functions, system, loop_compile_settings):
        """
        Returns an instance of the loop class specified in algorithm_class.

        Usage example:
        @pytest.mark.parametrize("system", ["ThreeChamber"], indirect=True)
        def test_loop_function(system, loop_under_test):
            ...
        """
        # Get the output functions from the output_functions fixture
        save_state = output_functions.save_state_func
        update_summaries = output_functions.update_summary_metrics_func
        save_summaries = output_functions.save_summary_metrics_func
        summary_temp_memory = output_functions.temp_memory_requirements

        algorithm_instance = algorithm_class(
            precision=from_dtype(precision),
            dxdt_func=system.dxdtfunc,
            n_states=system.num_states,
            n_obs=system.num_observables,
            n_par=system.num_parameters,
            n_drivers=system.num_drivers,
            dt_min=loop_compile_settings['dt_min'],
            dt_max=loop_compile_settings['dt_max'],
            dt_save=loop_compile_settings['dt_save'],
            dt_summarise=loop_compile_settings['dt_summarise'],
            atol=loop_compile_settings['atol'],
            rtol=loop_compile_settings['rtol'],
            save_state_func=save_state,  # These will be set by the ODEIntegratorLoop
            update_summary_func=update_summaries,
            save_summary_func=save_summaries,
            n_saved_states=len(loop_compile_settings['saved_states']),
            n_saved_observables=len(loop_compile_settings['saved_observables']),
            summary_temp_memory=summary_temp_memory,
        )

        return algorithm_instance

    @pytest.fixture(scope='function')
    def built_loop_function(self, loop_under_test):
        """Returns only the build loop function of the loop under test"""
        loop_under_test.build()
        return loop_under_test.loop_function

    @pytest.mark.parametrize("loop_compile_settings_overrides",
                             [{'dt_min': 0.01, 'dt_max': 0.1},
                              {'atol': 1e-5, 'rtol': 1e-4},
                              {"saved_states": [0, 1],
                               "saved_observables": [1, 2],
                               'output_functions': ["state", "observables"]},
                              {'dt_min': 0.002,
                               'dt_max': 0.02,
                               'dt_save': 0.02,
                               'dt_summarise': 0.2,
                               'atol': 1.0e-7,
                               'rtol': 1.0e-5,
                               'saved_states': [0, 1, 2],
                               'saved_observables': [0, 3],
                               'output_functions': ["state", "peaks"],
                               'n_peaks': 2}
                              ],
                             ids=['change_dts', 'change_tols', 'change_output_sizes', 'change_all'],
                             indirect=True)
    def test_loop_compile_settings_passed_successfully(self, loop_compile_settings_overrides,
                                                       loop_under_test, expected_summary_temp_memory):
        for key, value in loop_compile_settings_overrides.items():
            if key == "saved_states":
                assert loop_under_test.loop_parameters['n_saved_states'] == len(value), \
                    f"saved_states does not match expected value {len(value)}"
            elif key == "saved_observables":
                assert loop_under_test.loop_parameters['n_saved_observables'] == len(value), \
                    f"saved_states does not match expected value {len(value)}"
            elif key == "output_functions" or key == "n_peaks":
                assert loop_under_test.loop_parameters['summary_temp_memory'] == expected_summary_temp_memory, \
                    f"Summary temp memory requirement doesn't match expected - the loop_compile_settings change doesn't get through."
            else:
                assert key in loop_under_test.loop_parameters, f"{key} not found in loop parameters"
                assert loop_under_test.loop_parameters[key] == value, f"{key} does not match expected value {value}"

    @pytest.mark.parametrize("loop_compile_settings_overrides",
                             [{'dt_min': 0.01, 'dt_max': 0.1},
                              {'atol': 1e-5, 'rtol': 1e-4},
                              {"saved_states": [0, 1],
                               "saved_observables": [1, 2],
                               'output_functions': ["state", "observables"]},
                              {'dt_min': 0.002,
                               'dt_max': 0.02,
                               'dt_save': 0.02,
                               'dt_summarise': 0.2,
                               'atol': 1.0e-7,
                               'rtol': 1.0e-5,
                               'saved_states': [0, 1, 2],
                               'saved_observables': [0, 3],
                               'output_functions': ["state", "peaks"],
                               'n_peaks': 2}
                              ],
                             ids=['change_dts', 'change_tols', 'change_output_sizes', 'change_all'],
                             indirect=True)
    def test_loop_function_builds(self, built_loop_function):
        """
        Test that the loop function builds without errors.
        """
        assert callable(built_loop_function), "Loop function was not built successfully."

    #Don't call it test_kernel, as that is a reserved name in pytest
    @pytest.fixture(scope='function')
    def loop_test_kernel(self, precision, run_settings, loop_compile_settings, built_loop_function):
        loop_func = built_loop_function

        output_samples = int(run_settings['duration'] / loop_compile_settings['dt_save'])
        warmup_samples = int(run_settings['warmup'] / loop_compile_settings['dt_save'])
        numba_precision = from_dtype(precision)

        @cuda.jit()
        def test_kernel(inits,
                        params,
                        forcing_vector,
                        output,
                        observables,
                        summary_outputs,
                        summary_observables):
            c_forcing_vector = cuda.const.array_like(forcing_vector)

            shared_memory = cuda.shared.array(0, dtype=numba_precision)

            loop_func(
                inits,
                params,
                c_forcing_vector,
                shared_memory,
                output,
                observables,
                summary_outputs,
                summary_observables,
                output_samples,
                warmup_samples
            )

        return test_kernel


    @pytest.mark.parametrize("loop_compile_settings_overrides, inputs_override, run_settings_override",
                             [({'output_functions': ["state", "observables"], 'saved_states': [0, 1, 2]}, {}, {}),
                              ({'output_functions': ["state", "observables", "mean"],
                                'saved_states': [0, 1],
                                'saved_observables': [0, 1, 2]},
                               {},
                               {}),
                              ({}, {'initial_values': np.array([1.0, 2.0, 3.0])}, {}),
                              ({}, {}, {'duration': 5.0, 'warmup': 2.0}),
                             ({'output_functions': ["state", "observables", "mean", "max", "rms", "peaks"],
                               'n_peaks':3, 'saved_states': [0, 1, 2]}, {}, {})],
                             ids=['state_and_observables_empty_obs_list', 'state_observables_and_mean',
                                  'custom_initial_values', 'custom_run_settings', "all_summaries"],
                             indirect=True)
    def test_loop(self, loop_test_kernel, run_settings, loop_compile_settings, inputs, precision, output_functions,
                  loop_under_test, expected_answer):
        output_samples = int(run_settings['duration'] / loop_compile_settings['dt_save'])

        #TODO: Bring this logic into the ode integrator class, as it is a trap set by the divorcing of the output functions
        # from the output memory allocation.

        save_state_bool = "state" in loop_compile_settings['output_functions']
        save_observables_bool = "observables" in loop_compile_settings['output_functions']
        saved_states = np.asarray(loop_compile_settings['saved_states']) if save_state_bool else np.asarray([])
        saved_observables = np.asarray(loop_compile_settings['saved_observables']) if save_observables_bool else np.asarray([])
        n_saved_states = len(saved_states)
        n_saved_observables = len(saved_observables)

        output = cuda.pinned_array((n_saved_states, output_samples), dtype=precision)
        observables = np.zeros((n_saved_observables, output_samples), dtype=precision)

        summary_samples = int(
            np.ceil(output_samples * loop_compile_settings['dt_save'] / loop_compile_settings['dt_summarise']))
        num_state_summaries = output_functions.summary_output_length * n_saved_states
        num_observable_summaries = output_functions.summary_output_length * n_saved_observables

        summary_outputs = np.zeros((num_state_summaries, summary_samples), dtype=precision)
        summary_observables = np.zeros((num_observable_summaries, summary_samples), dtype=precision)
        forcing_vector = np.zeros(inputs['forcing_vectors'].shape, dtype=precision)

        inits = np.zeros(inputs['initial_values'].shape, dtype=precision)
        parameters = np.zeros(inputs['parameters'].shape, dtype=precision)
        forcing_vector[:, :] = inputs['forcing_vectors'][:, :]
        inits[:] = inputs['initial_values'][:]
        parameters[:] = inputs['parameters'][:]

        output[:, :] = precision(0.0)
        observables[:, :] = precision(0.0)
        summary_outputs[:, :] = precision(0.0)
        summary_observables[:, :] = precision(0.0)

        d_forcing = cuda.to_device(forcing_vector)
        d_inits = cuda.to_device(inits)
        d_params = cuda.to_device(parameters[:])
        d_output = cuda.to_device(output)
        d_observables = cuda.to_device(observables)
        d_summary_state = cuda.to_device(summary_outputs)
        d_summary_observables = cuda.to_device(summary_observables)

        # global sharedmem
        sharedmem = loop_under_test._calculate_loop_internal_shared_memory() + \
                    loop_under_test.loop_parameters['summary_temp_memory'] * (n_saved_states + n_saved_observables)

        loop_test_kernel[1, 1, 0, sharedmem](d_inits,
                                             d_params,
                                             d_forcing,
                                             d_output,
                                             d_observables,
                                             d_summary_state,
                                             d_summary_observables,
                                             )

        cuda.synchronize()
        output = d_output.copy_to_host()
        obs = d_observables.copy_to_host()
        summary_states = d_summary_state.copy_to_host()
        summary_observables = d_summary_observables.copy_to_host()

        expected_state_summaries, expected_obs_summaries = calculate_expected_summaries(*expected_answer,
                                                                                        loop_compile_settings,
                                                                                        output_functions,
                                                                                        precision)

        assert_allclose(expected_answer[0], output, err_msg="Output does not match expected.")
        assert_allclose(expected_answer[1], obs, err_msg="Observables do not match expected.")
        assert_allclose(expected_state_summaries, summary_states, err_msg="Summary states do not match expected.")
        assert_allclose(expected_obs_summaries, summary_observables, err_msg="Summary observables do not match expected.")



    @pytest.fixture(scope='function')
    def expected_summary_temp_memory(self, loop_compile_settings):
        """
        Calculate the expected temporary memory usage for the loop function.

        Usage example:
        @pytest.mark.parametrize("loop_compile_settings_overrides", [{'dt_min': 0.001, 'dt_max': 0.01}], indirect=True)
        def test_expected_temp_memory(expected_temp_memory):
            ...
        """
        from CuMC.ForwardSim.integrators.output_functions import _TempMemoryRequirements
        n_peaks = loop_compile_settings['n_peaks']
        outputs_list = loop_compile_settings['output_functions']
        return sum([_TempMemoryRequirements(n_peaks)[output_type] for output_type in outputs_list])


    @pytest.fixture(scope='function')
    def expected_summary_output_memory(self, loop_compile_settings):
        """
        Calculate the expected temporary memory usage for the loop function.

        Usage example:
        @pytest.mark.parametrize("loop_compile_settings_overrides", [{'dt_min': 0.001, 'dt_max': 0.01}], indirect=True)
        def test_expected_temp_memory(expected_temp_memory):
            ...
        """
        from CuMC.ForwardSim.integrators.output_functions import _OutputMemoryRequirements
        n_peaks = loop_compile_settings['n_peaks']
        outputs_list = loop_compile_settings['output_functions']
        return sum([_OutputMemoryRequirements(n_peaks)[output_type] for output_type in outputs_list])
