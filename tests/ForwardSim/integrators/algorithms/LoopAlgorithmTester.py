import pytest
import numpy as np
from numba import cuda, from_dtype
from numpy.testing import assert_allclose
from CuMC.ForwardSim.OutputHandling import summary_metrics
from tests._utils import calculate_expected_summaries

class LoopAlgorithmTester:
    """Base class for testing loop algorithms with different systems and configurations. Doesn't use the
    attribute or "self" system, but is set up as a class for easy inheritance and parameterization of tests.

    All parameters are set using fixtures in tests/_utils.py (general for all testers) and
    tests/integrators/algorithms/_utils.py (for things like inputs, that will change with larger arrays of runs in
    higher-level modules, which can be overridden by parametrizing (for example) precision_override,
    loop_compile_settings_overrides, inputs_override, etc. - check the fixtures in conftests.py for a full list and their
    contents"""

    @pytest.fixture(scope="module")
    def algorithm_class(self):
        """OVERRIDE THIS FIXTURE in subclasses to provide the algorithm class to test. """
        # e.g. return GenericIntegratorAlgorithm
        pass

    @pytest.fixture(scope='function')
    def expected_answer(self, system, loop_compile_settings, run_settings, inputs, precision):
        """OVERRIDE THIS FIXTURE with a python version of what your integrator loop should return - a scipy integrator
        or homebrew CPU loop should be able to provide an answer that matches the output of a given loop to within
        floating-point precision."""
        pass

    @pytest.fixture(scope='function')
    def loop_under_test(self, request, precision, algorithm_class: type, output_functions, system,
                        loop_compile_settings):
        """
        Returns an instance of the loop class specified in algorithm_class.

        Usage example:
        @pytest.mark.parametrize("system", ["ThreeChamber"], indirect=True)
        def test_loop_function(system, loop_under_test):
            ...
        """
        # Get the output functions from the output_functions fixture
        save_state = output_functions.save_state_func
        update_summaries = output_functions.update_summaries_func
        save_summaries = output_functions.save_summary_metrics_func
        summary_buffer_size = output_functions.memory_per_summarised_variable['buffer']
        save_time = output_functions.save_time

        dxdt_function = system.device_function

        algorithm_instance = algorithm_class(
                precision=from_dtype(precision),
                dxdt_func=dxdt_function,
                n_states=system.sizes.states,
                n_observables=system.sizes.observables,
                n_parameters=system.sizes.parameters,
                n_drivers=system.sizes.drivers,
                dt_min=loop_compile_settings['dt_min'],
                dt_max=loop_compile_settings['dt_max'],
                dt_save=loop_compile_settings['dt_save'],
                dt_summarise=loop_compile_settings['dt_summarise'],
                atol=loop_compile_settings['atol'],
                rtol=loop_compile_settings['rtol'],
                save_time=save_time,
                save_state_func=save_state,
                update_summary_func=update_summaries,
                save_summary_func=save_summaries,
                n_saved_states=len(loop_compile_settings['saved_states']),
                n_saved_observables=len(loop_compile_settings['saved_observables']),
                summary_buffer_size=summary_buffer_size,
                )

        return algorithm_instance

    @pytest.fixture(scope='function')
    def built_loop_function(self, loop_under_test):
        """Returns only the build loop function of the loop under test"""
        return loop_under_test.device_function

    def test_loop_compile_settings_passed_successfully(self, loop_compile_settings_overrides,
                                                       loop_under_test, expected_summary_buffer_memory):
        for key, value in loop_compile_settings_overrides.items():
            if key == "saved_states":
                assert loop_under_test.compile_settings.n_saved_states == len(value), \
                    f"saved_states does not match expected value {len(value)}"
            elif key == "saved_observables":
                assert loop_under_test.compile_settings.n_saved_observables == len(value), \
                    f"saved_states does not match expected value {len(value)}"
            elif key == "output_functions" or key == "n_peaks":
                assert loop_under_test.compile_settings.summaries_buffer_height == expected_summary_buffer_size, \
                    (f"Summary buffer requirement doesn't match expected - the loop_compile_settings change doesn't "
                     f"gethrough.")
            else:
                assert hasattr(loop_under_test.compile_settings, key), f"{key} not found in loop parameters"
                assert getattr(loop_under_test.compile_settings, key) == value, f"{key} does not match expected value {value}"

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

    @pytest.fixture(scope='function')
    def expected_summary_buffer_size(self, loop_compile_settings, output_functions):
        """
        Calculate the expected buffer memory usage for the loop function.

        Usage example:
        @pytest.mark.parametrize("loop_compile_settings_overrides", [{'dt_min': 0.001, 'dt_max': 0.01}], indirect=True)
        def test_expected_buffer_memory(expected_buffer_memory):
            ...
        """
        outputs_list = loop_compile_settings['output_functions']
        buffer_size = summary_metrics.summaries_buffer_height(outputs_list)
        return buffer_size

    @pytest.fixture(scope='function')
    def expected_summary_output_size(self, loop_compile_settings):
        """
        Calculate the expected output size usage for the loop function.

        Usage example:
        @pytest.mark.parametrize("loop_compile_settings_overrides", [{'dt_min': 0.001, 'dt_max': 0.01}], indirect=True)
        def test_expected_output_memory(expected_summary_output_memory):
            ...
        """
        outputs_list = loop_compile_settings['output_functions']
        output_size = sum(summary_metrics.output_sizes(outputs_list))


        return output_size

    def test_loop(self, loop_test_kernel, run_settings, loop_compile_settings, inputs, precision, output_functions,
                  loop_under_test, expected_answer):
        """Run the loop test kernel, checking the output against the expected answer. Override this in subclasses,
        but just call super().test_loop(...) to run the base test. with your own parameters added in"""
        output_samples = int(run_settings['duration'] / loop_compile_settings['dt_save'])

        save_state_bool = "state" in loop_compile_settings['output_functions']
        save_observables_bool = "observables" in loop_compile_settings['output_functions']
        saved_states = np.asarray(loop_compile_settings['saved_states']) if save_state_bool else np.asarray([])
        saved_observables = np.asarray(
                loop_compile_settings['saved_observables']) if save_observables_bool else np.asarray([])
        n_saved_states = len(saved_states)
        n_saved_observables = len(saved_observables)

        output = cuda.pinned_array((output_samples, n_saved_states), dtype=precision)
        observables = np.zeros((output_samples, n_saved_observables), dtype=precision)

        summary_samples = int(
                np.ceil(output_samples * loop_compile_settings['dt_save'] / loop_compile_settings['dt_summarise']))
        summary_output_memory = output_functions.memory_per_summarised_variable['output']

        num_state_summaries = summary_output_memory * n_saved_states
        num_observable_summaries = summary_output_memory * n_saved_observables

        summary_outputs = np.zeros((summary_samples, num_state_summaries), dtype=precision)
        summary_observables = np.zeros((summary_samples, num_observable_summaries), dtype=precision)
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

        # Shared memory requirements:
        loop_memory = loop_under_test.get_cached_output('loop_shared_memory')
        summary_memory = loop_under_test.compile_settings.summaries_buffer_height * (n_saved_states + n_saved_observables)
        floatsize = precision().itemsize

        dynamic_sharedmem = floatsize * (summary_memory + loop_memory)

        loop_test_kernel[1, 1, 0, dynamic_sharedmem](d_inits,
                                                     d_params,
                                                     d_forcing,
                                                     d_output,
                                                     d_observables,
                                                     d_summary_state,
                                                     d_summary_observables,
                                                     )

        cuda.synchronize()
        output = d_output.copy_to_host()
        observables = d_observables.copy_to_host()
        summary_states = d_summary_state.copy_to_host()
        summary_observables = d_summary_observables.copy_to_host()

        expected_state_summaries, expected_obs_summaries = calculate_expected_summaries(*expected_answer,
                                                                                        loop_compile_settings,
                                                                                        output_functions,
                                                                                        precision)

        if precision == np.float32:  #Allow for the numerical error expected in float32 calculations.
            atol = 1e-5
            rtol = 1e-5
        else:
            atol = 1e-12
            rtol = 1e-12

        if save_state_bool:
            assert_allclose(expected_answer[0], output, atol=atol, rtol=rtol, err_msg="Output does not match expected.")
        if save_observables_bool:
            assert_allclose(expected_answer[1], observables, atol=atol, rtol=rtol,
                            err_msg="Observables do not match expected.")
        assert_allclose(expected_state_summaries, summary_states, atol=atol, rtol=rtol,
                        err_msg="Summary states do not match expected.")
        assert_allclose(expected_obs_summaries, summary_observables, atol=atol, rtol=rtol,
                        err_msg="Summary observables do not match expected.")

    @pytest.fixture(scope='function')
    def expected_loop_shared_memory(self, system):
        """Override with your loops expected shared memory usage (not including summary memory, this is tested in
        test_output_functions.py)"""
        return 0

    def test_loop_shared_memory_calc(self, loop_under_test, expected_loop_shared_memory):
        """Test the calculate_shared_memory method of Euler."""

        shared_memory = loop_under_test.shared_memory_required
        assert shared_memory == expected_loop_shared_memory, f"Expected {expected_loop_shared_memory} shared memory items, got {shared_memory}"

