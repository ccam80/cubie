import pytest
import numpy as np
from numba import cuda, from_dtype
from numpy.testing import assert_allclose
from CuMC.ForwardSim.OutputHandling import summary_metrics
from CuMC.ForwardSim.OutputHandling.output_sizes import LoopBufferSizes, SingleRunOutputSizes
from CuMC.ForwardSim.integrators.algorithms.LoopStepConfig import LoopStepConfig
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
    def expected_answer(self, system, loop_compile_settings, run_settings, solver, inputs, precision):
        """OVERRIDE THIS FIXTURE with a python version of what your integrator loop should return - a scipy integrator
        or homebrew CPU loop should be able to provide an answer that matches the output of a given loop to within
        floating-point precision."""
        pass

    @pytest.fixture(scope='function')
    def expected_summaries(self, expected_answer, loop_under_test, loop_compile_settings,
                           output_functions, precision):
        """
        Calculate the expected summaries for the loop algorithm.

        Usage example:
        @pytest.mark.parametrize("summarise_every", [10], indirect=True)
        def test_expected_summaries(expected_summaries):
            ...
        """
        _, summarise_every, _ = loop_under_test.compile_settings.fixed_steps

        state, obs =  calculate_expected_summaries(
                *expected_answer,
                summarise_every,
                loop_compile_settings['output_functions'],
                output_functions.summaries_output_height_per_var,
                precision
                )
        return {'state': state, 'observables': obs}

    @pytest.fixture(scope='function')
    def buffer_sizes(self, system, output_functions):
        """Create LoopBufferSizes from system and output functions."""
        return LoopBufferSizes.from_system_and_output_fns(system, output_functions)


    @pytest.fixture(scope='function')
    def loop_under_test(self, request, precision, algorithm_class: type, output_functions, system,
                        buffer_sizes, run_settings):
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

        dxdt_function = system.device_function

        algorithm_instance = algorithm_class(
                precision=precision,
                dxdt_function=dxdt_function,
                buffer_sizes=buffer_sizes,
                loop_step_config=run_settings.loop_step_config,
                save_state_func=save_state,
                update_summaries_func=update_summaries,
                save_summaries_func=save_summaries,
                )

        return algorithm_instance

    @pytest.fixture(scope='function')
    def built_loop_function(self, loop_under_test):
        """Returns only the build loop function of the loop under test"""
        return loop_under_test.build()

    def test_loop_compile_settings_passed_successfully(self, loop_compile_settings_overrides, output_functions,
                                                       loop_under_test, expected_summary_buffer_size):
        """Test that compile settings are correctly passed to the loop algorithm."""
        compile_settings = loop_under_test.compile_settings

        for key, value in loop_compile_settings_overrides.items():
            if key == "saved_states":
                assert compile_settings.buffer_sizes.state >= len(value), \
                    f"saved_states buffer size does not accommodate expected value {len(value)}"
            elif key == "saved_observables":
                assert compile_settings.buffer_sizes.observables >= len(value), \
                    f"saved_observables buffer size does not accommodate expected value {len(value)}"
            elif key == "output_functions" or key == "n_peaks":
                # Check that summary buffer sizes match expectations
                assert compile_settings.buffer_sizes.state_summaries == output_functions.state_summaries_buffer_height, \
                    (f"Summary buffer requirement doesn't match expected - the loop_compile_settings change doesn't "
                     f"get through.")
            elif hasattr(compile_settings, key):
                assert getattr(compile_settings, key) == value, f"{key} does not match expected value {value}"

    def test_loop_function_builds(self, built_loop_function):
        """
        Test that the loop function builds without errors.
        """
        assert callable(built_loop_function), "Loop function was not built successfully."

    #Don't call it test_kernel, as that is a reserved name in pytest
    @pytest.fixture(scope='function')
    def loop_test_kernel(self, precision, built_loop_function, solver):
        loop_func = built_loop_function

        output_samples = solver.output_length
        warmup_samples = int(solver.warmup / solver.dt_save)
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

    @pytest.fixture(scope='function')
    def outputs(self, output_functions, precision, solver):
        output_shapes = SingleRunOutputSizes.from_solver(solver).nonzero
        state_output = cuda.pinned_array(output_shapes.state, dtype=precision)
        observables_output = cuda.pinned_array(output_shapes.observables, dtype=precision)
        state_summary_output = cuda.pinned_array(output_shapes.state_summaries, dtype=precision)
        observable_summary_output = cuda.pinned_array(output_shapes.observable_summaries,
                                                      dtype=precision)

        # Initialize output arrays to zero
        state_output[:, :] = precision(0.0)
        observables_output[:, :] = precision(0.0)
        state_summary_output[:, :] = precision(0.0)
        observable_summary_output[:, :] = precision(0.0)

        return {'state': state_output,
                'observables': observables_output,
                'state_summary': state_summary_output,
                'observable_summary': observable_summary_output,
                }

    def test_loop(self, loop_test_kernel, outputs, inputs, precision, output_functions,
                  loop_under_test, expected_answer, expected_summaries, solver):
        """Run the loop test kernel, checking the output against the expected answer."""
        save_state = output_functions.compile_settings.save_state
        save_observables = output_functions.compile_settings.save_observables
        summarise_state = output_functions.compile_settings.summarise_observables
        summarise_observables = output_functions.compile_settings.summarise_observables

        state_output = outputs['state']
        observables_output = outputs['observables']
        state_summary_output = outputs['state_summary']
        observable_summary_output = outputs['observable_summary']

        forcing_vector = inputs['forcing_vectors']
        inits = inputs['initial_values']
        parameters = inputs['parameters']

        # Transfer to GPU
        d_forcing = cuda.to_device(forcing_vector)
        d_inits = cuda.to_device(inits)
        d_params = cuda.to_device(parameters)
        d_output = cuda.to_device(state_output)
        d_observables = cuda.to_device(observables_output)
        d_summary_state = cuda.to_device(state_summary_output)
        d_summary_observables = cuda.to_device(observable_summary_output)

        # Calculate shared memory requirements
        loop_memory = loop_under_test.shared_memory_required
        summary_memory = (output_functions.state_summaries_buffer_height +
                         output_functions.observable_summaries_buffer_height)
        floatsize = precision().itemsize
        dynamic_sharedmem = floatsize * (summary_memory + loop_memory)

        # Run the kernel
        loop_test_kernel[1, 1, 0, dynamic_sharedmem](
            d_inits,
            d_params,
            d_forcing,
            d_output,
            d_observables,
            d_summary_state,
            d_summary_observables,
        )

        cuda.synchronize()

        # Copy results back to host
        state_output = d_output.copy_to_host()
        observables_output = d_observables.copy_to_host()
        state_summary_output = d_summary_state.copy_to_host()
        observable_summary_output = d_summary_observables.copy_to_host()

        # Calculate expected summaries
        expected_state_summaries = expected_summaries['state']
        expected_obs_summaries = expected_summaries['observables']


        # Set tolerance based on precision
        if precision == np.float32:
            atol = 1e-5
            rtol = 1e-5
        else:
            atol = 1e-12
            rtol = 1e-12

        # Assert results match expectations
        if save_state:
            assert_allclose(expected_answer[0], state_output, atol=atol, rtol=rtol,
                           err_msg="Output does not match expected.")
        if save_observables:
            assert_allclose(expected_answer[1], observables_output, atol=atol, rtol=rtol,
                           err_msg="Observables do not match expected.")
        if summarise_state:
            assert_allclose(expected_state_summaries, state_summary_output, atol=atol, rtol=rtol,
                           err_msg="Summary states do not match expected.")
        if summarise_observables:
            assert_allclose(expected_obs_summaries, observable_summary_output, atol=atol, rtol=rtol,
                           err_msg="Summary observables do not match expected.")

    @pytest.fixture(scope='function')
    def expected_loop_shared_memory(self, system, output_functions):
        """Override with your loops expected shared memory usage (not including summary memory, this is tested in
        test_output_functions.py)"""
        return 0

    def test_loop_shared_memory_calc(self, loop_under_test, expected_loop_shared_memory):
        """Test the calculate_shared_memory method of Euler."""

        shared_memory = loop_under_test.shared_memory_required
        assert shared_memory == expected_loop_shared_memory, f"Expected {expected_loop_shared_memory} shared memory items, got {shared_memory}"
