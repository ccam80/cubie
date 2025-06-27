import pytest
import numpy as np
from numba import cuda
from numpy.testing import assert_allclose


class LoopAlgorithmTester:
    """Class to test common functionality of integrator algorithms. Subclass this to add your algorithm as self.algorithm,
    and overload the "correct_answer" fixture to provide the expected output for a given input."""

    @pytest.fixture(scope="class")
    def algorithm_class(self, system_class, integrator_algorithm, system_settings, compile_settings, run_settings):
        """Override this fixture in subclasses to provide the algorithm class to test. """
        return None

    @pytest.fixture(scope="class")
    def system_instance(self, system_class, integrator_algorithm, system_settings, compile_settings, run_settings):
        """Override this fixture in subclasses to provide the system class to use for testing. Instantiate and build it
        in this step, we're not testing the system itself. """
        return None


    @pytest.fixture(scope="class")
    def integration_settings(self, system_class, integrator_algorithm, system_settings, compile_settings, run_settings):
        """Override this fixture in subclasses to provide integration settings."""
        return {
            'dt_min': 0.001,
            'dt_max': 0.01,
            'dt_save': 0.01,
            'dt_summarise': 0.1,
            'atol': 1e-6,
            'rtol': 1e-3,
            'saved_states': None,  # Default to all states
            'saved_observables': None,  # Default to no observables
            'output_functions': ["state"]
        }

    def instantiate_system(self, system_class, integrator_algorithm, system_settings, compile_settings, run_settings):
        """Instantiate the system to be used for testing."""
        self.system_instance = system_class(precision=precision)
        self.system_instance.build()
        return self.system_instance

    def instantiate_algorithm(self, system_class, integrator_algorithm, system_settings, compile_settings, run_settings):
        """Instantiate the algorithm to be tested."""
        self.algorithm_instance = algorithm_class(
            precision=system.precision,
            dxdt_func=system.dxdtfunc,
            n_states=system.num_states,
            n_obs=system.num_observables,
            n_par=system.num_parameters,
            n_drivers=system.num_drivers,
            dt_min=integration_settings['dt_min'],
            dt_max=integration_settings['dt_max'],
            dt_save=integration_settings['dt_save'],
            dt_summarise=integration_settings['dt_summarise'],
            atol=integration_settings['atol'],
            rtol=integration_settings['rtol'],
            save_state_func=None,  # These will be set by the ODEIntegratorLoop
            update_summary_func=None,
            save_summary_func=None,
            n_saved_states=system.num_states if integration_settings['saved_states'] is None else len(
                integration_settings['saved_states']),
            n_saved_observables=0 if integration_settings['saved_observables'] is None else len(
                integration_settings['saved_observables']),
            summary_temp_memory=1  # Default value, will be overridden by ODEIntegratorLoop
        )
        return self.algorithm_instance

    def instantiate_integrator(self, system_class, integrator_algorithm, system_settings, compile_settings, run_settings):
        """Instantiate the ODEIntegratorLoop with the algorithm and system."""
        from CuMC.ForwardSim.integrators.ODEIntegrator import ODEIntegratorLoop

        self.integrator_instance = ODEIntegratorLoop(
            system=system,
            algorithm=algorithm_class.__name__.lower(),
            saved_states=integration_settings['saved_states'],
            saved_observables=integration_settings['saved_observables'],
            dtmin=integration_settings['dt_min'],
            dtmax=integration_settings['dt_max'],
            atol=integration_settings['atol'],
            rtol=integration_settings['rtol'],
            dt_save=integration_settings['dt_save'],
            dt_summarise=integration_settings['dt_summarise'],
            output_functions=integration_settings['output_functions']
        )
        return self.integrator_instance

    def test_instantiation(self, system_class, integrator_algorithm, system_settings, compile_settings, run_settings, test_name):
        """Test that the algorithm instantiates correctly."""
        system = self.instantiate_system(system_class)
        algorithm = self.instantiate_algorithm(algorithm_class, system, integration_settings)

        assert isinstance(algorithm, algorithm_class), "Algorithm did not instantiate as expected."
        assert algorithm.loop_parameters[
                   'n_states'] == system.num_states, "Algorithm has misinterpreted the number of states."
        assert algorithm.loop_parameters[
                   'n_obs'] == system.num_observables, "Algorithm has misinterpreted the number of observables."
        assert algorithm.loop_parameters[
                   'n_par'] == system.num_parameters, "Algorithm has misinterpreted the number of parameters."
        assert algorithm.loop_parameters[
                   'n_drivers'] == system.num_drivers, "Algorithm has misinterpreted the number of drivers."

    def test_loop_function_creation(self, system_class, integrator_algorithm, system_settings, compile_settings, run_settings, test_name):
        """Test that the algorithm creates a loop function."""
        system = self.instantiate_system(system_class)
        algorithm = self.instantiate_algorithm(algorithm_class, system, integration_settings)

        # Build the loop function
        algorithm.build()

        assert algorithm.loop_function is not None, "Loop function was not created."

    def test_shared_memory_calculation(self, system_class, integrator_algorithm, system_settings, compile_settings, run_settings, test_name):
        """Test that the algorithm calculates shared memory requirements correctly."""
        system = self.instantiate_system(system_class)
        algorithm = self.instantiate_algorithm(algorithm_class, system, integration_settings)

        # Calculate shared memory
        shared_memory = algorithm.calculate_shared_memory()

        # For Euler, shared memory should be 2*n_states + n_obs + n_drivers
        expected_shared_memory = 2 * system.num_states + system.num_observables + system.num_drivers
        assert shared_memory == expected_shared_memory, f"Expected {expected_shared_memory} shared memory items, got {shared_memory}"

    def test_integration_with_system(self, system_class, integrator_algorithm, system_settings, compile_settings, run_settings, test_name):
        """Test that the algorithm integrates the system correctly."""
        system = self.instantiate_system(system_class)
        integrator = self.instantiate_integrator(system, algorithm_class, integration_settings)

        # Set up test parameters
        duration = 0.1
        warmup = 0.0
        dt_save = integration_settings['dt_save']

        output_samples = int(duration / dt_save)
        warmup_samples = int(warmup / dt_save)

        # Create test kernel
        @cuda.jit()
        def test_kernel(inits, params, forcing_vector, output, observables, summary_outputs, summary_observables):
            # Use constant memory for forcing vector
            c_forcing_vector = cuda.const.array_like(forcing_vector)

            # Allocate shared memory
            shared_memory = cuda.shared.array(0, dtype=system.precision)

            # Call the integrator loop function
            integrator.integrator_algorithm.loop_function(
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

        # Allocate memory for inputs and outputs
        output = np.zeros((system.num_states, output_samples), dtype=system.precision)
        observables = np.zeros((system.num_observables, output_samples), dtype=system.precision)
        summary_outputs = np.zeros((1, 1), dtype=system.precision)  # Placeholder
        summary_observables = np.zeros((1, 1), dtype=system.precision)  # Placeholder

        # Create forcing vector (all zeros for simplicity)
        forcing_vector = np.zeros((output_samples + warmup_samples, system.num_drivers), dtype=system.precision)

        # Copy to device
        d_inits = cuda.to_device(system.init_values.values_array)
        d_params = cuda.to_device(system.parameters.values_array)
        d_forcing = cuda.to_device(forcing_vector)
        d_output = cuda.to_device(output)
        d_observables = cuda.to_device(observables)
        d_summary_outputs = cuda.to_device(summary_outputs)
        d_summary_observables = cuda.to_device(summary_observables)

        # Get shared memory requirements
        shared_mem = integrator.dynamic_sharedmem

        # Run the kernel
        test_kernel[1, 1, 0, shared_mem](
            d_inits,
            d_params,
            d_forcing,
            d_output,
            d_observables,
            d_summary_outputs,
            d_summary_observables
        )

        # Copy results back to host
        cuda.synchronize()
        result_output = d_output.copy_to_host()
        result_observables = d_observables.copy_to_host()

        # Get expected results
        expected_output, expected_observables = self.correct_answer(system, integration_settings, duration, dt_save)

        # Compare results
        assert_allclose(result_output, expected_output, rtol=1e-3, atol=1e-6, err_msg="Output mismatch")
        assert_allclose(result_observables, expected_observables, rtol=1e-3, atol=1e-6, err_msg="Observables mismatch")

    def correct_answer(self, system, integration_settings, duration, dt_save):
        """Override this method in subclasses to provide the expected output for a given input.

        By default, returns zeros for all outputs.
        """
        output_samples = int(duration / dt_save)
        output = np.zeros((system.num_states, output_samples), dtype=system.precision)
        observables = np.zeros((system.num_observables, output_samples), dtype=system.precision)
        return output, observables