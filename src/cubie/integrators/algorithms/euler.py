from numba import cuda, int32, from_dtype

from cubie.integrators.algorithms.genericIntegratorAlgorithm import \
    GenericIntegratorAlgorithm


class Euler(GenericIntegratorAlgorithm):
    """Euler integrator algorithm for fixed-step integration.

    This is a simple, first-order integrator that uses the Euler method to
    update the state of the system.
    It is suitable for systems where the dynamics are not too stiff and where high accuracy is not required.
    """

    def __init__(self, precision, dxdt_function, buffer_sizes,
                 loop_step_config, save_state_func, update_summaries_func,
                 save_summaries_func, compile_flags=None, **kwargs, ):
        super().__init__(precision, dxdt_function, buffer_sizes,
                         loop_step_config, save_state_func,
                         update_summaries_func, save_summaries_func,
                         compile_flags=compile_flags, )

        self._threads_per_loop = 1

    def build_loop(self, precision, dxdt_function, save_state_func,
                   update_summaries_func, save_summaries_func, ):

        save_steps, summarise_steps, step_size = self.compile_settings.fixed_steps

        sizes = self.compile_settings.buffer_sizes
        flags = self.compile_settings.compile_flags
        save_observables_bool = flags.save_observables
        save_state_bool = flags.save_state
        summarise_observables_bool = flags.summarise_observables
        summarise_state_bool = flags.summarise_state

        state_buffer_size = sizes.state
        observables_buffer_size = sizes.observables
        dxdt_buffer_size = sizes.dxdt
        parameter_buffer_size = sizes.nonzero.parameters
        parameters_actual = sizes.parameters
        drivers_buffer_size = sizes.drivers
        state_summary_buffer_size = sizes.state_summaries
        observables_summary_buffer_size = sizes.observable_summaries

        # Generate indices into shared memory as compile-time constants
        dxdt_start_index = state_buffer_size
        observables_start_index = dxdt_start_index + dxdt_buffer_size
        drivers_start_index = observables_start_index + observables_buffer_size
        state_summaries_start_index = drivers_start_index + drivers_buffer_size
        observable_summaries_start_index = state_summaries_start_index + state_summary_buffer_size
        end_index = observable_summaries_start_index + observables_summary_buffer_size

        numba_precision = from_dtype(precision)

        @cuda.jit(
                (numba_precision[:], numba_precision[:], numba_precision[:, :],
                 numba_precision[:], numba_precision[:, :],
                 numba_precision[:, :], numba_precision[:, :],
                 numba_precision[:, :], int32, int32,), device=True,
                inline=True, )
        def euler_loop(inits, parameters, forcing_vec, shared_memory,
                       state_output, observables_output,
                       state_summaries_output, observables_summaries_output,
                       output_length, warmup_samples=0, ):
            """

            """

            # Allocate shared memory slices

            state_buffer = shared_memory[:dxdt_start_index]
            dxdt = shared_memory[dxdt_start_index:observables_start_index]
            observables_buffer = shared_memory[
                                 observables_start_index:drivers_start_index]
            drivers = shared_memory[
                      drivers_start_index: state_summaries_start_index]
            state_summary_buffer = shared_memory[
                                   state_summaries_start_index:observable_summaries_start_index]
            observable_summary_buffer = shared_memory[
                                        observable_summaries_start_index: end_index]

            driver_length = forcing_vec.shape[0]

            # Initialise/Assign values to allocated memory
            shared_memory[:end_index] = numba_precision(
                    0.0)  # initialise all shared memory before adding values
            for i in range(state_buffer_size):
                state_buffer[i] = inits[i]

            l_parameters = cuda.local.array((parameter_buffer_size),
                                            dtype=numba_precision, )

            for i in range(parameters_actual):
                l_parameters[i] = parameters[i]

            # Loop through output samples, one iteration per output sample
            for i in range(warmup_samples + output_length):

                # Euler loop - internal step size <= outout step size
                for j in range(save_steps):
                    for k in range(drivers_buffer_size):
                        drivers[k] = forcing_vec[
                            (i * save_steps + j) % driver_length, k]

                    # Calculate derivative at sample
                    dxdt_function(state_buffer, parameters, drivers,
                                  observables_buffer, dxdt, )

                    # Forward-step state using euler
                    for k in range(state_buffer_size):
                        state_buffer[k] += dxdt[k] * step_size

                # Start saving after the requested settling time has passed.
                if i > (warmup_samples - 1):
                    output_sample = i - warmup_samples
                    save_state_func(state_buffer, observables_buffer,
                                    state_output[
                                    output_sample * save_state_bool, :],
                                    observables_output[
                                    output_sample * save_observables_bool, :],
                                    output_sample, )
                    update_summaries_func(state_buffer, observables_buffer,
                                          state_summary_buffer,
                                          observable_summary_buffer,
                                          output_sample, )

                    if (i + 1) % summarise_steps == 0:
                        summary_sample = (
                                                 output_sample + 1) // summarise_steps - 1
                        save_summaries_func(state_summary_buffer,
                                            observable_summary_buffer,
                                            state_summaries_output[
                                            summary_sample * summarise_state_bool,
                                            :], observables_summaries_output[
                                                summary_sample * summarise_observables_bool,
                                                :], summarise_steps, )

        return euler_loop

    @property
    def shared_memory_required(self):
        """
        Calculate the number of items in shared memory required for the loop - don't include summaries_array, they are handled
        outside the loop as they are common to all algorithms. This is just the number of items stored in shared memory
        for state, dxdt, observables, drivers, which will change between algorithms.
        """

        sizes = self.compile_settings.buffer_sizes
        loop_shared_memory = (
                    sizes.state + sizes.dxdt + sizes.observables + sizes.drivers + sizes.state_summaries + sizes.observable_summaries)

        return loop_shared_memory
