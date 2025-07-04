from warnings import warn
from numba import cuda, int32
from CuMC.ForwardSim.integrators.algorithms.genericIntegratorAlgorithm import GenericIntegratorAlgorithm

class Euler(GenericIntegratorAlgorithm):
    """Euler integrator algorithm for fixed-step integration.

    This is a simple, first-order integrator that uses the Euler method to update the state of the system.
    It is suitable for systems where the dynamics are not too stiff and where high accuracy is not required.
    """

    def __init__(self,precision,
                 dxdt_func,
                 n_states,
                 n_obs,
                 n_par,
                 n_drivers,
                 dt_min,
                 dt_max,
                 dt_save,
                 dt_summarise,
                 atol,
                 rtol,
                 save_time,
                 save_state_func,
                 update_summary_func,
                 save_summary_func,
                 n_saved_states,
                 n_saved_observables,
                 summary_temp_memory,
                 **kwargs):
        super().__init__(precision,
                         dxdt_func,
                         n_states,
                         n_obs,
                         n_par,
                         n_drivers,
                         dt_min,
                         dt_max,
                         dt_save,
                         dt_summarise,
                         atol,
                         rtol,
                         save_time,
                         save_state_func,
                         update_summary_func,
                         save_summary_func,
                         n_saved_states,
                         n_saved_observables,
                         summary_temp_memory)


    def build_loop(self,
                   precision,
                   dxdt_func,
                   n_states,
                   n_obs,
                   n_par,
                   n_drivers,
                   dt_min,
                   dt_max,
                   dt_save,
                   dt_summarise,
                   atol,
                   rtol,
                   save_time,
                   save_state_func,
                   update_summary_func,
                   save_summary_func,
                   n_saved_states,
                   n_saved_observables,
                   summary_temp_memory
                   ):

        internal_step_size = dt_min
        save_every_samples, summarise_every_samples, internal_step_size = self._time_to_fixed_steps()
        if save_time:
            state_array_size = n_states + 1
        else:
            state_array_size = n_states

        @cuda.jit((precision[:],
                   precision[:],
                   precision[:, :],
                   precision[:],
                   precision[:, :],
                   precision[:, :],
                   precision[:, :],
                   precision[:, :],
                   int32,
                   int32,
                   ),
                  device=True,
                  inline=True)
        def euler_loop(inits,
                       parameters,
                       forcing_vec,
                       shared_memory,
                       state_output,
                       observables_output,
                       state_summaries_output,
                       observables_summaries_output,
                       output_length,
                       warmup_samples=0):
            """

            """

            # Allocate shared memory slices

            dxdt_start_index = state_array_size
            observables_start_index = n_states + dxdt_start_index
            drivers_start_index = observables_start_index + n_obs
            state_summaries_start_index = drivers_start_index + n_drivers
            observable_summaries_start_index = state_summaries_start_index + n_saved_states * summary_temp_memory
            end_index = observable_summaries_start_index + n_saved_observables * summary_temp_memory

            state = shared_memory[:dxdt_start_index]
            dxdt = shared_memory[dxdt_start_index:observables_start_index]
            observables = shared_memory[observables_start_index:drivers_start_index]
            drivers = shared_memory[drivers_start_index: state_summaries_start_index]
            state_summaries = shared_memory[
                              state_summaries_start_index:observable_summaries_start_index]  # Alias for drivers[-1] if no summaries.
            observables_summaries = shared_memory[observable_summaries_start_index: end_index]

            driver_length = forcing_vec.shape[0]

            # Initialise/Assign values to allocated memory
            shared_memory[:end_index] = precision(0.0)  # initialise all shared memory before adding values
            for i in range(n_states):
                state[i] = inits[i]

            # Optimise: Consider memory location of these parameters; it will probably differ based on system size.
            # HACK: This is a workaround for a zero-parameter system, which allocates a one-element array that we
            #  then don't use. There has to be a more elegant way to handle this.
            if n_par > 0:
                l_parameters = cuda.local.array((n_par),
                                                dtype=precision)
            else:
                l_parameters = cuda.local.array((1), dtype=precision)

            for i in range(n_par):
                l_parameters[i] = parameters[i]

            # Loop through output samples, one iteration per output sample
            for i in range(warmup_samples + output_length):

                # Euler loop - internal step size <= outout step size
                for j in range(save_every_samples):
                    for k in range(n_drivers):
                        drivers[k] = forcing_vec[(i * save_every_samples + j) % driver_length, k]

                    # Calculate derivative at sample
                    dxdt_func(state,
                              parameters,
                              drivers,
                              observables,
                              dxdt)

                    # Forward-step state using euler
                    for k in range(n_states):
                        state[k] += dxdt[k] * internal_step_size

                # Start saving only after warmup period (to get past transient behaviour)
                if i > (warmup_samples - 1):
                    output_sample = i - warmup_samples
                    save_state_func(state, observables, state_output[:, output_sample], observables_output[:, output_sample],
                                    output_sample)
                    update_summary_func(state, observables, state_summaries, observables_summaries, output_sample)

                    if (i + 1) % summarise_every_samples == 0:
                        summary_sample = (output_sample + 1) // summarise_every_samples - 1
                        save_summary_func(state_summaries, observables_summaries,
                                          state_summaries_output[:, summary_sample],
                                          observables_summaries_output[:, summary_sample], summarise_every_samples)

        return euler_loop

    def _calculate_loop_internal_shared_memory(self):
        """
        Calculate the number of items in shared memory required for the loop - don't include summaries, they are handled
        outside the loop as they are common to all algorithms. This is just the number of items stored in shared memory
        for state, dxdt, observables, drivers, which will change between algorithms.
        """

        n_states = self.loop_parameters['n_states']
        n_obs = self.loop_parameters['n_obs']
        n_drivers = self.loop_parameters['n_drivers']

        return n_states + n_states + n_obs + n_drivers

