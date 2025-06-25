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
                   save_state_func,
                   update_summary_func,
                   save_summary_func,
                   n_saved_states,
                   n_saved_observables,
                   summary_temp_memory
                   ):

        internal_step_size = dt_min
        save_every_samples, summarise_every_samples, internal_step_size = self._time_to_samples()

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
            # Optimise: use one or two running indices to reduce registers required if the compiler doesn't get it -
            # it should probably get it.
            dxdt_start_index = n_states
            observables_start_index = n_states + dxdt_start_index
            drivers_start_index = observables_start_index + n_obs
            state_summaries_start_index = drivers_start_index + n_drivers
            observable_summaries_start_index = state_summaries_start_index + n_saved_states * summary_temp_memory
            end_index = observable_summaries_start_index + n_saved_observables * summary_temp_memory

            state = shared_memory[:n_states]
            dxdt = shared_memory[n_states:observables_start_index]
            observables = shared_memory[observables_start_index:drivers_start_index]
            drivers = shared_memory[drivers_start_index: state_summaries_start_index]
            state_summaries = shared_memory[
                              state_summaries_start_index:observable_summaries_start_index]  # Alias for drivers[-1] if no summaries.
            observables_summaries = shared_memory[observable_summaries_start_index: end_index]

            driver_length = forcing_vec.shape[0]

            # Initialise/Assign values to allocated memory
            shared_memory[:end_index] = precision(0.0)  # initialise all shared memory before adding values
            for i in range(n_states):
                state[:] = inits[i]

            # Optimise: Consider memory location of these parameters; it will probably differ based on system size.
            l_parameters = cuda.local.array((n_par),
                                            dtype=precision)

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
                    save_state_func(state, observables, state_output[:, i], observables_output[:, i],
                                    i - warmup_samples)
                    update_summary_func(state, observables, state_summaries, observables_summaries, i - warmup_samples)

                    if (i + 1) % summarise_every_samples == 0:
                        summary_sample = (i + 1) // summarise_every_samples - 1
                        save_summary_func(state_summaries, observables_summaries,
                                          state_summaries_output[:, summary_sample],
                                          observables_summaries_output[:, summary_sample], i)

        return euler_loop

    def calculate_shared_memory(self):
        """
        Calculate the number of items in shared memory required for the loop - don't include summaries, they are handled
        outside the loop as they are common to all algorithms. This is just the number of items stored in shared memory
        for state, dxdt, observables, drivers, which will change between algorithms.
        """

        n_states = self.loop_parameters['n_states']
        n_obs = self.loop_parameters['n_obs']
        n_drivers = self.loop_parameters['n_drivers']

        return n_states + n_states + n_obs + n_drivers

    def _time_to_samples(self):
        """Fixed-step helper function: Convert the time-based compile_settings to sample-based compile_settings, which are used by
        fixed-step loop functions. Sanity-check values and warn the user if they don't work."""

        dt_min = self.loop_parameters['dt_min']
        dt_save = self.loop_parameters['dt_save']
        dt_summarise = self.loop_parameters['dt_summarise']

        save_every_samples = int32(round(dt_save / dt_min))
        summarise_every_samples = int32(round(dt_summarise / dt_save))

        # summarise every < 1 won't make much sense.
        if dt_save < dt_min:
            raise ValueError(
                "dt_save must be a longer period than dtmin, as it sets the number of loop-steps between saves, which must be >=1. ")
        if summarise_every_samples <= 1:
            raise ValueError(
                "dt_summarise must be greater than dt_save, as it sets the number of saved samples between summaries,"
                "which must be >1")


        # Update the actual save and summary intervals, which will differ from what was ordered if they are not
        # a multiple of the loop step size.
        # TODO: Figure this out for variable-step algorithms, or make a fixed-step only function.
        new_dt_save = save_every_samples * dt_min
        if new_dt_save != dt_save:
            self.loop_parameters['dt_save'] = new_dt_save
            warn(
                f"dt_save was set to {new_dt_save}s, because it is not a multiple of dtmin ({dt_min}s)."
                f"dt_save can only save a value after an integer number of steps in a fixed-step integrator", UserWarning)

        new_dt_summarise = summarise_every_samples * dt_save
        if new_dt_summarise != dt_summarise:
            self.loop_parameters['dt_summarise'] = new_dt_summarise
            warn(
                f"dt_summarise was set to {new_dt_summarise}s, because it is not a multiple of dt_save ({dt_save}s)."
                f"dt_summarise can only save a value after an integer number of steps in a fixed-step integrator", UserWarning)

        return save_every_samples, summarise_every_samples, dt_min

