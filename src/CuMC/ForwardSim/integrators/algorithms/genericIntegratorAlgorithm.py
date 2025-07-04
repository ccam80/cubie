from warnings import warn
from numba import cuda, int32
from CuMC.ForwardSim.integrators._utils import check_requested_timing_possible, convert_times_to_fixed_steps

from CuMC._utils import update_dicts_from_kwargs


class GenericIntegratorAlgorithm:
    """
    Base class for integrator algorithms.
    This class is not intended to be instantiated directly, but rather to be subclassed by specific integrator algorithms.

    Every integrator algorithm should have the following methods:
    - rebuild(**kwargs) - build the loop again with any modified parameters given in kwargs
    - build() - build the loop with the current parameters

    And the following attributes:
    - algo_shared_memory_items - the number of items (not memory, the number is used for indices before being converted to bytes)
    - loop_function - the compiled loop CUDA device function.

    Other helper methods can be added as needed, but this specifies the interface with higher level code.

    Subclasses should build on init.
    """

    def __init__(self,
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
                 summary_temp_memory):

        self.loop_function = None

        self.loop_parameters = {'precision': precision,
                                'n_states': n_states,
                                'n_obs': n_obs,
                                'n_par': n_par,
                                'n_drivers': n_drivers,
                                'dt_min': dt_min,
                                'dt_max': dt_max,
                                'dt_save': dt_save,
                                'dt_summarise': dt_summarise,
                                'atol': atol,
                                'rtol': rtol,
                                'save_time': save_time,
                                'n_saved_states': n_saved_states,
                                'n_saved_observables': n_saved_observables,
                                'summary_temp_memory': summary_temp_memory}

        self.functions = {'dxdt_func': dxdt_func,
                          'save_state_func': save_state_func,
                          'update_summary_func': update_summary_func,
                          'save_summary_func': save_summary_func,
                          }
        self.algo_shared_memory_items = self._calculate_loop_internal_shared_memory()
        self.needs_rebuild = True  # Set to False when the loop is built, so that an out-of-date loop isn't used

    def build(self):

        precision = self.loop_parameters['precision']
        dxdt_func = self.functions['dxdt_func']
        n_states = self.loop_parameters['n_states']
        n_obs = self.loop_parameters['n_obs']
        n_par = self.loop_parameters['n_par']
        n_drivers = self.loop_parameters['n_drivers']
        dt_min = self.loop_parameters['dt_min']
        dt_max = self.loop_parameters['dt_max']
        dt_save = self.loop_parameters['dt_save']
        atol = self.loop_parameters['atol']
        rtol = self.loop_parameters['rtol']
        dt_summarise = self.loop_parameters['dt_summarise']
        save_state_func = self.functions['save_state_func']
        save_time = self.loop_parameters['save_time']
        update_summary_func = self.functions['update_summary_func']
        save_summary_func = self.functions['save_summary_func']
        n_saved_states = self.loop_parameters['n_saved_states']
        n_saved_observables = self.loop_parameters['n_saved_observables']
        summary_temp_memory = self.loop_parameters['summary_temp_memory']

        self.loop_function = self.build_loop(precision,
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
        # Manipulate user-space time parameters into loop internal parameters here if you need to
        summarise_every_samples = int(dt_summarise / dt_save)

        temp_state_summary_shape = (n_saved_states * summary_temp_memory,)
        temp_observables_summary_shape = (n_saved_observables * summary_temp_memory,)

        if summary_temp_memory == 0:
            summaries_output = False
        else:
            summaries_output = True

        if n_saved_observables == 0:
            save_observables = False
        else:
            save_observables = True

        if n_saved_states == 0:
            save_states = False
        else:
            save_states = True

        if save_time:
            state_array_size = n_states + 1  # +1 for time
        else:
            state_array_size = n_states

        #Common template for all integrator loops - difficult to modify without changing all.
        # noinspection PyTypeChecker
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
        def dummy_loop(inits,
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
            Just a placeholder/template for an algorithm's loop function. This one just fills the output with the
            initial values.
            """
            l_state = cuda.local.array(shape=state_array_size, dtype=precision)
            l_obs = cuda.local.array(shape=n_obs, dtype=precision)
            l_obs[:] = precision(0.0)

            #HACK: if we have no saved states or observables, numba will error for creating empty local arrays.
            # Not an issue for shared memory, but sometimes we might not have enough shared memory for these.
            # This creation of an unused size-1 array feels like a hack, but avoids the error. Better to not create
            # the arrays at all, but we eould have to modify the summary functions to handle different combinations
            # of arrays and Nonetypes or similar.
            if summaries_output and save_states:
                l_temp_summaries = cuda.local.array(shape=temp_state_summary_shape, dtype=precision)
            else:
                l_temp_summaries = cuda.local.array(shape=1, dtype=precision)  # Dummy array to avoid errors
            if summaries_output and save_observables:
                l_temp_observables = cuda.local.array(shape=temp_observables_summary_shape, dtype=precision)
            else:
                l_temp_observables = cuda.local.array(shape=1, dtype=precision)

            for i in range(output_length):
                for j in range(n_states):
                    l_state[j] = inits[j]
                for j in range(n_obs):
                    l_obs[j] = inits[j % n_obs]
                # bp()
                save_state_func(l_state, l_obs, state_output[:, i], observables_output[:, i], i)
                if summaries_output:
                    update_summary_func(l_state, l_obs, l_temp_summaries, l_temp_observables, i)

                    if (i+1) % summarise_every_samples == 0:
                        summary_sample = (i + 1) // summarise_every_samples - 1
                        save_summary_func(l_temp_summaries, l_temp_observables,
                                          state_summaries_output[:, summary_sample],
                                          observables_summaries_output[:, summary_sample],
                                          summarise_every_samples)

        return dummy_loop

    def _calculate_loop_internal_shared_memory(self):
        """
        Calculate the number of items in shared memory required for the loop.
        The euler loop, for example, requires two arrays of size n_states for the state and dxdt,
        one for the current observables, and one for the drivers.

        Dummy loop uses 0 shared memory.
        """
        # Euler example:
        # n_states = self.loop_parameters['n_states']
        # n_obs = self.loop_parameters['n_obs']
        # n_drivers = self.loop_parameters['n_drivers']
        #
        #
        # return n_states*2 + n_obs + n_drivers
        return 0

    def rebuild(self, **kwargs):
        """
        Rebuild the loop with any modified parameters given in kwargs.

        Checks internal dictionaries, and if parameters have changed, rebuilds the loop.
        """
        # Check if any of the kwargs keys are in loop_parameters

        update_dicts_from_kwargs([self.loop_parameters, self.functions], **kwargs)

        # Rebuild the loop function
        self.build()

    def _time_to_fixed_steps(self):
        """Fixed-step helper function: Convert the time-based compile_settings to sample-based compile_settings,
        which are used by fixed-step loop functions. Sanity-check values and warn the user if they don't work.

        Returns:
            save_every_samples (int): The number of internal loop steps between saves.
            summarise_every_samples (int): The number of output samples between summary metric calculations.
            step_size (float): The internal time step size used in the loop (dt_min, by default).

        Raises:
            ValueError: If the user tries to save more often than they step, or summarise more often than they save.
            UserWarning: If the output rate or summary rate aren't an integer divisor of the internal loop frequency,
                update these values to be the actual time interval caused by stepping an integer number of steps. Warn
                the user that results aren't what they asked for.
        """

        dt_min = self.loop_parameters['dt_min']
        dt_max = self.loop_parameters['dt_max']
        dt_save = self.loop_parameters['dt_save']
        dt_summarise = self.loop_parameters['dt_summarise']

        check_requested_timing_possible(dt_min, dt_max, dt_save, dt_summarise)

        # Update the actual save and summary intervals, which will differ from what was ordered if they are not
        # a multiple of the loop step size.
        save_every_samples, summarise_every_samples, actual_dt_save, actual_dt_summarise = convert_times_to_fixed_steps(
            dt_min, dt_save, dt_summarise)

        # Update parameters if they differ from requested values and warn the user
        if actual_dt_save != dt_save:
            self.loop_parameters['dt_save'] = actual_dt_save
            warn(
                f"dt_save was set to {actual_dt_save}s, because it is not a multiple of dt_min ({dt_min}s). "
                f"dt_save can only save a value after an integer number of steps in a fixed-step integrator",
                UserWarning)

        if actual_dt_summarise != dt_summarise:
            self.loop_parameters['dt_summarise'] = actual_dt_summarise
            warn(
                f"dt_summarise was set to {actual_dt_summarise}s, because it is not a multiple of dt_save ({actual_dt_save}s). "
                f"dt_summarise can only save a value after an integer number of steps in a fixed-step integrator",
                UserWarning)

        return save_every_samples, summarise_every_samples, dt_min


