# from warnings import warn
from numba import cuda, int32
# from numba.parfors.parfor import dummy_return_in_loop_body
from CuMC._utils import update_dicts_from_kwargs

# from pdb import set_trace as bp

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


        #Common template for all integrator loops - difficult to modify without changing all.
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
            l_state = cuda.local.array(shape=n_states, dtype=precision)
            l_obs = cuda.local.array(shape=n_obs, dtype=precision)
            l_obs[:] = precision(0.0)

            #HACK: if we have no saved states or observables, numba will error for creating empty local arrays.
            # Not an issue for shared memory, but sometimes we might not have enough shared memory for these.
            # This creation of an unused size-1 array feels like a hack, but avoids the error.
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

        Dummy returns the number of states as a placeholder.
        """
        # Euler example:
        # n_states = self.loop_parameters['n_states']
        # n_obs = self.loop_parameters['n_obs']
        # n_drivers = self.loop_parameters['n_drivers']
        #
        #
        # return n_states*2 + n_obs + n_drivers
        return self.loop_parameters['n_states']

    def rebuild(self, **kwargs):
        """
        Rebuild the loop with any modified parameters given in kwargs.

        Checks internal dictionaries, and if parameters have changed, rebuilds the loop.
        """
        # Check if any of the kwargs keys are in loop_parameters

        update_dicts_from_kwargs([self.loop_parameters, self.functions], **kwargs)

        # Rebuild the loop function
        self.build()

