from warnings import warn
from numba import cuda, int32
from numba.parfors.parfor import dummy_return_in_loop_body


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
        self.algo_shared_memory_items = self.calculate_shared_memory()
        self.build()


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
            output_shape = state_output.shape

            for i in range(output_length):
                for j in range(output_shape[1]):
                    state_output[i, j] = inits[j]
                    observables_output[i, j] = inits[j]



        return dummy_loop

    def calculate_shared_memory(self):
        """
        Calculate the number of items in shared memory required for the loop.
        The euler loop, for example, requires two arrays of size n_states for the state and dxdt,
        one for the current observables, and one for the drivers.
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
        for key, value in kwargs.items():
            if key in self.loop_parameters:
                if self.loop_parameters[key] != value:
                    self.loop_parameters[key] = value
            elif key in self.functions:
                if self.functions[key] != value:
                    self.functions[key] = value
            else:
                warn(f"Keyerror: the parameter {key} was not found in the ODE algorithms dictionary"
                     "of parameters")

        # Rebuild the loop function
        self.build()

