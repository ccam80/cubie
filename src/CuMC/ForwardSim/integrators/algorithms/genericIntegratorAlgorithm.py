from warnings import warn
from numba import cuda, int32
from CuMC.ForwardSim.integrators._utils import check_requested_timing_possible, convert_times_to_fixed_steps
from CuMC.CUDAFactory import CUDAFactory


class GenericIntegratorAlgorithm(CUDAFactory):
    """
    Base class for integrator algorithms.
    This class is not intended to be instantiated directly except for by some basic tests, but rather to be
    subclassed by specific integrator algorithms.

    Every integrator algorithm subclass should override the following methods:
    - _calculate_loop_internal_shared_memory() - a method that calculates the number of items in shared memory
    required to run the loop internals - when designing the loop function, the developer may choose where to place the
    various working arrays. Shared memory is faster but is limited in size, so it's a trade-off.
    - build_loop() - a factory method that builds a CUDA device function that implements the loop. This function may
    call any other functions, and the subclass may define additional helper methods. There is allowance to pass extra
    parameters using kwargs - check kwargs for your specific parameters.

    And the following attributes:
    - loop_function - the compiled CUDA device function implementing the start-to-finish integration loop.
    This function must have the signature:
        loop(inits: 1d CUDA.deviceArray,
             parameters: 1d CUDA.deviceArray,
             forcing_vec: nd CUDA.deviceArray, where 'n' is the system's number of drivers,
             shared_memory: A pointer to the thread's slice of dynamic shared memory
             state_output: 2d CUDA.deviceArray, where the first dimension is the number of saved states and the
              second is the number of output samples,
             observables_output: 2d CUDA.deviceArray, where the first dimension is the number of saved
              observables and the second is the number of output samples
             state_summaries_output: 2d CUDA.deviceArray, where the first dimension is the number of saved
              state summaries and the second is the number of summary samples,
             observables_summaries_output: 2d CUDA.deviceArray, where the first dimension is the number of saved
              observable summaries and the second is the number of summary samples,
             output_length: int,
             warmup_samples:int = 0)

    The build_loop() has the system's dxdt function, the save_state function, the update_summary function, and the
    save_summary function as arguments. These remain in the local scope of the loop function, and so are compiled
    into the loop. The loop should call the save_state and update_summary function on each "save" step,
    and the save_summary function on each "summarise" step.
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
                 summary_temp_memory,
                 threads_per_loop=1,
                 ):
        super().__init__()
        compile_settings = {'precision':           precision,
                            'n_states':            n_states,
                            'n_obs':               n_obs,
                            'n_par':               n_par,
                            'n_drivers':           n_drivers,
                            'dt_min':              dt_min,
                            'dt_max':              dt_max,
                            'dt_save':             dt_save,
                            'dt_summarise':        dt_summarise,
                            'atol':                atol,
                            'rtol':                rtol,
                            'save_time':           save_time,
                            'n_saved_states':      n_saved_states,
                            'n_saved_observables': n_saved_observables,
                            'summary_temp_memory': summary_temp_memory,
                            'dxdt_func':           dxdt_func,
                            'save_state_func':     save_state_func,
                            'update_summary_func': update_summary_func,
                            'save_summary_func':   save_summary_func,
                            }
        self.setup_compile_settings(compile_settings)
        self._threads_per_loop = threads_per_loop
        self.integrator_loop = None
        self.is_current = False

    def build(self):
        """This wraps an algorithm's build_loop() method, unpacking the parameters and functions and passing them as
        arguments so that they are all in the local scope of the loop function."""

        precision = self.compile_settings['precision']
        dxdt_func = self.compile_settings['dxdt_func']
        n_states = self.compile_settings['n_states']
        n_obs = self.compile_settings['n_obs']
        n_par = self.compile_settings['n_par']
        n_drivers = self.compile_settings['n_drivers']
        dt_min = self.compile_settings['dt_min']
        dt_max = self.compile_settings['dt_max']
        dt_save = self.compile_settings['dt_save']
        atol = self.compile_settings['atol']
        rtol = self.compile_settings['rtol']
        dt_summarise = self.compile_settings['dt_summarise']
        save_state_func = self.compile_settings['save_state_func']
        save_time = self.compile_settings['save_time']
        update_summary_func = self.compile_settings['update_summary_func']
        save_summary_func = self.compile_settings['save_summary_func']
        n_saved_states = self.compile_settings['n_saved_states']
        n_saved_observables = self.compile_settings['n_saved_observables']
        summary_temp_memory = self.compile_settings['summary_temp_memory']

        integrator_loop = self.build_loop(precision,
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
                                          )

        return {'device_function':    integrator_loop,
                'loop_shared_memory': self.get_loop_internal_shared_memory()
                }

    @property
    def threads_per_loop(self):
        """The number of threads to use per loop iteration. Multi-thread algorithms will require different memory
        allocations."""
        return self._threads_per_loop

    @property
    def dt_save(self):
        """The time interval between saves, in seconds."""
        return self.compile_settings['dt_save']

    @property
    def dt_summarise(self):
        """The time interval between summary calculations, in seconds."""
        return self.compile_settings['dt_summarise']

    @property
    def n_saved_states(self):
        """The number of saved states."""
        return self.compile_settings['n_saved_states']

    @property
    def n_saved_observables(self):
        """The number of saved observables."""
        return self.compile_settings['n_saved_observables']


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
                   summary_temp_memory,
                   ):
        # Manipulate user-space time parameters into loop internal parameters here if you need to for your algorithm
        summarise_every_samples = int(dt_summarise / dt_save)
        #TODO: AI double-handling bollocks, all of this stuff appears to be done in the outputconfig module.
        # CUDA-safe array dimensions (minimum size 1 to prevent zero-sized arrays)
        cuda_safe_n_saved_states = max(1, n_saved_states)
        cuda_safe_n_saved_observables = max(1, n_saved_observables)
        cuda_safe_summary_temp_memory = max(1, summary_temp_memory)

        temp_state_summary_shape = (cuda_safe_n_saved_states * cuda_safe_summary_temp_memory,)
        temp_observables_summary_shape = (cuda_safe_n_saved_observables * cuda_safe_summary_temp_memory,)

        # Logical flags for what to actually save (independent of array sizes)
        summaries_output = summary_temp_memory > 0
        save_observables = n_saved_observables > 0
        save_states = n_saved_states > 0

        if save_time:
            state_array_size = n_states + 1
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
                  inline=True,
                  )
        def dummy_loop(inits,
                       parameters,
                       forcing_vec,
                       shared_memory,
                       state_output,
                       observables_output,
                       state_summaries_output,
                       observables_summaries_output,
                       output_length,
                       warmup_samples=0,
                       ):
            """
            Dummy integrator loop that doesn't integrate - it just fills the stae array and observables array with
            the initial values, repeatedly.
            """
            l_state = cuda.local.array(shape=state_array_size, dtype=precision)
            l_obs = cuda.local.array(shape=n_obs, dtype=precision)
            l_obs[:] = precision(0.0)

            for i in range(n_states):
                l_state[i] = inits[i]

            l_temp_summaries = cuda.local.array(shape=temp_state_summary_shape, dtype=precision)
            l_temp_observables = cuda.local.array(shape=temp_observables_summary_shape, dtype=precision)

            for i in range(output_length):
                for j in range(n_states):
                    l_state[j] = inits[j]
                for j in range(n_obs):
                    l_obs[j] = inits[j % n_obs]
                # bp()
                save_state_func(l_state, l_obs, state_output[:, i], observables_output[:, i], i)

                if summaries_output:
                    update_summary_func(l_state, l_obs, l_temp_summaries, l_temp_observables, i)

                    if (i + 1) % summarise_every_samples == 0:
                        summary_sample = (i + 1) // summarise_every_samples - 1
                        save_summary_func(l_temp_summaries, l_temp_observables,
                                          state_summaries_output[:, summary_sample],
                                          observables_summaries_output[:, summary_sample],
                                          summarise_every_samples,
                                          )

        return dummy_loop

    def update(self, **kwargs):
        """
        Update the loop parameters with the given kwargs.
        This will make any previously built loop invalid/not current.
        """
        self.update_compile_settings(**kwargs)

    def get_loop_internal_shared_memory(self):
        """
        Calculate the number of items in shared memory required for the loop.
        The euler loop, for example, requires two arrays of size n_states for the state and dxdt,
        one for the current observables, and one for the drivers.

        Dummy loop uses 0 shared memory.
        """
        return 0

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

        dt_min = self.compile_settings['dt_min']
        dt_max = self.compile_settings['dt_max']
        dt_save = self.compile_settings['dt_save']
        dt_summarise = self.compile_settings['dt_summarise']

        check_requested_timing_possible(dt_min, dt_max, dt_save, dt_summarise)

        # Update the actual save and summary intervals, which will differ from what was ordered if they are not
        # a multiple of the loop step size.
        save_every_samples, summarise_every_samples, actual_dt_save, actual_dt_summarise =  convert_times_to_fixed_steps(
                dt_min, dt_save, dt_summarise,
                )

        # Update parameters if they differ from requested values and warn the user
        if actual_dt_save != dt_save:
            self.compile_settings['dt_save'] = actual_dt_save
            warn(
                    f"dt_save was set to {actual_dt_save}s, because it is not a multiple of dt_min ({dt_min}s). "
                    f"dt_save can only save a value after an integer number of steps in a fixed-step integrator",
                    UserWarning,
                    )

        if actual_dt_summarise != dt_summarise:
            self.compile_settings['dt_summarise'] = actual_dt_summarise
            warn(
                    f"dt_summarise was set to {actual_dt_summarise}s, because it is not a multiple of dt_save ({actual_dt_save}s). "
                    f"dt_summarise can only save a value after an integer number of steps in a fixed-step integrator",
                    UserWarning,
                    )

        return save_every_samples, summarise_every_samples, dt_min