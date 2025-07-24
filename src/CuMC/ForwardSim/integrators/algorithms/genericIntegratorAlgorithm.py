from warnings import warn
from numba import cuda, int32
from CuMC.CUDAFactory import CUDAFactory
from CuMC.ForwardSim.integrators.algorithms.IntegratorLoopSettings import IntegratorLoopSettings


class GenericIntegratorAlgorithm(CUDAFactory):
    """
    Base class for integrator algorithms with improved separation of concerns.

    This class now uses IntegratorConfig for data handling and validation,
    following the same pattern as OutputFunctions/OutputConfig.

    Every integrator algorithm subclass should override:
    - get_loop_internal_shared_memory() - calculate shared memory requirements
    - build_loop() - factory method that builds the CUDA device function
    """

    def __init__(self,
                 precision,
                 dxdt_func,
                 n_states,
                 n_observables,
                 n_parameters,
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
                 summary_buffer_size,
                 threads_per_loop=1,
                 ):
        super().__init__()
        compile_settings = IntegratorLoopSettings(
                precision=precision,
                n_states=n_states,
                n_observables=n_observables,
                n_parameters=n_parameters,
                n_drivers=n_drivers,
                dt_min=dt_min,
                dt_max=dt_max,
                dt_save=dt_save,
                dt_summarise=dt_summarise,
                atol=atol,
                rtol=rtol,
                save_time=save_time,
                n_saved_states=n_saved_states,
                n_saved_observables=n_saved_observables,
                summary_buffer_size=summary_buffer_size,
                dxdt_func=dxdt_func,
                save_state_func=save_state_func,
                update_summary_func=update_summary_func,
                save_summary_func=save_summary_func,
                )
        self.setup_compile_settings(compile_settings)
        self._threads_per_loop = threads_per_loop
        self.integrator_loop = None
        self.is_current = False


    def build(self):
        """Build the integrator loop, unpacking config for local scope."""
        config = self.compile_settings

        integrator_loop = self.build_loop(
            precision=config.precision,
            dxdt_func=config.dxdt_func,
            n_states=config.n_states,
            n_observables=config.n_observables,
            n_parameters=config.n_parameters,
            n_drivers=config.n_drivers,
            dt_min=config.dt_min,
            dt_max=config.dt_max,
            dt_save=config.dt_save,
            dt_summarise=config.dt_summarise,
            atol=config.atol,
            rtol=config.rtol,
            save_time=config.save_time,
            save_state_func=config.save_state_func,
            update_summary_func=config.update_summary_func,
            save_summary_func=config.save_summary_func,
            n_saved_states=config.n_saved_states,
            n_saved_observables=config.n_saved_observables,
            summary_buffer_size=config.summary_buffer_size,
        )

        return {
            'device_function': integrator_loop,
            'loop_shared_memory': self.get_loop_internal_shared_memory()
        }

    # Properties for external access
    @property
    def threads_per_loop(self):
        """Number of threads required by loop algorithm."""
        return self._threads_per_loop

    @property
    def dt_save(self):
        """Time interval between saves."""
        return self.compile_settings.dt_save

    @property
    def dt_summarise(self):
        """Time interval between summary calculations."""
        return self.compile_settings.dt_summarise

    @property
    def n_saved_states(self):
        """Number of saved states."""
        return self.compile_settings.n_saved_states

    @property
    def n_saved_observables(self):
        """Number of saved observables."""
        return self.compile_settings.n_saved_observables


    def build_loop(self,
            precision,
            dxdt_func,
            n_states,
            n_observables,
            n_parameters,
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
            summary_buffer_size,
        ):
        save_steps, summary_steps, step_size = self.compile_settings.fixed_steps
        # Manipulate user-space time parameters into loop internal parameters here if you need to for your algorithm


        #Array Sizes
        state_summary_buffer_size = (n_saved_states * summary_buffer_size,)
        observables_summary_buffer_size = (n_saved_observables * summary_buffer_size,)

        # Logical flags for what to actually save (independent of array sizes)
        # Present in summary_metrics
        summaries_output = summary_buffer_size > 0
        save_observables = n_saved_observables > 0
        save_states = n_saved_states > 0


        ##Array sizes
        if save_time:
            state_array_size = n_states + 1
        else:
            state_array_size = n_states

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
            """Dummy integrator loop implementation."""
            l_state_buffer = cuda.local.array(shape=state_array_size, dtype=precision)
            l_obs_buffer = cuda.local.array(shape=n_observables, dtype=precision)
            l_obs_buffer[:] = precision(0.0)

            for i in range(n_states):
                l_state_buffer[i] = inits[i]

            state_summary_buffer = cuda.local.array(shape=state_summary_buffer_size, dtype=precision)
            obs_summary_buffer = cuda.local.array(shape=observables_summary_buffer_size, dtype=precision)

            for i in range(output_length):
                for j in range(n_states):
                    l_state_buffer[j] = inits[j]
                for j in range(n_observables):
                    l_obs_buffer[j] = inits[j % n_observables]

                save_state_func(l_state_buffer, l_obs_buffer, state_output[:, i], observables_output[:, i], i)

                if summaries_output:
                    update_summary_func(l_state_buffer, l_obs_buffer, state_summary_buffer, obs_summary_buffer, i)

                    if (i + 1) % summary_steps == 0:
                        summary_sample = (i + 1) // summary_steps - 1
                        save_summary_func(state_summary_buffer, obs_summary_buffer,
                                          state_summaries_output[:, summary_sample],
                                          observables_summaries_output[:, summary_sample],
                                          summary_steps,
                                          )

        return dummy_loop

    def update(self, **kwargs):
        """Update configuration parameters."""
        self.update_compile_settings(**kwargs)

    def get_loop_internal_shared_memory(self):
        """Calculate shared memory requirements. Dummy implementation returns 0."""
        return 0



    @classmethod
    def from_single_integrator_run(cls,
                                system,
                                output_functions,
                                run_settings,
                                output_settings):
        """Adapter to instantiate a loop algorithm from grouped parameters and other objects in the integrator
        architecture: systems and output functions."""
        pass

