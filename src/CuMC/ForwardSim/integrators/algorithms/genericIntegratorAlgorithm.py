from warnings import warn
from numba import cuda, int32
from CuMC.CUDAFactory import CUDAFactory
from CuMC.ForwardSim.integrators.algorithms.IntegratorLoopSettings import IntegratorLoopSettings


class GenericIntegratorAlgorithm(CUDAFactory):
    """
    Base class for the inner "loop" algorithm for an ODE solving algorithm. This class handles building and caching
    of the algorithm function, which is incorporated into a CUDA kernel (like the one in SolverKernel.py) for use.
    Any integration algorithms (e.g. Euler, Runge-Kutta) should subclass this class and override the following
    attributes/methods:

    - _threads_per_loop: How many threads does the algorithm use? A simple loop will use 1, but a computationally
    intensive algorithm might calculate dxdt at each point in its own thread, and then use a shuffle operation to add
    the parallel results together from shared memory.
    - build_loop() - factory method that builds the CUDA device function
    - _loop_shared_memory - How much shared memory your device allocates - usually a function of the number of states
     of our system, depending on where you store your numbers.

    Data used in compiling and controlling the loop is handled by the IntegratorLoopSettings class. This class
    presents a few relevant attributes of the data class to higher-level components as properties.

    """

    def __init__(self,
                 precision,
                 dxdt_func,
                 system_sizes,
                 loop_step_config,
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
                system_sizes=system_sizes,
                loop_step_config=loop_step_config,
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

        #Override this in subclasses!
        self._threads_per_loop = threads_per_loop

        self.integrator_loop = None
        self.is_current = False


    def build(self):
        """Build the integrator loop, unpacking config for local scope."""
        config = self.compile_settings

        integrator_loop = self.build_loop(
            precision=config.precision,
            dxdt_func=config.dxdt_func,
            buffer_sizes = config.buffer_sizes,
            loop_step_config = config.loop_step_config,
            save_state_func=config.save_state_func,
            update_summary_func=config.update_summary_func,
            save_summary_func=config.save_summary_func,
        )

        return {
            'device_function': integrator_loop,
            'loop_shared_memory': self.shared_memory_required
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
            buffer_sizes,
            save_state_func,
            update_summary_func,
            save_summary_func,
        ):
        save_steps, summary_steps, step_size = self.compile_settings.fixed_steps
        # Manipulate user-space time parameters into loop internal parameters here if you need to for your algorithm


        #Unpack sizes to keep compiler happy
        state_summary_buffer_size = buffer_sizes.state_summaries
        observables_summary_buffer_size = buffer_sizes.observables_summaries
        state_buffer_size = buffer_sizes.state
        observables_buffer_size = buffer_sizes.observables
        dxdt_buffer_size = buffer_sizes.dxdt
        parameters_buffer_size = buffer_sizes.parameters
        drivers_buffer_size = buffer_sizes.drivers

        # summaries_output = summary_buffer_size > 0 # Without this toggle, the compiler will try to compile in empty
        # functions. If it succeeds, that's a boon, it's just optimised out the calls. If it fails, we need to
        # propagate this flag across from output_functions

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
            l_state_buffer = cuda.local.array(shape=state_buffer_size, dtype=precision)
            l_obs_buffer = cuda.local.array(shape=observables_buffer_size, dtype=precision)
            l_obs_buffer[:] = precision(0.0)

            for i in range(state_buffer_size):
                l_state_buffer[i] = inits[i]

            state_summary_buffer = cuda.local.array(shape=state_summary_buffer_size, dtype=precision)
            obs_summary_buffer = cuda.local.array(shape=observables_summary_buffer_size, dtype=precision)

            for i in range(output_length):
                for j in range(state_buffer_size):
                    l_state_buffer[j] = inits[j]
                for j in range(observables_buffer_size):
                    l_obs_buffer[j] = inits[j % observables_buffer_size]

                save_state_func(l_state_buffer, l_obs_buffer, state_output[:, i], observables_output[:, i], i)

                # if summaries_output:
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

    @property
    def shared_memory_required(self):
        return self.get_cached_output("loop_shared_memory")

    def _loop_internal_shared_memory(self):
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

