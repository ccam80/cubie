from numba import cuda, int32, from_dtype

from cubie.CUDAFactory import CUDAFactory
from cubie.integrators.algorithms.IntegratorLoopSettings import \
    IntegratorLoopSettings
from cubie._utils import in_attr


class GenericIntegratorAlgorithm(CUDAFactory):
    """
    Base class for the inner "loop" algorithm for an ODE solving algorithm.
    This class handles building and caching
    of the algorithm function, which is incorporated into a CUDA kernel (like the one in BatchSolverKernel.py) for use.
    Any integration algorithms (e.g. Euler, Runge-Kutta) should subclass this class and override the following
    attributes/methods:

    - _threads_per_loop: How many threads does the algorithm use? A simple loop will use 1, but a computationally
    intensive algorithm might calculate dxdt at each point in its own thread, and then use a shuffle operation to add
    the parallel results together from shared memory.
    - build_loop() - factory method that builds the CUDA device function
    - shared_memory_required - How much shared memory your device allocates - usually a function of the number of states
     of our system, depending on where you store your numbers.

    Data used in compiling and controlling the loop is handled by the IntegratorLoopSettings class. This class
    presents a few relevant attributes of the data class to higher-level components as properties.

    """

    def __init__(self, precision, dxdt_function, buffer_sizes,
                 loop_step_config, save_state_func, update_summaries_func,
                 save_summaries_func, compile_flags, ):
        super().__init__()

        compile_settings = IntegratorLoopSettings(precision=precision,
                loop_step_config=loop_step_config, buffer_sizes=buffer_sizes,
                dxdt_function=dxdt_function, save_state_func=save_state_func,
                update_summaries_func=update_summaries_func,
                save_summaries_func=save_summaries_func,
                compile_flags=compile_flags, )
        self.setup_compile_settings(compile_settings)

        self.integrator_loop = None

        # Override this in subclasses!
        self._threads_per_loop = 1

    def build(self):
        """Build the integrator loop, unpacking config for local scope."""
        config = self.compile_settings

        integrator_loop = self.build_loop(precision=config.precision,
                                          dxdt_function=config.dxdt_function,
                                          save_state_func=config.save_state_func,
                                          update_summaries_func=config.update_summaries_func,
                                          save_summaries_func=config.save_summaries_func, )

        return integrator_loop

    @property
    def threads_per_loop(self):
        """Number of threads required by loop algorithm."""
        return self._threads_per_loop

    def build_loop(self, precision, dxdt_function, save_state_func,
                   update_summaries_func, save_summaries_func, ):
        save_steps, summary_steps, step_size = self.compile_settings.fixed_steps

        sizes = self.compile_settings.buffer_sizes.nonzero

        # Unpack sizes to keep compiler happy
        state_summary_buffer_size = sizes.state_summaries
        observables_summary_buffer_size = sizes.observable_summaries
        state_buffer_size = sizes.state
        observables_buffer_size = sizes.observables

        loop_sizes = self.compile_settings.buffer_sizes
        loop_states = loop_sizes.state
        loop_obs = loop_sizes.observables

        numba_precision = from_dtype(precision)

        @cuda.jit(
                (numba_precision[:], numba_precision[:], numba_precision[:, :],
                 numba_precision[:], numba_precision[:, :],
                 numba_precision[:, :], numba_precision[:, :],
                 numba_precision[:, :], int32, int32,), device=True,
                inline=True, )
        def dummy_loop(inits, parameters, forcing_vec, shared_memory,
                       state_output, observables_output,
                       state_summaries_output, observables_summaries_output,
                       output_length, warmup_samples=0, ):
            """Dummy integrator loop implementation."""
            l_state_buffer = cuda.local.array(shape=state_buffer_size,
                                              dtype=numba_precision)
            l_obs_buffer = cuda.local.array(shape=observables_buffer_size,
                                            dtype=numba_precision)
            l_obs_buffer[:] = numba_precision(0.0)

            for i in range(loop_states):
                l_state_buffer[i] = inits[i]

            state_summary_buffer = cuda.local.array(
                    shape=state_summary_buffer_size, dtype=numba_precision)
            obs_summary_buffer = cuda.local.array(
                    shape=observables_summary_buffer_size,
                    dtype=numba_precision)

            for i in range(output_length):
                for j in range(loop_states):
                    l_state_buffer[j] = inits[j]
                for j in range(loop_obs):
                    l_obs_buffer[j] = inits[j % observables_buffer_size]

                save_state_func(l_state_buffer, l_obs_buffer,
                                state_output[i, :], observables_output[i, :],
                                i)

                # if summaries_output:
                update_summaries_func(l_state_buffer, l_obs_buffer,
                                      state_summary_buffer, obs_summary_buffer,
                                      i)

                if (i + 1) % summary_steps == 0:
                    summary_sample = (i + 1) // summary_steps - 1
                    save_summaries_func(state_summary_buffer,
                                        obs_summary_buffer,
                                        state_summaries_output[summary_sample,
                                        :], observables_summaries_output[
                                            summary_sample, :],
                                        summary_steps, )

        return dummy_loop

    def update(self, updates_dict=None, silent=False, **kwargs):
        """
        Pass updates to compile settings through the CUDAFactory interface, which will invalidate cache if an update
        is successful. Pass silent=True if doing a bulk update with other component's params to suppress warnings
        about keys not found.

        Args:
            silent (bool): If True, suppress warnings about unrecognized parameters
            **kwargs: Parameter updates to apply

        Returns:
            list: recognized_params"""
        if updates_dict is None:
            updates_dict = {}
        if kwargs:
            updates_dict.update(kwargs)
        if updates_dict == {}:
            return set()

        recognised = self.update_compile_settings(updates_dict, silent=True)
        for key, value in updates_dict.items():
            if in_attr(key, self.compile_settings.loop_step_config):
                setattr(self.compile_settings, key, value)
                recognised.add(key)

        unrecognised = set(updates_dict.keys()) - recognised
        if not silent and unrecognised:
            raise KeyError(
                    f"Unrecognized parameters in update: {unrecognised}. "
                    "These parameters were not updated.", )
        return recognised

    @property
    def shared_memory_required(self):
        """Calculate shared memory requirements. Dummy implementation returns 0."""
        return 0

    @classmethod
    def from_single_integrator_run(cls, run_object):
        """Create an instance of the integrator algorithm from a SingleIntegratorRun object."""
        return cls(precision=run_object.precision,
                dxdt_function=run_object.dxdt_function,
                buffer_sizes=run_object.loop_buffer_sizes,
                loop_step_config=run_object.loop_step_config,
                save_state_func=run_object.save_state_func,
                update_summaries_func=run_object.update_summaries_func,
                save_summaries_func=run_object.save_summaries_func,
                compile_flags=run_object.compile_flags, )

    @property
    def fixed_step_size(self):
        """Return the fixed step size used in the loop."""
        return self.compile_settings.fixed_step_size
