"""
Base class for integration algorithm implementations.

This module provides the GenericIntegratorAlgorithm class, which serves as
the base class for all ODE integration algorithms. It handles building and
caching of algorithm functions that are incorporated into CUDA kernels,
and provides the interface that specific algorithms must implement.
"""

from numba import cuda, int32, from_dtype

from cubie.CUDAFactory import CUDAFactory
from cubie.integrators.algorithms.IntegratorLoopSettings import (
    IntegratorLoopSettings,
)
from cubie._utils import in_attr


class GenericIntegratorAlgorithm(CUDAFactory):
    """
    Base class for the inner "loop" algorithm for ODE solving algorithms.

    This class handles building and caching of the algorithm function, which
    is incorporated into a CUDA kernel for GPU execution. All integration
    algorithms (e.g. Euler, Runge-Kutta) should subclass this class and
    override specific attributes and methods.

    Parameters
    ----------
    precision : type
        Numerical precision type for computations.
    dxdt_function : CUDA device function
        Function that computes time derivatives of the state.
    buffer_sizes : LoopBufferSizes
        Configuration object specifying buffer sizes.
    loop_step_config : LoopStepConfig
        Configuration object for loop step parameters.
    save_state_func : CUDA device function
        Function for saving state values during integration.
    update_summaries_func : CUDA device function
        Function for updating summary statistics.
    save_summaries_func : CUDA device function
        Function for saving summary statistics.
    compile_flags : OutputCompileFlags
        Compilation flags for device function generation.

    Notes
    -----
    Subclasses must override:

    - `_threads_per_loop` : Number of threads the algorithm uses
    - `build_loop()` : Factory method that builds the CUDA device function
    - `shared_memory_required` : Amount of shared memory the device allocates

    Data used in compiling and controlling the loop is handled by the
    IntegratorLoopSettings class. This class presents relevant attributes
    of the data class to higher-level components as properties.

    See Also
    --------
    IntegratorLoopSettings : Configuration data for loop compilation
    CUDAFactory : Base factory class for CUDA device functions
    """

    def __init__(
        self,
        precision,
        dxdt_function,
        buffer_sizes,
        loop_step_config,
        save_state_func,
        update_summaries_func,
        save_summaries_func,
        compile_flags,
    ):
        super().__init__()

        compile_settings = IntegratorLoopSettings(
            precision=precision,
            loop_step_config=loop_step_config,
            buffer_sizes=buffer_sizes,
            dxdt_function=dxdt_function,
            save_state_func=save_state_func,
            update_summaries_func=update_summaries_func,
            save_summaries_func=save_summaries_func,
            compile_flags=compile_flags,
        )
        self.setup_compile_settings(compile_settings)

        self.integrator_loop = None

        # Override this in subclasses!
        self._threads_per_loop = 1

    def build(self):
        """
        Build the integrator loop, unpacking config for local scope.

        Returns
        -------
        callable
            The compiled integrator loop device function.
        """
        config = self.compile_settings

        integrator_loop = self.build_loop(
            precision=config.precision,
            dxdt_function=config.dxdt_function,
            save_state_func=config.save_state_func,
            update_summaries_func=config.update_summaries_func,
            save_summaries_func=config.save_summaries_func,
        )

        return integrator_loop

    @property
    def threads_per_loop(self):
        """
        Get the number of threads required by loop algorithm.

        Returns
        -------
        int
            Number of threads required per integration loop.
        """
        return self._threads_per_loop

    def build_loop(
        self,
        precision,
        dxdt_function,
        save_state_func,
        update_summaries_func,
        save_summaries_func,
    ):
        """
        Build the CUDA device function for the integration loop.

        This is a template method that provides a dummy implementation.
        Subclasses should override this method to implement specific
        integration algorithms.

        Parameters
        ----------
        precision : type
            Numerical precision type for the integration.
        dxdt_function : callable
            Function that computes time derivatives.
        save_state_func : callable
            Function for saving state values.
        update_summaries_func : callable
            Function for updating summary statistics.
        save_summaries_func : callable
            Function for saving summary statistics.

        Returns
        -------
        callable
            Compiled CUDA device function implementing the integration algorithm.

        Notes
        -----
        This base implementation provides a dummy loop that can be used
        for testing but does not perform actual integration. Real algorithms
        should override this method with their specific implementation.
        """
        save_steps, summary_steps, step_size = (
            self.compile_settings.fixed_steps
        )

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

        # no cover: start
        @cuda.jit(
            (
                numba_precision[:],
                numba_precision[:],
                numba_precision[:, :],
                numba_precision[:],
                numba_precision[:, :],
                numba_precision[:, :],
                numba_precision[:, :],
                numba_precision[:, :],
                int32,
                int32,
            ),
            device=True,
            inline=True,
        )
        def dummy_loop(
            inits,
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
            l_state_buffer = cuda.local.array(
                shape=state_buffer_size, dtype=numba_precision
            )
            l_obs_buffer = cuda.local.array(
                shape=observables_buffer_size, dtype=numba_precision
            )
            l_obs_buffer[:] = numba_precision(0.0)

            for i in range(loop_states):
                l_state_buffer[i] = inits[i]

            state_summary_buffer = cuda.local.array(
                shape=state_summary_buffer_size, dtype=numba_precision
            )
            obs_summary_buffer = cuda.local.array(
                shape=observables_summary_buffer_size, dtype=numba_precision
            )

            for i in range(output_length):
                for j in range(loop_states):
                    l_state_buffer[j] = inits[j]
                for j in range(loop_obs):
                    l_obs_buffer[j] = inits[j % observables_buffer_size]

                save_state_func(
                    l_state_buffer,
                    l_obs_buffer,
                    state_output[i, :],
                    observables_output[i, :],
                    i,
                )

                # if summaries_output:
                update_summaries_func(
                    l_state_buffer,
                    l_obs_buffer,
                    state_summary_buffer,
                    obs_summary_buffer,
                    i,
                )

                if (i + 1) % summary_steps == 0:
                    summary_sample = (i + 1) // summary_steps - 1
                    save_summaries_func(
                        state_summary_buffer,
                        obs_summary_buffer,
                        state_summaries_output[summary_sample, :],
                        observables_summaries_output[summary_sample, :],
                        summary_steps,
                    )

        return dummy_loop
        # no cover: stop

    def update(self, updates_dict=None, silent=False, **kwargs):
        """
        Pass updates to compile settings through the CUDAFactory interface.

        This method will invalidate the cache if an update is successful.
        Use silent=True when doing bulk updates with other component parameters
        to suppress warnings about unrecognized keys.

        Parameters
        ----------
        updates_dict : dict, optional
            Dictionary of parameters to update.
        silent : bool, default=False
            If True, suppress warnings about unrecognized parameters.
        **kwargs
            Parameter updates to apply as keyword arguments.

        Returns
        -------
        set
            Set of parameter names that were recognized and updated.
        """
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
                "These parameters were not updated.",
            )
        return recognised

    @property
    def shared_memory_required(self):
        """
        Calculate shared memory requirements for the integration algorithm.

        This is a dummy implementation that returns 0. Subclasses should
        override this method to calculate the actual shared memory requirements
        based on their specific algorithm needs.

        Returns
        -------
        int
            Number of shared memory elements required (dummy implementation returns 0).
        """
        return 0

    @classmethod
    def from_single_integrator_run(cls, run_object):
        """
        Create an instance of the integrator algorithm from a SingleIntegratorRun object.

        Parameters
        ----------
        run_object : SingleIntegratorRun
            The SingleIntegratorRun object containing configuration parameters.

        Returns
        -------
        GenericIntegratorAlgorithm
            New instance of the integrator algorithm configured with parameters
            from the run object.
        """
        return cls(
            precision=run_object.precision,
            dxdt_function=run_object.dxdt_function,
            buffer_sizes=run_object.loop_buffer_sizes,
            loop_step_config=run_object.loop_step_config,
            save_state_func=run_object.save_state_func,
            update_summaries_func=run_object.update_summaries_func,
            save_summaries_func=run_object.save_summaries_func,
            compile_flags=run_object.compile_flags,
        )

    @property
    def fixed_step_size(self):
        """
        Get the fixed step size used in the integration loop.

        Returns
        -------
        float
            The fixed step size from the compile settings.
        """
        return self.compile_settings.fixed_step_size
