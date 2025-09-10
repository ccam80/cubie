"""
Base class for integration algorithm Loops.

This module provides the IVPLoop class, which serves as
the base class for all ODE integration loops. This class provides default
update behaviour and properties for a unified interface and inherits build
and cache logic from CUDAFactory.

Integration loops handle the "outer" logic of an ODE integration, organising
steps and saving output, and call an algorithm-specific step function to do the
mathy end of the integration.
"""
from typing import Optional, Callable

from numba import cuda, from_dtype, int32

from cubie.CUDAFactory import CUDAFactory
from cubie._utils import in_attr
from cubie.outputhandling import OutputCompileFlags, LoopBufferSizes
from math import ceil

class IVPLoop(CUDAFactory):
    """
    Stepping loop for ODE solving algorithms.

    This class handles building and caching of the loop device function, which
    is incorporated into a CUDA kernel for GPU execution. The "meat" of the
    integration is completed in a step function, which contains the
    integration algorithm. This loop just assigns some memory, calls the
    step, calls a step-size controller to accept/reject the step, and calls
    output/save functions when appropriate. No subclasses are expected.

    Parameters
    ----------
    precision : type
        Numerical precision type for computations.
    buffer_sizes : LoopBufferSizes
        Configuration object specifying buffer sizes.
    compile_flags : LoopStepConfig
        Configuration object for loop step parameters.
    save_state_func : CUDA device function
        Function for saving state values during integration.
    update_summaries_func : CUDA device function
        Function for updating summary statistics.
    save_summaries_func : CUDA device function
        Function for saving summary statistics.
    step_controller_fn : device function
        A device function that adjusts the timestep and accepts or rejects a
        step. Signature [TODO]
    step_fn : device function
        A device function that executes a single step of the algorithm.
        Signature: [TODO]

    Notes
    -----

    """

    def __init__(
        self,
        precision: type,
        buffer_sizes: LoopBufferSizes,
        compile_flags: OutputCompileFlags,
        save_state_func: Callable,
        update_summaries_func: Callable,
        save_summaries_func: Callable,
        step_controller_fn: Callable,
        step_fn: Callable,
    ):
        super().__init__()
        self.precision = precision
        # consider breaking into compile_settings, as these are
        # compile-critical
        self.buffer_sizes = buffer_sizes
        self.compile_flags = compile_flags
        self.save_state_func = save_state_func
        self.update_summaries_func = update_summaries_func
        self.save_summaries_func = save_summaries_func
        self.step_controller_fn = step_controller_fn
        self.step_fn = step_fn
        self.loop_fn = None

    def build(self):
        """
        Build the integrator loop, unpacking config for local scope.

        Returns
        -------
        callable
            The compiled integrator loop device function.
        """
        config = self.compile_settings

        save_state = config.save_state_func
        update_summaries = config.update_summaries_func
        save_summaries = config.save_summaries_func
        step_controller = config.step_controller_fn
        step_fn = config.step_fn

        flags = config.compile_flags
        save_obs_bool = flags.save_observables
        save_state_bool = flags.save_state
        summarise_obs_bool = flags.summarise_observables
        summarise_state_bool = flags.summarise_state
        summarise = summarise_obs_bool or summarise_state_bool
        fixed_mode = config.is_fixed_step

        shared_indices = config.shared_memory_indices
        state_shared_ind = shared_indices.state
        obs_shared_ind = shared_indices.observables
        state_prop_shared_ind = shared_indices.state_proposal
        state_summ_shared_ind = shared_indices.state_summaries
        params_shared_ind = shared_indices.parameters
        obs_summ_shared_ind = shared_indices.observable_summaries
        drivers_shared_ind = shared_indices.drivers

        saves_per_summary = config.saves_per_summary
        total_saved_samples = config.total_saved_samples

        precision = from_dtype(self.precision)

        n_states = config.buffer_sizes.state
        n_parameters = config.buffer_sizes.nonzero.parameters

        dt0_default = config.dt0
        dt_save = config.dt_save
        dt_min = config.dt_min


        @cuda.jit(device=True, inline=True)
        def loop_fn(
            initial_states,
            parameters,
            drivers,
            shared_scratch,
            local_scratch,
            state_output,
            observables_output,
            state_summaries_output,
            observable_summaries_output,
            duration,
            settling_time,
            t0 = precision(0.0),
            dt0 = dt0_default,
        ):
            # Cap max iterations - all internal steps plus a bonus end/start
            max_steps =  (int32(
                          ceil(duration / max(dt_min, precision(1e-16))))
                          + int32(2))

            shared_scratch[shared_indices.all] = precision(0.0)

            state_buffer = shared_scratch[state_shared_ind]
            state_proposal_buffer = shared_scratch[state_prop_shared_ind]
            observables_buffer = shared_scratch[obs_shared_ind]
            parameters_buffer = shared_scratch[params_shared_ind]
            drivers_buffer = shared_scratch[drivers_shared_ind]
            state_summary_buffer = shared_scratch[state_summ_shared_ind]
            observable_summary_buffer = shared_scratch[obs_summ_shared_ind]

            for k in range(n_states):
                state_buffer[k] = initial_states[k]
            for k in range(n_parameters):
                parameters_buffer[k] = parameters[k]
            driver_length = drivers.shape[0]

            # Loop state
            t = t0
            dt = dt0
            next_save = precision(settling_time)
            error_integral = precision(0.0)
            status = int32(0)
            save_idx = int32(0)
            summary_idx = int32(0)

            # Get active threads to allow a warp-coherent exit without
            # waiting on inactive threads
            mask = cuda.activemask()

            for _ in range(max_steps):
                finished = t >= duration
                if cuda.all_sync(mask, finished):
                    return status

                if not finished:
                    # Schedule
                    if fixed_mode:
                        # Hit boundary when within half a step of the target time
                        do_save = abs(t - next_save) < (dt * precision(0.5))
                        dt_eff = dt
                    else:
                        do_save = (t + dt) >= next_save
                        dt_eff = cuda.selp(do_save, next_save - t, dt)

                    # Do a step with clamped step size
                    status |= step_fn(state_buffer,
                                      state_proposal_buffer,
                                      parameters_buffer,
                                      drivers_buffer,
                                      observables_buffer,
                                      shared_scratch, #dxdt, stage_buffers, etc
                                      local_scratch,
                                      dt_eff)

                    # Adjust dt if step rejected - auto-accepts if fixed-step
                    if fixed_mode:
                        accept, dt, error_integral, retcode = step_controller(
                            dt,
                            state_buffer,
                            state_proposal_buffer,
                            error_integral
                        )
                        status |= retcode
                    else:
                        accept = True
                        dt = dt

                    t_proposal = t + dt_eff
                    t = cuda.selp(accept, t_proposal, t)

                    if not fixed_mode:
                        for i in range(n_states):
                            newv = state_proposal_buffer[i]
                            oldv = state_buffer[i]
                            state_buffer = cuda.selp(accept, newv, oldv)

                    # Predicated update of next_save; update if save is accepted.
                    do_save = (accept & do_save)
                    next_save = cuda.selp(
                        do_save, next_save + dt_save, next_save
                    )

                    # infrequent save branch.
                    if do_save:
                        save_state(
                            state_buffer,
                            observables_buffer,
                            state_output[save_idx * save_state_bool, :],
                            observables_output[save_idx * save_obs_bool, :],
                            t)
                        save_idx += 1
                        if summarise:
                            update_summaries(
                                state_buffer,
                                observables_buffer,
                                state_summary_buffer,
                                observable_summary_buffer,
                                saves_per_summary)
                            if (save_idx + 1) % saves_per_summary == 0:
                                save_summaries(
                                    state_summary_buffer,
                                    observable_summary_buffer,
                                    state_summaries_output,
                                    observable_summaries_output,
                                    total_saved_samples)
                                summary_idx += 1
            #Max iterations exhausted
            status = int32(10)
            return status

        return loop_fn

    @property
    def shared_memory_elements(self):
        """
        Get the number of threads required by loop algorithm.

        Returns
        -------
        int
            Number of threads required per integration loop.
        """

    def update(self,
               updates_dict : Optional[dict] = None,
               silent: bool = False,
               **kwargs):
        """
        Pass updates to compile settings through the CUDAFactory interface.

        This method will invalidate the cache if an update is successful.
        Use silent=True when doing bulk updates with other component parameters
        to suppress warnings about unrecognized keys.

        Parameters
        ----------
        updates_dict
            Dictionary of parameters to update.
        silent
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
            if hasattr(self, key):
                # Update output and step functions
                setattr(self, key, value)
                recognised.add(key)
                self._invalidate_cache()

        unrecognised = set(updates_dict.keys()) - recognised
        if not silent and unrecognised:
            raise KeyError(
                f"Unrecognized parameters in update: {unrecognised}. "
                "These parameters were not updated.",
            )
        return recognised

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
        IVPLoop
            New instance of the integrator algorithm configured with parameters
            from the run object.
        """
        raise NotImplementedError

    @property
    def shared_memory_indices(self):
        return self.compile_settings.shared_memory_indices

    @property
    def constant_memory_indices(self):
        return self.compile_settings.constant_memory_indices

    @property
    def local_memory_indices(self):
        return self.compile_settings.local_memory_indices
