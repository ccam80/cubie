"""
Base class for integration algorithm Loops.

This module provides the IVPLoop class, which serves as
the base class for all ODE integration loops. This class provides default
update behaviour and properties for a unified interface and inherits build
and cache logic from CUDAFactory.

Integration loops handle the "outer" logic of an ODE integration, organising
algorithms_ and saving output, and call an algorithm-specific step function to do the
mathy end of the integration.
"""
from typing import Optional, Callable
from warnings import warn

from numba import cuda, from_dtype, int32

from cubie.CUDAFactory import CUDAFactory
from cubie.integrators.algorithms_.base_algorithm_step import BaseAlgorithmStep
from cubie.integrators.loops.ode_loop_config import LoopIndices, ODELoopConfig
from cubie.integrators.step_control.base_step_controller import \
    BaseStepController
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
        dt_save: float,
        dt_summarise: float,
        step_controller: BaseStepController,
        step_object: BaseAlgorithmStep,
        buffer_sizes: LoopBufferSizes,
        compile_flags: OutputCompileFlags,
        save_state_func: Callable,
        update_summaries_func: Callable,
        save_summaries_func: Callable,
    ):
        super().__init__()
        is_adaptive = step_controller.is_adaptive
        new_timing = self.validate_timing(step_controller.dt_min,
                                          step_controller.dt_max,
                                          dt_save,
                                          dt_summarise,
                                          is_adaptive=is_adaptive
                                          )
        dt_min, dt_max, dt_save, dt_summarise = new_timing
        step_controller.update(_dt_min=dt_min, _dt_max=dt_max)
        step_controller_fn = step_controller.device_function
        step_fn = step_object.device_function

        self.step_controller = step_controller
        self.algorithm = step_object

        shared_buffer_indices = LoopIndices.from_buffer_sizes(buffer_sizes)

        config = ODELoopConfig(buffer_indices=shared_buffer_indices,
                               save_state_func=save_state_func,
                               update_summaries_func=update_summaries_func,
                               save_summaries_func=save_summaries_func,
                               step_controller_fn=step_controller_fn,
                               step_fn=step_fn,
                               precision=precision,
                               compile_flags=compile_flags,
                               dt_save=dt_save,
                               dt_summarise=dt_summarise,
                               )
        self.setup_compile_settings(config)

    @property
    def is_adaptive(self):
        """Returns whether the loop is adaptive-step"""
        return self.step_controller.is_adaptive

    @property
    def precision(self):
        return self.compile_settings.precision

    def validate_timing(self,
                        dt_min: float,
                        dt_max: float,
                        dt_save: float,
                        dt_summarise: float,
                        is_adaptive: bool):
        """Check that user-provided step, save, and summary timings are valid.

        Returns modified dt_min, dt_save and dt_summarise if required to
        enforce consistency.
        """
        if dt_save < dt_min:
            dt_min = dt_save / 10.0
            warn(f"dt_save ({dt_save}s) must be >= dt_min ({dt_min}s). "
                 f"Setting dt_min to dt_save / 10 by default.")
        if dt_summarise < (2 * dt_save):
            dt_summarise = max(2 * dt_save)
            warn(f"dt_summarise ({dt_summarise}s) must be at least 2*dt_save "
                 f"to be meaningful, setting dt_summarise to ({2 * dt_save}s)",
            )
        if dt_summarise % dt_save > 0.01:
            dt_summarise = ceil(dt_summarise / dt_save) * dt_save
            warn(f"dt_summarise ({dt_summarise}s) is not an integer multiple of "
                 f"dt_save ({dt_save}s). Cubie estimates summaries every n "
                 f"saves, so dt_summarise has been rounded down to the "
                 f"nearest integer multiple of dt_save.")
        if is_adaptive:
            if dt_max > dt_save:
                dt_max = dt_save
                warn(f"dt_max ({dt_max}s) > dt_save ({dt_save}s). "
                     f"The loop will never be able to step that far before "
                     f"stopping to save, so dt_max has been set to dt_save.")

        return dt_min, dt_max, dt_save, dt_summarise

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

        shared_indices = config.shared_memory_indices
        state_shared_ind = shared_indices.state
        obs_shared_ind = shared_indices.observables
        state_prop_shared_ind = shared_indices.state_proposal
        state_summ_shared_ind = shared_indices.state_summaries
        params_shared_ind = shared_indices.parameters
        obs_summ_shared_ind = shared_indices.observable_summaries
        drivers_shared_ind = shared_indices.drivers

        saves_per_summary = config.saves_per_summary

        precision = from_dtype(self.precision)

        n_states = config.buffer_sizes.state
        n_parameters = config.buffer_sizes.nonzero.parameters

        dt_save = config.dt_save


        dt0_default = self.dt0
        dt_min = self.dt_min
        fixed_mode = not self.is_adaptive



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
            # Cap max iterations - all internal algorithms_ plus a bonus end/start
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
                                    state_summaries_output[
                                        summary_idx * summarise_state_bool, :
                                    ],
                                    observable_summaries_output[
                                        summary_idx * summarise_obs_bool, :
                                    ],
                                    saves_per_summary)
                                summary_idx += 1
            #Max iterations exhausted
            status = int32(10)
            return status

        return loop_fn

    @property
    def dt0(self):
        return self.step_controller.dt0

    @property
    def dt_min(self):
        return self.step_controller.dt_min

    @property
    def dt_max(self):
        return self.step_controller.dt_max

    @property
    def dt_save(self):
        return self.compile_settings.dt_save

    @property
    def dt_summarise(self):
        return self.compile_settings.dt_summarise

    @property
    def buffer_indices(self):
        return self.compile_settings.buffer_indices

    @property
    def shared_memory_elements(self):
        """
        Get the number of threads required by loop algorithm.

        Returns
        -------
        int
            Number of threads required per integration loop.
        """
        algo = self.algorithm.shared_memory_elements
        controller = self.step_controller.shared_memory_elements
        own = self.buffer_


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

        recognised = set()
        recognised |= self.step_controller.update(updates_dict, silent=True)
        recognised |= self.algorithm.update(updates_dict, silent=True)

        # Get fresh device functions
        updates_dict.update(
                {'step_controller_fn': self.step_controller.device_function,
                 'step_fn': self.algorithm.device_function}
        )
        recognised |= self.update_compile_settings(updates_dict, silent=True)

        unrecognised = set(updates_dict.keys()) - recognised
        if not silent and unrecognised:
            raise KeyError(
                f"Unrecognized parameters in update: {unrecognised}. "
                "These parameters were not updated.",
            )
        return recognised

    @property
    def shared_memory_indices(self):
        return self.compile_settings.shared_memory_indices

    # @property
    # def constant_memory_indices(self):
    #     return self.compile_settings.constant_memory_indices
    #
    # @property
    # def local_memory_indices(self):
    #     return self.compile_settings.local_memory_indices
