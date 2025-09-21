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

import numpy as np
from numba import cuda, int32

from cubie.cudasim_utils import activemask, all_sync
from cubie.CUDAFactory import CUDAFactory
from cubie.cudasim_utils import from_dtype as simsafe_dtype
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


        step_controller_fn = step_controller.device_function
        step_fn = step_object.step_function

        self.step_controller = step_controller
        self.algorithm = step_object

        shared_buffer_indices = LoopIndices.from_buffer_sizes(buffer_sizes)

        config = ODELoopConfig(buffer_sizes=buffer_sizes,
                               buffer_indices=shared_buffer_indices,
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

    def build(self):
        """
        Build the integrator loop, unpacking config for local scope.

        Returns
        -------
        callable
            The compiled integrator loop device function.
        """
        config = self.compile_settings

        precision = config.numba_precision
        simsafe_int32 = simsafe_dtype(np.int32)

        # Fetch device functions from config
        save_state = config.save_state_func
        update_summaries = config.update_summaries_func
        save_summaries = config.save_summaries_func
        step_controller = config.step_controller_fn
        step_fn = config.step_fn

        # Boolean toggles for saving observables and states
        flags = config.compile_flags
        save_obs_bool = flags.save_observables
        save_state_bool = flags.save_state
        summarise_obs_bool = flags.summarise_observables
        summarise_state_bool = flags.summarise_state
        summarise = summarise_obs_bool or summarise_state_bool

        # Indices into shared memory for work buffers
        shared_indices = config.buffer_indices
        state_shared_ind = shared_indices.state
        dxdt_shared_ind = shared_indices.dxdt
        obs_shared_ind = shared_indices.observables
        state_prop_shared_ind = shared_indices.proposed_state
        state_summ_shared_ind = shared_indices.state_summaries
        params_shared_ind = shared_indices.parameters
        obs_summ_shared_ind = shared_indices.observable_summaries
        drivers_shared_ind = shared_indices.drivers
        remaining_scratch_ind = shared_indices.scratch

        # Timing values
        saves_per_summary = config.saves_per_summary
        dt_save = precision(config.dt_save)
        dt0 = precision(self.dt0)
        dt_min = precision(self.dt_min)
        steps_per_save = int32(ceil(precision(dt_save) / precision(dt0)))

        # Loop sizes
        n_states = config.buffer_sizes.state
        n_parameters = config.buffer_sizes.nonzero.parameters
        n_drivers = config.buffer_sizes.drivers

        fixed_mode = not self.is_adaptive
        controller_scratch = self.step_controller.local_memory_elements
        algo_persistent = self.algorithm.persistent_local_required

        @cuda.jit(device=True, inline=True)
        def loop_fn(
            initial_states,
            parameters,
            drivers,
            shared_scratch,
            persistent_local,
            state_output,
            observables_output,
            state_summaries_output,
            observable_summaries_output,
            duration,
            settling_time,
            t0 = precision(0.0),
        ):
            # Cap max iterations - all internal algorithms_ plus a bonus end/start
            t = precision(t0)
            t_end = precision(settling_time + duration)
            max_steps =  (int32(
                          ceil(t_end / max(dt_min, precision(1e-16))))
                          + int32(2)) * 2
            n_output_samples = max(state_output.shape[0],
                                   observables_output.shape[0])

            shared_scratch[:] = precision(0.0)

            state_buffer = shared_scratch[state_shared_ind]
            state_proposal_buffer = shared_scratch[state_prop_shared_ind]
            work_buffer = shared_scratch[dxdt_shared_ind]
            observables_buffer = shared_scratch[obs_shared_ind]
            parameters_buffer = shared_scratch[params_shared_ind]
            drivers_buffer = shared_scratch[drivers_shared_ind]
            state_summary_buffer = shared_scratch[state_summ_shared_ind]
            observable_summary_buffer = shared_scratch[obs_summ_shared_ind]
            remaining_shared_scratch = shared_scratch[remaining_scratch_ind]

            for k in range(n_states):
                state_buffer[k] = initial_states[k]
                state_output[0,k] = initial_states[k]
            for k in range(n_parameters):
                parameters_buffer[k] = parameters[k]

            driver_length = drivers.shape[0]

            # Loop state

            #Start again from settling time if given, otherwise keep init val.
            if settling_time > precision(0.0):
                next_save = precision(settling_time)
                save_idx=int32(0)
            else:
                next_save = precision(dt_save)
                save_idx = int32(1)

            status = int32(0)
            summary_idx = int32(0)

            #TODO: break these up into slices in the factory
            local_index = 0
            dt = persistent_local[local_index:local_index + 1]
            local_index += 1
            error_integral = persistent_local[local_index:local_index+1]
            local_index += 1
            accept_step = persistent_local[local_index:local_index+1].view(
                    simsafe_int32)
            local_index += 1
            error = persistent_local[local_index:local_index + n_states]
            #TODO: This error array is right in the hot loop and persistent,
            # consider making it shared
            local_index += n_states
            controller_temp = persistent_local[
                local_index : local_index + controller_scratch
            ]
            local_index += controller_scratch
            algo_local = persistent_local[
               local_index:local_index + algo_persistent
            ]
            if fixed_mode:
                step_counter = int32(0)
            dt[0] = dt0
            dt_eff = dt[0]
            accept_step[0] = int32(0)

            mask = activemask()

            for _ in range(max_steps):
                finished = save_idx >= n_output_samples
                if all_sync(mask, finished):
                    return status
                if finished:
                    return status

                for k in range(n_drivers):
                    drivers_buffer[k] = drivers[
                        save_idx % driver_length, k
                    ]
                if not finished:

                    if fixed_mode:
                        # Save every N steps
                        step_counter += 1
                        do_save = (step_counter % steps_per_save) == 0
                        if do_save:
                            step_counter = int32(0)
                    else:
                        #If the next step would be at or past the save time,
                        # save.
                        do_save = (t + dt[0]) >= next_save # Add an equality
                        dt_eff = cuda.selp(do_save, next_save - t, dt[0])
                        status |= cuda.selp(dt_eff <= precision(0.0),
                                            int32(16), int32(0))

                    status |= step_fn(
                        state_buffer,
                        state_proposal_buffer,
                        work_buffer,
                        parameters_buffer,
                        drivers_buffer,
                        observables_buffer,
                        error,
                        dt_eff,
                        remaining_shared_scratch,
                        algo_local,
                    )

                    # Adjust dt if step rejected - auto-accepts if fixed-step
                    if fixed_mode:
                        accept = True
                    else:
                        status |= step_controller(
                            dt,
                            state_buffer,
                            state_proposal_buffer,
                            error,
                            accept_step,
                            controller_temp,
                        )
                        accept = accept_step[0] != int32(0)  # Convert int32 to boolean

                    t_proposal = t + dt_eff
                    t = cuda.selp(accept, t_proposal, t)

                    for i in range(n_states):
                        newv = state_proposal_buffer[i]
                        oldv = state_buffer[i]
                        state_buffer[i] = cuda.selp(accept, newv, oldv)

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
                                    saves_per_summary,
                                )
                                summary_idx += 1
            if status == int32(0):
                #Max iterations exhausted without other error
                status = int32(8)
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
        algo = self.algorithm.shared_memory_required
        # controller = self.step_controller.shared_memory_elements
        own = self.buffer_indices.local_end
        return algo + own

    @property
    def local_memory_elements(self):
        self_elements = self.compile_settings.buffer_sizes.state + 3
        algo_elements = self.algorithm.persistent_local_required
        controller_elements = self.step_controller.local_memory_elements
        return self_elements + algo_elements + controller_elements

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
                 'step_fn': self.algorithm.step_function}
        )
        recognised |= self.update_compile_settings(updates_dict, silent=True)

        unrecognised = set(updates_dict.keys()) - recognised
        if not silent and unrecognised:
            raise KeyError(
                f"Unrecognized parameters in update: {unrecognised}. "
                "These parameters were not updated.",
            )
        return recognised


    # @property
    # def constant_memory_indices(self):
    #     return self.compile_settings.constant_memory_indices
    #
    # @property
    # def local_memory_indices(self):
    #     return self.compile_settings.local_memory_indices
