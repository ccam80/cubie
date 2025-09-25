"""
Base class for integration algorithm Loops.

This module provides the IVPLoop class, which serves as
the base class for all ODE integration loops. This class provides default
update behaviour and properties for a unified interface and inherits build
and cache logic from CUDAFactory.

Integration loops handle the "outer" logic of an ODE integration, organising
algorithms and saving output, and call an algorithm-specific step function to do the
mathy end of the integration.
"""
from math import ceil
from typing import Callable, Optional

import numpy as np
from numba import cuda, int32

from cubie.CUDAFactory import CUDAFactory
from cubie.cudasim_utils import from_dtype as simsafe_dtype
from cubie.cudasim_utils import activemask, all_sync
from cubie.integrators.loops.ode_loop_config import (LoopSharedIndices,
                                                     ODELoopConfig,
                                                     LoopLocalIndices)
from cubie.outputhandling import OutputCompileFlags


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
        shared_indices: LoopSharedIndices,
        local_indices: LoopLocalIndices,
        compile_flags: OutputCompileFlags,
        dt_save: float,
        dt_summarise: float,
        dt0: Optional[float]=None,
        dt_min: Optional[float]=None,
        dt_max: Optional[float]=None,
        is_adaptive: Optional[bool]=None,
        save_state_func: Optional[Callable] = None,
        update_summaries_func: Optional[Callable] = None,
        save_summaries_func: Optional[Callable] = None,
        step_controller_fn: Optional[Callable] = None,
        step_fn: Optional[Callable] = None,
    ) -> None:
        super().__init__()

        config = ODELoopConfig(
            shared_buffer_indices=shared_indices,
            local_indices=local_indices,
            save_state_fn=save_state_func,
            update_summaries_fn=update_summaries_func,
            save_summaries_fn=save_summaries_func,
            step_controller_fn=step_controller_fn,
            step_fn=step_fn,
            precision=precision,
            compile_flags=compile_flags,
            dt_save=dt_save,
            dt_summarise=dt_summarise,
            dt0=dt0,
            dt_min=dt_min,
            dt_max=dt_max,
            is_adaptive=is_adaptive,
        )
        self.setup_compile_settings(config)

    # @property
    # def is_adaptive(self):
    #     """Returns whether the loop is adaptive-step"""
    #     return self.step_controller.is_adaptive

    @property
    def precision(self):
        """Returns numerical precision for system"""
        return self.compile_settings.precision

    @property
    def numba_precision(self):
        """Return the Numba compatible precision for the loop."""

        return self.compile_settings.numba_precision

    @property
    def simsafe_precision(self):
        """Return the simulator safe precision for the loop."""

        return self.compile_settings.simsafe_precision

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

        save_state = config.save_state_fn
        update_summaries = config.update_summaries_fn
        save_summaries = config.save_summaries_fn
        step_controller = config.step_controller_fn
        step_fn = config.step_fn

        flags = config.compile_flags
        save_obs_bool = flags.save_observables
        save_state_bool = flags.save_state
        summarise_obs_bool = flags.summarise_observables
        summarise_state_bool = flags.summarise_state
        summarise = summarise_obs_bool or summarise_state_bool

        # Indices into shared memory for work buffers
        shared_indices = config.shared_buffer_indices
        local_indices = config.local_indices
        
        state_shared_ind = shared_indices.state
        dxdt_shared_ind = shared_indices.dxdt
        obs_shared_ind = shared_indices.observables
        state_prop_shared_ind = shared_indices.proposed_state
        state_summ_shared_ind = shared_indices.state_summaries
        params_shared_ind = shared_indices.parameters
        obs_summ_shared_ind = shared_indices.observable_summaries
        drivers_shared_ind = shared_indices.drivers
        remaining_scratch_ind = shared_indices.scratch

        dt_slice = local_indices.dt
        accept_slice = local_indices.accept
        error_slice = local_indices.error
        controller_slice = local_indices.controller
        algorithm_slice = local_indices.algorithm

        # Timing values
        saves_per_summary = config.saves_per_summary
        dt_save = precision(config.dt_save)
        dt0 = precision(config.dt0)
        dt_min = precision(config.dt_min)
        steps_per_save = int32(ceil(precision(dt_save) / precision(dt0)))

        # Loop sizes
        n_states = shared_indices.n_states
        n_parameters = shared_indices.n_parameters
        n_drivers = shared_indices.n_drivers

        fixed_mode = not config.is_adaptive
        status_mask = int32(0xFFFF)

        equality_breaker = precision(1e-16)
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
            t0=precision(0.0),
        ):
            t = precision(t0)
            t_end = precision(settling_time + duration)

            # Cap max iterations - all internal steps at dt_min, plus a bonus
            # end/start, plus one failure per successful step.
            max_steps = (int32(ceil(t_end / dt_min)) + int32(2))
            max_steps = max_steps << 1

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

            dt = persistent_local[dt_slice]
            accept_step = persistent_local[accept_slice].view(simsafe_int32)
            error = persistent_local[error_slice]
            controller_temp = persistent_local[controller_slice]
            algo_local = persistent_local[algorithm_slice]
            
            for k in range(n_states):
                state_buffer[k] = initial_states[k]
            for k in range(n_parameters):
                parameters_buffer[k] = parameters[k]

            driver_length = drivers.shape[0]
            save_idx = int32(0)

            if settling_time > precision(0.0):
                next_save = precision(settling_time)
            else:
                next_save = precision(dt_save)
                save_state(
                    state_buffer,
                    observables_buffer,
                    state_output[save_idx * save_state_bool, :],
                    observables_output[save_idx * save_obs_bool, :],
                    t,
                )
                save_idx += int32(1)

            status = int32(0)
            summary_idx = int32(0)
            dt[0] = dt0
            dt_eff = dt[0]
            accept_step[0] = int32(0)

            if fixed_mode:
                step_counter = int32(0)

            mask = activemask()

            for _ in range(max_steps):
                finished = save_idx >= n_output_samples
                if all_sync(mask, finished):
                    return status

                for k in range(n_drivers):
                    drivers_buffer[k] = drivers[save_idx % driver_length, k]

                if not finished:
                    if fixed_mode:
                        step_counter += 1
                        accept = True
                        do_save = (step_counter % steps_per_save) == 0
                        if do_save:
                            step_counter = int32(0)
                    else:
                        do_save = (t + dt[0]  +equality_breaker) >= next_save
                        dt_eff = cuda.selp(do_save, next_save - t, dt[0])
                        status |= cuda.selp(dt_eff <= precision(0.0),
                                            int32(16), int32(0))

                    step_status = step_fn(
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
                    niters = (step_status >> 16) & status_mask
                    status |= step_status & status_mask

                    # Adjust dt if step rejected - auto-accepts if fixed-step
                    if not fixed_mode:
                        status |= step_controller(
                            dt,
                            state_buffer,
                            state_proposal_buffer,
                            error,
                            niters,
                            accept_step,
                            controller_temp,
                        )
                        accept = accept_step[0] != int32(0)

                    t_proposal = t + dt_eff
                    t = cuda.selp(accept, t_proposal, t)

                    for i in range(n_states):
                        newv = state_proposal_buffer[i]
                        oldv = state_buffer[i]
                        state_buffer[i] = cuda.selp(accept, newv, oldv)

                    # Predicated update of next_save; update if save is accepted.
                    do_save = accept and do_save
                    next_save = cuda.selp(
                        do_save, next_save + dt_save, next_save
                    )

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
                                        summary_idx * summarise_state_bool, :],
                                    observable_summaries_output[
                                        summary_idx * summarise_obs_bool, :],
                                    saves_per_summary,
                                )
                                summary_idx += 1
            if status == int32(0):
                #Max iterations exhausted without other error
                status = int32(8)
            return status

        return loop_fn

    @property
    def dt_save(self) -> float:
        """Return the save interval."""

        return self.compile_settings.dt_save

    @property
    def dt_summarise(self):
        return self.compile_settings.dt_summarise

    @property
    def shared_buffer_indices(self) -> LoopSharedIndices:
        """Return the shared buffer index layout."""

        return self.compile_settings.shared_buffer_indices

    @property
    def buffer_indices(self) -> LoopSharedIndices:
        """Return the shared buffer index layout."""

        return self.shared_buffer_indices

    @property
    def local_indices(self) -> LoopLocalIndices:
        """Return persistent local-memory indices."""

        return self.compile_settings.local_indices

    @property
    def shared_memory_elements(self) -> int:
        """Return the loop's shared-memory requirement."""
        return self.compile_settings.loop_shared_elements

    @property
    def local_memory_elements(self) -> int:
        """Return the loop's persistent local-memory requirement."""
        return self.compile_settings.loop_local_elements

    @property
    def compile_flags(self) -> OutputCompileFlags:
        """Return the output compile flags associated with the loop."""

        return self.compile_settings.compile_flags

    @property
    def save_state_fn(self) -> Optional[Callable]:
        """Return the cached state saving device function."""

        return self.compile_settings.save_state_fn

    @property
    def update_summaries_fn(self) -> Optional[Callable]:
        """Return the cached summary update device function."""

        return self.compile_settings.update_summaries_fn

    @property
    def save_summaries_fn(self) -> Optional[Callable]:
        """Return the cached summary saving device function."""

        return self.compile_settings.save_summaries_fn

    @property
    def step_controller_fn(self) -> Optional[Callable]:
        """Return the device function implementing step control."""

        return self.compile_settings.step_controller_fn

    @property
    def step_fn(self) -> Optional[Callable]:
        """Return the algorithm step device function used by the loop."""

        return self.compile_settings.step_fn

    @property
    def dt0(self) -> Optional[float]:
        """Return the initial step size provided to the loop."""

        return self.compile_settings.dt0

    @property
    def dt_min(self) -> Optional[float]:
        """Return the minimum allowable step size for the loop."""

        return self.compile_settings.dt_min

    @property
    def dt_max(self) -> Optional[float]:
        """Return the maximum allowable step size for the loop."""

        return self.compile_settings.dt_max

    @property
    def is_adaptive(self) -> Optional[bool]:
        """Return whether the loop operates in adaptive mode."""

        return self.compile_settings.is_adaptive

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
        updates_dict = updates_dict.copy()
if kwargs:
            updates_dict.update(kwargs)
        if updates_dict == {}:
            return set()

        recognised = self.update_compile_settings(updates_dict, silent=True)
        unrecognised = set(updates_dict.keys()) - recognised
        if not silent and unrecognised:
            raise KeyError(
                f"Unrecognized parameters in update: {unrecognised}. "
                "These parameters were not updated.",
            )
        return recognised
