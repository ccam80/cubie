"""Outer integration loops for running CUDA-based ODE solvers.

The :class:`IVPLoop` orchestrates an integration by coordinating device step
functions, output collectors, and adaptive controllers. The loop owns buffer
layout metadata and feeds the appropriate slices into each device call so that
compiled kernels only need to focus on algorithmic updates.
"""
from math import ceil
from typing import Callable, Optional, Set

import numpy as np
from numba import cuda, int16, int32

from cubie.CUDAFactory import CUDAFactory
from cubie.cuda_simsafe import from_dtype as simsafe_dtype
from cubie.cuda_simsafe import activemask, all_sync, selp
from cubie._utils import PrecisionDType
from cubie.integrators.loops.ode_loop_config import (LoopLocalIndices,
                                                     LoopSharedIndices,
                                                     ODELoopConfig)
from cubie.outputhandling import OutputCompileFlags


# Recognised compile-critical loop configuration parameters. These keys mirror
# the solver API so helper utilities can consistently merge keyword arguments
# into loop-specific settings dictionaries.
ALL_LOOP_SETTINGS = {
    "dt_save",
    "dt_summarise",
    "dt0",
    "dt_min",
    "dt_max",
    "is_adaptive",
}


class IVPLoop(CUDAFactory):
    """Factory for CUDA device loops that advance an IVP integration.

    Parameters
    ----------
    precision
        Precision used for state and observable updates.
    shared_indices
        Buffer layout describing slices of shared memory arrays.
    local_indices
        Buffer layout describing slices of persistent local memory.
    compile_flags
        Output configuration that drives save and summary behaviour.
    dt_save
        Interval between accepted saves. Defaults to ``0.1`` when not
        provided.
    dt_summarise
        Interval between summary accumulations. Defaults to ``1.0`` when not
        provided.
    dt0
        Initial timestep applied before controller feedback.
    dt_min
        Minimum allowable timestep.
    dt_max
        Maximum allowable timestep.
    is_adaptive
        Whether an adaptive controller is used.
    save_state_func
        Device function that writes state and observable snapshots.
    update_summaries_func
        Device function that accumulates summary statistics.
    save_summaries_func
        Device function that commits summary statistics to output buffers.
    step_controller_fn
        Device function that updates the timestep and accept flag.
    step_function
        Device function that advances the solution by one tentative step.
    driver_function
        Device function that evaluates drivers for a given time.
    observables_fn
        Device function that computes observables for proposed states.
    """

    def __init__(
        self,
        precision: PrecisionDType,
        shared_indices: LoopSharedIndices,
        local_indices: LoopLocalIndices,
        compile_flags: OutputCompileFlags,
        dt_save: float = 0.1,
        dt_summarise: float = 1.0,
        dt0: Optional[float]=None,
        dt_min: Optional[float]=None,
        dt_max: Optional[float]=None,
        is_adaptive: Optional[bool]=None,
        save_state_func: Optional[Callable] = None,
        update_summaries_func: Optional[Callable] = None,
        save_summaries_func: Optional[Callable] = None,
        step_controller_fn: Optional[Callable] = None,
        step_function: Optional[Callable] = None,
        driver_function: Optional[Callable] = None,
        observables_fn: Optional[Callable] = None,
    ) -> None:
        super().__init__()

        config = ODELoopConfig(
            shared_buffer_indices=shared_indices,
            local_indices=local_indices,
            save_state_fn=save_state_func,
            update_summaries_fn=update_summaries_func,
            save_summaries_fn=save_summaries_func,
            step_controller_fn=step_controller_fn,
            step_function=step_function,
            driver_function=driver_function,
            observables_fn=observables_fn,
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

    @property
    def precision(self) -> PrecisionDType:
        """Return the numerical precision used for the loop."""
        return self.compile_settings.precision

    @property
    def numba_precision(self) -> type:
        """Return the Numba compatible precision for the loop."""

        return self.compile_settings.numba_precision

    @property
    def simsafe_precision(self) -> type:
        """Return the simulator safe precision for the loop."""

        return self.compile_settings.simsafe_precision

    def build(self) -> Callable:
        """Compile the CUDA device loop.

        Returns
        -------
        Callable
            Compiled device function that executes the integration loop.
        """
        config = self.compile_settings

        precision = config.numba_precision
        simsafe_int32 = simsafe_dtype(np.int32)

        save_state = config.save_state_fn
        update_summaries = config.update_summaries_fn
        save_summaries = config.save_summaries_fn
        step_controller = config.step_controller_fn
        step_function = config.step_function
        driver_function = config.driver_function
        observables_fn = config.observables_fn

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
        obs_shared_ind = shared_indices.observables
        obs_prop_shared_ind = shared_indices.proposed_observables
        state_prop_shared_ind = shared_indices.proposed_state
        state_summ_shared_ind = shared_indices.state_summaries
        params_shared_ind = shared_indices.parameters
        obs_summ_shared_ind = shared_indices.observable_summaries
        drivers_shared_ind = shared_indices.drivers
        drivers_prop_shared_ind = shared_indices.proposed_drivers
        error_shared_ind = shared_indices.error
        remaining_scratch_ind = shared_indices.scratch

        dt_slice = local_indices.dt
        accept_slice = local_indices.accept
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
        n_observables = shared_indices.n_observables
        n_drivers = shared_indices.n_drivers

        fixed_mode = not config.is_adaptive
        status_mask = int32(0xFFFF)

        equality_breaker = precision(1e-7) if precision is np.float32 else (
        precision(1e-14))

        @cuda.jit(device=True, inline=True)
        def loop_fn(
            initial_states,
            parameters,
            driver_coefficients,
            shared_scratch,
            persistent_local,
            state_output,
            observables_output,
            state_summaries_output,
            observable_summaries_output,
            duration,
            settling_time,
            t0=precision(0.0),
        ): # pragma: no cover - CUDA fns not marked in coverage
            """Advance an integration using a compiled CUDA device loop.

            Parameters
            ----------
            initial_states
                Device array containing the initial state vector.
            parameters
                Device array containing static parameters.
            driver_coefficients
                Device array containing precomputed spline coefficients.
            shared_scratch
                Device array providing shared-memory work buffers.
            persistent_local
                Device array providing persistent local memory buffers.
            state_output
                Device array storing accepted state snapshots.
            observables_output
                Device array storing accepted observable snapshots.
            state_summaries_output
                Device array storing aggregated state summaries.
            observable_summaries_output
                Device array storing aggregated observable summaries.
            duration
                Total integration duration.
            settling_time
                Lead-in time before samples are collected.
            t0
                Initial integration time.

            Returns
            -------
            int
                Status code aggregating errors and iteration counts.
            """
            t = precision(t0)
            t_end = precision(settling_time + duration)

            # Cap max iterations - all internal steps at dt_min, plus a bonus
            # end/start, plus one failure per successful step.
            max_steps = (int32(ceil(t_end / dt_min)) + int32(2))
            max_steps = max_steps << 2

            n_output_samples = max(state_output.shape[0],
                                   observables_output.shape[0])

            shared_scratch[:] = precision(0.0)

            state_buffer = shared_scratch[state_shared_ind]
            state_proposal_buffer = shared_scratch[state_prop_shared_ind]
            observables_buffer = shared_scratch[obs_shared_ind]
            observables_proposal_buffer = shared_scratch[obs_prop_shared_ind]
            parameters_buffer = shared_scratch[params_shared_ind]
            drivers_buffer = shared_scratch[drivers_shared_ind]
            drivers_proposal_buffer = shared_scratch[drivers_prop_shared_ind]
            state_summary_buffer = shared_scratch[state_summ_shared_ind]
            observable_summary_buffer = shared_scratch[obs_summ_shared_ind]
            remaining_shared_scratch = shared_scratch[remaining_scratch_ind]

            dt = persistent_local[dt_slice]
            accept_step = persistent_local[accept_slice].view(simsafe_int32)
            # Non-adaptive algorithms map the error slice to length zero.
            error = shared_scratch[error_shared_ind]
            controller_temp = persistent_local[controller_slice]
            algo_local = persistent_local[algorithm_slice]

            first_step_flag = int16(1)
            prev_step_accepted_flag = int16(1)


            # --------------------------------------------------------------- #
            #                       Seed t=0 values                           #
            # --------------------------------------------------------------- #
            for k in range(n_states):
                state_buffer[k] = initial_states[k]
            for k in range(n_parameters):
                parameters_buffer[k] = parameters[k]

            # Seed initial observables from initial state.
            if driver_function is not None and n_drivers > 0:
                driver_function(
                    t,
                    driver_coefficients,
                    drivers_buffer,
                )
            if n_observables > 0:
                observables_fn(
                    state_buffer,
                    parameters_buffer,
                    drivers_buffer,
                    observables_buffer,
                    t,
                )

            save_idx = int32(0)
            summary_idx = int32(0)

            if settling_time > precision(0.0):
                #Don't save t0, wait until settling_time
                next_save = precision(settling_time)
            else:
                #Seed initial state and save/update summaries
                next_save = precision(dt_save)
                save_state(
                    state_buffer,
                    observables_buffer,
                    state_output[save_idx * save_state_bool, :],
                    observables_output[save_idx * save_obs_bool, :],
                    t,
                )
                if summarise:
                    #reset temp buffers to starting state - will be overwritten
                    save_summaries(state_summary_buffer,
                                   observable_summary_buffer,
                                   state_summaries_output[
                                       summary_idx * summarise_state_bool, :
                                   ],
                                   observable_summaries_output[
                                       summary_idx * summarise_obs_bool, :
                                   ],
                                   saves_per_summary)
                    
                    # Log first summary update
                    update_summaries(
                        state_buffer,
                        observables_buffer,
                        state_summary_buffer,
                        observable_summary_buffer,
                        save_idx,
                    )
                save_idx += int32(1)

            status = int32(0)
            dt[0] = dt0
            dt_eff = dt[0]
            accept_step[0] = int32(0)

            if fixed_mode:
                step_counter = int32(0)

            mask = activemask()

            # --------------------------------------------------------------- #
            #                        Main Loop                                #
            # --------------------------------------------------------------- #
            for _ in range(max_steps):
                finished = save_idx >= n_output_samples

                if all_sync(mask, finished):
                    return status

                if not finished:
                    if fixed_mode:
                        step_counter += 1
                        accept = True
                        do_save = (step_counter % steps_per_save) == 0
                        if do_save:
                            step_counter = int32(0)
                    else:
                        do_save = (t + dt[0]  +equality_breaker) >= next_save
                        dt_eff = selp(do_save, next_save - t, dt[0])

                        status |= selp(dt_eff <= precision(0.0), int32(16), int32(0))

                    step_status = step_function(
                        state_buffer,
                        state_proposal_buffer,
                        parameters_buffer,
                        driver_coefficients,
                        drivers_buffer,
                        drivers_proposal_buffer,
                        observables_buffer,
                        observables_proposal_buffer,
                        error,
                        dt_eff,
                        t,
                        first_step_flag,
                        prev_step_accepted_flag,
                        remaining_shared_scratch,
                        algo_local,
                    )

                    first_step_flag = int16(0)

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
                    t = selp(accept, t_proposal, t)

                    for i in range(n_states):
                        newv = state_proposal_buffer[i]
                        oldv = state_buffer[i]
                        state_buffer[i] = selp(accept, newv, oldv)

                    for i in range(n_drivers):
                        new_drv = drivers_proposal_buffer[i]
                        old_drv = drivers_buffer[i]
                        drivers_buffer[i] = selp(accept, new_drv, old_drv)

                    for i in range(n_observables):
                        new_obs = observables_proposal_buffer[i]
                        old_obs = observables_buffer[i]
                        observables_buffer[i] = selp(accept, new_obs, old_obs)

                    prev_step_accepted_flag = selp(
                        accept,
                        int16(1),
                        int16(0),
                    )

                    # Predicated update of next_save; update if save is accepted.
                    do_save = accept and do_save
                    next_save = selp(
                        do_save, next_save + dt_save, next_save
                    )

                    if do_save:
                        save_state(
                            state_buffer,
                            observables_buffer,
                            state_output[save_idx * save_state_bool, :],
                            observables_output[save_idx * save_obs_bool, :],
                            t,
                        )
                        if summarise:
                            update_summaries(
                                state_buffer,
                                observables_buffer,
                                state_summary_buffer,
                                observable_summary_buffer,
                                save_idx)

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
                        save_idx += 1

            if status == int32(0):
                #Max iterations exhausted without other error
                status = int32(32)
            return status

        return loop_fn

    @property
    def dt_save(self) -> float:
        """Return the save interval."""

        return self.compile_settings.dt_save

    @property
    def dt_summarise(self) -> float:
        """Return the summary interval."""

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
    def step_function(self) -> Optional[Callable]:
        """Return the algorithm step device function used by the loop."""

        return self.compile_settings.step_function

    @property
    def driver_function(self) -> Optional[Callable]:
        """Return the driver evaluation device function used by the loop."""

        return self.compile_settings.driver_function

    @property
    def observables_fn(self) -> Optional[Callable]:
        """Return the observables device function used by the loop."""

        return self.compile_settings.observables_fn

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

    def update(
        self,
        updates_dict: Optional[dict[str, object]] = None,
        silent: bool = False,
        **kwargs: object,
    ) -> Set[str]:
        """Update compile settings through the CUDAFactory interface.

        Parameters
        ----------
        updates_dict
            Mapping of configuration names to replacement values.
        silent
            When True, suppress warnings about unrecognized parameters.
        **kwargs
            Additional configuration updates applied as keyword arguments.

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
