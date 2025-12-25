"""Outer integration loops for running CUDA-based ODE solvers.

The :class:`IVPLoop` orchestrates an integration by coordinating device step
functions, output collectors, and adaptive controllers. The loop uses the
central buffer registry for memory allocation and provides slices into each
device call so that compiled kernels only focus on algorithmic updates.
"""
from typing import Callable, Optional, Set

import attrs
import numpy as np
from numba import cuda, int32, float64, bool_

from cubie.CUDAFactory import CUDAFactory, CUDAFunctionCache
from cubie.buffer_registry import buffer_registry
from cubie.cuda_simsafe import activemask, all_sync, compile_kwargs, selp
from cubie._utils import PrecisionDType, unpack_dict_values, build_config
from cubie.integrators.loops.ode_loop_config import (ODELoopConfig)
from cubie.outputhandling import OutputCompileFlags


@attrs.define
class IVPLoopCache(CUDAFunctionCache):
    """Cache for IVP loop device function.
    
    Attributes
    ----------
    loop_function
        Compiled CUDA device function that executes the integration loop.
    """
    loop_function: Callable = attrs.field()

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
    # Loop buffer location parameters
    "state_location",
    "proposed_state_location",
    "parameters_location",
    "drivers_location",
    "proposed_drivers_location",
    "observables_location",
    "proposed_observables_location",
    "error_location",
    "counters_location",
    "state_summary_location",
    "observable_summary_location",
    "dt_location",
    "accept_step_location",
    "proposed_counters_location",
}


class IVPLoop(CUDAFactory):
    """Factory for CUDA device loops that advance an IVP integration.

    Parameters
    ----------
    precision
        Precision used for state and observable updates.
    n_states
        Number of state variables.
    compile_flags
        Output configuration that drives save and summary behaviour.
    n_parameters
        Number of parameters.
    n_drivers
        Number of driver variables.
    n_observables
        Number of observable variables.
    n_error
        Number of error elements (typically equals n_states for adaptive).
    n_counters
        Number of counter elements.
    state_summaries_buffer_height
        Height of state summary buffer.
    observable_summaries_buffer_height
        Height of observable summary buffer.
    dt_save
        Interval between accepted saves. Defaults to ``0.1``.
    dt_summarise
        Interval between summary accumulations. Defaults to ``1.0``.
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
    **kwargs
        Optional parameters passed to ODELoopConfig. Available parameters
        include dt0, dt_min, dt_max, is_adaptive, controller_local_len,
        algorithm_local_len, and buffer location parameters (state_location,
        proposed_state_location, parameters_location, drivers_location,
        proposed_drivers_location, observables_location,
        proposed_observables_location, error_location, counters_location,
        state_summary_location, observable_summary_location, dt_location,
        accept_step_location). None values are ignored.
    """

    def __init__(
        self,
        precision: PrecisionDType,
        n_states: int,
        compile_flags: OutputCompileFlags,
        n_parameters: int = 0,
        n_drivers: int = 0,
        n_observables: int = 0,
        n_error: int = 0,
        n_counters: int = 0,
        state_summaries_buffer_height: int = 0,
        observable_summaries_buffer_height: int = 0,
        dt_save: float = 0.1,
        dt_summarise: float = 1.0,
        save_state_func: Optional[Callable] = None,
        update_summaries_func: Optional[Callable] = None,
        save_summaries_func: Optional[Callable] = None,
        step_controller_fn: Optional[Callable] = None,
        step_function: Optional[Callable] = None,
        driver_function: Optional[Callable] = None,
        observables_fn: Optional[Callable] = None,
        **kwargs,
    ) -> None:
        """Initialise the IVP loop configuration.

        Parameters
        ----------
        precision
            Precision used for state and observable updates.
        n_states
            Number of state variables.
        compile_flags
            Output configuration that drives save and summary behaviour.
        n_parameters
            Number of parameters.
        n_drivers
            Number of driver variables.
        n_observables
            Number of observable variables.
        n_error
            Number of error elements (typically equals n_states for adaptive).
        n_counters
            Number of counter elements.
        state_summary_buffer_height
            Height of state summary buffer.
        observable_summary_buffer_height
            Height of observable summary buffer.
        dt_save
            Interval between accepted saves.
        dt_summarise
            Interval between summary accumulations.
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
        **kwargs
            Optional parameters passed to ODELoopConfig. See ODELoopConfig
            for available parameters including dt0, dt_min, dt_max,
            is_adaptive, and buffer location parameters (state_location,
            proposed_state_location, etc.). None values are ignored.
        """
        super().__init__()

        config = build_config(
            ODELoopConfig,
            required={
                'n_states': n_states,
                'n_parameters': n_parameters,
                'n_drivers': n_drivers,
                'n_observables': n_observables,
                'n_error': n_error,
                'n_counters': n_counters,
                'state_summaries_buffer_height': state_summaries_buffer_height,
                'observable_summaries_buffer_height':
                    observable_summaries_buffer_height,
                'precision': precision,
                'compile_flags': compile_flags,
                'dt_save': dt_save,
                'dt_summarise': dt_summarise,
                'save_state_fn': save_state_func,
                'update_summaries_fn': update_summaries_func,
                'save_summaries_fn': save_summaries_func,
                'step_controller_fn': step_controller_fn,
                'step_function': step_function,
                'driver_function': driver_function,
                'observables_fn': observables_fn,
            },
            **kwargs
        )
        self.setup_compile_settings(config)
        self.register_buffers()

    def register_buffers(self) -> None:
        """Register buffers according to locations in compile settings."""
        config = self.compile_settings
        precision = config.precision
        n_states = config.n_states
        n_parameters = config.n_parameters
        n_drivers = config.n_drivers
        n_observables = config.n_observables
        n_error = config.n_error
        n_counters = config.n_counters
        state_summaries_buffer_height = config.state_summaries_buffer_height
        observable_summaries_buffer_height = config.observable_summaries_buffer_height

        # Register all loop buffers with central registry

        buffer_registry.register(
            'state', self, n_states, config.state_location,
            precision=precision
        )
        buffer_registry.register(
            'proposed_state', self, n_states, config.proposed_state_location,
            precision=precision
        )
        buffer_registry.register(
            'parameters', self, n_parameters,
            config.parameters_location, precision=precision
        )
        buffer_registry.register(
            'drivers', self, n_drivers, config.drivers_location,
            precision=precision
        )
        buffer_registry.register(
            'proposed_drivers', self, n_drivers,
            config.proposed_drivers_location, precision=precision
        )
        buffer_registry.register(
            'observables', self, n_observables,
            config.observables_location, precision=precision
        )
        buffer_registry.register(
            'proposed_observables', self, n_observables,
            config.proposed_observables_location, precision=precision
        )
        buffer_registry.register(
            'error', self, n_error, config.error_location,
            precision=precision
        )
        buffer_registry.register(
            'counters', self, n_counters, config.counters_location,
            precision=precision
        )
        buffer_registry.register(
            'state_summary', self, state_summaries_buffer_height,
            config.state_summary_location, precision=precision
        )
        buffer_registry.register(
            'observable_summary', self, observable_summaries_buffer_height,
            config.observable_summary_location, precision=precision
        )
        buffer_registry.register(
            'dt', self, 1, config.dt_location, precision=precision
        )
        buffer_registry.register(
            'accept_step', self, 1, config.accept_step_location, precision=precision
        )
        buffer_registry.register(
            'proposed_counters', self, 2, config.proposed_counters_location,
            precision=np.int32
        )

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
        save_counters_bool = flags.save_counters

        # Get allocators from buffer registry
        getalloc = buffer_registry.get_allocator
        alloc_state = getalloc('state', self, zero=True)
        alloc_proposed_state = getalloc('proposed_state', self, zero=True)
        alloc_parameters = getalloc('parameters', self, zero=True)
        alloc_drivers = getalloc('drivers', self, zero=True)
        alloc_proposed_drivers = getalloc('proposed_drivers', self, zero=True)
        alloc_observables = getalloc('observables', self, zero=True)
        alloc_proposed_observables = getalloc('proposed_observables', self, zero=True)
        alloc_error = getalloc('error', self, zero=True)
        alloc_counters = getalloc('counters', self, zero=True)
        alloc_state_summary = getalloc('state_summary', self, zero=True)
        alloc_observable_summary = getalloc('observable_summary', self, zero=True)
        alloc_algo_shared = getalloc('algorithm_shared', self, zero=True)
        alloc_algo_persistent = getalloc('algorithm_persistent', self, zero=True)
        alloc_controller_shared = getalloc('controller_shared', self, zero=True)
        alloc_controller_persistent = getalloc('controller_persistent', self, zero=True)
        alloc_dt = getalloc('dt', self, zero=True)
        alloc_accept_step = getalloc('accept_step', self, zero=True)
        alloc_proposed_counters = getalloc('proposed_counters', self)

        # Timing values
        saves_per_summary = config.saves_per_summary
        dt_save = precision(config.dt_save)
        dt0 = precision(config.dt0)
        # save_last is not yet piped up from this level, but is intended and
        # included in loop logic
        save_last = False

        # Loop sizes from config (sizes also used for iteration bounds)
        n_states = int32(config.n_states)
        n_parameters = int32(config.n_parameters)
        n_observables = int32(config.n_observables)
        n_drivers = int32(config.n_drivers)
        n_counters = int32(config.n_counters)
        
        fixed_mode = not config.is_adaptive

        @cuda.jit(
            # [
            #     (
            #         precision[::1],
            #         precision[::1],
            #         precision[:, :, ::1],
            #         precision[::1],
            #         precision[::1],
            #         precision[:, :],
            #         precision[:, :],
            #         precision[:, :],
            #         precision[:, :],
            #         precision[:,::1],
            #         float64,
            #         float64,
            #         float64,
            #     )
            # ],
            device=True,
            inline=True,
            **compile_kwargs,
        )
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
            iteration_counters_output,
            duration,
            settling_time,
            t0,
        ): # pragma: no cover - CUDA fns not marked in coverage
            """Advance an integration using a compiled CUDA device loop.

            The loop terminates when the time of the next saved sample
            exceeds the end time (t0 + settling_time + duration), or when
            the maximum number of iterations is reached.

            Parameters
            ----------
            initial_states
                1d Device array containing the initial state vector.
            parameters
                1d Device array containing static parameters.
            driver_coefficients
                3d Device array containing precomputed spline coefficients.
            shared_scratch
                1d Device array providing shared-memory work buffers.
            persistent_local
                1d Device array providing persistent local memory buffers.
            state_output
                2d Device array storing accepted state snapshots.
            observables_output
                2d Device array storing accepted observable snapshots.
            state_summaries_output
                Device array storing aggregated state summaries.
            observable_summaries_output
                Device array storing aggregated observable summaries.
            iteration_counters_output
                Device array storing iteration counter values at each save.
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
            t = float64(t0)
            t_prec = precision(t)
            t_end = precision(settling_time + t0 + duration)

            stagnant_counts = int32(0)

            persistent_local[:] = precision(0.0)
            shared_scratch[:] = precision(0.0)
            # ----------------------------------------------------------- #
            # Allocate buffers using registry allocators
            # ----------------------------------------------------------- #
            state_buffer = alloc_state(shared_scratch, persistent_local)
            state_proposal_buffer = alloc_proposed_state(
                    shared_scratch, persistent_local
            )
            observables_buffer = alloc_observables(shared_scratch, persistent_local)
            observables_proposal_buffer = alloc_proposed_observables(
                shared_scratch, persistent_local
            )
            parameters_buffer = alloc_parameters(shared_scratch, persistent_local)
            drivers_buffer = alloc_drivers(shared_scratch, persistent_local)
            drivers_proposal_buffer = alloc_proposed_drivers(
                shared_scratch, persistent_local
            )
            state_summary_buffer = alloc_state_summary(
                shared_scratch, persistent_local
            )
            observable_summary_buffer = alloc_observable_summary(
                shared_scratch, persistent_local
            )
            counters_since_save = alloc_counters(shared_scratch, persistent_local)
            error = alloc_error(shared_scratch, persistent_local)

            # Allocate child buffers for algorithm step
            algo_shared = alloc_algo_shared(shared_scratch, persistent_local)
            algo_persistent = alloc_algo_persistent(shared_scratch, persistent_local)
            ctrl_shared = alloc_controller_shared(shared_scratch, persistent_local)
            ctrl_persistent = alloc_controller_persistent(
                shared_scratch, persistent_local
            )
            dt = alloc_dt(shared_scratch, persistent_local)
            accept_step = alloc_accept_step(shared_scratch, persistent_local)
            # ----------------------------------------------------------- #

            proposed_counters = alloc_proposed_counters(
                shared_scratch, persistent_local
            )
            first_step_flag = True
            prev_step_accepted_flag = True

            # --------------------------------------------------------------- #
            #                       Seed t=0 values                           #
            # --------------------------------------------------------------- #
            for k in range(n_states):
                state_buffer[k] = initial_states[k]
            for k in range(n_parameters):
                parameters_buffer[k] = parameters[k]

            # Seed initial observables from initial state.
            if driver_function is not None and n_drivers > int32(0):
                driver_function(
                    t_prec,
                    driver_coefficients,
                    drivers_buffer,
                )
            if n_observables > int32(0):
                observables_fn(
                    state_buffer,
                    parameters_buffer,
                    drivers_buffer,
                    observables_buffer,
                    t_prec,
                )

            save_idx = int32(0)
            summary_idx = int32(0)

            # Set next save for settling time, or save first value if
            # starting at t0
            next_save = precision(settling_time + t0)
            if settling_time == 0.0:
                # Save initial state at t0, then advance to first interval save
                next_save += dt_save

                save_state(
                    state_buffer,
                    observables_buffer,
                    counters_since_save,
                    t_prec,
                    state_output[save_idx * save_state_bool, :],
                    observables_output[save_idx * save_obs_bool, :],
                    iteration_counters_output[save_idx * save_counters_bool, :],
                )
                if summarise:
                    #reset temp buffers to starting state
                    statesumm_idx = summary_idx * summarise_state_bool
                    obsumm_idx = summary_idx * summarise_obs_bool
                    save_summaries(state_summary_buffer,
                                   observable_summary_buffer,
                                   state_summaries_output[statesumm_idx, :],
                                   observable_summaries_output[obsumm_idx, :],
                                   saves_per_summary)
                save_idx += int32(1)

            status = int32(0)
            dt[0] = dt0
            dt_raw = dt0
            accept_step[0] = int32(0)

            # Initialize iteration counters
            for i in range(n_counters):
                counters_since_save[i] = int32(0)
                if i < int32(2):
                    proposed_counters[i] = int32(0)

            mask = activemask()

            # --------------------------------------------------------------- #
            #                        Main Loop                                #
            # --------------------------------------------------------------- #
            while True:
                # Exit as soon as we've saved the final step
                finished = bool_(next_save > t_end)
                if save_last:
                    # If last save requested, predicated commit dt, finished,
                    # do_save
                    at_last_save = finished and t_prec < t_end
                    finished = selp(at_last_save, False, True)
                    dt[0] = selp(at_last_save, precision(t_end - t),
                                 dt_raw)

                # Exit loop if finished, or min_step exceeded, or time stagnant
                finished = finished or bool_(status & int32(0x8)) or bool_(
                        status * int32(0x40))

                if all_sync(mask, finished):
                    return status

                if not finished:
                    do_save = bool_((t_prec + dt_raw) >= next_save)
                    dt_eff = selp(do_save, next_save - t_prec, dt_raw)

                    # Fixed mode auto-accepts all steps; adaptive uses controller

                    step_status = int32(
                        step_function(
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
                            t_prec,
                            first_step_flag,
                            prev_step_accepted_flag,
                            algo_shared,
                            algo_persistent,
                            proposed_counters,
                        )
                    )

                    first_step_flag = False

                    niters = proposed_counters[0]
                    status = int32(status | step_status)

                    # Adjust dt if step rejected - auto-accepts if fixed-step
                    if not fixed_mode:
                        controller_status = step_controller(
                            dt,
                            state_proposal_buffer,
                            state_buffer,
                            error,
                            niters,
                            accept_step,
                            ctrl_shared,
                            ctrl_persistent,
                        )

                        accept = bool_(accept_step[0] != int32(0))
                        status = int32(status | controller_status)

                    else:
                        accept = True

                    dt_raw = dt[0]

                    # Accumulate iteration counters if active
                    if save_counters_bool:
                        for i in range(n_counters):
                            if i < int32(2):
                                # Write newton, krylov iterations from buffer
                                counters_since_save[i] += proposed_counters[i]
                            elif i == int32(2):
                                # Increment total steps counter
                                counters_since_save[i] += int32(1)
                            elif not accept:
                                # Increment rejected steps counter
                                counters_since_save[i] += int32(1)

                    t_proposal = t + float64(dt_eff)
                    # test for stagnation - we might have one small step
                    # which doesn't nudge t if we're right up against a save
                    # boundary, so we call 2 stale t values in a row "stagnant"
                    if t_proposal == t:
                        stagnant_counts += int32(1)
                    else:
                        stagnant_counts = int32(0)

                    stagnant = bool_(stagnant_counts >= int32(2))
                    status = selp(
                            stagnant,
                            int32(status | int32(0x40)),
                            status
                    )

                    t = selp(accept, t_proposal, t)
                    t_prec = precision(t)

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
                        int32(1),
                        int32(0),
                    )

                    # Predicated update of next_save; update if save is accepted.
                    do_save = bool_(accept and do_save)
                    if do_save:
                        next_save = selp(do_save, next_save + dt_save, next_save)
                        save_state(
                            state_buffer,
                            observables_buffer,
                            counters_since_save,
                            t_prec,
                            state_output[save_idx * save_state_bool, :],
                            observables_output[save_idx * save_obs_bool, :],
                            iteration_counters_output[
                                save_idx * save_counters_bool, :
                            ],
                        )
                        if summarise:
                            statesumm_idx = summary_idx * summarise_state_bool
                            obssumm_idx = summary_idx * summarise_obs_bool
                            update_summaries(
                                state_buffer,
                                observables_buffer,
                                state_summary_buffer,
                                observable_summary_buffer,
                                save_idx)

                            if (save_idx % saves_per_summary == int32(0)):
                                save_summaries(
                                    state_summary_buffer,
                                    observable_summary_buffer,
                                    state_summaries_output[statesumm_idx,:],
                                    observable_summaries_output[obssumm_idx,:],
                                    saves_per_summary,
                                )
                                summary_idx += int32(1)
                        save_idx += int32(1)

                        # Reset iteration counters after save
                        if save_counters_bool:
                            for i in range(n_counters):
                                counters_since_save[i] = int32(0)

        # Attach critical shapes for dummy execution
        # Parameters in order: initial_states, parameters, driver_coefficients,
        # shared_scratch, persistent_local, state_output, observables_output,
        # state_summaries_output, observable_summaries_output,
        # iteration_counters_output, duration, settling_time, t0
        loop_fn.critical_shapes = (
            (n_states,),  # initial_states
            (n_parameters,),  # parameters
            (100,n_states,6),  # driver_coefficients
            (32768//8), # local persistent - arbitrary 32kb provided / float64
            (32768//8),  # persistent_local - arbitrary 32kb provided / float64
            (100, n_states), # state_output
            (100, n_observables), # observables_output
            (100, n_states),  # state_summaries_output
            (100, n_observables), # obs summ output
            (1, n_counters),  # iteration_counters_output
            None,  # duration - scalar
            None,  # settling_time - scalar
            None,  # t0 - scalar (optional)
        )
        loop_fn.critical_values = (
            None,  # initial_states
            None,  # parameters
            None,  # driver_coefficients
            None,  # local persistent - not really used
            None,  # persistent_local - arbitrary 32kb provided / float64
            None,  # state_output
            None,  # observables_output
            None,  # state_summaries_output
            None,  # obs summ output
            None,  # iteration_counters_output
            self.dt_save + 0.01,  # duration - scalar
            0.0,  # settling_time - scalar
            0.0,  # t0 - scalar (optional)
        )
        return IVPLoopCache(loop_function=loop_fn)

    @property
    def dt_save(self) -> float:
        """Return the save interval."""

        return self.compile_settings.dt_save

    @property
    def dt_summarise(self) -> float:
        """Return the summary interval."""

        return self.compile_settings.dt_summarise

    @property
    def shared_memory_elements(self) -> int:
        """Return the loop's shared-memory requirement."""
        return buffer_registry.shared_buffer_size(self)

    @property
    def local_memory_elements(self) -> int:
        """Return the loop's persistent local-memory requirement."""
        return buffer_registry.local_buffer_size(self)

    @property
    def persistent_local_elements(self) -> int:
        """Return the loop's persistent local-memory requirement."""
        return buffer_registry.persistent_local_buffer_size(self)

    @property
    def compile_flags(self) -> OutputCompileFlags:
        """Return the output compile flags associated with the loop."""

        return self.compile_settings.compile_flags

    @property
    def device_function(self):
        """Return the compiled CUDA loop function.
        
        Returns
        -------
        callable
            Compiled CUDA device function.
        """
        return self.get_cached_output('loop_function')

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

        # Flatten nested dict values (e.g., loop_settings={'dt_save': 0.01})
        # into top-level parameters before distributing to compile settings.
        # This ensures all configuration options are recognized and updated.
        # Example: {'loop_settings': {'dt_save': 0.01}, 'other': 5}
        #       -> {'dt_save': 0.01, 'other': 5}
        updates_dict, unpacked_keys = unpack_dict_values(updates_dict)

        recognised = self.update_compile_settings(updates_dict, silent=True)

        # Update buffer locations in registry
        recognised |= buffer_registry.update(self, updates_dict, silent=True)
        self.register_buffers()

        unrecognised = set(updates_dict.keys()) - recognised
        if not silent and unrecognised:
            raise KeyError(
                f"Unrecognized parameters in update: {unrecognised}. "
                "These parameters were not updated.",
            )
        # Include unpacked dict keys in recognized set
        return recognised | unpacked_keys
