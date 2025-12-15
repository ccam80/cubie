"""Outer integration loops for running CUDA-based ODE solvers.

The :class:`IVPLoop` orchestrates an integration by coordinating device step
functions, output collectors, and adaptive controllers. The loop owns buffer
layout metadata and feeds the appropriate slices into each device call so that
compiled kernels only need to focus on algorithmic updates.

Buffer settings classes for memory allocation configuration are also defined
here, providing selective allocation between shared and local memory.
"""
from typing import Callable, Optional, Set

import attrs
from attrs import validators
import numpy as np
from numba import cuda, int32, float64, bool_

from cubie.CUDAFactory import CUDAFactory, CUDAFunctionCache
from cubie.cuda_simsafe import from_dtype as simsafe_dtype
from cubie.cuda_simsafe import activemask, all_sync, compile_kwargs, selp
from cubie._utils import getype_validator, PrecisionDType, unpack_dict_values
from cubie.BufferSettings import BufferSettings, LocalSizes, SliceIndices
from cubie.integrators.loops.ode_loop_config import (LoopLocalIndices,
                                                     ODELoopConfig)
from cubie.outputhandling import OutputCompileFlags


@attrs.define
class LoopLocalSizes(LocalSizes):
    """Local array sizes for loop buffers with nonzero guarantees.

    Attributes
    ----------
    state : int
        State buffer size.
    proposed_state : int
        Proposed state buffer size.
    parameters : int
        Parameters buffer size.
    drivers : int
        Drivers buffer size.
    proposed_drivers : int
        Proposed drivers buffer size.
    observables : int
        Observables buffer size.
    proposed_observables : int
        Proposed observables buffer size.
    error : int
        Error buffer size.
    counters : int
        Counters buffer size.
    proposed_counters : int
        Proposed counters buffer size.
    state_summary : int
        State summary buffer size.
    observable_summary : int
        Observable summary buffer size.
    """

    state: int = attrs.field(validator=getype_validator(int, 0))
    proposed_state: int = attrs.field(validator=getype_validator(int, 0))
    parameters: int = attrs.field(validator=getype_validator(int, 0))
    drivers: int = attrs.field(validator=getype_validator(int, 0))
    proposed_drivers: int = attrs.field(validator=getype_validator(int, 0))
    observables: int = attrs.field(validator=getype_validator(int, 0))
    proposed_observables: int = attrs.field(validator=getype_validator(int, 0))
    error: int = attrs.field(validator=getype_validator(int, 0))
    counters: int = attrs.field(validator=getype_validator(int, 0))
    proposed_counters: int = attrs.field(validator=getype_validator(int, 0))
    state_summary: int = attrs.field(validator=getype_validator(int, 0))
    observable_summary: int = attrs.field(validator=getype_validator(int, 0))


@attrs.define
class LoopSliceIndices(SliceIndices):
    """Slice container for shared memory buffer layouts.

    Attributes
    ----------
    state : slice
        Slice covering the primary state buffer (empty if local).
    proposed_state : slice
        Slice covering the proposed state buffer.
    observables : slice
        Slice covering observable work buffers.
    proposed_observables : slice
        Slice covering the proposed observable buffer.
    parameters : slice
        Slice covering parameter storage.
    drivers : slice
        Slice covering driver storage.
    proposed_drivers : slice
        Slice covering the proposed driver storage.
    state_summaries : slice
        Slice covering aggregated state summaries.
    observable_summaries : slice
        Slice covering aggregated observable summaries.
    error : slice
        Slice covering the shared error buffer.
    counters : slice
        Slice covering the iteration counters buffer.
    proposed_counters : slice
        Slice covering the proposed iteration counters buffer.
    local_end : int
        Offset of the end of loop-managed shared memory.
    scratch : slice
        Slice covering any remaining shared-memory scratch space.
    all : slice
        Slice that spans the full shared-memory buffer.
    """

    state: slice = attrs.field()
    proposed_state: slice = attrs.field()
    observables: slice = attrs.field()
    proposed_observables: slice = attrs.field()
    parameters: slice = attrs.field()
    drivers: slice = attrs.field()
    proposed_drivers: slice = attrs.field()
    state_summaries: slice = attrs.field()
    observable_summaries: slice = attrs.field()
    error: slice = attrs.field()
    counters: slice = attrs.field()
    proposed_counters: slice = attrs.field()
    local_end: int = attrs.field()
    scratch: slice = attrs.field()
    all: slice = attrs.field()

    @property
    def loop_shared_elements(self) -> int:
        """Return the number of shared memory elements used by loop."""
        return self.local_end

    @property
    def n_states(self) -> int:
        """Return the number of states (from state slice width)."""
        return self.state.stop - self.state.start

    @property
    def n_parameters(self) -> int:
        """Return the number of parameters (from parameters slice width)."""
        return self.parameters.stop - self.parameters.start

    @property
    def n_drivers(self) -> int:
        """Return the number of drivers (from drivers slice width)."""
        return self.drivers.stop - self.drivers.start

    @property
    def n_observables(self) -> int:
        """Return the number of observables (from observables slice width)."""
        return self.observables.stop - self.observables.start

    @property
    def n_counters(self) -> int:
        """Return the number of counter elements (from counters slice)."""
        return self.counters.stop - self.counters.start


@attrs.define
class LoopBufferSettings(BufferSettings):
    """Configuration for loop buffer sizes and memory locations.

    Each buffer has a size attribute (number of elements) and a location
    attribute ('local' or 'shared'). The class provides computed properties
    for boolean flags, shared memory indices, and total memory requirements.

    Attributes
    ----------
    n_states : int
        Number of state variables.
    n_parameters : int
        Number of parameters.
    n_drivers : int
        Number of driver variables.
    n_observables : int
        Number of observable variables.
    state_summary_buffer_height : int
        Height of state summary buffer (0 if disabled).
    observable_summary_buffer_height : int
        Height of observable summary buffer (0 if disabled).
    n_counters : int
        Number of counter elements (typically 4 when enabled, 0 when disabled).
    n_error : int
        Number of error elements (typically equals n_states for adaptive).
    state_buffer_location : str
        Memory location for state buffer: 'local' or 'shared'.
    state_proposal_location : str
        Memory location for proposed state buffer.
    parameters_location : str
        Memory location for parameters buffer.
    drivers_location : str
        Memory location for drivers buffer.
    drivers_proposal_location : str
        Memory location for proposed drivers buffer.
    observables_location : str
        Memory location for observables buffer.
    observables_proposal_location : str
        Memory location for proposed observables buffer.
    error_location : str
        Memory location for error buffer.
    counters_location : str
        Memory location for counters buffer.
    state_summary_location : str
        Memory location for state summary buffer.
    observable_summary_location : str
        Memory location for observable summary buffer.
    scratch_location : str
        Memory location for algorithm scratch buffer.
    """

    # Size attributes
    n_states: int = attrs.field(validator=getype_validator(int, 0))
    n_parameters: int = attrs.field(
        default=0, validator=getype_validator(int, 0)
    )
    n_drivers: int = attrs.field(
        default=0, validator=getype_validator(int, 0)
    )
    n_observables: int = attrs.field(
        default=0, validator=getype_validator(int, 0)
    )
    state_summary_buffer_height: int = attrs.field(
        default=0, validator=getype_validator(int, 0)
    )
    observable_summary_buffer_height: int = attrs.field(
        default=0, validator=getype_validator(int, 0)
    )
    n_counters: int = attrs.field(
        default=0, validator=getype_validator(int, 0)
    )
    n_error: int = attrs.field(
        default=0, validator=getype_validator(int, 0)
    )

    # Location attributes with defaults matching all_in_one.py
    state_buffer_location: str = attrs.field(
        default='local', validator=validators.in_(["local", "shared"])
    )
    state_proposal_location: str = attrs.field(
        default='local', validator=validators.in_(["local", "shared"])
    )
    parameters_location: str = attrs.field(
        default='local', validator=validators.in_(["local", "shared"])
    )
    drivers_location: str = attrs.field(
        default='local', validator=validators.in_(["local", "shared"])
    )
    drivers_proposal_location: str = attrs.field(
        default='local', validator=validators.in_(["local", "shared"])
    )
    observables_location: str = attrs.field(
        default='local', validator=validators.in_(["local", "shared"])
    )
    observables_proposal_location: str = attrs.field(
        default='local', validator=validators.in_(["local", "shared"])
    )
    error_location: str = attrs.field(
        default='local', validator=validators.in_(["local", "shared"])
    )
    counters_location: str = attrs.field(
        default='local', validator=validators.in_(["local", "shared"])
    )
    state_summary_location: str = attrs.field(
        default='local', validator=validators.in_(["local", "shared"])
    )
    observable_summary_location: str = attrs.field(
        default='local', validator=validators.in_(["local", "shared"])
    )
    scratch_location: str = attrs.field(
        default='local', validator=validators.in_(["local", "shared"])
    )

    # Boolean properties for compile-time flags
    @property
    def use_shared_state(self) -> bool:
        """Return True if state buffer uses shared memory."""
        return self.state_buffer_location == 'shared'

    @property
    def use_shared_state_proposal(self) -> bool:
        """Return True if proposed state buffer uses shared memory."""
        return self.state_proposal_location == 'shared'

    @property
    def use_shared_parameters(self) -> bool:
        """Return True if parameters buffer uses shared memory."""
        return self.parameters_location == 'shared'

    @property
    def use_shared_drivers(self) -> bool:
        """Return True if drivers buffer uses shared memory."""
        return self.drivers_location == 'shared'

    @property
    def use_shared_drivers_proposal(self) -> bool:
        """Return True if proposed drivers buffer uses shared memory."""
        return self.drivers_proposal_location == 'shared'

    @property
    def use_shared_observables(self) -> bool:
        """Return True if observables buffer uses shared memory."""
        return self.observables_location == 'shared'

    @property
    def use_shared_observables_proposal(self) -> bool:
        """Return True if proposed observables buffer uses shared memory."""
        return self.observables_proposal_location == 'shared'

    @property
    def use_shared_error(self) -> bool:
        """Return True if error buffer uses shared memory."""
        return self.error_location == 'shared'

    @property
    def use_shared_counters(self) -> bool:
        """Return True if counters buffer uses shared memory."""
        return self.counters_location == 'shared'

    @property
    def use_shared_state_summary(self) -> bool:
        """Return True if state summary buffer uses shared memory."""
        return self.state_summary_location == 'shared'

    @property
    def use_shared_observable_summary(self) -> bool:
        """Return True if observable summary buffer uses shared memory."""
        return self.observable_summary_location == 'shared'

    @property
    def use_shared_scratch(self) -> bool:
        """Return True if scratch buffer uses shared memory."""
        return self.scratch_location == 'shared'

    # Computed size properties (return at least 1 for local arrays)
    @property
    def state_buffer_size(self) -> int:
        """Return state buffer size, minimum 1 for local arrays."""
        return max(1, self.n_states)

    @property
    def proposed_state_size(self) -> int:
        """Return proposed state buffer size, minimum 1 for local arrays."""
        return max(1, self.n_states)

    @property
    def parameters_size(self) -> int:
        """Return parameters buffer size, minimum 1 for local arrays."""
        return max(1, self.n_parameters)

    @property
    def drivers_size(self) -> int:
        """Return drivers buffer size, minimum 1 for local arrays."""
        return max(1, self.n_drivers)

    @property
    def proposed_drivers_size(self) -> int:
        """Return proposed drivers buffer size, minimum 1 for local arrays."""
        return max(1, self.n_drivers)

    @property
    def observables_size(self) -> int:
        """Return observables buffer size, minimum 1 for local arrays."""
        return max(1, self.n_observables)

    @property
    def proposed_observables_size(self) -> int:
        """Return proposed observables buffer size, minimum 1."""
        return max(1, self.n_observables)

    @property
    def error_size(self) -> int:
        """Return error buffer size, minimum 1 for local arrays."""
        return max(1, self.n_error)

    @property
    def counters_size(self) -> int:
        """Return counters buffer size, minimum 1 for local arrays."""
        return max(1, self.n_counters)

    @property
    def proposed_counters_size(self) -> int:
        """Return proposed counters buffer size (2 if active, else 1)."""
        return 2 if self.n_counters > 0 else 1

    @property
    def state_summary_size(self) -> int:
        """Return state summary buffer size, minimum 1 for local arrays."""
        return max(1, self.state_summary_buffer_height)

    @property
    def observable_summary_size(self) -> int:
        """Return observable summary buffer size, minimum 1."""
        return max(1, self.observable_summary_buffer_height)

    @property
    def shared_memory_elements(self) -> int:
        """Return total shared memory elements required by loop buffers.

        Only buffers configured with location='shared' contribute to
        this total.
        """
        total = 0
        if self.use_shared_state:
            total += self.n_states
        if self.use_shared_state_proposal:
            total += self.n_states
        if self.use_shared_parameters:
            total += self.n_parameters
        if self.use_shared_drivers:
            total += self.n_drivers
        if self.use_shared_drivers_proposal:
            total += self.n_drivers
        if self.use_shared_observables:
            total += self.n_observables
        if self.use_shared_observables_proposal:
            total += self.n_observables
        if self.use_shared_error:
            total += self.n_error
        if self.use_shared_counters:
            total += self.n_counters
            if self.n_counters > 0:
                total += 2  # proposed_counters (newton, krylov iters)
        if self.use_shared_state_summary:
            total += self.state_summary_buffer_height
        if self.use_shared_observable_summary:
            total += self.observable_summary_buffer_height
        return total

    @property
    def local_memory_elements(self) -> int:
        """Return total local memory elements required by loop buffers.

        Only buffers configured with location='local' contribute to
        this total.
        """
        total = 0
        if not self.use_shared_state:
            total += self.state_buffer_size
        if not self.use_shared_state_proposal:
            total += self.proposed_state_size
        if not self.use_shared_parameters:
            total += self.parameters_size
        if not self.use_shared_drivers:
            total += self.drivers_size
        if not self.use_shared_drivers_proposal:
            total += self.proposed_drivers_size
        if not self.use_shared_observables:
            total += self.observables_size
        if not self.use_shared_observables_proposal:
            total += self.proposed_observables_size
        if not self.use_shared_error:
            total += self.error_size
        if not self.use_shared_counters:
            total += self.counters_size
        if not self.use_shared_state_summary:
            total += self.state_summary_size
        if not self.use_shared_observable_summary:
            total += self.observable_summary_size
        return total

    @property
    def total_shared_elements(self) -> int:
        """Alias for shared_memory_elements for naming consistency."""
        return self.shared_memory_elements

    @property
    def total_local_elements(self) -> int:
        """Alias for local_memory_elements for naming consistency."""
        return self.local_memory_elements

    def calculate_shared_indices(self) -> "LoopSliceIndices":
        """Generate shared memory index slices based on location settings.

        Only buffers with location='shared' receive non-empty slices.
        Local buffers get zero-length slices (slice(0, 0)).

        The order matches the original LoopSharedIndices.from_sizes layout:
        state, proposed_state, observables, proposed_observables, parameters,
        drivers, proposed_drivers, state_summaries, observable_summaries,
        error, counters, proposed_counters.

        Returns
        -------
        LoopSliceIndices
            Object containing slice objects for each buffer.
        """
        ptr = 0

        # State buffer
        if self.use_shared_state:
            state_slice = slice(ptr, ptr + self.n_states)
            ptr += self.n_states
        else:
            state_slice = slice(0, 0)

        # Proposed state buffer
        if self.use_shared_state_proposal:
            proposed_state_slice = slice(ptr, ptr + self.n_states)
            ptr += self.n_states
        else:
            proposed_state_slice = slice(0, 0)

        # Observables buffer (order matches from_sizes)
        if self.use_shared_observables:
            observables_slice = slice(ptr, ptr + self.n_observables)
            ptr += self.n_observables
        else:
            observables_slice = slice(0, 0)

        # Proposed observables buffer
        if self.use_shared_observables_proposal:
            proposed_observables_slice = slice(ptr, ptr + self.n_observables)
            ptr += self.n_observables
        else:
            proposed_observables_slice = slice(0, 0)

        # Parameters buffer
        if self.use_shared_parameters:
            parameters_slice = slice(ptr, ptr + self.n_parameters)
            ptr += self.n_parameters
        else:
            parameters_slice = slice(0, 0)

        # Drivers buffer
        if self.use_shared_drivers:
            drivers_slice = slice(ptr, ptr + self.n_drivers)
            ptr += self.n_drivers
        else:
            drivers_slice = slice(0, 0)

        # Proposed drivers buffer
        if self.use_shared_drivers_proposal:
            proposed_drivers_slice = slice(ptr, ptr + self.n_drivers)
            ptr += self.n_drivers
        else:
            proposed_drivers_slice = slice(0, 0)

        # State summary buffer
        if self.use_shared_state_summary:
            state_summaries_slice = slice(
                ptr, ptr + self.state_summary_buffer_height
            )
            ptr += self.state_summary_buffer_height
        else:
            state_summaries_slice = slice(0, 0)

        # Observable summary buffer
        if self.use_shared_observable_summary:
            observable_summaries_slice = slice(
                ptr, ptr + self.observable_summary_buffer_height
            )
            ptr += self.observable_summary_buffer_height
        else:
            observable_summaries_slice = slice(0, 0)

        # Error buffer
        if self.use_shared_error:
            error_slice = slice(ptr, ptr + self.n_error)
            ptr += self.n_error
        else:
            error_slice = slice(0, 0)

        # Counters buffer
        if self.use_shared_counters:
            counters_slice = slice(ptr, ptr + self.n_counters)
            ptr += self.n_counters
            # proposed_counters: 2 elements (newton, krylov iters from step)
            proposed_counters_len = 2 if self.n_counters > 0 else 0
            proposed_counters_slice = slice(ptr, ptr + proposed_counters_len)
            ptr += proposed_counters_len
        else:
            counters_slice = slice(0, 0)
            proposed_counters_slice = slice(0, 0)

        local_end = ptr
        scratch_slice = slice(ptr, None)

        return LoopSliceIndices(
            state=state_slice,
            proposed_state=proposed_state_slice,
            observables=observables_slice,
            proposed_observables=proposed_observables_slice,
            parameters=parameters_slice,
            drivers=drivers_slice,
            proposed_drivers=proposed_drivers_slice,
            state_summaries=state_summaries_slice,
            observable_summaries=observable_summaries_slice,
            error=error_slice,
            counters=counters_slice,
            proposed_counters=proposed_counters_slice,
            local_end=local_end,
            scratch=scratch_slice,
            all=slice(None),
        )

    @property
    def local_sizes(self) -> LoopLocalSizes:
        """Return LoopLocalSizes instance with buffer sizes.

        The returned object provides nonzero sizes suitable for
        cuda.local.array allocation.
        """
        return LoopLocalSizes(
            state=self.n_states,
            proposed_state=self.n_states,
            parameters=self.n_parameters,
            drivers=self.n_drivers,
            proposed_drivers=self.n_drivers,
            observables=self.n_observables,
            proposed_observables=self.n_observables,
            error=self.n_error,
            counters=self.n_counters,
            proposed_counters=2 if self.n_counters > 0 else 0,
            state_summary=self.state_summary_buffer_height,
            observable_summary=self.observable_summary_buffer_height,
        )

    @property
    def shared_indices(self) -> LoopSliceIndices:
        """Return LoopSliceIndices instance with shared memory layout.

        The returned object contains slices for each buffer's region
        in shared memory. Local buffers receive empty slices.
        """
        return self.calculate_shared_indices()


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
}

# Buffer location parameters that can be specified at Solver level.
# These parameters control whether specific buffers are allocated in
# shared or local memory within CUDA device functions.
ALL_BUFFER_LOCATION_PARAMETERS = {
    "state_buffer_location",
    "state_proposal_location",
    "parameters_location",
    "drivers_location",
    "drivers_proposal_location",
    "observables_location",
    "observables_proposal_location",
    "error_location",
    "counters_location",
    "state_summary_location",
    "observable_summary_location",
    "scratch_location",
}


class IVPLoop(CUDAFactory):
    """Factory for CUDA device loops that advance an IVP integration.

    Parameters
    ----------
    precision
        Precision used for state and observable updates.
    buffer_settings
        Configuration for loop buffer sizes and memory locations.
    compile_flags
        Output configuration that drives save and summary behaviour.
    controller_local_len
        Number of persistent local memory elements for the controller.
    algorithm_local_len
        Number of persistent local memory elements for the algorithm.
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
        buffer_settings: LoopBufferSettings,
        compile_flags: OutputCompileFlags,
        controller_local_len: int = 0,
        algorithm_local_len: int = 0,
        dt_save: float = 0.1,
        dt_summarise: float = 1.0,
        dt0: Optional[float] = None,
        dt_min: Optional[float] = None,
        dt_max: Optional[float] = None,
        is_adaptive: Optional[bool] = None,
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
            buffer_settings=buffer_settings,
            controller_local_len=controller_local_len,
            algorithm_local_len=algorithm_local_len,
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
        save_counters_bool = flags.save_counters

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
        counters_shared_ind = shared_indices.counters
        proposed_counters_shared_ind = shared_indices.proposed_counters
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
        # save_last is not yet piped up from this level, but is intended and
        # included in loop logic
        save_last = False

        buffer_settings = config.buffer_settings

        # Loop sizes - computed from slice dimensions
        n_states = int32(buffer_settings.n_states)
        n_parameters = int32(buffer_settings.n_parameters)
        n_observables = int32(buffer_settings.n_observables)
        n_drivers = int32(buffer_settings.n_drivers)
        n_counters = int32(buffer_settings.n_counters)
        
        fixed_mode = not config.is_adaptive
        status_mask = int32(0xFFFF)

        # Buffer settings from compile_settings for selective shared/local.
        # IVPLoop.__init__ ensures buffer_settings is always set.

        # Unpack boolean flags as compile-time constants
        state_shared = buffer_settings.use_shared_state
        state_proposal_shared = buffer_settings.use_shared_state_proposal
        parameters_shared = buffer_settings.use_shared_parameters
        drivers_shared = buffer_settings.use_shared_drivers
        drivers_proposal_shared = buffer_settings.use_shared_drivers_proposal
        observables_shared = buffer_settings.use_shared_observables
        observables_proposal_shared = (
            buffer_settings.use_shared_observables_proposal
        )
        error_shared = buffer_settings.use_shared_error
        counters_shared = buffer_settings.use_shared_counters
        state_summary_shared = buffer_settings.use_shared_state_summary
        observable_summary_shared = buffer_settings.use_shared_observable_summary

        # Unpack local sizes for local array allocation
        local_sizes = buffer_settings.local_sizes
        state_local_size = local_sizes.nonzero('state')
        proposed_state_local_size = local_sizes.nonzero('proposed_state')
        parameters_local_size = local_sizes.nonzero('parameters')
        drivers_local_size = local_sizes.nonzero('drivers')
        proposed_drivers_local_size = local_sizes.nonzero('proposed_drivers')
        observables_local_size = local_sizes.nonzero('observables')
        proposed_observables_local_size = local_sizes.nonzero(
            'proposed_observables'
        )
        error_local_size = local_sizes.nonzero('error')
        counters_local_size = local_sizes.nonzero('counters')
        state_summary_local_size = local_sizes.nonzero('state_summary')
        observable_summary_local_size = local_sizes.nonzero(
            'observable_summary'
        )


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

            # ----------------------------------------------------------- #
            # Selective allocation from local or shared memory
            # ----------------------------------------------------------- #
            if state_shared:
                state_buffer = shared_scratch[state_shared_ind]
            else:
                state_buffer = cuda.local.array(state_local_size, precision)
                for _i in range(state_local_size):
                    state_buffer[_i] = precision(0.0)

            if state_proposal_shared:
                state_proposal_buffer = shared_scratch[state_prop_shared_ind]
            else:
                state_proposal_buffer = cuda.local.array(
                    proposed_state_local_size, precision
                )
                for _i in range(proposed_state_local_size):
                    state_proposal_buffer[_i] = precision(0.0)

            if observables_shared:
                observables_buffer = shared_scratch[obs_shared_ind]
            else:
                observables_buffer = cuda.local.array(
                    observables_local_size, precision
                )
                for _i in range(observables_local_size):
                    observables_buffer[_i] = precision(0.0)

            if observables_proposal_shared:
                observables_proposal_buffer = shared_scratch[obs_prop_shared_ind]
            else:
                observables_proposal_buffer = cuda.local.array(
                    proposed_observables_local_size, precision
                )
                for _i in range(proposed_observables_local_size):
                    observables_proposal_buffer[_i] = precision(0.0)

            if parameters_shared:
                parameters_buffer = shared_scratch[params_shared_ind]
            else:
                parameters_buffer = cuda.local.array(
                    parameters_local_size, precision
                )
                for _i in range(parameters_local_size):
                    parameters_buffer[_i] = precision(0.0)

            if drivers_shared:
                drivers_buffer = shared_scratch[drivers_shared_ind]
            else:
                drivers_buffer = cuda.local.array(drivers_local_size, precision)
                for _i in range(drivers_local_size):
                    drivers_buffer[_i] = precision(0.0)

            if drivers_proposal_shared:
                drivers_proposal_buffer = shared_scratch[drivers_prop_shared_ind]
            else:
                drivers_proposal_buffer = cuda.local.array(
                    proposed_drivers_local_size, precision
                )
                for _i in range(proposed_drivers_local_size):
                    drivers_proposal_buffer[_i] = precision(0.0)

            if state_summary_shared:
                state_summary_buffer = shared_scratch[state_summ_shared_ind]
            else:
                state_summary_buffer = cuda.local.array(
                    state_summary_local_size, precision
                )
                for _i in range(state_summary_local_size):
                    state_summary_buffer[_i] = precision(0.0)

            if observable_summary_shared:
                observable_summary_buffer = shared_scratch[obs_summ_shared_ind]
            else:
                observable_summary_buffer = cuda.local.array(
                    observable_summary_local_size, precision
                )
                for _i in range(observable_summary_local_size):
                    observable_summary_buffer[_i] = precision(0.0)

            if counters_shared:
                counters_since_save = shared_scratch[counters_shared_ind]
            else:
                counters_since_save = cuda.local.array(
                    counters_local_size, simsafe_int32
                )
                for _i in range(counters_local_size):
                    counters_since_save[_i] = simsafe_int32(0)

            if error_shared:
                error = shared_scratch[error_shared_ind]
            else:
                error = cuda.local.array(error_local_size, precision)
                for _i in range(error_local_size):
                    error[_i] = precision(0.0)

            remaining_shared_scratch = shared_scratch[remaining_scratch_ind]
            # ----------------------------------------------------------- #

            if save_counters_bool:
                # When enabled, use shared memory buffers
                proposed_counters = shared_scratch[proposed_counters_shared_ind]
            else:
                # When disabled, use a dummy local "proposed_counters" buffer
                proposed_counters = cuda.local.array(2, dtype=simsafe_int32)
                for _i in range(2):
                    proposed_counters[_i] = int32(0)

            dt = persistent_local[dt_slice]
            accept_step = persistent_local[accept_slice].view(simsafe_int32)
            controller_temp = persistent_local[controller_slice]
            algo_local = persistent_local[algorithm_slice]

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
                    # Summary accumulation starts with next accepted save to
                    # avoid double counting the t0 seed sample.
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

                    step_status = int32(step_function(
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
                        remaining_shared_scratch,
                        algo_local,
                        proposed_counters,
                    ))

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
                            controller_temp,
                        )

                        accept = accept_step[0] != int32(0)
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
                                    state_summaries_output[
                                        summary_idx * summarise_state_bool, :
                                    ],
                                    observable_summaries_output[
                                        summary_idx * summarise_obs_bool, :
                                    ],
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
            (32768//8), # local persistent - not really used
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
            None, # local persistent - not really used
            None,  # persistent_local - arbitrary 32kb provided / float64
            None, # state_output
            None, # observables_output
            None,  # state_summaries_output
            None, # obs summ output
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
    def shared_buffer_indices(self) -> LoopSliceIndices:
        """Return the shared buffer index layout."""

        return self.compile_settings.shared_buffer_indices

    @property
    def buffer_indices(self) -> LoopSliceIndices:
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
        unrecognised = set(updates_dict.keys()) - recognised
        if not silent and unrecognised:
            raise KeyError(
                f"Unrecognized parameters in update: {unrecognised}. "
                "These parameters were not updated.",
            )
        # Include unpacked dict keys in recognized set
        return recognised | unpacked_keys
