"""Buffer memory location settings for CUDA integration loops.

This module provides :class:`LoopBufferSettings`, an attrs class that
centralizes buffer sizing and memory location configuration for the
IVP integration loop. Each buffer can be independently configured to
use either shared or local (per-thread) memory.
"""

import attrs
from attrs import validators

from cubie._utils import getype_validator


@attrs.define
class LoopBufferSettings:
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
        default='shared', validator=validators.in_(["local", "shared"])
    )
    drivers_proposal_location: str = attrs.field(
        default='shared', validator=validators.in_(["local", "shared"])
    )
    observables_location: str = attrs.field(
        default='shared', validator=validators.in_(["local", "shared"])
    )
    observables_proposal_location: str = attrs.field(
        default='shared', validator=validators.in_(["local", "shared"])
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
        default='shared', validator=validators.in_(["local", "shared"])
    )
    scratch_location: str = attrs.field(
        default='shared', validator=validators.in_(["local", "shared"])
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
            total += (2 if self.n_counters > 0 else 0)  # proposed_counters
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

    def calculate_shared_indices(self) -> "LoopSharedIndicesFromSettings":
        """Generate shared memory index slices based on location settings.

        Only buffers with location='shared' receive non-empty slices.
        Local buffers get zero-length slices (slice(0, 0)).

        Returns
        -------
        LoopSharedIndicesFromSettings
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

        # Observables buffer
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
            proposed_counters_slice = slice(ptr, ptr + 2)
            ptr += 2
        else:
            counters_slice = slice(0, 0)
            proposed_counters_slice = slice(0, 0)

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

        local_end = ptr
        scratch_slice = slice(ptr, None)

        return LoopSharedIndicesFromSettings(
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


@attrs.define
class LoopSharedIndicesFromSettings:
    """Slice container for shared memory buffer layouts from BufferSettings.

    This class mirrors the structure of LoopSharedIndices but is generated
    dynamically from LoopBufferSettings based on memory location choices.

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
