"""Buffer memory location settings for algorithm step functions.

This module provides :class:`ERKBufferSettings` and :class:`DIRKBufferSettings`,
attrs classes that configure memory locations for algorithm-specific buffers
used during Runge-Kutta integration steps.
"""

import attrs
from attrs import validators

from cubie._utils import getype_validator


@attrs.define
class ERKBufferSettings:
    """Configuration for ERK step buffer sizes and memory locations.

    Controls whether stage_rhs and stage_accumulator buffers use shared
    or local memory. Also manages stage_cache aliasing logic for FSAL
    optimization.

    Attributes
    ----------
    n : int
        Number of state variables.
    stage_count : int
        Number of RK stages.
    stage_rhs_location : str
        Memory location for stage RHS buffer: 'local' or 'shared'.
    stage_accumulator_location : str
        Memory location for stage accumulator buffer: 'local' or 'shared'.
    """

    n: int = attrs.field(validator=getype_validator(int, 1))
    stage_count: int = attrs.field(validator=getype_validator(int, 1))
    stage_rhs_location: str = attrs.field(
        default='local', validator=validators.in_(["local", "shared"])
    )
    stage_accumulator_location: str = attrs.field(
        default='local', validator=validators.in_(["local", "shared"])
    )

    @property
    def use_shared_stage_rhs(self) -> bool:
        """Return True if stage_rhs buffer uses shared memory."""
        return self.stage_rhs_location == 'shared'

    @property
    def use_shared_stage_accumulator(self) -> bool:
        """Return True if stage_accumulator buffer uses shared memory."""
        return self.stage_accumulator_location == 'shared'

    @property
    def use_shared_stage_cache(self) -> bool:
        """Return True if stage_cache should use shared memory.

        stage_cache is shared if either stage_rhs or stage_accumulator
        is shared (it aliases onto one of them).
        """
        return self.use_shared_stage_rhs or self.use_shared_stage_accumulator

    @property
    def stage_cache_aliases_rhs(self) -> bool:
        """Return True if stage_cache aliases stage_rhs.

        stage_cache aliases stage_rhs when stage_rhs is in shared memory.
        """
        return self.use_shared_stage_rhs

    @property
    def stage_cache_aliases_accumulator(self) -> bool:
        """Return True if stage_cache aliases stage_accumulator.

        stage_cache aliases accumulator when stage_rhs is local but
        accumulator is shared.
        """
        return (not self.use_shared_stage_rhs
                and self.use_shared_stage_accumulator)

    @property
    def accumulator_length(self) -> int:
        """Return the length of the stage accumulator buffer."""
        return max(self.stage_count - 1, 0) * self.n

    @property
    def shared_memory_elements(self) -> int:
        """Return total shared memory elements required.

        Includes stage_rhs (n) if shared, and accumulator if shared.
        """
        total = 0
        if self.use_shared_stage_rhs:
            total += self.n
        if self.use_shared_stage_accumulator:
            total += self.accumulator_length
        return total

    @property
    def local_memory_elements(self) -> int:
        """Return total local memory elements required.

        Includes stage_rhs (n) if local, accumulator if local,
        plus persistent_local for stage_cache if not aliased.
        """
        total = 0
        if not self.use_shared_stage_rhs:
            total += self.n
        if not self.use_shared_stage_accumulator:
            total += self.accumulator_length
        # stage_cache needs persistent local if neither is shared
        if not self.use_shared_stage_cache:
            total += self.n
        return total

    @property
    def persistent_local_elements(self) -> int:
        """Return persistent local elements for stage_cache.

        Returns n if stage_cache cannot alias onto shared buffers.
        """
        if self.use_shared_stage_cache:
            return 0
        return self.n


@attrs.define
class DIRKBufferSettings:
    """Configuration for DIRK step buffer sizes and memory locations.

    Controls memory locations for stage_increment, stage_base, accumulator,
    and solver_scratch buffers used during DIRK integration steps.

    Attributes
    ----------
    n : int
        Number of state variables.
    stage_count : int
        Number of RK stages.
    stage_increment_location : str
        Memory location for stage increment buffer: 'local' or 'shared'.
    stage_base_location : str
        Memory location for stage base buffer: 'local' or 'shared'.
    accumulator_location : str
        Memory location for stage accumulator buffer: 'local' or 'shared'.
    solver_scratch_location : str
        Memory location for Newton solver scratch: 'local' or 'shared'.
    """

    n: int = attrs.field(validator=getype_validator(int, 1))
    stage_count: int = attrs.field(validator=getype_validator(int, 1))
    stage_increment_location: str = attrs.field(
        default='local', validator=validators.in_(["local", "shared"])
    )
    stage_base_location: str = attrs.field(
        default='shared', validator=validators.in_(["local", "shared"])
    )
    accumulator_location: str = attrs.field(
        default='shared', validator=validators.in_(["local", "shared"])
    )
    solver_scratch_location: str = attrs.field(
        default='shared', validator=validators.in_(["local", "shared"])
    )

    @property
    def use_shared_stage_increment(self) -> bool:
        """Return True if stage_increment buffer uses shared memory."""
        return self.stage_increment_location == 'shared'

    @property
    def use_shared_stage_base(self) -> bool:
        """Return True if stage_base buffer uses shared memory."""
        return self.stage_base_location == 'shared'

    @property
    def use_shared_accumulator(self) -> bool:
        """Return True if accumulator buffer uses shared memory."""
        return self.accumulator_location == 'shared'

    @property
    def use_shared_solver_scratch(self) -> bool:
        """Return True if solver_scratch buffer uses shared memory."""
        return self.solver_scratch_location == 'shared'

    @property
    def accumulator_length(self) -> int:
        """Return the length of the stage accumulator buffer."""
        return max(self.stage_count - 1, 0) * self.n

    @property
    def solver_scratch_elements(self) -> int:
        """Return the number of solver scratch elements (2 * n)."""
        return 2 * self.n

    @property
    def multistage(self) -> bool:
        """Return True if method has multiple stages."""
        return self.stage_count > 1

    @property
    def stage_base_aliases_accumulator(self) -> bool:
        """Return True if stage_base can alias first slice of accumulator.

        Only valid when multistage and accumulator is in shared memory.
        """
        return self.multistage and self.use_shared_accumulator

    @property
    def shared_memory_elements(self) -> int:
        """Return total shared memory elements required.

        Includes accumulator, solver_scratch, and stage_increment if shared.
        stage_base aliases accumulator when multistage, so not counted
        separately.
        """
        total = 0
        if self.use_shared_accumulator:
            total += self.accumulator_length
        if self.use_shared_solver_scratch:
            total += self.solver_scratch_elements
        if self.use_shared_stage_increment:
            total += self.n
        # stage_base aliases accumulator when multistage; only add if
        # single-stage and shared
        if not self.multistage and self.use_shared_stage_base:
            total += self.n
        return total

    @property
    def local_memory_elements(self) -> int:
        """Return total local memory elements required.

        Includes buffers configured with location='local'.
        """
        total = 0
        if not self.use_shared_accumulator:
            total += self.accumulator_length
        if not self.use_shared_solver_scratch:
            total += self.solver_scratch_elements
        if not self.use_shared_stage_increment:
            total += self.n
        # stage_base needs local storage when single-stage and local
        if not self.multistage and not self.use_shared_stage_base:
            total += self.n
        return total
