"""Buffer memory location settings for matrix-free linear solvers.

This module provides :class:`LinearSolverBufferSettings`, an attrs class
that configures memory locations for preconditioned_vec and temp buffers
used during Krylov iteration.
"""

import attrs
from attrs import validators

from cubie._utils import getype_validator


@attrs.define
class LinearSolverBufferSettings:
    """Configuration for linear solver buffer sizes and memory locations.

    Controls whether preconditioned_vec and temp buffers use shared or
    local memory during Krylov iteration.

    Attributes
    ----------
    n : int
        Number of state variables (length of vectors).
    preconditioned_vec_location : str
        Memory location for preconditioned vector: 'local' or 'shared'.
    temp_location : str
        Memory location for temporary vector: 'local' or 'shared'.
    """

    n: int = attrs.field(validator=getype_validator(int, 1))
    preconditioned_vec_location: str = attrs.field(
        default='local', validator=validators.in_(["local", "shared"])
    )
    temp_location: str = attrs.field(
        default='local', validator=validators.in_(["local", "shared"])
    )

    @property
    def use_shared_preconditioned_vec(self) -> bool:
        """Return True if preconditioned_vec uses shared memory."""
        return self.preconditioned_vec_location == 'shared'

    @property
    def use_shared_temp(self) -> bool:
        """Return True if temp buffer uses shared memory."""
        return self.temp_location == 'shared'

    @property
    def shared_memory_elements(self) -> int:
        """Return total shared memory elements required."""
        total = 0
        if self.use_shared_preconditioned_vec:
            total += self.n
        if self.use_shared_temp:
            total += self.n
        return total

    @property
    def local_memory_elements(self) -> int:
        """Return total local memory elements required."""
        total = 0
        if not self.use_shared_preconditioned_vec:
            total += self.n
        if not self.use_shared_temp:
            total += self.n
        return total
