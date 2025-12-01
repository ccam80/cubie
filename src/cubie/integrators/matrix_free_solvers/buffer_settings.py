"""Buffer memory location settings for matrix-free linear solvers.

This module provides :class:`LinearSolverBufferSettings`, an attrs class
that configures memory locations for preconditioned_vec and temp buffers
used during Krylov iteration.
"""

import attrs
from attrs import validators

from cubie._utils import getype_validator
from cubie.BufferSettings import BufferSettings, LocalSizes, SliceIndices


@attrs.define
class LinearSolverLocalSizes(LocalSizes):
    """Local array sizes for linear solver buffers with nonzero guarantees.

    Attributes
    ----------
    preconditioned_vec : int
        Preconditioned vector buffer size.
    temp : int
        Temporary vector buffer size.
    """

    preconditioned_vec: int = attrs.field(validator=getype_validator(int, 0))
    temp: int = attrs.field(validator=getype_validator(int, 0))


@attrs.define
class LinearSolverSliceIndices(SliceIndices):
    """Slice container for linear solver shared memory buffer layouts.

    Attributes
    ----------
    preconditioned_vec : slice
        Slice covering the preconditioned vector buffer (empty if local).
    temp : slice
        Slice covering the temporary vector buffer.
    local_end : int
        Offset of the end of solver-managed shared memory.
    """

    preconditioned_vec: slice = attrs.field()
    temp: slice = attrs.field()
    local_end: int = attrs.field()


@attrs.define
class LinearSolverBufferSettings(BufferSettings):
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

    @property
    def local_sizes(self) -> LinearSolverLocalSizes:
        """Return LinearSolverLocalSizes instance with buffer sizes.

        The returned object provides nonzero sizes suitable for
        cuda.local.array allocation.
        """
        return LinearSolverLocalSizes(
            preconditioned_vec=self.n,
            temp=self.n,
        )

    @property
    def shared_indices(self) -> LinearSolverSliceIndices:
        """Return LinearSolverSliceIndices instance with shared memory layout.

        The returned object contains slices for each buffer's region
        in shared memory. Local buffers receive empty slices.
        """
        ptr = 0

        if self.use_shared_preconditioned_vec:
            preconditioned_vec_slice = slice(ptr, ptr + self.n)
            ptr += self.n
        else:
            preconditioned_vec_slice = slice(0, 0)

        if self.use_shared_temp:
            temp_slice = slice(ptr, ptr + self.n)
            ptr += self.n
        else:
            temp_slice = slice(0, 0)

        return LinearSolverSliceIndices(
            preconditioned_vec=preconditioned_vec_slice,
            temp=temp_slice,
            local_end=ptr,
        )
