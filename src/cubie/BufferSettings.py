"""DEPRECATED: Buffer settings base classes are now defined locally in each module.

This module is maintained for backwards compatibility only. The base classes
LocalSizes, SliceIndices, and BufferSettings have been moved to local
definitions in the modules that use them.

For new code, import directly from:
- cubie.integrators.algorithms.generic_dirk (DIRKBufferSettings, etc.)
- cubie.integrators.algorithms.generic_erk (ERKBufferSettings, etc.)
- cubie.integrators.algorithms.generic_firk (FIRKBufferSettings, etc.)
- cubie.integrators.algorithms.generic_rosenbrock_w (RosenbrockBufferSettings)
- cubie.integrators.loops.ode_loop (LoopBufferSettings)
- cubie.integrators.matrix_free_solvers.linear_solver (LinearSolverBufferSettings)
- cubie.integrators.matrix_free_solvers.newton_krylov (NewtonBufferSettings)

The buffer_registry module provides a central registry for all buffer
management. See cubie.buffer_registry for the new API.
"""

from abc import ABC, abstractmethod


class LocalSizes:
    """Base class for local array sizes with nonzero guarantees."""

    def nonzero(self, attr_name: str) -> int:
        """Return max(value, 1) for cuda.local.array compatibility."""
        return max(getattr(self, attr_name), 1)


class SliceIndices:
    """Base class for shared memory slice indices."""
    pass


class BufferSettings(ABC):
    """Abstract base class for buffer memory configuration."""

    @property
    @abstractmethod
    def shared_memory_elements(self) -> int:
        """Return total shared memory elements required."""
        pass

    @property
    @abstractmethod
    def local_memory_elements(self) -> int:
        """Return total local memory elements required."""
        pass

    @property
    @abstractmethod
    def local_sizes(self) -> LocalSizes:
        """Return LocalSizes instance with buffer sizes."""
        pass

    @property
    @abstractmethod
    def shared_indices(self) -> SliceIndices:
        """Return SliceIndices instance with shared memory layout."""
        pass
