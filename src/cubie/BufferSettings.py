"""Base buffer settings infrastructure for CUDA memory management.

This module provides base classes for buffer settings across the cubie
library. Each subsystem (loops, algorithms, linear solvers) extends these
base classes to define their specific buffer requirements.

The architecture separates:
- Buffer sizes and memory locations (BufferSettings subclasses)
- Local array sizes with nonzero guarantees (LocalSizes subclasses)
- Shared memory slice indices (SliceIndices subclasses)
"""

from abc import ABC, abstractmethod

import attrs


@attrs.define
class LocalSizes(ABC):
    """Base class for local array sizes with nonzero guarantees.

    Subclasses define size attributes for buffers that may be allocated
    in local memory. The nonzero method provides a size suitable for
    cuda.local.array which requires size >= 1.

    Methods
    -------
    nonzero(attr_name)
        Return max(attribute_value, 1) for the named attribute.
    """

    def nonzero(self, attr_name: str) -> int:
        """Return the attribute value with minimum 1 for local arrays.

        Parameters
        ----------
        attr_name : str
            Name of the attribute to retrieve.

        Returns
        -------
        int
            max(attribute_value, 1) ensuring valid size for local arrays.
        """
        value = getattr(self, attr_name)
        return max(value, 1)


@attrs.define
class SliceIndices(ABC):
    """Base class for shared memory slice indices.

    Subclasses define slice attributes for partitioning shared memory
    among buffers. Each slice indicates the region of shared memory
    allocated to a specific buffer.

    Notes
    -----
    Buffers configured for local memory receive empty slices (slice(0, 0)).
    """
    pass


class BufferSettings(ABC):
    """Abstract base class for buffer memory configuration.

    BufferSettings subclasses centralize buffer sizing and memory location
    configuration for a subsystem. Each buffer can be independently
    configured to use either shared or local (per-thread) memory.

    Subclasses must implement:
    - shared_memory_elements: total shared memory elements required
    - local_memory_elements: total local memory elements required
    - local_sizes: LocalSizes instance with buffer sizes
    - shared_indices: SliceIndices instance with memory layout
    """

    @property
    @abstractmethod
    def shared_memory_elements(self) -> int:
        """Return total shared memory elements required.

        Only buffers configured with location='shared' contribute.
        """
        pass

    @property
    @abstractmethod
    def local_memory_elements(self) -> int:
        """Return total local memory elements required.

        Only buffers configured with location='local' contribute.
        """
        pass

    @property
    @abstractmethod
    def local_sizes(self) -> LocalSizes:
        """Return LocalSizes instance with buffer sizes.

        The returned object provides nonzero sizes suitable for
        cuda.local.array allocation.
        """
        pass

    @property
    @abstractmethod
    def shared_indices(self) -> SliceIndices:
        """Return SliceIndices instance with shared memory layout.

        The returned object contains slices for each buffer's region
        in shared memory. Local buffers receive empty slices.
        """
        pass

    @property
    @abstractmethod
    def persistent_local_elements(self) -> int:
        """Return persistent local memory elements required.

        Persistent local memory survives between step invocations,
        used for FSAL (First Same As Last) caching optimization.
        """
        pass
