"""DEPRECATED: Buffer settings functionality has been replaced by buffer_registry.

The old BufferSettings, LocalSizes, and SliceIndices classes are no longer
used. The buffer_registry module provides centralized buffer management for
all CUDA factories.

For new code, use the buffer_registry API:
- cubie.buffer_registry.buffer_registry.register()
- cubie.buffer_registry.buffer_registry.get_allocator()
- cubie.buffer_registry.buffer_registry.shared_buffer_size()
- cubie.buffer_registry.buffer_registry.local_buffer_size()
- cubie.buffer_registry.buffer_registry.persistent_local_buffer_size()

The base classes are kept for backwards compatibility with any external code
that may have subclassed them.
"""

from abc import ABC, abstractmethod


class LocalSizes:
    """Base class for local array sizes with nonzero guarantees.
    
    DEPRECATED: Use buffer_registry.get_allocator() instead.
    """

    def nonzero(self, attr_name: str) -> int:
        """Return max(value, 1) for cuda.local.array compatibility."""
        return max(getattr(self, attr_name), 1)


class SliceIndices:
    """Base class for shared memory slice indices.
    
    DEPRECATED: Use buffer_registry.get_allocator() instead.
    """
    pass


class BufferSettings(ABC):
    """Abstract base class for buffer memory configuration.
    
    DEPRECATED: Use buffer_registry API instead.
    """

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
