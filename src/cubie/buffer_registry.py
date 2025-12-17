"""Centralized buffer registry for CUDA memory management.

This module provides a package-wide singleton registry that manages
buffer metadata for all CUDA factories. Factories register their
buffer requirements during initialization, and the registry provides
CUDA-compatible allocator device functions during build.

The registry uses a lazy cached build pattern - slice layouts are
computed on demand and invalidated when any buffer is modified.
"""

from typing import Callable, Dict, Optional

import attrs
from attrs import validators
import numpy as np
from numba import cuda

from cubie._utils import getype_validator, ALLOWED_PRECISIONS, precision_validator
from cubie.cuda_simsafe import compile_kwargs


@attrs.define
class BufferEntry:
    """Immutable record describing a single buffer's requirements.

    Attributes
    ----------
    name : str
        Unique buffer name within factory context.
    factory : object
        Owning factory instance (CUDAFactory or similar).
    size : int
        Buffer size in elements.
    location : str
        Memory location for the buffer: 'shared' or 'local'.
    persistent : bool
        If True and location='local', use persistent_local.
    aliases : str or None
        Name of buffer to alias (must exist in same context).
    precision : type
        NumPy precision type for the buffer.
    """

    name: str = attrs.field(validator=validators.instance_of(str))
    factory: object = attrs.field()
    size: int = attrs.field(validator=getype_validator(int, 0))
    location: str = attrs.field(
        validator=validators.in_(["shared", "local"])
    )
    persistent: bool = attrs.field(
        default=False, validator=validators.instance_of(bool)
    )
    aliases: Optional[str] = attrs.field(
        default=None,
        validator=validators.optional(validators.instance_of(str))
    )
    precision: type = attrs.field(
        default=np.float32,
        validator=precision_validator
    )

    @property
    def is_shared(self) -> bool:
        """Return True if buffer uses shared memory."""
        return self.location == 'shared'

    @property
    def is_persistent_local(self) -> bool:
        """Return True if buffer uses persistent local memory."""
        return self.location == 'local' and self.persistent

    @property
    def is_local(self) -> bool:
        """Return True if buffer uses local (non-persistent) memory."""
        return self.location == 'local' and not self.persistent


@attrs.define
class BufferContext:
    """Groups all buffer entries for a single factory.

    Attributes
    ----------
    factory : object
        Factory instance that owns this context.
    entries : Dict[str, BufferEntry]
        Registered buffers by name.
    _shared_layout : Dict[str, slice] or None
        Cached shared memory slices (None when invalid).
    _persistent_layout : Dict[str, slice] or None
        Cached persistent_local slices (None when invalid).
    _local_sizes : Dict[str, int] or None
        Cached local sizes (None when invalid).
    _alias_offsets : Dict[str, int]
        Tracks consumed space in aliased buffers.
    """

    factory: object = attrs.field()
    entries: Dict[str, BufferEntry] = attrs.field(factory=dict)
    _shared_layout: Optional[Dict[str, slice]] = attrs.field(
        default=None, init=False
    )
    _persistent_layout: Optional[Dict[str, slice]] = attrs.field(
        default=None, init=False
    )
    _local_sizes: Optional[Dict[str, int]] = attrs.field(
        default=None, init=False
    )
    _alias_offsets: Dict[str, int] = attrs.field(
        factory=dict, init=False
    )

    def invalidate_layouts(self) -> None:
        """Set all cached layouts to None."""
        self._shared_layout = None
        self._persistent_layout = None
        self._local_sizes = None
        self._alias_offsets.clear()


@attrs.define
class BufferRegistry:
    """Central registry managing all buffer metadata for CUDA factories.

    The registry is a package-level singleton that tracks buffer
    requirements for all factories. It uses lazy cached builds for
    slice/layout computation - layouts are set to None on any change
    and regenerated on access.

    Attributes
    ----------
    _contexts : Dict[object, BufferContext]
        Maps factory instances to their buffer contexts.
    """

    _contexts: Dict[object, BufferContext] = attrs.field(
        factory=dict, init=False
    )

    def register(
        self,
        name: str,
        factory: object,
        size: int,
        location: str,
        persistent: bool = False,
        aliases: Optional[str] = None,
        precision: type = np.float32,
    ) -> None:
        """Register a buffer with the central registry.

        Parameters
        ----------
        name
            Unique buffer name within factory context.
        factory
            Factory instance that owns this buffer.
        size
            Buffer size in elements.
        location
            Memory location: 'shared' or 'local'.
        persistent
            If True and location='local', use persistent_local.
        aliases
            Name of buffer to alias (must exist in same context).
        precision
            NumPy precision type for the buffer.

        Raises
        ------
        ValueError
            If buffer name already registered for this factory.
        ValueError
            If aliases references non-existent buffer.
        ValueError
            If name is empty string.
        ValueError
            If buffer attempts to alias itself.
        ValueError
            If aliasing a shared buffer with non-shared location.
        ValueError
            If persistent buffer attempts to alias non-persistent local.
        ValueError
            If precision is not a supported NumPy floating-point type.
        """
        if not name:
            raise ValueError("Buffer name cannot be empty.")
        if aliases is not None and aliases == name:
            raise ValueError(
                f"Buffer '{name}' cannot alias itself."
            )

        # Get or create context
        if factory not in self._contexts:
            self._contexts[factory] = BufferContext(factory=factory)
        context = self._contexts[factory]

        # Check for duplicate
        if name in context.entries:
            raise ValueError(
                f"Buffer '{name}' already registered for this factory."
            )

        # Validate alias target exists
        if aliases is not None and aliases not in context.entries:
            raise ValueError(
                f"Alias target '{aliases}' not registered. "
                f"Register '{aliases}' before '{name}'."
            )

        # Validate cross-type aliasing constraints
        if aliases is not None:
            parent_entry = context.entries[aliases]
            # If parent is shared, child must also be shared
            if parent_entry.is_shared and location != 'shared':
                raise ValueError(
                    f"Buffer '{name}' cannot alias shared buffer "
                    f"'{aliases}' with location '{location}'."
                )
            # If parent is local (non-persistent), child cannot be persistent
            if parent_entry.is_local and persistent:
                raise ValueError(
                    f"Persistent buffer '{name}' cannot alias "
                    f"non-persistent local buffer '{aliases}'."
                )

        # Create and store entry
        entry = BufferEntry(
            name=name,
            factory=factory,
            size=size,
            location=location,
            persistent=persistent,
            aliases=aliases,
            precision=precision,
        )
        context.entries[name] = entry
        context.invalidate_layouts()

    def update_buffer(
        self,
        name: str,
        factory: object,
        **kwargs: object,
    ) -> None:
        """Update an existing buffer's properties.

        Parameters
        ----------
        name
            Buffer name to update.
        factory
            Factory instance that owns this buffer.
        **kwargs
            Properties to update (size, location, persistent, aliases).

        Notes
        -----
        Silently ignores updates for factories with no registered context.
        """
        if factory not in self._contexts:
            return
        context = self._contexts[factory]
        if name not in context.entries:
            return

        old_entry = context.entries[name]
        new_values = attrs.asdict(old_entry)
        new_values.update(kwargs)
        context.entries[name] = BufferEntry(**new_values)
        context.invalidate_layouts()

    def clear_layout(self, factory: object) -> None:
        """Invalidate cached slices for a factory.

        Parameters
        ----------
        factory
            Factory instance whose layouts should be cleared.
        """
        if factory in self._contexts:
            self._contexts[factory].invalidate_layouts()

    def clear_factory(self, factory: object) -> None:
        """Remove all buffer registrations for a factory.

        Parameters
        ----------
        factory
            Factory instance to remove.
        """
        if factory in self._contexts:
            del self._contexts[factory]

    def _build_shared_layout(
        self, context: BufferContext
    ) -> Dict[str, slice]:
        """Compute slice indices for shared memory buffers.

        Parameters
        ----------
        context
            BufferContext to compute layout for.

        Returns
        -------
        Dict[str, slice]
            Mapping of buffer names to shared memory slices.
        """
        offset = 0
        layout = {}
        alias_offsets = {}

        # Process non-aliased buffers first
        for name, entry in context.entries.items():
            if entry.location != 'shared' or entry.aliases is not None:
                continue
            layout[name] = slice(offset, offset + entry.size)
            alias_offsets[name] = 0
            offset += entry.size

        # Process aliased buffers
        for name, entry in context.entries.items():
            if entry.location != 'shared' or entry.aliases is None:
                continue
            parent_name = entry.aliases
            parent_slice = layout[parent_name]
            parent_start = parent_slice.start
            current_offset = alias_offsets.get(parent_name, 0)
            layout[name] = slice(
                parent_start + current_offset,
                parent_start + current_offset + entry.size
            )
            alias_offsets[parent_name] = current_offset + entry.size

        return layout

    def _build_persistent_layout(
        self, context: BufferContext
    ) -> Dict[str, slice]:
        """Compute slice indices for persistent local buffers.

        Parameters
        ----------
        context
            BufferContext to compute layout for.

        Returns
        -------
        Dict[str, slice]
            Mapping of buffer names to persistent_local slices.
        """
        offset = 0
        layout = {}
        alias_offsets = {}

        # Process non-aliased persistent buffers first
        for name, entry in context.entries.items():
            if not entry.is_persistent_local or entry.aliases is not None:
                continue
            layout[name] = slice(offset, offset + entry.size)
            alias_offsets[name] = 0
            offset += entry.size

        # Process aliased persistent buffers
        for name, entry in context.entries.items():
            if not entry.is_persistent_local or entry.aliases is None:
                continue
            parent_name = entry.aliases
            # Parent must be persistent (validated at registration)
            parent_slice = layout[parent_name]
            parent_start = parent_slice.start
            current_offset = alias_offsets.get(parent_name, 0)
            layout[name] = slice(
                parent_start + current_offset,
                parent_start + current_offset + entry.size
            )
            alias_offsets[parent_name] = current_offset + entry.size

        return layout

    def _build_local_sizes(
        self, context: BufferContext
    ) -> Dict[str, int]:
        """Compute sizes for local (non-persistent) buffers.

        Parameters
        ----------
        context
            BufferContext to compute sizes for.

        Returns
        -------
        Dict[str, int]
            Mapping of buffer names to local array sizes.
        """
        sizes = {}
        for name, entry in context.entries.items():
            if entry.is_local:
                # cuda.local.array requires size >= 1
                sizes[name] = max(entry.size, 1)
        return sizes

    def shared_buffer_size(self, factory: object) -> int:
        """Return total shared memory elements for a factory.

        Parameters
        ----------
        factory
            Factory instance to query.

        Returns
        -------
        int
            Total shared memory elements (excludes aliased buffers).
        """
        if factory not in self._contexts:
            return 0
        context = self._contexts[factory]

        # Rebuild layout if needed
        if context._shared_layout is None:
            context._shared_layout = self._build_shared_layout(context)

        # Sum sizes of non-aliased shared buffers
        total = 0
        for name, entry in context.entries.items():
            if entry.location == 'shared' and entry.aliases is None:
                total += entry.size
        return total

    def local_buffer_size(self, factory: object) -> int:
        """Return total local memory elements for a factory.

        Parameters
        ----------
        factory
            Factory instance to query.

        Returns
        -------
        int
            Total local memory elements (max(size, 1) for each).
        """
        if factory not in self._contexts:
            return 0
        context = self._contexts[factory]

        if context._local_sizes is None:
            context._local_sizes = self._build_local_sizes(context)

        return sum(context._local_sizes.values())

    def persistent_local_buffer_size(self, factory: object) -> int:
        """Return total persistent local elements for a factory.

        Parameters
        ----------
        factory
            Factory instance to query.

        Returns
        -------
        int
            Total persistent_local elements (excludes aliased buffers).
        """
        if factory not in self._contexts:
            return 0
        context = self._contexts[factory]

        if context._persistent_layout is None:
            context._persistent_layout = self._build_persistent_layout(
                context
            )

        # Sum sizes of non-aliased persistent buffers
        total = 0
        for name, entry in context.entries.items():
            if entry.is_persistent_local and entry.aliases is None:
                total += entry.size
        return total

    def get_allocator(
        self,
        name: str,
        factory: object,
    ) -> Callable:
        """Generate CUDA device function for buffer allocation.

        Parameters
        ----------
        name
            Buffer name to generate allocator for.
        factory
            Factory instance that owns the buffer.

        Returns
        -------
        Callable
            CUDA device function that allocates the buffer.

        Raises
        ------
        KeyError
            If factory or buffer name not registered.
        """
        if factory not in self._contexts:
            raise KeyError(
                f"Factory {factory} has no registered buffer context."
            )
        context = self._contexts[factory]
        if name not in context.entries:
            raise KeyError(
                f"Buffer '{name}' not registered for factory."
            )

        entry = context.entries[name]

        # Ensure layouts are computed
        if context._shared_layout is None:
            context._shared_layout = self._build_shared_layout(context)
        if context._persistent_layout is None:
            context._persistent_layout = self._build_persistent_layout(
                context
            )
        if context._local_sizes is None:
            context._local_sizes = self._build_local_sizes(context)

        # Compile-time constants
        _shared = entry.is_shared
        _persistent = entry.is_persistent_local
        _local = entry.is_local

        # Pre-computed values
        shared_slice = context._shared_layout.get(name, slice(0, 0))
        persistent_slice = context._persistent_layout.get(
            name, slice(0, 0)
        )
        local_size = context._local_sizes.get(name, 1)
        precision = entry.precision

        @cuda.jit(device=True, inline=True, ForceInline=True, **compile_kwargs)
        def allocate_buffer(shared_parent, persistent_parent):
            """Allocate buffer from appropriate memory region."""
            if _shared:
                array = shared_parent[shared_slice]
            elif _persistent:
                array = persistent_parent[persistent_slice]
            else:
                array = cuda.local.array(local_size, precision)
            return array

        return allocate_buffer


# Module-level singleton instance
buffer_registry = BufferRegistry()
