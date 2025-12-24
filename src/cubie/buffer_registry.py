"""Centralized buffer registry for CUDA memory management.

This module provides a package-wide singleton registry that manages
buffer metadata for all CUDA factories. Factories register their
buffer requirements during initialization, and the registry provides
CUDA-compatible allocator device functions during build.

The registry uses a lazy cached build pattern - slice layouts are
computed on demand and invalidated when any buffer is modified.
"""

from typing import Callable, Dict, Optional, Tuple, Any, Set

import attrs
from attrs import validators
import numpy as np
from numba import cuda, int32

from cubie._utils import getype_validator, precision_validator
from cubie.cuda_simsafe import compile_kwargs, local


@attrs.define
class CUDABuffer:
    """Immutable record describing a single buffer's requirements.

    Attributes
    ----------
    name : str
        Unique buffer name within parent context.
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

    def build_allocator(
        self,
        shared_slice: Optional[slice],
        persistent_slice: Optional[slice],
        local_size: Optional[int],
        zero: bool = False,
    ) -> Callable:
        """Compile CUDA device function for buffer allocation.

        Generates an inlined device function that allocates this buffer
        from the appropriate memory region based on which slice parameters
        are provided.

        Parameters
        ----------
        shared_slice
            Slice into shared memory, or None if not using shared.
        persistent_slice
            Slice into persistent local memory, or None if not using
            persistent.
        local_size
            Size for local array allocation, or None if not local.
        zero
            If True, initialize all elements to zero after allocation.

        Returns
        -------
        Callable
            CUDA device function: (shared, persistent) -> array

            The device function accepts shared and persistent memory
            arrays and returns a view/slice into the appropriate memory
            region, or allocates a fresh local array.

        Notes
        -----
        When a buffer aliases another buffer and aliasing succeeds, both
        buffers receive slices in the same memory region (shared or
        persistent) that overlap. The allocator transparently provides
        the correct view without needing a separate parent reference.
        """
        # Compile-time constants captured in closure
        _use_shared = shared_slice is not None
        _use_persistent = persistent_slice is not None
        _shared_slice = shared_slice if _use_shared else slice(0, 0)
        _persistent_slice = (
            persistent_slice if _use_persistent else slice(0, 0)
        )
        _local_size = local_size if local_size is not None else 1
        _precision = self.precision
        _zero = zero
        elements = int32(self.size)

        @cuda.jit(device=True, inline=True, **compile_kwargs)
        def allocate_buffer(shared, persistent):
            """Allocate buffer from appropriate memory region."""
            if _use_shared:
                array = shared[_shared_slice]
            elif _use_persistent:
                array = persistent[_persistent_slice]
            else:
                array = local.array(_local_size, _precision)
            if _zero:
                for i in range(elements):
                    array[i] = _precision(0.0)
            return array

        return allocate_buffer


@attrs.define
class BufferGroup:
    """Groups all buffer entries for a single parent object.

    Attributes
    ----------
    parent : object
        Parent instance that owns this group.
    entries : Dict[str, CUDABuffer]
        Registered buffers by name.
    _shared_layout : Dict[str, slice] or None
        Cached unified shared memory layout (None when invalid).
    _persistent_layout : Dict[str, slice] or None
        Cached persistent_local slices (None when invalid).
    _local_sizes : Dict[str, int] or None
        Cached local sizes (None when invalid).
    _alias_consumption : Dict[str, int]
        Tracks consumed space in aliased buffers for layout
        computation.
    """

    parent: object = attrs.field()
    entries: Dict[str, CUDABuffer] = attrs.field(factory=dict)
    _shared_layout: Optional[Dict[str, slice]] = attrs.field(
        default=None, init=False
    )
    _persistent_layout: Optional[Dict[str, slice]] = attrs.field(
        default=None, init=False
    )
    _local_sizes: Optional[Dict[str, int]] = attrs.field(
        default=None, init=False
    )
    _alias_consumption: Dict[str, int] = attrs.field(
        factory=dict, init=False
    )

    def invalidate_layouts(self) -> None:
        """Set all cached layouts to None and clear alias consumption."""
        self._shared_layout = None
        self._persistent_layout = None
        self._local_sizes = None
        self._alias_consumption.clear()

    def register(
        self,
        name: str,
        size: int,
        location: str,
        persistent: bool = False,
        aliases: Optional[str] = None,
        precision: type = np.float32,
    ) -> None:
        """Register a buffer with this group.

        Parameters
        ----------
        name
            Unique buffer name within this group.
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
            If buffer name already registered for this group.
        ValueError
            If aliases references non-existent buffer.
        ValueError
            If name is empty string.
        ValueError
            If buffer attempts to alias itself.
        """
        if not name:
            raise ValueError("Buffer name cannot be empty.")
        if aliases is not None and aliases == name:
            raise ValueError(f"Buffer '{name}' cannot alias itself.")
        if aliases is not None and aliases not in self.entries:
            raise ValueError(
                f"Alias target '{aliases}' not registered. "
                f"Register '{aliases}' before '{name}'."
            )

        entry = CUDABuffer(
            name=name,
            size=size,
            location=location,
            persistent=persistent,
            aliases=aliases,
            precision=precision,
        )
        self.entries[name] = entry
        self.invalidate_layouts()

    def update_buffer(self, name: str, **kwargs: object) -> Tuple[bool, bool]:
        """Update an existing buffer's properties.

        Parameters
        ----------
        name
            Buffer name to update.
        **kwargs
            Properties to update (size, location, persistent, aliases).

        Returns
        -------
        bool, bool
            Whether the buffer was recognized and updated, respectively.

        Notes
        -----
        Silently ignores updates for buffers not registered.
        """
        recognized = False
        changed = False

        if name not in self.entries:
            recognized = False
            changed = False
        else:
            recognized = True
            old_entry = self.entries[name]
            new_values = attrs.asdict(old_entry)
            new_values.update(kwargs)

            if new_values != attrs.asdict(old_entry):
                changed = True
                self.entries[name] = CUDABuffer(**new_values)
                self.invalidate_layouts()

        return recognized, changed

    def build_shared_layout(self) -> Dict[str, slice]:
        """Compute slice indices for shared memory buffers.

        Implements correct aliasing semantics where aliased children
        deliberately OVERLAP their parent buffers to reuse memory.
        When a child aliases a shared parent with sufficient space,
        the child slices directly into the parent's memory region.

        Aliasing behavior:
        - Child aliases shared parent with space: child OVERLAPS parent
        - Child aliases shared parent without space: fresh allocation
        - Child aliases local parent: fresh shared allocation
        - Non-aliased buffer: fresh allocation

        All allocations use the SAME shared array; aliased buffers
        reuse parent memory via deliberate overlap.

        Returns
        -------
        Dict[str, slice]
            Single unified mapping of buffer names to shared memory
            slices.
        """
        offset = 0
        layout = {}
        self._alias_consumption.clear()

        # Step 1: Allocate non-aliased shared buffers
        for name, entry in self.entries.items():
            if entry.location != 'shared' or entry.aliases is not None:
                continue
            layout[name] = slice(offset, offset + entry.size)
            self._alias_consumption[name] = 0
            offset += entry.size

        # Step 2: Process aliased buffers
        for name, entry in self.entries.items():
            if entry.aliases is None or entry.location != 'shared':
                continue

            parent_entry = self.entries[entry.aliases]

            if parent_entry.is_shared:
                # Parent is shared - check if we can alias it
                consumed = self._alias_consumption.get(entry.aliases, 0)
                available = parent_entry.size - consumed

                if entry.size <= available:
                    # Alias within parent (WITH OVERLAP - reuse memory)
                    parent_slice = layout[entry.aliases]
                    start = parent_slice.start + consumed
                    layout[name] = slice(start, start + entry.size)
                    self._alias_consumption[entry.aliases] = (
                        consumed + entry.size
                    )
                else:
                    # Parent full, allocate fresh shared space
                    layout[name] = slice(offset, offset + entry.size)
                    offset += entry.size
            else:
                # Parent is local, allocate fresh shared space for child
                layout[name] = slice(offset, offset + entry.size)
                offset += entry.size

        return layout

    @property
    def shared_layout(self) -> Dict[str, slice]:
        """Return shared memory layout.
        
        Returns
        -------
        Dict[str, slice]
            Mapping of buffer names to shared memory slices.
        """
        if self._shared_layout is None:
            self._shared_layout = self.build_shared_layout()
        return self._shared_layout

    def build_persistent_layout(self) -> Dict[str, slice]:
        """Compute slice indices for persistent local buffers.

        Implements cross-location aliasing for persistent buffers:
        - If parent is persistent and has sufficient remaining space,
          alias slices within parent
        - Otherwise allocate new persistent space

        Returns
        -------
        Dict[str, slice]
            Mapping of buffer names to persistent_local slices.
        """
        offset = 0
        layout = {}

        # Process non-aliased persistent buffers first
        for name, entry in self.entries.items():
            if not entry.is_persistent_local or entry.aliases is not None:
                continue
            layout[name] = slice(offset, offset + entry.size)
            self._alias_consumption[name] = 0
            offset += entry.size

        # Process aliased persistent buffers
        for name, entry in self.entries.items():
            if not entry.is_persistent_local or entry.aliases is None:
                continue

            parent_entry = self.entries[entry.aliases]

            if parent_entry.is_persistent_local:
                # Parent is persistent - check if space available
                consumed = self._alias_consumption.get(entry.aliases, 0)
                available = parent_entry.size - consumed

                if entry.size <= available:
                    # Alias fits within parent
                    parent_slice = layout[entry.aliases]
                    start = parent_slice.start + consumed
                    layout[name] = slice(start, start + entry.size)
                    self._alias_consumption[entry.aliases] = (
                        consumed + entry.size
                    )
                else:
                    # Parent too small, allocate new persistent space
                    layout[name] = slice(offset, offset + entry.size)
                    offset += entry.size
            else:
                # Parent is not persistent, allocate new persistent space
                layout[name] = slice(offset, offset + entry.size)
                offset += entry.size

        return layout

    def build_local_sizes(self) -> Dict[str, int]:
        """Compute sizes for local (non-persistent) buffers.

        Only allocates local memory for buffers that are not already
        allocated in shared or persistent memory via aliasing.

        Returns
        -------
        Dict[str, int]
            Mapping of buffer names to local array sizes.
        """
        # Ensure shared and persistent layouts are computed first
        if self._shared_layout is None:
            self._shared_layout = self.build_shared_layout()
        if self._persistent_layout is None:
            self._persistent_layout = self.build_persistent_layout()

        sizes = {}
        for name, entry in self.entries.items():
            if entry.is_local:
                # Check if this buffer was already allocated via aliasing
                if (name in self._shared_layout
                        or name in self._persistent_layout):
                    # Already allocated elsewhere, skip local allocation
                    continue
                # cuda.local.array requires size >= 1
                sizes[name] = max(entry.size, 1)
        return sizes

    def shared_buffer_size(self) -> int:
        """Return total shared memory elements.

        Returns
        -------
        int
            Total shared memory elements needed (end of last slice).
        """
        if self._shared_layout is None:
            self._shared_layout = self.build_shared_layout()

        if not self._shared_layout:
            return 0
        return max(s.stop for s in self._shared_layout.values())

    def local_buffer_size(self) -> int:
        """Return total local memory elements.

        Returns
        -------
        int
            Total local memory elements (max(size, 1) for each).
        """
        if self._local_sizes is None:
            self._local_sizes = self.build_local_sizes()

        return sum(self._local_sizes.values())

    def persistent_local_buffer_size(self) -> int:
        """Return total persistent local elements.

        Returns
        -------
        int
            Total persistent_local elements needed (end of last slice).
        """
        if self._persistent_layout is None:
            self._persistent_layout = self.build_persistent_layout()

        if not self._persistent_layout:
            return 0
        return max(s.stop for s in self._persistent_layout.values())

    def get_allocator(self, name: str, zero: bool = False) -> Callable:
        """Generate CUDA device function for buffer allocation.

        Retrieves the pre-computed memory slice for this buffer from the
        appropriate layout (shared, persistent, or local) and generates
        an allocator that uses that slice.

        Parameters
        ----------
        name
            Buffer name to generate allocator for.
        zero
            If True, initialize all elements to zero after allocation.

        Returns
        -------
        Callable
            CUDA device function that allocates the buffer.
            Signature: (shared, persistent) -> array

        Raises
        ------
        KeyError
            If buffer name not registered.

        Notes
        -----
        The layout building phase (build_shared_layout and
        build_persistent_layout) determines which memory region each
        buffer uses and assigns slices accordingly. This method simply
        retrieves those pre-computed slices and creates an allocator
        that uses them.

        For aliased buffers, the layout builder assigns slices that
        overlap the parent buffer, implementing aliasing transparently
        at the slice level.
        """
        if name not in self.entries:
            raise KeyError(f"Buffer '{name}' not registered for parent.")

        entry = self.entries[name]

        # Ensure layouts are computed
        if self._shared_layout is None:
            self._shared_layout = self.build_shared_layout()
        if self._persistent_layout is None:
            self._persistent_layout = self.build_persistent_layout()
        if self._local_sizes is None:
            self._local_sizes = self.build_local_sizes()

        # Get slice from appropriate layout
        shared_slice = self._shared_layout.get(name)
        persistent_slice = self._persistent_layout.get(name)
        local_size = self._local_sizes.get(name)

        return entry.build_allocator(
            shared_slice, persistent_slice, local_size, zero
        )


@attrs.define
class BufferRegistry:
    """Central registry managing all buffer metadata for CUDA parents.

    The registry is a package-level singleton that tracks buffer
    requirements for all parent objects. It uses lazy cached builds for
    slice/layout computation - layouts are set to None on any change
    and regenerated on access.

    Attributes
    ----------
    _groups : Dict[object, BufferGroup]
        Maps parent instances to their buffer groups.
    """

    _groups: Dict[object, BufferGroup] = attrs.field(
        factory=dict, init=False
    )

    def register(
        self,
        name: str,
        parent: object,
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
            Unique buffer name within parent context.
        parent
            Parent instance that owns this buffer.
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
            If buffer name already registered for this parent.
        ValueError
            If aliases references non-existent buffer.
        ValueError
            If name is empty string.
        ValueError
            If buffer attempts to alias itself.
        ValueError
            If precision is not a supported NumPy floating-point type.
        """
        if parent not in self._groups:
            self._groups[parent] = BufferGroup(parent=parent)
        self._groups[parent].register(
            name, size, location, persistent, aliases, precision
        )

    def update_buffer(
        self,
        name: str,
        parent: object,
        **kwargs: object,
    ) -> Tuple[bool, bool]:
        """Update an existing buffer's properties.

        Parameters
        ----------
        name
            Buffer name to update.
        parent
            Parent instance that owns this buffer.
        **kwargs
            Properties to update (size, location, persistent, aliases).

         Returns
        -------
        bool, bool
            Whether the buffer was recognized and updated, respectively.

        Notes
        -----
        Silently ignores updates for parents with no registered group.
        """
        if parent not in self._groups:
            return False, False

        recognized, changed = self._groups[parent].update_buffer(name,
                                                                 **kwargs)
        return recognized, changed

    def clear_layout(self, parent: object) -> None:
        """Invalidate cached slices for a parent.

        Parameters
        ----------
        parent
            Parent instance whose layouts should be cleared.
        """
        if parent in self._groups:
            self._groups[parent].invalidate_layouts()

    def clear_parent(self, parent: object) -> None:
        """Remove all buffer registrations for a parent.

        Parameters
        ----------
        parent
            Parent instance to remove.
        """
        if parent in self._groups:
            del self._groups[parent]

    def reset(self) -> None:
        """Clear all buffer registrations, for example when switching
        algorithms."""
        allparents = list(self._groups.keys())
        for parent in allparents:
            self.clear_parent(parent)

    def update(
        self,
        parent: object,
        updates_dict: Optional[Dict[str, Any]] = None,
        silent: bool = False,
        **kwargs: Any,
    ) -> Set[str]:
        """Update buffer locations from keyword arguments.

        For each key of the form '[buffer_name]_location', finds the
        corresponding buffer and updates its location. Mirrors the pattern
        of CUDAFactory.update_compile_settings().

        Parameters
        ----------
        parent
            Parent instance that owns the buffers to update.
        updates_dict
            Mapping of parameter names to new values.
        silent
            Suppress errors for unrecognized parameters.
        **kwargs
            Additional parameters merged into updates_dict.

        Returns
        -------
        Set[str]
            Names of parameters that were successfully recognized.

        Raises
        ------
        ValueError
            If a location value is not 'shared' or 'local'.

        Notes
        -----
        A parameter is recognized if it matches '[buffer_name]_location'
        where buffer_name is registered for the parent. The method
        silently ignores unrecognized parameters when silent=True.
        """
        if updates_dict is None:
            updates_dict = {}
        updates_dict = updates_dict.copy()
        if kwargs:
            updates_dict.update(kwargs)
        if not updates_dict:
            return set()

        if parent not in self._groups:
            return set()

        recognized = set()

        for key, value in updates_dict.items():
            if not key.endswith('_location'):
                continue

            buffer_name = key.removesuffix("_location")

            if value not in ('shared', 'local'):
                raise ValueError(
                    f"Invalid location '{value}' for buffer "
                    f"'{buffer_name}'. Must be 'shared' or 'local'."
                )

            buffer_recognized, buffer_changed = self.update_buffer(
                buffer_name, parent, location=value
            )
            if buffer_recognized:
                recognized.add(key)
            if buffer_changed:
                self.clear_layout(parent)

        return recognized

    def shared_buffer_size(self, parent: object) -> int:
        """Return total shared memory elements for a parent.

        Parameters
        ----------
        parent
            Parent instance to query.

        Returns
        -------
        int
            Total shared memory elements (excludes aliased buffers).
        """
        if parent not in self._groups:
            return 0
        return self._groups[parent].shared_buffer_size()

    def local_buffer_size(self, parent: object) -> int:
        """Return total local memory elements for a parent.

        Parameters
        ----------
        parent
            Parent instance to query.

        Returns
        -------
        int
            Total local memory elements (max(size, 1) for each).
        """
        if parent not in self._groups:
            return 0
        return self._groups[parent].local_buffer_size()

    def persistent_local_buffer_size(self, parent: object) -> int:
        """Return total persistent local elements for a parent.

        Parameters
        ----------
        parent
            Parent instance to query.

        Returns
        -------
        int
            Total persistent_local elements (excludes aliased buffers).
        """
        if parent not in self._groups:
            return 0
        return self._groups[parent].persistent_local_buffer_size()

    def get_allocator(
        self,
        name: str,
        parent: object,
        zero: bool = False,
    ) -> Callable:
        """Generate CUDA device function for buffer allocation.

        Parameters
        ----------
        name
            Buffer name to generate allocator for.
        parent
            Parent instance that owns the buffer.
        zero
            If True, initialize all elements to zero after allocation.

        Returns
        -------
        Callable
            CUDA device function that allocates the buffer.

        Raises
        ------
        KeyError
            If parent or buffer name not registered.
        """
        if parent not in self._groups:
            raise KeyError(
                f"Parent {parent} has no registered buffer group."
            )
        return self._groups[parent].get_allocator(name, zero)

    def get_child_allocators(
        self,
        parent: object,
        child: object,
        name: Optional[str] = None,
    ) -> Tuple[Callable, Callable]:
        """Register child buffers and return shared and persistent allocators.

        Registers buffers for a child device function within the parent's
        buffer group, then returns allocators that provide slices into the
        parent's shared and persistent memory regions.

        Parameters
        ----------
        parent
            Parent instance that will allocate memory for the child.
        child
            Child instance whose buffer requirements should be registered.
        name
            Optional base name for the buffer registrations. If not provided,
            uses 'child_{id}' as the base name.

        Returns
        -------
        Callable
            Allocator for child's shared memory (returns slice).
        Callable
            Allocator for child's persistent memory (returns slice).

        Notes
        -----
        This method computes the shared and persistent buffer sizes for the
        child, registers them with the parent's buffer group, and returns
        allocators that slice the parent's memory regions appropriately.
        The child's buffers are registered with names
        '{name}_shared' and '{name}_persistent' if name is provided, otherwise
        'child_{child_id}_shared' and 'child_{child_id}_persistent'.
        """
        # Get child buffer sizes
        child_shared_size = self.shared_buffer_size(child)
        child_persistent_size = self.persistent_local_buffer_size(child)

        # Generate buffer names
        if name is None:
            child_id = id(child)
            base_name = f'child_{child_id}'
        else:
            base_name = name
        
        shared_name = f'{base_name}_shared'
        persistent_name = f'{base_name}_persistent'

        # Get precision from parent
        precision = parent.precision

        # Register child buffers with parent
        self.register(
            shared_name,
            parent,
            child_shared_size,
            'shared',
            precision=precision
        )
        self.register(
            persistent_name,
            parent,
            child_persistent_size,
            'local',
            persistent=True,
            precision=precision
        )

        # Get and return allocators
        alloc_shared = self.get_allocator(shared_name, parent)
        alloc_persistent = self.get_allocator(persistent_name, parent)

        return alloc_shared, alloc_persistent


# Module-level singleton instance
buffer_registry = BufferRegistry()
