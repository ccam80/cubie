# Implementation Task List
# Feature: Buffer Registry Architecture Refactor - buffer_registry.py
# Plan Reference: .github/active_plans/buffer_registry_refactor/agent_plan.md

## Overview

This task list covers Task 1 of 3: Changes to `buffer_registry.py` ONLY.
Do NOT modify integration files (algorithms, loops, matrix-free solvers) or solver.py.

**Scope**: `src/cubie/buffer_registry.py` and `tests/test_buffer_registry.py`

---

## Task Group 1: Rename BufferEntry to CUDABuffer - SEQUENTIAL
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: src/cubie/buffer_registry.py (lines 24-77)

**Input Validation Required**:
- None (attrs validators already handle validation)

**Tasks**:

1. **Rename class BufferEntry → CUDABuffer**
   - File: src/cubie/buffer_registry.py
   - Action: Modify
   - Details:
     ```python
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
     ```
   - Changes:
     - Line 24: Change `class BufferEntry:` to `class CUDABuffer:`
     - Line 25-42: Update docstring to remove `factory` attribute documentation
   - Edge cases: None - pure rename
   - Integration: All internal references must also be updated

2. **Remove factory attribute from CUDABuffer**
   - File: src/cubie/buffer_registry.py
   - Action: Modify
   - Details:
     - Remove line 46: `factory: object = attrs.field()`
   - Edge cases: None
   - Integration: BufferGroup tracks ownership via `parent` attribute

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 2: Rename BufferContext to BufferGroup - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/buffer_registry.py (lines 79-120)

**Input Validation Required**:
- None (existing validators sufficient)

**Tasks**:

1. **Rename class BufferContext → BufferGroup and factory → parent**
   - File: src/cubie/buffer_registry.py
   - Action: Modify
   - Details:
     ```python
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
             Cached shared memory slices (None when invalid).
         _persistent_layout : Dict[str, slice] or None
             Cached persistent_local slices (None when invalid).
         _local_sizes : Dict[str, int] or None
             Cached local sizes (None when invalid).
         _alias_consumption : Dict[str, int]
             Tracks consumed space in aliased buffers.
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
     ```
   - Changes:
     - Line 79: `class BufferContext:` → `class BufferGroup:`
     - Line 80-96: Update docstring to change factory→parent and BufferEntry→CUDABuffer
     - Line 99: `factory: object` → `parent: object`
     - Line 100: `Dict[str, BufferEntry]` → `Dict[str, CUDABuffer]`
     - Line 110-112: Rename `_alias_offsets` → `_alias_consumption`
   - Edge cases: None - pure rename
   - Integration: BufferRegistry references must be updated

2. **Update invalidate_layouts to clear _alias_consumption**
   - File: src/cubie/buffer_registry.py
   - Action: Modify
   - Details:
     ```python
     def invalidate_layouts(self) -> None:
         """Set all cached layouts to None and clear alias consumption."""
         self._shared_layout = None
         self._persistent_layout = None
         self._local_sizes = None
         self._alias_consumption.clear()
     ```
   - Changes:
     - Line 119: `self._alias_offsets.clear()` → `self._alias_consumption.clear()`
   - Edge cases: None
   - Integration: None

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 3: Add Methods to BufferGroup - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Group 2

**Required Context**:
- File: src/cubie/buffer_registry.py (lines 79-120, 141-240, 296-403)
- File: .github/active_plans/buffer_registry_refactor/agent_plan.md (aliasing logic section)

**Input Validation Required**:
- register: name not empty, name not duplicate, name not self-alias, alias target exists
- update_buffer: none (silent ignore for missing)

**Tasks**:

1. **Add register method to BufferGroup**
   - File: src/cubie/buffer_registry.py
   - Action: Modify
   - Details: Add after invalidate_layouts method
     ```python
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
         if name in self.entries:
             raise ValueError(
                 f"Buffer '{name}' already registered for this parent."
             )
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
     ```
   - Edge cases:
     - Empty name: raise ValueError
     - Self-alias: raise ValueError
     - Duplicate name: raise ValueError
     - Alias target missing: raise ValueError
   - Integration: Called by BufferRegistry.register wrapper
   - **NOTE**: Cross-type aliasing constraints REMOVED - any buffer can alias any parent

2. **Add update_buffer method to BufferGroup**
   - File: src/cubie/buffer_registry.py
   - Action: Modify
   - Details: Add after register method
     ```python
     def update_buffer(self, name: str, **kwargs: object) -> None:
         """Update an existing buffer's properties.

         Parameters
         ----------
         name
             Buffer name to update.
         **kwargs
             Properties to update (size, location, persistent, aliases).

         Notes
         -----
         Silently ignores updates for buffers not registered.
         """
         if name not in self.entries:
             return

         old_entry = self.entries[name]
         new_values = attrs.asdict(old_entry)
         new_values.update(kwargs)
         self.entries[name] = CUDABuffer(**new_values)
         self.invalidate_layouts()
     ```
   - Edge cases: Silent ignore for missing buffer
   - Integration: Called by BufferRegistry.update_buffer wrapper

3. **Add build_shared_layout method to BufferGroup**
   - File: src/cubie/buffer_registry.py
   - Action: Modify
   - Details: Add after update_buffer method - NEW ALIASING LOGIC
     ```python
     def build_shared_layout(self) -> Dict[str, slice]:
         """Compute slice indices for shared memory buffers.

         Implements cross-location aliasing:
         - If parent is shared and has sufficient remaining space, alias
           slices within parent
         - If parent is shared but too small, allocate per buffer's own
           settings
         - If parent is local, allocate per buffer's own settings
         - Multiple aliases consume parent space first-come-first-serve

         Returns
         -------
         Dict[str, slice]
             Mapping of buffer names to shared memory slices.
         """
         offset = 0
         layout = {}
         self._alias_consumption.clear()

         # Process non-aliased shared buffers first
         for name, entry in self.entries.items():
             if entry.location != 'shared' or entry.aliases is not None:
                 continue
             layout[name] = slice(offset, offset + entry.size)
             self._alias_consumption[name] = 0
             offset += entry.size

         # Process aliased buffers
         for name, entry in self.entries.items():
             if entry.aliases is None:
                 continue

             parent_entry = self.entries[entry.aliases]

             if parent_entry.is_shared:
                 # Parent is shared - check if space available
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
                 elif entry.is_shared:
                     # Parent too small, allocate new shared space
                     layout[name] = slice(offset, offset + entry.size)
                     offset += entry.size
                 # else: not shared, will be handled by persistent/local
             elif entry.is_shared:
                 # Parent is local, allocate new shared space
                 layout[name] = slice(offset, offset + entry.size)
                 offset += entry.size
             # else: buffer not shared, skip

         return layout
     ```
   - Edge cases:
     - Zero-size buffer: gets slice(n, n) with 0 length
     - Alias larger than parent remaining: falls back to own allocation
     - Multiple aliases exhaust parent: first-come-first-serve
   - Integration: Called by BufferGroup.get_allocator and size methods

4. **Add build_persistent_layout method to BufferGroup**
   - File: src/cubie/buffer_registry.py
   - Action: Modify
   - Details: Add after build_shared_layout method
     ```python
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
         persistent_consumption = {}

         # Process non-aliased persistent buffers first
         for name, entry in self.entries.items():
             if not entry.is_persistent_local or entry.aliases is not None:
                 continue
             layout[name] = slice(offset, offset + entry.size)
             persistent_consumption[name] = 0
             offset += entry.size

         # Process aliased persistent buffers
         for name, entry in self.entries.items():
             if not entry.is_persistent_local or entry.aliases is None:
                 continue

             parent_entry = self.entries[entry.aliases]

             if parent_entry.is_persistent_local:
                 # Parent is persistent - check if space available
                 consumed = persistent_consumption.get(entry.aliases, 0)
                 available = parent_entry.size - consumed

                 if entry.size <= available:
                     # Alias fits within parent
                     parent_slice = layout[entry.aliases]
                     start = parent_slice.start + consumed
                     layout[name] = slice(start, start + entry.size)
                     persistent_consumption[entry.aliases] = (
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
     ```
   - Edge cases: Similar to shared layout
   - Integration: Called by BufferGroup.get_allocator and size methods

5. **Add build_local_sizes method to BufferGroup**
   - File: src/cubie/buffer_registry.py
   - Action: Modify
   - Details: Add after build_persistent_layout method
     ```python
     def build_local_sizes(self) -> Dict[str, int]:
         """Compute sizes for local (non-persistent) buffers.

         Returns
         -------
         Dict[str, int]
             Mapping of buffer names to local array sizes.
         """
         sizes = {}
         for name, entry in self.entries.items():
             if entry.is_local:
                 # cuda.local.array requires size >= 1
                 sizes[name] = max(entry.size, 1)
         return sizes
     ```
   - Edge cases: Zero-size buffer gets size=1 (cuda.local.array minimum)
   - Integration: Called by BufferGroup.get_allocator and size methods

6. **Add shared_buffer_size method to BufferGroup**
   - File: src/cubie/buffer_registry.py
   - Action: Modify
   - Details: Add after build_local_sizes method
     ```python
     def shared_buffer_size(self) -> int:
         """Return total shared memory elements.

         Returns
         -------
         int
             Total shared memory elements (excludes aliased buffers).
         """
         if self._shared_layout is None:
             self._shared_layout = self.build_shared_layout()

         total = 0
         for name, entry in self.entries.items():
             if entry.location == 'shared' and entry.aliases is None:
                 total += entry.size
         return total
     ```
   - Edge cases: Empty group returns 0
   - Integration: BufferRegistry wrapper calls this

7. **Add local_buffer_size method to BufferGroup**
   - File: src/cubie/buffer_registry.py
   - Action: Modify
   - Details: Add after shared_buffer_size method
     ```python
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
     ```
   - Edge cases: Empty group returns 0
   - Integration: BufferRegistry wrapper calls this

8. **Add persistent_local_buffer_size method to BufferGroup**
   - File: src/cubie/buffer_registry.py
   - Action: Modify
   - Details: Add after local_buffer_size method
     ```python
     def persistent_local_buffer_size(self) -> int:
         """Return total persistent local elements.

         Returns
         -------
         int
             Total persistent_local elements (excludes aliased buffers).
         """
         if self._persistent_layout is None:
             self._persistent_layout = self.build_persistent_layout()

         total = 0
         for name, entry in self.entries.items():
             if entry.is_persistent_local and entry.aliases is None:
                 total += entry.size
         return total
     ```
   - Edge cases: Empty group returns 0
   - Integration: BufferRegistry wrapper calls this

9. **Add get_allocator method to BufferGroup**
   - File: src/cubie/buffer_registry.py
   - Action: Modify
   - Details: Add after persistent_local_buffer_size method
     ```python
     def get_allocator(self, name: str) -> Callable:
         """Generate CUDA device function for buffer allocation.

         Parameters
         ----------
         name
             Buffer name to generate allocator for.

         Returns
         -------
         Callable
             CUDA device function that allocates the buffer.

         Raises
         ------
         KeyError
             If buffer name not registered.
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

         # Determine allocation source
         shared_slice = self._shared_layout.get(name)
         persistent_slice = self._persistent_layout.get(name)
         local_size = self._local_sizes.get(name)

         return entry.build_allocator(shared_slice, persistent_slice, local_size)
     ```
   - Edge cases: Missing buffer raises KeyError
   - Integration: BufferRegistry wrapper calls this; CUDABuffer.build_allocator handles compilation

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 4: Add build_allocator to CUDABuffer - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/buffer_registry.py (lines 24-77, 543-555)
- File: src/cubie/cuda_simsafe.py (compile_kwargs)

**Input Validation Required**:
- None (guarantee-by-design - BufferGroup ensures correct parameters)

**Tasks**:

1. **Add build_allocator method to CUDABuffer class**
   - File: src/cubie/buffer_registry.py
   - Action: Modify
   - Details: Add after is_local property in CUDABuffer class
     ```python
     def build_allocator(
         self,
         shared_slice: Optional[slice],
         persistent_slice: Optional[slice],
         local_size: Optional[int],
     ) -> Callable:
         """Compile CUDA device function for buffer allocation.

         Generates an inlined device function that allocates this buffer
         from the appropriate memory region based on which parameters are
         provided.

         Parameters
         ----------
         shared_slice
             Slice into shared memory, or None if not shared.
         persistent_slice
             Slice into persistent local memory, or None if not persistent.
         local_size
             Size for local array allocation, or None if not local.

         Returns
         -------
         Callable
             CUDA device function: (shared_parent, persistent_parent) -> array
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

         @cuda.jit(device=True, inline=True, **compile_kwargs)
         def allocate_buffer(shared_parent, persistent_parent):
             """Allocate buffer from appropriate memory region."""
             if _use_shared:
                 array = shared_parent[_shared_slice]
             elif _use_persistent:
                 array = persistent_parent[_persistent_slice]
             else:
                 array = cuda.local.array(_local_size, _precision)
             return array

         return allocate_buffer
     ```
   - Edge cases:
     - All parameters None: uses local array with size=1 (cuda minimum)
     - shared_slice takes priority over persistent_slice
   - Integration: Called by BufferGroup.get_allocator

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 5: Refactor BufferRegistry to Wrapper Pattern - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Groups 2, 3, 4

**Required Context**:
- File: src/cubie/buffer_registry.py (lines 122-559)
- File: .github/active_plans/buffer_registry_refactor/agent_plan.md

**Input Validation Required**:
- None (delegation to BufferGroup handles validation)

**Tasks**:

1. **Rename _contexts to _groups and factory to parent in BufferRegistry**
   - File: src/cubie/buffer_registry.py
   - Action: Modify
   - Details:
     ```python
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
     ```
   - Changes:
     - Line 133-138: Rename `_contexts` to `_groups` in docstring and attribute

2. **Simplify BufferRegistry.register to wrapper pattern**
   - File: src/cubie/buffer_registry.py
   - Action: Modify
   - Details: Replace entire register method with wrapper
     ```python
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
     ```
   - Edge cases: Creates new group if parent not seen before
   - Integration: Delegates to BufferGroup.register

3. **Simplify BufferRegistry.update_buffer to wrapper pattern**
   - File: src/cubie/buffer_registry.py
   - Action: Modify
   - Details:
     ```python
     def update_buffer(
         self,
         name: str,
         parent: object,
         **kwargs: object,
     ) -> None:
         """Update an existing buffer's properties.

         Parameters
         ----------
         name
             Buffer name to update.
         parent
             Parent instance that owns this buffer.
         **kwargs
             Properties to update (size, location, persistent, aliases).

         Notes
         -----
         Silently ignores updates for parents with no registered group.
         """
         if parent not in self._groups:
             return
         self._groups[parent].update_buffer(name, **kwargs)
     ```

4. **Update BufferRegistry.clear_layout to use _groups and parent**
   - File: src/cubie/buffer_registry.py
   - Action: Modify
   - Details:
     ```python
     def clear_layout(self, parent: object) -> None:
         """Invalidate cached slices for a parent.

         Parameters
         ----------
         parent
             Parent instance whose layouts should be cleared.
         """
         if parent in self._groups:
             self._groups[parent].invalidate_layouts()
     ```

5. **Rename clear_factory to clear_parent**
   - File: src/cubie/buffer_registry.py
   - Action: Modify
   - Details:
     ```python
     def clear_parent(self, parent: object) -> None:
         """Remove all buffer registrations for a parent.

         Parameters
         ----------
         parent
             Parent instance to remove.
         """
         if parent in self._groups:
             del self._groups[parent]
     ```
   - **NOTE**: Method renamed from `clear_factory` to `clear_parent`

6. **Simplify BufferRegistry.shared_buffer_size to wrapper pattern**
   - File: src/cubie/buffer_registry.py
   - Action: Modify
   - Details:
     ```python
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
     ```

7. **Simplify BufferRegistry.local_buffer_size to wrapper pattern**
   - File: src/cubie/buffer_registry.py
   - Action: Modify
   - Details:
     ```python
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
     ```

8. **Simplify BufferRegistry.persistent_local_buffer_size to wrapper pattern**
   - File: src/cubie/buffer_registry.py
   - Action: Modify
   - Details:
     ```python
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
     ```

9. **Simplify BufferRegistry.get_allocator to wrapper pattern**
   - File: src/cubie/buffer_registry.py
   - Action: Modify
   - Details:
     ```python
     def get_allocator(
         self,
         name: str,
         parent: object,
     ) -> Callable:
         """Generate CUDA device function for buffer allocation.

         Parameters
         ----------
         name
             Buffer name to generate allocator for.
         parent
             Parent instance that owns the buffer.

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
         return self._groups[parent].get_allocator(name)
     ```

10. **Remove private layout building methods from BufferRegistry**
    - File: src/cubie/buffer_registry.py
    - Action: Delete
    - Details: Remove these methods entirely (moved to BufferGroup):
      - `_build_shared_layout` (lines 296-337)
      - `_build_persistent_layout` (lines 339-381)
      - `_build_local_sizes` (lines 383-403)

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 6: Update Tests - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Groups 1-5

**Required Context**:
- File: tests/test_buffer_registry.py (entire file)
- File: .github/active_plans/buffer_registry_refactor/agent_plan.md (Test Updates Required section)

**Input Validation Required**:
- None (tests)

**Tasks**:

1. **Update imports to use new class names**
   - File: tests/test_buffer_registry.py
   - Action: Modify
   - Details:
     ```python
     from cubie.buffer_registry import (
         buffer_registry,
         CUDABuffer,
         BufferGroup,
         BufferRegistry,
     )
     ```

2. **Update TestBufferEntry class to TestCUDABuffer**
   - File: tests/test_buffer_registry.py
   - Action: Modify
   - Details:
     - Rename class: `TestBufferEntry` → `TestCUDABuffer`
     - Rename docstring: `"""Tests for BufferEntry attrs class."""` → `"""Tests for CUDABuffer attrs class."""`
     - In all test methods, change `BufferEntry(` to `CUDABuffer(`
     - Remove `factory=factory,` from all CUDABuffer instantiations

3. **Update all clear_factory calls to clear_parent**
   - File: tests/test_buffer_registry.py
   - Action: Modify
   - Details:
     - Line 151: `self.registry.clear_factory(self.factory)` → `self.registry.clear_parent(self.factory)`
     - Line 258: `self.registry.clear_factory(factory1)` → `self.registry.clear_parent(factory1)`

4. **Update test assertions using _contexts to _groups**
   - File: tests/test_buffer_registry.py
   - Action: Modify
   - Details: Change all `self.registry._contexts` to `self.registry._groups`
     - Lines 85-86, 114-115, 152, 167, 174, 187-188, 201, 260-261, 293-294, 302, 318, 324, 367-368, 372-373, 377, 384

5. **Remove cross-type aliasing restriction tests**
   - File: tests/test_buffer_registry.py
   - Action: Modify
   - Details: Remove these tests entirely (behavior changed):
     - `test_shared_alias_with_local_location_raises` (lines 344-350)
     - `test_persistent_alias_of_nonpersistent_local_raises` (lines 352-358)

6. **Add new cross-location aliasing tests**
   - File: tests/test_buffer_registry.py
   - Action: Modify
   - Details: Add new test class after TestCrossTypeAliasing
     ```python
     class TestCrossLocationAliasing:
         """Tests for cross-location aliasing behavior."""

         @pytest.fixture(autouse=True)
         def fresh_registry(self):
             self.registry = BufferRegistry()
             self.parent = MockFactory()
             yield

         def test_shared_buffer_can_alias_local_parent(self):
             """Shared buffer aliasing local parent falls back to own allocation."""
             self.registry.register('parent', self.parent, 100, 'local')
             self.registry.register(
                 'child', self.parent, 30, 'shared', aliases='parent'
             )
             # Child should be allocated in shared memory (fallback)
             size = self.registry.shared_buffer_size(self.parent)
             assert size == 30  # Child allocated separately

         def test_local_buffer_can_alias_shared_parent(self):
             """Local buffer aliasing shared parent uses local allocation."""
             self.registry.register('parent', self.parent, 100, 'shared')
             self.registry.register(
                 'child', self.parent, 30, 'local', aliases='parent'
             )
             # Child should be in local, not shared
             shared_size = self.registry.shared_buffer_size(self.parent)
             local_size = self.registry.local_buffer_size(self.parent)
             assert shared_size == 100  # Only parent
             assert local_size == 30  # Child in local

         def test_alias_fallback_when_parent_too_small(self):
             """Alias falls back to own allocation when parent insufficient."""
             self.registry.register('parent', self.parent, 50, 'shared')
             self.registry.register(
                 'child', self.parent, 80, 'shared', aliases='parent'
             )
             # Child needs 80 but parent only has 50, so allocates separately
             size = self.registry.shared_buffer_size(self.parent)
             assert size == 50  # Only non-aliased parent counts

             # But layout should have both
             group = self.registry._groups[self.parent]
             _ = self.registry.shared_buffer_size(self.parent)
             layout = group._shared_layout
             assert 'parent' in layout
             assert 'child' in layout
             # Child should have its own allocation after parent
             assert layout['parent'] == slice(0, 50)
             assert layout['child'] == slice(50, 130)

         def test_multiple_aliases_first_come_first_serve(self):
             """Multiple aliases consume parent space sequentially."""
             self.registry.register('parent', self.parent, 100, 'shared')
             self.registry.register(
                 'child1', self.parent, 40, 'shared', aliases='parent'
             )
             self.registry.register(
                 'child2', self.parent, 40, 'shared', aliases='parent'
             )
             self.registry.register(
                 'child3', self.parent, 40, 'shared', aliases='parent'
             )

             group = self.registry._groups[self.parent]
             _ = self.registry.shared_buffer_size(self.parent)
             layout = group._shared_layout

             # parent: slice(0, 100)
             # child1: slice(0, 40) within parent
             # child2: slice(40, 80) within parent
             # child3: doesn't fit (only 20 left), gets own allocation
             assert layout['parent'] == slice(0, 100)
             assert layout['child1'] == slice(0, 40)
             assert layout['child2'] == slice(40, 80)
             assert layout['child3'] == slice(100, 140)

         def test_persistent_alias_of_nonpersistent_local_allowed(self):
             """Persistent buffer can now alias non-persistent local."""
             self.registry.register('parent', self.parent, 100, 'local')
             self.registry.register(
                 'child', self.parent, 30, 'local',
                 persistent=True, aliases='parent'
             )
             # Should not raise; child gets own persistent allocation
             group = self.registry._groups[self.parent]
             assert group.entries['child'].aliases == 'parent'
             persist_size = self.registry.persistent_local_buffer_size(
                 self.parent
             )
             assert persist_size == 30
     ```

7. **Update test for error message text**
   - File: tests/test_buffer_registry.py
   - Action: Modify
   - Details: Update error message assertions that reference "factory" to "parent"
     - In `test_get_allocator_unregistered_factory_raises`: match text should be updated if needed

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Summary

| Group | Name | Status | Type | Estimated Complexity |
|-------|------|--------|------|---------------------|
| 1 | Rename BufferEntry to CUDABuffer | [ ] | SEQUENTIAL | Low |
| 2 | Rename BufferContext to BufferGroup | [ ] | SEQUENTIAL | Low |
| 3 | Add Methods to BufferGroup | [ ] | SEQUENTIAL | High |
| 4 | Add build_allocator to CUDABuffer | [ ] | SEQUENTIAL | Medium |
| 5 | Refactor BufferRegistry to Wrapper Pattern | [ ] | SEQUENTIAL | High |
| 6 | Update Tests | [ ] | SEQUENTIAL | Medium |

**Total Task Groups**: 6
**Dependency Chain**: 1 → 2 → 3 → 5, 1 → 4 → 5, 5 → 6
**Parallel Opportunities**: Groups 3 and 4 can be executed in parallel after Group 2 completes

**Key Changes**:
1. Class renames: BufferEntry → CUDABuffer, BufferContext → BufferGroup
2. Attribute renames: factory → parent, _contexts → _groups, _alias_offsets → _alias_consumption
3. Method rename: clear_factory → clear_parent
4. Method moves: Layout building, size calculation, and allocator generation move from BufferRegistry to BufferGroup
5. New method: CUDABuffer.build_allocator for CUDA device function compilation
6. Aliasing logic overhaul: Cross-location aliasing now allowed with first-come-first-serve fallback
