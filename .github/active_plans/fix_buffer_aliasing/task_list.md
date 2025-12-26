# Implementation Task List
# Feature: Buffer Aliasing Logic Refactor
# Plan Reference: .github/active_plans/fix_buffer_aliasing/agent_plan.md

## Task Group 1: Add build_layouts() Orchestration Method
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: src/cubie/buffer_registry.py (lines 145-180, BufferGroup class definition and attributes)
- File: src/cubie/buffer_registry.py (lines 181-186, invalidate_layouts method)
- File: src/cubie/buffer_registry.py (lines 284-348, build_shared_layout method)
- File: src/cubie/buffer_registry.py (lines 363-416, build_persistent_layout method)
- File: .github/context/cubie_internal_structure.md (Buffer Allocation Pattern section)

**Input Validation Required**:
- None (internal orchestration method with no external inputs)

**Tasks**:
1. **Add build_layouts() method to BufferGroup**
   - File: src/cubie/buffer_registry.py
   - Action: Create new method after invalidate_layouts() (around line 187)
   - Details:
     ```python
     def build_layouts(self) -> None:
         """Build all buffer layouts in deterministic order.

         Orchestrates layout building to ensure consistent results
         regardless of which property is accessed first:
         1. Build non-aliased shared buffers into _shared_layout
         2. Build non-aliased persistent buffers into _persistent_layout
         3. Call layout_aliases() to handle all aliased entries

         All three layout caches are fully populated after this
         method completes.
         """
         # If all layouts already built, nothing to do
         if (self._shared_layout is not None
                 and self._persistent_layout is not None
                 and self._local_sizes is not None):
             return

         # Clear state for fresh build
         self._alias_consumption.clear()
         self._shared_layout = {}
         self._persistent_layout = {}
         self._local_sizes = {}

         # Phase 1: Non-aliased shared buffers
         shared_offset = 0
         for name, entry in self.entries.items():
             if entry.is_shared and entry.aliases is None:
                 self._shared_layout[name] = slice(
                     shared_offset, shared_offset + entry.size
                 )
                 self._alias_consumption[name] = 0
                 shared_offset += entry.size

         # Phase 2: Non-aliased persistent buffers
         persistent_offset = 0
         for name, entry in self.entries.items():
             if entry.is_persistent_local and entry.aliases is None:
                 self._persistent_layout[name] = slice(
                     persistent_offset, persistent_offset + entry.size
                 )
                 self._alias_consumption[name] = 0
                 persistent_offset += entry.size

         # Phase 3: Process all aliased entries
         self.layout_aliases()
     ```
   - Edge cases:
     - Empty entries dict: method completes with all empty layout dicts
     - All layouts already populated: early return without changes
     - No non-aliased entries: phases 1 and 2 produce empty layouts, all work in phase 3
   - Integration: Called by property accessors when layouts are None

**Tests to Create**:
- Test file: tests/test_buffer_registry.py
- Test function: test_build_layouts_populates_all_caches
- Description: Verify that calling build_layouts() populates _shared_layout, _persistent_layout, and _local_sizes
- Test function: test_build_layouts_early_return_when_populated
- Description: Verify that build_layouts() returns early without rebuilding when all caches are populated

**Tests to Run**:
- tests/test_buffer_registry.py::TestLayoutComputation::test_build_layouts_populates_all_caches
- tests/test_buffer_registry.py::TestLayoutComputation::test_build_layouts_early_return_when_populated

**Outcomes**: 

---

## Task Group 2: Add layout_aliases() Method
**Status**: [ ]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/buffer_registry.py (lines 145-180, BufferGroup class and CUDABuffer)
- File: src/cubie/buffer_registry.py (lines 24-74, CUDABuffer class with is_shared, is_persistent_local, is_local properties)
- File: .github/active_plans/fix_buffer_aliasing/agent_plan.md (layout_aliases section, lines 93-124)

**Input Validation Required**:
- None (internal method operating on validated entries)

**Tasks**:
1. **Add layout_aliases() method to BufferGroup**
   - File: src/cubie/buffer_registry.py
   - Action: Create new method immediately after build_layouts()
   - Details:
     ```python
     def layout_aliases(self) -> None:
         """Process all aliased entries and assign to appropriate layouts.

         For each entry with aliases is not None:
         - If parent is shared with available space: overlap within parent
         - Else fallback based on entry's own type:
           - is_shared: allocate in _shared_layout
           - is_persistent_local: allocate in _persistent_layout
           - is_local: add to local pile (processed at end)

         Local pile entries are added to _local_sizes after all
         aliasing decisions are made.
         """
         local_pile = []

         # Compute current offsets from existing layouts
         shared_offset = 0
         if self._shared_layout:
             shared_offset = max(s.stop for s in self._shared_layout.values())

         persistent_offset = 0
         if self._persistent_layout:
             persistent_offset = max(
                 s.stop for s in self._persistent_layout.values()
             )

         # Process aliased entries
         for name, entry in self.entries.items():
             if entry.aliases is None:
                 continue

             parent_entry = self.entries[entry.aliases]
             aliased = False

             # Check if parent is in shared layout and has space
             if entry.aliases in self._shared_layout:
                 consumed = self._alias_consumption.get(entry.aliases, 0)
                 available = parent_entry.size - consumed

                 if entry.size <= available:
                     # Overlap within parent's shared memory
                     parent_slice = self._shared_layout[entry.aliases]
                     start = parent_slice.start + consumed
                     self._shared_layout[name] = slice(
                         start, start + entry.size
                     )
                     self._alias_consumption[entry.aliases] = (
                         consumed + entry.size
                     )
                     aliased = True

             # Fallback based on entry's own type
             if not aliased:
                 if entry.is_shared:
                     self._shared_layout[name] = slice(
                         shared_offset, shared_offset + entry.size
                     )
                     shared_offset += entry.size
                 elif entry.is_persistent_local:
                     self._persistent_layout[name] = slice(
                         persistent_offset, persistent_offset + entry.size
                     )
                     persistent_offset += entry.size
                 else:
                     # is_local: collect for batch processing
                     local_pile.append(entry)

         # Process local pile
         for entry in local_pile:
             self._local_sizes[entry.name] = max(entry.size, 1)
     ```
   - Edge cases:
     - Parent not in shared layout: fallback to entry's own type
     - Parent full (consumed >= size): fallback to entry's own type
     - Entry is local: added to local_pile, processed at end
     - No aliased entries: method completes without modifying layouts
   - Integration: Called by build_layouts() as phase 3

**Tests to Create**:
- Test file: tests/test_buffer_registry.py
- Test function: test_layout_aliases_overlaps_within_parent
- Description: Verify aliased buffer overlaps parent when space available
- Test function: test_layout_aliases_fallback_when_parent_full
- Description: Verify aliased buffer gets own allocation when parent is full
- Test function: test_layout_aliases_respects_entry_type
- Description: Verify fallback respects is_shared, is_persistent_local, is_local

**Tests to Run**:
- tests/test_buffer_registry.py::TestLayoutComputation::test_layout_aliases_overlaps_within_parent
- tests/test_buffer_registry.py::TestLayoutComputation::test_layout_aliases_fallback_when_parent_full
- tests/test_buffer_registry.py::TestLayoutComputation::test_layout_aliases_respects_entry_type

**Outcomes**: 

---

## Task Group 3: Update Property Accessors to Use build_layouts()
**Status**: [ ]
**Dependencies**: Task Groups 1, 2

**Required Context**:
- File: src/cubie/buffer_registry.py (lines 350-362, shared_layout property)
- File: src/cubie/buffer_registry.py (lines 418-446, build_local_sizes method)
- File: src/cubie/buffer_registry.py (lines 447-489, size methods using layouts)
- File: src/cubie/buffer_registry.py (lines 490-548, get_allocator method)

**Input Validation Required**:
- None (property accessors with no inputs)

**Tasks**:
1. **Update shared_layout property to call build_layouts()**
   - File: src/cubie/buffer_registry.py
   - Action: Modify existing shared_layout property (around line 350)
   - Details:
     ```python
     @property
     def shared_layout(self) -> Dict[str, slice]:
         """Return shared memory layout.

         Returns
         -------
         Dict[str, slice]
             Mapping of buffer names to shared memory slices.
         """
         if self._shared_layout is None:
             self.build_layouts()
         return self._shared_layout
     ```
   - Edge cases: None (build_layouts handles all edge cases)
   - Integration: Unchanged signature, triggers full layout build

2. **Add persistent_layout property**
   - File: src/cubie/buffer_registry.py
   - Action: Add new property after shared_layout property
   - Details:
     ```python
     @property
     def persistent_layout(self) -> Dict[str, slice]:
         """Return persistent local memory layout.

         Returns
         -------
         Dict[str, slice]
             Mapping of buffer names to persistent local slices.
         """
         if self._persistent_layout is None:
             self.build_layouts()
         return self._persistent_layout
     ```
   - Edge cases: None
   - Integration: New property, used by size methods and get_allocator

3. **Add local_sizes property**
   - File: src/cubie/buffer_registry.py
   - Action: Add new property after persistent_layout property
   - Details:
     ```python
     @property
     def local_sizes(self) -> Dict[str, int]:
         """Return local buffer sizes.

         Returns
         -------
         Dict[str, int]
             Mapping of buffer names to local array sizes.
         """
         if self._local_sizes is None:
             self.build_layouts()
         return self._local_sizes
     ```
   - Edge cases: None
   - Integration: New property, used by size methods and get_allocator

4. **Update shared_buffer_size() to use property**
   - File: src/cubie/buffer_registry.py
   - Action: Modify to use shared_layout property instead of direct access
   - Details:
     ```python
     def shared_buffer_size(self) -> int:
         """Return total shared memory elements.

         Returns
         -------
         int
             Total shared memory elements needed (end of last slice).
         """
         layout = self.shared_layout
         if not layout:
             return 0
         return max(s.stop for s in layout.values())
     ```
   - Edge cases: Empty layout returns 0
   - Integration: Uses property which triggers build_layouts if needed

5. **Update local_buffer_size() to use property**
   - File: src/cubie/buffer_registry.py
   - Action: Modify to use local_sizes property
   - Details:
     ```python
     def local_buffer_size(self) -> int:
         """Return total local memory elements.

         Returns
         -------
         int
             Total local memory elements (max(size, 1) for each).
         """
         return sum(self.local_sizes.values())
     ```
   - Edge cases: Empty sizes returns 0 (sum of empty dict)
   - Integration: Uses property which triggers build_layouts if needed

6. **Update persistent_local_buffer_size() to use property**
   - File: src/cubie/buffer_registry.py
   - Action: Modify to use persistent_layout property
   - Details:
     ```python
     def persistent_local_buffer_size(self) -> int:
         """Return total persistent local elements.

         Returns
         -------
         int
             Total persistent_local elements needed (end of last slice).
         """
         layout = self.persistent_layout
         if not layout:
             return 0
         return max(s.stop for s in layout.values())
     ```
   - Edge cases: Empty layout returns 0
   - Integration: Uses property which triggers build_layouts if needed

7. **Update get_allocator() to use properties**
   - File: src/cubie/buffer_registry.py
   - Action: Modify to use new properties instead of direct layout access
   - Details:
     ```python
     def get_allocator(self, name: str, zero: bool = False) -> Callable:
         # ... docstring unchanged ...
         if name not in self.entries:
             raise KeyError(f"Buffer '{name}' not registered for parent.")

         entry = self.entries[name]

         # Get slice from appropriate layout (properties trigger build)
         shared_slice = self.shared_layout.get(name)
         persistent_slice = self.persistent_layout.get(name)
         local_size = self.local_sizes.get(name)

         return entry.build_allocator(
             shared_slice, persistent_slice, local_size, zero
         )
     ```
   - Edge cases: KeyError for unregistered buffer
   - Integration: Uses properties which trigger build_layouts if needed

**Tests to Create**:
- Test file: tests/test_buffer_registry.py
- Test function: test_property_access_order_invariant
- Description: Verify accessing shared_layout, persistent_layout, local_sizes in any order produces same result

**Tests to Run**:
- tests/test_buffer_registry.py::TestLayoutComputation::test_property_access_order_invariant

**Outcomes**: 

---

## Task Group 4: Remove Aliasing Logic from Original Build Methods
**Status**: [ ]
**Dependencies**: Task Groups 1, 2, 3

**Required Context**:
- File: src/cubie/buffer_registry.py (lines 284-348, build_shared_layout method)
- File: src/cubie/buffer_registry.py (lines 363-416, build_persistent_layout method)
- File: src/cubie/buffer_registry.py (lines 418-446, build_local_sizes method)

**Input Validation Required**:
- None (removing methods that are no longer needed)

**Tasks**:
1. **Remove build_shared_layout() method**
   - File: src/cubie/buffer_registry.py
   - Action: Delete the entire build_shared_layout() method (lines 284-348)
   - Details: This method is replaced by the combination of build_layouts() phase 1 and layout_aliases(). Its functionality is now handled by build_layouts().
   - Edge cases: None
   - Integration: Method no longer called; shared_layout property now calls build_layouts()

2. **Remove build_persistent_layout() method**
   - File: src/cubie/buffer_registry.py
   - Action: Delete the entire build_persistent_layout() method (lines 363-416)
   - Details: This method is replaced by build_layouts() phase 2 and layout_aliases(). Its functionality is now handled by build_layouts().
   - Edge cases: None
   - Integration: Method no longer called; persistent_layout property now calls build_layouts()

3. **Remove build_local_sizes() method**
   - File: src/cubie/buffer_registry.py
   - Action: Delete the entire build_local_sizes() method (lines 418-446)
   - Details: This method is replaced by layout_aliases() which populates _local_sizes. Its functionality is now handled by layout_aliases().
   - Edge cases: None
   - Integration: Method no longer called; local_sizes property now calls build_layouts()

**Tests to Create**:
- None (removal task; existing tests will verify behavior is preserved)

**Tests to Run**:
- tests/test_buffer_registry.py (full file to ensure no regressions)

**Outcomes**: 

---

## Task Group 5: Add Deterministic Behavior Tests
**Status**: [ ]
**Dependencies**: Task Groups 1, 2, 3, 4

**Required Context**:
- File: tests/test_buffer_registry.py (entire file for test patterns)
- File: src/cubie/buffer_registry.py (lines 145-550, BufferGroup class)
- File: .github/active_plans/fix_buffer_aliasing/human_overview.md (User Stories section)

**Input Validation Required**:
- None (test file)

**Tasks**:
1. **Add test class for deterministic layout building**
   - File: tests/test_buffer_registry.py
   - Action: Create new test class TestDeterministicLayouts
   - Details:
     ```python
     class TestDeterministicLayouts:
         """Tests for deterministic layout building order."""

         @pytest.fixture(autouse=True)
         def fresh_registry(self):
             self.registry = BufferRegistry()
             self.parent = MockFactory()
             yield

         def test_property_access_order_shared_first(self):
             """Accessing shared_layout first produces same result."""
             self.registry.register('parent', self.parent, 100, 'shared')
             self.registry.register(
                 'child', self.parent, 30, 'shared', aliases='parent'
             )
             self.registry.register('local', self.parent, 20, 'local')

             group = self.registry._groups[self.parent]
             
             # Access shared first
             shared = group.shared_layout.copy()
             persistent = group.persistent_layout.copy()
             local = group.local_sizes.copy()

             # Invalidate and access in different order
             group.invalidate_layouts()
             
             # Access local first
             local2 = group.local_sizes.copy()
             persistent2 = group.persistent_layout.copy()
             shared2 = group.shared_layout.copy()

             assert shared == shared2
             assert persistent == persistent2
             assert local == local2

         def test_property_access_order_persistent_first(self):
             """Accessing persistent_layout first produces same result."""
             self.registry.register(
                 'persist', self.parent, 50, 'local', persistent=True
             )
             self.registry.register('shared', self.parent, 100, 'shared')
             self.registry.register(
                 'child', self.parent, 30, 'shared', aliases='shared'
             )

             group = self.registry._groups[self.parent]
             
             # Access persistent first
             persistent = group.persistent_layout.copy()
             shared = group.shared_layout.copy()
             local = group.local_sizes.copy()

             # Invalidate and access shared first
             group.invalidate_layouts()
             
             shared2 = group.shared_layout.copy()
             local2 = group.local_sizes.copy()
             persistent2 = group.persistent_layout.copy()

             assert shared == shared2
             assert persistent == persistent2
             assert local == local2
     ```
   - Edge cases: Test with mixed buffer types
   - Integration: Validates user story US-1

2. **Add test for centralized aliasing logic**
   - File: tests/test_buffer_registry.py
   - Action: Add tests to existing TestCrossLocationAliasing class
   - Details:
     ```python
     def test_aliased_shared_child_of_local_parent_gets_shared_allocation(self):
         """Shared child aliasing local parent allocates in shared layout."""
         self.registry.register('parent', self.parent, 100, 'local')
         self.registry.register(
             'child', self.parent, 30, 'shared', aliases='parent'
         )
         
         shared_size = self.registry.shared_buffer_size(self.parent)
         local_size = self.registry.local_buffer_size(self.parent)
         
         # Child should be in shared (fallback)
         assert shared_size == 30
         # Parent in local
         assert local_size == 100

     def test_aliased_persistent_child_of_local_parent_gets_persistent(self):
         """Persistent child aliasing local parent allocates in persistent."""
         self.registry.register('parent', self.parent, 100, 'local')
         self.registry.register(
             'child', self.parent, 30, 'local',
             persistent=True, aliases='parent'
         )
         
         persist_size = self.registry.persistent_local_buffer_size(
             self.parent
         )
         local_size = self.registry.local_buffer_size(self.parent)
         
         # Child should be in persistent (fallback)
         assert persist_size == 30
         # Parent in local
         assert local_size == 100

     def test_aliased_local_child_of_shared_parent_gets_local(self):
         """Local child aliasing shared parent allocates in local."""
         self.registry.register('parent', self.parent, 100, 'shared')
         self.registry.register(
             'child', self.parent, 30, 'local', aliases='parent'
         )
         
         shared_size = self.registry.shared_buffer_size(self.parent)
         local_size = self.registry.local_buffer_size(self.parent)
         
         # Parent in shared
         assert shared_size == 100
         # Child in local (fallback)
         assert local_size == 30
     ```
   - Edge cases: All cross-location combinations
   - Integration: Validates user stories US-2, US-3, US-4

3. **Add test for build_layouts method**
   - File: tests/test_buffer_registry.py
   - Action: Add tests to TestLayoutComputation class
   - Details:
     ```python
     def test_build_layouts_populates_all_caches(self):
         """build_layouts() populates all three layout caches."""
         self.registry.register('shared', self.factory, 100, 'shared')
         self.registry.register(
             'persist', self.factory, 50, 'local', persistent=True
         )
         self.registry.register('local', self.factory, 25, 'local')
         
         group = self.registry._groups[self.factory]
         
         # All caches should be None initially
         assert group._shared_layout is None
         assert group._persistent_layout is None
         assert group._local_sizes is None
         
         # Call build_layouts
         group.build_layouts()
         
         # All caches should be populated
         assert group._shared_layout is not None
         assert group._persistent_layout is not None
         assert group._local_sizes is not None
         
         # Verify contents
         assert 'shared' in group._shared_layout
         assert 'persist' in group._persistent_layout
         assert 'local' in group._local_sizes

     def test_build_layouts_early_return_when_populated(self):
         """build_layouts() returns early when all caches populated."""
         self.registry.register('shared', self.factory, 100, 'shared')
         group = self.registry._groups[self.factory]
         
         # First call populates
         group.build_layouts()
         original_layout = group._shared_layout
         
         # Second call should be no-op
         group.build_layouts()
         
         # Should be exact same object (not rebuilt)
         assert group._shared_layout is original_layout
     ```
   - Edge cases: Empty groups, already populated caches
   - Integration: Unit tests for new method

**Tests to Create**:
(Tests defined in task details above)

**Tests to Run**:
- tests/test_buffer_registry.py::TestDeterministicLayouts
- tests/test_buffer_registry.py::TestCrossLocationAliasing
- tests/test_buffer_registry.py::TestLayoutComputation

**Outcomes**: 

---

## Summary

| Task Group | Description | Dependencies | Estimated Complexity |
|------------|-------------|--------------|---------------------|
| 1 | Add build_layouts() method | None | Medium |
| 2 | Add layout_aliases() method | Group 1 | Medium-High |
| 3 | Update property accessors | Groups 1, 2 | Low |
| 4 | Remove old build methods | Groups 1-3 | Low |
| 5 | Add deterministic tests | Groups 1-4 | Medium |

**Dependency Chain**: 1 → 2 → 3 → 4 → 5

**Tests to Create (Total)**:
- test_build_layouts_populates_all_caches
- test_build_layouts_early_return_when_populated
- test_layout_aliases_overlaps_within_parent
- test_layout_aliases_fallback_when_parent_full
- test_layout_aliases_respects_entry_type
- test_property_access_order_invariant
- test_property_access_order_shared_first
- test_property_access_order_persistent_first
- test_aliased_shared_child_of_local_parent_gets_shared_allocation
- test_aliased_persistent_child_of_local_parent_gets_persistent
- test_aliased_local_child_of_shared_parent_gets_local

**Tests to Run (Final Verification)**:
- tests/test_buffer_registry.py (full file)
