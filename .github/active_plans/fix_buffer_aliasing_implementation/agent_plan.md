# Buffer Aliasing Implementation Rework - Agent Plan

## Overview

This plan corrects fundamental semantic errors in the current three-parameter allocator implementation. The core issue is a misunderstanding of what buffer aliasing means and how the allocator parameters should work.

## Problem Statement

### Semantic Errors in Current Implementation

**Error 1: Wrong Parameter Names**
```python
# CURRENT (WRONG):
def build_allocator(shared_parent, persistent_parent, shared_fallback):
    # shared_fallback implies a separate fallback array
    # This is confusing - there's only ONE shared array

# CORRECT:
def build_allocator(shared, persistent, aliased_parent):
    # shared: the ONE combined shared array
    # persistent: the persistent local array
    # aliased_parent: the parent buffer to slice (or None)
```

**Error 2: Misunderstanding Aliasing**
```python
# CURRENT (WRONG):
# Tries to avoid overlap between primary and fallback layouts
# Allocates child in "fallback" when parent is local

# CORRECT:
# Child DELIBERATELY OVERLAPS parent when aliasing succeeds
# Purpose: Reuse memory for buffers with disjoint lifetimes
# Example: parent[0:100], child aliases parent and uses parent[20:40]
#          This OVERLAPS parent - child reuses parent's memory
```

**Error 3: Manual Aliasing in Step Functions**
```python
# CURRENT (WRONG) - in generic_dirk.py:
stage_base_aliases_acc = (
    multistage
    and config.accumulator_location == 'shared'
    and config.stage_base_location == 'shared'
)
if stage_base_aliases_acc:
    buffer_registry.register('stage_base', self, n, 'local',
                            aliases='accumulator', ...)
else:
    buffer_registry.register('stage_base', self, n, 
                            config.stage_base_location, ...)

# CORRECT:
# Always register stage_base with aliases='accumulator'
# Let buffer_registry decide if aliasing is possible
buffer_registry.register('stage_base', self, n,
                        config.stage_base_location,
                        aliases='accumulator', ...)
```

**Error 4: "Fallback" Layout Confusion**
```python
# CURRENT (WRONG):
# Tracks two layouts: (primary, fallback)
# Implies two separate shared arrays

# CORRECT:
# Single shared layout
# Some buffers slice parents (aliases), others get fresh space
# All use the SAME shared array
```

## Architectural Changes Required

### Component 1: CUDABuffer.build_allocator()

**Current Signature:**
```python
def build_allocator(
    shared_slice: Optional[slice],
    persistent_slice: Optional[slice],
    shared_fallback_slice: Optional[slice],
    local_size: Optional[int],
) -> Callable:
    """Returns allocator: (shared_parent, persistent_parent, shared_fallback) -> array"""
```

**Target Signature:**
```python
def build_allocator(
    shared_slice: Optional[slice],
    persistent_slice: Optional[slice],
    aliased_parent_slice: Optional[slice],
    local_size: Optional[int],
) -> Callable:
    """Returns allocator: (shared, persistent, aliased_parent) -> array"""
```

**Changed Behavior:**
- Parameter 1 (`shared`): Slice into combined shared array (for fresh allocations)
- Parameter 2 (`persistent`): Slice into persistent array (unchanged)
- Parameter 3 (`aliased_parent`): Parent buffer to slice when aliasing succeeds (or None)

**Internal Logic:**
```python
@cuda.jit(device=True, inline=True)
def allocate_buffer(shared, persistent, aliased_parent):
    # Priority:
    # 1. If aliased_parent provided: slice it (aliasing succeeded)
    # 2. Else if persistent_slice: slice persistent array
    # 3. Else if shared_slice: slice shared array (fresh allocation)
    # 4. Else: local array
    
    if _use_aliased_parent:
        array = aliased_parent[_aliased_parent_slice]
    elif _use_persistent:
        array = persistent[_persistent_slice]
    elif _use_shared:
        array = shared[_shared_slice]
    else:
        array = cuda.local.array(_local_size, _precision)
    return array
```

### Component 2: BufferGroup.build_shared_layout()

**Current Behavior:**
Returns `Tuple[Dict[str, slice], Dict[str, slice]]` - (primary_layout, fallback_layout)
- Primary: Aliased buffers slicing parents
- Fallback: Buffers that couldn't alias, allocated separately

**Target Behavior:**
Returns `Dict[str, slice]` - Single unified layout
- All shared buffers get slices into THE shared array
- Aliased buffers consume parent space (tracked via _alias_consumption)
- Non-aliased buffers get sequential allocation

**Algorithm:**
```python
def build_shared_layout(self) -> Dict[str, slice]:
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
                # Alias within parent (WITH OVERLAP)
                parent_slice = layout[entry.aliases]
                start = parent_slice.start + consumed
                layout[name] = slice(start, start + entry.size)
                self._alias_consumption[entry.aliases] = consumed + entry.size
            else:
                # Parent full, allocate fresh
                layout[name] = slice(offset, offset + entry.size)
                offset += entry.size
        else:
            # Parent is local, allocate fresh shared space
            layout[name] = slice(offset, offset + entry.size)
            offset += entry.size
    
    return layout
```

**Key Change:** Single dict return, not tuple. Consumption tracking ensures we don't over-alias.

### Component 3: BufferGroup Property Updates

**Remove:**
- `shared_primary_layout` property
- `shared_fallback_layout` property
- `shared_fallback_buffer_size()` method

**Replace with:**
- `shared_layout` property - Returns single unified layout

**Update:**
- `shared_buffer_size()` - Returns max slice.stop from single layout

### Component 4: BufferGroup.get_allocator()

**Current Logic:**
```python
def get_allocator(self, name: str) -> Callable:
    # Determines source for buffer:
    primary_layout, fallback_layout = self._shared_layout
    
    shared_slice = primary_layout.get(name)
    shared_fallback_slice = fallback_layout.get(name)
    persistent_slice = self._persistent_layout.get(name)
    local_size = self._local_sizes.get(name)
    
    return entry.build_allocator(
        shared_slice, persistent_slice, shared_fallback_slice, local_size
    )
```

**Target Logic:**
```python
def get_allocator(self, name: str) -> Callable:
    entry = self.entries[name]
    
    # Ensure layouts computed
    if self._shared_layout is None:
        self._shared_layout = self.build_shared_layout()
    # ... persistent, local layouts
    
    # Determine allocation strategy
    shared_slice = None
    persistent_slice = None
    aliased_parent_slice = None
    local_size = None
    
    if entry.aliases is not None:
        # This buffer aliases another
        parent_entry = self.entries[entry.aliases]
        parent_name = entry.aliases
        
        if entry.is_shared and parent_entry.is_shared:
            # Both shared - check if aliasing succeeded
            if name in self._shared_layout:
                # Check if we're slicing parent or got fresh allocation
                child_slice = self._shared_layout[name]
                parent_slice = self._shared_layout[parent_name]
                
                # If child slice is WITHIN parent, we're aliasing
                if (child_slice.start >= parent_slice.start and 
                    child_slice.stop <= parent_slice.stop):
                    # Aliasing succeeded - pass parent buffer
                    # Compute slice RELATIVE to parent start
                    relative_start = child_slice.start - parent_slice.start
                    relative_stop = child_slice.stop - parent_slice.start
                    aliased_parent_slice = slice(relative_start, relative_stop)
                else:
                    # Got fresh allocation (parent was full)
                    shared_slice = child_slice
        elif entry.is_persistent_local and parent_entry.is_persistent_local:
            # Similar logic for persistent aliasing
            # ...
        else:
            # Cross-location: use appropriate layout
            if entry.is_shared:
                shared_slice = self._shared_layout.get(name)
            elif entry.is_persistent_local:
                persistent_slice = self._persistent_layout.get(name)
            else:
                local_size = self._local_sizes.get(name)
    else:
        # Non-aliased buffer: use appropriate layout
        if entry.is_shared:
            shared_slice = self._shared_layout.get(name)
        elif entry.is_persistent_local:
            persistent_slice = self._persistent_layout.get(name)
        else:
            local_size = self._local_sizes.get(name)
    
    return entry.build_allocator(
        shared_slice, persistent_slice, aliased_parent_slice, local_size
    )
```

**Critical Logic:** When aliasing succeeds, pass `aliased_parent_slice` with relative offsets. When aliasing fails, pass `shared_slice` or appropriate alternative.

### Component 5: Update Allocator Call Sites

**All call sites currently:**
```python
buffer = alloc_buffer(shared_scratch, persistent_local, shared_fallback)
```

**Should become:**
```python
buffer = alloc_buffer(shared, persistent, aliased_parent)
```

**Locations (49 sites):**
- `ode_loop.py`: ~16 calls in `loop_fn` device function
- `generic_dirk.py`: ~8 calls in `step` device function
- `generic_rosenbrock_w.py`: ~6 calls in step device function
- `generic_erk.py`: ~6 calls if exists
- `generic_firk.py`: ~6 calls if exists
- Other algorithm files as needed

**Search pattern:** `(shared, persistent_local, shared)` or similar patterns

### Component 6: Remove Manual Aliasing Logic

**File: generic_dirk.py**

**Current (lines ~254-268):**
```python
stage_base_aliases_acc = (
    multistage
    and config.accumulator_location == 'shared'
    and config.stage_base_location == 'shared'
)
if stage_base_aliases_acc:
    buffer_registry.register(
        'stage_base', self, n, 'local',
        aliases='accumulator', precision=precision
    )
else:
    buffer_registry.register(
        'stage_base', self, n, config.stage_base_location,
        precision=precision
    )
```

**Target:**
```python
# Always register with aliases, let registry decide
buffer_registry.register(
    'stage_base', self, n, config.stage_base_location,
    aliases='accumulator', precision=precision
)
```

**File: generic_rosenbrock_w.py**

**Current (lines ~252-263):**
```python
if config.stage_store_location == 'local':
    buffer_registry.register(
        'stage_cache', self, n, 'local',
        persistent=True, precision=precision
    )
else:
    buffer_registry.register(
        'stage_cache', self, n, 'shared',
        aliases='stage_store', precision=precision
    )
```

**Target:**
```python
# Always register with aliases='stage_store'
# If stage_store is local, registry will handle it appropriately
buffer_registry.register(
    'stage_cache', self, n, config.stage_cache_location,
    aliases='stage_store', precision=precision
)
```

**Note:** May need to add `stage_cache_location` to config if not present.

## Integration Points

### With ode_loop.py
- Loop allocates `shared_fallback` array (rename to just `shared`)
- Passes this array as first argument to all allocators
- No semantic change to loop logic, just parameter naming

### With Algorithm Step Functions
- Step functions call allocators with three arguments
- No conditional aliasing logic
- Clean, straightforward buffer allocation

### With buffer_registry Tests
- Update `test_buffer_registry.py` to reflect new semantics
- Test aliasing scenarios:
  - Parent shared + child shared with space → child slices parent
  - Parent shared + child shared no space → child gets fresh allocation
  - Parent local + child shared → child gets fresh allocation
  - Parent persistent + child persistent → similar logic

## Edge Cases to Consider

### Edge Case 1: Multiple Levels of Aliasing
**Scenario:** Buffer A, buffer B aliases A, buffer C aliases B

**Handling:** Not currently supported. Each buffer can alias only ONE parent. Transitive aliasing would require registry to resolve chains.

**Decision:** Document as not supported. Algorithm code should not attempt.

### Edge Case 2: Circular Aliases
**Scenario:** Buffer A aliases B, buffer B aliases A

**Handling:** Current validation prevents this (alias target must exist before aliasing buffer).

**Decision:** Keep current validation.

### Edge Case 3: Parent Changes Location After Child Registration
**Scenario:** Register parent as shared, register child with aliases, then update parent to local.

**Handling:** `update_buffer()` invalidates layouts, next `get_allocator()` recomputes.

**Decision:** This should work correctly with current invalidation logic.

### Edge Case 4: Precision Mismatch Between Parent and Child
**Scenario:** Parent is float32, child is float64

**Handling:** Currently not validated. Both would need same precision for slice to work.

**Decision:** Add validation? Or document as user responsibility?

## Expected Behavior Changes

### Behavioral Change 1: Aliasing Always Attempted
**Before:** Manual if/else in step functions decided whether to alias based on locations.

**After:** Registry always attempts aliasing when `aliases` parameter provided. Falls back to fresh allocation if parent location incompatible or insufficient space.

### Behavioral Change 2: No "Fallback" Array
**Before:** Code implied two shared arrays (primary and fallback).

**After:** Single shared array with unified layout.

### Behavioral Change 3: Deliberate Overlap
**Before:** Unclear whether overlap was intentional.

**After:** Overlap is INTENTIONAL. Child reuses parent memory. This is the point of aliasing.

## Dependencies and Data Structures

### Updated BufferGroup Attributes
```python
@attrs.define
class BufferGroup:
    parent: object
    entries: Dict[str, CUDABuffer]
    _shared_layout: Optional[Dict[str, slice]] = None  # Changed from Tuple
    _persistent_layout: Optional[Dict[str, slice]] = None
    _local_sizes: Optional[Dict[str, int]] = None
    _alias_consumption: Dict[str, int] = attrs.field(factory=dict, init=False)
```

### Updated BufferRegistry Methods
```python
class BufferRegistry:
    def shared_buffer_size(self, parent: object) -> int:
        """Total elements in THE shared array."""
        if parent not in self._groups:
            return 0
        layout = self._groups[parent].shared_layout
        if not layout:
            return 0
        return max(s.stop for s in layout.values())
    
    # REMOVE:
    # def shared_fallback_buffer_size(self, parent: object) -> int
```

## Validation Strategy

### Unit Tests
- `test_buffer_registry.py`:
  - Test aliasing with parent shared, child shared, sufficient space
  - Test aliasing with parent shared, child shared, insufficient space
  - Test aliasing with parent local, child shared
  - Test aliasing with parent persistent, child persistent
  - Test non-aliased buffer allocation

### Integration Tests
- Run existing algorithm tests with various buffer location configurations
- Verify no regressions in solver tests
- Test adaptive vs fixed step controllers with different buffer locations

### Manual Verification
- Inspect generated allocator device functions for correct parameter order
- Verify CUDA compilation succeeds after changes
- Check memory consumption matches expectations

## Migration Path

### Phase 1: Update buffer_registry.py
1. Rename parameters in `build_allocator` signature
2. Update allocator device function parameter names
3. Rewrite `build_shared_layout` to return single dict
4. Remove `shared_fallback_layout` property
5. Update `shared_buffer_size()` logic
6. Rewrite `get_allocator()` with new logic

### Phase 2: Update ode_loop.py
1. Rename `shared_fallback` variable to `shared`
2. Update all allocator calls: `alloc_X(shared, persistent, shared)` → `alloc_X(shared, persistent, aliased_parent)`
3. For most buffers: third argument is `shared` (no aliasing)
4. For child buffers: third argument from `alloc_parent_shared` etc.

### Phase 3: Update Algorithm Files
1. Remove conditional aliasing logic in `generic_dirk.py`
2. Remove conditional aliasing logic in `generic_rosenbrock_w.py`
3. Update allocator calls in algorithm step functions
4. Ensure each buffer registered exactly once

### Phase 4: Update Tests
1. Fix `test_buffer_registry.py` expectations
2. Run full test suite, fix failures
3. Add new aliasing scenario tests if missing

## Success Criteria

1. ✅ All allocator parameters named `(shared, persistent, aliased_parent)`
2. ✅ Single `shared_layout` property (no primary/fallback split)
3. ✅ Aliased buffers slice parent when parent is shared and has space
4. ✅ Aliased buffers get fresh allocation when parent local or full
5. ✅ No manual aliasing logic in algorithm step functions
6. ✅ All tests pass (excluding markers: batchsolving.test_run, specific_algos, no_cudasim)
7. ✅ No CUDA compilation errors
8. ✅ Memory consumption matches expectations
