# Buffer Aliasing Logic Refactor - Agent Plan

## Overview

This plan refactors the `BufferGroup` class in `src/cubie/buffer_registry.py` to fix incorrect aliasing logic. The core issue is that layout building methods have duplicated, intertwined aliasing logic that can produce inconsistent results depending on property access order.

## Component Architecture

### BufferGroup Class Structure (Current)

The `BufferGroup` class manages buffer entries for a single parent object:

```
BufferGroup
├── parent: object                     # Owner instance
├── entries: Dict[str, CUDABuffer]     # Registered buffers
├── _shared_layout: Optional[Dict]     # Cached shared slices
├── _persistent_layout: Optional[Dict] # Cached persistent slices
├── _local_sizes: Optional[Dict]       # Cached local sizes
├── _alias_consumption: Dict[str, int] # Consumed space per parent
│
├── invalidate_layouts()               # Clear all caches
├── register()                         # Add new buffer
├── update_buffer()                    # Modify existing buffer
│
├── build_shared_layout()              # PROBLEM: Contains aliasing logic
├── build_persistent_layout()          # PROBLEM: Contains aliasing logic
├── build_local_sizes()                # PROBLEM: Depends on above two
│
├── shared_layout (property)           # Accesses build_shared_layout
├── shared_buffer_size()               # Size calculation
├── persistent_local_buffer_size()     # Size calculation
├── local_buffer_size()                # Size calculation
└── get_allocator()                    # Generate device function
```

### BufferGroup Class Structure (Target)

```
BufferGroup
├── parent: object
├── entries: Dict[str, CUDABuffer]
├── _shared_layout: Optional[Dict]
├── _persistent_layout: Optional[Dict]
├── _local_sizes: Optional[Dict]
├── _alias_consumption: Dict[str, int]
├── _layouts_valid: bool               # NEW: Track if layouts built
│
├── invalidate_layouts()               # Clear all caches + _layouts_valid
├── register()                         # Unchanged
├── update_buffer()                    # Unchanged
│
├── build_layouts()                    # NEW: Orchestrates all layout building
├── layout_aliases()                   # NEW: Centralizes aliasing logic
│
├── shared_layout (property)           # Calls build_layouts() if needed
├── persistent_layout (property)       # Calls build_layouts() if needed
├── local_sizes (property)             # Calls build_layouts() if needed (NEW)
│
├── shared_buffer_size()               # Uses property (unchanged logic)
├── persistent_local_buffer_size()     # Uses property (unchanged logic)
├── local_buffer_size()                # Uses property (unchanged logic)
└── get_allocator()                    # Uses properties (unchanged logic)
```

---

## Detailed Component Descriptions

### build_layouts() Method

**Purpose**: Single entry point that ensures layouts are built in correct order.

**Behavior**:
1. Check if any layout is empty (None). If all are populated, return early.
2. Clear `_alias_consumption` dictionary.
3. Initialize `_shared_layout`, `_persistent_layout`, and `_local_sizes` as empty dicts.
4. **Phase 1**: Loop through entries. For each entry with `is_shared=True` AND `aliases is None`:
   - Allocate slice in `_shared_layout`
   - Initialize `_alias_consumption[name] = 0`
5. **Phase 2**: Loop through entries. For each entry with `is_persistent_local=True` AND `aliases is None`:
   - Allocate slice in `_persistent_layout`
   - Initialize `_alias_consumption[name] = 0`
6. **Phase 3**: Call `layout_aliases()` to handle all aliased entries.
7. Mark `_layouts_valid = True`.

**Key Invariant**: After `build_layouts()` completes, all three layout caches are fully populated. No partial state.

---

### layout_aliases() Method

**Purpose**: Process all entries with `aliases is not None` and assign them to appropriate layouts.

**Behavior**:
1. Create empty `local_pile` list to collect entries needing local allocation.
2. Loop through all entries where `entry.aliases is not None`:
   
   a. Get parent entry from `self.entries[entry.aliases]`
   
   b. Check if parent is in `_shared_layout` (i.e., parent is shared):
      - Calculate `consumed = _alias_consumption.get(entry.aliases, 0)`
      - Calculate `available = parent.size - consumed`
      - If `entry.size <= available`:
        - **Overlap**: Assign slice within parent's memory region
        - `parent_slice = _shared_layout[entry.aliases]`
        - `start = parent_slice.start + consumed`
        - `_shared_layout[name] = slice(start, start + entry.size)`
        - Update `_alias_consumption[entry.aliases] = consumed + entry.size`
        - Continue to next entry
   
   c. **Fallback**: Parent doesn't have space or isn't shared. Allocate based on entry's own type:
      - If `entry.is_shared`:
        - Add to `_shared_layout` at next available offset
      - Elif `entry.is_persistent_local`:
        - Add to `_persistent_layout` at next available offset
      - Else (`entry.is_local`):
        - Add entry to `local_pile` for later processing

3. After processing all aliased entries, process `local_pile`:
   - For each entry in `local_pile`:
     - `_local_sizes[entry.name] = max(entry.size, 1)`

**Important**: The offsets for shared and persistent layouts must be tracked during processing. This requires computing the "next available offset" for each layout type.

---

### Property Accessor Changes

**shared_layout property**:
```
@property
def shared_layout(self) -> Dict[str, slice]:
    if self._shared_layout is None:
        self.build_layouts()
    return self._shared_layout
```

**persistent_layout property** (NEW - was only method before):
```
@property  
def persistent_layout(self) -> Dict[str, slice]:
    if self._persistent_layout is None:
        self.build_layouts()
    return self._persistent_layout
```

**local_sizes property** (NEW):
```
@property
def local_sizes(self) -> Dict[str, int]:
    if self._local_sizes is None:
        self.build_layouts()
    return self._local_sizes
```

---

## Expected Interactions

### Registration Flow
1. `BufferRegistry.register()` calls `BufferGroup.register()`
2. `BufferGroup.register()` creates `CUDABuffer` and adds to `entries`
3. `BufferGroup.invalidate_layouts()` clears all caches
4. Next property access triggers `build_layouts()`

### Allocator Generation Flow
1. `BufferRegistry.get_allocator()` calls `BufferGroup.get_allocator()`
2. `get_allocator()` accesses `shared_layout`, `persistent_layout`, `local_sizes` properties
3. Properties trigger `build_layouts()` if needed
4. Allocator created with correct slice/size information

### Update Flow
1. `BufferRegistry.update_buffer()` calls `BufferGroup.update_buffer()`
2. If buffer changed, `invalidate_layouts()` called
3. Next property access triggers fresh `build_layouts()`

---

## Data Structures

### _alias_consumption Dictionary
- **Key**: Parent buffer name (str)
- **Value**: Number of elements already allocated to child aliases (int)
- **Lifecycle**: Cleared at start of `build_layouts()`, populated during layout phases
- **Usage**: Track how much parent buffer space remains for aliasing

### local_pile List (internal to layout_aliases)
- **Contents**: CUDABuffer entries that need local allocation
- **Lifecycle**: Created at start of `layout_aliases()`, consumed at end
- **Purpose**: Collect local entries for batch processing after aliasing decisions

---

## Edge Cases

### 1. Empty BufferGroup
- `entries` is empty dict
- All layouts should be empty dicts
- All size methods return 0

### 2. No Aliased Buffers
- `layout_aliases()` loops through entries but finds none with aliases
- Local pile remains empty
- Layouts only contain non-aliased entries

### 3. All Buffers Are Aliased
- Phase 1 and Phase 2 produce empty initial layouts
- All work done in `layout_aliases()`
- If all alias parents, circular reference should be caught at registration

### 4. Alias Chain (A aliases B aliases C)
- Registration order enforced: parent must exist before child
- Processing order respects registration order (dict iteration order)
- Deep chains work because parent is processed before children

### 5. Cross-Location Aliasing
- Shared child aliasing local parent: child goes to shared layout (fallback)
- Local child aliasing shared parent: child goes to local pile (fallback)
- Persistent child aliasing non-persistent: child goes to persistent layout (fallback)

### 6. Parent Too Small for All Children
- First children that fit get overlap slices
- Later children that don't fit get fallback allocation
- Order is registration order (dict iteration)

---

## Dependencies and Imports

No new imports required. Module already imports:
- `typing`: Dict, Optional, Tuple, Any, Set, Callable
- `attrs`: define, field, validators
- `numpy` as np
- `numba`: cuda, int32

---

## Integration Points

### BufferRegistry (unchanged interface)
- All public methods delegate to BufferGroup
- No changes needed to BufferRegistry class
- Singleton `buffer_registry` instance unchanged

### CUDABuffer (unchanged)
- Immutable record describing buffer requirements
- Properties `is_shared`, `is_persistent_local`, `is_local` used for fallback decisions
- `build_allocator()` method unchanged

### External Callers
- `get_allocator()` interface unchanged
- `shared_buffer_size()` interface unchanged
- `persistent_local_buffer_size()` interface unchanged
- `local_buffer_size()` interface unchanged

---

## Test Requirements

Tests should verify:

1. **Deterministic Order**: Accessing properties in different orders produces same layouts
2. **Aliasing Within Parent**: Child overlaps parent when space available
3. **Aliasing Overflow**: Child gets own allocation when parent full
4. **Cross-Location Fallback**: Entry respects its own type when parent wrong type
5. **Multiple Aliases**: Sequential consumption of parent space
6. **Empty Groups**: All methods handle empty entries gracefully
7. **Registration Order**: Later children respect earlier siblings' allocations
8. **Invalidation**: Layouts rebuild correctly after changes

---

## Backward Compatibility Notes

### Breaking Changes
- `build_shared_layout()` and `build_persistent_layout()` may be removed or made private
- `build_local_sizes()` may be removed or made private
- External code directly calling these methods will need updates

### Preserved Behavior
- All public property and method signatures unchanged
- Allocator device function signatures unchanged
- Registration and update semantics unchanged
- Buffer size calculations unchanged

---

## Implementation Order Recommendation

1. Add `build_layouts()` method with orchestration logic
2. Add `layout_aliases()` method with aliasing logic
3. Update property accessors to call `build_layouts()`
4. Remove aliasing logic from original build methods (or make them internal helpers)
5. Update/add tests for new behavior
6. Verify existing tests still pass
