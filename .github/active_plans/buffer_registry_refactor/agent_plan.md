# Buffer Registry Architecture Refactor - Agent Plan

## Scope

This plan covers Task 1 of 3 - changes to `buffer_registry.py` only. Do NOT modify:
- Integration files (algorithms, loops, matrix-free solvers)
- Solver plumbing (solver.py)
- Any files outside buffer_registry.py and its tests

## User Stories Reference

See `human_overview.md` for:
- US-1: Class naming (BufferEntry → CUDABuffer, BufferContext → BufferGroup)
- US-2: Remove factory from CUDABuffer, rename factory → parent
- US-3: Move methods to appropriate classes
- US-4: Cross-location aliasing support

---

## Component Specifications

### CUDABuffer (formerly BufferEntry)

**Purpose:** Immutable record describing a single buffer's requirements.

**Attributes:**
| Attribute | Type | Description |
|-----------|------|-------------|
| name | str | Unique buffer name within parent context |
| size | int | Buffer size in elements (≥0) |
| location | str | Memory location: 'shared' or 'local' |
| persistent | bool | If True and location='local', use persistent_local |
| aliases | str or None | Name of parent buffer to alias (if any) |
| precision | type | NumPy precision type (float16/32/64) |

**Removed:** `factory` attribute (ownership tracked at BufferGroup level)

**Properties (unchanged):**
- `is_shared` → bool
- `is_local` → bool  
- `is_persistent_local` → bool

**New Method:**

```
build_allocator(shared_slice, persistent_slice, local_size) -> Callable
```

Compiles and returns a CUDA device function that allocates this buffer from the appropriate memory region based on which parameters are non-None.

**Behavior:**
- If `shared_slice` is not None → return slice of shared_parent
- Elif `persistent_slice` is not None → return slice of persistent_parent
- Else → return cuda.local.array(local_size, precision)

The method captures compile-time constants in closure and returns an inlined CUDA device function.

---

### BufferGroup (formerly BufferContext)

**Purpose:** Groups all buffer entries for a single parent object.

**Attributes:**
| Attribute | Type | Init | Description |
|-----------|------|------|-------------|
| parent | object | Yes | Parent object that owns this group |
| entries | Dict[str, CUDABuffer] | factory=dict | Registered buffers by name |
| _shared_layout | Dict[str, slice] or None | init=False | Cached shared memory slices |
| _persistent_layout | Dict[str, slice] or None | init=False | Cached persistent_local slices |
| _local_sizes | Dict[str, int] or None | init=False | Cached local sizes |
| _alias_consumption | Dict[str, int] | init=False, factory=dict | Tracks consumed space in aliased buffers |

**Changed:** `factory` → `parent` (naming only, same purpose)

**Existing Method (unchanged):**
```
invalidate_layouts() -> None
```
Sets all cached layouts to None and clears _alias_consumption.

**Moved Methods (from BufferRegistry):**

```
register(name, size, location, persistent=False, aliases=None, precision=np.float32) -> None
```
Creates CUDABuffer and adds to entries. Validates:
- Name not empty
- Name not duplicate
- Name not self-aliasing
- Alias target exists (if specified)

**REMOVED validation:** Cross-type aliasing constraints (shared→shared, etc.) - these are now handled at layout build time.

```
update_buffer(name, **kwargs) -> None
```
Updates existing buffer properties. Ignores if name not found.

```
build_shared_layout() -> Dict[str, slice]
```
Computes slice indices for shared memory buffers with new aliasing logic:

1. Process non-aliased shared buffers first, assigning sequential slices
2. For each aliased buffer (regardless of its own location):
   - If parent is shared AND has sufficient remaining space:
     - Assign slice within parent
     - Update consumption tracker
   - Else:
     - If buffer's own location is shared: assign new slice at end
     - Otherwise: skip (will be handled by persistent or local layout)

```
build_persistent_layout() -> Dict[str, slice]
```
Computes slice indices for persistent local buffers:

1. Process non-aliased persistent buffers first
2. For each aliased buffer with persistent=True:
   - If parent is persistent AND has sufficient remaining space:
     - Assign slice within parent (similar logic to shared)
   - Else:
     - Assign new slice at end

Note: Non-persistent local buffers that failed shared aliasing are NOT included here.

```
build_local_sizes() -> Dict[str, int]
```
Computes sizes for non-persistent local buffers:
- Include all buffers with is_local=True
- Include aliased buffers that couldn't be sliced into shared parent
- Size is max(entry.size, 1) for each

```
shared_buffer_size() -> int
```
Returns total shared memory elements (excludes successfully aliased buffers).

```
local_buffer_size() -> int  
```
Returns total local memory elements.

```
persistent_local_buffer_size() -> int
```
Returns total persistent local elements (excludes successfully aliased buffers).

```
get_allocator(name) -> Callable
```
Generates CUDA device function for buffer allocation:
1. Ensure layouts are computed (lazy build)
2. Look up buffer entry
3. Determine which slice/size applies to this buffer
4. Call `buffer.build_allocator(shared_slice, persistent_slice, local_size)`
5. Return the compiled function

---

### BufferRegistry

**Purpose:** Central registry managing all buffer metadata for CUDA parents. Package-level singleton.

**Attributes:**
| Attribute | Type | Description |
|-----------|------|-------------|
| _groups | Dict[object, BufferGroup] | Maps parent instances to their buffer groups |

**Changed:** `_contexts` → `_groups` (naming consistency)

**Methods (wrapper pattern):**

All methods that took `factory` parameter now take `parent` parameter.

```
register(name, parent, size, location, persistent=False, aliases=None, precision=np.float32) -> None
```
Wrapper that:
1. Gets or creates BufferGroup for parent
2. Delegates to group.register(name, size, location, ...)

```
update_buffer(name, parent, **kwargs) -> None
```
Wrapper that:
1. Gets group for parent (returns silently if not found)
2. Delegates to group.update_buffer(name, **kwargs)

```
clear_layout(parent) -> None
```
Wrapper that calls group.invalidate_layouts() if group exists.

```
clear_parent(parent) -> None
```
Removes parent from _groups dict. **RENAMED** from `clear_factory`.

```
shared_buffer_size(parent) -> int
```
Wrapper returning 0 if no group, else group.shared_buffer_size().

```
local_buffer_size(parent) -> int
```
Wrapper returning 0 if no group, else group.local_buffer_size().

```
persistent_local_buffer_size(parent) -> int
```
Wrapper returning 0 if no group, else group.persistent_local_buffer_size().

```
get_allocator(name, parent) -> Callable
```
Wrapper that:
1. Raises KeyError if parent not registered
2. Delegates to group.get_allocator(name)

---

## Aliasing Logic Detail

### Registration Phase
At registration, we only validate:
- Alias target exists in same group
- No self-aliasing

We do NOT validate location compatibility. Any buffer can declare any other buffer as alias.

### Layout Build Phase (build_shared_layout)

```
For each buffer with aliases != None:
    parent = entries[buffer.aliases]
    if parent.is_shared:
        consumed = _alias_consumption.get(parent.name, 0)
        available = parent.size - consumed
        if buffer.size <= available:
            # Successful alias: slice within parent
            parent_slice = layout[parent.name]
            start = parent_slice.start + consumed
            layout[buffer.name] = slice(start, start + buffer.size)
            _alias_consumption[parent.name] = consumed + buffer.size
        else:
            # Parent too small: allocate per buffer's own settings
            if buffer.is_shared:
                layout[buffer.name] = slice(offset, offset + buffer.size)
                offset += buffer.size
            # else: not our concern, handled by persistent/local
    else:
        # Parent is local: allocate per buffer's own settings
        if buffer.is_shared:
            layout[buffer.name] = slice(offset, offset + buffer.size)
            offset += buffer.size
        # else: not our concern
```

### Layout Build Phase (build_persistent_layout)

Similar logic but for persistent buffers:
- Only process is_persistent_local buffers
- Check if aliased parent is also persistent and has space
- Fallback to own allocation if not

### Layout Build Phase (build_local_sizes)

```
For each buffer:
    if buffer.is_local:
        # Always allocate non-persistent local buffers
        sizes[name] = max(buffer.size, 1)
    elif buffer.aliases is not None:
        # Check if this aliased buffer failed to get space in shared/persistent
        parent = entries[buffer.aliases]
        if not parent.is_shared and not parent.is_persistent_local:
            # Parent is local, allocate this buffer locally too
            if buffer.location == 'local':
                sizes[name] = max(buffer.size, 1)
```

### Allocator Generation

```
def get_allocator(self, name):
    entry = self.entries[name]
    
    # Ensure layouts computed
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

---

## Integration Points

### External Interface (unchanged signatures except naming)

The following signatures change `factory` → `parent`:
- `buffer_registry.register(name, parent, size, location, ...)`
- `buffer_registry.update_buffer(name, parent, **kwargs)`
- `buffer_registry.clear_layout(parent)`
- `buffer_registry.clear_parent(parent)` ← renamed from clear_factory
- `buffer_registry.shared_buffer_size(parent)`
- `buffer_registry.local_buffer_size(parent)`
- `buffer_registry.persistent_local_buffer_size(parent)`
- `buffer_registry.get_allocator(name, parent)`

### Internal Dependencies

CUDABuffer.build_allocator depends on:
- `numba.cuda` for device function compilation
- `cubie.cuda_simsafe.compile_kwargs` for compilation options

BufferGroup and BufferRegistry have no new dependencies.

---

## Edge Cases

### 1. Alias Chain
Buffer A aliases B which aliases C. Currently not supported - the code only looks at direct alias. This remains unchanged; deep alias resolution is out of scope.

### 2. Zero-Size Buffer
A buffer with size=0:
- In shared: gets slice(n, n) with 0 length
- In local: gets allocated with size=1 (cuda.local.array minimum)
- As alias: consumes 0 space in parent

### 3. Alias Larger Than Parent
Under new logic:
- Aliased buffer requests 50 elements
- Parent only has 30 remaining
- Alias falls back to its own location settings
- If alias.location == 'shared', it gets new shared space
- If alias.location == 'local', it gets local array

### 4. Multiple Aliases Exhaust Parent
Parent size = 100
- Alias1 (size=40) → slices 0:40, consumed=40
- Alias2 (size=40) → slices 40:80, consumed=80
- Alias3 (size=40) → only 20 remaining, falls back to own allocation

---

## Test Updates Required

### Class Name Updates
- `BufferEntry` → `CUDABuffer`
- `BufferContext` → `BufferGroup`
- `MockFactory` → keep name, but conceptually is "parent"

### Removed Tests
- `test_shared_alias_with_local_location_raises` - no longer an error
- `test_persistent_alias_of_nonpersistent_local_raises` - no longer an error

### New Tests Needed
- `test_cross_location_aliasing_allowed` - shared buffer can alias local parent
- `test_alias_fallback_when_parent_not_shared` - alias gets own allocation
- `test_alias_fallback_when_parent_too_small` - alias gets own allocation
- `test_multiple_aliases_first_come_first_serve` - verify consumption tracking
- `test_alias_partially_fits` - first alias fits, second doesn't

### Modified Tests
- Update all `clear_factory` calls to `clear_parent`
- Update parameter names from `factory=` to `parent=` where applicable
- Verify layout building uses new logic

---

## Implementation Notes

### No Backwards Compatibility
Per project guidelines: "Never retain an obsolete feature or argument for API compatibility." All renames are breaking changes.

### No Optional Parameters
The buffer management system is compulsory. Do not add fallbacks for missing groups or optional parent parameters.

### Guarantee-by-Design
Do not add defensive guards for conditions that shouldn't occur when used properly. For example, if build_allocator is called with incompatible slice parameters, that's a bug in BufferGroup, not something to guard against.

### Error Messages
Error messages should reference `parent` not `factory`:
- "Parent {parent} has no registered buffer group."
- "Buffer '{name}' not registered for parent."
