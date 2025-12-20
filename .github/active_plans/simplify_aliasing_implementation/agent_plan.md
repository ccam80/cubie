# Simplify Aliasing Implementation - Agent Plan

## Overview

This plan simplifies the buffer aliasing implementation by reverting to a two-parameter allocator and removing complex bounds-checking logic from `get_allocator`. The layout-building phase already determines aliasing success by assigning overlapping slices, so the allocator just needs to use those slices.

## Component Changes

### 1. CUDABuffer.build_allocator (buffer_registry.py)

**Current Signature:** `build_allocator(shared_slice, persistent_slice, aliased_parent_slice, local_size)`

**New Signature:** `build_allocator(shared_slice, persistent_slice, local_size)`

**Behavior Changes:**
- Remove `aliased_parent_slice` parameter
- Remove `_use_aliased_parent` compile-time flag
- Remove `_aliased_parent_slice` captured value
- Device function signature becomes `allocate_buffer(shared, persistent)` (remove `aliased_parent` parameter)
- Allocation logic:
  1. If `shared_slice` is not None: `array = shared[shared_slice]`
  2. Elif `persistent_slice` is not None: `array = persistent[persistent_slice]`
  3. Else: `array = cuda.local.array(local_size, precision)`

**Key Insight:** When aliasing succeeds, the child buffer gets a slice that overlaps the parent in the same memory region (shared or persistent). The allocator doesn't need a separate parent reference - it just uses the child's slice directly.

### 2. BufferGroup.get_allocator (buffer_registry.py)

**Current Logic:** Complex bounds-checking to determine if child slice is within parent bounds, compute relative slice for `aliased_parent_slice` parameter.

**New Logic:** Simplified - just get the slice from the appropriate layout and pass it to `build_allocator`.

**Simplified Algorithm:**
```python
def get_allocator(self, name: str) -> Callable:
    # Ensure layouts are computed
    if self._shared_layout is None:
        self._shared_layout = self.build_shared_layout()
    if self._persistent_layout is None:
        self._persistent_layout = self.build_persistent_layout()
    if self._local_sizes is None:
        self._local_sizes = self.build_local_sizes()
    
    entry = self.entries[name]
    
    # Get slice from appropriate layout
    shared_slice = self._shared_layout.get(name)
    persistent_slice = self._persistent_layout.get(name)
    local_size = self._local_sizes.get(name)
    
    # That's it! No bounds checking needed.
    return entry.build_allocator(
        shared_slice, persistent_slice, local_size
    )
```

**Rationale:** The layout building phase (`build_shared_layout` and `build_persistent_layout`) already determines:
- Whether aliasing succeeds (by assigning overlapping slices)
- What slice each buffer gets (including aliased children)

The allocator just needs to use these pre-computed slices.

### 3. ERK Stage Cache Registration (generic_erk.py)

**Current Registration:**
```python
buffer_registry.register(
    'stage_cache',
    self,
    n,
    config.stage_cache_location,
    aliases='stage_accumulator',
    persistent=True,
    precision=precision
)
```

**New Registration (Restore Original Preference Order):**
```python
# stage_cache aliasing logic for FSAL optimization
use_shared_rhs = config.stage_rhs_location == 'shared'
use_shared_acc = config.stage_accumulator_location == 'shared'

if use_shared_rhs:
    # Prefer aliasing stage_rhs if it's shared
    buffer_registry.register(
        'stage_cache', self, n, 'shared',
        aliases='stage_rhs', precision=precision
    )
elif use_shared_acc:
    # Fall back to aliasing stage_accumulator if it's shared
    buffer_registry.register(
        'stage_cache', self, n, 'shared',
        aliases='stage_accumulator', precision=precision
    )
else:
    # Both local, use persistent local without aliasing
    buffer_registry.register(
        'stage_cache', self, n, 'local',
        persistent=True, precision=precision
    )
```

**Rationale:**
- FSAL (First Same As Last) optimization works best when stage_cache aliases the RHS buffer
- If RHS is not shared, try accumulator
- If neither is shared, use persistent local
- Buffer registry will handle the actual aliasing decision

**Note:** Remove `stage_cache_location` from `ERKStepConfig` since location is now computed from rhs/accumulator locations.

### 4. Rosenbrock Stage Cache Registration (generic_rosenbrock_w.py)

**Current Registration:**
```python
buffer_registry.register(
    'stage_cache',
    self,
    n,
    config.stage_cache_location,
    aliases='stage_store',
    persistent=True,
    precision=precision
)
```

**New Registration:**
```python
# stage_cache attempts to alias stage_store
use_shared_store = config.stage_store_location == 'shared'

if use_shared_store:
    buffer_registry.register(
        'stage_cache', self, n, 'shared',
        aliases='stage_store', precision=precision
    )
else:
    # stage_store is local, use persistent local without aliasing
    buffer_registry.register(
        'stage_cache', self, n, 'local',
        persistent=True, precision=precision
    )
```

**Rationale:**
- Stop setting `persistent=True` when location='shared' (contradictory)
- Let buffer_registry determine if aliasing succeeds
- Use same pattern as ERK: conditional registration based on parent location

**Note:** Remove `stage_cache_location` from Rosenbrock config since location is now computed from stage_store location.

### 5. All Allocator Call Sites

**Current Pattern:**
```python
stage_cache = alloc_stage_cache(
    shared, persistent_local, stage_accumulator
)
```

**New Pattern:**
```python
stage_cache = alloc_stage_cache(shared, persistent_local)
```

**Affected Files:**
- `generic_erk.py`: stage_cache allocation
- `generic_rosenbrock_w.py`: stage_cache allocation
- Any other files that call allocators

**Change:** Remove third argument from all allocator calls (it was the parent buffer reference).

## Data Structures

### BufferGroup Attributes (No Changes Needed)

The existing attributes are sufficient:
- `_shared_layout: Optional[Dict[str, slice]]` - Maps buffer names to shared memory slices
- `_persistent_layout: Optional[Dict[str, slice]]` - Maps buffer names to persistent local slices
- `_local_sizes: Dict[str, int]` - Maps buffer names to local array sizes

**Note:** Aliased buffers appear in the same layout as their parent (shared or persistent), with slices that may overlap the parent's slice. This overlap IS the aliasing.

### CUDABuffer.build_allocator Return Type

**Current:** Device function with signature `(shared, persistent, aliased_parent) -> array`

**New:** Device function with signature `(shared, persistent) -> array`

## Integration Points

### Integration with Layout Building

**build_shared_layout:**
- No changes needed
- Already assigns overlapping slices when aliasing succeeds
- Example: Parent at `slice(0, 100)`, child aliases parent gets `slice(20, 40)`

**build_persistent_layout:**
- No changes needed
- Same overlapping slice logic

### Integration with Allocator Usage

**Before (Three-Parameter):**
```python
# In device function
stage_cache = alloc_stage_cache(shared, persistent, stage_accumulator)
# If aliasing succeeded: stage_cache = stage_accumulator[relative_slice]
# If aliasing failed: stage_cache = shared[fresh_slice]
```

**After (Two-Parameter):**
```python
# In device function
stage_cache = alloc_stage_cache(shared, persistent)
# Always: stage_cache = shared[slice] or persistent[slice] or local array
# Aliasing is transparent - child slice overlaps parent in shared/persistent
```

**Key Difference:** With two parameters, the calling code doesn't need to know or care whether aliasing succeeded. The allocator uses the slice assigned during layout building, which automatically provides the correct memory (aliased or fresh).

### Integration with Config Classes

**ERKStepConfig:**
- Remove `stage_cache_location: str` field
- Keep `stage_rhs_location` and `stage_accumulator_location`
- Registration code determines stage_cache location from these

**RosenbrockStepConfig:**
- Remove `stage_cache_location: str` field
- Keep `stage_store_location`
- Registration code determines stage_cache location from this

**Rationale:** The location of an aliasing buffer is derived from its parent, not independently configured. Removing the config field prevents contradictory settings (e.g., `aliases='stage_rhs'` but `location='local'` when stage_rhs is shared).

## Expected Behavior

### Scenario 1: ERK with shared RHS

**Registration:**
```python
stage_rhs: location='shared', size=100
stage_accumulator: location='shared', size=100
stage_cache: location='shared', aliases='stage_rhs', size=100
```

**Layout Building:**
```python
shared_layout = {
    'stage_rhs': slice(0, 100),
    'stage_accumulator': slice(100, 200),
    'stage_cache': slice(0, 100),  # Overlaps stage_rhs
}
```

**Allocator Generation:**
```python
get_allocator('stage_cache')
# Returns device function that does: array = shared[slice(0, 100)]
```

**Device Function Execution:**
```python
stage_rhs = alloc_stage_rhs(shared, persistent)  # shared[0:100]
stage_cache = alloc_stage_cache(shared, persistent)  # shared[0:100]
# stage_rhs and stage_cache point to same memory - aliasing succeeded!
```

### Scenario 2: ERK with local RHS, shared accumulator

**Registration:**
```python
stage_rhs: location='local', size=100
stage_accumulator: location='shared', size=100
stage_cache: location='shared', aliases='stage_accumulator', size=100
```

**Layout Building:**
```python
shared_layout = {
    'stage_accumulator': slice(0, 100),
    'stage_cache': slice(0, 100),  # Overlaps stage_accumulator
}
local_sizes = {
    'stage_rhs': 100
}
```

**Allocator Generation:**
```python
get_allocator('stage_cache')
# Returns device function that does: array = shared[slice(0, 100)]
```

**Device Function Execution:**
```python
stage_rhs = alloc_stage_rhs(shared, persistent)  # cuda.local.array(100)
stage_accumulator = alloc_stage_accumulator(shared, persistent)  # shared[0:100]
stage_cache = alloc_stage_cache(shared, persistent)  # shared[0:100]
# stage_cache aliases stage_accumulator - correct fallback!
```

### Scenario 3: ERK with both local

**Registration:**
```python
stage_rhs: location='local', size=100
stage_accumulator: location='local', size=100
stage_cache: location='local', persistent=True, size=100
```

**Layout Building:**
```python
persistent_layout = {
    'stage_cache': slice(0, 100)
}
local_sizes = {
    'stage_rhs': 100,
    'stage_accumulator': 100
}
```

**Allocator Generation:**
```python
get_allocator('stage_cache')
# Returns device function that does: array = persistent[slice(0, 100)]
```

**Device Function Execution:**
```python
stage_rhs = alloc_stage_rhs(shared, persistent)  # cuda.local.array(100)
stage_accumulator = alloc_stage_accumulator(shared, persistent)  # cuda.local.array(100)
stage_cache = alloc_stage_cache(shared, persistent)  # persistent[0:100]
# stage_cache gets persistent local - no aliasing possible
```

### Scenario 4: Rosenbrock with shared stage_store

**Registration:**
```python
stage_store: location='shared', size=300
stage_cache: location='shared', aliases='stage_store', size=100
```

**Layout Building:**
```python
shared_layout = {
    'stage_store': slice(0, 300),
    'stage_cache': slice(0, 100),  # Overlaps stage_store
}
```

**Result:** stage_cache uses first 100 elements of stage_store - aliasing succeeded.

## Edge Cases

### Edge Case 1: Aliasing fails due to size

**Scenario:** Parent is size 50, child wants to alias with size 100.

**Layout Building:**
```python
# Parent not big enough
layout['parent'] = slice(0, 50)
layout['child'] = slice(50, 150)  # Fresh allocation, doesn't overlap
```

**Allocator:**
```python
# Child gets fresh allocation automatically
child_array = shared[slice(50, 150)]
```

**Handling:** Automatic - layout builder assigns fresh space, allocator uses it.

### Edge Case 2: Cross-location aliasing attempt

**Scenario:** Child wants to alias parent, but child is shared and parent is local.

**Layout Building:**
```python
# Parent in local, child requested shared
local_sizes['parent'] = 100
shared_layout['child'] = slice(0, 100)  # Fresh shared allocation
```

**Allocator:**
```python
# Child gets fresh shared allocation
child_array = shared[slice(0, 100)]
```

**Handling:** Automatic - layout builder places child in shared, allocator uses it.

### Edge Case 3: Buffer has no allocation

**Scenario:** Zero-size buffer registered (e.g., cached_auxiliaries before update).

**Layout Building:**
```python
# Skip zero-size buffers
if entry.size == 0:
    continue  # Not added to any layout
```

**Allocator:**
```python
# All slices are None, local_size is None
# What should happen here?
```

**Handling:** Need to handle this case - likely should allocate minimal local array or raise error. Check existing code for pattern.

### Edge Case 4: Multiple children alias same parent

**Scenario:** child1 and child2 both alias parent.

**Layout Building:**
```python
layout['parent'] = slice(0, 100)
layout['child1'] = slice(0, 50)  # First 50 of parent
layout['child2'] = slice(50, 100)  # Next 50 of parent
```

**Allocator:**
```python
child1 = shared[slice(0, 50)]
child2 = shared[slice(50, 100)]
```

**Handling:** Automatic - layout builder tracks consumption, assigns non-overlapping slices within parent.

## Dependencies and Imports

No new dependencies needed. All required functionality exists in:
- `numba.cuda` for device functions
- `cubie.cuda_simsafe` for compile_kwargs
- `attrs` for data classes

## Comments and Documentation

### Updated Docstrings

**CUDABuffer.build_allocator:**
```python
"""Compile CUDA device function for buffer allocation.

Generates an inlined device function that allocates this buffer
from the appropriate memory region based on which slice parameters
are provided.

Parameters
----------
shared_slice
    Slice into shared memory, or None if not using shared.
persistent_slice
    Slice into persistent local memory, or None if not using persistent.
local_size
    Size for local array allocation, or None if not local.

Returns
-------
Callable
    CUDA device function: (shared, persistent) -> array
    
    The device function accepts shared and persistent memory arrays
    and returns a view/slice into the appropriate memory region,
    or allocates a fresh local array.

Notes
-----
When a buffer aliases another buffer and aliasing succeeds, both
buffers receive slices in the same memory region (shared or persistent)
that overlap. The allocator transparently provides the correct view
without needing a separate parent reference.
"""
```

**BufferGroup.get_allocator:**
```python
"""Generate CUDA device function for buffer allocation.

Retrieves the pre-computed memory slice for this buffer from the
appropriate layout (shared, persistent, or local) and generates
an allocator that uses that slice.

Parameters
----------
name
    Buffer name to generate allocator for.

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
build_persistent_layout) determines which memory region each buffer
uses and assigns slices accordingly. This method simply retrieves
those pre-computed slices and creates an allocator that uses them.

For aliased buffers, the layout builder assigns slices that overlap
the parent buffer, implementing aliasing transparently at the slice
level.
"""
```

### Comments for ERK Registration

```python
# Stage cache registration with preference order for FSAL optimization.
# The buffer registry determines if aliasing succeeds based on parent
# availability and size. Preference order:
#   1. Alias stage_rhs if shared (best for FSAL)
#   2. Alias stage_accumulator if shared (fallback)
#   3. Use persistent local (no aliasing possible)
```

### Comments for Rosenbrock Registration

```python
# Stage cache attempts to alias stage_store for memory reuse.
# Buffer registry determines if aliasing succeeds based on
# stage_store location and available space.
```

## Architectural Rationale

### Why Two Parameters Instead of Three?

**Previous Approach (Three Parameters):**
- Allocator receives: `(shared, persistent, aliased_parent)`
- Logic: "If aliasing succeeded, slice the parent. Otherwise, use shared/persistent."
- Problem: Requires runtime decision in allocator, parent reference, complex logic

**New Approach (Two Parameters):**
- Allocator receives: `(shared, persistent)`
- Logic: "Use the slice assigned during layout building"
- Benefit: Layout building is the single source of truth; allocator is simple

**Key Realization:** When aliasing succeeds, the child buffer's slice IS ALREADY A SLICE OF THE PARENT in the same memory region. We don't need to pass the parent separately - the child's slice automatically provides access to the parent's memory.

Example:
```python
# Layout building assigns:
parent_slice = slice(0, 100)
child_slice = slice(20, 40)  # Deliberately overlaps parent

# In device function:
parent = shared[slice(0, 100)]  # shared[0:100]
child = shared[slice(20, 40)]   # shared[20:40] - same memory as parent[20:40]!

# No need to do: child = parent[slice(20, 40)]
# Because: shared[20:40] == shared[0:100][20:40] in memory terms
```

### Why Remove Location Config for Aliasing Buffers?

**Problem:** Config fields like `stage_cache_location` create contradictions:
- If `stage_cache` aliases `stage_rhs`, its location must match `stage_rhs`
- User could set `stage_rhs_location='shared'` and `stage_cache_location='local'`
- Which wins? Contradiction in intent.

**Solution:** Derive location from parent:
- If aliasing shared parent: child location is 'shared'
- If aliasing local parent: child location is 'local' + persistent flag
- If no parent: use explicit location

**Benefit:** Impossible to create contradictory configurations.

### Why Simplify get_allocator?

**Previous Logic:** 60+ lines of bounds-checking to determine if child slice is within parent bounds, compute relative slices, etc.

**Problem:** This recomputes what the layout builder already determined. Violates DRY (Don't Repeat Yourself).

**Solution:** Trust the layout builder. If a buffer appears in `_shared_layout`, it uses shared memory at that slice. Period.

**Benefit:** 
- Simpler code (6 lines instead of 60)
- Single source of truth (layout building)
- Easier to understand and maintain
- Fewer opportunities for bugs

## Testing Considerations

### Existing Tests Should Pass

All existing buffer allocation tests should pass without modification because:
- Layout building logic unchanged
- Final memory assignments unchanged
- Only allocator interface simplified

### Tests That May Need Updates

1. **Direct allocator call tests** - If any tests directly call allocators with three parameters, update to two parameters
2. **Config validation tests** - If any tests check `stage_cache_location` config field, remove those checks
3. **Mock/patch tests** - If any tests mock allocator calls, update mock signatures

### New Test Considerations

Consider adding tests for:
1. ERK preference order (rhs > accumulator > persistent)
2. Rosenbrock stage_store aliasing
3. Allocator with only shared_slice set
4. Allocator with only persistent_slice set
5. Allocator with neither slice set (local allocation)

However, these scenarios are likely already covered by existing integration tests.
