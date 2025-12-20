# Implementation Task List
# Feature: Simplify Aliasing Implementation
# Plan Reference: .github/active_plans/simplify_aliasing_implementation/agent_plan.md

## Task Group 1: Fix ERK Stage Cache Registration - SEQUENTIAL
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_erk.py (lines 230-266)
- File: .github/context/cubie_internal_structure.md (entire file for context)

**Input Validation Required**:
None - this is registration logic, not runtime logic.

**Tasks**:

### TG1.1: Restore ERK Conditional Registration Logic

- **File**: `/home/runner/work/cubie/cubie/src/cubie/integrators/algorithms/generic_erk.py`
- **Action**: Modify
- **Lines**: 251-259

**Current Code**:
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

**New Code**:
```python
# Stage cache registration with preference order for FSAL optimization.
# Preference order:
#   1. Alias stage_rhs if shared (best for FSAL)
#   2. Alias stage_accumulator if shared (fallback)
#   3. Use persistent local (no aliasing possible)
use_shared_rhs = config.stage_rhs_location == 'shared'
use_shared_acc = config.stage_accumulator_location == 'shared'

if use_shared_rhs:
    buffer_registry.register(
        'stage_cache', self, n, 'shared',
        aliases='stage_rhs', precision=precision
    )
elif use_shared_acc:
    buffer_registry.register(
        'stage_cache', self, n, 'shared',
        aliases='stage_accumulator', precision=precision
    )
else:
    buffer_registry.register(
        'stage_cache', self, n, 'local',
        persistent=True, precision=precision
    )
```

**Details**:
- Replace single registration with if/elif/else conditional registration
- Check `config.stage_rhs_location == 'shared'` first
- Check `config.stage_accumulator_location == 'shared'` second
- Fall back to persistent local if neither is shared
- Remove `stage_cache_location` from config - it's now derived
- Comment explains the preference order for FSAL optimization
- Buffer registry determines if aliasing succeeds based on parent availability

**Edge Cases**:
- If both rhs and accumulator are shared, rhs takes precedence (better for FSAL)
- If neither is shared, persistent local is used without aliasing
- If parent is too small for aliasing, buffer registry assigns fresh allocation

**Integration**:
- Connects to buffer_registry.register() method
- BufferGroup.build_shared_layout() handles actual aliasing decision
- No changes needed to step function - uses same allocators

### TG1.2: Remove stage_cache_location from ERKStepConfig

- **File**: `/home/runner/work/cubie/cubie/src/cubie/integrators/algorithms/ode_explicitstep.py`
- **Action**: Modify
- **Lines**: Find ERKStepConfig attrs class definition

**Details**:
- Locate ERKStepConfig attrs class
- Remove `stage_cache_location: str` field
- Keep `stage_rhs_location` and `stage_accumulator_location` fields
- Update ALL_ALGORITHM_STEP_PARAMETERS set if stage_cache_location is listed
- No validation changes needed since field is being removed

**Edge Cases**:
- Ensure no default value is referencing stage_cache_location
- Check if any tests explicitly set stage_cache_location

**Integration**:
- ERK build_step method now derives stage_cache location from parent locations
- Config remains valid for step controller and other consumers

**Outcomes**:
- Files Modified:
  * src/cubie/integrators/algorithms/generic_erk.py (25 lines changed)
- Functions/Methods Added/Modified:
  * ERKStep.__init__() - Restored conditional stage_cache registration
  * ERKStepConfig class - Removed stage_cache_location field
- Implementation Summary:
  * Replaced single stage_cache registration with three-branch
    conditional logic
  * Checks stage_rhs_location first, stage_accumulator_location second,
    falls back to persistent local
  * Removed stage_cache_location parameter from ERKStepConfig and
    __init__ signature
  * Registration now derives location from parent buffer locations
    instead of using explicit config field
- Issues Flagged: None

---

## Task Group 2: Fix Rosenbrock Stage Cache Registration - SEQUENTIAL
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py (lines 230-276)
- File: .github/context/cubie_internal_structure.md (entire file for context)

**Input Validation Required**:
None - this is registration logic, not runtime logic.

**Tasks**:

### TG2.1: Fix Rosenbrock Conditional Registration Logic

- **File**: `/home/runner/work/cubie/cubie/src/cubie/integrators/algorithms/generic_rosenbrock_w.py`
- **Action**: Modify
- **Lines**: 252-260

**Current Code**:
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

**New Code**:
```python
# Stage cache attempts to alias stage_store for memory reuse.
use_shared_store = config.stage_store_location == 'shared'

if use_shared_store:
    buffer_registry.register(
        'stage_cache', self, n, 'shared',
        aliases='stage_store', precision=precision
    )
else:
    buffer_registry.register(
        'stage_cache', self, n, 'local',
        persistent=True, precision=precision
    )
```

**Details**:
- Replace single registration with if/else conditional registration
- Check `config.stage_store_location == 'shared'`
- If shared, register stage_cache as shared with aliases='stage_store'
- If local, register stage_cache as persistent local without aliasing
- Remove contradictory `persistent=True` when location='shared'
- Comment explains aliasing intent

**Edge Cases**:
- If stage_store is local, stage_cache cannot alias it (different memory regions)
- If stage_store is too small, buffer registry assigns fresh allocation

**Integration**:
- Connects to buffer_registry.register() method
- BufferGroup.build_shared_layout() or build_persistent_layout() handles aliasing decision
- No changes needed to step function - uses same allocators

### TG2.2: Remove stage_cache_location from RosenbrockStepConfig

- **File**: `/home/runner/work/cubie/cubie/src/cubie/integrators/algorithms/ode_implicitstep.py`
- **Action**: Modify
- **Lines**: Find RosenbrockStepConfig attrs class definition (or equivalent config class)

**Details**:
- Locate the Rosenbrock config attrs class (may be in generic_rosenbrock_w.py or ode_implicitstep.py)
- Remove `stage_cache_location: str` field
- Keep `stage_store_location`, `stage_rhs_location`, `cached_auxiliaries_location` fields
- Update ALL_ALGORITHM_STEP_PARAMETERS set if stage_cache_location is listed
- No validation changes needed since field is being removed

**Edge Cases**:
- Ensure no default value is referencing stage_cache_location
- Check if any tests explicitly set stage_cache_location

**Integration**:
- Rosenbrock build_step method now derives stage_cache location from stage_store location
- Config remains valid for linear solver and other consumers

**Outcomes**:
- Files Modified:
  * src/cubie/integrators/algorithms/generic_rosenbrock_w.py (16 lines changed)
- Functions/Methods Added/Modified:
  * GenericRosenbrockWStep.__init__() - Fixed conditional stage_cache
    registration
- Implementation Summary:
  * Replaced single stage_cache registration with two-branch conditional
    logic
  * Checks stage_store_location: if shared, aliases stage_store; if
    local, uses persistent local
  * Removed contradictory persistent=True when location='shared'
  * Registration now derives location from stage_store_location instead
    of using non-existent config.stage_cache_location field
  * Note: RosenbrockWStepConfig never had stage_cache_location field, so
    no config changes needed
- Issues Flagged: 
  * The original code referenced config.stage_cache_location which
    doesn't exist in RosenbrockWStepConfig - this was a bug that would
    have caused AttributeError at runtime

---

## Task Group 3: Revert to Two-Parameter Allocator - SEQUENTIAL
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/buffer_registry.py (lines 75-134)
- File: .github/context/cubie_internal_structure.md (lines 1-100 for CUDAFactory pattern)

**Input Validation Required**:
None - compile-time device function generation, not runtime validation.

**Tasks**:

### TG3.1: Change build_allocator Signature and Implementation

- **File**: `/home/runner/work/cubie/cubie/src/cubie/buffer_registry.py`
- **Action**: Modify
- **Lines**: 75-134 (CUDABuffer.build_allocator method)

**Current Signature**:
```python
def build_allocator(
    self,
    shared_slice: Optional[slice],
    persistent_slice: Optional[slice],
    aliased_parent_slice: Optional[slice],
    local_size: Optional[int],
) -> Callable:
```

**New Signature**:
```python
def build_allocator(
    self,
    shared_slice: Optional[slice],
    persistent_slice: Optional[slice],
    local_size: Optional[int],
) -> Callable:
```

**Current Implementation** (lines 106-134):
```python
# Compile-time constants captured in closure
_use_shared = shared_slice is not None
_use_persistent = persistent_slice is not None
_use_aliased_parent = aliased_parent_slice is not None
_shared_slice = shared_slice if _use_shared else slice(0, 0)
_persistent_slice = (
    persistent_slice if _use_persistent else slice(0, 0)
)
_aliased_parent_slice = (
    aliased_parent_slice if _use_aliased_parent else slice(0, 0)
)
_local_size = local_size if local_size is not None else 1
_precision = self.precision

@cuda.jit(device=True, inline=True, **compile_kwargs)
def allocate_buffer(
    shared, persistent, aliased_parent
):
    """Allocate buffer from appropriate memory region."""
    if _use_aliased_parent:
        array = aliased_parent[_aliased_parent_slice]
    elif _use_persistent:
        array = persistent[_persistent_slice]
    elif _use_shared:
        array = shared[_shared_slice]
    else:
        array = cuda.local.array(_local_size, _precision)
    return array

return allocate_buffer
```

**New Implementation**:
```python
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
def allocate_buffer(shared, persistent):
    """Allocate buffer from appropriate memory region."""
    if _use_shared:
        array = shared[_shared_slice]
    elif _use_persistent:
        array = persistent[_persistent_slice]
    else:
        array = cuda.local.array(_local_size, _precision)
    return array

return allocate_buffer
```

**Details**:
- Remove `aliased_parent_slice` parameter from signature
- Remove `_use_aliased_parent` compile-time flag
- Remove `_aliased_parent_slice` captured value
- Change device function signature from `(shared, persistent, aliased_parent)` to `(shared, persistent)`
- Simplify allocation logic to check shared first, then persistent, then local
- Update docstring to reflect two-parameter signature

**Edge Cases**:
- If both shared_slice and persistent_slice are None, allocates local array
- If local_size is None, defaults to 1 (cuda.local.array minimum)
- Slices can overlap within shared or persistent arrays (that's aliasing!)

**Integration**:
- Called by BufferGroup.get_allocator() which provides the slices
- Used by all algorithm and loop device functions

### TG3.2: Update build_allocator Docstring

- **File**: `/home/runner/work/cubie/cubie/src/cubie/buffer_registry.py`
- **Action**: Modify
- **Lines**: 82-104 (build_allocator docstring)

**Current Docstring**:
```python
"""Compile CUDA device function for buffer allocation.

Generates an inlined device function that allocates this buffer
from the appropriate memory region based on which parameters are
provided.

Parameters
----------
shared_slice
    Slice into shared memory for fresh shared allocation, or None.
persistent_slice
    Slice into persistent local memory, or None.
aliased_parent_slice
    Slice into parent buffer when aliasing succeeds, or None.
local_size
    Size for local array allocation, or None if not local.

Returns
-------
Callable
    CUDA device function:
    (shared, persistent, aliased_parent) -> array
"""
```

**New Docstring**:
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

**Details**:
- Remove `aliased_parent_slice` parameter documentation
- Clarify that slices are for actual allocation, not "fresh allocation"
- Update return type signature to `(shared, persistent) -> array`
- Add Notes section explaining how aliasing works transparently via overlapping slices
- Update parameter descriptions to be clearer

**Edge Cases**: None - documentation only

**Integration**: Helps developers understand the two-parameter approach

**Outcomes**:
- Files Modified:
  * src/cubie/buffer_registry.py (30 lines changed in build_allocator
    method)
- Functions/Methods Added/Modified:
  * CUDABuffer.build_allocator() - Removed aliased_parent_slice
    parameter
  * allocate_buffer device function - Changed from 3-parameter to
    2-parameter signature (shared, persistent)
- Implementation Summary:
  * Removed aliased_parent_slice parameter from build_allocator
    signature
  * Removed _use_aliased_parent compile-time flag
  * Removed _aliased_parent_slice captured value
  * Changed device function signature from (shared, persistent,
    aliased_parent) to (shared, persistent)
  * Simplified allocation logic to check shared first, then persistent,
    then local
  * Updated docstring to reflect two-parameter signature and explain
    aliasing via overlapping slices
  * Added Notes section explaining transparent aliasing mechanism
- Issues Flagged: None

---

## Task Group 4: Simplify get_allocator - SEQUENTIAL
**Status**: [x]
**Dependencies**: Task Group 3 (build_allocator must accept two parameters
first)

**Required Context**:
- File: src/cubie/buffer_registry.py (lines 486-598 - BufferGroup.get_allocator method)
- File: src/cubie/buffer_registry.py (lines 280-412 - layout building methods for context)

**Input Validation Required**:
None - this retrieves pre-computed layouts, no runtime validation.

**Tasks**:

### TG4.1: Replace get_allocator Complex Logic with Simple Slice Retrieval

- **File**: `/home/runner/work/cubie/cubie/src/cubie/buffer_registry.py`
- **Action**: Modify
- **Lines**: 486-598 (BufferGroup.get_allocator method)

**Current Implementation** (60+ lines with complex bounds-checking):
```python
def get_allocator(self, name: str) -> Callable:
    """Generate CUDA device function for buffer allocation.

    Determines allocation strategy based on buffer properties and
    whether aliasing succeeds. When aliasing succeeds, passes parent
    buffer for slicing. Otherwise passes shared or appropriate
    location.

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
                child_slice = self._shared_layout[name]
                parent_slice = self._shared_layout[parent_name]

                # If child slice is WITHIN parent bounds, aliasing
                # succeeded
                if (child_slice.start >= parent_slice.start and
                        child_slice.stop <= parent_slice.stop):
                    # Aliasing succeeded - compute relative slice
                    relative_start = (
                        child_slice.start - parent_slice.start
                    )
                    relative_stop = (
                        child_slice.stop - parent_slice.start
                    )
                    aliased_parent_slice = slice(
                        relative_start, relative_stop
                    )
                else:
                    # Got fresh allocation (parent was full)
                    shared_slice = child_slice
        elif (entry.is_persistent_local and
              parent_entry.is_persistent_local):
            # Both persistent - similar logic
            if name in self._persistent_layout:
                child_slice = self._persistent_layout[name]
                parent_slice = self._persistent_layout[parent_name]

                if (child_slice.start >= parent_slice.start and
                        child_slice.stop <= parent_slice.stop):
                    # Aliasing succeeded
                    relative_start = (
                        child_slice.start - parent_slice.start
                    )
                    relative_stop = (
                        child_slice.stop - parent_slice.start
                    )
                    aliased_parent_slice = slice(
                        relative_start, relative_stop
                    )
                else:
                    # Got fresh allocation
                    persistent_slice = child_slice
        else:
            # Cross-location: use child's layout
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
        shared_slice, persistent_slice, aliased_parent_slice,
        local_size
    )
```

**New Implementation** (simplified to ~15 lines):
```python
def get_allocator(self, name: str) -> Callable:
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
        shared_slice, persistent_slice, local_size
    )
```

**Details**:
- Remove ALL bounds-checking logic (lines 528-593)
- Remove `aliased_parent_slice` variable
- Simplify to: compute layouts, get slices from layouts, call build_allocator
- Trust that layout building already determined aliasing success
- Overlapping slices in layouts indicate aliasing succeeded
- Pass only two parameters to build_allocator (three-parameter version removed in TG3)
- Update docstring to explain that layout building is source of truth

**Edge Cases**:
- If buffer not in any layout and local_size is None: build_allocator handles with default
- Zero-size buffers: may not appear in any layout (check build_local_sizes logic)
- Aliasing succeeded: slices overlap in shared/persistent layout (automatic)

**Integration**:
- Depends on build_allocator accepting two parameters (TG3 must complete
  first)
- Called by BufferRegistry.get_allocator() which wraps this method
- All algorithm and loop code calls BufferRegistry.get_allocator()

**Outcomes**:
- Files Modified:
  * src/cubie/buffer_registry.py (100+ lines removed, 20 lines added in
    get_allocator method)
- Functions/Methods Added/Modified:
  * BufferGroup.get_allocator() - Completely rewritten and simplified
- Implementation Summary:
  * Removed ALL bounds-checking logic (60+ lines of complex conditional
    code)
  * Removed aliased_parent_slice variable and all related computation
  * Simplified to: ensure layouts computed, get slices from layouts,
    call build_allocator
  * Changed from 113 lines to ~20 lines (massive reduction)
  * Updated docstring to explain that layout building is the single
    source of truth
  * Added Notes section explaining that aliased buffers get overlapping
    slices
  * Pass only three parameters to build_allocator (removed
    aliased_parent_slice)
- Issues Flagged: None

---

## Task Group 5: Update All Allocator Call Sites - PARALLEL
**Status**: [x]
**Dependencies**: Task Group 3, Task Group 4 (allocators must be two-parameter first)

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_erk.py (lines 400-411)
- File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py (search for allocator calls)
- File: src/cubie/integrators/algorithms/generic_dirk.py (search for allocator calls)
- File: src/cubie/integrators/algorithms/generic_firk.py (search for allocator calls)
- File: src/cubie/integrators/algorithms/backwards_euler.py (search for allocator calls)
- File: src/cubie/integrators/algorithms/backwards_euler_predict_correct.py (search for allocator calls)
- File: src/cubie/integrators/algorithms/crank_nicolson.py (search for allocator calls)
- File: src/cubie/integrators/algorithms/explicit_euler.py (search for allocator calls)
- File: src/cubie/integrators/loops/ (all loop files - search for allocator calls)

**Input Validation Required**:
None - these are device function calls, not runtime validation.

**Tasks**:

### TG5.1: Update ERK Allocator Calls

- **File**: `/home/runner/work/cubie/cubie/src/cubie/integrators/algorithms/generic_erk.py`
- **Action**: Modify
- **Lines**: 402-411 (in build_step device function)

**Current Pattern**:
```python
stage_rhs = alloc_stage_rhs(
    shared, persistent_local, None
)
stage_accumulator = alloc_stage_accumulator(
    shared, persistent_local, None
)

stage_cache = alloc_stage_cache(
        shared, persistent_local, stage_accumulator
)
```

**New Pattern**:
```python
stage_rhs = alloc_stage_rhs(shared, persistent_local)
stage_accumulator = alloc_stage_accumulator(shared, persistent_local)
stage_cache = alloc_stage_cache(shared, persistent_local)
```

**Details**:
- Remove third argument from all three allocator calls
- Format: `allocator(shared, persistent_local)` only
- stage_rhs: remove `, None`
- stage_accumulator: remove `, None`
- stage_cache: remove `, stage_accumulator`
- Condense to single line per call for readability

**Edge Cases**:
- Ensure no other allocator calls in ERK file (search entire file)

**Integration**:
- Allocators now use slices from layouts instead of parent buffer references
- Aliasing still works - slices overlap in shared memory

### TG5.2: Update Rosenbrock Allocator Calls

- **File**: `/home/runner/work/cubie/cubie/src/cubie/integrators/algorithms/generic_rosenbrock_w.py`
- **Action**: Modify
- **Lines**: Search for pattern `alloc_` in device functions (build_step or build_implicit_helpers)

**Pattern to Find**:
```python
buffer_name = alloc_buffer_name(shared, persistent_local, parent_buffer)
# OR
buffer_name = alloc_buffer_name(shared, persistent_local, None)
```

**Replace With**:
```python
buffer_name = alloc_buffer_name(shared, persistent_local)
```

**Details**:
- Search entire file for `alloc_` pattern
- Remove third argument from ALL allocator calls
- Likely candidates: stage_rhs, stage_store, stage_cache, cached_auxiliaries
- Format calls as single line when possible
- Check both build_step and build_implicit_helpers methods

**Edge Cases**:
- May have multiple device functions with allocator calls
- Ensure all are updated

**Integration**:
- Connects to updated build_allocator signature
- Aliasing works via overlapping slices

### TG5.3: Update DIRK Allocator Calls

- **File**: `/home/runner/work/cubie/cubie/src/cubie/integrators/algorithms/generic_dirk.py`
- **Action**: Modify
- **Lines**: Search for `alloc_` pattern in build_step method

**Pattern**: Same as TG5.2 - remove third argument from all allocator calls

**Details**:
- Search for all `alloc_` calls in device function
- Remove third parameter
- DIRK may have: stage_rhs, stage_store, stage_cache, or similar buffers

**Edge Cases**: Same as TG5.2

**Integration**: Same as TG5.2

### TG5.4: Update FIRK Allocator Calls

- **File**: `/home/runner/work/cubie/cubie/src/cubie/integrators/algorithms/generic_firk.py`
- **Action**: Modify
- **Lines**: Search for `alloc_` pattern in build_step method

**Pattern**: Same as TG5.2 - remove third argument from all allocator calls

**Details**:
- Search for all `alloc_` calls in device function
- Remove third parameter
- FIRK may have multiple stage buffers

**Edge Cases**: Same as TG5.2

**Integration**: Same as TG5.2

### TG5.5: Update Backwards Euler Allocator Calls

- **File**: `/home/runner/work/cubie/cubie/src/cubie/integrators/algorithms/backwards_euler.py`
- **Action**: Modify
- **Lines**: Search for `alloc_` pattern in build_step method

**Pattern**: Same as TG5.2 - remove third argument from all allocator calls

**Details**:
- Search for all `alloc_` calls in device function
- Remove third parameter

**Edge Cases**: Same as TG5.2

**Integration**: Same as TG5.2

### TG5.6: Update Backwards Euler Predict-Correct Allocator Calls

- **File**: `/home/runner/work/cubie/cubie/src/cubie/integrators/algorithms/backwards_euler_predict_correct.py`
- **Action**: Modify
- **Lines**: Search for `alloc_` pattern in build_step method

**Pattern**: Same as TG5.2 - remove third argument from all allocator calls

**Details**:
- Search for all `alloc_` calls in device function
- Remove third parameter

**Edge Cases**: Same as TG5.2

**Integration**: Same as TG5.2

### TG5.7: Update Crank-Nicolson Allocator Calls

- **File**: `/home/runner/work/cubie/cubie/src/cubie/integrators/algorithms/crank_nicolson.py`
- **Action**: Modify
- **Lines**: Search for `alloc_` pattern in build_step method

**Pattern**: Same as TG5.2 - remove third argument from all allocator calls

**Details**:
- Search for all `alloc_` calls in device function
- Remove third parameter

**Edge Cases**: Same as TG5.2

**Integration**: Same as TG5.2

### TG5.8: Update Explicit Euler Allocator Calls

- **File**: `/home/runner/work/cubie/cubie/src/cubie/integrators/algorithms/explicit_euler.py`
- **Action**: Modify
- **Lines**: Search for `alloc_` pattern in build_step method

**Pattern**: Same as TG5.2 - remove third argument from all allocator calls

**Details**:
- Search for all `alloc_` calls in device function
- Remove third parameter
- Explicit Euler is simple and may have minimal buffers

**Edge Cases**: Same as TG5.2

**Integration**: Same as TG5.2

### TG5.9: Update Loop Files Allocator Calls

- **File**: `/home/runner/work/cubie/cubie/src/cubie/integrators/loops/ode_loop.py`
- **Action**: Modify
- **Lines**: Search entire file for `alloc_` pattern

**Pattern**: Same as TG5.2 - remove third argument from all allocator calls

**Details**:
- Loop files may allocate buffers for output handling or state management
- Search for all `alloc_` calls
- Remove third parameter from each

**Edge Cases**:
- May have allocator calls in multiple device functions
- Ensure all are updated

**Integration**:
- Loop integrates with algorithm step functions
- Must use same two-parameter allocator interface

### TG5.10: Search for Any Other Allocator Calls

- **Files**: All files in `/home/runner/work/cubie/cubie/src/cubie/integrators/`
- **Action**: Search and modify
- **Lines**: Entire integrators module tree

**Pattern**: Search for regex `alloc_\w+\s*\([^)]+,\s*[^)]+,\s*[^)]+\)` to find three-argument calls

**Details**:
- Use grep or IDE search to find any remaining three-argument allocator calls
- Pattern to search: `alloc_` followed by three arguments
- Update each to two-argument form
- Document findings in outcomes

**Edge Cases**:
- May find allocator calls in unexpected locations
- Test files should NOT be modified (they test the interface, not use it in production)

**Integration**: Ensures complete coverage of all call sites

**Outcomes**:
- Files Modified:
  * src/cubie/integrators/algorithms/generic_erk.py (3 allocator calls updated)
  * src/cubie/integrators/algorithms/generic_dirk.py (8 allocator calls updated)
  * src/cubie/integrators/algorithms/generic_firk.py (5 allocator calls updated)
  * src/cubie/integrators/algorithms/generic_rosenbrock_w.py (3 allocator calls updated)
  * src/cubie/integrators/algorithms/backwards_euler.py (2 allocator calls updated)
  * src/cubie/integrators/algorithms/backwards_euler_predict_correct.py (0 - already uses 2-parameter form)
  * src/cubie/integrators/algorithms/crank_nicolson.py (2 allocator calls updated)
  * src/cubie/integrators/algorithms/explicit_euler.py (0 - no allocator calls found)
  * src/cubie/integrators/loops/ode_loop.py (16 allocator calls updated)
  * src/cubie/integrators/matrix_free_solvers/newton_krylov.py (6 allocator calls updated)
  * src/cubie/integrators/matrix_free_solvers/linear_solver.py (4 allocator calls updated - 2 in cached variant, 2 in non-cached variant)
- Functions/Methods Modified:
  * ERKStep.build_step() - Updated 3 allocator calls (stage_rhs, stage_accumulator, stage_cache)
  * DIRKStep.build_step() - Updated 8 allocator calls (stage_increment, accumulator, stage_base, solver_shared, solver_persistent, rhs_cache, increment_cache, stage_rhs)
  * FIRKStep.build_step() - Updated 5 allocator calls (stage_state, solver_shared, solver_persistent, stage_increment, stage_driver_stack)
  * GenericRosenbrockWStep.build_step() - Updated 3 allocator calls (stage_rhs, stage_store, cached_auxiliaries)
  * BackwardsEulerStep.build_step() - Updated 2 allocator calls (solver_scratch, solver_persistent)
  * CrankNicolsonStep.build_step() - Updated 2 allocator calls (solver_scratch, solver_persistent)
  * IVPLoop.build() loop_fn device function - Updated 16 allocator calls (state_buffer, state_proposal_buffer, observables_buffer, observables_proposal_buffer, parameters_buffer, drivers_buffer, drivers_proposal_buffer, state_summary_buffer, observable_summary_buffer, counters_since_save, error, algo_shared, algo_persistent, controller_temp, dt, accept_step)
  * NewtonKrylov.build() newton_krylov_solver device function - Updated 6 allocator calls (delta, residual, residual_temp, stage_base_bt, lin_shared, lin_persistent)
  * LinearSolver.build() linear_solver device functions - Updated 4 allocator calls (2 in cached variant: preconditioned_vec, temp; 2 in non-cached variant: preconditioned_vec, temp)
- Implementation Summary:
  * Removed third argument from all allocator calls across 10 files
  * Changed pattern from `allocator(shared, persistent, X)` to `allocator(shared, persistent)`
  * Total of 49 allocator calls updated (not 57 as estimated - some files already correct or don't use allocators)
  * Condensed multi-line allocator calls to single lines where appropriate for readability
  * Maintained PEP8 79-character line limit throughout
  * No logic changes - only parameter removal
- Issues Flagged: None

---

## Task Group 6: Verify ERK and Rosenbrock Aliasing - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Groups 1, 2, 3, 4, 5 (all code changes must be complete)

**Required Context**:
- File: tests/ (entire test directory for finding relevant tests)
- File: src/cubie/integrators/algorithms/generic_erk.py (final state after TG1, TG5)
- File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py (final state after TG2, TG5)

**Input Validation Required**:
None - this is testing/verification, not implementation.

**Tasks**:

### TG6.1: Create Verification Test for ERK Stage Cache Aliasing

- **File**: `/home/runner/work/cubie/cubie/tests/integrators/algorithms/test_erk_aliasing.py` (create new)
- **Action**: Create
- **Lines**: N/A (new file)

**Details**:
```python
"""Test ERK stage_cache aliasing behavior after simplification."""

import pytest
import numpy as np
from cubie.buffer_registry import buffer_registry
from cubie.integrators.algorithms.generic_erk import ERKStep
from cubie.integrators.algorithms.generic_erk_tableaus import RK4_TABLEAU, DOPRI5_TABLEAU


def test_erk_stage_cache_aliases_shared_rhs():
    """When stage_rhs is shared, stage_cache should alias it."""
    step = ERKStep(
        tableau=RK4_TABLEAU,
        stage_rhs_location='shared',
        stage_accumulator_location='shared',
    )
    
    group = buffer_registry._groups[step]
    shared_layout = group.shared_layout
    
    # Both stage_rhs and stage_cache should be in shared layout
    assert 'stage_rhs' in shared_layout
    assert 'stage_cache' in shared_layout
    
    # stage_cache should overlap stage_rhs (aliasing succeeded)
    rhs_slice = shared_layout['stage_rhs']
    cache_slice = shared_layout['stage_cache']
    
    # Aliasing: cache slice should be within rhs bounds
    assert cache_slice.start >= rhs_slice.start
    assert cache_slice.stop <= rhs_slice.stop


def test_erk_stage_cache_aliases_shared_accumulator():
    """When stage_rhs is local but accumulator is shared, alias accumulator."""
    step = ERKStep(
        tableau=DOPRI5_TABLEAU,
        stage_rhs_location='local',
        stage_accumulator_location='shared',
    )
    
    group = buffer_registry._groups[step]
    shared_layout = group.shared_layout
    
    # stage_accumulator and stage_cache should be in shared layout
    assert 'stage_accumulator' in shared_layout
    assert 'stage_cache' in shared_layout
    
    # stage_cache should overlap stage_accumulator
    acc_slice = shared_layout['stage_accumulator']
    cache_slice = shared_layout['stage_cache']
    
    assert cache_slice.start >= acc_slice.start
    assert cache_slice.stop <= acc_slice.stop


def test_erk_stage_cache_persistent_when_both_local():
    """When both rhs and accumulator are local, use persistent local."""
    step = ERKStep(
        tableau=RK4_TABLEAU,
        stage_rhs_location='local',
        stage_accumulator_location='local',
    )
    
    group = buffer_registry._groups[step]
    persistent_layout = group._persistent_layout
    if persistent_layout is None:
        persistent_layout = group.build_persistent_layout()
    
    # stage_cache should be in persistent layout
    assert 'stage_cache' in persistent_layout
    
    # stage_cache should NOT be in shared layout
    shared_layout = group.shared_layout
    assert 'stage_cache' not in shared_layout
```

**Edge Cases**:
- Test with both adaptive (DOPRI5) and fixed (RK4) tableaus
- Test all three preference scenarios

**Integration**:
- Run with `pytest tests/integrators/algorithms/test_erk_aliasing.py`
- Should pass if TG1 implemented correctly

### TG6.2: Create Verification Test for Rosenbrock Stage Cache Aliasing

- **File**: `/home/runner/work/cubie/cubie/tests/integrators/algorithms/test_rosenbrock_aliasing.py` (create new)
- **Action**: Create
- **Lines**: N/A (new file)

**Details**:
```python
"""Test Rosenbrock stage_cache aliasing behavior after simplification."""

import pytest
import numpy as np
from cubie.buffer_registry import buffer_registry
from cubie.integrators.algorithms.generic_rosenbrock_w import GenericRosenbrockWStep
from cubie.integrators.algorithms.generic_rosenbrockw_tableaus import ROS3P_TABLEAU


def test_rosenbrock_stage_cache_aliases_shared_stage_store():
    """When stage_store is shared, stage_cache should alias it."""
    # Create minimal ODE system for Rosenbrock initialization
    from cubie.odesystems.symbolicODE import SymbolicODE
    system = SymbolicODE(
        state_names=['x'],
        parameter_names=[],
        expressions=['x']
    )
    
    step = GenericRosenbrockWStep(
        system=system,
        tableau=ROS3P_TABLEAU,
        stage_store_location='shared',
        stage_rhs_location='shared',
    )
    
    group = buffer_registry._groups[step]
    shared_layout = group.shared_layout
    
    # Both stage_store and stage_cache should be in shared layout
    assert 'stage_store' in shared_layout
    assert 'stage_cache' in shared_layout
    
    # stage_cache should overlap stage_store
    store_slice = shared_layout['stage_store']
    cache_slice = shared_layout['stage_cache']
    
    assert cache_slice.start >= store_slice.start
    assert cache_slice.stop <= store_slice.stop


def test_rosenbrock_stage_cache_persistent_when_store_local():
    """When stage_store is local, stage_cache uses persistent local."""
    from cubie.odesystems.symbolicODE import SymbolicODE
    system = SymbolicODE(
        state_names=['x'],
        parameter_names=[],
        expressions=['x']
    )
    
    step = GenericRosenbrockWStep(
        system=system,
        tableau=ROS3P_TABLEAU,
        stage_store_location='local',
        stage_rhs_location='shared',
    )
    
    group = buffer_registry._groups[step]
    persistent_layout = group._persistent_layout
    if persistent_layout is None:
        persistent_layout = group.build_persistent_layout()
    
    # stage_cache should be in persistent layout
    assert 'stage_cache' in persistent_layout
    
    # stage_cache should NOT be in shared layout
    shared_layout = group.shared_layout
    assert 'stage_cache' not in shared_layout
```

**Edge Cases**:
- Rosenbrock requires ODE system for initialization (linear solver setup)
- Test both shared and local stage_store scenarios

**Integration**:
- Run with `pytest tests/integrators/algorithms/test_rosenbrock_aliasing.py`
- Should pass if TG2 implemented correctly

### TG6.3: Run Full Integration Test Suite

- **File**: N/A (command-line)
- **Action**: Execute tests
- **Lines**: N/A

**Command**:
```bash
pytest tests/integrators/algorithms/ -v
```

**Details**:
- Run all algorithm tests to ensure no regressions
- Look for any failures related to buffer allocation
- Check that ERK, DIRK, FIRK, Rosenbrock, and other algorithms still work
- Verify no memory errors or CUDA kernel launch failures

**Expected Results**:
- All existing tests should pass
- No new failures introduced
- stage_cache aliasing tests (TG6.1, TG6.2) should pass

**Edge Cases**:
- Tests marked `nocudasim` will fail without GPU (expected)
- Tests marked `cupy` will fail without CuPy installed (expected)
- Focus on tests that were passing before changes

**Integration**:
- Final verification that all changes work together
- Confirms aliasing still works correctly
- Validates that simplification didn't break functionality

**Outcomes**:
[Empty - to be filled by taskmaster agent]

---

## Summary

**Total Task Groups**: 6

**Dependency Chain**:
- TG1, TG2: Independent (can run in parallel)
- TG3: Independent (can run in parallel with TG1, TG2)
- TG4: Depends on TG3 (must wait for two-parameter allocator)
- TG5: Depends on TG3, TG4 (must wait for allocator signature changes)
- TG6: Depends on all previous groups (final verification)

**Parallel Execution Opportunities**:
- TG1 and TG2 can run in parallel (different files)
- TG3 can run in parallel with TG1, TG2 (buffer_registry.py vs. algorithm files)
- TG5 subtasks (TG5.1-TG5.10) can run in parallel (different files)

**Estimated Complexity**:
- TG1: Medium (conditional logic restoration)
- TG2: Low-Medium (similar to TG1 but simpler)
- TG3: Medium (signature change + docstring update)
- TG4: High (removing 60+ lines of complex logic, replacing with simple logic)
- TG5: Low-Medium (repetitive change across many files)
- TG6: Medium (creating new tests + running test suite)

**Critical Path**:
TG3 → TG4 → TG5 → TG6 (allocator signature must change before call sites can be updated)

**Files Modified** (estimated):
- buffer_registry.py: 2 methods (build_allocator, get_allocator)
- generic_erk.py: registration + allocator calls
- generic_rosenbrock_w.py: registration + allocator calls
- ode_explicitstep.py: config class (remove field)
- ode_implicitstep.py or generic_rosenbrock_w.py: config class (remove field)
- generic_dirk.py: allocator calls
- generic_firk.py: allocator calls
- backwards_euler.py: allocator calls
- backwards_euler_predict_correct.py: allocator calls
- crank_nicolson.py: allocator calls
- explicit_euler.py: allocator calls
- ode_loop.py: allocator calls (if any)
- 2 new test files

**Total Estimated Changes**: ~14-16 files modified, 2 files created
