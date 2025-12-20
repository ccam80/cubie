# Implementation Task List
# Feature: Fix Buffer Aliasing Implementation
# Plan Reference: .github/active_plans/fix_buffer_aliasing_implementation/agent_plan.md

## Overview

This task list corrects fundamental semantic errors in the three-parameter allocator implementation for buffer aliasing. The core issues are:
1. Wrong parameter names (`shared_fallback` should be `aliased_parent`)
2. Misunderstanding of aliasing semantics (should deliberately overlap, not avoid it)
3. Manual aliasing logic in step functions that belongs in buffer_registry
4. Confusing "fallback" layout when there's just one shared array

Total Task Groups: 6
Dependency Chain: TG1 → TG2 → TG3 → TG4, TG5 (parallel), TG6
Parallel Execution: TG5 can run independently before TG4

---

## Task Group 1: Rename Parameters in build_allocator - SEQUENTIAL
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: /home/runner/work/cubie/cubie/src/cubie/buffer_registry.py (lines 75-136)

**Input Validation Required**:
None (parameter renaming only, no runtime validation needed)

**Tasks**:

### TG1.1: Rename build_allocator signature parameter
- **File**: /home/runner/work/cubie/cubie/src/cubie/buffer_registry.py
- **Action**: Modify
- **Lines**: 75-81
- **Details**:
  Change parameter name from `shared_fallback_slice` to `aliased_parent_slice`:
  ```python
  def build_allocator(
      self,
      shared_slice: Optional[slice],
      persistent_slice: Optional[slice],
      aliased_parent_slice: Optional[slice],  # Changed from shared_fallback_slice
      local_size: Optional[int],
  ) -> Callable:
  ```
- **Edge cases**: None (simple renaming)
- **Integration**: Update all call sites in get_allocator and throughout

### TG1.2: Rename allocator device function parameter
- **File**: /home/runner/work/cubie/cubie/src/cubie/buffer_registry.py
- **Action**: Modify
- **Lines**: 122-124
- **Details**:
  Change device function parameter from `shared_fallback` to `aliased_parent`:
  ```python
  @cuda.jit(device=True, inline=True, **compile_kwargs)
  def allocate_buffer(
      shared, persistent, aliased_parent  # Changed from shared_parent, persistent_parent, shared_fallback
  ):
  ```
- **Edge cases**: None
- **Integration**: Matches new semantic meaning

### TG1.3: Rename internal closure variables
- **File**: /home/runner/work/cubie/cubie/src/cubie/buffer_registry.py
- **Action**: Modify
- **Lines**: 108-119
- **Details**:
  Rename closure-captured variables for consistency:
  ```python
  _use_shared = shared_slice is not None
  _use_persistent = persistent_slice is not None
  _use_aliased_parent = aliased_parent_slice is not None  # Changed from _use_shared_fallback
  _shared_slice = shared_slice if _use_shared else slice(0, 0)
  _persistent_slice = (
      persistent_slice if _use_persistent else slice(0, 0)
  )
  _aliased_parent_slice = (  # Changed from _shared_fallback_slice
      aliased_parent_slice if _use_aliased_parent else slice(0, 0)
  )
  _local_size = local_size if local_size is not None else 1
  _precision = self.precision
  ```
- **Edge cases**: None
- **Integration**: Used in device function body

### TG1.4: Update allocator device function body
- **File**: /home/runner/work/cubie/cubie/src/cubie/buffer_registry.py
- **Action**: Modify
- **Lines**: 125-135
- **Details**:
  Update conditional logic to use renamed variables:
  ```python
  def allocate_buffer(
      shared, persistent, aliased_parent
  ):
      """Allocate buffer from appropriate memory region."""
      if _use_aliased_parent:  # Priority 1: slice parent if provided
          array = aliased_parent[_aliased_parent_slice]
      elif _use_persistent:   # Priority 2: persistent local
          array = persistent[_persistent_slice]
      elif _use_shared:       # Priority 3: fresh shared allocation
          array = shared[_shared_slice]
      else:                   # Priority 4: local array
          array = cuda.local.array(_local_size, _precision)
      return array
  ```
- **Edge cases**: Order matters - aliased_parent has highest priority
- **Integration**: This is the core semantic fix

### TG1.5: Update docstring
- **File**: /home/runner/work/cubie/cubie/src/cubie/buffer_registry.py
- **Action**: Modify
- **Lines**: 82-106
- **Details**:
  Update parameter documentation:
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
- **Edge cases**: None
- **Integration**: Documentation matches implementation

**Outcomes**:
```
After TG1 completion:
- Files Modified:
  * src/cubie/buffer_registry.py (57 lines changed in build_allocator method)
  * src/cubie/buffer_registry.py (4 lines changed in get_allocator method)
- Functions/Methods Modified:
  * CUDABuffer.build_allocator() - signature parameter renamed
  * allocate_buffer device function - parameter names updated
  * BufferGroup.get_allocator() - call site updated
- Implementation Summary:
  * TG1.1: Renamed parameter shared_fallback_slice → aliased_parent_slice (line 79)
  * TG1.2: Renamed device function parameters: shared_parent → shared, 
    persistent_parent → persistent, shared_fallback → aliased_parent (lines 120-122)
  * TG1.3: Renamed internal flags: _use_shared_fallback → _use_aliased_parent,
    _shared_fallback_slice → _aliased_parent_slice (lines 108, 113-115)
  * TG1.4: Updated docstring to reflect new parameter names and semantics (lines 82-103)
  * TG1.5: Updated device function logic: changed conditional from 
    _use_shared_fallback to _use_aliased_parent and reordered priority 
    (aliased_parent now has highest priority) (lines 124-132)
  * Additional: Updated get_allocator call site to use aliased_parent_slice (lines 593, 598)
- Issues Flagged: None
```

---

## Task Group 2: Fix build_shared_layout to Return Single Dict - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Group 1

**Required Context**:
- File: /home/runner/work/cubie/cubie/src/cubie/buffer_registry.py (lines 282-361)
- File: /home/runner/work/cubie/cubie/src/cubie/buffer_registry.py (lines 139-173, BufferGroup attributes)

**Input Validation Required**:
None (internal layout computation, validated at registration time)

**Tasks**:

### TG2.1: Change BufferGroup._shared_layout type annotation
- **File**: /home/runner/work/cubie/cubie/src/cubie/buffer_registry.py
- **Action**: Modify
- **Lines**: 162-164
- **Details**:
  Change from Tuple to single Dict:
  ```python
  _shared_layout: Optional[Dict[str, slice]] = (  # Changed from Tuple[Dict[str, slice], Dict[str, slice]]
      attrs.field(default=None, init=False)
  )
  ```
- **Edge cases**: None
- **Integration**: Affects all properties and methods accessing _shared_layout

### TG2.2: Rewrite build_shared_layout to return single dict
- **File**: /home/runner/work/cubie/cubie/src/cubie/buffer_registry.py
- **Action**: Modify
- **Lines**: 282-361
- **Details**:
  Replace entire function with correct aliasing logic:
  ```python
  def build_shared_layout(self) -> Dict[str, slice]:
      """Compute slice indices for shared memory buffers.

      Implements correct aliasing semantics:
      - Child slices parent when parent is shared and has space
      - Child deliberately OVERLAPS parent (reuses memory)
      - Child gets fresh allocation when parent full or local
      - All allocations use the SAME shared array

      Returns
      -------
      Dict[str, slice]
          Single unified mapping of buffer names to shared memory slices.
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
                  # Alias within parent (WITH OVERLAP - reuses parent memory)
                  parent_slice = layout[entry.aliases]
                  start = parent_slice.start + consumed
                  layout[name] = slice(start, start + entry.size)
                  self._alias_consumption[entry.aliases] = consumed + entry.size
              else:
                  # Parent full, allocate fresh shared space
                  layout[name] = slice(offset, offset + entry.size)
                  offset += entry.size
          else:
              # Parent is local, allocate fresh shared space for child
              layout[name] = slice(offset, offset + entry.size)
              offset += entry.size

      return layout
  ```
- **Edge cases**:
  - Multiple children aliasing same parent (consumption tracking)
  - Child larger than parent (gets fresh allocation)
  - Parent is local (child gets fresh allocation)
- **Integration**: Replaces tuple return with single dict

### TG2.3: Update BufferGroup docstring
- **File**: /home/runner/work/cubie/cubie/src/cubie/buffer_registry.py
- **Action**: Modify
- **Lines**: 141-158
- **Details**:
  Update BufferGroup class docstring:
  ```python
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
      Tracks consumed space in aliased buffers for layout computation.
  """
  ```
- **Edge cases**: None
- **Integration**: Documentation matches new implementation

**Outcomes**:
```
After TG2 completion:
- build_shared_layout returns single Dict[str, slice]
- Aliased buffers slice parent WITH OVERLAP when parent is shared
- Fresh allocations when parent is local or full
- _alias_consumption tracks how much of parent is used
```

---

## Task Group 3: Remove Fallback Properties and Update shared_buffer_size - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Group 2

**Required Context**:
- File: /home/runner/work/cubie/cubie/src/cubie/buffer_registry.py (lines 363-398, 485-510, 540-558)

**Input Validation Required**:
None (internal helper methods)

**Tasks**:

### TG3.1: Remove shared_primary_layout property
- **File**: /home/runner/work/cubie/cubie/src/cubie/buffer_registry.py
- **Action**: Delete
- **Lines**: 363-377
- **Details**:
  Delete entire property:
  ```python
  @property
  def shared_primary_layout(self) -> Dict[str, slice]:
      """Return primary (aliased) shared memory layout...
      ...
      """
      if self._shared_layout is None:
          self._shared_layout = self.build_shared_layout()
      return self._shared_layout[0]
  ```
- **Edge cases**: None
- **Integration**: Replaced by single shared_layout property

### TG3.2: Remove shared_fallback_layout property
- **File**: /home/runner/work/cubie/cubie/src/cubie/buffer_registry.py
- **Action**: Delete
- **Lines**: 379-397
- **Details**:
  Delete entire property:
  ```python
  @property
  def shared_fallback_layout(self) -> Dict[str, slice]:
      """Return fallback shared memory layout...
      ...
      """
      if self._shared_layout is None:
          self._shared_layout = self.build_shared_layout()
      return self._shared_layout[1]
  ```
- **Edge cases**: None
- **Integration**: No longer needed with single layout

### TG3.3: Add single shared_layout property
- **File**: /home/runner/work/cubie/cubie/src/cubie/buffer_registry.py
- **Action**: Create
- **Insert After**: Line 361 (after build_shared_layout method)
- **Details**:
  Add new property:
  ```python
  @property
  def shared_layout(self) -> Dict[str, slice]:
      """Return unified shared memory layout.

      Computes layout on first access and caches result.
      Layout is invalidated when any buffer is modified.

      Returns
      -------
      Dict[str, slice]
          Mapping of buffer names to shared memory slices.
      """
      if self._shared_layout is None:
          self._shared_layout = self.build_shared_layout()
      return self._shared_layout
  ```
- **Edge cases**: None
- **Integration**: Single source of truth for shared layout

### TG3.4: Update shared_buffer_size method
- **File**: /home/runner/work/cubie/cubie/src/cubie/buffer_registry.py
- **Action**: Modify
- **Lines**: 485-510
- **Details**:
  Simplify to use single layout:
  ```python
  def shared_buffer_size(self) -> int:
      """Return total shared memory elements.

      Returns end of last slice in unified shared layout.

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
  ```
- **Edge cases**: Empty layout returns 0
- **Integration**: Uses single layout dict

### TG3.5: Remove shared_fallback_buffer_size method
- **File**: /home/runner/work/cubie/cubie/src/cubie/buffer_registry.py
- **Action**: Delete
- **Lines**: 540-558
- **Details**:
  Delete entire method:
  ```python
  def shared_fallback_buffer_size(self) -> int:
      """Return fallback shared memory elements for this parent...
      ...
      """
      if self._shared_layout is None:
          self._shared_layout = self.build_shared_layout()

      primary_layout, fallback_layout = self._shared_layout

      if not fallback_layout:
          return 0
      return max(s.stop for s in fallback_layout.values())
  ```
- **Edge cases**: None
- **Integration**: No longer needed - single shared_buffer_size suffices

### TG3.6: Update build_local_sizes to use single layout
- **File**: /home/runner/work/cubie/cubie/src/cubie/buffer_registry.py
- **Action**: Modify
- **Lines**: 454-483
- **Details**:
  Fix layout unpacking:
  ```python
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
              # Check if buffer already allocated via aliasing
              if (name in self._shared_layout  # Changed: no tuple unpacking
                      or name in self._persistent_layout):
                  # Already allocated elsewhere, skip local allocation
                  continue
              # cuda.local.array requires size >= 1
              sizes[name] = max(entry.size, 1)
      return sizes
  ```
- **Edge cases**: Local buffers that alias shared/persistent parents
- **Integration**: Uses single shared_layout dict

**Outcomes**:
```
After TG3 completion:
- shared_primary_layout and shared_fallback_layout properties removed
- Single shared_layout property added
- shared_buffer_size simplified to use single layout
- shared_fallback_buffer_size method removed
- build_local_sizes uses single layout
```

---

## Task Group 4: Fix get_allocator Logic - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Groups 1, 2, 3

**Required Context**:
- File: /home/runner/work/cubie/cubie/src/cubie/buffer_registry.py (lines 560-602)

**Input Validation Required**:
None (internal method, buffer existence validated at registration)

**Tasks**:

### TG4.1: Rewrite get_allocator method
- **File**: /home/runner/work/cubie/cubie/src/cubie/buffer_registry.py
- **Action**: Modify
- **Lines**: 560-602
- **Details**:
  Complete rewrite with correct aliasing logic:
  ```python
  def get_allocator(self, name: str) -> Callable:
      """Generate CUDA device function for buffer allocation.

      Determines allocation strategy based on buffer properties and
      whether aliasing succeeds. When aliasing succeeds, passes parent
      buffer for slicing. Otherwise passes shared or appropriate location.

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

                  # If child slice is WITHIN parent bounds, aliasing succeeded
                  if (child_slice.start >= parent_slice.start and
                      child_slice.stop <= parent_slice.stop):
                      # Aliasing succeeded - compute relative slice
                      relative_start = child_slice.start - parent_slice.start
                      relative_stop = child_slice.stop - parent_slice.start
                      aliased_parent_slice = slice(relative_start, relative_stop)
                  else:
                      # Got fresh allocation (parent was full)
                      shared_slice = child_slice
          elif entry.is_persistent_local and parent_entry.is_persistent_local:
              # Both persistent - similar logic
              if name in self._persistent_layout:
                  child_slice = self._persistent_layout[name]
                  parent_slice = self._persistent_layout[parent_name]

                  if (child_slice.start >= parent_slice.start and
                      child_slice.stop <= parent_slice.stop):
                      # Aliasing succeeded
                      relative_start = child_slice.start - parent_slice.start
                      relative_stop = child_slice.stop - parent_slice.start
                      aliased_parent_slice = slice(relative_start, relative_stop)
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
          shared_slice, persistent_slice, aliased_parent_slice, local_size
      )
  ```
- **Edge cases**:
  - Child larger than parent: child_slice won't be within parent bounds
  - Parent local, child shared: cross-location, use child's layout
  - Multiple children: each checks independently
- **Integration**: Critical method - determines what gets passed to allocators

**Outcomes**:
```
After TG4 completion:
- get_allocator determines aliased_parent_slice vs shared_slice correctly
- When aliasing succeeds: passes relative slice into parent
- When aliasing fails: passes fresh allocation from appropriate layout
- Handles cross-location aliasing (parent local, child shared)
```

---

## Task Group 5: Remove Manual Aliasing Logic from Step Functions - PARALLEL
**Status**: [ ]
**Dependencies**: None (can run in parallel with TG1-4, or after)

**Required Context**:
- File: /home/runner/work/cubie/cubie/src/cubie/integrators/algorithms/generic_dirk.py (lines 252-268)
- File: /home/runner/work/cubie/cubie/src/cubie/integrators/algorithms/generic_rosenbrock_w.py (find similar pattern)

**Input Validation Required**:
None (buffer registration, validation happens in buffer_registry)

**Tasks**:

### TG5.1: Remove manual aliasing in generic_dirk.py
- **File**: /home/runner/work/cubie/cubie/src/cubie/integrators/algorithms/generic_dirk.py
- **Action**: Modify
- **Lines**: 252-268
- **Details**:
  Replace conditional registration with single unconditional call:
  ```python
  # OLD (REMOVE):
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

  # NEW (REPLACE WITH):
  buffer_registry.register(
      'stage_base', self, n, config.stage_base_location,
      aliases='accumulator', precision=precision
  )
  ```
- **Edge cases**:
  - If accumulator is local, registry handles it (gives stage_base fresh allocation)
  - If both shared but accumulator too small, registry handles it
- **Integration**: Let buffer_registry decide whether aliasing succeeds

### TG5.2: Find and remove manual aliasing in generic_rosenbrock_w.py
- **File**: /home/runner/work/cubie/cubie/src/cubie/integrators/algorithms/generic_rosenbrock_w.py
- **Action**: Modify
- **Search Pattern**: Look for similar if/else blocks around stage_cache registration
- **Details**:
  Find code similar to:
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
  
  Replace with single unconditional register:
  ```python
  buffer_registry.register(
      'stage_cache', self, n, config.stage_cache_location,
      aliases='stage_store', precision=precision
  )
  ```
  
  Note: May need to add `stage_cache_location` to config if not present.
  Check `generic_rosenbrock_w.py` for current structure.
- **Edge cases**: stage_cache_location config parameter may not exist yet
- **Integration**: Verify config has stage_cache_location parameter

### TG5.3: Check other algorithm files for manual aliasing
- **Files**: 
  - /home/runner/work/cubie/cubie/src/cubie/integrators/algorithms/generic_erk.py
  - /home/runner/work/cubie/cubie/src/cubie/integrators/algorithms/generic_firk.py
  - /home/runner/work/cubie/cubie/src/cubie/integrators/algorithms/backwards_euler.py
  - /home/runner/work/cubie/cubie/src/cubie/integrators/algorithms/crank_nicolson.py
- **Action**: Search
- **Search Pattern**: `if.*location.*aliases`
- **Details**:
  Search each file for conditional aliasing logic. If found, apply same pattern:
  - Remove if/else checking locations
  - Use single register() call with aliases parameter
  - Let buffer_registry decide if aliasing succeeds
- **Edge cases**: Each algorithm may have different buffer names
- **Integration**: Consistent pattern across all algorithms

**Outcomes**:
```
After TG5 completion:
- No manual aliasing logic in generic_dirk.py
- No manual aliasing logic in generic_rosenbrock_w.py
- No manual aliasing logic in any algorithm files
- All algorithms register buffers once with aliases parameter
- buffer_registry makes all aliasing decisions
```

---

## Task Group 6: Update All Allocator Call Sites - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Groups 1, 2, 3, 4

**Required Context**:
- File: /home/runner/work/cubie/cubie/src/cubie/integrators/loops/ode_loop.py (lines 438-497)
- All algorithm step function files (any that call allocators)

**Input Validation Required**:
None (allocator calls, parameters validated by allocator itself)

**Tasks**:

### TG6.1: Update ode_loop.py allocator calls
- **File**: /home/runner/work/cubie/cubie/src/cubie/integrators/loops/ode_loop.py
- **Action**: Modify
- **Lines**: 438-497
- **Details**:
  For each allocator call, the third argument should be the correct buffer based on aliasing:
  
  **Pattern for non-aliased buffers** (most common):
  ```python
  buffer = alloc_buffer(
      shared_scratch, persistent_local, shared_scratch  # 3rd arg: shared_scratch for non-aliased
  )
  ```
  
  **Pattern for aliased buffers** (when buffer aliases another):
  The third argument should be the parent buffer variable, NOT shared_scratch.
  
  Example call sites to update (lines 448-497):
  ```python
  # Line 448-450: state_buffer (non-aliased)
  state_buffer = alloc_state(
      shared_scratch, persistent_local, shared_scratch
  )
  
  # Line 451-453: state_proposal_buffer (non-aliased)
  state_proposal_buffer = alloc_proposed_state(
      shared_scratch, persistent_local, shared_scratch
  )
  
  # Line 454-456: observables_buffer (non-aliased)
  observables_buffer = alloc_observables(
      shared_scratch, persistent_local, shared_scratch
  )
  
  # Continue for all 16 allocator calls in this section
  # For each one, determine if it aliases another buffer
  # If yes: third arg is parent buffer variable
  # If no: third arg is shared_scratch
  ```
  
  **Critical**: Need to trace which buffers alias which:
  - Check buffer_registry.register() calls in ode_loop.__init__()
  - If buffer has aliases parameter, third arg should be that parent buffer
  - Otherwise third arg is shared_scratch
  
- **Edge cases**:
  - Buffer aliases another that hasn't been allocated yet (order matters)
  - Buffer aliases parent in different location (registry handles this)
- **Integration**: Must coordinate with buffer_registry.register() calls in __init__

### TG6.2: Rename shared_fallback variable to shared_scratch
- **File**: /home/runner/work/cubie/cubie/src/cubie/integrators/loops/ode_loop.py
- **Action**: Modify
- **Lines**: 439-443
- **Details**:
  Variable is already named `shared_scratch` in parameters, but there's also a
  `shared_fallback` allocation. Check if they're the same or different arrays.
  
  If separate: rename `shared_fallback` allocation to just use `shared_scratch`:
  ```python
  # OLD:
  shared_fallback = cuda.shared.array(
      max(shared_fallback_size, 1), precision
  )
  
  # This might be redundant - check if shared_scratch is already allocated
  # If so, remove this allocation entirely and use shared_scratch everywhere
  ```
- **Edge cases**: Check if shared_scratch is already allocated as parameter
- **Integration**: Semantic fix - one shared array, not two

### TG6.3: Update algorithm step function allocator calls
- **Files**: All algorithm step function files that allocate buffers
  - /home/runner/work/cubie/cubie/src/cubie/integrators/algorithms/generic_dirk.py
  - /home/runner/work/cubie/cubie/src/cubie/integrators/algorithms/generic_rosenbrock_w.py
  - /home/runner/work/cubie/cubie/src/cubie/integrators/algorithms/generic_erk.py
  - /home/runner/work/cubie/cubie/src/cubie/integrators/algorithms/generic_firk.py
  - Others as needed
- **Action**: Search and Modify
- **Search Pattern**: `alloc_.*\(.*shared.*,.*persistent.*,`
- **Details**:
  For each file:
  1. Search for allocator calls in step device function
  2. For each call, determine if buffer aliases another:
     - Check __init__ buffer_registry.register() for aliases parameter
     - If aliases: third arg should be parent buffer variable
     - If no aliases: third arg should be shared (or shared_scratch)
  3. Update call accordingly
  
  Example in generic_dirk step function (if it exists):
  ```python
  # If stage_base aliases accumulator:
  stage_base = alloc_stage_base(
      shared, persistent, accumulator  # 3rd arg: parent buffer (accumulator)
  )
  
  # If stage_increment doesn't alias:
  stage_increment = alloc_stage_increment(
      shared, persistent, shared  # 3rd arg: shared array for fresh allocation
  )
  ```
- **Edge cases**:
  - Some algorithm files may not have allocator calls in step function
  - Some may call allocators in __init__ only
- **Integration**: Verify against buffer_registry.register() calls in each __init__

### TG6.4: Search entire codebase for remaining allocator calls
- **Files**: Entire repository
- **Action**: Search
- **Search Pattern**: `alloc_.*\(.*,.*,.*shared_fallback`
- **Details**:
  Final sweep to catch any remaining calls with old parameter name:
  1. Search for any remaining uses of `shared_fallback` as allocator argument
  2. Update to appropriate new argument (shared_scratch or parent buffer)
  3. Verify no calls have old three-parameter names
  
  If found, update pattern:
  ```python
  # OLD:
  buffer = alloc_X(shared_parent, persistent_parent, shared_fallback)
  
  # NEW (for non-aliased):
  buffer = alloc_X(shared, persistent, shared)
  
  # NEW (for aliased with parent_buffer):
  buffer = alloc_X(shared, persistent, parent_buffer)
  ```
- **Edge cases**: May find calls in test files (update those too)
- **Integration**: Complete migration across entire codebase

**Outcomes**:
```
After TG6 completion:
- All ode_loop.py allocator calls use correct third argument
- shared_fallback renamed/removed in favor of shared_scratch
- All algorithm step functions use correct third argument
- No remaining uses of shared_fallback parameter name
- All calls follow pattern: (shared, persistent, aliased_parent_or_shared)
```

---

## Testing Strategy

### Unit Tests
Test files to update/verify:
- `tests/test_buffer_registry.py` (if exists)

Required test scenarios:
1. **Aliasing with shared parent, sufficient space**:
   - Register parent (shared, size=100)
   - Register child (shared, size=20, aliases=parent)
   - Verify: child slice is WITHIN parent (overlaps parent memory)
   
2. **Aliasing with shared parent, insufficient space**:
   - Register parent (shared, size=100)
   - Register child1 (shared, size=100, aliases=parent)
   - Register child2 (shared, size=20, aliases=parent)
   - Verify: child1 slices parent, child2 gets fresh allocation
   
3. **Aliasing with local parent, shared child**:
   - Register parent (local)
   - Register child (shared, aliases=parent)
   - Verify: child gets fresh shared allocation (cannot alias local parent)
   
4. **Persistent aliasing**:
   - Register parent (local, persistent=True, size=100)
   - Register child (local, persistent=True, size=20, aliases=parent)
   - Verify: child slices parent persistent space
   
5. **Non-aliased shared buffer**:
   - Register buffer (shared, no aliases)
   - Verify: buffer gets fresh shared allocation

### Integration Tests
Run existing integration tests to verify no regressions:
- Algorithm tests with various buffer location configurations
- Solver tests with adaptive/fixed controllers
- Tests with different precision types

### Manual Verification
1. Inspect generated allocator device function signatures
2. Verify CUDA compilation succeeds
3. Check shared memory consumption matches expectations
4. Run performance benchmarks (aliasing should reduce memory usage)

---

## Summary

**Total Task Groups**: 6

**Dependency Chain**:
```
TG1 (Rename Parameters) 
  → TG2 (Fix build_shared_layout)
    → TG3 (Remove Fallback Properties)
      → TG4 (Fix get_allocator)
        → TG6 (Update Call Sites)

TG5 (Remove Manual Aliasing) - can run in parallel or after TG1-4
```

**Parallel Execution Opportunities**:
- TG5 can be done independently and merged after TG1-4 complete
- TG1, TG2, TG3, TG4 must be sequential

**Estimated Complexity**: Medium-High
- 6 major components requiring changes
- ~50+ allocator call sites to update
- Critical semantic changes to aliasing logic
- Requires careful testing to verify correctness

**Key Risks**:
1. Missing allocator call sites (mitigated by TG6.4 codebase search)
2. Incorrect parent buffer passed (mitigated by tracing register() calls)
3. Test failures due to changed semantics (expected, tests validate correct overlap)

**Success Criteria**:
1. ✅ All parameters named (shared, persistent, aliased_parent)
2. ✅ Single shared_layout property (no primary/fallback split)
3. ✅ Aliased buffers slice parent when parent is shared and has space
4. ✅ Aliased buffers get fresh allocation when parent local or full
5. ✅ No manual aliasing logic in algorithm step functions
6. ✅ All allocator calls updated with correct third argument
7. ✅ All tests pass
8. ✅ No CUDA compilation errors
