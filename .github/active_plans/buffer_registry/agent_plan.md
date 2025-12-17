# Buffer Registry Agent Plan

## Purpose

This document provides detailed technical specifications for implementing the BufferRegistry refactoring. It describes component behavior, architectural changes, integration points, and edge cases for the detailed_implementer and reviewer agents.

---

## Component Descriptions

### BufferRegistry (New)

**Location**: `src/cubie/BufferRegistry.py`

**Purpose**: Centralized package-wide singleton managing buffer registration, layout calculation, and allocator generation for all CUDAFactory subclasses.

**Key Responsibilities**:
1. Accept buffer registrations with name, size, and location
2. Track buffer ownership by factory instance
3. Calculate slice indices/offsets on demand (lazy, cached)
4. Provide allocator closures to factories
5. Integrate with update() pattern for location/size changes
6. Aggregate child buffer sizes for parent factories

**Public API**:
- `register(factory, name, size, location)` → buffer_id
- `unregister(factory, name)` → None
- `get_allocator(factory, name)` → Callable
- `get_aggregate_size(factory, location_type)` → int
- `get_child_aggregate_size(child_factory, location_type)` → int
- `update(factory, updates_dict)` → set[str] (recognized keys)
- `invalidate_layout(factory)` → None
- `build_layout(factory)` → None

**Instance Access**: 
```python
from cubie.BufferRegistry import buffer_registry
```

### BufferEntry (New Internal Class)

**Purpose**: Internal data structure representing a single registered buffer.

**Attributes**:
- `name: str` - Unique identifier within factory scope
- `size: int` - Number of elements (can be 0)
- `location: str` - One of "shared", "local", "persistent_local"
- `factory_id: int` - Owner factory's id()
- `offset: Optional[int]` - Computed slice start (None until build)
- `slice: Optional[slice]` - Computed slice object (None until build)

**Notes**:
- Immutable after creation (use unregister/register for changes)
- Offset/slice computed during build_layout(), not registration

### FactoryBufferContext (New Internal Class)

**Purpose**: Per-factory container holding registered buffers and layout state.

**Attributes**:
- `factory_id: int` - id() of owning factory
- `buffers: dict[str, BufferEntry]` - Registered buffers by name
- `layout_version: int` - Incremented on any change
- `cached_layout_version: int` - Version when layout was last computed
- `shared_indices: dict[str, slice]` - Computed shared memory layout
- `local_sizes: dict[str, int]` - Computed local array sizes
- `persistent_local_indices: dict[str, slice]` - Computed persistent layout

**Notes**:
- Layout recalculated when `layout_version != cached_layout_version`
- Parent factories query child contexts for aggregate sizes

---

## Architectural Changes

### CUDAFactory Base Class Modifications

Add the following to `CUDAFactory`:

**New Instance Variables**:
- `_buffer_context_id: Optional[int]` - ID linking to registry context

**New Methods**:
```python
def register_buffer(self, name: str, size: int, location: str) -> None:
    """Register a buffer with the central registry."""
    
def get_buffer_allocator(self, name: str) -> Callable:
    """Get allocator closure for named buffer."""
    
def get_aggregate_buffer_size(self, location: str) -> int:
    """Get total elements for location type."""

def get_child_buffer_size(self, child_factory: 'CUDAFactory', 
                          location: str) -> int:
    """Get child's aggregate elements for location type."""
```

**Modified Methods**:
- `update_compile_settings()` - Route buffer keys to registry
- `_invalidate_cache()` - Also invalidate buffer layout

### Buffer Location Parameter Recognition

The registry recognizes keys matching the pattern:
- `*_location` (e.g., `state_buffer_location`, `delta_location`)
- `*_size` for dynamic sizing (e.g., `scratch_size`)

When `update()` receives these keys:
1. Registry extracts buffer name from key prefix
2. Validates new location/size value
3. Updates BufferEntry if changed
4. Increments layout_version
5. Returns key as recognized

### Parent-Child Buffer Communication

**Information Flow**:
```
Child Factory                Registry                    Parent Factory
     |                          |                              |
     |-- register(buffers) ---->|                              |
     |                          |<-- get_child_aggregate() ----|
     |                          |---- returns total size ----->|
     |                          |<-- register(child_shared) ---|
     |                          |                              |
     |                          |<---- build_layout() ---------|
     |                          |                              |
     |<---- get_allocator() ----|                              |
     |                          |                              |
```

**Parent's Responsibilities**:
1. Query child's aggregate shared/persistent_local sizes
2. Register `child_shared` buffer sized to child's shared total
3. Register `child_persistent_local` buffer sized to child's persistent total
4. Pass unsliced arrays to child at call time

**Child's Responsibilities**:
1. Register all its buffers normally
2. Request allocators at device function compile time
3. Slice from parent-provided arrays using allocator offsets

### Default Child Buffers

When a factory has no children or child aggregate is 0:
- Register `child_shared` with size=0
- Register `child_persistent_local` with size=0

This ensures consistent API without special-casing childless factories.

---

## Integration Points

### Integration with update_compile_settings

```python
# In CUDAFactory.update_compile_settings:
def update_compile_settings(self, updates_dict=None, silent=False, **kwargs):
    # ... existing code ...
    
    # Route buffer-related keys to registry
    buffer_keys = buffer_registry.update(self, updates_dict)
    recognized_params.extend(buffer_keys)
    
    # ... rest of existing code ...
```

### Integration with build()

Layout calculation happens during the build phase:

```python
# In factory's build() method:
def build(self):
    # Ensure buffer layout is current before compiling
    buffer_registry.build_layout(self)
    
    # Now get allocators with correct offsets
    state_alloc = buffer_registry.get_allocator(self, "state")
    # ... use allocators in device function closures ...
```

### Integration with Device Function Compilation

Allocators are closures capturing computed offsets:

```python
# Generated allocator for shared buffer:
def state_allocator(shared_scratch):
    return shared_scratch[offset:offset+size]

# Generated allocator for local buffer:
def state_allocator():
    return cuda.local.array(size, dtype)

# Generated allocator for persistent_local buffer:
def state_allocator(persistent_local):
    return persistent_local[offset:offset+size]
```

---

## Data Structures

### Registry Storage

```python
@attrs.define
class BufferRegistry:
    _contexts: dict[int, FactoryBufferContext] = attrs.field(factory=dict)
    _update_keys: set[str] = attrs.field(factory=set)  # Recognized patterns
```

### Buffer Entry

```python
@attrs.define
class BufferEntry:
    name: str
    size: int
    location: str  # "shared" | "local" | "persistent_local"
    factory_id: int
    offset: Optional[int] = None
    computed_slice: Optional[slice] = None
```

### Factory Context

```python
@attrs.define
class FactoryBufferContext:
    factory_id: int
    buffers: dict[str, BufferEntry] = attrs.field(factory=dict)
    layout_version: int = 0
    cached_layout_version: int = -1
    # Computed during build_layout():
    shared_layout: dict[str, slice] = attrs.field(factory=dict)
    persistent_local_layout: dict[str, slice] = attrs.field(factory=dict)
    local_sizes: dict[str, int] = attrs.field(factory=dict)
```

---

## Expected Interactions

### Typical Factory Lifecycle

1. **Construction**:
   - Factory creates context in registry
   - Registers initial buffers with default locations

2. **Configuration**:
   - User calls `update(state_buffer_location="shared")`
   - Registry receives update, modifies entry, bumps version

3. **Build**:
   - Factory's `build()` triggers `registry.build_layout()`
   - Registry computes offsets for all buffers
   - Factory retrieves allocators for device function compilation

4. **Execution**:
   - Compiled device function uses captured offsets
   - Memory slicing happens at runtime using compile-time constants

### Parent-Child Coordination

1. **Child Construction**:
   - Child registers its buffers with registry

2. **Parent Construction**:
   - Parent queries `get_child_aggregate_size(child, "shared")`
   - Parent registers `child_shared` with returned size
   - Parent queries `get_child_aggregate_size(child, "persistent_local")`
   - Parent registers `child_persistent_local` with returned size

3. **Parent Build**:
   - Parent's layout includes child regions
   - Parent compiles device function passing child arrays

4. **Nested Build**:
   - Child's build uses registry for its own layout
   - Offsets relative to child's view of parent's arrays

---

## Edge Cases

### Edge Case: Zero-Size Buffers

**Scenario**: Buffer registered with size=0 (e.g., no observables)

**Handling**:
- Registry stores entry with size=0
- Layout calculation assigns empty slice `slice(ptr, ptr)`
- Allocator returns zero-length slice/array
- Device code uses `max(1, size)` for `cuda.local.array` (Numba requirement)

### Edge Case: All Buffers in Same Location

**Scenario**: All buffers configured as "local"

**Handling**:
- `get_aggregate_size("shared")` returns 0
- Parent still registers `child_shared` with size=0
- Device function receives empty shared array, ignored

### Edge Case: Location Change After Build

**Scenario**: `update(state_buffer_location="shared")` after device function compiled

**Handling**:
- Registry updates entry, bumps version
- Cache invalidation propagates to owning factory
- Next `build()` recalculates layout with new location

### Edge Case: Unregistered Buffer Access

**Scenario**: `get_allocator("nonexistent_buffer")`

**Handling**:
- Registry raises `KeyError` with descriptive message
- Factory code must register before requesting allocator

### Edge Case: Persistent Local Without Parent Support

**Scenario**: Factory registers `persistent_local` but parent doesn't provide array

**Handling**:
- This is a configuration error
- Registry documents requirement in docstrings
- Runtime: IndexError on device function (clear failure mode)

### Edge Case: Re-registration

**Scenario**: Factory calls `register("state", ...)` twice

**Handling**:
- Second registration overwrites first
- Version incremented, layout invalidated
- Alternative: Require explicit `unregister()` first (stricter)

### Edge Case: Child Factory Destroyed

**Scenario**: Child factory garbage collected before parent builds

**Handling**:
- Registry uses weak references for factory tracking (optional)
- Or: Registry requires explicit cleanup via `unregister_all(factory)`
- Parent's query for child aggregate returns 0 if child context gone

---

## Dependencies and Imports

### New Module Dependencies

`src/cubie/BufferRegistry.py`:
```python
from typing import Callable, Dict, Optional, Set
import attrs
from cubie.cuda_simsafe import from_dtype as simsafe_dtype
```

### Modified Module Imports

Each factory using registry adds:
```python
from cubie.BufferRegistry import buffer_registry
```

---

## Migration Strategy

### Phase 1: Registry Infrastructure
- Implement BufferRegistry singleton
- Add to CUDAFactory base class
- No behavioral changes yet

### Phase 2: Parallel Operation
- Factories register with both old and new systems
- Verify new system produces same layouts

### Phase 3: Gradual Migration
- Convert one factory at a time to registry-only
- Start with leaf factories (LinearSolver, Newton)
- Work up to IVPLoop, SingleIntegratorRun

### Phase 4: Cleanup
- Remove old BufferSettings pattern
- Delete `_layout_generated` flags
- Simplify factory code

---

## Validation Approach

### Unit Tests
- Registry registration/unregistration
- Layout calculation for various configurations
- Update propagation
- Parent-child aggregate queries

### Integration Tests
- Full solver run with registry-based allocation
- Buffer location changes via update()
- Memory size verification (shared, local, persistent_local)

### Regression Tests
- Compare output with old BufferSettings implementation
- Verify memory layouts match for equivalent configurations
