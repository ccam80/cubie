# Buffer Registry Refactor: Agent Implementation Plan

This document provides the technical specification for the detailed_implementer and reviewer agents.

---

## Component Descriptions

### 1. BufferRegistry Singleton

**Purpose:** Central registry managing all buffer metadata for CUDA factories.

**Location:** `src/cubie/buffer_registry.py`

**Behavior:**
- Package-level singleton instantiated at module load
- Maintains a dictionary mapping factory instances to their buffer contexts
- Provides lazy-cached slice/layout computation
- Generates CUDA-compatible allocator device functions

**Key Data Structures:**

```python
BufferEntry:
    name: str                          # Unique buffer name within factory
    factory: CUDAFactory               # Owning factory instance
    size: int                          # Buffer size in elements
    location: Literal['shared', 'local']  # Memory location
    persistent: bool = False           # If True and local, use persistent_local
    aliases: Optional[str] = None      # Name of buffer to alias
    precision: type                    # numpy precision type
    
BufferContext:
    factory: CUDAFactory               # Factory instance
    entries: Dict[str, BufferEntry]    # Registered buffers by name
    _shared_layout: Optional[Dict]     # Cached shared memory slices
    _local_layout: Optional[Dict]      # Cached local memory sizes
    _persistent_layout: Optional[Dict] # Cached persistent_local slices
    _alias_offsets: Dict[str, int]     # Tracks aliased regions per buffer
```

**Interactions:**
- Factories call `register()` during initialization
- Factories call `get_allocator()` in their `build()` methods
- Parent factories query child `shared_buffer_size` properties before registering their own buffers
- Lazy rebuild when accessing layout properties after changes

---

### 2. BufferEntry

**Purpose:** Immutable record describing a single buffer's requirements.

**Behavior:**
- Stores buffer metadata for registration
- Validates location is only 'shared' or 'local'
- Validates aliases reference existing buffers when resolved
- Computes size properties respecting aliasing

**Validation Rules:**
- `location` must be 'shared' or 'local'
- `size` must be >= 0
- `persistent` only meaningful when `location == 'local'`
- `aliases` must reference a registered buffer in the same factory context

---

### 3. Registry Registration Method

**Purpose:** Register a buffer with the central registry.

**Signature:**
```python
def register(
    name: str,
    factory: CUDAFactory,
    size: int,
    location: Literal['shared', 'local'],
    persistent: bool = False,
    aliases: Optional[str] = None,
    precision: type = np.float32,
) -> None
```

**Behavior:**
1. Get or create BufferContext for factory
2. Create BufferEntry with provided parameters
3. Add entry to context's entries dictionary
4. Invalidate cached layouts (set to None)
5. If aliases specified, validate target exists in same context

**Error Handling:**
- Raise ValueError if buffer name already registered for this factory
- Raise ValueError if aliases references non-existent buffer
- Silently ignore updates if factory has no registered context

---

### 4. Allocator Factory Method

**Purpose:** Generate CUDA device functions for buffer allocation.

**Signature:**
```python
def get_allocator(
    name: str,
    factory: CUDAFactory,
) -> Callable
```

**Behavior:**
1. Retrieve BufferContext for factory (error if not found)
2. Retrieve BufferEntry for name (error if not found)
3. Compute slice/size based on entry's location and aliasing
4. Generate and return CUDA device function

**Generated Function Pattern:**
```python
@cuda.jit(device=True, inline=True, ForceInline=True, **compile_kwargs)
def allocate_buffer(shared_parent, persistent_parent):
    if _shared:
        array = shared_parent[shared_slice]
    elif _persistent:
        array = persistent_parent[persistent_local_slice]
    else:
        array = cuda.local.array(local_size, precision)
    return array
```

**Critical:** Use compile-time constants for branching flags, captured in closure.

---

### 5. Size Properties

**Purpose:** Report memory requirements for each memory type.

**Properties per factory context:**

```python
@property
def shared_buffer_size(factory) -> int:
    """Sum of all non-aliased shared buffers."""
    
@property  
def local_buffer_size(factory) -> int:
    """Sum of all non-aliased, non-persistent local buffers.
    Returns max(size, 1) for each to satisfy cuda.local.array."""
    
@property
def persistent_local_buffer_size(factory) -> int:
    """Sum of all non-aliased persistent local buffers."""
```

**Behavior:**
- Exclude aliased buffers from size calculation (they share parent space)
- For local_buffer_size, return max(size, 1) for each buffer
- For others, return 0 if no applicable buffers

---

### 6. Layout Computation

**Purpose:** Compute slice indices for shared and persistent memory.

**Lazy Cached Pattern:**
1. On any buffer registration or update, set layouts to None
2. On size property access, check if layout is None
3. If None, rebuild layout by iterating entries
4. Cache and return computed layout

**Shared Layout Algorithm:**
```python
def _build_shared_layout(context) -> Dict[str, slice]:
    offset = 0
    layout = {}
    for name, entry in sorted(context.entries.items()):
        if entry.location != 'shared':
            continue
        if entry.aliases is not None:
            # Alias: compute slice within parent
            parent = context.entries[entry.aliases]
            parent_offset = layout[entry.aliases].start
            alias_offset = context._alias_offsets.get(entry.aliases, 0)
            layout[name] = slice(parent_offset + alias_offset, 
                                 parent_offset + alias_offset + entry.size)
            context._alias_offsets[entry.aliases] = alias_offset + entry.size
        else:
            layout[name] = slice(offset, offset + entry.size)
            offset += entry.size
    return layout
```

**Persistent Layout Algorithm:**
Similar to shared, but for persistent local buffers.

---

### 7. Aliasing System

**Purpose:** Allow buffers to share space with other buffers.

**Behavior:**
1. When buffer B aliases buffer A:
   - B does not contribute to total size calculation
   - B gets a slice starting where A's previous aliases ended
   - Track `_alias_offsets[A]` to know next available position
2. Check location compatibility:
   - If A is shared, B must also be shared
   - If A is local, B goes to local or persistent_local based on B's persistent flag
3. Error if alias would exceed parent's size

**DIRK Example:**
```python
# solver_scratch: 100 elements shared
registry.register('solver_scratch', self, 100, 'shared')

# stage_rhs: n elements, aliases first n of solver_scratch
registry.register('stage_rhs', self, n, 'shared', aliases='solver_scratch')

# increment_cache: n elements, aliases next n of solver_scratch
registry.register('increment_cache', self, n, 'shared', aliases='solver_scratch')
```

Result:
- solver_scratch: slice(0, 100)
- stage_rhs: slice(0, n)
- increment_cache: slice(n, 2*n)

---

### 8. Factory Integration Pattern

**Purpose:** Define how CUDAFactories interact with the registry.

**Registration Phase (in `__init__`):**
```python
def __init__(self, ...):
    super().__init__()
    
    # Register own buffers
    buffer_registry.register('state', self, n, 'local')
    buffer_registry.register('scratch', self, 100, 'shared')
    
    # For parent factories with children:
    child_shared = child_factory.shared_buffer_size
    buffer_registry.register('child_shared', self, child_shared, 'shared')
```

**Build Phase (in `build()`):**
```python
def build(self):
    # Get allocators for each buffer
    alloc_state = buffer_registry.get_allocator('state', self)
    alloc_scratch = buffer_registry.get_allocator('scratch', self)
    
    @cuda.jit(device=True, inline=True)
    def step(..., shared, persistent_local):
        state = alloc_state(shared, persistent_local)
        scratch = alloc_scratch(shared, persistent_local)
        # ... use buffers
```

---

### 9. Migration from BufferSettings

**Current Pattern (to remove):**
```python
class DIRKBufferSettings(BufferSettings):
    n: int
    stage_count: int
    stage_increment_location: str
    # ... many location attributes
    
    @property
    def shared_memory_elements(self) -> int:
        # Complex calculation
        
    @property
    def shared_indices(self) -> DIRKSliceIndices:
        # Complex slice computation
```

**New Pattern:**
```python
class DIRKStep(ODEImplicitStep):
    def __init__(self, ...):
        # Register all buffers with central registry
        buffer_registry.register('stage_increment', self, n, 
                                 stage_increment_location)
        buffer_registry.register('stage_base', self, n,
                                 stage_base_location)
        # ... etc
    
    @property
    def shared_memory_required(self) -> int:
        return buffer_registry.shared_buffer_size(self)
```

---

### 10. Error Handling

**Registration Errors:**
- Duplicate buffer name for same factory: ValueError
- Invalid location value: ValueError
- Invalid size (< 0): ValueError

**Retrieval Errors:**
- Unregistered factory: KeyError with helpful message
- Unregistered buffer name: KeyError with helpful message
- Alias exceeds parent size: RuntimeError

**Silent Handling:**
- Update requests from unregistered factories: silently ignore
- Empty factory context on deregistration: silently clean up

---

## Architectural Changes Required

### BufferSettings.py Removal
- Delete `src/cubie/BufferSettings.py` entirely
- Remove imports from all files that use it
- No deprecation warnings - complete removal

### CUDAFactory Base Class
- No changes required to base class
- Registry is independent subsystem

### Algorithm Files
Each algorithm file needs:
1. Remove BufferSettings class definition
2. Remove SliceIndices class definition
3. Remove LocalSizes class definition
4. Add buffer registrations in `__init__`
5. Update `build()` to use allocators
6. Update size properties to delegate to registry

### Loop Files
- Remove LoopBufferSettings
- Remove LoopSliceIndices
- Remove LoopLocalSizes
- Register all loop buffers with registry
- Update build to use allocators

### Matrix-Free Solvers
- Remove LinearSolverBufferSettings
- Remove NewtonBufferSettings
- Register solver buffers
- Update build methods

---

## Integration Points with Current Codebase

### SummaryMetrics Pattern Reference
Located at: `src/cubie/outputhandling/summarymetrics/metrics.py`

Key patterns to follow:
- Global singleton instance at module level
- Registration via method call (not decorator in this case)
- Factory pattern for generating functions

### MemoryManager Pattern Reference
Located at: `src/cubie/memory/mem_manager.py`

Key patterns to follow:
- Registry dictionary with object keys
- Settings classes per registered object
- Property-based configuration access

### CUDAFactory Pattern
Located at: `src/cubie/CUDAFactory.py`

Key patterns to follow:
- Lazy cached builds via `_build()` pattern
- Cache invalidation via `_invalidate_cache()`
- Property access triggers build if needed

---

## Data Structures and Their Purposes

### BufferEntry
- Immutable record of buffer requirements
- Used for registration and layout computation
- Contains all metadata needed for allocator generation

### BufferContext
- Groups all entries for a single factory
- Caches computed layouts
- Tracks alias offsets for multi-aliased buffers

### BufferRegistry (singleton)
- Maps factory instances to contexts
- Provides registration and query API
- Generates allocator device functions

---

## Dependencies and Imports Required

### New File (buffer_registry.py)
```python
from typing import Dict, Optional, Literal, Callable, Any
import attrs
import numpy as np
from numba import cuda

from cubie.cuda_simsafe import compile_kwargs
from cubie._utils import PrecisionDType
```

### Updated Files
All migrated files need:
```python
from cubie.buffer_registry import buffer_registry
```

Remove:
```python
from cubie.BufferSettings import BufferSettings, LocalSizes, SliceIndices
```

---

## Edge Cases to Consider

### 1. Empty Factory Context
- Factory registered but no buffers added
- All size properties return 0
- get_allocator raises KeyError

### 2. Self-Aliasing
- Buffer cannot alias itself
- Validate during registration

### 3. Circular Aliasing
- A aliases B, B aliases A
- Detect and error during layout computation

### 4. Alias Chain
- A is base, B aliases A, C aliases B
- Not supported - aliases must reference non-aliased buffers

### 5. Zero-Size Buffers
- Valid for shared and persistent_local
- local_buffer_size returns max(size, 1) for cuda.local.array

### 6. Factory Without Precision
- Allocator needs precision for cuda.local.array
- Must be provided at registration time

### 7. Multiple Factories Same Class
- Each instance is a unique key
- Independent contexts, independent layouts

### 8. Factory Garbage Collection
- When factory is deleted, context should be cleaned up
- Use weakref for factory keys if memory is a concern

---

## Expected Interactions Between Components

1. **Factory → Registry (registration):**
   Factory calls `register()` during `__init__`, providing buffer specs.

2. **Factory → Registry (query):**
   Factory calls `shared_buffer_size`, `local_buffer_size`, etc. to determine memory needs.

3. **Factory → Registry (allocation):**
   Factory calls `get_allocator()` in `build()` to get device functions.

4. **Parent → Child (size query):**
   Parent factory queries child's size properties, then registers its own buffer for child.

5. **Registry → Cache (invalidation):**
   Any registration sets cached layouts to None.

6. **Registry → Cache (rebuild):**
   Size property access triggers layout rebuild if None.

7. **Allocator → CUDA (runtime):**
   Generated device function called during kernel execution with shared/persistent arrays.

---

## Test Considerations

### Unit Tests for Registry
- Test registration creates context
- Test duplicate registration raises error
- Test aliasing computes correct slices
- Test size properties calculate correctly
- Test lazy rebuild pattern

### Integration Tests
- Test DIRK with aliased buffers
- Test loop with all buffer types
- Test parent-child buffer passing
- Test allocator generates valid device functions

### Migration Tests
- Ensure existing tests pass after migration
- No behavior changes expected - pure refactoring

---

## Notes for Reviewer Agent

The implementation should be validated against these user stories:

1. **US-1 (Centralized):** Verify single registry instance, factory instance keys, no merge functions
2. **US-2 (Aliasing):** Verify alias system works for DIRK patterns
3. **US-3 (Simplified):** Verify only two location values, persistent flag works
4. **US-4 (Lazy):** Verify no version tracking, nullable layouts
5. **US-5 (Allocators):** Verify generated functions match required pattern
6. **US-6 (Complete):** Verify all old code removed, no deprecation warnings
