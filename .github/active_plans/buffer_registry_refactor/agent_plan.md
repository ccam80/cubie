# Buffer Registry Refactor - Agent Plan

## Overview

This document provides detailed architectural specifications for implementing a `CUDABuffer` and `BufferRegistry` system to replace the scattered `BufferSettings` hierarchy in CuBIE.

---

## Component Descriptions

### 1. BufferLocation Enum

An enumeration representing the three valid memory locations for CUDA buffers.

**Module**: `src/cubie/buffer_registry.py` (new file)

**Behavior**:
- `SHARED`: Buffer resides in CUDA shared memory (per-block, fast, limited ~48KB)
- `LOCAL`: Buffer resides in thread-local memory (per-thread registers/local)
- `LOCAL_PERSISTENT`: Buffer in local memory that persists across function calls within a step

**Relationships**:
- Used by `CUDABuffer` to determine allocation strategy
- Replaces string literals `'local'`, `'shared'` in current BufferSettings

---

### 2. CUDABuffer Class

Represents a single buffer with its memory location, size, and slice information.

**Module**: `src/cubie/buffer_registry.py`

**Attributes**:
- `name: str` - Unique identifier for the buffer within a registry
- `location: BufferLocation` - Where the buffer should be allocated
- `size: int | Callable[[], int]` - Size in elements; callable for deferred evaluation
- `precision: PrecisionDType` - numpy dtype for the buffer elements
- `enabled: bool` - Whether buffer is active (for conditional allocation)
- `shared_slice: slice` - Assigned by registry for shared memory layout
- `local_slice: slice` - Assigned by registry for persistent local layout
- `_parent: Optional[CUDAFactory]` - Reference to owning factory (weak ref optional)

**Properties**:
- `shared_size: int` - Returns size if location is SHARED and enabled, else 0
- `local_size: int` - Returns max(size, 1) if location is LOCAL and enabled, else 0
- `persistent_size: int` - Returns size if location is LOCAL_PERSISTENT and enabled, else 0
- `resolved_size: int` - Evaluates size callable or returns int directly

**Methods**:
- `allocate()` - Returns a compiled CUDA device function that performs allocation

**Behavior**:
- Size callable is evaluated lazily when `resolved_size` is accessed
- `allocate()` captures location and slice as compile-time constants
- Disabled buffers return zero for all size properties
- `nonzero_size` property returns `max(resolved_size, 1)` for cuda.local.array compatibility

**Expected Interactions**:
- Created by CUDAFactory subclasses during `__init__` or buffer registration
- Registered with `BufferRegistry` for aggregate calculations
- Slice assigned by registry's `generate_shared_layout()` method
- `allocate()` called during factory's `build()` to get device functions

---

### 3. BufferRegistry Class

Aggregates multiple CUDABuffer instances and manages collective memory layouts.

**Module**: `src/cubie/buffer_registry.py`

**Attributes**:
- `buffers: Dict[str, CUDABuffer]` - Registry of buffers by name
- `precision: PrecisionDType` - Default precision for new buffers
- `_shared_layout_generated: bool` - Flag to track if layout is finalized

**Properties**:
- `shared_memory_elements: int` - Sum of all buffers' `shared_size`
- `local_memory_elements: int` - Sum of all buffers' `local_size`
- `persistent_local_elements: int` - Sum of all buffers' `persistent_size`
- `total_elements: int` - Sum of all memory requirements

**Methods**:
- `register(buffer: CUDABuffer) -> None` - Add buffer to registry
- `register_buffer(name, location, size, enabled=True) -> CUDABuffer` - Convenience method
- `get(name: str) -> CUDABuffer` - Retrieve buffer by name
- `generate_shared_layout() -> int` - Assign slices, return total shared size
- `generate_persistent_layout() -> int` - Assign persistent local slices
- `merge(child_registry: BufferRegistry) -> None` - Incorporate child's buffers
- `get_allocators() -> Dict[str, Callable]` - Return dict of allocation functions

**Behavior**:
- Names must be unique within a registry
- `merge()` prefixes child buffer names to avoid collisions (e.g., "newton.delta")
- Layout generation is idempotent - can be called multiple times safely
- After merge, re-generate layouts to get contiguous slices

**Expected Interactions**:
- Owned by CUDAFactory subclasses via composition
- Child factories pass their registry to parent's `merge()` call
- Factory's `build()` calls `generate_shared_layout()` before compilation
- Allocators retrieved during device function construction

---

### 4. CUDAFactory Integration

Modifications to the existing CUDAFactory base class to support buffer registration.

**Module**: `src/cubie/CUDAFactory.py` (modify existing)

**New Attributes**:
- `_buffer_registry: Optional[BufferRegistry]` - Lazy-initialized registry

**New Properties**:
- `buffer_registry: BufferRegistry` - Get or create registry

**New Methods**:
- `register_buffer(name, location, size, enabled=True) -> CUDABuffer` - Delegate to registry
- `merge_child_buffers(child: CUDAFactory) -> None` - Merge child's registry

**Behavior**:
- Registry is created on first access if not exists
- Precision inherited from compile_settings if available
- Methods are optional - factories not using buffers ignore them
- Backward compatible - existing factories unchanged

---

## Architectural Changes Required

### New Module

Create `src/cubie/buffer_registry.py` containing:
- `BufferLocation` enum
- `CUDABuffer` attrs class
- `BufferRegistry` attrs class

### Modified Modules

1. **CUDAFactory.py**: Add buffer_registry property and helper methods

2. **BufferSettings.py**: Eventually deprecated, but initially kept for backward compatibility

3. **Algorithm modules** (one at a time, in migration phase):
   - `linear_solver.py` - Remove LinearSolverBufferSettings
   - `newton_krylov.py` - Remove NewtonBufferSettings
   - `generic_erk.py` - Remove ERKBufferSettings
   - `generic_dirk.py` - Remove DIRKBufferSettings
   - `generic_firk.py` - Remove FIRKBufferSettings
   - `generic_rosenbrock_w.py` - Remove RosenbrockBufferSettings
   - `ode_loop.py` - Remove LoopBufferSettings

4. **Test files**:
   - Create `tests/test_buffer_registry.py`
   - Update `tests/test_buffer_settings.py` 
   - Update `tests/integrators/algorithms/test_buffer_settings.py`

---

## Integration Points with Current Codebase

### Pattern: Compile-time Constant Capture

The current ode_loop.py (lines 834-865) demonstrates the pattern:
```python
# Unpack boolean flags as compile-time constants
state_shared = buffer_settings.use_shared_state
state_proposal_shared = buffer_settings.use_shared_state_proposal
...
# Then in device function, these are compile-time known
if state_shared:
    state_buffer = shared_scratch[state_shared_ind]
else:
    state_buffer = cuda.local.array(state_local_size, precision)
```

The new `CUDABuffer.allocate()` encapsulates this pattern:
```python
# In CUDABuffer.allocate()
_is_shared = self.location == BufferLocation.SHARED
_shared_slice = self.shared_slice
_local_size = self.nonzero_size
_precision = self.precision

@cuda.jit(device=True, inline=True, ForceInline=True)
def allocate_buffer(shared_parent, persistent_parent):
    if _is_shared:
        return shared_parent[_shared_slice]
    else:
        return cuda.local.array(_local_size, _precision)
    
return allocate_buffer
```

### Pattern: Nested Buffer Composition

Current DIRK (generic_dirk.py, lines 521-541) composes Newton settings:
```python
linear_buffer_settings = LinearSolverBufferSettings(n=n)
newton_buffer_settings = NewtonBufferSettings(
    n=n,
    linear_solver_buffer_settings=linear_buffer_settings,
)
buffer_kwargs = {
    'n': n,
    'stage_count': tableau.stage_count,
    'newton_buffer_settings': newton_buffer_settings,
}
```

New pattern with BufferRegistry:
```python
# In DIRKStep.__init__
self.linear_solver = linear_solver_factory(...)  # Has its own registry
self.newton_solver = newton_krylov_solver_factory(...)  # Has its own registry

# Merge nested registries
self.buffer_registry.merge(self.linear_solver.buffer_registry, prefix="linear")
self.buffer_registry.merge(self.newton_solver.buffer_registry, prefix="newton")

# Now self.buffer_registry has all buffers with prefixed names
```

### Pattern: Solver Scratch Pass-through

Current pattern (generic_dirk.py, line 825):
```python
solver_scratch = shared[solver_scratch_slice]
```

The new system preserves this by having the parent factory know about child scratch requirements via merged registry.

---

## Data Structures and Their Purposes

### BufferLocation Enum

```python
class BufferLocation(Enum):
    SHARED = "shared"           # Per-block shared memory
    LOCAL = "local"             # Per-thread local arrays
    LOCAL_PERSISTENT = "local_persistent"  # Persists between step calls
```

Purpose: Type-safe replacement for string literals, enables validation.

### CUDABuffer 

```python
@attrs.define
class CUDABuffer:
    name: str
    location: BufferLocation
    size: Union[int, Callable[[], int]]
    precision: PrecisionDType
    enabled: bool = True
    shared_slice: slice = attrs.field(factory=lambda: slice(0, 0))
    local_slice: slice = attrs.field(factory=lambda: slice(0, 0))
    persistent_slice: slice = attrs.field(factory=lambda: slice(0, 0))
```

Purpose: Single source of truth for buffer configuration, generates allocation code.

### BufferRegistry

```python
@attrs.define
class BufferRegistry:
    buffers: Dict[str, CUDABuffer] = attrs.field(factory=dict)
    precision: PrecisionDType = np.float64
```

Purpose: Collect buffers, compute aggregates, generate layouts, merge nested registries.

---

## Dependencies and Imports Required

### buffer_registry.py

```python
from enum import Enum
from typing import Callable, Dict, Optional, Union

import attrs
from attrs import validators
import numpy as np
from numba import cuda

from cubie._utils import PrecisionDType
from cubie.cuda_simsafe import compile_kwargs
```

### CUDAFactory.py additions

```python
from cubie.buffer_registry import BufferRegistry, CUDABuffer, BufferLocation
```

---

## Edge Cases to Consider

### 1. Zero-size Buffers

- Some buffers have size 0 (e.g., accumulator for single-stage methods)
- `cuda.local.array` requires size >= 1
- Solution: `nonzero_size` property returns `max(resolved_size, 1)`

### 2. Disabled/Conditional Buffers

- DIRK increment_cache only needed for FSAL
- Pattern: `enabled=self.first_same_as_last`
- Disabled buffers contribute 0 to totals, get empty slices

### 3. Size Dependencies on Other Attributes

- Accumulator size depends on `stage_count` and `n`
- Solution: Size as callable that captures `self`
- Example: `size=lambda: max((self.stage_count - 1) * self.n, 0)`

### 4. Name Collisions in Merged Registries

- LinearSolver and Newton both might have "temp" buffer
- Solution: Prefix child buffer names on merge
- Result: "newton.temp", "linear.temp"

### 5. Buffer Aliasing

- ERK stage_cache can alias stage_rhs or accumulator
- Current: Complex boolean logic in ERKBufferSettings
- Solution: Separate buffers with `enabled` conditions, or keep aliasing logic in build()

### 6. Layout Regeneration After Merge

- Parent merges child, then child's slices are stale
- Solution: Call `generate_shared_layout()` after all merges complete
- Implementation: Layout generation is idempotent

### 7. Thread-local vs Persistent Local

- Regular local: fresh each device function call
- Persistent local: survives across calls in integration loop
- Solution: `LOCAL_PERSISTENT` location type, separate slice tracking

### 8. Solver Scratch Handoff

- Newton solver receives scratch from DIRK
- DIRK owns the shared memory slice
- Newton's internal buffers slice from that region
- Solution: Nested registries naturally compose; parent's slice becomes child's base

---

## Behavior Guidelines for Implementers

1. **Attrs patterns**: Use `@attrs.define`, validators from `cubie._utils`

2. **No guarding checks**: Follow CuBIE style - trust inputs, validate at boundaries

3. **Compile-time constants**: All allocation branching captured at JIT time

4. **Descriptive names**: `buffer_registry` not `buf_reg`, `shared_memory_elements` not `shared_size`

5. **PEP8 compliance**: 79-char lines, numpydoc docstrings

6. **Type hints**: On all function signatures, not on local variables

7. **Lazy evaluation**: Size callables evaluated when needed, not at registration

8. **Immutable after layout**: Once slices assigned, treat buffer config as frozen

---

## Success Criteria

1. All existing tests pass after migration
2. New BufferRegistry tests cover:
   - Single buffer registration
   - Multiple buffer aggregation
   - Nested registry merge
   - Slice layout generation
   - Allocator device function compilation
3. Memory totals match between old and new systems
4. Device functions compile to identical PTX (verify with Numba)
5. No runtime performance regression
