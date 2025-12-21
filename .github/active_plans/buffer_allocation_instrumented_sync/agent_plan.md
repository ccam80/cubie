# Buffer Allocation Instrumented Sync - Agent Plan

## Overview

This plan describes the changes needed to replicate buffer allocation refactoring from source algorithm implementations to their instrumented test counterparts. The goal is exact replication of buffer management patterns while preserving logging functionality.

## Key Patterns to Replicate

### Pattern 1: Buffer Registry Import
Source files import from `cubie.buffer_registry`:
```python
from cubie.buffer_registry import buffer_registry
```

### Pattern 2: `register_buffers()` Method
Source files have a `register_buffers()` method called at the end of `__init__()`:
```python
def register_buffers(self) -> None:
    """Register buffers with buffer_registry."""
    config = self.compile_settings
    buffer_registry.clear_parent(self)
    # Register buffers with specific locations
    buffer_registry.register('buffer_name', self, size, location, ...)
```

### Pattern 3: Get Allocator Functions
In `build_step()`, source files get allocator functions:
```python
getalloc = buffer_registry.get_allocator
alloc_buffer_name = getalloc('buffer_name', self)
```

### Pattern 4: Child Allocators for Newton Solver
For implicit methods with Newton solvers:
```python
alloc_solver_shared, alloc_solver_persistent = (
    buffer_registry.get_child_allocators(self, self.solver,
                                         name='solver_scratch')
)
```

### Pattern 5: Allocator Calls in step()
Replace manual slicing with allocator calls:
```python
# OLD: solver_scratch = shared[:solver_shared_elements]
# NEW:
solver_scratch = alloc_solver_shared(shared, persistent_local)
solver_persistent = alloc_solver_persistent(shared, persistent_local)
```

---

## File-by-File Changes

### 1. backwards_euler.py

**Current State:**
- Uses `solver_shared_elements` property for manual slicing
- No `buffer_registry` import
- `build_step()` slices shared memory manually

**Changes Required:**
1. Add import: `from cubie.buffer_registry import buffer_registry`
2. In `build_step()`:
   - Add child allocator acquisition:
     ```python
     alloc_solver_shared, alloc_solver_persistent = (
         buffer_registry.get_child_allocators(self, self.solver,
                                              name='solver_scratch')
     )
     ```
   - Replace `solver_scratch = shared[: solver_shared_elements]` with:
     ```python
     solver_scratch = alloc_solver_shared(shared, persistent_local)
     solver_persistent = alloc_solver_persistent(shared, persistent_local)
     ```
   - Remove `solver_shared_elements` capture
   - Remove `stage_rhs` local array (source doesn't use it)
   - Update Newton solver call to include `solver_persistent` parameter

**Preserve:**
- All logging arrays in step() signature
- All LOGGING: blocks
- Instrumented solver creation in `build_implicit_helpers()`
- Logging array parameters to `solver_fn()` call

---

### 2. crank_nicolson.py

**Current State:**
- Uses `solver_shared_elements` for manual slicing
- No `buffer_registry` import
- No config class with location options
- Uses `ImplicitStepConfig` directly

**Changes Required:**
1. Add import: `from cubie.buffer_registry import buffer_registry`
2. Add `import attrs`
3. Add `CrankNicolsonStepConfig` class extending `ImplicitStepConfig` with `dxdt_location` field
4. Update `__init__()` to:
   - Build `CrankNicolsonStepConfig` with location kwargs
   - Call `self.register_buffers()` at the end
5. Add `register_buffers()` method:
   - Register `cn_dxdt` buffer with aliases
6. In `build_step()`:
   - Get child allocators for Newton solver
   - Get allocator for `cn_dxdt`
   - Replace manual slicing with allocator calls
   - Update solver call to include `solver_persistent`

**Preserve:**
- All logging arrays and LOGGING blocks
- Instrumented solver creation
- Stage-specific dummy arrays for second BE solve

---

### 3. explicit_euler.py

**Current State:**
- Already minimal - no solver buffers needed
- No buffer allocation changes in source

**Changes Required:**
- None - explicit Euler has no solver buffers to manage

**Preserve:**
- All logging arrays and LOGGING blocks

---

### 4. generic_dirk.py

**Current State:**
- Has buffer registration in `__init__()` but with different naming
- Uses allocators but with `dirk_` prefix
- Missing `stage_rhs_location` config field
- Missing `solver` prefix registration

**Changes Required:**
1. Add `stage_rhs_location` field to `DIRKStepConfig`
2. Move buffer registration to dedicated `register_buffers()` method
3. Update buffer names to match source (shorter names):
   - `dirk_stage_increment` → `stage_increment`
   - `dirk_accumulator` → `accumulator`
   - `dirk_stage_base` → `stage_base`
   - etc.
4. Add `stage_rhs` buffer registration
5. Update allocator acquisitions in `build_step()` to match new names
6. Add child allocator registration for solver with name='solver'

**Preserve:**
- All logging arrays and LOGGING blocks
- Instrumented solver creation
- FSAL cache handling

---

### 5. generic_erk.py

**Current State:**
- Has buffer registration in `__init__()` with `erk_` prefix
- Missing `register_buffers()` method call

**Changes Required:**
1. Move buffer registration to `register_buffers()` method
2. Update buffer names to match source:
   - `erk_stage_rhs` → `stage_rhs`
   - `erk_stage_accumulator` → `stage_accumulator`
   - Remove `erk_stage_cache` (handle FSAL differently)
3. Update allocator names in `build_step()`
4. Add `super().__init__()` then `self.register_buffers()` pattern

**Preserve:**
- All logging arrays and LOGGING blocks
- FSAL caching logic

---

### 6. generic_firk.py

**Current State:**
- Has buffer registration in `__init__()` with `firk_` prefix
- Uses allocators but with different pattern

**Changes Required:**
1. Move buffer registration to `register_buffers()` method
2. Update buffer names to match source:
   - `firk_stage_increment` → `stage_increment`
   - `firk_stage_driver_stack` → `stage_driver_stack`
   - `firk_stage_state` → `stage_state`
   - Remove `firk_solver_scratch` registration
3. Add child allocator acquisition pattern
4. Update allocator acquisitions in `build_step()`

**Preserve:**
- All logging arrays and LOGGING blocks
- Instrumented solver creation

---

### 7. generic_rosenbrock_w.py

**Current State:**
- Has buffer registration in `__init__()` with `rosenbrock_` prefix
- Missing `register_buffers()` method

**Changes Required:**
1. Move buffer registration to `register_buffers()` method
2. Update buffer names to match source:
   - `rosenbrock_stage_rhs` → `stage_rhs`
   - `rosenbrock_stage_store` → `stage_store`
   - `rosenbrock_cached_auxiliaries` → `cached_auxiliaries`
   - `rosenbrock_stage_cache` → `stage_increment` (with aliases)
3. Add `super().__init__()` then `self.register_buffers()` pattern
4. Update `build_implicit_helpers()` to update buffer via `buffer_registry.update_buffer()`
5. Update allocator acquisitions in `build_step()`

**Preserve:**
- All logging arrays and LOGGING blocks
- Instrumented solver creation
- `build()` override with custom logic

---

## Integration Points

### Buffer Registry Module
All changes rely on `cubie.buffer_registry` providing:
- `buffer_registry.register(name, parent, size, location, **kwargs)`
- `buffer_registry.get_allocator(name, parent)`
- `buffer_registry.get_child_allocators(parent, child, name=...)`
- `buffer_registry.clear_parent(parent)`
- `buffer_registry.update_buffer(name, parent, size=...)`

### Solver Integration
Instrumented implicit methods create instrumented solvers:
- `InstrumentedLinearSolver` - from `.matrix_free_solvers`
- `InstrumentedNewtonKrylov` - from `.matrix_free_solvers`

Child allocators bridge parent algorithm and child solver buffers.

---

## Expected Behavior

After changes:
1. All instrumented files use consistent buffer allocation patterns
2. Buffer registration happens in `register_buffers()` method
3. Allocator functions obtained at compile time in `build_step()`
4. Allocator calls replace manual shared memory slicing
5. All logging functionality preserved unchanged
6. Tests using instrumented versions continue to work

---

## Dependencies

- `cubie.buffer_registry` module must be fully functional
- Source algorithm files must be stable (reference for changes)
- Instrumented matrix_free_solvers must support new signatures where needed

---

## Validation Approach

1. Compare each modified file side-by-side with source
2. Verify buffer names match source
3. Verify allocator patterns match source
4. Verify all LOGGING blocks preserved
5. Run instrumented tests (may require CUDA environment)
