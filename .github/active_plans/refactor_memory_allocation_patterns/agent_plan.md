# Refactor Memory Allocation Patterns - Agent Implementation Plan

## Overview

This refactoring removes the anti-pattern of storing memory allocation sizes in BatchSolverConfig and instead uses the centralized buffer_registry pattern consistently across all components. The changes ensure SingleIntegratorRun and BatchSolverKernel query buffer_registry directly for memory sizes rather than duplicating this information in compile settings.

## Component Changes Required

### 1. BatchSolverConfig (src/cubie/batchsolving/BatchSolverConfig.py)

**Current State:**
- Has `local_memory_elements` field (int, default=0)
- Has `shared_memory_elements` field (int, default=0)
- These fields are part of compile settings attrs class

**Required Changes:**
- Remove `local_memory_elements` field completely
- Remove `shared_memory_elements` field completely
- Keep `loop_fn`, `compile_flags`, and `precision` fields
- Update docstring to reflect that memory sizes are not part of config

**Expected Behavior:**
- BatchSolverConfig only contains true compile-time parameters
- Instantiation no longer requires memory element parameters
- Equality comparison no longer considers memory elements

### 2. BatchSolverKernel (src/cubie/batchsolving/BatchSolverKernel.py)

**Current State:**
- Creates BatchSolverConfig with memory elements from SingleIntegratorRun
- Updates config with memory elements in __init__ and update()
- Properties delegate to config.local_memory_elements and config.shared_memory_elements

**Required Changes:**

#### __init__ method (around lines 286-313)
- Remove local_memory_elements and shared_memory_elements from initial_config
- Remove the second update_compile_settings call that sets memory elements
- Keep loop_fn and compile_flags in config

#### build_kernel method (around line 700)
- Change `shared_elems_per_run = config.shared_memory_elements` to query buffer_registry
- Use `buffer_registry.shared_buffer_size(self.single_integrator)` instead

#### run method (around lines 527-539)
- Remove local_memory_elements and shared_memory_elements from update_compile_settings dict
- Keep loop_fn and precision updates

#### update method (around lines 877-888)
- Remove local_memory_elements and shared_memory_elements from updates_dict
- Add call to buffer_registry.update() to handle location parameters
- Ensure buffer sizes are queried fresh after SingleIntegratorRun update

#### local_memory_elements property (around line 912)
- Change from `return self.compile_settings.local_memory_elements`
- To: `return buffer_registry.persistent_local_buffer_size(self.single_integrator)`

#### shared_memory_elements property (around line 918)
- Change from `return self.compile_settings.shared_memory_elements`
- To: `return buffer_registry.shared_buffer_size(self.single_integrator)`

#### shared_memory_bytes property (around line 1036)
- Change from delegating to SingleIntegratorRun
- To: compute directly from buffer_registry query

**Expected Behavior:**
- BatchSolverKernel no longer stores memory sizes
- Properties compute memory sizes on-demand from buffer_registry
- Update method recognizes buffer location parameters
- Memory sizes always reflect current buffer_registry state
- Kernel build uses current buffer sizes from registry

### 3. SingleIntegratorRun (src/cubie/integrators/SingleIntegratorRun.py)

**Current State:**
- Properties delegate to internal _loop object's buffer_registry queries
- Properties: shared_memory_elements, shared_memory_bytes, local_memory_elements

**Required Changes:**
- **NO CHANGES NEEDED** - properties already query buffer_registry correctly
- These properties remain as they are (delegating to _loop)
- They serve as the bridge between BatchSolverKernel and buffer_registry

**Expected Behavior:**
- Properties continue to work as before
- BatchSolverKernel uses these properties (or queries buffer_registry directly)
- No behavioral changes in SingleIntegratorRun

### 4. buffer_registry (src/cubie/buffer_registry.py)

**Current State:**
- Provides get_child_allocators() method
- Provides shared_buffer_size(), persistent_local_buffer_size() methods
- Provides update() method for location parameters

**Required Changes:**
- **NO CHANGES NEEDED** - existing functionality is sufficient
- Padding is NOT part of buffer_registry (it's kernel-specific)

**Expected Behavior:**
- Works exactly as before
- BatchSolverKernel queries it for child buffer sizes
- No new functionality required

## Integration Points

### Between BatchSolverKernel and SingleIntegratorRun

**Current Integration:**
```python
# In BatchSolverKernel.__init__
self.single_integrator = SingleIntegratorRun(...)
config = BatchSolverConfig(
    local_memory_elements=self.single_integrator.local_memory_elements,
    shared_memory_elements=self.single_integrator.shared_memory_elements,
    ...
)
```

**New Integration:**
```python
# In BatchSolverKernel.__init__
self.single_integrator = SingleIntegratorRun(...)
config = BatchSolverConfig(
    loop_fn=None,
    compile_flags=self.single_integrator.output_compile_flags,
)
# Memory sizes queried via properties when needed
```

### Between BatchSolverKernel and buffer_registry

**Current Integration:**
- Indirect through SingleIntegratorRun properties
- No direct buffer_registry calls in BatchSolverKernel

**New Integration:**
```python
# In BatchSolverKernel properties
@property
def shared_memory_elements(self) -> int:
    return buffer_registry.shared_buffer_size(self.single_integrator)

@property
def local_memory_elements(self) -> int:
    return buffer_registry.persistent_local_buffer_size(self.single_integrator)
```

### Update Pattern

**Current Pattern:**
```python
# In BatchSolverKernel.update()
self.single_integrator.update(updates_dict, silent=True)
updates_dict.update({
    'local_memory_elements': self.single_integrator.local_memory_elements,
    'shared_memory_elements': self.single_integrator.shared_memory_elements,
})
self.update_compile_settings(updates_dict, silent=True)
```

**New Pattern:**
```python
# In BatchSolverKernel.update()
self.single_integrator.update(updates_dict, silent=True)
# Memory sizes are now queried via properties, not stored in config
updates_dict.update({
    'loop_fn': self.single_integrator.device_function,
    'compile_flags': self.single_integrator.output_compile_flags,
})
self.update_compile_settings(updates_dict, silent=True)
```

## Edge Cases to Consider

### 1. Cache Invalidation
**Scenario:** Memory sizes change but config doesn't track them
**Handling:** Config should invalidate based on loop_fn and compile_flags changes, not memory sizes. Memory sizes are metadata, not compile-time parameters affecting generated code.

### 2. Multiple Update Calls
**Scenario:** BatchSolverKernel.update() called multiple times with different settings
**Handling:** Each call updates SingleIntegratorRun, which updates buffer_registry. Properties always return current state.

### 3. Padding Calculation
**Scenario:** shared_memory_needs_padding property depends on shared_memory_elements
**Handling:** Property queries buffer_registry for current shared_memory_elements. Padding logic remains in BatchSolverKernel (not buffer_registry).

### 4. Kernel Build Timing
**Scenario:** build_kernel() needs current memory sizes
**Handling:** Queries buffer_registry at build time via properties. This ensures build uses current state, not stale config values.

### 5. Zero-Sized Buffers
**Scenario:** Loop has no shared memory (shared_memory_elements == 0)
**Handling:** buffer_registry.shared_buffer_size() returns 0. Padding property handles zero case correctly.

### 6. Chunking and Memory Allocation
**Scenario:** Chunking decisions based on memory sizes
**Handling:** Use properties to query current sizes. Memory manager already handles this correctly.

## Data Structure Changes

### BatchSolverConfig Before
```python
@attrs.define
class BatchSolverConfig(CUDAFactoryConfig):
    loop_fn: Optional[Callable]
    local_memory_elements: int = 0
    shared_memory_elements: int = 0
    compile_flags: Optional[OutputCompileFlags]
```

### BatchSolverConfig After
```python
@attrs.define
class BatchSolverConfig(CUDAFactoryConfig):
    loop_fn: Optional[Callable]
    compile_flags: Optional[OutputCompileFlags]
    # Memory elements removed - queried from buffer_registry
```

### BatchSolverKernel Properties Before
```python
@property
def shared_memory_elements(self) -> int:
    return self.compile_settings.shared_memory_elements
```

### BatchSolverKernel Properties After
```python
@property
def shared_memory_elements(self) -> int:
    return buffer_registry.shared_buffer_size(self.single_integrator)
```

## Expected Dependencies

### Import Changes
- BatchSolverKernel already imports buffer_registry (line 40)
- No new imports needed in any file

### Method Call Changes
- BatchSolverKernel will call buffer_registry methods directly
- No changes to buffer_registry API needed

## Testing Considerations

### Existing Tests Should Pass
- All existing BatchSolverKernel tests should pass unchanged
- Public API behavior is identical
- Memory size queries return same values

### Property Behavior
- shared_memory_elements should return same value as before
- local_memory_elements should return same value as before
- shared_memory_bytes should return same value as before

### Update Behavior
- Updating loop settings should invalidate cache correctly
- Memory sizes should reflect updates to SingleIntegratorRun
- buffer_registry should be queried fresh after updates

### Edge Case Coverage
- Zero shared memory should work correctly
- Padding calculation should work with new property implementation
- Chunking based on memory sizes should work unchanged

## Architectural Alignment

This refactoring aligns with the buffer_registry pattern described in `.github/context/cubie_internal_structure.md` lines 60-191:

1. **Buffer Registration:** IVPLoop registers buffers with buffer_registry
2. **Child Buffer Delegation:** Parent queries child buffer sizes via buffer_registry methods
3. **Allocation Pattern:** Buffers allocated at runtime using pre-generated allocators
4. **Update Pattern:** buffer_registry.update() handles location parameter updates

The change removes the anti-pattern of storing buffer sizes in compile settings and instead queries buffer_registry directly, matching the pattern used throughout CuBIE.

## Success Criteria

1. BatchSolverConfig has no memory element fields
2. BatchSolverKernel properties query buffer_registry directly
3. All existing tests pass without modification
4. Memory allocation behavior is identical to before
5. Update method recognizes buffer location parameters
6. Code follows established buffer_registry pattern
