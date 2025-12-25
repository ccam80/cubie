# Managed Buffer Refactor: Agent Plan

## Overview

This plan details the technical specification for refactoring compile-time CUDA_SIMULATION conditionals for `cuda.local.array` calls into the centralized buffer registry system. The goal is to consolidate all CUDASIM compatibility logic into `buffer_registry.py`.

## Architecture Context

### Buffer Registry Pattern

The buffer registry (`src/cubie/buffer_registry.py`) provides a centralized system for CUDA memory management:

1. **Registration Phase** - Buffers registered with name, parent, size, location, and precision
2. **Allocator Generation** - `get_allocator()` returns a compiled CUDA device function
3. **Runtime Allocation** - Device function calls allocator to get buffer

The critical CUDASIM handling already exists in `CUDABuffer.build_allocator()` (lines 128-158):
```python
if CUDA_SIMULATION:
    # Uses np.zeros instead of cuda.local.array
else:
    # Uses cuda.local.array
```

### Current Problem Locations

Five compile-time conditionals exist outside buffer_registry.py:

1. **BatchSolverKernel.py:578-584** - `local_scratch` (float32)
2. **ode_loop.py:487-490** - `proposed_counters` (int32, size 2)
3. **newton_krylov.py:383-386** - `krylov_iters_local` (int32, size 1)
4. **generic_rosenbrock_w.py:449-454** - `base_state_placeholder` (int32, size 1) and `krylov_iters_out` (int32, size 1)

---

## Component Changes

### 1. buffer_registry.py - Extend Precision Support

**Current Behavior**: The `CUDABuffer` class uses `precision_validator` which only accepts numpy floating-point types.

**Required Change**: Allow integer types (np.int32, np.int64) for precision field to support counter buffers.

**Details**:
- Modify validation in `CUDABuffer.precision` field to accept integer types
- Alternative: Add a separate `dtype` field that accepts any valid numpy dtype
- The `build_allocator()` method already uses `_precision` generically, so the allocator will work with int32

**Integration Points**:
- `CUDABuffer.__init__` - precision/dtype validation
- `CUDABuffer.build_allocator()` - dtype handling in allocation

### 2. BatchSolverKernel.py - local_scratch Buffer

**Current Code** (lines 578-584):
```python
if CUDA_SIMULATION:
    local_scratch = np.zeros(local_elements_per_run, dtype=np.float32)
else:
    local_scratch = cuda.local.array(local_elements_per_run, dtype=float32)
```

**Context**: This is inside `build_kernel()`, which creates a top-level kernel (not a device function). The `local_scratch` is passed to the loop function.

**Required Changes**:
- The `local_scratch` buffer should be allocated by `SingleIntegratorRun` through the buffer registry
- `BatchSolverKernel` should receive an allocator from `SingleIntegratorRun`
- Register `local_scratch` buffer in `SingleIntegratorRun.register_buffers()` or equivalent

**Alternative Approach**: Since `BatchSolverKernel` has access to `compile_settings.local_memory_elements`, it can register the buffer directly:
- Add buffer registration in `BatchSolverKernel.__init__()` or before `build_kernel()`
- Get allocator in `build_kernel()` and call it within the kernel

**Implementation Notes**:
- Buffer name: `'kernel_local_scratch'`
- Location: `'local'` (default)
- Size: `local_elements_per_run` (from `config.local_memory_elements`)
- Precision: `np.float32` (always 32-bit for local scratch)

### 3. ode_loop.py - proposed_counters Buffer

**Current Code** (lines 487-490):
```python
if CUDA_SIMULATION:
    proposed_counters = np.zeros(2, dtype=np.int32)
else:
    proposed_counters = cuda.local.array(2, dtype=simsafe_int32)
```

**Context**: Inside `build()` method of `IVPLoop`, used to track Newton/Krylov iterations.

**Required Changes**:
1. Add to `ODELoopConfig`:
   - `proposed_counters_location: str = field(default='local', ...)`
   
2. Add to `IVPLoop.register_buffers()`:
   ```python
   buffer_registry.register(
       'proposed_counters', self, 2, config.proposed_counters_location,
       precision=np.int32
   )
   ```

3. Update `IVPLoop.build()`:
   - Get allocator: `alloc_proposed_counters = getalloc('proposed_counters', self)`
   - In device function: `proposed_counters = alloc_proposed_counters(shared_scratch, persistent_local)`

4. Add `'proposed_counters_location'` to `ALL_LOOP_SETTINGS`

### 4. newton_krylov.py - krylov_iters_local Buffer

**Current Code** (lines 383-386):
```python
if CUDA_SIMULATION:
    krylov_iters_local = np.zeros(1, dtype=np.int32)
else:
    krylov_iters_local = cuda.local.array(1, int32)
```

**Context**: Inside `build()` method of `NewtonKrylov`, stores iteration count.

**Required Changes**:
1. Add to `NewtonKrylovConfig`:
   - `krylov_iters_local_location: str = field(default='local', ...)`

2. Add to `NewtonKrylov.register_buffers()`:
   ```python
   buffer_registry.register(
       'krylov_iters_local', self, 1, config.krylov_iters_local_location,
       precision=np.int32
   )
   ```

3. Update `NewtonKrylov.build()`:
   - Get allocator before device function: `alloc_krylov_iters = get_alloc('krylov_iters_local', self)`
   - In device function: `krylov_iters_local = alloc_krylov_iters(shared_scratch, persistent_scratch)`

### 5. generic_rosenbrock_w.py - base_state_placeholder and krylov_iters_out

**Current Code** (lines 449-454):
```python
if CUDA_SIMULATION:
    base_state_placeholder = np.zeros(1, dtype=np.int32)
    krylov_iters_out = np.zeros(1, dtype=np.int32)
else:
    base_state_placeholder = cuda.local.array(1, int32)
    krylov_iters_out = cuda.local.array(1, int32)
```

**Context**: Inside `build_step()` method of `GenericRosenbrockWStep`, used as placeholder arrays.

**Required Changes**:
1. Add to `RosenbrockWStepConfig`:
   - `base_state_placeholder_location: str = field(default='local', ...)`
   - `krylov_iters_out_location: str = field(default='local', ...)`

2. Add to `GenericRosenbrockWStep.register_buffers()`:
   ```python
   buffer_registry.register(
       'base_state_placeholder', self, 1, config.base_state_placeholder_location,
       precision=np.int32
   )
   buffer_registry.register(
       'krylov_iters_out', self, 1, config.krylov_iters_out_location,
       precision=np.int32
   )
   ```

3. Update `GenericRosenbrockWStep.build_step()`:
   - Get allocators before device function definition
   - Use allocators inside device function

---

## Dependencies and Imports

### New Imports Required

**buffer_registry.py**:
- None (already has numpy, numba imports)

**BatchSolverKernel.py**:
- Add: `from cubie.buffer_registry import buffer_registry`

**ODELoopConfig (ode_loop_config.py)**:
- None (already has validators)

### Import Removals

After refactoring, these files can remove CUDA_SIMULATION import if no other uses remain:
- `BatchSolverKernel.py` - Check if CUDA_SIMULATION used elsewhere
- `ode_loop.py` - Check if CUDA_SIMULATION used elsewhere  
- `newton_krylov.py` - Check if CUDA_SIMULATION used elsewhere
- `generic_rosenbrock_w.py` - Check if CUDA_SIMULATION used elsewhere

---

## Edge Cases

### 1. Integer Precision Validation

The buffer registry's `CUDABuffer` class uses `precision_validator` which validates against `ALLOWED_PRECISIONS` (float16, float32, float64). For int32 buffers:

**Option A**: Extend ALLOWED_PRECISIONS to include integer types
**Option B**: Add a separate `dtype` parameter with broader validation
**Option C**: Create integer-specific register method

**Recommended**: Option A or B - keep the API simple

### 2. Zero-Size Buffers

The `cached_auxiliaries` buffer in Rosenbrock is registered with size 0 initially. Ensure this doesn't conflict with the new integer buffers.

### 3. Kernel vs Device Function Context

`BatchSolverKernel.build_kernel()` creates a kernel, not a device function. The allocator pattern works the same way - capture allocator in closure, call at runtime.

### 4. Allocator Parameter Requirements

All allocators expect `(shared_scratch, persistent_local)` parameters. In contexts where these don't exist (top-level kernel), create dummy arrays or pass actual memory regions.

---

## Data Structures

### Updated Config Classes

**ODELoopConfig** (add field):
```python
proposed_counters_location: str = field(
    default='local',
    validator=validators.in_(['shared', 'local'])
)
```

**NewtonKrylovConfig** (add field):
```python
krylov_iters_local_location: str = field(
    default='local',
    validator=validators.in_(['shared', 'local'])
)
```

**RosenbrockWStepConfig** (add fields):
```python
base_state_placeholder_location: str = field(
    default='local',
    validator=validators.in_(['shared', 'local'])
)
krylov_iters_out_location: str = field(
    default='local',
    validator=validators.in_(['shared', 'local'])
)
```

---

## Expected Interactions

### Buffer Registry with Int32 Buffers

The `build_allocator()` method in `CUDABuffer`:
- Sets `_precision = self.precision` (will be np.int32)
- In CUDASIM: `array = np.zeros(_local_size, dtype=_precision)`
- In CUDA: `array = cuda.local.array(_local_size, _precision)`

Both paths work correctly with np.int32.

### Cross-Component Buffer Sizing

No changes needed to buffer sizing computations since:
- `shared_buffer_size()` only counts shared buffers
- `local_buffer_size()` counts local buffers  
- `persistent_local_buffer_size()` counts persistent local buffers

Local buffers (default for new buffers) don't contribute to shared/persistent counts.

---

## Validation Strategy

1. **Unit Tests**: Ensure buffer registration accepts int32 precision
2. **Integration Tests**: Run existing tests in both CUDA and CUDASIM modes
3. **Regression Tests**: Verify identical numerical results before/after refactor

---

## Implementation Order

1. **buffer_registry.py** - Extend precision validation for int32 (if needed)
2. **ode_loop.py** - Refactor proposed_counters (isolated change)
3. **newton_krylov.py** - Refactor krylov_iters_local (isolated change)
4. **generic_rosenbrock_w.py** - Refactor base_state_placeholder, krylov_iters_out
5. **BatchSolverKernel.py** - Refactor local_scratch (most complex due to kernel context)
6. **Cleanup** - Remove unused CUDA_SIMULATION imports

This order minimizes risk by starting with simpler, isolated changes.
