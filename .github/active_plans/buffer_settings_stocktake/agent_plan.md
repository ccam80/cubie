# BufferSettings Stocktake: Agent Plan

## Purpose

This document provides detailed technical specifications for auditing and fixing BufferSettings classes across the cubie codebase. The goal is to ensure that memory requirements calculated by BufferSettings match the actual allocations in device functions.

---

## Component Inventory

### 1. Base Classes (`src/cubie/BufferSettings.py`)

**Classes:**
- `LocalSizes`: Base class for local array sizes with `nonzero(attr_name)` method
- `SliceIndices`: Base class for shared memory slice indices
- `BufferSettings`: Abstract base class defining the interface

**Required Properties:**
- `shared_memory_elements` → int
- `local_memory_elements` → int
- `local_sizes` → LocalSizes subclass
- `shared_indices` → SliceIndices subclass

---

### 2. Loop BufferSettings (`src/cubie/integrators/loops/ode_loop.py`)

**Class:** `LoopBufferSettings`

**Buffers to Audit:**

| Buffer Name | Size | Location Toggle | Device Function Variable |
|-------------|------|-----------------|--------------------------|
| state | n_states | state_buffer_location | state_buffer |
| proposed_state | n_states | state_proposal_location | state_proposal_buffer |
| parameters | n_parameters | parameters_location | parameters_buffer |
| drivers | n_drivers | drivers_location | drivers_buffer |
| proposed_drivers | n_drivers | drivers_proposal_location | drivers_proposal_buffer |
| observables | n_observables | observables_location | observables_buffer |
| proposed_observables | n_observables | observables_proposal_location | observables_proposal_buffer |
| error | n_error | error_location | error |
| counters | n_counters | counters_location | counters_since_save |
| proposed_counters | 2 (when n_counters > 0) | counters_location | proposed_counters |
| state_summary | state_summary_buffer_height | state_summary_location | state_summary_buffer |
| observable_summary | observable_summary_buffer_height | observable_summary_location | observable_summary_buffer |

**Special Cases:**
- `proposed_counters` is tied to `counters_location` (same toggle)
- `proposed_counters` is only 2 elements when counters are active

**Expected Behavior:**
- When a buffer's location is 'shared', it contributes to `shared_memory_elements`
- When a buffer's location is 'local', it contributes to `local_memory_elements`
- `local_sizes` returns nonzero sizes even for zero-sized buffers (CUDA requirement)

---

### 3. ERK BufferSettings (`src/cubie/integrators/algorithms/generic_erk.py`)

**Class:** `ERKBufferSettings`

**Buffers to Audit:**

| Buffer Name | Size | Location Toggle | Device Function Variable |
|-------------|------|-----------------|--------------------------|
| stage_rhs | n | stage_rhs_location | stage_rhs |
| stage_accumulator | (stage_count - 1) * n | stage_accumulator_location | stage_accumulator |
| stage_cache | n (aliased or persistent) | (derived) | stage_cache |

**Aliasing Logic:**
- `stage_cache` aliases `stage_rhs` if `stage_rhs` is shared
- `stage_cache` aliases `stage_accumulator[0:n]` if `stage_rhs` is local but `stage_accumulator` is shared
- `stage_cache` uses `persistent_local` if neither is shared

**Key Properties:**
- `accumulator_length` = max(stage_count - 1, 0) * n
- `persistent_local_elements` = n when stage_cache cannot alias shared buffers

---

### 4. DIRK BufferSettings (`src/cubie/integrators/algorithms/generic_dirk.py`)

**Class:** `DIRKBufferSettings`

**Buffers to Audit:**

| Buffer Name | Size | Location Toggle | Device Function Variable |
|-------------|------|-----------------|--------------------------|
| stage_increment | n | stage_increment_location | stage_increment |
| stage_base | n (aliased when multistage) | stage_base_location | stage_base |
| accumulator | (stage_count - 1) * n | accumulator_location | stage_accumulator |
| solver_scratch | 2 * n | solver_scratch_location | solver_scratch |
| increment_cache | n (persistent) | (derived from solver_scratch) | increment_cache |
| rhs_cache | n (persistent) | (derived from solver_scratch) | rhs_cache |

**Aliasing Logic:**
- `stage_base` aliases `accumulator[0:n]` when multistage and accumulator is shared
- `increment_cache` and `rhs_cache` alias `solver_scratch[n:2n]` and `solver_scratch[0:n]` when shared
- Both caches use `persistent_local` when solver_scratch is local

**Key Properties:**
- `solver_scratch_elements` = 2 * n
- `persistent_local_elements` = 2 * n when solver_scratch is local (for FSAL)

**Newton Solver Integration:**
- Newton solver uses `solver_scratch[0:n]` for delta, `solver_scratch[n:2n]` for residual
- Linear solver uses `solver_scratch[2n:]` (which is empty in current implementation)
- **Issue:** DIRK's solver_scratch is only 2*n, but Newton needs 2*n + linear solver space

---

### 5. FIRK BufferSettings (`src/cubie/integrators/algorithms/generic_firk.py`)

**Class:** `FIRKBufferSettings`

**Buffers to Audit:**

| Buffer Name | Size | Location Toggle | Device Function Variable |
|-------------|------|-----------------|--------------------------|
| solver_scratch | 2 * all_stages_n | solver_scratch_location | solver_scratch |
| stage_increment | all_stages_n | stage_increment_location | stage_increment |
| stage_driver_stack | stage_count * n_drivers | stage_driver_stack_location | stage_driver_stack |
| stage_state | n | stage_state_location | stage_state |

**Key Properties:**
- `all_stages_n` = stage_count * n
- `solver_scratch_elements` = 2 * all_stages_n
- `stage_driver_stack_elements` = stage_count * n_drivers

---

### 6. Rosenbrock BufferSettings (`src/cubie/integrators/algorithms/generic_rosenbrock_w.py`)

**Class:** `RosenbrockBufferSettings`

**Buffers to Audit:**

| Buffer Name | Size | Location Toggle | Device Function Variable |
|-------------|------|-----------------|--------------------------|
| stage_rhs | n | stage_rhs_location | stage_rhs |
| stage_store | stage_count * n | stage_store_location | stage_store |
| cached_auxiliaries | cached_auxiliary_count | cached_auxiliaries_location | cached_auxiliaries |

**Special Cases:**
- `cached_auxiliary_count` is initially 0 and updated lazily when `build_implicit_helpers` is called
- Need to ensure buffer_settings is updated when cached_auxiliary_count changes

**Key Properties:**
- `stage_store_elements` = stage_count * n

---

### 7. Linear Solver BufferSettings (`src/cubie/integrators/matrix_free_solvers/linear_solver.py`)

**Class:** `LinearSolverBufferSettings`

**Buffers to Audit:**

| Buffer Name | Size | Location Toggle | Device Function Variable |
|-------------|------|-----------------|--------------------------|
| preconditioned_vec | n | preconditioned_vec_location | preconditioned_vec |
| temp | n | temp_location | temp |

**Key Properties:**
- Both buffers are size n
- Default is all local (no shared memory)

**Integration Issue:**
- Currently LinearSolverBufferSettings exists but is not passed from algorithms
- Need to wire it through DIRK/FIRK to linear solver factory

---

### 8. Newton Solver (`src/cubie/integrators/matrix_free_solvers/newton_krylov.py`)

**Current State:**
- No BufferSettings class exists
- Uses implicit 2*n carving from shared_scratch parameter
- Creates residual_temp as local array inside backtracking loop

**Buffers Used:**

| Buffer Name | Size | Source | Device Function Variable |
|-------------|------|--------|--------------------------|
| delta | n | shared_scratch[0:n] | delta |
| residual | n | shared_scratch[n:2n] | residual |
| krylov_iters_local | 1 (int32) | cuda.local.array | krylov_iters_local |
| residual_temp | n | cuda.local.array | residual_temp |

**Requirements:**
- Newton needs 2*n from shared_scratch for delta + residual
- Linear solver needs additional space starting at shared_scratch[2n:]
- Total Newton+Linear shared = 2*n + linear_solver.shared_memory_elements

---

## Integration Points

### SingleIntegratorRun Memory Aggregation

**Current Logic (SingleIntegratorRun.py lines 72-93):**
```python
@property
def shared_memory_elements(self) -> int:
    loop_shared = self._loop.shared_memory_elements
    algorithm_shared = self._algo_step.shared_memory_required
    return loop_shared + algorithm_shared

@property
def local_memory_elements(self) -> int:
    loop = self._loop.local_memory_elements
    algorithm = self._algo_step.persistent_local_required
    controller = self._step_controller.local_memory_elements
    return loop + algorithm + controller
```

**Expected Behavior:**
- Algorithm's `shared_memory_required` must include solver scratch space
- For implicit algorithms, this includes Newton's 2*n + linear solver's shared

### BatchSolverKernel Memory Usage

**Current Logic (BatchSolverKernel.py lines 157-181):**
- Gets `local_memory_elements` and `shared_memory_elements` from SingleIntegratorRun
- Passes to BatchSolverConfig for kernel compilation
- Allocates dynamic shared memory per block based on `shared_memory_bytes`

---

## Verification Strategy

### For Each BufferSettings Class:

1. **Extract all `cuda.local.array` calls** from the device function
2. **Extract all `shared[slice]` accesses** from the device function
3. **Map each allocation** to a BufferSettings entry
4. **Verify size calculations** match actual allocation sizes
5. **Verify slice indices** are contiguous and non-overlapping
6. **Verify local_sizes** provides correct nonzero values

### Cross-Component Verification:

1. **Trace shared memory flow** from kernel → loop → algorithm → solver
2. **Verify remaining_scratch_ind** correctly skips loop allocations
3. **Verify lin_shared** correctly skips Newton allocations
4. **Sum total shared** and compare to kernel allocation

---

## Edge Cases

1. **Zero-sized buffers**: Must still allocate 1 element for local arrays
2. **Single-stage algorithms**: Accumulator length is 0
3. **No drivers**: Driver buffers still need nonzero local sizes
4. **No counters**: proposed_counters handled separately
5. **Cached auxiliary lazy init**: Rosenbrock updates count after helper build
6. **FSAL disabled**: Persistent local still needed for stage_cache

---

## Dependencies

### Required Imports in Algorithm Modules:
```python
from cubie.BufferSettings import BufferSettings, LocalSizes, SliceIndices
```

### LinearSolverBufferSettings Integration:
- Add `buffer_settings` parameter to `linear_solver_factory`
- Add `buffer_settings` parameter to `linear_solver_cached_factory`
- Pass from DIRK/FIRK when constructing linear solver
- Update Newton solver to account for linear solver shared space

---

## Test Strategy

For each BufferSettings class, create tests that:

1. Verify `shared_memory_elements` matches sum of shared buffer sizes
2. Verify `local_memory_elements` matches sum of local buffer sizes
3. Verify slice indices are contiguous (slice[i].stop == slice[i+1].start)
4. Verify slice indices cover exactly `shared_memory_elements` range
5. Verify toggling locations correctly moves buffers between counts
6. Verify `local_sizes.nonzero()` returns ≥1 for all attributes
