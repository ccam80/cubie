# Iteration Counters Implementation Tasks

## Overview
Implement iteration_counters diagnostic output feature. Tasks ordered by dependency.

## Task Group 1: Foundation (Parallel - No Dependencies)

### Task 1.1: Linear Solver Iteration Count Return
**File**: `src/cubie/integrators/matrix_free_solvers/linear_solver.py`

**Changes**:
1. In `linear_solver_factory()` function:
   - Add iteration counter: `iter_count = int32(0)` before loop
   - Increment in loop: `iter_count += int32(1)`
   - Before final return, encode: `return_status |= (iter_count + int32(1)) << 16`
   - Return encoded status instead of plain status

2. Apply same pattern to `linear_solver_cached_factory()`

**Expected Behavior**:
- Lower 16 bits: status code (0=success, 4=max iters)
- Upper 16 bits: iteration count + 1
- Extract: `iters = (return_val >> 16) & 0xFFFF`

**Validation**: Unit test that linear solver return value has iterations in upper 16 bits

---

### Task 1.2: Output Configuration Flag
**File**: `src/cubie/outputhandling/output_config.py`

**Changes**:
1. Add to `OutputCompileFlags` attrs class:
   ```python
   output_iteration_counters: bool = False
   ```

2. Add "iteration_counters" to recognized output types (in validation/processing logic)

3. In `OutputConfig.__attrs_post_init__()`, set flag when "iteration_counters" in output_types

**Validation**: Test that output_types=["iteration_counters"] sets flag to True

---

## Task Group 2: Core Modifications (Sequential - Depends on Group 1)

### Task 2.1: Save State Signature Extension
**File**: `src/cubie/outputhandling/save_state.py`

**Changes** to `save_state_factory()`:

1. Add `output_iteration_counters: bool` parameter to factory

2. Extend device function signature:
   ```python
   def save_state_func(current_state, current_observables,
                       output_states_slice, output_observables_slice,
                       current_step,
                       output_counters_slice, counters_array):
   ```

3. Add compile-time conditional:
   ```python
   if output_iteration_counters:
       for i in range(4):
           output_counters_slice[i] = counters_array[i]
   ```

**Validation**: Test save_state writes counters array when flag enabled

---

### Task 2.2: Newton-Krylov Krylov Count Tracking
**Files**: 
- `src/cubie/integrators/matrix_free_solvers/newton_krylov.py`
- `tests/integrators/algorithms/instrumented/matrix_free_solvers.py`

**Changes** to `newton_krylov_solver_factory()`:

1. Add `counters` parameter to device function signature (size (2,) int32 array):
   ```python
   def newton_krylov_solver(
       stage_increment, parameters, drivers, t, h, a_ij, base_state,
       shared_scratch, counters):
   ```

2. Add local accumulator and track Krylov iterations:
   ```python
   total_krylov_iters = int32(0)
   # After each linear_solver() call:
   lin_return = linear_solver(...)
   krylov_iters = (lin_return >> 16) & int32(0xFFFF)
   total_krylov_iters += krylov_iters
   ```

3. Before return, write to counters array:
   ```python
   counters[0] = (newton_iters + 1)  # Newton iteration count
   counters[1] = total_krylov_iters  # Krylov iteration count
   ```

4. Return status as before (with Newton iters in upper 16 bits)

**Validation**: Test Newton solver writes to counters array correctly

---

### Task 2.3: Integration Loop Counter Tracking
**File**: `src/cubie/integrators/loops/ode_loop.py`

**Changes** to `IVPLoop.build()` method:

1. Add iteration_counters_output parameter to loop_fn signature

2. Allocate counters array:
   ```python
   counters_since_save = cuda.local.array(4, int32)
   for i in range(4):
       counters_since_save[i] = int32(0)
   ```

3. After each step, extract and accumulate:
   ```python
   step_status = step_function(...)
   newton_iters = (step_status >> 16) & status_mask
   counters_since_save[0] += newton_iters
   
   # Extract Krylov (method TBD based on Task 2.2)
   counters_since_save[1] += krylov_iters
   
   # Track steps
   counters_since_save[2] += int32(1)
   
   # Track rejections
   if not accept:
       counters_since_save[3] += int32(1)
   ```

4. Pass to save_state:
   ```python
   if do_save:
       save_state(
           ...,
           iteration_counters_output[save_idx, :],
           counters_since_save,
       )
       # Reset counters
       for i in range(4):
           counters_since_save[i] = int32(0)
   ```

**Validation**: Test loop accumulates all 4 counter types correctly

---

## Task Group 3: Infrastructure (Sequential - Depends on Group 2)

### Task 3.1: Loop Configuration Updates
**File**: `src/cubie/integrators/loops/ode_loop_config.py`

**Changes**:
- Verify compile flags propagate to loop
- Add iteration_counters_output to loop_fn parameters (handled in Task 2.3)
- No new attrs fields needed (flags already in OutputCompileFlags)

**Validation**: Test loop receives correct flag value

---

### Task 3.2: Output Array Allocation
**File**: `src/cubie/batchsolving/arrays/BatchOutputArrays.py`

**Changes**:

1. Add to `OutputArrayContainer`:
   ```python
   iteration_counters: Optional[NDArray] = None
   ```

2. In `OutputArrays` allocation logic:
   ```python
   if flags.output_iteration_counters:
       shape = (n_runs, n_saves, 4)
       self.iteration_counters = allocate_device_array(shape, dtype=np.int32)
   ```

3. Add transfer logic for iteration_counters (host<->device)

**Validation**: Test array allocated with correct shape when flag enabled

---

### Task 3.3: Batch Kernel Integration
**File**: `src/cubie/batchsolving/BatchSolverKernel.py`

**Changes**:
- Pass iteration_counters_output to loop_fn call
- Handle chunking for iteration_counters array (slice like state arrays)

**Validation**: Test kernel passes array correctly, chunking works

---

## Task Group 4: User Interface (Sequential - Depends on Group 3)

### Task 4.1: Solve Result Property
**File**: `src/cubie/batchsolving/solveresult.py`

**Changes**:
```python
@property
def iteration_counters(self) -> Optional[NDArray]:
    """Iteration counters at each save point.
    
    Returns array of shape (n_runs, n_saves, 4) where:
    - [:, :, 0]: Newton iteration counts
    - [:, :, 1]: Krylov iteration counts  
    - [:, :, 2]: Total steps between saves
    - [:, :, 3]: Rejected steps between saves
    
    Returns None if iteration_counters output was not requested.
    """
    return self._output_arrays.iteration_counters
```

**Validation**: Test property returns correct array or None

---

## Task Group 5: Testing

### Task 5.1: Unit Tests - Linear Solver
**File**: `tests/integrators/matrix_free_solvers/test_linear_solver.py`

**Tests**:
- Linear solver returns iterations in upper 16 bits
- Iteration count matches actual loop count
- Status code preserved in lower 16 bits

---

### Task 5.2: Unit Tests - Save State
**File**: `tests/outputhandling/test_save_state.py`

**Tests**:
- save_state writes counters when flag enabled
- save_state ignores counters when flag disabled
- All 4 counter values written correctly

---

### Task 5.3: Integration Tests - Full Loop
**File**: `tests/integrators/loops/test_iteration_counters.py`

**Tests**:
- End-to-end implicit solver with iteration_counters
- Verify Newton and Krylov counts reasonable
- Verify step counts and rejections tracked
- Verify counters reset between saves

---

### Task 5.4: Integration Tests - Batch Solver
**File**: `tests/batchsolving/test_iteration_counters_batch.py`

**Tests**:
- Batch solve with iteration_counters output
- Verify shape: (n_runs, n_saves, 4)
- Verify all runs have valid counter data
- Test chunking with iteration_counters

---

### Task 5.5: Performance Tests
**File**: `tests/performance/test_iteration_counters_overhead.py`

**Tests**:
- Benchmark with iteration_counters disabled (baseline)
- Benchmark with iteration_counters enabled
- Verify overhead < 2%

---

## Implementation Order Recommendation

1. **Phase 1** (Parallel): Tasks 1.1, 1.2
2. **Phase 2** (Sequential): Tasks 2.1, 2.2, 2.3
3. **Phase 3** (Sequential): Tasks 3.1, 3.2, 3.3
4. **Phase 4**: Task 4.1
5. **Phase 5** (Parallel): Tasks 5.1-5.5

## Notes

- Task 2.2 (Newton-Krylov) needs coordination with Task 2.3 (Loop) for Krylov count extraction
- Consider auxiliary buffer approach for Krylov count propagation
- All tests should use pytest fixtures from conftest.py
- Avoid mocks/patches - use real cubie objects in tests
