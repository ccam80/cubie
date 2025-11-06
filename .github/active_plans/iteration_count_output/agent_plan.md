# Iteration Count Output - Agent Implementation Plan

## Overview

This plan details the implementation of iteration count outputs for CuBIE's implicit solvers. The feature adds optional diagnostic outputs controlled by compile-time flags, following the existing output architecture pattern.

## Core Components to Modify

### 1. Linear Solver Enhancement (`src/cubie/integrators/matrix_free_solvers/linear_solver.py`)

**Current Behavior**:
- `linear_solver_factory()` returns `int32(0)` on convergence, `int32(4)` on max iterations
- Does NOT track or return iteration count
- Compatible with Newton-Krylov but provides no diagnostic info

**Required Changes**:
- Track iteration count in a local variable during the iteration loop
- Encode iteration count in upper 16 bits of return value (bits 31-16)
- Maintain status code in lower 16 bits (bits 15-0)
- Pattern: `status |= (iter_count + 1) << 16` (matches newton_krylov.py pattern)

**Behavior**:
- Return value structure: `(iterations << 16) | status_code`
- Iterations start at 0, incremented each loop iteration
- Add 1 before encoding (consistent with Newton solver pattern)
- Extract in caller: `iterations = (return_value >> 16) & 0xFFFF`

**Files**:
- `linear_solver_factory()` in `linear_solver.py`
- `linear_solver_cached_factory()` in `linear_solver.py` (same pattern)

**Integration Point**:
- Returned to Newton solver, which already extracts status
- Loop will need to extract Krylov count from linear solver return within Newton status

### 2. Output Configuration Extension (`src/cubie/outputhandling/output_config.py`)

**Current Behavior**:
- Recognizes output types: `"state"`, `"observables"`, `"time"`, plus summary metric names
- Creates compile flags for enabled outputs
- Validates indices and output selections

**Required Changes**:

**New Output Types** (add to recognized types):
- `"newton_iterations"` - Newton iteration counts
- `"krylov_iterations"` - Krylov/linear solver iteration counts
- `"step_counts"` - Total steps between saves
- `"rejected_steps"` - Rejected steps between saves (adaptive only)
- `"step_size"` - Step size at save point

**New Compile Flags** (add to `OutputCompileFlags` attrs class):
```python
output_newton_iterations: bool = False
output_krylov_iterations: bool = False
output_step_counts: bool = False
output_rejected_steps: bool = False
output_step_size: bool = False
```

**Behavior**:
- When user provides `output_types=["state", "newton_iterations"]`
- `OutputConfig.__attrs_post_init__()` sets corresponding flags to `True`
- Flags passed to `save_state_factory()` for compile-time branching
- Flags accessible via `OutputFunctions.compile_flags` property

**Validation**:
- No special validation needed (iteration outputs don't have indices)
- Accept any combination of iteration output types
- Warning if iteration outputs requested but algorithm is explicit (no iterations to track)

**Files**:
- `OutputCompileFlags` class definition
- `OutputConfig.__attrs_post_init__()` flag setting logic
- `ALL_OUTPUT_FUNCTION_PARAMETERS` set (add new type names)

### 3. Save State Function Signature Extension (`src/cubie/outputhandling/save_state.py`)

**Current Signature**:
```python
def save_state_func(current_state, current_observables,
                    output_states_slice, output_observables_slice,
                    current_step):
```

**New Signature**:
```python
def save_state_func(current_state, current_observables,
                    output_states_slice, output_observables_slice,
                    current_step,
                    output_newton_iters_slice, newton_iters,
                    output_krylov_iters_slice, krylov_iters,
                    output_steps_slice, steps_count,
                    output_rejected_slice, rejected_count,
                    output_stepsize_slice, step_size):
```

**Alternative Design** (cleaner, recommended):
Pass output slices as a dict-like structure or use compile-time selection to omit unused parameters. However, Numba device functions have limitations, so explicit parameters are safer.

**Behavior**:
- Compile-time branching based on `OutputCompileFlags`
- If `flags.output_newton_iterations`: write `newton_iters` to `output_newton_iters_slice[0]`
- If flag is `False`: entire branch is compiled away (zero overhead)
- Similar pattern for each iteration output type

**Implementation Pattern**:
```python
if flags.output_newton_iterations:
    output_newton_iters_slice[0] = newton_iters
if flags.output_krylov_iterations:
    output_krylov_iters_slice[0] = krylov_iters
# ... etc
```

**Files**:
- `save_state_factory()` function
- Flags passed in during construction

### 4. Integration Loop Modifications (`src/cubie/integrators/loops/ode_loop.py`)

**Current Behavior**:
- Main loop calls `step_function()`, gets status with Newton iters in upper 16 bits
- Extracts Newton iters: `niters = (step_status >> 16) & status_mask`
- Calls `save_state()` with state, observables, and time
- No accumulation of iteration counts between saves

**Required Changes**:

#### Local Memory Allocation
Add iteration counter variables in loop initialization:
```python
# Allocate counters in persistent_local or as local variables
newton_iters_since_save = int32(0)
krylov_iters_since_save = int32(0)
steps_since_save = int32(0)
rejected_steps_since_save = int32(0)
```

#### Iteration Tracking in Main Loop
After each step:
```python
step_status = step_function(...)

# Extract Newton iterations (already done)
niters = (step_status >> 16) & status_mask

# Extract Krylov iterations from Newton solver's linear solver return
# This requires Newton solver to pass through linear solver's upper 16 bits
# OR accumulate within Newton solver and encode in different bits
# OR have loop track separately by modifying step return structure
kriters = extracted_krylov_count  # See Newton Solver section

# Accumulate iterations
if flags.output_newton_iterations:
    newton_iters_since_save += niters
if flags.output_krylov_iterations:
    krylov_iters_since_save += kriters

# Track steps
if flags.output_step_counts or flags.output_rejected_steps:
    steps_since_save += int32(1)
    if not accept:
        rejected_steps_since_save += int32(1)
```

#### Modified Save Call
```python
if do_save:
    save_state(
        state_buffer, observables_buffer,
        state_output[save_idx * save_state_bool, :],
        observables_output[save_idx * save_obs_bool, :],
        t,
        # New iteration output slices
        newton_iters_output[save_idx, :],  # or just [save_idx]
        newton_iters_since_save,
        krylov_iters_output[save_idx, :],
        krylov_iters_since_save,
        steps_output[save_idx, :],
        steps_since_save,
        rejected_output[save_idx, :],
        rejected_steps_since_save,
        stepsize_output[save_idx, :],
        dt[0],  # current step size
    )
    
    # Reset accumulators
    newton_iters_since_save = int32(0)
    krylov_iters_since_save = int32(0)
    steps_since_save = int32(0)
    rejected_steps_since_save = int32(0)
```

**Key Consideration**: The loop function signature must receive new output array slices. This propagates up to `IVPLoop.__init__()` and `ODELoopConfig`.

**Files**:
- `IVPLoop.build()` method (loop_fn device function definition)
- Loop function signature in main loop
- Counter variable initialization
- Save call site

### 5. Newton-Krylov Solver Krylov Count Propagation (`src/cubie/integrators/matrix_free_solvers/newton_krylov.py`)

**Current Behavior**:
- Calls `linear_solver()` inside Newton loop
- Receives return status: `lin_return = linear_solver(...)`
- Checks if `lin_return != 0` to detect linear solver failure
- Does NOT extract or track Krylov iteration count

**Challenge**:
- Newton solver returns 32-bit status with Newton iters in upper 16 bits
- Linear solver now returns Krylov iters in upper 16 bits
- Need to propagate both counts to the loop

**Option A**: Extend Newton Solver Return to 64-bit
- Lower 16 bits: status code
- Bits 31-16: Newton iterations
- Bits 47-32: Total Krylov iterations (accumulated within Newton solver)
- Bits 63-48: Reserved

**Option B**: Separate Accumulation in Loop
- Newton solver continues returning 32-bit (Newton iters + status)
- Linear solver returns 32-bit (Krylov iters + status)
- Algorithm step function receives both and can return extended status OR
- Loop extracts Krylov count from algorithm local state
- **Problem**: Algorithm step is opaque to loop

**Option C**: Algorithm Step Extended Return
- Modify step function signature to return multiple values OR
- Use scratch buffer to communicate iteration counts back to loop
- Step function writes iteration counts to designated memory locations
- Loop reads and accumulates

**Recommended Approach** (Option C variant):
- Add optional iteration count output parameters to algorithm step signature
- When iteration output flags are enabled, allocate scratch space for counts
- Algorithm step writes counts to scratch before returning
- Loop reads and accumulates

**Implementation**:
```python
# In newton_krylov_solver:
total_krylov_iters = int32(0)
for newton_iter in range(max_iters):
    lin_return = linear_solver(...)
    krylov_count = (lin_return >> 16) & 0xFFFF
    total_krylov_iters += krylov_count
    # ... rest of logic

# Before return:
if krylov_output_enabled:  # compile-time flag
    krylov_output_buffer[0] = total_krylov_iters

# Return Newton count as before
status |= (newton_iters + 1) << 16
return status
```

Then algorithm step wrapper extracts from buffer and makes available to loop.

**Files**:
- `newton_krylov_solver_factory()` 
- Algorithm step wrappers that call Newton-Krylov
- May require compile settings to propagate iteration output flags

### 6. Loop Configuration and Buffer Management (`src/cubie/integrators/loops/ode_loop_config.py`)

**Current Behavior**:
- `ODELoopConfig` attrs class holds all loop compile settings
- `LoopSharedIndices` and `LoopLocalIndices` define buffer slices
- Loop allocates shared and local memory based on indices

**Required Changes**:

**New Loop Function Parameters**:
Add output array parameters for iteration counts:
```python
def loop_fn(initial_states, parameters, driver_coefficients,
            shared_scratch, persistent_local,
            state_output, observables_output,
            state_summaries_output, observable_summaries_output,
            # NEW:
            newton_iters_output, krylov_iters_output,
            steps_count_output, rejected_steps_output,
            step_size_output,
            duration, settling_time, t0):
```

**Conditional Parameters**:
Use compile-time flags to omit unused parameters. However, Numba may require consistent signatures. Alternative: Always pass arrays but make them size-0 when disabled.

**Size-0 Array Pattern**:
```python
# In loop construction:
if flags.output_newton_iterations:
    newton_iters_output = actual_array
else:
    newton_iters_output = cuda.local.array(0, int32)  # zero-size, compiled away

# In save call:
if flags.output_newton_iterations:
    newton_iters_output[save_idx, :] = ...
# Else: branch compiled away, no array access
```

**Files**:
- `ODELoopConfig` (add iteration output flags to config)
- `IVPLoop.build()` (loop signature and array handling)
- `LoopSharedIndices` (if iteration counters stored in shared memory)
- `LoopLocalIndices` (if iteration counters stored in local memory)

**Decision**: Store iteration counters in local variables (not shared/local memory slices) for simplicity.

### 7. Output Array Management (`src/cubie/batchsolving/arrays/BatchOutputArrays.py`)

**Current Behavior**:
- `OutputArrayContainer` attrs class holds state, observable, and summary arrays
- `ActiveOutputs` flags indicate which arrays are active
- `OutputArrays` manager allocates and transfers arrays

**Required Changes**:

**New Array Container Fields** (add to `OutputArrayContainer`):
```python
newton_iterations: Optional[NDArray] = None
krylov_iterations: Optional[NDArray] = None
step_counts: Optional[NDArray] = None
rejected_steps: Optional[NDArray] = None
step_sizes: Optional[NDArray] = None
```

**Array Allocation**:
In `OutputArrays._allocate_outputs()` or similar:
```python
if compile_flags.output_newton_iterations:
    shape = (n_runs, n_saves)
    self.newton_iterations = allocate_device_array(shape, dtype=np.int32)

if compile_flags.output_krylov_iterations:
    shape = (n_runs, n_saves)
    self.krylov_iterations = allocate_device_array(shape, dtype=np.int32)

# ... similar for other iteration types
```

**Array Transfer**:
Iteration arrays transferred to host after kernel execution, following same pattern as state/observable arrays.

**Files**:
- `OutputArrayContainer` class
- `ActiveOutputs` class (add iteration flags)
- `OutputArrays` class allocation and transfer logic

### 8. Solve Result Exposure (`src/cubie/batchsolving/solveresult.py`)

**Current Behavior**:
- `SolveResult` class exposes `state`, `observables`, `state_summaries`, `observable_summaries`
- Properties provide user-friendly access to output arrays

**Required Changes**:

**New Properties**:
```python
@property
def newton_iterations(self) -> Optional[NDArray]:
    """Newton iteration counts at each save point."""
    return self._output_arrays.newton_iterations

@property
def krylov_iterations(self) -> Optional[NDArray]:
    """Krylov/linear solver iteration counts at each save point."""
    return self._output_arrays.krylov_iterations

@property
def step_counts(self) -> Optional[NDArray]:
    """Total integration steps between save points."""
    return self._output_arrays.step_counts

@property
def rejected_steps(self) -> Optional[NDArray]:
    """Rejected steps between save points (adaptive controllers only)."""
    return self._output_arrays.rejected_steps

@property
def step_sizes(self) -> Optional[NDArray]:
    """Step size at each save point."""
    return self._output_arrays.step_sizes
```

**Behavior**:
- Returns `None` if corresponding iteration output was not requested
- Returns NumPy array of shape `(n_runs, n_saves)` if enabled
- Dtype is `np.int32` for iteration counts, `precision` for step_sizes

**Files**:
- `SolveResult` class

### 9. Output Sizes and Metadata (`src/cubie/outputhandling/output_sizes.py`)

**Current Behavior**:
- Helper classes calculate output array dimensions
- `OutputArrayHeights` determines per-variable output heights
- `BatchOutputSizes` calculates total sizes

**Required Changes**:

**New Height Calculations**:
Iteration count arrays are simpler than state/observable arrays:
- Always height=1 per save (single scalar value)
- No per-variable indexing (applies to entire integration)
- Calculated based on compile flags

**Behavior**:
May not require changes if iteration arrays are managed separately from state/observable sizing logic. Review and add helper methods if beneficial.

**Files**:
- Potentially `OutputArrayHeights` or new helper class
- Review sizing logic for consistency

### 10. Batch Solver Kernel Integration (`src/cubie/batchsolving/BatchSolverKernel.py`)

**Current Behavior**:
- Compiles batch kernel that calls loop function for each run
- Passes output arrays to loop function
- Manages chunking for memory constraints

**Required Changes**:

**Kernel Call Site**:
Pass new iteration output arrays to loop function:
```python
loop_fn(
    # ... existing parameters ...
    newton_iters_output=newton_iters_device,
    krylov_iters_output=krylov_iters_device,
    steps_count_output=steps_count_device,
    rejected_steps_output=rejected_steps_device,
    step_size_output=step_size_device,
    # ... rest ...
)
```

**Array Slicing for Chunking**:
When chunking, slice iteration arrays like state/observable arrays:
```python
chunk_newton_iters = newton_iters_output[chunk_start:chunk_end, :]
# Pass chunk slice to kernel
```

**Files**:
- `BatchSolverKernel.build()` or kernel compilation method
- Chunking logic

## Algorithm Step Return Value Consideration

Currently, algorithm step functions return a single `int32` status code with iteration count in upper 16 bits. With the addition of Krylov iteration tracking, we have options:

### Option A: Extended Return Value (64-bit)
- Return `int64` with multiple fields encoded
- Bits 15-0: Status code
- Bits 31-16: Newton iterations
- Bits 47-32: Krylov iterations
- Bits 63-48: Reserved

**Pros**: Clean, all info in one return
**Cons**: Requires changing algorithm return type throughout, potential compatibility issues

### Option B: Auxiliary Output Buffer
- Algorithm step writes iteration counts to provided buffer
- Loop reads from buffer after step call
- Return value remains `int32`

**Pros**: Minimal signature changes, flexible
**Cons**: Extra memory access, slightly complex

### Option C: Nested Status Extraction
- Newton solver accumulates Krylov iterations internally
- Returns both Newton and accumulated Krylov in extended bits
- Algorithm step passes through extended status

**Pros**: Encapsulation, natural accumulation point
**Cons**: Requires 64-bit or clever bit packing

**Recommendation**: Option B - use auxiliary buffer for iteration counts
- Least invasive to existing code
- Compile-time flag can disable buffer entirely
- Follows CuBIE's scratch buffer patterns

## Edge Cases and Error Handling

### 1. Explicit Algorithms (No Iterations)
**Scenario**: User requests `output_newton_iterations` but uses explicit Euler
**Handling**: 
- Compile-time flags still set (no harm)
- Iteration counts are zero (no Newton/Krylov calls occur)
- Arrays allocated but filled with zeros
- Optional: Warning at solver construction time

### 2. Fixed-Step Controllers (No Rejections)
**Scenario**: User requests `output_rejected_steps` with fixed-step controller
**Handling**:
- Array allocated but always zero
- No special handling needed
- Documentation should clarify applicability

### 3. Mixed Algorithm Batches
**Scenario**: Batch includes both explicit and implicit algorithms
**Handling**: Not currently supported by CuBIE (single algorithm per batch)
**Future**: If supported, iteration outputs valid only for implicit runs

### 4. Zero Saves (Summary-Only Output)
**Scenario**: `n_saves=0`, only summary outputs requested
**Handling**:
- Iteration output arrays have shape `(n_runs, 0)`
- No iteration data collected (no save calls)
- Consistent with current state/observable behavior

### 5. Memory Allocation Failures
**Scenario**: Iteration output arrays cause memory limit to be exceeded
**Handling**:
- Caught by existing memory manager chunking logic
- Batch automatically chunked to fit memory
- Same behavior as oversized state arrays

## Testing Strategy

### Unit Tests

1. **Linear Solver Iteration Count**
   - Test `linear_solver_factory()` returns iteration count correctly
   - Verify extraction: `(return_value >> 16) & 0xFFFF`
   - Test both convergence and max iterations scenarios

2. **Newton-Krylov Krylov Accumulation**
   - Test Newton solver accumulates Krylov iterations correctly
   - Verify total Krylov count across multiple Newton iterations
   - Test auxiliary buffer write (if Option B chosen)

3. **Save State Function**
   - Test compile-time branching (flags on/off)
   - Verify iteration values written to correct slices
   - Test all iteration output types independently

4. **Loop Iteration Tracking**
   - Test accumulation of Newton iterations between saves
   - Test accumulation of Krylov iterations between saves
   - Test step count accumulation
   - Test rejected step count (adaptive controller)
   - Verify reset after each save

5. **Output Configuration**
   - Test new output type recognition
   - Test flag creation from output types
   - Test invalid combinations (warn, don't error)

6. **Array Management**
   - Test iteration array allocation
   - Test host-device transfer
   - Test proper shapes and dtypes

### Integration Tests

1. **End-to-End Implicit Solver**
   - Run backward Euler with Newton iteration output enabled
   - Verify iteration counts are reasonable (e.g., 1-5 per step)
   - Compare against expected convergence behavior

2. **Adaptive Controller Step Tracking**
   - Run PI controller with step count and rejection tracking
   - Verify rejected_steps < total_steps
   - Verify step_sizes vary adaptively

3. **Multiple Output Types**
   - Enable all iteration outputs simultaneously
   - Verify no conflicts or memory issues
   - Check performance overhead

4. **Disabled vs. Enabled Performance**
   - Benchmark with all iteration outputs off (baseline)
   - Benchmark with all iteration outputs on
   - Verify overhead is minimal (<1-2%)

5. **Chunking with Iteration Outputs**
   - Force chunking with large batch + iteration outputs
   - Verify iteration arrays chunked correctly
   - Verify no data corruption across chunks

### Validation Tests

1. **Newton Convergence Theory**
   - For known problems, verify iteration counts match theoretical expectations
   - E.g., linear problem should converge in 1-2 iterations

2. **Krylov Iteration Counts**
   - Verify Krylov count >= Newton count (multiple Krylov per Newton)
   - Check scaling with system size

3. **Step Count Consistency**
   - For fixed-step: `step_count * dt ≈ dt_save` (within tolerance)
   - For adaptive: step_count varies based on error

4. **Cross-Platform Consistency**
   - Test on CUDA hardware and CUDASIM
   - Verify iteration counts identical (deterministic algorithms)

## Documentation Updates

### User-Facing Documentation

1. **Solver API Documentation**
   - Update `solve_ivp()` and `Solver` docstrings
   - Document new `output_types` options
   - Provide usage examples

2. **Output Types Reference**
   - Add section describing iteration output types
   - Explain what each output represents
   - Clarify applicability (implicit vs. explicit, adaptive vs. fixed)

3. **Tutorial/Examples**
   - Add example: "Tuning Implicit Solver Parameters Using Iteration Counts"
   - Show how to diagnose convergence issues
   - Demonstrate parameter optimization workflow

4. **FAQ**
   - "Why are my Newton iterations high?"
   - "What's a good Krylov iteration count?"
   - "How do I reduce iteration counts?"

### Developer Documentation

1. **Architecture Notes**
   - Update `cubie_internal_structure.md` with iteration output flow
   - Document status word encoding conventions
   - Explain auxiliary buffer pattern (if used)

2. **Implementation Notes**
   - Comment complex bit manipulation logic
   - Explain compile-time flag propagation
   - Document memory layout for iteration counters

## Implementation Order

### Phase 1: Foundation (Iteration Count Infrastructure)
1. Modify `linear_solver_factory()` to return iteration count
2. Add new output types to `OutputConfig`
3. Extend `OutputCompileFlags` with iteration flags
4. Update `ALL_OUTPUT_FUNCTION_PARAMETERS`

### Phase 2: Data Flow (Loop and Save)
5. Extend `save_state_factory()` signature and implementation
6. Modify `IVPLoop.build()` to track and pass iteration counts
7. Update `ODELoopConfig` for new parameters

### Phase 3: Algorithm Integration
8. Modify `newton_krylov_solver_factory()` to accumulate Krylov counts
9. Update algorithm step wrappers to pass iteration counts (if needed)
10. Test implicit algorithms with iteration tracking

### Phase 4: Output Management
11. Extend `OutputArrayContainer` with iteration arrays
12. Update `OutputArrays` allocation and transfer logic
13. Modify `BatchSolverKernel` to pass iteration arrays to loop
14. Extend `SolveResult` with iteration properties

### Phase 5: Testing and Validation
15. Write unit tests for each component
16. Write integration tests for end-to-end flow
17. Perform validation against theoretical expectations
18. Benchmark performance overhead

### Phase 6: Documentation
19. Update API documentation
20. Write user guide/tutorial
21. Update developer architecture docs

## Dependencies and Imports

### New Imports (Minimal)
- No new external dependencies
- Use existing: `numba`, `numpy`, `attrs`
- Internal dependencies already available

### Modified Imports
- Components using iteration outputs import updated `OutputConfig`
- Loop imports updated `save_state_factory`
- Solver imports updated `OutputArrays`

## Performance Considerations

### Compile-Time Optimization
- All iteration tracking guarded by compile-time flags
- Disabled features compiled away entirely
- Zero overhead when not used

### Runtime Cost (When Enabled)
- Iteration count extraction: bitwise operations (negligible)
- Accumulation: integer addition per step (negligible)
- Array writes on save: one int32 write per iteration type (~4-20 bytes)
- **Expected overhead**: <1% when all iteration outputs enabled

### Memory Cost
- Per-thread: 16 bytes (4 × int32 counters)
- Output arrays: `n_runs × n_saves × 4 bytes` per iteration type
- Example: 10k runs, 1k saves, 5 iteration types = 200 MB

### CUDA Considerations
- Integer operations highly efficient on GPU
- No warp divergence introduced (all threads track iterations)
- No shared memory contention (local counters or direct arrays)

## Potential Future Enhancements

### 1. Per-Newton-Iteration Krylov Counts
- Output array of Krylov counts per Newton iteration
- More detailed diagnostic, but variable-size output
- Defer to future feature request

### 2. Convergence Rate Metrics
- Calculate and output convergence rates
- Useful for research, but adds complexity
- Consider as separate feature

### 3. Timestep History
- Dense array of all timesteps taken
- Memory-intensive, conflicts with batching
- Probably not feasible for CuBIE's use case

### 4. Algorithm-Specific Diagnostics
- Tableau-specific metrics (stage counts, etc.)
- Requires per-algorithm customization
- Defer pending user demand

### 5. Real-Time Iteration Monitoring
- Stream iteration counts during execution
- Requires async kernel execution
- Advanced feature for future consideration

## Risks and Mitigation

### Risk 1: Signature Complexity
**Risk**: Too many parameters to `save_state` and loop functions  
**Mitigation**: Use compile-time selection to omit unused parameters, or auxiliary structures

### Risk 2: Bit Packing Confusion
**Risk**: Complex bit manipulation leads to bugs  
**Mitigation**: Extensive testing, clear documentation, helper macros/functions

### Risk 3: Memory Overhead
**Risk**: Iteration arrays too large for memory-constrained scenarios  
**Mitigation**: Chunking handles this automatically, consistent with existing outputs

### Risk 4: Performance Regression
**Risk**: Iteration tracking slows down even when disabled  
**Mitigation**: Compile-time flags ensure zero overhead, benchmark tests verify

### Risk 5: Incomplete Krylov Tracking
**Risk**: Krylov counts not propagated correctly from linear solver  
**Mitigation**: Careful design of propagation mechanism, thorough testing

## Summary

This feature adds comprehensive iteration count outputs to CuBIE following established patterns:
- **Compile-time flags** control feature activation
- **Status word encoding** carries iteration counts from solvers
- **Accumulation in loop** tracks iterations between saves
- **Extended save_state** writes counts to output arrays
- **Zero overhead** when disabled via compile-time branching

The implementation is modular, testable, and maintains compatibility with existing code. Users gain powerful diagnostic capabilities for tuning implicit solvers, while the architecture remains clean and performant.
