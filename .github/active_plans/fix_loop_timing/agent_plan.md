# Agent Plan: Fix Loop Timing and Precision Setting Issues

## Component Overview

This plan addresses fundamental flaws in how CuBIE calculates output array sizes and controls loop timing. The changes span multiple components but follow a consistent pattern: replace step-counting and rounded calculations with time-based comparisons and floor-based sizing.

## Component 1: BatchSolverKernel Output Length Calculation

**Location:** `src/cubie/batchsolving/BatchSolverKernel.py`

**Current Behavior:**
- Line 891: `output_length` property calculates `int(np.round(self.duration / self.single_integrator.dt_save)) + 1`
- This rounding can cause off-by-one errors depending on floating-point representation
- The `+1` is intended to capture both initial state and final state

**Expected New Behavior:**
- Change calculation to `int(np.floor(self.duration / self.single_integrator.dt_save)) + 1`
- `floor()` ensures conservative sizing - never underestimate
- Loop will save at t0, then every dt_save until t > t_end
- Array sized to hold all these saves without risk of overflow

**Related Properties to Review:**
- `warmup_length` (line 917): Uses `np.ceil()` - likely correct since warmup doesn't need +1 (see docstring)
- `summaries_length` (line 904): Uses `np.ceil()` - correct, as summaries represent intervals not snapshots

**Integration Points:**
- Used by `BatchOutputArrays` to size state_output and observables_output arrays
- Used by `BatchSolverKernel._get_chunk_params()` when chunking by time
- Accessed via `Solver.output_length` property

**Edge Cases:**
- When `duration / dt_save` is exactly an integer: floor gives exact count, +1 for initial point
- When `duration / dt_save` has fractional part: floor rounds down, +1 ensures final save included
- Very small dt_save relative to duration: ensure array size fits in memory and int32 range

---

## Component 2: IVPLoop Fixed-Step Save Logic Removal

**Location:** `src/cubie/integrators/loops/ode_loop.py`

**Current Behavior:**
- Line 209: `steps_per_save = int32(ceil(precision(dt_save) / precision(dt0)))`
- Line 421: `step_counter` initialized for fixed mode
- Line 440: Fixed-step save logic: `do_save = (step_counter % steps_per_save) == 0`
- Lines 441-442: Reset counter after save
- Line 438: Increment step counter

**Expected New Behavior:**
- Remove `steps_per_save` calculation (line 209)
- Remove `step_counter` variable and all references
- Fixed-step mode should use same logic as adaptive: `do_save = (t + dt[0]) >= next_save`
- When `do_save` is True in fixed mode:
  - Set `dt_eff = precision(next_save - t)` (reduced step)
  - Step algorithm runs with reduced dt
  - After successful step (implicit accept in fixed mode): update `next_save += dt_save`

**Loop Structure After Changes:**
```python
# Pseudocode for main loop
for _ in range(max_steps):
    finished = t > t_end
    if all_sync(mask, finished):
        return status
    
    if not finished:
        # Unified save logic for both adaptive and fixed
        do_save = (t + dt[0]) >= next_save
        dt_eff = selp(do_save, precision(next_save - t), dt[0])
        
        # Fixed mode has no additional logic here
        # Adaptive mode continues as before
        
        step_status = step_function(...)
        
        if not fixed_mode:
            step_controller(...)
            accept = accept_step[0] != int32(0)
        else:
            accept = True  # Fixed mode auto-accepts
        
        # Update time and state with predication
        t = selp(accept, t + dt_eff, t)
        # ... update buffers ...
        
        # Update next_save and save outputs
        do_save = accept and do_save
        next_save = selp(do_save, next_save + dt_save, next_save)
        
        if do_save:
            save_state(...)
            save_idx += 1
```

**Integration Points:**
- Interacts with `step_function` (algorithm step implementations)
- Interacts with `step_controller` (adaptive controllers only)
- Uses `save_state` device function from OutputFunctions
- Updates `next_save` timing variable

**Edge Cases:**
- When dt (fixed step size) is larger than dt_save: will take multiple reduced steps in a row
- When dt equals dt_save exactly: no reduced steps needed
- When t approaches t_end: may need reduced final step to exactly hit t_end (separate from save logic)
- Warp divergence: predicated updates ensure correct execution even when threads diverge

---

## Component 3: CPU Reference Loop Implementation

**Location:** `tests/integrators/cpu_reference/loops.py`

**Current Behavior:**
- Line 123: `max_save_samples = int(np.round(...)) + 1`
- Line 150: `fixed_steps_per_save = int(np.ceil(dt_save / controller.dt))`
- Line 164-168: Fixed-step save logic using modulo counting
- Line 151: `fixed_step_count` initialized

**Expected New Behavior:**
- Line 123: Change to `max_save_samples = int(np.floor(...)) + 1`
- Remove `fixed_steps_per_save` calculation (line 150)
- Remove `fixed_step_count` variable (line 151)
- Unify save logic for adaptive and fixed controllers:
  ```python
  # Both modes use same check
  do_save = (t + dt >= next_save_time)
  if do_save:
      dt = precision(next_save_time - t)
  ```
- Fixed controllers accept reduced steps just like adaptive controllers do before saves

**Function Signature:**
No changes to `run_reference_loop()` function signature.

**Integration Points:**
- Called by test fixtures in `conftest.py`
- Outputs compared against GPU loop results via `assert_integration_outputs`
- Uses `CPUAdaptiveController` which has `dt` property and `is_adaptive` flag

**Data Structures:**
- `state_history`: list of saved state snapshots
- `observable_history`: list of saved observable snapshots
- `time_history`: list of saved time values
- All should have length equal to `max_save_samples`

**Edge Cases:**
- Warmup phase: don't save during warmup, but track `next_save_time` correctly
- Initial save: if no warmup, save at t0; otherwise skip to end of warmup
- Controller step size changes: in adaptive mode, controller may propose different dt after each step

---

## Component 4: Test Utility Functions

**Location:** `tests/_utils.py`

**Current Behavior:**
- Line 605: `save_samples = int(np.round(duration / precision(dt_save))) + 1`
- Line 620: `summary_samples = int(np.ceil(duration / summarise_dt))`

**Expected New Behavior:**
- Line 605: Change to `save_samples = int(np.floor(duration / precision(dt_save))) + 1`
- Line 620: Keep as-is - summaries use different semantics (intervals not points)

**Function:** `run_device_loop()`

**Integration Points:**
- Used by tests to execute loops in CUDA simulator mode
- Must size arrays consistently with `BatchSolverKernel.output_length`
- Provides arrays to `loop_fn` device function

**Dependencies:**
- `OutputArrayHeights.from_output_fns(output_functions)` for width calculations
- `loop.precision` for dtype
- `loop.dt_save` and `loop.dt_summarise` for timing

---

## Component 5: Example Code Updates

**Location:** `docs/source/examples/controller_step_analysis.py`

**Current Behavior:**
- Uses `fixed_steps_per_save` pattern in example code
- May mislead users about proper implementation

**Expected New Behavior:**
- Update example to show time-based comparison approach
- Remove step-counting logic from examples
- Ensure examples demonstrate correct API usage

**Note:** This is documentation, not production code, but should reflect best practices.

---

## Architectural Changes Required

### Memory Layout
No changes to memory layout or array structures. Only the sizing calculation changes.

### Control Flow
Significant simplification of loop control flow:
- Remove conditional branching between fixed and adaptive save logic
- Reduce to single unified path with predicated updates
- May improve warp efficiency by reducing divergence

### Compilation Dependencies
No new compile-time dependencies. The `dt_save` value is already captured in closures.

### Data Structures
No new data structures. Existing buffers and variables reused.

---

## Expected Component Interactions

### Initialization Phase
1. `BatchSolverKernel` calculates `output_length` using floor + 1
2. `BatchOutputArrays` allocates arrays of size `output_length`
3. `IVPLoop.build()` compiles loop function with dt_save captured in closure
4. Loop function initializes `next_save = t0` (or `t0 + warmup`)

### Integration Phase
1. Loop checks `t > t_end` for exit condition
2. Loop checks `t + dt >= next_save` for save condition
3. If saving, reduces `dt_eff = next_save - t`
4. Algorithm step executes with `dt_eff`
5. Controller (if adaptive) proposes new dt for next step
6. If step accepted and `do_save`: increment `next_save += dt_save`, save outputs
7. Repeat until `t > t_end`

### Validation Phase
1. GPU loop produces `state_output[save_idx, :]` for save_idx in range(output_length)
2. CPU reference produces matching outputs
3. `assert_integration_outputs` compares timestamps, states, observables

---

## Dependencies and Imports

**No new imports required.** All necessary functions already imported:
- `numpy.floor` (standard numpy)
- `math.ceil` already imported in ode_loop.py (keep for max_steps calculation)
- `cuda.jit`, `selp`, `activemask`, `all_sync` already imported

---

## Edge Cases to Consider

### Numerical Precision
- **Issue:** `t + dt` may be slightly less than `next_save` due to floating-point rounding
- **Mitigation:** Use `>=` comparison, not `==`
- **Existing Code:** Already uses `>=` in adaptive path (line 451)

### Very Small dt_save
- **Issue:** If dt_save approaches machine epsilon, floor calculation may overflow int32
- **Mitigation:** Validate `dt_save > epsilon` in solver configuration
- **Note:** Existing code already has practical limits on dt_save

### Very Large Duration
- **Issue:** `floor(duration / dt_save)` could exceed int32 max value
- **Mitigation:** Memory manager already enforces practical limits via VRAM constraints
- **Note:** Array allocation would fail before integer overflow

### Warmup Phase Timing
- **Issue:** If warmup > 0, first save should be after warmup completes
- **Current Implementation:** Lines 373-376 handle this correctly
- **Changes Needed:** None - this logic is already correct

### Summary Intervals
- **Issue:** Summaries have different semantics - they aggregate over intervals
- **Current Implementation:** Uses `ceil()` without +1, which is correct
- **Changes Needed:** None - summaries unaffected by this fix

### Fixed-Step dt Larger Than dt_save
- **Issue:** If fixed dt > dt_save, multiple consecutive reduced steps needed
- **Behavior:** Each step will reduce to next_save, take step, save, increment next_save
- **Correctness:** Loop will correctly save at each interval despite large dt

### Final Step Timing
- **Issue:** Last step may need reduction to avoid overshooting t_end
- **Current Implementation:** Separate from save logic; loop exits when `t > t_end`
- **Behavior:** May take final step that slightly overshoots, but this is acceptable
- **Alternative:** Could add explicit `dt_eff = min(dt_eff, t_end - t)` check

---

## Testing Strategy

### Unit Tests to Update
- `tests/integrators/loops/test_ode_loop.py`: Parametrized tests for all algorithms and controllers
- Verify output length matches expected `floor(duration/dt_save) + 1`
- Verify save times align with expected intervals

### Integration Tests to Update
- `tests/batchsolving/test_solver.py`: End-to-end solver tests
- Verify CPU reference matches GPU outputs
- Verify chunking works correctly with new sizing

### Regression Tests
- Run full test suite to ensure no unexpected breakage
- Pay special attention to timing-sensitive tests
- Verify fixed-step controllers still produce deterministic results

### Test Data
- May need to regenerate golden outputs if timing changes affect results
- Document any changes in test expectations

---

## Implementation Order

Recommended order to minimize test failures during development:

1. **Update BatchSolverKernel.output_length** - Changes array sizing
2. **Update tests/_utils.py save_samples** - Aligns test utilities with new sizing
3. **Update cpu_reference/loops.py** - Makes CPU reference match new logic
4. **Update IVPLoop fixed-step logic** - Implements unified save logic in GPU code
5. **Update examples** - Documentation reflects new approach
6. **Run full test suite** - Validate all changes together

This order ensures that array sizing changes propagate before control flow changes, reducing the risk of array indexing errors during development.

---

## Performance Considerations

### GPU Performance
- **Reduced Divergence:** Unified save logic may improve warp efficiency
- **Reduced Steps:** Occasional reduced steps in fixed mode are negligible overhead
- **Memory Access:** No change to memory access patterns

### Compilation Time
- **No Impact:** Loop structure similar complexity, compilation time unchanged

### Runtime Performance
- **Negligible Impact:** Time-based comparison vs step counting both O(1)
- **Slight Improvement:** Less branching may help instruction cache

---

## Validation Criteria

The implementation is correct when:

1. **Array Sizing:** All arrays sized with `floor() + 1` calculation
2. **Loop Exit:** All loops exit based on `t > t_end` condition
3. **Save Timing:** All saves triggered by `t + dt >= next_save` comparison
4. **Step Counting Removed:** No references to `steps_per_save` or `fixed_steps_per_save` in loop control
5. **CPU/GPU Match:** CPU reference outputs match GPU outputs within tolerance
6. **Tests Pass:** All existing tests pass with new logic
7. **Examples Updated:** Documentation code examples use correct patterns
