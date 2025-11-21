# Loop Termination and Save Counting Fix - Agent Plan

## Architectural Overview

This plan addresses two interconnected issues in the integration loop:
1. Incorrect save count calculation that omits the final state at t_end
2. Lack of centralized save count logic leading to inconsistent calculations

The solution involves adding +1 to save counts and centralizing the logic in a single authoritative location.

## Component Analysis

### IVPLoop (src/cubie/integrators/loops/ode_loop.py)

**Current Behavior:**
- Loop terminates when `save_idx >= n_output_samples`
- `n_output_samples` comes from max(state_output.shape[0], observables_output.shape[0])
- Array shapes determined upstream by BatchSolverKernel.output_length
- No internal calculation of expected saves

**Required Changes:**
- Add property/method to calculate expected number of saves
- Formula: `ceil(duration / dt_save) + 1`
- Use float64 for duration, precision for dt_save (consistent with time precision fix)
- Property should accept duration and dt_save as parameters, or use compile_settings

**Integration Points:**
- Called by BatchSolverKernel.output_length
- Used by output_sizes.py for array sizing
- Could be static method or instance property depending on design choice

### BatchSolverKernel (src/cubie/batchsolving/BatchSolverKernel.py)

**Current Behavior (line 888):**
```python
def output_length(self) -> int:
    """Number of saved trajectory samples in the main run."""
    return int(np.ceil(self.duration / self.single_integrator.dt_save))
```

**Required Changes:**
- Update to: `int(np.ceil(self.duration / self.single_integrator.dt_save)) + 1`
- OR delegate to IVPLoop calculation method
- Maintain return type as int
- Keep using np.float64(self.duration) from property

**Dependencies:**
- Called by output_sizes.py via solver_instance.output_length
- Used for chunking calculations
- Affects array allocation sizes

**Testing Implications:**
- All tests that check output array sizes will need adjustment
- Chunking logic should handle +1 correctly
- warmup_length may need similar treatment (investigate separately)

### output_sizes.py (src/cubie/outputhandling/output_sizes.py)

**Current Behavior (lines 401-403):**
```python
output_samples = int(
    ceil(run_settings.duration / run_settings.dt_save)
)
```

**Required Changes:**
- Update to: `int(ceil(run_settings.duration / run_settings.dt_save)) + 1`
- OR delegate to IVPLoop/config calculation
- Ensure consistency with BatchSolverKernel.output_length

**Affected Classes:**
- `SingleRunOutputSizes.from_output_fns_and_run_settings`
- `BatchOutputSizes.from_solver` (uses SingleRunOutputSizes)

**Testing Implications:**
- Tests using these sizing classes will need adjustment
- Verify array allocation matches actual output

### ODELoopConfig (src/cubie/integrators/loops/ode_loop_config.py)

**Potential Addition:**
- Could add static method or property for save count calculation
- Would provide centralized logic accessible before IVPLoop instantiation
- Cleaner separation of concerns: config holds calculation, IVPLoop uses it

**Design Choice:**
Option A: Add to ODELoopConfig as static method
```python
@staticmethod
def calculate_n_saves(duration: float, dt_save: float) -> int:
    """Calculate number of saves including initial and final states."""
    return int(np.ceil(np.float64(duration) / dt_save)) + 1
```

Option B: Add to IVPLoop as property
```python
@property
def n_saves(self) -> int:
    """Expected number of saves for configured duration and dt_save."""
    # Requires duration to be passed or stored
    return int(np.ceil(np.float64(duration) / self.dt_save)) + 1
```

**Recommendation:** Option A (ODELoopConfig static method) for better separation and reusability

## Data Flow Changes

### Before Fix:
```
User specifies duration=1.0, dt_save=0.1
  ↓
BatchSolverKernel.output_length: ceil(1.0/0.1) = 10
  ↓
OutputArrays allocated with shape (10, ...)
  ↓
IVPLoop receives n_output_samples=10
  ↓
Loop saves at: t=0.0, 0.1, 0.2, ..., 0.9 (10 samples)
  ↓
Loop exits when save_idx >= 10
  ↓
PROBLEM: Final state at t=1.0 not saved
```

### After Fix:
```
User specifies duration=1.0, dt_save=0.1
  ↓
BatchSolverKernel.output_length: ceil(1.0/0.1) + 1 = 11
  ↓
OutputArrays allocated with shape (11, ...)
  ↓
IVPLoop receives n_output_samples=11
  ↓
Loop saves at: t=0.0, 0.1, 0.2, ..., 0.9, 1.0 (11 samples)
  ↓
Loop exits when save_idx >= 11
  ↓
CORRECT: Final state at t=1.0 saved
```

## Expected Behavior

### Save Point Calculation

**Fixed-Step Loop:**
- Steps taken: floor(duration / dt0)
- Saves occur every steps_per_save steps
- Final save at exactly t=t0+duration

**Adaptive Loop:**
- Steps taken: variable (depends on controller)
- Saves occur when t + dt >= next_save
- Final save when t >= t_end - epsilon

**With Settling Time:**
- No save at t=t0
- First save at t=settling_time (or first dt_save after settling_time)
- Last save at t=settling_time+duration
- Number of saves: `ceil(duration/dt_save) + 1`

**Without Settling Time:**
- First save at t=t0
- Last save at t=t0+duration
- Number of saves: `ceil(duration/dt_save) + 1`

### Loop Termination Logic

**Current (line 427):**
```python
finished = save_idx >= n_output_samples
```

**Remains Unchanged:**
- Logic is correct; problem is in n_output_samples calculation
- With corrected n_output_samples (using +1), termination will be correct

**Interaction with Time Checks:**
- Adaptive: `do_save = (t + dt[0] + equality_breaker) >= next_save`
- Fixed: `do_save = (step_counter % steps_per_save) == 0`
- Both should trigger final save correctly with updated n_output_samples

## Edge Cases to Handle

### Case 1: Exact Multiple
- duration=1.0, dt_save=0.1
- Expected: ceil(1.0/0.1) + 1 = 10 + 1 = 11 saves
- Times: 0.0, 0.1, 0.2, ..., 0.9, 1.0

### Case 2: Near-Integer Multiple (Rounding Up)
- duration=1.0 + 1e-10, dt_save=0.1
- Expected: ceil(10.00000001) + 1 = 11 + 1 = 12 saves
- Behavior: One extra sample beyond t=1.0 (acceptable trade-off)

### Case 3: Near-Integer Multiple (Rounding Down)
- duration=1.0 - 1e-10, dt_save=0.1
- Expected: ceil(9.99999999) + 1 = 10 + 1 = 11 saves
- Times: 0.0, 0.1, 0.2, ..., 0.9, ~1.0

### Case 4: With Settling Time
- settling_time=0.5, duration=1.0, dt_save=0.1
- Expected: ceil(1.0/0.1) + 1 = 11 saves
- Times: 0.5, 0.6, 0.7, ..., 1.4, 1.5 (no save at t=0.0)

### Case 5: Very Small dt_save
- duration=1.0, dt_save=0.001
- Expected: ceil(1.0/0.001) + 1 = 1000 + 1 = 1001 saves
- Final save at exactly t=1.0

### Case 6: Non-divisible Duration
- duration=1.23, dt_save=0.1
- Expected: ceil(1.23/0.1) + 1 = ceil(12.3) + 1 = 13 + 1 = 14 saves
- Times: 0.0, 0.1, 0.2, ..., 1.2, 1.23 (final save at t_end)

## Precision Considerations

### Float64 vs User Precision
Following the time precision fix architecture:
- **duration**: Always float64 (as per recent fix)
- **dt_save**: User precision (interval parameter)
- **Division**: `float64(duration) / precision(dt_save)` → float64 result
- **ceil()**: Operates on float64 → float64 result
- **int()**: Cast to int → int result

### Rounding Robustness
The ceiling function inherently handles floating-point imprecision:
- ceil(9.99999999) = 10 (rounds up from near-integer)
- ceil(10.0) = 10 (exact integer)
- ceil(10.00000001) = 11 (rounds up from just-above-integer)

Adding +1 after ceiling ensures:
- Always have space for final save
- Robust to floating-point representation issues
- Small cost: Occasionally one extra allocated sample

## Integration Dependencies

### Components That Calculate Save Count
1. **BatchSolverKernel.output_length** ← Must update
2. **output_sizes.SingleRunOutputSizes.from_output_fns_and_run_settings** ← Must update
3. **Tests that check output shapes** ← Must adjust expectations

### Components That Use Save Count
1. **OutputArrays allocation** ← Automatically uses updated output_length
2. **IVPLoop termination** ← Automatically uses updated n_output_samples from array shapes
3. **Chunking calculations** ← Uses output_length, should work correctly

### Components Not Affected
1. **Algorithm step functions** ← No changes needed
2. **Controllers** ← No changes needed
3. **Output functions** ← No changes needed
4. **Summary metrics** ← May benefit from additional save (better final-state metrics)

## Testing Strategy

### Unit Tests Needed

**Test 1: Save Count Calculation**
- Test ODELoopConfig.calculate_n_saves (or equivalent)
- Verify ceil(duration/dt_save) + 1 for various inputs
- Edge cases: exact multiples, near-integers, small dt_save

**Test 2: Final State Saved**
- Integration with duration=1.0, dt_save=0.1
- Verify 11 saves produced
- Verify last save time equals 1.0 (within epsilon)

**Test 3: With Settling Time**
- settling_time=0.5, duration=1.0, dt_save=0.1
- Verify 11 saves produced
- Verify first save at 0.5, last at 1.5

**Test 4: Array Sizing Consistency**
- Verify BatchSolverKernel.output_length matches actual saves
- Verify no buffer overruns
- Verify no wasted allocations

**Test 5: Fixed-Step Loop**
- Ensure fixed-step counter logic works with +1
- Verify steps_per_save calculation handles endpoint

### Integration Tests Needed

**Test 6: End-to-End Integration**
- Run solve_ivp with various duration/dt_save combinations
- Verify final output time equals t0+duration
- Verify output array sizes match allocated sizes

**Test 7: Chunked Execution**
- Test with chunking enabled (chunk_axis="time")
- Verify final state saved correctly across chunks
- Verify no double-saves at chunk boundaries

### Regression Tests

**All existing tests** must be updated to expect +1 additional sample:
- Adjust assertions on output array shapes
- Adjust assertions on number of saves
- Verify tests still validate correctness (not just shapes)

## Implementation Sequence

### Phase 1: Add Centralized Calculation
1. Add static method to ODELoopConfig (or IVPLoop)
2. Implement: `int(ceil(float64(duration) / dt_save)) + 1`
3. Add docstring explaining the +1 for endpoints
4. Add unit tests for the calculation

### Phase 2: Update BatchSolverKernel
1. Modify output_length property to use +1
2. Update docstring to reflect new behavior
3. Verify warmup_length and summaries_length don't need updates

### Phase 3: Update output_sizes.py
1. Update SingleRunOutputSizes.from_output_fns_and_run_settings
2. Use +1 in output_samples calculation
3. Verify BatchOutputSizes correctly uses updated SingleRunOutputSizes

### Phase 4: Update Tests
1. Identify all tests checking output shapes
2. Update expected shapes to include +1 sample
3. Add new tests for final state saving
4. Verify all edge cases covered

### Phase 5: Documentation
1. Update docstrings in affected methods
2. Add comment explaining +1 rationale
3. Update any architectural documentation

## Success Criteria

### Functional Correctness
- ✅ Final state at t=t_end always saved
- ✅ Save count calculation predictable across edge cases
- ✅ No buffer overruns or undersized allocations
- ✅ Loop terminates correctly at t=t_end

### Code Quality
- ✅ Single source of truth for save count calculation
- ✅ No duplicate logic in multiple files
- ✅ Clear documentation of +1 rationale
- ✅ Consistent use of float64 for duration

### Testing Coverage
- ✅ Unit tests for save count calculation
- ✅ Integration tests for final state saving
- ✅ Edge case tests for rounding scenarios
- ✅ Regression tests all passing with updated expectations

### Performance
- ✅ No performance regressions
- ✅ Minimal memory overhead (+1 sample per run)
- ✅ CUDA kernel efficiency unchanged

## Backward Compatibility

### Breaking Changes
- Output arrays will have one additional sample
- Users expecting N samples will now get N+1 samples
- This is technically a breaking change but improves correctness

### Migration Path
- Update documentation to explain new behavior
- Users should verify final time value in output arrays
- No code changes required in user applications
- Benefits: More accurate final-state analysis

### API Stability
- No changes to function signatures
- No changes to solve_ivp interface
- Internal implementation detail
- Properties and methods remain accessible

## Alternatives Considered

### Alternative 1: Don't Add +1, Use Floor Instead
- **Approach**: Use floor(duration/dt_save) and explicitly save at t_end
- **Pros**: Fewer saves in some cases
- **Cons**: Complex logic, needs explicit final-save check, harder to reason about
- **Rejected**: Ceiling + 1 is simpler and more robust

### Alternative 2: Track Target Time Instead of Save Count
- **Approach**: Loop until t >= t_end instead of save_idx >= n_output_samples
- **Pros**: More direct correspondence to integration goal
- **Cons**: Requires pre-allocating array of unknown size, wastes memory
- **Rejected**: Current approach (pre-sized arrays) is more GPU-friendly

### Alternative 3: Use Floating-Point Epsilon for Final Save
- **Approach**: Save when t >= t_end - epsilon
- **Pros**: Handles near-t_end situations
- **Cons**: Epsilon choice is arbitrary, can still miss final state
- **Rejected**: +1 approach is deterministic and doesn't rely on epsilon tuning

## Known Limitations

### Potential Over-Allocation
- When duration is a near-integer multiple of dt_save slightly above an integer
- Example: duration=1.0+1e-10, dt_save=0.1 → 12 saves instead of 11
- Impact: One extra unused sample per run (negligible memory cost)
- Mitigation: Acceptable trade-off for correctness guarantee

### Interaction with Summaries
- summaries_length also uses ceiling, may need similar treatment
- If summary intervals don't align with save intervals, could have mismatch
- Recommendation: Investigate summaries_length in separate task if needed

### Chunking Edge Cases
- When chunking by time, final chunk may have fewer saves than expected
- Chunk boundary logic should handle +1 correctly
- Requires careful testing of chunk_run method

## Questions for Reviewer

1. Should the centralized calculation be in ODELoopConfig or IVPLoop?
2. Does warmup_length need the same +1 treatment?
3. Does summaries_length need the same +1 treatment?
4. Should we add architectural documentation of the endpoints inclusion principle?
5. Are there other locations that calculate save counts we haven't identified?
