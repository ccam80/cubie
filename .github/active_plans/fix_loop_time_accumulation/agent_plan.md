# Agent Plan: Fix Loop Time Accumulation Precision Mismatch

## Problem Statement

The GPU integration loop has a type mismatch in the `next_save` accumulation at line 529 of `src/cubie/integrators/loops/ode_loop.py`. The variable `next_save` is `float64`, but `dt_save` is cast to `precision` type, causing mixed-precision arithmetic that diverges from the CPU reference implementation.

## Architectural Context

### Current Architecture

**File**: `src/cubie/integrators/loops/ode_loop.py`

**Key Variables**:
- `t`: Current integration time (float64)
- `t_end`: End time for integration (float64)
- `next_save`: Next scheduled save time (float64, implicitly)
- `dt_save`: Save interval (precision type after cast on line 206)
- `precision`: User-specified precision (float16/32/64)

**Type Handling Rules** (from `.github/context/cubie_internal_structure.md`):
1. Time accrues in float64
2. Intervals (dt_save, dt_summarise) stay in "precision" type
3. Cast accrued time back to precision before doing math with it

### CPU Reference Architecture

**File**: `tests/integrators/cpu_reference/loops.py`

**Key Variables**:
- `t`: Current integration time (float64)
- `end_time`: End time for integration (float64)
- `next_save_time`: Next scheduled save time (float64)
- `dt_save`: Save interval (precision type, line 91)

**Critical Line** (line 188):
```python
next_save_time = next_save_time + np.float64(dt_save)
```

## Required Changes

### Change Location

**File**: `src/cubie/integrators/loops/ode_loop.py`  
**Line**: 529  
**Current Code**:
```python
next_save = selp(do_save, next_save + dt_save, next_save)
```

**Required Change**:
```python
next_save = selp(do_save, next_save + float64(dt_save), next_save)
```

### Rationale

1. **Type Consistency**: `next_save` is `float64`, so accumulation should use `float64` arithmetic
2. **CPU Reference Match**: Exactly matches line 188 of CPU reference implementation
3. **Precision Rules**: Follows documented rule that time accrues in float64
4. **Minimal Impact**: Only affects the accumulation line, no other changes needed

## Integration Points

### Affected Components

1. **IVPLoop.build() method**: Contains the loop_fn device function where the change occurs
2. **Compile settings**: No changes needed; `dt_save` remains in compile settings as-is
3. **Output functions**: No changes needed; still receive `precision(t)` as expected
4. **Save logic**: No changes needed; `do_save` comparison logic remains unchanged

### Dependencies

**Imports**: Already available
- `float64` is imported from `numba` at the top of the file (line 13)

**No new dependencies required**

### Expected Interactions

1. **Loop exit condition** (line 430): Uses `next_save > t_end`, both float64, no change needed
2. **Save check** (line 445): Uses `(t + dt[0]) >= next_save`, comparison remains valid
3. **dt_eff calculation** (line 446): Uses `precision(next_save - t)`, cast to precision as required
4. **save_state call** (line 536): Receives `precision(t)`, not `next_save`, no change needed

## Data Structures

### Variable Types

Before change:
```python
next_save: float64 (implicit)
dt_save: precision (explicit cast line 206)
next_save + dt_save: mixed precision → float64 (with different rounding)
```

After change:
```python
next_save: float64 (implicit)
dt_save: precision (explicit cast line 206)
float64(dt_save): float64 (explicit cast in operation)
next_save + float64(dt_save): float64 (consistent arithmetic)
```

### No Buffer Changes

No shared memory, local memory, or output buffer changes required. The change is purely computational within the loop logic.

## Edge Cases

### Edge Case 1: Precision is float64

**Scenario**: User specifies `precision=np.float64`

**Behavior**: `float64(dt_save)` is a no-op cast, identical to current behavior

**Expected**: No difference in performance or results

### Edge Case 2: Precision is float32 or float16

**Scenario**: User specifies `precision=np.float32` or `precision=np.float16`

**Behavior**: `dt_save` is upcast from lower precision to float64 for accumulation

**Expected**: More accurate time accumulation, matching CPU reference

**Impact**: This is the fix - currently the mixed-precision arithmetic differs from CPU

### Edge Case 3: dt_save not exactly representable

**Scenario**: User provides `dt_save=0.1` in float32

**Behavior**: 
- `dt_save` stored as float32 approximation of 0.1
- Upcast to float64 for accumulation
- Each accumulation adds the float64 representation of the float32 value

**Expected**: Consistent rounding behavior with CPU reference

**Note**: Related to Issue #153, but not fully solved by this change alone

### Edge Case 4: Very long integrations

**Scenario**: Many accumulations of `next_save`

**Behavior**: Accumulation always in float64, matching CPU reference

**Expected**: Identical accumulated error patterns between GPU and CPU

### Edge Case 5: Save at exactly t_end

**Scenario**: `next_save` becomes exactly equal to `t_end`

**Behavior**: Loop exit condition `next_save > t_end` evaluates to False

**Expected**: One more iteration to handle the final save (if `save_last` is True)

**Note**: Existing `save_last` logic handles this correctly (lines 431-436)

## Testing Considerations

### Affected Tests

**Primary**: `tests/batchsolving/test_SolverKernel.py::test_run`
- Currently failing with state mismatch
- Should pass after fix
- Parametrized with "smoke_test" and "fire_test" scenarios

**Related**: Any test that compares GPU results to CPU reference
- Tests in `tests/integrators/loops/test_ode_loop.py`
- Tests using `assert_integration_outputs` utility

### Test Expectations

**Before Fix**:
- Time arrays: Match ✓
- State arrays: 75% mismatch (54/72 elements)
- Max absolute difference: 0.00957394
- Max relative difference: 0.01414536

**After Fix**:
- Time arrays: Match ✓ (unchanged)
- State arrays: Match ✓ (within tolerance)
- Max absolute difference: < 0.001
- Max relative difference: < 0.01

### Validation Strategy

1. Run `test_SolverKernel.py::test_run` to verify fix
2. Run full test suite for `tests/integrators/loops/` to ensure no regressions
3. Check that precision parameter works correctly for float16, float32, float64
4. Verify fixed-step and adaptive-step algorithms both work correctly

## Implementation Notes

### Numba CUDA Device Functions

The change occurs inside a Numba CUDA device function (the `loop_fn` function starting at line 245). Key considerations:

1. **Type casting**: `float64()` is a Numba-supported operation in device code
2. **Performance**: Cast is performed only when `do_save` is True, which is infrequent relative to integration steps
3. **Compilation**: No compilation settings changes needed; cast is inline in device code

### Closure Capture

The change does not affect any closure-captured variables. The `dt_save` variable is captured from the compile settings, but the cast occurs at runtime in the device code, not at compile time.

### Memory Access Patterns

No changes to memory access patterns:
- `next_save` is a scalar register variable
- `dt_save` is a scalar from closure (or register after compile-time evaluation)
- Cast operation is register-to-register

## Related Issues and Context

### Issue #153
Title: "bug: save cadence, fixed-step timing potentially incorrect"

**Relevant Quote**: "there must be a slight mismatch in CPU and device clamping to the 'next save' time"

**Relation**: This fix addresses the "slight mismatch" by ensuring type-consistent accumulation

**Not Fully Resolved**: Issue #153 also discusses non-representable dt_save values, which requires a broader solution involving warning users about precision limits

### Recent Commit d5b0250
Title: "fix: match loop exit condition in cpu reference"

**Relation**: Changed CPU reference to match GPU loop exit condition (`next_save_time < end_time` → consistent with GPU's `next_save > t_end`)

**Implication**: Exit condition logic is now aligned; this fix completes the alignment by ensuring time accumulation is also consistent

## Architecture Compliance

### CUDAFactory Pattern
- Change occurs in `IVPLoop.build()` method
- No changes to compile settings or cache invalidation logic
- Device function is rebuilt with new logic automatically

### Precision System
- Follows rules from `src/cubie/_utils.py`
- Time accumulation in float64 ✓
- Intervals in precision type ✓
- Cast to precision before device function calls ✓

### Comment Style
No new comments needed; the change is self-documenting. If a comment were added, it should describe the current behavior, not explain the change:

**Bad**: "Changed to float64 to fix precision mismatch"
**Good**: "Accumulate in float64 for consistent time representation"

However, no comment is necessary as the cast is standard practice for type-consistent arithmetic.

## Success Criteria

1. `test_SolverKernel.py::test_run` passes all parametrizations
2. State arrays match CPU reference within tolerance
3. Time arrays continue to match (no regression)
4. No new test failures introduced
5. Code follows repository style and conventions
