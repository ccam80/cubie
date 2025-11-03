# Implementation Plan for Issue #141: Output Functions Spree

## Executive Summary

This document provides a comprehensive technical plan for implementing all child issues under #141. The plan follows the constraint that each new metric must be a separate `summary_metric` function with uniform signatures, and device functions can only return a single int with output data passed as device arrays.

## Current Architecture

### SummaryMetric System

**Base Class: `SummaryMetric`**
```python
class SummaryMetric(CUDAFactory):
    buffer_size: Union[int, Callable]  # Per-variable buffer slots
    output_size: Union[int, Callable]  # Per-variable output slots
    name: str
    
    def build(self) -> MetricFuncCache:
        # Returns: update_func, save_func
```

**Required Device Function Signatures:**
```cuda
@cuda.jit([float32/float64 signatures], device=True, inline=True)
def update(value, buffer, current_index, customisable_variable):
    # value: current scalar value
    # buffer: device array slice for this metric
    # current_index: integration step counter
    # customisable_variable: metric parameter (e.g., n_peaks)
    # Returns: None (mutates buffer in place)

@cuda.jit([float32/float64 signatures], device=True, inline=True)
def save(buffer, output_array, summarise_every, customisable_variable):
    # buffer: accumulated data from update phase
    # output_array: destination for final metric values
    # summarise_every: window size
    # customisable_variable: metric parameter
    # Returns: None (mutates output_array in place, resets buffer)
```

### Integration Points

1. **update_summaries.py**: Chains update functions into solver loop
2. **save_summaries.py**: Chains save functions for periodic output
3. **metrics.py**: Registry managing all metrics
4. **__init__.py**: Auto-registers metrics on import

## Detailed Metric Specifications

### Phase 1: Simple Single-Value Metrics

#### Issue #63: Minimum (`min`)

**Complexity:** Low  
**Similar to:** `max.py`

**Specification:**
- `buffer_size`: 1
- `output_size`: 1
- **Buffer layout:** `[current_min]`
- **Update logic:** `if value < buffer[0]: buffer[0] = value`
- **Save logic:** `output_array[0] = buffer[0]; buffer[0] = 1.0e30`  # large sentinel
- **Initial buffer state:** `1.0e30`

**Files to create:**
- `src/cubie/outputhandling/summarymetrics/min.py`

**Tests:**
- `tests/outputhandling/summarymetrics/test_min.py`

---

#### Issue #61: Max Magnitude (`max_magnitude`)

**Complexity:** Low  
**Similar to:** `max.py`

**Specification:**
- `buffer_size`: 1
- `output_size`: 1
- **Buffer layout:** `[current_max_abs]`
- **Update logic:** `abs_val = abs(value); if abs_val > buffer[0]: buffer[0] = abs_val`
- **Save logic:** `output_array[0] = buffer[0]; buffer[0] = -1.0e30`
- **Initial buffer state:** `-1.0e30`

**Files to create:**
- `src/cubie/outputhandling/summarymetrics/max_magnitude.py`

**Tests:**
- `tests/outputhandling/summarymetrics/test_max_magnitude.py`

---

### Phase 2: Statistical Metrics

#### Issue #62: Standard Deviation (`std`)

**Complexity:** Medium  
**Similar to:** `rms.py` + `mean.py`

**Specification:**
- `buffer_size`: 2
- `output_size`: 1
- **Buffer layout:** `[sum, sum_of_squares]`
- **Update logic:**
  ```cuda
  buffer[0] += value           # sum
  buffer[1] += value * value   # sum_of_squares
  ```
- **Save logic:**
  ```cuda
  mean = buffer[0] / summarise_every
  variance = (buffer[1] / summarise_every) - (mean * mean)
  output_array[0] = sqrt(variance)
  buffer[0] = 0.0
  buffer[1] = 0.0
  ```
- **Notes:** Can be optimized with mean/rms if requested together

**Files to create:**
- `src/cubie/outputhandling/summarymetrics/std.py`

**Tests:**
- `tests/outputhandling/summarymetrics/test_std.py`

**Buffer Optimization Opportunity:**
Create optional combined device function when mean + rms + std requested:
- Single buffer: `[sum, sum_of_squares]` (shared)
- All three metrics derive from same accumulated data
- Reduces buffer size from 3 to 2
- Implementation: Add optional `combined_stats.py` for future optimization

---

### Phase 3: Peak Detection Variants

#### Issue #64: Negative Peak (`negative_peak`)

**Complexity:** Medium  
**Similar to:** `peaks.py` (inverted logic)

**Specification:**
- `buffer_size`: `lambda n: 3 + n`
- `output_size`: `lambda n: n`
- **Buffer layout:** `[prev, prev_prev, counter, times[n]]`
- **Update logic:**
  ```cuda
  if (current_index >= 2) and (peak_counter < npeaks) and (prev_prev != 0.0):
      # Detect minimum: prev < value AND prev < prev_prev
      if prev < value and prev_prev > prev:
          buffer[3 + peak_counter] = current_index - 1
          buffer[2] += 1.0
  buffer[0] = value
  buffer[1] = prev
  ```
- **Save logic:** Same as peaks - copy indices and reset

**Files to create:**
- `src/cubie/outputhandling/summarymetrics/negative_peak.py`

**Tests:**
- `tests/outputhandling/summarymetrics/test_negative_peak.py`

---

#### Issue #65: Both Extrema (`extrema`)

**Complexity:** Medium  
**Combines:** `peaks.py` + `negative_peak.py`

**Specification:**
- `buffer_size`: `lambda n: 6 + 2*n`
- `output_size`: `lambda n: 2*n`
- **Buffer layout:**
  ```
  [prev, prev_prev,                      # 2 slots for state
   max_counter, min_counter,             # 2 slots for counters
   max_times[n], min_times[n]]           # 2*n slots for indices
  ```
- **Update logic:**
  ```cuda
  # Check for maximum
  if prev > value and prev_prev < prev:
      buffer[4 + max_counter] = current_index - 1
      buffer[2] += 1.0
  # Check for minimum
  if prev < value and prev_prev > prev:
      buffer[4 + npeaks + min_counter] = current_index - 1
      buffer[3] += 1.0
  ```
- **Save logic:**
  ```cuda
  # Copy max peaks to output[0:n]
  # Copy min peaks to output[n:2n]
  # Reset counters and buffers
  ```

**Files to create:**
- `src/cubie/outputhandling/summarymetrics/extrema.py`

**Tests:**
- `tests/outputhandling/summarymetrics/test_extrema.py`

---

### Phase 4: Derivative-Based Metrics

#### Issue #66: dxdt Extrema (`dxdt_extrema`)

**Complexity:** High  
**Requires:** Numerical differentiation + peak detection

**Specification:**
- `buffer_size`: `lambda n: 5 + n`
- `output_size`: `lambda n: n`
- **Buffer layout:**
  ```
  [value_prev, value_prev_prev,          # 2 slots for x(t) history
   dxdt_prev, dxdt_prev_prev,            # 2 slots for dx/dt history
   counter, times[n]]                     # 1+n slots for peak times
  ```
- **Update logic:**
  ```cuda
  # Calculate derivative: dxdt = (value - value_prev)
  dxdt = value - buffer[0]
  
  # Detect peak in derivative (like peaks.py but on dxdt)
  if (current_index >= 2):
      if buffer[2] > dxdt and buffer[3] < buffer[2]:  # dxdt peak
          buffer[5 + counter] = current_index - 1
          buffer[4] += 1.0
  
  # Shift buffers
  buffer[1] = buffer[0]  # value_prev_prev
  buffer[0] = value      # value_prev
  buffer[3] = buffer[2]  # dxdt_prev_prev
  buffer[2] = dxdt       # dxdt_prev
  ```
- **Save logic:** Copy peak times and reset

**Files to create:**
- `src/cubie/outputhandling/summarymetrics/dxdt_extrema.py`

**Tests:**
- `tests/outputhandling/summarymetrics/test_dxdt_extrema.py`

---

#### Issue #67: d2xdt2 Extrema (`d2xdt2_extrema`)

**Complexity:** High  
**Requires:** Second derivative + peak detection

**Specification:**
- `buffer_size`: `lambda n: 8 + n`
- `output_size`: `lambda n: n`
- **Buffer layout:**
  ```
  [value_prev, value_prev_prev, value_prev_prev_prev,     # 3 for x(t)
   dxdt_prev, dxdt_prev_prev,                             # 2 for dx/dt
   d2xdt2_prev, d2xdt2_prev_prev,                         # 2 for d²x/dt²
   counter, times[n]]                                      # 1+n for peaks
  ```
- **Update logic:**
  ```cuda
  # Calculate first derivative
  dxdt = value - buffer[0]
  
  # Calculate second derivative: d2xdt2 = dxdt - dxdt_prev
  d2xdt2 = dxdt - buffer[3]
  
  # Detect peak in second derivative
  if (current_index >= 3):
      if buffer[5] > d2xdt2 and buffer[6] < buffer[5]:
          buffer[8 + counter] = current_index - 1
          buffer[7] += 1.0
  
  # Shift all buffers
  buffer[2] = buffer[1]  # value_prev_prev_prev
  buffer[1] = buffer[0]  # value_prev_prev
  buffer[0] = value      # value_prev
  buffer[4] = buffer[3]  # dxdt_prev_prev
  buffer[3] = dxdt       # dxdt_prev
  buffer[6] = buffer[5]  # d2xdt2_prev_prev
  buffer[5] = d2xdt2     # d2xdt2_prev
  ```
- **Save logic:** Copy peak times and reset

**Files to create:**
- `src/cubie/outputhandling/summarymetrics/d2xdt2_extrema.py`

**Tests:**
- `tests/outputhandling/summarymetrics/test_d2xdt2_extrema.py`

---

#### Issue #68: dxdt Output (`dxdt`)

**Complexity:** Medium (Lower priority per issue description)  
**Note:** "only add if required in real-time"

**Specification:**
- `buffer_size`: 1
- `output_size`: 1
- **Buffer layout:** `[value_prev]`
- **Update logic:**
  ```cuda
  # Just store the previous value
  buffer[0] = value
  ```
- **Save logic:**
  ```cuda
  # Output the derivative at save time
  # Note: This is approximate since we don't know exact timing
  output_array[0] = value - buffer[0]  # Approximate derivative
  buffer[0] = value
  ```
- **Limitations:** Without dense time information, this is approximate
- **Consider:** May need different approach if exact derivatives required

**Files to create:**
- `src/cubie/outputhandling/summarymetrics/dxdt.py`

**Tests:**
- `tests/outputhandling/summarymetrics/test_dxdt.py`

---

### Phase 5: State Management (Non-Summary Metrics)

#### Issue #76: Save Exit State as Initial Condition

**Complexity:** High  
**Architecture Change:** Yes

**This is NOT a summary_metric** - requires different implementation:

**Approach:**
1. Add new device function to `save_state.py`:
   ```python
   def save_exit_state_factory(num_states: int) -> Callable:
       @cuda.jit(device=True, inline=True)
       def save_exit_state_func(current_state, d_inits):
           for i in range(num_states):
               d_inits[i] = current_state[i]
       return save_exit_state_func
   ```

2. Modify `BatchSolverKernel`:
   - Add `continuation` flag to config
   - Pass `d_inits` array to kernel
   - Call `save_exit_state_func` at end of integration

3. Modify `BatchInputArrays`:
   - Add `fetch_inits()` method to retrieve saved states
   - Ensure `d_inits` is accessible from host

**Files to modify:**
- `src/cubie/outputhandling/save_state.py` - add factory
- `src/cubie/batchsolving/BatchSolverKernel.py` - add continuation logic
- `src/cubie/batchsolving/arrays/BatchInputArrays.py` - add fetch_inits

**New signature requirements:**
- Kernel needs `d_inits` parameter
- Config needs `continuation` boolean

**Tests:**
- `tests/outputhandling/test_save_exit_state.py`
- `tests/batchsolving/test_continuation.py`

---

#### Issue #125: Output Iteration Counts

**Complexity:** High  
**Architecture Change:** Yes

**This is NOT a summary_metric** - requires iteration counter propagation:

**Approach:**
1. Add compile-time flag for iteration counting
2. Pass counter arrays through solver chain:
   - Newton iteration counter (per step)
   - Krylov iteration counter (per Newton iteration)
   - Step controller rejections (per output window)

3. New output arrays:
   - `iteration_counts[n_windows, 3]` where:
     - `[0]`: cumulative Newton iterations since last save
     - `[1]`: cumulative Krylov iterations since last save
     - `[2]`: cumulative step rejections since last save

**Files to modify:**
- Algorithm files (Newton, Krylov) - propagate counters
- `BatchSolverKernel.py` - add counter arrays
- `BatchOutputArrays.py` - add iteration_counts array
- Compile settings - add iteration counting flag

**Signature changes:**
- Device functions return int status codes
- Iteration counts passed as device array parameters
- Each level increments appropriate counter

**Tests:**
- `tests/integrators/test_iteration_counts.py`

**Note:** Keep disabled by default for performance

---

## Buffer Size Summary

| Metric | Buffer Size | Output Size | Complexity |
|--------|-------------|-------------|------------|
| mean (existing) | 1 | 1 | Low |
| max (existing) | 1 | 1 | Low |
| rms (existing) | 1 | 1 | Low |
| peaks (existing) | 3+n | n | Medium |
| **min** | 1 | 1 | Low |
| **max_magnitude** | 1 | 1 | Low |
| **std** | 2 | 1 | Medium |
| **negative_peak** | 3+n | n | Medium |
| **extrema** | 6+2n | 2n | Medium |
| **dxdt_extrema** | 5+n | n | High |
| **d2xdt2_extrema** | 8+n | n | High |
| **dxdt** | 1 | 1 | Medium |

**Combined stats optimization (future):**
- If mean + rms + std requested: buffer = 2 (vs 3 separate)

## Testing Strategy

### Unit Tests (per metric)

Each metric needs:
1. **Basic functionality test:**
   - Known input sequence → expected output
   - Example: `[1, 2, 3, 4, 5]` → `mean=3.0`, `std=1.41...`

2. **Edge cases:**
   - Empty window behavior
   - Single value
   - All same values
   - Very large/small values

3. **Parameter validation:**
   - For parameterized metrics (peaks, extrema)
   - Test different n values

4. **Buffer reset test:**
   - Multiple save cycles
   - Verify buffer clears correctly

### Integration Tests

1. **Multiple metrics simultaneously:**
   - Request 3+ metrics at once
   - Verify correct buffer offsets
   - Check output array layout

2. **Float32 vs Float64:**
   - Both precision modes
   - Verify numerical accuracy

3. **Registry tests:**
   - Metric auto-registration
   - Duplicate name detection
   - Parameter parsing

### System Tests

1. **Full solver integration:**
   - Run actual ODE with new metrics
   - Verify against numpy calculations
   - Performance benchmarks

2. **Continuation test (#76):**
   - Two-phase integration
   - Final state of phase 1 = initial state of phase 2

3. **Iteration counts test (#125):**
   - Verify counts match expected solver behavior
   - Test with/without compile flag

## Implementation Order

### Sprint 1: Foundation (Low-hanging fruit)
1. **min** - simplest, validates workflow
2. **max_magnitude** - builds on min
3. Update __init__.py to register new metrics
4. Basic unit tests

### Sprint 2: Statistics
1. **std** - introduces multi-slot buffers
2. Tests for correctness vs numpy
3. Document buffer optimization opportunity

### Sprint 3: Peak Detection Variants
1. **negative_peak** - reuses peaks logic
2. **extrema** - combines both
3. Comprehensive peak detection tests

### Sprint 4: Derivatives
1. **dxdt_extrema** - first derivative peaks
2. **d2xdt2_extrema** - second derivative peaks
3. **dxdt** (optional, lower priority)
4. Numerical accuracy tests

### Sprint 5: State Management
1. **save_exit_state (#76)**
   - New device function
   - Kernel modifications
   - Continuation tests
2. **iteration_counts (#125)**
   - Counter propagation
   - Compile-time flag
   - Performance validation

## Risks and Mitigation

### Risk 1: Buffer Size Explosion
**Impact:** Large n values in parameterized metrics
**Mitigation:**
- Document recommended n values
- Add warnings for large buffers
- Consider max_n validation

### Risk 2: Numerical Accuracy (Derivatives)
**Impact:** Finite differences accumulate errors
**Mitigation:**
- Use central differences where possible
- Document limitations
- Provide accuracy tests
- Consider higher-order schemes if needed

### Risk 3: Performance Degradation
**Impact:** Many metrics = many device function calls
**Mitigation:**
- Keep inline=True for all device functions
- Implement combined_stats optimization
- Profile before/after
- Make iteration_counts optional

### Risk 4: Breaking Changes (#76, #125)
**Impact:** Kernel signature changes
**Mitigation:**
- Make changes backward compatible
- Use optional parameters
- Comprehensive regression tests
- Version the API

## Success Criteria

1. ✓ All 12 issues implemented and tested
2. ✓ All metrics registered and accessible
3. ✓ Unit tests pass with >95% coverage
4. ✓ Integration tests verify multi-metric usage
5. ✓ No performance regression in existing metrics
6. ✓ Documentation updated with new metrics
7. ✓ Examples demonstrating each new metric

## File Checklist

### New Files (Summary Metrics)
- [ ] `src/cubie/outputhandling/summarymetrics/min.py`
- [ ] `src/cubie/outputhandling/summarymetrics/max_magnitude.py`
- [ ] `src/cubie/outputhandling/summarymetrics/std.py`
- [ ] `src/cubie/outputhandling/summarymetrics/negative_peak.py`
- [ ] `src/cubie/outputhandling/summarymetrics/extrema.py`
- [ ] `src/cubie/outputhandling/summarymetrics/dxdt_extrema.py`
- [ ] `src/cubie/outputhandling/summarymetrics/d2xdt2_extrema.py`
- [ ] `src/cubie/outputhandling/summarymetrics/dxdt.py`

### Modified Files
- [ ] `src/cubie/outputhandling/summarymetrics/__init__.py` - register new metrics
- [ ] `src/cubie/outputhandling/save_state.py` - add save_exit_state_factory (#76)
- [ ] `src/cubie/batchsolving/BatchSolverKernel.py` - continuation + counters
- [ ] `src/cubie/batchsolving/arrays/BatchInputArrays.py` - fetch_inits (#76)
- [ ] `src/cubie/batchsolving/arrays/BatchOutputArrays.py` - iteration counts (#125)

### New Test Files
- [ ] `tests/outputhandling/summarymetrics/test_min.py`
- [ ] `tests/outputhandling/summarymetrics/test_max_magnitude.py`
- [ ] `tests/outputhandling/summarymetrics/test_std.py`
- [ ] `tests/outputhandling/summarymetrics/test_negative_peak.py`
- [ ] `tests/outputhandling/summarymetrics/test_extrema.py`
- [ ] `tests/outputhandling/summarymetrics/test_dxdt_extrema.py`
- [ ] `tests/outputhandling/summarymetrics/test_d2xdt2_extrema.py`
- [ ] `tests/outputhandling/summarymetrics/test_dxdt.py`
- [ ] `tests/outputhandling/test_save_exit_state.py`
- [ ] `tests/batchsolving/test_continuation.py`
- [ ] `tests/integrators/test_iteration_counts.py`

## References

- Issue #141: https://github.com/ccam80/cubie/issues/141
- Existing metric implementations: `src/cubie/outputhandling/summarymetrics/`
- Numba device functions: https://numba.readthedocs.io/en/stable/cuda/device-functions.html
