# Derivative-Based Output Metrics - Agent Implementation Plan

## Executive Summary

Implement three new summary metrics (dxdt_extrema, d2xdt2_extrema, and optionally dxdt) that compute derivatives via finite differences of consecutive saved state values. This architectural approach avoids invasive changes to the integrator while providing real-time derivative-based feature detection.

## Component 1: dxdt_extrema Metric (Issue #66)

### Purpose
Track maximum and minimum first derivative (slope) values over each summary interval for ECG waveform analysis and similar applications.

### Behavioral Requirements

**Initialization:**
- Metric name: "dxdt_extrema"
- Buffer size: 4 elements per variable
  - buffer[0]: previous state value (x[n-1])
  - buffer[1]: maximum dxdt encountered
  - buffer[2]: minimum dxdt encountered
  - buffer[3]: initialization flag (0.0 = uninitialized, 1.0 = ready)
- Output size: 2 elements (max_dxdt, min_dxdt)

**Update Behavior:**
1. On first call (buffer[3] == 0.0):
   - Store current value in buffer[0]
   - Initialize buffer[1] = -1e30 (max sentinel)
   - Initialize buffer[2] = 1e30 (min sentinel)
   - Set buffer[3] = 1.0
   - Return (no derivative computed yet)

2. On subsequent calls:
   - Compute dxdt = (current_value - buffer[0]) / dt_save
   - Update buffer[1] = max(buffer[1], dxdt)
   - Update buffer[2] = min(buffer[2], dxdt)
   - Store current_value in buffer[0] for next iteration

**Save Behavior:**
1. Write buffer[1] to output_array[0] (max_dxdt)
2. Write buffer[2] to output_array[1] (min_dxdt)
3. Reset buffer[1] = -1e30
4. Reset buffer[2] = 1e30
5. Keep buffer[0] and buffer[3] unchanged (preserve continuity across summaries)

**dt_save Acquisition:**
- Pass dt_save through customisable_variable parameter
- Requires coordination with OutputFunctions compilation
- dt_save is available in output_functions.py from OutputConfig

### Integration Architecture

**Registration:**
- Use @register_metric(summary_metrics) decorator
- Auto-registration on module import

**File Location:**
- `src/cubie/outputhandling/summarymetrics/dxdt_extrema.py`

**Dependencies:**
- Inherit from SummaryMetric
- Import MetricFuncCache, register_metric from metrics.py
- Import cuda from numba

**CUDA Signatures:**
- Update: `(float32/64, float32/64[::1], int32, int32)` device function
- Save: `(float32/64[::1], float32/64[::1], int32, int32)` device function
- Both must be inline=True, device=True

### Edge Cases

1. **First save point:** No derivative computed, extrema remain at sentinel values
   - Behavior: Output sentinels, document in docstring
   
2. **Zero dt_save:** Division by zero
   - Assumption: dt_save > 0 enforced by solver configuration
   - No additional checks needed (fail-fast acceptable)

3. **Variable dt_save:** If dt_save changes between summaries
   - Behavior: Each derivative uses its corresponding dt_save
   - Continuity maintained via buffer[0] preservation

4. **Single variable vs multiple:** Metrics applied per-variable independently
   - No special handling needed

## Component 2: d2xdt2_extrema Metric (Issue #67)

### Purpose
Track maximum and minimum second derivative (acceleration) values over each summary interval for pulse wave analysis and inflection point detection.

### Behavioral Requirements

**Initialization:**
- Metric name: "d2xdt2_extrema"
- Buffer size: 5 elements per variable
  - buffer[0]: previous state value (x[n-1])
  - buffer[1]: previous-previous state value (x[n-2])
  - buffer[2]: maximum d2xdt2 encountered
  - buffer[3]: minimum d2xdt2 encountered
  - buffer[4]: initialization counter (0.0, 1.0, 2.0 = warmup stages)
- Output size: 2 elements (max_d2xdt2, min_d2xdt2)

**Update Behavior:**
1. On first call (buffer[4] == 0.0):
   - Store current value in buffer[0]
   - Set buffer[4] = 1.0
   - Return (insufficient history)

2. On second call (buffer[4] == 1.0):
   - Store buffer[0] in buffer[1] (shift history)
   - Store current value in buffer[0]
   - Initialize buffer[2] = -1e30 (max sentinel)
   - Initialize buffer[3] = 1e30 (min sentinel)
   - Set buffer[4] = 2.0
   - Return (still insufficient history)

3. On subsequent calls (buffer[4] == 2.0):
   - Compute d2xdt2 = (current_value - 2*buffer[0] + buffer[1]) / (dt_save**2)
   - Update buffer[2] = max(buffer[2], d2xdt2)
   - Update buffer[3] = min(buffer[3], d2xdt2)
   - Shift history: buffer[1] = buffer[0], buffer[0] = current_value

**Save Behavior:**
1. Write buffer[2] to output_array[0] (max_d2xdt2)
2. Write buffer[3] to output_array[1] (min_d2xdt2)
3. Reset buffer[2] = -1e30
4. Reset buffer[3] = 1e30
5. Keep buffer[0], buffer[1], buffer[4] unchanged (preserve continuity)

**dt_save Acquisition:**
- Same as dxdt_extrema: pass through customisable_variable
- Square dt_save in computation: dt_save_sq = dt_save * dt_save

### Integration Architecture

**Registration:**
- Use @register_metric(summary_metrics) decorator
- Auto-registration on module import

**File Location:**
- `src/cubie/outputhandling/summarymetrics/d2xdt2_extrema.py`

**Dependencies:**
- Identical to dxdt_extrema

**CUDA Signatures:**
- Identical structure to dxdt_extrema
- Update and save device functions with same signatures

### Edge Cases

1. **First two save points:** No second derivative computed
   - Behavior: Output sentinels, document warmup requirement
   
2. **Numerical stability:** dt_save^2 in denominator
   - Assumption: dt_save >> 1e-8 for numerical stability
   - Document minimum recommended dt_save

3. **Warmup across summaries:** If summary interval < 3 saves
   - Behavior: First summary may contain sentinels
   - Document: requires saves_per_summary >= 3 for meaningful output

## Component 3: dxdt Output Metric (Issue #68 - CONDITIONAL)

### Decision Point
**RECOMMEND: Defer implementation until user confirms real-time requirement.**

Issue #68 explicitly states: "easily done offline, so only add if required in real-time"

### Conditional Implementation (IF REQUESTED)

**Purpose:**
Output full time series of first derivatives at each save point.

**Behavioral Requirements:**

**Initialization:**
- Metric name: "dxdt"
- Buffer size: `lambda n: n + 1` (parameterized)
  - buffer[0]: previous state value
  - buffer[1:n+1]: accumulated derivative series
- Output size: `lambda n: n` (full series)
- Parameter: number of saves per summary period

**Update Behavior:**
1. On first call:
   - Store current value in buffer[0]
   - Return

2. On subsequent calls:
   - Compute dxdt = (current_value - buffer[0]) / dt_save
   - Determine save index within summary: idx = (current_index % saves_per_summary)
   - Store dxdt in buffer[1 + idx]
   - Update buffer[0] = current_value

**Save Behavior:**
1. Copy buffer[1:n+1] to output_array[0:n]
2. Clear buffer[1:n+1] = 0.0
3. Keep buffer[0] unchanged

**Memory Implications:**
- Same footprint as full state output (if all states tracked)
- Significant for large systems with many save points
- This is why offline post-processing is typically preferred

## Cross-Cutting Concerns

### dt_save Parameter Passing

**Current Architecture:**
- customisable_variable is int32 in metric signatures
- dt_save is a float (precision-dependent)

**Solution Approaches:**

**Option A (Recommended): Type punning via int32**
- Cast dt_save float bits to int32 in OutputFunctions
- Cast back to float in metric update function
- CUDA device code example:
  ```python
  # In output_functions compilation
  dt_save_bits = cuda.as_dtype(int32)(dt_save_value)
  
  # In metric update
  dt_save_float = cuda.as_dtype(precision)(customisable_variable)
  ```

**Option B: Store in metric buffer**
- Each metric stores dt_save in buffer[last_element]
- Increases buffer size by 1
- More straightforward but wastes memory

**Option C: Extend metric signature (NOT RECOMMENDED)**
- Add dt_save as explicit parameter
- Breaking change to all metrics
- Violates existing architecture

**Recommendation:** Implement Option A if possible, fallback to Option B

### OutputFunctions Integration

**Required Changes:**
To pass dt_save to derivative metrics, OutputFunctions needs to:

1. Extract dt_save from OutputConfig
2. Pass dt_save value when calling derivative metric update functions
3. Use appropriate type conversion (per Option A or B above)

**File:** `src/cubie/outputhandling/output_functions.py`

**Modification Point:**
- update_summary_factory function
- chain_metrics call site
- function_params tuple construction

**Expected Change:**
- Add conditional logic: if metric_name in ['dxdt_extrema', 'd2xdt2_extrema', 'dxdt'], pass dt_save
- Otherwise pass 0 (existing behavior)

**Alternative:** 
- All metrics could receive dt_save (simpler)
- Unused by non-derivative metrics (no harm)
- **RECOMMENDED** for cleaner implementation

### Testing Strategy

**Unit Tests (per metric):**
1. Buffer initialization check
2. Single update warmup behavior
3. Multiple updates with known dt_save
4. Save and reset behavior
5. Continuity across multiple summary periods
6. Precision variants (float32/64)

**Integration Tests:**
1. Simple linear system: x(t) = t → dxdt = 1, d2xdt2 = 0
2. Quadratic system: x(t) = t² → dxdt = 2t, d2xdt2 = 2
3. Sinusoidal system: x(t) = sin(t) → validate extrema locations
4. Comparison with NumPy gradient (post-processing)
5. Multi-variable system (independence check)

**Performance Tests:**
1. Overhead measurement vs existing extrema metric
2. Memory footprint validation
3. Large batch scaling

**Test Files:**
- Extend `tests/outputhandling/summarymetrics/test_summary_metrics.py`
- Add derivative_metrics fixture
- Parameterize with test signals

### Documentation Requirements

**Per-Metric Docstrings:**
- Module docstring explaining finite difference approach
- Class docstring with Notes on warmup behavior
- Update function docstring with parameter details
- Save function docstring with reset behavior

**Usage Examples:**
- How to request derivative metrics in solver
- Interpretation of warmup periods
- Recommended dt_save values for accuracy

**User Guide Updates:**
- Add section on derivative metrics
- Explain numerical differentiation limitations
- Compare to offline post-processing

## Dependencies and Imports

**All Metrics:**
```python
from numba import cuda
from cubie.outputhandling.summarymetrics import summary_metrics
from cubie.outputhandling.summarymetrics.metrics import (
    SummaryMetric,
    register_metric,
    MetricFuncCache,
)
```

**No New Dependencies:**
- All required packages already in project
- NumPy/SciPy only for testing (already dev dependencies)

## Implementation Sequence

**Phase 1: Core Metrics (Required)**
1. Implement dxdt_extrema.py
2. Implement d2xdt2_extrema.py
3. Solve dt_save parameter passing (Option A or B)
4. Modify OutputFunctions if needed

**Phase 2: Testing**
1. Unit tests for each metric
2. Integration tests with simple ODE systems
3. Validation against NumPy/SciPy

**Phase 3: Optional dxdt Output (If Requested)**
1. Confirm user requirement
2. Implement dxdt.py
3. Add corresponding tests

**Phase 4: Documentation**
1. Complete all docstrings
2. Add usage examples
3. Update user guide if necessary

## Potential Gotchas

1. **CUDA float/int casting:** Ensure precision-safe type punning for dt_save
2. **Buffer persistence:** Update functions must not reset buffers inappropriately
3. **Warmup handling:** Clear documentation of initialization behavior
4. **Summary boundaries:** Ensure state history preserved across summary saves
5. **Precision modes:** Test both float32 and float64 thoroughly
6. **Combined metrics:** Future optimization may combine dxdt + d2xdt2 for shared computation

## Success Criteria

1. ✅ All three metrics (or two if dxdt deferred) implemented and tested
2. ✅ Pass validation against SciPy/NumPy derivatives on test signals
3. ✅ Performance overhead < 5% (dxdt_extrema) and < 7% (d2xdt2_extrema)
4. ✅ Work with all precision modes (float32/64)
5. ✅ Work with all integration algorithms
6. ✅ Auto-register and integrate seamlessly with existing metric system
7. ✅ Complete docstrings and usage documentation
8. ✅ All tests passing in CI

## Non-Goals (Out of Scope)

- ❌ Higher-order derivatives (d3xdt3, etc.)
- ❌ Adaptive finite difference methods
- ❌ Richardson extrapolation for accuracy
- ❌ Accessing ODE solver analytical derivatives
- ❌ Modifying integration loop architecture
- ❌ GPU-based spline differentiation
- ❌ Combined derivative metrics (defer to future optimization)
