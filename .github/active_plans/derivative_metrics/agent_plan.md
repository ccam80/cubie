# Derivative-Based Output Metrics - Agent Implementation Plan

## Executive Summary

Implement six new summary metrics (dxdt_max, dxdt_min, dxdt_extrema, d2xdt2_max, d2xdt2_min, d2xdt2_extrema) that compute derivatives via finite differences of consecutive saved state values. Combined metrics (extrema) follow the existing combined_metrics pattern. This architectural approach avoids invasive changes to the integrator while providing real-time derivative-based feature detection.

**Issue #68 (dxdt time series) is NOT implemented** per user request.

## Component 1: dxdt_max Metric (Issue #66 - part 1)

### Purpose
Track maximum first derivative (slope) value over each summary interval for ECG waveform analysis.

### Behavioral Requirements

**Initialization:**
- Metric name: "dxdt_max"
- Buffer size: 3 elements per variable
  - buffer[0]: previous state value (x[n-1])
  - buffer[1]: maximum dxdt encountered (unscaled)
  - buffer[2]: dt_save value (stored at first update for scaling at save)
- Output size: 1 element (max_dxdt)

**Update Behavior:**
1. Guard: if buffer[0] == 0.0, this is first call or after save
   - Store current value in buffer[0]
   - Store dt_save in buffer[2]
   - Do not update max (no derivative yet)
   
2. Otherwise (buffer[0] != 0.0):
   - Compute unscaled_dxdt = current_value - buffer[0]
   - Update buffer[1] = max(buffer[1], unscaled_dxdt)
   - Store current_value in buffer[0] for next iteration

**Save Behavior:**
1. Scale and write buffer[1] / buffer[2] to output_array[0] (max_dxdt)
2. Reset buffer[1] = 0.0
3. Reset buffer[0] = 0.0 (signals reinitialization for next period)
4. Keep buffer[2] unchanged (dt_save persists)

**dt_save Acquisition:**
- dt_save passed as compile-time constant to factory build() method
- Factory signature modified to accept dt_save parameter
- All summary metrics factory signatures updated for consistency

## Component 2: dxdt_min Metric (Issue #66 - part 2)

### Purpose
Track minimum first derivative (slope) value over each summary interval.

### Behavioral Requirements

**Initialization:**
- Metric name: "dxdt_min"
- Buffer size: 3 elements per variable
  - buffer[0]: previous state value (x[n-1])
  - buffer[1]: minimum dxdt encountered (unscaled)
  - buffer[2]: dt_save value (stored at first update for scaling at save)
- Output size: 1 element (min_dxdt)

**Update Behavior:**
1. Guard: if buffer[0] == 0.0, this is first call or after save
   - Store current value in buffer[0]
   - Store dt_save in buffer[2]
   - Do not update min (no derivative yet)
   
2. Otherwise (buffer[0] != 0.0):
   - Compute unscaled_dxdt = current_value - buffer[0]
   - Update buffer[1] = min(buffer[1], unscaled_dxdt)
   - Store current_value in buffer[0] for next iteration

**Save Behavior:**
1. Scale and write buffer[1] / buffer[2] to output_array[0] (min_dxdt)
2. Reset buffer[1] = 0.0
3. Reset buffer[0] = 0.0 (signals reinitialization for next period)
4. Keep buffer[2] unchanged (dt_save persists)

## Component 3: dxdt_extrema Combined Metric (Issue #66 - combined)

### Purpose
Efficiently compute both max and min first derivatives when both are requested, following the combined_metrics pattern.

### Behavioral Requirements

**Initialization:**
- Metric name: "dxdt_extrema"
- Buffer size: 4 elements per variable
  - buffer[0]: previous state value (x[n-1])
  - buffer[1]: maximum dxdt encountered (unscaled)
  - buffer[2]: minimum dxdt encountered (unscaled)
  - buffer[3]: dt_save value
- Output size: 2 elements (max_dxdt, min_dxdt)

**Update Behavior:**
1. Guard: if buffer[0] == 0.0
   - Store current value in buffer[0]
   - Store dt_save in buffer[3]
   - Do not update extrema
   
2. Otherwise:
   - Compute unscaled_dxdt = current_value - buffer[0]
   - Update buffer[1] = max(buffer[1], unscaled_dxdt)
   - Update buffer[2] = min(buffer[2], unscaled_dxdt)
   - Store current_value in buffer[0]

**Save Behavior:**
1. Write buffer[1] / buffer[3] to output_array[0] (max_dxdt)
2. Write buffer[2] / buffer[3] to output_array[1] (min_dxdt)
3. Reset buffer[1] = 0.0
4. Reset buffer[2] = 0.0
5. Reset buffer[0] = 0.0

**Combined Metrics Registry:**
- Add to _combined_metrics dict: `frozenset(["dxdt_max", "dxdt_min"]): "dxdt_extrema"`
- Auto-substitution when both dxdt_max and dxdt_min requested

### Integration Architecture

**File Locations:**
- `src/cubie/outputhandling/summarymetrics/dxdt_max.py`
- `src/cubie/outputhandling/summarymetrics/dxdt_min.py`
- `src/cubie/outputhandling/summarymetrics/dxdt_extrema.py`
- `src/cubie/outputhandling/summarymetrics/d2xdt2_max.py`
- `src/cubie/outputhandling/summarymetrics/d2xdt2_min.py`
- `src/cubie/outputhandling/summarymetrics/d2xdt2_extrema.py`

**Registration:**
- Use @register_metric(summary_metrics) decorator
- Auto-registration on module import
- Add to __init__.py imports

**Dependencies:**
- Inherit from SummaryMetric
- Import MetricFuncCache, register_metric from metrics.py
- Import cuda from numba

**CUDA Signatures:**
- Update: `(float32/64, float32/64[::1], int32, float32/64)` device function
- Save: `(float32/64[::1], float32/64[::1], int32, float32/64)` device function
- Both must be inline=True, device=True
- **Note:** customisable_variable changes from int32 to float32/64 for dt_save

### Edge Cases

1. **First save point:** No derivative computed, output is 0.0 (from zeroed buffer)
   - Behavior: Document in docstring that first point may be invalid
   
2. **Zero dt_save:** Division by zero
   - Assumption: dt_save > 0 enforced by solver configuration
   - No additional checks needed (fail-fast acceptable)

3. **Variable dt_save:** If dt_save changes between summaries
   - Behavior: Each period uses its current dt_save value
   - Stored in buffer at first update, used at save

4. **No conditional returns:** All CUDA device code paths must reach end
   - Guards use if statements but always continue to end of function

## Cross-Cutting Concerns

### dt_save Parameter Passing

**Architecture Change:**
- Modify ALL summary metric factory build() method signatures to accept dt_save
- dt_save passed as compile-time constant during metric compilation
- Stored in metric build() and passed to device functions

**Factory Signature Change:**
```python
def build(self, dt_save: float) -> MetricFuncCache:
    # dt_save is available here at compile time
    # Pass to device functions via closure or as parameter
```

**Modified Files:**
1. `src/cubie/outputhandling/summarymetrics/metrics.py`
   - Update SummaryMetric.build() signature (abstract method)
   - Update SummaryMetric property accessors to pass dt_save
   
2. `src/cubie/outputhandling/output_functions.py`
   - Modify metric compilation to pass dt_save from OutputConfig
   - Access dt_save via self.output_config.dt_save

3. ALL existing metric files (mean.py, max.py, rms.py, etc.)
   - Update build(self) to build(self, dt_save)
   - Even if not used, maintain consistent signature

**Device Function Signature:**
- Change customisable_variable from int32 to float32/64
- Pass dt_save directly as this parameter
- Consistent across all metrics for uniformity

**Example Implementation:**
```python
def build(self, dt_save: float) -> MetricFuncCache:
    @cuda.jit([
        "float32, float32[::1], int32, float32",
        "float64, float64[::1], int32, float64",
    ], device=True, inline=True)
    def update(value, buffer, current_index, dt_save_param):
        # dt_save_param is the dt_save value
        if buffer[0] == 0.0:
            buffer[0] = value
            buffer[2] = dt_save_param
        else:
            unscaled_diff = value - buffer[0]
            if unscaled_diff > buffer[1]:
                buffer[1] = unscaled_diff
            buffer[0] = value
    
    @cuda.jit([
        "float32[::1], float32[::1], int32, float32",
        "float64[::1], float64[::1], int32, float64",
    ], device=True, inline=True)
    def save(buffer, output_array, summarise_every, dt_save_param):
        output_array[0] = buffer[1] / buffer[2]
        buffer[1] = 0.0
        buffer[0] = 0.0
    
    return MetricFuncCache(update=update, save=save)
```

### Combined Metrics Registry Updates

**File:** `src/cubie/outputhandling/summarymetrics/metrics.py`

**Modifications to __attrs_post_init__:**
```python
def __attrs_post_init__(self) -> None:
    self._params = {}
    self._combined_metrics = {
        frozenset(["mean", "std", "rms"]): "mean_std_rms",
        frozenset(["mean", "std"]): "mean_std",
        frozenset(["std", "rms"]): "std_rms",
        frozenset(["max", "min"]): "extrema",
        frozenset(["dxdt_max", "dxdt_min"]): "dxdt_extrema",  # NEW
        frozenset(["d2xdt2_max", "d2xdt2_min"]): "d2xdt2_extrema",  # NEW
    }
```

### Testing Strategy

**DO NOT create new test files or performance tests.**

**Modifications to existing tests:**

1. **test_all_summaries_long_run** in `tests/outputhandling/test_output_functions.py`
   - Add new metrics to the test case
   - Metrics auto-discovered via summary_metrics object
   
2. **test_all_summary_metrics_numerical_check** in `tests/outputhandling/summarymetrics/test_summary_metrics.py`
   - Add parametrized test cases for combined metrics
   - Add separate test cases for individual metrics (dxdt_max, dxdt_min, etc.)
   - Validate against numpy gradient for first derivatives
   - Validate against finite differences for second derivatives

**Test Implementation Details:**
```python
# In test_all_summary_metrics_numerical_check
# Add to parametrized cases:
@pytest.mark.parametrize("metric_name,validation_func", [
    ("dxdt_max", validate_dxdt_max),
    ("dxdt_min", validate_dxdt_min),
    ("d2xdt2_max", validate_d2xdt2_max),
    ("d2xdt2_min", validate_d2xdt2_min),
    # Combined versions tested automatically
])
```

**Validation Functions:**
```python
def validate_dxdt_max(state_values, dt_save):
    # Use numpy gradient or manual diff
    dxdt = np.diff(state_values) / dt_save
    return np.max(dxdt)

def validate_d2xdt2_max(state_values, dt_save):
    # Three-point finite difference
    d2xdt2 = (state_values[2:] - 2*state_values[1:-1] + state_values[:-2]) / (dt_save**2)
    return np.max(d2xdt2)
```

### Implementation Sequence

1. **Update metric infrastructure** (metrics.py)
   - Add dt_save parameter to build() abstract method
   - Update property accessors
   
2. **Update output_functions.py**
   - Modify metric compilation to pass dt_save
   
3. **Update ALL existing metrics**
   - Add dt_save parameter to each build() method
   - Update device function signatures (int32 -> float32/64 for last param)
   
4. **Implement new metrics** (in order)
   - dxdt_max.py
   - dxdt_min.py
   - dxdt_extrema.py
   - d2xdt2_max.py
   - d2xdt2_min.py
   - d2xdt2_extrema.py
   
5. **Update combined metrics registry**
   - Add new entries in metrics.py __attrs_post_init__
   
6. **Update imports**
   - Add to summarymetrics/__init__.py
   
7. **Update tests**
   - Add validation functions
   - Add parametrized test cases
   
8. **Validate**
   - Run full test suite
   - Check combined metrics substitution works

### Success Criteria

1. All existing tests pass (after updating build() signatures)
2. New metrics correctly compute derivatives via finite differences
3. Combined metrics auto-substitute when both requested
4. Numerical validation passes against numpy/scipy references
5. No conditional returns in CUDA device code
6. Buffer usage minimized (no flags, use zero guards)
7. Scaling done at save time, not during accumulation

## Files to Create

1. `src/cubie/outputhandling/summarymetrics/dxdt_max.py`
2. `src/cubie/outputhandling/summarymetrics/dxdt_min.py`
3. `src/cubie/outputhandling/summarymetrics/dxdt_extrema.py`
4. `src/cubie/outputhandling/summarymetrics/d2xdt2_max.py`
5. `src/cubie/outputhandling/summarymetrics/d2xdt2_min.py`
6. `src/cubie/outputhandling/summarymetrics/d2xdt2_extrema.py`

## Files to Modify

1. `src/cubie/outputhandling/summarymetrics/metrics.py`
   - Add dt_save to build() signature
   - Add new combined metrics to registry
   - Update customisable_variable type in signatures

2. `src/cubie/outputhandling/output_functions.py`
   - Pass dt_save to metric build() methods
   - Update metric compilation

3. `src/cubie/outputhandling/summarymetrics/__init__.py`
   - Import new metric modules

4. ALL existing metric files:
   - `mean.py` - Update build(self, dt_save)
   - `max.py` - Update build(self, dt_save)
   - `min.py` - Update build(self, dt_save)
   - `rms.py` - Update build(self, dt_save)
   - `std.py` - Update build(self, dt_save)
   - `peaks.py` - Update build(self, dt_save)
   - `negative_peaks.py` - Update build(self, dt_save)
   - `max_magnitude.py` - Update build(self, dt_save)
   - `extrema.py` - Update build(self, dt_save)
   - `mean_std.py` - Update build(self, dt_save)
   - `std_rms.py` - Update build(self, dt_save)
   - `mean_std_rms.py` - Update build(self, dt_save)

5. `tests/outputhandling/summarymetrics/test_summary_metrics.py`
   - Add validation functions for derivatives
   - Add parametrized test cases

## NOT Implemented

- Issue #68 (dxdt full time series) - Deferred per user request
- Performance testing - Not required per user
- Examples and user guides - Not required per user
