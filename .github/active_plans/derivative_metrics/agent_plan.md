# Derivative-Based Output Metrics - Agent Implementation Plan

## Executive Summary

Implement six new summary metrics (dxdt_max, dxdt_min, dxdt_extrema, d2xdt2_max, d2xdt2_min, d2xdt2_extrema) that compute derivatives via finite differences of consecutive saved state values. Combined metrics (extrema) follow the existing combined_metrics pattern. This architectural approach avoids invasive changes to the integrator while providing real-time derivative-based feature detection.

**Issue #68 (dxdt time series) is NOT implemented** per user request.

## dt_save Access Pattern (CRITICAL ARCHITECTURE)

**New Simplified Approach:**
1. Replace `CompileSettingsPlaceholder` with `MetricConfig` attrs class containing dt_save
2. Add `update()` method to `SummaryMetrics` that calls update on all registered metric_objects
3. Add `update()` method to `SummaryMetric` that calls `update_compile_settings()`
4. In `output_functions.build()`, call `summary_metrics.update(dt_save=dt_save)`
5. Metrics access dt_save via `self.compile_settings.dt_save` in their `build()` method
6. Capture dt_save in closure when compiling device functions

**Key Benefits:**
- NO changes to build() signatures (stays `build(self)`)
- NO changes to existing metrics at all
- Uses existing CUDAFactory cache invalidation pattern
- dt_save changes automatically trigger metric recompilation

## Component 1: dxdt_max Metric (Issue #66 - part 1)

### Purpose
Track maximum first derivative (slope) value over each summary interval for ECG waveform analysis.

### Behavioral Requirements

**Initialization:**
- Metric name: "dxdt_max"
- Buffer size: 2 elements per variable
  - buffer[0]: previous state value (x[n-1])
  - buffer[1]: maximum dxdt encountered (unscaled)
- Output size: 1 element (max_dxdt)

**Update Behavior (Predicated Commit Pattern):**
```python
unscaled_dxdt = value - buffer[0]
update_max = (buffer[0] != 0.0) and (unscaled_dxdt > buffer[1])
buffer[1] = selp(update_max, unscaled_dxdt, buffer[1])
buffer[0] = value
```
- No if/else statements
- All code paths reach function end
- dt_save accessed from closure: `dt_save = self.compile_settings.dt_save` in build()

**Save Behavior:**
1. Scale and write buffer[1] / dt_save to output_array[0] (max_dxdt)
2. Reset buffer[1] = 0.0
3. Reset buffer[0] = 0.0 (signals reinitialization for next period)

**dt_save Acquisition:**
- dt_save accessed via `self.compile_settings.dt_save` in build() method
- Captured in closure when compiling device functions
- NOT passed as parameter, NOT stored in buffer

## Component 2: dxdt_min Metric (Issue #66 - part 2)

### Purpose
Track minimum first derivative (slope) value over each summary interval.

### Behavioral Requirements

**Initialization:**
- Metric name: "dxdt_min"
- Buffer size: 2 elements per variable
  - buffer[0]: previous state value (x[n-1])
  - buffer[1]: minimum dxdt encountered (unscaled)
- Output size: 1 element (min_dxdt)

**Update Behavior (Predicated Commit Pattern):**
```python
unscaled_dxdt = value - buffer[0]
update_min = (buffer[0] != 0.0) and (unscaled_dxdt < buffer[1])
buffer[1] = selp(update_min, unscaled_dxdt, buffer[1])
buffer[0] = value
```

**Save Behavior:**
1. Scale and write buffer[1] / dt_save to output_array[0] (min_dxdt)
2. Reset buffer[1] = 0.0
3. Reset buffer[0] = 0.0

## Component 3: dxdt_extrema Combined Metric (Issue #66 - combined)

### Purpose
Efficiently compute both max and min first derivatives when both are requested, following the combined_metrics pattern.

### Behavioral Requirements

**Initialization:**
- Metric name: "dxdt_extrema"
- Buffer size: 3 elements per variable
  - buffer[0]: previous state value (x[n-1])
  - buffer[1]: maximum dxdt encountered (unscaled)
  - buffer[2]: minimum dxdt encountered (unscaled)
- Output size: 2 elements (max_dxdt, min_dxdt)

**Update Behavior (Predicated Commit Pattern):**
```python
unscaled_dxdt = value - buffer[0]
update_max = (buffer[0] != 0.0) and (unscaled_dxdt > buffer[1])
update_min = (buffer[0] != 0.0) and (unscaled_dxdt < buffer[2])
buffer[1] = selp(update_max, unscaled_dxdt, buffer[1])
buffer[2] = selp(update_min, unscaled_dxdt, buffer[2])
buffer[0] = value
```

**Save Behavior:**
1. Write buffer[1] / dt_save to output_array[0] (max_dxdt)
2. Write buffer[2] / dt_save to output_array[1] (min_dxdt)
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


### Integration Architecture

**dt_save Access (SIMPLIFIED APPROACH - NO BUILD SIGNATURE CHANGES):**

**Infrastructure Changes:**
1. Replace `CompileSettingsPlaceholder` with `MetricConfig` attrs class
   - Add dt_save field to MetricConfig
   - MetricConfig becomes the compile_settings for all metrics

2. Add `update()` method to `SummaryMetrics` class
   - Calls `update()` on all registered metric_objects
   - Propagates dt_save to all metrics

3. Add `update()` method to `SummaryMetric` class
   - Calls `update_compile_settings()` with new MetricConfig
   - Triggers cache invalidation when dt_save changes

4. Update `OutputFunctions.build()` method
   - Call `summary_metrics.update(dt_save=dt_save)` before requesting metric functions
   - Ensures current dt_save is propagated to all metrics

5. Metrics access dt_save in their build() method
   - Via `self.compile_settings.dt_save`
   - Capture in closure when compiling device functions
   - NO parameter passing needed

**Key Benefits:**
- NO changes to build() signatures (stays `build(self)`)
- NO changes to existing metrics
- NO changes to device function signatures (customisable_variable stays int32)
- Uses existing CUDAFactory cache invalidation pattern
- dt_save changes automatically trigger metric recompilation

**Example in metric build():**
```python
def build(self) -> MetricFuncCache:
    # Access dt_save from compile_settings
    dt_save = self.compile_settings.dt_save
    
    # Capture in closure
    @cuda.jit([...], device=True, inline=True)
    def update(value, buffer, current_index, customisable_variable):
        unscaled_dxdt = value - buffer[0]
        update_max = (buffer[0] != 0.0) and (unscaled_dxdt > buffer[1])
        buffer[1] = selp(update_max, unscaled_dxdt, buffer[1])
        buffer[0] = value
    
    @cuda.jit([...], device=True, inline=True)
    def save(buffer, output_array, summarise_every, customisable_variable):
        # dt_save captured from closure
        output_array[0] = buffer[1] / dt_save
        buffer[1] = 0.0
        buffer[0] = 0.0
    
    return MetricFuncCache(update=update, save=save)
```

**Files to Modify:**
1. `src/cubie/outputhandling/summarymetrics/metrics.py`
   - Replace CompileSettingsPlaceholder with MetricConfig (add dt_save field)
   - Add update() method to SummaryMetrics
   - Add update() method to SummaryMetric
   - Add new combined metrics to _combined_metrics registry

2. `src/cubie/outputhandling/output_functions.py`
   - Call summary_metrics.update(dt_save=...) in build() method
   - Add dt_save to ALL_OUTPUT_FUNCTION_PARAMETERS (for completeness)

3. `src/cubie/outputhandling/output_config.py`
   - Add dt_save field to OutputConfig attrs class

4. `src/cubie/outputhandling/summarymetrics/__init__.py`
   - Import new metric modules

5. `tests/outputhandling/summarymetrics/test_summary_metrics.py`
   - Add validation logic to existing calculate_single_summary_array
   - Add new metrics to special cases in test_all_summary_metrics_numerical_check

**Files to Create:**
1. `src/cubie/outputhandling/summarymetrics/dxdt_max.py`
2. `src/cubie/outputhandling/summarymetrics/dxdt_min.py`
3. `src/cubie/outputhandling/summarymetrics/dxdt_extrema.py`
4. `src/cubie/outputhandling/summarymetrics/d2xdt2_max.py`
5. `src/cubie/outputhandling/summarymetrics/d2xdt2_min.py`
6. `src/cubie/outputhandling/summarymetrics/d2xdt2_extrema.py`

**NO Changes to Existing Metrics:**
- All 12 existing metric files remain unchanged
- No build() signature changes
- No device function signature changes

### Combined Metrics Registry Updates

Add to `_combined_metrics` dict in `SummaryMetrics.__attrs_post_init__()`:
```python
frozenset(["dxdt_max", "dxdt_min"]): "dxdt_extrema",
frozenset(["d2xdt2_max", "d2xdt2_min"]): "d2xdt2_extrema",
```

### Testing Strategy

**DO NOT create new test files.**

Add to existing tests:
1. **calculate_single_summary_array** - Add validation logic for derivative metrics
2. **test_all_summary_metrics_numerical_check** - Add to special cases (combinations and no combinations)
3. **test_all_summaries_long_run** - Metrics auto-discovered via summary_metrics object

Validate against numpy.diff for first derivatives and finite difference formula for second derivatives.

## Implementation Sequence

1. **Update MetricConfig** in metrics.py
   - Replace CompileSettingsPlaceholder with MetricConfig
   - Add dt_save field with default value

2. **Add update methods**
   - SummaryMetrics.update() - propagate to all metrics
   - SummaryMetric.update() - call update_compile_settings

3. **Update OutputConfig** - Add dt_save field

4. **Update OutputFunctions** - Call summary_metrics.update()

5. **Update combined metrics registry**

6. **Implement new metrics** (can be done in parallel)
   - dxdt_max.py
   - dxdt_min.py
   - dxdt_extrema.py
   - d2xdt2_max.py
   - d2xdt2_min.py
   - d2xdt2_extrema.py

7. **Update imports** in __init__.py

8. **Add tests** to existing test functions

## Success Criteria

1. All existing tests pass (NO changes to existing metrics)
2. New metrics correctly compute derivatives via finite differences
3. Combined metrics auto-substitute when both requested
4. Numerical validation passes against numpy references
5. No conditional returns in CUDA device code
6. Buffer usage minimized (no dt_save storage, no flags)
7. Scaling done at save time

## Files Summary

**Modified:** 5 files
- metrics.py (MetricConfig, update methods, registry)
- output_config.py (dt_save field)
- output_functions.py (call summary_metrics.update)
- __init__.py (import new metrics)
- test_summary_metrics.py (add validation logic)

**Created:** 6 files
- dxdt_max.py, dxdt_min.py, dxdt_extrema.py
- d2xdt2_max.py, d2xdt2_min.py, d2xdt2_extrema.py

**Unchanged:** 12 existing metric files
- No signature changes, no functional changes
