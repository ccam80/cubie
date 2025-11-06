# Implementation Task List
# Feature: Derivative-Based Output Metrics (SIMPLIFIED APPROACH)
# Plan Reference: .github/active_plans/derivative_metrics/agent_plan.md

## Overview

This task list implements six new summary metrics (dxdt_max, dxdt_min, dxdt_extrema, d2xdt2_max, d2xdt2_min, d2xdt2_extrema) that compute derivatives via finite differences. The SIMPLIFIED implementation uses the CUDAFactory compile_settings pattern to propagate dt_save via update() methods, avoiding signature changes to all existing metrics.

**Key Architecture Decisions (SIMPLIFIED):**
- dt_save passed via `summary_metrics.update(dt_save=dt_save)` call in OutputFunctions.build()
- Replace `CompileSettingsPlaceholder` with `MetricConfig` attrs class containing dt_save
- Add `update()` method to `SummaryMetrics` (propagates to all metric_objects)
- Add `update()` method to `SummaryMetric` (calls update_compile_settings)
- Metrics access dt_save via `self.compile_settings.dt_save` in build()
- dt_save captured in closure when compiling device functions
- NO changes to build() signatures (stays `build(self)`)
- NO changes to existing metrics at all
- NO changes to device function signatures (customisable_variable stays int32)
- dt_save NOT stored in buffers (smaller memory footprint)
- Predicated commit pattern (selp) used instead of if/else for warp efficiency

**Buffer Sizes (REDUCED - dt_save NOT stored):**
- dxdt_max/min: 2 floats (prev_value, max/min_unscaled)
- dxdt_extrema: 3 floats (prev_value, max_unscaled, min_unscaled)
- d2xdt2_max/min: 3 floats (prev_value, prev_prev_value, max/min_unscaled)
- d2xdt2_extrema: 4 floats (prev_value, prev_prev_value, max_unscaled, min_unscaled)

**Dependencies:**
Task groups can be executed with some parallelism where noted.

---

## Task Group 1: Replace CompileSettingsPlaceholder with MetricConfig - SEQUENTIAL
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: src/cubie/outputhandling/summarymetrics/metrics.py (lines 31-36, CompileSettingsPlaceholder definition)
- File: src/cubie/outputhandling/summarymetrics/metrics.py (lines 117, setup_compile_settings call)

**Input Validation Required**:
- dt_save: Type float, validate dt_save > 0.0

**Tasks**:

### 1.1 Replace CompileSettingsPlaceholder with MetricConfig
   - File: src/cubie/outputhandling/summarymetrics/metrics.py
   - Action: Modify
   - Details:
     Replace the CompileSettingsPlaceholder class (lines 31-36) with:
     ```python
     @attrs.define
     class MetricConfig:
         """Configuration for summary metric compilation.
         
         Attributes
         ----------
         dt_save
             Time interval between saved states. Used by derivative
             metrics to scale finite differences. Defaults to 0.01.
         """
         
         _dt_save: float = attrs.field(
             default=0.01,
             validator=attrs.validators.instance_of(float)
         )
         
         @property
         def dt_save(self) -> float:
             """Time interval between saved states."""
             return self._dt_save
     ```
   - Edge cases: Default value 0.01 ensures backwards compatibility
   - Integration: All metrics get this via setup_compile_settings

### 1.2 Update SummaryMetric.__init__ to use MetricConfig
   - File: src/cubie/outputhandling/summarymetrics/metrics.py
   - Action: Modify
   - Details:
     Change line 117 from:
     ```python
     self.setup_compile_settings(CompileSettingsPlaceholder())
     ```
     to:
     ```python
     self.setup_compile_settings(MetricConfig())
     ```
   - Edge cases: None
   - Integration: All metrics now have MetricConfig as compile_settings

**Outcomes**:
[Empty - to be filled by do_task agent]

---

## Task Group 2: Add update() Methods to SummaryMetrics and SummaryMetric - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Group 1

**Required Context**:
- File: src/cubie/outputhandling/summarymetrics/metrics.py (lines 65-152, SummaryMetric class)
- File: src/cubie/outputhandling/summarymetrics/metrics.py (lines 154-217, SummaryMetrics class and __attrs_post_init__)
- File: src/cubie/CUDAFactory.py (lines 129-194, update_compile_settings method)

**Input Validation Required**:
- dt_save: Type float, validate dt_save > 0.0 in MetricConfig validator

**Tasks**:

### 2.1 Add update() Method to SummaryMetric
   - File: src/cubie/outputhandling/summarymetrics/metrics.py
   - Action: Modify
   - Details:
     Add method after save_device_func property (after line 151):
     ```python
     def update(self, **kwargs) -> None:
         """Update metric compile settings.
         
         Parameters
         ----------
         **kwargs
             Compile settings to update (e.g., dt_save=0.02).
             
         Notes
         -----
         Updates the MetricConfig and invalidates cache if values change.
         Triggers recompilation on next device_function access.
         """
         self.update_compile_settings(kwargs, silent=True)
     ```
   - Edge cases: silent=True allows unrelated kwargs to pass through
   - Integration: Called by SummaryMetrics.update()

### 2.2 Add update() Method to SummaryMetrics
   - File: src/cubie/outputhandling/summarymetrics/metrics.py
   - Action: Modify
   - Details:
     Add method after __attrs_post_init__ (after line 216):
     ```python
     def update(self, **kwargs) -> None:
         """Update compile settings for all registered metrics.
         
         Parameters
         ----------
         **kwargs
             Compile settings to update (e.g., dt_save=0.02).
             
         Notes
         -----
         Propagates updates to all registered metric objects.
         Each metric invalidates its cache if values change.
         """
         for metric in self._metric_objects.values():
             metric.update(**kwargs)
     ```
   - Edge cases: No metrics case - loops over empty dict (safe)
   - Integration: Called by OutputFunctions.build()

**Outcomes**:
[Empty - to be filled by do_task agent]

---

## Task Group 3: Add dt_save to OutputConfig and Update Combined Metrics Registry - SEQUENTIAL
**Status**: [ ]
**Dependencies**: None (can run parallel with Groups 1-2)

**Required Context**:
- File: src/cubie/outputhandling/output_config.py (lines 94-163, OutputConfig class)
- File: src/cubie/outputhandling/summarymetrics/metrics.py (lines 203-216, __attrs_post_init__)

**Input Validation Required**:
- dt_save: Type float, NO validation (assume validated at higher level)

**Tasks**:

### 3.1 Add dt_save Field to OutputConfig
   - File: src/cubie/outputhandling/output_config.py
   - Action: Modify
   - Details:
     Add dt_save field after _max_observables (around line 131):
     ```python
     _dt_save: float = attrs.field(
         default=0.01,
         validator=attrs.validators.instance_of(float)
     )
     ```
     Add property after other properties:
     ```python
     @property
     def dt_save(self) -> float:
         """Time interval between saved states."""
         return self._dt_save
     ```
   - Edge cases: Default 0.01 matches typical usage
   - Integration: Available via self.compile_settings.dt_save in OutputFunctions

### 3.2 Add dt_save to ALL_OUTPUT_FUNCTION_PARAMETERS
   - File: src/cubie/outputhandling/output_functions.py
   - Action: Modify
   - Details:
     Add "dt_save" to the ALL_OUTPUT_FUNCTION_PARAMETERS set (line 29):
     ```python
     ALL_OUTPUT_FUNCTION_PARAMETERS = {
         "output_types",
         "saved_states", "saved_observables",
         "summarised_states", "summarised_observables",
         "saved_state_indices",
         "saved_observable_indices",
         "summarised_state_indices",
         "summarised_observable_indices",
         "dt_save",  # NEW
     }
     ```
   - Edge cases: None
   - Integration: Allows dt_save to flow through configuration

### 3.3 Update Combined Metrics Registry
   - File: src/cubie/outputhandling/summarymetrics/metrics.py
   - Action: Modify
   - Details:
     In __attrs_post_init__ method (around line 211), add two entries to _combined_metrics dict:
     ```python
     self._combined_metrics = {
         frozenset(["mean", "std", "rms"]): "mean_std_rms",
         frozenset(["mean", "std"]): "mean_std",
         frozenset(["std", "rms"]): "std_rms",
         frozenset(["max", "min"]): "extrema",
         frozenset(["dxdt_max", "dxdt_min"]): "dxdt_extrema",  # NEW
         frozenset(["d2xdt2_max", "d2xdt2_min"]): "d2xdt2_extrema",  # NEW
     }
     ```
   - Edge cases: Both constituents must be present for substitution (existing logic handles this)
   - Integration: Auto-substitution when both requested

**Outcomes**:
[Empty - to be filled by do_task agent]

---

## Task Group 4: Update OutputFunctions.build() to Call summary_metrics.update() - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Groups 1, 2, 3

**Required Context**:
- File: src/cubie/outputhandling/output_functions.py (lines 176-222, build() method)
- File: src/cubie/outputhandling/summarymetrics/__init__.py (line 14, summary_metrics singleton)

**Input Validation Required**:
- None (dt_save from compile_settings already validated)

**Tasks**:

### 4.1 Call summary_metrics.update() in build()
   - File: src/cubie/outputhandling/output_functions.py
   - Action: Modify
   - Details:
     In build() method, add call before building summary functions (around line 191):
     ```python
     def build(self) -> OutputFunctionCache:
         """Compile output functions and calculate memory requirements."""
         config = self.compile_settings
         
         # Update all metrics with current dt_save
         from cubie.outputhandling.summarymetrics import summary_metrics
         summary_metrics.update(dt_save=config.dt_save)
         
         buffer_sizes = self.summaries_buffer_sizes
         
         # Build functions using output sizes objects
         save_state_func = save_state_factory(...)
         # ... rest of method unchanged
     ```
   - Edge cases: Import inside method avoids circular import issues
   - Integration: Metrics recompile with new dt_save when accessed

**Outcomes**:
[Empty - to be filled by do_task agent]

---

## Task Group 5: Implement dxdt_max Metric - PARALLEL (with 5.2-5.6)
**Status**: [ ]
**Dependencies**: Groups 1, 2

**Required Context**:
- File: src/cubie/outputhandling/summarymetrics/mean.py (entire file - pattern reference)
- File: src/cubie/outputhandling/summarymetrics/extrema.py (entire file - pattern reference)
- File: src/cubie/cuda_simsafe.py (selp function for predicated commit)

**Input Validation Required**:
- None (operates on buffer and value from integration loop)

**Tasks**:

### 5.1 Create dxdt_max.py Metric File
   - File: src/cubie/outputhandling/summarymetrics/dxdt_max.py
   - Action: Create
   - Details:
     ```python
     """
     Maximum first derivative (dxdt) summary metric for CUDA batch integration.
     
     This module implements a summary metric that tracks the maximum first
     derivative computed via backward finite differences of saved state values.
     """
     
     from numba import cuda
     from cubie.cuda_simsafe import selp
     
     from cubie.outputhandling.summarymetrics import summary_metrics
     from cubie.outputhandling.summarymetrics.metrics import (
         SummaryMetric,
         register_metric,
         MetricFuncCache,
     )
     
     
     @register_metric(summary_metrics)
     class DxdtMax(SummaryMetric):
         """Summary metric that tracks maximum first derivative.
         
         Notes
         -----
         Uses 2 buffer slots: buffer[0] for previous value, buffer[1] for
         maximum unscaled dxdt. Scaling by dt_save done at save time.
         """
         
         def __init__(self) -> None:
             """Initialise the DxdtMax summary metric."""
             super().__init__(
                 name="dxdt_max",
                 buffer_size=2,
                 output_size=1,
             )
         
         def build(self) -> MetricFuncCache:
             """Generate CUDA device functions for dxdt_max calculation.
             
             Returns
             -------
             MetricFuncCache
                 Cache containing the device update and save callbacks.
             
             Notes
             -----
             Accesses dt_save from self.compile_settings.dt_save and captures
             it in closure. Uses predicated commit pattern with selp().
             """
             
             # Access dt_save from compile_settings
             dt_save = self.compile_settings.dt_save
             
             # no cover: start
             @cuda.jit(
                 [
                     "float32, float32[::1], int32, int32",
                     "float64, float64[::1], int32, int32",
                 ],
                 device=True,
                 inline=True,
             )
             def update(
                 value,
                 buffer,
                 current_index,
                 customisable_variable,
             ):
                 """Update maximum dxdt with new value.
                 
                 Parameters
                 ----------
                 value
                     float. New state value.
                 buffer
                     device array. Storage for [prev_value, max_dxdt_unscaled].
                 current_index
                     int. Current integration step index (unused).
                 customisable_variable
                     int. Metric parameter placeholder (unused).
                 
                 Notes
                 -----
                 Uses predicated commit pattern - no conditional returns.
                 buffer[0] == 0.0 guards against first call.
                 """
                 unscaled_dxdt = value - buffer[0]
                 update_max = (buffer[0] != 0.0) and (unscaled_dxdt > buffer[1])
                 buffer[1] = selp(update_max, unscaled_dxdt, buffer[1])
                 buffer[0] = value
             
             @cuda.jit(
                 [
                     "float32[::1], float32[::1], int32, int32",
                     "float64[::1], float64[::1], int32, int32",
                 ],
                 device=True,
                 inline=True,
             )
             def save(
                 buffer,
                 output_array,
                 summarise_every,
                 customisable_variable,
             ):
                 """Save scaled maximum dxdt and reset buffer.
                 
                 Parameters
                 ----------
                 buffer
                     device array. Buffer containing [prev_value, max_dxdt_unscaled].
                 output_array
                     device array. Output location for max_dxdt.
                 summarise_every
                     int. Number of steps between saves (unused).
                 customisable_variable
                     int. Metric parameter placeholder (unused).
                 
                 Notes
                 -----
                 Scales unscaled difference by dt_save (captured from closure).
                 Resets buffer to signal reinitialization.
                 """
                 output_array[0] = buffer[1] / dt_save
                 buffer[1] = 0.0
                 buffer[0] = 0.0
             
             # no cover: end
             return MetricFuncCache(update=update, save=save)
     ```
   - Edge cases: buffer[0] == 0.0 check handles first call and post-save
   - Integration: Auto-registers via @register_metric decorator

**Outcomes**:
[Empty - to be filled by do_task agent]

---

## Task Group 5.2: Implement dxdt_min Metric - PARALLEL
**Status**: [ ]
**Dependencies**: Groups 1, 2

**Required Context**:
- File: src/cubie/outputhandling/summarymetrics/dxdt_max.py (pattern reference from 5.1)

**Input Validation Required**:
- None

**Tasks**:

### 5.2.1 Create dxdt_min.py Metric File
   - File: src/cubie/outputhandling/summarymetrics/dxdt_min.py
   - Action: Create
   - Details:
     Copy dxdt_max.py and modify:
     - Class name: DxdtMin
     - name="dxdt_min"
     - In update(): change `unscaled_dxdt > buffer[1]` to `unscaled_dxdt < buffer[1]`
     - Docstrings updated to reference minimum instead of maximum
   - Edge cases: Same as dxdt_max
   - Integration: Auto-registers

**Outcomes**:
[Empty - to be filled by do_task agent]

---

## Task Group 5.3: Implement dxdt_extrema Combined Metric - PARALLEL
**Status**: [ ]
**Dependencies**: Groups 1, 2

**Required Context**:
- File: src/cubie/outputhandling/summarymetrics/extrema.py (combined metric pattern)
- File: src/cubie/outputhandling/summarymetrics/dxdt_max.py (derivative pattern)

**Input Validation Required**:
- None

**Tasks**:

### 5.3.1 Create dxdt_extrema.py Metric File
   - File: src/cubie/outputhandling/summarymetrics/dxdt_extrema.py
   - Action: Create
   - Details:
     Combine dxdt_max and dxdt_min patterns:
     - Class name: DxdtExtrema
     - name="dxdt_extrema"
     - buffer_size=3 (prev_value, max_unscaled, min_unscaled)
     - output_size=2
     - In update():
       ```python
       unscaled_dxdt = value - buffer[0]
       update_max = (buffer[0] != 0.0) and (unscaled_dxdt > buffer[1])
       update_min = (buffer[0] != 0.0) and (unscaled_dxdt < buffer[2])
       buffer[1] = selp(update_max, unscaled_dxdt, buffer[1])
       buffer[2] = selp(update_min, unscaled_dxdt, buffer[2])
       buffer[0] = value
       ```
     - In save():
       ```python
       output_array[0] = buffer[1] / dt_save  # max
       output_array[1] = buffer[2] / dt_save  # min
       buffer[1] = 0.0
       buffer[2] = 0.0
       buffer[0] = 0.0
       ```
   - Edge cases: Same as individual metrics
   - Integration: Auto-substituted when both dxdt_max and dxdt_min requested

**Outcomes**:
[Empty - to be filled by do_task agent]

---

## Task Group 5.4: Implement d2xdt2_max Metric - PARALLEL
**Status**: [ ]
**Dependencies**: Groups 1, 2

**Required Context**:
- File: src/cubie/outputhandling/summarymetrics/dxdt_max.py (first derivative pattern)

**Input Validation Required**:
- None

**Tasks**:

### 5.4.1 Create d2xdt2_max.py Metric File
   - File: src/cubie/outputhandling/summarymetrics/d2xdt2_max.py
   - Action: Create
   - Details:
     Extend dxdt_max pattern for second derivative:
     - Class name: D2xdt2Max
     - name="d2xdt2_max"
     - buffer_size=3 (prev_value, prev_prev_value, max_unscaled)
     - output_size=1
     - In update():
       ```python
       # Central difference: d2xdt2 = (x[n] - 2*x[n-1] + x[n-2])
       unscaled_d2xdt2 = value - 2.0 * buffer[0] + buffer[1]
       # Only update if we have 2 previous values
       update_max = (buffer[1] != 0.0) and (unscaled_d2xdt2 > buffer[2])
       buffer[2] = selp(update_max, unscaled_d2xdt2, buffer[2])
       # Shift values
       buffer[1] = buffer[0]
       buffer[0] = value
       ```
     - In save():
       ```python
       # Scale by dt_save^2
       output_array[0] = buffer[2] / (dt_save * dt_save)
       buffer[2] = 0.0
       buffer[1] = 0.0
       buffer[0] = 0.0
       ```
   - Edge cases: buffer[1] == 0.0 guard handles first two calls
   - Integration: Auto-registers

**Outcomes**:
[Empty - to be filled by do_task agent]

---

## Task Group 5.5: Implement d2xdt2_min Metric - PARALLEL
**Status**: [ ]
**Dependencies**: Groups 1, 2

**Required Context**:
- File: src/cubie/outputhandling/summarymetrics/d2xdt2_max.py (pattern reference from 5.4)

**Input Validation Required**:
- None

**Tasks**:

### 5.5.1 Create d2xdt2_min.py Metric File
   - File: src/cubie/outputhandling/summarymetrics/d2xdt2_min.py
   - Action: Create
   - Details:
     Copy d2xdt2_max.py and modify:
     - Class name: D2xdt2Min
     - name="d2xdt2_min"
     - In update(): change `unscaled_d2xdt2 > buffer[2]` to `unscaled_d2xdt2 < buffer[2]`
     - Docstrings updated to reference minimum
   - Edge cases: Same as d2xdt2_max
   - Integration: Auto-registers

**Outcomes**:
[Empty - to be filled by do_task agent]

---

## Task Group 5.6: Implement d2xdt2_extrema Combined Metric - PARALLEL
**Status**: [ ]
**Dependencies**: Groups 1, 2

**Required Context**:
- File: src/cubie/outputhandling/summarymetrics/d2xdt2_max.py (second derivative pattern)
- File: src/cubie/outputhandling/summarymetrics/dxdt_extrema.py (combined pattern)

**Input Validation Required**:
- None

**Tasks**:

### 5.6.1 Create d2xdt2_extrema.py Metric File
   - File: src/cubie/outputhandling/summarymetrics/d2xdt2_extrema.py
   - Action: Create
   - Details:
     Combine d2xdt2_max and d2xdt2_min patterns:
     - Class name: D2xdt2Extrema
     - name="d2xdt2_extrema"
     - buffer_size=4 (prev_value, prev_prev_value, max_unscaled, min_unscaled)
     - output_size=2
     - In update():
       ```python
       unscaled_d2xdt2 = value - 2.0 * buffer[0] + buffer[1]
       update_max = (buffer[1] != 0.0) and (unscaled_d2xdt2 > buffer[2])
       update_min = (buffer[1] != 0.0) and (unscaled_d2xdt2 < buffer[3])
       buffer[2] = selp(update_max, unscaled_d2xdt2, buffer[2])
       buffer[3] = selp(update_min, unscaled_d2xdt2, buffer[3])
       buffer[1] = buffer[0]
       buffer[0] = value
       ```
     - In save():
       ```python
       dt_save_sq = dt_save * dt_save
       output_array[0] = buffer[2] / dt_save_sq  # max
       output_array[1] = buffer[3] / dt_save_sq  # min
       buffer[2] = 0.0
       buffer[3] = 0.0
       buffer[1] = 0.0
       buffer[0] = 0.0
       ```
   - Edge cases: Same as individual metrics
   - Integration: Auto-substituted when both d2xdt2_max and d2xdt2_min requested

**Outcomes**:
[Empty - to be filled by do_task agent]

---

## Task Group 6: Update Metric Imports - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Groups 5, 5.2, 5.3, 5.4, 5.5, 5.6

**Required Context**:
- File: src/cubie/outputhandling/summarymetrics/__init__.py (lines 16-28, imports)

**Input Validation Required**:
- None

**Tasks**:

### 6.1 Add New Metric Imports
   - File: src/cubie/outputhandling/summarymetrics/__init__.py
   - Action: Modify
   - Details:
     Add imports after line 28:
     ```python
     from cubie.outputhandling.summarymetrics import dxdt_max  # noqa
     from cubie.outputhandling.summarymetrics import dxdt_min  # noqa
     from cubie.outputhandling.summarymetrics import dxdt_extrema  # noqa
     from cubie.outputhandling.summarymetrics import d2xdt2_max  # noqa
     from cubie.outputhandling.summarymetrics import d2xdt2_min  # noqa
     from cubie.outputhandling.summarymetrics import d2xdt2_extrema  # noqa
     ```
   - Edge cases: Import triggers @register_metric decoration
   - Integration: Metrics available in summary_metrics registry

**Outcomes**:
[Empty - to be filled by do_task agent]

---

## Task Group 7: Add Tests for New Metrics - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Groups 5, 5.2, 5.3, 5.4, 5.5, 5.6, 6

**Required Context**:
- File: tests/outputhandling/summarymetrics/test_summary_metrics.py (entire file - understand test patterns)
- Specific functions to update will be identified after reading file

**Input Validation Required**:
- None (tests validate metric behavior)

**Tasks**:

### 7.1 Add Derivative Metrics to Test Infrastructure
   - File: tests/outputhandling/summarymetrics/test_summary_metrics.py
   - Action: Modify
   - Details:
     AFTER reading the file, update the following (exact locations TBD):
     
     1. Add validation logic to `calculate_single_summary_array` helper:
        - For dxdt_max: `np.max(np.diff(values) / dt_save)`
        - For dxdt_min: `np.min(np.diff(values) / dt_save)`
        - For dxdt_extrema: both max and min in array
        - For d2xdt2_max: `np.max((values[2:] - 2*values[1:-1] + values[:-2]) / dt_save**2)`
        - For d2xdt2_min: `np.min((values[2:] - 2*values[1:-1] + values[:-2]) / dt_save**2)`
        - For d2xdt2_extrema: both max and min in array
     
     2. Add to `test_all_summary_metrics_numerical_check` parametrization:
        - Add "dxdt_max", "dxdt_min", "d2xdt2_max", "d2xdt2_min" to individual test cases
        - Add combined cases: ["dxdt_max", "dxdt_min"], ["d2xdt2_max", "d2xdt2_min"]
     
     3. The `test_all_summaries_long_run` should auto-discover new metrics via summary_metrics object
   - Edge cases: First/last points may need special handling for finite differences
   - Integration: Validates against numpy reference implementations

**Outcomes**:
[Empty - to be filled by do_task agent]

---

## Summary

**Total Task Groups**: 7 main groups (1-4, 5 with 6 parallel subgroups, 6, 7)

**Dependency Chain Overview**:
```
Group 1 (MetricConfig) ─┬─> Group 2 (update methods) ─┬─> Group 4 (OutputFunctions) ─┐
                        │                               │                              │
Group 3 (OutputConfig) ─┴──────────────────────────────┘                              │
                                                                                        │
Group 1 + Group 2 ──> Groups 5.1-5.6 (6 new metrics in parallel) ──> Group 6 (imports) ─┤
                                                                                        │
                                                                                        ├─> Group 7 (tests)
                                                                                        │
                                                                                        └─> DONE
```

**Parallel Execution Opportunities**:
- Group 3 can run parallel with Groups 1-2
- Groups 5.1, 5.2, 5.3, 5.4, 5.5, 5.6 can ALL run in parallel (6 metrics simultaneously)

**Estimated Complexity**: Medium
- Infrastructure changes: Simple (replace placeholder, add update methods, add field)
- New metrics: Straightforward (follow existing patterns with predicated commit)
- Testing: Moderate (add validation logic and test cases)
- NO changes to existing metrics (major simplification vs old approach)

**Files Modified**: 5
- src/cubie/outputhandling/summarymetrics/metrics.py
- src/cubie/outputhandling/output_config.py
- src/cubie/outputhandling/output_functions.py
- src/cubie/outputhandling/summarymetrics/__init__.py
- tests/outputhandling/summarymetrics/test_summary_metrics.py

**Files Created**: 6
- src/cubie/outputhandling/summarymetrics/dxdt_max.py
- src/cubie/outputhandling/summarymetrics/dxdt_min.py
- src/cubie/outputhandling/summarymetrics/dxdt_extrema.py
- src/cubie/outputhandling/summarymetrics/d2xdt2_max.py
- src/cubie/outputhandling/summarymetrics/d2xdt2_min.py
- src/cubie/outputhandling/summarymetrics/d2xdt2_extrema.py
