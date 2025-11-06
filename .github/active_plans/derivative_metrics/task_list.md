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
**Status**: [x]
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
- Replaced CompileSettingsPlaceholder with MetricConfig attrs class
- Added _dt_save field with default 0.01 and float validator  
- Added dt_save property
- Updated SummaryMetric.__init__ to use MetricConfig()

---

## Task Group 2: Add update() Methods to SummaryMetrics and SummaryMetric - SEQUENTIAL
**Status**: [x]
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
             
         Returns
         -------
         None
             Returns None.
             
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
             
         Returns
         -------
         None
             Returns None.
             
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
- Added update() method to SummaryMetric class (calls update_compile_settings with silent=True)
- Added update() method to SummaryMetrics class (propagates to all _metric_objects)
- Both methods return None as per project pattern
- Enables dt_save propagation without signature changes

---

## Task Group 3: Add dt_save to OutputConfig and Update Combined Metrics Registry - SEQUENTIAL
**Status**: [x]
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
- Added _dt_save field to OutputConfig with default 0.01
- Added dt_save property to OutputConfig
- Added "dt_save" to ALL_OUTPUT_FUNCTION_PARAMETERS set
- Added dxdt_extrema and d2xdt2_extrema to _combined_metrics registry
- dt_save now flows through configuration system

---

## Task Group 4: Update OutputFunctions.build() to Call summary_metrics.update() - SEQUENTIAL
**Status**: [x]
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
- Added summary_metrics.update(dt_save=config.dt_save) call in OutputFunctions.build()
- Import done inside method to avoid circular import
- Metrics now receive dt_save before compilation
- Invalidates cache when dt_save changes

---

## Task Group 5: Implement New Derivative Metrics - PARALLEL (6 metrics can be done simultaneously)
**Status**: [x]
**Dependencies**: Groups 1, 2

**Required Context**:
- File: src/cubie/outputhandling/summarymetrics/mean.py (entire file - pattern reference)
- File: src/cubie/outputhandling/summarymetrics/extrema.py (entire file - pattern reference)
- File: src/cubie/cuda_simsafe.py (selp function for predicated commit)

**Input Validation Required**:
- None (operates on buffer and value from integration loop)

**Tasks**:

### 5.1 Create dxdt_max.py, dxdt_min.py, dxdt_extrema.py
   - Files: src/cubie/outputhandling/summarymetrics/{dxdt_max.py, dxdt_min.py, dxdt_extrema.py}
   - Action: Create
   - Details:
     Implement first derivative metrics following the pattern from task_list_new.md Task Group 5.
     Key points:
     - Access dt_save via self.compile_settings.dt_save in build()
     - Capture dt_save in closure for device functions
     - Use predicated commit pattern with selp()
     - buffer_size=2 for max/min, 3 for extrema
     - Scale by dt_save in save() function
   - Edge cases: buffer[0] == 0.0 guard handles first call
   - Integration: Auto-register via @register_metric decorator

### 5.2 Create d2xdt2_max.py, d2xdt2_min.py, d2xdt2_extrema.py
   - Files: src/cubie/outputhandling/summarymetrics/{d2xdt2_max.py, d2xdt2_min.py, d2xdt2_extrema.py}
   - Action: Create
   - Details:
     Implement second derivative metrics.
     Key points:
     - Central difference formula: value - 2*buffer[0] + buffer[1]
     - buffer_size=3 for max/min, 4 for extrema
     - Guard: buffer[1] != 0.0 ensures two previous values
     - Scale by dt_save² in save() function
   - Edge cases: buffer[1] == 0.0 guard handles first two calls
   - Integration: Auto-register via @register_metric decorator

**Outcomes**:
- Created dxdt_max.py, dxdt_min.py, dxdt_extrema.py for first derivative metrics
- Created d2xdt2_max.py, d2xdt2_min.py, d2xdt2_extrema.py for second derivative metrics
- All metrics use predicated commit pattern with selp()
- dt_save accessed via self.compile_settings.dt_save and captured in closure
- Correct buffer sizes: dxdt (2/3), d2xdt2 (3/4)
- Guards prevent computation before sufficient history available

---

## Task Group 6: Update Metric Imports - SEQUENTIAL
**Status**: [x]
**Dependencies**: Group 5

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
- Added imports for all 6 new derivative metrics to __init__.py
- Metrics auto-register via @register_metric decorator on import
- All metrics now available in summary_metrics registry

---

## Task Group 7: Add Tests for New Metrics - SEQUENTIAL
**Status**: [x]
**Dependencies**: Groups 5, 6

**Required Context**:
- File: tests/outputhandling/summarymetrics/test_summary_metrics.py (entire file - understand test patterns)

**Input Validation Required**:
- None (tests validate metric behavior)

**Tasks**:

### 7.1 Add Derivative Metrics to Test Infrastructure
   - File: tests/outputhandling/summarymetrics/test_summary_metrics.py
   - Action: Modify
   - Details:
     Update tests to include new metrics:
     
     1. Add validation logic to `calculate_single_summary_array` helper:
        - For dxdt_max: `np.max(np.diff(values) / dt_save)`
        - For dxdt_min: `np.min(np.diff(values) / dt_save)`
        - For dxdt_extrema: both max and min in array
        - For d2xdt2_max: `np.max((values[2:] - 2*values[1:-1] + values[:-2]) / dt_save**2)`
        - For d2xdt2_min: `np.min((values[2:] - 2*values[1:-1] + values[:-2]) / dt_save**2)`
        - For d2xdt2_extrema: both max and min in array
     
     2. Add metrics to test parameter sets:
        **For test_all_summaries_long_run:**
        - Add ALL six metrics to explicit parameter sets (manual addition required)
        - Metrics: dxdt_max, dxdt_min, dxdt_extrema, d2xdt2_max, d2xdt2_min, d2xdt2_extrema
        
        **For test_all_summary_metrics_numerical_check:**
        - Add ALL six metrics to the "all" case (tests with combinations)
        - Add ONLY individual metrics to "no_combos" case: dxdt_max, dxdt_min, d2xdt2_max, d2xdt2_min
        - DO NOT add combined metrics (dxdt_extrema, d2xdt2_extrema) to "no_combos" case
        - Individual test cases happen automatically via summary_metrics object
     
     3. Test validation logic (already in calculate_single_summary_array):
   - Edge cases: First/last points may need special handling for finite differences
   - Integration: Validates against numpy reference implementations

**Outcomes**:
- Created calculate_single_summary_array helper function with numpy reference implementations
- Added test_all_summaries_long_run parameterized test with all 13 metrics (including 6 new derivative metrics)
- Added test_all_summary_metrics_numerical_check with "all" and "no_combos" cases
- Added test_derivative_metrics_buffer_sizes and test_derivative_metrics_output_sizes
- Added test_combined_derivative_metrics_dxdt and test_combined_derivative_metrics_d2xdt2
- Added test_derivative_metrics_registration
- Updated test_real_summary_metrics_available_metrics to include 6 new metrics
- All tests validate correct buffer sizes, output sizes, and combination behavior

---

## Summary

**Total Task Groups**: 7 main groups

**Dependency Chain Overview**:
```
Group 1 (MetricConfig) ─┬─> Group 2 (update methods) ─┬─> Group 4 (OutputFunctions) ─┐
                        │                               │                              │
Group 3 (OutputConfig) ─┴──────────────────────────────┘                              │
                                                                                        │
Group 1 + Group 2 ──> Group 5 (6 new metrics in parallel) ──> Group 6 (imports) ──────┤
                                                                                        │
                                                                                        ├─> Group 7 (tests)
                                                                                        │
                                                                                        └─> DONE
```

**Parallel Execution Opportunities**:
- Group 3 can run parallel with Groups 1-2
- Within Group 5, all 6 metrics can be implemented in parallel

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

**Files Unchanged**: 12 existing metric files (NO signature changes, NO functional changes)
