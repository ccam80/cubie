# Implementation Task List
# Feature: Derivative-Based Output Metrics
# Plan Reference: .github/active_plans/derivative_metrics/agent_plan.md

## Overview

This task list implements six new summary metrics (dxdt_max, dxdt_min, dxdt_extrema, d2xdt2_max, d2xdt2_min, d2xdt2_extrema) that compute derivatives via finite differences. The implementation requires breaking architectural changes to ALL metric signatures to pass dt_save as a compile-time parameter.

**Critical Breaking Changes:**
- ALL 12 existing metric build() methods must change from `build(self)` to `build(self, dt_save: float)`
- Device function customisable_variable changes from int32 to float32/64 in ALL metrics
- OutputFunctions must pass dt_save to metric build() calls

**Dependencies:**
Task groups must be executed SEQUENTIALLY due to architectural changes affecting all components.

---

## Task Group 1: Update Base Infrastructure - SEQUENTIAL
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: src/cubie/outputhandling/summarymetrics/metrics.py (lines 65-152)
- File: src/cubie/outputhandling/output_functions.py (lines 176-222)
- File: src/cubie/outputhandling/summarymetrics/__init__.py (entire file)

**Input Validation Required**:
- dt_save: Check type is float, value > 0.0

**Tasks**:

### 1.1 Update SummaryMetric Abstract Base Class
   - File: src/cubie/outputhandling/summarymetrics/metrics.py
   - Action: Modify
   - Details:
     ```python
     # Update abstract method signature at line 121
     @abstractmethod
     def build(self, dt_save: float) -> MetricFuncCache:
         """Generate CUDA device functions for the metric.
         
         Parameters
         ----------
         dt_save
             float. Time interval between consecutive saved states.
         
         Returns
         -------
         MetricFuncCache
             Cache containing device update and save functions compiled for
             CUDA execution.
         
         Notes
         -----
         Implementations must return functions with the signatures
         ``update(value, buffer, current_index, customisable_variable)`` and
         ``save(buffer, output_array, summarise_every, customisable_variable)``.
         Each callback needs ``@cuda.jit(..., device=True, inline=True)``
         decoration supporting both single- and double-precision input.
         
         The customisable_variable parameter is now float32/64 (changed from
         int32) to accommodate dt_save passing for derivative metrics.
         """
         pass
     ```
   - Edge cases: 
     - Ensure dt_save parameter is documented as required
     - Note type change for customisable_variable in docstring
   - Integration: All subclasses must update to match this signature

### 1.2 Add New Combined Metrics to Registry
   - File: src/cubie/outputhandling/summarymetrics/metrics.py
   - Action: Modify
   - Details:
     ```python
     # Update __attrs_post_init__ method around line 203
     def __attrs_post_init__(self) -> None:
         """Reset the parsed parameter cache and define combined metrics."""
         
         self._params = {}
         # Define combined metrics registry:
         # Maps frozenset of individual metrics to the combined metric name
         # Only combine when ALL constituent parts are requested
         # This ensures user gets exactly what they requested
         self._combined_metrics = {
             frozenset(["mean", "std", "rms"]): "mean_std_rms",
             frozenset(["mean", "std"]): "mean_std",
             frozenset(["std", "rms"]): "std_rms",
             frozenset(["max", "min"]): "extrema",
             frozenset(["dxdt_max", "dxdt_min"]): "dxdt_extrema",  # NEW
             frozenset(["d2xdt2_max", "d2xdt2_min"]): "d2xdt2_extrema",  # NEW
         }
     ```
   - Edge cases: Ensure both constituents must be present for substitution
   - Integration: Auto-substitution logic in preprocess_request() already handles this

### 1.3 Update OutputFunctions to Pass dt_save to Metrics
   - File: src/cubie/outputhandling/output_functions.py
   - Action: Modify
   - Details:
     ```python
     # In build() method around lines 176-222
     # Modify update_summary_factory call around line 204
     # Add dt_save parameter extraction and passing
     
     def build(self) -> OutputFunctionCache:
         """Compile output functions and calculate memory requirements.
         
         Returns
         -------
         OutputFunctionCache
             Container with compiled functions that target the current
             configuration.
         
         Notes
         -----
         This method is invoked lazily by :class:`cubie.CUDAFactory` the first
         time a compiled function is requested. The resulting cache is reused
         until configuration settings change.
         """
         config = self.compile_settings
         
         buffer_sizes = self.summaries_buffer_sizes
         
         # Extract dt_save from compile settings
         # dt_save is available via config which should have access to loop settings
         # Need to access from self.compile_settings or parent context
         # This requires investigation of how dt_save flows to OutputFunctions
         
         # Build functions using output sizes objects
         save_state_func = save_state_factory(
             config.saved_state_indices,
             config.saved_observable_indices,
             config.save_state,
             config.save_observables,
             config.save_time,
         )
         
         # CRITICAL: Need to determine how to access dt_save here
         # Options:
         # 1. Add dt_save to OutputConfig
         # 2. Add dt_save to OutputFunctions.__init__
         # 3. Access from parent context (SingleIntegratorRunCore)
         # 
         # Recommended: Add dt_save to OutputConfig and OutputFunctions.__init__
         
         update_summary_metrics_func = update_summary_factory(
             buffer_sizes,
             config.summarised_state_indices,
             config.summarised_observable_indices,
             config.summary_types,
             # dt_save,  # NEW PARAMETER - Need to pass through
         )
         
         save_summary_metrics_func = save_summary_factory(
             buffer_sizes,
             config.summarised_state_indices,
             config.summarised_observable_indices,
             config.summary_types,
             # dt_save,  # NEW PARAMETER - Need to pass through
         )
         
         return OutputFunctionCache(
             save_state_function=save_state_func,
             update_summaries_function=update_summary_metrics_func,
             save_summaries_function=save_summary_metrics_func,
         )
     ```
   - Edge cases:
     - **CRITICAL DESIGN DECISION NEEDED**: How to access dt_save in OutputFunctions.build()?
     - User must clarify preferred approach before implementation
   - Integration: This affects update_summary_factory and save_summary_factory signatures

**Outcomes**:
[Empty - to be filled by do_task agent]

---

## Task Group 2: Update ALL Existing Metric Signatures - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Group 1

**Required Context**:
- File: src/cubie/outputhandling/summarymetrics/max.py (lines 36-121)
- File: src/cubie/outputhandling/summarymetrics/min.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/mean.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/rms.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/std.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/peaks.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/negative_peaks.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/max_magnitude.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/extrema.py (lines 37-127)
- File: src/cubie/outputhandling/summarymetrics/mean_std.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/std_rms.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/mean_std_rms.py (entire file)

**Input Validation Required**:
- dt_save: Check type is float (validation in base class, not individual metrics)
- No additional validation needed in individual metrics (parameter unused)

**Tasks**:

### 2.1 Update max.py
   - File: src/cubie/outputhandling/summarymetrics/max.py
   - Action: Modify
   - Details:
     ```python
     # Change build() signature at line 36
     def build(self, dt_save: float) -> MetricFuncCache:
         """Generate CUDA device functions for maximum value calculation.
         
         Parameters
         ----------
         dt_save
             float. Time interval between saved states (unused for max).
         
         Returns
         -------
         MetricFuncCache
             Cache containing the device update and save callbacks.
         
         Notes
         -----
         The update callback keeps the running maximum while the save callback
         writes the result and resets the buffer sentinel.
         """
         
         # Update device function signatures (int32 -> float32/64 for last param)
         @cuda.jit(
             [
                 "float32, float32[::1], int32, float32",  # Changed from int32
                 "float64, float64[::1], int32, float64",  # Changed from int32
             ],
             device=True,
             inline=True,
         )
         def update(
             value,
             buffer,
             current_index,
             customisable_variable,  # Now float32/64
         ):
             """Update the running maximum with a new value."""
             if value > buffer[0]:
                 buffer[0] = value
         
         @cuda.jit(
             [
                 "float32[::1], float32[::1], int32, float32",  # Changed from int32
                 "float64[::1], float64[::1], int32, float64",  # Changed from int32
             ],
             device=True,
             inline=True,
         )
         def save(
             buffer,
             output_array,
             summarise_every,
             customisable_variable,  # Now float32/64
         ):
             """Save the maximum value to output and reset the buffer."""
             output_array[0] = buffer[0]
             buffer[0] = -1.0e30
         
         return MetricFuncCache(update=update, save=save)
     ```
   - Edge cases: None (dt_save unused but must accept parameter)
   - Integration: Matches new base class signature

### 2.2 Update min.py
   - File: src/cubie/outputhandling/summarymetrics/min.py
   - Action: Modify
   - Details: Same pattern as max.py - add dt_save parameter, change customisable_variable to float32/64
   - Edge cases: None
   - Integration: Matches new base class signature

### 2.3 Update mean.py
   - File: src/cubie/outputhandling/summarymetrics/mean.py
   - Action: Modify
   - Details: Same pattern - add dt_save parameter, change customisable_variable to float32/64
   - Edge cases: None
   - Integration: Matches new base class signature

### 2.4 Update rms.py
   - File: src/cubie/outputhandling/summarymetrics/rms.py
   - Action: Modify
   - Details: Same pattern - add dt_save parameter, change customisable_variable to float32/64
   - Edge cases: None
   - Integration: Matches new base class signature

### 2.5 Update std.py
   - File: src/cubie/outputhandling/summarymetrics/std.py
   - Action: Modify
   - Details: Same pattern - add dt_save parameter, change customisable_variable to float32/64
   - Edge cases: None
   - Integration: Matches new base class signature

### 2.6 Update peaks.py
   - File: src/cubie/outputhandling/summarymetrics/peaks.py
   - Action: Modify
   - Details: Same pattern - add dt_save parameter, change customisable_variable to float32/64
   - Edge cases: None
   - Integration: Matches new base class signature

### 2.7 Update negative_peaks.py
   - File: src/cubie/outputhandling/summarymetrics/negative_peaks.py
   - Action: Modify
   - Details: Same pattern - add dt_save parameter, change customisable_variable to float32/64
   - Edge cases: None
   - Integration: Matches new base class signature

### 2.8 Update max_magnitude.py
   - File: src/cubie/outputhandling/summarymetrics/max_magnitude.py
   - Action: Modify
   - Details: Same pattern - add dt_save parameter, change customisable_variable to float32/64
   - Edge cases: None
   - Integration: Matches new base class signature

### 2.9 Update extrema.py
   - File: src/cubie/outputhandling/summarymetrics/extrema.py
   - Action: Modify
   - Details: Same pattern - add dt_save parameter, change customisable_variable to float32/64
   - Edge cases: None
   - Integration: Matches new base class signature

### 2.10 Update mean_std.py
   - File: src/cubie/outputhandling/summarymetrics/mean_std.py
   - Action: Modify
   - Details: Same pattern - add dt_save parameter, change customisable_variable to float32/64
   - Edge cases: None
   - Integration: Matches new base class signature

### 2.11 Update std_rms.py
   - File: src/cubie/outputhandling/summarymetrics/std_rms.py
   - Action: Modify
   - Details: Same pattern - add dt_save parameter, change customisable_variable to float32/64
   - Edge cases: None
   - Integration: Matches new base class signature

### 2.12 Update mean_std_rms.py
   - File: src/cubie/outputhandling/summarymetrics/mean_std_rms.py
   - Action: Modify
   - Details: Same pattern - add dt_save parameter, change customisable_variable to float32/64
   - Edge cases: None
   - Integration: Matches new base class signature

**Outcomes**:
[Empty - to be filled by do_task agent]

---

## Task Group 3: Implement First Derivative Metrics - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Groups 1, 2

**Required Context**:
- File: src/cubie/outputhandling/summarymetrics/max.py (lines 1-122) - Reference pattern
- File: src/cubie/outputhandling/summarymetrics/extrema.py (lines 1-128) - Combined metric pattern
- File: .github/active_plans/derivative_metrics/agent_plan.md (lines 9-126) - Specifications

**Input Validation Required**:
- None (all validation done at factory level)
- Assumption: dt_save > 0.0 enforced by solver configuration
- Assumption: Buffers are zeroed before first use

**Tasks**:

### 3.1 Create dxdt_max.py
   - File: src/cubie/outputhandling/summarymetrics/dxdt_max.py
   - Action: Create
   - Details:
     ```python
     """
     Maximum first derivative summary metric for CUDA-accelerated batch integration.
     
     This module implements a summary metric that tracks the maximum first
     derivative (slope) encountered during integration using backward finite
     differences of consecutive saved state values.
     """
     
     from numba import cuda
     
     from cubie.outputhandling.summarymetrics import summary_metrics
     from cubie.outputhandling.summarymetrics.metrics import (
         SummaryMetric,
         register_metric,
         MetricFuncCache,
     )
     
     
     @register_metric(summary_metrics)
     class DxdtMax(SummaryMetric):
         """Summary metric that tracks the maximum first derivative.
         
         Notes
         -----
         Uses finite differences: dxdt ≈ (x[n] - x[n-1]) / dt_save
         
         Buffer layout (3 elements per variable):
         - buffer[0]: previous state value (x[n-1])
         - buffer[1]: maximum unscaled dxdt encountered
         - buffer[2]: dt_save value (stored at first update)
         
         Scaling by dt_save done at save time to reduce roundoff error.
         
         First save point output is 0.0 (no previous value for derivative).
         """
         
         def __init__(self) -> None:
             """Initialise the DxdtMax summary metric with fixed buffer sizes."""
             super().__init__(
                 name="dxdt_max",
                 buffer_size=3,
                 output_size=1,
             )
         
         def build(self, dt_save: float) -> MetricFuncCache:
             """Generate CUDA device functions for maximum first derivative.
             
             Parameters
             ----------
             dt_save
                 float. Time interval between consecutive saved states.
             
             Returns
             -------
             MetricFuncCache
                 Cache containing the device update and save callbacks.
             
             Notes
             -----
             The update callback computes backward differences and tracks the
             maximum unscaled value. The save callback scales by dt_save and
             writes the result.
             """
             
             # no cover: start
             @cuda.jit(
                 [
                     "float32, float32[::1], int32, float32",
                     "float64, float64[::1], int32, float64",
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
                 """Update the running maximum first derivative.
                 
                 Parameters
                 ----------
                 value
                     float. Current state value.
                 buffer
                     device array. Buffer [prev_value, max_dxdt, dt_save].
                 current_index
                     int. Current integration step index (unused).
                 customisable_variable
                     float. Metric parameter placeholder (unused).
                 
                 Notes
                 -----
                 Uses buffer[0] == 0.0 as guard for first call or post-save.
                 Computes unscaled difference and updates max.
                 All code paths reach end (no early returns).
                 """
                 if buffer[0] == 0.0:
                     buffer[0] = value
                     buffer[2] = customisable_variable
                 else:
                     unscaled_dxdt = value - buffer[0]
                     if unscaled_dxdt > buffer[1]:
                         buffer[1] = unscaled_dxdt
                     buffer[0] = value
             
             @cuda.jit(
                 [
                     "float32[::1], float32[::1], int32, float32",
                     "float64[::1], float64[::1], int32, float64",
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
                 """Save scaled maximum first derivative to output.
                 
                 Parameters
                 ----------
                 buffer
                     device array. Buffer [prev_value, max_dxdt, dt_save].
                 output_array
                     device array. Output location for max_dxdt.
                 summarise_every
                     int. Number of steps between saves (unused).
                 customisable_variable
                     float. Metric parameter placeholder (unused).
                 
                 Notes
                 -----
                 Scales unscaled maximum by dt_save and writes to output.
                 Resets max and previous value for next period.
                 """
                 output_array[0] = buffer[1] / buffer[2]
                 buffer[1] = 0.0
                 buffer[0] = 0.0
             
             # no cover: end
             return MetricFuncCache(update=update, save=save)
     ```
   - Edge cases:
     - First save outputs 0.0 (buffer[1] is zeroed, no derivative yet)
     - Division by buffer[2] (dt_save) assumed safe (> 0.0 by solver validation)
     - Zero guard (buffer[0] == 0.0) works if buffers zeroed or after reset
   - Integration: Auto-registers via decorator, follows existing metric pattern

### 3.2 Create dxdt_min.py
   - File: src/cubie/outputhandling/summarymetrics/dxdt_min.py
   - Action: Create
   - Details: Same as dxdt_max.py but track minimum instead of maximum
     ```python
     # Key differences from dxdt_max:
     # - Class name: DxdtMin
     # - Metric name: "dxdt_min"
     # - In update(): if unscaled_dxdt < buffer[1]: (instead of >)
     # - Docstrings reference minimum instead of maximum
     ```
   - Edge cases: Same as dxdt_max
   - Integration: Auto-registers via decorator

### 3.3 Create dxdt_extrema.py
   - File: src/cubie/outputhandling/summarymetrics/dxdt_extrema.py
   - Action: Create
   - Details:
     ```python
     """
     First derivative extrema (max and min) summary metric for CUDA-accelerated
     batch integration.
     
     This module implements a combined summary metric that tracks both maximum
     and minimum first derivatives when both are requested.
     """
     
     from numba import cuda
     
     from cubie.outputhandling.summarymetrics import summary_metrics
     from cubie.outputhandling.summarymetrics.metrics import (
         SummaryMetric,
         register_metric,
         MetricFuncCache,
     )
     
     
     @register_metric(summary_metrics)
     class DxdtExtrema(SummaryMetric):
         """Summary metric that tracks both max and min first derivatives.
         
         Notes
         -----
         Uses finite differences: dxdt ≈ (x[n] - x[n-1]) / dt_save
         
         Buffer layout (4 elements per variable):
         - buffer[0]: previous state value (x[n-1])
         - buffer[1]: maximum unscaled dxdt encountered
         - buffer[2]: minimum unscaled dxdt encountered
         - buffer[3]: dt_save value (stored at first update)
         
         Output layout (2 elements):
         - output[0]: max_dxdt
         - output[1]: min_dxdt
         
         Auto-substituted when both dxdt_max and dxdt_min are requested.
         """
         
         def __init__(self) -> None:
             """Initialise the DxdtExtrema summary metric."""
             super().__init__(
                 name="dxdt_extrema",
                 buffer_size=4,
                 output_size=2,
             )
         
         def build(self, dt_save: float) -> MetricFuncCache:
             """Generate CUDA device functions for first derivative extrema.
             
             Parameters
             ----------
             dt_save
                 float. Time interval between consecutive saved states.
             
             Returns
             -------
             MetricFuncCache
                 Cache containing the device update and save callbacks.
             """
             
             # no cover: start
             @cuda.jit(
                 [
                     "float32, float32[::1], int32, float32",
                     "float64, float64[::1], int32, float64",
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
                 """Update the running max and min first derivatives.
                 
                 Parameters
                 ----------
                 value
                     float. Current state value.
                 buffer
                     device array. Buffer [prev, max_dxdt, min_dxdt, dt_save].
                 current_index
                     int. Current integration step index (unused).
                 customisable_variable
                     float. Metric parameter placeholder (unused).
                 """
                 if buffer[0] == 0.0:
                     buffer[0] = value
                     buffer[3] = customisable_variable
                 else:
                     unscaled_dxdt = value - buffer[0]
                     if unscaled_dxdt > buffer[1]:
                         buffer[1] = unscaled_dxdt
                     if unscaled_dxdt < buffer[2]:
                         buffer[2] = unscaled_dxdt
                     buffer[0] = value
             
             @cuda.jit(
                 [
                     "float32[::1], float32[::1], int32, float32",
                     "float64[::1], float64[::1], int32, float64",
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
                 """Save scaled max and min first derivatives to output.
                 
                 Parameters
                 ----------
                 buffer
                     device array. Buffer [prev, max_dxdt, min_dxdt, dt_save].
                 output_array
                     device array. Output location for [max_dxdt, min_dxdt].
                 summarise_every
                     int. Number of steps between saves (unused).
                 customisable_variable
                     float. Metric parameter placeholder (unused).
                 """
                 output_array[0] = buffer[1] / buffer[3]
                 output_array[1] = buffer[2] / buffer[3]
                 buffer[1] = 0.0
                 buffer[2] = 0.0
                 buffer[0] = 0.0
             
             # no cover: end
             return MetricFuncCache(update=update, save=save)
     ```
   - Edge cases: Same as individual metrics
   - Integration: Auto-substituted via _combined_metrics registry

**Outcomes**:
[Empty - to be filled by do_task agent]

---

## Task Group 4: Implement Second Derivative Metrics - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Groups 1, 2, 3

**Required Context**:
- File: src/cubie/outputhandling/summarymetrics/dxdt_max.py (entire file from Group 3)
- File: .github/active_plans/derivative_metrics/agent_plan.md (lines 9-126)

**Input Validation Required**:
- None (same assumptions as first derivative metrics)

**Tasks**:

### 4.1 Create d2xdt2_max.py
   - File: src/cubie/outputhandling/summarymetrics/d2xdt2_max.py
   - Action: Create
   - Details:
     ```python
     """
     Maximum second derivative summary metric for CUDA-accelerated batch integration.
     
     This module implements a summary metric that tracks the maximum second
     derivative (acceleration) using three-point central finite differences.
     """
     
     from numba import cuda
     
     from cubie.outputhandling.summarymetrics import summary_metrics
     from cubie.outputhandling.summarymetrics.metrics import (
         SummaryMetric,
         register_metric,
         MetricFuncCache,
     )
     
     
     @register_metric(summary_metrics)
     class D2xdt2Max(SummaryMetric):
         """Summary metric that tracks the maximum second derivative.
         
         Notes
         -----
         Uses finite differences: d2xdt2 ≈ (x[n] - 2*x[n-1] + x[n-2]) / dt_save²
         
         Buffer layout (4 elements per variable):
         - buffer[0]: previous state value (x[n-1])
         - buffer[1]: previous previous state value (x[n-2])
         - buffer[2]: maximum unscaled d2xdt2 encountered
         - buffer[3]: dt_save value
         
         Scaling by dt_save² done at save time.
         
         First TWO save points output 0.0 (need 3 points for second derivative).
         """
         
         def __init__(self) -> None:
             """Initialise the D2xdt2Max summary metric."""
             super().__init__(
                 name="d2xdt2_max",
                 buffer_size=4,
                 output_size=1,
             )
         
         def build(self, dt_save: float) -> MetricFuncCache:
             """Generate CUDA device functions for maximum second derivative.
             
             Parameters
             ----------
             dt_save
                 float. Time interval between consecutive saved states.
             
             Returns
             -------
             MetricFuncCache
                 Cache containing the device update and save callbacks.
             """
             
             # no cover: start
             @cuda.jit(
                 [
                     "float32, float32[::1], int32, float32",
                     "float64, float64[::1], int32, float64",
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
                 """Update the running maximum second derivative.
                 
                 Parameters
                 ----------
                 value
                     float. Current state value.
                 buffer
                     device array. Buffer [x[n-1], x[n-2], max_d2xdt2, dt_save].
                 current_index
                     int. Current integration step index (unused).
                 customisable_variable
                     float. Metric parameter placeholder (unused).
                 
                 Notes
                 -----
                 Uses buffer[0] == 0.0 as guard for initialization.
                 Requires two previous values before computing derivative.
                 """
                 if buffer[0] == 0.0:
                     buffer[0] = value
                     buffer[3] = customisable_variable
                 elif buffer[1] == 0.0:
                     buffer[1] = buffer[0]
                     buffer[0] = value
                 else:
                     unscaled_d2xdt2 = value - 2.0 * buffer[0] + buffer[1]
                     if unscaled_d2xdt2 > buffer[2]:
                         buffer[2] = unscaled_d2xdt2
                     buffer[1] = buffer[0]
                     buffer[0] = value
             
             @cuda.jit(
                 [
                     "float32[::1], float32[::1], int32, float32",
                     "float64[::1], float64[::1], int32, float64",
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
                 """Save scaled maximum second derivative to output.
                 
                 Parameters
                 ----------
                 buffer
                     device array. Buffer [x[n-1], x[n-2], max_d2xdt2, dt_save].
                 output_array
                     device array. Output location for max_d2xdt2.
                 summarise_every
                     int. Number of steps between saves (unused).
                 customisable_variable
                     float. Metric parameter placeholder (unused).
                 
                 Notes
                 -----
                 Scales by dt_save² (buffer[3] * buffer[3]).
                 Resets max but preserves x[n-1] for continuity.
                 """
                 dt_save_sq = buffer[3] * buffer[3]
                 output_array[0] = buffer[2] / dt_save_sq
                 buffer[2] = 0.0
                 buffer[1] = 0.0
                 buffer[0] = 0.0
             
             # no cover: end
             return MetricFuncCache(update=update, save=save)
     ```
   - Edge cases:
     - First two saves output 0.0 (need 3 points for second derivative)
     - Three-state guard: buffer[0]==0.0, then buffer[1]==0.0, then compute
     - Reset all buffers at save to avoid stale x[n-2] values
   - Integration: Auto-registers via decorator

### 4.2 Create d2xdt2_min.py
   - File: src/cubie/outputhandling/summarymetrics/d2xdt2_min.py
   - Action: Create
   - Details: Same as d2xdt2_max.py but track minimum
     ```python
     # Key differences:
     # - Class name: D2xdt2Min
     # - Metric name: "d2xdt2_min"
     # - In update(): if unscaled_d2xdt2 < buffer[2]: (instead of >)
     ```
   - Edge cases: Same as d2xdt2_max
   - Integration: Auto-registers via decorator

### 4.3 Create d2xdt2_extrema.py
   - File: src/cubie/outputhandling/summarymetrics/d2xdt2_extrema.py
   - Action: Create
   - Details:
     ```python
     """
     Second derivative extrema (max and min) summary metric for CUDA-accelerated
     batch integration.
     """
     
     from numba import cuda
     
     from cubie.outputhandling.summarymetrics import summary_metrics
     from cubie.outputhandling.summarymetrics.metrics import (
         SummaryMetric,
         register_metric,
         MetricFuncCache,
     )
     
     
     @register_metric(summary_metrics)
     class D2xdt2Extrema(SummaryMetric):
         """Summary metric that tracks both max and min second derivatives.
         
         Notes
         -----
         Uses finite differences: d2xdt2 ≈ (x[n] - 2*x[n-1] + x[n-2]) / dt_save²
         
         Buffer layout (5 elements per variable):
         - buffer[0]: previous state value (x[n-1])
         - buffer[1]: previous previous state value (x[n-2])
         - buffer[2]: maximum unscaled d2xdt2
         - buffer[3]: minimum unscaled d2xdt2
         - buffer[4]: dt_save value
         
         Output layout (2 elements):
         - output[0]: max_d2xdt2
         - output[1]: min_d2xdt2
         """
         
         def __init__(self) -> None:
             """Initialise the D2xdt2Extrema summary metric."""
             super().__init__(
                 name="d2xdt2_extrema",
                 buffer_size=5,
                 output_size=2,
             )
         
         def build(self, dt_save: float) -> MetricFuncCache:
             """Generate CUDA device functions for second derivative extrema.
             
             Parameters
             ----------
             dt_save
                 float. Time interval between consecutive saved states.
             
             Returns
             -------
             MetricFuncCache
                 Cache containing the device update and save callbacks.
             """
             
             # no cover: start
             @cuda.jit(
                 [
                     "float32, float32[::1], int32, float32",
                     "float64, float64[::1], int32, float64",
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
                 """Update the running max and min second derivatives.
                 
                 Parameters
                 ----------
                 value
                     float. Current state value.
                 buffer
                     device array. [x[n-1], x[n-2], max, min, dt_save].
                 current_index
                     int. Current integration step index (unused).
                 customisable_variable
                     float. Metric parameter placeholder (unused).
                 """
                 if buffer[0] == 0.0:
                     buffer[0] = value
                     buffer[4] = customisable_variable
                 elif buffer[1] == 0.0:
                     buffer[1] = buffer[0]
                     buffer[0] = value
                 else:
                     unscaled_d2xdt2 = value - 2.0 * buffer[0] + buffer[1]
                     if unscaled_d2xdt2 > buffer[2]:
                         buffer[2] = unscaled_d2xdt2
                     if unscaled_d2xdt2 < buffer[3]:
                         buffer[3] = unscaled_d2xdt2
                     buffer[1] = buffer[0]
                     buffer[0] = value
             
             @cuda.jit(
                 [
                     "float32[::1], float32[::1], int32, float32",
                     "float64[::1], float64[::1], int32, float64",
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
                 """Save scaled max and min second derivatives to output.
                 
                 Parameters
                 ----------
                 buffer
                     device array. [x[n-1], x[n-2], max, min, dt_save].
                 output_array
                     device array. Output for [max_d2xdt2, min_d2xdt2].
                 summarise_every
                     int. Number of steps between saves (unused).
                 customisable_variable
                     float. Metric parameter placeholder (unused).
                 """
                 dt_save_sq = buffer[4] * buffer[4]
                 output_array[0] = buffer[2] / dt_save_sq
                 output_array[1] = buffer[3] / dt_save_sq
                 buffer[2] = 0.0
                 buffer[3] = 0.0
                 buffer[1] = 0.0
                 buffer[0] = 0.0
             
             # no cover: end
             return MetricFuncCache(update=update, save=save)
     ```
   - Edge cases: Same as individual d2xdt2 metrics
   - Integration: Auto-substituted via _combined_metrics registry

**Outcomes**:
[Empty - to be filled by do_task agent]

---

## Task Group 5: Update Module Imports - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Groups 1, 2, 3, 4

**Required Context**:
- File: src/cubie/outputhandling/summarymetrics/__init__.py (entire file)

**Input Validation Required**:
- None

**Tasks**:

### 5.1 Add New Metric Imports to __init__.py
   - File: src/cubie/outputhandling/summarymetrics/__init__.py
   - Action: Modify
   - Details:
     ```python
     # Add imports for new metrics (alphabetical order)
     # These imports trigger auto-registration via @register_metric decorator
     
     from cubie.outputhandling.summarymetrics.d2xdt2_extrema import D2xdt2Extrema
     from cubie.outputhandling.summarymetrics.d2xdt2_max import D2xdt2Max
     from cubie.outputhandling.summarymetrics.d2xdt2_min import D2xdt2Min
     from cubie.outputhandling.summarymetrics.dxdt_extrema import DxdtExtrema
     from cubie.outputhandling.summarymetrics.dxdt_max import DxdtMax
     from cubie.outputhandling.summarymetrics.dxdt_min import DxdtMin
     # ... existing imports ...
     ```
   - Edge cases: None
   - Integration: Import triggers decorator execution, registering metrics

**Outcomes**:
[Empty - to be filled by do_task agent]

---

## Task Group 6: Update Summary Factory Functions - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Groups 1-5

**Required Context**:
- File: src/cubie/outputhandling/update_summaries.py (lines 60-150)
- File: src/cubie/outputhandling/save_summaries.py (entire file)

**Input Validation Required**:
- dt_save: Check type is float, value > 0.0 (validation at factory call site)

**Tasks**:

### 6.1 Update update_summary_factory to Pass dt_save
   - File: src/cubie/outputhandling/update_summaries.py
   - Action: Modify
   - Details:
     ```python
     # Add dt_save parameter to factory function signature
     def update_summary_factory(
         buffer_sizes: SummariesBufferSizes,
         summarised_state_indices: ArrayLike,
         summarised_observable_indices: ArrayLike,
         summary_types: Sequence[str],
         dt_save: float,  # NEW PARAMETER
     ) -> Callable:
         """Build CUDA device function to update summary metrics.
         
         Parameters
         ----------
         buffer_sizes
             SummariesBufferSizes. Buffer sizing metadata.
         summarised_state_indices
             ArrayLike. Indices of state variables to summarise.
         summarised_observable_indices
             ArrayLike. Indices of observable variables to summarise.
         summary_types
             Sequence[str]. Metric names to compile.
         dt_save
             float. Time interval between saved states.
         
         Returns
         -------
         Callable
             CUDA device function for updating summaries.
         """
         
         # Get metric functions from registry
         metric_update_funcs = summary_metrics.update_functions(summary_types)
         metric_params = summary_metrics.params(summary_types)
         
         # CHANGE: Pass dt_save instead of metric_params to device functions
         # Replace metric_params with dt_save for all metrics
         # dt_save replaces the customisable_variable parameter
         
         # Build chained update function
         chained_func = chain_metrics(
             metric_update_funcs,
             buffer_offsets,
             buffer_sizes_list,
             [dt_save] * len(metric_update_funcs),  # CHANGED: dt_save for all
             inner_chain=do_nothing,
         )
         
         # Return compiled device function...
     ```
   - Edge cases:
     - **CRITICAL**: This changes how customisable_variable is used
     - All metrics now receive dt_save instead of their parameter
     - Parametrized metrics (e.g., peaks[n]) lose parameter passing
     - **DESIGN ISSUE**: May need to rethink parameter passing strategy
   - Integration: Affects all metric update calls

### 6.2 Update save_summary_factory to Pass dt_save
   - File: src/cubie/outputhandling/save_summaries.py
   - Action: Modify
   - Details: Same pattern as update_summary_factory - add dt_save parameter, pass to metrics
   - Edge cases: Same parameter passing issue as update_summary_factory
   - Integration: Affects all metric save calls

**CRITICAL DESIGN DECISION NEEDED:**
The agent_plan states customisable_variable changes to float32/64 for dt_save, but this breaks parametrized metrics (peaks[n]) that use customisable_variable for n. Two options:

1. **Option A**: Add separate dt_save parameter to device functions
   - Signature: `update(value, buffer, index, param, dt_save)`
   - Signature: `save(buffer, output, summarise_every, param, dt_save)`
   - Preserves parametrized metrics

2. **Option B**: Use customisable_variable for dt_save only
   - Parametrized metrics must get parameter another way (e.g., stored in buffer)
   - Breaking change for any custom parametrized metrics

**User must clarify before Group 6 implementation.**

**Outcomes**:
[Empty - to be filled by do_task agent]

---

## Task Group 7: Add Tests for New Metrics - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Groups 1-6

**Required Context**:
- File: tests/outputhandling/summarymetrics/test_summary_metrics.py (entire file)
- File: tests/outputhandling/test_output_functions.py (search for test_all_summaries)

**Input Validation Required**:
- None (test fixtures handle validation)

**Tasks**:

### 7.1 Add Validation Functions for Derivative Metrics
   - File: tests/outputhandling/summarymetrics/test_summary_metrics.py
   - Action: Modify
   - Details:
     ```python
     import numpy as np
     
     def validate_dxdt_max(state_values, dt_save):
         """Validate maximum first derivative.
         
         Parameters
         ----------
         state_values
             np.ndarray. Time series of state values.
         dt_save
             float. Time interval between saved states.
         
         Returns
         -------
         float
             Expected maximum first derivative.
         
         Notes
         -----
         Uses backward finite differences: dxdt[i] = (x[i] - x[i-1]) / dt_save
         First value is 0.0 (no previous state).
         """
         if len(state_values) < 2:
             return 0.0
         dxdt = np.diff(state_values) / dt_save
         return np.max(dxdt)
     
     def validate_dxdt_min(state_values, dt_save):
         """Validate minimum first derivative."""
         if len(state_values) < 2:
             return 0.0
         dxdt = np.diff(state_values) / dt_save
         return np.min(dxdt)
     
     def validate_d2xdt2_max(state_values, dt_save):
         """Validate maximum second derivative.
         
         Notes
         -----
         Uses three-point central differences:
         d2xdt2[i] = (x[i+1] - 2*x[i] + x[i-1]) / dt_save²
         First two values are 0.0 (need 3 points).
         """
         if len(state_values) < 3:
             return 0.0
         d2xdt2 = (state_values[2:] - 2 * state_values[1:-1] + 
                   state_values[:-2]) / (dt_save ** 2)
         return np.max(d2xdt2)
     
     def validate_d2xdt2_min(state_values, dt_save):
         """Validate minimum second derivative."""
         if len(state_values) < 3:
             return 0.0
         d2xdt2 = (state_values[2:] - 2 * state_values[1:-1] + 
                   state_values[:-2]) / (dt_save ** 2)
         return np.min(d2xdt2)
     
     def validate_dxdt_extrema(state_values, dt_save):
         """Validate first derivative extrema.
         
         Returns
         -------
         tuple[float, float]
             (max_dxdt, min_dxdt)
         """
         return (validate_dxdt_max(state_values, dt_save),
                 validate_dxdt_min(state_values, dt_save))
     
     def validate_d2xdt2_extrema(state_values, dt_save):
         """Validate second derivative extrema.
         
         Returns
         -------
         tuple[float, float]
             (max_d2xdt2, min_d2xdt2)
         """
         return (validate_d2xdt2_max(state_values, dt_save),
                 validate_d2xdt2_min(state_values, dt_save))
     ```
   - Edge cases:
     - First derivative: first output is 0.0
     - Second derivative: first two outputs are 0.0
     - Empty/short arrays handled gracefully
   - Integration: Called by parametrized tests

### 7.2 Add Parametrized Test Cases for New Metrics
   - File: tests/outputhandling/summarymetrics/test_summary_metrics.py
   - Action: Modify
   - Details:
     ```python
     # Add to existing parametrized test (if test_all_summary_metrics_numerical_check exists)
     # Otherwise create new test following existing pattern
     
     @pytest.mark.parametrize("metric_name,validation_func", [
         ("dxdt_max", validate_dxdt_max),
         ("dxdt_min", validate_dxdt_min),
         ("dxdt_extrema", validate_dxdt_extrema),
         ("d2xdt2_max", validate_d2xdt2_max),
         ("d2xdt2_min", validate_d2xdt2_min),
         ("d2xdt2_extrema", validate_d2xdt2_extrema),
         # ... existing metrics ...
     ])
     def test_derivative_metrics_numerical_validation(
         metric_name, validation_func, test_system, dt_save
     ):
         """Test derivative metrics against numpy reference implementations.
         
         Parameters
         ----------
         metric_name
             str. Name of metric to test.
         validation_func
             Callable. Reference implementation for validation.
         test_system
             Fixture. ODE system for testing.
         dt_save
             Fixture. Time interval between saves.
         
         Notes
         -----
         Runs integration with metric enabled and compares output against
         numpy-based reference calculation.
         """
         # Setup solver with metric
         # Run integration
         # Extract metric output and state time series
         # Compute reference values using validation_func
         # Assert np.allclose(metric_output, reference, rtol=1e-6)
     ```
   - Edge cases:
     - First save point(s) may be 0.0 for derivatives
     - Numerical precision tolerance (rtol=1e-6)
     - Both float32 and float64 precision
   - Integration: Uses existing test infrastructure

### 7.3 Verify Combined Metric Substitution
   - File: tests/outputhandling/summarymetrics/test_summary_metrics.py
   - Action: Modify
   - Details:
     ```python
     def test_dxdt_combined_metric_substitution(real_metrics):
         """Test that dxdt_max + dxdt_min auto-substitutes to dxdt_extrema.
         
         Parameters
         ----------
         real_metrics
             Fixture. SummaryMetrics registry instance.
         """
         request = ["dxdt_max", "dxdt_min"]
         processed = real_metrics.preprocess_request(request)
         assert "dxdt_extrema" in processed
         assert "dxdt_max" not in processed
         assert "dxdt_min" not in processed
     
     def test_d2xdt2_combined_metric_substitution(real_metrics):
         """Test that d2xdt2_max + d2xdt2_min auto-substitutes to d2xdt2_extrema."""
         request = ["d2xdt2_max", "d2xdt2_min"]
         processed = real_metrics.preprocess_request(request)
         assert "d2xdt2_extrema" in processed
         assert "d2xdt2_max" not in processed
         assert "d2xdt2_min" not in processed
     ```
   - Edge cases: Order independence of request list
   - Integration: Tests registry substitution logic

**Outcomes**:
[Empty - to be filled by do_task agent]

---

## Summary

**Total Task Groups**: 7
**Execution Strategy**: SEQUENTIAL (all groups depend on previous groups)
**Estimated Complexity**: High

**Dependency Chain:**
1. Group 1: Infrastructure (base class + registry + dt_save passing)
2. Group 2: Update ALL existing metrics (12 files)
3. Group 3: First derivative metrics (3 files)
4. Group 4: Second derivative metrics (3 files)
5. Group 5: Module imports (1 file)
6. Group 6: Factory functions (2 files)
7. Group 7: Tests (1-2 files)

**Critical Design Decisions Needed Before Implementation:**

1. **How to access dt_save in OutputFunctions.build()?**
   - Option A: Add dt_save to OutputConfig attributes
   - Option B: Add dt_save to OutputFunctions.__init__ parameters
   - Option C: Access from parent SingleIntegratorRunCore context
   
2. **How to handle customisable_variable parameter conflict?**
   - Option A: Add separate dt_save parameter to device functions (5 params instead of 4)
   - Option B: Use customisable_variable for dt_save, break parametrized metrics
   - Option C: Store parameter in buffer for parametrized metrics

**Files to Create**: 6 (dxdt_max, dxdt_min, dxdt_extrema, d2xdt2_max, d2xdt2_min, d2xdt2_extrema)

**Files to Modify**: 
- Infrastructure: 3 (metrics.py, output_functions.py, __init__.py)
- Existing metrics: 12 (all current metric files)
- Factories: 2 (update_summaries.py, save_summaries.py)
- Tests: 1-2 (test_summary_metrics.py, possibly test_output_functions.py)
- **Total: 18-19 files**

**Breaking Changes**: Yes - ALL metric signatures change, affecting any custom metrics

**Rollback Strategy**: 
1. Revert Groups 6-7 first (factories and tests)
2. Revert Groups 3-5 (new metrics and imports)
3. Revert Group 2 (existing metric signatures)
4. Revert Group 1 (infrastructure)

**Parallel Execution Opportunities**: None - architectural changes require sequential execution
