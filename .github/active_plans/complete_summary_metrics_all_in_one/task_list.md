# Implementation Task List
# Feature: Complete Summary Metrics Integration in all_in_one.py
# Plan Reference: .github/active_plans/complete_summary_metrics_all_in_one/agent_plan.md

---

## Task Group 1: Metric Registry Simulation Infrastructure - SEQUENTIAL
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/outputhandling/summarymetrics/mean.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/max.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/min.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/rms.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/std.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/mean_std.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/mean_std_rms.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/std_rms.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/extrema.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/peaks.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/negative_peaks.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/max_magnitude.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/dxdt_max.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/dxdt_min.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/dxdt_extrema.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/d2xdt2_max.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/d2xdt2_min.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/d2xdt2_extrema.py (entire file)

**Input Validation Required**:
- summaries_list: Validate it is a sequence (list or tuple), not None
- Each metric name in summaries_list: Validate it exists in implemented_metrics list

**Tasks**:

1. **Add implemented_metrics list**
   - File: tests/all_in_one.py
   - Action: Create
   - Location: After line 3378 (before "# SUMMARY METRIC FUNCTIONS" comment)
   - Details:
     ```python
     # =========================================================================
     # SUMMARY METRICS REGISTRY SIMULATION
     # =========================================================================
     
     # List of all implemented summary metrics (matches package registry)
     implemented_metrics = [
         "mean", "max", "min", "rms", "std",
         "mean_std", "mean_std_rms", "std_rms",
         "extrema", "peaks", "negative_peaks", "max_magnitude",
         "dxdt_max", "dxdt_min", "dxdt_extrema",
         "d2xdt2_max", "d2xdt2_min", "d2xdt2_extrema"
     ]
     ```
   - Edge cases: None
   - Integration: Used by configuration derivation and lookup functions

2. **Add metric buffer sizes dictionary**
   - File: tests/all_in_one.py
   - Action: Create
   - Location: After implemented_metrics list
   - Details:
     ```python
     # Buffer sizes per metric (number of slots needed per variable)
     METRIC_BUFFER_SIZES = {
         "mean": 1,
         "max": 1,
         "min": 1,
         "rms": 1,
         "std": 3,
         "mean_std": 3,
         "mean_std_rms": 3,
         "std_rms": 3,
         "extrema": 2,
         "peaks": 3,  # Base size, increases with customisable_variable
         "negative_peaks": 3,  # Base size, increases with customisable_variable
         "max_magnitude": 1,
         "dxdt_max": 2,
         "dxdt_min": 2,
         "dxdt_extrema": 3,
         "d2xdt2_max": 3,
         "d2xdt2_min": 3,
         "d2xdt2_extrema": 4,
     }
     ```
   - Edge cases: None
   - Integration: Used by buffer_sizes() lookup function

3. **Add metric output sizes dictionary**
   - File: tests/all_in_one.py
   - Action: Create
   - Location: After METRIC_BUFFER_SIZES
   - Details:
     ```python
     # Output sizes per metric (number of values written per variable)
     METRIC_OUTPUT_SIZES = {
         "mean": 1,
         "max": 1,
         "min": 1,
         "rms": 1,
         "std": 1,
         "mean_std": 2,
         "mean_std_rms": 3,
         "std_rms": 2,
         "extrema": 2,
         "peaks": 0,  # Base size, increases with customisable_variable
         "negative_peaks": 0,  # Base size, increases with customisable_variable
         "max_magnitude": 1,
         "dxdt_max": 1,
         "dxdt_min": 1,
         "dxdt_extrema": 2,
         "d2xdt2_max": 1,
         "d2xdt2_min": 1,
         "d2xdt2_extrema": 2,
     }
     ```
   - Edge cases: None
   - Integration: Used by output_sizes() lookup function

4. **Add buffer_sizes() lookup function**
   - File: tests/all_in_one.py
   - Action: Create
   - Location: After METRIC_OUTPUT_SIZES
   - Details:
     ```python
     def buffer_sizes(summaries_list):
         """Return list of buffer sizes for each metric in summaries_list.
         
         Parameters
         ----------
         summaries_list
             Sequence of summary metric names.
         
         Returns
         -------
         list
             Buffer size for each metric.
         """
         return [METRIC_BUFFER_SIZES[name] for name in summaries_list]
     ```
   - Edge cases: Empty summaries_list returns empty list
   - Integration: Called by update_summary_factory and save_summary_factory

5. **Add output_sizes() lookup function**
   - File: tests/all_in_one.py
   - Action: Create
   - Location: After buffer_sizes()
   - Details:
     ```python
     def output_sizes(summaries_list):
         """Return list of output sizes for each metric in summaries_list.
         
         Parameters
         ----------
         summaries_list
             Sequence of summary metric names.
         
         Returns
         -------
         list
             Output size for each metric.
         """
         return [METRIC_OUTPUT_SIZES[name] for name in summaries_list]
     ```
   - Edge cases: Empty summaries_list returns empty list
   - Integration: Called by save_summary_factory

6. **Add buffer_offsets() lookup function**
   - File: tests/all_in_one.py
   - Action: Create
   - Location: After output_sizes()
   - Details:
     ```python
     def buffer_offsets(summaries_list):
         """Return cumulative buffer offsets for each metric.
         
         Parameters
         ----------
         summaries_list
             Sequence of summary metric names.
         
         Returns
         -------
         list
             Starting buffer offset for each metric.
         """
         sizes = buffer_sizes(summaries_list)
         offsets = []
         cumulative = 0
         for size in sizes:
             offsets.append(cumulative)
             cumulative += size
         return offsets
     ```
   - Edge cases: Empty summaries_list returns empty list
   - Integration: Called by update_summary_factory and save_summary_factory

7. **Add output_offsets() lookup function**
   - File: tests/all_in_one.py
   - Action: Create
   - Location: After buffer_offsets()
   - Details:
     ```python
     def output_offsets(summaries_list):
         """Return cumulative output offsets for each metric.
         
         Parameters
         ----------
         summaries_list
             Sequence of summary metric names.
         
         Returns
         -------
         list
             Starting output offset for each metric.
         """
         sizes = output_sizes(summaries_list)
         offsets = []
         cumulative = 0
         for size in sizes:
             offsets.append(cumulative)
             cumulative += size
         return offsets
     ```
   - Edge cases: Empty summaries_list returns empty list
   - Integration: Called by save_summary_factory

8. **Add params() lookup function**
   - File: tests/all_in_one.py
   - Action: Create
   - Location: After output_offsets()
   - Details:
     ```python
     def params(summaries_list):
         """Return customisable_variable parameters for each metric.
         
         Parameters
         ----------
         summaries_list
             Sequence of summary metric names.
         
         Returns
         -------
         list
             Parameter value for each metric (0 for all current metrics).
         """
         return [0 for _ in summaries_list]
     ```
   - Edge cases: Empty summaries_list returns empty list
   - Integration: Called by update_summary_factory and save_summary_factory

**Outcomes**: 
- Files Modified:
  * tests/all_in_one.py (~150 lines added for registry infrastructure)
- Functions/Methods Added:
  * implemented_metrics list (18 metric names)
  * METRIC_BUFFER_SIZES dict (18 entries)
  * METRIC_OUTPUT_SIZES dict (18 entries)
  * buffer_sizes() lookup function
  * output_sizes() lookup function
  * buffer_offsets() lookup function
  * output_offsets() lookup function
  * params() lookup function
- Implementation Summary:
  Added complete registry simulation infrastructure matching package pattern.
  All lookup functions use dictionary-based approach for metric properties.
  Provides foundation for dynamic metric configuration.
- Issues Flagged: None

---

## Task Group 2: Summary Metric Device Functions (Basic Stats) - PARALLEL
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/outputhandling/summarymetrics/mean.py (lines 56-125)
- File: src/cubie/outputhandling/summarymetrics/max.py (lines 55-125)
- File: src/cubie/outputhandling/summarymetrics/min.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/rms.py (lines 57-134)
- File: src/cubie/outputhandling/summarymetrics/std.py (lines 59-149)
- File: tests/all_in_one.py (lines 3380-3446 for current mean implementation)

**Input Validation Required**:
None - device functions do not perform input validation

**Tasks**:

1. **Replace update_mean device function**
   - File: tests/all_in_one.py
   - Action: Modify
   - Location: Lines 3387-3410
   - Details: Replace existing update_mean with verbatim copy from mean.py lines 64-87
     ```python
     @cuda.jit(
         # [
         #     "float32, float32[::1], int32, int32",
         #     "float64, float64[::1], int32, int32",
         # ],
         device=True,
         inline=True,
         **compile_kwargs
     )
     def update_mean(
         value,
         buffer,
         current_index,
         customisable_variable,
     ):
         """Update the running sum with a new value.

         Parameters
         ----------
         value
             float. New value to add to the running sum.
         buffer
             device array. Location containing the running sum.
         current_index
             int. Current integration step index (unused for mean).
         customisable_variable
             int. Metric parameter placeholder (unused for mean).

         Notes
         -----
         Adds the new value to ``buffer[0]`` to maintain the running sum.
         """
         buffer[0] += value
     ```
   - Edge cases: None
   - Integration: Called by chained update functions

2. **Replace save_mean device function**
   - File: tests/all_in_one.py
   - Action: Modify
   - Location: Lines 3413-3445
   - Details: Replace existing save_mean with verbatim copy from mean.py lines 89-122
     ```python
     @cuda.jit(
         # [
         #     "float32[::1], float32[::1], int32, int32",
         #     "float64[::1], float64[::1], int32, int32",
         # ],
         device=True,
         inline=True,
         **compile_kwargs
     )
     def save_mean(
         buffer,
         output_array,
         summarise_every,
         customisable_variable,
     ):
         """Calculate the mean and reset the buffer.

         Parameters
         ----------
         buffer
             device array. Location containing the running sum of values.
         output_array
             device array. Location for saving the mean value.
         summarise_every
             int. Number of integration steps contributing to each summary.
         customisable_variable
             int. Metric parameter placeholder (unused for mean).

         Notes
         -----
         Divides the accumulated sum by ``summarise_every`` and saves the
         result to ``output_array[0]`` before resetting ``buffer[0]``.
         """
         output_array[0] = buffer[0] / summarise_every
         buffer[0] = precision(0.0)
     ```
   - Edge cases: None
   - Integration: Called by chained save functions

3. **Add update_max and save_max device functions**
   - File: tests/all_in_one.py
   - Action: Create
   - Location: After save_mean function
   - Details: Verbatim copy from max.py lines 55-125
     ```python
     @cuda.jit(
         # [
         #     "float32, float32[::1], int32, int32",
         #     "float64, float64[::1], int32, int32",
         # ],
         device=True,
         inline=True,
         **compile_kwargs
     )
     def update_max(
         value,
         buffer,
         current_index,
         customisable_variable,
     ):
         """Update the running maximum with a new value.

         Parameters
         ----------
         value
             float. New value to compare against the current maximum.
         buffer
             device array. Storage for the current maximum value.
         current_index
             int. Current integration step index (unused for this metric).
         customisable_variable
             int. Metric parameter placeholder (unused for max).

         Notes
         -----
         Updates ``buffer[0]`` if the new value exceeds the current maximum.
         """
         if value > buffer[0]:
             buffer[0] = value

     @cuda.jit(
         # [
         #     "float32[::1], float32[::1], int32, int32",
         #     "float64[::1], float64[::1], int32, int32",
         # ],
         device=True,
         inline=True,
         **compile_kwargs
     )
     def save_max(
         buffer,
         output_array,
         summarise_every,
         customisable_variable,
     ):
         """Save the maximum value to output and reset the buffer.

         Parameters
         ----------
         buffer
             device array. Buffer containing the current maximum value.
         output_array
             device array. Output location for saving the maximum value.
         summarise_every
             int. Number of steps between saves (unused for max).
         customisable_variable
             int. Metric parameter placeholder (unused for max).

         Notes
         -----
         Copies ``buffer[0]`` to ``output_array[0]`` and resets the buffer
         sentinel to ``-1.0e30`` for the next period.
         """
         output_array[0] = buffer[0]
         buffer[0] = precision(-1.0e30)
     ```
   - Edge cases: None
   - Integration: Added to metric lookup tables

4. **Add update_min and save_min device functions**
   - File: tests/all_in_one.py
   - Action: Create
   - Location: After save_max function
   - Details: Verbatim copy from min.py (copy pattern from max.py with inversions)
   - Edge cases: None
   - Integration: Added to metric lookup tables

5. **Add update_rms and save_rms device functions**
   - File: tests/all_in_one.py
   - Action: Create
   - Location: After save_min function
   - Details: Verbatim copy from rms.py lines 57-134, ensure `from math import sqrt` is in imports
   - Edge cases: None
   - Integration: Added to metric lookup tables

6. **Add update_std and save_std device functions**
   - File: tests/all_in_one.py
   - Action: Create
   - Location: After save_rms function
   - Details: Verbatim copy from std.py lines 59-149, ensure `from math import sqrt` is in imports
   - Edge cases: None
   - Integration: Added to metric lookup tables

**Outcomes**: 
[To be filled by taskmaster agent]

---

## Task Group 3: Summary Metric Device Functions (Composite Stats) - PARALLEL
**Status**: [ ]
**Dependencies**: Group 2

**Required Context**:
- File: src/cubie/outputhandling/summarymetrics/mean_std.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/mean_std_rms.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/std_rms.py (entire file)

**Input Validation Required**:
None - device functions do not perform input validation

**Tasks**:

1. **Add update_mean_std and save_mean_std device functions**
   - File: tests/all_in_one.py
   - Action: Create
   - Location: After save_std function
   - Details: Verbatim copy from mean_std.py, ensure `from math import sqrt` is in imports
   - Edge cases: None
   - Integration: Added to metric lookup tables

2. **Add update_mean_std_rms and save_mean_std_rms device functions**
   - File: tests/all_in_one.py
   - Action: Create
   - Location: After save_mean_std function
   - Details: Verbatim copy from mean_std_rms.py, ensure `from math import sqrt` is in imports
   - Edge cases: None
   - Integration: Added to metric lookup tables

3. **Add update_std_rms and save_std_rms device functions**
   - File: tests/all_in_one.py
   - Action: Create
   - Location: After save_mean_std_rms function
   - Details: Verbatim copy from std_rms.py, ensure `from math import sqrt` is in imports
   - Edge cases: None
   - Integration: Added to metric lookup tables

**Outcomes**: 
[To be filled by taskmaster agent]

---

## Task Group 4: Summary Metric Device Functions (Extrema Tracking) - PARALLEL
**Status**: [ ]
**Dependencies**: Group 2

**Required Context**:
- File: src/cubie/outputhandling/summarymetrics/extrema.py (lines 56-131)
- File: src/cubie/outputhandling/summarymetrics/peaks.py (lines 58-149)
- File: src/cubie/outputhandling/summarymetrics/negative_peaks.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/max_magnitude.py (entire file)

**Input Validation Required**:
None - device functions do not perform input validation

**Tasks**:

1. **Add update_extrema and save_extrema device functions**
   - File: tests/all_in_one.py
   - Action: Create
   - Location: After save_std_rms function
   - Details: Verbatim copy from extrema.py lines 56-131
   - Edge cases: None
   - Integration: Added to metric lookup tables

2. **Add update_peaks and save_peaks device functions**
   - File: tests/all_in_one.py
   - Action: Create
   - Location: After save_extrema function
   - Details: Verbatim copy from peaks.py lines 58-149
   - Edge cases: None
   - Integration: Added to metric lookup tables

3. **Add update_negative_peaks and save_negative_peaks device functions**
   - File: tests/all_in_one.py
   - Action: Create
   - Location: After save_peaks function
   - Details: Verbatim copy from negative_peaks.py (mirror of peaks.py with sign inversions)
   - Edge cases: None
   - Integration: Added to metric lookup tables

4. **Add update_max_magnitude and save_max_magnitude device functions**
   - File: tests/all_in_one.py
   - Action: Create
   - Location: After save_negative_peaks function
   - Details: Verbatim copy from max_magnitude.py, ensure `from math import fabs` or use `abs()` builtin
   - Edge cases: None
   - Integration: Added to metric lookup tables

**Outcomes**: 
[To be filled by taskmaster agent]

---

## Task Group 5: Summary Metric Device Functions (First Derivative Tracking) - PARALLEL
**Status**: [ ]
**Dependencies**: Group 2

**Required Context**:
- File: src/cubie/outputhandling/summarymetrics/dxdt_max.py (lines 60-135)
- File: src/cubie/outputhandling/summarymetrics/dxdt_min.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/dxdt_extrema.py (entire file)

**Input Validation Required**:
None - device functions do not perform input validation

**Tasks**:

1. **Add update_dxdt_max and save_dxdt_max device functions**
   - File: tests/all_in_one.py
   - Action: Create
   - Location: After save_max_magnitude function
   - Details: Verbatim copy from dxdt_max.py lines 60-135, uses `selp` from cubie.cuda_simsafe
     - Note: dt_save is captured from configuration, not compile_settings
   - Edge cases: Predicated commit pattern to avoid warp divergence
   - Integration: Added to metric lookup tables, requires dt_save from configuration

2. **Add update_dxdt_min and save_dxdt_min device functions**
   - File: tests/all_in_one.py
   - Action: Create
   - Location: After save_dxdt_max function
   - Details: Verbatim copy from dxdt_min.py (mirror of dxdt_max.py with min instead of max)
   - Edge cases: Predicated commit pattern to avoid warp divergence
   - Integration: Added to metric lookup tables, requires dt_save from configuration

3. **Add update_dxdt_extrema and save_dxdt_extrema device functions**
   - File: tests/all_in_one.py
   - Action: Create
   - Location: After save_dxdt_min function
   - Details: Verbatim copy from dxdt_extrema.py (combines dxdt_max and dxdt_min)
   - Edge cases: Predicated commit pattern to avoid warp divergence
   - Integration: Added to metric lookup tables, requires dt_save from configuration

**Outcomes**: 
[To be filled by taskmaster agent]

---

## Task Group 6: Summary Metric Device Functions (Second Derivative Tracking) - PARALLEL
**Status**: [ ]
**Dependencies**: Group 5

**Required Context**:
- File: src/cubie/outputhandling/summarymetrics/d2xdt2_max.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/d2xdt2_min.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/d2xdt2_extrema.py (entire file)

**Input Validation Required**:
None - device functions do not perform input validation

**Tasks**:

1. **Add update_d2xdt2_max and save_d2xdt2_max device functions**
   - File: tests/all_in_one.py
   - Action: Create
   - Location: After save_dxdt_extrema function
   - Details: Verbatim copy from d2xdt2_max.py, uses `selp` from cubie.cuda_simsafe
     - Note: dt_save is captured from configuration, not compile_settings
   - Edge cases: Predicated commit pattern to avoid warp divergence
   - Integration: Added to metric lookup tables, requires dt_save from configuration

2. **Add update_d2xdt2_min and save_d2xdt2_min device functions**
   - File: tests/all_in_one.py
   - Action: Create
   - Location: After save_d2xdt2_max function
   - Details: Verbatim copy from d2xdt2_min.py (mirror of d2xdt2_max.py with min instead of max)
   - Edge cases: Predicated commit pattern to avoid warp divergence
   - Integration: Added to metric lookup tables, requires dt_save from configuration

3. **Add update_d2xdt2_extrema and save_d2xdt2_extrema device functions**
   - File: tests/all_in_one.py
   - Action: Create
   - Location: After save_d2xdt2_min function
   - Details: Verbatim copy from d2xdt2_extrema.py (combines d2xdt2_max and d2xdt2_min)
   - Edge cases: Predicated commit pattern to avoid warp divergence
   - Integration: Added to metric lookup tables, requires dt_save from configuration

**Outcomes**: 
[To be filled by taskmaster agent]

---

## Task Group 7: Metric Lookup Tables - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Groups 2, 3, 4, 5, 6

**Required Context**:
- All metric device functions added in groups 2-6
- Implementation pattern in registry simulation (Group 1)

**Input Validation Required**:
- summaries_list: Validate each metric name exists in lookup table

**Tasks**:

1. **Add update_functions() lookup function**
   - File: tests/all_in_one.py
   - Action: Create
   - Location: After params() function
   - Details:
     ```python
     def update_functions(summaries_list):
         """Return list of update function references for each metric.
         
         Parameters
         ----------
         summaries_list
             Sequence of summary metric names.
         
         Returns
         -------
         list
             Update function for each metric.
         """
         function_map = {
             "mean": update_mean,
             "max": update_max,
             "min": update_min,
             "rms": update_rms,
             "std": update_std,
             "mean_std": update_mean_std,
             "mean_std_rms": update_mean_std_rms,
             "std_rms": update_std_rms,
             "extrema": update_extrema,
             "peaks": update_peaks,
             "negative_peaks": update_negative_peaks,
             "max_magnitude": update_max_magnitude,
             "dxdt_max": update_dxdt_max,
             "dxdt_min": update_dxdt_min,
             "dxdt_extrema": update_dxdt_extrema,
             "d2xdt2_max": update_d2xdt2_max,
             "d2xdt2_min": update_d2xdt2_min,
             "d2xdt2_extrema": update_d2xdt2_extrema,
         }
         return [function_map[name] for name in summaries_list]
     ```
   - Edge cases: Empty summaries_list returns empty list
   - Integration: Called by update_summary_factory

2. **Add save_functions() lookup function**
   - File: tests/all_in_one.py
   - Action: Create
   - Location: After update_functions()
   - Details:
     ```python
     def save_functions(summaries_list):
         """Return list of save function references for each metric.
         
         Parameters
         ----------
         summaries_list
             Sequence of summary metric names.
         
         Returns
         -------
         list
             Save function for each metric.
         """
         function_map = {
             "mean": save_mean,
             "max": save_max,
             "min": save_min,
             "rms": save_rms,
             "std": save_std,
             "mean_std": save_mean_std,
             "mean_std_rms": save_mean_std_rms,
             "std_rms": save_std_rms,
             "extrema": save_extrema,
             "peaks": save_peaks,
             "negative_peaks": save_negative_peaks,
             "max_magnitude": save_max_magnitude,
             "dxdt_max": save_dxdt_max,
             "dxdt_min": save_dxdt_min,
             "dxdt_extrema": save_dxdt_extrema,
             "d2xdt2_max": save_d2xdt2_max,
             "d2xdt2_min": save_d2xdt2_min,
             "d2xdt2_extrema": save_d2xdt2_extrema,
         }
         return [function_map[name] for name in summaries_list]
     ```
   - Edge cases: Empty summaries_list returns empty list
   - Integration: Called by save_summary_factory

**Outcomes**: 
[To be filled by taskmaster agent]

---

## Task Group 8: Update Chaining Factory Functions - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Group 7

**Required Context**:
- File: src/cubie/outputhandling/update_summaries.py (lines 29-280)
- Metric lookup functions from Group 7

**Input Validation Required**:
- metric_functions: Validate it is a sequence
- buffer_offsets: Validate same length as metric_functions
- buffer_sizes: Validate same length as metric_functions
- function_params: Validate same length as metric_functions

**Tasks**:

1. **Replace existing chain_update_metrics with do_nothing (update version)**
   - File: tests/all_in_one.py
   - Action: Modify
   - Location: Lines 3448-3471 (current chain_update_metrics function)
   - Details: Replace with verbatim copy from update_summaries.py lines 29-61
     ```python
     @cuda.jit(
         device=True,
         inline=True,
         **compile_kwargs,
     )
     def do_nothing_update(
         values,
         buffer,
         current_step,
     ):
         """Provide a no-op device function for empty metric chains.

         Parameters
         ----------
         values
             device array containing the current scalar value (unused).
         buffer
             device array slice reserved for summary accumulation (unused).
         current_step
             Integer or scalar step identifier (unused).

         Returns
         -------
         None
             The device function intentionally performs no operations.

         Notes
         -----
         This function serves as the base case for the recursive chain when no
         summary metrics are configured or as the initial ``inner_chain`` function
         for update operations.
         """
         pass
     ```
   - Edge cases: No-op function, no edge cases
   - Integration: Base case for chain_metrics_update recursion

2. **Add chain_metrics_update recursive function**
   - File: tests/all_in_one.py
   - Action: Create
   - Location: After do_nothing_update function
   - Details: Verbatim copy from update_summaries.py lines 64-161
     - Function name: `chain_metrics_update` (to distinguish from save version)
     - Default inner_chain parameter: `do_nothing_update`
   - Edge cases: Empty metric_functions list returns do_nothing_update
   - Integration: Called by update_summary_factory

3. **Add update_summary_factory function**
   - File: tests/all_in_one.py
   - Action: Create
   - Location: After chain_metrics_update function
   - Details: Verbatim copy from update_summaries.py lines 164-280
     - Replace `summary_metrics.buffer_offsets(summaries_list)` with `buffer_offsets(summaries_list)`
     - Replace `summary_metrics.update_functions(summaries_list)` with `update_functions(summaries_list)`
     - Replace `summary_metrics.buffer_sizes(summaries_list)` with `buffer_sizes(summaries_list)`
     - Replace `summary_metrics.params(summaries_list)` with `params(summaries_list)`
     - Replace `chain_metrics` with `chain_metrics_update`
   - Edge cases: Empty summaries_list produces do_nothing function
   - Integration: Called during configuration to build update chain

**Outcomes**: 
[To be filled by taskmaster agent]

---

## Task Group 9: Save Chaining Factory Functions - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Group 7

**Required Context**:
- File: src/cubie/outputhandling/save_summaries.py (lines 29-326)
- Metric lookup functions from Group 7

**Input Validation Required**:
- metric_functions: Validate it is a sequence
- buffer_offsets: Validate same length as metric_functions
- buffer_sizes: Validate same length as metric_functions
- output_offsets: Validate same length as metric_functions
- output_sizes: Validate same length as metric_functions
- function_params: Validate same length as metric_functions

**Tasks**:

1. **Add do_nothing (save version)**
   - File: tests/all_in_one.py
   - Action: Create
   - Location: After update_summary_factory function
   - Details: Verbatim copy from save_summaries.py lines 29-60
     ```python
     @cuda.jit(
         device=True,
         inline=True,
         **compile_kwargs,
     )
     def do_nothing_save(
         buffer,
         output,
         summarise_every,
     ):
         """Provide a no-op device function for empty metric chains.

         Parameters
         ----------
         buffer
             device array slice containing accumulated metric values (unused).
         output
             device array slice that would receive saved results (unused).
         summarise_every
             Integer interval between summary exports (unused).

         Returns
         -------
         None
             The device function intentionally performs no operations.

         Notes
         -----
         This function serves as the base case for the recursive chain when no
         summary metrics are configured or as the initial ``inner_chain`` function.
         """
         pass
     ```
   - Edge cases: No-op function, no edge cases
   - Integration: Base case for chain_metrics_save recursion

2. **Add chain_metrics_save recursive function**
   - File: tests/all_in_one.py
   - Action: Create
   - Location: After do_nothing_save function
   - Details: Verbatim copy from save_summaries.py lines 63-180
     - Function name: `chain_metrics_save` (to distinguish from update version)
     - Default inner_chain parameter: `do_nothing_save`
   - Edge cases: Empty metric_functions list returns do_nothing_save
   - Integration: Called by save_summary_factory

3. **Add save_summary_factory function**
   - File: tests/all_in_one.py
   - Action: Create
   - Location: After chain_metrics_save function
   - Details: Verbatim copy from save_summaries.py lines 183-326
     - Replace `summary_metrics.save_functions(summaries_list)` with `save_functions(summaries_list)`
     - Replace `summary_metrics.summaries_output_height(summaries_list)` with `sum(output_sizes(summaries_list))`
     - Replace `summary_metrics.buffer_offsets(summaries_list)` with `buffer_offsets(summaries_list)`
     - Replace `summary_metrics.buffer_sizes(summaries_list)` with `buffer_sizes(summaries_list)`
     - Replace `summary_metrics.output_offsets(summaries_list)` with `output_offsets(summaries_list)`
     - Replace `summary_metrics.output_sizes(summaries_list)` with `output_sizes(summaries_list)`
     - Replace `summary_metrics.params(summaries_list)` with `params(summaries_list)`
     - Replace `chain_metrics` with `chain_metrics_save`
   - Edge cases: Empty summaries_list produces do_nothing function
   - Integration: Called during configuration to build save chain

**Outcomes**: 
[To be filled by taskmaster agent]

---

## Task Group 10: List-Based Configuration System - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Groups 1, 8, 9

**Required Context**:
- File: src/cubie/outputhandling/output_config.py (lines 818-874)
- File: tests/all_in_one.py (lines 140-158 for current configuration)

**Input Validation Required**:
- output_types: Validate it is a list, can be empty
- Each entry in output_types: Validate it is a string

**Tasks**:

1. **Add output_types list configuration**
   - File: tests/all_in_one.py
   - Action: Modify
   - Location: Lines 148-156 (replace existing boolean flags)
   - Details:
     ```python
     # -------------------------------------------------------------------------
     # Output Configuration
     # -------------------------------------------------------------------------
     # List-based output configuration (matches package pattern)
     # Available types: 'state', 'observables', 'time', 'iteration_counters'
     # Plus any metric name from implemented_metrics list
     output_types = ['state', 'mean', 'max', 'rms']
     ```
   - Edge cases: Empty list is valid
   - Integration: Used by configuration derivation

2. **Add configuration derivation logic**
   - File: tests/all_in_one.py
   - Action: Create
   - Location: After output_types definition
   - Details: Verbatim logic from output_config.py lines 840-872
     ```python
     # Derive boolean toggles from output_types list
     if not output_types:
         summary_types = tuple()
         save_state_bool = False
         save_obs_bool = False
         save_time_bool = False
         save_counters_bool = False
     else:
         save_state_bool = "state" in output_types
         save_obs_bool = "observables" in output_types
         save_time_bool = "time" in output_types
         save_counters_bool = "iteration_counters" in output_types
         
         summary_types_list = []
         for output_type in output_types:
             if any(
                 (
                     output_type.startswith(name)
                     for name in implemented_metrics
                 )
             ):
                 summary_types_list.append(output_type)
             elif output_type in ["state", "observables", "time", "iteration_counters"]:
                 continue
             else:
                 print(
                     f"Warning: Summary type '{output_type}' is not implemented. "
                     f"Ignoring."
                 )
         
         summary_types = tuple(summary_types_list)
     
     # Derive summarise booleans
     summarise_state_bool = len(summary_types) > 0 and save_state_bool
     summarise_obs_bool = len(summary_types) > 0 and save_obs_bool
     ```
   - Edge cases: Empty output_types, unknown metric names (print warning)
   - Integration: Replaces hardcoded boolean flags

3. **Remove old hardcoded boolean definitions**
   - File: tests/all_in_one.py
   - Action: Delete
   - Location: Lines 150-155 (current hardcoded booleans)
   - Details: Delete the following lines:
     ```python
     save_obs_bool = False
     save_state_bool = True
     summarise_obs_bool = False
     summarise_state_bool = False
     save_counters_bool = False
     ```
   - Edge cases: None
   - Integration: Replaced by derived values

**Outcomes**: 
[To be filled by taskmaster agent]

---

## Task Group 11: Buffer Size Calculations - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Group 10

**Required Context**:
- Configuration section with summary_types derived
- Registry lookup functions from Group 1
- File: tests/all_in_one.py (lines 3530, 3572-3573 for current buffer sizing)

**Input Validation Required**:
- summary_types: Validate it is a tuple or list

**Tasks**:

1. **Add dynamic buffer size calculations**
   - File: tests/all_in_one.py
   - Action: Create
   - Location: After configuration derivation (after summarise_obs_bool)
   - Details:
     ```python
     # Calculate buffer and output sizes based on enabled metrics
     if len(summary_types) > 0:
         summaries_buffer_height_per_var = sum(buffer_sizes(summary_types))
         summaries_output_height_per_var = sum(output_sizes(summary_types))
     else:
         summaries_buffer_height_per_var = 0
         summaries_output_height_per_var = 0
     ```
   - Edge cases: Empty summary_types produces 0 sizes
   - Integration: Used by loop buffer allocation

2. **Update hardcoded buffer size in update_summaries_inline**
   - File: tests/all_in_one.py
   - Action: Modify
   - Location: Line 3530 (total_buffer_size = int32(1))
   - Details: Replace hardcoded 1 with calculated value:
     ```python
     total_buffer_size = int32(summaries_buffer_height_per_var)
     ```
   - Edge cases: 0 when no metrics enabled
   - Integration: Matches dynamic buffer allocation

3. **Update hardcoded buffer size in save_summaries_inline**
   - File: tests/all_in_one.py
   - Action: Modify
   - Location: Lines 3572-3573 (total_buffer_size and total_output_size)
   - Details: Replace hardcoded values with calculated:
     ```python
     total_buffer_size = int32(summaries_buffer_height_per_var)
     total_output_size = int32(summaries_output_height_per_var)
     ```
   - Edge cases: Both 0 when no metrics enabled
   - Integration: Matches dynamic buffer allocation

**Outcomes**: 
[To be filled by taskmaster agent]

---

## Task Group 12: Replace Chain Functions with Factory-Generated - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Groups 8, 9, 10, 11

**Required Context**:
- File: tests/all_in_one.py (lines 3448-3498 for current chain functions)
- Factory functions from Groups 8 and 9
- Configuration from Group 10

**Input Validation Required**:
None - factory functions handle validation

**Tasks**:

1. **Generate and assign update chain function**
   - File: tests/all_in_one.py
   - Action: Modify
   - Location: After buffer size calculations (from Group 11)
   - Details:
     ```python
     # Generate chained update and save functions for enabled metrics
     if len(summary_types) > 0:
         # Create indices for all state variables (all summarised)
         summarised_state_indices = list(range(n_states))
         summarised_observable_indices = list(range(n_observables))
         
         # Generate update chain
         update_summaries_chain = update_summary_factory(
             summaries_buffer_height_per_var,
             summarised_state_indices,
             summarised_observable_indices,
             summary_types,
         )
         
         # Generate save chain
         save_summaries_chain = save_summary_factory(
             summaries_buffer_height_per_var,
             summarised_state_indices,
             summarised_observable_indices,
             summary_types,
         )
     else:
         # No metrics enabled, use do_nothing functions
         update_summaries_chain = do_nothing_update
         save_summaries_chain = do_nothing_save
     ```
   - Edge cases: Empty summary_types uses do_nothing functions
   - Integration: Replaces manual chain functions

2. **Replace update_summaries_inline to call factory-generated chain**
   - File: tests/all_in_one.py
   - Action: Modify
   - Location: Lines 3501-3538 (current update_summaries_inline function)
   - Details: Simplify to call factory-generated function:
     ```python
     @cuda.jit(device=True, inline=True, **compile_kwargs)
     def update_summaries_inline(
         current_state,
         current_observables,
         state_summary_buffer,
         obs_summary_buffer,
         current_step,
     ):
         """Accumulate summary metrics from the current state sample.

         Parameters
         ----------
         current_state
             device array. Current state vector.
         current_observables
             device array. Current observable vector.
         state_summary_buffer
             device array. Buffer for state summary accumulation.
         obs_summary_buffer
             device array. Buffer for observable summary accumulation.
         current_step
             int. Current integration step index.

         Notes
         -----
         Delegates to factory-generated chained update function for all
         enabled summary metrics.
         """
         update_summaries_chain(
             current_state,
             current_observables,
             state_summary_buffer,
             obs_summary_buffer,
             current_step,
         )
     ```
   - Edge cases: Calls do_nothing if no metrics enabled
   - Integration: Wrapper for factory-generated chain

3. **Replace save_summaries_inline to call factory-generated chain**
   - File: tests/all_in_one.py
   - Action: Modify
   - Location: Lines 3541-3581 (current save_summaries_inline function)
   - Details: Simplify to call factory-generated function:
     ```python
     @cuda.jit(device=True, inline=True, **compile_kwargs)
     def save_summaries_inline(
         buffer_state,
         buffer_obs,
         output_state,
         output_obs,
         summarise_every,
     ):
         """Export summary metrics from buffers to output windows.

         Parameters
         ----------
         buffer_state
             device array. State summary accumulation buffer.
         buffer_obs
             device array. Observable summary accumulation buffer.
         output_state
             device array. State summary output array.
         output_obs
             device array. Observable summary output array.
         summarise_every
             int. Number of integration steps in each summary window.

         Notes
         -----
         Delegates to factory-generated chained save function for all
         enabled summary metrics.
         """
         save_summaries_chain(
             buffer_state,
             buffer_obs,
             output_state,
             output_obs,
             summarise_every,
         )
     ```
   - Edge cases: Calls do_nothing if no metrics enabled
   - Integration: Wrapper for factory-generated chain

4. **Delete old chain_update_metrics and chain_save_metrics functions**
   - File: tests/all_in_one.py
   - Action: Delete
   - Location: Lines 3448-3498 (if not already replaced)
   - Details: Remove the old manual chain functions that hardcoded mean metric only
   - Edge cases: None
   - Integration: Replaced by factory-generated chains

**Outcomes**: 
[To be filled by taskmaster agent]

---

## Task Group 13: Verification and Documentation - SEQUENTIAL
**Status**: [ ]
**Dependencies**: All previous groups

**Required Context**:
- All modified files from previous groups
- Original plan files for comparison

**Input Validation Required**:
None - verification task

**Tasks**:

1. **Verify completeness of metric implementations**
   - File: tests/all_in_one.py
   - Action: Verify
   - Details:
     - Confirm all 18 metrics have both update and save functions
     - Confirm all metrics are in lookup dictionaries
     - Confirm all metrics are in implemented_metrics list
     - List: mean, max, min, rms, std, mean_std, mean_std_rms, std_rms, extrema, peaks, negative_peaks, max_magnitude, dxdt_max, dxdt_min, dxdt_extrema, d2xdt2_max, d2xdt2_min, d2xdt2_extrema
   - Edge cases: None
   - Integration: Ensures all requirements met

2. **Verify verbatim copying**
   - File: tests/all_in_one.py
   - Action: Verify
   - Details:
     - Compare each metric device function against source file
     - Verify factory functions match source (accounting for allowed substitutions)
     - Verify configuration logic matches output_config.py
   - Edge cases: None
   - Integration: Ensures behavioral parity with package

3. **Verify integration points unchanged**
   - File: tests/all_in_one.py
   - Action: Verify
   - Details:
     - Confirm loop integration code around lines 3500-3550 still works
     - Confirm function signatures match expected interfaces
     - Confirm buffer layouts are consistent
   - Edge cases: None
   - Integration: Ensures no breaking changes

4. **Add configuration example comment**
   - File: tests/all_in_one.py
   - Action: Modify
   - Location: After output_types definition
   - Details:
     ```python
     # Examples of output_types configurations:
     # output_types = ['state']  # Only save states, no summaries
     # output_types = ['state', 'mean']  # Save states and mean summary
     # output_types = ['state', 'mean', 'max', 'rms']  # Multiple summaries
     # output_types = ['state', 'dxdt_max', 'd2xdt2_extrema']  # Derivative metrics
     # output_types = []  # No outputs (valid but not useful)
     ```
   - Edge cases: None
   - Integration: Helps users understand configuration

**Outcomes**: 
[To be filled by taskmaster agent]

---

## Summary

**Total Task Groups**: 13
**Total Individual Tasks**: 50+

**Dependency Chain Overview**:
1. Group 1 (Registry Infrastructure) → Groups 2-6 (Metric Functions) in parallel
2. Groups 2-6 → Group 7 (Lookup Tables)
3. Group 7 → Groups 8-9 (Factory Functions) in parallel
4. Groups 1, 8, 9 → Group 10 (Configuration)
5. Group 10 → Group 11 (Buffer Sizing)
6. Groups 8, 9, 10, 11 → Group 12 (Replace Chains)
7. All → Group 13 (Verification)

**Parallel Execution Opportunities**:
- Groups 2, 3, 4, 5, 6 can execute in parallel (different metric categories)
- Groups 8 and 9 can execute in parallel (update vs save chains)

**Estimated Complexity**: High
- 18+ metric pairs (36+ device functions) requiring verbatim copying
- 6 factory functions requiring careful adaptation
- Configuration system requiring exact logic replication
- Integration requiring surgical precision to avoid breaking existing code
