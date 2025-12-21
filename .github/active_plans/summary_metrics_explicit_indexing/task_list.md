# Implementation Task List
# Feature: Summary Metrics Explicit Indexing Refactor
# Plan Reference: .github/active_plans/summary_metrics_explicit_indexing/agent_plan.md

## Task Group 1: Core Infrastructure - SEQUENTIAL
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/outputhandling/update_summaries.py (entire file, lines 1-281)
- File: src/cubie/cuda_simsafe.py (lines 200-260 for selp definition)

**Input Validation Required**:
- None required for this task (internal signature changes only)

**Tasks**:

1. **Update `do_nothing` function signature**
   - File: src/cubie/outputhandling/update_summaries.py
   - Action: Modify
   - Details:
     ```python
     @cuda.jit(
         device=True,
         inline=True,
         **compile_kwargs,
     )
     def do_nothing(
         value,
         buffer,
         buffer_offset,
         current_step,
     ):
         """Provide a no-op device function for empty metric chains.

         Parameters
         ----------
         value
             device array element being summarised (unused).
         buffer
             device array containing metric working storage (unused).
         buffer_offset
             Integer offset into the buffer for this variable (unused).
         current_step
             Integer or scalar step identifier (unused).

         Returns
         -------
         None
             The device function intentionally performs no operations.

         Notes
         -----
         This function serves as the base case for the recursive chain when no
         summary metrics are configured or as the initial ``inner_chain``
         function for update operations.
         """
         pass
     ```
   - Edge cases: Must handle being called with any buffer_offset value
   - Integration: Used as base case when metric_functions is empty

2. **Update `chain_metrics` function to pass offset instead of sliced buffer**
   - File: src/cubie/outputhandling/update_summaries.py
   - Action: Modify
   - Details:
     ```python
     def chain_metrics(
         metric_functions: Sequence[Callable],
         buffer_offsets: Sequence[int],
         buffer_sizes: Sequence[int],
         function_params: Sequence[object],
         inner_chain: Callable = do_nothing,
     ) -> Callable:
         """
         Recursively chain summary metric update functions for CUDA execution.

         This function builds a recursive chain of summary metric update
         functions, where each function in the sequence is wrapped with the
         previous functions to create a single callable that updates all
         metrics.

         Parameters
         ----------
         metric_functions
             Sequence of CUDA device functions for updating summary metrics.
         buffer_offsets
             Sequence of offsets into the metric buffer for each function.
         buffer_sizes
             Sequence of per-metric buffer lengths.
         function_params
             Sequence of parameter payloads passed to each metric function.
         inner_chain
             Callable executed before the current metric; defaults to
             ``do_nothing``.

         Returns
         -------
         Callable
             CUDA device function that executes all chained metric updates.

         Notes
         -----
         The function uses recursion to build a chain where each level executes
         the inner chain first, then the current metric update function. Each
         wrapper passes the full buffer plus the computed offset (base_offset +
         metric_offset) to the metric function, eliminating array slicing.
         """
         if len(metric_functions) == 0:
             return do_nothing

         current_fn = metric_functions[0]
         current_offset = buffer_offsets[0]
         current_param = function_params[0]

         remaining_functions = metric_functions[1:]
         remaining_offsets = buffer_offsets[1:]
         remaining_sizes = buffer_sizes[1:]
         remaining_params = function_params[1:]

         # no cover: start
         @cuda.jit(
             device=True,
             inline=True,
             **compile_kwargs,
         )
         def wrapper(
             value,
             buffer,
             buffer_offset,
             current_step,
         ):
             """Apply the accumulated metric chain before invoking the
             current metric.

             Parameters
             ----------
             value
                 device array element being summarised.
             buffer
                 device array containing the full metric working storage.
             buffer_offset
                 Integer base offset for this variable's buffer segment.
             current_step
                 Integer or scalar step identifier passed through the chain.

             Returns
             -------
             None
                 The device function mutates the metric buffer in place.
             """
             inner_chain(value, buffer, buffer_offset, current_step)
             current_fn(
                 value,
                 buffer,
                 buffer_offset + current_offset,
                 current_step,
                 current_param,
             )

         if remaining_functions:
             return chain_metrics(
                 remaining_functions,
                 remaining_offsets,
                 remaining_sizes,
                 remaining_params,
                 wrapper,
             )
         else:
             return wrapper
         # no cover: stop
     ```
   - Edge cases: Empty metric_functions returns do_nothing (with new signature)
   - Integration: Called by update_summary_factory to build the chain

3. **Update `update_summary_factory` to pass offset instead of sliced buffer**
   - File: src/cubie/outputhandling/update_summaries.py
   - Action: Modify
   - Details:
     ```python
     def update_summary_factory(
         summaries_buffer_height_per_var: int,
         summarised_state_indices: Union[Sequence[int], ArrayLike],
         summarised_observable_indices: Union[Sequence[int], ArrayLike],
         summaries_list: Sequence[str],
     ) -> Callable:
         """
         Factory function for creating CUDA device functions to update summary
         metrics.

         This factory generates an optimized CUDA device function that applies
         chained summary metric updates to all requested state and observable
         variables during each integration step.

         Parameters
         ----------
         summaries_buffer_height_per_var
             Number of buffer slots required per tracked variable.
         summarised_state_indices
             Sequence of state indices to include in summary calculations.
         summarised_observable_indices
             Sequence of observable indices to include in summary calculations.
         summaries_list
             Ordered list of summary metric identifiers registered with
             :mod:`cubie.outputhandling.summarymetrics`.

         Returns
         -------
         Callable
             CUDA device function for updating summary metrics.

         Notes
         -----
         The generated function iterates through all specified state and
         observable variables, applying the chained summary metric updates to
         accumulate data in the appropriate buffer locations during each
         integration step. Uses explicit offset parameters instead of buffer
         slicing to improve register efficiency.
         """
         num_summarised_states = int32(len(summarised_state_indices))
         num_summarised_observables = int32(len(summarised_observable_indices))
         buff_per_var = summaries_buffer_height_per_var
         total_buffer_size = int32(buff_per_var)
         buffer_offsets = summary_metrics.buffer_offsets(summaries_list)
         num_metrics = len(buffer_offsets)

         summarise_states = (num_summarised_states > 0) and (num_metrics > 0)
         summarise_observables = (num_summarised_observables > 0) and (
             num_metrics > 0
         )

         update_fns = summary_metrics.update_functions(summaries_list)
         buffer_sizes_list = summary_metrics.buffer_sizes(summaries_list)
         params = summary_metrics.params(summaries_list)
         chain_fn = chain_metrics(
             update_fns, buffer_offsets, buffer_sizes_list, params
         )

         # no cover: start
         @cuda.jit(
             device=True,
             inline=True,
             **compile_kwargs,
         )
         def update_summary_metrics_func(
             current_state,
             current_observables,
             state_summary_buffer,
             observable_summary_buffer,
             current_step,
         ):
             """Accumulate summary metrics from the current state sample.

             Parameters
             ----------
             current_state
                 device array holding the latest integrator state values.
             current_observables
                 device array holding the latest observable values.
             state_summary_buffer
                 device array used to accumulate state summary data.
             observable_summary_buffer
                 device array used to accumulate observable summary data.
             current_step
                 Integer or scalar step identifier associated with the sample.

             Returns
             -------
             None
                 The device function mutates the supplied summary buffers in
                 place.

             Notes
             -----
             The chained metric function is executed for each selected state or
             observable entry, passing the full buffer with a base offset for
             that variable's segment.
             """
             if summarise_states:
                 for idx in range(num_summarised_states):
                     base_offset = idx * total_buffer_size
                     chain_fn(
                         current_state[summarised_state_indices[idx]],
                         state_summary_buffer,
                         base_offset,
                         current_step,
                     )

             if summarise_observables:
                 for idx in range(num_summarised_observables):
                     base_offset = idx * total_buffer_size
                     chain_fn(
                         current_observables[
                             summarised_observable_indices[idx]
                         ],
                         observable_summary_buffer,
                         base_offset,
                         current_step,
                     )

         # no cover: stop
         return update_summary_metrics_func
     ```
   - Edge cases: 
     - Zero summarised states or observables: loops don't execute
     - Zero metrics: chain_fn is do_nothing
   - Integration: Entry point called from output_functions.py

**Outcomes**: 
- Files Modified: 
  * src/cubie/outputhandling/update_summaries.py (76 lines changed)
- Functions/Methods Added/Modified:
  * do_nothing() - Updated signature to add buffer_offset parameter, updated docstring
  * chain_metrics() - Updated wrapper to pass buffer_offset instead of sliced buffer, removed current_size usage, updated docstring
  * update_summary_factory() - Updated update_summary_metrics_func to pass base_offset instead of sliced buffer, updated docstring
- Implementation Summary:
  Modified core infrastructure to use explicit offset parameters. The do_nothing function now accepts (value, buffer, buffer_offset, current_step). The chain_metrics wrapper now passes buffer_offset and computes buffer_offset + current_offset for metric calls. The update_summary_factory now computes base_offset = idx * total_buffer_size and passes the full buffer with offset to chain_fn.
- Issues Flagged: None

---

## Task Group 2: Simple Accumulator Metrics - PARALLEL
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/outputhandling/summarymetrics/mean.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/rms.py (entire file)

**Input Validation Required**:
- None (CUDA device functions, no validation needed)

**Tasks**:

1. **Update `mean.py` update function signature**
   - File: src/cubie/outputhandling/summarymetrics/mean.py
   - Action: Modify
   - Details:
     ```python
     @cuda.jit(
         device=True,
         inline=True,
     )
     def update(
         value,
         buffer,
         offset,
         current_index,
         customisable_variable,
     ):
         """Update the running sum with a new value.

         Parameters
         ----------
         value
             float. New value to add to the running sum.
         buffer
             device array. Full buffer containing metric working storage.
         offset
             int. Offset to this metric's storage within the buffer.
         current_index
             int. Current integration step index (unused for mean).
         customisable_variable
             int. Metric parameter placeholder (unused for mean).

         Notes
         -----
         Adds the new value to ``buffer[offset + 0]`` to maintain the
         running sum.
         """
         buffer[offset + 0] += value
     ```
   - Edge cases: None
   - Integration: Called from chain_metrics wrapper

2. **Update `rms.py` update function signature**
   - File: src/cubie/outputhandling/summarymetrics/rms.py
   - Action: Modify
   - Details:
     ```python
     @cuda.jit(
         device=True,
         inline=True,
         **compile_kwargs,
     )
     def update(
         value,
         buffer,
         offset,
         current_index,
         customisable_variable,
     ):
         """Update the running sum of squares with a new value.

         Parameters
         ----------
         value
             float. New value to square and add to the running sum.
         buffer
             device array. Full buffer containing metric working storage.
         offset
             int. Offset to this metric's storage within the buffer.
         current_index
             int. Current integration step index, used to reset the sum.
         customisable_variable
             int. Metric parameter placeholder (unused for RMS).

         Notes
         -----
         Resets ``buffer[offset + 0]`` on the first step of a period before
         adding the squared value.
         """
         sum_of_squares = buffer[offset + 0]
         if current_index == 0:
             sum_of_squares = precision(0.0)
         sum_of_squares += value * value
         buffer[offset + 0] = sum_of_squares
     ```
   - Edge cases: Reset on current_index == 0
   - Integration: Called from chain_metrics wrapper

**Outcomes**: 
- Files Modified: 
  * src/cubie/outputhandling/summarymetrics/mean.py (7 lines changed)
  * src/cubie/outputhandling/summarymetrics/rms.py (8 lines changed)
- Functions/Methods Added/Modified:
  * update() in mean.py - Added offset parameter, updated buffer indexing
  * update() in rms.py - Added offset parameter, updated buffer indexing
- Implementation Summary:
  Updated both simple accumulator metrics (mean and rms) to use explicit offset indexing. Added offset as the third parameter to the update function signature. Changed buffer access from buffer[0] to buffer[offset + 0] for all read/write operations. Updated docstrings to document the new offset parameter.
- Issues Flagged: None

---

## Task Group 3: Conditional Metrics with selp - PARALLEL
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/outputhandling/summarymetrics/max.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/min.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/max_magnitude.py (entire file)
- File: src/cubie/cuda_simsafe.py (lines 200-260 for selp import)

**Input Validation Required**:
- None (CUDA device functions, no validation needed)

**Tasks**:

1. **Update `max.py` with explicit indexing and selp pattern**
   - File: src/cubie/outputhandling/summarymetrics/max.py
   - Action: Modify
   - Details:
     Add import for selp:
     ```python
     from cubie.cuda_simsafe import selp
     ```
     Update update function:
     ```python
     @cuda.jit(
         device=True,
         inline=True,
     )
     def update(
         value,
         buffer,
         offset,
         current_index,
         customisable_variable,
     ):
         """Update the running maximum with a new value.

         Parameters
         ----------
         value
             float. New value to compare against the current maximum.
         buffer
             device array. Full buffer containing metric working storage.
         offset
             int. Offset to this metric's storage within the buffer.
         current_index
             int. Current integration step index (unused for this metric).
         customisable_variable
             int. Metric parameter placeholder (unused for max).

         Notes
         -----
         Uses predicated commit to update ``buffer[offset + 0]`` if the new
         value exceeds the current maximum, avoiding warp divergence.
         """
         update_flag = value > buffer[offset + 0]
         buffer[offset + 0] = selp(update_flag, value, buffer[offset + 0])
     ```
   - Edge cases: None
   - Integration: Called from chain_metrics wrapper

2. **Update `min.py` with explicit indexing and selp pattern**
   - File: src/cubie/outputhandling/summarymetrics/min.py
   - Action: Modify
   - Details:
     Add import for selp:
     ```python
     from cubie.cuda_simsafe import selp, compile_kwargs
     ```
     Update update function:
     ```python
     @cuda.jit(
         device=True,
         inline=True,
         **compile_kwargs,
     )
     def update(
         value,
         buffer,
         offset,
         current_index,
         customisable_variable,
     ):
         """Update the running minimum with a new value.

         Parameters
         ----------
         value
             float. New value to compare against the current minimum.
         buffer
             device array. Full buffer containing metric working storage.
         offset
             int. Offset to this metric's storage within the buffer.
         current_index
             int. Current integration step index (unused for this metric).
         customisable_variable
             int. Metric parameter placeholder (unused for min).

         Notes
         -----
         Uses predicated commit to update ``buffer[offset + 0]`` if the new
         value is less than the current minimum, avoiding warp divergence.
         """
         update_flag = value < buffer[offset + 0]
         buffer[offset + 0] = selp(update_flag, value, buffer[offset + 0])
     ```
   - Edge cases: None
   - Integration: Called from chain_metrics wrapper

3. **Update `max_magnitude.py` with explicit indexing and selp pattern**
   - File: src/cubie/outputhandling/summarymetrics/max_magnitude.py
   - Action: Modify
   - Details:
     Add import for selp:
     ```python
     from cubie.cuda_simsafe import selp
     ```
     Update update function:
     ```python
     @cuda.jit(
         device=True,
         inline=True,
     )
     def update(
         value,
         buffer,
         offset,
         current_index,
         customisable_variable,
     ):
         """Update the running maximum magnitude with a new value.

         Parameters
         ----------
         value
             float. New value whose absolute value is compared.
         buffer
             device array. Full buffer containing metric working storage.
         offset
             int. Offset to this metric's storage within the buffer.
         current_index
             int. Current integration step index (unused).
         customisable_variable
             int. Metric parameter placeholder (unused).

         Notes
         -----
         Uses predicated commit to update ``buffer[offset + 0]`` if
         ``abs(value)`` exceeds the current maximum magnitude, avoiding
         warp divergence.
         """
         abs_value = fabs(value)
         update_flag = abs_value > buffer[offset + 0]
         buffer[offset + 0] = selp(update_flag, abs_value, buffer[offset + 0])
     ```
   - Edge cases: None
   - Integration: Called from chain_metrics wrapper

**Outcomes**: 
- Files Modified: 
  * src/cubie/outputhandling/summarymetrics/max.py (9 lines changed)
  * src/cubie/outputhandling/summarymetrics/min.py (9 lines changed)
  * src/cubie/outputhandling/summarymetrics/max_magnitude.py (10 lines changed)
- Functions/Methods Added/Modified:
  * update() in max.py - Added offset parameter, updated buffer indexing with selp pattern
  * update() in min.py - Added offset parameter, updated buffer indexing with selp pattern
  * update() in max_magnitude.py - Added offset parameter, updated buffer indexing with selp pattern
- Implementation Summary:
  Updated all three conditional metrics (max, min, max_magnitude) to use explicit offset indexing and selp predicated commit pattern. Added selp import from cubie.cuda_simsafe. Changed update function signatures to include offset as the third parameter. Replaced if-condition buffer writes with predicated selp pattern to avoid warp divergence. Updated docstrings to document the new offset parameter and selp usage.
- Issues Flagged: None

---

## Task Group 4: Multi-Slot Statistical Metrics - PARALLEL
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/outputhandling/summarymetrics/std.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/extrema.py (entire file)
- File: src/cubie/cuda_simsafe.py (lines 200-260 for selp import)

**Input Validation Required**:
- None (CUDA device functions, no validation needed)

**Tasks**:

1. **Update `std.py` with explicit indexing**
   - File: src/cubie/outputhandling/summarymetrics/std.py
   - Action: Modify
   - Details:
     ```python
     @cuda.jit(
         device=True,
         inline=True,
         **compile_kwargs,
     )
     def update(
         value,
         buffer,
         offset,
         current_index,
         customisable_variable,
     ):
         """Update the running sum and sum of squares with shifted values.

         Parameters
         ----------
         value
             float. New value to add to the running statistics.
         buffer
             device array. Full buffer containing metric working storage.
         offset
             int. Offset to this metric's storage within the buffer.
         current_index
             int. Current integration step index within the summary period.
         customisable_variable
             int. Metric parameter placeholder (unused for std).

         Notes
         -----
         On first sample (current_index == 0), stores the value as shift
         and resets accumulators. For all samples including the first,
         computes shifted_value = value - shift and adds it to
         buffer[offset + 1] (sum) and shifted_value^2 to buffer[offset + 2]
         (sum of squares). This shifting improves numerical stability.
         """
         if current_index == 0:
             buffer[offset + 0] = value  # Store shift value
             buffer[offset + 1] = precision(0.0)    # Reset sum
             buffer[offset + 2] = precision(0.0)    # Reset sum of squares
         
         shifted_value = value - buffer[offset + 0]
         buffer[offset + 1] += shifted_value
         buffer[offset + 2] += shifted_value * shifted_value
     ```
   - Edge cases: Reset on current_index == 0
   - Integration: Called from chain_metrics wrapper

2. **Update `extrema.py` with explicit indexing and selp pattern**
   - File: src/cubie/outputhandling/summarymetrics/extrema.py
   - Action: Modify
   - Details:
     Add import for selp:
     ```python
     from cubie.cuda_simsafe import selp
     ```
     Update update function:
     ```python
     @cuda.jit(
         device=True,
         inline=True,
     )
     def update(
         value,
         buffer,
         offset,
         current_index,
         customisable_variable,
     ):
         """Update the running maximum and minimum with a new value.

         Parameters
         ----------
         value
             float. New value to compare against current extrema.
         buffer
             device array. Full buffer containing metric working storage.
         offset
             int. Offset to this metric's storage within the buffer.
         current_index
             int. Current integration step index (unused).
         customisable_variable
             int. Metric parameter placeholder (unused).

         Notes
         -----
         Uses predicated commit to update ``buffer[offset + 0]`` (max) if
         value exceeds it, and ``buffer[offset + 1]`` (min) if value is
         less than it, avoiding warp divergence.
         """
         update_max = value > buffer[offset + 0]
         update_min = value < buffer[offset + 1]
         buffer[offset + 0] = selp(update_max, value, buffer[offset + 0])
         buffer[offset + 1] = selp(update_min, value, buffer[offset + 1])
     ```
   - Edge cases: None
   - Integration: Called from chain_metrics wrapper

**Outcomes**: 
- Files Modified: 
  * src/cubie/outputhandling/summarymetrics/std.py (12 lines changed)
  * src/cubie/outputhandling/summarymetrics/extrema.py (13 lines changed)
- Functions/Methods Added/Modified:
  * update() in std.py - Added offset parameter, updated buffer indexing
  * update() in extrema.py - Added offset parameter, updated buffer indexing with selp pattern
- Implementation Summary:
  Updated both multi-slot statistical metrics (std and extrema) to use explicit offset indexing. Added offset as the third parameter to the update function signatures. Changed buffer access from buffer[N] to buffer[offset + N] for all read/write operations. In extrema.py, added selp import and replaced if-conditions with predicated commit pattern to avoid warp divergence. Updated docstrings to document the new offset parameter and usage patterns.
- Issues Flagged: None

---

## Task Group 5: Composite Statistical Metrics - PARALLEL
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/outputhandling/summarymetrics/mean_std.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/mean_std_rms.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/std_rms.py (entire file)

**Input Validation Required**:
- None (CUDA device functions, no validation needed)

**Tasks**:

1. **Update `mean_std.py` with explicit indexing**
   - File: src/cubie/outputhandling/summarymetrics/mean_std.py
   - Action: Modify
   - Details:
     ```python
     @cuda.jit(
         device=True,
         inline=True,
     )
     def update(
         value,
         buffer,
         offset,
         current_index,
         customisable_variable,
     ):
         """Update running sums with a new value using shifted data.

         Parameters
         ----------
         value
             float. New value to add to the running statistics.
         buffer
             device array. Full buffer containing metric working storage.
         offset
             int. Offset to this metric's storage within the buffer.
         current_index
             int. Current integration step index within summary period.
         customisable_variable
             int. Metric parameter placeholder (unused).

         Notes
         -----
         On first sample (current_index == 0), stores value as shift.
         Computes shifted_value = value - shift and adds it to
         buffer[offset + 1] (sum) and shifted_value^2 to buffer[offset + 2]
         (sum of squares).
         """
         if current_index == 0:
             buffer[offset + 0] = value
             buffer[offset + 1] = precision(0.0)
             buffer[offset + 2] = precision(0.0)
         
         shifted_value = value - buffer[offset + 0]
         buffer[offset + 1] += shifted_value
         buffer[offset + 2] += shifted_value * shifted_value
     ```
   - Edge cases: Reset on current_index == 0
   - Integration: Called from chain_metrics wrapper

2. **Update `mean_std_rms.py` with explicit indexing**
   - File: src/cubie/outputhandling/summarymetrics/mean_std_rms.py
   - Action: Modify
   - Details:
     ```python
     @cuda.jit(
         device=True,
         inline=True,
         **compile_kwargs,
     )
     def update(
         value,
         buffer,
         offset,
         current_index,
         customisable_variable,
     ):
         """Update running sums with a new value using shifted data.

         Parameters
         ----------
         value
             float. New value to add to the running statistics.
         buffer
             device array. Full buffer containing metric working storage.
         offset
             int. Offset to this metric's storage within the buffer.
         current_index
             int. Current integration step index within summary period.
         customisable_variable
             int. Metric parameter placeholder (unused).

         Notes
         -----
         On first sample (current_index == 0), stores value as shift.
         Computes shifted_value = value - shift and adds it to
         buffer[offset + 1] (sum) and shifted_value^2 to buffer[offset + 2]
         (sum of squares).
         """
         if current_index == 0:
             buffer[offset + 0] = value
             buffer[offset + 1] = precision(0.0)
             buffer[offset + 2] = precision(0.0)
         
         shifted_value = value - buffer[offset + 0]
         buffer[offset + 1] += shifted_value
         buffer[offset + 2] += shifted_value * shifted_value
     ```
   - Edge cases: Reset on current_index == 0
   - Integration: Called from chain_metrics wrapper

3. **Update `std_rms.py` with explicit indexing**
   - File: src/cubie/outputhandling/summarymetrics/std_rms.py
   - Action: Modify
   - Details:
     ```python
     @cuda.jit(
         device=True,
         inline=True,
         **compile_kwargs,
     )
     def update(
         value,
         buffer,
         offset,
         current_index,
         customisable_variable,
     ):
         """Update running sums with a new value using shifted data.

         Parameters
         ----------
         value
             float. New value to add to the running statistics.
         buffer
             device array. Full buffer containing metric working storage.
         offset
             int. Offset to this metric's storage within the buffer.
         current_index
             int. Current integration step index within summary period.
         customisable_variable
             int. Metric parameter placeholder (unused).

         Notes
         -----
         On first sample (current_index == 0), stores value as shift.
         Computes shifted_value = value - shift and adds it to
         buffer[offset + 1] (sum) and shifted_value^2 to buffer[offset + 2]
         (sum of squares).
         """
         if current_index == 0:
             buffer[offset + 0] = value
             buffer[offset + 1] = precision(0.0)
             buffer[offset + 2] = precision(0.0)
         
         shifted_value = value - buffer[offset + 0]
         buffer[offset + 1] += shifted_value
         buffer[offset + 2] += shifted_value * shifted_value
     ```
   - Edge cases: Reset on current_index == 0
   - Integration: Called from chain_metrics wrapper

**Outcomes**: 
- Files Modified: 
  * src/cubie/outputhandling/summarymetrics/mean_std.py (11 lines changed)
  * src/cubie/outputhandling/summarymetrics/mean_std_rms.py (11 lines changed)
  * src/cubie/outputhandling/summarymetrics/std_rms.py (11 lines changed)
- Functions/Methods Added/Modified:
  * update() in mean_std.py - Added offset parameter, updated buffer indexing
  * update() in mean_std_rms.py - Added offset parameter, updated buffer indexing
  * update() in std_rms.py - Added offset parameter, updated buffer indexing
- Implementation Summary:
  Updated all three composite statistical metrics (mean_std, mean_std_rms, std_rms) to use explicit offset indexing. Added offset as the third parameter to the update function signatures. Changed buffer access from buffer[N] to buffer[offset + N] for all read/write operations (shift at offset+0, sum at offset+1, sum of squares at offset+2). Updated docstrings to document the new offset parameter.
- Issues Flagged: None

---

## Task Group 6: First Derivative Metrics - PARALLEL
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/outputhandling/summarymetrics/dxdt_max.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/dxdt_min.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/dxdt_extrema.py (entire file)

**Input Validation Required**:
- None (CUDA device functions, no validation needed)

**Tasks**:

1. **Update `dxdt_max.py` with explicit indexing**
   - File: src/cubie/outputhandling/summarymetrics/dxdt_max.py
   - Action: Modify
   - Details:
     ```python
     @cuda.jit(
         device=True,
         inline=True,
     )
     def update(
         value,
         buffer,
         offset,
         current_index,
         customisable_variable,
     ):
         """Update the maximum first derivative with a new value.

         Parameters
         ----------
         value
             float. New value to compute derivative from.
         buffer
             device array. Full buffer containing metric working storage.
         offset
             int. Offset to this metric's storage within the buffer.
         current_index
             int. Current integration step index (unused).
         customisable_variable
             int. Metric parameter placeholder (unused).

         Notes
         -----
         Computes unscaled derivative as (value - buffer[offset + 0]) and
         updates buffer[offset + 1] if larger. Uses predicated commit pattern
         to avoid warp divergence.
         """
         derivative_unscaled = value - buffer[offset + 0]
         update_flag = (derivative_unscaled > buffer[offset + 1]) and (
             buffer[offset + 0] != precision(0.0)
         )
         buffer[offset + 1] = selp(
             update_flag, derivative_unscaled, buffer[offset + 1]
         )
         buffer[offset + 0] = value
     ```
   - Edge cases: Guards against initial zero value
   - Integration: Called from chain_metrics wrapper

2. **Update `dxdt_min.py` with explicit indexing**
   - File: src/cubie/outputhandling/summarymetrics/dxdt_min.py
   - Action: Modify
   - Details:
     ```python
     @cuda.jit(
         device=True,
         inline=True,
     )
     def update(
         value,
         buffer,
         offset,
         current_index,
         customisable_variable,
     ):
         """Update the minimum first derivative with a new value.

         Parameters
         ----------
         value
             float. New value to compute derivative from.
         buffer
             device array. Full buffer containing metric working storage.
         offset
             int. Offset to this metric's storage within the buffer.
         current_index
             int. Current integration step index (unused).
         customisable_variable
             int. Metric parameter placeholder (unused).

         Notes
         -----
         Computes unscaled derivative as (value - buffer[offset + 0]) and
         updates buffer[offset + 1] if smaller. Uses predicated commit pattern
         to avoid warp divergence.
         """
         derivative_unscaled = value - buffer[offset + 0]
         update_flag = (derivative_unscaled < buffer[offset + 1]) and (
             buffer[offset + 0] != precision(0.0)
         )
         buffer[offset + 1] = selp(
             update_flag, derivative_unscaled, buffer[offset + 1]
         )
         buffer[offset + 0] = value
     ```
   - Edge cases: Guards against initial zero value
   - Integration: Called from chain_metrics wrapper

3. **Update `dxdt_extrema.py` with explicit indexing**
   - File: src/cubie/outputhandling/summarymetrics/dxdt_extrema.py
   - Action: Modify
   - Details:
     ```python
     @cuda.jit(
         device=True,
         inline=True,
     )
     def update(
         value,
         buffer,
         offset,
         current_index,
         customisable_variable,
     ):
         """Update maximum and minimum first derivatives with a new value.

         Parameters
         ----------
         value
             float. New value to compute derivative from.
         buffer
             device array. Full buffer containing metric working storage.
         offset
             int. Offset to this metric's storage within the buffer.
         current_index
             int. Current integration step index (unused).
         customisable_variable
             int. Metric parameter placeholder (unused).

         Notes
         -----
         Computes unscaled derivative as (value - buffer[offset + 0]) and
         updates buffer[offset + 1] if larger and buffer[offset + 2] if
         smaller. Uses predicated commit pattern to avoid warp divergence.
         """
         derivative_unscaled = value - buffer[offset + 0]
         update_max = (derivative_unscaled > buffer[offset + 1]) and (
             buffer[offset + 0] != precision(0.0)
         )
         update_min = (derivative_unscaled < buffer[offset + 2]) and (
             buffer[offset + 0] != precision(0.0)
         )
         buffer[offset + 1] = selp(
             update_max, derivative_unscaled, buffer[offset + 1]
         )
         buffer[offset + 2] = selp(
             update_min, derivative_unscaled, buffer[offset + 2]
         )
         buffer[offset + 0] = value
     ```
   - Edge cases: Guards against initial zero value
   - Integration: Called from chain_metrics wrapper

**Outcomes**: 
- Files Modified: 
  * src/cubie/outputhandling/summarymetrics/dxdt_max.py (11 lines changed)
  * src/cubie/outputhandling/summarymetrics/dxdt_min.py (11 lines changed)
  * src/cubie/outputhandling/summarymetrics/dxdt_extrema.py (14 lines changed)
- Functions/Methods Added/Modified:
  * update() in dxdt_max.py - Added offset parameter, updated buffer indexing with selp pattern
  * update() in dxdt_min.py - Added offset parameter, updated buffer indexing with selp pattern
  * update() in dxdt_extrema.py - Added offset parameter, updated buffer indexing with selp pattern
- Implementation Summary:
  Updated all three first derivative metrics (dxdt_max, dxdt_min, dxdt_extrema) to use explicit offset indexing. Added offset as the third parameter to the update function signatures. Changed buffer access from buffer[N] to buffer[offset + N] for all read/write operations (previous value at offset+0, max/min derivative at offset+1, min derivative for extrema at offset+2). Updated docstrings to document the new offset parameter. The selp predicated commit pattern was already in use and continues to be used with the new indexing.
- Issues Flagged: None

---

## Task Group 7: Second Derivative Metrics - PARALLEL
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/outputhandling/summarymetrics/d2xdt2_max.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/d2xdt2_min.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/d2xdt2_extrema.py (entire file)

**Input Validation Required**:
- None (CUDA device functions, no validation needed)

**Tasks**:

1. **Update `d2xdt2_max.py` with explicit indexing**
   - File: src/cubie/outputhandling/summarymetrics/d2xdt2_max.py
   - Action: Modify
   - Details:
     ```python
     @cuda.jit(
         device=True,
         inline=True,
     )
     def update(
         value,
         buffer,
         offset,
         current_index,
         customisable_variable,
     ):
         """Update the maximum second derivative with a new value.

         Parameters
         ----------
         value
             float. New value to compute second derivative from.
         buffer
             device array. Full buffer containing metric working storage.
         offset
             int. Offset to this metric's storage within the buffer.
         current_index
             int. Current integration step index (unused).
         customisable_variable
             int. Metric parameter placeholder (unused).

         Notes
         -----
         Computes unscaled second derivative using central difference formula
         (value - 2*buffer[offset + 0] + buffer[offset + 1]) and updates
         buffer[offset + 2] if larger. Uses predicated commit pattern to avoid
         warp divergence. Guard on buffer[offset + 1] ensures two previous
         values are available.
         """
         second_derivative_unscaled = (
             value
             - precision(2.0) * buffer[offset + 0]
             + buffer[offset + 1]
         )
         update_flag = (second_derivative_unscaled > buffer[offset + 2]) and (
             buffer[offset + 1] != precision(0.0)
         )
         buffer[offset + 2] = selp(
             update_flag, second_derivative_unscaled, buffer[offset + 2]
         )
         buffer[offset + 1] = buffer[offset + 0]
         buffer[offset + 0] = value
     ```
   - Edge cases: Guards against insufficient previous values
   - Integration: Called from chain_metrics wrapper

2. **Update `d2xdt2_min.py` with explicit indexing**
   - File: src/cubie/outputhandling/summarymetrics/d2xdt2_min.py
   - Action: Modify
   - Details:
     ```python
     @cuda.jit(
         device=True,
         inline=True,
     )
     def update(
         value,
         buffer,
         offset,
         current_index,
         customisable_variable,
     ):
         """Update the minimum second derivative with a new value.

         Parameters
         ----------
         value
             float. New value to compute second derivative from.
         buffer
             device array. Full buffer containing metric working storage.
         offset
             int. Offset to this metric's storage within the buffer.
         current_index
             int. Current integration step index (unused).
         customisable_variable
             int. Metric parameter placeholder (unused).

         Notes
         -----
         Computes unscaled second derivative using central difference formula
         (value - 2*buffer[offset + 0] + buffer[offset + 1]) and updates
         buffer[offset + 2] if smaller. Uses predicated commit pattern to avoid
         warp divergence. Guard on buffer[offset + 1] ensures two previous
         values are available.
         """
         second_derivative_unscaled = (
             value
             - precision(2.0) * buffer[offset + 0]
             + buffer[offset + 1]
         )
         update_flag = (second_derivative_unscaled < buffer[offset + 2]) and (
             buffer[offset + 1] != precision(0.0)
         )
         buffer[offset + 2] = selp(
             update_flag, second_derivative_unscaled, buffer[offset + 2]
         )
         buffer[offset + 1] = buffer[offset + 0]
         buffer[offset + 0] = value
     ```
   - Edge cases: Guards against insufficient previous values
   - Integration: Called from chain_metrics wrapper

3. **Update `d2xdt2_extrema.py` with explicit indexing**
   - File: src/cubie/outputhandling/summarymetrics/d2xdt2_extrema.py
   - Action: Modify
   - Details:
     ```python
     @cuda.jit(
         device=True,
         inline=True,
     )
     def update(
         value,
         buffer,
         offset,
         current_index,
         customisable_variable,
     ):
         """Update maximum and minimum second derivatives with a new value.

         Parameters
         ----------
         value
             float. New value to compute second derivative from.
         buffer
             device array. Full buffer containing metric working storage.
         offset
             int. Offset to this metric's storage within the buffer.
         current_index
             int. Current integration step index (unused).
         customisable_variable
             int. Metric parameter placeholder (unused).

         Notes
         -----
         Computes unscaled second derivative using central difference formula
         (value - 2*buffer[offset + 0] + buffer[offset + 1]) and updates
         buffer[offset + 2] if larger and buffer[offset + 3] if smaller.
         Uses predicated commit pattern to avoid warp divergence. Guard on
         buffer[offset + 1] ensures two previous values are available.
         """
         second_derivative_unscaled = (
             value
             - precision(2.0) * buffer[offset + 0]
             + buffer[offset + 1]
         )
         update_max = (second_derivative_unscaled > buffer[offset + 2]) and (
             buffer[offset + 1] != precision(0.0)
         )
         update_min = (second_derivative_unscaled < buffer[offset + 3]) and (
             buffer[offset + 1] != precision(0.0)
         )
         buffer[offset + 2] = selp(
             update_max, second_derivative_unscaled, buffer[offset + 2]
         )
         buffer[offset + 3] = selp(
             update_min, second_derivative_unscaled, buffer[offset + 3]
         )
         buffer[offset + 1] = buffer[offset + 0]
         buffer[offset + 0] = value
     ```
   - Edge cases: Guards against insufficient previous values
   - Integration: Called from chain_metrics wrapper

**Outcomes**: 
- Files Modified: 
  * src/cubie/outputhandling/summarymetrics/d2xdt2_max.py (14 lines changed)
  * src/cubie/outputhandling/summarymetrics/d2xdt2_min.py (14 lines changed)
  * src/cubie/outputhandling/summarymetrics/d2xdt2_extrema.py (21 lines changed)
- Functions/Methods Added/Modified:
  * update() in d2xdt2_max.py - Added offset parameter, updated buffer indexing with selp pattern
  * update() in d2xdt2_min.py - Added offset parameter, updated buffer indexing with selp pattern
  * update() in d2xdt2_extrema.py - Added offset parameter, updated buffer indexing with selp pattern
- Implementation Summary:
  Updated all three second derivative metrics (d2xdt2_max, d2xdt2_min, d2xdt2_extrema) to use explicit offset indexing. Added offset as the third parameter to the update function signatures. Changed buffer access from buffer[N] to buffer[offset + N] for all read/write operations (current value at offset+0, previous value at offset+1, max/min derivative at offset+2, min derivative for extrema at offset+3). Updated docstrings to document the new offset parameter. The selp predicated commit pattern was already in use and continues to be used with the new indexing.
- Issues Flagged: None

---

## Task Group 8: Peak Detection Metrics - PARALLEL
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/outputhandling/summarymetrics/peaks.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/negative_peaks.py (entire file)
- File: src/cubie/cuda_simsafe.py (lines 200-260 for selp import)

**Input Validation Required**:
- None (CUDA device functions, no validation needed)

**Tasks**:

1. **Update `peaks.py` with explicit indexing and selp pattern**
   - File: src/cubie/outputhandling/summarymetrics/peaks.py
   - Action: Modify
   - Details:
     Add import for selp:
     ```python
     from cubie.cuda_simsafe import selp, compile_kwargs
     ```
     Update update function:
     ```python
     @cuda.jit(
         device=True,
         inline=True,
         **compile_kwargs,
     )
     def update(
         value,
         buffer,
         offset,
         current_index,
         customisable_variable,
     ):
         """Update peak detection with a new value.

         Parameters
         ----------
         value
             float. New value to analyse for peak detection.
         buffer
             device array. Full buffer containing metric working storage.
         offset
             int. Offset to this metric's storage within the buffer.
             Layout at offset: ``[prev, prev_prev, counter, times...]``.
         current_index
             int. Current integration step index, used to record peaks.
         customisable_variable
             int. Maximum number of peaks to detect.

         Notes
         -----
         Detects peaks when the prior value exceeds both the current and
         second-prior values. Peak indices are stored from
         ``buffer[offset + 3]`` onward. Uses predicated commit pattern
         for all buffer writes to avoid warp divergence.
         """
         npeaks = customisable_variable
         prev = buffer[offset + 0]
         prev_prev = buffer[offset + 1]
         peak_counter = int32(buffer[offset + 2])

         is_valid = (
             (current_index >= 2)
             and (peak_counter < npeaks)
             and (prev_prev != precision(0.0))
         )
         is_peak = prev > value and prev_prev < prev
         should_record = is_valid and is_peak

         # Predicated commit for peak recording
         new_counter = precision(int32(buffer[offset + 2]) + 1)
         buffer[offset + 3 + peak_counter] = selp(
             should_record,
             precision(current_index - 1),
             buffer[offset + 3 + peak_counter],
         )
         buffer[offset + 2] = selp(
             should_record, new_counter, buffer[offset + 2]
         )
         buffer[offset + 0] = value
         buffer[offset + 1] = prev
     ```
   - Edge cases: 
     - current_index < 2: no peaks detected yet
     - peak_counter >= npeaks: stop recording
   - Integration: Called from chain_metrics wrapper

2. **Update `negative_peaks.py` with explicit indexing and selp pattern**
   - File: src/cubie/outputhandling/summarymetrics/negative_peaks.py
   - Action: Modify
   - Details:
     Add import for selp:
     ```python
     from cubie.cuda_simsafe import selp, compile_kwargs
     ```
     Update update function:
     ```python
     @cuda.jit(
         device=True,
         inline=True,
         **compile_kwargs,
     )
     def update(
         value,
         buffer,
         offset,
         current_index,
         customisable_variable,
     ):
         """Update negative peak detection with a new value.

         Parameters
         ----------
         value
             float. New value to analyse for negative peak detection.
         buffer
             device array. Full buffer containing metric working storage.
         offset
             int. Offset to this metric's storage within the buffer.
             Layout at offset: ``[prev, prev_prev, counter, times...]``.
         current_index
             int. Current integration step index, used to record peaks.
         customisable_variable
             int. Maximum number of negative peaks to detect.

         Notes
         -----
         Detects negative peaks (local minima) when the prior value is
         less than both the current and second-prior values. Peak indices
         are stored from ``buffer[offset + 3]`` onward. Uses predicated
         commit pattern for all buffer writes to avoid warp divergence.
         """
         npeaks = customisable_variable
         prev = buffer[offset + 0]
         prev_prev = buffer[offset + 1]
         peak_counter = int32(buffer[offset + 2])

         is_valid = (
             (current_index >= 2)
             and (peak_counter < npeaks)
             and (prev_prev != precision(0.0))
         )
         is_peak = prev < value and prev_prev > prev
         should_record = is_valid and is_peak

         # Predicated commit for peak recording
         new_counter = precision(int32(buffer[offset + 2]) + 1)
         buffer[offset + 3 + peak_counter] = selp(
             should_record,
             precision(current_index - 1),
             buffer[offset + 3 + peak_counter],
         )
         buffer[offset + 2] = selp(
             should_record, new_counter, buffer[offset + 2]
         )
         buffer[offset + 0] = value
         buffer[offset + 1] = prev
     ```
   - Edge cases: 
     - current_index < 2: no peaks detected yet
     - peak_counter >= npeaks: stop recording
   - Integration: Called from chain_metrics wrapper

**Outcomes**: 
- Files Modified: 
  * src/cubie/outputhandling/summarymetrics/peaks.py (26 lines changed)
  * src/cubie/outputhandling/summarymetrics/negative_peaks.py (26 lines changed)
- Functions/Methods Added/Modified:
  * update() in peaks.py - Added offset parameter, updated buffer indexing with selp pattern
  * update() in negative_peaks.py - Added offset parameter, updated buffer indexing with selp pattern
- Implementation Summary:
  Updated both peak detection metrics (peaks and negative_peaks) to use explicit offset indexing and selp predicated commit pattern. Added selp import from cubie.cuda_simsafe. Changed update function signatures to include offset as the third parameter. Changed buffer access from buffer[N] to buffer[offset + N] for all read/write operations (prev at offset+0, prev_prev at offset+1, counter at offset+2, peak times at offset+3+peak_counter). Replaced nested if-conditions with predicated selp pattern using is_valid, is_peak, and should_record flags to avoid warp divergence. Updated docstrings to document the new offset parameter and selp usage.
- Issues Flagged: None

---

## Task Group 9: Documentation Update - SEQUENTIAL
**Status**: [x]
**Dependencies**: Task Groups 2-8

**Required Context**:
- File: src/cubie/outputhandling/summarymetrics/metrics.py (lines 127-196)

**Input Validation Required**:
- None (documentation only)

**Tasks**:

1. **Update SummaryMetric docstring to reflect new signature**
   - File: src/cubie/outputhandling/summarymetrics/metrics.py
   - Action: Modify
   - Details:
     Update the class docstring's Notes section (around line 127):
     ```python
     """Abstract base class for summary metrics.

     Attributes
     ----------
     buffer_size
         int or Callable. Memory required per metric buffer entry.
         Parameterised metrics should supply a callable that accepts
         the metric parameter.
     output_size
         int or Callable. Memory required for persisted metric results.
     name
         str. Identifier used in registries and configuration strings.
     unit_modification
         str. Format string for unit modification in legends.
     update_device_func
         Callable. Compiled CUDA device update function for the metric.
     save_device_func
         Callable. Compiled CUDA device save function for the metric.
     dt_save
         save interval. Defaults to 0.01.
     precision
         Numerical precision for metric calculations. Defaults to
         np.float32.

     Notes
     -----
     Subclasses must implement :meth:`build` to provide CUDA device
     callbacks with the signatures ``update(value, buffer, offset,
     current_index, customisable_variable)`` and ``save(buffer,
     output_array, summarise_every, customisable_variable)``. The
     ``offset`` parameter indicates the starting position within the
     full buffer for this metric's storage, eliminating the need for
     buffer slicing. Metrics should use explicit indexing patterns
     like ``buffer[offset + 0]`` for optimal register efficiency.
     Metrics need to be imported only after the global registry has
     been created so that decoration registers the implementation.
     """
     ```
   - Edge cases: None
   - Integration: Documentation only

**Outcomes**: 
- Files Modified: 
  * src/cubie/outputhandling/summarymetrics/metrics.py (4 lines changed)
- Functions/Methods Added/Modified:
  * SummaryMetric class docstring - Updated Notes section
- Implementation Summary:
  Updated the SummaryMetric class docstring Notes section to reflect the new update function signature. Changed the documented signature from `update(value, buffer, current_index, customisable_variable)` to `update(value, buffer, offset, current_index, customisable_variable)`. Added documentation explaining that the `offset` parameter indicates the starting position within the full buffer for this metric's storage, eliminating the need for buffer slicing. Added note recommending explicit indexing patterns like `buffer[offset + 0]` for optimal register efficiency.
- Issues Flagged: None

---

## Summary

### Total Task Groups: 9
### Dependency Chain:
1. Task Group 1 (Core Infrastructure) - MUST complete first
2. Task Groups 2-8 (Metric Updates) - Can run in PARALLEL after Group 1
3. Task Group 9 (Documentation) - Should run after all metric updates

### Parallel Execution Opportunities:
- After Task Group 1 completes, Task Groups 2-8 can all execute in parallel
- Within each parallel group, individual file updates are independent

### Estimated Complexity:
- Task Group 1: HIGH (core architectural change)
- Task Groups 2-8: MEDIUM (repetitive pattern application)
- Task Group 9: LOW (documentation only)

### Files Modified Summary:
| File | Task Group |
|------|------------|
| update_summaries.py | 1 |
| mean.py | 2 |
| rms.py | 2 |
| max.py | 3 |
| min.py | 3 |
| max_magnitude.py | 3 |
| std.py | 4 |
| extrema.py | 4 |
| mean_std.py | 5 |
| mean_std_rms.py | 5 |
| std_rms.py | 5 |
| dxdt_max.py | 6 |
| dxdt_min.py | 6 |
| dxdt_extrema.py | 6 |
| d2xdt2_max.py | 7 |
| d2xdt2_min.py | 7 |
| d2xdt2_extrema.py | 7 |
| peaks.py | 8 |
| negative_peaks.py | 8 |
| metrics.py | 9 |
