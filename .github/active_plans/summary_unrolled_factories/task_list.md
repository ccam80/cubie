# Implementation Task List
# Feature: Summary Metrics Unrolled Factories
# Plan Reference: .github/active_plans/summary_unrolled_factories/agent_plan.md

## Task Group 1: Fully Unrolled Update Factory

**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: tests/all_in_one.py (lines 4000-4041 for INLINE_UPDATE_FUNCTIONS dict)
- File: tests/all_in_one.py (lines 4044-4071 for inline_update_functions helper)
- File: tests/all_in_one.py (lines 4104-4191 for chain_metrics_update and do_nothing_update)
- File: tests/all_in_one.py (lines 4193-4251 for update_summary_factory)
- File: tests/all_in_one.py (lines 14-21 for imports including compile_kwargs)
- File: src/cubie/outputhandling/summarymetrics/metrics.py (lines 409-445 for preprocess_request)
- File: src/cubie/outputhandling/summarymetrics/metrics.py (lines 477-523 for buffer_offsets and buffer_sizes)
- File: src/cubie/outputhandling/summarymetrics/metrics.py (lines 770-784 for params method)

**Input Validation Required**:
- None (use existing validation from summary_metrics.preprocess_request)

**Tasks**:
1. **Create unrolled_update_summary_factory function**
   - File: tests/all_in_one.py
   - Action: Create new function after save_summary_factory (around line 4415)
   - Details:
     ```python
     def unrolled_update_summary_factory(
         summaries_buffer_height_per_var: int,
         summarised_state_indices: tuple,
         summarised_observable_indices: tuple,
         summaries_list: tuple,
     ):
         """Generate unrolled summary update device function.
         
         Uses compile-time boolean flags to conditionally execute each
         metric update within a single device function body. Each metric
         is guarded by a boolean captured in closure.
         
         Parameters
         ----------
         summaries_buffer_height_per_var
             Total buffer size per variable.
         summarised_state_indices
             Array of state indices to summarize.
         summarised_observable_indices
             Array of observable indices to summarize.
         summaries_list
             Tuple of metric names to enable.
             
         Returns
         -------
         Callable
             CUDA device function for updating summary metrics.
         """
         # Implementation logic:
         # 1. Convert indices to int32 and compute basic sizes
         num_summarised_states = int32(len(summarised_state_indices))
         num_summarised_observables = int32(len(summarised_observable_indices))
         total_buffer_size = int32(summaries_buffer_height_per_var)
         
         # 2. Use preprocess_request to get the canonical metric list
         parsed_summaries = tuple(
             summary_metrics.preprocess_request(list(summaries_list))
         )
         num_metrics = len(parsed_summaries)
         
         # 3. Compute compile-time boolean flags for each known metric
         summarise_states = (num_summarised_states > 0) and (num_metrics > 0)
         summarise_observables = (num_summarised_observables > 0) and (
             num_metrics > 0
         )
         
         # Handle empty case early
         if num_metrics == 0:
             @cuda.jit(device=True, inline=True, **compile_kwargs)
             def do_nothing_update_summary(...):
                 pass
             return do_nothing_update_summary
         
         # 4. Get buffer metadata from summary_metrics registry
         buffer_offsets_list = summary_metrics.buffer_offsets(summaries_list)
         buffer_sizes_list = summary_metrics.buffer_sizes(summaries_list)
         params_list = summary_metrics.params(summaries_list)
         
         # 5. Extract per-metric enable flags, offsets, sizes, params
         #    For each metric in INLINE_UPDATE_FUNCTIONS:
         enable_mean = 'mean' in parsed_summaries
         enable_max = 'max' in parsed_summaries
         enable_min = 'min' in parsed_summaries
         # ... (all 18 metrics)
         
         # Get index positions and capture offset/size/param for enabled metrics
         # If enabled, capture: offset=buffer_offsets_list[idx], etc.
         # If not enabled, set dummy values (won't be used)
         
         # 6. Capture inline update functions in closure
         # (reference INLINE_UPDATE_FUNCTIONS dict values)
         
         # 7. Define the device function with conditional execution
         @cuda.jit(device=True, inline=True, forceinline=True, **compile_kwargs)
         def update_summary_metrics_func(
             current_state,
             current_observables,
             state_summary_buffer,
             observable_summary_buffer,
             current_step,
         ):
             if summarise_states:
                 for idx in range(num_summarised_states):
                     base = idx * total_buffer_size
                     value = current_state[summarised_state_indices[idx]]
                     buf = state_summary_buffer[base:base + total_buffer_size]
                     
                     # Conditional calls guarded by compile-time booleans
                     if enable_mean:
                         update_mean(value, buf[mean_offset:mean_offset+mean_size],
                                     current_step, mean_param)
                     if enable_max:
                         update_max(value, buf[max_offset:max_offset+max_size],
                                    current_step, max_param)
                     # ... repeat for all metrics
             
             if summarise_observables:
                 for idx in range(num_summarised_observables):
                     base = idx * total_buffer_size
                     value = current_observables[
                         summarised_observable_indices[idx]
                     ]
                     buf = observable_summary_buffer[base:base + total_buffer_size]
                     
                     # Same conditional calls for observables
         
         return update_summary_metrics_func
     ```
   - Edge cases:
     - Empty summaries_list: return do-nothing function
     - No states to summarize: skip state loop entirely
     - No observables to summarize: skip observable loop entirely
     - Parameterized metrics (peaks[N]): extract param value from params_list
     - Combined metrics (extrema): handled by preprocess_request substitution
   - Integration: Uses same signature as existing update_summary_factory

2. **Implement all 18 metric enable flags and their metadata extraction**
   - File: tests/all_in_one.py
   - Action: Within unrolled_update_summary_factory
   - Details:
     ```python
     # Build lookup from parsed_summaries for index positions
     metric_indices = {m: i for i, m in enumerate(parsed_summaries)}
     
     # For each known metric, extract enable flag and metadata
     KNOWN_METRICS = (
         'mean', 'max', 'min', 'rms', 'std', 'mean_std', 'mean_std_rms',
         'std_rms', 'extrema', 'peaks', 'negative_peaks', 'max_magnitude',
         'dxdt_max', 'dxdt_min', 'dxdt_extrema', 'd2xdt2_max', 'd2xdt2_min',
         'd2xdt2_extrema',
     )
     
     # Create dictionaries for enabled flags, offsets, sizes, params
     # Capture as local variables for closure
     enable_flags = {}
     offsets = {}
     sizes = {}
     params = {}
     
     for metric in KNOWN_METRICS:
         if metric in metric_indices:
             idx = metric_indices[metric]
             enable_flags[metric] = True
             offsets[metric] = int32(buffer_offsets_list[idx])
             sizes[metric] = int32(buffer_sizes_list[idx])
             params[metric] = int32(params_list[idx])
         else:
             enable_flags[metric] = False
             offsets[metric] = int32(0)
             sizes[metric] = int32(0)
             params[metric] = int32(0)
     
     # Extract individual variables for closure capture
     enable_mean = enable_flags['mean']
     mean_offset = offsets['mean']
     mean_size = sizes['mean']
     mean_param = params['mean']
     # ... repeat for all 18 metrics
     ```
   - Edge cases: Unknown metrics filtered out by preprocess_request
   - Integration: Values captured in device function closure

**Tests to Create**:
- Test file: tests/all_in_one.py (inline validation, not separate test file)
- Description: The unrolled factory should be validated by comparing outputs with existing chaining implementation when running all_in_one.py

**Tests to Run**:
- Manual validation by running tests/all_in_one.py and comparing output arrays

**Outcomes**: 
- Files Modified: 
  * tests/all_in_one.py (~660 lines added)
- Functions/Methods Added/Modified:
  * unrolled_update_summary_factory() in tests/all_in_one.py
- Implementation Summary:
  Created `unrolled_update_summary_factory` function at line 4417 in tests/all_in_one.py.
  The function uses compile-time boolean flags captured in closure for each of the 18 known metrics.
  Each metric has its own enable flag, offset, size, and param variables extracted at factory call time.
  The device function iterates over states and observables with conditional execution guarded by boolean flags.
  Edge cases handled: empty summaries_list returns do_nothing_update_summary, empty state/observable indices skip respective loops.
  Uses same signature as existing update_summary_factory for drop-in replacement.
- Issues Flagged: None

---

## Task Group 2: Fully Unrolled Save Factory

**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: tests/all_in_one.py (lines 4022-4041 for INLINE_SAVE_FUNCTIONS dict)
- File: tests/all_in_one.py (lines 4074-4101 for inline_save_functions helper)
- File: tests/all_in_one.py (lines 4254-4264 for do_nothing_save)
- File: tests/all_in_one.py (lines 4267-4329 for chain_metrics_save)
- File: tests/all_in_one.py (lines 4332-4414 for save_summary_factory)
- File: src/cubie/outputhandling/summarymetrics/metrics.py (lines 525-549 for output_offsets)
- File: src/cubie/outputhandling/summarymetrics/metrics.py (lines 577-599 for summaries_output_height)
- File: src/cubie/outputhandling/summarymetrics/metrics.py (lines 702-722 for output_sizes)
- Completed: Task Group 1 for pattern reference

**Input Validation Required**:
- None (use existing validation from summary_metrics.preprocess_request)

**Tasks**:
1. **Create unrolled_save_summary_factory function**
   - File: tests/all_in_one.py
   - Action: Create new function after unrolled_update_summary_factory
   - Details:
     ```python
     def unrolled_save_summary_factory(
         summaries_buffer_height_per_var: int,
         summarised_state_indices: tuple,
         summarised_observable_indices: tuple,
         summaries_list: tuple,
     ):
         """Generate unrolled summary save device function.
         
         Uses compile-time boolean flags to conditionally execute each
         metric save within a single device function body. Each metric
         is guarded by a boolean captured in closure.
         
         Parameters
         ----------
         summaries_buffer_height_per_var
             Total buffer size per variable.
         summarised_state_indices
             Array of state indices to summarize.
         summarised_observable_indices
             Array of observable indices to summarize.
         summaries_list
             Tuple of metric names to enable.
             
         Returns
         -------
         Callable
             CUDA device function for saving summary metrics.
         """
         # Implementation logic:
         # 1. Convert indices to int32 and compute basic sizes
         num_summarised_states = int32(len(summarised_state_indices))
         num_summarised_observables = int32(len(summarised_observable_indices))
         total_buffer_size = int32(summaries_buffer_height_per_var)
         total_output_size = int32(
             summary_metrics.summaries_output_height(summaries_list)
         )
         
         # 2. Use preprocess_request to get the canonical metric list
         parsed_summaries = tuple(
             summary_metrics.preprocess_request(list(summaries_list))
         )
         num_metrics = len(parsed_summaries)
         
         # 3. Compute compile-time boolean flags
         summarise_states = (num_summarised_states > 0) and (num_metrics > 0)
         summarise_observables = (num_summarised_observables > 0) and (
             num_metrics > 0
         )
         
         # Handle empty case
         if num_metrics == 0:
             @cuda.jit(device=True, inline=True, **compile_kwargs)
             def do_nothing_save_summary(...):
                 pass
             return do_nothing_save_summary
         
         # 4. Get buffer and output metadata
         buffer_offsets_list = summary_metrics.buffer_offsets(summaries_list)
         buffer_sizes_list = summary_metrics.buffer_sizes(summaries_list)
         output_offsets_list = summary_metrics.output_offsets(summaries_list)
         output_sizes_list = summary_metrics.output_sizes(summaries_list)
         params_list = summary_metrics.params(summaries_list)
         
         # 5. Extract per-metric metadata (same pattern as update factory)
         # enable_mean, mean_buffer_offset, mean_buffer_size,
         # mean_output_offset, mean_output_size, mean_param
         # ... for all 18 metrics
         
         # 6. Capture inline save functions in closure
         
         # 7. Define the device function
         @cuda.jit(device=True, inline=True, forceinline=True, **compile_kwargs)
         def save_summary_metrics_func(
             buffer_state_summaries,
             buffer_observable_summaries,
             output_state_summaries_window,
             output_observable_summaries_window,
         ):
             if summarise_states:
                 for state_index in range(num_summarised_states):
                     buf_start = state_index * total_buffer_size
                     out_start = state_index * total_output_size
                     buf = buffer_state_summaries[
                         buf_start:buf_start + total_buffer_size
                     ]
                     out = output_state_summaries_window[
                         out_start:out_start + total_output_size
                     ]
                     
                     # Conditional save calls guarded by compile-time booleans
                     if enable_mean:
                         save_mean(
                             buf[mean_buf_off:mean_buf_off + mean_buf_sz],
                             out[mean_out_off:mean_out_off + mean_out_sz],
                             mean_param,
                         )
                     # ... repeat for all metrics
             
             if summarise_observables:
                 for obs_index in range(num_summarised_observables):
                     buf_start = obs_index * total_buffer_size
                     out_start = obs_index * total_output_size
                     # Same pattern for observables
         
         return save_summary_metrics_func
     ```
   - Edge cases:
     - Empty summaries_list: return do-nothing function
     - No states/observables: skip respective loops
     - Parameterized metrics: param extracted from params_list
   - Integration: Uses same signature as existing save_summary_factory

2. **Implement all 18 metric enable flags with buffer and output metadata**
   - File: tests/all_in_one.py
   - Action: Within unrolled_save_summary_factory
   - Details:
     ```python
     # Same pattern as update factory, but include output offsets/sizes
     KNOWN_METRICS = (
         'mean', 'max', 'min', 'rms', 'std', 'mean_std', 'mean_std_rms',
         'std_rms', 'extrema', 'peaks', 'negative_peaks', 'max_magnitude',
         'dxdt_max', 'dxdt_min', 'dxdt_extrema', 'd2xdt2_max', 'd2xdt2_min',
         'd2xdt2_extrema',
     )
     
     metric_indices = {m: i for i, m in enumerate(parsed_summaries)}
     
     # For each metric, extract:
     # - enable flag
     # - buffer offset and size
     # - output offset and size
     # - param value
     
     # Create individual closure-captured variables:
     enable_mean = 'mean' in metric_indices
     mean_buf_off = int32(buffer_offsets_list[metric_indices['mean']]) \
         if enable_mean else int32(0)
     mean_buf_sz = int32(buffer_sizes_list[metric_indices['mean']]) \
         if enable_mean else int32(0)
     mean_out_off = int32(output_offsets_list[metric_indices['mean']]) \
         if enable_mean else int32(0)
     mean_out_sz = int32(output_sizes_list[metric_indices['mean']]) \
         if enable_mean else int32(0)
     mean_param = int32(params_list[metric_indices['mean']]) \
         if enable_mean else int32(0)
     # ... repeat for all 18 metrics
     ```
   - Edge cases: Metrics not in parsed_summaries get dummy values
   - Integration: Values captured in device function closure

**Tests to Create**:
- Test file: tests/all_in_one.py (inline validation)
- Description: Compare unrolled save outputs with existing chaining implementation

**Tests to Run**:
- Manual validation by running tests/all_in_one.py

**Outcomes**: 
- Files Modified: 
  * tests/all_in_one.py (~540 lines added)
- Functions/Methods Added/Modified:
  * unrolled_save_summary_factory() in tests/all_in_one.py
- Implementation Summary:
  Created `unrolled_save_summary_factory` function after `unrolled_update_summary_factory` in tests/all_in_one.py.
  The function uses compile-time boolean flags captured in closure for each of the 18 known metrics.
  Each metric has its own enable flag, buffer offset/size, output offset/size, and param variables extracted at factory call time.
  Uses `summary_metrics.summaries_output_height()` to compute total_output_size for output slicing.
  The device function iterates over states and observables with conditional execution guarded by boolean flags.
  Edge cases handled: empty summaries_list returns do_nothing_save_summary, empty state/observable indices skip respective loops.
  Uses same signature as existing save_summary_factory for drop-in replacement.
- Issues Flagged: None

---

## Task Group 3: SymPy Codegen Update Factory

**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/odesystems/symbolic/codegen/dxdt.py (lines 25-45 for DXDT_TEMPLATE pattern)
- File: src/cubie/odesystems/symbolic/codegen/dxdt.py (lines 180-223 for generate_dxdt_fac_code pattern)
- File: tests/all_in_one.py (lines 4000-4041 for INLINE_UPDATE_FUNCTIONS dict)
- File: tests/all_in_one.py (lines 4193-4251 for update_summary_factory signature and logic)
- Completed: Task Group 1 for understanding metric structure

**Input Validation Required**:
- None (use existing validation from summary_metrics.preprocess_request)

**Tasks**:
1. **Define UPDATE_SUMMARY_TEMPLATE string constant**
   - File: tests/all_in_one.py
   - Action: Create constant after save_summary_factory (around line 4415)
   - Details:
     ```python
     UPDATE_SUMMARY_TEMPLATE = '''
     def {func_name}():
         """Auto-generated summary update factory."""
         from numba import cuda, int32
         from cubie.cuda_simsafe import compile_kwargs
     {constant_assignments}
         
         @cuda.jit(device=True, inline=True, forceinline=True, **compile_kwargs)
         def update_summary_metrics_func(
             current_state,
             current_observables,
             state_summary_buffer,
             observable_summary_buffer,
             current_step,
         ):
     {body}
         
         return update_summary_metrics_func
     '''
     ```
   - Edge cases: Template must handle proper indentation
   - Integration: Follows DXDT_TEMPLATE pattern from dxdt.py

2. **Create codegen_update_summary_factory function**
   - File: tests/all_in_one.py
   - Action: Create function after template constant
   - Details:
     ```python
     def codegen_update_summary_factory(
         summaries_buffer_height_per_var: int,
         summarised_state_indices: tuple,
         summarised_observable_indices: tuple,
         summaries_list: tuple,
     ):
         """Generate summary update device function via code generation.
         
         Generates Python source code as a string and uses exec() to
         compile the generated code into a CUDA device function.
         
         Parameters
         ----------
         summaries_buffer_height_per_var
             Total buffer size per variable.
         summarised_state_indices
             Array of state indices to summarize.
         summarised_observable_indices
             Array of observable indices to summarize.
         summaries_list
             Tuple of metric names to enable.
             
         Returns
         -------
         Callable
             CUDA device function for updating summary metrics.
         """
         # Implementation logic:
         # 1. Preprocess metrics and compute metadata
         parsed_summaries = tuple(
             summary_metrics.preprocess_request(list(summaries_list))
         )
         num_metrics = len(parsed_summaries)
         
         num_summarised_states = len(summarised_state_indices)
         num_summarised_observables = len(summarised_observable_indices)
         summarise_states = (num_summarised_states > 0) and (num_metrics > 0)
         summarise_observables = (num_summarised_observables > 0) and (
             num_metrics > 0
         )
         total_buffer_size = summaries_buffer_height_per_var
         
         # Handle empty case
         if num_metrics == 0:
             @cuda.jit(device=True, inline=True, **compile_kwargs)
             def do_nothing_update_summary(...):
                 pass
             return do_nothing_update_summary
         
         # 2. Get metadata from registry
         buffer_offsets_list = summary_metrics.buffer_offsets(summaries_list)
         buffer_sizes_list = summary_metrics.buffer_sizes(summaries_list)
         params_list = summary_metrics.params(summaries_list)
         
         # 3. Generate constant assignments
         const_lines = []
         const_lines.append(
             f"    num_summarised_states = int32({num_summarised_states})"
         )
         const_lines.append(
             f"    num_summarised_observables = int32({num_summarised_observables})"
         )
         const_lines.append(
             f"    total_buffer_size = int32({total_buffer_size})"
         )
         const_lines.append(
             f"    summarised_state_indices = {summarised_state_indices}"
         )
         const_lines.append(
             f"    summarised_observable_indices = {summarised_observable_indices}"
         )
         
         # Add per-metric constants
         for i, metric in enumerate(parsed_summaries):
             const_lines.append(
                 f"    {metric}_offset = int32({buffer_offsets_list[i]})"
             )
             const_lines.append(
                 f"    {metric}_size = int32({buffer_sizes_list[i]})"
             )
             const_lines.append(
                 f"    {metric}_param = int32({params_list[i]})"
             )
         
         constant_assignments = '\n'.join(const_lines)
         
         # 4. Generate body code
         body_lines = []
         indent = '        '
         
         if summarise_states:
             body_lines.append(f"{indent}for idx in range(num_summarised_states):")
             body_lines.append(f"{indent}    base = idx * total_buffer_size")
             body_lines.append(
                 f"{indent}    value = current_state[summarised_state_indices[idx]]"
             )
             for metric in parsed_summaries:
                 # Generate inline call to update function
                 body_lines.append(
                     f"{indent}    update_{metric}("
                 )
                 body_lines.append(
                     f"{indent}        value,"
                 )
                 body_lines.append(
                     f"{indent}        state_summary_buffer[base + {metric}_offset:"
                     f"base + {metric}_offset + {metric}_size],"
                 )
                 body_lines.append(
                     f"{indent}        current_step,"
                 )
                 body_lines.append(
                     f"{indent}        {metric}_param,"
                 )
                 body_lines.append(f"{indent}    )")
         
         if summarise_observables:
             body_lines.append(f"{indent}for idx in range(num_summarised_observables):")
             body_lines.append(f"{indent}    base = idx * total_buffer_size")
             body_lines.append(
                 f"{indent}    value = current_observables["
                 "summarised_observable_indices[idx]]"
             )
             for metric in parsed_summaries:
                 body_lines.append(
                     f"{indent}    update_{metric}("
                 )
                 body_lines.append(
                     f"{indent}        value,"
                 )
                 body_lines.append(
                     f"{indent}        observable_summary_buffer[base + {metric}_offset:"
                     f"base + {metric}_offset + {metric}_size],"
                 )
                 body_lines.append(
                     f"{indent}        current_step,"
                 )
                 body_lines.append(
                     f"{indent}        {metric}_param,"
                 )
                 body_lines.append(f"{indent}    )")
         
         if not body_lines:
             body_lines.append(f"{indent}pass")
         
         body = '\n'.join(body_lines)
         
         # 5. Generate source code from template
         source_code = UPDATE_SUMMARY_TEMPLATE.format(
             func_name='_generated_update_factory',
             constant_assignments=constant_assignments,
             body=body,
         )
         
         # 6. Build namespace with required imports and functions
         namespace = {
             'cuda': cuda,
             'int32': int32,
             'compile_kwargs': compile_kwargs,
         }
         # Add all update functions to namespace
         for metric in parsed_summaries:
             namespace[f'update_{metric}'] = INLINE_UPDATE_FUNCTIONS[metric]
         
         # 7. Execute generated source
         exec(source_code, namespace)
         
         # 8. Retrieve and call the factory function
         factory_func = namespace['_generated_update_factory']
         return factory_func()
     ```
   - Edge cases:
     - Empty summaries_list: return do-nothing function before codegen
     - No states/observables: generate appropriate pass or skip loop
     - Parameterized metrics: param values embedded as literals in generated code
   - Integration: Uses exec() pattern similar to dxdt.py codegen

**Tests to Create**:
- Test file: tests/all_in_one.py (inline validation)
- Description: Compare codegen update outputs with existing chaining implementation

**Tests to Run**:
- Manual validation by running tests/all_in_one.py

**Outcomes**: 
- Files Modified: 
  * tests/all_in_one.py (~200 lines added)
- Functions/Methods Added/Modified:
  * UPDATE_SUMMARY_TEMPLATE constant in tests/all_in_one.py
  * codegen_update_summary_factory() in tests/all_in_one.py
- Implementation Summary:
  Created `UPDATE_SUMMARY_TEMPLATE` string constant following the DXDT_TEMPLATE pattern from dxdt.py.
  Created `codegen_update_summary_factory` function at line 5796 in tests/all_in_one.py.
  The function uses code generation with exec() to create CUDA device functions at runtime.
  Generates constant assignments for all metadata (offsets, sizes, params) and loop body code.
  Edge cases handled: empty summaries_list returns do_nothing_update_summary before codegen.
  No states/observables to summarize generates appropriate empty loops or pass.
  Uses exec() pattern similar to dxdt.py codegen with namespace containing metric update functions.
  Uses same signature as existing update_summary_factory for drop-in replacement.
- Issues Flagged: None

---

## Task Group 4: SymPy Codegen Save Factory

**Status**: [x]
**Dependencies**: Task Groups 2 and 3

**Required Context**:
- File: src/cubie/odesystems/symbolic/codegen/dxdt.py (lines 47-65 for OBSERVABLES_TEMPLATE pattern)
- File: tests/all_in_one.py (lines 4022-4041 for INLINE_SAVE_FUNCTIONS dict)
- File: tests/all_in_one.py (lines 4332-4414 for save_summary_factory signature and logic)
- Completed: Task Group 2 for save metadata extraction
- Completed: Task Group 3 for codegen pattern

**Input Validation Required**:
- None (use existing validation from summary_metrics.preprocess_request)

**Tasks**:
1. **Define SAVE_SUMMARY_TEMPLATE string constant**
   - File: tests/all_in_one.py
   - Action: Create constant after UPDATE_SUMMARY_TEMPLATE
   - Details:
     ```python
     SAVE_SUMMARY_TEMPLATE = '''
     def {func_name}():
         """Auto-generated summary save factory."""
         from numba import cuda, int32
         from cubie.cuda_simsafe import compile_kwargs
     {constant_assignments}
         
         @cuda.jit(device=True, inline=True, forceinline=True, **compile_kwargs)
         def save_summary_metrics_func(
             buffer_state_summaries,
             buffer_observable_summaries,
             output_state_summaries_window,
             output_observable_summaries_window,
         ):
     {body}
         
         return save_summary_metrics_func
     '''
     ```
   - Edge cases: Template must handle proper indentation
   - Integration: Follows same pattern as UPDATE_SUMMARY_TEMPLATE

2. **Create codegen_save_summary_factory function**
   - File: tests/all_in_one.py
   - Action: Create function after SAVE_SUMMARY_TEMPLATE
   - Details:
     ```python
     def codegen_save_summary_factory(
         summaries_buffer_height_per_var: int,
         summarised_state_indices: tuple,
         summarised_observable_indices: tuple,
         summaries_list: tuple,
     ):
         """Generate summary save device function via code generation.
         
         Generates Python source code as a string and uses exec() to
         compile the generated code into a CUDA device function.
         
         Parameters
         ----------
         summaries_buffer_height_per_var
             Total buffer size per variable.
         summarised_state_indices
             Array of state indices to summarize.
         summarised_observable_indices
             Array of observable indices to summarize.
         summaries_list
             Tuple of metric names to enable.
             
         Returns
         -------
         Callable
             CUDA device function for saving summary metrics.
         """
         # Implementation logic:
         # 1. Preprocess metrics and compute metadata
         parsed_summaries = tuple(
             summary_metrics.preprocess_request(list(summaries_list))
         )
         num_metrics = len(parsed_summaries)
         
         num_summarised_states = len(summarised_state_indices)
         num_summarised_observables = len(summarised_observable_indices)
         summarise_states = (num_summarised_states > 0) and (num_metrics > 0)
         summarise_observables = (num_summarised_observables > 0) and (
             num_metrics > 0
         )
         total_buffer_size = summaries_buffer_height_per_var
         total_output_size = summary_metrics.summaries_output_height(
             summaries_list
         )
         
         # Handle empty case
         if num_metrics == 0:
             @cuda.jit(device=True, inline=True, **compile_kwargs)
             def do_nothing_save_summary(...):
                 pass
             return do_nothing_save_summary
         
         # 2. Get metadata from registry
         buffer_offsets_list = summary_metrics.buffer_offsets(summaries_list)
         buffer_sizes_list = summary_metrics.buffer_sizes(summaries_list)
         output_offsets_list = summary_metrics.output_offsets(summaries_list)
         output_sizes_list = summary_metrics.output_sizes(summaries_list)
         params_list = summary_metrics.params(summaries_list)
         
         # 3. Generate constant assignments
         const_lines = []
         const_lines.append(
             f"    num_summarised_states = int32({num_summarised_states})"
         )
         const_lines.append(
             f"    num_summarised_observables = int32({num_summarised_observables})"
         )
         const_lines.append(
             f"    total_buffer_size = int32({total_buffer_size})"
         )
         const_lines.append(
             f"    total_output_size = int32({total_output_size})"
         )
         
         # Add per-metric constants
         for i, metric in enumerate(parsed_summaries):
             const_lines.append(
                 f"    {metric}_buf_off = int32({buffer_offsets_list[i]})"
             )
             const_lines.append(
                 f"    {metric}_buf_sz = int32({buffer_sizes_list[i]})"
             )
             const_lines.append(
                 f"    {metric}_out_off = int32({output_offsets_list[i]})"
             )
             const_lines.append(
                 f"    {metric}_out_sz = int32({output_sizes_list[i]})"
             )
             const_lines.append(
                 f"    {metric}_param = int32({params_list[i]})"
             )
         
         constant_assignments = '\n'.join(const_lines)
         
         # 4. Generate body code
         body_lines = []
         indent = '        '
         
         if summarise_states:
             body_lines.append(
                 f"{indent}for state_index in range(num_summarised_states):"
             )
             body_lines.append(
                 f"{indent}    buf_start = state_index * total_buffer_size"
             )
             body_lines.append(
                 f"{indent}    out_start = state_index * total_output_size"
             )
             for metric in parsed_summaries:
                 body_lines.append(
                     f"{indent}    save_{metric}("
                 )
                 body_lines.append(
                     f"{indent}        buffer_state_summaries["
                     f"buf_start + {metric}_buf_off:"
                     f"buf_start + {metric}_buf_off + {metric}_buf_sz],"
                 )
                 body_lines.append(
                     f"{indent}        output_state_summaries_window["
                     f"out_start + {metric}_out_off:"
                     f"out_start + {metric}_out_off + {metric}_out_sz],"
                 )
                 body_lines.append(
                     f"{indent}        {metric}_param,"
                 )
                 body_lines.append(f"{indent}    )")
         
         if summarise_observables:
             body_lines.append(
                 f"{indent}for obs_index in range(num_summarised_observables):"
             )
             body_lines.append(
                 f"{indent}    buf_start = obs_index * total_buffer_size"
             )
             body_lines.append(
                 f"{indent}    out_start = obs_index * total_output_size"
             )
             for metric in parsed_summaries:
                 body_lines.append(
                     f"{indent}    save_{metric}("
                 )
                 body_lines.append(
                     f"{indent}        buffer_observable_summaries["
                     f"buf_start + {metric}_buf_off:"
                     f"buf_start + {metric}_buf_off + {metric}_buf_sz],"
                 )
                 body_lines.append(
                     f"{indent}        output_observable_summaries_window["
                     f"out_start + {metric}_out_off:"
                     f"out_start + {metric}_out_off + {metric}_out_sz],"
                 )
                 body_lines.append(
                     f"{indent}        {metric}_param,"
                 )
                 body_lines.append(f"{indent}    )")
         
         if not body_lines:
             body_lines.append(f"{indent}pass")
         
         body = '\n'.join(body_lines)
         
         # 5. Generate source code from template
         source_code = SAVE_SUMMARY_TEMPLATE.format(
             func_name='_generated_save_factory',
             constant_assignments=constant_assignments,
             body=body,
         )
         
         # 6. Build namespace with required imports and functions
         namespace = {
             'cuda': cuda,
             'int32': int32,
             'compile_kwargs': compile_kwargs,
         }
         # Add all save functions to namespace
         for metric in parsed_summaries:
             namespace[f'save_{metric}'] = INLINE_SAVE_FUNCTIONS[metric]
         
         # 7. Execute generated source
         exec(source_code, namespace)
         
         # 8. Retrieve and call the factory function
         factory_func = namespace['_generated_save_factory']
         return factory_func()
     ```
   - Edge cases:
     - Empty summaries_list: return do-nothing function
     - No states/observables: skip respective loops in generated code
     - Parameterized metrics: param values embedded as literals
   - Integration: Uses same exec() pattern as codegen_update_summary_factory

**Tests to Create**:
- Test file: tests/all_in_one.py (inline validation)
- Description: Compare codegen save outputs with existing chaining implementation

**Tests to Run**:
- Manual validation by running tests/all_in_one.py

**Outcomes**: 
- Files Modified: 
  * tests/all_in_one.py (~200 lines added)
- Functions/Methods Added/Modified:
  * SAVE_SUMMARY_TEMPLATE constant in tests/all_in_one.py
  * codegen_save_summary_factory() in tests/all_in_one.py
- Implementation Summary:
  Created `SAVE_SUMMARY_TEMPLATE` string constant following the same pattern as UPDATE_SUMMARY_TEMPLATE.
  Created `codegen_save_summary_factory` function after `codegen_update_summary_factory` in tests/all_in_one.py.
  The function uses code generation with exec() to create CUDA device functions at runtime.
  Generates constant assignments for all metadata (buffer offsets/sizes, output offsets/sizes, params) and loop body code.
  Edge cases handled: empty summaries_list returns do_nothing_save_summary before codegen.
  No states/observables to summarize generates appropriate empty loops or pass.
  Uses exec() pattern with namespace containing metric save functions from INLINE_SAVE_FUNCTIONS.
  Uses same signature as existing save_summary_factory for drop-in replacement.
- Issues Flagged: None

---

## Summary

### Total Task Groups: 4

### Dependency Chain:
```
Task Group 1 (Unrolled Update) ─┬─→ Task Group 2 (Unrolled Save)
                                │
                                └─→ Task Group 3 (Codegen Update) ─→ Task Group 4 (Codegen Save)
```

### Tests to Create:
- All 4 factories should be validated by running tests/all_in_one.py and comparing outputs with the existing chaining implementation
- No separate test files needed as this is exploratory/debug code

### Tests to Run:
- Manual execution of tests/all_in_one.py
- Compare numerical outputs between chaining and new factory implementations

### Estimated Complexity:
- Task Group 1: Medium (establish pattern for all 18 metrics)
- Task Group 2: Low (follows same pattern as Task Group 1)
- Task Group 3: Medium (code generation with string manipulation)
- Task Group 4: Low (follows same pattern as Task Group 3)

### Key Implementation Notes:
1. All factories use `summary_metrics.preprocess_request()` for metric discovery
2. Combined metrics (extrema, mean_std, etc.) are handled by preprocess_request substitution
3. Parameterized metrics (peaks[N]) extract param from `summary_metrics.params()`
4. The INLINE_UPDATE_FUNCTIONS and INLINE_SAVE_FUNCTIONS dicts already exist and should be used
5. All new code goes in tests/all_in_one.py after the existing save_summary_factory (around line 4415)
6. Use `compile_kwargs` from cuda_simsafe for CUDA JIT consistency
