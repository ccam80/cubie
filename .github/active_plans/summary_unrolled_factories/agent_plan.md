# Summary Metrics Unrolled Factories - Agent Plan

## Overview

This document provides detailed technical specifications for implementing two alternative summary metrics factory approaches in `tests/all_in_one.py`. The implementations replace the recursive chaining pattern with compiler-friendly alternatives.

---

## Component 1: Fully Unrolled Update Factory

### Description
A factory function that generates a CUDA device function for updating summary metrics. Instead of recursively chaining wrapper functions, it produces a single device function with compile-time boolean guards for each enabled metric.

### Expected Behavior

1. **Factory Invocation**: Called with the same parameters as `update_summary_factory()`:
   - `summaries_buffer_height_per_var`: Total buffer size per variable
   - `summarised_state_indices`: Array of state indices to summarize
   - `summarised_observable_indices`: Array of observable indices to summarize
   - `summaries_list`: Tuple of metric names to enable

2. **Compile-Time Setup**:
   - Extract individual enable flags for each known metric (mean, max, rms, peaks, etc.)
   - Compute buffer offsets for each enabled metric
   - Compute buffer sizes for each enabled metric
   - Extract function parameters for each enabled metric
   - Capture individual metric update functions in closure

3. **Generated Device Function**:
   - Signature: `update_summary_metrics_func(current_state, current_observables, state_summary_buffer, observable_summary_buffer, current_step)`
   - Single loop over state indices when `summarise_states` is True
   - Single loop over observable indices when `summarise_observables` is True
   - Inside each loop: conditional calls to each metric's update function, guarded by compile-time boolean

### Buffer Indexing Pattern

The device function uses a marching base pointer pattern:
```
buffer_base = variable_index * total_buffer_size
For each metric (if enabled):
    metric_buffer = buffer[buffer_base + metric_offset : buffer_base + metric_offset + metric_size]
    metric_update(value, metric_buffer, current_step, metric_param)
```

### Integration Points
- Uses `summary_metrics.buffer_offsets()` for offset calculation
- Uses `summary_metrics.buffer_sizes()` for size calculation
- Uses `summary_metrics.params()` for parameter values
- Uses `summary_metrics.update_functions()` for device function references

---

## Component 2: Fully Unrolled Save Factory

### Description
A factory function that generates a CUDA device function for saving summary metrics. Follows the same unrolled pattern as the update factory.

### Expected Behavior

1. **Factory Invocation**: Called with the same parameters as `save_summary_factory()`:
   - `summaries_buffer_height_per_var`: Total buffer size per variable
   - `summarised_state_indices`: Array of state indices to summarize
   - `summarised_observable_indices`: Array of observable indices to summarize
   - `summaries_list`: Tuple of metric names to enable

2. **Compile-Time Setup**:
   - Extract individual enable flags for each known metric
   - Compute buffer offsets and sizes for each enabled metric
   - Compute output offsets and sizes for each enabled metric
   - Extract function parameters for each enabled metric
   - Capture individual metric save functions in closure

3. **Generated Device Function**:
   - Signature: `save_summary_metrics_func(buffer_state_summaries, buffer_observable_summaries, output_state_summaries_window, output_observable_summaries_window)`
   - Single loop over state indices when `summarise_states` is True
   - Single loop over observable indices when `summarise_observables` is True
   - Inside each loop: conditional calls to each metric's save function, guarded by compile-time boolean

### Integration Points
- Uses `summary_metrics.output_offsets()` for output offset calculation
- Uses `summary_metrics.output_sizes()` for output size calculation
- Uses `summary_metrics.save_functions()` for device function references

---

## Component 3: SymPy Codegen Update Factory

### Description
A factory function that generates Python source code as a string, following the pattern in `odesystems/symbolic/codegen/dxdt.py`. The generated code is executed via `exec()` to produce a device function.

### Expected Behavior

1. **Template Structure**:
   Similar to `DXDT_TEMPLATE`, define an `UPDATE_SUMMARY_TEMPLATE`:
   ```python
   UPDATE_SUMMARY_TEMPLATE = '''
   def {func_name}():
       """Auto-generated summary update factory."""
       {constant_assignments}
       
       @cuda.jit(device=True, inline=True, forceinline=True)
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

2. **Code Generation**:
   - Generate constant assignments for offsets, sizes, and parameters
   - Generate loop code for iterating over variables
   - Generate metric call code inline within the loop body
   - No function references in generated code - all calls are direct

3. **Compilation**:
   - Build namespace with required imports (cuda, int32, metric functions)
   - Execute generated source with `exec(source_code, namespace)`
   - Extract and call the factory function from namespace
   - Return the compiled device function

### Generated Code Structure
```python
# Example generated body:
if summarise_states:
    for idx in range(num_summarised_states):
        base = idx * total_buffer_size
        value = current_state[state_indices[idx]]
        # Mean update (inline)
        state_summary_buffer[base + 0] += value
        # Max update (inline)
        if value > state_summary_buffer[base + 1]:
            state_summary_buffer[base + 1] = value
        # ... more metrics
```

### Dependencies
- `numba.cuda` for JIT decoration
- Metric device functions captured in generation namespace
- `compile_kwargs` for consistent compilation settings

---

## Component 4: SymPy Codegen Save Factory

### Description
A factory function that generates Python source code for the save operation, following the same codegen pattern.

### Expected Behavior

1. **Template Structure**:
   Similar template for save operations:
   ```python
   SAVE_SUMMARY_TEMPLATE = '''
   def {func_name}():
       """Auto-generated summary save factory."""
       {constant_assignments}
       
       @cuda.jit(device=True, inline=True, forceinline=True)
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

2. **Code Generation**:
   - Generate constant assignments
   - Generate loop code for iterating over variables
   - Generate metric save calls inline within the loop body

3. **Compilation**:
   - Same exec pattern as update factory

---

## Architectural Considerations

### Metric Registration Discovery
The factories must handle any combination of registered metrics. The unrolled factory should explicitly check for each known metric. The codegen factory should iterate through the provided metrics list.

### Known Metrics to Handle
Based on the `outputhandling/summarymetrics/` directory:
- mean
- max
- min
- rms
- std
- peaks (parameterized)
- negative_peaks (parameterized)
- extrema (combined max+min)
- mean_std (combined)
- mean_std_rms (combined)
- std_rms (combined)
- max_magnitude
- dxdt_max
- dxdt_min
- dxdt_extrema (combined)
- d2xdt2_max
- d2xdt2_min
- d2xdt2_extrema (combined)

### Combined Metrics
Some metrics are combined versions (e.g., `extrema` = `max` + `min`). The factories should use the combined versions when detected by `summary_metrics.preprocess_request()`.

### Compile-Time Constants
All offsets, sizes, and boolean flags should be captured in the device function's closure so they become compile-time constants. This is essential for LLVM optimization.

### Buffer Layout
The buffer layout follows this structure:
```
For each variable (state or observable):
    [metric_1_buffer | metric_2_buffer | ... | metric_n_buffer]
    
Total buffer per variable = sum of all metric buffer sizes
```

### Output Layout
Similar to buffer layout but uses output sizes:
```
For each variable:
    [metric_1_output | metric_2_output | ... | metric_n_output]
```

---

## Data Structures

### Required Inputs (Both Factories)
- `summaries_buffer_height_per_var`: int - Total buffer size per variable
- `summarised_state_indices`: np.ndarray or tuple - Indices of states to summarize
- `summarised_observable_indices`: np.ndarray or tuple - Indices of observables to summarize
- `summaries_list`: tuple[str, ...] - Metric names to enable

### Computed Values (Unrolled Factory)
For each metric:
- `enable_{metric}`: bool - Whether this metric is enabled
- `{metric}_offset`: int32 - Buffer offset for this metric
- `{metric}_size`: int32 - Buffer size for this metric
- `{metric}_param`: int32 - Parameter value for this metric
- `{metric}_update_fn`: Callable - Device update function
- `{metric}_save_fn`: Callable - Device save function

### Computed Values (Codegen Factory)
Same logical values, but rendered as literal code in the generated source string.

---

## Expected Function Signatures

### Unrolled Update Factory
```python
def unrolled_update_summary_factory(
    summaries_buffer_height_per_var: int,
    summarised_state_indices: tuple,
    summarised_observable_indices: tuple,
    summaries_list: tuple,
) -> Callable:
    """Generate unrolled summary update device function."""
    ...
    return update_summary_metrics_func
```

### Unrolled Save Factory
```python
def unrolled_save_summary_factory(
    summaries_buffer_height_per_var: int,
    summarised_state_indices: tuple,
    summarised_observable_indices: tuple,
    summaries_list: tuple,
) -> Callable:
    """Generate unrolled summary save device function."""
    ...
    return save_summary_metrics_func
```

### Codegen Update Factory
```python
def codegen_update_summary_factory(
    summaries_buffer_height_per_var: int,
    summarised_state_indices: tuple,
    summarised_observable_indices: tuple,
    summaries_list: tuple,
) -> Callable:
    """Generate summary update device function via code generation."""
    ...
    return update_summary_metrics_func
```

### Codegen Save Factory
```python
def codegen_save_summary_factory(
    summaries_buffer_height_per_var: int,
    summarised_state_indices: tuple,
    summarised_observable_indices: tuple,
    summaries_list: tuple,
) -> Callable:
    """Generate summary save device function via code generation."""
    ...
    return save_summary_metrics_func
```

---

## Edge Cases

1. **Empty summaries_list**: Return a do-nothing function
2. **No states to summarize**: Skip state summarization loop
3. **No observables to summarize**: Skip observable summarization loop
4. **Parameterized metrics** (e.g., `peaks[3]`): Must handle parameter extraction
5. **Combined metrics** (e.g., `extrema`): Should work with preprocessed list
6. **Unknown metrics**: Should be filtered out by `preprocess_request()`

---

## Testing Strategy

Both new factory implementations should be validated by:
1. Instantiating alongside the current chaining implementation
2. Running integration with identical inputs
3. Comparing output arrays for numerical equality
4. Verifying all edge cases (empty lists, single metrics, all metrics)

---

## File Location

All implementations go in: `/home/runner/work/cubie/cubie/tests/all_in_one.py`

The new factories should be placed near the existing `update_summary_factory()` and `save_summary_factory()` functions (around lines 4193-4414).
