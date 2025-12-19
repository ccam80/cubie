# Agent Plan: Complete Summary Metrics Integration in all_in_one.py

## Overview

This plan details the implementation of complete summary metrics integration in the all_in_one.py debug script. The implementation requires verbatim copying of device functions and factory code from the package source to maintain exact behavioral parity for NVIDIA profiler analysis.

## Component 1: List-Based Output Configuration System

### Purpose
Replace hardcoded boolean flags with list-based configuration matching `output_config.py`.

### Expected Behavior
- Configuration starts with an `output_types` list containing strings like `['state', 'mean', 'max']`
- Boolean toggles are derived programmatically from the list
- Summary types are extracted by checking against `summary_metrics.implemented_metrics`
- Summarise booleans are derived from summary_types and save flags

### Integration Points
- Configuration section near lines 150-160 in all_in_one.py
- Replaces existing hardcoded boolean variables

### Data Structures
```python
# Input structure
output_types = ['state', 'mean', 'max', 'rms']

# Derived outputs
save_state_bool = "state" in output_types
save_obs_bool = "observables" in output_types  
save_time_bool = "time" in output_types
save_counters_bool = "iteration_counters" in output_types

summary_types = []  # Built by iterating output_types
# For each item, check if it starts with any metric name from implemented_metrics
# Add to summary_types if match found

summary_types = tuple(summary_types)  # Convert to tuple

summarise_state_bool = len(summary_types) > 0 and save_state_bool
summarise_obs_bool = len(summary_types) > 0 and save_obs_bool
```

### Dependencies
Requires knowledge of implemented metric names. In package this comes from `summary_metrics.implemented_metrics`. For all_in_one.py, this should be a hardcoded list matching the available metrics.

### Edge Cases
- Empty output_types list: All booleans should be False
- Unknown metric names: Should warn but not raise (matching package behavior)
- Mix of summary and non-summary outputs: Both types coexist correctly

## Component 2: Summary Metric Device Functions

### Purpose
Provide inline CUDA device functions for all 18+ summary metrics, exactly matching package source.

### Expected Behavior
Each metric provides two device functions:
1. **update function**: Accumulates data during integration steps
2. **save function**: Computes final metric, writes output, resets buffer

### Metrics to Implement

#### Basic Statistics
1. **mean** (`mean.py`): Accumulates sum, divides by count on save
   - Buffer: 1 slot (running sum)
   - Output: 1 value (mean)
   
2. **max** (`max.py`): Tracks maximum value
   - Buffer: 1 slot (current max, initialized to -1.0e30)
   - Output: 1 value (maximum)
   
3. **min** (`min.py`): Tracks minimum value
   - Buffer: 1 slot (current min, initialized to 1.0e30)
   - Output: 1 value (minimum)
   
4. **rms** (`rms.py`): Root mean square
   - Buffer: 1 slot (sum of squares)
   - Output: 1 value (sqrt(sum_sq / count))
   
5. **std** (`std.py`): Standard deviation
   - Buffer: 2 slots (sum, sum_of_squares)
   - Output: 1 value (std)

#### Composite Statistics
6. **mean_std** (`mean_std.py`): Both mean and std together
   - Buffer: 2 slots (sum, sum_of_squares)
   - Output: 2 values (mean, std)
   
7. **mean_std_rms** (`mean_std_rms.py`): Mean, std, and rms together
   - Buffer: 2 slots (sum, sum_of_squares)
   - Output: 3 values (mean, std, rms)
   
8. **std_rms** (`std_rms.py`): Std and rms together
   - Buffer: 2 slots (sum, sum_of_squares)
   - Output: 2 values (std, rms)

#### Extrema Tracking
9. **extrema** (`extrema.py`): Both max and min
   - Buffer: 2 slots (max, min)
   - Output: 2 values (max, min)
   
10. **peaks** (`peaks.py`): Local maxima detection
    - Buffer: 3 slots (prev_value, current_peak, peak_candidate)
    - Output: 1 value (highest peak)
    
11. **negative_peaks** (`negative_peaks.py`): Local minima detection
    - Buffer: 3 slots (prev_value, current_valley, valley_candidate)
    - Output: 1 value (deepest valley)
    
12. **max_magnitude** (`max_magnitude.py`): Maximum absolute value
    - Buffer: 1 slot (current max magnitude)
    - Output: 1 value (max |value|)

#### First Derivative Tracking
13. **dxdt_max** (`dxdt_max.py`): Maximum first derivative
    - Buffer: 2 slots (prev_value, max_unscaled_derivative)
    - Output: 1 value (max dx/dt scaled by dt_save)
    
14. **dxdt_min** (`dxdt_min.py`): Minimum first derivative
    - Buffer: 2 slots (prev_value, min_unscaled_derivative)
    - Output: 1 value (min dx/dt scaled by dt_save)
    
15. **dxdt_extrema** (`dxdt_extrema.py`): Both max and min derivatives
    - Buffer: 3 slots (prev_value, max_unscaled, min_unscaled)
    - Output: 2 values (max dx/dt, min dx/dt)

#### Second Derivative Tracking
16. **d2xdt2_max** (`d2xdt2_max.py`): Maximum second derivative
    - Buffer: 3 slots (prev_prev_value, prev_value, max_unscaled)
    - Output: 1 value (max d²x/dt² scaled by dt_save²)
    
17. **d2xdt2_min** (`d2xdt2_min.py`): Minimum second derivative
    - Buffer: 3 slots (prev_prev_value, prev_value, min_unscaled)
    - Output: 1 value (min d²x/dt² scaled by dt_save²)
    
18. **d2xdt2_extrema** (`d2xdt2_extrema.py`): Both max and min second derivatives
    - Buffer: 4 slots (prev_prev_value, prev_value, max_unscaled, min_unscaled)
    - Output: 2 values (max d²x/dt², min d²x/dt²)

### Function Signatures
All update functions:
```python
def update_<metric>(value, buffer, current_index, customisable_variable):
    # Exact implementation from source file
```

All save functions:
```python
def save_<metric>(buffer, output_array, summarise_every, customisable_variable):
    # Exact implementation from source file
```

### Integration Points
- Add after line ~3380 in all_in_one.py
- Replace existing minimal stubs
- Each function decorated with `@cuda.jit(device=True, inline=True, **compile_kwargs)`

### Dependencies
- `from cubie.cuda_simsafe import selp` (for predicated commits in derivative metrics)
- `from math import sqrt` (for rms and std calculations)
- Access to `precision` variable for type conversions
- Access to `dt_save` for derivative scaling

### Special Considerations

#### Derivative Metrics (dxdt_*, d2xdt2_*)
These metrics require `dt_save` from compile settings. In all_in_one.py, this should come from the configuration section where `dt_save` is already defined.

#### Predicated Commits
Several metrics use predicated commit patterns to avoid warp divergence:
```python
update_flag = condition
buffer[0] = selp(update_flag, new_value, buffer[0])
```
This is preferred over `if/else` branches in CUDA device code.

#### Buffer Initialization
Some metrics require specific initial buffer values (e.g., max starts at -1.0e30, min at 1.0e30). These are set in save functions after writing output.

## Component 3: Update Chaining Factory Functions

### Purpose
Build recursive chains of update functions that execute all enabled metrics sequentially.

### Expected Behavior

#### do_nothing (update version)
```python
@cuda.jit(device=True, inline=True, **compile_kwargs)
def do_nothing(values, buffer, current_step):
    pass
```
- Serves as base case for recursion
- Called when no metrics are configured
- No-op function that does nothing

#### chain_metrics (update version)
```python
def chain_metrics(
    metric_functions: Sequence[Callable],
    buffer_offsets: Sequence[int],
    buffer_sizes: Sequence[int],
    function_params: Sequence[object],
    inner_chain: Callable = do_nothing,
) -> Callable:
```
- Recursively builds a chain of metric update functions
- Each level wraps the inner chain and adds one metric
- Returns a single callable that executes all metrics in sequence

**Recursion Pattern:**
1. Base case: Empty metric_functions list returns do_nothing
2. Recursive case:
   - Extract first metric and its parameters
   - Create wrapper that calls inner_chain then current metric
   - Recurse with remaining metrics, passing wrapper as new inner_chain
   - Return final wrapper

#### update_summary_factory
```python
def update_summary_factory(
    summaries_buffer_height_per_var: int,
    summarised_state_indices: Union[Sequence[int], ArrayLike],
    summarised_observable_indices: Union[Sequence[int], ArrayLike],
    summaries_list: Sequence[str],
) -> Callable:
```
- Factory that creates a CUDA device function for updating all metrics
- Uses chain_metrics to build the metric chain
- Generates function that iterates variables and applies chain to each

**Generated Function Signature:**
```python
def update_summary_metrics_func(
    current_state,
    current_observables,
    state_summary_buffer,
    observable_summary_buffer,
    current_step,
):
```

### Integration Points
- Add in summary metrics section before device function definitions
- Replace existing `chain_update_metrics` stub
- Called during loop integration

### Data Flow
```
summaries_list ['mean', 'max'] 
    ↓
Get update functions, buffer info, params from registry
    ↓
chain_metrics builds recursive chain
    ↓
update_summary_factory wraps chain for all variables
    ↓
Generated device function integrates into loop
```

### Dependencies
- Requires metric registry access (in package: `summary_metrics.update_functions()`, etc.)
- For all_in_one.py: Will need manual lookup tables mapping metric names to functions
- Sequences for buffer_offsets, buffer_sizes, params

## Component 4: Save Chaining Factory Functions

### Purpose
Build recursive chains of save functions that compute and export all enabled metrics.

### Expected Behavior

#### do_nothing (save version)
```python
@cuda.jit(device=True, inline=True, **compile_kwargs)
def do_nothing(buffer, output, summarise_every):
    pass
```
- Base case for save chain recursion
- Different signature than update do_nothing

#### chain_metrics (save version)
```python
def chain_metrics(
    metric_functions: Sequence[Callable],
    buffer_offsets: Sequence[int],
    buffer_sizes: Sequence[int],
    output_offsets: Sequence[int],
    output_sizes: Sequence[int],
    function_params: Sequence[object],
    inner_chain: Callable = do_nothing,
) -> Callable:
```
- Similar recursion pattern to update version
- Additional output_offsets and output_sizes for save operations
- Each metric writes to specific output window

#### save_summary_factory
```python
def save_summary_factory(
    summaries_buffer_height_per_var: int,
    summarised_state_indices: Union[Sequence[int], ArrayLike],
    summarised_observable_indices: Union[Sequence[int], ArrayLike],
    summaries_list: Sequence[str],
) -> Callable:
```
- Factory that creates CUDA device function for saving all metrics
- Uses chain_metrics to build the save chain
- Generates function that iterates variables and applies chain to each

**Generated Function Signature:**
```python
def save_summary_metrics_func(
    buffer_state_summaries,
    buffer_observable_summaries,
    output_state_summaries_window,
    output_observable_summaries_window,
    summarise_every,
):
```

### Integration Points
- Add in summary metrics section after update factories
- Replace existing `chain_save_metrics` stub
- Called during loop integration at save intervals

### Data Flow
```
summaries_list ['mean', 'max']
    ↓
Get save functions, buffer/output info, params from registry
    ↓
chain_metrics builds recursive chain
    ↓
save_summary_factory wraps chain for all variables
    ↓
Generated device function integrates into loop
```

### Dependencies
- Requires metric registry access (in package: `summary_metrics.save_functions()`, etc.)
- For all_in_one.py: Manual lookup tables for metric names
- Sequences for buffer_offsets, buffer_sizes, output_offsets, output_sizes, params

## Component 5: Metric Registry Simulation

### Purpose
Since all_in_one.py can't import from the package, we need inline structures that simulate the metric registry.

### Expected Behavior
Provide lookup functions that return metric information based on metric names.

### Required Lookup Functions

#### implemented_metrics
```python
implemented_metrics = [
    "mean", "max", "min", "rms", "std",
    "mean_std", "mean_std_rms", "std_rms",
    "extrema", "peaks", "negative_peaks", "max_magnitude",
    "dxdt_max", "dxdt_min", "dxdt_extrema",
    "d2xdt2_max", "d2xdt2_min", "d2xdt2_extrema"
]
```

#### buffer_sizes
```python
def buffer_sizes(summaries_list):
    # Return list of buffer sizes for each metric
    # Example: ['mean', 'max'] -> [1, 1]
```

#### output_sizes
```python
def output_sizes(summaries_list):
    # Return list of output sizes for each metric
    # Example: ['mean', 'extrema'] -> [1, 2]
```

#### buffer_offsets
```python
def buffer_offsets(summaries_list):
    # Return cumulative buffer offsets
    # Example: ['mean', 'max', 'rms'] -> [0, 1, 2]
```

#### output_offsets
```python
def output_offsets(summaries_list):
    # Return cumulative output offsets
    # Example: ['mean', 'extrema'] -> [0, 1]
```

#### update_functions
```python
def update_functions(summaries_list):
    # Return list of update function references
```

#### save_functions
```python
def save_functions(summaries_list):
    # Return list of save function references
```

#### params
```python
def params(summaries_list):
    # Return list of customisable_variable values (usually 0)
```

### Integration Points
- Add in configuration/setup section
- Used by factory functions to build chains

### Data Structures
Likely implemented as dictionaries mapping metric names to properties:
```python
METRIC_BUFFER_SIZES = {
    "mean": 1,
    "max": 1,
    # ... etc
}

METRIC_OUTPUT_SIZES = {
    "mean": 1,
    "extrema": 2,
    # ... etc
}
```

## Component 6: Buffer Size Calculations

### Purpose
Calculate total buffer and output sizes based on enabled metrics and tracked variables.

### Expected Behavior
```python
# For buffer sizing
summaries_buffer_height_per_var = sum(buffer_sizes(summary_types))

# For output sizing
summaries_output_height_per_var = sum(output_sizes(summary_types))

# Total buffer sizes
total_state_summary_buffer = num_summarised_states * summaries_buffer_height_per_var
total_obs_summary_buffer = num_summarised_observables * summaries_buffer_height_per_var
```

### Integration Points
- Configuration section where buffer sizes are calculated
- Used for array allocation in loop

### Edge Cases
- Empty summary_types: Buffer sizes should be 0
- Single metric: Buffer size = that metric's buffer_size
- Multiple metrics: Sum of all buffer sizes

## Component 7: Integration with Existing Loop Code

### Expected Behavior
The existing loop integration code should work without modification. The factories produce device functions with signatures matching what the loop expects.

### Verification Points
1. `update_summaries_inline` calls the factory-generated update function
2. `save_summaries_inline` calls the factory-generated save function
3. Buffer layouts match expected strides
4. Function signatures are compatible

### No Changes Required
The current integration pattern around lines 3500-3550 is correct and should be preserved. Only the implementations of the called functions change.

## Implementation Sequence

1. **Add metric registry simulation** - Lookup tables and helper functions
2. **Add all metric device functions** - 18+ update/save function pairs
3. **Add update chaining factories** - do_nothing, chain_metrics, update_summary_factory
4. **Add save chaining factories** - do_nothing, chain_metrics, save_summary_factory  
5. **Update configuration section** - List-based derivation system
6. **Update buffer calculations** - Dynamic sizing based on enabled metrics
7. **Test integration** - Verify loop code works with new factories

## Architectural Interactions

### With Configuration System
- Configuration provides `summary_types` list
- Registry simulation uses list to look up metrics
- Factory functions build chains based on list

### With Integration Loop
- Loop calls factory-generated update function at each step
- Loop calls factory-generated save function at save intervals
- Loop provides buffers sized according to enabled metrics

### With Buffer Management
- Buffer sizes calculated from sum of metric buffer_sizes
- Output sizes calculated from sum of metric output_sizes
- Strides computed for multi-variable arrays

## Key Constraints

1. **Verbatim Requirement**: All device functions and factories must be exact word-for-word copies
2. **Completeness Requirement**: All 18+ metrics must be implemented, no skipping
3. **No Imports**: Cannot import from package, must inline everything
4. **Behavioral Parity**: Must match production behavior exactly for profiling validity
5. **No Breaking Changes**: Existing loop integration must continue working

## Validation Approach

1. **Completeness Check**: Verify all 18+ metrics have update and save functions
2. **Verbatim Check**: Diff each function against package source
3. **Configuration Check**: Verify list-based derivation produces correct booleans
4. **Integration Check**: Verify loop code works without modification
5. **Sizing Check**: Verify buffer calculations produce correct sizes
