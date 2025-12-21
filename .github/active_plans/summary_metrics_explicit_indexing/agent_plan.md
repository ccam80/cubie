# Agent Plan: Summary Metrics Explicit Indexing Refactor

## Overview

This plan refactors the summary metrics system to eliminate buffer slicing operations that cause register spilling. The refactoring occurs at three levels:

1. **Chain level**: Modify `chain_metrics` to pass offset parameters instead of sliced buffers
2. **Factory level**: Modify `update_summary_factory` to move the per-variable loop inside the chain
3. **Metric level**: Update all metric update functions to use explicit indexing with offsets

## Component Descriptions

### 1. Updated `chain_metrics` Function

**Current Behavior**: 
- Takes a sequence of metric functions and creates a recursive chain
- Each wrapper slices the buffer: `buffer[current_offset : current_offset + current_size]`
- Passes the sliced view to the metric update function

**New Behavior**:
- Takes the same sequence of metric functions
- Each wrapper passes the full buffer plus the current metric's offset
- Metric receives `(value, buffer, base_offset + metric_offset, current_step, param)`
- No slicing occurs; offset is an integer parameter

**Signature Change for Wrapper**:
```python
# Current wrapper signature:
def wrapper(value, buffer, current_step):
    inner_chain(value, buffer, current_step)
    current_fn(value, buffer[offset:offset+size], current_step, param)

# New wrapper signature:
def wrapper(value, buffer, buffer_offset, current_step):
    inner_chain(value, buffer, buffer_offset, current_step)
    current_fn(value, buffer, buffer_offset + metric_offset, current_step, param)
```

**Key Points**:
- `buffer_offset` parameter is the base offset for the current variable
- `metric_offset` is captured in closure for each metric in the chain
- The chain builds up cumulative offsets through the recursive structure

### 2. Updated `do_nothing` Function

**Current Signature**: `do_nothing(values, buffer, current_step)`

**New Signature**: `do_nothing(value, buffer, buffer_offset, current_step)`

**Purpose**: Base case for empty chains; must match new wrapper signature

### 3. Updated `update_summary_factory` Function

**Current Behavior**:
- Loops over summarized state indices
- For each index, slices the buffer and calls chain_fn
- Same for observable indices

**New Behavior**:
- Calculates `total_buffer_size` per variable (already done)
- Loops over indices but passes base offset instead of sliced buffer
- For variable `idx`, base offset is `idx * total_buffer_size`

**Key Change**:
```python
# Current:
for idx in range(num_summarised_states):
    start = idx * total_buffer_size
    end = start + total_buffer_size
    chain_fn(
        current_state[summarised_state_indices[idx]],
        state_summary_buffer[start:end],  # SLICING
        current_step,
    )

# New:
for idx in range(num_summarised_states):
    base_offset = idx * total_buffer_size
    chain_fn(
        current_state[summarised_state_indices[idx]],
        state_summary_buffer,  # FULL BUFFER
        base_offset,           # OFFSET PARAMETER
        current_step,
    )
```

### 4. Updated Metric Update Function Signature

**Current Signature** (in each metric's `build()` method):
```python
def update(value, buffer, current_index, customisable_variable):
    buffer[0] += value  # Direct indexing into pre-sliced buffer
```

**New Signature**:
```python
def update(value, buffer, offset, current_index, customisable_variable):
    buffer[offset + 0] += value  # Explicit offset + index
```

**Impact**: All metrics must update their `update` function to:
1. Accept `offset` as third parameter
2. Replace `buffer[k]` with `buffer[offset + k]` for all accesses

### 5. Predicated Commit Pattern for Conditional Metrics

Metrics that conditionally update buffers (max, min, extrema, peaks) must use `selp`:

**Current Pattern** (e.g., max.py):
```python
if value > buffer[0]:
    buffer[0] = value
```

**New Pattern**:
```python
update_flag = value > buffer[offset + 0]
buffer[offset + 0] = selp(update_flag, value, buffer[offset + 0])
```

**Rationale**: Conditional branching causes warp divergence; predicated execution maintains warp efficiency.

## Expected Behavior Changes

### Chain Function Execution

1. Factory calls `chain_fn(value, buffer, base_offset, step)` for each variable
2. Chain wrapper executes `inner_chain(value, buffer, base_offset, step)` first
3. Chain wrapper calls metric: `metric_fn(value, buffer, base_offset + metric_offset, step, param)`
4. Metric function accesses `buffer[offset + 0]`, `buffer[offset + 1]`, etc.

### Memory Access Pattern

Before: Multiple array slice views created, each requiring memory operations
After: Single buffer reference, all accesses via integer offset arithmetic

## Architectural Changes Required

### 1. Signature Propagation

The `buffer_offset` parameter must propagate through:
- `do_nothing` base function
- All chain wrappers
- All metric update functions

### 2. Offset Calculation in Chain

Each level of the chain captures its metric's offset:
- First metric: offset 0
- Second metric: offset = first_metric.buffer_size
- Third metric: offset = first_metric.buffer_size + second_metric.buffer_size

This is already computed by `buffer_offsets` in the registry; pass to `chain_metrics`.

### 3. Base Offset Calculation in Factory

For variable `idx`:
- State buffer base: `idx * total_buffer_size`
- Observable buffer base: `idx * total_buffer_size`

## Integration Points

### With metrics.py Registry

No changes to the registry interface. The registry continues to provide:
- `buffer_offsets()`: Returns tuple of metric offsets within a variable's buffer
- `buffer_sizes()`: Returns tuple of metric buffer sizes
- `update_functions()`: Returns tuple of update function callables

### With output_functions.py

The `update_summary_metrics_func` signature remains unchanged:
```python
def update_summary_metrics_func(
    current_state,
    current_observables,
    state_summary_buffer,
    observable_summary_buffer,
    current_step,
)
```

Callers of this function are unaffected.

### With Integration Loop (ode_loop.py)

No changes required. The loop calls `update_summary_metrics_func` with the same parameters.

## Data Structures

### Buffer Layout (Unchanged)

For a configuration with metrics [mean, max, std] and 3 summarized variables:

```
Buffer Layout:
|-- Variable 0 --|-- Variable 1 --|-- Variable 2 --|
|mean|max|std    |mean|max|std    |mean|max|std    |
|0   |1  |2,3,4  |5   |6  |7,8,9  |10  |11 |12,13,14|

total_buffer_size = 5 (1 + 1 + 3)
num_variables = 3
```

### Offset Calculation

For variable `idx` and metric with `metric_offset`:
- Buffer position = `idx * total_buffer_size + metric_offset`

## Dependencies and Imports

### update_summaries.py

Current imports remain, plus add:
```python
from numba import int32  # Already imported
```

### Individual Metrics

Metrics using `selp` need:
```python
from cubie.cuda_simsafe import selp
```

Some metrics already import this (e.g., dxdt_max.py).

## Edge Cases

### Empty Metric List

When `metric_functions` is empty, `chain_metrics` returns `do_nothing`.
The updated `do_nothing` must handle the new signature.

### Single Metric

Chain with one metric works identically to multiple metrics.

### Zero Summarized Variables

When `num_summarised_states == 0` or `num_summarised_observables == 0`, the loops don't execute. No special handling needed.

### Metrics with Variable Buffer Sizes (e.g., peaks)

Parameterized metrics receive their buffer size from the registry. The offset calculation uses the actual size returned by `buffer_sizes()`.

## Metrics Requiring Updates

### Simple Accumulator Metrics
- `mean.py`: `buffer[offset + 0] += value`
- `rms.py`: Reset on index 0, accumulate sum of squares

### Conditional Metrics (Need selp)
- `max.py`: `buffer[offset + 0] = selp(value > buffer[offset + 0], value, buffer[offset + 0])`
- `min.py`: `buffer[offset + 0] = selp(value < buffer[offset + 0], value, buffer[offset + 0])`
- `max_magnitude.py`: Similar pattern with absolute value

### Multi-Slot Metrics
- `std.py`: Three slots at `offset + 0`, `offset + 1`, `offset + 2`
- `extrema.py`: Two slots at `offset + 0` (max), `offset + 1` (min)

### Composite Metrics
- `mean_std.py`: Three slots for shift, sum, sum_sq
- `mean_std_rms.py`: Same three slots
- `std_rms.py`: Same three slots

### Derivative Metrics
- `dxdt_max.py`: Already uses selp; update indexing
- `dxdt_min.py`: Two slots, needs selp
- `dxdt_extrema.py`: Three slots (prev_value, max, min)
- `d2xdt2_*`: Similar patterns to dxdt

### Peak Finding Metrics
- `peaks.py`: Variable size buffer, complex logic with selp
- `negative_peaks.py`: Same pattern as peaks

## Notes for Implementer

1. **Order of Implementation**: Start with `update_summaries.py` changes, then update `do_nothing`, then update one simple metric (mean) to validate the pattern, then update remaining metrics.

2. **Testing Strategy**: Existing tests in `test_summary_metrics.py` should continue to pass. The tests validate buffer sizes, offsets, and function retrieval - these remain unchanged.

3. **Compile-Time Constants**: All offsets calculated by `buffer_offsets()` should be captured in closures, not passed as runtime parameters to metrics.

4. **The `selp` Import**: Add import from `cubie.cuda_simsafe` to metrics that don't already have it.

5. **Save Functions**: Save functions also need updating with the same offset pattern. Their signature also changes to include offset parameter.

## Summary of Signature Changes

| Component | Current Signature | New Signature |
|-----------|-------------------|---------------|
| do_nothing | `(value, buffer, current_step)` | `(value, buffer, buffer_offset, current_step)` |
| chain wrapper | `(value, buffer, current_step)` | `(value, buffer, buffer_offset, current_step)` |
| metric update | `(value, buffer, current_index, param)` | `(value, buffer, offset, current_index, param)` |
| metric save | `(buffer, output, summarise_every, param)` | `(buffer, offset, output, output_offset, summarise_every, param)` |

Note: Save functions also benefit from explicit indexing but are called less frequently than update functions. They should follow the same pattern for consistency.
