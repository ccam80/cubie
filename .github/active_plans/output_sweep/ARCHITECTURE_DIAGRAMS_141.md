# Architecture Diagrams for Issue #141

## Summary Metric Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Integration Loop (CUDA Kernel)                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────────┐                                                    │
│  │   Integrator │                                                    │
│  │  (algorithm) │                                                    │
│  └──────┬───────┘                                                    │
│         │                                                            │
│         ├─── state[n] ──────────────────────┐                        │
│         │                                    │                        │
│         └─── observables[m] ────────────┐   │                        │
│                                         │   │                        │
│                                         ▼   ▼                        │
│                              ┌────────────────────┐                  │
│                              │  update_summaries  │                  │
│                              │   (chain metrics)  │                  │
│                              └────────┬───────────┘                  │
│                                       │                              │
│                           ┌───────────┼───────────┐                  │
│                           ▼           ▼           ▼                  │
│                      ┌──────┐    ┌──────┐    ┌──────┐              │
│                      │ mean │    │ max  │    │ rms  │   ...         │
│                      │update│    │update│    │update│              │
│                      └───┬──┘    └───┬──┘    └───┬──┘              │
│                          │           │           │                   │
│                          ▼           ▼           ▼                   │
│                   ┌────────────────────────────────┐                 │
│                   │   Summary Buffers (device)     │                 │
│                   │  [metric1_buf | metric2_buf |  │                 │
│                   │   metric3_buf | ... ]          │                 │
│                   └────────────┬───────────────────┘                 │
│                                │                                      │
│                    (every summarise_every steps)                     │
│                                │                                      │
│                                ▼                                      │
│                   ┌─────────────────────┐                            │
│                   │   save_summaries    │                            │
│                   │   (chain metrics)   │                            │
│                   └──────────┬──────────┘                            │
│                              │                                        │
│                   ┌──────────┼──────────┐                            │
│                   ▼          ▼          ▼                            │
│              ┌──────┐   ┌──────┐   ┌──────┐                        │
│              │ mean │   │ max  │   │ rms  │   ...                   │
│              │ save │   │ save │   │ save │                         │
│              └───┬──┘   └───┬──┘   └───┬──┘                        │
│                  │          │          │                             │
│                  └──────────┼──────────┘                             │
│                             ▼                                         │
│              ┌──────────────────────────┐                            │
│              │  Output Arrays (device)  │                            │
│              │ [summary results per var]│                            │
│              └──────────────┬───────────┘                            │
│                             │                                         │
└─────────────────────────────┼─────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Host (Python)   │
                    │  Retrieve results│
                    └──────────────────┘
```

## Buffer Layout Examples

### Simple Metrics (1-2 slots)

```
min:           [current_min]
max_magnitude: [current_max_abs]
std:           [sum | sum_of_squares]
```

### Peak Detection (3+n to 6+2n slots)

```
peaks:         [prev | prev_prev | counter | time₁ | time₂ | ... | timeₙ]

negative_peak: [prev | prev_prev | counter | time₁ | time₂ | ... | timeₙ]

extrema:       [prev | prev_prev | max_counter | min_counter | 
                max_time₁ | ... | max_timeₙ | min_time₁ | ... | min_timeₙ]
```

### Derivative Metrics (5+n to 8+n slots)

```
dxdt_extrema:  [value_prev | value_prev_prev | 
                dxdt_prev | dxdt_prev_prev | 
                counter | time₁ | ... | timeₙ]

d2xdt2_extrema: [value_prev | value_prev_prev | value_prev_prev_prev |
                 dxdt_prev | dxdt_prev_prev |
                 d2xdt2_prev | d2xdt2_prev_prev |
                 counter | time₁ | ... | timeₙ]
```

## Combined Statistics Optimization

### Without Optimization (3 separate buffers)
```
mean: [sum]                  = 1 slot
rms:  [sum_of_squares]       = 1 slot  
std:  [sum | sum_of_squares] = 2 slots
                        Total = 4 slots
```

### With Optimization (shared buffer)
```
combined_stats: [sum | sum_of_squares] = 2 slots
                                  Total = 2 slots
                                  
At save time:
  mean = sum / n
  rms  = sqrt(sum_of_squares / n)
  std  = sqrt((sum_of_squares / n) - (mean)²)
```

**Savings: 50% buffer reduction when all 3 requested**

## Registry and Compilation Flow

```
Python Import Time:
┌──────────────────────────────────────┐
│ summary_metrics = SummaryMetrics()   │
│                                      │
│ @register_metric(summary_metrics)   │
│ class Mean(SummaryMetric):          │
│     def build(self): ...            │
│                                      │
│ @register_metric(summary_metrics)   │
│ class Max(SummaryMetric):           │
│     def build(self): ...            │
│                                      │
│ ... (all metrics auto-register)      │
└──────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────┐
│  Registry populated:                 │
│  _names = ["mean", "max", "rms", ...]│
│  _metric_objects = {name: instance}  │
│  _buffer_sizes = {name: size}        │
│  _output_sizes = {name: size}        │
└──────────────────────────────────────┘

Runtime (User requests metrics):
┌──────────────────────────────────────┐
│ solver.configure(                    │
│   summaries=["mean", "max", "std"]   │
│ )                                    │
└──────────┬───────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│ summary_metrics.preprocess_request() │
│   - Parse parameters                 │
│   - Validate metric names            │
│   - Calculate buffer offsets         │
└──────────┬───────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│ Build device functions:              │
│   - metric.build() for each          │
│   - Chain into update_summary_func   │
│   - Chain into save_summary_func     │
└──────────┬───────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│ Allocate device arrays:              │
│   - summary_buffer[total_buf_size]   │
│   - output_array[total_out_size]     │
└──────────┬───────────────────────────┘
           │
           ▼
      ┌────────┐
      │  Run!  │
      └────────┘
```

## Architecture Changes for Non-Summary Metrics

### Issue #76: Save Exit State

```
┌──────────────────────────────────────┐
│  Integration Complete                │
└──────────┬───────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│  if continuation_mode:               │
│    save_exit_state_func(             │
│      current_state,                  │
│      d_inits                         │
│    )                                 │
└──────────┬───────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│  d_inits[i] = current_state[i]       │
│  for all i in state_indices          │
└──────────┬───────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│  BatchInputArrays.fetch_inits()      │
│  Returns: initial_values array       │
└──────────┬───────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│  Next integration run uses           │
│  fetched values as initial state     │
└──────────────────────────────────────┘
```

### Issue #125: Iteration Counts

```
Algorithm Level:
┌──────────────────────────────────────┐
│  newton_iteration():                 │
│    solve...                          │
│    if counting_enabled:              │
│      iteration_counter[0] += 1       │
│    return status                     │
└──────────┬───────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│  krylov_solve():                     │
│    solve...                          │
│    if counting_enabled:              │
│      iteration_counter[1] += n_iters │
│    return status                     │
└──────────┬───────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│  step_controller():                  │
│    if step_rejected:                 │
│      iteration_counter[2] += 1       │
└──────────┬───────────────────────────┘
           │
           ▼ (every save window)
┌──────────────────────────────────────┐
│  save_iteration_counts():            │
│    output[0] = iteration_counter[0]  │
│    output[1] = iteration_counter[1]  │
│    output[2] = iteration_counter[2]  │
│    reset counters for next window    │
└──────────────────────────────────────┘
```

## Device Function Chaining (Numba Pattern)

```python
# Problem: Can't pass list of functions to CUDA device function
# Solution: Recursive chaining at compile time

def chain_metrics(fns, offsets, sizes, params, inner=do_nothing):
    if len(fns) == 0:
        return do_nothing
    
    current_fn = fns[0]
    current_offset = offsets[0]
    current_size = sizes[0]
    current_param = params[0]
    
    @cuda.jit(device=True, inline=True)
    def wrapper(value, buffer, step):
        inner(value, buffer, step)  # Execute accumulated chain
        current_fn(                 # Execute current metric
            value,
            buffer[current_offset:current_offset+current_size],
            step,
            current_param
        )
    
    return chain_metrics(fns[1:], offsets[1:], sizes[1:], params[1:], wrapper)

# Result: Single device function that calls all metrics in sequence
# Example:
#   chain = chain_metrics([mean.update, max.update, rms.update], ...)
#   # Compiles to:
#   @cuda.jit(device=True, inline=True)
#   def chain(value, buffer, step):
#       mean.update(value, buffer[0:1], step, 0)
#       max.update(value, buffer[1:2], step, 0)
#       rms.update(value, buffer[2:3], step, 0)
```

## Memory Layout in Device Arrays

```
For a system with:
  - 2 state variables to summarize
  - 1 observable to summarize
  - 3 metrics: mean (buf=1), max (buf=1), peaks[3] (buf=6)
  - Buffer size per variable = 1 + 1 + 6 = 8

State Summary Buffer:
┌─────────────────┬─────────────────┐
│   Variable 0    │   Variable 1    │
│ [0:8]           │ [8:16]          │
├───┬───┬─────────┼───┬───┬─────────┤
│ M │ M │ Peaks   │ M │ M │ Peaks   │
│ e │ a │ [6]     │ e │ a │ [6]     │
│ a │ x │         │ a │ x │         │
│ n │   │         │ n │   │         │
└───┴───┴─────────┴───┴───┴─────────┘
 [0] [1] [2:8]     [8] [9] [10:16]

Observable Summary Buffer:
┌─────────────────┐
│   Observable 0  │
│ [0:8]           │
├───┬───┬─────────┤
│ M │ M │ Peaks   │
│ e │ a │ [6]     │
│ a │ x │         │
│ n │   │         │
└───┴───┴─────────┘
 [0] [1] [2:8]

Output Arrays (after save):
State outputs: [mean₀, max₀, peak₀₁, peak₀₂, peak₀₃,
                mean₁, max₁, peak₁₁, peak₁₂, peak₁₃]
               
Observable outputs: [mean₀, max₀, peak₀₁, peak₀₂, peak₀₃]
```

## Implementation Phases by Dependency

```
Phase 1: Architecture Changes (Must be first)
├─ save_exit_state      [●●●●●●] High complexity
│  ├─ Device function   [●●●●○]
│  ├─ Kernel changes    [●●●●●]
│  └─ Tests             [●●●●○]
└─ iteration_counts     [●●●●●●●] High complexity
   ├─ Counter prop.     [●●●●●●]
   ├─ Compile flag      [●●●○○]
   └─ Tests             [●●●●●]

Phase 2: Simple Metrics (Validates new architecture)
├─ min.py               [●●●○○] Low complexity
├─ max_magnitude.py     [●●●○○] Low complexity
├─ std.py               [●●●●○] Medium complexity
└─ Tests                [●●●○○] Basic validation

Phase 3: Peak Detection (Builds on simple metrics)
├─ negative_peak.py     [●●●●○] Medium complexity
├─ extrema.py           [●●●●●] Medium complexity
└─ Tests                [●●●●○] Comprehensive

Phase 4: Derivatives (Most complex, builds on peaks)
├─ dxdt_extrema.py      [●●●●●●] High complexity
├─ d2xdt2_extrema.py    [●●●●●●●] High complexity
├─ dxdt.py              [●●●○○] Medium (optional)
└─ Tests                [●●●●●●] Numerical accuracy

[●] = Relative complexity indicator
```

## Testing Strategy Flow

```
┌─────────────────────────────────────┐
│  Unit Tests (per metric)            │
│  ├─ Known input → expected output   │
│  ├─ Edge cases (empty, single, etc) │
│  ├─ Parameter validation            │
│  └─ Buffer reset verification       │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  Integration Tests                  │
│  ├─ Multiple metrics simultaneously │
│  ├─ Correct buffer offsets          │
│  ├─ Float32 vs Float64              │
│  └─ Registry tests                  │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  System Tests                       │
│  ├─ Full ODE integration            │
│  ├─ Verify vs offline calculation   │
│  ├─ Performance benchmarks          │
│  └─ Continuation test (#76)         │
└─────────────────────────────────────┘
```

---

**Note:** These diagrams use ASCII art for maximum compatibility.
For production documentation, consider converting to:
- PlantUML (for auto-generated diagrams)
- Mermaid (for GitHub-rendered diagrams)
- draw.io (for manual diagrams)
