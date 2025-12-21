# Summary Metrics Explicit Indexing Refactor

## User Stories

### User Story 1: Eliminate Buffer Slicing for Register Efficiency
**As a** CuBIE developer optimizing GPU performance  
**I want** the summary metrics system to use explicit buffer indexing instead of array slicing  
**So that** the CUDA compiler can keep values in registers instead of spilling to memory

**Acceptance Criteria:**
- [ ] `chain_metrics` wrapper passes buffer base reference plus offset parameters instead of pre-sliced arrays
- [ ] Individual metric update functions receive the full buffer and an offset, using direct indexing like `buffer[offset + 0]`
- [ ] No array slice operations (`buffer[start:end]`) occur in the update path
- [ ] External API signature of `update_summary_factory` remains unchanged

### User Story 2: Batch Processing of Summarized Indices
**As a** CuBIE developer optimizing integration loops  
**I want** the summary metrics chain to process all summarized variable indices together  
**So that** per-variable loop overhead is eliminated and the compiler has better optimization opportunities

**Acceptance Criteria:**
- [ ] The `update_summary_metrics_func` passes all summarized indices to the chain in a single call
- [ ] Metric functions are aware of the total number of summarized variables and buffer layout at compile time
- [ ] Loop iteration over summarized indices occurs inside the chained metric functions, not outside
- [ ] The chain can efficiently compute summaries for multiple values using compile-time-known offsets

### User Story 3: Buffer-Structure-Aware Metric Functions
**As a** metric implementer  
**I want** metric update functions to have compile-time knowledge of buffer structure  
**So that** the compiler can optimize memory access patterns and eliminate runtime offset calculations

**Acceptance Criteria:**
- [ ] Update function signature accepts buffer base, offset, and size parameters
- [ ] Individual metrics use explicit indexing: `buffer[offset + 0]`, `buffer[offset + 1]`, etc.
- [ ] Buffer layout information (offsets, sizes, variable count) is captured in closure at compile time
- [ ] Existing metrics (mean, max, min, std, rms, extrema, etc.) are updated to use new signature

### User Story 4: Maintain Predicated Commit Pattern
**As a** CUDA performance engineer  
**I want** all metric update operations to use predicated commit rather than conditional branching  
**So that** warp efficiency is maximized by avoiding divergence

**Acceptance Criteria:**
- [ ] Metrics using conditionals (max, min, extrema) use `selp` for predicated assignment
- [ ] No `if/else` statements that conditionally write to buffers
- [ ] Pattern: `buffer[offset] = selp(condition, new_value, buffer[offset])`

---

## Executive Summary

This refactoring addresses a performance issue in the summary metrics system where buffer slicing operations cause the CUDA compiler to spill values from registers to memory. The current implementation slices buffer arrays at two levels:

1. **Chain level**: `buffer[current_offset : current_offset + current_size]`
2. **Factory level**: `state_summary_buffer[start:end]` in per-variable loops

The solution replaces slicing with explicit index passing, making functions "buffer-structure-aware" at compile time. This allows the compiler to keep intermediate values in registers and apply better optimizations.

---

## Architecture Overview

### Current Data Flow (Problematic)

```mermaid
flowchart TD
    subgraph Factory["update_summary_factory"]
        A[For each summarized variable idx] --> B[Calculate start/end offsets]
        B --> C[Slice buffer: buffer[start:end]]
        C --> D[Call chain_fn with sliced buffer]
    end
    
    subgraph Chain["chain_metrics wrapper"]
        D --> E[Call inner_chain with buffer]
        E --> F[Slice metric buffer: buffer[offset:offset+size]]
        F --> G[Call metric update with sliced buffer]
    end
    
    subgraph Metric["Individual Metric"]
        G --> H[Access buffer[0], buffer[1], etc.]
    end
    
    style C fill:#ff9999
    style F fill:#ff9999
```

### Proposed Data Flow (Optimized)

```mermaid
flowchart TD
    subgraph Factory["update_summary_factory"]
        A[Pass full buffers + compile-time layout info] --> B[Call chain_fn once]
    end
    
    subgraph Chain["chain_metrics wrapper"]
        B --> C[For each variable idx - loop inside chain]
        C --> D[Calculate base offset for variable]
        D --> E[Call inner_chain with buffer + base_offset]
        E --> F[Call metric update with buffer + metric_offset]
    end
    
    subgraph Metric["Individual Metric"]
        F --> G[Access buffer[offset + 0], buffer[offset + 1], etc.]
    end
    
    style C fill:#99ff99
    style G fill:#99ff99
```

### Component Interaction

```mermaid
sequenceDiagram
    participant Loop as Integration Loop
    participant Factory as update_summary_factory
    participant Chain as chain_fn
    participant Metric as Metric Update
    participant Buffer as Summary Buffer

    Loop->>Factory: update_summary_metrics_func(state, buffers, step)
    Factory->>Chain: chain_fn(values, buffer, step, indices, layout)
    
    loop For each variable (inside chain)
        Chain->>Chain: Calculate variable base offset
        loop For each metric (recursive chain)
            Chain->>Metric: update(value, buffer, base + metric_offset, step, param)
            Metric->>Buffer: buffer[offset + 0] += value
            Metric->>Buffer: buffer[offset + 1] = selp(cond, new, old)
        end
    end
```

---

## Key Technical Decisions

### 1. Explicit Index Parameters vs. Sliced Views

**Decision**: Pass full buffer reference + integer offset instead of sliced array views

**Rationale**: 
- Array slicing in Numba CUDA creates new array views that require memory operations
- Integer offsets can be computed at compile time and kept in registers
- Direct indexing `buffer[offset + k]` compiles to efficient pointer arithmetic

**Trade-offs**:
- Slightly more verbose metric implementations
- Offset calculations must be correct (no bounds checking from slices)

### 2. Loop Inversion (Outside → Inside)

**Decision**: Move the per-variable loop from `update_summary_factory` into the chain

**Rationale**:
- Current approach: `for idx in range(n): chain_fn(state[indices[idx]], buffer[slice])`
- Proposed: `chain_fn(state, buffer, indices, n)` with loop inside
- Reduces function call overhead per variable
- Allows compiler to see full iteration pattern for optimization

**Trade-offs**:
- Chain becomes more complex, but complexity is compile-time
- All metrics must handle multi-variable iteration

### 3. Compile-Time Buffer Layout Capture

**Decision**: Capture all layout constants (offsets, sizes, variable count) in closure

**Rationale**:
- `buffer_per_var`, `num_variables`, and metric offsets are known when factory builds
- Capturing in closure makes them compile-time constants
- Enables the compiler to unroll loops and optimize memory access patterns

### 4. Predicated Commit for All Conditional Writes

**Decision**: Use `selp(condition, new_value, old_value)` instead of `if condition: buffer[i] = new_value`

**Rationale**:
- Conditional branches cause warp divergence when threads take different paths
- Predicated execution lets all threads execute the same instruction
- `selp` maps to PTX `selp` instruction which is divergence-free

---

## Impact on Existing Architecture

### Files to Modify

| File | Changes |
|------|---------|
| `update_summaries.py` | Refactor `chain_metrics` and `update_summary_factory` |
| `metrics.py` | Update `SummaryMetric.build()` signature documentation |
| `mean.py` | Update to explicit indexing signature |
| `max.py` | Update to explicit indexing + selp pattern |
| `min.py` | Update to explicit indexing + selp pattern |
| `std.py` | Update to explicit indexing signature |
| `rms.py` | Update to explicit indexing signature |
| `extrema.py` | Update to explicit indexing + selp pattern |
| `mean_std.py` | Update to explicit indexing signature |
| `mean_std_rms.py` | Update to explicit indexing signature |
| `std_rms.py` | Update to explicit indexing signature |
| `peaks.py` | Update to explicit indexing + selp pattern |
| `negative_peaks.py` | Update to explicit indexing + selp pattern |
| `max_magnitude.py` | Update to explicit indexing + selp pattern |
| `dxdt_max.py` | Already uses selp; update indexing |
| `dxdt_min.py` | Update to explicit indexing + selp pattern |
| `dxdt_extrema.py` | Update to explicit indexing + selp pattern |
| `d2xdt2_max.py` | Update to explicit indexing + selp pattern |
| `d2xdt2_min.py` | Update to explicit indexing + selp pattern |
| `d2xdt2_extrema.py` | Update to explicit indexing + selp pattern |

### Backwards Compatibility

The external API of `update_summary_factory` remains unchanged:
- Same parameters
- Same return type (CUDA device function)
- Same device function signature for callers

Internal implementation details change but are not part of the public API.

---

## References

- Numba Issue #3405: Chain approach for dynamic CUDA function composition
- CuBIE `cuda_simsafe.py`: `selp` wrapper for predicated selection
- Existing derivative metrics (`dxdt_max.py`) already demonstrate the `selp` pattern

---

## Alternatives Considered

### 1. Shared Memory Staging
**Rejected**: Would add complexity and shared memory pressure without addressing the fundamental slicing issue.

### 2. Separate Kernels per Metric
**Rejected**: Would increase kernel launch overhead and lose the efficiency of the chain approach.

### 3. Code Generation Instead of Closures
**Considered but deferred**: Could generate optimized code per configuration, but adds significant complexity. The explicit indexing approach achieves similar benefits with less architectural change.
