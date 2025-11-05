# User Stories: Warp Divergence at Step-Save

## User Personas

### 1. Performance-Conscious GPU Developer
A researcher or engineer using CuBIE for large-scale batch ODE/SDE integration who needs maximum GPU efficiency and is sensitive to warp divergence issues that could degrade performance.

### 2. Scientific Computing Researcher
A scientist running likelihood-free inference or other computationally intensive simulations that require millions of integration runs with infrequent saves, where performance bottlenecks directly impact research throughput.

### 3. CuBIE Library Maintainer
A developer maintaining the CuBIE codebase who needs to understand the performance implications of different implementation strategies for save operations and make informed architectural decisions.

## User Stories

### Story 1: Investigate Warp Divergence Impact
**As a** CuBIE library maintainer  
**I want to** understand the actual performance impact of warp divergence at save boundaries  
**So that** I can make an evidence-based decision about whether optimization is necessary

**Acceptance Criteria:**
- Analysis of current save logic identifies where warp divergence occurs
- Documentation explains the three potential approaches:
  1. Current branching approach (threads save independently)
  2. Predicated commit approach (all threads execute save logic, commit conditionally)
  3. Warp sync approach (threads wait at save boundary for slowest thread)
- Performance characteristics of each approach are documented
- A recommendation is provided based on analysis

### Story 2: Quantify Performance Under Typical Workloads
**As a** performance-conscious GPU developer  
**I want to** know how warp divergence at save points affects my integration performance  
**So that** I can understand if this is a bottleneck in my workflow

**Acceptance Criteria:**
- Benchmark or analysis tool exists to measure the impact of save divergence
- Documentation provides guidance on when save divergence is likely vs unlikely to be a bottleneck
- Workload characteristics that amplify/minimize the issue are identified (e.g., steps-per-save ratio, system complexity)

### Story 3: Document Architectural Decision
**As a** CuBIE library maintainer  
**I want to** have a clear architectural decision record about the save divergence issue  
**So that** future developers understand why the current implementation was chosen

**Acceptance Criteria:**
- Documentation clearly explains the warp divergence issue at save boundaries
- Trade-offs of each approach are documented with reasoning
- The chosen approach (whether changed or status quo) is justified
- Future optimization opportunities are noted if applicable

## Success Metrics

### Primary Metrics
- **Clarity**: Issue is thoroughly analyzed with documented findings
- **Decision Quality**: Architectural decision is evidence-based and well-justified
- **Documentation**: Future developers can understand the trade-offs

### Secondary Metrics
- **Performance Impact**: If changes are made, they should not regress performance for typical workloads
- **Code Maintainability**: Any changes should maintain or improve code clarity

## Edge Cases and Constraints

### Edge Cases to Consider
1. **High save frequency**: When `dt_save` is small relative to `dt0`, saves happen frequently
2. **Low save frequency**: When saves are very infrequent (many steps between saves)
3. **Varied convergence rates**: Adaptive stepping with widely varying step sizes across batch
4. **Fixed vs adaptive stepping**: Different behavior in fixed-step mode

### Constraints
1. Must maintain backward compatibility of results (same numerical outputs)
2. Must work in both CUDA and CUDASIM modes
3. Should not increase memory usage significantly
4. Must preserve thread safety and correctness

## Context and Background

The issue arises in the main integration loop (`ode_loop.py` lines 454-483) where threads make a runtime decision to save based on whether they've crossed a save boundary. In a warp of 32 threads executing the same code:

- **Current behavior**: Each thread independently checks `if do_save:` and executes save logic only if true
- **Problem**: Threads may diverge - some execute the save branch, others skip it
- **Impact**: Unknown - depends on how CUDA handles this divergence

The owner's analysis suggests that for typical workloads (many steps between saves), the current approach may already be optimal because:
- Warp divergence at save is infrequent (happens every N steps where N >> 32)
- Threads quickly rejoin after the divergent save operation
- Alternative approaches (predicated commit or warp sync) may have higher overhead

This investigation should validate or refute these assumptions with evidence.
