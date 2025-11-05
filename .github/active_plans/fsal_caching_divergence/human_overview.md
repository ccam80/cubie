# FSAL Caching Warp Divergence Fix

## User Stories

### User Story 1: Eliminate Warp Divergence from FSAL Caching
**As a** performance-conscious user running batch integrations on GPUs  
**I want** the FSAL (First-Same-As-Last) caching mechanism to avoid warp divergence  
**So that** my integrations achieve optimal GPU performance without branching inefficiency

**Acceptance Criteria:**
- FSAL caching does not cause threads within a warp to take different execution paths
- Performance with divergent acceptance patterns is at least as good as without FSAL caching
- Correctness of integration results is maintained across all test cases
- Solution works for both ERK (explicit) and DIRK (diagonally-implicit) algorithms

### User Story 2: Data-Driven Decision on FSAL Caching Value
**As a** library maintainer  
**I want** empirical data on FSAL caching performance under realistic divergent conditions  
**So that** I can make an informed decision about keeping, modifying, or removing FSAL caching

**Acceptance Criteria:**
- Benchmark tests measure FSAL caching benefit under uniform acceptance (all threads accept)
- Benchmark tests measure FSAL caching cost under divergent acceptance (mixed accept/reject)
- Performance comparison available for multiple problem sizes and tableau types
- Clear recommendation documented based on empirical results

### User Story 3: Graceful Fallback or Convergence-Based Caching
**As a** user with heterogeneous integration scenarios  
**I want** the integration algorithm to automatically choose the most efficient execution path  
**So that** I get optimal performance without manual tuning

**Acceptance Criteria:**
- System detects when all threads in a warp have accepted their previous step
- When uniformly accepted, FSAL cache is used to skip redundant RHS computation
- When divergent, all threads compute fresh RHS to avoid branching overhead
- Transition between modes is seamless and correct

## Overview

### Problem Statement

CuBIE's generic ERK (explicit Runge-Kutta) and DIRK (diagonally-implicit Runge-Kutta) algorithms currently implement FSAL (First-Same-As-Last) caching to avoid redundant right-hand-side (RHS) evaluations. When a tableau has the FSAL property (first stage at t=0, last stage at t=1, and specific coefficient structure), the last stage's RHS from step N can be reused as the first stage's RHS for step N+1.

**Current Implementation:**
```
use_cached_rhs = ((not first_step_flag) and accepted_flag and first_same_as_last)
if use_cached_rhs:
    # Use cached RHS
else:
    # Compute fresh RHS
```

**The Divergence Issue:**
In adaptive stepping, `accepted_flag` varies per thread based on local error estimates. Within a single warp (32 threads on NVIDIA GPUs), some threads may have accepted their previous step while others rejected it. This creates warp divergence:
- Threads with `accepted_flag=True` execute the caching branch
- Threads with `accepted_flag=False` execute the computation branch
- GPU must serialize both paths, reducing parallelism efficiency

### Architectural Decision: Warp-Synchronized FSAL Caching

The solution evaluates whether FSAL caching provides net benefit under realistic divergence. Three approaches are considered:

#### Option A: Conditional Warp-Synchronized Caching
Use CUDA's `all_sync()` warp vote to cache only when all threads in a warp have accepted:
```
all_threads_accepted = all_sync(mask, accepted_flag != 0)
use_cached_rhs = all_threads_accepted and first_same_as_last and not first_step
```

**Pros:**
- Eliminates warp divergence
- Retains FSAL benefit when acceptance is uniform
- Minimal code change

**Cons:**
- Adds warp synchronization overhead
- Cache hit rate lower than per-thread decision
- Still requires benchmark validation

#### Option B: Predicated Computation with Selective Commit
Always compute the RHS but use predicated writes to commit either cached or fresh values:
```
// Always compute
compute_rhs(...)

// Selectively commit
for idx in range(n):
    stage_rhs[idx] = selp(use_cached_rhs, stage_cache[idx], computed_rhs[idx])
```

**Pros:**
- Zero divergence (all threads execute same path)
- Simpler control flow

**Cons:**
- Defeats purpose of caching (always pays computation cost)
- Likely slower than Option A

#### Option C: Remove FSAL Caching Entirely
Eliminate FSAL caching and always compute fresh RHS values.

**Pros:**
- Simplest solution
- Zero divergence
- No synchronization overhead

**Cons:**
- Loses performance benefit when acceptance is uniform
- Requires validation that loss is acceptable

### Recommended Approach

**Phase 1:** Implement Option A (warp-synchronized caching) as the primary solution
**Phase 2:** Add benchmarks comparing:
- Option A vs. Option C (no caching)
- Scenarios with 100% acceptance, 50% acceptance, 0% acceptance
- Multiple tableau types (DP54, Tsit5, etc.)

**Phase 3:** Based on empirical data, decide whether to:
- Keep Option A
- Switch to Option C
- Add configuration option for users to choose

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    IVP Loop (ode_loop.py)                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Per-Thread State:                                   │  │
│  │    - accepted_flag (varies per thread)               │  │
│  │    - first_step_flag (uniform across threads)        │  │
│  └──────────────────────────────────────────────────────┘  │
│                          │                                  │
│                          ▼                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Call step_function() with:                          │  │
│  │    - state_buffer                                    │  │
│  │    - accepted_flag (per-thread)                      │  │
│  │    - first_step_flag                                 │  │
│  │    - shared memory (includes stage_cache)            │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│     Generic ERK/DIRK Step (generic_erk.py, generic_dirk.py) │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  CURRENT (Divergent):                                │  │
│  │    use_cached_rhs = accepted_flag and ...            │  │
│  │    if use_cached_rhs:                                │  │
│  │      // Branch A (some threads)                      │  │
│  │    else:                                             │  │
│  │      // Branch B (other threads)                     │  │
│  └──────────────────────────────────────────────────────┘  │
│                          │                                  │
│                          ▼                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  PROPOSED (Warp-Synchronized):                       │  │
│  │    mask = activemask()                               │  │
│  │    all_accepted = all_sync(mask, accepted_flag != 0) │  │
│  │    use_cached_rhs = all_accepted and ...             │  │
│  │    if use_cached_rhs:                                │  │
│  │      // All threads take this path OR                │  │
│  │    else:                                             │  │
│  │      // All threads take this path                   │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Key Technical Decisions

1. **Warp Vote Primitive**: Use `all_sync(mask, condition)` from `cuda_simsafe` module
   - Already imported and used in matrix-free solvers and IVP loop
   - Handles both real CUDA and CUDASIM modes
   - `mask = activemask()` captures active thread set

2. **Cache Hit Rate vs. Divergence Cost**: 
   - Uniform acceptance (100% cache hit): FSAL saves ~1 RHS evaluation per step
   - Mixed acceptance (0% warp-level cache hit): all_sync() overhead is minimal
   - Trade-off favors warp-synchronized approach

3. **Backward Compatibility**: 
   - No API changes required
   - Compile settings unchanged
   - Existing tests should pass with improved performance characteristics

4. **Testing Strategy**:
   - Functional tests: verify correctness across acceptance patterns
   - Performance benchmarks: measure warp-level efficiency gains
   - Divergence validation: confirm no branching in profiler

### Expected Impact

**Performance:**
- Eliminate serialization overhead from warp divergence
- Retain FSAL benefits when steps are uniformly accepted
- Net improvement in realistic adaptive scenarios (mixed acceptance)

**Code Complexity:**
- Minimal increase: 2-3 lines per algorithm
- Leverages existing `all_sync()` infrastructure
- No new dependencies

**Architecture:**
- Aligns with CUDA best practices (minimize divergence)
- Consistent with existing warp-vote patterns in matrix-free solvers
- Sets precedent for future divergence-sensitive optimizations

### Alternative Considered: Per-Tableau Configuration

Allow users to disable FSAL caching via tableau attribute:
```python
tableau = ERKTableau(..., enable_fsal_caching=True)
```

**Rejected because:**
- Adds configuration complexity without clear user benefit
- Requires API changes and documentation updates
- Should make optimal choice automatically based on hardware constraints

### References to Research

- Issue #149: Original bug report identifying warp divergence
- CHANGELOG line 19: "minimal FSAL caching added to DIRK, ERK, Rosenbrock" (v0.0.5)
- Existing `all_sync()` usage in:
  - `integrators/matrix_free_solvers/newton_krylov.py` (convergence checks)
  - `integrators/matrix_free_solvers/linear_solver.py` (convergence checks)
  - `integrators/loops/ode_loop.py` (early termination)
- NVIDIA CUDA Programming Guide: Warp divergence costs and mitigation strategies
