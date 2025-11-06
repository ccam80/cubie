# Iteration Count Output Feature - Overview

## User Stories

### Story 1: Newton Iteration Diagnostics
**As a** user performing implicit ODE integration with Newton-Krylov methods  
**I want to** see the number of Newton iterations performed at each save point  
**So that I can** diagnose convergence issues, tune solver parameters (newton_tolerance, max_newton_iters), and identify problematic regions in my integration

**Acceptance Criteria:**
- Newton iteration counts are optionally output at each save point
- Counts are controlled by a compile-time flag (e.g., `output_newton_iterations=True`)
- Output array has shape `(n_saves,)` or `(n_saves, 1)` consistent with other outputs
- Zero or minimal performance overhead when disabled
- Iteration count accurately reflects the number of Newton iterations performed

### Story 2: Linear Solver (Krylov) Iteration Diagnostics
**As a** user performing implicit ODE integration  
**I want to** see the number of linear solver (GMRES/MR) iterations performed  
**So that I can** tune linear solver parameters (krylov_tolerance, max_linear_iters, preconditioner_order) and understand computational cost

**Acceptance Criteria:**
- Krylov iteration counts are optionally output at each save point
- Counts are controlled by a compile-time flag (e.g., `output_krylov_iterations=True`)
- Output array has shape `(n_saves,)` consistent with other diagnostic outputs
- Can be enabled independently of Newton iteration tracking
- Minimal performance overhead when disabled

### Story 3: Step Controller Diagnostics
**As a** user using adaptive step controllers  
**I want to** see step rejection counts and total step counts  
**So that I can** tune controller parameters and understand the adaptive behavior

**Acceptance Criteria:**
- Total step count (accepted + rejected) is optionally output at each save point
- Step rejection count is optionally output at each save point
- Counts are controlled by compile-time flags (e.g., `output_step_counts=True`)
- Works with all adaptive controllers (I, PI, PID, Gustafsson)
- Fixed-step controllers can also report total step count

### Story 4: Integration Step Information
**As a** user analyzing solver performance  
**I want to** see the number of integration steps between save points  
**So that I can** understand the computational work distribution and adaptive behavior over time

**Acceptance Criteria:**
- Number of steps since last save is optionally output
- Memory-efficient approach (no dense time-step arrays)
- Controlled by compile-time flags

## Executive Summary

This feature adds optional diagnostic iteration count outputs to CuBIE's implicit solvers. The implementation follows CuBIE's existing output architecture using compile-time flags and optional output arrays, ensuring zero overhead when disabled.

### Key Technical Decisions

1. **Output Architecture**: Use existing `OutputConfig` + `OutputCompileFlags` pattern
2. **Data Flow**: Iteration counts accumulate in loop, passed to `save_state` function as array
3. **Memory Strategy**: Single `counters` array with 4 int32 values, reset between saves
4. **Status Word Enhancement**: Linear solver returns iteration count like Newton solver
5. **User Interface**: Single "iteration_counters" output type (all-or-nothing)

### Implementation Scope

**New Output Type** (added to `output_types` list):
- `"iteration_counters"` - All iteration diagnostics in one array per save
  - Index 0: Newton iteration counts
  - Index 1: Krylov/linear solver iteration counts
  - Index 2: Total integration steps between saves
  - Index 3: Rejected steps between saves (adaptive only, 0 for fixed-step)

**Modified Components**:
- `save_state_factory()` - Extended signature to accept counters array
- `linear_solver_factory()` - Return iteration count in status word
- `IVPLoop.build()` - Track iteration counts in array, pass to save_state
- `OutputConfig` - Recognize "iteration_counters" output type, create flag
- `OutputArrays` - Allocate counters output array (shape: n_saves × 4)
- `SolveResult` - Expose iteration_counters array property

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Integration Loop (ode_loop.py)               │
│                                                                 │
│  Iteration Counters Array (local memory):                      │
│  ┌────────────────────────────────────────────────────────┐   │
│  │ counters_since_save: int32[4]                           │   │
│  │   [0] newton_iters        : int32 (accumulator)         │   │
│  │   [1] krylov_iters        : int32 (accumulator)         │   │
│  │   [2] steps               : int32 (counter)             │   │
│  │   [3] rejected_steps      : int32 (counter)             │   │
│  └────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Main Loop:                                                     │
│  ┌────────────────────────────────────────────────────────┐   │
│  │ 1. step_status = step_function(...)                     │   │
│  │ 2. Extract Newton iters from status: (status >> 16)     │   │
│  │ 3. counters_since_save[0] += niters                     │   │
│  │ 4. Extract Krylov iters: (lin_status >> 16)             │   │
│  │ 5. counters_since_save[1] += kriters                    │   │
│  │ 6. counters_since_save[2] += 1                          │   │
│  │ 7. if not accept: counters_since_save[3] += 1           │   │
│  │                                                          │   │
│  │ 8. if do_save:                                           │   │
│  │      save_state(..., counters_output_slice,             │   │
│  │                     counters_since_save)                │   │
│  │      for i in range(4):                                  │   │
│  │          counters_since_save[i] = 0  # reset            │   │
│  └────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              save_state_func (save_state.py)                    │
│                                                                 │
│  New Signature:                                                 │
│  save_state_func(current_state, current_observables,           │
│                  output_states_slice, output_observables_slice,│
│                  current_step,                                  │
│                  output_counters_slice,    ← NEW               │
│                  counters_array)           ← NEW               │
│                                                                 │
│  Compile-time branching based on OutputCompileFlags:           │
│  if flags.output_iteration_counters:                           │
│      for i in range(4):                                         │
│          output_counters_slice[i] = counters_array[i]          │
└─────────────────────────────────────────────────────────────────┘
```

## Status Word Encoding (Extended)

Newton-Krylov solver currently uses 32 bits:
```
Current (newton_krylov.py):
┌────────────────┬────────────────┐
│ Bits 31-16     │ Bits 15-0      │
│ Newton iters   │ Status code    │
└────────────────┴────────────────┘
```

**Option A**: Use 64-bit status word (cleaner but requires broader changes)
```
Proposed (64-bit):
┌────────────────┬────────────────┬────────────────┬────────────────┐
│ Bits 63-48     │ Bits 47-32     │ Bits 31-16     │ Bits 15-0      │
│ Reserved       │ Krylov iters   │ Newton iters   │ Status code    │
└────────────────┴────────────────┴────────────────┴────────────────┘
```

**Option B**: Separate return (simpler, chosen approach)
```
- Newton solver: Returns 32-bit with iters in upper 16 bits (unchanged)
- Linear solver: Modified to return iters in upper 16 bits of 32-bit return
- Loop extracts both and accumulates separately
```

**Decision**: Use Option B - separate accumulation in loop
- Less invasive change
- Maintains compatibility
- Linear solver factory modified similarly to Newton solver

## Data Flow Summary

1. **Compilation Phase**:
   - User specifies `output_types=["state", "iteration_counters"]`
   - `OutputConfig` creates flag: `output_iteration_counters=True`
   - `save_state_factory()` receives flag, compiles with conditional branch
   - `IVPLoop` allocates counters array (4 int32 values) based on flag

2. **Execution Phase**:
   - Loop tracks iterations in `counters_since_save` array
   - On save: passes counters array to `save_state_func()`
   - `save_state_func()` writes all 4 values to device output array (compile-time branched)
   - All 4 counters reset to zero after save

3. **Output Phase**:
   - `OutputArrays` transfers counters array to host (shape: n_saves × 4)
   - `SolveResult` exposes array via property
   - User accesses: `result.iteration_counters[:, 0]` for Newton, etc.

## Memory Impact

Per-thread memory additions (only when enabled):
- Counters array: 4 × 4 bytes = 16 bytes (int32[4])
- **Total**: 16 bytes local memory per thread (negligible)

Output array memory (host + device):
- Single counters array: `n_saves × 4 × sizeof(int32)` bytes per run
- Example: 1000 saves × 4 × 4 bytes = 16 KB per run
- For 1M runs: ~16 GB (significant but comparable to state outputs)

## Trade-offs Considered

### Dense vs. Sparse Time Step Arrays
**Decision**: Do NOT output dense timestep arrays
- **Reasoning**: Memory prohibitive, unpredictable size, defeats batching
- **Alternative**: Output steps-since-last-save (sparse, predictable size)

### Accumulation vs. Per-Step Storage
**Decision**: Accumulate between saves
- **Reasoning**: Matches user need ("how much work between saves?")
- **Alternative**: Store per-step arrays (rejected: too much memory)

### 64-bit vs. Separate Status Words
**Decision**: Separate accumulation in loop
- **Reasoning**: Simpler implementation, less invasive
- **Alternative**: 64-bit status (rejected: requires changes throughout)

### Compile-time vs. Runtime Flags
**Decision**: Compile-time flag (existing pattern)
- **Reasoning**: Zero overhead when disabled, follows CuBIE conventions
- **Alternative**: Runtime checks (rejected: performance cost)

### All-or-Nothing vs. Individual Counter Flags
**Decision**: Single "iteration_counters" output type
- **Reasoning**: Simplifies flag logic, signature changes, and user interface
- **Reasoning**: Diagnostic outputs are typically used together
- **Alternative**: Individual flags (rejected: complexity, multiple arrays)

## Expected Impact on Existing Architecture

**Minimal Breaking Changes**:
- `save_state` signature extension is managed internally
- Existing code without iteration output continues to work
- Zero performance impact when disabled (compile-time branching)

**New Capabilities**:
- Diagnostic-driven parameter tuning
- Convergence analysis
- Performance profiling
- Identification of problematic time regions

## Success Metrics

1. **Functionality**: All iteration types correctly tracked and output
2. **Performance**: <1% overhead when enabled, 0% when disabled
3. **Usability**: Simple flag-based interface matching existing output patterns
4. **Accuracy**: Iteration counts match expected theoretical values in tests
5. **Memory**: Predictable, bounded memory usage

## References

### Internal Architecture
- `.github/context/cubie_internal_structure.md` - CuBIE architecture patterns
- Issue #125 - Original feature request
- `newton_krylov.py` - Current iteration count encoding pattern
- `output_config.py` - Output configuration system
- `save_state.py` - State saving infrastructure

### Research Findings
- **SciPy**: Returns solve statistics in result object (nfev, njev, nlu)
- **Sundials**: Extensive diagnostics via CVodeGetStats family
- **PETSc**: KSPGetIterationNumber, SNESGetIterationNumber APIs
- **Pattern**: Iteration counts are standard diagnostic outputs in scientific computing
