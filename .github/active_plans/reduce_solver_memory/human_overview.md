# Reduce Nonlinear Solver Memory Footprint

## User Stories

### Story 1: Memory-Efficient Nonlinear Solver
**As a** CuBIE user running large-scale batch integrations  
**I want** the Newton-Krylov solver to use minimal shared memory  
**So that** I can run more concurrent integrations and achieve higher throughput

**Acceptance Criteria:**
- Newton-Krylov solver uses 2 n-sized buffers instead of 3
- Shared memory usage reduced by ~33% (from 3n to 2n)
- All existing tests pass with new implementation
- No performance regression in solver convergence
- Both single-stage (DIRK) and multi-stage (FIRK) methods work correctly

### Story 2: Inline Evaluation State Computation
**As a** developer maintaining the CuBIE codebase  
**I want** the eval_state computation moved inline to operators and residuals  
**So that** the code is clearer and eliminates unnecessary buffer copying

**Acceptance Criteria:**
- Linear operators compute `base_state + a_ij * stage_increment` inline when accessing state
- Nonlinear residuals compute `base_state + a_ij * stage_increment` inline when accessing state
- No eval_state buffer allocation in Newton-Krylov solver
- Code is self-documenting about where state evaluation happens

### Story 3: No Compatibility Breakage
**As a** CuBIE maintainer  
**I want** the refactor to maintain algorithm correctness  
**So that** all existing use cases continue to work without modification

**Acceptance Criteria:**
- All existing DIRK algorithms (Backwards Euler, Crank-Nicolson, generic DIRK) work correctly
- All existing FIRK algorithms (generic FIRK) work correctly
- Compatibility with cached and non-cached operator modes
- No changes required to user-facing APIs

## Overview

This plan addresses the excessive shared memory footprint in the Newton-Krylov solver by eliminating the `eval_state` buffer through inline computation in codegen templates.

### Current Architecture (3 Buffers)

```
Newton-Krylov Solver (shared_scratch: 3n elements)
├── delta[0:n]           - Newton search direction
├── residual[n:2n]       - Current residual
└── eval_state[2n:3n]    - Evaluation state (base_state + a_ij * stage_increment)
    │
    ├─> Passed to linear_solver()
    └─> Passed to residual_function()
```

**Problem:** The `eval_state` buffer stores `base_state[i] + a_ij * stage_increment[i]` computed once per Newton iteration, then passed to operators/residuals. This is wasteful since the computation is simple and infrequent.

### Proposed Architecture (2 Buffers)

```
Newton-Krylov Solver (shared_scratch: 2n elements)
├── delta[0:n]           - Newton search direction
└── residual[n:2n]       - Current residual

Linear Operators & Residuals:
  Compute inline: state[i] = base_state[i % n_base] + a_ij * stage_increment[i]
  Use computed state immediately in Jacobian/ODE evaluations
```

**Benefit:** 33% reduction in shared memory = 33% increase in potential concurrent warps = significant performance improvement (project is memory-bound).

### Data Flow Diagram

```
Before (3 buffers):
  Newton iteration:
    1. Compute eval_state[i] = base_state[i] + a_ij * stage_increment[i]
    2. Call linear_solver(eval_state, ..., delta)
       └─> operator_apply(eval_state, ..., v, out)  [uses eval_state]
    3. Update stage_increment[i] += scale * delta[i]
    4. Call residual_function(..., stage_increment, residual)
       └─> Uses eval_state from closure or computes inline

After (2 buffers):
  Newton iteration:
    1. Call linear_solver(..., stage_increment, base_state, a_ij, delta)
       └─> operator_apply(..., stage_increment, base_state, a_ij, v, out)
           └─> Computes: state[i] = base_state[i] + a_ij * stage_increment[i]
    2. Update stage_increment[i] += scale * delta[i]
    3. Call residual_function(..., stage_increment, base_state, a_ij, residual)
        └─> Computes: state[i] = base_state[i] + a_ij * stage_increment[i]
```

### Key Technical Decisions

#### Decision 1: Inline Computation in Codegen (CHOSEN)
Move `base_state[i] + a_ij * stage_increment[i]` computation to SymPy code generation templates.

**Rationale:**
- Computation happens in-place during state variable substitution
- No runtime overhead (compiled once, executed many times)
- Cleaner than runtime buffer reuse tricks
- Makes state evaluation location explicit in generated code

**Alternative Rejected:** Buffer reuse in Newton-Krylov solver
- Complex sequencing to avoid data clobbering
- Error-prone and hard to maintain
- Doesn't reduce actual memory usage during critical sections

#### Decision 2: Modify Codegen Substitution Pattern
Update `state_subs` dictionaries in both linear operator and nonlinear residual builders.

**Current Pattern (residuals):**
```python
state_subs[state_sym] = base[i] + aij_sym * u[i]
```

**New Pattern:**
```python
# Same! Already uses this pattern correctly.
# Just need to ensure operators follow suit.
```

**Impact:** Minimal changes to existing codegen logic; leverages existing substitution infrastructure.

#### Decision 3: Signature Changes
Change Newton-Krylov call signatures to pass `stage_increment` and `base_state` separately instead of pre-computed `eval_state`.

**Before:**
```python
linear_solver(eval_state, parameters, drivers, base_state, t, h, a_ij, ...)
operator_apply(state, parameters, drivers, base_state, t, h, a_ij, v, out)
```

**After:**
```python
linear_solver(stage_increment, parameters, drivers, base_state, t, h, a_ij, ...)
operator_apply(stage_increment, parameters, drivers, base_state, t, h, a_ij, v, out)
```

### Expected Impact

**Performance:**
- 33% reduction in shared memory per solver invocation
- Enables more concurrent blocks on GPU
- Potential 20-30% throughput increase for memory-bound workloads

**Correctness:**
- No change to mathematical operations
- Same convergence behavior
- Identical numerical results (verified via tests)

**Maintainability:**
- Simpler Newton-Krylov implementation (2 buffers instead of 3)
- State evaluation logic centralized in codegen
- Easier to understand data flow

### Research Findings

1. **Compatibility with a_ij usage:** Verified that algorithm applies `stage_base += a_ij * stage_increment` AFTER solver convergence. This is separate from the inline `base_state + a_ij * stage_increment` computation during Jacobian evaluation. No conflicts.

2. **Existing codegen patterns:** Nonlinear residual codegen already uses inline state substitution (`base[i] + aij_sym * u[i]`). Linear operators currently use pre-computed state in some paths but can be updated to match.

3. **Single vs multi-stage:** Both DIRK (single-stage) and FIRK (multi-stage) use same Newton-Krylov solver with different residual/operator codegen. Changes apply uniformly.

### Trade-offs

**Pros:**
- 33% memory reduction = significant benefit (memory-bound project)
- Cleaner code architecture
- No runtime performance penalty (computation is minimal)
- Leverages existing codegen infrastructure

**Cons:**
- Slight increase in operator/residual complexity (inline computation)
- Requires careful testing to ensure correctness
- Touches multiple critical codegen templates

**Why pros outweigh cons:** In a memory-bound GPU application, 33% reduction in shared memory is transformative. The code complexity increase is minimal and isolated to codegen templates that are already doing similar substitutions.
