# Reduce Nonlinear Solver Memory Footprint

## User Stories

### As a CuBIE User
**I want** the nonlinear solver to use less shared memory  
**So that** I can run larger systems or achieve higher throughput on memory-constrained GPUs

**Acceptance Criteria:**
- Nonlinear solver reduces from 3 n-sized buffers to 2 n-sized buffers in shared memory
- All existing tests pass with the modified implementation
- Performance is maintained or improved (due to reduced memory pressure)
- Single-stage (DIRK) and multi-stage (FIRK) integrators both benefit from the change

### As a Performance-Conscious Developer
**I want** the eval_state computation to be performed inline  
**So that** we avoid unnecessary memory allocations without sacrificing code clarity

**Acceptance Criteria:**
- Linear operator and nonlinear residual functions compute eval_state inline
- No performance degradation from inline computation
- Code remains maintainable and follows CUDA best practices

## Overview

The nonlinear Newton-Krylov solver currently allocates 3 n-sized buffers in shared memory:
1. `delta` - Newton direction vector
2. `residual` - Residual vector
3. `eval_state` - Stage evaluation state (base_state + a_ij * stage_increment)

The third buffer (`eval_state`) was added to support multistage methods but can be eliminated by computing the evaluation state inline within the linear operator and nonlinear residual functions.

### Current Architecture

```
Newton-Krylov Solver (newton_krylov.py)
├── Allocates shared_scratch[3*n]
│   ├── delta = shared_scratch[:n]
│   ├── residual = shared_scratch[n:2*n]
│   └── eval_state = shared_scratch[2*n:3*n]
│
├── Computes eval_state (line 177):
│   eval_state[i] = base_state[i % n_base] + a_ij * stage_increment[i]
│
└── Passes eval_state to:
    ├── linear_solver (lines 178-188)
    └── residual_function (implicitly via operator calls)
```

### Proposed Architecture

```
Newton-Krylov Solver (newton_krylov.py)
├── Allocates shared_scratch[2*n]  ← REDUCED
│   ├── delta = shared_scratch[:n]
│   └── residual = shared_scratch[n:2*n]
│
└── Passes stage_increment + base_state + a_ij to:
    ├── linear_solver → operator (computes eval inline)
    └── residual_function (computes eval inline)

Linear Operator (linear_operators.py codegen)
└── Computes inline: state_eval[i] = base_state[i % n_base] + a_ij * stage_increment[i]
    └── Uses state_eval for Jacobian evaluation

Nonlinear Residual (nonlinear_residuals.py codegen)
└── Computes inline: state_eval[i] = base_state[i % n_base] + a_ij * stage_increment[i]
    └── Uses state_eval for f(t, state) evaluation
```

### Memory Impact

**Before:** 3n elements in shared memory  
**After:** 2n elements in shared memory  
**Reduction:** 33% fewer elements = 33% more throughput (memory-bound workloads)

For a typical system with n=50 states and float32 precision:
- Before: 3 × 50 × 4 bytes = 600 bytes
- After: 2 × 50 × 4 bytes = 400 bytes
- Savings: 200 bytes per thread

With 256 threads per block:
- Before: 153,600 bytes (exceeds 48KB on some GPUs)
- After: 102,400 bytes (fits comfortably)

### Key Technical Decisions

**Decision 1: Inline Computation vs Buffer Reuse**
- **Chosen:** Inline computation
- **Rationale:** 
  - Cleaner separation of concerns
  - No risk of data clobbering
  - Compiler can optimize redundant computations
  - Matches the pattern used elsewhere in codegen

**Decision 2: Signature Changes**
- **Operator signature change:** Add `stage_increment` parameter, remove `state` parameter
- **Residual signature:** Already has all needed parameters (`u` is stage_increment)
- **Linear solver:** Pass `stage_increment` instead of `eval_state`

**Decision 3: Both Single-Stage and Multi-Stage**
- Both DIRK (single stage, n elements) and FIRK (multi-stage, s*n elements) benefit
- FIRK has even larger impact: 3*s*n → 2*s*n elements
- For FIRK with s=3 stages, n=50: 1800 bytes → 1200 bytes per thread

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────┐
│  Newton-Krylov Iteration                             │
│  ┌────────────┐                                      │
│  │ Initialize │                                      │
│  │ residual   │                                      │
│  └─────┬──────┘                                      │
│        │                                             │
│        ▼                                             │
│  ┌────────────────────────────────┐                 │
│  │ Linear Solver (Krylov)         │                 │
│  │  ┌──────────────────────────┐  │                 │
│  │  │ Operator Apply           │  │                 │
│  │  │  1. Compute inline:      │  │                 │
│  │  │     eval = base + a*inc  │  │  ← NEW         │
│  │  │  2. Evaluate J @ v       │  │                 │
│  │  └──────────────────────────┘  │                 │
│  │         ▼                       │                 │
│  │  ┌──────────────────────────┐  │                 │
│  │  │ Preconditioner           │  │                 │
│  │  │  (also inline if needed) │  │                 │
│  │  └──────────────────────────┘  │                 │
│  └───────────┬────────────────────┘                 │
│              │ delta                                 │
│              ▼                                       │
│  ┌────────────────────────────────┐                 │
│  │ Backtracking Line Search       │                 │
│  │  ┌──────────────────────────┐  │                 │
│  │  │ Residual Evaluation      │  │                 │
│  │  │  1. Compute inline:      │  │                 │
│  │  │     eval = base + a*inc  │  │  ← NEW         │
│  │  │  2. Evaluate f(t, eval)  │  │                 │
│  │  └──────────────────────────┘  │                 │
│  └────────────────────────────────┘                 │
└─────────────────────────────────────────────────────┘
```

### Integration Points

**Files Modified:**
1. `newton_krylov.py` - Remove eval_state buffer, update calls
2. `linear_operators.py` - Modify codegen templates to compute eval inline
3. `nonlinear_residuals.py` - Modify codegen templates to compute eval inline
4. `linear_solver.py` - Update signature to pass stage_increment
5. `generic_firk.py` - Update solver_shared_elements property
6. `generic_dirk.py` - Update solver_shared_elements property

**Backward Compatibility:**
- Breaking change to generated code (recompilation required)
- No API changes for end users
- Package is in development, breaking changes acceptable

### Alternatives Considered

**Alternative 1: Reuse delta buffer for eval_state**
- Pros: No codegen changes
- Cons: Complex lifetime management, risk of data corruption, harder to understand
- Decision: Rejected due to complexity and fragility

**Alternative 2: Reuse residual buffer for eval_state**
- Pros: No codegen changes
- Cons: Even more complex sequencing, residual needed after eval_state
- Decision: Rejected as infeasible

**Alternative 3: Use local memory for eval_state**
- Pros: Simple change
- Cons: Local memory spills to global if too large, performance penalty
- Decision: Rejected per project constraints (no local unless immediate assign→consume)

### Expected Impact

**Performance:**
- 33% reduction in shared memory per solver invocation
- Enables larger systems on memory-constrained GPUs
- May improve occupancy on some GPU architectures
- Inline computation cost is negligible (compiler optimizes)

**Code Quality:**
- Clearer separation: solver manages iteration, operators/residuals manage evaluation
- More maintainable: eval computation near where it's used
- Follows existing patterns in other codegen modules

**Testing:**
- All existing solver tests should pass unchanged
- No new test cases needed (behavior is identical)
- Instrumented tests verify buffer usage
