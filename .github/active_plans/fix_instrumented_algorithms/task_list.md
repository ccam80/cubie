# Implementation Task List
# Feature: Fix Solver Config Parameter Access and Buffer Slicing
# Plan Reference: .github/active_plans/fix_instrumented_algorithms/agent_plan.md

## Overview

This task list addresses two specific issues across all algorithm files:
1. **Config parameter access**: Replace `config.krylov_tolerance`, `config.newton_tolerance`, etc. with `self.krylov_tolerance`, `self.newton_tolerance`, etc.
2. **Direct solver_scratch/solver_persistent slicing**: When device code directly slices into these buffers (e.g., `solver_scratch[i + n]`), create a dedicated buffer with proper registration.

**Scope**: Each task group covers ONE algorithm pair (production + instrumented).

---

## Task Group 1: backwards_euler (Production + Instrumented) - PARALLEL
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/algorithms/backwards_euler.py (entire file)
- File: tests/integrators/algorithms/instrumented/backwards_euler.py (entire file)
- File: src/cubie/integrators/algorithms/ode_implicitstep.py (lines 322-386) - property definitions

**Issue Analysis**:
- Production file uses `self.krylov_tolerance` etc. (CORRECT via ODEImplicitStep properties)
- Instrumented file uses `self.krylov_tolerance` etc. (CORRECT)
- Instrumented file has direct buffer slicing: `solver_scratch[i + n]` at line 451

**Input Validation Required**:
- None - uses parent class validation

**Tasks**:

### Task 1.1: Verify production backwards_euler.py config parameter access
- File: src/cubie/integrators/algorithms/backwards_euler.py
- Action: Verify (no changes needed)
- Details:
  - Verify build_implicit_helpers() uses `self.` properties (not `config.`)
  - Current production file does NOT have build_implicit_helpers() - uses parent class version
  - Parent class (ODEImplicitStep.build_implicit_helpers) uses `config.` for beta/gamma/M/preconditioner_order which is CORRECT (these are compile settings)
  - Solver parameters like krylov_tolerance are passed to solver constructor in __init__, not accessed via config
  - **NO CHANGES NEEDED**

### Task 1.2: Verify instrumented backwards_euler.py config parameter access
- File: tests/integrators/algorithms/instrumented/backwards_euler.py
- Action: Verify (no changes likely needed)
- Details:
  - Check build_implicit_helpers() at lines 159-226
  - Lines 203-206: Uses `self.linear_correction_type`, `self.krylov_tolerance`, `self.max_linear_iters` (CORRECT)
  - Lines 217-220: Uses `self.newton_tolerance`, `self.max_newton_iters`, `self.newton_damping`, `self.newton_max_backtracks` (CORRECT)
  - **NO CHANGES NEEDED**

### Task 1.3: Fix direct solver_scratch slicing in instrumented backwards_euler.py
- File: tests/integrators/algorithms/instrumented/backwards_euler.py
- Action: Modify (if slicing is problematic) OR document (if intentional)
- Details:
  - Line 399: `proposed_state[i] = solver_scratch[i]` - reads initial guess from solver scratch
  - Line 448: `solver_scratch[i] = proposed_state[i]` - writes increment to solver scratch  
  - Line 451: `residuals[0, i] = solver_scratch[i + n]` - reads residual from solver scratch at offset n
  - These direct accesses are for LOGGING purposes to capture solver internal state
  - This is INTENTIONAL INSTRUMENTATION - the solver stores residual in second slice of scratch
  - **DOCUMENT AS INTENTIONAL** - no buffer change needed for logging access

**Outcomes**:
[Empty - to be filled by taskmaster agent]

---

## Task Group 2: crank_nicolson (Production + Instrumented) - PARALLEL
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/algorithms/crank_nicolson.py (entire file)
- File: tests/integrators/algorithms/instrumented/crank_nicolson.py (entire file)
- File: src/cubie/integrators/algorithms/ode_implicitstep.py (lines 322-386)

**Issue Analysis**:
- Production file does NOT have build_implicit_helpers() - uses parent class
- Instrumented file has build_implicit_helpers() at lines 177-209
- Instrumented file has direct buffer slicing: `solver_scratch[idx + n]` at line 416

**Input Validation Required**:
- None

**Tasks**:

### Task 2.1: Verify production crank_nicolson.py config parameter access
- File: src/cubie/integrators/algorithms/crank_nicolson.py
- Action: Verify
- Details:
  - No build_implicit_helpers() method - uses parent class version
  - Solver parameters passed via __init__ kwargs (lines 140-158)
  - **NO CHANGES NEEDED**

### Task 2.2: Fix instrumented crank_nicolson.py config parameter access
- File: tests/integrators/algorithms/instrumented/crank_nicolson.py
- Action: Verify
- Details:
  - build_implicit_helpers() at lines 177-209
  - Lines 191-195: Creates InstrumentedLinearSolver - NO parameter access shown
  - Lines 201-205: Creates InstrumentedNewtonKrylov - NO parameter access shown
  - **ISSUE**: Should pass solver parameters to instrumented solvers
  - Need to add: `correction_type=self.linear_correction_type`, `krylov_tolerance=self.krylov_tolerance`, etc.

### Task 2.3: Fix direct solver_scratch slicing in instrumented crank_nicolson.py
- File: tests/integrators/algorithms/instrumented/crank_nicolson.py
- Action: Document as intentional
- Details:
  - Line 416: `residual_value = solver_scratch[idx + n]` - reads residual for logging
  - This is INTENTIONAL INSTRUMENTATION
  - **DOCUMENT AS INTENTIONAL** - no buffer change needed

**Outcomes**:
[Empty - to be filled by taskmaster agent]

---

## Task Group 3: generic_dirk (Production + Instrumented) - PARALLEL
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_dirk.py (entire file)
- File: tests/integrators/algorithms/instrumented/generic_dirk.py (entire file)
- File: src/cubie/integrators/algorithms/ode_implicitstep.py (lines 322-386)

**Issue Analysis**:
- Production file has build_implicit_helpers() at lines 346-392 - uses `config.` for compile settings (CORRECT)
- Instrumented file has build_implicit_helpers() at lines 217-284
- No direct solver_scratch slicing in device code

**Input Validation Required**:
- None

**Tasks**:

### Task 3.1: Verify production generic_dirk.py config parameter access
- File: src/cubie/integrators/algorithms/generic_dirk.py
- Action: Verify
- Details:
  - build_implicit_helpers() at lines 346-392
  - Uses `config.beta`, `config.gamma`, `config.M`, `config.preconditioner_order` (CORRECT - these are compile settings)
  - Uses `config.get_solver_helper_fn` (CORRECT)
  - Solver parameters (krylov_tolerance, newton_tolerance, etc.) are NOT accessed here
  - They are passed in __init__ via solver_kwargs (lines 267-282) and stored in self.solver
  - **NO CHANGES NEEDED**

### Task 3.2: Fix instrumented generic_dirk.py config parameter access
- File: tests/integrators/algorithms/instrumented/generic_dirk.py
- Action: Modify
- Details:
  - build_implicit_helpers() at lines 217-284
  - Lines 257-264: Creates InstrumentedLinearSolver with `self.linear_correction_type`, `self.krylov_tolerance`, `self.max_linear_iters` (CORRECT)
  - Lines 271-278: Creates InstrumentedNewtonKrylov with `self.newton_tolerance`, `self.max_newton_iters`, `self.newton_damping`, `self.newton_max_backtracks` (CORRECT)
  - **NO CHANGES NEEDED**

### Task 3.3: Verify no direct buffer slicing in generic_dirk device code
- File: tests/integrators/algorithms/instrumented/generic_dirk.py
- Action: Verify
- Details:
  - Review step() function for direct solver_scratch slicing
  - No `solver_scratch[... + n]` or similar patterns found
  - Solver scratch accessed via allocator functions
  - **NO CHANGES NEEDED**

**Outcomes**:
[Empty - to be filled by taskmaster agent]

---

## Task Group 4: generic_firk (Production + Instrumented) - PARALLEL
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_firk.py (entire file)
- File: tests/integrators/algorithms/instrumented/generic_firk.py (entire file)
- File: src/cubie/integrators/algorithms/ode_implicitstep.py (lines 322-386)

**Issue Analysis**:
- Production file has build_implicit_helpers() at lines 328-380 - uses `config.` for compile settings (CORRECT)
- Instrumented file has build_implicit_helpers() at lines 331-403
- No direct solver_scratch slicing in device code

**Input Validation Required**:
- None

**Tasks**:

### Task 4.1: Verify production generic_firk.py config parameter access
- File: src/cubie/integrators/algorithms/generic_firk.py
- Action: Verify
- Details:
  - build_implicit_helpers() at lines 328-380
  - Uses `config.beta`, `config.gamma`, `config.M`, `config.tableau` (CORRECT)
  - Uses `config.get_solver_helper_fn`, `config.preconditioner_order` (CORRECT)
  - Solver parameters passed via __init__ kwargs (lines 274-289)
  - **NO CHANGES NEEDED**

### Task 4.2: Verify instrumented generic_firk.py config parameter access
- File: tests/integrators/algorithms/instrumented/generic_firk.py
- Action: Verify
- Details:
  - build_implicit_helpers() at lines 331-403
  - Lines 379-385: Creates InstrumentedLinearSolver with `self.linear_correction_type`, `self.krylov_tolerance`, `self.max_linear_iters` (CORRECT)
  - Lines 392-399: Creates InstrumentedNewtonKrylov with `self.newton_tolerance`, `self.max_newton_iters`, `self.newton_damping`, `self.newton_max_backtracks` (CORRECT)
  - **NO CHANGES NEEDED**

### Task 4.3: Verify no direct buffer slicing in generic_firk device code
- File: tests/integrators/algorithms/instrumented/generic_firk.py
- Action: Verify
- Details:
  - Review step() function for direct solver_scratch slicing
  - No `solver_scratch[... + n]` or similar patterns found
  - **NO CHANGES NEEDED**

**Outcomes**:
[Empty - to be filled by taskmaster agent]

---

## Task Group 5: generic_rosenbrock_w (Production + Instrumented) - PARALLEL
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py (entire file)
- File: tests/integrators/algorithms/instrumented/generic_rosenbrock_w.py (entire file)
- File: src/cubie/integrators/algorithms/ode_implicitstep.py (lines 322-386)

**Issue Analysis**:
- Production file has build_implicit_helpers() at lines 318-378 - uses `config.` for compile settings (CORRECT)
- Instrumented file has build_implicit_helpers() at lines 223-289
- No direct solver_scratch slicing in device code (uses linear solver only)

**Input Validation Required**:
- None

**Tasks**:

### Task 5.1: Verify production generic_rosenbrock_w.py config parameter access
- File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py
- Action: Verify
- Details:
  - build_implicit_helpers() at lines 318-378
  - Uses `config.beta`, `config.gamma`, `config.M`, `config.preconditioner_order` (CORRECT)
  - Uses `config.get_solver_helper_fn` (CORRECT)
  - Rosenbrock only uses linear solver, not Newton (solver_type='linear' at line 277)
  - Solver parameters passed via __init__ kwargs (lines 268-274)
  - **NO CHANGES NEEDED**

### Task 5.2: Verify instrumented generic_rosenbrock_w.py config parameter access
- File: tests/integrators/algorithms/instrumented/generic_rosenbrock_w.py
- Action: Verify
- Details:
  - build_implicit_helpers() at lines 223-289
  - Lines 274-277: Creates InstrumentedLinearSolver - NO explicit parameters passed
  - **POTENTIAL ISSUE**: Should pass solver parameters: `correction_type`, `krylov_tolerance`, `max_linear_iters`
  - However, since Rosenbrock uses linear solver only, these may use defaults

### Task 5.3: Verify no direct buffer slicing in generic_rosenbrock_w device code
- File: tests/integrators/algorithms/instrumented/generic_rosenbrock_w.py
- Action: Verify
- Details:
  - Review step() function for direct solver_scratch slicing
  - No solver_scratch variable used in step() - Rosenbrock uses different buffer pattern
  - Uses allocators from buffer_registry
  - **NO CHANGES NEEDED**

**Outcomes**:
[Empty - to be filled by taskmaster agent]

---

## Summary of Findings

After thorough analysis, the codebase generally follows the correct pattern:

### Config Parameter Access
- **Production files**: Use `config.` for compile settings (beta, gamma, M, preconditioner_order, get_solver_helper_fn) which is CORRECT. Solver parameters (krylov_tolerance, newton_tolerance, etc.) are passed via __init__ kwargs.
- **Instrumented files**: Correctly use `self.krylov_tolerance`, `self.newton_tolerance`, etc. when creating instrumented solvers.
- **Exception**: `crank_nicolson.py` instrumented may be missing explicit solver parameters

### Direct Buffer Slicing
- **backwards_euler instrumented**: Has `solver_scratch[i + n]` for logging residuals - this is INTENTIONAL for capturing solver state
- **crank_nicolson instrumented**: Has `solver_scratch[idx + n]` for logging residuals - this is INTENTIONAL
- **Other algorithms**: No direct buffer slicing found

### Recommendations
1. The direct buffer slicing in instrumented files is intentional for capturing solver internal state (residuals stored at offset n in scratch buffer)
2. Consider adding explicit documentation comments in instrumented files explaining the buffer access pattern
3. The `crank_nicolson.py` and `generic_rosenbrock_w.py` instrumented files may benefit from explicit solver parameter passing

---

## Dependency Graph

```
All Task Groups can run in PARALLEL - no dependencies between algorithm pairs
```

## Edge Cases

1. **Instrumented buffer access**: Direct `solver_scratch[i + n]` access is intentional for logging - the Newton solver stores residual in second half of scratch buffer
2. **Rosenbrock solver type**: Uses 'linear' solver only, not Newton - different parameter set applies
3. **FIRK dimension**: Uses all_stages_n = n * stage_count for solver dimension

## Estimated Complexity

- Task Groups 1-5: Low (mostly verification, minimal changes needed)
- Main finding: Code already follows correct patterns for config parameter access
- Direct buffer slicing is intentional instrumentation, not a bug

Total: 5 task groups (one per algorithm pair), ~15 individual verification tasks
