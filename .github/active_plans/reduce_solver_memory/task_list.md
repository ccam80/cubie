# Implementation Task List
# Feature: Reduce Nonlinear Solver Memory Footprint
# Plan Reference: .github/active_plans/reduce_solver_memory/agent_plan.md

## Overview

This task list implements the reduction of Newton-Krylov solver shared memory usage from 3n to 2n elements by eliminating the `eval_state` buffer and computing evaluation states inline within linear operators and nonlinear residuals.

**Total Task Groups**: 5  
**Dependency Chain**: Sequential (Groups 1 → 2 → 3 → 4 → 5)  
**Estimated Complexity**: Medium (core architecture changes, no new functionality)

---

## Task Group 1: Update Base Implicit Step Class - SEQUENTIAL
**Status**: [ ]  
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/algorithms/ode_implicitstep.py (lines 300-305)

**Input Validation Required**:
- None (property change only)

**Tasks**:

1. **Update solver_shared_elements property in ODEImplicitStep**
   - File: src/cubie/integrators/algorithms/ode_implicitstep.py
   - Action: Modify
   - Line: 304
   - Details:
     ```python
     @property
     def solver_shared_elements(self) -> int:
         """Return shared scratch dedicated to the Newton--Krylov solver."""
         
         return self.compile_settings.n * 2  # Changed from 3 to 2
     ```
   - Edge cases: None (simple constant change)
   - Integration: Base class for all implicit steps (DIRK, Backwards Euler, Crank-Nicolson)
   - Rationale: Reduces allocation from 3n to 2n elements (removes eval_state buffer)

**Outcomes**:

---

## Task Group 2: Update FIRK Algorithm Shared Memory - SEQUENTIAL
**Status**: [ ]  
**Dependencies**: Group 1

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_firk.py (lines 621-626)

**Input Validation Required**:
- None (property change only)

**Tasks**:

1. **Update solver_shared_elements property in FIRKStep**
   - File: src/cubie/integrators/algorithms/generic_firk.py
   - Action: Modify
   - Line: 625
   - Details:
     ```python
     @property
     def solver_shared_elements(self) -> int:
         """Return solver scratch elements accounting for flattened stages."""
         
         return 2 * self.compile_settings.all_stages_n  # Changed from 3 to 2
     ```
   - Edge cases: Multi-stage methods (s*n elements, where s is stage count)
   - Integration: FIRKStep overrides base class for multi-stage allocation
   - Rationale: For FIRK with s stages, reduces from 3*s*n to 2*s*n elements

**Outcomes**:

---

## Task Group 3: Update Newton-Krylov Solver - SEQUENTIAL
**Status**: [ ]  
**Dependencies**: Groups 1, 2

**Required Context**:
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py (lines 50-90, 140-195)

**Input Validation Required**:
- None (internal solver changes only)

**Tasks**:

1. **Update shared memory partitioning in newton_krylov_solver_factory**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify
   - Lines: 60-75 (approximately, in factory function setup)
   - Details:
     ```python
     # Update buffer allocation comment and slicing
     # Current allocation creates 3 buffers:
     # shared_scratch = cuda.shared.array(3 * n, precision_scalar)
     # delta = shared_scratch[:n]
     # residual = shared_scratch[n:2*n]
     # eval_state = shared_scratch[2*n:3*n]
     
     # New allocation creates 2 buffers:
     shared_scratch = cuda.shared.array(2 * n, precision_scalar)
     delta = shared_scratch[:n]
     residual = shared_scratch[n:2*n]
     # Remove eval_state assignment
     ```
   - Edge cases: Buffer used in Newton iteration loop
   - Integration: Buffers passed to linear solver and residual function
   - Note: Shared memory size controlled by algorithm (Groups 1-2), factory uses passed size

2. **Remove eval_state computation loop**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Delete
   - Lines: 176-177 (within Newton iteration, before linear solver call)
   - Details:
     ```python
     # DELETE these lines:
     # for i in range(n):
     #     eval_state[i] = base_state[i % n_base] + a_ij * stage_increment[i]
     ```
   - Edge cases: None (removing entire computation)
   - Integration: Computation moved inline to operators/residuals
   - Rationale: eval_state now computed on-demand in operators

3. **Update linear solver call signature**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify
   - Line: 178 (lin_return = linear_solver(...))
   - Details:
     ```python
     # Change first parameter from eval_state to stage_increment
     lin_return = linear_solver(
         stage_increment,  # Changed from: eval_state
         parameters,
         drivers,
         base_state,
         t,
         h,
         a_ij,
         residual,
         delta,
     )
     ```
   - Edge cases: Must match updated linear_solver signature (Group 4)
   - Integration: Passes stage_increment for inline eval computation
   - Rationale: Linear solver will compute eval inline from stage_increment + base_state + a_ij

4. **Update function docstring**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify
   - Lines: 27-50 (docstring for newton_krylov_solver_factory)
   - Details:
     Update docstring to reflect that linear_solver receives stage_increment rather than pre-computed eval_state. Update any implementation notes about buffer usage (3n → 2n).
   - Edge cases: Documentation only
   - Integration: Keeps documentation in sync with implementation

**Outcomes**:

---

## Task Group 4: Update Linear Solver Signature - SEQUENTIAL
**Status**: [ ]  
**Dependencies**: Group 3

**Required Context**:
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py (lines 18-95, 125-195)
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py (lines 213-340)

**Input Validation Required**:
- None (signature change only, validation unchanged)

**Tasks**:

1. **Update linear_solver_factory function signature and calls**
   - File: src/cubie/integrators/matrix_free_solvers/linear_solver.py
   - Action: Modify
   - Lines: 79-89, 133, 162
   - Details:
     ```python
     # Line 79: Change parameter name in device function signature
     @cuda.jit(device=True)
     def linear_solver(
         stage_increment,  # Changed from: state
         parameters,
         drivers,
         base_state,
         t,
         h,
         a_ij,
         rhs,
         x,
     ):
     
     # Line 133: Update operator_apply call (first parameter)
     operator_apply(stage_increment, parameters, drivers, base_state, t, h, a_ij, x, temp)
     # Changed from: operator_apply(state, parameters, ...)
     
     # Line 162 (approx): Update second operator_apply call if present
     operator_apply(stage_increment, parameters, drivers, base_state, t, h, a_ij, preconditioned_vec, temp)
     # Changed from: operator_apply(state, parameters, ...)
     ```
   - Edge cases: Two operator_apply calls (minimal_residual mode), both need update
   - Integration: Operator will compute eval inline from stage_increment
   - Rationale: Parameter rename for clarity, operator receives raw stage_increment

2. **Update linear_solver docstring**
   - File: src/cubie/integrators/matrix_free_solvers/linear_solver.py
   - Action: Modify
   - Lines: 90-128 (docstring within linear_solver device function)
   - Details:
     ```python
     # Update parameter documentation:
     """Run one preconditioned steepest-descent or minimal-residual solve.
     
     Parameters
     ----------
     stage_increment
         Stage increment vector forwarded to the operator and preconditioner.
         Operator will compute eval_state inline as:
         eval_state[i] = base_state[i % n_base] + a_ij * stage_increment[i]
     parameters
         Model parameters forwarded to the operator and preconditioner.
     ...
     """
     ```
   - Edge cases: Documentation only
   - Integration: Clarifies new calling convention

3. **Update linear_solver_cached_factory function signature and calls**
   - File: src/cubie/integrators/matrix_free_solvers/linear_solver.py
   - Action: Modify
   - Lines: 239, 257, 290
   - Details:
     ```python
     # Line 239: Change parameter name in device function signature
     @cuda.jit(device=True)
     def linear_solver(
         stage_increment,  # Changed from: state
         parameters,
         drivers,
         cached_aux,
         base_state,
         t,
         h,
         a_ij,
         rhs,
         x,
     ):
     
     # Line 257: Update operator_apply call
     operator_apply(stage_increment, parameters, drivers, cached_aux, base_state, t, h, a_ij, x, temp)
     
     # Line 290: Update second operator_apply call if present
     operator_apply(stage_increment, parameters, drivers, cached_aux, base_state, t, h, a_ij, preconditioned_vec, temp)
     ```
   - Edge cases: Cached variant has extra cached_aux parameter
   - Integration: Same pattern as non-cached variant
   - Rationale: Consistent signature across both linear solver variants

4. **Update linear_solver_cached_factory docstring**
   - File: src/cubie/integrators/matrix_free_solvers/linear_solver.py
   - Action: Modify
   - Lines: Within linear_solver_cached_factory function
   - Details:
     Update docstring to match linear_solver_factory changes (stage_increment parameter).
   - Edge cases: Documentation only
   - Integration: Consistent documentation across variants

**Outcomes**:

---

## Task Group 5: Update Linear Operator Code Generation - SEQUENTIAL
**Status**: [ ]  
**Dependencies**: Groups 1, 2, 3, 4

**Required Context**:
- File: src/cubie/odesystems/symbolic/codegen/linear_operators.py (lines 25-86, 175-205, 467-600)
- File: src/cubie/odesystems/symbolic/codegen/nonlinear_residuals.py (lines 100-120) - for reference pattern

**Input Validation Required**:
- None (code generation templates only)

**Tasks**:

1. **Update OPERATOR_APPLY_TEMPLATE signature**
   - File: src/cubie/odesystems/symbolic/codegen/linear_operators.py
   - Action: Modify
   - Line: 83
   - Details:
     ```python
     # Change signature in template string:
     "    def operator_apply(stage_increment, parameters, drivers, base_state, t, h, a_ij, v, out):\n"
     # Changed from: def operator_apply(state, parameters, drivers, ...)
     ```
   - Edge cases: Template string modification only
   - Integration: Generated code will have new signature
   - Rationale: Matches updated linear solver calls

2. **Update CACHED_OPERATOR_APPLY_TEMPLATE signature**
   - File: src/cubie/odesystems/symbolic/codegen/linear_operators.py
   - Action: Modify
   - Lines: 52-54
   - Details:
     ```python
     # Change signature in template string:
     "    def operator_apply(\n"
     "        stage_increment, parameters, drivers, cached_aux, base_state, t, h, a_ij, v, out\n"
     "    ):\n"
     # Changed from: state, parameters, drivers, cached_aux, ...
     ```
   - Edge cases: Template string modification only
   - Integration: Cached variant matches non-cached
   - Rationale: Consistent signature across operator variants

3. **Update N_STAGE_OPERATOR_TEMPLATE signature**
   - File: src/cubie/odesystems/symbolic/codegen/linear_operators.py
   - Action: Modify
   - Line: 707
   - Details:
     ```python
     # Change signature in template string:
     "    def operator_apply(stage_increment, parameters, drivers, base_state, t, h, a_ij, v, out):\n"
     # Changed from: def operator_apply(state, parameters, drivers, ...)
     ```
   - Edge cases: Multi-stage (FIRK) operator template
   - Integration: All three templates now consistent
   - Rationale: FIRK operators use same signature

4. **Modify _build_operator_body to compute eval_state inline**
   - File: src/cubie/odesystems/symbolic/codegen/linear_operators.py
   - Action: Modify
   - Function: _build_operator_body (lines 175-205 approximately)
   - Details:
     ```python
     def _build_operator_body(
         cached_assigns: List[Tuple[sp.Symbol, sp.Expr]],
         runtime_assigns: List[Tuple[sp.Symbol, sp.Expr]],
         jvp_terms: Dict[int, sp.Expr],
         index_map: IndexedBases,
         M: sp.Matrix,
         use_cached_aux: bool = False,
         prepare_assigns: Optional[List[Tuple[sp.Symbol, sp.Expr]]] = None,
     ) -> str:
         """Build the CUDA body computing J·v with optional mass matrix and auxiliaries."""
         
         n_out = len(index_map.dxdt.ref_map)
         
         # NEW: Create symbolic references for inline eval computation
         stage_increment = sp.IndexedBase("stage_increment", shape=(sp.Integer(n_out),))
         base_state_vec = sp.IndexedBase("base_state")
         a_ij = sp.Symbol("a_ij")
         
         # NEW: Build state substitution that computes eval inline
         # For each state variable, substitute with: base_state[i % n_base] + a_ij * stage_increment[i]
         # This matches the pattern in nonlinear_residuals.py lines 115-116
         state_symbols = list(index_map.states.index_map.keys())
         n_states = len(state_symbols)
         state_subs = {}
         for i, state_sym in enumerate(state_symbols):
             # Inline eval computation (modulo handled at code generation)
             eval_point = base_state_vec[i] + a_ij * stage_increment[i]
             state_subs[state_sym] = eval_point
         
         # Apply state substitution to auxiliary assignments
         runtime_assigns_subst = [
             (lhs, rhs.subs(state_subs)) for lhs, rhs in runtime_assigns
         ]
         
         # Continue with existing mass matrix and JVP logic...
         # (rest of function unchanged, except use runtime_assigns_subst instead of runtime_assigns)
     ```
   - Edge cases:
     - Single-stage: i % n_base simplifies to i when n_base == n
     - Multi-stage: modulo correctly wraps for each stage
     - Explicit stages (a_ij=0): compiler optimizes multiplication by zero
   - Integration: State substitution applied before Jacobian evaluation
   - Rationale: Matches nonlinear_residuals.py pattern (lines 114-116)
   - Note: Symbolic computation, actual modulo arithmetic in generated CUDA code

5. **Modify _build_cached_jvp_body to compute eval_state inline**
   - File: src/cubie/odesystems/symbolic/codegen/linear_operators.py
   - Action: Modify
   - Function: _build_cached_jvp_body (lines 207-237 approximately)
   - Details:
     ```python
     def _build_cached_jvp_body(
         cached_assigns: List[Tuple[sp.Symbol, sp.Expr]],
         runtime_assigns: List[Tuple[sp.Symbol, sp.Expr]],
         jvp_terms: Dict[int, sp.Expr],
         index_map: IndexedBases,
     ) -> str:
         """Build the CUDA body computing J·v with optional cached auxiliaries."""
         
         n_out = len(index_map.dxdt.ref_map)
         
         # NEW: Create symbolic references for inline eval computation
         stage_increment = sp.IndexedBase("stage_increment", shape=(sp.Integer(n_out),))
         base_state_vec = sp.IndexedBase("base_state")
         a_ij = sp.Symbol("a_ij")
         
         # NEW: Build state substitution that computes eval inline
         state_symbols = list(index_map.states.index_map.keys())
         state_subs = {}
         for i, state_sym in enumerate(state_symbols):
             eval_point = base_state_vec[i] + a_ij * stage_increment[i]
             state_subs[state_sym] = eval_point
         
         # Apply state substitution to runtime auxiliary assignments
         runtime_assigns_subst = [
             (lhs, rhs.subs(state_subs)) for lhs, rhs in runtime_assigns
         ]
         
         # Continue with existing cached auxiliary and output logic...
         # (rest of function unchanged, except use runtime_assigns_subst)
     ```
   - Edge cases: Same as _build_operator_body
   - Integration: Cached operators compute eval inline like non-cached
   - Rationale: Consistent pattern across operator variants

6. **Update _build_n_stage_operator_lines for FIRK operators**
   - File: src/cubie/odesystems/symbolic/codegen/linear_operators.py
   - Action: Modify
   - Function: _build_n_stage_operator_lines (lines 467-600 approximately)
   - Lines: 494, 538-548
   - Details:
     ```python
     # Line 494: Rename state_vec to stage_increment for clarity
     stage_increment_vec = sp.IndexedBase("stage_increment", shape=(total_states,))
     # Changed from: state_vec = sp.IndexedBase("state", shape=(total_states,))
     
     # Lines 538-548: Update variable name in stage state substitution
     # Already computes inline (correct pattern), just rename for consistency:
     stage_state_subs = {}
     for state_idx, state_sym in enumerate(state_symbols):
         expr = base_state[state_idx]
         for contrib_idx in range(stage_count):
             coeff_value = stage_coefficients[stage_idx, contrib_idx]
             if coeff_value == 0:
                 continue
             coeff_sym = coeff_symbols[stage_idx][contrib_idx]
             expr += coeff_sym * stage_increment_vec[  # Changed from: state_vec
                 contrib_idx * state_count + state_idx
             ]
         stage_state_subs[state_sym] = expr
     
     # Update all other references to state_vec → stage_increment_vec throughout function
     ```
   - Edge cases: Multi-stage (FIRK) methods with s*n unknowns
   - Integration: Already has inline computation, just renaming for consistency
   - Rationale: Variable name should reflect that it's stage_increment, not eval state
   - Note: This function already implements inline computation correctly (lines 539-548)

7. **Update template docstrings**
   - File: src/cubie/odesystems/symbolic/codegen/linear_operators.py
   - Action: Modify
   - Lines: 29-31, 64-69, 690-692 (template docstrings)
   - Details:
     Update template docstrings to reflect new signature:
     ```python
     # CACHED_OPERATOR_APPLY_TEMPLATE docstring (lines 29-37):
     "      operator_apply(\n"
     "          stage_increment, parameters, drivers, cached_aux, base_state, t, h, a_ij, v, out\n"
     "      )\n"
     
     # OPERATOR_APPLY_TEMPLATE docstring (lines 64-69):
     "      operator_apply(stage_increment, parameters, drivers, base_state, t, h, a_ij, v, out)\n"
     
     # N_STAGE_OPERATOR_TEMPLATE docstring (lines 690-692):
     # Already mentions flattened stages, add note about inline eval computation
     ```
   - Edge cases: Documentation only
   - Integration: Keeps generated code documentation accurate

**Outcomes**:

---

## Verification Strategy

**After all groups complete:**

1. **Regenerate all ODE systems** - Generated code will be recompiled automatically due to template changes
2. **Run existing test suite**:
   - `pytest tests/integrators/matrix_free_solvers/` - Solver unit tests
   - `pytest tests/integrators/algorithms/test_generic_firk.py` - FIRK integration tests
   - `pytest tests/integrators/algorithms/test_generic_dirk.py` - DIRK integration tests
   - `pytest tests/integrators/algorithms/test_backwards_euler.py` - Backwards Euler tests
   - `pytest tests/integrators/algorithms/test_crank_nicolson.py` - Crank-Nicolson tests
3. **Check instrumented tests** - `tests/integrators/algorithms/instrumented/matrix_free_solvers.py` validates buffer usage
4. **Verify shared memory reduction** - Inspect generated code for 2n allocation vs 3n

**Expected outcomes:**
- All existing tests pass unchanged (behavior identical)
- Shared memory usage reduced by 33% (3n → 2n for DIRK, 3*s*n → 2*s*n for FIRK)
- No performance regression (inline computation is negligible cost)

---

## Notes

**Dependency Rationale:**
- Group 1 must complete first (base class sets contract)
- Group 2 depends on Group 1 (overrides base class)
- Group 3 depends on Groups 1-2 (uses updated allocation sizes)
- Group 4 depends on Group 3 (receives new calling convention from Newton-Krylov)
- Group 5 depends on all previous (generates code matching updated signatures)

**Parallel Execution:** Not recommended - changes are tightly coupled through function signatures

**Code Regeneration:** All symbolic ODE systems will regenerate on next import due to template hash changes (automatic via CUDAFactory cache invalidation)

**Backward Compatibility:** Breaking change to generated code, but no user-facing API changes. Package is in development, breaking changes are acceptable per repository guidelines.

**Performance Impact:**
- Memory: 33% reduction in solver shared memory usage
- Compute: Negligible (one additional load per element, base_state likely cached)
- Occupancy: May improve on memory-constrained GPUs

**Testing Philosophy:**
- Behavior is identical to current implementation
- All existing tests provide coverage
- No new tests needed (functional equivalence)
- Instrumented tests validate memory layout changes
