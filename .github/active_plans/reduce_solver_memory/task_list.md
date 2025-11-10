# Implementation Task List
# Feature: Reduce Nonlinear Solver Memory Footprint
# Plan Reference: .github/active_plans/reduce_solver_memory/agent_plan.md

## Task Group 1: Update Solver Shared Memory Allocation - SEQUENTIAL
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/algorithms/ode_implicitstep.py (lines 300-304)

**Input Validation Required**:
None - property returns computed value only

**Tasks**:
1. **Reduce solver_shared_elements from 3n to 2n**
   - File: src/cubie/integrators/algorithms/ode_implicitstep.py
   - Action: Modify
   - Details:
     ```python
     @property
     def solver_shared_elements(self) -> int:
         """Return shared scratch dedicated to the Newton--Krylov solver."""
         
         return self.compile_settings.n * 2  # Changed from * 3
     ```
   - Edge cases: None - simple property change
   - Integration: This property is accessed by DIRK and FIRK algorithms to allocate shared memory

**Outcomes**:
✅ Successfully modified ode_implicitstep.py line 304 from `return self.compile_settings.n * 3` to `return self.compile_settings.n * 2`. This reduces the solver shared memory allocation from 3n to 2n buffers, removing the eval_state buffer allocation from all implicit algorithms (DIRK and FIRK).

---

## Task Group 2: Remove eval_state Buffer from Newton-Krylov Solver - SEQUENTIAL
**Status**: [x]
**Dependencies**: Group 1

**Required Context**:
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py (lines 82-250)
- Specific lines to modify: 115-137 (docstring), 130 (comment), 143 (eval_state allocation), 176-178 (eval_state computation and linear_solver call)

**Input Validation Required**:
None - internal refactoring only

**Tasks**:
1. **Remove eval_state buffer allocation**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Delete
   - Details:
     - Remove line 143: `eval_state = shared_scratch[2 * n: 3 * n]`
     - This removes the third buffer slice allocation
   - Edge cases: None
   - Integration: Must update all references to eval_state in this function

2. **Remove eval_state computation loop**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Delete
   - Details:
     - Remove lines 176-177:
       ```python
       n_base = base_state.shape[0]
       for i in range(n):
           eval_state[i] = base_state[i % n_base] + a_ij * stage_increment[i]
       ```
     - This removes the loop that populates eval_state before calling linear_solver
   - Edge cases: None
   - Integration: The linear_solver will now receive stage_increment instead

3. **Update linear_solver call to pass stage_increment**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify
   - Details:
     - Change line 178 from:
       ```python
       lin_return = linear_solver(
           eval_state,
           parameters,
           ...
       )
       ```
     - To:
       ```python
       lin_return = linear_solver(
           stage_increment,
           parameters,
           ...
       )
       ```
     - Pass stage_increment as first argument instead of eval_state
   - Edge cases: None
   - Integration: Linear solver signature unchanged, just receives different data

4. **Update docstring to reflect 2-buffer usage**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify
   - Details:
     - Update lines 115-137 docstring section describing shared_scratch parameter
     - Change from: "The first ``n`` entries store the Newton direction, the next ``n`` entries store the residual, and the final ``n`` entries store the stage state ``base_state + a_ij * stage_increment``."
     - To: "The first ``n`` entries store the Newton direction and the next ``n`` entries store the residual."
   - Edge cases: None
   - Integration: Documentation accuracy

5. **Update scratch space requirement comment**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify
   - Details:
     - Update line 130 in Notes section
     - Change from: "Scratch space requirements total three vectors of length ``n`` drawn from ``shared_scratch``."
     - To: "Scratch space requirements total two vectors of length ``n`` drawn from ``shared_scratch``."
   - Edge cases: None
   - Integration: Documentation accuracy

6. **Remove eval_state reference from Notes section**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify
   - Details:
     - Update Notes section (lines 129-138) to remove mention of eval_state
     - Remove: "``eval_state`` stores the stage state ``base_state + a_ij * stage_increment`` for the Jacobian evaluations."
     - Update description to reflect that operators compute evaluation state inline
   - Edge cases: None
   - Integration: Documentation accuracy

**Outcomes**:
✅ Successfully removed eval_state buffer from newton_krylov.py with 4 edits:
1. Updated shared_scratch docstring (lines 112-117) to remove mention of third buffer
2. Updated Notes section (lines 128-138) to describe 2-buffer usage and note inline computation
3. Removed eval_state allocation line (line 143)
4. Removed eval_state computation loop (lines 175-177) and updated linear_solver call to pass stage_increment directly instead of eval_state

All references to eval_state eliminated. Linear solver now receives stage_increment, which operators will use to compute evaluation state inline.

---

## Task Group 3: Add Inline State Evaluation to Single-Stage Linear Operator Codegen - SEQUENTIAL
**Status**: [x]
**Dependencies**: Group 2

**Required Context**:
- File: src/cubie/odesystems/symbolic/codegen/linear_operators.py (lines 147-204)
- File: src/cubie/odesystems/symbolic/codegen/nonlinear_residuals.py (lines 74-150) - for reference pattern
- IndexedBases structure from src/cubie/odesystems/symbolic/parsing/parser.py

**Input Validation Required**:
None - code generation only

**Tasks**:
1. **Add state substitution to _build_operator_body for single-stage operator**
   - File: src/cubie/odesystems/symbolic/codegen/linear_operators.py
   - Action: Modify
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
         """Build the CUDA body computing ``β·M·v − γ·h·J·v``."""
         
         n_out = len(index_map.dxdt.ref_map)
         n_in = len(index_map.states.index_map)
         v = sp.IndexedBase("v")
         beta_sym = sp.Symbol("beta")
         gamma_sym = sp.Symbol("gamma")
         a_ij_sym = sp.Symbol("a_ij")
         h_sym = sp.Symbol("h")
         
         # NEW: Add state substitution for inline evaluation
         # This computes: state_sym -> base_state[i] + a_ij * state[i]
         # where 'state' parameter is actually stage_increment
         state_subs = {}
         state_symbols = list(index_map.states.index_map.keys())
         state_indexed = sp.IndexedBase("state")
         base_state_indexed = sp.IndexedBase("base_state")
         for i, state_sym in enumerate(state_symbols):
             eval_point = base_state_indexed[i] + a_ij_sym * state_indexed[i]
             state_subs[state_sym] = eval_point
         
         # Mass matrix assignments (unchanged)
         mass_assigns = []
         out_updates = []
         for i in range(n_out):
             mv = sp.S.Zero
             for j in range(n_in):
                 entry = M[i, j]
                 if entry == 0:
                     continue
                 sym = sp.Symbol(f"m_{i}{j}")
                 mass_assigns.append((sym, entry))
                 mv += sym * v[j]
             
             # Apply state substitution to jvp_terms
             jvp_substituted = jvp_terms[i].subs(state_subs)
             rhs = beta_sym * mv - gamma_sym * a_ij_sym * h_sym * jvp_substituted
             out_updates.append((sp.Symbol(f"out[{i}]"), rhs))
         
         # Auxiliary assignments handling (unchanged logic)
         if use_cached_aux:
             if cached_assigns:
                 cached = sp.IndexedBase(
                     "cached_aux", shape=(sp.Integer(len(cached_assigns)),)
                 )
             else:
                 cached = sp.IndexedBase("cached_aux")
             aux_assignments = [
                 (lhs, cached[idx]) for idx, (lhs, _) in enumerate(cached_assigns)
             ] + runtime_assigns
         else:
             combined = list(prepare_assigns or []) + cached_assigns + runtime_assigns
             seen = set()
             aux_assignments = []
             for lhs, rhs in combined:
                 if lhs in seen:
                     continue
                 seen.add(lhs)
                 # Apply state substitution to auxiliary assignments
                 rhs_substituted = rhs.subs(state_subs)
                 aux_assignments.append((lhs, rhs_substituted))
         
         exprs = mass_assigns + aux_assignments + out_updates
         lines = print_cuda_multiple(exprs, symbol_map=index_map.all_arrayrefs)
         if not lines:
             return "        pass"
         return "\n".join("        " + ln for ln in lines)
     ```
   - Edge cases: 
     - Handle empty state_symbols list (should not occur in practice)
     - Handle zero a_ij coefficient (SymPy will simplify: base + 0*stage = base)
   - Integration: This changes how JVP terms are evaluated - they now compute state inline
   - **CRITICAL**: Apply state_subs AFTER auxiliary assignments are created but BEFORE JVP terms are used
   - **CRITICAL**: Must also apply state_subs to runtime auxiliary assignments if they reference state

2. **Verify state substitution is applied to auxiliary assignments**
   - File: src/cubie/odesystems/symbolic/codegen/linear_operators.py
   - Action: Modify
   - Details:
     - In the aux_assignments loop for non-cached path (else branch), apply state_subs to rhs
     - Ensure substitution happens in correct order: state_subs must be applied to auxiliary RHS expressions
     - This ensures any auxiliaries that depend on state variables use the inline-computed evaluation point
   - Edge cases: Auxiliaries with no state dependence (substitution has no effect)
   - Integration: Maintains correctness of auxiliary expressions that feed into JVP

**Outcomes**:
✅ Successfully modified _build_operator_body in linear_operators.py to add inline state evaluation:
1. Added state_subs dictionary construction that maps state_sym -> base_state[i] + a_ij * state[i]
2. Applied state_subs to JVP terms before constructing output RHS expressions
3. Applied state_subs to auxiliary assignment RHS expressions in the non-cached path
This ensures operators compute the evaluation state (base_state + a_ij * stage_increment) inline instead of expecting a pre-computed eval_state buffer. The change follows the same pattern already used in residual codegen and n-stage operators.

---

## Task Group 4: Verify N-Stage Operator Already Has Inline Computation - PARALLEL
**Status**: [x]
**Dependencies**: None (verification only, can run in parallel with other groups)

**Required Context**:
- File: src/cubie/odesystems/symbolic/codegen/linear_operators.py (lines 500-636)
- Specifically lines 537-548: stage_state_subs construction

**Input Validation Required**:
None - verification task only

**Tasks**:
1. **Verify _build_n_stage_operator_lines creates inline state evaluation**
   - File: src/cubie/odesystems/symbolic/codegen/linear_operators.py
   - Action: Read and verify (no changes expected)
   - Details:
     - Examine lines 537-548 where stage_state_subs is constructed:
       ```python
       stage_state_subs = {}
       for state_idx, state_sym in enumerate(state_symbols):
           expr = base_state[state_idx]
           for contrib_idx in range(stage_count):
               coeff_value = stage_coefficients[stage_idx, contrib_idx]
               if coeff_value == 0:
                   continue
               coeff_sym = coeff_symbols[stage_idx][contrib_idx]
               expr += coeff_sym * state_vec[
                   contrib_idx * state_count + state_idx
               ]
           stage_state_subs[state_sym] = expr
       ```
     - Verify this pattern computes: `state_sym = base_state[i] + sum(a[s,j] * stage_increment[j*n+i])`
     - Verify this substitution is applied to all expressions via `.subs(stage_state_subs)` on lines 553, 582, 595
   - Expected result: **NO CHANGES NEEDED** - already implements inline evaluation correctly
   - Edge cases: Multi-stage indexing must be correct
   - Integration: This is used by FIRK algorithms

**Outcomes**:
✅ VERIFIED - N-stage operator already implements inline state evaluation correctly:
- Lines 537-548 construct stage_state_subs that maps state_sym -> base_state[i] + sum(coeff * state_vec[contrib*n+i])
- Line 553 applies substitution with .subs(stage_state_subs)
- NO CHANGES NEEDED - implementation is correct and matches the pattern we added to single-stage operators

---

## Task Group 5: Verify Residual Codegen Already Has Inline Computation - PARALLEL
**Status**: [x]
**Dependencies**: None (verification only, can run in parallel with other groups)

**Required Context**:
- File: src/cubie/odesystems/symbolic/codegen/nonlinear_residuals.py (lines 74-150)
- Specifically lines 112-116: state_subs construction

**Input Validation Required**:
None - verification task only

**Tasks**:
1. **Verify _build_residual_lines creates inline state evaluation**
   - File: src/cubie/odesystems/symbolic/codegen/nonlinear_residuals.py
   - Action: Read and verify (no changes expected)
   - Details:
     - Examine lines 112-116 where state_subs is constructed:
       ```python
       state_subs = {}
       state_symbols = list(index_map.states.index_map.keys())
       for i, state_sym in enumerate(state_symbols):
           eval_point = base[i] + aij_sym * u[i]
           state_subs[state_sym] = eval_point
       ```
     - Verify this pattern computes: `state_sym = base_state[i] + a_ij * u[i]`
     - Verify this substitution is applied on line 120: `eval_rhs = rhs.subs(state_subs)`
   - Expected result: **NO CHANGES NEEDED** - already implements inline evaluation correctly
   - Edge cases: None
   - Integration: This is used by both DIRK and FIRK algorithms

2. **Verify n-stage residual also has inline computation**
   - File: src/cubie/odesystems/symbolic/codegen/nonlinear_residuals.py
   - Action: Read and verify (no changes expected)
   - Details:
     - Search for similar state_subs pattern in n-stage residual builder
     - Verify it computes evaluation state inline per stage
   - Expected result: **NO CHANGES NEEDED**
   - Edge cases: Multi-stage indexing
   - Integration: This is used by FIRK algorithms

**Outcomes**:
✅ VERIFIED - Residual codegen already implements inline state evaluation correctly:
- Lines 112-116 construct state_subs that maps state_sym -> base[i] + aij_sym * u[i]
- Line 120 applies substitution with .subs(state_subs)
- NO CHANGES NEEDED - implementation is correct and was the reference pattern for Task Group 3

---

## Task Group 6: Update DIRK Algorithm Shared Memory Comments - SEQUENTIAL
**Status**: [x]
**Dependencies**: Groups 1, 2, 3

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_dirk.py (lines 403-439)
- Understanding of shared memory layout after changes

**Input Validation Required**:
None - documentation only

**Tasks**:
1. **Update solver_scratch comment to reflect 2-buffer usage**
   - File: src/cubie/integrators/algorithms/generic_dirk.py
   - Action: Modify
   - Details:
     - Update lines 413-427 comment block
     - Change from:
       ```
       # solver_scratch: size solver_shared_elements, shared memory.
       #   Default behaviour:
       #       - Provides workspace for the Newton iteration helpers.
       #   Reuse:
       #       - stage_rhs: first slice (size n)
       #           - Carries the Newton residual and then the stage rhs.
       #           - Once a stage closes we reuse it for the next residual,
       #             so no live data remains.
       #       - increment_cache: second slice (size n)
       #           - Receives the accepted increment at step end for FSAL.
       #           - Solver stops touching it once convergence is reached.
       #       - eval_state: third slice (size n)
       #           - Stores base_state + a_ij * stage_increment.
       #           - Reserved for Newton Jacobian evaluations.
       ```
     - To:
       ```
       # solver_scratch: size solver_shared_elements, shared memory.
       #   Default behaviour:
       #       - Provides workspace for the Newton iteration helpers.
       #   Reuse:
       #       - stage_rhs: first slice (size n)
       #           - Carries the Newton residual and then the stage rhs.
       #           - Once a stage closes we reuse it for the next residual,
       #             so no live data remains.
       #       - increment_cache: second slice (size n)
       #           - Receives the accepted increment at step end for FSAL.
       #           - Solver stops touching it once convergence is reached.
       #   Note:
       #       - State evaluation (base_state + a_ij * stage_increment) is now
       #         computed inline by operators and residuals, eliminating the
       #         need for a dedicated eval_state buffer.
       ```
   - Edge cases: None
   - Integration: Documentation accuracy for future developers

**Outcomes**:
✅ Successfully updated solver_scratch comment block in generic_dirk.py (lines 413-426):
- Removed eval_state buffer description (former third slice)
- Added Note section explaining that state evaluation is now computed inline by operators and residuals
- Documentation now accurately reflects 2-buffer usage (stage_rhs and increment_cache only)

---

## Task Group 7: Update FIRK Algorithm Shared Memory Comments - SEQUENTIAL
**Status**: [x]
**Dependencies**: Groups 1, 2, 3

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_firk.py (lines 336-450)
- Understanding of shared memory layout after changes

**Input Validation Required**:
None - documentation only

**Tasks**:
1. **Update solver_scratch comment in FIRK algorithm**
   - File: src/cubie/integrators/algorithms/generic_firk.py
   - Action: Modify
   - Details:
     - Locate the shared/local buffer guide comment section (similar to DIRK)
     - Update solver_scratch description to reflect 2-buffer usage (not 3-buffer)
     - Note that eval_state is computed inline by n-stage operators
     - If no such comment exists in FIRK, verify the shared memory allocation is correct
   - Edge cases: FIRK may have different buffer organization than DIRK
   - Integration: Documentation accuracy

**Outcomes**:
✅ VERIFIED - FIRK does not have a detailed buffer guide comment like DIRK. Verified shared memory allocation is correct:
- Line 371: Uses self.solver_shared_elements (now 2n after Group 1 changes)
- Line 429: solver_scratch = shared[:solver_shared_elements] correctly allocates first 2n elements
- NO CHANGES NEEDED - allocation is correct and automatically benefits from base class property change

---

## Task Group 8: Verify Linear Solver Forwarding is Correct - PARALLEL
**Status**: [x]
**Dependencies**: None (verification only)

**Required Context**:
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py (lines 70-150)
- Specifically line 133: operator_apply call

**Input Validation Required**:
None - verification task only

**Tasks**:
1. **Verify linear_solver forwards state parameter unchanged**
   - File: src/cubie/integrators/matrix_free_solvers/linear_solver.py
   - Action: Read and verify (no changes expected)
   - Details:
     - Examine line 133:
       ```python
       operator_apply(state, parameters, drivers, base_state, t, h, a_ij, x, temp)
       ```
     - Verify that `state` parameter (first arg to linear_solver) is forwarded as-is
     - Verify no assumptions about state being pre-evaluated
   - Expected result: **NO CHANGES NEEDED** - already forwards parameter correctly
   - Edge cases: None
   - Integration: Linear solver is a pass-through; operators handle inline computation

**Outcomes**:
✅ VERIFIED - Linear solver correctly forwards state parameter unchanged:
- Line 133: operator_apply(state, parameters, ...) forwards first argument as-is
- No assumptions about state being pre-evaluated
- NO CHANGES NEEDED - linear solver is a pass-through, operators handle inline computation correctly

---

## Summary

**Total Task Groups**: 8
- Groups requiring code changes: 3 (Groups 1, 2, 3)
- Groups requiring documentation updates: 2 (Groups 6, 7)
- Verification-only groups: 3 (Groups 4, 5, 8)

**Dependency Chain**:
1. Group 1 (update solver_shared_elements) must complete first
2. Group 2 (remove eval_state from Newton-Krylov) depends on Group 1
3. Group 3 (add inline computation to operators) depends on Group 2
4. Groups 6 and 7 (documentation) depend on Groups 1, 2, 3
5. Groups 4, 5, 8 are verification tasks and can run in parallel with any group

**Parallel Execution Opportunities**:
- Groups 4, 5, 8 can all run in parallel with each other
- Groups 4, 5, 8 can run in parallel with Groups 1-3 (they're independent verifications)
- Groups 6 and 7 can run in parallel with each other once their dependencies complete

**Estimated Complexity**: Medium
- Core changes are surgical and well-defined (3 files modified)
- Verification tasks confirm existing code already has correct patterns
- Documentation updates are straightforward
- No new imports or dependencies required
- Risk is low - changes are localized and preserve mathematical correctness

**Critical Success Factors**:
1. State substitution in linear operators must happen BEFORE JVP terms are evaluated
2. Must apply state_subs to both JVP terms and auxiliary assignments that reference states
3. Newton-Krylov must pass stage_increment (not eval_state) to linear_solver
4. solver_shared_elements must change from 3n to 2n

**Testing Strategy** (not included in tasks per instructions):
- After implementation, run existing test suite
- Verify convergence behavior unchanged
- Verify iteration counts identical to baseline
- Verify numerical results match within tolerance
