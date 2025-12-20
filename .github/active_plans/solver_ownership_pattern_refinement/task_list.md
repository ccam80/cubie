# Implementation Task List
# Feature: Solver Ownership Pattern Refinement
# Plan Reference: .github/active_plans/solver_ownership_pattern_refinement/agent_plan.md

## Task Group 1: Add Pass-through Properties to NewtonKrylov - SEQUENTIAL
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py (lines 540-578)
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py (lines 560-630)

**Input Validation Required**:
None - these are read-only properties returning values from nested LinearSolver

**Tasks**:
1. **Add krylov_tolerance pass-through property**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Create
   - Location: After line 573 (after newton_max_backtracks property)
   - Details:
     ```python
     @property
     def krylov_tolerance(self) -> float:
         """Return krylov tolerance from nested linear solver."""
         return self.linear_solver.krylov_tolerance
     ```
   - Edge cases: None - linear_solver is always present in NewtonKrylov
   - Integration: Allows ODEImplicitStep to access uniformly via self.solver.krylov_tolerance

2. **Add max_linear_iters pass-through property**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Create
   - Location: After krylov_tolerance property (newly created)
   - Details:
     ```python
     @property
     def max_linear_iters(self) -> int:
         """Return max linear iterations from nested linear solver."""
         return self.linear_solver.max_linear_iters
     ```
   - Edge cases: LinearSolver may store as np.integer, but its property already returns int
   - Integration: Allows ODEImplicitStep to access uniformly via self.solver.max_linear_iters

3. **Add correction_type pass-through property**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Create
   - Location: After max_linear_iters property (newly created)
   - Details:
     ```python
     @property
     def correction_type(self) -> str:
         """Return correction type from nested linear solver."""
         return self.linear_solver.correction_type
     ```
   - Edge cases: None - correction_type is always a string
   - Integration: Allows ODEImplicitStep to access uniformly via self.solver.correction_type

**Outcomes**:
- Files Modified:
  * src/cubie/integrators/matrix_free_solvers/newton_krylov.py (15 lines added)
- Functions/Methods Added/Modified:
  * krylov_tolerance property in NewtonKrylov class
  * max_linear_iters property in NewtonKrylov class
  * correction_type property in NewtonKrylov class
- Implementation Summary:
  Added three pass-through properties that delegate to nested linear_solver, enabling uniform property access pattern for ODEImplicitStep
- Issues Flagged: None

---

## Task Group 2: Update LinearSolver Constructor Pattern - SEQUENTIAL
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py (lines 160-177)
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py (lines 204-226) - reference pattern

**Input Validation Required**:
None - validation handled by LinearSolverConfig validators

**Tasks**:
1. **Replace ternary operators with conditional kwargs pattern**
   - File: src/cubie/integrators/matrix_free_solvers/linear_solver.py
   - Action: Modify
   - Location: Lines 166-175 (config instantiation)
   - Details:
     Replace:
     ```python
     config = LinearSolverConfig(
         precision=precision,
         n=n,
         correction_type=correction_type if correction_type is not None else "minimal_residual",
         krylov_tolerance=krylov_tolerance if krylov_tolerance is not None else 1e-6,
         max_linear_iters=max_linear_iters if max_linear_iters is not None else 100,
         preconditioned_vec_location=preconditioned_vec_location,
         temp_location=temp_location,
     )
     ```
     With:
     ```python
     linear_kwargs = {}
     if correction_type is not None:
         linear_kwargs['correction_type'] = correction_type
     if krylov_tolerance is not None:
         linear_kwargs['krylov_tolerance'] = krylov_tolerance
     if max_linear_iters is not None:
         linear_kwargs['max_linear_iters'] = max_linear_iters
     
     config = LinearSolverConfig(
         precision=precision,
         n=n,
         preconditioned_vec_location=preconditioned_vec_location,
         temp_location=temp_location,
         **linear_kwargs
     )
     ```
   - Edge cases: All three parameters (correction_type, krylov_tolerance, max_linear_iters) are optional; LinearSolverConfig provides defaults
   - Integration: Matches pattern in NewtonKrylov.__init__; centralizes default management in config class

**Outcomes**:
- Files Modified:
  * src/cubie/integrators/matrix_free_solvers/linear_solver.py (9 lines changed in __init__)
- Functions/Methods Added/Modified:
  * LinearSolver.__init__ method
- Implementation Summary:
  Replaced ternary operators with conditional kwargs pattern, matching NewtonKrylov pattern
- Issues Flagged: None

---

## Task Group 3: Update LinearSolver.update Method Pattern - SEQUENTIAL
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py (lines 529-558)
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py (lines 518-538) - reference pattern
- File: src/cubie/integrators/algorithms/ode_implicitstep.py (lines 179-199) - reference pattern

**Input Validation Required**:
None - update_compile_settings handles parameter filtering and validation internally

**Tasks**:
1. **Simplify update method to match delegation pattern**
   - File: src/cubie/integrators/matrix_free_solvers/linear_solver.py
   - Action: Modify
   - Location: Lines 529-558 (entire update method body)
   - Details:
     Replace:
     ```python
     # Merge updates
     all_updates = {}
     if updates_dict:
         all_updates.update(updates_dict)
     all_updates.update(kwargs)
     
     if not all_updates:
         return set()
     
     # Extract linear solver parameters
     linear_keys = {
         'krylov_tolerance', 'max_linear_iters',
         'linear_correction_type', 'correction_type',
         'operator_apply', 'preconditioner',
         'use_cached_auxiliaries'
     }
     linear_params = {k: all_updates[k] for k in linear_keys & all_updates.keys()}
     
     recognized = set()
     
     # Update buffer registry with full dict (extracts buffer location params)
     buffer_registry.update(self, updates_dict=all_updates, silent=True)
     
     # Update compile settings with recognized params only
     if linear_params:
         recognized = self.update_compile_settings(
             updates_dict=linear_params, silent=silent
         )
     
     return recognized
     ```
     With:
     ```python
     # Merge updates
     all_updates = {}
     if updates_dict:
         all_updates.update(updates_dict)
     all_updates.update(kwargs)
     
     if not all_updates:
         return set()
     
     recognized = set()
     # No delegation to child solvers (LinearSolver has no children)
     # No device function update (LinearSolver has no child device functions)
     
     recognized |= self.update_compile_settings(updates_dict=all_updates, silent=True)
     
     # Buffer locations will trigger cache invalidation in compile settings
     buffer_registry.update(self, updates_dict=all_updates, silent=True)
     
     return recognized
     ```
   - Edge cases: 
     - Empty all_updates returns early with empty set
     - update_compile_settings filters parameters internally, no need for pre-filtering
     - buffer_registry.update doesn't contribute to recognized set
   - Integration: 
     - Matches pattern in NewtonKrylov.update and ODEImplicitStep.update
     - Preserves cache invalidation behavior
     - Delegates parameter filtering to update_compile_settings

**Outcomes**:
- Files Modified:
  * src/cubie/integrators/matrix_free_solvers/linear_solver.py (14 lines simplified in update method)
- Functions/Methods Added/Modified:
  * LinearSolver.update method
- Implementation Summary:
  Simplified update logic to match delegation pattern, removed pre-filtering, pass full dict to update_compile_settings
- Issues Flagged: None

---

## Task Group 4: Simplify ODEImplicitStep Linear Properties - SEQUENTIAL
**Status**: [x]
**Dependencies**: Task Group 1 (pass-through properties must exist)

**Required Context**:
- File: src/cubie/integrators/algorithms/ode_implicitstep.py (lines 348-369)
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py (lines added in Task Group 1)
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py (lines 560-630)

**Input Validation Required**:
None - properties are read-only accessors

**Tasks**:
1. **Simplify krylov_tolerance property**
   - File: src/cubie/integrators/algorithms/ode_implicitstep.py
   - Action: Modify
   - Location: Lines 348-353
   - Details:
     Replace:
     ```python
     @property
     def krylov_tolerance(self) -> float:
         """Return the tolerance used for the linear solve."""
         if hasattr(self.solver, 'krylov_tolerance'):
             return self.solver.krylov_tolerance
         # For NewtonKrylov, forward to nested linear_solver
         return self.solver.linear_solver.krylov_tolerance
     ```
     With:
     ```python
     @property
     def krylov_tolerance(self) -> float:
         """Return the tolerance used for the linear solve."""
         return self.solver.krylov_tolerance
     ```
   - Edge cases: Works for both LinearSolver (direct property) and NewtonKrylov (pass-through property)
   - Integration: Relies on NewtonKrylov.krylov_tolerance pass-through property added in Task Group 1

2. **Simplify max_linear_iters property**
   - File: src/cubie/integrators/algorithms/ode_implicitstep.py
   - Action: Modify
   - Location: Lines 355-361
   - Details:
     Replace:
     ```python
     @property
     def max_linear_iters(self) -> int:
         """Return the maximum number of linear iterations allowed."""
         if hasattr(self.solver, 'max_linear_iters'):
             return int(self.solver.max_linear_iters)
         # For NewtonKrylov, forward to nested linear_solver
         return int(self.solver.linear_solver.max_linear_iters)
     ```
     With:
     ```python
     @property
     def max_linear_iters(self) -> int:
         """Return the maximum number of linear iterations allowed."""
         return int(self.solver.max_linear_iters)
     ```
   - Edge cases: int() cast handles np.integer types from either solver
   - Integration: Relies on NewtonKrylov.max_linear_iters pass-through property added in Task Group 1

3. **Simplify linear_correction_type property**
   - File: src/cubie/integrators/algorithms/ode_implicitstep.py
   - Action: Modify
   - Location: Lines 363-369
   - Details:
     Replace:
     ```python
     @property
     def linear_correction_type(self) -> str:
         """Return the linear correction strategy identifier."""
         if hasattr(self.solver, 'correction_type'):
             return self.solver.correction_type
         # For NewtonKrylov, forward to nested linear_solver
         return self.solver.linear_solver.correction_type
     ```
     With:
     ```python
     @property
     def linear_correction_type(self) -> str:
         """Return the linear correction strategy identifier."""
         return self.solver.correction_type
     ```
   - Edge cases: None - both solvers return string
   - Integration: Relies on NewtonKrylov.correction_type pass-through property added in Task Group 1

**Outcomes**:
- Files Modified:
  * src/cubie/integrators/algorithms/ode_implicitstep.py (12 lines simplified)
- Functions/Methods Added/Modified:
  * krylov_tolerance property
  * max_linear_iters property
  * linear_correction_type property
- Implementation Summary:
  Simplified properties to direct access via self.solver.[property], leveraging pass-through properties from Task Group 1
- Issues Flagged: None

---

## Task Group 5: Update ODEImplicitStep Nonlinear Properties - SEQUENTIAL
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/algorithms/ode_implicitstep.py (lines 371-405)
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py (lines 556-573)

**Input Validation Required**:
None - properties are read-only accessors

**Tasks**:
1. **Update newton_tolerance property to use getattr**
   - File: src/cubie/integrators/algorithms/ode_implicitstep.py
   - Action: Modify
   - Location: Lines 371-378
   - Details:
     Replace:
     ```python
     @property
     def newton_tolerance(self) -> float:
         """Return the Newton solve tolerance."""
         if hasattr(self.solver, 'newton_tolerance'):
             return self.solver.newton_tolerance
         raise AttributeError(
             f"{type(self.solver).__name__} does not have newton_tolerance"
         )
     ```
     With:
     ```python
     @property
     def newton_tolerance(self) -> Optional[float]:
         """Return the Newton solve tolerance."""
         return getattr(self.solver, 'newton_tolerance', None)
     ```
   - Edge cases: Returns None when solver is LinearSolver (no newton_tolerance attribute)
   - Integration: Cleaner access pattern; callers should check for None if needed

2. **Update max_newton_iters property to use getattr**
   - File: src/cubie/integrators/algorithms/ode_implicitstep.py
   - Action: Modify
   - Location: Lines 380-387
   - Details:
     Replace:
     ```python
     @property
     def max_newton_iters(self) -> int:
         """Return the maximum allowed Newton iterations."""
         if hasattr(self.solver, 'max_newton_iters'):
             return int(self.solver.max_newton_iters)
         raise AttributeError(
             f"{type(self.solver).__name__} does not have max_newton_iters"
         )
     ```
     With:
     ```python
     @property
     def max_newton_iters(self) -> Optional[int]:
         """Return the maximum allowed Newton iterations."""
         val = getattr(self.solver, 'max_newton_iters', None)
         return int(val) if val is not None else None
     ```
   - Edge cases: Returns None when solver is LinearSolver; handles np.integer types
   - Integration: Cleaner access pattern; callers should check for None if needed

3. **Update newton_damping property to use getattr**
   - File: src/cubie/integrators/algorithms/ode_implicitstep.py
   - Action: Modify
   - Location: Lines 389-396
   - Details:
     Replace:
     ```python
     @property
     def newton_damping(self) -> float:
         """Return the Newton damping factor."""
         if hasattr(self.solver, 'newton_damping'):
             return self.solver.newton_damping
         raise AttributeError(
             f"{type(self.solver).__name__} does not have newton_damping"
         )
     ```
     With:
     ```python
     @property
     def newton_damping(self) -> Optional[float]:
         """Return the Newton damping factor."""
         return getattr(self.solver, 'newton_damping', None)
     ```
   - Edge cases: Returns None when solver is LinearSolver
   - Integration: Cleaner access pattern; callers should check for None if needed

4. **Update newton_max_backtracks property to use getattr**
   - File: src/cubie/integrators/algorithms/ode_implicitstep.py
   - Action: Modify
   - Location: Lines 398-405
   - Details:
     Replace:
     ```python
     @property
     def newton_max_backtracks(self) -> int:
         """Return the maximum number of Newton backtracking steps."""
         if hasattr(self.solver, 'newton_max_backtracks'):
             return int(self.solver.newton_max_backtracks)
         raise AttributeError(
             f"{type(self.solver).__name__} does not have newton_max_backtracks"
         )
     ```
     With:
     ```python
     @property
     def newton_max_backtracks(self) -> Optional[int]:
         """Return the maximum number of Newton backtracking steps."""
         val = getattr(self.solver, 'newton_max_backtracks', None)
         return int(val) if val is not None else None
     ```
   - Edge cases: Returns None when solver is LinearSolver; handles np.integer types
   - Integration: Cleaner access pattern; callers should check for None if needed

5. **Add Optional import if not present**
   - File: src/cubie/integrators/algorithms/ode_implicitstep.py
   - Action: Modify
   - Location: Top of file (imports section)
   - Details:
     Ensure `Optional` is imported from typing:
     ```python
     from typing import Callable, Optional, Set
     ```
     If Optional is already imported, no change needed.
   - Edge cases: Check existing imports first to avoid duplicate
   - Integration: Required for Optional[float] and Optional[int] type hints

**Outcomes**:
- Files Modified:
  * src/cubie/integrators/algorithms/ode_implicitstep.py (20 lines simplified)
- Functions/Methods Added/Modified:
  * newton_tolerance property
  * max_newton_iters property
  * newton_damping property
  * newton_max_backtracks property
- Implementation Summary:
  Replaced hasattr/raise pattern with getattr(..., None) pattern for cleaner access. Optional already imported.
- Issues Flagged: None

---

## Task Group 6: Update Instrumented LinearSolver - SEQUENTIAL
**Status**: [x]
**Dependencies**: Task Groups 2, 3 (source changes must be complete)

**Required Context**:
- File: tests/integrators/algorithms/instrumented/matrix_free_solvers.py (lines 200-250, 600-700)
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py (modified in Task Groups 2, 3)

**Input Validation Required**:
None - mirror source file changes

**Tasks**:
1. **Update instrumented LinearSolver.__init__ pattern**
   - File: tests/integrators/algorithms/instrumented/matrix_free_solvers.py
   - Action: Modify
   - Location: LinearSolver.__init__ method (find and update config instantiation)
   - Details:
     Apply same changes as Task Group 2, Task 1:
     - Replace ternary operators with conditional kwargs pattern
     - Exact same code structure as source file
     - Preserve any additional logging parameters unique to instrumented version
   - Edge cases: Instrumented version may have extra parameters for logging; preserve those
   - Integration: Keep signatures synchronized with source for test compatibility

2. **Update instrumented LinearSolver.update pattern**
   - File: tests/integrators/algorithms/instrumented/matrix_free_solvers.py
   - Action: Modify
   - Location: LinearSolver.update method (find and update body)
   - Details:
     Apply same changes as Task Group 3, Task 1:
     - Simplify to match delegation pattern
     - Pass full dict to update_compile_settings
     - Call buffer_registry.update last
     - Exact same code structure as source file
     - Preserve any additional logging unique to instrumented version
   - Edge cases: Instrumented version may have logging arrays; preserve those
   - Integration: Keep logic synchronized with source for test compatibility

**Outcomes**:
- Files Modified: None
- Functions/Methods Added/Modified: None
- Implementation Summary:
  Verified that InstrumentedLinearSolver does not override __init__ logic (takes config and passes to super) or update() method (inherits from parent). Changes from Task Groups 2 and 3 automatically apply through inheritance. No code changes needed.
- Issues Flagged: None

---

## Task Group 7: Update Instrumented NewtonKrylov - SEQUENTIAL
**Status**: [x]
**Dependencies**: Task Group 1 (source changes must be complete)

**Required Context**:
- File: tests/integrators/algorithms/instrumented/matrix_free_solvers.py (NewtonKrylov class)
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py (modified in Task Group 1)

**Input Validation Required**:
None - mirror source file changes

**Tasks**:
1. **Add instrumented NewtonKrylov pass-through properties**
   - File: tests/integrators/algorithms/instrumented/matrix_free_solvers.py
   - Action: Modify
   - Location: NewtonKrylov class properties section
   - Details:
     Apply same changes as Task Group 1, Tasks 1-3:
     - Add krylov_tolerance property
     - Add max_linear_iters property
     - Add correction_type property
     - Exact same code as source file
     - Insert at same relative location (after existing properties)
   - Edge cases: None - simple property pass-through
   - Integration: Keep signatures synchronized with source for test compatibility

**Outcomes**:
- Files Modified: None
- Functions/Methods Added/Modified: None
- Implementation Summary:
  Verified that InstrumentedNewtonKrylov does not override properties (inherits from parent). Changes from Task Group 1 (pass-through properties) automatically apply through inheritance. No code changes needed.
- Issues Flagged: None

---

## Summary

**Total Task Groups**: 7

**Dependency Chain**:
```
Group 1 (NewtonKrylov properties) ────┐
                                       ├──> Group 4 (ODEImplicitStep linear properties)
Group 2 (LinearSolver.__init__) ──────┼──> Group 6 (Instrumented LinearSolver)
Group 3 (LinearSolver.update) ────────┘
Group 5 (ODEImplicitStep nonlinear properties) [Independent]
Group 1 ──────────────────────────────────────> Group 7 (Instrumented NewtonKrylov)
```

**Parallel Execution Opportunities**:
- Groups 2 and 3 can be done in parallel with Group 1
- Group 5 can be done in parallel with any other group
- Groups 6 and 7 must wait for their corresponding source changes but can be done in parallel with each other

**Estimated Complexity**: Low
- All changes are pattern-based refactoring
- No new functionality
- No algorithm logic changes
- Straightforward property additions and simplifications
- Well-defined patterns from reference implementation

**Key Success Criteria**:
1. All property accesses work uniformly for both LinearSolver and NewtonKrylov
2. Constructor parameter handling matches established pattern
3. Update method delegation follows consistent pattern
4. Instrumented test files mirror source file changes exactly
5. No behavioral changes - existing tests pass unchanged
