# Implementation Task List
# Feature: Fix Instrumented Solver Instantiation
# Plan Reference: .github/active_plans/fix_instrumented_solvers/agent_plan.md

## Task Group 1: matrix_free_solvers.py (Instrumented) - SEQUENTIAL
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: `tests/integrators/algorithms/instrumented/matrix_free_solvers.py` (entire file)
- File: `src/cubie/integrators/matrix_free_solvers/linear_solver.py` (lines 147-208, class LinearSolver.__init__)
- File: `src/cubie/integrators/matrix_free_solvers/newton_krylov.py` (lines 174-256, class NewtonKrylov.__init__)

**Input Validation Required**:
- None - rely on parent class validators

**Tasks**:
1. **Update InstrumentedLinearSolver.__init__ to match production LinearSolver**
   - File: `tests/integrators/algorithms/instrumented/matrix_free_solvers.py`
   - Action: Modify
   - Details:
     ```python
     def __init__(
         self,
         precision: PrecisionDType,
         n: int,
         correction_type: Optional[str] = None,
         krylov_tolerance: Optional[float] = None,
         max_linear_iters: Optional[int] = None,
         preconditioned_vec_location: Optional[str] = None,
         temp_location: Optional[str] = None,
     ) -> None:
         """Initialize InstrumentedLinearSolver with parameters.
         
         Parameters
         ----------
         precision : PrecisionDType
             Numerical precision for computations.
         n : int
             Length of residual and search-direction vectors.
         correction_type : str, optional
             Line-search strategy ('steepest_descent' or 'minimal_residual').
         krylov_tolerance : float, optional
             Target on squared residual norm for convergence.
         max_linear_iters : int, optional
             Maximum iterations permitted.
         preconditioned_vec_location : str, optional
             Memory location for preconditioned_vec buffer.
         temp_location : str, optional
             Memory location for temp buffer.
         """
         super().__init__(
             precision=precision,
             n=n,
             correction_type=correction_type,
             krylov_tolerance=krylov_tolerance,
             max_linear_iters=max_linear_iters,
             preconditioned_vec_location=preconditioned_vec_location,
             temp_location=temp_location,
         )
     ```
   - Edge cases: None - parent handles all validation
   - Integration: Child class now matches parent signature exactly

2. **Update InstrumentedLinearSolver.build() to use correct property names**
   - File: `tests/integrators/algorithms/instrumented/matrix_free_solvers.py`
   - Action: Modify
   - Details:
     - Change `config.tolerance` to `self.krylov_tolerance` (line ~102)
     - Change `config.max_iters` to `self.max_linear_iters` (line ~103)
     - Change `max_iters` to `max_linear_iters` in subsequent uses
     - Update buffer allocator key names if they differ from production
   - Edge cases: Property access patterns differ between instrumented and production
   - Integration: Aligns with production property naming

3. **Update InstrumentedNewtonKrylov.__init__ to match production NewtonKrylov**
   - File: `tests/integrators/algorithms/instrumented/matrix_free_solvers.py`
   - Action: Modify
   - Details:
     ```python
     def __init__(
         self,
         precision: PrecisionDType,
         n: int,
         linear_solver: LinearSolver,
         newton_tolerance: Optional[float] = None,
         max_newton_iters: Optional[int] = None,
         newton_damping: Optional[float] = None,
         newton_max_backtracks: Optional[int] = None,
         delta_location: Optional[str] = None,
         residual_location: Optional[str] = None,
         residual_temp_location: Optional[str] = None,
         stage_base_bt_location: Optional[str] = None,
     ) -> None:
         """Initialize InstrumentedNewtonKrylov with parameters.
         
         Parameters match production NewtonKrylov exactly.
         """
         super().__init__(
             precision=precision,
             n=n,
             linear_solver=linear_solver,
             newton_tolerance=newton_tolerance,
             max_newton_iters=max_newton_iters,
             newton_damping=newton_damping,
             newton_max_backtracks=newton_max_backtracks,
             delta_location=delta_location,
             residual_location=residual_location,
             residual_temp_location=residual_temp_location,
             stage_base_bt_location=stage_base_bt_location,
         )
     ```
   - Edge cases: linear_solver must be InstrumentedLinearSolver or compatible
   - Integration: Child class now matches parent signature exactly

4. **Update InstrumentedNewtonKrylov.build() to use correct property names**
   - File: `tests/integrators/algorithms/instrumented/matrix_free_solvers.py`
   - Action: Modify
   - Details:
     - Change `config.linear_solver` to `self.linear_solver`
     - Change `config.tolerance` to `self.newton_tolerance`
     - Change `config.max_iters` to `self.max_newton_iters`
     - Change `config.damping` to `self.newton_damping`
     - Change `config.max_backtracks` to `self.newton_max_backtracks`
     - Update buffer allocator key names to match buffer_registry patterns
   - Edge cases: Property naming patterns may differ
   - Integration: Aligns with production naming conventions

5. **Clean up unused imports**
   - File: `tests/integrators/algorithms/instrumented/matrix_free_solvers.py`
   - Action: Modify
   - Details:
     - Remove `LinearSolverConfig` import (no longer passed to __init__)
     - Remove `NewtonKrylovConfig` import (no longer passed to __init__)
     - Add `PrecisionDType` import from cubie._utils
     - Ensure `Optional` from typing is imported
   - Edge cases: None
   - Integration: Reduces confusion about expected __init__ signature

**Outcomes**: 
- InstrumentedLinearSolver and InstrumentedNewtonKrylov accept same parameters as production classes
- No TypeError when instantiating with keyword arguments

---

## Task Group 2: backwards_euler.py (Instrumented) - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Group 1

**Required Context**:
- File: `tests/integrators/algorithms/instrumented/backwards_euler.py` (entire file)
- File: `src/cubie/integrators/algorithms/backwards_euler.py` (entire file for reference)
- File: `src/cubie/integrators/algorithms/ode_implicitstep.py` (lines 266-315, build_implicit_helpers)

**Input Validation Required**:
- None - rely on parent class validators

**Tasks**:
1. **Remove custom build_implicit_helpers() method**
   - File: `tests/integrators/algorithms/instrumented/backwards_euler.py`
   - Action: Delete
   - Details:
     - Remove the entire `build_implicit_helpers()` method (lines 131-207)
     - The parent `ODEImplicitStep.build_implicit_helpers()` will be used instead
     - Instrumented logging is already in `build_step()` - that method stays unchanged
   - Edge cases: Ensure `build_step()` still receives solver_function correctly from parent build()
   - Integration: Relies on parent class to set up solver correctly

2. **Clean up unused imports**
   - File: `tests/integrators/algorithms/instrumented/backwards_euler.py`
   - Action: Modify
   - Details:
     - Remove `InstrumentedNewtonKrylov` import (no longer used after removing build_implicit_helpers)
     - Remove `LinearSolverConfig` import
     - Remove `NewtonKrylovConfig` import
   - Edge cases: None
   - Integration: Simplifies file to match production structure

**Outcomes**: 
- BackwardsEulerStep instrumented version matches production __init__ and build_implicit_helpers()
- Only build_step() has instrumented device function with logging arrays

---

## Task Group 3: crank_nicolson.py (Instrumented) - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Group 1

**Required Context**:
- File: `tests/integrators/algorithms/instrumented/crank_nicolson.py` (entire file)
- File: `src/cubie/integrators/algorithms/crank_nicolson.py` (entire file for reference)

**Input Validation Required**:
- None - rely on parent class validators

**Tasks**:
1. **Remove custom build_implicit_helpers() method**
   - File: `tests/integrators/algorithms/instrumented/crank_nicolson.py`
   - Action: Delete
   - Details:
     - Remove the entire `build_implicit_helpers()` method (lines 183-258)
     - Parent class handles solver setup correctly
   - Edge cases: None
   - Integration: Relies on parent class for solver configuration

2. **Update register_buffers() to add child allocator registration**
   - File: `tests/integrators/algorithms/instrumented/crank_nicolson.py`
   - Action: Modify
   - Details:
     - Add `buffer_registry.get_child_allocators()` call before registering cn_dxdt buffer
     - Match production pattern exactly:
       ```python
       def register_buffers(self) -> None:
           """Register buffers with buffer_registry."""
           config = self.compile_settings
           alloc_solver_shared, alloc_solver_persistent = (
               buffer_registry.get_child_allocators(self, self.solver,
                                                    name='solver_scratch')
           )
           buffer_registry.register(
               'cn_dxdt',
               self,
               config.n,
               config.dxdt_location,
               aliases='solver_scratch_shared',
               precision=config.precision
           )
       ```
   - Edge cases: None
   - Integration: Matches production buffer registration pattern

3. **Clean up unused imports**
   - File: `tests/integrators/algorithms/instrumented/crank_nicolson.py`
   - Action: Modify
   - Details:
     - Remove `InstrumentedLinearSolver` import
     - Remove `InstrumentedNewtonKrylov` import
     - Remove `LinearSolverConfig` import
     - Remove `NewtonKrylovConfig` import
   - Edge cases: None
   - Integration: Simplifies imports to match production

**Outcomes**: 
- CrankNicolsonStep instrumented version matches production pattern
- Uses parent build_implicit_helpers() without custom solver instantiation

---

## Task Group 4: generic_dirk.py (Instrumented) - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Group 1

**Required Context**:
- File: `tests/integrators/algorithms/instrumented/generic_dirk.py` (entire file)
- File: `src/cubie/integrators/algorithms/generic_dirk.py` (entire file for reference)

**Input Validation Required**:
- None - rely on parent class validators

**Tasks**:
1. **Remove solver replacement code from __init__**
   - File: `tests/integrators/algorithms/instrumented/generic_dirk.py`
   - Action: Delete
   - Details:
     - Remove lines 158-176 that replace solver with instrumented version:
       ```python
       # REMOVE THIS BLOCK:
       # Replace solver with instrumented version that has logging
       original_linear_solver = self.solver.linear_solver
       instrumented_linear = InstrumentedLinearSolver(...)
       instrumented_newton = InstrumentedNewtonKrylov(...)
       self.solver = instrumented_newton
       ```
     - The `__init__` should now end with:
       ```python
       super().__init__(config, controller_defaults, **solver_kwargs)
       self.register_buffers()
       ```
   - Edge cases: None
   - Integration: Matches production __init__ exactly

2. **Clean up unused imports**
   - File: `tests/integrators/algorithms/instrumented/generic_dirk.py`
   - Action: Modify
   - Details:
     - Remove `InstrumentedLinearSolver` import
     - Remove `InstrumentedNewtonKrylov` import
   - Edge cases: None
   - Integration: Simplifies imports

**Outcomes**: 
- DIRKStep instrumented version has __init__ matching production exactly
- build_implicit_helpers() already uses correct pattern (self.solver.update())

---

## Task Group 5: generic_firk.py (Instrumented) - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Group 1

**Required Context**:
- File: `tests/integrators/algorithms/instrumented/generic_firk.py` (entire file)
- File: `src/cubie/integrators/algorithms/generic_firk.py` (entire file for reference)

**Input Validation Required**:
- None - rely on parent class validators

**Tasks**:
1. **Remove _replace_with_instrumented_solvers() method**
   - File: `tests/integrators/algorithms/instrumented/generic_firk.py`
   - Action: Delete
   - Details:
     - Remove the entire `_replace_with_instrumented_solvers()` method (lines 169-193)
   - Edge cases: None
   - Integration: Method no longer needed

2. **Remove call to _replace_with_instrumented_solvers() in __init__**
   - File: `tests/integrators/algorithms/instrumented/generic_firk.py`
   - Action: Modify
   - Details:
     - Remove line 165: `self._replace_with_instrumented_solvers()`
     - The `__init__` should now end with:
       ```python
       super().__init__(config, controller_defaults, **solver_kwargs)
       self.register_buffers()
       ```
   - Edge cases: None
   - Integration: Matches production __init__ exactly

3. **Clean up unused imports**
   - File: `tests/integrators/algorithms/instrumented/generic_firk.py`
   - Action: Modify
   - Details:
     - Remove `InstrumentedLinearSolver` import
     - Remove `InstrumentedNewtonKrylov` import
   - Edge cases: None
   - Integration: Simplifies imports

**Outcomes**: 
- FIRKStep instrumented version has __init__ matching production exactly
- No custom solver instantiation anywhere

---

## Task Group 6: generic_rosenbrock_w.py (Instrumented) - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Group 1

**Required Context**:
- File: `tests/integrators/algorithms/instrumented/generic_rosenbrock_w.py` (entire file)
- File: `src/cubie/integrators/algorithms/generic_rosenbrock_w.py` (entire file for reference)

**Input Validation Required**:
- None - rely on parent class validators

**Tasks**:
1. **Remove solver replacement from build_implicit_helpers()**
   - File: `tests/integrators/algorithms/instrumented/generic_rosenbrock_w.py`
   - Action: Modify
   - Details:
     - Remove lines 276-277:
       ```python
       # REMOVE:
       instrumented_solver = InstrumentedLinearSolver(self.solver.compile_settings)
       self.solver = instrumented_solver
       ```
     - The build_implicit_helpers() should now match production exactly:
       ```python
       def build_implicit_helpers(self) -> Callable:
           # ... get device functions as before ...
           
           # Update linear solver with device functions
           self.solver.update(
               operator_apply=operator,
               preconditioner=preconditioner,
               use_cached_auxiliaries=True,
           )
           
           # Return linear solver device function
           self.update_compile_settings(
               {'solver_function': self.solver.device_function,
                'time_derivative_function': time_derivative_function,
                'prepare_jacobian_function': prepare_jacobian}
           )
       ```
   - Edge cases: None
   - Integration: Matches production build_implicit_helpers() exactly

2. **Clean up unused imports**
   - File: `tests/integrators/algorithms/instrumented/generic_rosenbrock_w.py`
   - Action: Modify
   - Details:
     - Remove `InstrumentedLinearSolver` import
   - Edge cases: None
   - Integration: Simplifies imports

**Outcomes**: 
- GenericRosenbrockWStep instrumented version matches production pattern exactly
- Uses production LinearSolver, only build_step() has logging

---

## Summary

### Total Task Groups: 6
### Dependency Chain: 
```
Task Group 1 (matrix_free_solvers.py) 
    ↓
Task Groups 2-6 (algorithm files) [can run in PARALLEL after Group 1]
```

### Parallel Execution Opportunities:
- Task Groups 2, 3, 4, 5, 6 can all be executed in parallel after Task Group 1 completes
- Task Group 1 must complete first as it fixes the solver class signatures

### Estimated Complexity:
- Task Group 1: High (multiple signature changes, property name updates)
- Task Group 2: Low (delete method, update imports)
- Task Group 3: Medium (delete method, update register_buffers, update imports)
- Task Group 4: Low (delete code block, update imports)
- Task Group 5: Low (delete method, delete call, update imports)
- Task Group 6: Low (remove 2 lines, update imports)
