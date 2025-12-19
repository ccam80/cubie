# Implementation Task List
# Feature: Instrumented Matrix-Free Solvers Refactor
# Plan Reference: .github/active_plans/instrumented_matrix_free_solvers/agent_plan.md

## Task Group 1: Create InstrumentedLinearSolverCache - SEQUENTIAL
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: src/cubie/CUDAFactory.py (lines 1-50) - CUDAFunctionCache base class
- File: src/cubie/_utils.py (lines 1-100) - is_device_validator
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py (lines 126-138) - LinearSolverCache pattern

**Input Validation Required**:
- linear_solver: Validate with is_device_validator (attrs handles this)

**Tasks**:
1. **Create InstrumentedLinearSolverCache attrs class**
   - File: tests/integrators/algorithms/instrumented/matrix_free_solvers.py
   - Action: Create (add at top of file after imports)
   - Details:
     ```python
     @attrs.define
     class InstrumentedLinearSolverCache(CUDAFunctionCache):
         """Cache container for InstrumentedLinearSolver outputs.
         
         Attributes
         ----------
         linear_solver : Callable
             Compiled CUDA device function with logging signature.
         """
         
         linear_solver: Callable = attrs.field(
             validator=is_device_validator
         )
     ```
   - Edge cases: None (simple data container)
   - Integration: Returned by InstrumentedLinearSolver.build()

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 2: Create InstrumentedLinearSolver Class - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Group 1

**Required Context**:
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py (lines 141-587) - LinearSolver complete implementation
- File: tests/integrators/algorithms/instrumented/matrix_free_solvers.py (lines 11-152) - inst_linear_solver_factory non-cached
- File: tests/integrators/algorithms/instrumented/matrix_free_solvers.py (lines 154-308) - inst_linear_solver_cached_factory cached variant
- File: src/cubie/CUDAFactory.py (lines 1-200) - CUDAFactory base class methods

**Input Validation Required**:
- None (inherits validation from LinearSolver parent and config attrs validators)
- Config validation already handled by LinearSolverConfig attrs class

**Tasks**:
1. **Create InstrumentedLinearSolver class skeleton**
   - File: tests/integrators/algorithms/instrumented/matrix_free_solvers.py
   - Action: Create (add after InstrumentedLinearSolverCache)
   - Details:
     ```python
     class InstrumentedLinearSolver(LinearSolver):
         """Factory for instrumented linear solver device functions.
         
         Inherits from LinearSolver and adds iteration logging to device function.
         Logging arrays are passed as device function parameters and populated
         during iteration. Uses buffer_registry for production buffers
         (preconditioned_vec, temp) but logging arrays are caller-allocated.
         """
         
         def __init__(self, config: LinearSolverConfig) -> None:
             """Initialize InstrumentedLinearSolver with configuration.
             
             Parameters
             ----------
             config : LinearSolverConfig
                 Configuration containing all compile-time parameters.
             """
             super().__init__(config)
         
         def build(self) -> InstrumentedLinearSolverCache:
             """Compile instrumented linear solver device function.
             
             Returns
             -------
             InstrumentedLinearSolverCache
                 Container with compiled linear_solver device function including
                 logging parameters.
             
             Raises
             ------
             ValueError
                 If operator_apply is None when build() is called.
             """
             # Implementation in next task
             pass
     ```
   - Edge cases: None at this level (handled by parent __init__)
   - Integration: Instantiated in test infrastructure with LinearSolverConfig

2. **Implement build() method - non-cached variant**
   - File: tests/integrators/algorithms/instrumented/matrix_free_solvers.py
   - Action: Modify (InstrumentedLinearSolver.build method)
   - Details:
     - Extract config parameters (copy from parent build() lines 206-248)
     - Compute flags: sd_flag, mr_flag, preconditioned
     - Type conversions: n_val, max_iters_val, precision_numba, typed_zero, tol_squared
     - Get allocators from buffer_registry (alloc_precond, alloc_temp)
     - Branch on use_cached_auxiliaries flag
     - For non-cached variant, compile device function with signature:
       ```python
       def linear_solver(
           state, parameters, drivers, base_state,
           t, h, a_ij, rhs, x, shared, krylov_iters_out,
           # Logging parameters:
           slot_index,
           linear_initial_guesses,
           linear_iteration_guesses,
           linear_residuals,
           linear_squared_norms,
           linear_preconditioned_vectors,
       ):
       ```
     - Copy device logic from src/cubie/integrators/matrix_free_solvers/linear_solver.py lines 426-510
     - Inject logging statements from tests/integrators/algorithms/instrumented/matrix_free_solvers.py lines 60-148:
       - After initial residual computation: Record x[i] to linear_initial_guesses[slot_index, i]
       - Inside iteration loop after convergence update: Record to linear_iteration_guesses, linear_residuals, linear_preconditioned_vectors, linear_squared_norms
     - Use log_iter = iteration - int32(1) for 0-based indexing
     - Return InstrumentedLinearSolverCache(linear_solver=linear_solver)
   - Edge cases:
     - Converged on first iteration: still log initial guess and first iteration
     - Max iterations reached: log all iterations up to max
   - Integration: Device function callable from Newton-Krylov and test code

3. **Implement build() method - cached variant**
   - File: tests/integrators/algorithms/instrumented/matrix_free_solvers.py
   - Action: Modify (InstrumentedLinearSolver.build method, add cached branch)
   - Details:
     - In build() method, add if use_cached_auxiliaries branch
     - For cached variant, compile device function with signature:
       ```python
       def linear_solver_cached(
           state, parameters, drivers, base_state, cached_aux,
           t, h, a_ij, rhs, x, shared, krylov_iters_out,
           # Logging parameters:
           slot_index,
           linear_initial_guesses,
           linear_iteration_guesses,
           linear_residuals,
           linear_squared_norms,
           linear_preconditioned_vectors,
       ):
       ```
     - Copy device logic from src/cubie/integrators/matrix_free_solvers/linear_solver.py lines 275-357
     - Inject same logging statements as non-cached variant
     - Only difference: operator_apply and preconditioner calls include cached_aux parameter
     - Return InstrumentedLinearSolverCache(linear_solver=linear_solver_cached)
   - Edge cases: Same as non-cached variant
   - Integration: Device function callable from Newton-Krylov and test code

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 3: Create InstrumentedNewtonKrylovCache - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Groups 1, 2

**Required Context**:
- File: src/cubie/CUDAFactory.py (lines 1-50) - CUDAFunctionCache base class
- File: src/cubie/_utils.py (lines 1-100) - is_device_validator
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py (lines 160-172) - NewtonKrylovCache pattern

**Input Validation Required**:
- newton_krylov_solver: Validate with is_device_validator (attrs handles this)

**Tasks**:
1. **Create InstrumentedNewtonKrylovCache attrs class**
   - File: tests/integrators/algorithms/instrumented/matrix_free_solvers.py
   - Action: Create (add after InstrumentedLinearSolver class)
   - Details:
     ```python
     @attrs.define
     class InstrumentedNewtonKrylovCache(CUDAFunctionCache):
         """Cache container for InstrumentedNewtonKrylov outputs.
         
         Attributes
         ----------
         newton_krylov_solver : Callable
             Compiled CUDA device function with logging signature.
         """
         
         newton_krylov_solver: Callable = attrs.field(
             validator=is_device_validator
         )
     ```
   - Edge cases: None (simple data container)
   - Integration: Returned by InstrumentedNewtonKrylov.build()

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 4: Create InstrumentedNewtonKrylov Class - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Groups 1, 2, 3

**Required Context**:
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py (lines 175-570) - NewtonKrylov complete implementation
- File: tests/integrators/algorithms/instrumented/matrix_free_solvers.py (lines 311-563) - inst_newton_krylov_solver_factory
- File: src/cubie/CUDAFactory.py (lines 1-200) - CUDAFactory base class methods

**Input Validation Required**:
- None (inherits validation from NewtonKrylov parent and config attrs validators)
- Config validation already handled by NewtonKrylovConfig attrs class
- linear_solver must be InstrumentedLinearSolver instance (validate in build())

**Tasks**:
1. **Create InstrumentedNewtonKrylov class skeleton**
   - File: tests/integrators/algorithms/instrumented/matrix_free_solvers.py
   - Action: Create (add after InstrumentedNewtonKrylovCache)
   - Details:
     ```python
     class InstrumentedNewtonKrylov(NewtonKrylov):
         """Factory for instrumented Newton-Krylov solver device functions.
         
         Inherits from NewtonKrylov and adds iteration logging to device function.
         Logging arrays are passed as device function parameters and populated
         during Newton iteration. Embeds InstrumentedLinearSolver for nested
         linear solve logging.
         """
         
         def __init__(self, config: NewtonKrylovConfig) -> None:
             """Initialize InstrumentedNewtonKrylov with configuration.
             
             Parameters
             ----------
             config : NewtonKrylovConfig
                 Configuration containing all compile-time parameters.
                 linear_solver must be InstrumentedLinearSolver instance.
             """
             super().__init__(config)
         
         def build(self) -> InstrumentedNewtonKrylovCache:
             """Compile instrumented Newton-Krylov solver device function.
             
             Returns
             -------
             InstrumentedNewtonKrylovCache
                 Container with compiled newton_krylov_solver device function
                 including logging parameters.
             
             Raises
             ------
             ValueError
                 If residual_function or linear_solver is None when build() is called.
             TypeError
                 If linear_solver is not InstrumentedLinearSolver instance.
             """
             # Implementation in next task
             pass
     ```
   - Edge cases: None at this level (handled by parent __init__)
   - Integration: Instantiated in test infrastructure with NewtonKrylovConfig

2. **Implement build() method - device function compilation**
   - File: tests/integrators/algorithms/instrumented/matrix_free_solvers.py
   - Action: Modify (InstrumentedNewtonKrylov.build method)
   - Details:
     - Extract config parameters (copy from parent build() lines 236-256)
     - Validate linear_solver is InstrumentedLinearSolver:
       ```python
       if not isinstance(config.linear_solver, InstrumentedLinearSolver):
           raise TypeError(
               "InstrumentedNewtonKrylov requires linear_solver to be "
               "InstrumentedLinearSolver instance"
           )
       ```
     - Get linear solver device function: linear_solver_fn = linear_solver.device_function
     - Type conversions: precision_dtype, numba_precision, tol_squared, typed_zero, typed_one, typed_damping, n_val, max_iters_val, max_backtracks_val
     - Get allocators from buffer_registry: alloc_delta, alloc_residual, alloc_residual_temp, alloc_stage_base_bt
     - Compute lin_shared_offset = buffer_registry.shared_buffer_size(self)
     - Compile device function with signature:
       ```python
       def newton_krylov_solver(
           stage_increment, parameters, drivers,
           t, h, a_ij, base_state, shared_scratch, counters,
           # Logging parameters:
           stage_index,
           newton_initial_guesses,
           newton_iteration_guesses,
           newton_residuals,
           newton_squared_norms,
           newton_iteration_scale,
           linear_initial_guesses,
           linear_iteration_guesses,
           linear_residuals,
           linear_squared_norms,
           linear_preconditioned_vectors,
       ):
       ```
     - Copy device logic from src/cubie/integrators/matrix_free_solvers/newton_krylov.py lines 336-467
     - Inject logging statements from tests/integrators/algorithms/instrumented/matrix_free_solvers.py lines 401-547:
       - After initial residual evaluation: Record stage_increment[i] to newton_initial_guesses[stage_index, i]
       - Create residual_copy local array and log first iteration
       - Inside Newton loop: compute linear_slot_base and call linear solver with logging arrays
       - Create stage_increment_snapshot and residual_snapshot local arrays
       - Inside backtracking loop: snapshot values when snapshot_ready=True
       - After backtracking: log iteration state and scale factor if snapshot_ready
     - Linear solver call must include logging parameters:
       ```python
       lin_shared = shared_scratch[lin_shared_offset:]
       lin_status = linear_solver_fn(
           stage_increment, parameters, drivers, base_state,
           t, h, a_ij, residual, delta, lin_shared, krylov_iters_local,
           # Logging parameters:
           linear_slot_base + iter_slot,
           linear_initial_guesses,
           linear_iteration_guesses,
           linear_residuals,
           linear_squared_norms,
           linear_preconditioned_vectors,
       )
       ```
     - Return InstrumentedNewtonKrylovCache(newton_krylov_solver=newton_krylov_solver)
   - Edge cases:
     - Linear solver fails: propagate error status, still log Newton state
     - Backtracking fails: revert to stage_base_bt, still log scale
     - Convergence on first iteration: log initial and first iteration state
     - No snapshot_ready: don't log iteration (backtrack failed before evaluation)
   - Integration: Device function callable from implicit algorithms

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 5: Remove Old Factory Functions - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Groups 1, 2, 3, 4

**Required Context**:
- File: tests/integrators/algorithms/instrumented/matrix_free_solvers.py (lines 11-563) - All factory functions to remove

**Input Validation Required**:
- None (deletion task)

**Tasks**:
1. **Remove inst_linear_solver_factory**
   - File: tests/integrators/algorithms/instrumented/matrix_free_solvers.py
   - Action: Delete
   - Details: Remove function definition from lines 11-151 (approximately)
   - Edge cases: None
   - Integration: Replaced by InstrumentedLinearSolver class

2. **Remove inst_linear_solver_cached_factory**
   - File: tests/integrators/algorithms/instrumented/matrix_free_solvers.py
   - Action: Delete
   - Details: Remove function definition from lines 154-308 (approximately)
   - Edge cases: None
   - Integration: Replaced by InstrumentedLinearSolver class (cached variant)

3. **Remove inst_newton_krylov_solver_factory**
   - File: tests/integrators/algorithms/instrumented/matrix_free_solvers.py
   - Action: Delete
   - Details: Remove function definition from lines 311-563 (approximately)
   - Edge cases: None
   - Integration: Replaced by InstrumentedNewtonKrylov class

4. **Update __all__ export list**
   - File: tests/integrators/algorithms/instrumented/matrix_free_solvers.py
   - Action: Modify
   - Details:
     ```python
     __all__ = [
         "InstrumentedLinearSolver",
         "InstrumentedLinearSolverCache",
         "InstrumentedNewtonKrylov",
         "InstrumentedNewtonKrylovCache",
     ]
     ```
   - Edge cases: None
   - Integration: Exports new class-based API

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 6: Update Import Statements in matrix_free_solvers.py - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Groups 1, 2, 3, 4, 5

**Required Context**:
- File: tests/integrators/algorithms/instrumented/matrix_free_solvers.py (lines 1-9) - Current imports
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py (lines 1-30) - Required imports
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py (lines 1-30) - Required imports

**Input Validation Required**:
- None (import updates)

**Tasks**:
1. **Add production class imports**
   - File: tests/integrators/algorithms/instrumented/matrix_free_solvers.py
   - Action: Modify (import section at top)
   - Details:
     ```python
     import attrs
     from typing import Callable, Optional
     
     import numpy as np
     from numba import cuda, int32, from_dtype
     
     from cubie._utils import is_device_validator, PrecisionDType
     from cubie.buffer_registry import buffer_registry
     from cubie.CUDAFactory import CUDAFunctionCache
     from cubie.cuda_simsafe import activemask, all_sync, selp, any_sync, compile_kwargs
     from cubie.integrators.matrix_free_solvers.linear_solver import (
         LinearSolver,
         LinearSolverConfig,
     )
     from cubie.integrators.matrix_free_solvers.newton_krylov import (
         NewtonKrylov,
         NewtonKrylovConfig,
     )
     ```
   - Edge cases: None
   - Integration: Provides base classes and config types

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 7: Update Instrumented Algorithm - backwards_euler.py - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Groups 1-6

**Required Context**:
- File: tests/integrators/algorithms/instrumented/backwards_euler.py (lines 1-176) - Complete file
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py (lines 32-98) - LinearSolverConfig
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py (lines 32-111) - NewtonKrylovConfig

**Input Validation Required**:
- None (validation handled by config classes)

**Tasks**:
1. **Update imports in backwards_euler.py**
   - File: tests/integrators/algorithms/instrumented/backwards_euler.py
   - Action: Modify (import section)
   - Details:
     ```python
     from .matrix_free_solvers import (
         InstrumentedLinearSolver,
         InstrumentedNewtonKrylov,
     )
     from cubie.integrators.matrix_free_solvers.linear_solver import LinearSolverConfig
     from cubie.integrators.matrix_free_solvers.newton_krylov import NewtonKrylovConfig
     ```
   - Edge cases: None
   - Integration: Imports new class-based API

2. **Update build_implicit_helpers() method**
   - File: tests/integrators/algorithms/instrumented/backwards_euler.py
   - Action: Modify (lines 109-176 approximately)
   - Details:
     - Replace inst_linear_solver_factory call (lines 150-158) with:
       ```python
       linear_solver_config = LinearSolverConfig(
           precision=precision,
           n=n,
           operator_apply=operator,
           preconditioner=preconditioner,
           correction_type=correction_type,
           tolerance=krylov_tolerance,
           max_iters=max_linear_iters,
           use_cached_auxiliaries=False,
       )
       linear_solver_instance = InstrumentedLinearSolver(linear_solver_config)
       ```
     - Replace inst_newton_krylov_solver_factory call (lines 165-174) with:
       ```python
       newton_config = NewtonKrylovConfig(
           precision=precision,
           n=n,
           residual_function=residual,
           linear_solver=linear_solver_instance,
           tolerance=newton_tolerance,
           max_iters=max_newton_iters,
           damping=newton_damping,
           max_backtracks=newton_max_backtracks,
       )
       newton_instance = InstrumentedNewtonKrylov(newton_config)
       ```
     - Change return statement to:
       ```python
       return newton_instance.device_function
       ```
   - Edge cases: None (config validation handles parameter checking)
   - Integration: Returns device function to build_step() method

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 8: Update Instrumented Algorithm - backwards_euler_predict_correct.py - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Groups 1-6

**Required Context**:
- File: tests/integrators/algorithms/instrumented/backwards_euler_predict_correct.py (entire file)
- Pattern from Group 7 Task 2

**Input Validation Required**:
- None (validation handled by config classes)

**Tasks**:
1. **Update imports**
   - File: tests/integrators/algorithms/instrumented/backwards_euler_predict_correct.py
   - Action: Modify (import section)
   - Details: Same pattern as Group 7 Task 1
   - Edge cases: None
   - Integration: Imports new class-based API

2. **Update build_implicit_helpers() method**
   - File: tests/integrators/algorithms/instrumented/backwards_euler_predict_correct.py
   - Action: Modify
   - Details: Same pattern as Group 7 Task 2
   - Edge cases: None
   - Integration: Returns device function to build_step() method

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 9: Update Instrumented Algorithm - crank_nicolson.py - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Groups 1-6

**Required Context**:
- File: tests/integrators/algorithms/instrumented/crank_nicolson.py (entire file)
- Pattern from Group 7 Task 2

**Input Validation Required**:
- None (validation handled by config classes)

**Tasks**:
1. **Update imports**
   - File: tests/integrators/algorithms/instrumented/crank_nicolson.py
   - Action: Modify (import section)
   - Details: Same pattern as Group 7 Task 1
   - Edge cases: None
   - Integration: Imports new class-based API

2. **Update build_implicit_helpers() method**
   - File: tests/integrators/algorithms/instrumented/crank_nicolson.py
   - Action: Modify
   - Details: Same pattern as Group 7 Task 2
   - Edge cases: None
   - Integration: Returns device function to build_step() method

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 10: Update Instrumented Algorithm - generic_dirk.py - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Groups 1-6

**Required Context**:
- File: tests/integrators/algorithms/instrumented/generic_dirk.py (entire file)
- Pattern from Group 7 Task 2
- Note: DIRK may use cached_auxiliaries variant

**Input Validation Required**:
- None (validation handled by config classes)

**Tasks**:
1. **Update imports**
   - File: tests/integrators/algorithms/instrumented/generic_dirk.py
   - Action: Modify (import section)
   - Details: Same pattern as Group 7 Task 1
   - Edge cases: None
   - Integration: Imports new class-based API

2. **Update build_implicit_helpers() method**
   - File: tests/integrators/algorithms/instrumented/generic_dirk.py
   - Action: Modify
   - Details:
     - Same pattern as Group 7 Task 2
     - Check if use_cached_auxiliaries is set in config
     - Pass use_cached_auxiliaries to LinearSolverConfig
   - Edge cases: Must handle both cached and non-cached variants
   - Integration: Returns device function to build_step() method

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 11: Update Instrumented Algorithm - generic_firk.py - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Groups 1-6

**Required Context**:
- File: tests/integrators/algorithms/instrumented/generic_firk.py (entire file)
- Pattern from Group 7 Task 2

**Input Validation Required**:
- None (validation handled by config classes)

**Tasks**:
1. **Update imports**
   - File: tests/integrators/algorithms/instrumented/generic_firk.py
   - Action: Modify (import section)
   - Details: Same pattern as Group 7 Task 1
   - Edge cases: None
   - Integration: Imports new class-based API

2. **Update build_implicit_helpers() method**
   - File: tests/integrators/algorithms/instrumented/generic_firk.py
   - Action: Modify
   - Details: Same pattern as Group 7 Task 2
   - Edge cases: None
   - Integration: Returns device function to build_step() method

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 12: Update Instrumented Algorithm - generic_rosenbrock_w.py - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Groups 1-6

**Required Context**:
- File: tests/integrators/algorithms/instrumented/generic_rosenbrock_w.py (entire file)
- Pattern from Group 7 Task 2
- Note: Rosenbrock may use cached_auxiliaries variant

**Input Validation Required**:
- None (validation handled by config classes)

**Tasks**:
1. **Update imports**
   - File: tests/integrators/algorithms/instrumented/generic_rosenbrock_w.py
   - Action: Modify (import section)
   - Details: Same pattern as Group 7 Task 1
   - Edge cases: None
   - Integration: Imports new class-based API

2. **Update build_implicit_helpers() method**
   - File: tests/integrators/algorithms/instrumented/generic_rosenbrock_w.py
   - Action: Modify
   - Details:
     - Same pattern as Group 7 Task 2
     - Check if use_cached_auxiliaries is set in config
     - Pass use_cached_auxiliaries to LinearSolverConfig
   - Edge cases: Must handle both cached and non-cached variants
   - Integration: Returns device function to build_step() method

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 13: Verify All Tests Pass - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Groups 1-12

**Required Context**:
- File: tests/integrators/algorithms/instrumented/test_instrumented.py (entire file)
- All updated instrumented algorithm files

**Input Validation Required**:
- None (test execution)

**Tasks**:
1. **Run instrumented algorithm tests**
   - File: N/A (test execution)
   - Action: Execute tests
   - Details:
     - Run: `pytest tests/integrators/algorithms/instrumented/test_instrumented.py -v`
     - Verify all tests pass
     - Check logged data matches expected patterns
   - Edge cases:
     - CUDA not available: tests should skip gracefully
     - CUDASIM mode: tests marked nocudasim should skip
   - Integration: Validates entire refactor is functional

2. **Run linters**
   - File: N/A (linting)
   - Action: Execute linters
   - Details:
     - Run: `flake8 tests/integrators/algorithms/instrumented/ --count --select=E9,F63,F7,F82 --show-source --statistics`
     - Run: `ruff check tests/integrators/algorithms/instrumented/`
     - Fix any issues found
   - Edge cases: None
   - Integration: Ensures code quality standards

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Summary

**Total Task Groups**: 13

**Dependency Chain**:
```
1 → 2 → 3 → 4 → 5 → 6 → 7, 8, 9, 10, 11, 12 → 13
         ↓
    (Groups 7-12 can execute in parallel after Group 6)
```

**Parallel Execution Opportunities**:
- Groups 7-12 (instrumented algorithm updates) can run in parallel after Group 6 completes
- All other groups must run sequentially

**Estimated Complexity**:
- **High complexity**: Groups 2, 4 (core implementation with device function logic)
- **Medium complexity**: Groups 7-12 (algorithmic updates following pattern)
- **Low complexity**: Groups 1, 3, 5, 6, 13 (boilerplate, cleanup, validation)

**Critical Path**:
Groups 1 → 2 → 3 → 4 → 5 → 6 → (any of 7-12) → 13

**Key Integration Points**:
1. InstrumentedLinearSolver inherits from LinearSolver (Group 2)
2. InstrumentedNewtonKrylov embeds InstrumentedLinearSolver (Group 4)
3. All instrumented algorithms call .device_function property (Groups 7-12)
4. Linear solver device function receives logging arrays via parameters (Groups 2, 4)
5. Newton-Krylov computes linear solver slot indices (Group 4)

**Validation Strategy**:
1. Build and test InstrumentedLinearSolver standalone first
2. Then build and test InstrumentedNewtonKrylov with embedded linear solver
3. Update one algorithm (backwards_euler) and test incrementally
4. Update remaining algorithms in parallel
5. Run full test suite to validate integration
