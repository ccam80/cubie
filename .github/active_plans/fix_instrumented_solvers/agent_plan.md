# Agent Implementation Plan

## Task Group 1: Fix InstrumentedLinearSolver and InstrumentedNewtonKrylov

### File: `tests/integrators/algorithms/instrumented/matrix_free_solvers.py`

**Problem**: The `InstrumentedLinearSolver.__init__` accepts a `LinearSolverConfig` object, but the parent `LinearSolver.__init__` accepts individual parameters (`precision`, `n`, `correction_type`, etc.). Same issue with `InstrumentedNewtonKrylov`.

**Required Changes**:

1. Update `InstrumentedLinearSolver.__init__` to accept the same signature as `LinearSolver`:
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

2. Update `InstrumentedNewtonKrylov.__init__` to accept the same signature as `NewtonKrylov`:
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

3. Remove old imports: `LinearSolverConfig`, `NewtonKrylovConfig`

4. Add necessary imports: `PrecisionDType` from cubie._utils, `Optional` from typing

---

## Task Group 2: Fix Backwards Euler Instrumented

### File: `tests/integrators/algorithms/instrumented/backwards_euler.py`

**Problem**: 
- Has custom `build_implicit_helpers()` that creates `InstrumentedNewtonKrylov` with wrong parameters
- Imports unused `LinearSolverConfig` and `NewtonKrylovConfig`

**Required Changes**:

1. Remove unused imports:
   - Remove `InstrumentedNewtonKrylov` import
   - Remove `LinearSolverConfig` import  
   - Remove `NewtonKrylovConfig` import

2. Remove the entire custom `build_implicit_helpers()` method (lines 131-207)
   - The parent `ODEImplicitStep.build_implicit_helpers()` should be used
   - Instrumented behavior only needs to be in `build_step()` where logging arrays are used

3. The `build_step()` method already has the instrumented device function with logging - this should remain unchanged

---

## Task Group 3: Fix Crank-Nicolson Instrumented

### File: `tests/integrators/algorithms/instrumented/crank_nicolson.py`

**Problem**:
- Has custom `build_implicit_helpers()` creating `InstrumentedLinearSolver` and `InstrumentedNewtonKrylov` with wrong parameters
- Imports unused config classes

**Required Changes**:

1. Remove unused imports:
   - Remove `InstrumentedLinearSolver` import
   - Remove `InstrumentedNewtonKrylov` import
   - Remove `LinearSolverConfig` import
   - Remove `NewtonKrylovConfig` import

2. Remove the entire custom `build_implicit_helpers()` method (lines 183-258)
   - Production pattern from parent class should be used

3. Update `register_buffers()` to match production exactly (check for any differences with parent child allocator registration)

---

## Task Group 4: Fix Generic DIRK Instrumented

### File: `tests/integrators/algorithms/instrumented/generic_dirk.py`

**Problem**:
- Creates `InstrumentedLinearSolver` and `InstrumentedNewtonKrylov` in `__init__` with wrong parameters
- The `__init__` should be EXACT copy of production

**Required Changes**:

1. Remove unused imports:
   - Remove `InstrumentedLinearSolver` import
   - Remove `InstrumentedNewtonKrylov` import

2. In `__init__`: Remove the entire block that replaces solver with instrumented version (lines 158-176):
   ```python
   # Replace solver with instrumented version that has logging
   original_linear_solver = self.solver.linear_solver
   instrumented_linear = InstrumentedLinearSolver(...)
   instrumented_newton = InstrumentedNewtonKrylov(...)
   self.solver = instrumented_newton
   ```

3. `build_implicit_helpers()` already uses `self.solver.update()` pattern correctly - no changes needed

4. `build_step()` already has instrumented device function with logging - no changes needed

---

## Task Group 5: Fix Generic FIRK Instrumented

### File: `tests/integrators/algorithms/instrumented/generic_firk.py`

**Problem**:
- Has `_replace_with_instrumented_solvers()` method that creates instrumented solvers with wrong parameters
- Calls this method in `__init__`

**Required Changes**:

1. Remove unused imports:
   - Remove `InstrumentedLinearSolver` import
   - Remove `InstrumentedNewtonKrylov` import

2. Remove the `_replace_with_instrumented_solvers()` method entirely (lines 169-193)

3. In `__init__`: Remove the call to `self._replace_with_instrumented_solvers()` (line 165)

4. The `__init__` should now match production exactly

---

## Task Group 6: Fix Generic Rosenbrock-W Instrumented

### File: `tests/integrators/algorithms/instrumented/generic_rosenbrock_w.py`

**Problem**:
- In `build_implicit_helpers()`: Creates `InstrumentedLinearSolver(self.solver.compile_settings)` which passes a config object
- Should follow production pattern

**Required Changes**:

1. Remove the `InstrumentedLinearSolver` import

2. In `build_implicit_helpers()`: Remove the lines that create and assign instrumented solver (lines 276-277):
   ```python
   instrumented_solver = InstrumentedLinearSolver(self.solver.compile_settings)
   self.solver = instrumented_solver
   ```

3. The method should now match production exactly (just calling `self.solver.update()` and `self.update_compile_settings()`)

---

## Integration Notes

### Solver Hierarchy
Production flow:
1. `ODEImplicitStep.__init__()` creates `LinearSolver` → `NewtonKrylov` (or just `LinearSolver` for Rosenbrock)
2. `build_implicit_helpers()` calls `self.solver.update()` with device functions
3. `build_implicit_helpers()` calls `self.update_compile_settings(solver_function=...)`

Instrumented flow should be IDENTICAL except:
- `build_step()` method has additional logging array parameters in device function signature
- Device function inside `build_step()` records data to logging arrays

### Why This Fixes the Error
The `TypeError: InstrumentedLinearSolver.__init__() got an unexpected keyword argument 'precision'` occurs because:
- Some code is calling `InstrumentedLinearSolver(precision=..., n=..., ...)` 
- But `InstrumentedLinearSolver.__init__` only accepts `config` parameter
- After fixing, both signatures will match and calls will work

### Testing Verification
After fixes, run tests to ensure:
1. No `TypeError` on solver instantiation
2. Instrumented algorithms produce correct logging data
3. Algorithm behavior matches production (same numerical results)
