# Agent Implementation Plan: Post-Refactor Cleanup

## Context

This plan addresses cleanup tasks following a major refactor of the buffer allocation system in CuBIE. The refactor converted factory functions to CUDAFactory subclasses and replaced BufferSettings with buffer_registry.

Reference: `.github/context/cubie_internal_structure.md` for architectural patterns.

## Component Changes Required

### 1. Import Statement Updates

**Location:** `src/cubie/integrators/__init__.py`

**Current State:**
```python
from cubie.integrators.matrix_free_solvers import (
    linear_solver_factory,
    newton_krylov_solver_factory,
)
```

**Required Change:**
```python
from cubie.integrators.matrix_free_solvers import (
    LinearSolver,
    NewtonKrylov,
)
```

**Behavior:**
- These are now CUDAFactory subclasses, not factory functions
- Instantiated with config objects: `LinearSolver(config=LinearSolverConfig(...))`
- Return compiled device functions via `.device_function` property

**Integration Points:**
- Used by implicit algorithm steps (BackwardsEulerStep, CrankNicolsonStep, etc.)
- Imported in algorithm modules via `from cubie.integrators.matrix_free_solvers import LinearSolver`

### 2. Export Updates in __init__.py Files

**Location:** `src/cubie/integrators/__init__.py`

**Current __all__ entries to remove:**
- `"linear_solver_factory"`
- `"newton_krylov_solver_factory"`

**New __all__ entries (already present in matrix_free_solvers/__init__.py):**
- `"LinearSolver"`
- `"NewtonKrylov"`
- `"LinearSolverConfig"`
- `"LinearSolverCache"`
- `"NewtonKrylovConfig"`
- `"NewtonKrylovCache"`

**Behavior:**
- Config classes hold compilation parameters (precision, n, tolerances, etc.)
- Cache classes hold compiled outputs (device functions)
- Main classes inherit from CUDAFactory and expose `.device_function` property

**Note:** The matrix_free_solvers/__init__.py already has correct exports. The integrators/__init__.py needs to re-export them or remove the old names.

### 3. File Removals

**Files to Delete:**

1. **`src/cubie/BufferSettings.py`**
   - Currently contains only deprecation message
   - Replaced by `src/cubie/buffer_registry.py`
   - No remaining imports detected in src/ or tests/

2. **`tests/test_buffer_settings.py`**
   - Deprecated test file with notice pointing to test_buffer_registry.py
   - Replacement test file exists and is comprehensive

3. **`tests/integrators/algorithms/test_buffer_settings.py`**
   - Tests old BufferSettings with algorithm steps
   - Functionality covered by algorithm tests + buffer_registry tests

4. **`tests/integrators/loops/test_buffer_settings.py`**
   - Tests old BufferSettings with loop compilation
   - Functionality covered by loop tests + buffer_registry tests

5. **`tests/integrators/matrix_free_solvers/test_buffer_settings.py`**
   - Tests old BufferSettings with linear solver
   - Functionality covered by solver tests + buffer_registry tests

6. **`tests/integrators/matrix_free_solvers/test_newton_buffer_settings.py`**
   - Tests old BufferSettings with Newton-Krylov solver
   - Functionality covered by solver tests + buffer_registry tests

**Verification:** Confirm no imports of these files remain after deletion

### 4. Parameter Naming Consistency

**Search Scope:** All `.py` files in `src/cubie/` and `tests/`

**Parameters to Verify:**

#### 4.1 krylov_tolerance
**Status:** Appears correctly spelled throughout
**Locations:**
- `linear_solver.py`: Config class attribute, property, factory parameter
- Algorithm steps: BackwardsEuler, CrankNicolson, DIRK, FIRK, RosenbrockW
- `base_algorithm_step.py`: Listed in ALL_ALGORITHM_STEP_PARAMETERS

**Verification:** Search for misspellings like `kyrlov_tolerance`

#### 4.2 Buffer Location Parameters
**Pattern:** `[buffer_name]_location`

**Known Examples:**
- `state_location` (loop state buffer)
- `proposed_state_location` (loop proposed state buffer)
- `preconditioned_vec_location` (linear solver buffer)
- `stage_state_location` (FIRK algorithm buffer)

**Consistency Check:**
- All location parameters should end with `_location`
- Values should be `'shared'` or `'local'`
- Should be listed in appropriate ALL_*_PARAMETERS sets

**ALL_*_PARAMETERS Sets to Verify:**
1. `ALL_ALGORITHM_STEP_PARAMETERS` (base_algorithm_step.py)
2. `ALL_LOOP_SETTINGS` (ode_loop.py)
3. `ALL_BUFFER_LOCATION_PARAMETERS` (SingleIntegratorRunCore.py)
4. `ALL_STEP_CONTROLLER_PARAMETERS` (base_step_controller.py)
5. `ALL_OUTPUT_FUNCTION_PARAMETERS` (output_functions.py)
6. `ALL_MEMORY_MANAGER_PARAMETERS` (mem_manager.py)

### 5. Buffer Registry Integration

**Current State:**
- `buffer_registry` singleton exists in `src/cubie/buffer_registry.py`
- Used by LinearSolver, NewtonKrylov, algorithm steps, loops
- Replaces old BufferSettings pattern

**Expected Behavior:**
- Components register buffers during `__init__`:
  ```python
  buffer_registry.register(
      name='buffer_name',
      parent=self,
      size=n,
      location='local',  # or 'shared'
      persistent=False,
      aliases=None,
      precision=self.precision
  )
  ```

- Retrieve allocators during `build()`:
  ```python
  allocate_buffer = buffer_registry.get_allocator('buffer_name', self)
  ```

- Update locations via `buffer_registry.update()`:
  ```python
  buffer_registry.update(
      parent=self,
      buffer_name_location='shared'
  )
  ```

**Verification Points:**
- LinearSolver registers: 'preconditioned_vec', 'residual', 'search_direction'
- NewtonKrylov registers: 'state', 'residual', 'rhs', 'residual_norm'
- IVPLoop registers: 'loop_state', 'proposed_state'
- Algorithm steps register stage-specific buffers

### 6. Test Suite Considerations

**CUDA Simulated Tests:**
- Run with: `NUMBA_ENABLE_CUDASIM="1" pytest -m "not nocudasim and not cupy"`
- Focus: Init and parameter errors only
- Ignore: Numerical errors, memory errors (out of scope)

**Expected Failures to Fix:**
1. **ImportError** - Old factory function imports
2. **TypeError** - Wrong class instantiation pattern
3. **AttributeError** - Missing or renamed attributes
4. **KeyError** - Missing parameter names

**Acceptable Failures (do not fix):**
1. Numerical accuracy differences
2. Memory allocation errors
3. Convergence failures
4. Performance issues

### 7. Documentation References

**Files to Check:**
- Any docstrings mentioning `linear_solver_factory` or `newton_krylov_solver_factory`
- References to `BufferSettings` in docstrings or comments
- Parameter documentation in config classes

**Update Pattern:**
- Replace factory function references with class names
- Update usage examples to show class instantiation
- Correct import statements in examples

## Edge Cases to Consider

### Case 1: Circular Imports
**Scenario:** NewtonKrylov imports LinearSolver  
**Current State:** Working correctly  
**Verification:** Import cubie.integrators should succeed

### Case 2: Test Fixtures
**Scenario:** Tests may use indirect parameterization with old factory names  
**Check:** `tests/conftest.py` for fixture definitions  
**Action:** Update any fixtures referencing old names

### Case 3: Instrumented Tests
**Scenario:** `tests/integrators/algorithms/instrumented/` contains copies  
**Check:** If linear_solver or newton_krylov instrumented variants exist  
**Action:** Update instrumented versions to match refactored structure  
**Note:** Internal structure document mentions these must stay in sync

### Case 4: Dynamic Imports
**Scenario:** get_algorithm_step() dynamically instantiates steps  
**Check:** Does it pass correct parameters to LinearSolver/NewtonKrylov?  
**Verification:** Run algorithm tests with various implicit methods

### Case 5: Partial Updates
**Scenario:** Some files may import from submodules directly  
**Check:** `from cubie.integrators.matrix_free_solvers.linear_solver import ...`  
**Action:** These should continue working; focus on __init__.py imports

## Dependencies and Constraints

### Internal Dependencies
- `buffer_registry` module must be imported before use
- `CUDAFactory` base class must be available
- Config classes (LinearSolverConfig, etc.) must be importable
- Precision system in `_utils.py` must be consistent

### External Dependencies
- Numba CUDA compilation
- Attrs for config classes
- NumPy for array operations

### Constraints
- Cannot modify instrumented test files without updating source files
- Cannot change public API without deprecation period (not applicable here)
- Must maintain Python 3.8+ compatibility
- Must work in both CUDA and CUDASIM modes

## Validation Strategy

### Phase 1: Import Validation
1. Fix `integrators/__init__.py` imports
2. Run: `python -c "import cubie"`
3. Success: No ImportError

### Phase 2: Deprecated File Removal
1. Delete BufferSettings.py
2. Delete deprecated test files
3. Search codebase for any remaining references
4. Run: `grep -r "BufferSettings" src/ tests/`
5. Success: Only comments/docstrings found

### Phase 3: Parameter Consistency
1. Search for all `*_tolerance` parameters
2. Search for all `*_location` parameters
3. Verify ALL_*_PARAMETERS sets are complete
4. Success: No inconsistencies found

### Phase 4: Test Execution
1. Run: `NUMBA_ENABLE_CUDASIM="1" pytest -m "not nocudasim and not cupy" --tb=short`
2. Check for init/parameter errors only
3. Success: No ImportError, TypeError, AttributeError, or KeyError related to refactor

### Phase 5: Spot Check
1. Instantiate LinearSolver and NewtonKrylov directly
2. Verify buffer_registry.get_allocator() works
3. Run single algorithm test with implicit method
4. Success: Code executes without errors

## Implementation Order

1. **Fix Critical Import** (blocks everything else)
   - Update `src/cubie/integrators/__init__.py`
   - Update `__all__` list

2. **Remove Deprecated Files** (clean slate)
   - Delete src/cubie/BufferSettings.py
   - Delete 5 deprecated test files

3. **Verify Parameters** (ensure consistency)
   - Search and verify krylov_tolerance
   - Search and verify *_location parameters
   - Check ALL_*_PARAMETERS sets

4. **Run Tests** (validate changes)
   - Execute CUDA simulated test suite
   - Fix any additional import/parameter issues discovered

5. **Final Verification** (confirm completion)
   - Full codebase search for old names
   - Verify no broken imports remain
   - Document any intentional exceptions

## Success Criteria

1. ✓ Package imports without errors
2. ✓ No deprecated files remain in repository
3. ✓ All __init__.py files export current names only
4. ✓ Parameter names are consistent throughout
5. ✓ ALL_*_PARAMETERS sets are accurate and complete
6. ✓ CUDA simulated tests run without init/parameter errors
7. ✓ No references to removed files exist in code
8. ✓ Documentation reflects current API
