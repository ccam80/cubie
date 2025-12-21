# Implementation Task List
# Feature: settings_dict_properties
# Plan Reference: .github/active_plans/settings_dict_properties/agent_plan.md

## Task Group 1: LinearSolver Settings Dict Properties - SEQUENTIAL
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py (lines 32-108 for LinearSolverConfig, lines 125-607 for LinearSolver)
- File: .github/context/cubie_internal_structure.md (entire file for patterns)

**Input Validation Required**:
None - properties are read-only, no user input

**Tasks**:

### Task 1.1: Add LinearSolverConfig.settings_dict Property
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py
- Action: Add property after line 107 (after simsafe_precision property)
- Details:
  ```python
  @property
  def settings_dict(self) -> Dict[str, Any]:
      """Return linear solver configuration as dictionary.
      
      Returns
      -------
      dict
          Configuration dictionary containing:
          - krylov_tolerance: Convergence tolerance for linear solver
          - max_linear_iters: Maximum iterations permitted
          - correction_type: Line-search strategy
          - preconditioned_vec_location: Buffer location for preconditioned vector
          - temp_location: Buffer location for temporary vector
      """
      return {
          'krylov_tolerance': self.krylov_tolerance,
          'max_linear_iters': self.max_linear_iters,
          'correction_type': self.correction_type,
          'preconditioned_vec_location': self.preconditioned_vec_location,
          'temp_location': self.temp_location,
      }
  ```
- Edge cases:
  - krylov_tolerance is a property that applies precision conversion (line 95-97)
  - max_linear_iters is a direct attribute access
  - correction_type is a string attribute
  - Buffer locations are string attributes
- Integration:
  - Uses existing property pattern from class (krylov_tolerance property)
  - Returns primitive values and strings (no mutable references)
  - Keys match parameter names in __init__ (lines 136-141)

### Task 1.2: Add LinearSolver.settings_dict Property
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py
- Action: Add property after line 607 (after persistent_local_buffer_size property)
- Details:
  ```python
  @property
  def settings_dict(self) -> Dict[str, Any]:
      """Return linear solver configuration as dictionary.
      
      Delegates to compile_settings for configuration state.
      
      Returns
      -------
      dict
          Configuration dictionary from LinearSolverConfig.settings_dict
      """
      return self.compile_settings.settings_dict
  ```
- Edge cases:
  - compile_settings is always present (set in __init__ via setup_compile_settings)
  - Returns copy of config dict (LinearSolverConfig creates new dict)
- Integration:
  - Follows pattern of existing properties that delegate to compile_settings
  - Enables: `newton_solver.linear_solver.settings_dict` access chain
  - Consistent with krylov_tolerance property pattern (line 580-582)

**Outcomes**:
- Files Modified:
  * src/cubie/integrators/matrix_free_solvers/linear_solver.py (28 lines added)
- Functions/Methods Added/Modified:
  * LinearSolverConfig.settings_dict property added (lines 110-132)
  * LinearSolver.settings_dict property added (lines 609-621)
- Implementation Summary:
  * Added settings_dict property to LinearSolverConfig that returns dictionary with krylov_tolerance, max_linear_iters, correction_type, and buffer locations
  * Added settings_dict property to LinearSolver that delegates to compile_settings.settings_dict
  * Both properties follow existing patterns in the codebase
  * Properties return new dict instances (no mutable references)
- Issues Flagged: None

---

## Task Group 2: NewtonKrylov Settings Dict Properties - SEQUENTIAL
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py (lines 32-128 for NewtonKrylovConfig, lines 146-610 for NewtonKrylov)
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py (lines 559-587 for LinearSolver properties as reference)
- File: .github/active_plans/settings_dict_properties/agent_plan.md (lines 98-150 for merge strategy)

**Input Validation Required**:
None - properties are read-only, no user input

**Tasks**:

### Task 2.1: Add NewtonKrylovConfig.settings_dict Property
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
- Action: Add property after line 128 (after simsafe_precision property)
- Details:
  ```python
  @property
  def settings_dict(self) -> Dict[str, Any]:
      """Return Newton-Krylov configuration as dictionary.
      
      Returns
      -------
      dict
          Configuration dictionary containing:
          - newton_tolerance: Residual norm threshold for convergence
          - max_newton_iters: Maximum Newton iterations permitted
          - newton_damping: Step shrink factor for backtracking
          - newton_max_backtracks: Maximum damping attempts per Newton step
          - delta_location: Buffer location for delta
          - residual_location: Buffer location for residual
          - residual_temp_location: Buffer location for residual_temp
          - stage_base_bt_location: Buffer location for stage_base_bt
      """
      return {
          'newton_tolerance': self.newton_tolerance,
          'max_newton_iters': self.max_newton_iters,
          'newton_damping': self.newton_damping,
          'newton_max_backtracks': self.newton_max_backtracks,
          'delta_location': self.delta_location,
          'residual_location': self.residual_location,
          'residual_temp_location': self.residual_temp_location,
          'stage_base_bt_location': self.stage_base_bt_location,
      }
  ```
- Edge cases:
  - newton_tolerance and newton_damping are properties that apply precision conversion (lines 111-118)
  - max_newton_iters and newton_max_backtracks are direct attribute access
  - All buffer location parameters are string attributes
- Integration:
  - Uses existing property pattern from class
  - Returns primitive values and strings (no mutable references)
  - Keys match parameter names in __init__ (lines 158-165)
  - Does NOT include linear solver settings (merged at factory level)

### Task 2.2: Add NewtonKrylov.settings_dict Property
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
- Action: Add property after line 610 (after persistent_local_buffer_size property)
- Details:
  ```python
  @property
  def settings_dict(self) -> Dict[str, Any]:
      """Return merged Newton and linear solver configuration.
      
      Combines Newton-level settings from compile_settings with
      linear solver settings from nested linear_solver instance.
      
      Returns
      -------
      dict
          Merged configuration dictionary containing both Newton
          parameters and linear solver parameters
      """
      combined = dict(self.linear_solver.settings_dict)
      combined.update(self.compile_settings.settings_dict)
      return combined
  ```
- Edge cases:
  - No key conflicts expected (Newton params vs linear params are distinct)
  - Newton settings override if conflict occurs (update order)
  - linear_solver is always present (stored in __init__ line 200)
- Integration:
  - Accesses self.linear_solver.settings_dict (requires Task 1.2)
  - Merge order: linear solver first, then Newton config
  - Creates new dict via dict() constructor (no shared references)
  - Enables: `implicit_step.solver.settings_dict` with complete settings

**Outcomes**:
- Files Modified:
  * src/cubie/integrators/matrix_free_solvers/newton_krylov.py (48 lines added)
- Functions/Methods Added/Modified:
  * NewtonKrylovConfig.settings_dict property added (lines 130-156)
  * NewtonKrylov.settings_dict property added (lines 612-630)
- Implementation Summary:
  * Added settings_dict property to NewtonKrylovConfig that returns dictionary with newton_tolerance, max_newton_iters, newton_damping, newton_max_backtracks, and buffer locations
  * Added settings_dict property to NewtonKrylov that merges linear_solver.settings_dict with compile_settings.settings_dict
  * Merge order: linear solver settings first, then Newton settings (Newton takes precedence)
  * Properties return new dict instances (no mutable references)
- Issues Flagged: None

---

## Task Group 3: ODEImplicitStep Settings Dict Override - SEQUENTIAL
**Status**: [x]
**Dependencies**: Task Group 2

**Required Context**:
- File: src/cubie/integrators/algorithms/ode_implicitstep.py (lines 26-83 for ImplicitStepConfig, lines 86-382 for ODEImplicitStep)
- File: src/cubie/integrators/algorithms/base_algorithm_step.py (lines 329-396 for BaseStepConfig.settings_dict)
- File: .github/active_plans/settings_dict_properties/agent_plan.md (lines 154-205 for override pattern)

**Input Validation Required**:
None - properties are read-only, no user input

**Tasks**:

### Task 3.1: Add ODEImplicitStep.settings_dict Property Override
- File: src/cubie/integrators/algorithms/ode_implicitstep.py
- Action: Add property after line 382 (after newton_max_backtracks property)
- Details:
  ```python
  @property
  def settings_dict(self) -> dict:
      """Return merged algorithm and solver settings.
      
      Combines implicit step configuration (beta, gamma, M, etc.)
      with solver settings (Newton and linear solver parameters).
      
      Returns
      -------
      dict
          Merged configuration dictionary containing:
          - Base step settings (n, n_drivers, precision) from BaseStepConfig
          - Implicit step settings (beta, gamma, M, preconditioner_order,
            get_solver_helper_fn) from ImplicitStepConfig
          - Solver settings (newton_tolerance, krylov_tolerance, etc.)
            from NewtonKrylov or LinearSolver
          - All buffer location parameters from solver hierarchy
      """
      settings = super().settings_dict
      settings.update(self.solver.settings_dict)
      return settings
  ```
- Edge cases:
  - super().settings_dict calls ImplicitStepConfig.settings_dict (line 71-83)
  - ImplicitStepConfig.settings_dict calls super().settings_dict (BaseStepConfig)
  - No key conflicts expected between algorithm and solver settings
  - self.solver is either NewtonKrylov or LinearSolver instance
- Integration:
  - Inherits from BaseAlgorithmStep which uses ImplicitStepConfig
  - ImplicitStepConfig.settings_dict already exists and returns algorithm params
  - Override merges algorithm settings with solver settings
  - Merge order: algorithm settings first (via super()), then solver settings
  - Creates new dict reference via update (no shared state)

**Outcomes**:
- Files Modified:
  * src/cubie/integrators/algorithms/ode_implicitstep.py (25 lines added)
- Functions/Methods Added/Modified:
  * ODEImplicitStep.settings_dict property override added (lines 384-407)
- Implementation Summary:
  * Added settings_dict property override in ODEImplicitStep class
  * Calls super().settings_dict to get base implicit step settings (beta, gamma, M, preconditioner_order, etc.)
  * Updates with self.solver.settings_dict to merge in solver parameters
  * Merge order: implicit config first, then solver config (solver settings added to base)
  * Property returns complete settings for hot-swapping algorithms
- Issues Flagged: None

---

## Task Group 4: Integration Verification - PARALLEL
**Status**: [x]
**Dependencies**: Task Group 3

**Required Context**:
- File: src/cubie/integrators/SingleIntegratorRunCore.py (lines 450-490 for _switch_algos method)
- File: src/cubie/integrators/algorithms/get_algorithm_step.py (entire file for factory function)
- All files from Task Groups 1-3

**Input Validation Required**:
None - verification only, no changes

**Tasks**:

### Task 4.1: Verify Settings Flow Through Algorithm Hot-Swap
- File: N/A (verification task)
- Action: Code inspection and validation
- Details:
  - Review SingleIntegratorRunCore._switch_algos() implementation
  - Verify that `old_settings = self._algo_step.settings_dict` captures enhanced dict
  - Confirm that old_settings is passed to get_algorithm_step()
  - Check that new algorithm instance receives and applies solver settings
  - Verify no code changes needed in SingleIntegratorRunCore
- Expected behavior:
  - _switch_algos calls self._algo_step.settings_dict
  - For implicit algorithms, this now returns merged algorithm + solver settings
  - get_algorithm_step() receives complete settings dict
  - New algorithm factory (BackwardsEulerStep, CrankNicolsonStep, etc.) applies recognized solver parameters
  - Hot-swap preserves solver configuration between algorithm changes
- Integration verification:
  - Settings chain: LinearSolverConfig → LinearSolver → NewtonKrylovConfig → NewtonKrylov → ImplicitStepConfig → ODEImplicitStep
  - Dictionary merge sequence:
    1. BaseStepConfig.settings_dict: {n, n_drivers, precision}
    2. ImplicitStepConfig.settings_dict: adds {beta, gamma, M, preconditioner_order, get_solver_helper_fn}
    3. LinearSolverConfig.settings_dict: {krylov_tolerance, max_linear_iters, correction_type, buffer_locations}
    4. LinearSolver.settings_dict: pass-through from config
    5. NewtonKrylovConfig.settings_dict: {newton_tolerance, max_newton_iters, newton_damping, newton_max_backtracks, buffer_locations}
    6. NewtonKrylov.settings_dict: merges linear + Newton
    7. ODEImplicitStep.settings_dict: merges implicit config + solver
  - Final result: Complete settings snapshot for hot-swapping

### Task 4.2: Document Expected Settings Dict Structure
- File: .github/active_plans/settings_dict_properties/task_list.md (this file)
- Action: Add documentation section
- Details:
  - Document the complete settings_dict structure for implicit algorithms
  - List all keys expected in final merged dictionary
  - Provide example showing settings flow through hot-swap
- Expected structure:
  ```python
  # For ODEImplicitStep with NewtonKrylov solver:
  {
      # From BaseStepConfig
      'n': int,
      'n_drivers': int,
      'precision': np.dtype,
      
      # From ImplicitStepConfig
      'beta': float,
      'gamma': float,
      'M': np.ndarray or sp.Matrix,
      'preconditioner_order': int,
      'get_solver_helper_fn': Callable or None,
      
      # From LinearSolverConfig (via NewtonKrylov)
      'krylov_tolerance': float,
      'max_linear_iters': int,
      'correction_type': str,
      'preconditioned_vec_location': str,
      'temp_location': str,
      
      # From NewtonKrylovConfig
      'newton_tolerance': float,
      'max_newton_iters': int,
      'newton_damping': float,
      'newton_max_backtracks': int,
      'delta_location': str,
      'residual_location': str,
      'residual_temp_location': str,
      'stage_base_bt_location': str,
  }
  ```

**Outcomes**:
- Files Modified: None (verification task)
- Verification Results:
  * Task 4.1 - Settings Flow Verification:
    - Reviewed SingleIntegratorRunCore._switch_algos() at line 468
    - Confirmed that `old_settings = self._algo_step.settings_dict` captures enhanced dict
    - Verified old_settings is passed to get_algorithm_step() at line 470-473
    - Checked get_algorithm_step() in algorithms/__init__.py (lines 116-191)
    - Confirmed settings dict is filtered via split_applicable_settings() to apply recognized parameters
    - Algorithm hot-swap will correctly preserve solver settings between algorithm changes
    - No code changes needed in SingleIntegratorRunCore
  * Task 4.2 - Settings Dict Structure Documentation:
    - Complete settings_dict structure for ODEImplicitStep with NewtonKrylov solver documented above
    - Settings chain verified:
      1. LinearSolverConfig.settings_dict → {krylov params + buffer locations}
      2. LinearSolver.settings_dict → pass-through from config
      3. NewtonKrylovConfig.settings_dict → {newton params + buffer locations}
      4. NewtonKrylov.settings_dict → merged linear + newton
      5. ImplicitStepConfig.settings_dict → {beta, gamma, M, preconditioner_order}
      6. ODEImplicitStep.settings_dict → merged implicit config + solver
    - Final dict contains all hot-swappable parameters for implicit algorithms
- Issues Flagged: None

---

## Summary

### Total Task Groups: 4
### Total Individual Tasks: 7

### Dependency Chain Overview:
1. **Task Group 1** (LinearSolver) → Foundation layer, no dependencies
2. **Task Group 2** (NewtonKrylov) → Depends on Task Group 1 (accesses linear_solver.settings_dict)
3. **Task Group 3** (ODEImplicitStep) → Depends on Task Group 2 (accesses solver.settings_dict)
4. **Task Group 4** (Verification) → Depends on Task Group 3 (validates complete chain)

### Parallel Execution Opportunities:
- Task Group 4 contains parallel verification tasks (independent inspections)
- All other groups must execute sequentially due to dependency chain

### Estimated Complexity:
- **Task Group 1**: Low complexity (2 simple property additions)
- **Task Group 2**: Low-Medium complexity (2 properties, one with merge logic)
- **Task Group 3**: Low complexity (1 property override with merge)
- **Task Group 4**: Low complexity (verification and documentation)

### Key Implementation Notes:
1. All properties are read-only (no setters required)
2. All dictionaries return copies, not references to mutable state
3. Merge order is critical: child settings first, parent overrides
4. No cache invalidation impact (properties are read-only views)
5. No changes required to SingleIntegratorRunCore (existing code already supports enhanced settings_dict)
6. Buffer location parameters are included as configuration settings
7. Property pattern follows existing conventions in codebase (see BaseStepConfig.settings_dict, step controller settings_dict)

### Expected Behavior After Implementation:
- Implicit algorithms export complete solver configuration via settings_dict
- Algorithm hot-swap in SingleIntegratorRun preserves solver settings
- Settings flow correctly through the property chain
- All buffer locations are captured in settings snapshot
- New algorithm instances receive applicable solver parameters
- Unrecognized parameters for a given algorithm are ignored (existing behavior)
