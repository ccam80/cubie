# Agent Plan: Settings Dict Properties Implementation

## Component Specifications

### 1. LinearSolverConfig.settings_dict Property

**Location:** `src/cubie/integrators/matrix_free_solvers/linear_solver.py`

**Expected Behavior:**
- Return a dictionary containing all linear solver configuration parameters
- Include solver convergence parameters: krylov_tolerance, max_linear_iters, correction_type
- Include all buffer location parameters: preconditioned_vec_location, temp_location
- Return copies or primitive values, not references to mutable internal state
- Property should be read-only (no setter)

**Integration:**
- Add property after the existing properties in LinearSolverConfig class (around line 98)
- Use existing property pattern from the class (krylov_tolerance, numba_precision, etc.)
- Dictionary keys should match parameter names used in __init__ and update()

**Data Structure:**
```python
{
    'krylov_tolerance': float,  # from self.krylov_tolerance property
    'max_linear_iters': int,    # from self.max_linear_iters
    'correction_type': str,     # from self.correction_type
    'preconditioned_vec_location': str,  # from self.preconditioned_vec_location
    'temp_location': str,       # from self.temp_location
}
```

**Architectural Notes:**
- LinearSolverConfig is an attrs class, so all parameters are already accessible as attributes
- Other config parameters (precision, n, operator_apply, preconditioner, use_cached_auxiliaries) are compile-time or runtime context, not hot-swappable settings
- precision and n are handled at higher levels (BaseStepConfig.settings_dict)

---

### 2. LinearSolver.settings_dict Property

**Location:** `src/cubie/integrators/matrix_free_solvers/linear_solver.py`

**Expected Behavior:**
- Pass through the compile_settings.settings_dict without modification
- Return value should be equivalent to `return self.compile_settings.settings_dict`
- Property should be read-only (no setter)

**Integration:**
- Add property after existing properties in LinearSolver class (around line 603)
- Follows same pattern as existing properties that delegate to compile_settings
- Complements existing property getters (krylov_tolerance, max_linear_iters, etc.)

**Architectural Notes:**
- LinearSolver is a CUDAFactory subclass
- Compile settings are managed via self.compile_settings (LinearSolverConfig instance)
- Factory class acts as a thin wrapper around config for most properties
- This property enables: `newton_solver.linear_solver.settings_dict`

---

### 3. NewtonKrylovConfig.settings_dict Property

**Location:** `src/cubie/integrators/matrix_free_solvers/newton_krylov.py`

**Expected Behavior:**
- Return a dictionary containing all Newton-Krylov configuration parameters
- Include Newton convergence parameters: newton_tolerance, max_newton_iters, newton_damping, newton_max_backtracks
- Include all buffer location parameters: delta_location, residual_location, residual_temp_location, stage_base_bt_location
- Return copies or primitive values, not references to mutable internal state
- Property should be read-only (no setter)

**Integration:**
- Add property after existing properties in NewtonKrylovConfig class (around line 129)
- Use existing property pattern from the class (newton_tolerance, newton_damping, etc.)
- Dictionary keys should match parameter names used in __init__ and update()

**Data Structure:**
```python
{
    'newton_tolerance': float,  # from self.newton_tolerance property
    'max_newton_iters': int,    # from self.max_newton_iters
    'newton_damping': float,    # from self.newton_damping property
    'newton_max_backtracks': int,  # from self.newton_max_backtracks
    'delta_location': str,      # from self.delta_location
    'residual_location': str,   # from self.residual_location
    'residual_temp_location': str,  # from self.residual_temp_location
    'stage_base_bt_location': str,  # from self.stage_base_bt_location
}
```

**Architectural Notes:**
- NewtonKrylovConfig is an attrs class
- Other config parameters (precision, n, residual_function, linear_solver_function) are compile-time or runtime context
- Linear solver settings are NOT included here - they're merged at the factory level

---

### 4. NewtonKrylov.settings_dict Property

**Location:** `src/cubie/integrators/matrix_free_solvers/newton_krylov.py`

**Expected Behavior:**
- Fetch self.compile_settings.settings_dict (Newton-level settings)
- Fetch self.linear_solver.settings_dict (linear solver settings)
- Merge the two dictionaries, with Newton-level settings taking precedence on conflicts
- Return the combined dictionary

**Integration:**
- Add property after existing properties in NewtonKrylov class (around line 610)
- Must access nested linear_solver instance: `self.linear_solver.settings_dict`
- Merge strategy: create new dict from linear solver, update with Newton dict

**Data Structure (merged):**
```python
{
    # From linear solver
    'krylov_tolerance': float,
    'max_linear_iters': int,
    'correction_type': str,
    'preconditioned_vec_location': str,
    'temp_location': str,
    # From Newton solver
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

**Merge Implementation Pattern:**
```python
@property
def settings_dict(self) -> Dict[str, Any]:
    """Return merged Newton and linear solver settings."""
    # Start with linear solver settings
    combined = dict(self.linear_solver.settings_dict)
    # Add/override with Newton-level settings
    combined.update(self.compile_settings.settings_dict)
    return combined
```

**Architectural Notes:**
- NewtonKrylov owns a LinearSolver instance (self.linear_solver)
- No key conflicts expected between linear and Newton settings
- If conflicts occur, Newton settings should take precedence (parent overrides child)
- This property enables: `implicit_step.solver.settings_dict`

---

### 5. ODEImplicitStep.settings_dict Property Override

**Location:** `src/cubie/integrators/algorithms/ode_implicitstep.py`

**Expected Behavior:**
- ODEImplicitStep already inherits settings_dict from BaseAlgorithmStep
- Currently returns only ImplicitStepConfig.settings_dict
- Must override to merge solver settings with implicit step config settings
- Solver settings should be added to the base settings (not override)

**Integration:**
- ODEImplicitStep does NOT currently override settings_dict
- Must add property that calls super().settings_dict and merges with self.solver.settings_dict
- Position: Add after existing properties (around line 382, after newton_max_backtracks property)

**Current Behavior (via ImplicitStepConfig):**
```python
# From ImplicitStepConfig.settings_dict (around line 71-83)
@property
def settings_dict(self) -> dict:
    settings_dict = super().settings_dict  # Gets n, n_drivers, precision from BaseStepConfig
    settings_dict.update(
        {
            'beta': self.beta,
            'gamma': self.gamma,
            'M': self.M,
            'preconditioner_order': self.preconditioner_order,
            'get_solver_helper_fn': self.get_solver_helper_fn,
        }
    )
    return settings_dict
```

**Required Override:**
```python
# In ODEImplicitStep class
@property
def settings_dict(self) -> dict:
    """Return merged algorithm and solver settings."""
    # Get base implicit step settings (beta, gamma, M, etc.)
    settings = super().settings_dict
    # Merge with solver settings (newton, krylov parameters, buffer locations)
    settings.update(self.solver.settings_dict)
    return settings
```

**Architectural Notes:**
- ODEImplicitStep inherits from BaseAlgorithmStep
- self.solver is either NewtonKrylov or LinearSolver instance
- Merge order: base settings first, then solver settings
- No key conflicts expected (algorithm params vs solver params are distinct)

---

### 6. SingleIntegratorRunCore Integration

**Location:** `src/cubie/integrators/SingleIntegratorRunCore.py`

**Expected Behavior:**
- No code changes required in SingleIntegratorRunCore
- _switch_algos() already captures `old_settings = self._algo_step.settings_dict`
- Enhanced settings_dict from ODEImplicitStep will automatically flow through
- New algorithm instance created via get_algorithm_step() will receive complete settings

**Integration Points:**
- _switch_algos() at line 468: captures old settings
- get_algorithm_step() receives settings dict and applies recognized parameters
- Algorithm factories (BackwardsEulerStep, etc.) already accept solver parameters in __init__

**Validation:**
- After implementation, verify that _switch_algos() propagates solver parameters
- Confirm that old_settings dict contains solver parameters after merge
- Test that new algorithm instance receives and applies solver settings

**Architectural Notes:**
- SingleIntegratorRunCore orchestrates algorithm and controller hot-swapping
- Current implementation assumes settings_dict contains all hot-swappable parameters
- This implementation fulfills that assumption for implicit algorithms

---

## Expected Interactions Between Components

### Component Call Chain

```
SingleIntegratorRunCore._switch_algos()
    └─> self._algo_step.settings_dict
        └─> ODEImplicitStep.settings_dict
            ├─> super().settings_dict  (ImplicitStepConfig)
            │   └─> super().settings_dict  (BaseStepConfig)
            │       └─> {n, n_drivers, precision}
            │   └─> {beta, gamma, M, preconditioner_order, get_solver_helper_fn}
            └─> self.solver.settings_dict
                └─> NewtonKrylov.settings_dict (or LinearSolver.settings_dict)
                    ├─> self.linear_solver.settings_dict
                    │   └─> LinearSolver.settings_dict
                    │       └─> self.compile_settings.settings_dict
                    │           └─> LinearSolverConfig.settings_dict
                    │               └─> {krylov_tolerance, max_linear_iters, 
                    │                    correction_type, preconditioned_vec_location,
                    │                    temp_location}
                    └─> self.compile_settings.settings_dict
                        └─> NewtonKrylovConfig.settings_dict
                            └─> {newton_tolerance, max_newton_iters, newton_damping,
                                 newton_max_backtracks, delta_location, 
                                 residual_location, residual_temp_location,
                                 stage_base_bt_location}
```

### Dictionary Merge Sequence

1. **BaseStepConfig.settings_dict** returns `{n, n_drivers, precision}`
2. **ImplicitStepConfig.settings_dict** merges base + `{beta, gamma, M, preconditioner_order, get_solver_helper_fn}`
3. **LinearSolverConfig.settings_dict** returns `{krylov_tolerance, max_linear_iters, correction_type, buffer_locations}`
4. **LinearSolver.settings_dict** passes through config dict
5. **NewtonKrylovConfig.settings_dict** returns `{newton_tolerance, max_newton_iters, newton_damping, newton_max_backtracks, buffer_locations}`
6. **NewtonKrylov.settings_dict** merges linear solver dict + Newton config dict
7. **ODEImplicitStep.settings_dict** merges implicit config dict + solver dict

Final result: Complete settings snapshot containing all hot-swappable parameters

---

## Edge Cases to Consider

### 1. Linear-Only Solver (No Newton Wrapper)
**Scenario:** ODEImplicitStep.solver is a LinearSolver instance (not NewtonKrylov)  
**Expected Behavior:** 
- self.solver.settings_dict returns only linear solver parameters
- No Newton parameters in final settings_dict
- Algorithm hot-swap still works, new algorithm gets linear solver settings

**Handling:** No special code needed - duck typing handles this naturally

---

### 2. Buffer Location Parameter Conflicts
**Scenario:** Both LinearSolver and NewtonKrylov have buffer location parameters  
**Expected Behavior:**
- Linear solver: preconditioned_vec_location, temp_location
- Newton solver: delta_location, residual_location, residual_temp_location, stage_base_bt_location
- Different parameter names, so no conflicts

**Handling:** No special code needed - parameter names are distinct by design

---

### 3. Missing Solver Instance
**Scenario:** ODEImplicitStep.solver is None (shouldn't happen but defensive)  
**Expected Behavior:** AttributeError when accessing self.solver.settings_dict  
**Handling:** Not required - ODEImplicitStep.__init__ always creates a solver instance

---

### 4. Hot-Swap Between Implicit and Explicit Algorithms
**Scenario:** Swap from BackwardsEuler (implicit) to ExplicitEuler (explicit)  
**Expected Behavior:**
- Explicit algorithm receives solver parameters in settings dict
- ExplicitEuler.update() doesn't recognize solver parameters
- Parameters ignored with warning (if not silent)

**Handling:** Already handled by BaseAlgorithmStep.update() - valid but inapplicable parameters are recognized and warned

---

### 5. Dict Mutability
**Scenario:** Caller modifies returned settings_dict  
**Expected Behavior:** Should not affect internal state  
**Handling:** Each level creates new dict via dict() constructor or dict.update() - returns copies, not references

---

## Dependencies and Imports

All required imports already exist in the target files:
- `Dict` and `Any` from typing (already imported)
- No new dependencies required
- All classes are already defined

## Testing Considerations

### Unit Tests Needed:
1. Test LinearSolverConfig.settings_dict returns expected keys and values
2. Test LinearSolver.settings_dict passes through config dict
3. Test NewtonKrylovConfig.settings_dict returns expected keys and values
4. Test NewtonKrylov.settings_dict correctly merges linear and Newton dicts
5. Test ODEImplicitStep.settings_dict includes solver parameters
6. Integration test: hot-swap BackwardsEuler to CrankNicolson preserves solver settings

### Test Assertions:
- Verify all expected keys present in returned dicts
- Verify values match corresponding property getters
- Verify buffer location parameters included
- Verify no unintended side effects (dict mutations don't affect internal state)
- Verify merge order (parent settings don't get overwritten by child settings inappropriately)

## Implementation Order

1. **LinearSolverConfig.settings_dict** - Foundation, no dependencies
2. **LinearSolver.settings_dict** - Depends on LinearSolverConfig property
3. **NewtonKrylovConfig.settings_dict** - Foundation, no dependencies
4. **NewtonKrylov.settings_dict** - Depends on LinearSolver and NewtonKrylovConfig properties
5. **ODEImplicitStep.settings_dict** - Depends on NewtonKrylov property
6. **Validation** - Test SingleIntegratorRunCore hot-swap behavior

This order ensures each component has its dependencies available when implemented.

## Rollback Considerations

If issues arise:
- All changes are additive (new properties, one override)
- No existing functionality removed
- Can roll back by removing added properties
- No database migrations or persistent state changes
- Cache invalidation unaffected (read-only properties)
