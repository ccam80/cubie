# Agent Plan: Solver Ownership Pattern Refinement

## Objective

Replicate coding patterns established in newton_krylov.py and ode_implicitstep.py across all related solver and algorithm files to ensure consistency.

## Pattern Descriptions

### Pattern 1: Conditional Constructor Parameters (LinearSolver)

**Current state (linear_solver.py lines 167-175):**
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

**Desired state (match newton_krylov.py lines 204-226):**
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

**Why:** Defaults managed in LinearSolverConfig, not in constructor. Cleaner separation of concerns.

### Pattern 2: Update Method Delegation (LinearSolver)

**Current state (linear_solver.py lines 529-558):**
```python
# Merge updates
all_updates = {}
if updates_dict:
    all_updates.update(updates_dict)
all_updates.update(kwargs)

if not all_updates:
    return set()

# Extract linear solver parameters
linear_keys = {...}
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

**Desired state (match newton_krylov.py lines 518-538 and ode_implicitstep.py lines 179-199):**
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

**Why:** 
- Pass full dict to update_compile_settings (it filters internally)
- Call buffer_registry.update last (doesn't contribute to recognized set)
- Consistent pattern with NewtonKrylov and ODEImplicitStep

### Pattern 3: Pass-through Properties (NewtonKrylov)

**Add to NewtonKrylov (after existing properties):**
```python
@property
def krylov_tolerance(self) -> float:
    """Return krylov tolerance from nested linear solver."""
    return self.linear_solver.krylov_tolerance

@property
def max_linear_iters(self) -> int:
    """Return max linear iterations from nested linear solver."""
    return self.linear_solver.max_linear_iters

@property
def correction_type(self) -> str:
    """Return correction type from nested linear solver."""
    return self.linear_solver.correction_type
```

**Why:** Allows ODEImplicitStep to access linear properties uniformly regardless of solver type.

### Pattern 4: Simplified Property Access (ODEImplicitStep)

**Current state (ode_implicitstep.py lines 348-369):**
```python
@property
def krylov_tolerance(self) -> float:
    """Return the tolerance used for the linear solve."""
    if hasattr(self.solver, 'krylov_tolerance'):
        return self.solver.krylov_tolerance
    # For NewtonKrylov, forward to nested linear_solver
    return self.solver.linear_solver.krylov_tolerance

@property
def max_linear_iters(self) -> int:
    """Return the maximum number of linear iterations allowed."""
    if hasattr(self.solver, 'max_linear_iters'):
        return int(self.solver.max_linear_iters)
    # For NewtonKrylov, forward to nested linear_solver
    return int(self.solver.linear_solver.max_linear_iters)
```

**Desired state (after Pattern 3 applied):**
```python
@property
def krylov_tolerance(self) -> float:
    """Return the tolerance used for the linear solve."""
    return self.solver.krylov_tolerance

@property
def max_linear_iters(self) -> int:
    """Return the maximum number of linear iterations allowed."""
    return int(self.solver.max_linear_iters)

@property
def linear_correction_type(self) -> str:
    """Return the linear correction strategy identifier."""
    return self.solver.correction_type
```

**For nonlinear properties:**
```python
@property
def newton_tolerance(self) -> Optional[float]:
    """Return the Newton solve tolerance."""
    return getattr(self.solver, 'newton_tolerance', None)

@property
def max_newton_iters(self) -> Optional[int]:
    """Return the maximum allowed Newton iterations."""
    val = getattr(self.solver, 'max_newton_iters', None)
    return int(val) if val is not None else None

@property
def newton_damping(self) -> Optional[float]:
    """Return the Newton damping factor."""
    return getattr(self.solver, 'newton_damping', None)

@property
def newton_max_backtracks(self) -> Optional[int]:
    """Return the maximum number of Newton backtracking steps."""
    val = getattr(self.solver, 'newton_max_backtracks', None)
    return int(val) if val is not None else None
```

**Why:** 
- Linear properties: direct access (works for both LinearSolver and NewtonKrylov after pass-through)
- Nonlinear properties: getattr with None default (only NewtonKrylov has these)
- No hasattr checks or conditional delegation

## Component Interactions

### Ownership Hierarchy
```
ODEImplicitStep
  ├─ solver (NewtonKrylov or LinearSolver)
  │   └─ linear_solver (if NewtonKrylov)
```

### Update Flow
```
1. User calls step.update(krylov_tolerance=1e-7, newton_tolerance=1e-6)
2. ODEImplicitStep.update delegates full dict to self.solver.update
3. NewtonKrylov.update:
   - Delegates full dict to self.linear_solver.update (recognizes krylov_tolerance)
   - Updates linear_solver_function in dict
   - Calls self.update_compile_settings (recognizes newton_tolerance)
   - Calls buffer_registry.update
4. Returns union of recognized parameters
```

### Property Access Flow
```
# Linear property
step.krylov_tolerance
  → solver.krylov_tolerance
    → linear_solver.krylov_tolerance (if NewtonKrylov)
    → linear_solver.compile_settings.krylov_tolerance (if LinearSolver)

# Nonlinear property  
step.newton_tolerance
  → getattr(solver, 'newton_tolerance', None)
    → value (if NewtonKrylov)
    → None (if LinearSolver)
```

## Files Requiring Changes

### Source Files

**1. src/cubie/integrators/matrix_free_solvers/linear_solver.py**
- Apply Pattern 1 to __init__ (lines ~167-175)
- Apply Pattern 2 to update method (lines ~529-558)

**2. src/cubie/integrators/matrix_free_solvers/newton_krylov.py**
- Apply Pattern 3: Add 3 pass-through properties after existing properties

**3. src/cubie/integrators/algorithms/ode_implicitstep.py**
- Apply Pattern 4 to properties (lines ~348-405)
- Simplify linear properties (direct access)
- Update nonlinear properties (getattr with None)

**4. Review other algorithm files (update methods only if they exist):**
- src/cubie/integrators/algorithms/backwards_euler.py
- src/cubie/integrators/algorithms/crank_nicolson.py
- src/cubie/integrators/algorithms/generic_dirk.py
- src/cubie/integrators/algorithms/generic_firk.py
- src/cubie/integrators/algorithms/generic_rosenbrock_w.py

Note: These files inherit from ODEImplicitStep, so they may not need changes unless they override update methods.

### Test Files

**5. tests/integrators/algorithms/instrumented/matrix_free_solvers.py**
- Mirror changes from linear_solver.py and newton_krylov.py
- Instrumented versions must match source signatures

**6. tests/integrators/algorithms/instrumented/*.py**
- Mirror any changes from source algorithm files
- Specific files:
  - backwards_euler.py
  - crank_nicolson.py
  - generic_dirk.py
  - generic_firk.py
  - generic_rosenbrock_w.py

## Edge Cases and Considerations

### Edge Case 1: Optional Type Annotations
ODEImplicitStep nonlinear properties should return Optional[float] or Optional[int] since LinearSolver doesn't have these attributes.

### Edge Case 2: Int Conversion
max_linear_iters and max_newton_iters may be stored as np.integer types; properties should cast to int.

### Edge Case 3: Precision Handling
krylov_tolerance returns precision-converted value from config; pass-through should maintain this.

### Edge Case 4: Property Naming
ODEImplicitStep has `linear_correction_type` property, NewtonKrylov pass-through should be `correction_type` to match LinearSolver.

### Edge Case 5: Instrumented Signatures
Instrumented versions may have additional logging parameters; preserve these while matching core signatures.

## Expected Behavior

### Before Changes
- LinearSolver uses ternary operators in __init__
- LinearSolver.update filters params before passing to update_compile_settings
- NewtonKrylov lacks pass-through properties for linear solver settings
- ODEImplicitStep uses hasattr checks and conditional delegation

### After Changes
- LinearSolver uses conditional kwargs in __init__ (matches NewtonKrylov)
- LinearSolver.update passes full dict to update_compile_settings
- NewtonKrylov has krylov_tolerance, max_linear_iters, correction_type properties
- ODEImplicitStep uses direct access for linear properties, getattr for nonlinear

### Behavioral Equivalence
All changes are refactors - external behavior should be identical:
- Same defaults applied
- Same parameters recognized
- Same cache invalidation triggers
- Same property values returned

## Integration Points

### With Buffer Registry
- buffer_registry.update called last in update methods
- Does not contribute to recognized parameter set
- Handles buffer location updates (e.g., preconditioned_vec_location)

### With CUDAFactory Base Class
- update_compile_settings called with full dict
- Returns set of recognized parameters
- Triggers cache invalidation if settings change

### With Config Classes
- LinearSolverConfig provides defaults for optional parameters
- NewtonKrylovConfig delegates to LinearSolverConfig for linear params
- Attrs validators enforce constraints

## Data Structures

### Recognized Parameter Sets
```python
# LinearSolver recognizes:
{'krylov_tolerance', 'max_linear_iters', 'correction_type', 
 'operator_apply', 'preconditioner', 'use_cached_auxiliaries',
 'preconditioned_vec_location', 'temp_location'}

# NewtonKrylov recognizes (including delegated):
{'newton_tolerance', 'max_newton_iters', 'newton_damping', 
 'newton_max_backtracks', 'residual_function',
 'delta_location', 'residual_location', 'residual_temp_location',
 'stage_base_bt_location'} | linear_solver_recognized
```

### Property Return Types
```python
# Linear properties (always available)
krylov_tolerance: float
max_linear_iters: int
correction_type: str

# Nonlinear properties (NewtonKrylov only)
newton_tolerance: Optional[float]
max_newton_iters: Optional[int]
newton_damping: Optional[float]
newton_max_backtracks: Optional[int]
```

## Dependencies

### Imports Required
No new imports needed; all patterns use existing imports.

### Module Dependencies
- linear_solver.py: No changes to imports
- newton_krylov.py: Already imports LinearSolver
- ode_implicitstep.py: Already imports Optional from typing

## Testing Considerations

### What to Test
This is a refactor - existing tests should pass unchanged:
- Solver initialization with various parameter combinations
- Update method parameter recognition
- Property access for both solver types
- Cache invalidation on parameter changes

### What NOT to Test
- New functionality (there is none)
- Different behavior (behavior should be identical)

### Validation Strategy
- Run existing test suite
- Verify no behavioral changes
- Check that recognized parameter sets remain the same
- Confirm cache invalidation triggers are preserved
