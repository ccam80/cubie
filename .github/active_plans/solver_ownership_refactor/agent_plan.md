# Solver Ownership Refactor - Agent Plan

## Overview

This plan details the structural changes needed to establish proper ownership of solver objects within the algorithm step hierarchy. The refactor corrects antipatterns introduced when solvers were converted from factory functions to CUDAFactory subclasses.

## Component Behavior Descriptions

### ODEImplicitStep Base Class

**Current Behavior:**
- Creates solver instances in __init__ with hardcoded default parameters
- Stores solvers as `_linear_solver` and `_newton_solver` instance variables
- build_implicit_helpers() updates solvers and returns newton solver device function
- build() calls build_implicit_helpers() then passes solver_fn to build_step()
- Properties expose solver parameters by delegating to solver instances

**Expected Behavior:**
- Accept solver parameters as optional kwargs in __init__ with defaults
- Pass these parameters to solver constructors during instantiation
- update() method accepts solver parameters and delegates to owned solvers
- build_step() signature removes solver_fn parameter
- build_step() accesses self._newton_solver.device_function or self._linear_solver.device_function directly
- Compile settings updated with solver device function to trigger rebuild when solver changes

**Key Methods:**
```python
def __init__(self, config, _controller_defaults, 
             krylov_tolerance=1e-3, max_linear_iters=100, 
             linear_correction_type="minimal_residual",
             newton_tolerance=1e-3, max_newton_iters=100,
             newton_damping=0.5, newton_max_backtracks=10):
    # Create LinearSolver with passed parameters
    self._linear_solver = LinearSolver(
        precision=config.precision,
        n=config.n,
        correction_type=linear_correction_type,
        krylov_tolerance=krylov_tolerance,
        max_linear_iters=max_linear_iters,
    )
    
    # Create NewtonKrylov with passed parameters
    self._newton_solver = NewtonKrylov(
        precision=config.precision,
        n=config.n,
        linear_solver=self._linear_solver,
        newton_tolerance=newton_tolerance,
        max_newton_iters=max_newton_iters,
        newton_damping=newton_damping,
        newton_max_backtracks=newton_max_backtracks,
    )

def update(self, updates_dict=None, silent=False, **kwargs):
    # Filter solver parameters and delegate
    # Update linear solver parameters
    # Update newton solver parameters
    # Call parent update for non-solver parameters
    # Return set of recognized parameters

def build(self):
    solver_fn = self.build_implicit_helpers()
    config = self.compile_settings
    
    # Update compile settings with solver device function
    # This ensures cache invalidates when solver changes
    self.update_compile_settings(
        solver_device_function=solver_fn
    )
    
    return self.build_step(
        # solver_fn parameter removed
        dxdt_fn=config.dxdt_function,
        observables_function=config.observables_function,
        driver_function=config.driver_function,
        numba_precision=config.numba_precision,
        n=config.n,
        n_drivers=config.n_drivers,
    )
```

### BackwardsEulerStep and Other Implicit Newton Steps

**Current Behavior:**
- __init__ accepts solver parameters but doesn't use them (passed to config then ignored)
- build_step() accepts solver_fn as parameter
- Calls solver_fn inside step device function

**Expected Behavior:**
- __init__ accepts solver parameters and passes to super().__init__()
- build_step() signature removes solver_fn parameter
- Accesses self._newton_solver.device_function inside build_step()
- Before calling device function, updates compile settings to include solver reference

**Signature Changes:**
```python
# OLD
def build_step(self, solver_fn, dxdt_fn, observables_function, ...):
    # solver_fn passed as parameter
    status = solver_fn(...)

# NEW  
def build_step(self, dxdt_fn, observables_function, ...):
    # Access owned solver's device function
    solver_fn = self._newton_solver.device_function
    status = solver_fn(...)
```

**Steps Affected:**
- BackwardsEulerStep
- BackwardsEulerPCStep
- CrankNicolsonStep
- DIRKStep
- FIRKStep

### GenericRosenbrockWStep

**Current Behavior:**
- Creates LinearSolver in __init__ (no NewtonKrylov since Rosenbrock is linearly implicit)
- build_step() accesses solver differently than Newton-based methods
- Has additional complexity for cached auxiliaries

**Expected Behavior:**
- Accept linear solver parameters in __init__ and pass to LinearSolver constructor
- build_implicit_helpers() updates and returns LinearSolver device function
- build_step() removes solver_fn parameter, accesses self._linear_solver.device_function
- update() method filters and delegates linear solver parameters

**Special Considerations:**
- Rosenbrock uses LinearSolver directly, not NewtonKrylov
- Cached auxiliaries variant has different solver signature
- Must handle both cached and non-cached code paths

### ImplicitStepConfig

**Current Behavior:**
- settings_dict property returns hardcoded solver defaults
- These defaults duplicate values that should live in solver __init__

**Expected Behavior:**
- settings_dict property REMOVES solver parameter defaults
- Only returns algorithm-specific configuration
- Solver parameters flow through constructor kwargs, not config dict

**Removed from settings_dict:**
```python
# REMOVE these lines from settings_dict property:
'krylov_tolerance': 1e-3,
'max_linear_iters': 100,
'linear_correction_type': "minimal_residual",
'newton_tolerance': 1e-3,
'max_newton_iters': 100,
'newton_damping': 0.5,
'newton_max_backtracks': 10,
```

### Algorithm Update Method Pattern

**Expected Behavior for All Implicit Steps:**

```python
def update(self, updates_dict=None, silent=False, **kwargs):
    """Update algorithm and owned solver parameters."""
    
    # Merge updates
    all_updates = {}
    if updates_dict:
        all_updates.update(updates_dict)
    all_updates.update(kwargs)
    
    # Separate solver parameters from algorithm parameters
    solver_params = {}
    for key in ['newton_tolerance', 'max_newton_iters', 'newton_damping',
                'newton_max_backtracks', 'krylov_tolerance', 'max_linear_iters',
                'linear_correction_type', 'correction_type', 'preconditioner_order']:
        if key in all_updates:
            solver_params[key] = all_updates.pop(key)
    
    # Update owned solver(s)
    recognized = set()
    if hasattr(self, '_newton_solver') and solver_params:
        recognized.update(self._newton_solver.update(solver_params, silent=True))
    elif hasattr(self, '_linear_solver') and solver_params:
        recognized.update(self._linear_solver.update(solver_params, silent=True))
    
    # Update buffer registry for any buffer location changes
    buffer_registry.update(self, updates_dict=all_updates, silent=True)
    
    # Update algorithm compile settings
    recognized.update(
        self.update_compile_settings(updates_dict=all_updates, silent=silent)
    )
    
    return recognized
```

## Architectural Changes Required

### Ownership Hierarchy

**Before:**
```
SingleIntegratorRunCore
├── OutputFunctions
├── StepController  
├── Algorithm (owns nothing)
└── IVPLoop
```

**After:**
```
SingleIntegratorRunCore
├── OutputFunctions
├── StepController
├── Algorithm
│   ├── NewtonKrylov (for implicit newton steps)
│   │   └── LinearSolver
│   └── LinearSolver (for rosenbrock steps)
└── IVPLoop
```

### Parameter Flow

**Current (Incorrect):**
```
User kwargs → Solver
             → algorithm_settings dict
             → Algorithm constructor
             → Config object (defaults set in settings_dict property)
             → Ignored!
```

**New (Correct):**
```
User kwargs → Solver
             → algorithm_settings dict
             → Algorithm constructor kwargs
             → super().__init__(kwargs for solvers)
             → ODEImplicitStep.__init__(solver kwargs)
             → LinearSolver.__init__(krylov params)
             → NewtonKrylov.__init__(newton params + linear solver)
```

### Cache Invalidation Chain

**Scenario:** User calls `solver.update(newton_tolerance=1e-5)`

**Invalidation Sequence:**
1. SingleIntegratorRunCore.update() receives {newton_tolerance: 1e-5}
2. Filters and calls algorithm.update({newton_tolerance: 1e-5})
3. Algorithm.update() calls self._newton_solver.update({newton_tolerance: 1e-5})
4. NewtonKrylov.update() invalidates its cache (settings changed)
5. Algorithm.build() is called on next device_function access
6. build() calls build_implicit_helpers() → gets NEW solver device function
7. build() calls self.update_compile_settings(solver_device_function=new_fn)
8. Compile settings comparison detects change → invalidates algorithm cache
9. SingleIntegratorRunCore rebuild triggered on next access
10. Full integration loop recompiled with new solver

## Integration Points with Current Codebase

### SingleIntegratorRunCore Integration

**No Changes Required:**
- Already calls `get_algorithm_step()` with algorithm_settings dict
- Already has update() method that delegates to components
- Already registers child allocators for algorithm buffers

**Parameter Flow Already Works:**
```python
# In SingleIntegratorRunCore.__init__:
algorithm_settings["n"] = n
algorithm_settings["driver_function"] = driver_function
self._algo_step = get_algorithm_step(
    precision=precision,
    settings=algorithm_settings,  # Contains solver params if provided
)
```

### get_algorithm_step Integration

**No Changes Required:**
- Already accepts arbitrary kwargs in settings dict
- Already filters parameters for specific algorithm type
- Already passes filtered kwargs to algorithm constructor
- Solver parameters already in ALL_ALGORITHM_STEP_PARAMETERS

### Buffer Registry Integration

**No Changes Required:**
- Solvers already register buffers in their __init__
- get_child_allocators() already establishes parent-child relationships
- Algorithm already calls get_child_allocators for newton solver
- Hierarchical allocation already works: algorithm → newton → linear

### NewtonKrylov and LinearSolver Integration

**No Changes Required:**
- Already accept parameters in __init__
- Already implement update() methods
- Already register buffers with buffer_registry
- Already implement CUDAFactory pattern correctly
- NewtonKrylov already accepts linear_solver parameter

## Data Structures and Purposes

### Solver Parameter Sets

**Newton-Krylov Parameters:**
- newton_tolerance: Convergence threshold for residual norm
- max_newton_iters: Maximum iterations before failure
- newton_damping: Backtracking step size reduction factor
- newton_max_backtracks: Maximum backtracking attempts

**Linear Solver Parameters:**
- krylov_tolerance: Convergence threshold for linear solve
- max_linear_iters: Maximum Krylov iterations
- linear_correction_type / correction_type: "steepest_descent" or "minimal_residual"

**Preconditioner Parameters:**
- preconditioner_order: Neumann series truncation order

### Config Classes

**BaseStepConfig:** Algorithm-specific compile settings (precision, n, dxdt_function, etc.)

**ImplicitStepConfig:** Adds beta, gamma, M, preconditioner_order (NOT solver parameters)

**LinearSolverConfig:** Compile settings for linear solver (precision, n, operator_apply, etc.)

**NewtonKrylovConfig:** Compile settings for Newton solver (precision, n, residual_function, linear_solver, etc.)

## Dependencies and Imports

### No New Dependencies Required

All necessary imports already exist:
- LinearSolver already imported in ode_implicitstep.py
- NewtonKrylov already imported in ode_implicitstep.py
- buffer_registry already imported in algorithm files
- split_applicable_settings available for parameter filtering

### Import Locations

```python
# ode_implicitstep.py (already has these)
from cubie.integrators.matrix_free_solvers.linear_solver import LinearSolver
from cubie.integrators.matrix_free_solvers.newton_krylov import NewtonKrylov
from cubie.buffer_registry import buffer_registry
```

## Edge Cases to Consider

### Edge Case 1: User Provides No Solver Parameters

**Handling:** Defaults in algorithm __init__ signature provide fallback values
**Example:**
```python
def __init__(self, ..., newton_tolerance=1e-3, ...):
    # Uses default if not provided
```

### Edge Case 2: User Provides Only Some Solver Parameters

**Handling:** Kwargs with defaults handle partial specification
**Example:**
```python
# User provides:
Solver(system, method="backwards_euler", newton_tolerance=1e-5)
# Other params use defaults:
# max_newton_iters=100, newton_damping=0.5, etc.
```

### Edge Case 3: Parameter Update After Initialization

**Handling:** Update method filters and delegates to owned solvers
**Example:**
```python
solver = Solver(system, method="backwards_euler")
solver.update(newton_tolerance=1e-6)  # Delegated to _newton_solver
```

### Edge Case 4: Invalid Parameter Names

**Handling:** split_applicable_settings catches unrecognized parameters
**Example:**
```python
# Typo in parameter name
Solver(system, newtn_tolerance=1e-5)  # Raises error if silent=False
```

### Edge Case 5: Rosenbrock (No NewtonKrylov)

**Handling:** Rosenbrock only creates LinearSolver, not NewtonKrylov
**Example:**
```python
class GenericRosenbrockWStep(ODEImplicitStep):
    def __init__(self, ..., krylov_tolerance=1e-6, ...):
        # Only create linear solver
        self._linear_solver = LinearSolver(...)
        # No self._newton_solver created
```

### Edge Case 6: Cached vs Non-Cached Auxiliaries (Rosenbrock)

**Handling:** LinearSolver.build() already handles this via use_cached_auxiliaries flag
**Example:**
```python
# LinearSolver compiles different signatures based on config
if use_cached_auxiliaries:
    # Signature includes cached_aux parameter
else:
    # Standard signature
```

### Edge Case 7: Multiple Updates in Sequence

**Handling:** Each update invalidates cache; rebuild happens on next device_function access
**Example:**
```python
solver.update(newton_tolerance=1e-5)  # Invalidates
solver.update(krylov_tolerance=1e-7)  # Invalidates again  
result = solver.solve()  # Rebuild happens here
```

### Edge Case 8: Step Controller Parameter Collision

**Handling:** Step controller params filtered separately from solver params
**Example:**
```python
# Both dt and newton_tolerance provided
Solver(system, dt=1e-3, newton_tolerance=1e-5)
# dt → step_control_settings
# newton_tolerance → algorithm_settings
```

## Expected Interactions Between Components

### Initialization Sequence

```
1. User creates Solver with solver parameters
2. Solver creates BatchSolverKernel with algorithm_settings
3. BatchSolverKernel creates SingleIntegratorRunCore
4. SingleIntegratorRunCore calls get_algorithm_step with settings
5. get_algorithm_step filters parameters and creates algorithm instance
6. Algorithm __init__ receives filtered solver parameters
7. Algorithm calls super().__init__ passing solver kwargs
8. ODEImplicitStep.__init__ creates LinearSolver with krylov params
9. ODEImplicitStep.__init__ creates NewtonKrylov with newton params + linear solver
10. Solvers register buffers with buffer_registry
11. Algorithm build() deferred until first device_function access
```

### Build Sequence

```
1. User calls solver.solve()
2. BatchSolverKernel.device_function accessed
3. SingleIntegratorRunCore.device_function accessed
4. Algorithm.device_function accessed → triggers build()
5. Algorithm.build() calls build_implicit_helpers()
6. build_implicit_helpers() updates solver with ODE functions
7. build_implicit_helpers() returns solver.device_function
8. Algorithm.build() updates compile_settings with solver_device_function
9. Algorithm.build() calls build_step() WITHOUT solver_fn parameter
10. build_step() accesses self._newton_solver.device_function
11. Step device function compiled with solver reference in closure
12. Compile settings compared; cache marked valid
```

### Update Sequence

```
1. User calls solver.update(newton_tolerance=1e-5)
2. Solver.update() filters and delegates to single_integrator.update()
3. SingleIntegratorRunCore.update() delegates to algorithm.update()
4. Algorithm.update() extracts solver parameters
5. Algorithm.update() calls self._newton_solver.update({newton_tolerance: 1e-5})
6. NewtonKrylov.update() updates compile_settings
7. NewtonKrylov cache invalidated (settings changed)
8. Algorithm cache REMAINS VALID (its settings unchanged)
9. Next device_function access:
   - Algorithm.build() called
   - Gets new solver.device_function (triggers NewtonKrylov rebuild)
   - Updates own compile_settings with new solver reference
   - Detects change → invalidates algorithm cache
   - Rebuilds step with new solver
```

## Behavioral Contracts

### Algorithm Classes Must:
- Accept solver parameters as optional kwargs with defaults
- Create owned solver instances in __init__
- Pass solver parameters to owned solver constructors  
- Implement update() method that filters and delegates solver parameters
- Access owned solver.device_function in build_step(), not accept as parameter
- Update compile_settings with solver device function reference

### Solver Classes Must:
- Accept parameters in __init__ with defaults
- Store parameters in config attrs class
- Implement update() method that invalidates cache on change
- Register buffers with buffer_registry
- Return device function from device_function property

### Config Classes Must NOT:
- Store solver instances (they're frozen attrs classes)
- Return solver default values from settings_dict property
- Handle solver parameter initialization

### Update Methods Must:
- Accept updates_dict and **kwargs
- Filter applicable parameters
- Delegate to owned child objects
- Return set of recognized parameter names
- Support silent=True for suppressing warnings

## Compile Settings Update Pattern

### Problem to Solve
When solver parameters change, the algorithm must detect this and rebuild even though the algorithm's own parameters haven't changed.

### Solution
Algorithm compile settings must include a reference to the solver device function:

```python
def build(self):
    solver_fn = self.build_implicit_helpers()
    
    # Update compile settings to include solver reference
    # This ensures cache invalidates when solver device function changes
    self.update_compile_settings(
        solver_device_function=solver_fn
    )
    
    # Now build step (solver_fn no longer a parameter)
    return self.build_step(
        dxdt_fn=...,
        # solver_fn removed from parameters
    )
```

**Why This Works:**
- Solver.device_function is a Callable object
- When solver rebuilds (due to parameter change), new Callable returned
- Algorithm compile_settings comparison detects new Callable ≠ old Callable
- Algorithm cache automatically invalidated
- Next access rebuilds algorithm with new solver

### Alternative Considered: Manual Invalidation

```python
# REJECTED APPROACH
def update(self, ...):
    if solver_params:
        self._newton_solver.update(solver_params)
        self.invalidate_cache()  # Manual invalidation
```

**Why Rejected:**
- Requires manual cache management
- Error-prone (easy to forget)
- Doesn't leverage CUDAFactory automatic invalidation
- Violates separation of concerns

## Testing Considerations

### Existing Tests Should Continue Working

**Reason:** Parameter flow pattern matches current usage, just internal routing changes

**Example:**
```python
# Test code (unchanged)
solver = Solver(system, method="backwards_euler", newton_tolerance=1e-5)
result = solver.solve()

# Internally:
# OLD: parameter ignored, default used
# NEW: parameter flows to NewtonKrylov constructor
# Both: test passes, but NEW actually respects the parameter
```

### New Tests Recommended

1. **Parameter Flow Test:** Verify solver params reach solver objects
2. **Update Test:** Verify solver.update() with solver params invalidates correctly
3. **Multiple Updates Test:** Verify sequential updates work correctly
4. **Cache Invalidation Test:** Verify rebuild triggered by solver param change

### Instrumented Tests

**Must Update:** All instrumented versions must mirror source changes
**Files:**
- `tests/integrators/algorithms/instrumented/backwards_euler.py`
- `tests/integrators/algorithms/instrumented/crank_nicolson.py`
- `tests/integrators/algorithms/instrumented/generic_dirk.py`
- `tests/integrators/algorithms/instrumented/generic_firk.py`
- `tests/integrators/algorithms/instrumented/generic_rosenbrock_w.py`
- `tests/integrators/algorithms/instrumented/backwards_euler_predict_correct.py`

**Changes:**
- build_step() signature: remove solver_fn parameter
- Access self._newton_solver.device_function or self._linear_solver.device_function
- Same pattern as source files

## Implementation Order

### Phase 1: Base Class Changes
1. Modify ODEImplicitStep.__init__ to accept solver kwargs
2. Modify ODEImplicitStep.update to filter and delegate solver params
3. Modify ImplicitStepConfig.settings_dict to remove solver defaults
4. Add compile settings update in ODEImplicitStep.build()

### Phase 2: Algorithm Concrete Classes  
5. Modify BackwardsEulerStep.__init__ to accept and pass solver kwargs
6. Modify BackwardsEulerStep.build_step to remove solver_fn parameter
7. Repeat for BackwardsEulerPCStep
8. Repeat for CrankNicolsonStep
9. Repeat for DIRKStep
10. Repeat for FIRKStep

### Phase 3: Rosenbrock Special Case
11. Modify GenericRosenbrockWStep.__init__ for linear solver kwargs
12. Modify GenericRosenbrockWStep.build_step to remove solver_fn parameter
13. Handle cached/non-cached auxiliary variants

### Phase 4: Instrumented Tests
14. Update all instrumented algorithm files to match source changes

### Phase 5: Verification
15. Run test suite to verify no regressions
16. Verify parameter flow works end-to-end
17. Verify cache invalidation triggers rebuilds

## Success Criteria

### Functional Requirements Met
- ✓ Implicit steps own NewtonKrylov solvers
- ✓ Rosenbrock steps own LinearSolver
- ✓ NewtonKrylov owns LinearSolver
- ✓ Parameters flow from user to solver objects
- ✓ Update methods delegate to owned solvers
- ✓ Cache invalidates on parameter changes
- ✓ build_step() no longer accepts solver_fn

### Code Quality Requirements Met
- ✓ No solver defaults in config.settings_dict
- ✓ Clear ownership hierarchy
- ✓ Consistent with step controller pattern
- ✓ All implicit algorithms follow same pattern
- ✓ Instrumented tests mirror source changes

### No Regressions
- ✓ Existing tests pass
- ✓ Public API unchanged
- ✓ Parameter names unchanged
- ✓ Default values unchanged (just location)
- ✓ Buffer allocation unchanged
