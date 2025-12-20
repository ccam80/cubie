# Algorithm Pattern Replication - Agent Implementation Plan

## Overview

This plan ensures all algorithm step implementations follow the patterns established in `ODEImplicitStep` and `NewtonKrylov`. The work is divided into six distinct modification areas.

## Component Descriptions

### 1. NewtonKrylov Typo Fix
**File**: `src/cubie/integrators/matrix_free_solvers/newton_krylov.py`

**Current Behavior**: Line 248 uses `self.nn` which doesn't exist

**Expected Behavior**: Line 248 should use `self.n` to reference vector size

**Change Required**: Single character replacement

### 2. BackwardsEulerStep __init__ Refactor
**File**: `src/cubie/integrators/algorithms/backwards_euler.py`

**Current Behavior**: 
- __init__ has hardcoded defaults (lines 38, 41-44)
- Passes all values directly to super().__init__()

**Expected Behavior**:
- Parameters use Optional[type] = None pattern
- Build kwargs dict conditionally
- Only pass non-None values to super()

**Parameters to Change**:
- preconditioner_order (default in ImplicitStepConfig)
- krylov_tolerance (default in LinearSolverConfig)
- max_linear_iters (default in LinearSolverConfig)
- linear_correction_type (default in LinearSolverConfig)
- newton_tolerance (default in NewtonKrylovConfig)
- max_newton_iters (default in NewtonKrylovConfig)
- newton_damping (default in NewtonKrylovConfig)
- newton_max_backtracks (default in NewtonKrylovConfig)

### 3. BackwardsEulerStep build_step Signature
**File**: `src/cubie/integrators/algorithms/backwards_euler.py`

**Current Signature** (lines 106-113):
```python
def build_step(
    self,
    dxdt_fn: Callable,
    observables_function: Callable,
    driver_function: Optional[Callable],
    numba_precision: type,
    n: int,
    n_drivers: int,
) -> StepCache:
```

**Expected Signature**:
```python
def build_step(
    self,
    dxdt_fn: Callable,
    observables_function: Callable,
    driver_function: Optional[Callable],
    solver_function: Callable,  # <-- Add this
    numba_precision: type,
    n: int,
    n_drivers: int,
) -> StepCache:
```

**Usage Change**: Replace `solver_fn = self.solver.device_function` (line 149) with `solver_fn = solver_function`

### 4. BackwardsEulerPCStep build_step Signature
**File**: `src/cubie/integrators/algorithms/backwards_euler_predict_correct.py`

**Current Signature** (lines 15-22): Missing solver_function parameter

**Expected Behavior**: Same changes as BackwardsEulerStep build_step

**Usage Change**: Replace access to `self._newton_solver.device_function` (line 58) with `solver_function` parameter

### 5. CrankNicolsonStep Refactor
**File**: `src/cubie/integrators/algorithms/crank_nicolson.py`

**Changes Required**:
1. __init__ method: Apply Optional=None pattern (same as BackwardsEulerStep)
2. build_step signature: Add solver_function parameter
3. build_step usage: Use solver_function instead of self.solver.device_function

### 6. DIRKStep Refactor
**File**: `src/cubie/integrators/algorithms/generic_dirk.py`

**Additional Complexity**: Has custom buffers for multi-stage algorithm

**Changes Required**:
1. __init__ method: Apply Optional=None pattern
   - Additional parameters: stage_increment_location, stage_base_location, accumulator_location
2. register_buffers() method: Extract buffer registration
3. build_step signature: Add solver_function parameter
4. build_step usage: Use solver_function instead of self.solver.device_function

**Buffer Registration Pattern**:
```python
def register_buffers(self) -> None:
    """Register buffers according to locations in compile settings."""
    config = self.compile_settings
    precision = config.precision
    
    buffer_registry.register(
        'dirk_stage_increment',
        self,
        self.n,
        config.stage_increment_location,
        precision=precision
    )
    # ... more registrations
```

### 7. FIRKStep Refactor
**File**: `src/cubie/integrators/algorithms/generic_firk.py`

**Similar to DIRKStep** - has custom buffers for coupled stage system

**Changes Required**:
1. __init__ method: Apply Optional=None pattern
   - Additional parameters: stage_increment_location, stage_driver_stack_location, stage_state_location
2. register_buffers() method: Extract buffer registration
3. build_step signature: Add solver_function parameter
4. build_step usage: Use solver_function instead of self.solver.device_function

### 8. GenericRosenbrockWStep Refactor
**File**: `src/cubie/integrators/algorithms/generic_rosenbrock_w.py`

**Special Case**: Uses LinearSolver instead of NewtonKrylov

**Changes Required**:
1. __init__ method: Apply Optional=None pattern
   - Rosenbrock-specific parameters: tableau, driver_del_t
   - Buffer location parameters: stage_increment_location, etc.
2. register_buffers() method: Extract buffer registration
3. **Remove build() override entirely** (lines 330-364)
4. build_step signature: Add solver_function parameter (note: already has driver_del_t)
5. build_step usage: Use solver_function instead of self.solver.device_function

**build_step Signature** should become:
```python
def build_step(
    self,
    dxdt_fn: Callable,
    observables_function: Callable,
    driver_function: Optional[Callable],
    solver_function: Callable,  # <-- Add this
    driver_del_t: Callable,      # Rosenbrock-specific, already present
    numba_precision: type,
    n: int,
    n_drivers: int,
) -> StepCache:
```

## Integration Points

### With ODEImplicitStep
- All algorithm steps inherit from ODEImplicitStep
- Must maintain compatibility with base class build() method
- __init__ calls super().__init__() with conditional kwargs

### With NewtonKrylov/LinearSolver
- Algorithm steps own a solver instance (self.solver)
- Solver device function obtained via self.solver.device_function
- After changes, device function passed explicitly via build_step parameter

### With Buffer Registry
- Custom buffers registered via buffer_registry.register()
- Allocators obtained via buffer_registry.get_allocator()
- Child allocators for nested solvers via buffer_registry.get_child_allocators()

## Data Structures

### Config Classes
- **ImplicitStepConfig**: Has preconditioner_order default
- **LinearSolverConfig**: Has krylov_tolerance, max_linear_iters, correction_type defaults
- **NewtonKrylovConfig**: Has newton_tolerance, max_newton_iters, newton_damping, newton_max_backtracks defaults
- **Algorithm-specific configs** (DIRKStepConfig, FIRKStepConfig, etc.): Have buffer location defaults

### Kwargs Pattern
```python
# Build kwargs conditionally
solver_kwargs = {}
if preconditioner_order is not None:
    solver_kwargs['preconditioner_order'] = preconditioner_order
if krylov_tolerance is not None:
    solver_kwargs['krylov_tolerance'] = krylov_tolerance
# ... etc

super().__init__(config, defaults, **solver_kwargs)
```

## Dependencies and Imports

No new imports required. All necessary imports already present:
- typing.Optional
- typing.Callable
- attrs for config classes
- buffer_registry
- Base classes (ODEImplicitStep, etc.)

## Expected Interactions

### Initialization Flow
1. User creates algorithm instance with optional parameters
2. Algorithm __init__ builds config with required parameters
3. Algorithm __init__ builds kwargs dict from optional parameters
4. Algorithm __init__ calls super().__init__(config, defaults, **kwargs)
5. ODEImplicitStep.__init__ creates solver with parameters from kwargs

### Build Flow
1. User calls algorithm.build() or accesses cached output
2. ODEImplicitStep.build() calls build_implicit_helpers()
3. build_implicit_helpers() updates solver and stores solver_function in config
4. ODEImplicitStep.build() extracts solver_function from config
5. ODEImplicitStep.build() calls algorithm.build_step() with solver_function
6. algorithm.build_step() uses solver_function parameter directly

## Edge Cases

### Case 1: User Passes All Parameters Explicitly
**Behavior**: All values go into kwargs dict, override config defaults
**Handling**: Normal flow, no special handling needed

### Case 2: User Passes No Optional Parameters
**Behavior**: kwargs dict is empty or contains only required values
**Handling**: Config defaults apply, no special handling needed

### Case 3: Rosenbrock with LinearSolver
**Behavior**: Rosenbrock uses LinearSolver not NewtonKrylov
**Handling**: Same pattern applies, solver_function comes from LinearSolver.device_function

### Case 4: Build Called Multiple Times
**Behavior**: Should use cached result if config unchanged
**Handling**: CUDAFactory caching handles this, no change needed

### Case 5: Buffer Locations Changed After Init
**Behavior**: Buffer registry should handle re-registration
**Handling**: register_buffers() called when needed, registry manages lifecycle

## Implementation Order

1. Fix typo in newton_krylov.py (independent, quick win)
2. Refactor BackwardsEulerStep (simplest algorithm, template for others)
3. Refactor CrankNicolsonStep (similar to BackwardsEuler)
4. Refactor BackwardsEulerPCStep (inherits from BackwardsEuler)
5. Refactor DIRKStep (adds buffer registration pattern)
6. Refactor FIRKStep (similar to DIRK)
7. Refactor GenericRosenbrockWStep (most complex, has special build_step signature)

## Testing Considerations

### Existing Tests Should Pass
- No behavioral changes, only refactoring
- Existing integration tests validate correctness
- Unit tests for individual algorithms remain valid

### Areas to Validate
- Algorithm initialization with various parameter combinations
- Build process completes successfully
- Solver device functions are correctly passed
- Buffer registration works correctly
- Config defaults apply when parameters omitted

### Test Files to Update
Mirror source changes in instrumented test versions:
- tests/integrators/algorithms/instrumented/backwards_euler.py
- tests/integrators/algorithms/instrumented/crank_nicolson.py
- tests/integrators/algorithms/instrumented/generic_dirk.py
- tests/integrators/algorithms/instrumented/generic_firk.py
- tests/integrators/algorithms/instrumented/generic_rosenbrock_w.py

## Success Criteria

1. All algorithm __init__ methods use Optional=None pattern
2. All build_step signatures include solver_function parameter
3. No algorithm accesses self.solver.device_function in build_step
4. Rosenbrock build() override is removed
5. Algorithm steps with custom buffers have register_buffers() methods
6. Typo in newton_krylov.py is fixed
7. All existing tests pass
8. No new linter warnings
