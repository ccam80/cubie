# Solver Ownership Refinement - Agent Implementation Plan

This document provides detailed technical specifications for implementing the solver ownership architectural changes. It describes component behavior, integration points, and implementation requirements.

## Component Descriptions

### ODEImplicitStep Class

**Location:** `src/cubie/integrators/algorithms/ode_implicitstep.py`

**Current Behavior:**
- Owns both `_newton_solver` and `_linear_solver` attributes
- Splits update parameters manually
- Always invalidates cache after solver updates

**Expected Behavior:**
- Owns single `self.solver` attribute (type determined by `solver_type` parameter)
- Delegates all updates to `self.solver` without splitting
- Conditionally invalidates cache based on `self.solver.cache_valid`

**Constructor Changes:**
- Add `solver_type: str = 'newton'` parameter
- Validate `solver_type in ['newton', 'linear']`
- Create LinearSolver instance first
- If `solver_type == 'newton'`: pass LinearSolver to NewtonKrylov constructor, store NewtonKrylov in `self.solver`
- If `solver_type == 'linear'`: store LinearSolver directly in `self.solver`

**Update Method Changes:**
- Accept full `updates_dict` parameter
- Call `recognized = self.solver.update(updates_dict, silent=True)` with full dict
- Check `if not self.solver.cache_valid: self.invalidate_cache()`
- Delegate buffer registry updates
- Delegate algorithm compile settings updates
- Return accumulated recognized set

**Build Method Changes:**
- Access solver via `self.solver.device_function` (not `self._newton_solver`)
- No changes to build_implicit_helpers logic

**Property Changes:**
- Update solver-related properties to access `self.solver` instead of `_newton_solver` or `_linear_solver`
- Properties: `newton_tolerance`, `max_newton_iters`, `newton_damping`, `newton_max_backtracks`, `krylov_tolerance`, `max_linear_iters`, `linear_correction_type`
- For Newton-based solvers: forward to `self.solver.<property>`
- For linear-only solvers: newton properties should raise AttributeError or return None

**Removed Methods:**
- Delete `_split_solver_params()` method entirely

### GenericRosenbrockWStep Class

**Location:** `src/cubie/integrators/algorithms/generic_rosenbrock_w.py`

**Current Behavior:**
- Calls `BaseAlgorithmStep.__init__` directly, skipping ODEImplicitStep
- Creates only `_linear_solver`
- Has custom update() method splitting linear parameters

**Expected Behavior:**
- Calls `super().__init__(..., solver_type='linear')`
- No longer creates `_linear_solver` directly (parent class creates it)
- Access linear solver via `self.solver` (not `self._linear_solver`)
- No custom update() method needed (inherits from ODEImplicitStep)
- Sets `use_cached_auxiliaries=True` on LinearSolver

**Constructor Changes:**
- Remove direct `BaseAlgorithmStep.__init__()` call
- Call `super().__init__(config, controller_defaults, solver_type='linear', krylov_tolerance=..., max_linear_iters=..., linear_correction_type=...)`
- Remove manual `_linear_solver` creation
- After super().__init__, configure cached auxiliaries: `self.solver.update(use_cached_auxiliaries=True)`

**Update Method Changes:**
- Delete custom `update()` method entirely
- Inherit ODEImplicitStep.update() behavior

**Build Methods Changes:**
- Change `self._linear_solver` references to `self.solver`
- In `build_implicit_helpers()`: update `self.solver.update_compile_settings()` instead of `self._linear_solver.update_compile_settings()`
- In `build_step()`: access `linear_solver = self.solver.device_function`

### NewtonKrylov Class

**Location:** `src/cubie/integrators/matrix_free_solvers/newton_krylov.py`

**Current Behavior:**
- Receives LinearSolver instance as constructor parameter
- Has update() method that calls update_compile_settings()

**Expected Behavior:**
- Continue receiving LinearSolver instance as constructor parameter
- update() method accepts full updates dict, delegates to linear_solver
- Returns accumulated recognized parameters

**Constructor Changes:**
- No changes to signature (already receives linear_solver parameter)

**Update Method Changes:**
- Accept full `updates_dict` with all parameters
- Extract Newton-specific parameters: `newton_tolerance`, `max_newton_iters`, `newton_damping`, `newton_max_backtracks`
- Call `linear_recognized = self.linear_solver.update(updates_dict, silent=True)` with full dict
- Call `newton_recognized = self.update_compile_settings(newton_params, silent=True)` with Newton-only params
- Return `linear_recognized | newton_recognized`
- Update buffer registry for location changes

**No Changes:**
- Build method unchanged
- Buffer registration unchanged
- Device function compilation unchanged

### LinearSolver Class

**Location:** `src/cubie/integrators/matrix_free_solvers/linear_solver.py`

**Current Behavior:**
- Has update() method that calls update_compile_settings()
- Does not support cached auxiliaries parameter

**Expected Behavior:**
- update() method accepts full updates dict
- Extracts relevant parameters, ignores irrelevant ones
- Returns set of recognized parameters
- Supports `use_cached_auxiliaries` parameter

**Update Method Changes:**
- Accept full `updates_dict` with all parameters
- Extract LinearSolver-specific parameters: `krylov_tolerance`, `max_linear_iters`, `linear_correction_type`, `correction_type`, `operator_apply`, `preconditioner`, `use_cached_auxiliaries`
- Call `self.update_compile_settings(linear_params, silent=silent)` with relevant params only
- Return set of recognized parameter names
- Update buffer registry for location changes

**No Changes:**
- Constructor unchanged
- Build method unchanged (already supports use_cached_auxiliaries)
- Buffer registration unchanged

## Architectural Changes

### Solver Type Parameter Flow

1. User creates algorithm step (e.g., BackwardsEulerStep)
2. BackwardsEulerStep calls `super().__init__()` (ODEImplicitStep) with default `solver_type='newton'`
3. ODEImplicitStep.__init__:
   - Creates LinearSolver
   - Checks solver_type
   - If 'newton': creates NewtonKrylov with LinearSolver, stores in self.solver
   - If 'linear': stores LinearSolver directly in self.solver

4. Rosenbrock overrides this:
   - GenericRosenbrockWStep calls `super().__init__(..., solver_type='linear')`
   - ODEImplicitStep creates LinearSolver, stores directly in self.solver
   - Rosenbrock configures: `self.solver.update(use_cached_auxiliaries=True)`

### Update Delegation Chain

```
Step.update({'newton_tolerance': 1e-4, 'krylov_tolerance': 1e-6, 'some_algo_param': 10})
│
├─> self.solver.update({'newton_tolerance': 1e-4, 'krylov_tolerance': 1e-6, 'some_algo_param': 10})
│   │
│   ├─> [if NewtonKrylov]
│   │   ├─> Extract newton_* params → {'newton_tolerance': 1e-4}
│   │   ├─> self.update_compile_settings({'newton_tolerance': 1e-4}) → {'newton_tolerance'}
│   │   ├─> self.linear_solver.update({'newton_tolerance': 1e-4, 'krylov_tolerance': 1e-6, 'some_algo_param': 10})
│   │   │   ├─> Extract krylov_* params → {'krylov_tolerance': 1e-6}
│   │   │   ├─> self.update_compile_settings({'krylov_tolerance': 1e-6}) → {'krylov_tolerance'}
│   │   │   └─> return {'krylov_tolerance', 'correction_type'}
│   │   └─> return {'newton_tolerance', 'krylov_tolerance', 'correction_type'}
│   │
│   └─> [if LinearSolver]
│       ├─> Extract krylov_* params → {'krylov_tolerance': 1e-6}
│       ├─> self.update_compile_settings({'krylov_tolerance': 1e-6}) → {'krylov_tolerance'}
│       └─> return {'krylov_tolerance', 'correction_type'}
│
├─> recognized = {'newton_tolerance', 'krylov_tolerance', 'correction_type'}
├─> Check self.solver.cache_valid
├─> If invalid: self.invalidate_cache()
├─> buffer_registry.update(self, {'some_algo_param': 10})
├─> self.update_compile_settings({'some_algo_param': 10})
└─> return recognized | buffer_recognized | algo_recognized
```

### Cache Invalidation Logic

```python
# In ODEImplicitStep.update()
recognized = set()

# Delegate to solver with full dict
solver_recognized = self.solver.update(updates_dict, silent=True)
recognized.update(solver_recognized)

# Check if solver cache was invalidated
if not self.solver.cache_valid:
    # Solver params changed, invalidate step cache
    self.invalidate_cache()

# Continue with buffer registry and algorithm settings
buffer_recognized = buffer_registry.update(self, updates_dict, silent=True)
recognized.update(buffer_recognized)

algo_recognized = self.update_compile_settings(updates_dict, silent=silent)
recognized.update(algo_recognized)

return recognized
```

## Integration Points

### With Existing Code

**BaseAlgorithmStep:**
- ODEImplicitStep continues to inherit from BaseAlgorithmStep
- No changes to BaseAlgorithmStep itself
- Step controller defaults continue to work

**Buffer Registry:**
- Step continues to register buffers for algorithm-specific arrays
- Solvers register their own buffers
- No changes to buffer registry itself

**Build Chain:**
- Step.build() calls build_implicit_helpers()
- build_implicit_helpers() configures solvers, returns device function
- build_step() accesses `self.solver.device_function` for compilation
- No changes to device function signatures

**Instrumented Tests:**
- All algorithm step changes must be mirrored in `tests/integrators/algorithms/instrumented/`
- Instrumented versions add logging, but keep same solver ownership pattern
- Device function signatures unchanged

### Parameter Namespace

**Newton Parameters (recognized by NewtonKrylov):**
- `newton_tolerance`: Convergence threshold for residual norm
- `max_newton_iters`: Maximum Newton iterations
- `newton_damping`: Backtracking damping factor
- `newton_max_backtracks`: Maximum backtracking steps
- `residual_function`: Device function for residual evaluation

**Linear Solver Parameters (recognized by LinearSolver):**
- `krylov_tolerance`: Convergence threshold for Krylov iteration
- `max_linear_iters`: Maximum Krylov iterations
- `linear_correction_type` or `correction_type`: 'steepest_descent' or 'minimal_residual'
- `operator_apply`: Device function for operator application
- `preconditioner`: Device function for preconditioning
- `use_cached_auxiliaries`: Boolean flag for cached Jacobian data

**Algorithm Parameters (recognized by Step):**
- All parameters in step config (beta, gamma, M, preconditioner_order, etc.)
- Buffer location parameters (via buffer_registry)

### Import Organization

**Move all imports to module top:**

`ode_implicitstep.py`:
```python
from abc import abstractmethod
from typing import Callable, Optional, Union, Set

import attrs
import numpy as np
import sympy as sp

from cubie._utils import inrangetype_validator
from cubie.integrators.matrix_free_solvers.linear_solver import LinearSolver
from cubie.integrators.matrix_free_solvers.newton_krylov import NewtonKrylov
from cubie.integrators.algorithms.base_algorithm_step import (
    BaseAlgorithmStep,
    BaseStepConfig,
    StepCache, 
    StepControlDefaults,
)
from cubie.buffer_registry import buffer_registry
```

`generic_rosenbrock_w.py`:
```python
from typing import Callable, Optional, Tuple, Set

import attrs
import numpy as np
from numba import cuda, int32

from cubie._utils import PrecisionDType
from cubie.integrators.algorithms.base_algorithm_step import (
    StepCache,
    StepControlDefaults,
)
from cubie.integrators.algorithms.ode_implicitstep import (
    ImplicitStepConfig,
    ODEImplicitStep,
)
from cubie.integrators.algorithms.generic_rosenbrockw_tableaus import (
    DEFAULT_ROSENBROCK_TABLEAU,
    RosenbrockTableau,
)
from cubie.buffer_registry import buffer_registry
```

## Data Structures

### ODEImplicitStep Attributes

**Before:**
```python
self._linear_solver: LinearSolver
self._newton_solver: NewtonKrylov
```

**After:**
```python
self.solver: Union[NewtonKrylov, LinearSolver]
```

### NewtonKrylov.update() Signature

**Before:**
```python
def update(self, updates_dict=None, silent=False, **kwargs) -> Set[str]:
    buffer_registry.update(self, updates_dict=updates_dict, silent=True, **kwargs)
    return self.update_compile_settings(
        updates_dict=updates_dict, silent=silent, **kwargs
    )
```

**After:**
```python
def update(self, updates_dict=None, silent=False, **kwargs) -> Set[str]:
    # Merge updates
    all_updates = {}
    if updates_dict:
        all_updates.update(updates_dict)
    all_updates.update(kwargs)
    
    if not all_updates:
        return set()
    
    # Extract Newton parameters
    newton_keys = {
        'newton_tolerance', 'max_newton_iters',
        'newton_damping', 'newton_max_backtracks',
        'residual_function'
    }
    newton_params = {k: all_updates[k] for k in newton_keys & all_updates.keys()}
    
    # Delegate to linear solver with full dict
    recognized = set()
    linear_recognized = self.linear_solver.update(all_updates, silent=True)
    recognized.update(linear_recognized)
    
    # Update Newton parameters
    if newton_params:
        buffer_registry.update(self, updates_dict=newton_params, silent=True)
        newton_recognized = self.update_compile_settings(
            updates_dict=newton_params, silent=True
        )
        recognized.update(newton_recognized)
    
    # Update buffer registry for other params
    buffer_registry.update(self, updates_dict=all_updates, silent=True)
    
    return recognized
```

### LinearSolver.update() Signature

**Before:**
```python
def update(self, updates_dict=None, silent=False, **kwargs) -> Set[str]:
    buffer_registry.update(self, updates_dict=updates_dict, silent=True, **kwargs)
    return self.update_compile_settings(
        updates_dict=updates_dict, silent=silent, **kwargs
    )
```

**After:**
```python
def update(self, updates_dict=None, silent=False, **kwargs) -> Set[str]:
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
    
    # Update buffer registry
    buffer_registry.update(self, updates_dict=all_updates, silent=True)
    
    # Update compile settings with recognized params only
    if linear_params:
        recognized = self.update_compile_settings(
            updates_dict=linear_params, silent=silent
        )
    
    return recognized
```

## Dependencies

### New Dependencies
- None (all components already exist)

### Modified Dependencies
- ODEImplicitStep depends on both NewtonKrylov and LinearSolver (already does)
- NewtonKrylov depends on LinearSolver (already does)
- GenericRosenbrockWStep depends on ODEImplicitStep (already does)

### Import Changes
- Move buffer_registry import to top of ode_implicitstep.py
- Remove BaseAlgorithmStep import from generic_rosenbrock_w.py

## Edge Cases

### Case 1: Rosenbrock with Newton Parameters
**Scenario:** User calls `rosenbrock_step.update(newton_tolerance=1e-4)`

**Expected Behavior:**
- ODEImplicitStep.update() delegates to LinearSolver
- LinearSolver doesn't recognize `newton_tolerance`
- If silent=True: parameter ignored, not in recognized set
- If silent=False: KeyError raised by update_compile_settings

**Implementation:** No special handling needed, natural behavior of recognition filtering

### Case 2: Update with Empty Dict
**Scenario:** User calls `step.update({})`

**Expected Behavior:**
- Early return with empty set
- No cache invalidation
- No solver calls

**Implementation:** Check `if not all_updates: return set()` at start of update()

### Case 3: Update with Only Unrecognized Params
**Scenario:** User calls `step.update(unknown_param=10, silent=True)`

**Expected Behavior:**
- Solver returns empty set
- Step update_compile_settings returns empty set
- Total recognized set is empty
- Cache stays valid (no recognized params changed)

**Implementation:** Accumulate recognized sets, only invalidate if solver cache invalid

### Case 4: Multiple Update Calls
**Scenario:** 
```python
step.update(newton_tolerance=1e-4)
step.update(krylov_tolerance=1e-6)
```

**Expected Behavior:**
- First call: Newton cache invalidates, step cache invalidates
- Second call: Linear cache invalidates, step cache invalidates again
- Both parameters take effect

**Implementation:** Each update() call is independent, cache invalidation cascades naturally

### Case 5: Accessing Newton Properties on Rosenbrock
**Scenario:** User calls `rosenbrock_step.newton_tolerance`

**Expected Behavior:**
- Rosenbrock solver is LinearSolver, which doesn't have newton_tolerance property
- Should raise AttributeError

**Implementation:** Properties forward to `self.solver.<property>`, LinearSolver doesn't have Newton properties

## Testing Considerations

### Unit Tests Needed
- ODEImplicitStep with solver_type='newton' creates NewtonKrylov
- ODEImplicitStep with solver_type='linear' creates LinearSolver
- Update delegation passes full dict to solver
- Cache invalidates when solver params change
- Cache stays valid when only algo params change
- Recognized parameter sets accumulate correctly

### Integration Tests Needed
- BackwardsEulerStep (Newton-based) works correctly
- GenericRosenbrockWStep works correctly
- Parameter updates flow through ownership chain
- Cached auxiliaries work for Rosenbrock

### Instrumented Tests
- Update instrumented versions to mirror source changes
- Verify logging still works with new solver ownership
- Check device function signatures unchanged

## Implementation Notes

### Constructor Order of Operations

**ODEImplicitStep.__init__ sequence:**
1. Create ImplicitStepConfig
2. Create LinearSolver with parameters
3. Register LinearSolver buffers
4. If solver_type == 'newton':
   - Create NewtonKrylov with LinearSolver
   - Register NewtonKrylov buffers
   - Store NewtonKrylov in self.solver
5. If solver_type == 'linear':
   - Store LinearSolver in self.solver
6. Call super().__init__(config, controller_defaults)

**GenericRosenbrockWStep.__init__ sequence:**
1. Create RosenbrockWStepConfig
2. Register buffers (stage_rhs, stage_store, etc.)
3. Call super().__init__(config, controller_defaults, solver_type='linear', ...)
4. Configure cached auxiliaries: self.solver.update(use_cached_auxiliaries=True)

### Property Forwarding Pattern

```python
@property
def newton_tolerance(self) -> float:
    """Return the Newton solve tolerance."""
    if hasattr(self.solver, 'newton_tolerance'):
        return self.solver.newton_tolerance
    raise AttributeError(f"{type(self.solver).__name__} does not have newton_tolerance")

# Or simpler, let AttributeError propagate naturally:
@property
def newton_tolerance(self) -> float:
    """Return the Newton solve tolerance."""
    return self.solver.newton_tolerance
```

The simpler approach is preferred unless specific error messages are needed.

### Cached Auxiliaries Configuration

After Rosenbrock creates solver:
```python
# In GenericRosenbrockWStep.__init__, after super().__init__
self.solver.update(use_cached_auxiliaries=True)
```

This configures LinearSolver to compile the cached auxiliaries variant of the device function.

## Success Criteria

Implementation is complete when:
1. All implicit algorithm steps own single `self.solver` attribute
2. Update methods accept full dicts, return recognized sets
3. Cache invalidation is conditional on solver parameter changes
4. All imports are at module top
5. Rosenbrock uses cached auxiliaries
6. All tests pass
7. Instrumented versions mirror source changes
