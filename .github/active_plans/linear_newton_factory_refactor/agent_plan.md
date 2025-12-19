# Technical Specification: Linear and Newton-Krylov Solver Refactor

## Component Descriptions

### LinearSolverConfig (attrs class)

Configuration container for LinearSolver following BaseStepConfig pattern.

**Attributes**:
- `precision: PrecisionDType` - Numerical precision (float16/32/64), uses `precision_converter` and `precision_validator`
- `n: int` - Length of residual and search-direction vectors, validator: `getype_validator(int, 1)`
- `operator_apply: Optional[Callable]` - Device function applying operator `F @ v`, validator: `validators.optional(is_device_validator)`
- `preconditioner: Optional[Callable]` - Device function for approximate inverse preconditioner, validator: `validators.optional(is_device_validator)`
- `correction_type: str` - Line-search strategy ("steepest_descent" or "minimal_residual"), validator: `validators.in_(["steepest_descent", "minimal_residual"])`
- `_tolerance: float` - Target on squared residual norm, validator: `gttype_validator(float, 0)`
- `max_iters: int` - Maximum iterations, validator: `inrangetype_validator(int, 1, 32767)`
- `preconditioned_vec_location: str` - Memory location ('local' or 'shared'), validator: `validators.in_(["local", "shared"])`
- `temp_location: str` - Memory location ('local' or 'shared'), validator: `validators.in_(["local", "shared"])`
- `use_cached_auxiliaries: bool` - Whether to use cached auxiliary arrays (determines signature), default: False

**Properties**:
- `tolerance` - Returns `self.precision(self._tolerance)`
- `numba_precision` - Returns Numba type via `numba.from_dtype(np.dtype(self.precision))`
- `simsafe_precision` - Returns CUDA-sim-safe type via `simsafe_dtype(np.dtype(self.precision))`
- `settings_dict` - Returns dict of all attributes for external access

**Validators**:
- Use `cubie._utils` validators (tolerant of NumPy types)
- Floating-point with leading underscore pattern for precision conversion

### LinearSolverCache (attrs class)

Cache container inheriting from CUDAFunctionCache.

**Attributes**:
- `linear_solver: Callable` - Compiled device function, validator: `is_device_validator`

### LinearSolver (CUDAFactory subclass)

Main factory class implementing cached compilation of linear solver device functions.

**Methods**:

**`__init__(self, config: LinearSolverConfig)`**:
- Call `super().__init__()`
- Call `self.setup_compile_settings(config)`
- Register buffers with buffer_registry:
  - If `use_cached_auxiliaries`: register `'lin_cached_preconditioned_vec'` and `'lin_cached_temp'`
  - Else: register `'lin_preconditioned_vec'` and `'lin_temp'`
  - Use config attributes for size, location, precision
  - Pass `self` as factory parameter

**`build(self) -> LinearSolverCache`**:
- Extract all parameters from `self.compile_settings`
- Compute flags: `sd_flag`, `mr_flag`, `preconditioned`
- Convert types: `n_val = int32(n)`, `max_iters = int32(max_iters)`, `precision_numba`, `typed_zero`, `tol_squared`
- Get allocators from buffer_registry using appropriate buffer names
- Define device function with closure capturing all compile-time constants
- Return `LinearSolverCache(linear_solver=device_func)`

**`update(self, updates_dict=None, silent=False, **kwargs) -> Set[str]`**:
- Call `self.update_compile_settings(updates_dict, silent=silent, **kwargs)`
- Return set of recognized keys

**Properties**:
- `device_function` - Returns `self.get_cached_output("linear_solver")`
- `precision` - Returns `self.compile_settings.precision`
- `n` - Returns `self.compile_settings.n`
- `correction_type` - Returns `self.compile_settings.correction_type`
- `tolerance` - Returns `self.compile_settings.tolerance`
- `max_iters` - Returns `self.compile_settings.max_iters`
- `use_cached_auxiliaries` - Returns `self.compile_settings.use_cached_auxiliaries`
- `shared_buffer_size` - Returns total shared memory elements required (computed from buffer_registry)
- `local_buffer_size` - Returns total local memory elements required (computed from buffer_registry)

**Expected Behavior**:
- First access to `.device_function` triggers `build()`
- Subsequent accesses return cached function
- Calling `update()` with changed parameters invalidates cache
- Next `.device_function` access triggers rebuild

### NewtonKrylovConfig (attrs class)

Configuration container for NewtonKrylov solver.

**Attributes**:
- `precision: PrecisionDType` - Numerical precision, uses `precision_converter` and `precision_validator`
- `n: int` - Size of state vectors, validator: `getype_validator(int, 1)`
- `residual_function: Optional[Callable]` - Device function evaluating residuals, validator: `validators.optional(is_device_validator)`
- `linear_solver: Optional[LinearSolver]` - LinearSolver instance, validator: `validators.optional(validators.instance_of(LinearSolver))`
- `_tolerance: float` - Residual norm threshold, validator: `gttype_validator(float, 0)`
- `max_iters: int` - Maximum Newton iterations, validator: `inrangetype_validator(int, 1, 32767)`
- `_damping: float` - Step shrink factor, validator: `inrangetype_validator(float, 0, 1)`
- `max_backtracks: int` - Maximum damping attempts, validator: `inrangetype_validator(int, 1, 32767)`
- `delta_location: str` - Memory location, validator: `validators.in_(["local", "shared"])`
- `residual_location: str` - Memory location, validator: `validators.in_(["local", "shared"])`
- `residual_temp_location: str` - Memory location, validator: `validators.in_(["local", "shared"])`
- `stage_base_bt_location: str` - Memory location, validator: `validators.in_(["local", "shared"])`

**Properties**:
- `tolerance` - Returns `self.precision(self._tolerance)`
- `damping` - Returns `self.precision(self._damping)`
- `numba_precision` - Returns Numba type
- `simsafe_precision` - Returns CUDA-sim-safe type
- `settings_dict` - Returns dict of all attributes

### NewtonKrylovCache (attrs class)

Cache container inheriting from CUDAFunctionCache.

**Attributes**:
- `newton_krylov_solver: Callable` - Compiled device function, validator: `is_device_validator`

### NewtonKrylov (CUDAFactory subclass)

Factory class for Newton-Krylov solver device functions.

**Methods**:

**`__init__(self, config: NewtonKrylovConfig)`**:
- Call `super().__init__()`
- Call `self.setup_compile_settings(config)`
- Register buffers with buffer_registry:
  - `'newton_delta'`, `'newton_residual'`, `'newton_residual_temp'`, `'newton_stage_base_bt'`
  - Use config attributes for size, location, precision
  - Pass `self` as factory parameter

**`build(self) -> NewtonKrylovCache`**:
- Extract parameters from `self.compile_settings`
- Get linear_solver device function: `linear_solver_fn = self.compile_settings.linear_solver.device_function`
- Convert types: `numba_precision`, `tol_squared`, `typed_zero`, `typed_one`, `typed_damping`, `n_val`, `max_iters`, `max_backtracks`
- Get allocators from buffer_registry
- Define device function with closure capturing all compile-time constants
- Include logic for computing `lin_start` (shared buffer offset for linear solver)
- Return `NewtonKrylovCache(newton_krylov_solver=device_func)`

**`update(self, updates_dict=None, silent=False, **kwargs) -> Set[str]`**:
- Handle `linear_solver` updates by checking if LinearSolver config changed
- Call `self.update_compile_settings()` for other parameters
- Return set of recognized keys

**Properties**:
- `device_function` - Returns `self.get_cached_output("newton_krylov_solver")`
- `precision` - Returns `self.compile_settings.precision`
- `n` - Returns `self.compile_settings.n`
- `tolerance` - Returns `self.compile_settings.tolerance`
- `max_iters` - Returns `self.compile_settings.max_iters`
- `damping` - Returns `self.compile_settings.damping`
- `max_backtracks` - Returns `self.compile_settings.max_backtracks`
- `linear_solver` - Returns `self.compile_settings.linear_solver`
- `shared_buffer_size` - Returns Newton buffers + linear solver shared size
- `local_buffer_size` - Returns Newton buffers + linear solver local size

## Architectural Changes Required

### File Structure Changes

**New files**:
- None (refactoring existing files)

**Modified files**:
- `src/cubie/integrators/matrix_free_solvers/linear_solver.py` - Convert to class-based
- `src/cubie/integrators/matrix_free_solvers/newton_krylov.py` - Convert to class-based
- `src/cubie/integrators/algorithms/ode_implicitstep.py` - Update `build_implicit_helpers()`
- All implicit algorithm step implementations (backwards_euler.py, crank_nicolson.py, etc.)
- `tests/integrators/matrix_free_solvers/test_linear_solver.py` - Update fixtures and tests
- `tests/integrators/matrix_free_solvers/test_newton_krylov.py` - Update fixtures and tests
- `tests/integrators/matrix_free_solvers/conftest.py` - Add new fixtures

**Deprecated elements**:
- `linear_solver_factory()` function - Replace with LinearSolver class
- `linear_solver_cached_factory()` function - Merged into LinearSolver with config flag
- `newton_krylov_solver_factory()` function - Replace with NewtonKrylov class

### Integration Points with Current Codebase

**ODEImplicitStep.build_implicit_helpers()**:

Current approach (call sites):
```python
def build_implicit_helpers(self) -> Callable:
    config = self.compile_settings
    # Extract parameters
    # Call get_solver_helper_fn for operator, preconditioner, residual
    linear_solver = linear_solver_factory(operator, n=n, ...)
    nonlinear_solver = newton_krylov_solver_factory(
        residual_function=residual, linear_solver=linear_solver, ...
    )
    return nonlinear_solver
```

New approach:
```python
def __init__(self, config: ImplicitStepConfig, _controller_defaults):
    super().__init__(config, _controller_defaults)
    # Create solver instances with initial config
    linear_config = LinearSolverConfig(
        precision=config.precision,
        n=config.n,
        correction_type=config.linear_correction_type,
        tolerance=config.krylov_tolerance,
        max_iters=config.max_linear_iters,
    )
    self._linear_solver = LinearSolver(linear_config)
    
    newton_config = NewtonKrylovConfig(
        precision=config.precision,
        n=config.n,
        linear_solver=self._linear_solver,
        tolerance=config.newton_tolerance,
        max_iters=config.max_newton_iters,
        damping=config.newton_damping,
        max_backtracks=config.newton_max_backtracks,
    )
    self._newton_solver = NewtonKrylov(newton_config)

def build_implicit_helpers(self) -> Callable:
    config = self.compile_settings
    # Get device functions from system
    get_fn = config.get_solver_helper_fn
    operator = get_fn('linear_operator', ...)
    preconditioner = get_fn('neumann_preconditioner', ...)
    residual = get_fn('stage_residual', ...)
    
    # Update solvers with device functions
    self._linear_solver.update(
        operator_apply=operator,
        preconditioner=preconditioner,
    )
    self._newton_solver.update(
        residual_function=residual,
    )
    
    # Return device function
    return self._newton_solver.device_function
```

**Buffer Registry Coordination**:

Current flow:
- Factory function registers buffers during execution
- Allocators retrieved inline
- Factory parameter identifies context

New flow:
- Solver `__init__` registers buffers immediately
- `build()` retrieves allocators from registry
- `self` passed as factory parameter
- Buffer size queries available before building

**Test Fixtures**:

Current pattern (test_linear_solver.py):
```python
@pytest.fixture
def solver_device(request, placeholder_operator, precision):
    return linear_solver_factory(
        placeholder_operator, 3, precision=precision, ...
    )
```

New pattern:
```python
@pytest.fixture
def linear_solver_instance(request, solver_settings, placeholder_operator):
    config = LinearSolverConfig(
        operator_apply=placeholder_operator,
        n=solver_settings['n'],
        precision=solver_settings['precision'],
        **request.param  # indirect override
    )
    return LinearSolver(config)

@pytest.fixture
def solver_device(linear_solver_instance):
    return linear_solver_instance.device_function
```

## Expected Interactions Between Components

### LinearSolver ↔ NewtonKrylov

1. **Instantiation**: NewtonKrylov stores LinearSolver instance in config
2. **Compilation**: NewtonKrylov accesses `linear_solver.device_function` during `build()`
3. **Cache Invalidation**: If LinearSolver config changes, NewtonKrylov cache should also invalidate
4. **Buffer Coordination**: NewtonKrylov queries `linear_solver.shared_buffer_size` to compute offset for linear solver shared scratch

### NewtonKrylov ↔ ODEImplicitStep

1. **Instantiation**: ODEImplicitStep creates NewtonKrylov in `__init__`
2. **Update**: ODEImplicitStep calls `newton_solver.update()` with residual function during `build()`
3. **Access**: ODEImplicitStep accesses `newton_solver.device_function` to get compiled solver
4. **Buffer Sizing**: ODEImplicitStep queries `newton_solver.shared_buffer_size` for memory planning

### LinearSolver ↔ ODEImplicitStep

1. **Instantiation**: ODEImplicitStep creates LinearSolver in `__init__`
2. **Update**: ODEImplicitStep calls `linear_solver.update()` with operator and preconditioner during `build()`
3. **Indirect Access**: ODEImplicitStep accesses LinearSolver only through NewtonKrylov

### Solvers ↔ buffer_registry

1. **Registration**: Solvers register buffers during `__init__` with unique names per factory instance
2. **Allocation**: Solvers get allocators during `build()` using registered names
3. **Size Queries**: External components query `shared_buffer_size()` and `local_buffer_size()` via buffer_registry

## Data Structures and Their Purposes

### LinearSolverConfig

**Purpose**: Store all compile-time configuration for linear solver

**Key Fields**:
- `operator_apply`, `preconditioner` - Device functions that define the linear system
- `correction_type` - Algorithm variant (compile-time branch selection)
- `tolerance`, `max_iters` - Convergence parameters captured in closure
- Buffer locations - Memory placement decisions

**Usage Pattern**: Created once, updated when solver parameters change, triggers rebuild via cache invalidation

### NewtonKrylovConfig

**Purpose**: Store all compile-time configuration for Newton-Krylov solver

**Key Fields**:
- `residual_function` - Device function defining nonlinear system
- `linear_solver` - Nested solver instance (enables composition)
- `tolerance`, `max_iters`, `damping`, `max_backtracks` - Convergence and backtracking parameters
- Buffer locations - Memory placement decisions

**Usage Pattern**: Created once, updated when solver parameters change, invalidates cache when linear_solver or other parameters change

### LinearSolverCache / NewtonKrylovCache

**Purpose**: Store compiled device function outputs

**Structure**: Single `Callable` field with device function

**Lifetime**: Created by `build()`, cached until config changes, automatically invalidated by CUDAFactory base class

## Dependencies and Imports Required

### linear_solver.py

**New imports**:
```python
from typing import Callable, Optional, Set, Dict, Any
import attrs
from attrs import validators
from cubie._utils import (
    PrecisionDType,
    getype_validator,
    gttype_validator,
    inrangetype_validator,
    is_device_validator,
    precision_converter,
    precision_validator,
)
from cubie.CUDAFactory import CUDAFactory, CUDAFunctionCache
import numba
```

**Existing imports retained**:
```python
from numba import cuda, int32, from_dtype
import numpy as np
from cubie.buffer_registry import buffer_registry
from cubie.cuda_simsafe import activemask, all_sync, compile_kwargs, selp
```

### newton_krylov.py

**New imports**:
```python
from typing import Callable, Optional, Set, Dict, Any
import attrs
from attrs import validators
from cubie._utils import (
    PrecisionDType,
    getype_validator,
    gttype_validator,
    inrangetype_validator,
    is_device_validator,
    precision_converter,
    precision_validator,
)
from cubie.CUDAFactory import CUDAFactory, CUDAFunctionCache
import numba
```

**Modified imports**:
```python
# Remove direct import of factory function
# from cubie.integrators.matrix_free_solvers import linear_solver_factory

# Import class instead
from cubie.integrators.matrix_free_solvers.linear_solver import LinearSolver
```

**Existing imports retained**:
```python
from numba import cuda, int32, from_dtype
import numpy as np
from cubie._utils import ALLOWED_PRECISIONS, PrecisionDType
from cubie.buffer_registry import buffer_registry
from cubie.cuda_simsafe import activemask, all_sync, selp, any_sync, compile_kwargs
```

### ode_implicitstep.py

**New imports**:
```python
from cubie.integrators.matrix_free_solvers.linear_solver import (
    LinearSolver,
    LinearSolverConfig,
)
from cubie.integrators.matrix_free_solvers.newton_krylov import (
    NewtonKrylov,
    NewtonKrylovConfig,
)
```

**Removed imports**:
```python
# Remove factory function imports
# from cubie.integrators.matrix_free_solvers import (
#     linear_solver_factory,
#     newton_krylov_solver_factory,
# )
```

## Edge Cases to Consider

### 1. Operator/Preconditioner/Residual Not Set

**Scenario**: LinearSolver or NewtonKrylov `build()` called before device functions provided

**Handling**: Config validators allow `Optional[Callable]`, but `build()` should check and raise informative error if None

### 2. LinearSolver Config Changed After NewtonKrylov Built

**Scenario**: User updates LinearSolver parameters after NewtonKrylov has cached its device function

**Handling**: NewtonKrylov must detect LinearSolver config change and invalidate its own cache. Implement custom comparison in `update()` method.

### 3. Buffer Location Conflicts

**Scenario**: User specifies 'shared' location but total shared memory exceeds GPU limits

**Handling**: Document behavior - buffer_registry will handle allocation failure. Solvers don't need special handling.

### 4. Precision Mismatch

**Scenario**: LinearSolver and NewtonKrylov instantiated with different precisions

**Handling**: NewtonKrylov `__init__` should validate that `linear_solver.precision == self.precision`, raise ValueError if mismatch.

### 5. Empty Operator Result

**Scenario**: Operator returns zero vector (division by zero in correction calculation)

**Handling**: Existing logic handles this with `selp(denominator != typed_zero, ...)` - preserve this pattern.

### 6. Max Iterations = 0

**Scenario**: User sets `max_iters=0` for immediate return

**Handling**: Validators enforce `min=1`, so this is prevented at config level.

### 7. Cached vs Non-Cached Signature Mismatch

**Scenario**: Test expects cached signature but gets non-cached (or vice versa)

**Handling**: `use_cached_auxiliaries` flag in config makes this explicit. Tests must specify correct flag.

### 8. Indirect Test Parameterization

**Scenario**: Test fixture needs to override solver config parameters

**Handling**: Follow pytest indirect pattern - fixture accepts `request.param` dict and merges with defaults.

### 9. Factory Instance Identity

**Scenario**: Buffer registry uses factory instance as key, must be stable across updates

**Handling**: Store `self` reference, never create new LinearSolver/NewtonKrylov instances during updates.

### 10. Nested Cache Invalidation Timing

**Scenario**: LinearSolver rebuild triggered inside NewtonKrylov.build()

**Handling**: This is expected behavior. NewtonKrylov accesses `linear_solver.device_function`, which may trigger LinearSolver.build() first. Both builds complete before NewtonKrylov returns.
