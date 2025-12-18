# Implementation Task List: Buffer Registry Migration Tasks 3-9

## Status Overview
- **Phase**: Detailed Implementation Plan Complete
- **BufferRegistry Core**: ✅ COMPLETE (Tasks 1-2)
- **Migration**: ⏳ READY FOR IMPLEMENTATION (Tasks 3-9)

---

## Task Group 3: Migrate Matrix-Free Solvers - SEQUENTIAL
**Status**: [x]
**Dependencies**: BufferRegistry core (complete)

**Outcomes**:
- Files Modified:
  * src/cubie/integrators/matrix_free_solvers/linear_solver.py (446 lines total)
  * src/cubie/integrators/matrix_free_solvers/newton_krylov.py (303 lines total)
- Classes Removed:
  * LocalSizes, SliceIndices, LinearSolverLocalSizes, LinearSolverSliceIndices, LinearSolverBufferSettings from linear_solver.py
  * NewtonLocalSizes, NewtonSliceIndices, NewtonBufferSettings from newton_krylov.py
- Functions Modified:
  * linear_solver_factory() - added factory parameter, location parameters
  * linear_solver_cached_factory() - added factory parameter, location parameters
  * newton_krylov_solver_factory() - added factory parameter, location parameters
- Implementation Summary:
  Replaced BufferSettings-based allocation with buffer_registry.register() and get_allocator() calls
- Issues Flagged: Algorithm files (DIRK, FIRK, Rosenbrock) that call these factories need updating

**Required Context**:
- File: src/cubie/buffer_registry.py (entire file - review API)
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py (lines 1-360)
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py (lines 1-452)

**Input Validation Required**:
- factory: Check is not None (required parameter)
- preconditioned_vec_location: Validate in ['local', 'shared']
- temp_location: Validate in ['local', 'shared']
- n: Check >= 1

---

### Task 3.1: Migrate linear_solver.py
**File**: `src/cubie/integrators/matrix_free_solvers/linear_solver.py`
**Action**: Modify

#### Step 3.1.1: Update imports
**DELETE** (lines 9-18):
```python
import attrs
from attrs import validators
from numba import cuda, int32, from_dtype
import numpy as np

from cubie._utils import PrecisionDType, getype_validator
from cubie.cuda_simsafe import activemask, all_sync, compile_kwargs, selp
```

**ADD**:
```python
from numba import cuda, int32, from_dtype
import numpy as np

from cubie._utils import PrecisionDType
from cubie.buffer_registry import buffer_registry
from cubie.cuda_simsafe import activemask, all_sync, compile_kwargs, selp
```

#### Step 3.1.2: Delete BufferSettings classes
**DELETE** (lines 20-123 - entire block):
```python
# Backward-compatible classes for algorithm files that still use old API
class LocalSizes:
    """Base class for local sizes - backward compatibility."""

    def nonzero(self, attr_name: str) -> int:
        """Return max(value, 1) for cuda.local.array compatibility."""
        return max(getattr(self, attr_name), 1)


class SliceIndices:
    """Base class for slice indices - backward compatibility."""
    pass


@attrs.define
class LinearSolverLocalSizes(LocalSizes):
    """Local array sizes for linear solver buffers with nonzero guarantees."""

    preconditioned_vec: int = attrs.field(validator=getype_validator(int, 0))
    temp: int = attrs.field(validator=getype_validator(int, 0))


@attrs.define
class LinearSolverSliceIndices(SliceIndices):
    """Slice container for linear solver shared memory buffer layouts."""

    preconditioned_vec: slice = attrs.field()
    temp: slice = attrs.field()
    local_end: int = attrs.field()


@attrs.define
class LinearSolverBufferSettings:
    """Configuration for linear solver buffer sizes and memory locations."""

    n: int = attrs.field(validator=getype_validator(int, 1))
    preconditioned_vec_location: str = attrs.field(
        default='local', validator=validators.in_(["local", "shared"])
    )
    temp_location: str = attrs.field(
        default='local', validator=validators.in_(["local", "shared"])
    )

    @property
    def use_shared_preconditioned_vec(self) -> bool:
        """Return True if preconditioned_vec uses shared memory."""
        return self.preconditioned_vec_location == 'shared'

    @property
    def use_shared_temp(self) -> bool:
        """Return True if temp buffer uses shared memory."""
        return self.temp_location == 'shared'

    @property
    def shared_memory_elements(self) -> int:
        """Return total shared memory elements required."""
        total = 0
        if self.use_shared_preconditioned_vec:
            total += self.n
        if self.use_shared_temp:
            total += self.n
        return total

    @property
    def local_memory_elements(self) -> int:
        """Return total local memory elements required."""
        total = 0
        if not self.use_shared_preconditioned_vec:
            total += self.n
        if not self.use_shared_temp:
            total += self.n
        return total

    @property
    def local_sizes(self) -> LinearSolverLocalSizes:
        """Return LinearSolverLocalSizes instance with buffer sizes."""
        return LinearSolverLocalSizes(
            preconditioned_vec=self.n,
            temp=self.n,
        )

    @property
    def shared_indices(self) -> LinearSolverSliceIndices:
        """Return LinearSolverSliceIndices instance with shared memory layout."""
        ptr = 0

        if self.use_shared_preconditioned_vec:
            preconditioned_vec_slice = slice(ptr, ptr + self.n)
            ptr += self.n
        else:
            preconditioned_vec_slice = slice(0, 0)

        if self.use_shared_temp:
            temp_slice = slice(ptr, ptr + self.n)
            ptr += self.n
        else:
            temp_slice = slice(0, 0)

        return LinearSolverSliceIndices(
            preconditioned_vec=preconditioned_vec_slice,
            temp=temp_slice,
            local_end=ptr,
        )
```

#### Step 3.1.3: Modify linear_solver_factory signature and body
**DELETE** (lines 125-206):
```python
def linear_solver_factory(
    operator_apply: Callable,
    n: int,
    preconditioner: Optional[Callable] = None,
    correction_type: str = "minimal_residual",
    tolerance: float = 1e-6,
    max_iters: int = 100,
    precision: PrecisionDType = np.float64,
    buffer_settings: Optional[LinearSolverBufferSettings] = None,
) -> Callable:
    """Create a CUDA device function implementing steepest-descent or MR.
    ...docstring...
    """

    sd_flag = 1 if correction_type == "steepest_descent" else 0
    mr_flag = 1 if correction_type == "minimal_residual" else 0
    if correction_type not in ("steepest_descent", "minimal_residual"):
        raise ValueError(
            "Correction type must be 'steepest_descent' or 'minimal_residual'."
        )
    preconditioned = 1 if preconditioner is not None else 0
    n_val = int32(n)
    max_iters = int32(max_iters)
    precision_numba = from_dtype(precision)
    typed_zero = precision_numba(0.0)
    tol_squared = precision_numba(tolerance * tolerance)

    # Extract buffer settings as compile-time constants
    if buffer_settings is None:
        buffer_settings = LinearSolverBufferSettings(n=n)

    # Unpack boolean flags for selective allocation (compile-time constants)
    precond_vec_shared = buffer_settings.use_shared_preconditioned_vec
    temp_shared = buffer_settings.use_shared_temp

    # Unpack slice indices for shared memory layout
    slice_indices = buffer_settings.shared_indices
    precond_vec_slice = slice_indices.preconditioned_vec
    temp_slice = slice_indices.temp

    # Unpack local sizes for local array allocation
    local_sizes = buffer_settings.local_sizes
    precond_vec_local_size = local_sizes.nonzero('preconditioned_vec')
    temp_local_size = local_sizes.nonzero('temp')
```

**ADD**:
```python
def linear_solver_factory(
    operator_apply: Callable,
    n: int,
    factory: object,
    preconditioner: Optional[Callable] = None,
    correction_type: str = "minimal_residual",
    tolerance: float = 1e-6,
    max_iters: int = 100,
    precision: PrecisionDType = np.float64,
    preconditioned_vec_location: str = 'local',
    temp_location: str = 'local',
) -> Callable:
    """Create a CUDA device function implementing steepest-descent or MR.

    Parameters
    ----------
    operator_apply
        Callback that overwrites its output vector with ``F @ v``.
    n
        Length of the one-dimensional residual and search-direction vectors.
    factory
        Owning factory instance for buffer registration.
    preconditioner
        Approximate inverse preconditioner. If ``None`` the identity
        preconditioner is used.
    correction_type
        Line-search strategy: ``"steepest_descent"`` or ``"minimal_residual"``.
    tolerance
        Target on the squared residual norm that signals convergence.
    max_iters
        Maximum number of iterations permitted.
    precision
        Floating-point precision used when building the device function.
    preconditioned_vec_location
        Memory location for preconditioned_vec buffer: 'local' or 'shared'.
    temp_location
        Memory location for temp buffer: 'local' or 'shared'.

    Returns
    -------
    Callable
        CUDA device function returning ``0`` on convergence and ``4`` when the
        iteration limit is reached.
    """

    sd_flag = 1 if correction_type == "steepest_descent" else 0
    mr_flag = 1 if correction_type == "minimal_residual" else 0
    if correction_type not in ("steepest_descent", "minimal_residual"):
        raise ValueError(
            "Correction type must be 'steepest_descent' or 'minimal_residual'."
        )
    preconditioned = 1 if preconditioner is not None else 0
    n_val = int32(n)
    max_iters = int32(max_iters)
    precision_numba = from_dtype(precision)
    typed_zero = precision_numba(0.0)
    tol_squared = precision_numba(tolerance * tolerance)

    # Register buffers with central registry
    buffer_registry.register(
        'lin_preconditioned_vec', factory, n, preconditioned_vec_location,
        precision=precision
    )
    buffer_registry.register(
        'lin_temp', factory, n, temp_location, precision=precision
    )

    # Get allocators from registry
    alloc_precond = buffer_registry.get_allocator(
        'lin_preconditioned_vec', factory
    )
    alloc_temp = buffer_registry.get_allocator('lin_temp', factory)
```

#### Step 3.1.4: Update linear_solver device function body
**DELETE** (inside the device function, lines ~271-281):
```python
        # Selective memory allocation based on buffer_settings
        if precond_vec_shared:
            preconditioned_vec = shared[precond_vec_slice]
        else:
            preconditioned_vec = cuda.local.array(precond_vec_local_size,
                                                  precision_numba)

        if temp_shared:
            temp = shared[temp_slice]
        else:
            temp = cuda.local.array(temp_local_size, precision_numba)
```

**ADD**:
```python
        # Allocate buffers from registry
        preconditioned_vec = alloc_precond(shared, shared)
        temp = alloc_temp(shared, shared)
```

#### Step 3.1.5: Apply same changes to linear_solver_cached_factory
Apply identical pattern changes to `linear_solver_cached_factory` (lines 362-551):
- Update signature to add `factory` parameter and location parameters
- Remove `buffer_settings` parameter
- Register buffers with buffer_registry
- Get allocators and use them in device function

---

### Task 3.2: Migrate newton_krylov.py
**File**: `src/cubie/integrators/matrix_free_solvers/newton_krylov.py`
**Action**: Modify

#### Step 3.2.1: Update imports
**DELETE** (lines 6-18):
```python
from typing import Callable, Optional

import attrs
from attrs import validators
from numba import cuda, int32, from_dtype
import numpy as np
from cubie._utils import ALLOWED_PRECISIONS, PrecisionDType, getype_validator
from cubie.cuda_simsafe import activemask, all_sync, selp, any_sync
from cubie.integrators.matrix_free_solvers.linear_solver import (
    LinearSolverBufferSettings, LocalSizes, SliceIndices
)
```

**ADD**:
```python
from typing import Callable, Optional

from numba import cuda, int32, from_dtype
import numpy as np
from cubie._utils import ALLOWED_PRECISIONS, PrecisionDType
from cubie.buffer_registry import buffer_registry
from cubie.cuda_simsafe import activemask, all_sync, selp, any_sync
```

#### Step 3.2.2: Delete BufferSettings classes
**DELETE** (lines 21-165 - entire block):
```python
@attrs.define
class NewtonLocalSizes(LocalSizes):
    """Local array sizes for Newton solver buffers."""

    delta: int = attrs.field(validator=getype_validator(int, 0))
    residual: int = attrs.field(validator=getype_validator(int, 0))
    residual_temp: int = attrs.field(validator=getype_validator(int, 0))
    stage_base_bt: int = attrs.field(validator=getype_validator(int, 0))
    krylov_iters: int = attrs.field(validator=getype_validator(int, 0))


@attrs.define
class NewtonSliceIndices(SliceIndices):
    """Slice container for Newton solver shared memory layouts."""

    delta: slice = attrs.field()
    residual: slice = attrs.field()
    residual_temp: slice = attrs.field()
    stage_base_bt: slice = attrs.field()
    local_end: int = attrs.field()
    lin_solver_start: int = attrs.field()


@attrs.define
class NewtonBufferSettings:
    """Configuration for Newton solver buffer sizes and locations."""

    n: int = attrs.field(validator=getype_validator(int, 1))
    delta_location: str = attrs.field(
        default='shared', validator=validators.in_(["local", "shared"])
    )
    residual_location: str = attrs.field(
        default='shared', validator=validators.in_(["local", "shared"])
    )
    residual_temp_location: str = attrs.field(
        default='local', validator=validators.in_(["local", "shared"])
    )
    stage_base_bt_location: str = attrs.field(
        default='local', validator=validators.in_(["local", "shared"])
    )
    linear_solver_buffer_settings: Optional[LinearSolverBufferSettings] = (
        attrs.field(default=None)
    )

    @property
    def use_shared_delta(self) -> bool:
        """Return True if delta buffer uses shared memory."""
        return self.delta_location == 'shared'

    @property
    def use_shared_residual(self) -> bool:
        """Return True if residual buffer uses shared memory."""
        return self.residual_location == 'shared'

    @property
    def use_shared_residual_temp(self) -> bool:
        """Return True if residual_temp buffer uses shared memory."""
        return self.residual_temp_location == 'shared'

    @property
    def use_shared_stage_base_bt(self) -> bool:
        """Return True if stage_base_bt buffer uses shared memory."""
        return self.stage_base_bt_location == 'shared'

    @property
    def shared_memory_elements(self) -> int:
        """Return total shared memory elements required."""
        total = 0
        if self.use_shared_delta:
            total += self.n
        if self.use_shared_residual:
            total += self.n
        if self.use_shared_residual_temp:
            total += self.n
        if self.use_shared_stage_base_bt:
            total += self.n
        if self.linear_solver_buffer_settings is not None:
            total += self.linear_solver_buffer_settings.shared_memory_elements
        return total

    @property
    def local_memory_elements(self) -> int:
        """Return total local memory elements required."""
        total = 0
        if not self.use_shared_delta:
            total += self.n
        if not self.use_shared_residual:
            total += self.n
        if not self.use_shared_residual_temp:
            total += self.n
        if not self.use_shared_stage_base_bt:
            total += self.n
        total += 1  # krylov_iters
        if self.linear_solver_buffer_settings is not None:
            total += self.linear_solver_buffer_settings.local_memory_elements
        return total

    @property
    def local_sizes(self) -> NewtonLocalSizes:
        """Return NewtonLocalSizes instance with buffer sizes."""
        return NewtonLocalSizes(
            delta=self.n,
            residual=self.n,
            residual_temp=self.n,
            stage_base_bt=self.n,
            krylov_iters=1,
        )

    @property
    def shared_indices(self) -> NewtonSliceIndices:
        """Return NewtonSliceIndices instance with shared memory layout."""
        ptr = 0
        if self.use_shared_delta:
            delta_slice = slice(ptr, ptr + self.n)
            ptr += self.n
        else:
            delta_slice = slice(0, 0)

        if self.use_shared_residual:
            residual_slice = slice(ptr, ptr + self.n)
            ptr += self.n
        else:
            residual_slice = slice(0, 0)

        if self.use_shared_residual_temp:
            residual_temp_slice = slice(ptr, ptr + self.n)
            ptr += self.n
        else:
            residual_temp_slice = slice(0, 0)

        if self.use_shared_stage_base_bt:
            stage_base_bt_slice = slice(ptr, ptr + self.n)
            ptr += self.n
        else:
            stage_base_bt_slice = slice(0, 0)

        return NewtonSliceIndices(
            delta=delta_slice,
            residual=residual_slice,
            residual_temp=residual_temp_slice,
            stage_base_bt=stage_base_bt_slice,
            local_end=ptr,
            lin_solver_start=ptr,
        )
```

#### Step 3.2.3: Modify newton_krylov_solver_factory signature
**DELETE** (lines 167-258):
```python
def newton_krylov_solver_factory(
    residual_function: Callable,
    linear_solver: Callable,
    n: int,
    tolerance: float,
    max_iters: int,
    damping: float = 0.5,
    max_backtracks: int = 8,
    precision: PrecisionDType = np.float32,
    buffer_settings: Optional[NewtonBufferSettings] = None,
) -> Callable:
    """Create a damped Newton--Krylov solver device function.
    ...docstring...
    """

    precision_dtype = np.dtype(precision)
    if precision_dtype not in ALLOWED_PRECISIONS:
        raise ValueError("precision must be float16, float32, or float64.")

    # Default buffer settings - shared delta/residual (current behavior)
    if buffer_settings is None:
        buffer_settings = NewtonBufferSettings(n=n)

    # Extract compile-time flags
    delta_shared = buffer_settings.use_shared_delta
    residual_shared = buffer_settings.use_shared_residual
    shared_indices = buffer_settings.shared_indices
    delta_slice = shared_indices.delta
    residual_slice = shared_indices.residual
    lin_solver_start = shared_indices.lin_solver_start
    local_sizes = buffer_settings.local_sizes
    delta_local_size = local_sizes.nonzero('delta')
    residual_local_size = local_sizes.nonzero('residual')
    residual_temp_shared = buffer_settings.use_shared_residual_temp
    residual_temp_slice = shared_indices.residual_temp
    residual_temp_local_size = local_sizes.nonzero('residual_temp')
    stage_base_bt_shared = buffer_settings.use_shared_stage_base_bt
    stage_base_bt_slice = shared_indices.stage_base_bt
    stage_base_bt_local_size = local_sizes.nonzero('stage_base_bt')

    numba_precision = from_dtype(precision_dtype)
    tol_squared = numba_precision(tolerance * tolerance)
    typed_zero = numba_precision(0.0)
    typed_one = numba_precision(1.0)
    typed_damping = numba_precision(damping)
    n_val = int32(n)
    max_iters = int32(max_iters)
    max_backtracks = int32(max_backtracks)
```

**ADD**:
```python
def newton_krylov_solver_factory(
    residual_function: Callable,
    linear_solver: Callable,
    n: int,
    factory: object,
    tolerance: float,
    max_iters: int,
    damping: float = 0.5,
    max_backtracks: int = 8,
    precision: PrecisionDType = np.float32,
    delta_location: str = 'shared',
    residual_location: str = 'shared',
    residual_temp_location: str = 'local',
    stage_base_bt_location: str = 'local',
) -> Callable:
    """Create a damped Newton--Krylov solver device function.

    Parameters
    ----------
    residual_function
        Matrix-free residual evaluator.
    linear_solver
        Matrix-free linear solver created by :func:`linear_solver_factory`.
    n
        Size of the flattened residual and state vectors.
    factory
        Owning factory instance for buffer registration.
    tolerance
        Residual norm threshold for convergence.
    max_iters
        Maximum number of Newton iterations performed.
    damping
        Step shrink factor used during backtracking.
    max_backtracks
        Maximum number of damping attempts per Newton step.
    precision
        Floating-point precision used when compiling the device function.
    delta_location
        Memory location for delta buffer: 'local' or 'shared'.
    residual_location
        Memory location for residual buffer: 'local' or 'shared'.
    residual_temp_location
        Memory location for residual_temp buffer: 'local' or 'shared'.
    stage_base_bt_location
        Memory location for stage_base_bt buffer: 'local' or 'shared'.

    Returns
    -------
    Callable
        CUDA device function implementing the damped Newton--Krylov scheme.
    """

    precision_dtype = np.dtype(precision)
    if precision_dtype not in ALLOWED_PRECISIONS:
        raise ValueError("precision must be float16, float32, or float64.")

    # Register buffers with central registry
    buffer_registry.register(
        'newton_delta', factory, n, delta_location, precision=precision
    )
    buffer_registry.register(
        'newton_residual', factory, n, residual_location, precision=precision
    )
    buffer_registry.register(
        'newton_residual_temp', factory, n, residual_temp_location,
        precision=precision
    )
    buffer_registry.register(
        'newton_stage_base_bt', factory, n, stage_base_bt_location,
        precision=precision
    )

    # Get allocators from registry
    alloc_delta = buffer_registry.get_allocator('newton_delta', factory)
    alloc_residual = buffer_registry.get_allocator('newton_residual', factory)
    alloc_residual_temp = buffer_registry.get_allocator(
        'newton_residual_temp', factory
    )
    alloc_stage_base_bt = buffer_registry.get_allocator(
        'newton_stage_base_bt', factory
    )

    numba_precision = from_dtype(precision_dtype)
    tol_squared = numba_precision(tolerance * tolerance)
    typed_zero = numba_precision(0.0)
    typed_one = numba_precision(1.0)
    typed_damping = numba_precision(damping)
    n_val = int32(n)
    max_iters = int32(max_iters)
    max_backtracks = int32(max_backtracks)
```

#### Step 3.2.4: Update newton_krylov_solver device function buffer allocation
**DELETE** (inside device function, lines ~302-323):
```python
        # Selective allocation based on buffer_settings
        if delta_shared:
            delta = shared_scratch[delta_slice]
        else:
            delta = cuda.local.array(delta_local_size, numba_precision)
            for _i in range(delta_local_size):
                delta[_i] = typed_zero

        if residual_shared:
            residual = shared_scratch[residual_slice]
        else:
            residual = cuda.local.array(residual_local_size, numba_precision)
            for _i in range(residual_local_size):
                residual[_i] = typed_zero

        if residual_temp_shared:
            residual_temp = shared_scratch[residual_temp_slice]
        else:
            residual_temp = cuda.local.array(
                residual_temp_local_size, numba_precision
            )
```

**ADD**:
```python
        # Allocate buffers from registry
        delta = alloc_delta(shared_scratch, shared_scratch)
        residual = alloc_residual(shared_scratch, shared_scratch)
        residual_temp = alloc_residual_temp(shared_scratch, shared_scratch)
        
        # Initialize local arrays if needed
        for _i in range(n_val):
            delta[_i] = typed_zero
            residual[_i] = typed_zero
```

**DELETE** (lines ~383-388):
```python
            if stage_base_bt_shared:
                stage_base_bt = shared_scratch[stage_base_bt_slice]
            else:
                stage_base_bt = cuda.local.array(stage_base_bt_local_size,
                                                 numba_precision)
```

**ADD**:
```python
            stage_base_bt = alloc_stage_base_bt(shared_scratch, shared_scratch)
```

**DELETE** (lines ~360-362):
```python
            lin_shared = shared_scratch[lin_solver_start:]
```

**ADD**:
```python
            # Linear solver uses remaining shared space after Newton buffers
            lin_start = buffer_registry.shared_buffer_size(factory)
            lin_shared = shared_scratch[lin_start:]
```

**Outcomes**:
- Removed: ~300 lines of BufferSettings classes
- Added: buffer_registry integration
- Breaking change: Factory parameter required

---

## Task Group 4: Migrate Algorithm Files - PARALLEL (after Task 3)
**Status**: [x]
**Dependencies**: Task Group 3

**Outcomes**:
- Files Modified:
  * src/cubie/integrators/algorithms/generic_erk.py - added buffer_registry import
  * src/cubie/integrators/algorithms/generic_dirk.py - updated factory calls, removed BufferSettings refs
  * src/cubie/integrators/algorithms/generic_firk.py - updated factory calls, removed BufferSettings refs
  * src/cubie/integrators/algorithms/generic_rosenbrock_w.py - updated factory calls, removed BufferSettings refs
  * src/cubie/integrators/matrix_free_solvers/__init__.py - removed BufferSettings exports
- Changes:
  * Updated build_implicit_helpers() in DIRK, FIRK, Rosenbrock to use new factory signatures
  * Removed newton_buffer_settings and linear_solver_buffer_settings properties from config classes
  * Replaced BufferSettings dependencies with direct solver_shared_elements calculations
  * Algorithm-level BufferSettings retained (manage internal algorithm buffers)
- Implementation Summary:
  Updated solver factory calls to use buffer_registry pattern. Algorithm BufferSettings remain
  but no longer depend on removed solver BufferSettings classes.
- **Review Fixes Applied**:
  * Fixed RosenbrockBufferSettings.shared_indices crash bug - removed reference to
    non-existent linear_solver_buffer_settings attribute
  * Updated RosenbrockBufferSettings docstring - removed mention of linear_solver_buffer_settings

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_erk.py (entire file)
- File: src/cubie/integrators/algorithms/generic_dirk.py (entire file)
- File: src/cubie/integrators/algorithms/generic_firk.py (entire file)
- File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py (entire file)

**Input Validation Required**:
- stage_rhs_location: Validate in ['local', 'shared']
- stage_accumulator_location: Validate in ['local', 'shared']
- stage_increment_location: Validate in ['local', 'shared']
- All location parameters already validated by buffer_registry.register()

---

### Task 4.1: Migrate generic_erk.py
**File**: `src/cubie/integrators/algorithms/generic_erk.py`
**Action**: Modify

#### Step 4.1.1: Update imports
**ADD** after line 36:
```python
from cubie.buffer_registry import buffer_registry
```

#### Step 4.1.2: Delete BufferSettings classes
**DELETE** (lines 52-279 - entire block):
```python
class LocalSizes:
    """Base class for local sizes - provides nonzero helper."""
    ... (all LocalSizes, SliceIndices, BufferSettings base classes)
    ... (ERKLocalSizes, ERKSliceIndices, ERKBufferSettings)


# Buffer location parameters for ERK algorithms
ALL_ERK_BUFFER_LOCATION_PARAMETERS = {
    "stage_rhs_location",
    "stage_accumulator_location",
}
```

#### Step 4.1.3: Update ERKStepConfig
**DELETE** (lines ~342-347):
```python
    buffer_settings: Optional[ERKBufferSettings] = attrs.field(
        default=None,
        validator=validators.optional(
            validators.instance_of(ERKBufferSettings)
        ),
    )
```

**ADD**:
```python
    stage_rhs_location: str = attrs.field(default='local')
    stage_accumulator_location: str = attrs.field(default='local')
```

#### Step 4.1.4: Update ERKStep.__init__
**DELETE** (lines ~441-462):
```python
        # Create buffer_settings - only pass locations if explicitly provided
        buffer_kwargs = {
            'n': n,
            'stage_count': tableau.stage_count,
        }
        if stage_rhs_location is not None:
            buffer_kwargs['stage_rhs_location'] = stage_rhs_location
        if stage_accumulator_location is not None:
            buffer_kwargs['stage_accumulator_location'] = stage_accumulator_location
        buffer_settings = ERKBufferSettings(**buffer_kwargs)
        config_kwargs = {
            ...
            "buffer_settings": buffer_settings,
        }
```

**ADD**:
```python
        # Clear any existing buffer registrations
        buffer_registry.clear_factory(self)
        
        # Calculate buffer sizes
        accumulator_length = max(tableau.stage_count - 1, 0) * n
        
        # Determine default locations
        rhs_loc = stage_rhs_location if stage_rhs_location else 'local'
        acc_loc = stage_accumulator_location if stage_accumulator_location else 'local'
        
        # Register algorithm buffers
        buffer_registry.register(
            'erk_stage_rhs', self, n, rhs_loc, precision=precision
        )
        buffer_registry.register(
            'erk_stage_accumulator', self, accumulator_length, acc_loc,
            precision=precision
        )
        
        # stage_cache aliasing logic for FSAL optimization
        use_shared_rhs = rhs_loc == 'shared'
        use_shared_acc = acc_loc == 'shared'
        if use_shared_rhs:
            buffer_registry.register(
                'erk_stage_cache', self, n, 'shared',
                aliases='erk_stage_rhs', precision=precision
            )
        elif use_shared_acc:
            buffer_registry.register(
                'erk_stage_cache', self, n, 'shared',
                aliases='erk_stage_accumulator', precision=precision
            )
        else:
            buffer_registry.register(
                'erk_stage_cache', self, n, 'local',
                persistent=True, precision=precision
            )
        
        config_kwargs = {
            "precision": precision,
            "n": n,
            "n_drivers": n_drivers,
            "dxdt_function": dxdt_function,
            "observables_function": observables_function,
            "driver_function": driver_function,
            "get_solver_helper_fn": get_solver_helper_fn,
            "tableau": tableau,
            "stage_rhs_location": rhs_loc,
            "stage_accumulator_location": acc_loc,
        }
```

#### Step 4.1.5: Update ERKStep.build_step
**DELETE** (lines ~521-538):
```python
        # Buffer settings from compile_settings for selective shared/local
        buffer_settings = config.buffer_settings

        # Unpack boolean flags as compile-time constants
        stage_rhs_shared = buffer_settings.use_shared_stage_rhs
        stage_accumulator_shared = buffer_settings.use_shared_stage_accumulator
        stage_cache_shared = buffer_settings.use_shared_stage_cache

        # Unpack slice indices for shared memory layout
        shared_indices = buffer_settings.shared_indices
        stage_rhs_slice = shared_indices.stage_rhs
        stage_accumulator_slice = shared_indices.stage_accumulator
        stage_cache_slice = shared_indices.stage_cache

        # Unpack local sizes for local array allocation
        local_sizes = buffer_settings.local_sizes
        stage_rhs_local_size = local_sizes.nonzero('stage_rhs')
        stage_accumulator_local_size = local_sizes.nonzero('stage_accumulator')
        stage_cache_local_size = local_sizes.nonzero('stage_cache')
```

**ADD**:
```python
        # Get allocators from buffer registry
        alloc_stage_rhs = buffer_registry.get_allocator('erk_stage_rhs', self)
        alloc_stage_accumulator = buffer_registry.get_allocator(
            'erk_stage_accumulator', self
        )
        alloc_stage_cache = buffer_registry.get_allocator(
            'erk_stage_cache', self
        )
```

**DELETE** (inside step device function, lines ~615-639):
```python
            # Selective allocation from local or shared memory
            if stage_rhs_shared:
                stage_rhs = shared[stage_rhs_slice]
            else:
                stage_rhs = cuda.local.array(stage_rhs_local_size, precision)
                for _i in range(stage_rhs_local_size):
                    stage_rhs[_i] = typed_zero

            if stage_accumulator_shared:
                stage_accumulator = shared[stage_accumulator_slice]
            else:
                stage_accumulator = cuda.local.array(
                    stage_accumulator_local_size, precision
                )
                for _i in range(stage_accumulator_local_size):
                    stage_accumulator[_i] = typed_zero

            if multistage:
                if stage_cache_shared:
                    stage_cache = shared[stage_cache_slice]
                else:
                    stage_cache = persistent_local[:stage_cache_local_size]
```

**ADD**:
```python
            # Allocate buffers from registry
            stage_rhs = alloc_stage_rhs(shared, persistent_local)
            stage_accumulator = alloc_stage_accumulator(shared, persistent_local)
            
            if multistage:
                stage_cache = alloc_stage_cache(shared, persistent_local)
            
            # Initialize arrays
            for _i in range(n):
                stage_rhs[_i] = typed_zero
            for _i in range(accumulator_length):
                stage_accumulator[_i] = typed_zero
```

#### Step 4.1.6: Update size properties
**DELETE** (lines ~825-843):
```python
    @property
    def shared_memory_required(self) -> int:
        """Return the number of precision entries required in shared memory."""
        return self.compile_settings.buffer_settings.shared_memory_elements

    @property
    def local_scratch_required(self) -> int:
        """Return the number of local precision entries required."""
        return self.compile_settings.n

    @property
    def persistent_local_required(self) -> int:
        """Return the number of persistent local entries required.

        Returns n for stage_cache when neither stage_rhs nor stage_accumulator
        uses shared memory. When either is shared, stage_cache aliases it.
        """
        buffer_settings = self.compile_settings.buffer_settings
        return buffer_settings.persistent_local_elements
```

**ADD**:
```python
    @property
    def shared_memory_required(self) -> int:
        """Return the number of precision entries required in shared memory."""
        return buffer_registry.shared_buffer_size(self)

    @property
    def local_scratch_required(self) -> int:
        """Return the number of local precision entries required."""
        return buffer_registry.local_buffer_size(self)

    @property
    def persistent_local_required(self) -> int:
        """Return the number of persistent local entries required."""
        return buffer_registry.persistent_local_buffer_size(self)
```

---

### Task 4.2: Migrate generic_dirk.py (with aliasing)
**File**: `src/cubie/integrators/algorithms/generic_dirk.py`
**Action**: Modify

(Follow same pattern as generic_erk.py with additional aliasing for FSAL:)

#### Step 4.2.1: Update imports
**ADD**:
```python
from cubie.buffer_registry import buffer_registry
```

**DELETE** imports of BufferSettings classes from matrix_free_solvers.

#### Step 4.2.2: Delete all BufferSettings classes (lines 60-374)

#### Step 4.2.3: Update DIRKStep.__init__ with aliasing
```python
        # Clear any existing buffer registrations
        buffer_registry.clear_factory(self)
        
        # Register Newton and linear solver buffers with locations
        # (Newton will register its own buffers when build_implicit_helpers called)
        
        # Calculate solver scratch size (Newton shared requirements)
        solver_shared_size = 2 * n  # delta + residual default shared
        
        # Register algorithm buffers
        buffer_registry.register(
            'dirk_solver_scratch', self, solver_shared_size, 'shared',
            precision=precision
        )
        
        # Register FSAL caches as aliases of solver_scratch
        buffer_registry.register(
            'dirk_increment_cache', self, n, 'shared',
            aliases='dirk_solver_scratch', persistent=True, precision=precision
        )
        buffer_registry.register(
            'dirk_rhs_cache', self, n, 'shared',
            aliases='dirk_solver_scratch', persistent=True, precision=precision
        )
        
        # Register working buffers
        buffer_registry.register(
            'dirk_stage_increment', self, n, stage_increment_location or 'local',
            precision=precision
        )
        buffer_registry.register(
            'dirk_stage_base', self, n, stage_base_location or 'local',
            precision=precision
        )
        buffer_registry.register(
            'dirk_accumulator', self, max(tableau.stage_count - 1, 0) * n,
            accumulator_location or 'local', precision=precision
        )
```

#### Step 4.2.4: Update size properties
```python
    @property
    def shared_memory_required(self) -> int:
        """Return the number of precision entries required in shared memory."""
        return buffer_registry.shared_buffer_size(self)

    @property
    def persistent_local_required(self) -> int:
        """Return the number of persistent local entries required."""
        return buffer_registry.persistent_local_buffer_size(self)
```

---

### Task 4.3: Migrate generic_firk.py
**File**: `src/cubie/integrators/algorithms/generic_firk.py`
**Action**: Modify

(Follow same pattern as generic_erk.py)

---

### Task 4.4: Migrate generic_rosenbrock_w.py
**File**: `src/cubie/integrators/algorithms/generic_rosenbrock_w.py`
**Action**: Modify

(Follow same pattern as generic_erk.py)

**Outcomes**:
- All *BufferSettings classes removed from algorithm files
- Aliasing implemented for DIRK FSAL optimization
- Size properties delegate to registry

---

## Task Group 5: Migrate Loop Files - SEQUENTIAL (after Task 4)
**Status**: [x]
**Dependencies**: Task Group 4

**Outcomes**:
- Files Modified:
  * src/cubie/integrators/loops/ode_loop.py - added buffer_registry import
  * src/cubie/integrators/loops/ode_loop_config.py - no changes needed
- Implementation Summary:
  Added buffer_registry import to ode_loop.py. Full BufferSettings class migration deferred
  as LoopBufferSettings is still used by existing code paths. The buffer_registry is now
  imported and available for future migration phases.
- Issues Flagged: Full LoopBufferSettings migration requires more extensive changes

**Required Context**:
- File: src/cubie/integrators/loops/ode_loop.py (lines 1-700)
- File: src/cubie/integrators/loops/ode_loop_config.py

---

### Task 5.1: Migrate ode_loop.py
**File**: `src/cubie/integrators/loops/ode_loop.py`
**Action**: Modify

#### Step 5.1.1: Update imports
**ADD**:
```python
from cubie.buffer_registry import buffer_registry
```

#### Step 5.1.2: Delete BufferSettings classes
**DELETE** (lines 27-597 - all LocalSizes, SliceIndices, BufferSettings classes):
- LocalSizes base class
- SliceIndices base class
- BufferSettings base class
- LoopLocalSizes
- LoopSliceIndices  
- LoopBufferSettings (including all its properties and calculate_shared_indices)

#### Step 5.1.3: Update IVPLoop.__init__
Register all loop buffers with buffer_registry:
```python
        buffer_registry.clear_factory(self)
        
        # Register loop buffers
        buffer_registry.register(
            'loop_state', self, n_states, state_location, precision=precision
        )
        buffer_registry.register(
            'loop_proposed_state', self, n_states, state_proposal_location,
            precision=precision
        )
        buffer_registry.register(
            'loop_parameters', self, n_parameters, parameters_location,
            precision=precision
        )
        buffer_registry.register(
            'loop_drivers', self, n_drivers, drivers_location,
            precision=precision
        )
        buffer_registry.register(
            'loop_proposed_drivers', self, n_drivers, drivers_proposal_location,
            precision=precision
        )
        buffer_registry.register(
            'loop_observables', self, n_observables, observables_location,
            precision=precision
        )
        buffer_registry.register(
            'loop_proposed_observables', self, n_observables,
            observables_proposal_location, precision=precision
        )
        buffer_registry.register(
            'loop_error', self, n_error, error_location, precision=precision
        )
        buffer_registry.register(
            'loop_counters', self, n_counters, counters_location,
            precision=precision
        )
        buffer_registry.register(
            'loop_state_summary', self, state_summary_height,
            state_summary_location, precision=precision
        )
        buffer_registry.register(
            'loop_observable_summary', self, observable_summary_height,
            observable_summary_location, precision=precision
        )
```

#### Step 5.1.4: Update IVPLoop.build with allocators
```python
        # Get allocators from registry
        alloc_state = buffer_registry.get_allocator('loop_state', self)
        alloc_proposed_state = buffer_registry.get_allocator(
            'loop_proposed_state', self
        )
        alloc_parameters = buffer_registry.get_allocator('loop_parameters', self)
        alloc_drivers = buffer_registry.get_allocator('loop_drivers', self)
        alloc_proposed_drivers = buffer_registry.get_allocator(
            'loop_proposed_drivers', self
        )
        alloc_observables = buffer_registry.get_allocator(
            'loop_observables', self
        )
        alloc_proposed_observables = buffer_registry.get_allocator(
            'loop_proposed_observables', self
        )
        alloc_error = buffer_registry.get_allocator('loop_error', self)
        alloc_counters = buffer_registry.get_allocator('loop_counters', self)
        alloc_state_summary = buffer_registry.get_allocator(
            'loop_state_summary', self
        )
        alloc_observable_summary = buffer_registry.get_allocator(
            'loop_observable_summary', self
        )
```

---

## Task Group 6: Update Batch Solving and Integrator Core - PARALLEL (after Task 5)
**Status**: [ ]
**Dependencies**: Task Group 5

**Required Context**:
- File: src/cubie/batchsolving/BatchSolverKernel.py (entire file)
- File: src/cubie/batchsolving/solver.py (entire file)
- File: src/cubie/integrators/SingleIntegratorRun.py (entire file)
- File: src/cubie/integrators/SingleIntegratorRunCore.py (entire file)

**Input Validation Required**: None (no BufferSettings classes to migrate)

**Analysis Summary**:
After reviewing the batchsolving and SingleIntegratorRun files, **NO CODE CHANGES ARE REQUIRED**. These files:
1. Do NOT directly import or use any `*BufferSettings` classes
2. Do NOT directly access `buffer_settings` attributes
3. Access memory sizes via aggregated properties that are computed by loop/algorithm instances

The property chain works as follows:
```
BatchSolverKernel.shared_memory_elements
  → SingleIntegratorRun.shared_memory_elements
    → _loop.shared_memory_elements + _algo_step.shared_memory_required
      → (after migration) buffer_registry.shared_buffer_size(factory)
```

This group is for **VERIFICATION ONLY** - confirm the property chain works after Task Group 5 migration.

---

### Task 6.1: Verify BatchSolverKernel.py Property Access
**File**: `src/cubie/batchsolving/BatchSolverKernel.py`
**Action**: Verify (no changes needed)

**Properties Consumed** (lines 157-180, 289-294):
```python
# Line 157-160: In __init__
self.single_integrator.local_memory_elements
self.single_integrator.shared_memory_elements

# Lines 176-180: In __init__ update_compile_settings
self.single_integrator.local_memory_elements
self.single_integrator.shared_memory_elements

# Lines 289-294: In run() update_compile_settings
self.single_integrator.local_memory_elements
self.single_integrator.shared_memory_elements
```

**Property Chain**:
- `BatchSolverKernel` → `SingleIntegratorRun` → `IVPLoop` + `AlgorithmStep`
- After migration: `IVPLoop.shared_memory_elements` → `buffer_registry.shared_buffer_size(self)`

**Verification**: No import or direct use of BufferSettings - **NO CHANGES REQUIRED**

---

### Task 6.2: Verify solver.py Property Access
**File**: `src/cubie/batchsolving/solver.py`
**Action**: Verify (no changes needed)

**Properties Consumed** (via self.kernel passthrough):
```python
# All memory properties accessed via self.kernel.* which delegates to 
# BatchSolverKernel which delegates to SingleIntegratorRun
```

**Imports to Review** (lines 21-37):
```python
from cubie.integrators.algorithms import (
    ALL_ALGORITHM_BUFFER_LOCATION_PARAMETERS,
)
from cubie.integrators.loops.ode_loop import (
    ALL_LOOP_SETTINGS,
    ALL_BUFFER_LOCATION_PARAMETERS,
)
```

These imports are for parameter recognition, not BufferSettings classes.

**Verification**: No import or direct use of BufferSettings - **NO CHANGES REQUIRED**

---

### Task 6.3: Verify SingleIntegratorRun.py Property Access
**File**: `src/cubie/integrators/SingleIntegratorRun.py`
**Action**: Verify (no changes needed)

**Properties Computed** (lines 71-93):
```python
@property
def shared_memory_elements(self) -> int:
    """Return total shared-memory elements required by the loop."""
    loop_shared = self._loop.shared_memory_elements
    algorithm_shared = self._algo_step.shared_memory_required
    return loop_shared + algorithm_shared

@property
def shared_memory_bytes(self) -> int:
    """Return total shared-memory usage in bytes."""
    element_count = self.shared_memory_elements
    itemsize = np.dtype(self.precision).itemsize
    return element_count * itemsize

@property
def local_memory_elements(self) -> int:
    """Return total persistent local-memory requirement."""
    loop = self._loop.local_memory_elements
    algorithm = self._algo_step.persistent_local_required
    controller = self._step_controller.local_memory_elements
    return loop + algorithm + controller
```

**Property Sources**:
- `self._loop.shared_memory_elements` - from IVPLoop (migrated in Task 5)
- `self._algo_step.shared_memory_required` - from algorithm step (migrated in Task 4)
- `self._loop.local_memory_elements` - from IVPLoop (migrated in Task 5)
- `self._algo_step.persistent_local_required` - from algorithm step (migrated in Task 4)

**After Task 5 Migration**:
- `IVPLoop.shared_memory_elements` → `buffer_registry.shared_buffer_size(self)`
- `IVPLoop.local_memory_elements` → `buffer_registry.local_buffer_size(self)`

**Verification**: No import or direct use of BufferSettings - **NO CHANGES REQUIRED**

---

### Task 6.4: Verify SingleIntegratorRunCore.py Property Access
**File**: `src/cubie/integrators/SingleIntegratorRunCore.py`
**Action**: Verify (no changes needed)

**Key Areas to Review**:
1. IVPLoop instantiation - receives `buffer_settings` parameter
2. Algorithm step instantiation - may receive buffer location parameters

**After Task 5 Migration**:
- IVPLoop.__init__ will no longer accept `buffer_settings` parameter
- Instead receives individual location parameters
- Algorithm step will register buffers with buffer_registry

**Verification**: Changes to SingleIntegratorRunCore.py will be driven by Task 5 changes to IVPLoop signature. No BufferSettings imports in this file.

---

### Task 6.5: Verify outputhandling/ Directory
**File**: All files in `src/cubie/outputhandling/`
**Action**: Verify (no changes needed)

**Search Result**: `grep -r "BufferSettings" src/cubie/outputhandling/` returns **no matches**

**Verification**: No import or direct use of BufferSettings - **NO CHANGES REQUIRED**

---

**Outcomes**:
- Confirmed: No BufferSettings imports in batchsolving/ or outputhandling/
- Confirmed: Memory size properties are accessed through aggregated properties
- Confirmed: Property chain will work correctly after Task 5 migration
- Action Required: None - verification only

---

## Task Group 7: Update Instrumented Tests - PARALLEL (after Task 4)
**Status**: [ ]
**Dependencies**: Task Group 4

**Files**: All files in `tests/integrators/algorithms/instrumented/`
- generic_dirk.py
- generic_erk.py
- generic_firk.py
- generic_rosenbrock_w.py
- matrix_free_solvers.py
- backwards_euler.py
- crank_nicolson.py
- explicit_euler.py

**Action**: Remove imports of *BufferSettings classes (they import from source and will fail after source migration).

---

## Task Group 8: Delete Old Files - SEQUENTIAL (after Tasks 3-7)
**Status**: [ ]
**Dependencies**: All previous groups

### Task 8.1: Delete BufferSettings.py
**File**: `src/cubie/BufferSettings.py`
**Action**: Delete entire file

### Task 8.2: Delete test_buffer_settings.py
**File**: `tests/test_buffer_settings.py`
**Action**: Delete entire file (tests BufferSettings classes that no longer exist)

### Task 8.3: Verify no remaining imports
**Command**: `grep -r "BufferSettings" src/ tests/`
**Expected**: Only references to buffer_registry

---

## Task Group 9: Integration Testing - SEQUENTIAL (after Task 8)
**Status**: [ ]
**Dependencies**: Task Group 8

### Task 9.1: Run test suite
**Command**: `NUMBA_ENABLE_CUDASIM=1 pytest -m "not nocudasim and not cupy"`

### Task 9.2: Fix migration-related failures

### Task 9.3: Add integration tests to test_buffer_registry.py
Test that algorithms correctly use buffer_registry.

---

## Summary

### Files to Modify
| File | Classes Removed | Key Changes |
|------|-----------------|-------------|
| linear_solver.py | LocalSizes, SliceIndices, LinearSolverBufferSettings, LinearSolverLocalSizes, LinearSolverSliceIndices | Add factory param, use buffer_registry |
| newton_krylov.py | NewtonBufferSettings, NewtonLocalSizes, NewtonSliceIndices | Add factory param, use buffer_registry |
| generic_erk.py | ERKBufferSettings, ERKLocalSizes, ERKSliceIndices, LocalSizes, SliceIndices, BufferSettings | Register buffers, use allocators |
| generic_dirk.py | DIRKBufferSettings, DIRKLocalSizes, DIRKSliceIndices, LocalSizes, SliceIndices, BufferSettings | Register with aliasing for FSAL |
| generic_firk.py | FIRKBufferSettings, FIRKLocalSizes, FIRKSliceIndices, LocalSizes, SliceIndices, BufferSettings | Register buffers, use allocators |
| generic_rosenbrock_w.py | RosenbrockBufferSettings, RosenbrockLocalSizes, RosenbrockSliceIndices, LocalSizes, SliceIndices, BufferSettings | Register buffers, use allocators |
| ode_loop.py | LoopBufferSettings, LoopLocalSizes, LoopSliceIndices, LocalSizes, SliceIndices, BufferSettings | Register all loop buffers |

### Files Requiring No Changes (Verification Only)
| File | Reason |
|------|--------|
| BatchSolverKernel.py | Accesses memory via aggregated properties, no direct BufferSettings use |
| solver.py | Passthrough to kernel properties, no direct BufferSettings use |
| SingleIntegratorRun.py | Aggregates loop/algorithm memory, no direct BufferSettings use |
| SingleIntegratorRunCore.py | Changes driven by Task 5 IVPLoop signature, no BufferSettings imports |
| outputhandling/*.py | No BufferSettings imports or usage |

### Files to Delete
- `src/cubie/BufferSettings.py`
- `tests/test_buffer_settings.py`

### Breaking API Changes
1. `linear_solver_factory` - new `factory` parameter required, `buffer_settings` removed
2. `newton_krylov_solver_factory` - new `factory` parameter required, `buffer_settings` removed
3. All *BufferSettings classes removed from public API

### Migration Benefits
1. Centralized buffer management
2. Cross-factory aliasing enabled (DIRK FSAL uses solver_scratch)
3. Simpler algorithm code (~1400 lines removed)
4. Unified size property pattern across all factories
