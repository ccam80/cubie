# Agent Plan: Scaled Tolerance Refactoring Improvements

## Overview

This plan addresses review feedback on PR #475 (refactor: extract scaled tolerance to ScaledNorm CUDAFactory). The goal is to improve the architecture of tolerance handling, cache invalidation, and code organization in the Newton and Krylov matrix-free solvers.

## Component Descriptions

### 1. MatrixFreeSolverConfig Base Class (Enhanced)

**Location**: `src/cubie/integrators/matrix_free_solvers/base_solver.py`

**Current State**: Contains only `precision`, `n`, and type accessor properties.

**Target State**:
- Add `max_iters` field with validator `inrangetype_validator(int, 1, 32767)` 
- Add `norm_device_function` field (Optional[Callable]) with `eq=False` to store the compiled norm function
- This enables cache invalidation when norm changes

**Expected Behavior**:
- Subclasses inherit `max_iters` instead of defining it separately
- Norm device function stored in config enables automatic cache invalidation when updated

### 2. MatrixFreeSolver CUDAFactory Base Class (New)

**Location**: `src/cubie/integrators/matrix_free_solvers/base_solver.py`

**Purpose**: Provide shared update scaffolding for tolerance parameter mapping

**Attributes**:
- `settings_prefix: str` - e.g., `"krylov_"` or `"newton_"`
- `norm: ScaledNorm` - Factory for scaled norm computation (owned by base)

**Expected Behavior**:
- Constructor accepts `precision`, `n`, and tolerance kwargs
- Creates and owns `self.norm = ScaledNorm(precision, n, atol=..., rtol=...)`
- `update()` method handles prefix mapping:
  - Receives update dict with e.g., `krylov_atol`, `newton_atol`
  - Uses `settings_prefix` to identify tolerance keys
  - Maps `{prefix}atol` → `atol`, `{prefix}rtol` → `rtol`, `{prefix}max_iters` → `max_iters`
  - Updates norm factory with unprefixed tolerance values
  - Updates config with new `norm_device_function` reference
  - Cache invalidates automatically through config update mechanism

### 3. LinearSolverConfig (Modified)

**Location**: `src/cubie/integrators/matrix_free_solvers/linear_solver.py`

**Changes**:
- Remove `krylov_atol` and `krylov_rtol` fields (move to norm factory)
- Keep `_krylov_tolerance` for legacy scalar tolerance
- Rename `max_linear_iters` to inherit from base `max_iters` or keep as alias
- Add `norm_device_function` reference from base class

**Expected Behavior**:
- `settings_dict` property retrieves `krylov_atol`/`krylov_rtol` from solver's norm factory
- Validation still occurs through norm's field validators

### 4. LinearSolver (Modified)

**Location**: `src/cubie/integrators/matrix_free_solvers/linear_solver.py`

**Changes**:
- Inherit from `MatrixFreeSolver` instead of `CUDAFactory` directly
- Set `settings_prefix = "krylov_"`
- Remove direct `self.norm` creation (handled by base)
- Update `update()` method:
  - Copy updates dict before modifying
  - Pop `krylov_atol`, `krylov_rtol` from copy
  - Call base class update with norm updates
  - Remove manual `_invalidate_cache()` call
- Update `build()`:
  - Get `scaled_norm_fn` from `self.compile_settings.norm_device_function` instead of `self.norm.device_function`

**Expected Behavior**:
- `krylov_atol`/`krylov_rtol` properties delegate to `self.norm.atol`/`self.norm.rtol`
- Cache invalidation happens automatically when config's `norm_device_function` is updated

### 5. NewtonKrylovConfig (Modified)

**Location**: `src/cubie/integrators/matrix_free_solvers/newton_krylov.py`

**Changes**:
- Remove `newton_atol` and `newton_rtol` fields (move to norm factory)
- Keep `_newton_tolerance` for legacy scalar tolerance
- Inherit `max_iters` from base or alias as `max_newton_iters`

**Expected Behavior**:
- `settings_dict` property retrieves `newton_atol`/`newton_rtol` from solver's norm factory

### 6. NewtonKrylov (Modified)

**Location**: `src/cubie/integrators/matrix_free_solvers/newton_krylov.py`

**Changes**:
- Inherit from `MatrixFreeSolver` instead of `CUDAFactory` directly
- Set `settings_prefix = "newton_"`
- Remove direct `self.norm` creation (handled by base)
- Update `update()` method:
  - Copy updates dict before modifying
  - Pop `newton_atol`, `newton_rtol` from copy
  - Call base class update with norm updates
  - Remove manual `_invalidate_cache()` call
- Update `build()`:
  - Get `scaled_norm_fn` from `self.compile_settings.norm_device_function`

## Integration Points

### With ScaledNorm Factory

- `MatrixFreeSolver` creates `ScaledNorm` in constructor
- Stores `self.norm.device_function` in `compile_settings.norm_device_function`
- When `update()` is called with tolerance changes:
  1. Tolerance updates passed to `self.norm.update()`
  2. New `self.norm.device_function` retrieved
  3. Config updated with new device function
  4. Cache invalidates automatically

### With Step Controllers

- Step controllers also use `atol`/`rtol` parameters
- By copying the updates dict before popping solver-specific keys, the original `atol`/`rtol` values remain available for step controller updates
- This prevents solvers from "consuming" tolerance values meant for controllers

### With CUDAFactory Pattern

- `update_compile_settings()` is the standard way to trigger cache invalidation
- By including `norm_device_function` in compile settings, changes to it (when norm is rebuilt) trigger automatic invalidation
- No need for manual `_invalidate_cache()` calls

## Edge Cases

### Empty Updates Dict
- `update({})` should be a no-op, returning empty set of recognized params

### Mixed Prefixed and Unprefixed Tolerance
- If both `krylov_atol` and `atol` are in updates dict:
  - Pop `krylov_atol` for solver's norm
  - Leave `atol` for step controller
  - This is the designed behavior

### Nested LinearSolver in NewtonKrylov
- `NewtonKrylov` owns a `LinearSolver` instance
- `NewtonKrylov.update()` forwards krylov-prefixed params to `self.linear_solver.update()`
- Each solver handles its own tolerance updates independently

### Updates Dict with Unrecognized Keys
- Silent mode ignores unrecognized keys
- Non-silent mode raises KeyError for truly invalid keys
- Valid solver params not applicable to this solver type are recognized but ignored

## Dependencies

### Required Imports in base_solver.py
```python
from typing import Callable, Optional, Set, Dict, Any
from cubie.CUDAFactory import CUDAFactory
from cubie.integrators.norms import ScaledNorm
```

### Required Imports in linear_solver.py
```python
from cubie.integrators.matrix_free_solvers.base_solver import (
    MatrixFreeSolverConfig,
    MatrixFreeSolver,
)
```

## Data Structures

### Tolerance Update Flow
```
User Input: {"newton_atol": array([1e-6]), "atol": array([1e-5])}
           ↓
NewtonKrylov.update():
  1. all_updates = input.copy()
  2. norm_atol = all_updates.pop("newton_atol")  # removes from copy
  3. self.norm.update({"atol": norm_atol})
  4. self.update_compile_settings({"norm_device_function": self.norm.device_function})
           ↓
Original dict unchanged: still has "atol" for step controller
```

### Config Structure (After Refactoring)
```python
@define
class MatrixFreeSolverConfig:
    precision: PrecisionDType
    n: int
    max_iters: int = field(default=100, validator=inrangetype_validator(int, 1, 32767))
    norm_device_function: Optional[Callable] = field(default=None, eq=False)

@define
class LinearSolverConfig(MatrixFreeSolverConfig):
    operator_apply: Optional[Callable]
    preconditioner: Optional[Callable]
    linear_correction_type: str
    _krylov_tolerance: float  # legacy scalar
    # krylov_atol, krylov_rtol REMOVED - use norm factory
    preconditioned_vec_location: str
    temp_location: str
    use_cached_auxiliaries: bool
```

## Instrumented Test Files

The following instrumented test files must be updated to mirror source changes:
- `tests/integrators/algorithms/instrumented/matrix_free_solvers.py`

Changes needed:
- Update `InstrumentedLinearSolver` and `InstrumentedNewtonKrylov` to inherit from new base
- Mirror tolerance handling changes
- Mirror build() changes to use config's norm_device_function
