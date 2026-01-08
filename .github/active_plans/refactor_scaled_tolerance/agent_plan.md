# Agent Plan: Refactoring Scaled Tolerance in Newton and Krylov Solvers

## Overview

This plan describes the architectural changes needed to refactor the scaled tolerance implementation in CuBIE's Newton and Krylov solvers. The refactoring eliminates code duplication and establishes reusable components that follow existing CuBIE patterns.

---

## Component 1: tol_converter in _utils.py

### Current State
The `tol_converter` function exists in three locations:
1. `src/cubie/integrators/step_control/adaptive_step_controller.py` (lines 22-55)
2. `src/cubie/integrators/matrix_free_solvers/newton_krylov.py` (lines 35-66)
3. `src/cubie/integrators/matrix_free_solvers/linear_solver.py` (lines 33-64)

All three implementations are nearly identical, converting a scalar or array tolerance value into an ndarray of shape `(n,)` with the appropriate precision dtype.

### Expected Behavior After Refactor
- A single `tol_converter` function in `cubie._utils`
- Function signature: `tol_converter(value, self_) -> ndarray` where `self_` has `n` and `precision` attributes
- The three original locations import from `_utils` instead of defining locally
- Behavior is identical to current implementations

### Integration Points
- Used as an attrs `Converter` with `takes_self=True`
- Called during attrs class initialization
- Requires access to `self_.n` (int) and `self_.precision` (PrecisionDType)

### Dependencies
- `numpy.isscalar`, `numpy.full`, `numpy.asarray`, `numpy.ndarray`
- No CUDA dependencies

---

## Component 2: MatrixFreeSolverConfig Base Class

### Current State
`LinearSolverConfig` and `NewtonKrylovConfig` share common attributes:
- `precision` (PrecisionDType)
- `n` (int, state vector size)
- Numba precision properties
- Simsafe precision properties

However, they duplicate these definitions independently.

### Expected Behavior After Refactor

#### MatrixFreeSolverConfig (base attrs class)
Located in: `integrators/matrix_free_solvers/base_solver.py`

Common attributes extracted to base:
- `precision: PrecisionDType` - with converter and validator
- `n: int` - vector size with `getype_validator(int, 1)`
- Property `numba_precision` - returns Numba type via `from_dtype`
- Property `simsafe_precision` - returns simsafe type via `simsafe_dtype`

#### MatrixFreeSolver (optional base CUDAFactory class)
Base class for matrix-free solver factories:
- Common `update()` pattern
- Shared property accessors for `precision`, `n`
- Simplified inheritance for `LinearSolver` and `NewtonKrylov`

### Integration Points
- `LinearSolverConfig` inherits from `MatrixFreeSolverConfig`
- `NewtonKrylovConfig` inherits from `MatrixFreeSolverConfig`
- Maintains full backward compatibility with current config APIs

### Dependencies
- `cubie._utils`: precision_converter, precision_validator, getype_validator
- `numba.from_dtype`
- `cubie.cuda_simsafe.from_dtype` (as simsafe_dtype)

---

## Component 3: ScaledNorm CUDAFactory

### Current State
The scaled norm computation appears inline in multiple device functions:
1. `adaptive_PID_controller.py` - controller_PID (lines 234-242)
2. `linear_solver.py` - linear_solver (lines 524-538, and repeated in loop)
3. `newton_krylov.py` - newton_krylov_solver (lines 438-450, and in backtrack loop)

The pattern is:
```python
nrm2 = typed_zero
for i in range(n):
    value_i = values[i]
    ref_i = reference[i]
    abs_ref = abs(ref_i)
    tol_i = atol[i] + rtol[i] * abs_ref
    tol_i = max(tol_i, floor)
    abs_val = abs(value_i)
    ratio = abs_val / tol_i
    nrm2 += ratio * ratio
nrm2 = nrm2 * inv_n
```

### Expected Behavior After Refactor

#### ScaledNormConfig (attrs class)
Located in: `integrators/norms.py`

Attributes:
- `precision: PrecisionDType` - numerical precision
- `n: int` - vector size
- `atol: ndarray` - absolute tolerance (uses `tol_converter` from `_utils`)
- `rtol: ndarray` - relative tolerance (uses `tol_converter` from `_utils`)

Properties:
- `numba_precision` - Numba type
- `inv_n` - precomputed `1.0 / n` in precision type
- `tol_floor` - minimum tolerance floor (e.g., 1e-16)

#### ScaledNorm (CUDAFactory subclass)
Located in: `integrators/norms.py`

Behavior:
- Inherits from `CUDAFactory`
- Implements `build()` returning `ScaledNormCache`
- Compiles a `scaled_norm` CUDA device function

#### scaled_norm Device Function
Signature: `scaled_norm(values, reference_values) -> float`

The device function:
1. Iterates over vector elements
2. Computes per-element scaled error: `|values[i]| / (atol[i] + rtol[i] * |ref[i]|)`
3. Accumulates squared ratio
4. Returns mean squared norm (sum / n)

Return value interpretation:
- `<= 1.0`: Within tolerance
- `> 1.0`: Exceeds tolerance

### Integration Points

Each solver owns its own `ScaledNorm` factory instance:

```
LinearSolver
├── compile_settings: LinearSolverConfig
├── norm: ScaledNorm (owns krylov_atol, krylov_rtol)
└── device_function: linear_solver

NewtonKrylov
├── compile_settings: NewtonKrylovConfig
├── norm: ScaledNorm (owns newton_atol, newton_rtol)
├── linear_solver: LinearSolver
└── device_function: newton_krylov_solver
```

#### Update Propagation
When `update(atol=..., rtol=...)` is called:
1. Solver delegates to `norm.update(atol=..., rtol=...)`
2. Norm factory invalidates its cache
3. Solver invalidates its cache
4. Next `device_function` access rebuilds with new norm

### Dependencies
- `cubie.CUDAFactory` and `CUDAFunctionCache`
- `cubie._utils.tol_converter`
- `cubie.cuda_simsafe` for CUDA types
- `numba.cuda` for device function compilation

---

## Component 4: Updated LinearSolver

### Changes Required
1. Config inherits from `MatrixFreeSolverConfig`
2. Remove local `tol_converter` - import from `_utils`
3. Move `krylov_atol` and `krylov_rtol` to owned `ScaledNorm` factory
4. Update `build()` to capture `norm.device_function` in closure
5. Replace inline norm computation with call to `scaled_norm`

### Expected Interactions
- `__init__` creates `ScaledNorm` instance
- `update()` propagates atol/rtol updates to norm factory
- `build()` uses `norm.device_function` for convergence checks

### Tolerance Handling
Currently: `LinearSolverConfig` has `krylov_atol`, `krylov_rtol`, `_krylov_tolerance`

After: 
- `krylov_tolerance` remains on config (legacy scalar tolerance, deprecated)
- `krylov_atol`, `krylov_rtol` move to `norm.compile_settings`
- Config exposes delegating properties: `krylov_atol` -> `norm.atol`

---

## Component 5: Updated NewtonKrylov

### Changes Required
1. Config inherits from `MatrixFreeSolverConfig`
2. Remove local `tol_converter` - import from `_utils`
3. Move `newton_atol` and `newton_rtol` to owned `ScaledNorm` factory
4. Update `build()` to capture `norm.device_function` in closure
5. Replace inline norm computation with call to `scaled_norm`

### Expected Interactions
- `__init__` creates `ScaledNorm` instance (separate from LinearSolver's)
- Each level (Newton, Krylov) has independent tolerance configuration
- `update()` propagates to both `norm` and `linear_solver`

### Tolerance Handling
Currently: `NewtonKrylovConfig` has `newton_atol`, `newton_rtol`, `_newton_tolerance`

After:
- `newton_tolerance` remains on config (legacy scalar tolerance, deprecated)
- `newton_atol`, `newton_rtol` move to `norm.compile_settings`
- Config exposes delegating properties: `newton_atol` -> `norm.atol`

---

## Component 6: Updated Instrumented Test Files

### Files Requiring Changes
- `tests/integrators/algorithms/instrumented/matrix_free_solvers.py`

### Changes Required
1. Import `tol_converter` from `cubie._utils` (if used)
2. Update `InstrumentedLinearSolver` to match production `LinearSolver` changes
3. Update `InstrumentedNewtonKrylov` to match production `NewtonKrylov` changes
4. Ensure scaled norm calls match new pattern

### Key Constraint
The instrumented versions must remain functionally identical to production except for logging additions. Any structural change to the production code must be replicated.

---

## Edge Cases

### Array Broadcasting
`tol_converter` handles:
- Scalar input → broadcasted to `(n,)` array
- Single-element array `[val]` with `n > 1` → broadcasted to `(n,)` 
- Full array `(n,)` → passed through with dtype conversion
- Wrong size array → raises `ValueError`

### Zero Reference Values
When `reference_values[i] = 0`, the scaled norm uses:
```python
tol_i = atol[i] + rtol[i] * 0.0 = atol[i]
```
This requires `atol > 0` to avoid division by zero. The tolerance floor (`1e-16`) provides a safety net.

### Convergence Threshold
The norm returns `sum(ratio²) / n`. Convergence test is:
```python
converged = norm_squared <= 1.0
```
This differs from `norm <= tolerance` because tolerance is embedded in the ratio computation.

---

## File Change Summary

### New Files
- `src/cubie/integrators/matrix_free_solvers/base_solver.py`
- `src/cubie/integrators/norms.py`

### Modified Files
- `src/cubie/_utils.py` (add `tol_converter`)
- `src/cubie/integrators/step_control/adaptive_step_controller.py` (import `tol_converter`)
- `src/cubie/integrators/matrix_free_solvers/linear_solver.py` (major refactor)
- `src/cubie/integrators/matrix_free_solvers/newton_krylov.py` (major refactor)
- `src/cubie/integrators/matrix_free_solvers/__init__.py` (export base classes)
- `tests/integrators/algorithms/instrumented/matrix_free_solvers.py` (sync with production)

### Files to Verify (Tests)
- `tests/integrators/matrix_free_solvers/test_linear_solver.py`
- `tests/integrators/matrix_free_solvers/test_newton_krylov.py`
- `tests/integrators/step_control/` (all controller tests)

---

## Architectural Constraints

1. **CUDAFactory Pattern**: All CUDA-generating components must inherit from `CUDAFactory`, use `setup_compile_settings()`, and return `CUDAFunctionCache` subclass from `build()`

2. **Buffer Registry**: No changes to buffer registration patterns. Norm factory has no buffers (pure computation)

3. **Cache Invalidation**: When norm settings change, both norm and parent solver caches must invalidate

4. **Attrs Conventions**: 
   - Leading underscore for float attributes with precision wrapping
   - Use `tol_converter` with `Converter(tol_converter, takes_self=True)`
   - Validators from `cubie._utils`

5. **CUDA Device Code**: 
   - Prefer predicated commit over conditional branching
   - Inline device functions where possible
   - Use compile-time constants captured in closure
