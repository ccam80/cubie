# Agent Plan: Scaled Tolerance in Newton-Krylov Solver

## Problem Statement

The Newton-Krylov solver currently uses a simple L2 norm with scalar tolerance for convergence checking. This approach fails for poorly-scaled ODE systems where state variables have vastly different magnitudes (e.g., nA currents alongside V voltages). The step controller already implements per-element scaled tolerances successfully; this feature applies the same pattern to the Newton and Krylov solvers.

## Target Components

### 1. LinearSolverConfig (linear_solver.py)

**Current State:**
- Has `_krylov_tolerance: float` field with property returning `self.precision(self._krylov_tolerance)`
- Single scalar tolerance used for all state elements

**Expected Behavior:**
- Add `krylov_atol: ndarray` field with `tol_converter` and `float_array_validator`
- Add `krylov_rtol: ndarray` field with `tol_converter` and `float_array_validator`
- The converter must have access to `self.n` and `self.precision` (use `Converter(tol_converter, takes_self=True)`)
- Keep `_krylov_tolerance` as fallback/default value generator when atol/rtol not explicitly set
- Add `settings_dict` entries for the new tolerance arrays

### 2. LinearSolver.build() (linear_solver.py)

**Current State:**
```python
tol_squared = precision_numba(krylov_tolerance * krylov_tolerance)
# ... in loop:
acc += residual_value * residual_value
# ... 
converged = acc <= tol_squared
```

**Expected Behavior:**
- Capture `krylov_atol` and `krylov_rtol` arrays in closure
- Replace norm computation with scaled norm:
  ```python
  inv_n = precision_numba(1.0 / n)
  # In convergence check loop:
  for i in range(n_val):
      error_i = max(abs(residual_value), precision(1e-16))
      tol_i = krylov_atol[i] + krylov_rtol[i] * max(abs(x[i]), typed_zero)
      ratio = error_i / tol_i
      acc += ratio * ratio
  acc = acc * inv_n
  converged = acc <= typed_one
  ```
- The reference value for rtol scaling should be the current iterate `x`

### 3. NewtonKrylovConfig (newton_krylov.py)

**Current State:**
- Has `_newton_tolerance: float` field with property
- Single scalar tolerance for Newton convergence

**Expected Behavior:**
- Add `newton_atol: ndarray` field with `tol_converter` and `float_array_validator`
- Add `newton_rtol: ndarray` field with `tol_converter` and `float_array_validator`
- Converter needs `self.n` and `self.precision` access
- Keep `_newton_tolerance` as fallback/default
- Add `settings_dict` entries for new tolerance arrays

### 4. NewtonKrylov.build() (newton_krylov.py)

**Current State:**
```python
tol_squared = numba_precision(newton_tolerance * newton_tolerance)
# ...
norm2_prev = typed_zero
for i in range(n_val):
    residual_value = residual[i]
    # ...
    norm2_prev += residual_value * residual_value
# ...
converged = norm2_prev <= tol_squared
```

**Expected Behavior:**
- Capture `newton_atol` and `newton_rtol` arrays in closure
- Replace norm computation with scaled norm pattern
- Reference value for rtol should be current `stage_increment` values
- Modify both the initial convergence check and the backtracking residual check

### 5. tol_converter Function Location

**Current Location:** `adaptive_step_controller.py` (local to that module)

**Expected Behavior:**
- The converter pattern should be replicated in the solver modules OR
- A shared converter could be created in `_utils.py` if reuse is desired
- The converter must work with any attrs config class that has `n` and `precision` attributes

**Converter Pattern Reference:**
```python
def tol_converter(
    value: Union[float, ArrayLike],
    self_: "ConfigClass",
) -> ndarray:
    if isscalar(value):
        tol = full(self_.n, value, dtype=self_.precision)
    else:
        tol = asarray(value, dtype=self_.precision)
        if tol.shape[0] == 1 and self_.n > 1:
            tol = full(self_.n, tol[0], dtype=self_.precision)
        elif tol.shape[0] != self_.n:
            raise ValueError("tol must have shape (n,).")
    return tol
```

## Integration Points

### Parameter Flow from Algorithms

Implicit algorithms (e.g., `BackwardsEulerStep`, `CrankNicolsonStep`, `generic_dirk.py`) create the Newton-Krylov solver chain. The new tolerance parameters must flow through:

1. **Algorithm __init__** accepts `newton_atol`, `newton_rtol`, `krylov_atol`, `krylov_rtol` via `**kwargs`
2. **ODEImplicitStep** filters these to appropriate solver constructors
3. **NewtonKrylov/LinearSolver** receive and configure tolerance arrays

The `_NEWTON_KRYLOV_PARAMS` and `_LINEAR_SOLVER_PARAMS` frozensets in `ode_implicitstep.py` must be updated to include the new parameter names.

### Update Method Chain

When tolerances are updated at runtime:
1. `algorithm.update(newton_atol=..., krylov_atol=...)`
2. `ODEImplicitStep.update()` delegates to `self.solver.update()`
3. `NewtonKrylov.update()` delegates krylov params to `self.linear_solver.update()`
4. Cache invalidation triggers rebuild with new tolerance arrays

### Buffer Registry (No Changes Expected)

Tolerance arrays are factory-scope constants captured in closures, not runtime buffers. No buffer registry changes needed.

## Edge Cases

1. **Scalar input (most common)**: Convert to array of length `n` filled with scalar value
2. **Single-element array**: Broadcast to array of length `n`
3. **Array of wrong length**: Raise `ValueError` with clear message
4. **Zero tolerance element**: Could cause division by zero; ensure minimum floor (e.g., 1e-16)
5. **Precision mismatch**: Converter ensures array dtype matches config precision
6. **n changes after init**: Cache invalidation via `update_compile_settings()` handles this

## Dependencies and Imports

New imports needed in solver modules:
```python
from numpy import asarray, full, isscalar, ndarray
from numpy.typing import ArrayLike
from attrs import Converter
from cubie._utils import float_array_validator
```

## Expected Data Structures

### NewtonKrylovConfig (updated)
```
newton_atol: ndarray  # shape (n,), dtype matches precision
newton_rtol: ndarray  # shape (n,), dtype matches precision
_newton_tolerance: float  # kept for backward compat/defaults
```

### LinearSolverConfig (updated)
```
krylov_atol: ndarray  # shape (n,), dtype matches precision  
krylov_rtol: ndarray  # shape (n,), dtype matches precision
_krylov_tolerance: float  # kept for backward compat/defaults
```

## Instrumented Test File Updates

The following instrumented test files must be updated to match source changes:
- `tests/integrators/algorithms/instrumented/matrix_free_solvers.py`

These files contain logging-enabled copies of the device functions and must reflect any changes to convergence check loops.

## ALL_*_PARAMETERS Sets

Update the following parameter sets to include new tolerance parameters:

1. **In newton_krylov.py or ode_implicitstep.py**:
   - Add `'newton_atol'`, `'newton_rtol'` to Newton params set

2. **In linear_solver.py or ode_implicitstep.py**:
   - Add `'krylov_atol'`, `'krylov_rtol'` to linear solver params set

This ensures the update() methods recognize and route these parameters correctly.
