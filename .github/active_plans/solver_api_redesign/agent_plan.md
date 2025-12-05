# Solver API Redesign: Agent Plan

## Overview

This plan describes the redesign of the Solver API to provide a single user-facing `solve()` method with automatic fast paths based on input detection. The goal is to make `solve()` the most basic call while keeping BatchSolverKernel methods internal.

---

## Component Descriptions

### 1. Input Type Classifier (New)

A new internal mechanism within `Solver` to classify input types at the start of `solve()`.

**Expected Behavior:**
- Examine `initial_values` and `parameters` arguments
- Return classification indicating which path to take:
  - `DICT_INPUT`: One or both inputs are dictionaries → use grid builder
  - `ARRAY_FAST_PATH`: Both are numpy arrays with correct shapes → skip grid building
  - `DEVICE_ARRAY`: Both have `__cuda_array_interface__` → minimal processing

**Classification Logic:**
```
If either input is a dict:
    return DICT_INPUT
    
If both inputs are np.ndarray:
    Check shape: (n_vars, n_runs) format expected
    If initial_values.shape[0] == n_states AND parameters.shape[0] == n_params:
        If initial_values.shape[1] == parameters.shape[1]:
            return ARRAY_FAST_PATH
    # If shapes don't match expectations, fall through to grid builder
    
If hasattr(input, '__cuda_array_interface__'):
    return DEVICE_ARRAY
    
Default: return DICT_INPUT (will attempt grid building)
```

### 2. Modified solve() Method

The `solve()` method in `Solver` class becomes the intelligent entry point.

**Expected Behavior:**
1. Call input type classifier on `initial_values` and `parameters`
2. Branch based on classification:
   - **DICT_INPUT**: Follow current path (grid_builder → kernel.run)
   - **ARRAY_FAST_PATH**: Validate arrays, skip grid_builder, call kernel.run directly
   - **DEVICE_ARRAY**: Minimal validation, call kernel.run with device arrays

**Current vs New Flow:**

Current (always):
```
solve() → grid_builder() → kernel.run()
```

New (conditional):
```
solve() → classify_inputs()
    → DICT_INPUT → grid_builder() → kernel.run()
    → ARRAY_FAST_PATH → validate_arrays() → kernel.run()
    → DEVICE_ARRAY → kernel.run()
```

### 3. build_grid() Method (New)

A new public method on `Solver` that exposes grid creation without solving.

**Expected Behavior:**
- Accept same dict inputs as `solve()` for `initial_values` and `parameters`
- Accept `grid_type` parameter ("combinatorial" or "verbatim")
- Return tuple: `(initial_values_array, parameters_array)`
- Arrays are in `(n_vars, n_runs)` format with system precision
- Does NOT execute the kernel

**Relationship to Existing Components:**
- Wraps `self.grid_builder()` call from current `solve()` implementation
- Uses same `BatchGridBuilder` instance already on `Solver`

### 4. Array Validation Helper (New Internal)

Internal method to validate pre-built arrays before fast path execution.

**Expected Behavior:**
- Verify arrays are contiguous (`np.ndarray` with correct flags)
- Verify dtype matches system precision (or cast if needed)
- Verify shape `(n_vars, n_runs)` with correct variable counts
- Return validated/cast arrays ready for kernel

**Edge Cases to Handle:**
- Arrays with wrong dtype → cast to `self.precision`
- Non-contiguous arrays → make contiguous copy
- Wrong number of variables → warning + adjustment (trim/extend)
- Mismatched run counts between inits and params → error

---

## Architectural Changes Required

### Changes to solver.py

1. **Add input classification logic** at start of `solve()`
2. **Add conditional branching** based on classification
3. **Add `build_grid()` public method**
4. **Add array validation helper** (private method)

### No Changes Required

- `BatchGridBuilder.py` - Already has fast paths for arrays
- `BatchSolverKernel.py` - `run()` already accepts arrays
- `solveresult.py` - No changes needed
- `SystemInterface.py` - No changes needed

---

## Integration Points

### solve() Integration with Grid Builder

Current integration (line 380-382 in solver.py):
```python
inits, params = self.grid_builder(
    states=initial_values, params=parameters, kind=grid_type
)
```

New integration:
```python
input_type = self._classify_inputs(initial_values, parameters)
if input_type == 'dict':
    inits, params = self.grid_builder(
        states=initial_values, params=parameters, kind=grid_type
    )
elif input_type == 'array':
    inits, params = self._validate_arrays(initial_values, parameters)
# ... continue with kernel.run()
```

### solve() Integration with Kernel

Current integration (line 396-406 in solver.py):
```python
self.kernel.run(
    inits=inits,
    params=params,
    driver_coefficients=self.driver_interpolator.coefficients,
    duration=duration,
    warmup=settling_time,
    t0=t0,
    blocksize=blocksize,
    stream=stream,
    chunk_axis=chunk_axis,
)
```

This integration point remains unchanged - all paths converge to calling `kernel.run()` with array inputs.

### build_grid() Integration

```python
def build_grid(
    self,
    initial_values: Union[np.ndarray, Dict[str, Union[float, np.ndarray]]],
    parameters: Union[np.ndarray, Dict[str, Union[float, np.ndarray]]],
    grid_type: str = "verbatim",
) -> Tuple[np.ndarray, np.ndarray]:
    """Build parameter and state grids for external use.
    
    Returns arrays that can be passed to solve() for fast-path execution.
    """
    return self.grid_builder(
        states=initial_values, params=parameters, kind=grid_type
    )
```

---

## Expected Interactions Between Components

### Flow 1: Dictionary Input (Existing Behavior)

```
User: solver.solve({"x": [1,2,3]}, {"p": [0.1, 0.2]})
  ↓
Solver._classify_inputs() → returns 'dict'
  ↓
Solver calls grid_builder() with dicts
  ↓
BatchGridBuilder.__call__() generates (n_vars, n_runs) arrays
  ↓
Solver calls kernel.run() with arrays
  ↓
BatchSolverKernel executes integration
  ↓
Returns SolveResult
```

### Flow 2: Pre-built Arrays (New Fast Path)

```
User: solver.solve(inits_array, params_array)  # both (n_vars, n_runs)
  ↓
Solver._classify_inputs() → returns 'array'
  ↓
Solver._validate_arrays() → validates/casts arrays
  ↓
Solver calls kernel.run() directly (grid_builder skipped!)
  ↓
BatchSolverKernel executes integration
  ↓
Returns SolveResult
```

### Flow 3: Using build_grid() Helper

```
User: inits, params = solver.build_grid({"x": [1,2,3]}, {"p": [0.1, 0.2]})
  ↓
Solver.build_grid() calls grid_builder() internally
  ↓
Returns arrays to user

# Later...
User: solver.solve(inits, params)  # Uses fast path since arrays provided
```

---

## Data Structures

### Input Type Classification

Return value from `_classify_inputs()`:
- Type: `str` (could be enum in future)
- Values: `'dict'`, `'array'`, `'device'`

### Validated Array Tuple

Return value from `_validate_arrays()`:
- Type: `Tuple[np.ndarray, np.ndarray]`
- Both arrays: dtype=`self.precision`, C-contiguous, shape=`(n_vars, n_runs)`

---

## Dependencies and Imports

### Existing (No Changes)

```python
from cubie.batchsolving.BatchGridBuilder import BatchGridBuilder
from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel
import numpy as np
```

### Potential New Imports

None required - all needed functionality exists in current imports.

---

## Edge Cases to Consider

### 1. Mixed Input Types
**Scenario**: User passes dict for initial_values, array for parameters
**Behavior**: Classify as `'dict'`, use grid_builder (handles mixed types)

### 2. Wrong Array Dimensions
**Scenario**: User passes 1D array instead of 2D
**Behavior**: Grid builder already handles this via `_sanitise_arraylike()`

### 3. Mismatched Run Counts
**Scenario**: `initial_values.shape[1] != parameters.shape[1]`
**Behavior**: 
- If `kind='verbatim'`: Error (current behavior)
- If `kind='combinatorial'`: Apply Cartesian product

### 4. Wrong Variable Counts
**Scenario**: `initial_values.shape[0] != n_states`
**Behavior**: Warning + trim/extend with defaults (existing behavior in BatchGridBuilder)

### 5. Non-Contiguous Arrays
**Scenario**: User passes sliced array that's not contiguous
**Behavior**: Make contiguous copy before passing to kernel

### 6. Wrong Dtype
**Scenario**: User passes float32 array when system is float64
**Behavior**: Cast to system precision

### 7. Empty Inputs
**Scenario**: User passes empty dict or empty array
**Behavior**: Use defaults, single-run execution

### 8. Driver Updates with Fast Path
**Scenario**: User passes arrays but also provides `drivers` dict
**Behavior**: Driver interpolator still updates (orthogonal to input detection)

---

## Testing Considerations

Tests should verify:

1. **Fast path detection**: Arrays skip grid_builder
2. **Dict path preserved**: Existing dict behavior unchanged
3. **build_grid() returns correct arrays**: Shape, dtype, values
4. **Fast path arrays work**: solve() with build_grid() output succeeds
5. **Mixed inputs handled**: dict + array → grid_builder called
6. **Edge cases**: wrong shapes, dtypes, run counts

---

## Performance Expectations

### Fast Path Benefits
- Skip grid_builder when arrays already built
- Avoid redundant array construction
- Reduce Python overhead for repeated solves with same grid

### No Performance Regression
- Dict inputs follow existing path (unchanged)
- Minimal overhead from type checking (O(1) isinstance checks)
