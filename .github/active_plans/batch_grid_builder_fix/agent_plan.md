# BatchGridBuilder Bug Fix: Agent Implementation Plan

## Overview

This document provides the technical specification for fixing the `BatchGridBuilder` class. The goal is to simplify the interface by removing the `request` parameter and fixing edge-case bugs with single-parameter sweeps and 1D input handling.

## Component Changes

### 1. BatchGridBuilder Class (`src/cubie/batchsolving/BatchGridBuilder.py`)

#### 1.1 Remove `request` Parameter from `__call__()`

**Current signature:**
```python
def __call__(
    self,
    request: Optional[Dict[str, Union[float, ArrayLike, np.ndarray]]] = None,
    params: Optional[Union[Dict, ArrayLike]] = None,
    states: Optional[Union[Dict, ArrayLike]] = None,
    kind: str = "combinatorial",
) -> tuple[np.ndarray, np.ndarray]:
```

**New signature:**
```python
def __call__(
    self,
    params: Optional[Union[Dict, ArrayLike]] = None,
    states: Optional[Union[Dict, ArrayLike]] = None,
    kind: str = "combinatorial",
) -> tuple[np.ndarray, np.ndarray]:
```

**Behavior changes:**
- Remove all `request` handling logic
- Simplify the branching to only handle `params` and `states`
- Remove the `grid_arrays()` method (or keep internal if useful)
- Update error messages to not reference `request`

#### 1.2 Fix Single-Dict Single-Parameter Bug

**Problem**: When `params={'p1': np.linspace(0,1,100)}` is passed, the code path doesn't correctly generate default states.

**Solution**: In the branch where only `params` (dict) is provided:
1. Generate params array using `generate_array(params_dict, self.parameters, kind)`
2. Create states array by tiling defaults: `np.tile(self.states.values_array[:, np.newaxis], (1, n_runs))`

#### 1.3 Fix 1D Array Handling

**Problem**: 1D arrays with single varied parameter raise errors.

**Root cause**: `_sanitise_arraylike()` converts 1D to column vector but subsequent logic may fail.

**Solution**: Ensure `_sanitise_arraylike()` always returns 2D arrays in (variable, run) format:
- 1D array of length N → reshape to (N, 1) for single run
- Verify downstream code handles single-run case correctly

#### 1.4 Fix `combine_grids()` Edge Cases

**Problem**: When one grid has only 1 run (from scalar or single value), combining fails.

**Solution**: In `combine_grids()`:
- If either grid has shape (n_vars, 1), broadcast it to match the other grid's run count
- For combinatorial: single-run grid should tile, multi-run grid should repeat
- For verbatim: single-run grid broadcasts to match other grid

### 2. verbatim_grid Function Bug

**Problem**: `verbatim_grid()` may fail when arrays have different lengths.

**Current behavior**: Only checks after building grid.

**Solution**: Add early validation before grid construction.

### 3. extend_grid_to_array Bug

**Problem**: Function doesn't handle empty indices correctly.

**Current code:**
```python
if grid.ndim == 1:
    array = default_values[:, np.newaxis]
```

**Issue**: When `indices` is empty but grid isn't 1D, behavior is undefined.

**Solution**: Handle empty indices case explicitly:
```python
if indices.size == 0:
    # No sweep, return defaults tiled to match grid run count
    n_runs = grid.shape[1] if grid.ndim > 1 else 1
    return np.tile(default_values[:, np.newaxis], (1, n_runs))
```

## Integration Points

### Solver.build_grid() (`src/cubie/batchsolving/solver.py`)

**Current code** (duplicate method definition issue):
```python
def build_grid(
    self,
    initial_values: ...,
    parameters: ...,
    grid_type: str = "verbatim",
) -> Tuple[np.ndarray, np.ndarray]:
    return self.grid_builder(
        states=initial_values, params=parameters, kind=grid_type
    )
```

**Note**: There are two identical `build_grid()` method definitions in solver.py (lines 529-568 and 570-609). This is a bug that should be fixed by removing the duplicate.

**Changes needed**: None for signature, but remove the duplicate definition.

### Solver.solve() Internal Call

**Current code:**
```python
inits, params = self.grid_builder(
    states=initial_values, params=parameters, kind=grid_type
)
```

**Changes needed**: None - already uses `states` and `params` keywords.

## Test Requirements

### New Test Cases Needed

1. **test_single_param_dict_sweep**
   - Input: `params={'p1': np.linspace(0,1,100)}`
   - Expected: 100 runs, p1 varies, all else defaults

2. **test_single_state_dict_single_run**
   - Input: `states={'x': 0.5}`
   - Expected: 1 run with x=0.5

3. **test_states_dict_params_sweep**
   - Input: `states={'x': 0.2}, params={'p1': np.linspace(0,3,300)}`
   - Expected: 300 runs, x=0.2 for all, p1 varies

4. **test_combinatorial_states_params**
   - Input: `states={'y': [0.1, 0.2]}, params={'p1': np.linspace(0,1,100)}, kind='combinatorial'`
   - Expected: 200 runs

5. **test_1d_param_array**
   - Input: `params=np.array([1.0, 2.0, 3.0])` (length matches n_params)
   - Expected: 1 run with custom parameters

6. **test_1d_state_array_partial**
   - Input: `states=np.array([1.0, 2.0])` (length < n_states)
   - Expected: Warning, defaults fill missing values

7. **test_empty_inputs_returns_defaults**
   - Input: `params=None, states=None`
   - Expected: Single run with all defaults

8. **test_verbatim_single_run_broadcast**
   - Input: `states={'x': 0.5}, params={'p1': [1,2,3]}, kind='verbatim'`
   - Expected: 3 runs, x=0.5 broadcasts

### Existing Tests to Verify

- `test_call_with_request` → Remove or update (request removed)
- `test_call_input_types` → Update valid_combos list
- `test_docstring_examples` → Update examples for new API

## Data Structures

### Input Processing Flow

```
User Input                    Internal Format           Output
-----------                   ---------------           ------
dict{'p1': [1,2,3]}    →     (indices, grid)      →    (n_params, 3)
np.array([1,2,3])      →     reshape to (3,1)     →    (n_vars, 1) 
np.array([[1,2],[3,4]])→     pass through         →    (2, 2)
scalar 0.5             →     np.atleast_1d()      →    (1,) → (n_vars, 1)
```

### Grid Combination Logic

For `kind='combinatorial'`:
```
states: (n_states, S_runs)  params: (n_params, P_runs)
Result: states expanded to (n_states, S_runs * P_runs)
        params tiled to (n_params, S_runs * P_runs)
```

For `kind='verbatim'`:
```
states: (n_states, S_runs)  params: (n_params, P_runs)
If S_runs == P_runs: return as-is
If S_runs == 1: broadcast states to P_runs
If P_runs == 1: broadcast params to S_runs
Else: ValueError
```

## Edge Cases

### Empty Dictionary
```python
grid_builder(params={}, states={})  # Returns single run with all defaults
```

### All Defaults for One Category
```python
grid_builder(params={'p1': [1,2,3]}, states={})  # 3 runs, states default
grid_builder(params={}, states={'x': [1,2,3]})  # 3 runs, params default
```

### Scalar Values in Dict
```python
grid_builder(params={'p1': 0.5})  # Single value, 1 run with p1=0.5
```

## Dependencies

- `numpy` for array operations
- `SystemValues` for accessing default values and name resolution
- `SystemInterface` for wrapping system access

## Backward Compatibility Notes

This is a **breaking change**:
- `request` parameter removed from `__call__()`
- `grid_arrays()` method behavior may change or be removed
- Users must update calls using `request` to use `params` and `states` separately

Per project guidelines: "No backwards compatibility enforcement - breaking changes expected during development"
