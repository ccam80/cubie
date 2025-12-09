# Technical Implementation Plan: NaN Error Trajectories

## Component Overview

This feature requires modifications to three key components in the batch solving pipeline:

1. **solve_ivp()** - User-facing convenience function
2. **Solver.solve()** - High-level solver orchestration
3. **SolveResult.from_solver()** - Result packaging and processing
4. **SolveResult** - Result container attrs class

## Detailed Component Specifications

### 1. solve_ivp() Function

**Location:** `src/cubie/batchsolving/solver.py`

**Current Signature:**
```python
def solve_ivp(
    system: BaseODE,
    y0: Union[np.ndarray, Dict[str, np.ndarray]],
    parameters: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
    drivers: Optional[Dict[str, object]] = None,
    dt_save: Optional[float] = None,
    method: str = "euler",
    duration: float = 1.0,
    settling_time: float = 0.0,
    t0: float = 0.0,
    grid_type: str = "combinatorial",
    time_logging_level: Optional[str] = 'default',
    **kwargs: Any,
) -> SolveResult:
```

**Required Changes:**
- Add parameter `nan_error_trajectories: bool = True` after `time_logging_level`
- Forward this parameter to `Solver.solve()` via kwargs

**Expected Behavior:**
- Parameter is passed through kwargs to Solver.solve()
- Default value of True enables safe behavior for typical users
- Docstring updated to explain this parameter

**Integration Points:**
- Currently creates Solver and calls solver.solve() with kwargs
- New parameter flows naturally through existing kwargs mechanism

### 2. Solver.solve() Method

**Location:** `src/cubie/batchsolving/solver.py`

**Current Signature:**
```python
def solve(
    self,
    initial_values: Union[np.ndarray, Dict[str, Union[float, np.ndarray]]],
    parameters: Union[np.ndarray, Dict[str, Union[float, np.ndarray]]],
    drivers: Optional[Dict[str, Any]] = None,
    duration: float = 1.0,
    settling_time: float = 0.0,
    t0: float = 0.0,
    blocksize: int = 256,
    stream: Any = None,
    chunk_axis: str = "run",
    grid_type: str = "verbatim",
    results_type: str = "full",
    **kwargs: Any,
) -> SolveResult:
```

**Required Changes:**
- Add parameter `nan_error_trajectories: bool = True` after `results_type`
- Forward to `SolveResult.from_solver(solver, results_type, nan_error_trajectories)`

**Expected Behavior:**
- Accepts nan_error_trajectories from solve_ivp or direct calls
- Passes it to SolveResult.from_solver() along with results_type
- No other logic changes needed in this method

**Integration Points:**
- Currently calls `SolveResult.from_solver(self, results_type=results_type)`
- Needs to pass additional parameter: `nan_error_trajectories=nan_error_trajectories`

### 3. SolveResult.from_solver() Class Method

**Location:** `src/cubie/batchsolving/solveresult.py`

**Current Signature:**
```python
@classmethod
def from_solver(
    cls,
    solver: Union["Solver", "BatchSolverKernel"],
    results_type: str = "full",
) -> Union["SolveResult", dict[str, Any]]:
```

**Required Changes:**
- Add parameter `nan_error_trajectories: bool = True` after `results_type`
- Add logic to retrieve status_codes from solver
- Add logic to process arrays based on status_codes when enabled
- Include status_codes in returned SolveResult

**Expected Behavior:**

#### When results_type == 'raw':
- Skip all processing (existing behavior)
- Return dict without status_codes
- Ignore nan_error_trajectories parameter entirely

#### When results_type != 'raw' and nan_error_trajectories == False:
- Include status_codes in SolveResult
- Do not modify array values
- Return results with original data

#### When results_type != 'raw' and nan_error_trajectories == True:
- Retrieve status_codes from solver: `solver.status_codes`
- Include status_codes in SolveResult
- For each run where status_codes[run_idx] != 0:
  - Set all values in time_domain_array for that run to NaN
  - Set all values in summaries_array for that run to NaN
- Return processed results

**Processing Algorithm:**

```python
# Pseudo-code for NaN processing
status_codes = solver.status_codes  # shape: (n_runs,)
run_index = stride_order.index("run")
ndim = len(stride_order)

# Find runs with errors
error_run_indices = np.where(status_codes != 0)[0]

for run_idx in error_run_indices:
    # Create slice for this run across all time points and variables
    run_slice = slice_variable_dimension(
        slice(run_idx, run_idx + 1, None),
        run_index,
        ndim
    )
    
    # Set entire trajectory to NaN
    time_domain_array[run_slice] = np.nan
    
    # Set summaries to NaN if present
    if summaries_array.size > 0:
        summaries_array[run_slice] = np.nan
```

**Data Structures:**
- status_codes: numpy array, shape (n_runs,), dtype int32
- time_domain_array: numpy array, shape depends on stride_order
- summaries_array: numpy array, shape depends on stride_order
- run_index: int, position of "run" in stride_order tuple

**Integration Points:**
- Access status_codes: `solver.status_codes` (property on Solver/BatchSolverKernel)
- Access stride_order: `solver.state_stride_order` (property returns tuple)
- Use existing helper: `slice_variable_dimension()` from `cubie._utils`
- Process arrays before calling result type conversions (as_numpy, as_pandas, etc.)

**Edge Cases:**
- Empty summaries_array (size 0): Skip summaries processing
- All runs successful (no nonzero status codes): No modifications needed
- Single run (n_runs=1): Still process correctly via slice
- Different stride orders: Run index may be 0, 1, or 2 - handled by index lookup

### 4. SolveResult Class

**Location:** `src/cubie/batchsolving/solveresult.py`

**Current Definition:**
```python
@attrs.define
class SolveResult:
    time_domain_array: Optional[NDArray] = ...
    summaries_array: Optional[NDArray] = ...
    time: Optional[NDArray] = ...
    iteration_counters: Optional[NDArray] = ...
    time_domain_legend: Optional[dict[int, str]] = ...
    summaries_legend: Optional[dict[int, str]] = ...
    solve_settings: Optional[SolveSpec] = ...
    _singlevar_summary_legend: Optional[dict[int, str]] = ...
    _active_outputs: Optional[ActiveOutputs] = ...
    _stride_order: Union[tuple[str, ...], list[str]] = ...
```

**Required Changes:**
- Add new attribute: `status_codes: Optional[NDArray]`
- Place after `iteration_counters` attribute
- Include in `from_solver()` instantiation
- No property wrapper needed (direct array access is appropriate)

**Expected Behavior:**
- status_codes stored as numpy array when results_type != 'raw'
- Defaults to None for backward compatibility
- Shape: (n_runs,), dtype: int32
- Values: 0 for success, nonzero for various error conditions

**Integration Points:**
- Populated in `from_solver()` when not using 'raw' results type
- Available for inspection via `result.status_codes`
- Not modified by NaN processing (remains original values)

## Architectural Considerations

### Memory and Performance

**Host-Side Processing:**
- All NaN assignment happens on host arrays after GPU→host transfer
- NumPy operations are efficient for setting slices to NaN
- No additional memory allocation needed (modifies existing arrays in-place)

**Typical Case Performance:**
- Most runs succeed (status_code == 0)
- Only failed runs need processing
- Cost: O(failed_runs × time_points × variables)
- Negligible compared to GPU kernel execution time

### Stride Order Independence

The implementation must work with any valid stride order:
- `("time", "variable", "run")` - default
- `("time", "run", "variable")`
- `("variable", "time", "run")`
- `("variable", "run", "time")`
- `("run", "time", "variable")`
- `("run", "variable", "time")`

**Implementation Strategy:**
1. Query `stride_order` from solver
2. Find run dimension: `run_idx = stride_order.index("run")`
3. Use `slice_variable_dimension()` to create proper slice
4. This helper automatically handles any dimension ordering

### Status Code Values

**Expected Status Codes:**
- 0: Success (no error)
- Nonzero: Various error conditions from integrator

**Sources:**
- Generated by integration loop in `SingleIntegratorRun`
- Passed through `BatchSolverKernel.integration_kernel()`
- Stored in `OutputArrays.status_codes`

**Error Conditions (nonzero codes):**
- Newton solver failures
- Maximum iterations exceeded
- Step size fell below minimum
- Divergence detected
- Other algorithm-specific failures

### Backward Compatibility

**Breaking Change (Intentional):**
- Default behavior changes to null out error trajectories
- This is a safety improvement, not a regression

**Migration Path:**
- Users wanting old behavior: set `nan_error_trajectories=False`
- Users already checking status codes: no change needed
- Users not checking status codes: now protected by default

**Compatibility Matrix:**
```
| User Code                  | Old Behavior      | New Behavior              |
|----------------------------|-------------------|---------------------------|
| No status checking         | Gets invalid data | Gets NaN (protected)      |
| Checks status codes        | Filters manually  | Gets NaN (automatic)      |
| nan_error_trajectories=False| N/A              | Gets invalid data (opted) |
```

### Dependencies

**Required Imports:**
- `numpy as np` (already imported)
- `slice_variable_dimension` from `cubie._utils` (already imported)

**No New Dependencies:**
- All required functionality exists in current codebase
- No new external packages needed

### Testing Strategy

**Test Coverage Areas:**

1. **Parameter Propagation:**
   - Test that nan_error_trajectories flows from solve_ivp → Solver.solve → from_solver
   - Test default value (True) is applied
   - Test explicit False value is respected

2. **Status Code Handling:**
   - Test status_codes included in non-raw results
   - Test status_codes excluded in raw results
   - Test status_codes values are unmodified

3. **NaN Processing:**
   - Test runs with status_code == 0 are unchanged
   - Test runs with status_code != 0 are all-NaN
   - Test partial failures (some runs succeed, some fail)
   - Test all runs fail
   - Test all runs succeed

4. **Array Dimensions:**
   - Test with different stride orders
   - Test with 3D arrays (time, variable, run)
   - Test with both time_domain_array and summaries_array
   - Test empty summaries_array (size 0)

5. **Results Types:**
   - Test 'raw' skips processing entirely
   - Test 'full' includes status_codes and processes arrays
   - Test 'numpy' dict includes status_codes
   - Test 'numpy_per_summary' dict includes status_codes
   - Test 'pandas' DataFrame handling of NaN values

6. **Edge Cases:**
   - Single run (n_runs=1) with error
   - No runs with errors (all status_codes == 0)
   - All runs with errors
   - Mixed error codes (different nonzero values)

## Implementation Order

1. Add `status_codes` attribute to SolveResult class
2. Modify `SolveResult.from_solver()` to retrieve and include status_codes
3. Add NaN-processing logic in `from_solver()` when enabled
4. Add `nan_error_trajectories` parameter to `Solver.solve()`
5. Add `nan_error_trajectories` parameter to `solve_ivp()`
6. Update docstrings for all modified functions
7. Write tests covering all scenarios

## Expected Interactions

### Between Components:

**solve_ivp → Solver.solve:**
- Passes nan_error_trajectories via kwargs
- No other changes to interaction

**Solver.solve → SolveResult.from_solver:**
- Passes nan_error_trajectories as explicit parameter
- Solver provides status_codes via property

**SolveResult.from_solver → Arrays:**
- Reads status_codes from solver
- Modifies time_domain_array and summaries_array in place
- Uses stride_order for proper indexing

### With Existing Systems:

**Memory Management:**
- No changes needed
- Processing happens on host after transfer

**Output Arrays:**
- status_codes already tracked and transferred
- No changes to OutputArrays class needed

**Result Type System:**
- 'raw' type bypasses all processing (existing behavior preserved)
- Other types get new processing step (enhancement)

**Solver Properties:**
- `solver.status_codes` already available
- `solver.state_stride_order` already available
- No new properties needed

## Validation Criteria

Implementation is successful when:

1. ✓ Tests pass for all result types with nan_error_trajectories=True
2. ✓ Tests pass for all result types with nan_error_trajectories=False
3. ✓ Status codes are included in non-raw results
4. ✓ Invalid trajectories contain all NaN when enabled
5. ✓ Valid trajectories remain unchanged
6. ✓ Implementation works with all stride orders
7. ✓ Docstrings clearly explain new parameter
8. ✓ No regression in existing tests (except expected behavior change)
