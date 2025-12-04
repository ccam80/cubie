# Solver API Stratification - Agent Plan

## Architectural Overview

This plan stratifies the solver API into three levels to allow users with different expertise and performance requirements to access appropriate entry points. The stratification follows a cascading pattern where higher-level methods call lower-level methods.

## Component Architecture

### Level 1: Solver.solve() - High-Level API

**Purpose**: Maximum convenience for novice users with labeled dictionary inputs

**Current Behavior (to retain)**:
- Accepts dictionaries with variable names as keys
- Calls `grid_builder()` to construct parameter grids
- Handles driver interpolation via `driver_interpolator`
- Processes `**kwargs` via `update()` method
- Calls `BatchSolverKernel.run()` with processed arrays
- Synchronizes memory and constructs `SolveResult`

**Expected Changes**:
- Extract array-handling logic into new `solve_arrays()` method
- `solve()` becomes a thin wrapper that:
  1. Resolves labels to indices
  2. Builds grids from dictionaries
  3. Sets up driver interpolation
  4. Calls `solve_arrays()` with numpy arrays

**Integration Points**:
- `BatchGridBuilder.__call__()` for grid construction
- `ArrayInterpolator` for driver setup
- `convert_output_labels()` for label resolution
- New `solve_arrays()` method

### Level 2: Solver.solve_arrays() - Mid-Level API

**Purpose**: Reduced overhead for users providing pre-built numpy arrays

**Expected Behavior**:
- Accepts pre-validated numpy arrays in (variable, run) format
- Skips label resolution and grid construction
- Handles settings updates via `update()` if kwargs provided
- Validates array shapes match system expectations
- Validates array precision matches system precision
- Handles memory allocation and host/device transfers via array managers
- Calls `BatchSolverKernel.execute()` for actual execution
- Synchronizes memory and constructs `SolveResult`

**Input Requirements**:
- `initial_values`: numpy array shape (n_states, n_runs), dtype matching system precision
- `parameters`: numpy array shape (n_params, n_runs), dtype matching system precision  
- `driver_coefficients`: Optional numpy array from `ArrayInterpolator.coefficients`
- `duration`, `warmup`, `t0`: float time parameters
- `blocksize`, `stream`: CUDA execution parameters
- `chunk_axis`: Memory chunking strategy

**Validation Scope**:
- Array shape compatibility with system sizes
- Array dtype matches system precision
- Contiguous memory layout
- No label resolution or content validation

**Integration Points**:
- `InputArrays.update()` for input array management
- `OutputArrays.update()` for output array setup
- `BatchSolverKernel.execute()` for kernel execution
- `MemoryManager.sync_stream()` for synchronization
- `SolveResult.from_solver()` for result construction

### Level 3: BatchSolverKernel.execute() - Low-Level API

**Purpose**: Minimal overhead for advanced users with pre-allocated device arrays

**Expected Behavior**:
- Accepts pre-allocated device arrays
- Skips all array management (no allocation, no transfers)
- Performs chunk calculation and parameter setup
- Launches CUDA kernel directly
- Writes results to provided output arrays
- Returns status code array (or void with in-place writes)

**Input Requirements**:
- `device_initial_values`: Device array (n_states, n_runs)
- `device_parameters`: Device array (n_params, n_runs)
- `device_driver_coefficients`: Device array (n_segments, n_drivers, order+1)
- `device_state_output`: Device array for state trajectories
- `device_observables_output`: Device array for observable trajectories
- `device_state_summaries_output`: Device array for state summaries
- `device_observable_summaries_output`: Device array for observable summaries
- `device_iteration_counters`: Device array for iteration counters
- `device_status_codes`: Device array for per-run status codes
- `duration`, `warmup`, `t0`: float64 time parameters
- `blocksize`: CUDA block size
- `stream`: CUDA stream for execution

**Validation Scope**:
- Minimal or no validation (trusts caller)
- Optional shape validation via debug flag

**Integration Points**:
- `limit_blocksize()` for block size adjustment
- `chunk_run()` for chunk parameter calculation
- `self.kernel[]` for CUDA kernel launch
- Profiling hooks if enabled

### Refactored BatchSolverKernel.run() 

**Current Behavior (to modify)**:
- Sets time parameters
- Updates input/output arrays via managers
- Refreshes compile settings
- Allocates memory
- Calculates chunking
- Limits blocksize
- Executes kernel loop
- Handles profiling

**Expected Changes**:
- `run()` becomes a thin wrapper that:
  1. Validates and converts input arrays
  2. Manages input/output array lifecycle
  3. Handles memory allocation via managers
  4. Calls `execute()` with device arrays
  5. Handles device→host transfers

## Data Structures

### Device Array Container (for Level 3)

Level 3 users need a container to hold pre-allocated device arrays. The existing `InputArrayContainer` and `OutputArrayContainer` structures can be reused, but Level 3 callers will pass device arrays directly rather than through managers.

**Approach**: Level 3 (`execute()`) accepts individual device arrays as parameters rather than a container, keeping the signature explicit and avoiding new data structure requirements.

### Input Validation Helper

A helper function for Level 2 validation:

```python
def validate_solver_arrays(
    initial_values: np.ndarray,
    parameters: np.ndarray,
    system_sizes: SystemSizes,
    precision: PrecisionDType,
) -> None:
    """Validate array shapes and types for solve_arrays().
    
    Raises ValueError with descriptive message on validation failure.
    """
```

## Integration Architecture

### Method Call Flow

```
Solver.solve(dicts, **kwargs)
    ├─ convert_output_labels()
    ├─ grid_builder(dicts) → arrays
    ├─ driver_interpolator.update_from_dict()
    └─ Solver.solve_arrays(arrays, **kwargs)
           ├─ update(kwargs) if kwargs
           ├─ validate_solver_arrays()
           └─ BatchSolverKernel.run(arrays)
                  ├─ input_arrays.update()
                  ├─ output_arrays.update()
                  ├─ memory_manager.allocate_queue()
                  └─ BatchSolverKernel.execute(device_arrays)
                         ├─ chunk_run()
                         ├─ limit_blocksize()
                         └─ kernel[grid, block](...)
```

### Memory Flow

```
Level 1 (solve):      Host dicts → numpy arrays → 
Level 2 (solve_arrays):                numpy arrays → managed device arrays →
Level 3 (execute):                                    device arrays → kernel → device arrays
Level 2 (solve_arrays):                ← managed host arrays ←
Level 1 (solve):      ← SolveResult
```

## Edge Cases

1. **Single Run Batches**: Arrays with shape (..., 1) must work at all levels
2. **Empty Driver Coefficients**: None handling consistent across levels  
3. **Precision Mismatch**: Level 2 should reject; Level 3 may produce undefined behavior
4. **Memory Exhaustion**: Chunking handled at Level 2+; Level 3 caller's responsibility
5. **Invalid Block Size**: `limit_blocksize()` called at Level 3

## Dependencies

### Required Imports (for new methods)

**Solver.solve_arrays()**:
- numpy as np
- numpy.typing.NDArray
- cubie.batchsolving.solveresult.SolveResult

**BatchSolverKernel.execute()**:
- numba.cuda
- numpy as np
- cubie.cuda_simsafe (for DeviceNDArrayBase type checking)

### Affected Modules

1. `src/cubie/batchsolving/solver.py` - Add `solve_arrays()` method
2. `src/cubie/batchsolving/BatchSolverKernel.py` - Add `execute()` method, refactor `run()`
3. `src/cubie/batchsolving/__init__.py` - Export new methods if needed

## Expected Interactions

### Between Solver and BatchSolverKernel

- `Solver.solve_arrays()` calls `self.kernel.run()` (unchanged)
- `BatchSolverKernel.run()` calls `self.execute()` (new)
- Array managers continue to work through `run()`
- Direct `execute()` calls bypass managers entirely

### With Memory Manager

- Level 1 & 2: Memory manager handles allocation and chunking
- Level 3: Caller must have already allocated; chunking parameters still calculated

### With Array Managers

- Level 1 & 2: `InputArrays` and `OutputArrays` manage host/device lifecycle
- Level 3: Array managers not used; caller provides device arrays directly

## Testing Considerations

### Test Categories

1. **Level 1 Tests**: Existing `test_solver.py` tests continue to pass
2. **Level 2 Tests**: New tests with numpy array inputs
3. **Level 3 Tests**: New tests with device array inputs (marked `nocudasim`)
4. **Cross-Level Tests**: Verify same results from all levels with equivalent inputs

### Test Fixtures

- Reuse existing system fixtures from `tests/conftest.py`
- Add fixtures for pre-built arrays matching system sizes
- Add fixtures for pre-allocated device arrays

## Documentation Requirements

### Docstrings

Each new method requires comprehensive numpydoc-style docstrings explaining:
- Purpose and use case
- What the method handles vs. what caller must handle
- Parameter requirements and validation behavior
- Return type and error conditions

### Usage Examples

Documentation should include examples showing:
- When to use each level
- How to transition from Level 1 to Level 2
- How to transition from Level 2 to Level 3
- Common patterns for tight loop usage with Level 3

## Backward Compatibility

- `Solver.solve()` signature unchanged
- `BatchSolverKernel.run()` signature unchanged  
- Existing tests pass without modification
- New methods are additive, not replacements
