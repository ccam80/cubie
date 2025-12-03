# Agent Plan: Run-Contiguous Array Memory Layout

## Overview

This plan transforms CuBIE's array memory layout so that the run dimension is always the rightmost (innermost) dimension, enabling CUDA memory coalescing. The change affects stride_order declarations, array indexing, and CUDA kernel signatures.

## Target Layout

| Array Type | Current stride_order | Target stride_order |
|------------|---------------------|---------------------|
| 3D output (state, observables, summaries) | `("time", "run", "variable")` | `("time", "variable", "run")` |
| 2D input (initial_values, parameters) | `("run", "variable")` | `("variable", "run")` |
| 3D driver_coefficients | `("time", "run", "variable")` | Keep as-is or update based on access pattern |

## Component Descriptions

### 1. MemoryManager Default Stride Order

**File**: `src/cubie/memory/mem_manager.py`

**Current Behavior**: The `_stride_order` field defaults to `("time", "variable", "run")` at line 300-302.

**Target Behavior**: Default remains `("time", "variable", "run")` - this is already the target layout. However, verify that all consumers interpret this correctly.

**Key Attributes**:
- `_stride_order`: tuple[str, str, str] defining the logical dimension order
- `get_strides()`: Calculates byte strides from the stride_order
- `create_host_array()`: Creates arrays with compatible strides

### 2. ArrayRequest Fallback Orders

**File**: `src/cubie/memory/array_requests.py`

**Current Behavior**: 
- 3D arrays default to `("time", "run", "variable")` at line 92
- 2D arrays default to `("variable", "run")` at line 94

**Target Behavior**:
- 3D arrays should default to `("time", "variable", "run")`
- 2D arrays should default to `("variable", "run")` - already correct

### 3. InputArrayContainer Stride Orders

**File**: `src/cubie/batchsolving/arrays/BatchInputArrays.py`

**Current Behavior**:
- `initial_values`: `stride_order=("run", "variable")` at line 29
- `parameters`: `stride_order=("run", "variable")` at line 36
- `driver_coefficients`: `stride_order=("time", "run", "variable")` at line 43

**Target Behavior**:
- `initial_values`: `stride_order=("variable", "run")`
- `parameters`: `stride_order=("variable", "run")`
- `driver_coefficients`: `stride_order=("time", "variable", "run")`

### 4. OutputArrayContainer Stride Orders

**File**: `src/cubie/batchsolving/arrays/BatchOutputArrays.py`

**Current Behavior**:
- `state`: `stride_order=("time", "run", "variable")` at line 32
- `observables`: `stride_order=("time", "run", "variable")` at line 39
- `state_summaries`: `stride_order=("time", "run", "variable")` at line 46
- `observable_summaries`: `stride_order=("time", "run", "variable")` at line 53
- `status_codes`: `stride_order=("run",)` at line 60
- `iteration_counters`: `stride_order=("run", "time", "variable")` at line 68

**Target Behavior**:
- `state`: `stride_order=("time", "variable", "run")`
- `observables`: `stride_order=("time", "variable", "run")`
- `state_summaries`: `stride_order=("time", "variable", "run")`
- `observable_summaries`: `stride_order=("time", "variable", "run")`
- `status_codes`: `stride_order=("run",)` - unchanged (1D)
- `iteration_counters`: `stride_order=("time", "variable", "run")`

### 5. Output Sizing Classes

**File**: `src/cubie/outputhandling/output_sizes.py`

**Current Behavior**:
- `SingleRunOutputSizes.stride_order`: `("time", "variable")` at line 329
- `BatchInputSizes.stride_order`: `("run", "variable")` at line 405
- `BatchOutputSizes.stride_order`: `("time", "run", "variable")` at line 483

**Target Behavior**:
- `SingleRunOutputSizes.stride_order`: `("time", "variable")` - unchanged (2D, no run)
- `BatchInputSizes.stride_order`: `("variable", "run")`
- `BatchOutputSizes.stride_order`: `("time", "variable", "run")`

Additionally, update docstrings that describe array shapes from "(time × run × variable)" to "(time × variable × run)".

### 6. BatchSolverKernel Array Indexing

**File**: `src/cubie/batchsolving/BatchSolverKernel.py`

**Current Indexing** (lines 587-600):
```python
rx_inits = inits[run_index, :]
rx_params = params[run_index, :]
rx_state = state_output[:, run_index * save_state, :]
rx_observables = observables_output[:, run_index * save_observables, :]
rx_state_summaries = state_summaries_output[:, run_index * save_state_summaries, :]
rx_observables_summaries = observables_summaries_output[:, run_index * save_observable_summaries, :]
rx_iteration_counters = iteration_counters_output[run_index, :, :]
```

**Target Indexing**:
```python
rx_inits = inits[:, run_index]
rx_params = params[:, run_index]
rx_state = state_output[:, :, run_index * save_state]
rx_observables = observables_output[:, :, run_index * save_observables]
rx_state_summaries = state_summaries_output[:, :, run_index * save_state_summaries]
rx_observables_summaries = observables_summaries_output[:, :, run_index * save_observable_summaries]
rx_iteration_counters = iteration_counters_output[:, :, run_index]
```

**Kernel Signature Update** (line 497-513):
Current signature uses generic slices. Update to use contiguity annotations:
```python
@cuda.jit(
    (
        precision[:, ::1],    # inits: (variable, run)
        precision[:, ::1],    # params: (variable, run)
        precision[:, :, ::1], # d_coefficients: (time, variable, run)
        precision[:, :, :],   # state_output (non-contiguous view possible)
        precision[:, :, :],   # observables_output
        precision[:, :, :],   # state_summaries_output
        precision[:, :, :],   # observables_summaries_output
        int32[:, :, :],       # iteration_counters_output
        int32[::1],           # status_codes_output
        float64,
        float64,
        float64,
        int32,
    ),
    **compile_kwargs,
)
```

### 7. IVPLoop Array Indexing

**File**: `src/cubie/integrators/loops/ode_loop.py`

**Current Behavior** (lines 887-903):
The loop function signature shows output arrays as 2D `(time, variable)` slices passed from the kernel. These are already single-run views so no run index is present.

The signature annotation at lines 887-903:
```python
precision[:, :],  # state_output (2D slice for one run)
precision[:, :],  # observables_output
precision[:, :],  # state_summaries_output
precision[:, :],  # observable_summaries_output
precision[:,::1], # iteration_counters_output
```

The 2D slices `state_output[save_idx, :]` access time as first dimension and variable as second, which is consistent with the new layout.

**No changes needed** in ode_loop.py as it operates on single-run 2D slices.

### 8. SolveResult Stride Order

**File**: `src/cubie/batchsolving/solveresult.py`

**Current Behavior**:
- `_stride_order` default: `("time", "run", "variable")` at line 187
- `concat_results()` uses `stride_order = ["time", "run", "variable"]` at line 470

**Target Behavior**:
- `_stride_order` default: `("time", "variable", "run")`
- Update `concat_results()` to use `["time", "variable", "run"]`

Also update:
- `run_property()` at line 307: index lookup for "run" dimension
- Documentation describing array shapes

## Integration Points

### Memory Manager ↔ Array Containers
- Array containers declare their stride_order
- Memory manager respects stride_order when allocating
- `get_strides()` uses stride_order to compute byte strides

### BatchSolverKernel ↔ Loop Function
- Kernel slices 3D arrays by run index before passing to loop
- Loop receives 2D `(time, variable)` views
- No run indexing needed inside loop

### Host ↔ Device Transfers
- `BaseArrayManager._convert_to_device_strides()` ensures host arrays match device layout
- Memory manager's `to_device()` and `from_device()` operate on matching strides

## Edge Cases

1. **Chunking**: The chunk axis is typically "run". With run as the rightmost dimension, chunking creates contiguous memory slices.

2. **Single-run case**: When num_runs=1, arrays are still 3D but with size 1 in the run dimension.

3. **Empty outputs**: When an output type is disabled, the array size may be 1×1×1.

4. **iteration_counters**: Has shape `(run, time, 4)` currently, needs `(time, 4, run)` or similar.

## Test Files Requiring Updates

### Primary Test Files

1. **tests/memory/test_memmgmt.py**
   - Line 117: defaults dict `stride_order`
   - Lines 379, 387, 526, 534: stride_order in test fixtures
   - Lines 396-403: `test_set_global_stride_ordering`
   - Lines 708, 714: stride_order assertions

2. **tests/memory/test_array_requests.py**
   - Lines 38, 48, 53: stride_order default assertions

3. **tests/memory/conftest.py**
   - Line 19: stride_order in test settings

4. **tests/batchsolving/arrays/test_basearraymanager.py**
   - Lines 47, 55, 63, 71, 84, 92: stride_order declarations
   - Line 124: `_stride_order` settings dict
   - Lines 172, 177, 206, 211, etc.: stride_order in fixtures

5. **tests/batchsolving/arrays/test_batchinputarrays.py**
   - Line 96: stride_order assertion

6. **tests/batchsolving/arrays/test_batchoutputarrays.py**
   - Various stride_order declarations

7. **tests/outputhandling/test_output_sizes.py**
   - Lines 562-565: stride_order default test

8. **tests/all_in_one.py**
   - Lines 263-293: stride order comments and index mapping
   - Lines 2390-2391: array indexing patterns
   - Lines 2443, 2547-2548: dimension index comments

9. **tests/tests/all_in_one_nblocks.py**
   - Lines 42-72: stride order comments
   - Lines 1868-1869: array indexing
   - Lines 1922, 2032-2033: dimension comments

10. **tests/_utils.py**
    - Lines 819-823: state_output indexing patterns

## Validation Steps

1. After modifying each source file, run related unit tests
2. After all source changes, run integration tests
3. Verify array shapes in test assertions match new layout
4. Check kernel signatures compile without errors
5. Run a full solve and verify output array shapes

## Dependencies

The changes should be applied in this order:
1. Memory manager defaults (affects all downstream)
2. Array request fallbacks
3. Input/Output array container declarations
4. Output sizing classes
5. BatchSolverKernel indexing and signatures
6. SolveResult updates
7. Test file updates
