# Remove Chunk Axis Feature - Agent Implementation Plan

## Overview

This plan details the complete removal of the `chunk_axis` parameter and all associated infrastructure from CuBIE. The feature allowed selecting whether to chunk arrays along the "run", "time", or "variable" axis when memory is limited. Due to implementation complexity (summary arrays having different lengths than save arrays) and minimal use case, we are removing this feature and hardcoding chunking to occur only along the "run" axis.

## Architectural Context

### Current Architecture

The chunk_axis feature flows through the following components:

1. **User API Layer** (`solver.py`, `solve_ivp()`)
   - Accepts `chunk_axis` parameter (default: "run")
   - Validates and forwards to kernel

2. **Kernel Layer** (`BatchSolverKernel.py`)
   - Stores in `FullRunParams` and `ChunkParams` data structures
   - Has `_chunk_axis` attribute and `chunk_axis` property
   - Passes to memory manager and array managers

3. **Memory Management** (`mem_manager.py`)
   - `allocate_queue()` accepts chunk_axis parameter
   - Uses axis to calculate chunking parameters
   - Returns axis in `ArrayResponse`

4. **Array Management** (`BaseArrayManager.py`, `BatchInputArrays.py`, `BatchOutputArrays.py`)
   - Stores `_chunk_axis` attribute
   - Uses for computing per-chunk slices
   - Validates consistency between input/output managers

5. **Data Structures** (`array_requests.py`)
   - `ArrayRequest` has `stride_order` that includes axis labels
   - `ArrayResponse` contains `chunk_axis` field

### Target Architecture

After removal:
- All `chunk_axis` parameters removed from public and internal APIs
- Chunking logic hardcoded to use `"run"` axis
- Data structures simplified to remove axis tracking
- Properties and validators for chunk_axis removed

## Component-by-Component Changes

### 1. src/cubie/batchsolving/solver.py

**Expected Behavior:**
- `Solver.solve()` no longer accepts `chunk_axis` parameter
- `solve_ivp()` no longer accepts `chunk_axis` parameter  
- `Solver.chunk_axis` property removed (was returning `self.kernel.chunk_axis`)

**Key Integration Points:**
- Calls `self.kernel.run()` - remove chunk_axis argument
- Property access removed from public API

**Implementation Notes:**
- Update docstrings to remove chunk_axis parameter documentation
- Remove chunk_axis from parameter list (line ~351)
- Remove chunk_axis from docstring "Parameters" section (line ~384-386)
- Remove chunk_axis property (lines ~892-894)
- Remove chunk_axis=chunk_axis in kernel.run() call (line ~450)

### 2. src/cubie/batchsolving/BatchSolverKernel.py

**Expected Behavior:**
- `BatchSolverKernel.run()` no longer accepts chunk_axis parameter
- `FullRunParams` dataclass no longer has chunk_axis field
- `ChunkParams` dataclass no longer has _chunk_axis field or time-axis logic
- `chunk_axis` property removed from BatchSolverKernel
- Hardcoded "run" axis used throughout

**Key Integration Points:**
- Creates `FullRunParams` without chunk_axis
- Calls `memory_manager.allocate_queue()` without chunk_axis
- Sets `_chunk_axis` on array managers (remove these assignments)

**Data Structures:**
- `FullRunParams` (line ~64-85): Remove `chunk_axis: str` field
- `ChunkParams` (line ~88-125): Remove `_chunk_axis: str` field
- `ChunkParams.__getitem__()` (line ~162-200): Remove time-axis branching logic (lines 187-196)
- `ChunkParams.from_allocation_response()` (line ~128-160): Remove chunk_axis parameter passing

**Implementation Notes:**
- Remove chunk_axis parameter from `run()` method (line ~801)
- Remove chunk_axis from docstring
- Remove time-axis chunking logic in ChunkParams.__getitem__
- Simplify __getitem__ to only handle run-axis case
- Remove chunk_axis property (lines ~1153-1177)
- Remove chunk_axis validation logic
- Remove assignments like `self.input_arrays._chunk_axis = chunk_axis`
- Remove chunk_axis from FullRunParams construction

### 3. src/cubie/batchsolving/arrays/BaseArrayManager.py

**Expected Behavior:**
- `BaseArrayManager` no longer has `_chunk_axis` attribute
- No validation of chunk_axis values
- Chunking logic assumes "run" axis

**Key Integration Points:**
- Receives `ArrayResponse` from memory manager (no longer has chunk_axis field)
- Uses hardcoded "run" axis for all chunking calculations

**Implementation Notes:**
- Remove `_chunk_axis` attribute (line ~261-263)
- Remove `_chunk_axis` from class docstring (lines ~226-228)
- Remove assignment `self._chunk_axis = response.chunk_axis` (line ~368)
- Update any methods that reference self._chunk_axis to use "run" literal

### 4. src/cubie/memory/mem_manager.py

**Expected Behavior:**
- `allocate_queue()` no longer accepts chunk_axis parameter - hardcoded to "run"
- `get_chunk_parameters()` no longer accepts chunk_axis parameter
- Helper functions like `get_chunk_axis_length()`, `is_request_chunkable()`, etc. hardcoded to "run"

**Key Integration Points:**
- Called by BatchSolverKernel without chunk_axis
- Creates ArrayResponse without chunk_axis field
- Uses "run" axis for all chunking calculations

**Implementation Notes:**
- Remove chunk_axis parameter from `allocate_queue()` (line ~1159)
- Hardcode `chunk_axis = "run"` at start of method
- Remove chunk_axis parameter from `get_chunk_parameters()` 
- Remove chunk_axis parameter from `get_portioned_request_size()`
- Remove chunk_axis parameter from `get_chunk_axis_length()`
- Remove chunk_axis parameter from `is_request_chunkable()`
- Remove chunk_axis parameter from `compute_per_chunk_slice()`
- Remove chunk_axis from ArrayResponse construction
- Update all internal calls to these functions

### 5. src/cubie/memory/array_requests.py

**Expected Behavior:**
- `ArrayResponse` no longer has `chunk_axis` field
- Default chunking behavior hardcoded to "run" axis

**Implementation Notes:**
- Remove `chunk_axis` field from ArrayResponse (lines ~141-143)
- Update validator list if chunk_axis had validation
- Remove chunk_axis from class docstring

**Data Structure:**
```python
# Before:
@attrs.define
class ArrayResponse:
    arr: dict[str, DeviceNDArrayBase]
    chunks: int
    chunk_axis: str = "run"  # REMOVE THIS
    ...

# After:
@attrs.define  
class ArrayResponse:
    arr: dict[str, DeviceNDArrayBase]
    chunks: int
    # chunk_axis removed, implicitly "run"
    ...
```

### 6. Tests - General Strategy

**Expected Behavior:**
- All `chunk_axis` parametrization removed
- Tests validate run-axis chunking only
- No tests for time-axis or variable-axis chunking
- All chunk_axis parameters removed from test calls

**Common Changes Across All Test Files:**
- Remove `@pytest.mark.parametrize("chunk_axis", ["run", "time"])` decorators
- Remove `chunk_axis` parameter from function signatures
- Remove `chunk_axis=chunk_axis` from all solve() calls
- Remove `chunk_axis="run"` or `chunk_axis="time"` from direct calls
- Remove chunk_axis fixture usage

### 7. tests/batchsolving/arrays/conftest.py

**Implementation Notes:**
- Remove `chunk_axis` fixture (lines ~60-63)
- Remove chunk_axis parameter from `chunked_solved_solver` (line ~68, ~103)
- Remove chunk_axis parameter from `unchunked_solved_solver` (line ~114, ~133)
- Update docstrings that mention chunk_axis

### 8. tests/batchsolving/arrays/test_chunking.py

**Implementation Notes:**
- Remove `TestChunkAxisProperty` class entirely (tests property that will be removed)
- Remove all `@pytest.mark.parametrize("chunk_axis", ...)` decorators
- Remove chunk_axis parameters from test function signatures
- Remove assertions about chunk_axis values
- Keep tests that validate run-axis chunking behavior
- Remove tests specific to time-axis chunking

**Specific Removals:**
- `test_chunk_axis_property_returns_consistent_value` - entire test
- `test_chunk_axis_property_raises_on_inconsistency` - entire test  
- Remove chunk_axis parametrization from remaining tests
- Update test names if they specifically mention "chunk_axis"

### 9. tests/batchsolving/arrays/test_basearraymanager.py

**Implementation Notes:**
- Remove `chunk_axis="run"` from all `allocate_queue()` calls
- Remove assignments like `test_manager_with_sizing._chunk_axis = chunk_axis`
- Remove assertions like `assert manager._chunk_axis == "run"`
- Update test logic that validates chunk_axis behavior

### 10. tests/batchsolving/arrays/test_batchinputarrays.py

**Implementation Notes:**
- Remove chunk_axis from `allocate_queue()` calls
- Search for chunk_axis references and remove

### 11. tests/batchsolving/arrays/test_batchoutputarrays.py

**Implementation Notes:**
- Remove chunk_axis from all `allocate_queue()` calls (many occurrences)
- Remove `_chunk_axis` attribute assignments
- Remove assertions validating chunk_axis values

### 12. tests/batchsolving/test_solver.py

**Implementation Notes:**
- Remove chunk_axis from solve() calls
- Remove any tests that specifically validate chunk_axis property

### 13. tests/batchsolving/test_config_plumbing.py

**Implementation Notes:**
- Remove chunk_axis references if present
- Verify no configuration tests rely on chunk_axis

### 14. tests/memory/test_memmgmt.py

**Implementation Notes:**
- Remove chunk_axis parameter from `allocate_queue()` calls
- Remove chunk_axis from test docstrings
- Update tests like `test_compute_per_chunk_slice_handles_arrays_without_chunk_axis`
  - This test verifies behavior when chunk_axis not in stride_order
  - After changes, all chunking is on "run" axis, so update test accordingly

### 15. tests/memory/test_array_requests.py

**Implementation Notes:**
- Remove chunk_axis from ArrayResponse instantiation
- Update assertions to not check chunk_axis field

## Edge Cases and Special Considerations

### 1. ChunkParams Time-Axis Logic

The `ChunkParams.__getitem__()` method has complex logic for time-axis chunking:
- Calculates per-chunk duration
- Adjusts t0 for subsequent chunks
- Sets warmup to 0 for chunks after the first

**Action:** Remove all time-axis branching (lines ~187-196), keep only run-axis logic

### 2. Chunk Axis Validation in BatchSolverKernel

The `chunk_axis` property validates consistency between input and output arrays:

```python
input_axis = self.input_arrays._chunk_axis
output_axis = self.output_arrays._chunk_axis
if input_axis != output_axis:
    raise ValueError(...)
```

**Action:** Remove entire property - no validation needed with hardcoded value

### 3. ArrayResponse Default Values

ArrayResponse has default values for chunk_axis. Removing the field means:
- Consumers must not reference `response.chunk_axis`
- Hardcode "run" where needed

### 4. Stride Order Validation

Functions like `is_request_chunkable()` check if chunk_axis is in stride_order:

```python
if chunk_axis not in request.stride_order:
    return False
```

**Action:** Replace with `if "run" not in request.stride_order`

### 5. Documentation in Docstrings

Many docstrings reference chunk_axis in:
- Parameter lists
- Return value descriptions
- Notes sections
- Examples

**Action:** Remove all such references, update to describe run-axis-only behavior

## Dependencies Between Changes

### Phase 1: Data Structures (can be done in parallel)
- Remove chunk_axis from `ArrayResponse`
- Remove chunk_axis from `FullRunParams`
- Remove chunk_axis from `ChunkParams`

### Phase 2: Core Logic (depends on Phase 1)
- Update `mem_manager.py` functions
- Update `BaseArrayManager`
- Update `BatchSolverKernel`

### Phase 3: Public API (depends on Phase 2)
- Update `Solver` class
- Update `solve_ivp()` function

### Phase 4: Tests (depends on all above)
- Update all test files
- Remove parametrization
- Update fixtures

## Validation Strategy

### 1. Static Checks
```bash
# Verify no chunk_axis references remain in source
grep -r "chunk_axis" src/cubie/

# Should return empty or only in comments explaining removal
```

### 2. Test Execution
```bash
# Run CUDASIM tests (no GPU required)
pytest -m "not nocudasim and not cupy"

# Run full test suite with CUDA
pytest

# Run specific chunking tests
pytest tests/batchsolving/arrays/test_chunking.py -v
```

### 3. Integration Validation
- Verify chunked solver still works with memory constraints
- Verify unchunked solver works normally
- Verify all array transfers work correctly

### 4. Property Verification
- Verify `Solver.chunks` property still works (returns number of chunks)
- Verify chunking calculations still produce correct chunk counts
- Verify per-chunk parameters correctly calculated for run-axis

## Implementation Order

1. **Start with data structures** - least dependent
2. **Update memory manager** - central coordinator
3. **Update array managers** - depend on ArrayResponse
4. **Update kernel** - depends on managers and data structures
5. **Update solver** - depends on kernel
6. **Update tests** - depends on all source changes
7. **Final verification** - grep and test execution

## Testing Before Review

Before submitting to reviewer:

1. Run full CUDASIM test suite: `pytest -m "not nocudasim and not cupy"`
2. Run targeted chunking tests: `pytest tests/batchsolving/arrays/test_chunking.py`
3. Run memory management tests: `pytest tests/memory/`
4. Verify grep: `grep -r "chunk_axis" src/` returns empty
5. Check docstrings manually for lingering references

---
*This plan provides comprehensive removal strategy for detailed_implementer agent to execute*
