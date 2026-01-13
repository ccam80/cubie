# Memory Refactor Cleanup - Agent Plan

## Context

The user has refactored the memory manager to centralize chunking responsibility and remove unused allocation modes. This plan addresses the cleanup tasks required to complete the refactor.

## Task 1: Fix `ensure_nonzero_size` Bug

### Location
`src/cubie/_utils.py`, function `ensure_nonzero_size`

### Current Behavior
```python
if any(v == 0 for v in value):
    return tuple(1 for v in value)  # Replaces ENTIRE tuple
```

### Expected Behavior
Replace only zero elements while preserving non-zero elements:
```python
# (2, 0, 2) → (2, 1, 2), not (1, 1, 1)
```

### Why This Matters
- When `save_variables` selects a subset of states, observable dimensions may be 0
- Shape `(num_saves, 0, num_runs)` becomes `(1, 1, 1)` instead of `(num_saves, 1, num_runs)`
- This causes shape mismatch between host and device arrays during copy

---

## Task 2: Remove Tests for Deleted Functions

### File
`tests/memory/test_memmgmt.py`

### Functions to Remove Tests For

These methods were removed from `MemoryManager`:

1. **`single_request`** - Replaced by `queue_request` + `allocate_queue`
2. **`get_chunks`** - Replaced by `get_chunk_parameters`
3. **`get_available_single`** - Removed (instance mode gone)
4. **`get_available_group`** - Now `get_available_memory`
5. **`chunk_arrays`** - Chunking now internal to `allocate_queue`
6. **`get_chunkable_request_size`** - Now `get_portioned_request_size`
7. **`get_total_request_size`** - Removed (not needed externally)

### Tests to Remove

1. `test_get_chunks` - Tests removed `get_chunks` method
2. `test_get_available_single` - Tests removed instance-based method
3. `test_get_available_group` - Method signature changed
4. `test_chunk_arrays` - Tests removed method
5. `test_chunk_arrays_skips_missing_axis` - Tests removed method
6. `test_single_request` - Tests removed method
7. `test_get_available_single_low_memory_warning` - Tests removed method
8. `test_get_available_group_low_memory_warning` - Tests removed method
9. `test_get_total_request_size` - Tests removed function
10. `test_get_chunkable_request_size_excludes_unchunkable` - Tests removed function
11. `test_get_chunkable_request_size_excludes_wrong_axis` - Tests removed function
12. `test_get_chunkable_request_size_returns_zero_all_unchunkable` - Tests removed function
13. `test_compute_chunked_shapes_uses_floor_division` - Tests old signature
14. `test_compute_chunked_shapes_preserves_unchunkable` - Tests old signature
15. `test_compute_chunked_shapes_empty_requests` - Tests old signature
16. `test_compute_chunked_shapes_single_chunk` - Tests old signature
17. `test_chunk_arrays_uses_floor_division` - Tests removed method
18. `test_chunk_arrays_min_one_element` - Tests removed method
19. `test_single_request_populates_chunked_shapes` - Tests removed method
20. `test_allocate_queue_populates_chunked_shapes` - Tests old signature (`limit_type`)
21. `test_chunk_calculation_5_runs_4_chunks` - Tests removed method/old signature
22. `test_all_arrays_unchunkable_produces_one_chunk` - Tests removed function
23. `test_final_chunk_has_correct_indices` - Tests removed method/old signature
24. `test_uneven_chunk_division_7_runs_3_chunks` - Tests removed method/old signature
25. `test_chunk_size_minimum_one_when_runs_less_than_chunks` - Tests removed method/old signature

### Test to Fix (Collection Error)

`test_allocate_queue_empty_queue` - Has parametrized `limit_type` which no longer exists
- Remove the `limit_type` and `chunk_axis` parameters from the parametrize decorator
- These are now set directly via `allocate_queue(instance, chunk_axis=...)` call

---

## Task 3: Update Tests with New Method Signatures

### `compute_chunked_shapes` Signature Change

Old:
```python
mgr.compute_chunked_shapes(requests, num_chunks=4, chunk_axis="run")
```

New:
```python
mgr.compute_chunked_shapes(requests, chunk_axis="run", chunk_size=12)
```

If any tests using this method should remain, update the call signature.

### `allocate_queue` Signature Change

Old:
```python
mgr.allocate_queue(instance, limit_type="group")
```

New:
```python
mgr.allocate_queue(instance, chunk_axis="run")
```

Remove `limit_type` parameter usage from all tests.

---

## Task 4: Verify Solver Test Passes

### Test to Verify
`tests/batchsolving/test_solver.py::test_solver_solve_with_save_variables`

After fixing `ensure_nonzero_size`, this test should pass. Run to confirm.

---

## Dependencies and Imports

### Removed Functions (do not import)
- `get_chunkable_request_size` - removed
- `get_total_request_size` - removed

### Available Functions
- `get_portioned_request_size` - returns (chunkable, unchunkable) bytes
- `is_request_chunkable` - checks if request can be chunked
- `replace_with_chunked_size` - modifies shape for chunking
- `compute_per_chunk_slice` - calculates slices for chunks
- `get_chunk_axis_length` - gets length of chunk axis

---

## Edge Cases to Consider

1. **Empty save_variables**: All state/observable dimensions could be 0
2. **No chunking needed**: `chunks=1` case should preserve original shapes
3. **Time-axis chunking**: Different from run-axis chunking in behavior

---

## Integration Points

1. **BatchOutputArrays.update_from_solver** - Uses `BatchOutputSizes.nonzero`
2. **BatchInputArrays.update** - Uses sizing from solver
3. **MemoryManager.allocate_queue** - Central allocation with chunking
4. **BaseArrayManager._on_allocation_complete** - Receives chunked shapes

---

## Validation

After implementation:
1. Run `pytest tests/memory/test_memmgmt.py -v` - all tests should pass
2. Run `pytest tests/batchsolving/test_solver.py::test_solver_solve_with_save_variables -v` - should pass
3. Run `pytest tests/batchsolving/ -v` - verify no regressions
