# Chunked Transfer Shape Fix - Agent Plan

## Problem Statement

The `needs_chunked_transfer` property in `ManagedArray` returns `False` when it should return `True` for chunked array scenarios. This causes:

1. **44 test failures**: Buffer pool paths are skipped, leading to shape mismatches during copy operations
2. **7 test failures**: Tests asserting `needs_chunked_transfer=True` fail
3. **2 test failures**: Memory limits too low for test scenarios

The root cause is that `ManagedArray.array.setter` updates the `shape` attribute to match the actual array dimensions. When a chunked device array is attached, `shape` becomes equal to `chunked_shape`, making `needs_chunked_transfer` return `False`.

---

## Architecture Context

### ManagedArray Structure (Current)
- `shape`: Tuple representing array dimensions, auto-updated by `array.setter`
- `chunked_shape`: Tuple representing per-chunk dimensions (set after allocation)
- `_array`: The actual numpy/CUDA array reference
- `needs_chunked_transfer`: Property returning `shape != chunked_shape`

### Data Flow in Chunked Allocation
1. `MemoryManager.allocate_queue()` allocates device arrays with chunked dimensions
2. `ArrayResponse` contains `chunked_shapes` dict mapping array names to per-chunk shapes
3. `BaseArrayManager._on_allocation_complete()` is called with response
4. For each array needing reallocation:
   - `attach(label, allocated_array)` is called
   - `chunked_shape` and `chunked_slice_fn` are set on ManagedArray
5. **BUG**: `attach()` calls `array.setter` which overwrites `shape` with chunked dimensions

### Key Files

| File | Role |
|------|------|
| `src/cubie/batchsolving/arrays/BaseArrayManager.py` | Contains `ManagedArray`, `ArrayContainer`, `BaseArrayManager` |
| `src/cubie/batchsolving/arrays/BatchInputArrays.py` | `InputArrays.initialise()` uses buffer pool for chunked H2D |
| `src/cubie/batchsolving/arrays/BatchOutputArrays.py` | `OutputArrays.finalise()` uses buffer pool for chunked D2H |
| `tests/batchsolving/arrays/test_basearraymanager.py` | Tests for ManagedArray and chunked shape propagation |
| `tests/batchsolving/arrays/test_batchinputarrays.py` | Tests for InputArrays buffer pool integration |
| `tests/batchsolving/arrays/test_batchoutputarrays.py` | Tests for OutputArrays buffer pool integration |

---

## Component Behavior Changes

### Task Group A: Fix ManagedArray.array.setter Shape Update

**File**: `src/cubie/batchsolving/arrays/BaseArrayManager.py`

**Current Behavior**:
```python
@array.setter
def array(self, value):
    self._array = value
    if value is not None:
        self.shape = tuple(value.shape)  # BUG: Overwrites logical shape
```

**Expected Behavior**:
The setter should NOT update `shape` when an array is attached for chunked operation. The `shape` attribute should represent the logical (full) dimensions, not the actual allocation dimensions.

**Approach Options**:

1. **Simple fix**: Remove the shape update from the setter entirely. The `shape` is already set during initialization and should remain the logical shape.

2. **Conditional fix**: Only update shape if `chunked_shape` is None (indicating non-chunked operation).

3. **Preserve behavior for host arrays**: Only skip shape update for device arrays.

**Recommended**: Option 1 (remove shape update) is cleanest. The `shape` attribute is initialized correctly and should represent logical dimensions. Consumers who need actual array dimensions can access `array.shape` directly.

**Integration Points**:
- No other code should depend on `array.setter` updating `shape`
- Tests may need adjustment if they relied on auto-sync behavior

---

### Task Group B: Verify Test Fixtures Create Proper Chunked Scenarios

**Files**: 
- `tests/batchsolving/arrays/test_batchinputarrays.py`
- `tests/batchsolving/arrays/test_batchoutputarrays.py`
- `tests/batchsolving/arrays/test_basearraymanager.py`

**Issue**: Some tests set `chunked_shape` on device arrays AFTER the array is attached, which is correct. However, the fixture setup may still result in equal shapes if:
- `num_runs` is too small (e.g., 5 runs with 3 chunks = 1 run per chunk)
- Host array shape ends up matching device chunked shape

**Verification Tasks**:
1. Ensure test fixtures use sufficient `num_runs` so `chunk_size < num_runs`
2. Ensure host arrays have full shape, device arrays have chunked shape in allocation response
3. Ensure `shape` attribute is NOT modified after `chunked_shape` is set

**Tests to Check**:
- `test_chunked_shape_propagates_through_allocation`
- `test_convert_host_to_numpy_uses_needs_chunked_transfer`
- `test_finalise_uses_needs_chunked_transfer`
- `test_initialise_uses_buffer_pool_when_chunked`
- `test_release_buffers_returns_to_pool`
- `test_reset_clears_buffer_pool_and_active_buffers`
- `test_input_arrays_buffer_pool_used_in_chunked_mode`

---

### Task Group C: Increase Memory Limits for Memory Tests

**Issue**: Two tests fail with "Can't fit a single run in GPU VRAM":
- `test_watcher_completes_all_tasks` - Needs 71860 bytes, has 65536
- `test_large_batch_produces_correct_results` - Needs 86232 bytes, has 65536

**Fix**: Increase `MockMemoryManager` total memory from 64KB (65536 bytes) to 256KB (262144 bytes) or larger.

**Files to Update**:

1. `tests/batchsolving/test_pinned_memory_refactor.py` (lines 20-25)
2. `tests/batchsolving/test_chunked_solver.py` (lines ~17-22)

**Current Code** (both files):
```python
class MockMemoryManager(MemoryManager):
    """Mock memory manager for testing with controlled memory info."""

    def get_memory_info(self):
        return int(65536), int(131072)  # 64kb free, 128kb total
```

**Change**: Increase values to at least 256KB:
```python
class MockMemoryManager(MemoryManager):
    """Mock memory manager for testing with controlled memory info."""

    def get_memory_info(self):
        return int(262144), int(524288)  # 256kb free, 512kb total
```

---

## Detailed Implementation Guidance

### A1: Remove shape update from ManagedArray.array.setter

**Location**: `src/cubie/batchsolving/arrays/BaseArrayManager.py`, lines 108-116

**Change**: Remove or comment out line 116 (`self.shape = tuple(value.shape)`)

**Rationale**: The `shape` attribute represents the logical dimensions expected by the user. It is set during ManagedArray initialization and should not change when the physical array (which may be chunked) is attached.

**Edge Cases**:
- Host arrays: `shape` is already set correctly during initialization, no change needed
- Non-chunked device arrays: `shape` equals actual array shape, still works correctly
- Chunked device arrays: `shape` (full) != `chunked_shape` (per-chunk), works correctly

### A2: Verify _on_allocation_complete preserves logical shape

**Location**: `src/cubie/batchsolving/arrays/BaseArrayManager.py`, lines 360-384

The current implementation:
1. Calls `attach(label, array)` which sets the array
2. Then sets `chunked_shape` from response

After the A1 fix, `attach()` will no longer overwrite `shape`, so `shape` will remain at logical dimensions. The `chunked_shape` will be set correctly from the response.

---

## Dependency Analysis

### Components Using needs_chunked_transfer

1. **BatchOutputArrays._convert_host_to_numpy()** (line ~248-258)
   - Uses `device_slot.needs_chunked_transfer` to decide if host array should be converted to regular numpy

2. **BatchOutputArrays.finalise()** (line ~450)
   - Uses `device_slot.needs_chunked_transfer` to decide if buffer pool should be used

3. **BatchInputArrays.initialise()** (line ~315)
   - Uses `host_obj.needs_chunked_transfer` to decide if buffer pool should be used

### No Changes Needed to These Components
After the ManagedArray fix, `needs_chunked_transfer` will return correct values, and these components will work as expected.

---

## Expected Test Results After Fix

| Test Category | Before | After |
|--------------|--------|-------|
| `test_chunked_shape_propagates_through_allocation` | FAIL | PASS |
| `test_convert_host_to_numpy_uses_needs_chunked_transfer` | FAIL | PASS |
| `test_finalise_uses_needs_chunked_transfer` | FAIL | PASS |
| `test_initialise_uses_buffer_pool_when_chunked` | FAIL | PASS |
| `test_release_buffers_returns_to_pool` | FAIL | PASS |
| `test_reset_clears_buffer_pool_and_active_buffers` | FAIL | PASS |
| All copy_to_host shape mismatch tests | FAIL | PASS |

---

## Risk Assessment

### Low Risk
- The change is localized to `ManagedArray.array.setter`
- No production code depends on shape auto-sync behavior
- Tests explicitly verify the new behavior

### Verification Steps
1. Run `test_basearraymanager.py` tests for ManagedArray behavior
2. Run `test_batchinputarrays.py` and `test_batchoutputarrays.py` for integration
3. Run `test_chunked_solver.py` for end-to-end chunked operation
4. Run full `tests/batchsolving/` suite to verify no regressions
