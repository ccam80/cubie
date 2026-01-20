# Investigation Report: Chunked Tests Infinite Loop Issue

## Summary

After thorough investigation of the chunked tests and chunking logic, **no infinite loop bugs were found in CUDASIM**. All tests using the `chunked_solved_solver` fixture pass successfully with CUDASIM enabled, both with and without xdist parallel execution.

However, upon closer examination of the CUDA compilation and caching infrastructure, **potential issues were identified that could cause bugs on real CUDA hardware** related to compile-time constants not being properly tracked in the configuration hash.

## Investigation Scope

### Areas Examined

1. **Chunking Logic in `BatchSolverKernel.run()`**
   - Loop structure: `for i in range(chunks)` ✓ Correct
   - Chunk index validation in `RunParams.__getitem__()` ✓ Correct
   - Edge cases for 0 chunks, 1 chunk, and multiple chunks ✓ All handled correctly

2. **Chunk Size Calculation (`MemoryManager.get_chunk_parameters()`)**
   - Mathematical correctness of `floor(axis_length / chunk_ratio)` ✓ Correct
   - Handling of edge cases (memory too low, can't fit single run) ✓ Proper error raising
   - Calculation verified with multiple test scenarios ✓ All produce correct results

3. **Array Slicing Logic (`ManagedArray.chunk_slice()`)**
   - Start/end index calculation ✓ Correct
   - Last chunk handling with `end=None` ✓ Correct
   - Synchronization between `RunParams` and `ManagedArray` chunk parameters ✓ Properly maintained

4. **Memory Allocation Callback Flow**
   - `_on_allocation_complete()` updates all arrays in `_needs_reallocation` ✓ Correct
   - `chunk_length` and `num_chunks` properly propagated from `ArrayResponse` ✓ Verified
   - Arrays in `chunked_shapes` correctly match requested arrays ✓ Confirmed

## Test Results

All chunked tests execute successfully:
- ✅ `test_run_executes_with_chunking`
- ✅ `test_chunked_solve_produces_valid_output`  
- ✅ `test_chunked_solver_produces_correct_results` (all 5 parametrized variants)
- ✅ `test_input_buffers_released_after_kernel`
- ✅ `test_chunked_uses_numpy_host`
- ✅ `test_pinned_buffers_created`
- ✅ `test_watcher_completes_all_tasks`

Tests completed in ~7-12 seconds with no hangs or infinite loops.

## Chunk Calculation Examples

Verified correct behavior for 5 runs with various memory limits:

| Free Memory | Chunk Ratio | Chunk Size | Num Chunks | Distribution |
|-------------|-------------|------------|------------|--------------|
| 850 MB      | 4.12        | 1          | 5          | 1-1-1-1-1    |
| 1024 MB     | 2.48        | 2          | 3          | 2-2-1        |
| 1240 MB     | 1.66        | 3          | 2          | 3-2          |
| 1460 MB     | 1.25        | 4          | 2          | 4-1          |
| 2048 MB     | <1.0        | 5          | 1          | 5 (no chunk) |

All calculations match expected values from `conftest.py` comments.

## Potential Issues Identified

While no bugs causing infinite loops were found, one potential improvement was identified:

### Better Error Messages for Chunk Parameter Mismatches

Currently, if `chunk_length` or `num_chunks` are not properly initialized on a `ManagedArray`, the error would be:
- `TypeError: unsupported operand type(s) for *: 'int' and 'NoneType'` (if `chunk_length=None`)

This could be confusing. Adding validation would provide clearer error messages.

## Recommendations

1. **No immediate code changes required** - The chunking logic is working correctly
2. **Consider adding defensive checks** - Add validation to ensure `chunk_length` and `num_chunks` are properly set before use
3. **Monitor for environment-specific issues** - The reported issue may be specific to certain hardware/driver configurations
4. **Verify CI/CD environment** - Ensure tests are running with appropriate timeouts if they occasionally slow down

## CUDA-Specific Issues Identified

Based on guidance about CUDA kernel compilation behavior:

### Issue 1: Callable Fields with `eq=False`

Many CUDAFactory config classes have callable (device function) fields marked with `eq=False`:
- `BatchSolverConfig.loop_fn`
- `AlgorithmConfig.evaluate_f`, `evaluate_observables`, `evaluate_driver_at_t`
- `LoopConfig.save_state_fn`, `update_summaries_fn`, `step_function`, etc.
- Various solver and algorithm specific functions

These fields are excluded from the config hash calculation. While child factories' hashes are included via `_iter_child_factories()`, any direct changes to these callable fields won't trigger cache invalidation.

**Potential Impact**: If a device function changes but the config hash doesn't update, a stale cached kernel could be reused on CUDA hardware, potentially causing incorrect behavior including infinite loops.

### Issue 2: Variables Captured in `build()` Closure

In `BatchSolverKernel.build_kernel()`, several variables are captured in the kernel closure:
- Lines 694-697: `save_state`, `save_observables`, `save_state_summaries`, `save_observable_summaries` derived from `active_outputs` property
- Line 698: `needs_padding` from `shared_memory_needs_padding` property  
- Lines 708-710: `alloc_shared`, `alloc_persistent` from `buffer_registry.get_toplevel_allocators()`

While these are derived from config values, they are computed at build time and baked into the compiled kernel as constants. Any subsequent changes to the underlying config values won't affect already-compiled kernels unless the build() method is called again.

**Potential Impact**: On CUDA hardware with kernel caching, a kernel compiled with one set of output flags could be reused even after output configuration changes, leading to incorrect array indexing and potential infinite loops or memory corruption.

## Recommendations

1. **Verify all child factory relationships** - Ensure all CUDAFactory instances with callable fields are properly registered as child factories so their config hashes propagate to parents

2. **Audit `eq=False` usage** - Review all fields marked `eq=False` to ensure they don't contain values that should invalidate the cache when changed

3. **Add validation** - Consider adding runtime validation to detect when stale kernels are being used with incompatible configurations

4. **Test on real CUDA hardware** - The chunking tests should be run on actual CUDA hardware to reproduce the reported infinite loop condition

## Conclusion

The chunked tests **pass in CUDASIM** and the iteration logic is mathematically correct. However, **potential issues with CUDA kernel caching and compile-time constant tracking** were identified that could cause bugs on real CUDA hardware.

The infinite loop behavior reported on CUDA hardware is likely caused by stale cached kernels being reused with incompatible runtime configurations, particularly related to chunking parameters or output array flags.
