# Test Results Summary

## Overview
- **Tests Run**: 1306
- **Passed**: 1290
- **Failed**: 16
- **Errors**: 0
- **Skipped**: 0

## Command Executed
```bash
NUMBA_ENABLE_CUDASIM=1 pytest -m "not nocudasim and not specific_algos" -v --tb=short
```

## Failures

All 16 failures share a common root cause: **incompatible array shapes** during device-to-host copy operations. The error occurs when copying arrays from device to host where the destination array size does not match the source array size.

### tests/batchsolving/arrays/test_batchoutputarrays.py

#### TestBufferPoolAndWatcherIntegration::test_finalise_uses_buffer_pool_when_chunked
**Type**: ValueError
**Message**: incompatible shape: (51, 3, 5) vs. (51, 3, 2)

#### TestBufferPoolAndWatcherIntegration::test_finalise_submits_to_watcher_when_chunked
**Type**: ValueError
**Message**: incompatible shape: (51, 3, 5) vs. (51, 3, 2)

#### TestBufferPoolAndWatcherIntegration::test_wait_pending_blocks_until_complete
**Type**: ValueError
**Message**: incompatible shape: (51, 3, 5) vs. (51, 3, 2)

#### TestBufferPoolAndWatcherIntegration::test_reset_clears_buffer_pool_and_watcher
**Type**: ValueError
**Message**: incompatible shape: (51, 3, 5) vs. (51, 3, 2)

### tests/batchsolving/test_chunked_solver.py

#### TestChunkedSolverExecution::test_chunked_solve_produces_valid_output[run]
**Type**: ValueError
**Message**: incompatible shape: (5,) vs. (2,)

#### TestChunkedSolverExecution::test_chunked_solve_produces_valid_output[time]
**Type**: ValueError
**Message**: incompatible shape: (1, 1, 1) vs. (0, 1, 1)

#### TestSyncStreamRemoval::test_solver_completes_without_sync_stream
**Type**: ValueError
**Message**: incompatible shape: (5,) vs. (1,)

#### TestSyncStreamRemoval::test_chunked_solver_produces_correct_results
**Type**: ValueError
**Message**: incompatible shape: (3,) vs. (1,)

#### TestSyncStreamRemoval::test_input_buffers_released_after_kernel
**Type**: ValueError
**Message**: incompatible shape: (5,) vs. (1,)

### tests/batchsolving/test_pinned_memory_refactor.py

#### TestTwoTierMemoryStrategy::test_chunked_uses_numpy_host
**Type**: ValueError
**Message**: incompatible shape: (5,) vs. (2,)

#### TestTwoTierMemoryStrategy::test_total_pinned_memory_bounded
**Type**: ValueError
**Message**: incompatible shape: (5,) vs. (1,)

#### TestEventBasedSynchronization::test_wait_pending_blocks_correctly
**Type**: ValueError
**Message**: incompatible shape: (5,) vs. (1,)

#### TestWatcherThreadBehavior::test_watcher_starts_on_first_chunk
**Type**: ValueError
**Message**: incompatible shape: (5,) vs. (1,)

#### TestWatcherThreadBehavior::test_watcher_completes_all_tasks
**Type**: ValueError
**Message**: incompatible shape: (5,) vs. (1,)

#### TestRegressionChunkedPath::test_large_batch_produces_correct_results
**Type**: ValueError
**Message**: incompatible shape: (5,) vs. (1,)

#### TestRegressionChunkedPath::test_input_arrays_buffer_pool_used_in_chunked_mode
**Type**: ValueError
**Message**: incompatible shape: (3, 6) vs. (3, 2)

## Root Cause Analysis

The failures all occur during the `finalise()` or `initialise()` methods of BatchOutputArrays and BatchInputArrays respectively. The call stack shows:

1. `finalise(host_indices)` → `from_device(from_, to_)`
2. `from_device()` → `memory_manager.from_device()`
3. `from_array.copy_to_host(to_arrays[i], stream=stream)`
4. **Error**: Array shape mismatch

The issue is that when using the buffer pool in chunked mode:
- The **source array** (device array or pool buffer) has a shape based on the **total batch size** or the **pool buffer size**
- The **destination array** (host slice) has a shape based on the **current chunk size**

For example, error `(51, 3, 5) vs. (51, 3, 2)` shows:
- 51 = time samples
- 3 = state variables  
- 5 = total systems (pool buffer dimension)
- 2 = systems in current chunk

The buffer pool returns full-sized buffers but the code tries to copy to chunk-sized destination arrays.

## Recommendations

1. **Fix buffer slicing in `from_device` and `to_device` methods**: When using buffer pool, slice the buffer to match the chunk size before copying:
   ```python
   # Instead of copying full buffer to chunk-sized destination
   from_array.copy_to_host(to_arrays[i], stream=stream)
   
   # Slice the buffer to match destination size
   chunk_size = to_arrays[i].shape[-1]  # or appropriate dimension
   from_array[..., :chunk_size].copy_to_host(to_arrays[i], stream=stream)
   ```

2. **Review ChunkBufferPool.get_buffer()**: Consider if buffers should be pre-sliced when acquired based on the current chunk requirements.

3. **Check host_indices handling**: Ensure `_get_host_slice()` returns arrays sized for the current chunk, and that the buffer selection logic accounts for this.

4. **Test fixture review**: Verify that test fixtures correctly set up chunked scenarios where chunk_size < total_systems.
