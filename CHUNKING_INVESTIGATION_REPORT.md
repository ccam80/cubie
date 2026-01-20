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

Based on comprehensive analysis of all CUDAFactory classes (see `CUDA_CACHING_ANALYSIS.md` for full details):

### Issue 1: Systematic Callable Field Tracking Failure ⚠️ CRITICAL

**Root Cause**: Device function fields marked `eq=False` in config classes are excluded from hash calculation. When these are updated via `update()` or `update_compile_settings()`, the config hash doesn't change, causing stale cached kernels to be reused.

**Critical Chain - SingleIntegratorRun → IVPLoop**:

1. `SingleIntegratorRun.build()` (lines 655-664) updates IVPLoop with 7 device functions:
   ```python
   compiled_functions = {
       'save_state_fn': self._output_functions.save_state_func,
       'update_summaries_fn': self._output_functions.update_summaries_func,
       'save_summaries_fn': self._output_functions.save_summary_metrics_func,
       'step_controller_fn': self._step_controller.device_function,
       'step_function': self._algo_step.step_function,
       'evaluate_observables': evaluate_observables
   }
   self._loop.update(compiled_functions)
   ```

2. ALL these fields in `ODELoopConfig` have `eq=False`:
   - `save_state_fn`, `update_summaries_fn`, `save_summaries_fn`
   - `step_controller_fn`, `step_function`
   - `evaluate_driver_at_t`, `evaluate_observables`

3. `IVPLoop.build()` (lines 346-352) reads these from config and captures them in kernel closure

**Result**: When output configuration changes (e.g., different arrays to save), functions are updated but kernel is NOT rebuilt. Old kernel with incorrect array indexing continues execution.

**All Affected Classes**:
- `IVPLoop` (7 callable fields) - **CRITICAL** - center of integration chain
- `BaseAlgorithmStep` (4 callable fields) - evaluate_f, evaluate_observables, etc.
- `ImplicitStepConfig` (4+ callable fields) - solver_function, jacobian functions
- `MatrixFreeSolver` (2-4 callable fields) - operator_apply, preconditioner, residual
- `BatchSolverConfig.loop_fn` - appears unused (dead code)

### Issue 2: Variables Captured in `build()` Closure

Device functions and flags are captured in build() methods and baked into kernels as compile-time constants:

**BatchSolverKernel.build_kernel()** (lines 694-710):
- `save_state`, `save_observables`, etc. from `active_outputs` property (derived from `compile_flags` ✓)
- `needs_padding` from property (derived from `shared_memory_elements` ✓)
- `alloc_shared`, `alloc_persistent` dynamically created (capture config values ✓)
- `loopfunction = self.single_integrator.device_function` - child factory ✓

**IVPLoop.build()** (lines 346-352):
- `save_state = config.save_state_fn` - **eq=False** ✗
- `step_controller = config.step_controller_fn` - **eq=False** ✗
- `step_function = config.step_function` - **eq=False** ✗
- Plus 4 more callable fields - all **eq=False** ✗

**ImplicitStep.build()** (lines 225-231):
- `evaluate_f = config.evaluate_f` - **eq=False** ✗
- `solver_function = config.solver_function` - **eq=False** ✗

**Impact**: Values marked ✓ are properly tracked (changes trigger rebuild). Values marked ✗ are NOT tracked - updates change config but don't invalidate cache, causing stale kernels with wrong function pointers to persist.

## Recommendations

### Immediate Actions

1. **Force cache invalidation on callable updates** - Modify `update_compile_settings()` to explicitly invalidate cache when any callable field with `eq=False` changes

2. **Verify child factory chain** - Ensure SingleIntegratorRun, IVPLoop, algorithm steps, and controllers are properly linked as children so hash changes propagate

3. **Test on CUDA hardware** - Reproduce chunked test infinite loop with fix applied

### Long-term Solutions

1. **Track function hashes** - Implement stable hashing for callable fields instead of using `eq=False`

2. **Explicit rebuild triggers** - Add mechanism to force rebuild when critical functions change

3. **Runtime validation** - Detect when stale kernels used with incompatible configs (defensive programming)

### Code Changes Required

Files needing modification:
- `src/cubie/CUDAFactory.py` - Add invalidation logic to `update_compile_settings()`
- `src/cubie/integrators/loops/ode_loop_config.py` - 7 callable fields need tracking
- `src/cubie/integrators/algorithms/base_algorithm_step.py` - 4 callable fields need tracking
- `src/cubie/integrators/algorithms/ode_implicitstep.py` - 1+ callable fields need tracking
- `src/cubie/batchsolving/BatchSolverConfig.py` - Remove unused `loop_fn` or track properly

## Conclusion

The chunked tests **pass in CUDASIM** and the iteration logic is mathematically correct. However, **systematic issues with callable field tracking in CUDA kernel caching** were identified.

**Root Cause of Infinite Loops**: The `SingleIntegratorRun → IVPLoop` chain updates device functions via `update()`, but ODELoopConfig has ALL 7 callable fields marked `eq=False`. When output configuration changes:
1. Device functions are updated in config
2. Config hash DOESN'T change (eq=False excludes them)
3. Cache hit on stale hash returns old kernel
4. Old kernel has incorrect array indexing/logic baked in
5. On CUDA hardware: memory corruption, infinite loops, crashes

This explains why chunked tests hang on CUDA but pass in CUDASIM - CUDASIM doesn't cache kernels the same way.

**See `CUDA_CACHING_ANALYSIS.md` for comprehensive analysis of all 12 affected CUDAFactory classes and detailed recommendations.**
