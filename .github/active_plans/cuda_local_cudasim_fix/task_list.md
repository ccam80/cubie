# Implementation Task List
# Feature: CUDA Local Array CUDASIM Fix
# Plan Reference: .github/active_plans/cuda_local_cudasim_fix/agent_plan.md

## Task Group 1: Core Infrastructure - Add CUDASIM-safe local_array pattern to cuda_simsafe.py
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/cuda_simsafe.py (entire file - see lines 1-349 for existing patterns)

**Input Validation Required**:
- None (no new parameters being added)

**Tasks**:
1. **Export CUDA_SIMULATION constant in __all__**
   - File: src/cubie/cuda_simsafe.py
   - Action: Verify (already exported at line 317)
   - Details: The `CUDA_SIMULATION` constant is already exported in `__all__`. No changes needed here, but taskmaster should verify this is correctly exported for use by other modules.

**Outcomes**:
- Files Modified: 
  * None (verification only)
- Verification Summary:
  * `CUDA_SIMULATION` constant defined at line 19: `CUDA_SIMULATION: bool = os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"`
  * `CUDA_SIMULATION` correctly exported in `__all__` at line 317 (first item in the list)
  * The constant is ready for import by other modules that need CUDASIM-safe local array patterns
- Issues Flagged: None

**Tests to Run**:
- None (verification task only)

---

## Task Group 2: Fix buffer_registry.py - cuda.local.array in CUDABuffer.build_allocator()
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/buffer_registry.py (lines 1-143)
- File: src/cubie/cuda_simsafe.py (lines 1-35 for CUDA_SIMULATION constant and compile_kwargs)

**Input Validation Required**:
- None (existing validation in CUDABuffer attrs class is sufficient)

**Tasks**:
1. **Import CUDA_SIMULATION from cuda_simsafe**
   - File: src/cubie/buffer_registry.py
   - Action: Modify
   - Details: Add `CUDA_SIMULATION` to the import from `cubie.cuda_simsafe`:
     ```python
     from cubie.cuda_simsafe import compile_kwargs, CUDA_SIMULATION
     ```
   - Current import at line 20: `from cubie.cuda_simsafe import compile_kwargs`

2. **Apply compile-time conditional pattern in CUDABuffer.build_allocator()**
   - File: src/cubie/buffer_registry.py
   - Action: Modify
   - Details: Replace the single `allocate_buffer` device function (lines 128-141) with compile-time conditional definitions:
     ```python
     if CUDA_SIMULATION:
         @cuda.jit(device=True, inline=True, **compile_kwargs)
         def allocate_buffer(shared, persistent):
             """Allocate buffer from appropriate memory region."""
             if _use_shared:
                 array = shared[_shared_slice]
             elif _use_persistent:
                 array = persistent[_persistent_slice]
             else:
                 # CUDASIM: use numpy.zeros instead of cuda.local.array
                 array = np.zeros(_local_size, dtype=_precision)
             if _zero:
                 for i in range(elements):
                     array[i] = _precision(0.0)
             return array
     else:
         @cuda.jit(device=True, inline=True, **compile_kwargs)
         def allocate_buffer(shared, persistent):
             """Allocate buffer from appropriate memory region."""
             if _use_shared:
                 array = shared[_shared_slice]
             elif _use_persistent:
                 array = persistent[_persistent_slice]
             else:
                 array = cuda.local.array(_local_size, _precision)
             if _zero:
                 for i in range(elements):
                     array[i] = _precision(0.0)
             return array
     ```
   - Edge cases: 
     - Zero-size arrays: Already handled by `sizes[name] = max(entry.size, 1)` pattern elsewhere
     - Precision types: Both numpy and cuda.local.array handle np.float32/float64/int32
   - Integration: The returned `allocate_buffer` function is used identically regardless of mode

**Tests to Create**:
- None (existing tests will verify correct behavior)

**Tests to Run**:
- tests/test_buffer_registry.py

**Outcomes**:
- Files Modified: 
  * src/cubie/buffer_registry.py (16 lines changed)
- Functions/Methods Added/Modified:
  * CUDABuffer.build_allocator() in buffer_registry.py - Added compile-time conditional for CUDASIM mode
- Implementation Summary:
  * Added CUDA_SIMULATION import from cubie.cuda_simsafe
  * Applied compile-time conditional pattern: when CUDA_SIMULATION is True, uses np.zeros() instead of cuda.local.array()
  * Both branches define identical allocate_buffer device function signatures, only differing in local array allocation method
- Issues Flagged: None

---

## Task Group 3: Fix BatchSolverKernel.py - cuda.local.array in kernel
**Status**: [x]
**Dependencies**: Task Group 1 (complete)

**Required Context**:
- File: src/cubie/batchsolving/BatchSolverKernel.py (lines 1-30 for imports, lines 540-620 for build_kernel method with cuda.local.array at line 578-580)
- File: src/cubie/cuda_simsafe.py (lines 1-35 for CUDA_SIMULATION constant)

**Input Validation Required**:
- None

**Tasks**:
1. **Import CUDA_SIMULATION from cuda_simsafe**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details: Add `CUDA_SIMULATION` to the import from `cubie.cuda_simsafe`:
     ```python
     from cubie.cuda_simsafe import is_cudasim_enabled, compile_kwargs, CUDA_SIMULATION
     ```
   - Current import at lines 13-14: `from cubie.cuda_simsafe import is_cudasim_enabled, compile_kwargs`

2. **Apply compile-time conditional for local_scratch allocation**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details: Replace the `cuda.local.array` call at lines 578-580 with conditional:
     ```python
     # Inside the kernel function, replace:
     # local_scratch = cuda.local.array(local_elements_per_run, dtype=float32)
     # With compile-time conditional captured in closure:
     if CUDA_SIMULATION:
         local_scratch = np.zeros(local_elements_per_run, dtype=np.float32)
     else:
         local_scratch = cuda.local.array(local_elements_per_run, dtype=float32)
     ```
   - Note: This must be done INSIDE the kernel function definition, using the CUDA_SIMULATION constant that was captured in the enclosing build() method's closure.
   - Edge cases: `local_elements_per_run` is computed earlier and guaranteed to be >= 1
   - Integration: The local_scratch array is used identically in both modes

**Tests to Create**:
- None (existing solver tests will verify correct behavior)

**Tests to Run**:
- tests/batchsolving/test_solver.py
- tests/batchsolving/test_solveresult.py

**Outcomes**:
- Files Modified: 
  * src/cubie/batchsolving/BatchSolverKernel.py (7 lines changed)
- Functions/Methods Added/Modified:
  * Import statement at lines 13-14 - Added CUDA_SIMULATION to imports
  * batch_solver_kernel() in BatchSolverKernel.py - Added compile-time conditional for local_scratch allocation
- Implementation Summary:
  * Added CUDA_SIMULATION import from cubie.cuda_simsafe
  * Applied compile-time conditional pattern at lines 578-584: when CUDA_SIMULATION is True, uses np.zeros() instead of cuda.local.array()
  * Both branches use identical array semantics, only differing in allocation method
- Issues Flagged: None

---

## Task Group 4: Fix ode_loop.py - cuda.local.array in IVPLoop.build()
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/integrators/loops/ode_loop.py (lines 1-25 for imports, lines 480-490 for cuda.local.array at line 487)
- File: src/cubie/cuda_simsafe.py (lines 1-35 for CUDA_SIMULATION constant)

**Input Validation Required**:
- None

**Tasks**:
1. **Import CUDA_SIMULATION from cuda_simsafe**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Details: Add `CUDA_SIMULATION` to the import from `cubie.cuda_simsafe`:
     ```python
     from cubie.cuda_simsafe import from_dtype as simsafe_dtype
     from cubie.cuda_simsafe import activemask, all_sync, compile_kwargs, selp, CUDA_SIMULATION
     ```
   - Current imports at lines 16-17:
     ```python
     from cubie.cuda_simsafe import from_dtype as simsafe_dtype
     from cubie.cuda_simsafe import activemask, all_sync, compile_kwargs, selp
     ```

2. **Apply compile-time conditional for proposed_counters allocation**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Details: Replace line 487 `proposed_counters = cuda.local.array(2, dtype=simsafe_int32)` with:
     ```python
     if CUDA_SIMULATION:
         proposed_counters = np.zeros(2, dtype=np.int32)
     else:
         proposed_counters = cuda.local.array(2, dtype=simsafe_int32)
     ```
   - Note: This must be inside the device function definition, using the CUDA_SIMULATION constant that was captured in the enclosing build() method's closure.
   - Edge cases: Fixed size of 2, always valid
   - Integration: proposed_counters is used identically in both modes

**Tests to Create**:
- None (existing loop tests will verify correct behavior)

**Tests to Run**:
- tests/integrators/loops/test_ode_loop.py

**Outcomes**:
- Files Modified: 
  * src/cubie/integrators/loops/ode_loop.py (5 lines changed)
- Functions/Methods Added/Modified:
  * Import statement at line 17 - Added CUDA_SIMULATION to imports
  * ivp_loop device function in IVPLoop.build() - Added compile-time conditional for proposed_counters allocation
- Implementation Summary:
  * Added CUDA_SIMULATION import from cubie.cuda_simsafe
  * Applied compile-time conditional pattern at lines 487-490: when CUDA_SIMULATION is True, uses np.zeros() instead of cuda.local.array()
  * Both branches use identical array semantics, only differing in allocation method
- Issues Flagged: None

---

## Task Group 5: Fix newton_krylov.py - cuda.local.array in solver
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py (lines 1-35 for imports, lines 375-390 for cuda.local.array at line 383)
- File: src/cubie/cuda_simsafe.py (lines 1-35 for CUDA_SIMULATION constant)

**Input Validation Required**:
- None

**Tasks**:
1. **Import CUDA_SIMULATION from cuda_simsafe**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify
   - Details: Add `CUDA_SIMULATION` to the import from `cubie.cuda_simsafe`:
     ```python
     from cubie.cuda_simsafe import activemask, all_sync, selp, any_sync, compile_kwargs, CUDA_SIMULATION
     ```
   - Current import at line 27: `from cubie.cuda_simsafe import activemask, all_sync, selp, any_sync, compile_kwargs`

2. **Apply compile-time conditional for krylov_iters_local allocation**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify
   - Details: Replace line 383 `krylov_iters_local = cuda.local.array(1, int32)` with:
     ```python
     if CUDA_SIMULATION:
         krylov_iters_local = np.zeros(1, dtype=np.int32)
     else:
         krylov_iters_local = cuda.local.array(1, int32)
     ```
   - Note: This must be inside the device function definition, using the CUDA_SIMULATION constant that was captured in the enclosing build() method's closure.
   - Edge cases: Fixed size of 1, always valid
   - Integration: krylov_iters_local is used identically in both modes

**Tests to Create**:
- None (existing solver tests will verify correct behavior)

**Tests to Run**:
- tests/integrators/algorithms/test_newton_krylov.py (if exists)
- tests/batchsolving/test_solver.py (exercises newton_krylov through implicit algorithms)

**Outcomes**:
- Files Modified: 
  * src/cubie/integrators/matrix_free_solvers/newton_krylov.py (5 lines changed)
- Functions/Methods Added/Modified:
  * Import statement at line 27 - Added CUDA_SIMULATION to imports
  * newton_krylov_solve device function in NewtonKrylov.build() - Added compile-time conditional for krylov_iters_local allocation
- Implementation Summary:
  * Added CUDA_SIMULATION import from cubie.cuda_simsafe
  * Applied compile-time conditional pattern at lines 383-386: when CUDA_SIMULATION is True, uses np.zeros() instead of cuda.local.array()
  * Both branches use identical array semantics, only differing in allocation method
- Issues Flagged: None

---

## Task Group 6: Fix generic_rosenbrock_w.py - cuda.local.array in step function
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py (lines 1-60 for imports, lines 440-455 for cuda.local.array at lines 448-449)
- File: src/cubie/cuda_simsafe.py (lines 1-35 for CUDA_SIMULATION constant)

**Input Validation Required**:
- None

**Tasks**:
1. **Import CUDA_SIMULATION and numpy for CUDASIM fallback**
   - File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py
   - Action: Modify
   - Details: Add `CUDA_SIMULATION` import. Check if numpy is already imported; if not, add it. Look for existing imports around lines 33-53.
     - Add to imports: `from cubie.cuda_simsafe import CUDA_SIMULATION`
     - Verify `import numpy as np` exists (should already be present at line 37)

2. **Apply compile-time conditional for base_state_placeholder and krylov_iters_out allocations**
   - File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py
   - Action: Modify
   - Details: Replace lines 448-449:
     ```python
     base_state_placeholder = cuda.local.array(1, int32)
     krylov_iters_out = cuda.local.array(1, int32)
     ```
     With:
     ```python
     if CUDA_SIMULATION:
         base_state_placeholder = np.zeros(1, dtype=np.int32)
         krylov_iters_out = np.zeros(1, dtype=np.int32)
     else:
         base_state_placeholder = cuda.local.array(1, int32)
         krylov_iters_out = cuda.local.array(1, int32)
     ```
   - Note: This must be inside the step device function definition, using the CUDA_SIMULATION constant that was captured in the enclosing build_step() method's closure.
   - Edge cases: Fixed size of 1, always valid
   - Integration: Both arrays are used identically in both modes

**Tests to Create**:
- None (existing algorithm tests will verify correct behavior)

**Tests to Run**:
- tests/integrators/algorithms/test_generic_rosenbrock_w.py (if exists)
- tests/batchsolving/test_solver.py (exercises rosenbrock through algorithm tests)

**Outcomes**:
- Files Modified: 
  * src/cubie/integrators/algorithms/generic_rosenbrock_w.py (7 lines changed)
- Functions/Methods Added/Modified:
  * Import statement at line 54 - Added CUDA_SIMULATION import from cubie.cuda_simsafe
  * step device function in GenericRosenbrockWStep.build_step() - Added compile-time conditional for base_state_placeholder and krylov_iters_out allocations
- Implementation Summary:
  * Added CUDA_SIMULATION import from cubie.cuda_simsafe
  * Verified numpy as np already imported at line 37
  * Applied compile-time conditional pattern at lines 449-454: when CUDA_SIMULATION is True, uses np.zeros() instead of cuda.local.array()
  * Both branches use identical array semantics, only differing in allocation method
- Issues Flagged: None

---

## Task Group 7: Add CUDASIM-specific Tests
**Status**: [x]
**Dependencies**: Task Groups 2-6

**Required Context**:
- File: tests/test_cuda_simsafe.py (entire file - see lines 1-43 for existing test patterns)
- File: tests/test_buffer_registry.py (lines 1-100 for existing buffer registry test patterns)
- File: src/cubie/buffer_registry.py (lines 75-143 for CUDABuffer.build_allocator method)

**Input Validation Required**:
- None (tests do not require input validation)

**Tasks**:
1. **Add test for buffer_registry allocator in CUDASIM mode**
   - File: tests/test_buffer_registry.py
   - Action: Modify (add new test)
   - Details: Add a test that verifies the allocator works correctly when the buffer uses local memory:
     ```python
     @pytest.mark.sim_only
     def test_local_buffer_allocator_cudasim():
         """Test that local buffer allocator works in CUDASIM mode.
         
         Verifies that the conditional cuda.local.array fix produces
         a valid array in simulation mode.
         """
         from cubie.buffer_registry import BufferRegistry
         import numpy as np
         
         registry = BufferRegistry()
         factory = MockFactory()
         factory.precision = np.float32
         
         # Register a local buffer
         registry.register('local_buf', factory, 10, 'local')
         
         # Get allocator - this should work without cuda.local.array errors
         allocator = registry.get_allocator('local_buf', factory)
         
         # In CUDASIM, allocator should return a numpy array
         # when called with dummy shared/persistent arrays
         shared = np.zeros(1, dtype=np.float32)
         persistent = np.zeros(1, dtype=np.float32)
         result = allocator(shared, persistent)
         
         assert result is not None
         assert len(result) == 10
     ```
   - Edge cases: Test covers local memory allocation path specifically
   - Integration: Uses existing MockFactory pattern from test file

**Tests to Create**:
- tests/test_buffer_registry.py::test_local_buffer_allocator_cudasim

**Tests to Run**:
- tests/test_buffer_registry.py
- tests/test_cuda_simsafe.py

**Outcomes**:
- Files Modified: 
  * tests/test_buffer_registry.py (22 lines added)
- Functions/Methods Added/Modified:
  * test_local_buffer_allocator_cudasim() in test_buffer_registry.py
- Implementation Summary:
  * Added standalone test function marked with @pytest.mark.sim_only
  * Test creates a BufferRegistry and MockFactory, registers a local buffer
  * Test verifies that get_allocator() returns a callable that produces a valid array
  * Uses existing MockFactory pattern from the test file
  * Test verifies the CUDASIM-safe local array allocation works correctly
- Issues Flagged: None

---

## Summary

**Total Task Groups**: 7
**Dependency Chain**: 
- Group 1 (infrastructure) → Groups 2-6 (parallel fixes) → Group 7 (tests)

**Files Modified**:
1. `src/cubie/buffer_registry.py` - Add CUDA_SIMULATION import, apply conditional pattern
2. `src/cubie/batchsolving/BatchSolverKernel.py` - Add CUDA_SIMULATION import, apply conditional pattern
3. `src/cubie/integrators/loops/ode_loop.py` - Add CUDA_SIMULATION import, apply conditional pattern
4. `src/cubie/integrators/matrix_free_solvers/newton_krylov.py` - Add CUDA_SIMULATION import, apply conditional pattern
5. `src/cubie/integrators/algorithms/generic_rosenbrock_w.py` - Add CUDA_SIMULATION import, apply conditional pattern
6. `tests/test_buffer_registry.py` - Add CUDASIM-specific test

**Estimated Complexity**: Low-Medium
- Each fix follows the same pattern
- No API changes
- Compile-time conditional selection ensures no runtime overhead
- Tests should pass without modification (new test added for explicit verification)
