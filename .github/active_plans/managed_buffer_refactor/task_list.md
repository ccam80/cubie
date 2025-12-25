# Implementation Task List
# Feature: Managed Buffer Refactor - Centralize CUDASIM Compatibility
# Plan Reference: .github/active_plans/managed_buffer_refactor/agent_plan.md

---

## Task Group 1: Extend Buffer Registry for Integer Types
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/_utils.py (lines 24-62) - precision_validator and ALLOWED_PRECISIONS
- File: src/cubie/buffer_registry.py (lines 24-60, 116-158) - CUDABuffer class and build_allocator

**Input Validation Required**:
- precision: Extend to accept np.int32, np.int64 in addition to float types
- The existing validation pattern should be extended, not replaced

**Tasks**:
1. **Extend ALLOWED_PRECISIONS in _utils.py**
   - File: src/cubie/_utils.py
   - Action: Modify
   - Details:
     ```python
     # Current (lines 33-37):
     ALLOWED_PRECISIONS = {
         np.dtype(np.float16),
         np.dtype(np.float32),
         np.dtype(np.float64),
     }
     
     # Add integer dtypes for buffer registry support:
     ALLOWED_BUFFER_DTYPES = {
         np.dtype(np.float16),
         np.dtype(np.float32),
         np.dtype(np.float64),
         np.dtype(np.int32),
         np.dtype(np.int64),
     }
     ```
   - Edge cases: Keep ALLOWED_PRECISIONS unchanged for precision_validator (floats only)
   - Integration: New set used only by buffer_registry

2. **Add buffer_dtype_validator to _utils.py**
   - File: src/cubie/_utils.py
   - Action: Modify
   - Details:
     ```python
     def buffer_dtype_validator(
         _: object,
         __: Attribute,
         value: type,
     ) -> None:
         """Validate that value is a supported buffer dtype (float or int)."""
         if np.dtype(value) not in ALLOWED_BUFFER_DTYPES:
             raise ValueError(
                 "Buffer dtype must be one of float16, float32, float64, "
                 "int32, or int64",
             )
     ```
   - Edge cases: Accept numpy scalar types like np.int32 as well as dtypes
   - Integration: Used in buffer_registry.py CUDABuffer class

3. **Update CUDABuffer to use buffer_dtype_validator**
   - File: src/cubie/buffer_registry.py
   - Action: Modify
   - Details:
     ```python
     # Import the new validator
     from cubie._utils import getype_validator, buffer_dtype_validator
     
     # Change CUDABuffer.precision field (line 55-58):
     precision: type = attrs.field(
         default=np.float32,
         validator=buffer_dtype_validator  # Was: precision_validator
     )
     ```
   - Edge cases: Ensure build_allocator works with int32 precision
   - Integration: All existing usages continue to work

**Tests to Create**:
- Test file: tests/test_buffer_registry.py
- Test function: test_cudabuffer_accepts_int32_precision (add to TestPrecisionValidation class)
- Description: Verify CUDABuffer can be created with np.int32 precision
- Test function: test_cudabuffer_accepts_int64_precision (add to TestPrecisionValidation class)
- Description: Verify CUDABuffer can be created with np.int64 precision

**Tests to Modify**:
- Test file: tests/test_buffer_registry.py
- Test function: test_invalid_precision_raises (lines 454-461)
- Description: Update to use a truly invalid dtype (e.g., np.complex64) since int32 will now be valid

**Tests to Run**:
- tests/test_buffer_registry.py

**Outcomes**: 
- Files Modified:
  * src/cubie/_utils.py (14 lines added)
  * src/cubie/buffer_registry.py (1 line changed)
  * tests/test_buffer_registry.py (17 lines added)
- Functions/Methods Added/Modified:
  * ALLOWED_BUFFER_DTYPES (new set) in _utils.py
  * buffer_dtype_validator() (new function) in _utils.py
  * CUDABuffer.precision field validator changed in buffer_registry.py
  * test_valid_int32_precision() (new test) in test_buffer_registry.py
  * test_valid_int64_precision() (new test) in test_buffer_registry.py
  * test_invalid_precision_raises() (modified) in test_buffer_registry.py
- Implementation Summary:
  Added ALLOWED_BUFFER_DTYPES set containing float16, float32, float64, int32, int64.
  Created buffer_dtype_validator function to validate buffer dtypes.
  Updated CUDABuffer.precision field to use buffer_dtype_validator.
  Added tests for int32 and int64 precision acceptance.
  Updated invalid precision test to use np.complex64.
- Issues Flagged: None

---

## Task Group 2: Add location Parameters to Frozensets
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/integrators/algorithms/base_algorithm_step.py (lines 24-47) - ALL_ALGORITHM_STEP_PARAMETERS
- File: src/cubie/integrators/step_control/base_step_controller.py (lines 27-34) - ALL_STEP_CONTROLLER_PARAMETERS
- File: src/cubie/integrators/loops/ode_loop.py (lines 37-58) - ALL_LOOP_SETTINGS

**Input Validation Required**:
- None - this is additive only

**Tasks**:
1. **Add proposed_counters_location to ALL_LOOP_SETTINGS**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Details:
     ```python
     # Add to ALL_LOOP_SETTINGS set (after line 57):
     ALL_LOOP_SETTINGS = {
         "dt_save",
         "dt_summarise",
         "dt0",
         "dt_min",
         "dt_max",
         "is_adaptive",
         # Loop buffer location parameters
         "state_location",
         "proposed_state_location",
         "parameters_location",
         "drivers_location",
         "proposed_drivers_location",
         "observables_location",
         "proposed_observables_location",
         "error_location",
         "counters_location",
         "state_summary_location",
         "observable_summary_location",
         "dt_location",
         "accept_step_location",
         "proposed_counters_location",  # NEW
     }
     ```
   - Edge cases: None
   - Integration: Enables proposed_counters_location to be passed through update()

2. **Add krylov_iters_local_location to ALL_ALGORITHM_STEP_PARAMETERS**
   - File: src/cubie/integrators/algorithms/base_algorithm_step.py
   - Action: Modify
   - Details:
     ```python
     # Add to ALL_ALGORITHM_STEP_PARAMETERS set (line 46):
     ALL_ALGORITHM_STEP_PARAMETERS = {
         # ... existing entries ...
         'delta_location',
         'residual_location', 'residual_temp_location', 'stage_base_bt_location',
         'krylov_iters_local_location',  # NEW - for newton_krylov.py
     }
     ```
   - Edge cases: None
   - Integration: Enables location param to pass through algorithm update()

3. **Add base_state_placeholder_location and krylov_iters_out_location to ALL_ALGORITHM_STEP_PARAMETERS**
   - File: src/cubie/integrators/algorithms/base_algorithm_step.py
   - Action: Modify
   - Details:
     ```python
     # Add after krylov_iters_local_location:
     ALL_ALGORITHM_STEP_PARAMETERS = {
         # ... existing entries ...
         'krylov_iters_local_location',
         'base_state_placeholder_location',  # NEW - for rosenbrock
         'krylov_iters_out_location',  # NEW - for rosenbrock
     }
     ```
   - Edge cases: None
   - Integration: Enables location params for Rosenbrock buffers

**Tests to Create**:
- None required - parameter sets are tested implicitly through update() calls

**Tests to Run**:
- None

**Outcomes**: 
- Files Modified:
  * src/cubie/integrators/loops/ode_loop.py (1 line added)
  * src/cubie/integrators/algorithms/base_algorithm_step.py (4 lines added)
- Functions/Methods Added/Modified:
  * ALL_LOOP_SETTINGS set in ode_loop.py - added "proposed_counters_location"
  * ALL_ALGORITHM_STEP_PARAMETERS set in base_algorithm_step.py - added 'krylov_iters_local_location', 'base_state_placeholder_location', 'krylov_iters_out_location'
- Implementation Summary:
  Added location parameters to both frozensets to enable them to pass through update() methods. These parameters will be used by Task Groups 3, 4, and 5 when registering and allocating int32 buffers.
- Issues Flagged: None

---

## Task Group 3: Refactor ode_loop.py - proposed_counters Buffer
**Status**: [x]
**Dependencies**: Task Groups 1, 2

**Required Context**:
- File: src/cubie/integrators/loops/ode_loop.py (entire file)
- File: src/cubie/integrators/loops/ode_loop_config.py (entire file) 
- File: src/cubie/buffer_registry.py (lines 116-158, 585-631) - build_allocator, register

**Input Validation Required**:
- proposed_counters_location: Validate in_(['shared', 'local'])

**Tasks**:
1. **Add proposed_counters_location to ODELoopConfig**
   - File: src/cubie/integrators/loops/ode_loop_config.py
   - Action: Modify
   - Details:
     ```python
     # Add field to ODELoopConfig class:
     proposed_counters_location: str = attrs.field(
         default='local',
         validator=validators.in_(['shared', 'local'])
     )
     ```
   - Edge cases: Default to 'local' to preserve existing behavior
   - Integration: Field accessed in IVPLoop.register_buffers()

2. **Register proposed_counters buffer in IVPLoop.register_buffers()**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Details:
     ```python
     # Add to register_buffers() method after line 281:
     buffer_registry.register(
         'proposed_counters', self, 2, config.proposed_counters_location,
         precision=np.int32
     )
     ```
   - Edge cases: Size is fixed at 2, precision is int32
   - Integration: Imports np at top of file (already imported)

3. **Replace inline CUDA_SIMULATION conditional with allocator in IVPLoop.build()**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Details:
     ```python
     # Add allocator get (after line 347, with other allocators):
     alloc_proposed_counters = getalloc('proposed_counters', self)
     
     # Replace lines 487-490 (inside loop_fn device function):
     # OLD:
     # if CUDA_SIMULATION:
     #     proposed_counters = np.zeros(2, dtype=np.int32)
     # else:
     #     proposed_counters = cuda.local.array(2, dtype=simsafe_int32)
     
     # NEW:
     proposed_counters = alloc_proposed_counters(shared_scratch, persistent_local)
     ```
   - Edge cases: The buffer is int32, allocator handles CUDASIM swap
   - Integration: Remove CUDA_SIMULATION import if no longer needed

4. **Remove unused CUDA_SIMULATION import if applicable**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify (conditional)
   - Details: Check if CUDA_SIMULATION is used elsewhere in file. If not, remove from import.
   - Edge cases: May still be needed for other purposes
   - Integration: Clean import

**Tests to Create**:
- None - existing tests cover functionality

**Tests to Run**:
- tests/integrators/loops/test_ode_loop.py

**Outcomes**: 
- Files Modified:
  * src/cubie/integrators/loops/ode_loop_config.py (4 lines added)
  * src/cubie/integrators/loops/ode_loop.py (8 lines changed)
- Functions/Methods Added/Modified:
  * proposed_counters_location field added to ODELoopConfig in ode_loop_config.py
  * register_buffers() method in IVPLoop - added proposed_counters registration
  * build() method in IVPLoop - added alloc_proposed_counters allocator
  * loop_fn device function - replaced CUDA_SIMULATION conditional with allocator call
- Implementation Summary:
  Added proposed_counters_location config field with 'local' default.
  Registered proposed_counters buffer with size 2 and np.int32 precision.
  Replaced inline CUDA_SIMULATION conditional with managed buffer allocator.
  Removed unused CUDA_SIMULATION import and simsafe_dtype import.
  Removed unused simsafe_int32 variable.
- Issues Flagged: None

---

## Task Group 4: Refactor newton_krylov.py - krylov_iters_local Buffer
**Status**: [x]
**Dependencies**: Task Groups 1, 2

**Required Context**:
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py (entire file)
- File: src/cubie/buffer_registry.py (lines 116-158, 585-631)

**Input Validation Required**:
- krylov_iters_local_location: Validate in_(['shared', 'local'])

**Tasks**:
1. **Add krylov_iters_local_location to NewtonKrylovConfig**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify
   - Details:
     ```python
     # Add field to NewtonKrylovConfig class (after line 111):
     krylov_iters_local_location: str = attrs.field(
         default='local',
         validator=validators.in_(["local", "shared"])
     )
     ```
   - Edge cases: Default to 'local' to preserve existing behavior
   - Integration: Accessed in register_buffers() and settings_dict

2. **Add krylov_iters_local_location to settings_dict property**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify
   - Details:
     ```python
     # Add to settings_dict (inside the return dict, after line 158):
     'krylov_iters_local_location': self.krylov_iters_local_location,
     ```
   - Edge cases: None
   - Integration: Ensures parameter is exposed to parent components

3. **Register krylov_iters_local buffer in NewtonKrylov.register_buffers()**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify
   - Details:
     ```python
     # Add to register_buffers() method (after line 256):
     buffer_registry.register(
         'krylov_iters_local',
         self,
         1,
         config.krylov_iters_local_location,
         precision=np.int32
     )
     ```
   - Edge cases: Size is 1, precision is int32
   - Integration: Buffer registry handles CUDASIM swap

4. **Replace inline CUDA_SIMULATION conditional with allocator in build()**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify
   - Details:
     ```python
     # Add allocator get (after line 302, with other allocators):
     alloc_krylov_iters_local = get_alloc('krylov_iters_local', self)
     
     # Replace lines 383-386 (inside newton_krylov_solver device function):
     # OLD:
     # if CUDA_SIMULATION:
     #     krylov_iters_local = np.zeros(1, dtype=np.int32)
     # else:
     #     krylov_iters_local = cuda.local.array(1, int32)
     
     # NEW:
     krylov_iters_local = alloc_krylov_iters_local(shared_scratch, persistent_scratch)
     ```
   - Edge cases: The buffer is int32, allocator handles CUDASIM swap
   - Integration: Uses existing shared_scratch and persistent_scratch params

5. **Remove unused CUDA_SIMULATION import if applicable**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify (conditional)
   - Details: Check if CUDA_SIMULATION is used elsewhere. If not, remove from import line 27.
   - Edge cases: May still be needed for other purposes
   - Integration: Clean import

**Tests to Create**:
- None - existing tests cover functionality

**Tests to Run**:
- tests/integrators/matrix_free_solvers/test_newton_krylov.py

**Outcomes**: 
- Files Modified:
  * src/cubie/integrators/matrix_free_solvers/newton_krylov.py (10 lines changed)
- Functions/Methods Added/Modified:
  * krylov_iters_local_location field added to NewtonKrylovConfig
  * settings_dict property in NewtonKrylovConfig - added krylov_iters_local_location
  * register_buffers() method - added krylov_iters_local buffer registration
  * build() method - added alloc_krylov_iters_local allocator retrieval
  * newton_krylov_solver device function - replaced CUDA_SIMULATION conditional
- Implementation Summary:
  Added krylov_iters_local_location config field with 'local' default and in_ validator.
  Added the location to settings_dict for proper exposure to parent components.
  Registered krylov_iters_local buffer with size 1 and np.int32 precision.
  Replaced inline CUDA_SIMULATION conditional with managed buffer allocator call.
  Removed unused CUDA_SIMULATION import from cuda_simsafe.
- Issues Flagged: None

---

## Task Group 5: Refactor generic_rosenbrock_w.py - base_state_placeholder and krylov_iters_out Buffers
**Status**: [x]
**Dependencies**: Task Groups 1, 2

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py (entire file)
- File: src/cubie/buffer_registry.py (lines 116-158, 585-631)

**Input Validation Required**:
- base_state_placeholder_location: Validate in_(['shared', 'local'])
- krylov_iters_out_location: Validate in_(['shared', 'local'])

**Tasks**:
1. **Add location fields to RosenbrockWStepConfig**
   - File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py
   - Action: Modify
   - Details:
     ```python
     # Add fields to RosenbrockWStepConfig class (after line 141):
     base_state_placeholder_location: str = attrs.field(
         default='local',
         validator=attrs.validators.in_(['local', 'shared'])
     )
     krylov_iters_out_location: str = attrs.field(
         default='local',
         validator=attrs.validators.in_(['local', 'shared'])
     )
     ```
   - Edge cases: Default to 'local' to preserve existing behavior
   - Integration: Accessed in register_buffers()

2. **Register int32 buffers in GenericRosenbrockWStep.register_buffers()**
   - File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py
   - Action: Modify
   - Details:
     ```python
     # Add to register_buffers() method (after line 275):
     buffer_registry.register(
         'base_state_placeholder', self, 1,
         config.base_state_placeholder_location,
         precision=np.int32
     )
     buffer_registry.register(
         'krylov_iters_out', self, 1,
         config.krylov_iters_out_location,
         precision=np.int32
     )
     ```
   - Edge cases: Size is 1 for each, precision is int32
   - Integration: Buffer registry handles CUDASIM swap

3. **Replace inline CUDA_SIMULATION conditional with allocators in build_step()**
   - File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py
   - Action: Modify
   - Details:
     ```python
     # Add allocator gets (after line 400, with other allocators):
     alloc_base_state_placeholder = getalloc('base_state_placeholder', self)
     alloc_krylov_iters_out = getalloc('krylov_iters_out', self)
     
     # Replace lines 449-454 (inside step device function):
     # OLD:
     # if CUDA_SIMULATION:
     #     base_state_placeholder = np.zeros(1, dtype=np.int32)
     #     krylov_iters_out = np.zeros(1, dtype=np.int32)
     # else:
     #     base_state_placeholder = cuda.local.array(1, int32)
     #     krylov_iters_out = cuda.local.array(1, int32)
     
     # NEW:
     base_state_placeholder = alloc_base_state_placeholder(shared, persistent_local)
     krylov_iters_out = alloc_krylov_iters_out(shared, persistent_local)
     ```
   - Edge cases: The buffers are int32, allocator handles CUDASIM swap
   - Integration: Uses existing shared and persistent_local params

4. **Remove unused CUDA_SIMULATION import if applicable**
   - File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py
   - Action: Modify (conditional)
   - Details: Check if CUDA_SIMULATION is used elsewhere. If not, remove from import line 54.
   - Edge cases: May still be needed for other purposes
   - Integration: Clean import

**Tests to Create**:
- None - existing tests cover functionality

**Tests to Run**:
- tests/integrators/algorithms/test_generic_rosenbrock_w.py

**Outcomes**: 
- Files Modified:
  * src/cubie/integrators/algorithms/generic_rosenbrock_w.py (18 lines changed)
- Functions/Methods Added/Modified:
  * base_state_placeholder_location field added to RosenbrockWStepConfig
  * krylov_iters_out_location field added to RosenbrockWStepConfig
  * register_buffers() method - added base_state_placeholder and krylov_iters_out buffer registrations with np.int32 precision
  * build_step() method - added alloc_base_state_placeholder and alloc_krylov_iters_out allocator retrievals
  * step device function - replaced CUDA_SIMULATION conditional with managed buffer allocator calls
- Implementation Summary:
  Added two location config fields with 'local' default and in_ validators.
  Registered both int32 buffers with size 1 in register_buffers().
  Retrieved allocators in build_step() using getalloc pattern.
  Replaced inline CUDA_SIMULATION conditional with allocator calls.
  Removed unused CUDA_SIMULATION import from cuda_simsafe.
- Issues Flagged: None

---

## Task Group 6: Refactor BatchSolverKernel.py - local_scratch Buffer
**Status**: [x]
**Dependencies**: Task Groups 1, 2

**Required Context**:
- File: src/cubie/batchsolving/BatchSolverKernel.py (lines 473-630) - build_kernel method
- File: src/cubie/buffer_registry.py (lines 116-158) - build_allocator with CUDASIM handling

**Input Validation Required**:
- None - size is determined at runtime from local_memory_elements

**Tasks**:
1. **Create standalone allocator using buffer_registry pattern in build_kernel()**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     The BatchSolverKernel is a CUDA kernel (not device function) with a unique
     structure. The `local_scratch` is allocated inside the kernel and passed
     to the loop function. The simplest approach is to create a standalone
     allocator that mirrors the buffer_registry's CUDASIM handling pattern.
     
     ```python
     # Before the kernel definition (around line 498), create an allocator:
     # This follows the same pattern as buffer_registry.CUDABuffer.build_allocator
     
     if CUDA_SIMULATION:
         @cuda.jit(device=True, inline=True, **compile_kwargs)
         def alloc_local_scratch():
             return np.zeros(local_elements_per_run, dtype=np.float32)
     else:
         @cuda.jit(device=True, inline=True, **compile_kwargs)
         def alloc_local_scratch():
             return cuda.local.array(local_elements_per_run, dtype=float32)
     
     # Replace lines 578-584 inside integration_kernel:
     # OLD:
     # if CUDA_SIMULATION:
     #     local_scratch = np.zeros(local_elements_per_run, dtype=np.float32)
     # else:
     #     local_scratch = cuda.local.array(
     #         local_elements_per_run, dtype=float32
     #     )
     
     # NEW:
     local_scratch = alloc_local_scratch()
     ```
   - Edge cases: The CUDA_SIMULATION check is now at compile-time in build_kernel
   - Integration: Keeps CUDASIM handling consolidated in build-time code

2. **Verify CUDA_SIMULATION import is retained**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Review
   - Details: CUDA_SIMULATION import is needed for the build-time conditional.
     Unlike device functions which use buffer_registry allocators, the kernel
     pattern requires the compile-time check to remain in build_kernel.
   - Edge cases: Import already exists on line 13-14
   - Integration: No change needed

**Note**: This approach keeps ONE CUDASIM check in build_kernel rather than
zero, but it moves the check from inside the kernel to the allocator factory.
This is consistent with the buffer_registry pattern where CUDASIM handling
happens at build time (in build_allocator) rather than runtime. The difference
is that BatchSolverKernel creates its own allocator instead of using
buffer_registry because it operates in a kernel context.

**Tests to Create**:
- None - existing tests cover functionality

**Tests to Run**:
- tests/batchsolving/test_BatchSolverKernel.py

**Outcomes**: 
- Files Modified:
  * src/cubie/batchsolving/BatchSolverKernel.py (5 lines changed)
- Functions/Methods Added/Modified:
  * alloc_local_scratch() device function added in build_kernel()
  * integration_kernel() - replaced CUDA_SIMULATION conditional with allocator call
- Implementation Summary:
  Created standalone allocator device function using compile-time CUDA_SIMULATION
  conditional before the kernel definition. The allocator follows the buffer_registry
  pattern where CUDASIM handling happens at build time. Replaced inline conditional
  inside integration_kernel with simple allocator call. CUDA_SIMULATION import retained
  as needed for the build-time conditional.
- Issues Flagged: None

---

## Task Group 7: Sync Instrumented Test Files
**Status**: [x]
**Dependencies**: Task Groups 3, 4, 5

**Required Context**:
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py (entire file)
- File: tests/integrators/algorithms/instrumented/ (directory listing)
- File: tests/integrators/algorithms/instrumented/matrix_free_solvers.py (if exists)

**Input Validation Required**:
- None

**Tasks**:
1. **Check if instrumented newton_krylov exists and needs update**
   - File: tests/integrators/algorithms/instrumented/
   - Action: Review
   - Details: Check if there's an instrumented version of newton_krylov.py or matrix_free_solvers.py that needs to be updated with the same changes
   - Edge cases: May not exist
   - Integration: Per cubie_internal_structure.md, instrumented files must mirror source

2. **Update instrumented file if it exists**
   - File: tests/integrators/algorithms/instrumented/matrix_free_solvers.py (if exists)
   - Action: Modify (conditional)
   - Details: Apply same buffer registration and allocator changes as source file
   - Edge cases: Instrumented version may have different structure
   - Integration: Keep instrumentation (logging) intact

**Tests to Create**:
- None

**Tests to Run**:
- tests/integrators/algorithms/instrumented/ (all tests using instrumented files)

**Outcomes**: 
- Files Modified:
  * tests/integrators/algorithms/instrumented/generic_rosenbrock_w.py (14 lines changed)
  * tests/integrators/algorithms/instrumented/matrix_free_solvers.py (6 lines changed)
- Functions/Methods Added/Modified:
  * register_buffers() in InstrumentedRosenbrockWStep - added base_state_placeholder and krylov_iters_out buffer registrations with np.int32 precision
  * build_step() in InstrumentedRosenbrockWStep - added alloc_base_state_placeholder and alloc_krylov_iters_out allocator retrievals; replaced cuda.local.array() calls with allocator calls
  * build() in InstrumentedNewtonKrylov - added alloc_krylov_iters_local allocator retrieval; replaced cuda.local.array(1, int32) with allocator call
- Implementation Summary:
  Updated instrumented test files to match the managed buffer pattern changes from Task Groups 4 and 5. For generic_rosenbrock_w.py, added buffer registrations for base_state_placeholder and krylov_iters_out with np.int32 precision, retrieved allocators using getalloc pattern, and replaced direct cuda.local.array() calls with allocator calls. For matrix_free_solvers.py, added allocator retrieval for krylov_iters_local and replaced direct cuda.local.array() call with allocator call. Logging-specific arrays (residual_copy, stage_increment_snapshot, residual_snapshot) were left unchanged as they are instrumentation-only and not part of the production buffer registry pattern. No instrumented ode_loop.py file exists (Task Group 3), so no changes were needed for that file.
- Issues Flagged: None

---

## Task Group 8: Final Verification and Cleanup
**Status**: [ ]
**Dependencies**: All previous task groups

**Required Context**:
- All modified files from previous task groups
- File: src/cubie/buffer_registry.py (lines 128-156) - CUDASIM handling

**Input Validation Required**:
- None

**Tasks**:
1. **Verify CUDA_SIMULATION inline conditionals are refactored in target files**
   - Files: 
     - src/cubie/integrators/loops/ode_loop.py
     - src/cubie/integrators/matrix_free_solvers/newton_krylov.py
     - src/cubie/integrators/algorithms/generic_rosenbrock_w.py
   - Action: Review
   - Details: Confirm no `if CUDA_SIMULATION:` blocks remain INSIDE device functions
     for local array allocation. The CUDASIM handling is now in buffer_registry.
   - Edge cases: CUDA_SIMULATION may be used for other purposes in these files
   - Integration: buffer_registry.py handles CUDASIM for managed buffers

2. **Verify BatchSolverKernel uses allocator pattern**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Review
   - Details: Confirm the CUDA_SIMULATION conditional has been moved from inside
     the kernel to the allocator factory (before kernel definition). The pattern
     should create an allocator device function that is called inside the kernel.
   - Edge cases: This is the exception - kernel context requires build-time allocator
   - Integration: Allocator pattern mirrors buffer_registry approach

3. **Verify buffer_registry.py is the primary source of CUDASIM handling**
   - File: src/cubie/buffer_registry.py
   - Action: Review
   - Details: Confirm build_allocator() at lines 128-156 handles CUDASIM for
     all managed buffer allocations (device functions use this through allocators)
   - Edge cases: BatchSolverKernel has its own allocator for kernel context
   - Integration: Central pattern for device function buffer allocation

4. **Run full test suite to verify no regressions**
   - Action: Execute
   - Details: Run `pytest` to verify all tests pass
   - Edge cases: Some tests may be marked nocudasim
   - Integration: Full integration verification

**Tests to Create**:
- None

**Tests to Run**:
- Full test suite: pytest

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

# Summary

## Total Task Groups: 8

## Dependency Chain:
```
Task Group 1 (Buffer Registry Extension - int32 support)
        ↓
Task Group 2 (Frozenset Updates - location parameters)
        ↓
    ┌───┴───┬───────┬───────┐
    ↓       ↓       ↓       ↓
   TG3     TG4     TG5     TG6
(ode_loop) (newton) (rosenbrock) (kernel)
    └───────┴───────┴───────┘
                ↓
        Task Group 7 (Instrumented Files)
                ↓
        Task Group 8 (Verification)
```

## Tests to Create/Modify:
- tests/test_buffer_registry.py::test_cudabuffer_accepts_int32_precision (new)
- tests/test_buffer_registry.py::test_cudabuffer_accepts_int64_precision (new)
- tests/test_buffer_registry.py::test_invalid_precision_raises (modify - use complex64)

## Tests to Run (per group):
- TG1: tests/test_buffer_registry.py
- TG2: None (implicit validation through later tests)
- TG3: tests/integrators/loops/test_ode_loop.py
- TG4: tests/integrators/matrix_free_solvers/test_newton_krylov.py
- TG5: tests/integrators/algorithms/test_generic_rosenbrock_w.py
- TG6: tests/batchsolving/test_BatchSolverKernel.py
- TG7: tests/integrators/algorithms/instrumented/ (if files exist)
- TG8: Full pytest run

## Estimated Complexity: Medium
- Core pattern already exists in buffer_registry.py
- Main work is applying consistent pattern across 4 files
- Need to extend buffer dtype validation for int32 support (not precision validation)
- Need to update frozensets for new location parameters
- BatchSolverKernel requires special handling (kernel context vs device function)

## Key Design Decisions:
1. **Buffer Registry Extension**: Add ALLOWED_BUFFER_DTYPES set in _utils.py and
   buffer_dtype_validator for CUDABuffer.precision field. Keep ALLOWED_PRECISIONS
   and precision_validator unchanged (floats only for other uses).

2. **Frozenset Updates**: Add location parameters to ALL_LOOP_SETTINGS and
   ALL_ALGORITHM_STEP_PARAMETERS so they pass through update() methods correctly.

3. **Device Function Pattern**: For ode_loop.py, newton_krylov.py, and
   generic_rosenbrock_w.py - register buffers, get allocators, call allocators
   inside device functions. CUDASIM handling is in buffer_registry.

4. **Kernel Pattern**: For BatchSolverKernel.py - create standalone allocator
   device function with CUDASIM handling at build time, call allocator inside
   kernel. This is the exception to the centralized pattern due to kernel context.

