# Implementation Task List
# Feature: Step Controller Buffer Allocation Refactor
# Plan Reference: .github/active_plans/step_controller_buffer_allocation/agent_plan.md

## Task Group 1: Add timestep_memory Configuration Field - SEQUENTIAL
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/step_control/base_step_controller.py (entire file)
- File: src/cubie/buffer_registry.py (lines 1-60 for CUDABuffer reference)
- File: src/cubie/integrators/algorithms/generic_dirk.py (lines 105-128 for location field pattern)

**Input Validation Required**:
- timestep_memory: Validate value is either 'local' or 'shared' using attrs.validators.in_

**Tasks**:
1. **Add timestep_memory field to BaseStepControllerConfig**
   - File: src/cubie/integrators/step_control/base_step_controller.py
   - Action: Modify
   - Details:
     ```python
     # Add after the 'n' field (around line 55)
     timestep_memory: str = field(
         default='local',
         validator=validators.in_(['local', 'shared'])
     )
     ```
   - Edge cases: Field must be accessible to child classes via inheritance
   - Integration: Add `from attrs import validators` to imports if not present

2. **Add timestep_memory to ALL_STEP_CONTROLLER_PARAMETERS**
   - File: src/cubie/integrators/step_control/base_step_controller.py
   - Action: Modify
   - Details:
     ```python
     # Add 'timestep_memory' to the set (around line 26-32)
     ALL_STEP_CONTROLLER_PARAMETERS = {
         'precision', 'n', 'step_controller', 'dt',
         'dt_min', 'dt_max', 'atol', 'rtol', 'algorithm_order',
         'min_gain', 'max_gain', 'safety',
         'kp', 'ki', 'kd', 'deadband_min', 'deadband_max',
         'gamma', 'max_newton_iters',
         'timestep_memory'  # Added for buffer location configuration
     }
     ```
   - Edge cases: None
   - Integration: Enables buffer location updates via update() method

**Outcomes**: 
- Files Modified:
  * src/cubie/integrators/step_control/base_step_controller.py (~15 lines changed)
- Functions/Methods Added/Modified:
  * Added `timestep_memory` field to BaseStepControllerConfig
  * Added `timestep_memory` to ALL_STEP_CONTROLLER_PARAMETERS set
  * Added `validators` to attrs imports
  * Added `buffer_registry` import
- Implementation Summary:
  Added timestep_memory configuration field with validation to BaseStepControllerConfig.
  Field defaults to 'local' and validates against ['local', 'shared'].
- Issues Flagged: None

---

## Task Group 2: Add register_buffers Method to BaseStepController - SEQUENTIAL
**Status**: [x]
**Dependencies**: Group 1

**Required Context**:
- File: src/cubie/integrators/step_control/base_step_controller.py (entire file)
- File: src/cubie/buffer_registry.py (lines 559-606 for register() signature)
- File: src/cubie/integrators/loops/ode_loop.py (lines 246-313 for register_buffers pattern)
- File: src/cubie/integrators/algorithms/generic_dirk.py (lines 289-345 for register_buffers pattern)

**Input Validation Required**:
- None (validation handled by buffer_registry.register())

**Tasks**:
1. **Add buffer_registry import to base_step_controller.py**
   - File: src/cubie/integrators/step_control/base_step_controller.py
   - Action: Modify
   - Details:
     ```python
     # Add import at top of file (after existing imports)
     from cubie.buffer_registry import buffer_registry
     ```
   - Edge cases: None
   - Integration: Required for register_buffers() and build() methods

2. **Add register_buffers() method to BaseStepController**
   - File: src/cubie/integrators/step_control/base_step_controller.py
   - Action: Modify
   - Details:
     ```python
     # Add after the __init__ method (around line 104)
     def register_buffers(self) -> None:
         """Register controller buffers with the central buffer registry.

         Registers the timestep_buffer using size from local_memory_elements
         and location from compile_settings.timestep_memory. Controllers
         with zero buffer requirements still register to maintain consistent
         interface.
         """
         config = self.compile_settings
         precision = config.precision
         size = self.local_memory_elements

         # Clear any existing buffer registrations
         buffer_registry.clear_parent(self)

         # Register timestep buffer
         buffer_registry.register(
             'timestep_buffer',
             self,
             size,
             config.timestep_memory,
             persistent=True,
             precision=precision
         )
     ```
   - Edge cases: Controllers with local_memory_elements == 0 still register with size 0 (buffer_registry handles this via max(size, 1) in allocator)
   - Integration: Must be called after setup_compile_settings() in all controller __init__ methods

**Outcomes**: 
- Files Modified:
  * src/cubie/integrators/step_control/base_step_controller.py (~25 lines changed)
- Functions/Methods Added/Modified:
  * Added register_buffers() method to BaseStepController
- Implementation Summary:
  Added register_buffers() method that registers 'timestep_buffer' with the central
  buffer_registry using size from local_memory_elements and location from 
  compile_settings.timestep_memory. Buffer is registered as persistent.
- Issues Flagged: None

---

## Task Group 3: Update Controller __init__ Methods to Call register_buffers - PARALLEL
**Status**: [x]
**Dependencies**: Group 2

**Required Context**:
- File: src/cubie/integrators/step_control/adaptive_step_controller.py (lines 193-210)
- File: src/cubie/integrators/step_control/fixed_step_controller.py (lines 70-94)
- File: src/cubie/integrators/step_control/adaptive_I_controller.py (lines 19-78)
- File: src/cubie/integrators/step_control/adaptive_PI_controller.py (lines 47-115)
- File: src/cubie/integrators/step_control/adaptive_PID_controller.py (lines 34-106)
- File: src/cubie/integrators/step_control/gustafsson_controller.py (lines 58-126)

**Input Validation Required**:
- None (register_buffers validates via buffer_registry)

**Tasks**:
1. **Add register_buffers() call to BaseAdaptiveStepController.__init__**
   - File: src/cubie/integrators/step_control/adaptive_step_controller.py
   - Action: Modify
   - Details:
     ```python
     # In __init__, after super().__init__() and setup_compile_settings(config)
     # (around line 208-209)
     def __init__(
         self,
         config: AdaptiveStepControlConfig,
     ) -> None:
         """..."""
         super().__init__()
         self.setup_compile_settings(config)
         self.register_buffers()  # Add this line
     ```
   - Edge cases: All adaptive controller subclasses (I, PI, PID, Gustafsson) call super().__init__(config) which will now register buffers
   - Integration: Covers AdaptiveIController, AdaptivePIController, AdaptivePIDController, GustafssonController

2. **Add register_buffers() call to FixedStepController.__init__**
   - File: src/cubie/integrators/step_control/fixed_step_controller.py
   - Action: Modify
   - Details:
     ```python
     # In __init__, after setup_compile_settings(config) (around line 93)
     def __init__(
         self,
         precision: PrecisionDType,
         dt: float,
         n: int = 1,
     ) -> None:
         """..."""
         super().__init__()
         config = FixedStepControlConfig(precision=precision, n=n, dt=dt)
         self.setup_compile_settings(config)
         self.register_buffers()  # Add this line
     ```
   - Edge cases: FixedStepController has local_memory_elements == 0, but register_buffers handles this
   - Integration: Fixed controllers participate in buffer registry like adaptive ones

**Outcomes**: 
- Files Modified:
  * src/cubie/integrators/step_control/adaptive_step_controller.py (~1 line changed)
  * src/cubie/integrators/step_control/fixed_step_controller.py (~1 line changed)
- Functions/Methods Added/Modified:
  * BaseAdaptiveStepController.__init__() - added register_buffers() call
  * FixedStepController.__init__() - added register_buffers() call
- Implementation Summary:
  Added register_buffers() call after setup_compile_settings() in both 
  BaseAdaptiveStepController and FixedStepController. All adaptive controllers
  (I, PI, PID, Gustafsson) inherit from BaseAdaptiveStepController, so they
  automatically get buffer registration through super().__init__(config).
- Issues Flagged: None

---

## Task Group 4: Update Controller build() Methods with Allocator - PARALLEL
**Status**: [x]
**Dependencies**: Group 2

**Required Context**:
- File: src/cubie/integrators/step_control/adaptive_I_controller.py (lines 86-225)
- File: src/cubie/integrators/step_control/adaptive_PI_controller.py (lines 141-281)
- File: src/cubie/integrators/step_control/adaptive_PID_controller.py (lines 155-315)
- File: src/cubie/integrators/step_control/gustafsson_controller.py (lines 145-296)
- File: src/cubie/integrators/step_control/fixed_step_controller.py (lines 95-149)
- File: src/cubie/buffer_registry.py (lines 788-816 for get_allocator signature)
- File: src/cubie/integrators/algorithms/generic_dirk.py (lines 445-510 for allocator usage pattern in build)

**Input Validation Required**:
- None (allocator handles all memory allocation internally)

**Tasks**:
1. **Update AdaptiveIController.build_controller() to use allocator**
   - File: src/cubie/integrators/step_control/adaptive_I_controller.py
   - Action: Modify
   - Details:
     ```python
     # Add buffer_registry import at top of file
     from cubie.buffer_registry import buffer_registry
     
     # In build_controller method (around line 86-100), get allocator:
     def build_controller(
         self,
         precision: PrecisionDType,
         clamp: Callable,
         # ... other params ...
     ) -> ControllerCache:
         """..."""
         # Add at start of method, before existing code:
         alloc_timestep_buffer = buffer_registry.get_allocator(
             'timestep_buffer', self
         )
         
         # ... existing setup code ...
         
         # Modify the device function signature and body (around line 149):
         @cuda.jit(
             device=True,
             inline=True,
             **compile_kwargs,
         )
         def controller_I(
             dt,
             state,
             state_prev,
             error,
             niters,
             accept_out,
             shared_scratch,      # Changed from local_temp
             persistent_local,    # Added new parameter
         ):  # pragma: no cover - CUDA
             """Integral accept/step-size controller.
             
             Parameters
             ----------
             # ... update docstring parameters ...
             shared_scratch : device array
                 Shared memory scratch space.
             persistent_local : device array
                 Persistent local memory for controller state.
             """
             # Allocate buffer at start of function body
             timestep_buffer = alloc_timestep_buffer(
                 shared_scratch, persistent_local
             )
             # Note: I controller has local_memory_elements=0, 
             # timestep_buffer not used but allocation maintains interface
             
             # ... rest of existing implementation unchanged ...
     ```
   - Edge cases: I controller doesn't use the buffer (local_memory_elements=0), but signature change is required for consistency
   - Integration: Works with IVPLoop passing shared_scratch and persistent_local

2. **Update AdaptivePIController.build_controller() to use allocator**
   - File: src/cubie/integrators/step_control/adaptive_PI_controller.py
   - Action: Modify
   - Details:
     ```python
     # Add buffer_registry import at top of file
     from cubie.buffer_registry import buffer_registry
     
     # In build_controller method, get allocator:
     def build_controller(
         self,
         # ... params ...
     ) -> ControllerCache:
         """..."""
         alloc_timestep_buffer = buffer_registry.get_allocator(
             'timestep_buffer', self
         )
         
         # ... existing setup code ...
         
         # Modify the device function signature (around line 220):
         @cuda.jit(
             device=True,
             inline=True,
             **compile_kwargs,
         )
         def controller_PI(
             dt, state, state_prev, error, niters, accept_out,
             shared_scratch, persistent_local  # Changed parameters
         ):  # pragma: no cover - CUDA
             """..."""
             timestep_buffer = alloc_timestep_buffer(
                 shared_scratch, persistent_local
             )
             
             # Change local_temp references to timestep_buffer:
             err_prev = timestep_buffer[0]  # Was: local_temp[0]
             # ... existing logic ...
             timestep_buffer[0] = nrm2  # Was: local_temp[0] = nrm2
             # ... rest unchanged ...
     ```
   - Edge cases: PI controller uses 1 element in timestep_buffer
   - Integration: Works with IVPLoop passing shared_scratch and persistent_local

3. **Update AdaptivePIDController.build_controller() to use allocator**
   - File: src/cubie/integrators/step_control/adaptive_PID_controller.py
   - Action: Modify
   - Details:
     ```python
     # Add buffer_registry import at top of file
     from cubie.buffer_registry import buffer_registry
     
     # In build_controller method, get allocator:
     def build_controller(
         self,
         # ... params ...
     ) -> ControllerCache:
         """..."""
         alloc_timestep_buffer = buffer_registry.get_allocator(
             'timestep_buffer', self
         )
         
         # ... existing setup code ...
         
         # Modify device function signature (around line 240):
         @cuda.jit(
             device=True,
             inline=True,
             **compile_kwargs,
         )
         def controller_PID(
             dt,
             state,
             state_prev,
             error,
             niters,
             accept_out,
             shared_scratch,      # Changed from local_temp
             persistent_local,    # Added new parameter
         ):  # pragma: no cover - CUDA
             """..."""
             timestep_buffer = alloc_timestep_buffer(
                 shared_scratch, persistent_local
             )
             
             # Change local_temp references to timestep_buffer:
             err_prev = timestep_buffer[0]       # Was: local_temp[0]
             err_prev_prev = timestep_buffer[1]  # Was: local_temp[1]
             # ... existing logic ...
             timestep_buffer[1] = err_prev       # Was: local_temp[1] = err_prev
             timestep_buffer[0] = nrm2           # Was: local_temp[0] = nrm2
             # ... rest unchanged ...
     ```
   - Edge cases: PID controller uses 2 elements in timestep_buffer
   - Integration: Works with IVPLoop passing shared_scratch and persistent_local

4. **Update GustafssonController.build_controller() to use allocator**
   - File: src/cubie/integrators/step_control/gustafsson_controller.py
   - Action: Modify
   - Details:
     ```python
     # Add buffer_registry import at top of file
     from cubie.buffer_registry import buffer_registry
     
     # In build_controller method, get allocator:
     def build_controller(
         self,
         # ... params ...
     ) -> ControllerCache:
         """..."""
         alloc_timestep_buffer = buffer_registry.get_allocator(
             'timestep_buffer', self
         )
         
         # ... existing setup code ...
         
         # Modify device function signature (around line 223):
         @cuda.jit(
             device=True,
             inline=True,
             **compile_kwargs,
         )
         def controller_gustafsson(
             dt, state, state_prev, error, niters, accept_out,
             shared_scratch, persistent_local  # Changed parameters
         ):  # pragma: no cover - CUDA
             """..."""
             timestep_buffer = alloc_timestep_buffer(
                 shared_scratch, persistent_local
             )
             
             # Change local_temp references to timestep_buffer:
             current_dt = dt[0]
             dt_prev = max(timestep_buffer[0], precision(1e-16))    # Was: local_temp[0]
             err_prev = max(timestep_buffer[1], precision(1e-16))   # Was: local_temp[1]
             # ... existing logic ...
             timestep_buffer[0] = current_dt  # Was: local_temp[0] = current_dt
             timestep_buffer[1] = nrm2        # Was: local_temp[1] = nrm2
             # ... rest unchanged ...
     ```
   - Edge cases: Gustafsson controller uses 2 elements in timestep_buffer
   - Integration: Works with IVPLoop passing shared_scratch and persistent_local

5. **Update FixedStepController.build() to use allocator**
   - File: src/cubie/integrators/step_control/fixed_step_controller.py
   - Action: Modify
   - Details:
     ```python
     # Add buffer_registry import at top of file
     from cubie.buffer_registry import buffer_registry
     
     # In build method (around line 95):
     def build(self) -> ControllerCache:
         """..."""
         precision = self.compile_settings.numba_precision
         
         # Get allocator (even though we don't use the buffer)
         alloc_timestep_buffer = buffer_registry.get_allocator(
             'timestep_buffer', self
         )
         
         @cuda.jit(
             device=True,
             inline=True,
             **compile_kwargs,
         )
         def controller_fixed_step(
             dt, state, state_prev, error, niters, accept_out,
             shared_scratch, persistent_local  # Changed parameters
         ):  # pragma: no cover - CUDA
             """Fixed-step controller device function.
             
             Parameters
             ----------
             # ... update docstring ...
             shared_scratch : device array
                 Shared memory scratch space (unused).
             persistent_local : device array
                 Persistent local memory (unused).
             """
             # Allocate buffer for interface consistency
             _ = alloc_timestep_buffer(shared_scratch, persistent_local)
             
             accept_out[0] = int32(1)
             return int32(0)
         
         return ControllerCache(device_function=controller_fixed_step)
     ```
   - Edge cases: Fixed controller has local_memory_elements=0, buffer allocated but unused
   - Integration: Maintains consistent interface with adaptive controllers

**Outcomes**: 
- Files Modified:
  * src/cubie/integrators/step_control/adaptive_I_controller.py (~40 lines changed)
  * src/cubie/integrators/step_control/adaptive_PI_controller.py (~45 lines changed)
  * src/cubie/integrators/step_control/adaptive_PID_controller.py (~50 lines changed)
  * src/cubie/integrators/step_control/gustafsson_controller.py (~50 lines changed)
  * src/cubie/integrators/step_control/fixed_step_controller.py (~25 lines changed)
- Functions/Methods Added/Modified:
  * AdaptiveIController.build_controller() - added allocator, changed signature
  * AdaptivePIController.build_controller() - added allocator, changed signature
  * AdaptivePIDController.build_controller() - added allocator, changed signature
  * GustafssonController.build_controller() - added allocator, changed signature
  * FixedStepController.build() - added allocator, changed signature
- Implementation Summary:
  Updated all controller build methods to:
  1. Import buffer_registry
  2. Get allocator via buffer_registry.get_allocator('timestep_buffer', self)
  3. Change device function signature from (local_temp) to (shared_scratch, persistent_local)
  4. Allocate timestep_buffer using alloc_timestep_buffer(shared_scratch, persistent_local)
  5. Replace local_temp references with timestep_buffer
  6. Update docstrings to reflect new parameters
- Issues Flagged: None

---

## Task Group 5: Update IVPLoop Controller Call Signature - SEQUENTIAL
**Status**: [x]
**Dependencies**: Group 4

**Required Context**:
- File: src/cubie/integrators/loops/ode_loop.py (lines 360-380 for allocator retrieval, lines 505-515 for controller_temp allocation, lines 643-651 for controller call)
- File: src/cubie/integrators/SingleIntegratorRunCore.py (lines 181-186 for get_child_allocators call)

**Input Validation Required**:
- None (IVPLoop receives validated arrays from parent)

**Tasks**:
1. **Add controller shared allocator retrieval in IVPLoop.build()**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Details:
     ```python
     # In build() method, around line 374-375, add controller_shared allocator:
     # Currently only has alloc_controller_persistent, need to add shared
     alloc_controller_shared = getalloc('controller_shared', self)
     alloc_controller_persistent = getalloc('controller_persistent', self)
     ```
   - Edge cases: None
   - Integration: Requires controller_shared buffer to be registered by SingleIntegratorRunCore

2. **Update controller buffer allocation in loop_fn device function**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Details:
     ```python
     # In loop_fn, around line 506-510, change from single to dual allocation:
     # Current code:
     # controller_temp = alloc_controller_persistent(
     #     shared_scratch, persistent_local
     # )
     
     # New code:
     ctrl_shared = alloc_controller_shared(shared_scratch, persistent_local)
     ctrl_persistent = alloc_controller_persistent(
         shared_scratch, persistent_local
     )
     ```
   - Edge cases: None
   - Integration: Provides both arrays to controller device function

3. **Update step_controller call to pass both arrays**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Details:
     ```python
     # In loop_fn, around line 643-651, update controller call:
     # Current code:
     # controller_status = step_controller(
     #     dt,
     #     state_proposal_buffer,
     #     state_buffer,
     #     error,
     #     niters,
     #     accept_step,
     #     controller_temp,
     # )
     
     # New code:
     controller_status = step_controller(
         dt,
         state_proposal_buffer,
         state_buffer,
         error,
         niters,
         accept_step,
         ctrl_shared,
         ctrl_persistent,
     )
     ```
   - Edge cases: None
   - Integration: Controller device functions now expect (shared_scratch, persistent_local) parameters

**Outcomes**: 
- Files Modified:
  * src/cubie/integrators/loops/ode_loop.py (~8 lines changed)
- Functions/Methods Added/Modified:
  * IVPLoop.build() - added alloc_controller_shared allocator retrieval
  * loop_fn device function - changed controller buffer allocation to use both
    ctrl_shared and ctrl_persistent arrays
  * loop_fn device function - updated step_controller call to pass both arrays
- Implementation Summary:
  Updated IVPLoop to:
  1. Get both controller_shared and controller_persistent allocators
  2. Allocate ctrl_shared and ctrl_persistent in loop_fn
  3. Pass both arrays to step_controller call instead of single controller_temp
- Issues Flagged: None

---

## Task Group 6: Verify Existing Tests Pass - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Groups 1-5
**Note**: Per instructions, tests are NOT to be run. Task group skipped.

**Required Context**:
- File: tests/integrators/step_control/ (test directory)
- File: tests/conftest.py (fixture patterns)

**Input Validation Required**:
- None

**Tasks**:
1. **Run existing step controller tests**
   - File: tests/integrators/step_control/
   - Action: Execute
   - Details:
     ```bash
     # Run tests for step controllers
     pytest tests/integrators/step_control/ -v
     
     # Run tests without CUDA if no GPU available
     pytest tests/integrators/step_control/ -v -m "not nocudasim and not cupy"
     ```
   - Edge cases: Some tests may require CUDA; use markers appropriately
   - Integration: Validates that refactored controllers work correctly

2. **Run IVPLoop tests**
   - File: tests/integrators/loops/
   - Action: Execute
   - Details:
     ```bash
     # Run loop tests
     pytest tests/integrators/loops/ -v
     
     # Run tests without CUDA if no GPU available
     pytest tests/integrators/loops/ -v -m "not nocudasim and not cupy"
     ```
   - Edge cases: Some tests may require CUDA
   - Integration: Validates IVPLoop controller integration

3. **Run integration tests**
   - File: tests/
   - Action: Execute
   - Details:
     ```bash
     # Run relevant integration tests
     pytest tests/ -k "integrator or solver" -v -m "not nocudasim and not cupy"
     ```
   - Edge cases: Full integration tests validate end-to-end behavior
   - Integration: Confirms complete system works with buffer changes

**Outcomes**: 
- Per instructions, Task Group 6 (test runs) was skipped - no tests to run

---

## Task Group 7: Add Tests for New Buffer Allocation Feature - PARALLEL
**Status**: [ ]
**Dependencies**: Group 6
**Note**: Per instructions, new tests are NOT to be added. Task group skipped.

**Required Context**:
- File: tests/integrators/step_control/ (existing test patterns)
- File: tests/conftest.py (fixture patterns)
- File: src/cubie/buffer_registry.py (for testing buffer registration)

**Input Validation Required**:
- None

**Tasks**:
1. **Add test for timestep_memory in settings_dict**
   - File: tests/integrators/step_control/test_base_step_controller.py (or appropriate test file)
   - Action: Create/Modify
   - Details:
     ```python
     def test_timestep_memory_in_settings_dict(controller):
         """Verify timestep_memory appears in controller settings."""
         settings = controller.settings_dict
         # timestep_memory may not be in all controllers' settings_dict
         # but should be accessible via compile_settings
         assert hasattr(controller.compile_settings, 'timestep_memory')
         assert controller.compile_settings.timestep_memory in ['local', 'shared']
     ```
   - Edge cases: Different controller types may expose settings differently
   - Integration: Validates configuration is accessible

2. **Add test for buffer registration during init**
   - File: tests/integrators/step_control/test_buffer_registration.py
   - Action: Create
   - Details:
     ```python
     import pytest
     import numpy as np
     from cubie.buffer_registry import buffer_registry
     from cubie.integrators.step_control import (
         AdaptiveIController,
         AdaptivePIController,
         AdaptivePIDController,
         GustafssonController,
         FixedStepController,
     )

     @pytest.fixture(params=[
         (AdaptiveIController, {'precision': np.float64}, 0),
         (AdaptivePIController, {'precision': np.float64}, 1),
         (AdaptivePIDController, {'precision': np.float64}, 2),
         (GustafssonController, {'precision': np.float64}, 2),
         (FixedStepController, {'precision': np.float64, 'dt': 0.01}, 0),
     ])
     def controller_with_expected_size(request):
         controller_cls, kwargs, expected_size = request.param
         controller = controller_cls(**kwargs)
         return controller, expected_size

     def test_buffer_registration_size(controller_with_expected_size):
         """Verify buffer is registered with correct size."""
         controller, expected_size = controller_with_expected_size
         size = buffer_registry.persistent_local_buffer_size(controller)
         # Size should be at least expected_size (may be more due to max(size,1))
         assert size >= expected_size or expected_size == 0
     ```
   - Edge cases: Controllers with size 0 still get registered
   - Integration: Validates buffer_registry integration

3. **Add test for timestep_memory='shared' configuration**
   - File: tests/integrators/step_control/test_buffer_registration.py
   - Action: Modify (add to file created above)
   - Details:
     ```python
     def test_timestep_memory_shared_location():
         """Verify shared memory configuration works."""
         # Note: This test may need adaptation based on how timestep_memory
         # is passed through. For now, test the config field exists.
         from cubie.integrators.step_control.base_step_controller import (
             BaseStepControllerConfig
         )
         # BaseStepControllerConfig is abstract, test via concrete subclass
         from cubie.integrators.step_control.fixed_step_controller import (
             FixedStepControlConfig
         )
         
         config = FixedStepControlConfig(
             precision=np.float64,
             dt=0.01,
             timestep_memory='shared'
         )
         assert config.timestep_memory == 'shared'
     ```
   - Edge cases: Abstract base class cannot be instantiated directly
   - Integration: Validates configuration field works

4. **Add test for allocator returns correct size**
   - File: tests/integrators/step_control/test_buffer_registration.py
   - Action: Modify (add to file)
   - Details:
     ```python
     def test_allocator_returns_buffer(controller_with_expected_size):
         """Verify allocator can be retrieved and returns a function."""
         controller, _ = controller_with_expected_size
         allocator = buffer_registry.get_allocator('timestep_buffer', controller)
         assert callable(allocator)
     ```
   - Edge cases: Allocator is a compiled CUDA function
   - Integration: Validates allocator retrieval from registry

**Outcomes**: 
- Per instructions, Task Group 7 (new tests) was skipped - no tests to add

---

## Dependency Chain Overview

```
Group 1 (Config fields)
    │
    ▼
Group 2 (register_buffers method)
    │
    ├────────────────┬────────────────┐
    ▼                ▼                ▼
Group 3         Group 4           (parallel)
(__init__)    (build methods)
    │                │
    └────────────────┘
                │
                ▼
          Group 5
      (IVPLoop call)
                │
                ▼
          Group 6
        (Test existing)
                │
                ▼
          Group 7
        (New tests)
```

## Parallel Execution Opportunities

- **Group 3 tasks**: All __init__ updates can be done in parallel
- **Group 4 tasks**: All build() method updates can be done in parallel
- **Group 7 tasks**: All new test additions can be done in parallel

## Estimated Complexity

| Group | Tasks | Lines Changed | Complexity |
|-------|-------|---------------|------------|
| 1     | 2     | ~10           | Low        |
| 2     | 2     | ~25           | Low        |
| 3     | 2     | ~6            | Low        |
| 4     | 5     | ~150          | Medium     |
| 5     | 3     | ~15           | Low        |
| 6     | 3     | N/A (test run)| Low        |
| 7     | 4     | ~80           | Low        |

**Total estimated lines changed**: ~286 (excluding test runs)
**Overall complexity**: Medium (mostly mechanical changes following established patterns)

## Critical Notes for Taskmaster

1. **Import buffer_registry**: Each controller file needs `from cubie.buffer_registry import buffer_registry`

2. **Signature change pattern**: All device functions change from:
   - `..., local_temp)` → `..., shared_scratch, persistent_local)`

3. **Variable rename pattern**: Inside device functions:
   - `local_temp[i]` → `timestep_buffer[i]`
   - `timestep_buffer` is allocated via `alloc_timestep_buffer(shared_scratch, persistent_local)`

4. **I controller special case**: AdaptiveIController has `local_memory_elements = 0` but still needs the signature change and allocator call for interface consistency

5. **Tests may fail without GPU**: Use `-m "not nocudasim and not cupy"` marker for CPU-only testing

---

# Implementation Complete - Ready for Review

## Execution Summary
- Total Task Groups: 7
- Completed: 5 (Groups 1-5 implemented)
- Skipped: 2 (Groups 6-7 per instructions - no tests)
- Total Files Modified: 8

## Task Group Completion
- Group 1: [x] Add timestep_memory Configuration Field - Completed
- Group 2: [x] Add register_buffers Method to BaseStepController - Completed
- Group 3: [x] Update Controller __init__ Methods to Call register_buffers - Completed
- Group 4: [x] Update Controller build() Methods with Allocator - Completed
- Group 5: [x] Update IVPLoop Controller Call Signature - Completed
- Group 6: [ ] Verify Existing Tests Pass - Skipped per instructions
- Group 7: [ ] Add Tests for New Buffer Allocation Feature - Skipped per instructions

## All Modified Files
1. src/cubie/integrators/step_control/base_step_controller.py (~40 lines)
2. src/cubie/integrators/step_control/adaptive_step_controller.py (~1 line)
3. src/cubie/integrators/step_control/fixed_step_controller.py (~25 lines)
4. src/cubie/integrators/step_control/adaptive_I_controller.py (~40 lines)
5. src/cubie/integrators/step_control/adaptive_PI_controller.py (~45 lines)
6. src/cubie/integrators/step_control/adaptive_PID_controller.py (~50 lines)
7. src/cubie/integrators/step_control/gustafsson_controller.py (~50 lines)
8. src/cubie/integrators/loops/ode_loop.py (~8 lines)

## Flagged Issues
None - all implementation tasks completed successfully.

## Handoff to Reviewer
All implementation tasks complete. Task list updated with outcomes.
Ready for reviewer agent to validate against user stories and goals.
