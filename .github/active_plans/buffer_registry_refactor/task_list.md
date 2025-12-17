# Implementation Task List
# Feature: Buffer Registry Refactor
# Plan Reference: .github/active_plans/buffer_registry_refactor/agent_plan.md

---

## Task Group 1: Core Buffer Registry Infrastructure - SEQUENTIAL
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: src/cubie/BufferSettings.py (entire file - understand existing patterns)
- File: src/cubie/CUDAFactory.py (lines 441-550 - base class structure)
- File: src/cubie/_utils.py (lines 24-61 - PrecisionDType and validators)
- File: src/cubie/cuda_simsafe.py (lines 19-34 - compile_kwargs)

**Input Validation Required**:
- BufferLocation: Validate enum membership (SHARED, LOCAL, LOCAL_PERSISTENT)
- CUDABuffer.name: str, non-empty
- CUDABuffer.location: BufferLocation enum instance
- CUDABuffer.size: int >= 0 OR callable returning int >= 0
- CUDABuffer.precision: member of ALLOWED_PRECISIONS
- CUDABuffer.enabled: bool
- BufferRegistry.precision: member of ALLOWED_PRECISIONS

**Tasks**:
1. **Create buffer_registry.py module with BufferLocation enum**
   - File: src/cubie/buffer_registry.py
   - Action: Create
   - Details:
     ```python
     from enum import Enum

     class BufferLocation(Enum):
         """Memory location for CUDA buffer allocation.
         
         Attributes
         ----------
         SHARED : str
             Per-block shared memory (fast, limited ~48KB).
         LOCAL : str
             Per-thread local arrays (fresh each call).
         LOCAL_PERSISTENT : str
             Per-thread local memory persisting across calls.
         """
         SHARED = "shared"
         LOCAL = "local"
         LOCAL_PERSISTENT = "local_persistent"
     ```
   - Edge cases: Ensure string values match existing BufferSettings patterns
   - Integration: Will replace string literals in location attributes

2. **Add CUDABuffer attrs class**
   - File: src/cubie/buffer_registry.py
   - Action: Modify (append to file)
   - Details:
     ```python
     @attrs.define
     class CUDABuffer:
         """Buffer configuration for CUDA memory allocation.
         
         Attributes
         ----------
         name : str
             Unique identifier for the buffer.
         location : BufferLocation
             Memory location (SHARED, LOCAL, or LOCAL_PERSISTENT).
         size : int or Callable[[], int]
             Size in elements; callable for deferred evaluation.
         precision : PrecisionDType
             NumPy dtype for buffer elements.
         enabled : bool
             Whether buffer is active for allocation.
         shared_slice : slice
             Assigned by registry for shared memory layout.
         local_slice : slice
             Assigned by registry for local memory layout.
         persistent_slice : slice
             Assigned by registry for persistent local layout.
         """
         name: str = attrs.field(validator=validators.instance_of(str))
         location: BufferLocation = attrs.field(
             validator=validators.instance_of(BufferLocation)
         )
         size: Union[int, Callable[[], int]] = attrs.field()
         precision: PrecisionDType = attrs.field(
             converter=precision_converter,
             validator=precision_validator
         )
         enabled: bool = attrs.field(default=True)
         shared_slice: slice = attrs.field(
             factory=lambda: slice(0, 0), eq=False
         )
         local_slice: slice = attrs.field(
             factory=lambda: slice(0, 0), eq=False
         )
         persistent_slice: slice = attrs.field(
             factory=lambda: slice(0, 0), eq=False
         )

         @property
         def resolved_size(self) -> int:
             # Return size if int, else call callable
             pass

         @property
         def nonzero_size(self) -> int:
             # max(resolved_size, 1) for cuda.local.array
             pass

         @property
         def shared_size(self) -> int:
             # size if SHARED and enabled, else 0
             pass

         @property
         def local_size(self) -> int:
             # nonzero_size if LOCAL and enabled, else 0
             pass

         @property
         def persistent_size(self) -> int:
             # size if LOCAL_PERSISTENT and enabled, else 0
             pass

         def allocate(self) -> Callable:
             # Return compiled CUDA device function
             pass
     ```
   - Edge cases: 
     - Size callable must be evaluated lazily
     - Disabled buffers return 0 for all size properties
     - nonzero_size returns max(size, 1) for cuda.local.array compatibility
   - Integration: Used by factories to define buffer requirements

3. **Add BufferRegistry attrs class**
   - File: src/cubie/buffer_registry.py
   - Action: Modify (append to file)
   - Details:
     ```python
     @attrs.define
     class BufferRegistry:
         """Registry aggregating CUDABuffer instances.
         
         Manages collective memory layouts and provides aggregate
         memory requirements for CUDA kernel compilation.
         
         Attributes
         ----------
         buffers : Dict[str, CUDABuffer]
             Registry of buffers keyed by name.
         precision : PrecisionDType
             Default precision for new buffers.
         """
         buffers: Dict[str, CUDABuffer] = attrs.field(factory=dict)
         precision: PrecisionDType = attrs.field(
             default=np.float64,
             converter=precision_converter,
             validator=precision_validator
         )
         _shared_layout_generated: bool = attrs.field(
             default=False, init=False, eq=False
         )
         _persistent_layout_generated: bool = attrs.field(
             default=False, init=False, eq=False
         )

         @property
         def shared_memory_elements(self) -> int:
             # Sum of all buffers' shared_size
             pass

         @property
         def local_memory_elements(self) -> int:
             # Sum of all buffers' local_size
             pass

         @property
         def persistent_local_elements(self) -> int:
             # Sum of all buffers' persistent_size
             pass

         def register(self, buffer: CUDABuffer) -> None:
             # Add buffer to registry, validate unique name
             pass

         def register_buffer(
             self,
             name: str,
             location: BufferLocation,
             size: Union[int, Callable[[], int]],
             enabled: bool = True,
         ) -> CUDABuffer:
             # Convenience method to create and register buffer
             pass

         def get(self, name: str) -> CUDABuffer:
             # Retrieve buffer by name
             pass

         def generate_shared_layout(self) -> int:
             # Assign shared_slice to each buffer, return total
             pass

         def generate_persistent_layout(self) -> int:
             # Assign persistent_slice to each buffer, return total
             pass

         def merge(
             self, 
             child_registry: "BufferRegistry", 
             prefix: str = ""
         ) -> None:
             # Incorporate child's buffers with prefixed names
             pass

         def get_allocators(self) -> Dict[str, Callable]:
             # Return dict of allocation device functions
             pass
     ```
   - Edge cases:
     - Name collisions on register() should raise KeyError
     - Merge with prefix handles nested component buffers
     - Layout generation is idempotent
   - Integration: Owned by CUDAFactory subclasses

**Outcomes**: 
[ ] BufferLocation enum created with SHARED, LOCAL, LOCAL_PERSISTENT values
[ ] CUDABuffer class with all properties and allocate() method
[ ] BufferRegistry class with all methods for buffer management

---

## Task Group 2: CUDABuffer.allocate() Implementation - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/buffer_registry.py (CUDABuffer class from Group 1)
- File: src/cubie/integrators/loops/ode_loop.py (lines 954-1046 - existing allocation pattern)
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py (lines 239-325 - allocation pattern)
- File: src/cubie/cuda_simsafe.py (lines 19-34 - compile_kwargs)

**Input Validation Required**:
- None (method operates on validated CUDABuffer instance)

**Tasks**:
1. **Implement CUDABuffer.allocate() method**
   - File: src/cubie/buffer_registry.py
   - Action: Modify (implement allocate method in CUDABuffer)
   - Details:
     ```python
     def allocate(self) -> Callable:
         """Return compiled CUDA device function for buffer allocation.
         
         The returned function branches on compile-time constants,
         yielding a single CUDA instruction at runtime.
         
         Returns
         -------
         Callable
             CUDA device function with signature:
             allocate_buffer(shared_parent, persistent_parent) -> array
         """
         # Capture compile-time constants in closure
         _is_shared = self.location == BufferLocation.SHARED
         _is_persistent = self.location == BufferLocation.LOCAL_PERSISTENT
         _shared_slice = self.shared_slice
         _persistent_slice = self.persistent_slice
         _local_size = self.nonzero_size
         _precision = from_dtype(self.precision)
         _enabled = self.enabled

         @cuda.jit(device=True, inline=True, **compile_kwargs)
         def allocate_buffer(shared_parent, persistent_parent):
             if not _enabled:
                 # Disabled buffer - return minimal local array
                 return cuda.local.array(1, _precision)
             if _is_shared:
                 return shared_parent[_shared_slice]
             elif _is_persistent:
                 return persistent_parent[_persistent_slice]
             else:
                 return cuda.local.array(_local_size, _precision)

         return allocate_buffer
     ```
   - Edge cases:
     - Disabled buffers return minimal 1-element local array
     - Empty slices (0,0) handled gracefully
     - Must use from_dtype for Numba precision type
   - Integration: Called during factory build() to get allocation functions

**Outcomes**: 
[ ] allocate() returns compiled CUDA device function
[ ] Compile-time constants baked into closure
[ ] All three location types handled correctly

---

## Task Group 3: Core Unit Tests for Buffer Registry - PARALLEL
**Status**: [ ]
**Dependencies**: Task Groups 1, 2

**Required Context**:
- File: src/cubie/buffer_registry.py (entire file from Groups 1-2)
- File: tests/test_buffer_settings.py (existing test patterns)
- File: tests/conftest.py (fixture patterns)

**Input Validation Required**:
- None (tests validate the implementation)

**Tasks**:
1. **Create test_buffer_registry.py with BufferLocation tests**
   - File: tests/test_buffer_registry.py
   - Action: Create
   - Details:
     ```python
     """Tests for buffer registry infrastructure."""
     import pytest
     import numpy as np
     from numba import cuda

     from cubie.buffer_registry import (
         BufferLocation,
         CUDABuffer,
         BufferRegistry,
     )


     class TestBufferLocation:
         """Tests for BufferLocation enum."""

         def test_shared_value(self):
             assert BufferLocation.SHARED.value == "shared"

         def test_local_value(self):
             assert BufferLocation.LOCAL.value == "local"

         def test_local_persistent_value(self):
             assert BufferLocation.LOCAL_PERSISTENT.value == "local_persistent"
     ```
   - Edge cases: Ensure enum values match string literals in existing code
   - Integration: Basic enum validation

2. **Add CUDABuffer unit tests**
   - File: tests/test_buffer_registry.py
   - Action: Modify (append)
   - Details:
     ```python
     class TestCUDABuffer:
         """Tests for CUDABuffer class."""

         def test_resolved_size_with_int(self):
             buffer = CUDABuffer(
                 name="test",
                 location=BufferLocation.LOCAL,
                 size=10,
                 precision=np.float64,
             )
             assert buffer.resolved_size == 10

         def test_resolved_size_with_callable(self):
             buffer = CUDABuffer(
                 name="test",
                 location=BufferLocation.LOCAL,
                 size=lambda: 20,
                 precision=np.float64,
             )
             assert buffer.resolved_size == 20

         def test_nonzero_size_returns_one_for_zero(self):
             buffer = CUDABuffer(
                 name="test",
                 location=BufferLocation.LOCAL,
                 size=0,
                 precision=np.float64,
             )
             assert buffer.nonzero_size == 1

         def test_shared_size_when_shared(self):
             buffer = CUDABuffer(
                 name="test",
                 location=BufferLocation.SHARED,
                 size=10,
                 precision=np.float64,
             )
             assert buffer.shared_size == 10

         def test_shared_size_when_local(self):
             buffer = CUDABuffer(
                 name="test",
                 location=BufferLocation.LOCAL,
                 size=10,
                 precision=np.float64,
             )
             assert buffer.shared_size == 0

         def test_disabled_buffer_returns_zero_sizes(self):
             buffer = CUDABuffer(
                 name="test",
                 location=BufferLocation.SHARED,
                 size=10,
                 precision=np.float64,
                 enabled=False,
             )
             assert buffer.shared_size == 0
             assert buffer.local_size == 0
             assert buffer.persistent_size == 0
     ```
   - Edge cases: Zero size, disabled, callable size
   - Integration: Validates CUDABuffer properties

3. **Add BufferRegistry unit tests**
   - File: tests/test_buffer_registry.py
   - Action: Modify (append)
   - Details:
     ```python
     class TestBufferRegistry:
         """Tests for BufferRegistry class."""

         def test_register_buffer(self):
             registry = BufferRegistry(precision=np.float64)
             buffer = registry.register_buffer(
                 name="state",
                 location=BufferLocation.SHARED,
                 size=10,
             )
             assert "state" in registry.buffers
             assert buffer.name == "state"

         def test_register_duplicate_name_raises(self):
             registry = BufferRegistry(precision=np.float64)
             registry.register_buffer("test", BufferLocation.LOCAL, 5)
             with pytest.raises(KeyError):
                 registry.register_buffer("test", BufferLocation.LOCAL, 10)

         def test_shared_memory_elements(self):
             registry = BufferRegistry(precision=np.float64)
             registry.register_buffer("a", BufferLocation.SHARED, 10)
             registry.register_buffer("b", BufferLocation.SHARED, 5)
             registry.register_buffer("c", BufferLocation.LOCAL, 20)
             assert registry.shared_memory_elements == 15

         def test_generate_shared_layout(self):
             registry = BufferRegistry(precision=np.float64)
             registry.register_buffer("a", BufferLocation.SHARED, 10)
             registry.register_buffer("b", BufferLocation.SHARED, 5)
             total = registry.generate_shared_layout()
             assert total == 15
             assert registry.get("a").shared_slice == slice(0, 10)
             assert registry.get("b").shared_slice == slice(10, 15)

         def test_merge_prefixes_names(self):
             parent = BufferRegistry(precision=np.float64)
             parent.register_buffer("state", BufferLocation.SHARED, 10)

             child = BufferRegistry(precision=np.float64)
             child.register_buffer("temp", BufferLocation.SHARED, 5)

             parent.merge(child, prefix="child")
             assert "child.temp" in parent.buffers
             assert parent.shared_memory_elements == 15

         def test_get_returns_buffer(self):
             registry = BufferRegistry(precision=np.float64)
             registry.register_buffer("test", BufferLocation.LOCAL, 10)
             buffer = registry.get("test")
             assert buffer.name == "test"

         def test_get_missing_raises(self):
             registry = BufferRegistry(precision=np.float64)
             with pytest.raises(KeyError):
                 registry.get("nonexistent")
     ```
   - Edge cases: Duplicate names, empty registry, merge collisions
   - Integration: Validates BufferRegistry functionality

**Outcomes**: 
[ ] BufferLocation enum tests pass
[ ] CUDABuffer property tests pass
[ ] BufferRegistry method tests pass

---

## Task Group 4: CUDAFactory Integration - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Groups 1, 2

**Required Context**:
- File: src/cubie/CUDAFactory.py (lines 441-564 - CUDAFactory class)
- File: src/cubie/buffer_registry.py (entire file from Groups 1-2)

**Input Validation Required**:
- None (methods operate on validated objects)

**Tasks**:
1. **Add buffer_registry property to CUDAFactory**
   - File: src/cubie/CUDAFactory.py
   - Action: Modify
   - Details:
     - Add import: `from cubie.buffer_registry import BufferRegistry, CUDABuffer, BufferLocation`
     - Add to __init__: `self._buffer_registry = None`
     - Add property:
       ```python
       @property
       def buffer_registry(self) -> BufferRegistry:
           """Return the buffer registry, creating if needed.
           
           Returns
           -------
           BufferRegistry
               Registry for managing CUDA buffer allocations.
           """
           if self._buffer_registry is None:
               # Get precision from compile_settings if available
               precision = np.float64
               if self._compile_settings is not None:
                   if hasattr(self._compile_settings, 'precision'):
                       precision = self._compile_settings.precision
               self._buffer_registry = BufferRegistry(precision=precision)
           return self._buffer_registry
       ```
   - Edge cases: Handle case where compile_settings not yet set
   - Integration: Lazy initialization maintains backward compatibility

2. **Add register_buffer helper method to CUDAFactory**
   - File: src/cubie/CUDAFactory.py
   - Action: Modify (add method)
   - Details:
     ```python
     def register_buffer(
         self,
         name: str,
         location: BufferLocation,
         size: Union[int, Callable[[], int]],
         enabled: bool = True,
     ) -> CUDABuffer:
         """Register a buffer with the factory's registry.
         
         Parameters
         ----------
         name
             Unique identifier for the buffer.
         location
             Memory location (SHARED, LOCAL, or LOCAL_PERSISTENT).
         size
             Size in elements; callable for deferred evaluation.
         enabled
             Whether buffer is active for allocation.
             
         Returns
         -------
         CUDABuffer
             The registered buffer instance.
         """
         return self.buffer_registry.register_buffer(
             name=name,
             location=location,
             size=size,
             enabled=enabled,
         )
     ```
   - Edge cases: None (delegates to registry)
   - Integration: Convenience wrapper for factory subclasses

3. **Add merge_child_buffers helper method to CUDAFactory**
   - File: src/cubie/CUDAFactory.py
   - Action: Modify (add method)
   - Details:
     ```python
     def merge_child_buffers(
         self,
         child: "CUDAFactory",
         prefix: str = "",
     ) -> None:
         """Merge a child factory's buffer registry.
         
         Parameters
         ----------
         child
             Child CUDAFactory whose buffers to incorporate.
         prefix
             Prefix to add to child buffer names.
         """
         if child._buffer_registry is not None:
             self.buffer_registry.merge(child._buffer_registry, prefix=prefix)
     ```
   - Edge cases: Child with no registry (skip merge)
   - Integration: Enables nested buffer composition (DIRK → Newton → Linear)

**Outcomes**: 
[ ] buffer_registry property added to CUDAFactory
[ ] register_buffer helper method added
[ ] merge_child_buffers helper method added

---

## Task Group 5: CUDAFactory Integration Tests - PARALLEL
**Status**: [ ]
**Dependencies**: Task Group 4

**Required Context**:
- File: src/cubie/CUDAFactory.py (modified in Group 4)
- File: tests/test_CUDAFactory.py (existing test patterns)

**Input Validation Required**:
- None (tests validate the implementation)

**Tasks**:
1. **Add buffer_registry property tests**
   - File: tests/test_CUDAFactory.py
   - Action: Modify (append tests)
   - Details:
     ```python
     class TestCUDAFactoryBufferRegistry:
         """Tests for CUDAFactory buffer registry integration."""

         def test_buffer_registry_lazy_initialization(self):
             """buffer_registry should create registry on first access."""
             # Use a concrete CUDAFactory subclass from fixtures
             factory = create_test_factory()
             assert factory._buffer_registry is None
             registry = factory.buffer_registry
             assert registry is not None
             assert factory._buffer_registry is registry

         def test_buffer_registry_uses_compile_settings_precision(self):
             """Registry should inherit precision from compile_settings."""
             factory = create_test_factory(precision=np.float32)
             registry = factory.buffer_registry
             assert registry.precision == np.float32

         def test_register_buffer_adds_to_registry(self):
             """register_buffer should add buffer to factory's registry."""
             factory = create_test_factory()
             buffer = factory.register_buffer(
                 name="test",
                 location=BufferLocation.SHARED,
                 size=10,
             )
             assert "test" in factory.buffer_registry.buffers

         def test_merge_child_buffers_incorporates_child(self):
             """merge_child_buffers should add child buffers with prefix."""
             parent = create_test_factory()
             child = create_test_factory()
             child.register_buffer("temp", BufferLocation.SHARED, 5)

             parent.merge_child_buffers(child, prefix="child")
             assert "child.temp" in parent.buffer_registry.buffers
     ```
   - Edge cases: Factory without compile_settings
   - Integration: Validates CUDAFactory integration

**Outcomes**: 
[ ] buffer_registry property tests pass
[ ] register_buffer tests pass
[ ] merge_child_buffers tests pass

---

## Task Group 6: LinearSolver Migration - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Groups 1-4

**Required Context**:
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py (entire file)
- File: src/cubie/buffer_registry.py (entire file)

**Input Validation Required**:
- Preserve existing validation from LinearSolverBufferSettings

**Tasks**:
1. **Add BufferRegistry to linear_solver_factory**
   - File: src/cubie/integrators/matrix_free_solvers/linear_solver.py
   - Action: Modify
   - Details:
     - Add import: `from cubie.buffer_registry import BufferRegistry, BufferLocation, CUDABuffer`
     - Create registry in factory function:
       ```python
       # Create buffer registry for linear solver
       registry = BufferRegistry(precision=precision)
       
       # Register buffers
       precond_vec_loc = (BufferLocation.SHARED 
                          if buffer_settings.use_shared_preconditioned_vec 
                          else BufferLocation.LOCAL)
       registry.register_buffer("preconditioned_vec", precond_vec_loc, n)
       
       temp_loc = (BufferLocation.SHARED 
                   if buffer_settings.use_shared_temp 
                   else BufferLocation.LOCAL)
       registry.register_buffer("temp", temp_loc, n)
       ```
     - Extract slices from registry after generate_shared_layout()
   - Edge cases: Maintain backward compatibility with buffer_settings parameter
   - Integration: First algorithm to use new system

2. **Add shared_memory_elements property to factory return**
   - File: src/cubie/integrators/matrix_free_solvers/linear_solver.py
   - Action: Modify
   - Details:
     - Attach registry to returned device function as attribute:
       ```python
       linear_solver.buffer_registry = registry
       linear_solver.shared_memory_elements = registry.shared_memory_elements
       return linear_solver
       ```
   - Edge cases: Ensure attribute access doesn't break existing code
   - Integration: Enables parent factories to query memory requirements

**Outcomes**: 
[ ] linear_solver_factory uses BufferRegistry internally
[ ] Existing tests continue to pass
[ ] Memory totals match old and new systems

---

## Task Group 7: Newton-Krylov Migration - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Groups 1-4, 6

**Required Context**:
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py (entire file)
- File: src/cubie/buffer_registry.py (entire file)
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py (modified in Group 6)

**Input Validation Required**:
- Preserve existing validation from NewtonBufferSettings

**Tasks**:
1. **Add BufferRegistry to newton_krylov_solver_factory**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify
   - Details:
     - Add import: `from cubie.buffer_registry import BufferRegistry, BufferLocation`
     - Create registry:
       ```python
       registry = BufferRegistry(precision=precision)
       
       # Register Newton's own buffers
       delta_loc = (BufferLocation.SHARED 
                    if buffer_settings.use_shared_delta 
                    else BufferLocation.LOCAL)
       registry.register_buffer("delta", delta_loc, n)
       
       residual_loc = (BufferLocation.SHARED 
                       if buffer_settings.use_shared_residual 
                       else BufferLocation.LOCAL)
       registry.register_buffer("residual", residual_loc, n)
       
       residual_temp_loc = (BufferLocation.SHARED 
                            if buffer_settings.use_shared_residual_temp 
                            else BufferLocation.LOCAL)
       registry.register_buffer("residual_temp", residual_temp_loc, n)
       
       stage_base_bt_loc = (BufferLocation.SHARED 
                            if buffer_settings.use_shared_stage_base_bt 
                            else BufferLocation.LOCAL)
       registry.register_buffer("stage_base_bt", stage_base_bt_loc, n)
       ```
     - Merge linear solver's registry:
       ```python
       # linear_solver already has buffer_registry attached
       registry.merge(linear_solver.buffer_registry, prefix="linear")
       ```
   - Edge cases: Handle nested buffer composition
   - Integration: Demonstrates merge pattern for nested solvers

2. **Attach registry to returned device function**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify
   - Details:
     ```python
     newton_krylov_solver.buffer_registry = registry
     newton_krylov_solver.shared_memory_elements = registry.shared_memory_elements
     return newton_krylov_solver
     ```
   - Edge cases: Ensure all nested buffers included in totals
   - Integration: Enables DIRK to query total memory requirements

**Outcomes**: 
[ ] newton_krylov_solver_factory uses BufferRegistry
[ ] Nested linear solver buffers properly merged
[ ] Memory totals match old and new systems

---

## Task Group 8: ERK Algorithm Migration - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Groups 1-4

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_erk.py (lines 1-350)
- File: src/cubie/buffer_registry.py (entire file)

**Input Validation Required**:
- Preserve existing validation from ERKBufferSettings

**Tasks**:
1. **Add BufferRegistry usage to ERKStep**
   - File: src/cubie/integrators/algorithms/generic_erk.py
   - Action: Modify
   - Details:
     - Add import: `from cubie.buffer_registry import BufferRegistry, BufferLocation`
     - In __init__, after buffer_settings creation:
       ```python
       # Create buffer registry
       self._buffer_registry = BufferRegistry(precision=precision)
       
       # Register stage_rhs buffer
       rhs_loc = (BufferLocation.SHARED 
                  if buffer_settings.use_shared_stage_rhs 
                  else BufferLocation.LOCAL)
       self._buffer_registry.register_buffer("stage_rhs", rhs_loc, n)
       
       # Register stage_accumulator buffer
       acc_loc = (BufferLocation.SHARED 
                  if buffer_settings.use_shared_stage_accumulator 
                  else BufferLocation.LOCAL)
       accumulator_size = lambda: max((self.stage_count - 1) * n, 0)
       self._buffer_registry.register_buffer(
           "stage_accumulator", acc_loc, accumulator_size
       )
       
       # Register stage_cache (persistent for FSAL)
       cache_enabled = self.first_same_as_last
       cache_loc = BufferLocation.LOCAL_PERSISTENT
       self._buffer_registry.register_buffer(
           "stage_cache", cache_loc, n, enabled=cache_enabled
       )
       ```
   - Edge cases: 
     - stage_cache only needed for FSAL tableaus
     - Accumulator size depends on stage_count
   - Integration: Explicit algorithm using new buffer system

**Outcomes**: 
[ ] ERKStep uses BufferRegistry
[ ] Conditional FSAL buffer handled correctly
[ ] Memory totals match old system

---

## Task Group 9: DIRK Algorithm Migration - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Groups 1-4, 6, 7

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_dirk.py (entire file)
- File: src/cubie/buffer_registry.py (entire file)
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py (modified in Group 7)

**Input Validation Required**:
- Preserve existing validation from DIRKBufferSettings

**Tasks**:
1. **Add BufferRegistry usage to DIRKStep**
   - File: src/cubie/integrators/algorithms/generic_dirk.py
   - Action: Modify
   - Details:
     - Add import: `from cubie.buffer_registry import BufferRegistry, BufferLocation`
     - In __init__, after creating newton solver:
       ```python
       # Create buffer registry
       self._buffer_registry = BufferRegistry(precision=precision)
       
       # Register DIRK's own buffers
       increment_loc = (BufferLocation.SHARED 
                        if buffer_settings.use_shared_stage_increment 
                        else BufferLocation.LOCAL)
       self._buffer_registry.register_buffer("stage_increment", increment_loc, n)
       
       base_loc = (BufferLocation.SHARED 
                   if buffer_settings.use_shared_stage_base 
                   else BufferLocation.LOCAL)
       self._buffer_registry.register_buffer("stage_base", base_loc, n)
       
       acc_loc = (BufferLocation.SHARED 
                  if buffer_settings.use_shared_accumulator 
                  else BufferLocation.LOCAL)
       accumulator_size = lambda: max((self.stage_count - 1) * n, 0)
       self._buffer_registry.register_buffer(
           "accumulator", acc_loc, accumulator_size
       )
       
       # Merge Newton solver's registry
       self._buffer_registry.merge(
           newton_solver.buffer_registry, prefix="newton"
       )
       ```
   - Edge cases:
     - Newton solver brings linear solver buffers
     - FSAL increment/rhs caches for stiff-accurate tableaus
   - Integration: Most complex algorithm - tests full nesting

**Outcomes**: 
[ ] DIRKStep uses BufferRegistry
[ ] Nested Newton → Linear solver buffers merged
[ ] Memory totals match old system

---

## Task Group 10: FIRK and Rosenbrock Migration - PARALLEL
**Status**: [ ]
**Dependencies**: Task Groups 1-4, 6, 7

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_firk.py (buffer settings section)
- File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py (buffer settings section)
- File: src/cubie/buffer_registry.py (entire file)

**Input Validation Required**:
- Preserve existing validation from respective BufferSettings classes

**Tasks**:
1. **Add BufferRegistry usage to FIRKStep**
   - File: src/cubie/integrators/algorithms/generic_firk.py
   - Action: Modify
   - Details: Similar pattern to DIRKStep (nested solver integration)
   - Edge cases: FIRK has different stage structure than DIRK
   - Integration: Validates pattern works for all implicit methods

2. **Add BufferRegistry usage to RosenbrockStep**
   - File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py
   - Action: Modify
   - Details: Similar pattern to DIRKStep
   - Edge cases: Rosenbrock-W methods have specific buffer requirements
   - Integration: Completes implicit algorithm migration

**Outcomes**: 
[ ] FIRKStep uses BufferRegistry
[ ] RosenbrockStep uses BufferRegistry
[ ] All implicit algorithms migrated

---

## Task Group 11: Loop Buffer Migration - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Groups 1-4, 8, 9, 10

**Required Context**:
- File: src/cubie/integrators/loops/ode_loop.py (lines 1-600)
- File: src/cubie/integrators/loops/ode_loop_config.py (entire file)
- File: src/cubie/buffer_registry.py (entire file)

**Input Validation Required**:
- Preserve existing validation from LoopBufferSettings

**Tasks**:
1. **Add BufferRegistry usage to IVPLoop**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Details:
     - Add import: `from cubie.buffer_registry import BufferRegistry, BufferLocation`
     - In __init__ or build():
       ```python
       registry = BufferRegistry(precision=precision)
       
       # Register all loop buffers
       state_loc = (BufferLocation.SHARED 
                    if buffer_settings.use_shared_state 
                    else BufferLocation.LOCAL)
       registry.register_buffer("state", state_loc, n_states)
       
       proposal_loc = (BufferLocation.SHARED 
                       if buffer_settings.use_shared_state_proposal 
                       else BufferLocation.LOCAL)
       registry.register_buffer("proposed_state", proposal_loc, n_states)
       
       # ... register all 12+ loop buffers
       
       # Merge algorithm's registry
       registry.merge(algorithm.buffer_registry, prefix="algorithm")
       ```
   - Edge cases:
     - Most buffers in LoopBufferSettings (12+ location attributes)
     - Must merge child algorithm buffers
   - Integration: Top-level loop owns complete buffer hierarchy

2. **Update ODELoopConfig to use BufferRegistry**
   - File: src/cubie/integrators/loops/ode_loop_config.py
   - Action: Modify
   - Details:
     - Add buffer_registry reference to config
     - Update shared_buffer_indices to use registry slices
   - Edge cases: Maintain backward compatibility with existing configs
   - Integration: Config carries registry through compilation

**Outcomes**: 
[ ] IVPLoop uses BufferRegistry
[ ] All nested algorithm buffers properly merged
[ ] Complete buffer hierarchy managed by single registry

---

## Task Group 12: Migration Integration Tests - PARALLEL
**Status**: [ ]
**Dependencies**: Task Groups 6-11

**Required Context**:
- File: tests/integrators/algorithms/ (existing algorithm tests)
- File: tests/integrators/loops/ (existing loop tests)

**Input Validation Required**:
- None (tests validate the migration)

**Tasks**:
1. **Add memory total comparison tests**
   - File: tests/test_buffer_registry.py
   - Action: Modify (append)
   - Details:
     ```python
     class TestMemoryTotalEquivalence:
         """Verify new BufferRegistry matches old BufferSettings totals."""

         def test_linear_solver_memory_totals_match(self):
             """LinearSolver registry should match old settings totals."""
             # Create with old settings
             old_settings = LinearSolverBufferSettings(n=10)
             old_shared = old_settings.shared_memory_elements
             old_local = old_settings.local_memory_elements
             
             # Create with new registry
             linear_solver = linear_solver_factory(
                 operator_apply=mock_operator,
                 n=10,
                 precision=np.float64,
                 buffer_settings=old_settings,
             )
             new_shared = linear_solver.buffer_registry.shared_memory_elements
             
             assert old_shared == new_shared

         def test_newton_solver_memory_totals_match(self):
             """Newton solver registry should match old settings totals."""
             # Similar comparison test
             pass

         def test_dirk_memory_totals_match(self):
             """DIRK step registry should match old settings totals."""
             # Similar comparison test
             pass
     ```
   - Edge cases: All buffer configurations tested
   - Integration: Ensures migration preserves behavior

2. **Add allocator compilation tests**
   - File: tests/test_buffer_registry.py
   - Action: Modify (append)
   - Details:
     ```python
     @pytest.mark.nocudasim
     class TestAllocatorCompilation:
         """Test that allocators compile to valid CUDA functions."""

         def test_shared_buffer_allocator_compiles(self):
             """Shared buffer allocator should compile successfully."""
             buffer = CUDABuffer(
                 name="test",
                 location=BufferLocation.SHARED,
                 size=10,
                 precision=np.float64,
             )
             buffer.shared_slice = slice(0, 10)
             allocator = buffer.allocate()
             
             # Verify it's a CUDA device function
             assert hasattr(allocator, 'py_func')

         def test_local_buffer_allocator_compiles(self):
             """Local buffer allocator should compile successfully."""
             # Similar test for LOCAL location
             pass
     ```
   - Edge cases: All three location types
   - Integration: Validates allocate() produces working code

**Outcomes**: 
[ ] Memory totals match between old and new systems
[ ] All allocators compile successfully
[ ] Integration tests pass with new buffer system

---

## Task Group 13: Cleanup and Deprecation - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Groups 1-12 (all prior groups complete)

**Required Context**:
- File: src/cubie/BufferSettings.py (entire file)
- All files containing BufferSettings references

**Input Validation Required**:
- None (cleanup task)

**Tasks**:
1. **Mark BufferSettings as deprecated (optional first step)**
   - File: src/cubie/BufferSettings.py
   - Action: Modify
   - Details:
     - Add deprecation warning to BufferSettings classes:
       ```python
       import warnings
       
       class BufferSettings(ABC):
           """Abstract base class for buffer memory configuration.
           
           .. deprecated::
               Use :class:`~cubie.buffer_registry.BufferRegistry` instead.
           """
           
           def __init_subclass__(cls, **kwargs):
               warnings.warn(
                   f"{cls.__name__} is deprecated. "
                   "Use BufferRegistry instead.",
                   DeprecationWarning,
                   stacklevel=2,
               )
               super().__init_subclass__(**kwargs)
       ```
   - Edge cases: Ensure existing code still works during transition
   - Integration: Signals migration path to developers

2. **Update module exports**
   - File: src/cubie/__init__.py
   - Action: Modify (if needed)
   - Details:
     - Add BufferRegistry, CUDABuffer, BufferLocation to public exports
   - Edge cases: None
   - Integration: Public API includes new buffer system

**Outcomes**: 
[ ] BufferSettings marked deprecated with warnings
[ ] New buffer registry classes exported
[ ] Migration path documented

---

## Summary

### Total Task Groups: 13
### Dependency Chain:
```
Group 1 (Core Infrastructure)
    ↓
Group 2 (allocate() Implementation)
    ↓
Group 3 (Core Unit Tests) ←──────────────────────┐
    ↓                                            │
Group 4 (CUDAFactory Integration)                │
    ↓                                            │
Group 5 (CUDAFactory Tests) ←────────────────────┤
    ↓                                            │
Group 6 (LinearSolver Migration)                 │
    ↓                                            │
Group 7 (Newton-Krylov Migration)                │
    ↓                                            │
Group 8 (ERK Migration) ←────────────────────────┤
    ↓                                            │
Group 9 (DIRK Migration)                         │
    ↓                                            │
Group 10 (FIRK/Rosenbrock Migration) ←───────────┤
    ↓                                            │
Group 11 (Loop Migration)                        │
    ↓                                            │
Group 12 (Integration Tests) ←───────────────────┘
    ↓
Group 13 (Cleanup/Deprecation)
```

### Parallel Execution Opportunities:
- Groups 3, 5: Tests can run in parallel after respective implementation
- Groups 8, 10: ERK and FIRK/Rosenbrock don't depend on each other
- Group 12: Integration tests can run components in parallel

### Estimated Complexity:
- **High**: Groups 1, 2, 11 (core infrastructure and loop migration)
- **Medium**: Groups 4, 6, 7, 9 (factory integration and implicit algorithms)
- **Low**: Groups 3, 5, 8, 10, 12, 13 (tests and simpler algorithms)

### Critical Path:
Groups 1 → 2 → 4 → 6 → 7 → 9 → 11 → 13
