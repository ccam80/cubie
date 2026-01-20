# Implementation Task List
# Feature: Refactor Memory Allocation Patterns
# Plan Reference: .github/active_plans/refactor_memory_allocation_patterns/agent_plan.md

## Task Group 1: Remove Memory Elements from BatchSolverConfig
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/batchsolving/BatchSolverConfig.py (entire file)
- File: .github/context/cubie_internal_structure.md (lines 195-199 for attrs pattern)

**Input Validation Required**:
- None (removing fields, not adding validation)

**Tasks**:
1. **Remove local_memory_elements field from BatchSolverConfig**
   - File: src/cubie/batchsolving/BatchSolverConfig.py
   - Action: Modify
   - Details:
     - Remove lines 119-122 (local_memory_elements field definition)
     - Field to remove:
       ```python
       local_memory_elements: int = attrs.field(
           default=0,
           validator=getype_validator(int, 0),
       )
       ```
   - Edge cases: None - this is a straightforward field removal
   - Integration: BatchSolverKernel will no longer pass this parameter during instantiation

2. **Remove shared_memory_elements field from BatchSolverConfig**
   - File: src/cubie/batchsolving/BatchSolverConfig.py
   - Action: Modify
   - Details:
     - Remove lines 123-126 (shared_memory_elements field definition)
     - Field to remove:
       ```python
       shared_memory_elements: int = attrs.field(
           default=0,
           validator=getype_validator(int, 0),
       )
       ```
   - Edge cases: None - this is a straightforward field removal
   - Integration: BatchSolverKernel will no longer pass this parameter during instantiation

3. **Update BatchSolverConfig docstring**
   - File: src/cubie/batchsolving/BatchSolverConfig.py
   - Action: Modify
   - Details:
     - Remove documentation for local_memory_elements and shared_memory_elements
     - Update docstring (lines 98-112) to remove:
       ```
       local_memory_elements
           Number of precision elements required in local memory per run.
       shared_memory_elements
           Number of precision elements required in shared memory per run.
       ```
     - Keep documentation for: precision, loop_fn, compile_flags
   - Edge cases: None - documentation update only
   - Integration: Users will no longer see these parameters in API documentation

**Tests to Create**:
- Test file: tests/batchsolving/test_config_plumbing.py
- Test function: test_batch_solver_config_no_memory_fields
- Description: Verify BatchSolverConfig can be instantiated without local_memory_elements or shared_memory_elements, and that these fields do not exist in the attrs class

**Outcomes**: 
- Files Modified:
  * src/cubie/batchsolving/BatchSolverConfig.py (10 lines removed, docstring updated)
  * tests/batchsolving/test_config_plumbing.py (30 lines added for test, 2 lines removed from existing test)
- Functions/Methods Added/Modified:
  * BatchSolverConfig class in BatchSolverConfig.py (removed two fields)
  * test_batch_solver_config_no_memory_fields() in test_config_plumbing.py (new test)
  * assert_solverkernel_config() in test_config_plumbing.py (removed assertions for deleted fields)
- Implementation Summary:
  Removed local_memory_elements and shared_memory_elements fields from BatchSolverConfig attrs class.
  Updated docstring to remove documentation for these fields. Created test to verify fields do not exist
  and that config can be instantiated without them. Updated existing test to remove assertions that
  checked these fields in compile_settings.
- Issues Flagged: None

**Tests to Run**:
- tests/batchsolving/test_config_plumbing.py::test_batch_solver_config_no_memory_fields
- tests/batchsolving/test_config_plumbing.py::test_comprehensive_config_plumbing

---

## Task Group 2: Update BatchSolverKernel.__init__ to Remove Memory Elements
**Status**: [x]
**Dependencies**: Groups [1]

**Required Context**:
- File: src/cubie/batchsolving/BatchSolverKernel.py (lines 280-320)
- File: src/cubie/batchsolving/BatchSolverConfig.py (entire file - post Group 1 changes)
- File: src/cubie/integrators/SingleIntegratorRun.py (lines 59-80)

**Input Validation Required**:
- None (removing code, not adding validation)

**Tasks**:
1. **Remove memory elements from initial_config in __init__**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     - In __init__ method around lines 286-296, modify initial_config creation:
     - Remove these lines from BatchSolverConfig instantiation:
       ```python
       local_memory_elements=(
           self.single_integrator.local_memory_elements
       ),
       shared_memory_elements=(
           self.single_integrator.shared_memory_elements
       ),
       ```
     - Keep:
       ```python
       initial_config = BatchSolverConfig(
           precision=precision,
           loop_fn=None,
           compile_flags=self.single_integrator.output_compile_flags,
       )
       ```
   - Edge cases: None - straightforward parameter removal
   - Integration: BatchSolverConfig no longer accepts these parameters (removed in Group 1)

2. **Remove second update_compile_settings call with memory elements**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     - Remove the entire second update_compile_settings call (lines 303-313):
       ```python
       self.update_compile_settings(
           {
               "local_memory_elements": (
                   self.single_integrator.local_memory_elements
               ),
               "shared_memory_elements": (
                   self.single_integrator.shared_memory_elements
               ),
               "precision": self.single_integrator.precision,
           }
       )
       ```
     - This call is redundant - precision is already in initial_config
   - Edge cases: Verify precision is set in initial_config before removing this
   - Integration: Config settings are complete after setup_compile_settings call on line 297

**Tests to Create**:
- Test file: tests/batchsolving/test_SolverKernel.py
- Test function: test_batch_solver_kernel_init_without_memory_elements
- Description: Verify BatchSolverKernel initializes correctly without setting memory elements in config

**Outcomes**: 
- Files Modified:
  * src/cubie/batchsolving/BatchSolverKernel.py (11 lines removed)
  * tests/batchsolving/test_SolverKernel.py (62 lines added for test)
- Functions/Methods Added/Modified:
  * BatchSolverKernel.__init__() in BatchSolverKernel.py (removed memory element parameters from config instantiation and removed redundant update_compile_settings call)
  * test_batch_solver_kernel_init_without_memory_elements() in test_SolverKernel.py (new test)
- Implementation Summary:
  Removed local_memory_elements and shared_memory_elements parameters from BatchSolverConfig
  instantiation in BatchSolverKernel.__init__. Removed redundant update_compile_settings call
  that was setting these same memory parameters plus precision (precision was already in
  initial_config). Created comprehensive test to verify kernel initializes successfully without
  these parameters in config, and that memory element properties remain accessible (for future
  Task Group 3 work where they'll query buffer_registry).
- Issues Flagged: None

**Tests to Run**:
- tests/batchsolving/test_SolverKernel.py::test_batch_solver_kernel_init_without_memory_elements

---

## Task Group 3: Update BatchSolverKernel Properties to Query buffer_registry
**Status**: [x]
**Dependencies**: Groups [1, 2]

**Required Context**:
- File: src/cubie/batchsolving/BatchSolverKernel.py (lines 911-922, 1036-1040)
- File: src/cubie/buffer_registry.py (lines 746-795)
- File: src/cubie/integrators/SingleIntegratorRun.py (lines 59-80)

**Input Validation Required**:
- None (properties compute from existing validated data)

**Tasks**:
1. **Update local_memory_elements property to query buffer_registry**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     - Replace property implementation at line 912-915:
     - Old implementation:
       ```python
       @property
       def local_memory_elements(self) -> int:
           """Number of precision elements required in local memory per run."""
           return self.compile_settings.local_memory_elements
       ```
     - New implementation:
       ```python
       @property
       def local_memory_elements(self) -> int:
           """Number of precision elements required in local memory per run."""
           return buffer_registry.persistent_local_buffer_size(
               self.single_integrator._loop
           )
       ```
     - Note: Query buffer_registry for the _loop object (the actual buffer owner)
   - Edge cases: Returns 0 if no persistent local buffers registered
   - Integration: SingleIntegratorRun._loop is the object that registers buffers with buffer_registry

2. **Update shared_memory_elements property to query buffer_registry**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     - Replace property implementation at line 917-921:
     - Old implementation:
       ```python
       @property
       def shared_memory_elements(self) -> int:
           """Number of precision elements required in shared memory per run."""
           return self.compile_settings.shared_memory_elements
       ```
     - New implementation:
       ```python
       @property
       def shared_memory_elements(self) -> int:
           """Number of precision elements required in shared memory per run."""
           return buffer_registry.shared_buffer_size(
               self.single_integrator._loop
           )
       ```
     - Note: Query buffer_registry for the _loop object (the actual buffer owner)
   - Edge cases: Returns 0 if no shared buffers registered
   - Integration: SingleIntegratorRun._loop is the object that registers buffers with buffer_registry

3. **Update shared_memory_bytes property to compute from buffer_registry query**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     - Replace property implementation at line 1035-1039:
     - Old implementation:
       ```python
       @property
       def shared_memory_bytes(self) -> int:
           """Shared-memory footprint per run for the compiled kernel."""
           return self.single_integrator.shared_memory_bytes
       ```
     - New implementation:
       ```python
       @property
       def shared_memory_bytes(self) -> int:
           """Shared-memory footprint per run for the compiled kernel."""
           element_count = self.shared_memory_elements
           itemsize = np_dtype(self.precision).itemsize
           return element_count * itemsize
       ```
     - Add import at top of file: `from numpy import dtype as np_dtype`
   - Edge cases: Returns 0 if shared_memory_elements is 0
   - Integration: Computes bytes from elements, matching pattern in SingleIntegratorRun (lines 65-70)

**Tests to Create**:
- Test file: tests/batchsolving/test_SolverKernel.py
- Test function: test_batch_solver_kernel_properties_query_buffer_registry
- Description: Verify that local_memory_elements, shared_memory_elements, and shared_memory_bytes properties return correct values queried from buffer_registry

**Outcomes**: 
- Files Modified:
  * src/cubie/batchsolving/BatchSolverKernel.py (18 lines changed: 1 import added, 3 properties updated with educational comments)
  * tests/batchsolving/test_SolverKernel.py (59 lines added for new test)
- Functions/Methods Added/Modified:
  * local_memory_elements property in BatchSolverKernel.py (now queries buffer_registry.persistent_local_buffer_size())
  * shared_memory_elements property in BatchSolverKernel.py (now queries buffer_registry.shared_buffer_size())
  * shared_memory_bytes property in BatchSolverKernel.py (now computes from shared_memory_elements * itemsize)
  * test_batch_solver_kernel_properties_query_buffer_registry() in test_SolverKernel.py (new test)
- Implementation Summary:
  Updated all three memory-related properties to query buffer_registry instead of reading from
  compile_settings. The local_memory_elements and shared_memory_elements properties now call
  buffer_registry methods with self.single_integrator._loop (the actual buffer owner object).
  The shared_memory_bytes property now computes bytes from shared_memory_elements multiplied by
  itemsize, matching the pattern in SingleIntegratorRun. Added np_dtype import for itemsize access.
  Created comprehensive test that verifies properties return values matching direct buffer_registry
  queries, validates byte computation, and checks edge cases for zero elements.
- Issues Flagged: None

**Tests to Run**:
- tests/batchsolving/test_SolverKernel.py::test_batch_solver_kernel_properties_query_buffer_registry

---

## Task Group 4: Update BatchSolverKernel.run Method to Remove Memory Elements
**Status**: [x]
**Dependencies**: Groups [1, 2, 3]

**Required Context**:
- File: src/cubie/batchsolving/BatchSolverKernel.py (lines 520-540)

**Input Validation Required**:
- None (removing code, not adding validation)

**Tasks**:
1. **Remove memory elements from update_compile_settings in run method**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     - In run method, modify update_compile_settings call (lines 528-539):
     - Remove these lines:
       ```python
       "local_memory_elements": (
           self.single_integrator.local_memory_elements
       ),
       "shared_memory_elements": (
           self.single_integrator.shared_memory_elements
       ),
       ```
     - Keep:
       ```python
       self.update_compile_settings(
           {
               "loop_fn": self.single_integrator.compiled_loop_function,
               "precision": self.single_integrator.precision,
           }
       )
       ```
   - Edge cases: None - straightforward parameter removal
   - Integration: Config no longer tracks memory elements (removed in Group 1)

**Tests to Create**:
- Test file: tests/batchsolving/test_SolverKernel.py
- Test function: test_batch_solver_kernel_run_updates_without_memory_elements
- Description: Verify run method updates compile settings without attempting to set memory elements

**Outcomes**: 
- Files Modified:
  * src/cubie/batchsolving/BatchSolverKernel.py (6 lines removed from run method)
  * tests/batchsolving/test_SolverKernel.py (78 lines added for new test)
- Functions/Methods Added/Modified:
  * BatchSolverKernel.run() in BatchSolverKernel.py (removed local_memory_elements and shared_memory_elements from update_compile_settings call)
  * test_batch_solver_kernel_run_updates_without_memory_elements() in test_SolverKernel.py (new test)
- Implementation Summary:
  Removed local_memory_elements and shared_memory_elements parameters from the
  update_compile_settings call in BatchSolverKernel.run() method. The method now only
  updates loop_fn and precision in compile_settings, which aligns with the refactoring
  that removed these fields from BatchSolverConfig in Task Group 1. Memory element
  properties remain accessible through the properties that query buffer_registry
  (implemented in Task Group 3). Created comprehensive test to verify run() executes
  successfully without attempting to update memory elements in config, and that memory
  properties still return valid values from buffer_registry.
- Issues Flagged: None

**Tests to Run**:
- tests/batchsolving/test_SolverKernel.py::test_batch_solver_kernel_run_updates_without_memory_elements

---

## Task Group 5: Update BatchSolverKernel.build_kernel to Query buffer_registry
**Status**: [x]
**Dependencies**: Groups [1, 2, 3]

**Required Context**:
- File: src/cubie/batchsolving/BatchSolverKernel.py (lines 690-710)

**Input Validation Required**:
- None (properties already provide validated data)

**Tasks**:
1. **Update build_kernel to use property instead of config**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     - In build_kernel method around line 700, change:
     - Old code:
       ```python
       shared_elems_per_run = config.shared_memory_elements
       ```
     - New code:
       ```python
       shared_elems_per_run = self.shared_memory_elements
       ```
     - This uses the property defined in Group 3 which queries buffer_registry
   - Edge cases: None - property handles all edge cases
   - Integration: Property queries buffer_registry for current shared buffer size at build time

**Tests to Create**:
- Test file: tests/batchsolving/test_SolverKernel.py
- Test function: test_batch_solver_kernel_build_uses_current_buffer_sizes
- Description: Verify build_kernel uses current buffer_registry state via properties

**Outcomes**: 
- Files Modified:
  * src/cubie/batchsolving/BatchSolverKernel.py (4 lines changed: 1 line modified, 3 comment lines added)
  * tests/batchsolving/test_SolverKernel.py (76 lines added for new test)
- Functions/Methods Added/Modified:
  * build_kernel() in BatchSolverKernel.py (changed to use self.shared_memory_elements property instead of config.shared_memory_elements)
  * test_batch_solver_kernel_build_uses_current_buffer_sizes() in test_SolverKernel.py (new test)
- Implementation Summary:
  Updated build_kernel method to query shared_memory_elements via the property
  (which queries buffer_registry) instead of reading from config.shared_memory_elements
  (which no longer exists after Task Group 1). Added educational comments explaining
  that this ensures build_kernel uses the actual registered buffer size from
  buffer_registry. Created comprehensive test that verifies build_kernel successfully
  uses buffer_registry-derived sizes, validates consistency between property and
  registry queries, and confirms kernel compilation and execution work correctly
  with the refactored memory allocation pattern.
- Issues Flagged: None

**Tests to Run**:
- tests/batchsolving/test_SolverKernel.py::test_batch_solver_kernel_build_uses_current_buffer_sizes

---

## Task Group 6: Update BatchSolverKernel.update Method to Remove Memory Elements
**Status**: [x]
**Dependencies**: Groups [1, 2, 3]

**Required Context**:
- File: src/cubie/batchsolving/BatchSolverKernel.py (lines 870-905)
- File: src/cubie/buffer_registry.py (lines 671-744)

**Input Validation Required**:
- None (buffer_registry.update handles validation)

**Tasks**:
1. **Remove memory elements from update_compile_settings in update method**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     - In update method, modify the updates_dict.update call (lines 877-888):
     - Remove these lines:
       ```python
       "local_memory_elements": (
           self.single_integrator.local_memory_elements
       ),
       "shared_memory_elements": (
           self.single_integrator.shared_memory_elements
       ),
       ```
     - Keep:
       ```python
       updates_dict.update(
           {
               "loop_fn": self.single_integrator.device_function,
               "compile_flags": self.single_integrator.output_compile_flags,
           }
       )
       ```
   - Edge cases: None - straightforward parameter removal
   - Integration: Config no longer accepts memory element parameters

2. **Add buffer_registry.update call to recognize location parameters**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     - After the self.single_integrator.update() call (line 873-875), add buffer_registry update:
     - Add after line 875:
       ```python
       all_unrecognized -= buffer_registry.update(
           self.single_integrator._loop, updates_dict, silent=True
       )
       ```
     - This allows BatchSolverKernel.update() to recognize buffer location parameters like 'state_location', 'proposed_state_location', etc.
     - The buffer_registry.update() call returns set of recognized parameter names
   - Edge cases: buffer_registry.update() silently ignores non-location parameters
   - Integration: Follows same pattern as IVPLoop.update() (lines 984-986 in ode_loop.py)

**Tests to Create**:
- Test file: tests/batchsolving/test_SolverKernel.py
- Test function: test_batch_solver_kernel_update_recognizes_buffer_locations
- Description: Verify update method recognizes buffer location parameters and delegates to buffer_registry

**Outcomes**: 
- Files Modified:
  * src/cubie/batchsolving/BatchSolverKernel.py (12 lines changed: 6 lines removed, 6 comment/code lines added)
  * tests/batchsolving/test_SolverKernel.py (107 lines added for new test)
- Functions/Methods Added/Modified:
  * BatchSolverKernel.update() in BatchSolverKernel.py (removed local_memory_elements and shared_memory_elements from updates_dict, added buffer_registry.update() call)
  * test_batch_solver_kernel_update_recognizes_buffer_locations() in test_SolverKernel.py (new test)
- Implementation Summary:
  Removed local_memory_elements and shared_memory_elements parameters from the
  updates_dict.update() call in BatchSolverKernel.update() method. Added
  buffer_registry.update() call after single_integrator.update() to recognize
  and handle buffer location parameters (e.g., 'state_location', 'proposed_state_location').
  This delegates buffer location management to buffer_registry, following the same
  pattern as IVPLoop.update(). Added educational comments explaining the delegation.
  Created comprehensive test that verifies: (1) valid buffer location parameters are
  recognized by buffer_registry, (2) invalid location values raise ValueError,
  (3) bogus parameters raise KeyError when silent=False, (4) memory elements are
  not in compile_settings, (5) update() still updates loop_fn and compile_flags,
  and (6) memory properties remain accessible via buffer_registry queries.
- Issues Flagged: None

**Tests to Run**:
- tests/batchsolving/test_SolverKernel.py::test_batch_solver_kernel_update_recognizes_buffer_locations

---

## Task Group 7: Integration Testing
**Status**: [x]
**Dependencies**: Groups [1, 2, 3, 4, 5, 6]

**Required Context**:
- File: tests/batchsolving/test_SolverKernel.py (entire file)
- File: tests/batchsolving/test_solver.py (entire file)
- File: src/cubie/batchsolving/BatchSolverKernel.py (entire file - post all changes)

**Input Validation Required**:
- None (testing only)

**Tasks**:
1. **Create integration test verifying end-to-end behavior**
   - File: tests/batchsolving/test_refactored_memory_allocation.py
   - Action: Create
   - Details:
     ```python
     """Integration tests for refactored memory allocation patterns.
     
     Verifies that BatchSolverKernel correctly queries buffer_registry
     for memory sizes and that behavior is identical to pre-refactoring.
     """
     import pytest
     from cubie import Solver
     from cubie.buffer_registry import buffer_registry
     
     
     def test_memory_allocation_via_buffer_registry(three_state_linear):
         """Verify BatchSolverKernel queries buffer_registry for memory sizes."""
         solver = Solver(
             system=three_state_linear,
             algorithm='explicit_euler',
             dt=0.001,
         )
         
         # Memory sizes should be queryable via properties
         shared_elements = solver.kernel.shared_memory_elements
         local_elements = solver.kernel.local_memory_elements
         shared_bytes = solver.kernel.shared_memory_bytes
         
         # These should match buffer_registry queries
         loop = solver.kernel.single_integrator._loop
         assert shared_elements == buffer_registry.shared_buffer_size(loop)
         assert local_elements == buffer_registry.persistent_local_buffer_size(loop)
         
         # shared_bytes should be elements * itemsize
         import numpy as np
         expected_bytes = shared_elements * np.dtype(solver.kernel.precision).itemsize
         assert shared_bytes == expected_bytes
     
     
     def test_memory_sizes_update_with_buffer_changes(three_state_linear):
         """Verify memory sizes reflect buffer_registry changes."""
         solver = Solver(
             system=three_state_linear,
             algorithm='explicit_euler',
             dt=0.001,
         )
         
         initial_shared = solver.kernel.shared_memory_elements
         
         # Update buffer location (if implementation supports it)
         # This would trigger buffer re-registration
         solver.kernel.single_integrator.update({'dt0': 0.002})
         
         # Memory sizes should still be accessible
         updated_shared = solver.kernel.shared_memory_elements
         
         # Size should be consistent (may or may not change depending on algorithm)
         assert isinstance(updated_shared, int)
         assert updated_shared >= 0
     
     
     def test_solver_run_with_refactored_allocation(three_state_linear):
         """Verify solver.solve() works correctly with refactored allocation."""
         solver = Solver(
             system=three_state_linear,
             algorithm='explicit_euler',
             dt=0.001,
         )
         
         # Run a solve to verify everything works end-to-end
         result = solver.solve(
             duration=1.0,
             runs=10,
         )
         
         # Verify result is valid
         assert result.state is not None
         assert result.state.shape[0] == 10  # 10 runs
         assert result.status_codes is not None
     ```
   - Edge cases: Test with different algorithms, adaptive vs fixed step
   - Integration: Validates entire refactoring works correctly in real usage

**Tests to Create**:
- Test file: tests/batchsolving/test_refactored_memory_allocation.py
- Test function: test_memory_allocation_via_buffer_registry
- Description: Verify memory sizes are correctly queried from buffer_registry
- Test function: test_memory_sizes_update_with_buffer_changes
- Description: Verify memory sizes remain accessible after updates
- Test function: test_solver_run_with_refactored_allocation
- Description: End-to-end integration test verifying solve() works correctly

**Outcomes**: 
- Files Modified:
  * tests/batchsolving/test_refactored_memory_allocation.py (136 lines created)
- Functions/Methods Added/Modified:
  * three_state_linear fixture in test_refactored_memory_allocation.py (function-scoped fixture for clean buffer_registry state)
  * test_memory_allocation_via_buffer_registry() in test_refactored_memory_allocation.py (new test)
  * test_memory_sizes_update_with_buffer_changes() in test_refactored_memory_allocation.py (new test)
  * test_solver_run_with_refactored_allocation() in test_refactored_memory_allocation.py (new test)
- Implementation Summary:
  Created comprehensive integration test file with three tests validating the refactored
  memory allocation pattern. Test 1 verifies that kernel memory properties query
  buffer_registry and match direct registry queries. Test 2 verifies memory properties
  remain accessible after parameter updates. Test 3 is the critical end-to-end test that
  runs a full solve() with the refactored allocation, confirming kernel builds correctly
  using buffer_registry sizes and executes batch runs successfully. Added three_state_linear
  fixture with function scope to ensure clean buffer_registry state between tests.
  All tests include educational comments explaining what they validate and why.
- Issues Flagged: None

**Tests to Run**:
- tests/batchsolving/test_refactored_memory_allocation.py::test_memory_allocation_via_buffer_registry
- tests/batchsolving/test_refactored_memory_allocation.py::test_memory_sizes_update_with_buffer_changes
- tests/batchsolving/test_refactored_memory_allocation.py::test_solver_run_with_refactored_allocation

---

## Summary

**Total Task Groups**: 7

**Dependency Chain**:
```
Group 1 (Config cleanup)
   ↓
Group 2 (Init cleanup) ──┐
   ↓                     │
Group 3 (Properties) ────┤
   ↓                     │
Group 4 (Run method) ────┤
   ↓                     │
Group 5 (Build kernel) ──┤
   ↓                     │
Group 6 (Update method) ─┤
   ↓                     │
Group 7 (Integration) ←──┘
```

**Tests to Create**: 7 test functions across 3 test files

**Estimated Complexity**: Medium
- Straightforward field and parameter removals (Groups 1, 2, 4, 6)
- Property updates with clear buffer_registry API (Group 3, 5)
- Integration testing to validate behavior (Group 7)

**Breaking Changes**: None (internal refactoring only)
- Public API unchanged (properties have same names and return same values)
- BatchSolverConfig internal changes (users don't instantiate directly)
- Memory allocation behavior identical from user perspective

**Key Risks**:
- Accessing SingleIntegratorRun._loop (private attribute) - but this is internal code
- Ensuring buffer_registry is queried at correct time (after buffer registration)
- Properties must return same values as before refactoring

**Validation Strategy**:
- Each task group includes specific test creation
- Integration tests verify end-to-end behavior
- Existing tests should pass without modification (proves behavior unchanged)
