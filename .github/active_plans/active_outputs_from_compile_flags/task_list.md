# Implementation Task List
# Feature: ActiveOutputs Derivation from OutputCompileFlags
# Plan Reference: .github/active_plans/active_outputs_from_compile_flags/agent_plan.md

## Task Group 1: Add Factory Method to ActiveOutputs - SEQUENTIAL
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/batchsolving/arrays/BatchOutputArrays.py (lines 109-191)
- File: src/cubie/outputhandling/output_config.py (lines 64-102 for OutputCompileFlags)

**Input Validation Required**:
- flags parameter: Check isinstance(flags, OutputCompileFlags)

**Tasks**:

1. **Add import for OutputCompileFlags**
   - File: src/cubie/batchsolving/arrays/BatchOutputArrays.py
   - Action: Modify
   - Details:
     Add import statement at top of file after existing imports:
     ```python
     from cubie.outputhandling.output_config import OutputCompileFlags
     ```
   - Edge cases: None
   - Integration: Required for type annotation and isinstance check

2. **Add from_compile_flags class method to ActiveOutputs**
   - File: src/cubie/batchsolving/arrays/BatchOutputArrays.py
   - Action: Modify (add method to ActiveOutputs class, after line 146, before update_from_outputarrays)
   - Details:
     ```python
     @classmethod
     def from_compile_flags(cls, flags: OutputCompileFlags) -> "ActiveOutputs":
         """
         Create ActiveOutputs from compile flags.

         Parameters
         ----------
         flags
             The compile flags determining which outputs are active.

         Returns
         -------
         ActiveOutputs
             Instance with flags derived from compile flags.

         Notes
         -----
         Maps OutputCompileFlags to ActiveOutputs:
         - save_state → state
         - save_observables → observables  
         - summarise_state → state_summaries
         - summarise_observables → observable_summaries
         - save_counters → iteration_counters
         - status_codes is always True (always written during execution)
         """
         return cls(
             state=flags.save_state,
             observables=flags.save_observables,
             state_summaries=flags.summarise_state,
             observable_summaries=flags.summarise_observables,
             status_codes=True,
             iteration_counters=flags.save_counters,
         )
     ```
   - Edge cases:
     - Empty output types: All flags False except status_codes (always True)
     - Single-run batches: status_codes=True regardless of array size
     - Single-variable summaries: Derived from flags, not array size
   - Integration: This factory method will be called by BatchSolverKernel

**Outcomes**: 
- Files Modified: 
  * src/cubie/batchsolving/arrays/BatchOutputArrays.py (35 lines added)
- Functions/Methods Added/Modified:
  * from_compile_flags() classmethod in ActiveOutputs
  * Added import for OutputCompileFlags
- Implementation Summary:
  Added factory method that maps OutputCompileFlags to ActiveOutputs.
  status_codes is always True as it's always written during execution.
- Issues Flagged: None

---

## Task Group 2: Update OutputArrays to Accept External ActiveOutputs - SEQUENTIAL
**Status**: [x]
**Dependencies**: Group 1

**Required Context**:
- File: src/cubie/batchsolving/arrays/BatchOutputArrays.py (lines 193-281)

**Input Validation Required**:
- active_outputs parameter in set_active_outputs: Check isinstance(active_outputs, ActiveOutputs)

**Tasks**:

1. **Add set_active_outputs method to OutputArrays**
   - File: src/cubie/batchsolving/arrays/BatchOutputArrays.py
   - Action: Modify (add method after active_outputs property, around line 281)
   - Details:
     ```python
     def set_active_outputs(self, active_outputs: ActiveOutputs) -> None:
         """
         Set active outputs from external source.

         Parameters
         ----------
         active_outputs
             ActiveOutputs instance to store.

         Returns
         -------
         None
             Stores the provided ActiveOutputs instance.
         """
         self._active_outputs = active_outputs
     ```
   - Edge cases: None
   - Integration: Called by BatchSolverKernel.__init__ and update()

2. **Modify active_outputs property to return stored value without recalculation**
   - File: src/cubie/batchsolving/arrays/BatchOutputArrays.py
   - Action: Modify (lines 277-281)
   - Details:
     Change from:
     ```python
     @property
     def active_outputs(self) -> ActiveOutputs:
         """Active output configuration derived from host allocations."""
         self._active_outputs.update_from_outputarrays(self)
         return self._active_outputs
     ```
     To:
     ```python
     @property
     def active_outputs(self) -> ActiveOutputs:
         """Active output configuration."""
         return self._active_outputs
     ```
   - Edge cases: 
     - Property now returns whatever was set; relies on external caller to set correctly
   - Integration: BatchSolverKernel will set via set_active_outputs before accessing

**Outcomes**: 
- Files Modified: 
  * src/cubie/batchsolving/arrays/BatchOutputArrays.py (17 lines modified)
- Functions/Methods Added/Modified:
  * set_active_outputs() method in OutputArrays
  * active_outputs property modified to return stored value
- Implementation Summary:
  Added setter method and simplified property to return stored ActiveOutputs instance.
- Issues Flagged: None

---

## Task Group 3: Update BatchSolverKernel.__init__ to Use Factory Method - SEQUENTIAL
**Status**: [x]
**Dependencies**: Groups 1, 2

**Required Context**:
- File: src/cubie/batchsolving/BatchSolverKernel.py (lines 109-184)
- File: src/cubie/integrators/SingleIntegratorRun.py (lines 529-533 for output_compile_flags property)

**Input Validation Required**:
- None (SingleIntegratorRun.output_compile_flags already validated)

**Tasks**:

1. **Modify __init__ to create ActiveOutputs from compile flags**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify (lines 151-184)
   - Details:
     Replace the current initialization flow. After line 151 where single_integrator is created, and before initial_config creation (line 153):
     
     Current code (lines 153-183):
     ```python
         initial_config = BatchSolverConfig(
             precision=precision,
             loop_fn=None,
             local_memory_elements=(
                 self.single_integrator.local_memory_elements
             ),
             shared_memory_elements=(
                 self.single_integrator.shared_memory_elements
             ),
             ActiveOutputs=ActiveOutputs(),
             # placeholder, updated after arrays allocate
         )
         self.setup_compile_settings(initial_config)

         self.input_arrays = InputArrays.from_solver(self)
         self.output_arrays = OutputArrays.from_solver(self)

         # Allocate/update to set active outputs then refresh compile settings
         self.output_arrays.update(self)
         self.update_compile_settings(
             {
                 "ActiveOutputs": self.output_arrays.active_outputs,
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
     
     Replace with:
     ```python
         # Derive ActiveOutputs from compile flags (authoritative source)
         compile_flags = self.single_integrator.output_compile_flags
         active_outputs = ActiveOutputs.from_compile_flags(compile_flags)

         initial_config = BatchSolverConfig(
             precision=precision,
             loop_fn=None,
             local_memory_elements=(
                 self.single_integrator.local_memory_elements
             ),
             shared_memory_elements=(
                 self.single_integrator.shared_memory_elements
             ),
             ActiveOutputs=active_outputs,
         )
         self.setup_compile_settings(initial_config)

         self.input_arrays = InputArrays.from_solver(self)
         self.output_arrays = OutputArrays.from_solver(self)

         # Set active outputs on output_arrays and update arrays
         self.output_arrays.set_active_outputs(active_outputs)
         self.output_arrays.update(self)
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
   - Edge cases:
     - Single-run batches: status_codes=True from factory method
     - No output types configured: Validation handled by OutputConfig
   - Integration: Uses SingleIntegratorRun.output_compile_flags property

**Outcomes**: 
- Files Modified: 
  * src/cubie/batchsolving/BatchSolverKernel.py (11 lines modified)
- Functions/Methods Added/Modified:
  * __init__() method modified to derive ActiveOutputs from compile flags
- Implementation Summary:
  Derived ActiveOutputs from compile flags in __init__ and set it on output_arrays.
  Removed placeholder comment and now properly initializes with correct flags.
- Issues Flagged: None

---

## Task Group 4: Update BatchSolverKernel.update() to Use Factory Method - SEQUENTIAL
**Status**: [x]
**Dependencies**: Groups 1, 2, 3

**Required Context**:
- File: src/cubie/batchsolving/BatchSolverKernel.py (lines 667-740)

**Input Validation Required**:
- None (output_compile_flags already validated)

**Tasks**:

1. **Modify update() to derive ActiveOutputs from compile flags**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify (lines 714-727)
   - Details:
     Current code:
     ```python
         all_unrecognized -= self.single_integrator.update(
                 updates_dict, silent=True
         )
         updates_dict.update({
             "loop_function": self.single_integrator.device_function,
             "local_memory_elements": (
                 self.single_integrator.local_memory_elements
             ),
             "shared_memory_elements": (
                 self.single_integrator.shared_memory_elements
             ),
              "ActiveOutputs": self.output_arrays.active_outputs,
         })
     ```
     
     Replace with:
     ```python
         all_unrecognized -= self.single_integrator.update(
                 updates_dict, silent=True
         )
         # Derive ActiveOutputs from updated compile flags
         compile_flags = self.single_integrator.output_compile_flags
         active_outputs = ActiveOutputs.from_compile_flags(compile_flags)
         self.output_arrays.set_active_outputs(active_outputs)
         
         updates_dict.update({
             "loop_function": self.single_integrator.device_function,
             "local_memory_elements": (
                 self.single_integrator.local_memory_elements
             ),
             "shared_memory_elements": (
                 self.single_integrator.shared_memory_elements
             ),
             "ActiveOutputs": active_outputs,
         })
     ```
   - Edge cases:
     - Update without changing outputs: compile_flags unchanged, ActiveOutputs same
     - Update enabling summaries: compile_flags.summarise_state becomes True
   - Integration: Ensures ActiveOutputs reflects updated compile flags before kernel rebuild

**Outcomes**: 
- Files Modified: 
  * src/cubie/batchsolving/BatchSolverKernel.py (6 lines added)
- Functions/Methods Added/Modified:
  * update() method modified to derive ActiveOutputs from compile flags
- Implementation Summary:
  After single_integrator.update(), now derives ActiveOutputs from compile flags
  and sets it on output_arrays before updating compile settings.
- Issues Flagged: None

---

## Task Group 5: Update BatchSolverKernel.run() to Use Derived ActiveOutputs - SEQUENTIAL
**Status**: [x]
**Dependencies**: Groups 1, 2, 3, 4

**Required Context**:
- File: src/cubie/batchsolving/BatchSolverKernel.py (lines 281-297)

**Input Validation Required**:
- None

**Tasks**:

1. **Modify run() to derive ActiveOutputs from compile flags instead of output_arrays**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify (lines 281-297)
   - Details:
     Current code:
     ```python
         # Queue allocations
         self.input_arrays.update(self, inits, params, driver_coefficients)
         self.output_arrays.update(self)

         # Refresh compile-critical settings (may trigger rebuild)
         self.update_compile_settings(
             {
                 "loop_fn": self.single_integrator.compiled_loop_function,
                 "precision": self.single_integrator.precision,
                 "local_memory_elements": (
                     self.single_integrator.local_memory_elements
                 ),
                 "shared_memory_elements": (
                     self.single_integrator.shared_memory_elements
                 ),
                 "ActiveOutputs": self.output_arrays.active_outputs,
             }
         )
     ```
     
     Replace with:
     ```python
         # Queue allocations
         self.input_arrays.update(self, inits, params, driver_coefficients)
         self.output_arrays.update(self)

         # Derive ActiveOutputs from compile flags (authoritative source)
         compile_flags = self.single_integrator.output_compile_flags
         active_outputs = ActiveOutputs.from_compile_flags(compile_flags)
         self.output_arrays.set_active_outputs(active_outputs)

         # Refresh compile-critical settings (may trigger rebuild)
         self.update_compile_settings(
             {
                 "loop_fn": self.single_integrator.compiled_loop_function,
                 "precision": self.single_integrator.precision,
                 "local_memory_elements": (
                     self.single_integrator.local_memory_elements
                 ),
                 "shared_memory_elements": (
                     self.single_integrator.shared_memory_elements
                 ),
                 "ActiveOutputs": active_outputs,
             }
         )
     ```
   - Edge cases:
     - Repeated runs: Each run derives fresh ActiveOutputs from current compile flags
   - Integration: Ensures run() uses authoritative source for ActiveOutputs

**Outcomes**: 
- Files Modified: 
  * src/cubie/batchsolving/BatchSolverKernel.py (5 lines added)
- Functions/Methods Added/Modified:
  * run() method modified to derive ActiveOutputs from compile flags
- Implementation Summary:
  After output_arrays.update(), now derives ActiveOutputs from compile flags
  and sets it on output_arrays before updating compile settings.
- Issues Flagged: None

---

## Task Group 6: Add Unit Tests for ActiveOutputs.from_compile_flags - PARALLEL
**Status**: [x]
**Dependencies**: Groups 1-5

**Required Context**:
- File: tests/batchsolving/test_SolverKernel.py (entire file)
- File: src/cubie/batchsolving/arrays/BatchOutputArrays.py (ActiveOutputs class)
- File: src/cubie/outputhandling/output_config.py (OutputCompileFlags class)

**Input Validation Required**:
- None (tests validate behavior, not inputs)

**Tasks**:

1. **Add test for ActiveOutputs.from_compile_flags factory method**
   - File: tests/batchsolving/test_BatchOutputArrays.py (create new file if not exists, or add to existing)
   - Action: Create/Modify
   - Details:
     ```python
     import pytest
     from cubie.batchsolving.arrays.BatchOutputArrays import ActiveOutputs
     from cubie.outputhandling.output_config import OutputCompileFlags


     class TestActiveOutputsFromCompileFlags:
         """Tests for ActiveOutputs.from_compile_flags factory method."""

         def test_all_flags_true(self):
             """Test mapping when all compile flags are enabled."""
             flags = OutputCompileFlags(
                 save_state=True,
                 save_observables=True,
                 summarise=True,
                 summarise_observables=True,
                 summarise_state=True,
                 save_counters=True,
             )
             active = ActiveOutputs.from_compile_flags(flags)
             
             assert active.state is True
             assert active.observables is True
             assert active.state_summaries is True
             assert active.observable_summaries is True
             assert active.iteration_counters is True
             assert active.status_codes is True

         def test_all_flags_false(self):
             """Test mapping when all compile flags are disabled."""
             flags = OutputCompileFlags(
                 save_state=False,
                 save_observables=False,
                 summarise=False,
                 summarise_observables=False,
                 summarise_state=False,
                 save_counters=False,
             )
             active = ActiveOutputs.from_compile_flags(flags)
             
             assert active.state is False
             assert active.observables is False
             assert active.state_summaries is False
             assert active.observable_summaries is False
             assert active.iteration_counters is False
             # status_codes is ALWAYS True
             assert active.status_codes is True

         def test_status_codes_always_true(self):
             """Verify status_codes is always True regardless of flags."""
             flags = OutputCompileFlags()  # All defaults (False)
             active = ActiveOutputs.from_compile_flags(flags)
             assert active.status_codes is True

         def test_partial_flags(self):
             """Test with only some flags enabled."""
             flags = OutputCompileFlags(
                 save_state=True,
                 save_observables=False,
                 summarise=True,
                 summarise_observables=False,
                 summarise_state=True,
                 save_counters=False,
             )
             active = ActiveOutputs.from_compile_flags(flags)
             
             assert active.state is True
             assert active.observables is False
             assert active.state_summaries is True
             assert active.observable_summaries is False
             assert active.iteration_counters is False
             assert active.status_codes is True
     ```
   - Edge cases: Covered in test cases above
   - Integration: Tests the factory method in isolation

2. **Add test to verify updated solver matches fresh solver (extends test_all_lower_plumbing)**
   - File: tests/batchsolving/test_SolverKernel.py
   - Action: Verify existing test_all_lower_plumbing passes with changes
   - Details:
     The existing `test_all_lower_plumbing` test at lines 148-226 already validates that an updated solver's compile_settings match a freshly instantiated solver with the same configuration. This test should now pass once the fix is implemented.
     
     Key assertions that will be validated:
     - `freshsolver.compile_settings == solverkernel.compile_settings` (line 207)
     - This includes `BatchSolverConfig.ActiveOutputs` equality
   - Edge cases: None (existing test covers the scenario)
   - Integration: Validates end-to-end fix

**Outcomes**: 
- Files Modified: 
  * tests/batchsolving/arrays/test_batchoutputarrays.py (98 lines added)
- Functions/Methods Added/Modified:
  * TestActiveOutputsFromCompileFlags test class added
  * TestOutputArraysSetActiveOutputs test class added
  * Import for OutputCompileFlags added
- Implementation Summary:
  Added comprehensive tests for from_compile_flags factory method covering:
  all flags true, all flags false, status_codes always true, partial flags.
  Also added test for set_active_outputs method.
- Issues Flagged: None

---

# Implementation Complete - Ready for Review

## Execution Summary
- Total Task Groups: 6
- Completed: 6
- Failed: 0
- Total Files Modified: 3

## Task Group Completion
- Group 1: [x] Add Factory Method to ActiveOutputs - COMPLETED
- Group 2: [x] Update OutputArrays to Accept External ActiveOutputs - COMPLETED
- Group 3: [x] Update BatchSolverKernel.__init__ to Use Factory Method - COMPLETED
- Group 4: [x] Update BatchSolverKernel.update() to Use Factory Method - COMPLETED
- Group 5: [x] Update BatchSolverKernel.run() to Use Derived ActiveOutputs - COMPLETED
- Group 6: [x] Add Unit Tests for ActiveOutputs.from_compile_flags - COMPLETED

## All Modified Files
1. src/cubie/batchsolving/arrays/BatchOutputArrays.py (52 lines changed)
2. src/cubie/batchsolving/BatchSolverKernel.py (22 lines changed)
3. tests/batchsolving/arrays/test_batchoutputarrays.py (108 lines changed)

## Flagged Issues
None - all implementations followed specifications.

## Summary

**Total Task Groups**: 6
**Dependency Chain**: Groups 1 → 2 → 3 → 4 → 5 → 6 (sequential chain with Group 6 parallelizable internally)
**Parallel Execution Opportunities**: Tasks within Group 6 can run in parallel
**Estimated Complexity**: Medium - well-scoped changes to 2 source files with clear mapping

### Key Changes Summary

| File | Change Type | Description |
|------|-------------|-------------|
| BatchOutputArrays.py | Add import | Import OutputCompileFlags |
| BatchOutputArrays.py | Add method | ActiveOutputs.from_compile_flags() factory |
| BatchOutputArrays.py | Add method | OutputArrays.set_active_outputs() setter |
| BatchOutputArrays.py | Modify property | OutputArrays.active_outputs returns stored value |
| BatchSolverKernel.py | Modify __init__ | Use factory method for ActiveOutputs |
| BatchSolverKernel.py | Modify update() | Derive ActiveOutputs from compile flags |
| BatchSolverKernel.py | Modify run() | Use factory method for ActiveOutputs |
| test_batchoutputarrays.py | Add tests | Unit tests for factory method and set_active_outputs |

### Field Mapping Reference

| OutputCompileFlags | ActiveOutputs | Notes |
|-------------------|---------------|-------|
| save_state | state | Direct mapping |
| save_observables | observables | Direct mapping |
| summarise_state | state_summaries | Direct mapping |
| summarise_observables | observable_summaries | Direct mapping |
| save_counters | iteration_counters | Direct mapping |
| (always True) | status_codes | Always active during kernel execution |

## Handoff to Reviewer
All implementation tasks complete. Task list updated with outcomes.
Ready for reviewer agent to validate against user stories and goals.
