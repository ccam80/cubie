# Implementation Task List
# Feature: Variable Interface Consolidation
# Plan Reference: .github/active_plans/consolidate_variable_interface/agent_plan.md

## Task Group 1: Add Variable Resolution Methods to SystemInterface
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: src/cubie/batchsolving/SystemInterface.py (entire file)
- File: src/cubie/batchsolving/solver.py (lines 292-404) - Current implementation to migrate
- File: .github/context/cubie_internal_structure.md (lines 1-100) - Architecture patterns

**Input Validation Required**:
- `labels`: Check if None, empty list, or non-empty list; use `is None` check (not truthiness)
- `state_indices`: Check if None, empty array, or non-empty array; use `is None` check
- `observable_indices`: Check if None, empty array, or non-empty array; use `is None` check
- `var_labels`: Check if None, empty list, or non-empty list; use `is None` check
- `max_states`: Validate is int >= 0
- `max_observables`: Validate is int >= 0

**Tasks**:
1. **Add resolve_variable_labels method**
   - File: src/cubie/batchsolving/SystemInterface.py
   - Action: Add method after existing `observable_labels` method (around line 258)
   - Details:
     ```python
     def resolve_variable_labels(
         self,
         labels: Optional[List[str]],
         silent: bool = False,
     ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
         """Resolve variable labels to separate state and observable indices.

         Parameters
         ----------
         labels
             Variable names that may be states or observables. If ``None``,
             returns ``(None, None)`` to signal "use defaults". If empty
             list, returns empty arrays for both.
         silent
             If ``True``, suppresses errors for unrecognized labels.

         Returns
         -------
         Tuple[Optional[np.ndarray], Optional[np.ndarray]]
             Tuple of (state_indices, observable_indices). Returns
             ``(None, None)`` when labels is None.

         Raises
         ------
         ValueError
             If any label is not found in states or observables and
             silent is False.
         """
         # Implementation logic:
         # 1. If labels is None, return (None, None) to signal defaults
         # 2. If labels is empty, return (empty array, empty array)
         # 3. Resolve each label to state or observable index
         # 4. Validate all labels found (unless silent)
         # 5. Return tuple of state and observable index arrays
     ```
   - Edge cases:
     - Empty labels list → returns (empty int32 array, empty int32 array)
     - None labels → returns (None, None)
     - Mixed valid/invalid labels (silent=False) → raises ValueError listing invalid
     - Mixed valid/invalid labels (silent=True) → returns only valid indices
   - Integration: Uses existing `state_indices()` and `observable_indices()` methods with silent=True

2. **Add merge_variable_inputs method**
   - File: src/cubie/batchsolving/SystemInterface.py
   - Action: Add method after `resolve_variable_labels`
   - Details:
     ```python
     def merge_variable_inputs(
         self,
         var_labels: Optional[List[str]],
         state_indices: Optional[Union[List[int], np.ndarray]],
         observable_indices: Optional[Union[List[int], np.ndarray]],
         max_states: int,
         max_observables: int,
     ) -> Tuple[np.ndarray, np.ndarray]:
         """Merge label-based selections with index-based selections.

         Parameters
         ----------
         var_labels
             Variable names to resolve. None means "not provided".
         state_indices
             Direct state index selection. None means "not provided".
         observable_indices
             Direct observable index selection. None means "not provided".
         max_states
             Total number of states in the system.
         max_observables
             Total number of observables in the system.

         Returns
         -------
         Tuple[np.ndarray, np.ndarray]
             Final (state_indices, observable_indices) arrays.

         Notes
         -----
         When all three inputs are None, returns full range arrays
         (all states, all observables). When any input is explicitly
         empty ([] or empty array), that emptiness is preserved.
         Union of resolved labels and provided indices is returned.
         """
         # Implementation logic:
         # 1. Resolve var_labels using resolve_variable_labels()
         # 2. Handle "all defaults" case: all three inputs None → full range
         # 3. Handle explicit empty: if any is empty array/list, respect it
         # 4. Compute union of resolved label indices with provided indices
         # 5. Return final (state_indices, observable_indices) as int32 arrays
     ```
   - Edge cases:
     - All None → returns (arange(max_states), arange(max_observables))
     - Empty labels [] with None indices → returns (empty, empty)
     - None labels with empty indices [] → returns (empty for that type, full for other if None)
     - Union with duplicates → deduplicated via np.union1d
   - Integration: Calls `resolve_variable_labels()` internally

3. **Add convert_variable_labels method**
   - File: src/cubie/batchsolving/SystemInterface.py
   - Action: Add method after `merge_variable_inputs`
   - Details:
     ```python
     def convert_variable_labels(
         self,
         output_settings: Dict[str, Any],
         max_states: int,
         max_observables: int,
     ) -> None:
         """Convert variable label settings to index arrays in-place.

         Parameters
         ----------
         output_settings
             Settings dict containing ``save_variables``,
             ``summarise_variables``, and their index counterparts.
             Modified in-place.
         max_states
             Total number of states in the system.
         max_observables
             Total number of observables in the system.

         Returns
         -------
         None
             Modifies output_settings in-place.

         Raises
         ------
         ValueError
             If any variable labels are not found in states or observables.

         Notes
         -----
         Pops ``save_variables`` and ``summarise_variables`` from the dict
         and replaces index parameters with final resolved arrays. For
         summarised indices, defaults to saved indices when both labels
         and indices are None.
         """
         # Implementation logic:
         # 1. Extract save_variables, saved_state_indices, saved_observable_indices
         # 2. Call merge_variable_inputs for save variables
         # 3. Extract summarise_variables, summarised_*_indices
         # 4. Handle "summarised defaults to saved" when all summarise inputs None
         # 5. Call merge_variable_inputs for summarise variables
         # 6. Pop label keys, update dict with final index arrays
     ```
   - Edge cases:
     - No keys present → sets all to full range
     - Only save_variables present → summarised defaults to saved
     - Empty save_variables=[] → explicit no variables saved
   - Integration: Called by Solver.convert_output_labels()

4. **Add required imports to SystemInterface**
   - File: src/cubie/batchsolving/SystemInterface.py
   - Action: Modify imports at top of file
   - Details:
     ```python
     from typing import Any, Dict, List, Optional, Set, Tuple, Union
     ```
   - Edge cases: None
   - Integration: Required for new method type hints

**Tests to Create**:
- Test file: tests/batchsolving/test_system_interface.py
- Test function: test_resolve_variable_labels_none_returns_none_tuple
- Description: Verify that None input returns (None, None)
- Test function: test_resolve_variable_labels_empty_returns_empty_arrays
- Description: Verify that empty list returns two empty int32 arrays
- Test function: test_resolve_variable_labels_states_only
- Description: Verify resolution of pure state labels
- Test function: test_resolve_variable_labels_observables_only
- Description: Verify resolution of pure observable labels
- Test function: test_resolve_variable_labels_mixed
- Description: Verify resolution of mixed state/observable labels
- Test function: test_resolve_variable_labels_invalid_raises
- Description: Verify ValueError for invalid labels when silent=False
- Test function: test_resolve_variable_labels_silent_mode
- Description: Verify silent mode returns empty for invalid labels
- Test function: test_merge_variable_inputs_all_none_returns_full
- Description: Verify all-None returns full range
- Test function: test_merge_variable_inputs_empty_labels
- Description: Verify empty labels with None indices returns empty arrays
- Test function: test_merge_variable_inputs_empty_indices
- Description: Verify None labels with empty indices respects empty
- Test function: test_merge_variable_inputs_union
- Description: Verify union of labels and indices
- Test function: test_merge_variable_inputs_deduplication
- Description: Verify duplicate indices are deduplicated
- Test function: test_convert_variable_labels_mutates_dict
- Description: Verify dict is mutated in-place with correct keys
- Test function: test_convert_variable_labels_pops_label_keys
- Description: Verify save_variables and summarise_variables are popped
- Test function: test_convert_variable_labels_summarised_defaults_to_saved
- Description: Verify summarised defaults to saved when all None

**Tests to Run**:
- tests/batchsolving/test_system_interface.py::test_resolve_variable_labels_none_returns_none_tuple
- tests/batchsolving/test_system_interface.py::test_resolve_variable_labels_empty_returns_empty_arrays
- tests/batchsolving/test_system_interface.py::test_resolve_variable_labels_states_only
- tests/batchsolving/test_system_interface.py::test_resolve_variable_labels_observables_only
- tests/batchsolving/test_system_interface.py::test_resolve_variable_labels_mixed
- tests/batchsolving/test_system_interface.py::test_resolve_variable_labels_invalid_raises
- tests/batchsolving/test_system_interface.py::test_resolve_variable_labels_silent_mode
- tests/batchsolving/test_system_interface.py::test_merge_variable_inputs_all_none_returns_full
- tests/batchsolving/test_system_interface.py::test_merge_variable_inputs_empty_labels
- tests/batchsolving/test_system_interface.py::test_merge_variable_inputs_empty_indices
- tests/batchsolving/test_system_interface.py::test_merge_variable_inputs_union
- tests/batchsolving/test_system_interface.py::test_merge_variable_inputs_deduplication
- tests/batchsolving/test_system_interface.py::test_convert_variable_labels_mutates_dict
- tests/batchsolving/test_system_interface.py::test_convert_variable_labels_pops_label_keys
- tests/batchsolving/test_system_interface.py::test_convert_variable_labels_summarised_defaults_to_saved

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 2: Update Solver to Delegate to SystemInterface
**Status**: [ ]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/batchsolving/solver.py (lines 1-420)
- File: src/cubie/batchsolving/SystemInterface.py (entire file after Task Group 1)
- File: tests/batchsolving/test_solver.py (lines 1-150) - Test patterns

**Input Validation Required**:
- No additional validation needed; delegation to SystemInterface handles validation

**Tasks**:
1. **Simplify Solver.convert_output_labels to delegate**
   - File: src/cubie/batchsolving/solver.py
   - Action: Modify method at lines 336-404
   - Details:
     ```python
     def convert_output_labels(
         self,
         output_settings: Dict[str, Any],
     ) -> None:
         """Convert variable labels to indices by delegating to SystemInterface.

         Parameters
         ----------
         output_settings
             Output configuration kwargs. Entries used are ``save_variables``,
             ``summarise_variables``, ``saved_state_indices``,
             ``saved_observable_indices``, ``summarised_state_indices``,
             and ``summarised_observable_indices``.

         Returns
         -------
         None
             Modifies ``output_settings`` in-place.

         Raises
         ------
         ValueError
             If variable labels are not recognized by the system.
         """
         self.system_interface.convert_variable_labels(
             output_settings,
             self.system_sizes.states,
             self.system_sizes.observables,
         )
     ```
   - Edge cases: None - all handled by SystemInterface
   - Integration: Replaces existing implementation

2. **Remove Solver._resolve_labels method**
   - File: src/cubie/batchsolving/solver.py
   - Action: Delete method at lines 292-315
   - Details: Remove the entire `_resolve_labels` method
   - Edge cases: None
   - Integration: Logic moved to SystemInterface.resolve_variable_labels()

3. **Remove Solver._merge_vars_and_indices method**
   - File: src/cubie/batchsolving/solver.py
   - Action: Delete method at lines 317-334
   - Details: Remove the entire `_merge_vars_and_indices` method
   - Edge cases: None
   - Integration: Logic moved to SystemInterface.merge_variable_inputs()

**Tests to Create**:
- Test file: tests/batchsolving/test_solver.py
- Test function: test_convert_output_labels_delegates_to_system_interface
- Description: Verify Solver.convert_output_labels calls SystemInterface method
- Test function: test_solver_with_empty_save_variables
- Description: Verify empty save_variables=[] results in no variables saved
- Test function: test_solver_with_empty_summarise_variables
- Description: Verify empty summarise_variables=[] results in no variables summarised
- Test function: test_solver_save_variables_and_indices_union
- Description: Verify union of save_variables and saved_*_indices

**Tests to Run**:
- tests/batchsolving/test_solver.py::test_convert_output_labels_delegates_to_system_interface
- tests/batchsolving/test_solver.py::test_solver_with_empty_save_variables
- tests/batchsolving/test_solver.py::test_solver_with_empty_summarise_variables
- tests/batchsolving/test_solver.py::test_solver_save_variables_and_indices_union
- tests/batchsolving/test_solver.py::test_solver_initialization
- tests/batchsolving/test_solver.py::test_saved_variables_properties
- tests/batchsolving/test_solver.py::test_summarised_variables_properties

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 3: Fix OutputConfig Empty Array Handling
**Status**: [ ]
**Dependencies**: Task Group 1, Task Group 2

**Required Context**:
- File: src/cubie/outputhandling/output_config.py (entire file)
- File: tests/outputhandling/test_output_config.py (entire file) - Test patterns
- File: .github/active_plans/consolidate_variable_interface/agent_plan.md (lines 75-100) - Behavioral spec

**Input Validation Required**:
- No additional validation needed; empty arrays should be preserved as empty, not expanded to full range

**Tasks**:
1. **Modify _check_saved_indices to preserve empty arrays**
   - File: src/cubie/outputhandling/output_config.py
   - Action: Modify method at lines 261-289
   - Details:
     ```python
     def _check_saved_indices(self) -> None:
         """Convert saved indices to numpy arrays.

         Returns
         -------
         None
             Returns ``None``.

         Notes
         -----
         Converts index collections to numpy int arrays. Empty arrays are
         preserved as empty; defaults are handled upstream in SystemInterface.
         """
         # Convert to numpy array, preserving empty as empty
         self._saved_state_indices = np.asarray(
             self._saved_state_indices, dtype=np.int_
         )
         self._saved_observable_indices = np.asarray(
             self._saved_observable_indices, dtype=np.int_
         )
     ```
   - Edge cases:
     - Empty list/array → preserved as empty numpy array
     - Non-empty list → converted to numpy array
   - Integration: Defaults now applied in SystemInterface, not here

2. **Modify _check_summarised_indices to preserve empty arrays**
   - File: src/cubie/outputhandling/output_config.py
   - Action: Modify method at lines 291-317
   - Details:
     ```python
     def _check_summarised_indices(self) -> None:
         """Convert summarised indices to numpy arrays.

         Returns
         -------
         None
             Returns ``None``.

         Notes
         -----
         Converts index collections to numpy int arrays. Empty arrays are
         preserved as empty; defaults are handled upstream in SystemInterface.
         """
         # Convert to numpy array, preserving empty as empty
         self._summarised_state_indices = np.asarray(
             self._summarised_state_indices, dtype=np.int_
         )
         self._summarised_observable_indices = np.asarray(
             self._summarised_observable_indices, dtype=np.int_
         )
     ```
   - Edge cases:
     - Empty list/array → preserved as empty numpy array
     - Non-empty list → converted to numpy array
   - Integration: "Summarised defaults to saved" now handled in SystemInterface

3. **Update from_loop_settings to not convert None to empty**
   - File: src/cubie/outputhandling/output_config.py
   - Action: Modify class method at lines 894-970
   - Details:
     ```python
     @classmethod
     def from_loop_settings(
         cls,
         output_types: List[str],
         precision: PrecisionDType,
         saved_state_indices: Union[Sequence[int], NDArray[np.int_], None] = None,
         saved_observable_indices: Union[Sequence[int], NDArray[np.int_], None] = None,
         summarised_state_indices: Union[Sequence[int], NDArray[np.int_], None] = None,
         summarised_observable_indices: Union[Sequence[int], NDArray[np.int_], None] = None,
         max_states: int = 0,
         max_observables: int = 0,
         dt_save: Optional[float] = 0.01,
     ) -> "OutputConfig":
         # ... docstring unchanged ...
         output_types = output_types.copy()

         # Convert None to empty arrays; SystemInterface has already applied defaults
         if saved_state_indices is None:
             saved_state_indices = np.asarray([], dtype=np.int_)
         if saved_observable_indices is None:
             saved_observable_indices = np.asarray([], dtype=np.int_)
         if summarised_state_indices is None:
             summarised_state_indices = np.asarray([], dtype=np.int_)
         if summarised_observable_indices is None:
             summarised_observable_indices = np.asarray([], dtype=np.int_)

         return cls(
             max_states=max_states,
             max_observables=max_observables,
             saved_state_indices=saved_state_indices,
             saved_observable_indices=saved_observable_indices,
             summarised_state_indices=summarised_state_indices,
             summarised_observable_indices=summarised_observable_indices,
             output_types=output_types,
             dt_save=dt_save,
             precision=precision,
         )
     ```
   - Edge cases: None - already handles None appropriately
   - Integration: None values from upstream converted to empty; SystemInterface provides defaults

**Tests to Create**:
- Test file: tests/outputhandling/test_output_config.py
- Test function: test_empty_saved_state_indices_preserved
- Description: Verify empty saved_state_indices stays empty (not expanded to full range)
- Test function: test_empty_saved_observable_indices_preserved
- Description: Verify empty saved_observable_indices stays empty
- Test function: test_empty_summarised_state_indices_preserved
- Description: Verify empty summarised_state_indices stays empty
- Test function: test_empty_summarised_observable_indices_preserved
- Description: Verify empty summarised_observable_indices stays empty
- Test function: test_from_loop_settings_preserves_empty_indices
- Description: Verify from_loop_settings preserves explicit empty arrays

**Tests to Run**:
- tests/outputhandling/test_output_config.py::test_empty_saved_state_indices_preserved
- tests/outputhandling/test_output_config.py::test_empty_saved_observable_indices_preserved
- tests/outputhandling/test_output_config.py::test_empty_summarised_state_indices_preserved
- tests/outputhandling/test_output_config.py::test_empty_summarised_observable_indices_preserved
- tests/outputhandling/test_output_config.py::test_from_loop_settings_preserves_empty_indices
- tests/outputhandling/test_output_config.py::TestInitialization
- tests/outputhandling/test_output_config.py::TestFromLoopSettings

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 4: Integration Testing and Edge Cases
**Status**: [ ]
**Dependencies**: Task Group 1, Task Group 2, Task Group 3

**Required Context**:
- File: src/cubie/batchsolving/solver.py (entire file after Task Groups 1-3)
- File: src/cubie/batchsolving/SystemInterface.py (entire file after Task Group 1)
- File: src/cubie/outputhandling/output_config.py (entire file after Task Group 3)
- File: tests/batchsolving/test_solver.py (entire file)
- File: tests/batchsolving/conftest.py (entire file) - Fixture patterns

**Input Validation Required**:
- Test all combinations from truth table in agent_plan.md

**Tasks**:
1. **Add integration tests for Solver variable handling**
   - File: tests/batchsolving/test_solver.py
   - Action: Add new test class at end of file
   - Details:
     ```python
     class TestVariableResolutionIntegration:
         """Integration tests for variable resolution through Solver."""

         def test_none_inputs_default_to_all(self, solver):
             """Test that None for both labels and indices saves all variables."""
             # When save_variables=None and saved_*_indices=None
             # Result should be all states and observables saved
             pass

         def test_empty_labels_explicit_none(self, solver, system):
             """Test that empty save_variables=[] means no variables."""
             # Create solver with save_variables=[]
             # Verify saved_state_indices and saved_observable_indices are empty
             pass

         def test_empty_indices_explicit_none(self, system):
             """Test that empty indices means no variables for that type."""
             # Create solver with saved_state_indices=[]
             # Verify saved_state_indices is empty
             pass

         def test_labels_and_indices_union(self, solver, system):
             """Test union of labels and indices."""
             # Provide both save_variables and saved_state_indices
             # Verify union is computed correctly
             pass

         def test_summarised_defaults_to_saved(self, system):
             """Test summarised defaults to saved when not specified."""
             # Provide save_variables but no summarise_variables
             # Verify summarised indices match saved indices
             pass

         def test_explicit_empty_summarised_independent_of_saved(self, system):
             """Test explicit empty summarised is independent of saved."""
             # Provide save_variables=["x"] and summarise_variables=[]
             # Verify summarised is empty while saved has x
             pass
     ```
   - Edge cases: All cases from truth table in agent_plan.md
   - Integration: Tests full pipeline from Solver through SystemInterface to OutputConfig

2. **Add edge case tests for systems with no observables**
   - File: tests/batchsolving/test_solver.py
   - Action: Add tests to TestVariableResolutionIntegration class
   - Details:
     ```python
         def test_system_no_observables_default(self, system_no_observables):
             """Test default behavior with system having no observables."""
             # Verify observable_indices is empty array, not error
             pass

         def test_system_no_observables_with_labels(self, system_no_observables):
             """Test label resolution with system having no observables."""
             # Providing observable labels should work but resolve to empty
             pass
     ```
   - Edge cases: System with max_observables=0
   - Integration: Requires fixture for system with no observables

3. **Update existing tests that may rely on old behavior**
   - File: tests/outputhandling/test_output_config.py
   - Action: Review and update tests in TestInitialization and TestFromLoopSettings
   - Details: 
     - Tests expecting empty indices to expand to full range must be updated
     - `test_none_indices_conversion` specifically tests old behavior
   - Edge cases: Tests that verify "empty expands to all" must be changed
   - Integration: Ensure backward compatibility for users not passing indices

**Tests to Create**:
- Test file: tests/batchsolving/test_solver.py
- Test function: TestVariableResolutionIntegration::test_none_inputs_default_to_all
- Description: Verify None labels and indices defaults to all variables
- Test function: TestVariableResolutionIntegration::test_empty_labels_explicit_none
- Description: Verify empty save_variables=[] results in no variables
- Test function: TestVariableResolutionIntegration::test_empty_indices_explicit_none
- Description: Verify empty indices means no variables for that type
- Test function: TestVariableResolutionIntegration::test_labels_and_indices_union
- Description: Verify union of labels and indices
- Test function: TestVariableResolutionIntegration::test_summarised_defaults_to_saved
- Description: Verify summarised defaults to saved when not specified
- Test function: TestVariableResolutionIntegration::test_explicit_empty_summarised_independent_of_saved
- Description: Verify explicit empty summarised is independent of saved
- Test function: TestVariableResolutionIntegration::test_system_no_observables_default
- Description: Verify systems with no observables work correctly

**Tests to Run**:
- tests/batchsolving/test_solver.py::TestVariableResolutionIntegration
- tests/batchsolving/test_solver.py::test_saved_variables_properties
- tests/batchsolving/test_solver.py::test_summarised_variables_properties
- tests/batchsolving/test_solver.py::test_variable_indices_methods
- tests/outputhandling/test_output_config.py::TestInitialization
- tests/outputhandling/test_output_config.py::TestFromLoopSettings

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 5: Update Documentation and Cleanup
**Status**: [ ]
**Dependencies**: Task Group 1, Task Group 2, Task Group 3, Task Group 4

**Required Context**:
- File: src/cubie/batchsolving/SystemInterface.py (entire file after all changes)
- File: src/cubie/batchsolving/solver.py (entire file after all changes)
- File: src/cubie/outputhandling/output_config.py (entire file after all changes)
- File: .github/active_plans/consolidate_variable_interface/human_overview.md (lines 37-60) - Data flow diagrams

**Input Validation Required**:
- None - documentation only

**Tasks**:
1. **Update Solver class docstring**
   - File: src/cubie/batchsolving/solver.py
   - Action: Update class docstring (lines 166-206)
   - Details:
     ```
     Add note about variable resolution being delegated to SystemInterface.
     Clarify that None means "use all" while [] means "use none".
     ```
   - Edge cases: None
   - Integration: Documentation for users

2. **Verify solve_ivp docstring accuracy**
   - File: src/cubie/batchsolving/solver.py
   - Action: Review docstring (lines 51-124)
   - Details:
     ```
     Ensure save_variables and summarise_variables documentation
     accurately reflects the new None vs empty behavior.
     ```
   - Edge cases: None
   - Integration: Documentation for users

3. **Add SystemInterface docstring notes**
   - File: src/cubie/batchsolving/SystemInterface.py
   - Action: Update class docstring (lines 23-41)
   - Details:
     ```
     Add notes about new variable resolution methods and their role
     as the single source of truth for variable label handling.
     ```
   - Edge cases: None
   - Integration: Documentation for maintainers

**Tests to Create**:
- None - documentation changes only

**Tests to Run**:
- Run full test suite to verify no regressions:
  - tests/batchsolving/test_solver.py
  - tests/batchsolving/test_system_interface.py
  - tests/outputhandling/test_output_config.py

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

# Summary

## Total Task Groups: 5

## Dependency Chain:
```
Task Group 1 (SystemInterface methods)
    ↓
Task Group 2 (Solver delegation)
    ↓
Task Group 3 (OutputConfig fixes)
    ↓
Task Group 4 (Integration tests)
    ↓
Task Group 5 (Documentation)
```

## Tests to be Created:
- 15 unit tests for SystemInterface new methods
- 4 tests for Solver delegation
- 5 tests for OutputConfig empty handling
- 7+ integration tests for full pipeline

## Estimated Complexity:
- Task Group 1: Medium (3 new methods with detailed logic)
- Task Group 2: Low (simplification/deletion)
- Task Group 3: Low (simplification)
- Task Group 4: Medium (integration testing)
- Task Group 5: Low (documentation)

## Key Behavioral Changes:
1. `None` inputs → "use all" (default behavior)
2. `[]` or empty array → "explicitly no variables"
3. Union of labels and indices when both provided
4. Single source of truth in SystemInterface
5. OutputConfig no longer expands empty to full range
