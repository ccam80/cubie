# Implementation Task List
# Feature: BatchGridBuilder Complete Refactoring
# Plan Reference: .github/active_plans/batch_grid_refactor/agent_plan.md

## Overview

This refactoring eliminates the combined dictionary pattern in BatchGridBuilder, keeping `params` and `states` separate throughout processing. The key changes are:

1. Remove `grid_arrays()` method entirely
2. Add `_process_input()` helper for unified input processing
3. Add `_align_run_counts()` helper for final alignment
4. Refactor `__call__()` to use separate processing paths
5. Remove static method wrappers (tests import module directly)
6. Update tests to reflect new internal structure

---

## Task Group 1: Add New Private Helper Methods
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: src/cubie/batchsolving/BatchGridBuilder.py (lines 1-106, 350-429, 649-763)
- File: .github/context/cubie_internal_structure.md (entire file for conventions)

**Input Validation Required**:
- `input_data` in `_process_input`: Check if None, dict, or array-like (list, tuple, np.ndarray)
- `kind` parameter: Validate is "combinatorial" or "verbatim"
- Type errors for unsupported input types

**Tasks**:
1. **Add `_process_input()` method**
   - File: src/cubie/batchsolving/BatchGridBuilder.py
   - Action: Create new method after `_sanitise_arraylike()` (around line 740)
   - Details:
     ```python
     def _process_input(
         self,
         input_data: Optional[Union[Dict, ArrayLike]],
         values_object: SystemValues,
         kind: str,
     ) -> np.ndarray:
         """Process a single input category to a 2D array.
         
         Handles None, dict, or array-like inputs for either params
         or states, returning a complete 2D array in (variable, run)
         format with all variables included.
         
         Parameters
         ----------
         input_data
             Input as None (use defaults), dict (expand to grid), 
             or array-like (sanitize).
         values_object
             SystemValues instance for this category (params or states).
         kind
             Grid type: "combinatorial" or "verbatim".
             
         Returns
         -------
         np.ndarray
             2D array in (variable, run) format with all variables.
         """
         # None -> single-column defaults
         if input_data is None:
             return values_object.values_array[:, np.newaxis]
         
         # Dict -> expand to grid, extend with defaults
         if isinstance(input_data, dict):
             indices, grid = generate_grid(input_data, values_object, kind=kind)
             return extend_grid_to_array(
                 grid, indices, values_object.values_array
             )
         
         # Array-like -> sanitize to 2D
         if isinstance(input_data, (list, tuple, np.ndarray)):
             return self._sanitise_arraylike(input_data, values_object)
         
         # Unsupported type
         raise TypeError(
             f"Input must be None, dict, or array-like, got {type(input_data)}"
         )
     ```
   - Edge cases:
     - Empty dict `{}` produces single-run defaults (handled by generate_grid)
     - Dict with empty values `{"p0": []}` filters empty before grid generation
     - 1D array converted to column vector by `_sanitise_arraylike`
   - Integration: Called by refactored `__call__()` for each input category

2. **Add `_align_run_counts()` method**
   - File: src/cubie/batchsolving/BatchGridBuilder.py
   - Action: Create new method after `_process_input()` 
   - Details:
     ```python
     def _align_run_counts(
         self,
         states_array: np.ndarray,
         params_array: np.ndarray,
         kind: str,
     ) -> tuple[np.ndarray, np.ndarray]:
         """Align run counts between states and params arrays.
         
         For combinatorial: computes Cartesian product of runs.
         For verbatim: pairs directly (single-run broadcasts).
         
         Parameters
         ----------
         states_array
             States in (variable, run) format.
         params_array
             Params in (variable, run) format.
         kind
             Grid type: "combinatorial" or "verbatim".
             
         Returns
         -------
         tuple[np.ndarray, np.ndarray]
             Aligned (states_array, params_array) with matching run counts.
         """
         return combine_grids(states_array, params_array, kind=kind)
     ```
   - Edge cases:
     - Both single-run -> no expansion needed for either kind
     - Verbatim mismatch raises ValueError (handled by combine_grids)
   - Integration: Called by refactored `__call__()` as final alignment step

**Tests to Create**:
- Test file: tests/batchsolving/test_batch_grid_builder.py
- Test function: test_process_input_none
- Description: Verify _process_input with None returns single-column defaults
- Test function: test_process_input_dict_combinatorial
- Description: Verify _process_input with dict expands correctly for combinatorial
- Test function: test_process_input_dict_verbatim
- Description: Verify _process_input with dict expands correctly for verbatim
- Test function: test_process_input_array
- Description: Verify _process_input with array sanitizes correctly
- Test function: test_process_input_invalid_type
- Description: Verify _process_input raises TypeError for invalid input
- Test function: test_align_run_counts_combinatorial
- Description: Verify _align_run_counts produces Cartesian product
- Test function: test_align_run_counts_verbatim
- Description: Verify _align_run_counts pairs directly with broadcast

**Tests to Run**:
- tests/batchsolving/test_batch_grid_builder.py::test_process_input_none
- tests/batchsolving/test_batch_grid_builder.py::test_process_input_dict_combinatorial
- tests/batchsolving/test_batch_grid_builder.py::test_process_input_dict_verbatim
- tests/batchsolving/test_batch_grid_builder.py::test_process_input_array
- tests/batchsolving/test_batch_grid_builder.py::test_process_input_invalid_type
- tests/batchsolving/test_batch_grid_builder.py::test_align_run_counts_combinatorial
- tests/batchsolving/test_batch_grid_builder.py::test_align_run_counts_verbatim

**Outcomes**: 

---

## Task Group 2: Refactor `__call__()` Method
**Status**: [ ]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/batchsolving/BatchGridBuilder.py (lines 519-648, new methods from Group 1)
- File: src/cubie/batchsolving/BatchGridBuilder.py (lines 291-348 for combine_grids)
- File: .github/active_plans/batch_grid_refactor/agent_plan.md (Expected Behavior section)

**Input Validation Required**:
- `kind`: Must be "combinatorial" or "verbatim" (delegated to combine_grids)
- `params`: None, dict, or array-like (delegated to _process_input)
- `states`: None, dict, or array-like (delegated to _process_input)

**Tasks**:
1. **Refactor `__call__()` to use separate processing paths**
   - File: src/cubie/batchsolving/BatchGridBuilder.py
   - Action: Replace lines 519-648 with new implementation
   - Details:
     ```python
     def __call__(
         self,
         params: Optional[Union[Dict, ArrayLike]] = None,
         states: Optional[Union[Dict, ArrayLike]] = None,
         kind: str = "combinatorial",
     ) -> tuple[np.ndarray, np.ndarray]:
         """Process user input to generate parameter and state arrays.

         Parameters
         ----------
         params
             Optional dictionary or array describing parameter sweeps.
         states
             Optional dictionary or array describing initial state sweeps.
         kind
             Strategy for grid assembly. ``"combinatorial"`` expands
             all combinations while ``"verbatim"`` preserves pairings.

         Returns
         -------
         tuple[np.ndarray, np.ndarray]
             Initial state and parameter arrays aligned for batch execution.

         Notes
         -----
         Passing ``params`` and ``states`` as arrays treats each as a
         complete grid. ``kind="combinatorial"`` computes the Cartesian
         product of both grids. When arrays already describe paired runs,
         set ``kind`` to ``"verbatim"`` to keep them aligned.
         """
         # Update precision from current system state
         self.precision = self.states.precision

         # Process each category independently
         states_array = self._process_input(states, self.states, kind)
         params_array = self._process_input(params, self.parameters, kind)

         # Align run counts
         states_array, params_array = self._align_run_counts(
             states_array, params_array, kind
         )

         # Cast to system precision
         return self._cast_to_precision(states_array, params_array)
     ```
   - Edge cases:
     - Both None: Each produces single-column defaults, alignment is no-op
     - One None, one dict: None produces defaults, dict expands, then align
     - Both arrays: Sanitize both, then align per kind
     - Mixed dict/array: Process each appropriately, then align
   - Integration: This replaces the entire existing `__call__()` implementation

**Tests to Create**:
- None (existing tests cover the public API behavior)

**Tests to Run**:
- tests/batchsolving/test_batch_grid_builder.py::test_call_with_request
- tests/batchsolving/test_batch_grid_builder.py::test_call_input_types
- tests/batchsolving/test_batch_grid_builder.py::test_call_outputs
- tests/batchsolving/test_batch_grid_builder.py::test_docstring_examples
- tests/batchsolving/test_batch_grid_builder.py::test_empty_inputs_returns_defaults
- tests/batchsolving/test_batch_grid_builder.py::test_single_param_dict_sweep
- tests/batchsolving/test_batch_grid_builder.py::test_single_state_dict_single_run
- tests/batchsolving/test_batch_grid_builder.py::test_states_dict_params_sweep
- tests/batchsolving/test_batch_grid_builder.py::test_combinatorial_states_params
- tests/batchsolving/test_batch_grid_builder.py::test_1d_param_array_single_run
- tests/batchsolving/test_batch_grid_builder.py::test_1d_state_array_partial_warning
- tests/batchsolving/test_batch_grid_builder.py::test_verbatim_single_run_broadcast

**Outcomes**: 

---

## Task Group 3: Remove Deprecated Code
**Status**: [ ]
**Dependencies**: Task Group 2

**Required Context**:
- File: src/cubie/batchsolving/BatchGridBuilder.py (lines 475-517 for grid_arrays, lines 765-826 for static wrappers)
- File: tests/batchsolving/test_batch_grid_builder.py (lines 1-10 for imports, lines 87-95 for test_grid_arrays)

**Input Validation Required**:
- None (removal only)

**Tasks**:
1. **Remove `grid_arrays()` method**
   - File: src/cubie/batchsolving/BatchGridBuilder.py
   - Action: Delete lines 475-517 (the entire `grid_arrays` method)
   - Details: This method exists solely to process a combined dictionary and is no longer needed with the refactored `__call__()`.
   - Edge cases: None
   - Integration: Tests that call `grid_arrays()` directly must be removed or updated

2. **Remove static method wrappers**
   - File: src/cubie/batchsolving/BatchGridBuilder.py
   - Action: Delete lines 765-826 (all static method wrappers and their comment block starting at line 765)
   - Details: 
     - Remove `unique_cartesian_product` static wrapper
     - Remove `combinatorial_grid` static wrapper
     - Remove `verbatim_grid` static wrapper
     - Remove `generate_grid` static wrapper
     - Remove `combine_grids` static wrapper
     - Remove `extend_grid_to_array` static wrapper
     - Remove `generate_array` static wrapper
     - Remove the explanatory comment block (lines 765-776)
   - Edge cases: Tests import module directly (`import cubie.batchsolving.BatchGridBuilder as batchgridmodule`) so this is safe
   - Integration: Verify tests use module-level functions, not class static methods

**Tests to Create**:
- None (removal only)

**Tests to Run**:
- tests/batchsolving/test_batch_grid_builder.py (full file to verify nothing breaks)

**Outcomes**: 

---

## Task Group 4: Update Tests for Removed Methods
**Status**: [ ]
**Dependencies**: Task Group 3

**Required Context**:
- File: tests/batchsolving/test_batch_grid_builder.py (entire file)
- File: src/cubie/batchsolving/BatchGridBuilder.py (refactored version)

**Input Validation Required**:
- None (test updates only)

**Tasks**:
1. **Remove `test_grid_arrays` test**
   - File: tests/batchsolving/test_batch_grid_builder.py
   - Action: Delete lines 87-95 (the `test_grid_arrays` function)
   - Details: This test directly calls the removed `grid_arrays()` method
   - Edge cases: None
   - Integration: The behavior is now tested via `__call__()` tests

2. **Remove `test_grid_arrays_1each` test**
   - File: tests/batchsolving/test_batch_grid_builder.py
   - Action: Delete lines 243-276 (parametrized test using `grid_arrays`)
   - Details: This test directly calls the removed `grid_arrays()` method
   - Edge cases: None
   - Integration: The behavior is now tested via `__call__()` tests

3. **Remove `test_grid_arrays_2and2` test**
   - File: tests/batchsolving/test_batch_grid_builder.py
   - Action: Delete lines 278-318 (parametrized test using `grid_arrays`)
   - Details: This test directly calls the removed `grid_arrays()` method
   - Edge cases: None
   - Integration: The behavior is now tested via `__call__()` tests

4. **Remove `test_grid_arrays_verbatim_mismatch` test**
   - File: tests/batchsolving/test_batch_grid_builder.py
   - Action: Delete lines 320-345 (parametrized test using `grid_arrays`)
   - Details: This test directly calls the removed `grid_arrays()` method
   - Edge cases: None
   - Integration: Mismatch behavior tested via `__call__()` with verbatim kind

5. **Remove `test_grid_arrays_empty_inputs` test**
   - File: tests/batchsolving/test_batch_grid_builder.py
   - Action: Delete lines 347-388 (parametrized test using `grid_arrays`)
   - Details: This test directly calls the removed `grid_arrays()` method
   - Edge cases: None
   - Integration: Empty input behavior tested via existing `__call__()` tests

6. **Add replacement tests for removed grid_arrays functionality**
   - File: tests/batchsolving/test_batch_grid_builder.py
   - Action: Add new tests that verify the same behaviors through `__call__()`
   - Details:
     ```python
     def test_call_combinatorial_1each(grid_builder, system):
         """Test combinatorial expansion with 1 state and 1 param swept."""
         state_names = list(system.initial_values.names)
         param_names = list(system.parameters.names)
         states = {state_names[0]: [0, 1]}
         params = {param_names[1]: np.arange(10)}
         inits, params_out = grid_builder(
             params=params, states=states, kind="combinatorial"
         )
         assert inits.shape == (system.sizes.states, 20)
         assert params_out.shape == (system.sizes.parameters, 20)


     def test_call_verbatim_mismatch_raises(grid_builder, system):
         """Test verbatim with mismatched lengths raises ValueError."""
         state_names = list(system.initial_values.names)
         param_names = list(system.parameters.names)
         states = {state_names[0]: [0, 1, 2]}
         params = {param_names[0]: [0, 1]}
         with pytest.raises(ValueError):
             grid_builder(params=params, states=states, kind="verbatim")


     def test_call_empty_dict_values(grid_builder, system):
         """Test that empty dict values are filtered correctly."""
         state_names = list(system.initial_values.names)
         param_names = list(system.parameters.names)
         states = {state_names[0]: [0, 1, 2], state_names[1]: []}
         params = {param_names[0]: np.arange(3), param_names[1]: np.array([])}
         inits, params_out = grid_builder(
             params=params, states=states, kind="verbatim"
         )
         assert inits.shape[1] == 3
         assert params_out.shape[1] == 3
     ```
   - Edge cases: Covers all behaviors previously tested via `grid_arrays()`
   - Integration: These tests use the public API

**Tests to Create**:
- Test file: tests/batchsolving/test_batch_grid_builder.py
- Test function: test_call_combinatorial_1each
- Description: Verify combinatorial expansion with 1 state and 1 param swept
- Test function: test_call_verbatim_mismatch_raises
- Description: Verify verbatim with mismatched lengths raises ValueError
- Test function: test_call_empty_dict_values
- Description: Verify empty dict values are filtered correctly

**Tests to Run**:
- tests/batchsolving/test_batch_grid_builder.py (full file)

**Outcomes**: 

---

## Task Group 5: Update Module Docstring
**Status**: [ ]
**Dependencies**: Task Group 4

**Required Context**:
- File: src/cubie/batchsolving/BatchGridBuilder.py (lines 1-94 module docstring)
- File: .github/active_plans/batch_grid_refactor/human_overview.md (Architecture Overview)

**Input Validation Required**:
- None (documentation only)

**Tasks**:
1. **Update module docstring to reflect new architecture**
   - File: src/cubie/batchsolving/BatchGridBuilder.py
   - Action: Update lines 1-94 to describe the separate processing architecture
   - Details:
     - Remove references to "combined request dictionary"
     - Add note about parallel processing paths for params and states
     - Keep examples unchanged (they test the public API which is unchanged)
     - Update the Notes section to describe the new flow:
       ```
       Notes
       -----
       ``BatchGridBuilder.__call__`` processes params and states through
       independent paths, combining only at the final alignment step:
       
       1. Each input is processed via ``_process_input()`` to produce
          a 2D array in (variable, run) format
       2. Arrays are aligned via ``_align_run_counts()`` using the
          specified ``kind`` strategy
       3. Results are cast to system precision
       
       This architecture keeps params and states separate throughout,
       improving code clarity and reducing unnecessary transformations.
       ```
   - Edge cases: None
   - Integration: Documentation only, no functional changes

**Tests to Create**:
- None (documentation only)

**Tests to Run**:
- tests/batchsolving/test_batch_grid_builder.py::test_docstring_examples

**Outcomes**: 

---

## Task Group 6: Final Validation and Solver Integration
**Status**: [ ]
**Dependencies**: Task Group 5

**Required Context**:
- File: src/cubie/batchsolving/solver.py (lines 480-568)
- File: src/cubie/batchsolving/BatchGridBuilder.py (entire refactored file)
- File: tests/batchsolving/test_batch_grid_builder.py (entire file)

**Input Validation Required**:
- None (validation only)

**Tasks**:
1. **Verify solver.py call sites work unchanged**
   - File: src/cubie/batchsolving/solver.py
   - Action: Review (no changes expected)
   - Details:
     - Line 489-491: `self.grid_builder(states=initial_values, params=parameters, kind=grid_type)` - unchanged API
     - Line 566-568: `self.grid_builder(states=initial_values, params=parameters, kind=grid_type)` - unchanged API
   - Edge cases: None
   - Integration: The public API is unchanged, solver.py requires no modifications

2. **Run full test suite for batchsolving**
   - File: tests/batchsolving/
   - Action: Run all tests in the batchsolving directory
   - Details: Verify no regressions in solver integration
   - Edge cases: None
   - Integration: Full integration test

**Tests to Create**:
- None (validation only)

**Tests to Run**:
- tests/batchsolving/test_batch_grid_builder.py (full file)
- tests/batchsolving/test_solver.py (if exists, for integration)

**Outcomes**: 

---

## Summary

| Task Group | Description | Estimated Changes |
|------------|-------------|-------------------|
| 1 | Add new private helper methods | +50 lines |
| 2 | Refactor `__call__()` method | -100 lines, +30 lines |
| 3 | Remove deprecated code | -95 lines |
| 4 | Update tests for removed methods | -100 lines, +30 lines |
| 5 | Update module docstring | ~10 line edits |
| 6 | Final validation | 0 lines |

**Net result**: Approximately 185 fewer lines of code in the module.

## Dependency Chain

```
Group 1 (Add helpers)
    ↓
Group 2 (Refactor __call__)
    ↓
Group 3 (Remove deprecated)
    ↓
Group 4 (Update tests)
    ↓
Group 5 (Update docs)
    ↓
Group 6 (Final validation)
```

## Validation Criteria

1. ✓ All existing public API tests pass
2. ✓ `__call__()` never combines params and states into single dict
3. ✓ Data flow is traceable: params in → params out, states in → states out
4. ✓ No new public API (only internal restructuring)
5. ✓ Code line count is reduced from current implementation
6. ✓ `grid_arrays()` method is completely removed
7. ✓ Static method wrappers are completely removed
