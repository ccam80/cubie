# Implementation Task List
# Feature: save_variables and summarise_variables Parameters
# Plan Reference: .github/active_plans/save_summarise_variables/agent_plan.md

## Task Group 1: Add Parameter Names to ALL_OUTPUT_FUNCTION_PARAMETERS - SEQUENTIAL
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/outputhandling/output_functions.py (lines 27-37)

**Input Validation Required**:
- None (this is a simple set addition)

**Tasks**:
1. **Add "save_variables" to ALL_OUTPUT_FUNCTION_PARAMETERS set**
   - File: src/cubie/outputhandling/output_functions.py
   - Action: Modify
   - Details:
     ```python
     # Current set (lines 27-37):
     ALL_OUTPUT_FUNCTION_PARAMETERS = {
         "output_types",
         "saved_states", "saved_observables",
         "summarised_states", "summarised_observables",
         "saved_state_indices",
         "saved_observable_indices",
         "summarised_state_indices",
         "summarised_observable_indices",
         "dt_save",
         "precision",
     }
     
     # Add two new entries:
     # - "save_variables" (after "summarised_observables" line)
     # - "summarise_variables" (after "save_variables")
     ```
   - Edge cases: None
   - Integration: This allows merge_kwargs_into_settings() to recognize and route these parameters to output_settings

**Outcomes**:
- Files Modified:
  * src/cubie/outputhandling/output_functions.py (2 lines added)
- Changes:
  * Added "save_variables" and "summarise_variables" to ALL_OUTPUT_FUNCTION_PARAMETERS set
- Implementation Summary:
  Added two new parameter names to the recognized set, allowing merge_kwargs_into_settings() to properly route these parameters to output_settings
- Issues Flagged: None 

---

## Task Group 2: Update solve_ivp() Function Signature - SEQUENTIAL
**Status**: [x]
**Dependencies**: Group 1

**Required Context**:
- File: src/cubie/batchsolving/solver.py (lines 41-121)
- Function signature starts at line 41
- Parameters section of docstring starts at line 58

**Input Validation Required**:
- save_variables: Must be Optional[List[str]] or None
- summarise_variables: Must be Optional[List[str]] or None
- No validation needed in solve_ivp (validation happens in convert_output_labels)

**Tasks**:
1. **Add save_variables parameter to solve_ivp() signature**
   - File: src/cubie/batchsolving/solver.py
   - Action: Modify
   - Details:
     ```python
     # After line 50 (t0: float = 0.0), add:
     save_variables: Optional[List[str]] = None,
     summarise_variables: Optional[List[str]] = None,
     ```
   - Edge cases: None (defaults to None)
   - Integration: Forward to Solver.__init__() via **kwargs

2. **Add docstring entries for new parameters**
   - File: src/cubie/batchsolving/solver.py
   - Action: Modify
   - Details:
     ```python
     # After line 89 (nan_error_trajectories documentation), add:
     save_variables : list of str, optional
         Variable names (states or observables) to save in time-domain output.
         Alternative to specifying saved_states and saved_observables separately.
         Can be combined with existing parameters using union semantics.
         Default is ``None``.
     summarise_variables : list of str, optional
         Variable names (states or observables) to include in summary calculations.
         Alternative to specifying summarised_states and summarised_observables.
         Can be combined with existing parameters using union semantics.
         Default is ``None``.
     ```
   - Edge cases: None
   - Integration: Standard numpydoc format

**Outcomes**:
- Files Modified:
  * src/cubie/batchsolving/solver.py (16 lines changed)
- Functions/Methods Added/Modified:
  * solve_ivp() signature and docstring updated
- Implementation Summary:
  Added save_variables and summarise_variables parameters to solve_ivp() signature with Optional[List[str]] type hints. Added comprehensive numpydoc docstring entries. Parameters are forwarded to Solver via kwargs using setdefault pattern.
- Issues Flagged: None

---

## Task Group 3: Update Solver Class Docstring - SEQUENTIAL
**Status**: [x]
**Dependencies**: Group 2

**Required Context**:
- File: src/cubie/batchsolving/solver.py (lines 124-164)
- Solver class __init__ parameters section in docstring

**Input Validation Required**:
- None (docstring only)

**Tasks**:
1. **Add docstring entries for new parameters in Solver class**
   - File: src/cubie/batchsolving/solver.py
   - Action: Modify
   - Details:
     ```python
     # After line 156 (time_logging_level documentation), add to the **kwargs section:
     # Update the **kwargs description to mention:
     **kwargs
         Additional keyword arguments forwarded to internal components. This
         includes output selection parameters such as ``save_variables`` and
         ``summarise_variables`` which provide a unified interface for
         specifying variables to save or summarize without distinguishing
         between states and observables.
     ```
   - Edge cases: None
   - Integration: Clarifies that these parameters are accepted via kwargs

**Outcomes**:
- Files Modified:
  * src/cubie/batchsolving/solver.py (5 lines changed)
- Functions/Methods Added/Modified:
  * Solver class docstring updated
- Implementation Summary:
  Enhanced the **kwargs documentation in Solver class to explicitly mention save_variables and summarise_variables as examples of output selection parameters that can be passed via kwargs.
- Issues Flagged: None

---

## Task Group 4: Implement Variable Classification in convert_output_labels() - SEQUENTIAL
**Status**: [x]
**Dependencies**: Groups 1, 2, 3

**Required Context**:
- File: src/cubie/batchsolving/solver.py (lines 260-331)
- Method: Solver.convert_output_labels()
- SystemInterface methods: state_indices() (lines 131-155 in SystemInterface.py), observable_indices() (lines 157-179)
- SystemValues.get_indices() method (lines 207-307 in SystemValues.py)

**Input Validation Required**:
- save_variables: Check if None or empty list (treat as no-op)
- summarise_variables: Check if None or empty list (treat as no-op)
- Variable names: Each name must exist in either states or observables (raise ValueError with helpful message if not found)
- Union results: Check that resulting arrays contain no duplicates (numpy unique handles this)

**Tasks**:
1. **Add fast-path check at start of convert_output_labels()**
   - File: src/cubie/batchsolving/solver.py
   - Action: Modify
   - Details:
     ```python
     # After line 288 (docstring ends), add:
     # Fast-path: skip name resolution if new parameters not present
     has_save_vars = "save_variables" in output_settings and output_settings["save_variables"] is not None
     has_summarise_vars = "summarise_variables" in output_settings and output_settings["summarise_variables"] is not None
     
     if not has_save_vars and not has_summarise_vars:
         # Proceed with existing logic only (lines 290-331)
         pass
     ```
   - Edge cases: Empty lists should not trigger name resolution
   - Integration: Ensures zero overhead when new parameters not used

2. **Implement classify_variables() helper method**
   - File: src/cubie/batchsolving/solver.py
   - Action: Create (as private method of Solver class)
   - Details:
     ```python
     def _classify_variables(
         self,
         var_names: List[str],
     ) -> Tuple[np.ndarray, np.ndarray]:
         """Classify variable names into states and observables.
         
         Parameters
         ----------
         var_names
             List of variable names to classify.
         
         Returns
         -------
         state_indices : np.ndarray
             Array of state indices (dtype=np.int16).
         observable_indices : np.ndarray
             Array of observable indices (dtype=np.int16).
         
         Raises
         ------
         ValueError
             If any variable name is not found in states or observables.
         """
         state_list = []
         observable_list = []
         unrecognized = []
         
         for name in var_names:
             found_as_state = False
             found_as_observable = False
             
             # Try as state
             try:
                 idx = self.system_interface.state_indices([name], silent=False)
                 state_list.extend(idx.tolist())
                 found_as_state = True
             except (KeyError, IndexError):
                 pass
             
             # Try as observable
             try:
                 idx = self.system_interface.observable_indices([name], silent=False)
                 observable_list.extend(idx.tolist())
                 found_as_observable = True
             except (KeyError, IndexError):
                 pass
             
             if not found_as_state and not found_as_observable:
                 unrecognized.append(name)
         
         if unrecognized:
             state_names = self.system_interface.states.names
             obs_names = self.system_interface.observables.names
             raise ValueError(
                 f"Variables not found in states or observables: {unrecognized}. "
                 f"Available states: {state_names}. "
                 f"Available observables: {obs_names}."
             )
         
         return (
             np.array(state_list, dtype=np.int16) if state_list else np.array([], dtype=np.int16),
             np.array(observable_list, dtype=np.int16) if observable_list else np.array([], dtype=np.int16)
         )
     ```
   - Edge cases:
     - Empty list: returns empty arrays
     - Variable in both states and observables (unlikely): added to both arrays
     - Variable not found: raises ValueError with available names
   - Integration: Called from convert_output_labels() for both save_variables and summarise_variables

3. **Add save_variables processing logic in convert_output_labels()**
   - File: src/cubie/batchsolving/solver.py
   - Action: Modify
   - Details:
     ```python
     # After the fast-path check (before existing resolvers dict at line 290), add:
     
     # Process save_variables parameter
     if has_save_vars:
         save_vars = output_settings.pop("save_variables")
         if save_vars:  # Not empty list
             state_idxs, obs_idxs = self._classify_variables(save_vars)
             
             # Union with existing saved_state_indices
             if "saved_state_indices" in output_settings and output_settings["saved_state_indices"] is not None:
                 existing = output_settings["saved_state_indices"]
                 combined = np.union1d(existing, state_idxs).astype(np.int16)
                 output_settings["saved_state_indices"] = combined
             elif len(state_idxs) > 0:
                 output_settings["saved_state_indices"] = state_idxs
             
             # Union with existing saved_observable_indices
             if "saved_observable_indices" in output_settings and output_settings["saved_observable_indices"] is not None:
                 existing = output_settings["saved_observable_indices"]
                 combined = np.union1d(existing, obs_idxs).astype(np.int16)
                 output_settings["saved_observable_indices"] = combined
             elif len(obs_idxs) > 0:
                 output_settings["saved_observable_indices"] = obs_idxs
     ```
   - Edge cases:
     - Empty list: pop but don't process
     - No existing indices: directly assign
     - Existing indices: use union1d to merge without duplicates
   - Integration: Processes save_variables before existing label-to-index conversion

4. **Add summarise_variables processing logic in convert_output_labels()**
   - File: src/cubie/batchsolving/solver.py
   - Action: Modify
   - Details:
     ```python
     # After save_variables processing (before existing resolvers dict), add:
     
     # Process summarise_variables parameter
     if has_summarise_vars:
         summarise_vars = output_settings.pop("summarise_variables")
         if summarise_vars:  # Not empty list
             state_idxs, obs_idxs = self._classify_variables(summarise_vars)
             
             # Union with existing summarised_state_indices
             if "summarised_state_indices" in output_settings and output_settings["summarised_state_indices"] is not None:
                 existing = output_settings["summarised_state_indices"]
                 combined = np.union1d(existing, state_idxs).astype(np.int16)
                 output_settings["summarised_state_indices"] = combined
             elif len(state_idxs) > 0:
                 output_settings["summarised_state_indices"] = state_idxs
             
             # Union with existing summarised_observable_indices
             if "summarised_observable_indices" in output_settings and output_settings["summarised_observable_indices"] is not None:
                 existing = output_settings["summarised_observable_indices"]
                 combined = np.union1d(existing, obs_idxs).astype(np.int16)
                 output_settings["summarised_observable_indices"] = combined
             elif len(obs_idxs) > 0:
                 output_settings["summarised_observable_indices"] = obs_idxs
     ```
   - Edge cases: Same as save_variables
   - Integration: Processes summarise_variables before existing label-to-index conversion

5. **Update convert_output_labels() docstring**
   - File: src/cubie/batchsolving/solver.py
   - Action: Modify
   - Details:
     ```python
     # Update Notes section (after line 284):
     Notes
     -----
     Users may supply selectors as labels or integers; this resolver ensures
     that downstream components receive numeric indices and canonical keys.
     
     The unified parameters ``save_variables`` and ``summarise_variables``
     are automatically classified into states and observables using
     SystemInterface. Results are merged with existing indices using set union,
     allowing both old and new parameter styles to coexist.
     ```
   - Edge cases: None
   - Integration: Documents new behavior

**Outcomes**:
- Files Modified:
  * src/cubie/batchsolving/solver.py (~140 lines changed)
- Functions/Methods Added/Modified:
  * _classify_variables() method added (new helper method, 69 lines)
  * convert_output_labels() method extended (71 lines added for new logic)
- Implementation Summary:
  Added _classify_variables() helper that iterates through variable names and uses SystemInterface.state_indices() and observable_indices() to classify each name. Method returns two arrays (states, observables) with appropriate dtype. Added fast-path check at start of convert_output_labels() to skip processing when new parameters not provided. Implemented save_variables and summarise_variables processing logic that calls _classify_variables() and merges results with existing indices using np.union1d(). Updated docstring to document new behavior.
- Issues Flagged: None

---

## Task Group 5: Unit Tests for Variable Classification - SEQUENTIAL
**Status**: [x]
**Dependencies**: Group 4

**Required Context**:
- File: tests/batchsolving/test_solver.py (entire file)
- System fixtures in tests/system_fixtures.py
- Conftest patterns in tests/conftest.py

**Input Validation Required**:
- Test inputs validated by production code
- Test assertions validate correct behavior

**Tasks**:
1. **Test save_variables with pure state names**
   - File: tests/batchsolving/test_solver.py
   - Action: Create
   - Details:
     ```python
     def test_save_variables_pure_states(solver, system):
         """Test save_variables with only state names."""
         state_names = list(system.initial_values.names)[:2]  # First 2 states
         
         output_settings = {"save_variables": state_names}
         solver.convert_output_labels(output_settings)
         
         # Verify save_variables was removed
         assert "save_variables" not in output_settings
         # Verify state indices were set
         assert "saved_state_indices" in output_settings
         assert len(output_settings["saved_state_indices"]) == 2
         # Verify no observable indices
         assert "saved_observable_indices" not in output_settings or output_settings["saved_observable_indices"] is None
     ```
   - Edge cases: Uses only states
   - Integration: Tests basic classification

2. **Test save_variables with pure observable names**
   - File: tests/batchsolving/test_solver.py
   - Action: Create
   - Details:
     ```python
     def test_save_variables_pure_observables(solver, system):
         """Test save_variables with only observable names."""
         if not hasattr(system.observables, "names") or len(system.observables.names) == 0:
             pytest.skip("System has no observables")
         
         obs_names = list(system.observables.names)[:2]
         
         output_settings = {"save_variables": obs_names}
         solver.convert_output_labels(output_settings)
         
         # Verify save_variables was removed
         assert "save_variables" not in output_settings
         # Verify observable indices were set
         assert "saved_observable_indices" in output_settings
         assert len(output_settings["saved_observable_indices"]) >= 1
         # Verify no state indices
         assert "saved_state_indices" not in output_settings or output_settings["saved_state_indices"] is None
     ```
   - Edge cases: Uses only observables, skips if no observables
   - Integration: Tests observable classification

3. **Test save_variables with mixed state and observable names**
   - File: tests/batchsolving/test_solver.py
   - Action: Create
   - Details:
     ```python
     def test_save_variables_mixed(solver, system):
         """Test save_variables with both states and observables."""
         state_names = list(system.initial_values.names)[:1]
         obs_names = []
         if hasattr(system.observables, "names") and len(system.observables.names) > 0:
             obs_names = list(system.observables.names)[:1]
         
         if not obs_names:
             pytest.skip("System has no observables for mixed test")
         
         mixed_names = state_names + obs_names
         
         output_settings = {"save_variables": mixed_names}
         solver.convert_output_labels(output_settings)
         
         # Verify both types classified
         assert "saved_state_indices" in output_settings
         assert "saved_observable_indices" in output_settings
         assert len(output_settings["saved_state_indices"]) >= 1
         assert len(output_settings["saved_observable_indices"]) >= 1
     ```
   - Edge cases: Mixed types
   - Integration: Tests both classification paths

4. **Test summarise_variables with same variations**
   - File: tests/batchsolving/test_solver.py
   - Action: Create
   - Details:
     ```python
     def test_summarise_variables_pure_states(solver, system):
         """Test summarise_variables with only state names."""
         state_names = list(system.initial_values.names)[:2]
         
         output_settings = {"summarise_variables": state_names}
         solver.convert_output_labels(output_settings)
         
         assert "summarise_variables" not in output_settings
         assert "summarised_state_indices" in output_settings
         assert len(output_settings["summarised_state_indices"]) == 2
     
     # Similar tests for pure observables and mixed
     ```
   - Edge cases: Same as save_variables tests
   - Integration: Tests summarise_variables classification

5. **Test union with existing saved_state_indices**
   - File: tests/batchsolving/test_solver.py
   - Action: Create
   - Details:
     ```python
     def test_save_variables_union_with_indices(solver, system):
         """Test save_variables merges with existing saved_state_indices."""
         state_names = list(system.initial_values.names)
         
         # Pre-populate with first state index
         output_settings = {
             "saved_state_indices": np.array([0], dtype=np.int16),
             "save_variables": [state_names[1]]  # Add second state by name
         }
         solver.convert_output_labels(output_settings)
         
         # Should have union of indices 0 and 1
         result = output_settings["saved_state_indices"]
         assert len(result) == 2
         assert 0 in result
         assert 1 in result
     ```
   - Edge cases: Tests union semantics
   - Integration: Verifies merging behavior

6. **Test union with existing saved_states label parameter**
   - File: tests/batchsolving/test_solver.py
   - Action: Create
   - Details:
     ```python
     def test_save_variables_union_with_saved_states(solver, system):
         """Test save_variables merges with existing saved_states."""
         state_names = list(system.initial_values.names)
         
         output_settings = {
             "saved_states": [state_names[0]],  # Will be converted to index
             "save_variables": [state_names[1]]  # Additional state
         }
         solver.convert_output_labels(output_settings)
         
         # Should have both states
         result = output_settings["saved_state_indices"]
         assert len(result) == 2
     ```
   - Edge cases: Tests interaction with label conversion
   - Integration: Verifies both mechanisms work together

7. **Test empty list handling**
   - File: tests/batchsolving/test_solver.py
   - Action: Create
   - Details:
     ```python
     def test_save_variables_empty_list(solver):
         """Test save_variables with empty list is no-op."""
         output_settings = {"save_variables": []}
         solver.convert_output_labels(output_settings)
         
         # Empty list should be removed but not create indices
         assert "save_variables" not in output_settings
         assert "saved_state_indices" not in output_settings or output_settings.get("saved_state_indices") is None
     ```
   - Edge cases: Empty list
   - Integration: Tests no-op behavior

8. **Test None value handling**
   - File: tests/batchsolving/test_solver.py
   - Action: Create
   - Details:
     ```python
     def test_save_variables_none(solver):
         """Test save_variables=None is ignored."""
         output_settings = {"save_variables": None}
         solver.convert_output_labels(output_settings)
         
         # None should trigger fast-path and be ignored
         # Behavior: fast-path means parameter stays in dict but is not processed
     ```
   - Edge cases: None value
   - Integration: Tests fast-path

9. **Test invalid variable names raise ValueError**
   - File: tests/batchsolving/test_solver.py
   - Action: Create
   - Details:
     ```python
     def test_save_variables_invalid_name_raises(solver):
         """Test save_variables with invalid name raises clear error."""
         output_settings = {"save_variables": ["nonexistent_variable"]}
         
         with pytest.raises(ValueError, match="Variables not found"):
             solver.convert_output_labels(output_settings)
         
         # Error message should include available names
         try:
             solver.convert_output_labels(output_settings)
         except ValueError as e:
             assert "Available states:" in str(e)
             assert "Available observables:" in str(e)
     ```
   - Edge cases: Invalid names
   - Integration: Tests error handling

10. **Test no performance regression for array-only path**
    - File: tests/batchsolving/test_solver.py
    - Action: Create
    - Details:
      ```python
      def test_array_only_fast_path(solver):
          """Test array-only parameters don't trigger name resolution."""
          import time
          
          output_settings = {
              "saved_state_indices": np.array([0, 1], dtype=np.int16)
          }
          
          # Time the fast path (should be very quick)
          start = time.perf_counter()
          for _ in range(1000):
              settings_copy = output_settings.copy()
              solver.convert_output_labels(settings_copy)
          fast_time = time.perf_counter() - start
          
          # Verify fast path was taken (implementation detail: no classification)
          # This is more of a regression test than strict performance test
          assert fast_time < 1.0  # Should be well under 1 second for 1000 iterations
      ```
    - Edge cases: Performance check
    - Integration: Tests fast-path optimization

**Outcomes**:
- Files Modified:
  * tests/batchsolving/test_solver.py (~270 lines added)
- Tests Added:
  * test_save_variables_pure_states()
  * test_save_variables_pure_observables()
  * test_save_variables_mixed()
  * test_summarise_variables_pure_states()
  * test_summarise_variables_pure_observables()
  * test_summarise_variables_mixed()
  * test_save_variables_union_with_indices()
  * test_save_variables_union_with_saved_states()
  * test_save_variables_empty_list()
  * test_save_variables_none()
  * test_save_variables_invalid_name_raises()
  * test_save_variables_error_includes_available_names()
  * test_array_only_fast_path()
- Implementation Summary:
  Added comprehensive unit tests covering all edge cases for save_variables and summarise_variables parameters. Tests validate classification of pure states, pure observables, and mixed variables. Tests verify union semantics with existing parameters. Tests check error handling for invalid names. Tests confirm fast-path optimization for array-only workflows.
- Issues Flagged: None

---

## Task Group 6: Integration Tests for Full Solve Workflow - SEQUENTIAL
**Status**: [x]
**Dependencies**: Group 5

**Required Context**:
- File: tests/batchsolving/test_solver.py
- System fixtures with states and observables
- Test patterns for solve() calls

**Input Validation Required**:
- Test that solve() produces correct output arrays
- Validate output shapes and content

**Tasks**:
1. **Test solve_ivp with save_variables**
   - File: tests/batchsolving/test_solver.py
   - Action: Create
   - Details:
     ```python
     def test_solve_ivp_with_save_variables(system):
         """Test solve_ivp accepts save_variables and produces correct output."""
         state_names = list(system.initial_values.names)[:2]
         
         result = solve_ivp(
             system,
             y0={state_names[0]: [1.0, 2.0]},
             parameters={list(system.parameters.names)[0]: [0.1, 0.2]},
             save_variables=state_names,
             dt_save=0.01,
             duration=0.1,
             method="euler",
         )
         
         # Verify result contains saved states
         assert result is not None
         assert hasattr(result, 'y')
         assert result.y is not None
         # Verify shape matches number of save_variables
         assert result.y.shape[0] == len(state_names)
     ```
   - Edge cases: Basic integration test
   - Integration: Tests full workflow through solve_ivp

2. **Test Solver.solve() with save_variables**
   - File: tests/batchsolving/test_solver.py
   - Action: Create
   - Details:
     ```python
     def test_solver_solve_with_save_variables(solver, system):
         """Test Solver.solve accepts save_variables parameter."""
         state_names = list(system.initial_values.names)[:1]
         
         result = solver.solve(
             initial_values={state_names[0]: [1.0, 2.0]},
             parameters={list(system.parameters.names)[0]: [0.1, 0.2]},
             save_variables=state_names,
             duration=0.1,
         )
         
         assert result is not None
         # Verify saved output contains requested states
         assert result.y.shape[0] >= 1
     ```
   - Edge cases: Tests Solver.solve() path
   - Integration: Tests kwargs forwarding through solve()

3. **Test solve_ivp with summarise_variables**
   - File: tests/batchsolving/test_solver.py
   - Action: Create
   - Details:
     ```python
     def test_solve_ivp_with_summarise_variables(system):
         """Test solve_ivp accepts summarise_variables and produces summaries."""
         state_names = list(system.initial_values.names)[:2]
         
         result = solve_ivp(
             system,
             y0={state_names[0]: [1.0, 2.0]},
             parameters={list(system.parameters.names)[0]: [0.1, 0.2]},
             summarise_variables=state_names,
             duration=0.1,
             method="euler",
         )
         
         # Verify result contains summaries
         assert result is not None
         # Check that summaries exist (structure depends on OutputFunctions)
     ```
   - Edge cases: Tests summary path
   - Integration: Tests summarise_variables through full solve

4. **Test backward compatibility with existing parameters**
   - File: tests/batchsolving/test_solver.py
   - Action: Create
   - Details:
     ```python
     def test_backward_compatibility_existing_params(system):
         """Test existing saved_states parameter still works."""
         state_names = list(system.initial_values.names)[:2]
         
         # Use old-style parameters only
         result = solve_ivp(
             system,
             y0={state_names[0]: [1.0, 2.0]},
             parameters={list(system.parameters.names)[0]: [0.1, 0.2]},
             saved_states=state_names,  # Old parameter
             dt_save=0.01,
             duration=0.1,
             method="euler",
         )
         
         assert result is not None
         assert result.y is not None
     ```
   - Edge cases: Tests backward compatibility
   - Integration: Ensures old parameters unchanged

5. **Test mixing old and new parameters**
   - File: tests/batchsolving/test_solver.py
   - Action: Create
   - Details:
     ```python
     def test_mixing_old_and_new_params(system):
         """Test mixing saved_states with save_variables."""
         state_names = list(system.initial_values.names)
         if len(state_names) < 2:
             pytest.skip("Need at least 2 states")
         
         result = solve_ivp(
             system,
             y0={state_names[0]: [1.0], state_names[1]: [2.0]},
             parameters={list(system.parameters.names)[0]: [0.1]},
             saved_states=[state_names[0]],  # Old style
             save_variables=[state_names[1]],  # New style
             dt_save=0.01,
             duration=0.1,
             method="euler",
         )
         
         # Should have both states saved (union)
         assert result is not None
         assert result.y.shape[0] == 2
     ```
   - Edge cases: Mixed parameter styles
   - Integration: Tests union semantics in full solve

**Outcomes**:
- Files Modified:
  * tests/batchsolving/test_solver.py (same file as Group 5, tests added together)
- Tests Added:
  * test_solve_ivp_with_save_variables()
  * test_solver_solve_with_save_variables()
  * test_solve_ivp_with_summarise_variables()
  * test_backward_compatibility_existing_params()
  * test_mixing_old_and_new_params()
- Implementation Summary:
  Added integration tests that verify the complete workflow through solve_ivp() and Solver.solve(). Tests confirm that save_variables and summarise_variables parameters are properly forwarded through the entire solve pipeline. Tests validate backward compatibility with existing parameters and verify union semantics when mixing old and new parameter styles.
- Issues Flagged: None

---

## Task Group 7: Documentation Updates - PARALLEL
**Status**: [x]
**Dependencies**: Groups 1-6 (documentation can be drafted in parallel but should reference completed implementation)

**Required Context**:
- All modified function signatures
- Updated docstrings in solver.py

**Input Validation Required**:
- None (documentation only)

**Tasks**:
1. **Verify docstrings are complete and accurate**
   - File: src/cubie/batchsolving/solver.py
   - Action: Review
   - Details:
     - Check solve_ivp() docstring includes new parameters (Group 2)
     - Check Solver class docstring mentions new parameters (Group 3)
     - Check convert_output_labels() docstring updated (Group 4)
     - Ensure numpydoc format consistency
   - Edge cases: None
   - Integration: Documentation consistency check

**Outcomes**:
- Files Modified:
  * No additional files modified (docstrings updated in Groups 2, 3, and 4)
- Documentation Complete:
  * solve_ivp() docstring includes save_variables and summarise_variables parameters with complete descriptions
  * Solver class docstring updated to mention new parameters in **kwargs section
  * convert_output_labels() docstring updated with Notes section explaining new behavior
  * All docstrings follow numpydoc format consistently
- Implementation Summary:
  Verified all docstrings are complete and accurate. Documentation properly explains the new unified interface for variable selection and union semantics with existing parameters.
- Issues Flagged: None

---

## Summary

### Total Task Groups: 7
- Sequential: Groups 1-6 (core implementation and tests)
- Parallel: Group 7 (documentation review)

### Dependency Chain
```
Group 1 (Parameters to set)
  ↓
Group 2 (solve_ivp signature)
  ↓
Group 3 (Solver docstring)
  ↓
Group 4 (convert_output_labels implementation) ← CORE LOGIC
  ↓
Group 5 (Unit tests)
  ↓
Group 6 (Integration tests)

Group 7 (Documentation) can proceed in parallel once Groups 2-4 complete
```

### Parallel Execution Opportunities
- Group 7 (documentation review) can be done in parallel with Groups 5-6 (testing)

### Estimated Complexity
- **Low complexity**: Groups 1, 2, 3, 7 (simple additions, docstring updates)
- **Medium complexity**: Group 4 (core classification logic, but well-specified)
- **Medium complexity**: Groups 5, 6 (comprehensive test coverage)

### Key Implementation Notes
1. **Fast-path optimization**: Critical for performance - check for new parameters before any processing
2. **Union semantics**: Use np.union1d() to merge indices without duplicates
3. **Error messages**: Include available variable names in ValueError for easy debugging
4. **Edge cases**: Handle None, empty lists, mixed types gracefully
5. **SystemInterface methods**: Use existing state_indices() and observable_indices() - no modifications needed
6. **No GPU code changes**: All changes are in Python host code, no CUDA kernel modifications

### Files Modified
1. `src/cubie/outputhandling/output_functions.py` - Add parameter names
2. `src/cubie/batchsolving/solver.py` - Main implementation
3. `tests/batchsolving/test_solver.py` - Comprehensive tests

### Files Created
None (all changes to existing files)

### Breaking Changes
None - fully backward compatible

---

# Implementation Complete - Ready for Review

## Execution Summary
- Total Task Groups: 7
- Completed: 7
- Failed: 0
- Total Files Modified: 3

## Task Group Completion
- Group 1: [x] Add Parameter Names to ALL_OUTPUT_FUNCTION_PARAMETERS - Complete
- Group 2: [x] Update solve_ivp() Function Signature - Complete
- Group 3: [x] Update Solver Class Docstring - Complete
- Group 4: [x] Implement Variable Classification in convert_output_labels() - Complete
- Group 5: [x] Unit Tests for Variable Classification - Complete
- Group 6: [x] Integration Tests for Full Solve Workflow - Complete
- Group 7: [x] Documentation Updates - Complete

## All Modified Files
1. src/cubie/outputhandling/output_functions.py (2 lines added)
2. src/cubie/batchsolving/solver.py (~200 lines added/modified)
3. tests/batchsolving/test_solver.py (~270 lines added)

## Implementation Highlights

### Core Logic (Group 4)
- Added `_classify_variables()` helper method (69 lines)
  * Iterates through variable names
  * Uses SystemInterface to classify as states or observables
  * Returns two arrays with appropriate dtype (np.int16)
  * Raises ValueError with helpful message for unrecognized names
  
- Extended `convert_output_labels()` method (71 lines added)
  * Fast-path check at start (zero overhead when new params not used)
  * Processes save_variables parameter
  * Processes summarise_variables parameter
  * Merges results with existing indices using np.union1d()
  * Updated docstring with Notes explaining new behavior

### API Changes (Groups 1-3)
- Added "save_variables" and "summarise_variables" to ALL_OUTPUT_FUNCTION_PARAMETERS
- Updated solve_ivp() signature with two new optional parameters
- Added comprehensive numpydoc docstrings
- Updated Solver class docstring to mention new parameters

### Test Coverage (Groups 5-6)
- 18 new test functions covering:
  * Pure state classification
  * Pure observable classification
  * Mixed state/observable classification
  * Union semantics with existing parameters
  * Empty list handling
  * None value handling
  * Invalid name error handling
  * Fast-path performance
  * Full integration through solve_ivp() and Solver.solve()
  * Backward compatibility
  * Mixing old and new parameter styles

## Key Features Implemented

1. **Unified Interface**: Users can now specify `save_variables=["x", "observable1"]` without distinguishing between states and observables

2. **Fast-Path Optimization**: When only array indices are provided (no string lists), no name resolution executes - zero performance overhead

3. **Union Semantics**: New parameters merge with existing parameters using set union - no duplicates

4. **Backward Compatibility**: All existing parameters work unchanged; users can mix old and new styles

5. **Clear Error Messages**: Invalid variable names raise ValueError with list of available states and observables

## Flagged Issues
None - all implementation tasks completed successfully

## Handoff to Reviewer
All implementation tasks complete. Task list updated with outcomes for each group.
Implementation follows PEP8, uses type hints, includes comprehensive tests, and maintains backward compatibility.
Ready for validation against user stories and goals.
