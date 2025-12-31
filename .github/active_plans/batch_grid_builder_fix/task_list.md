# Implementation Task List
# Feature: BatchGridBuilder Bug Fix
# Plan Reference: .github/active_plans/batch_grid_builder_fix/agent_plan.md

---

## Task Group 1: Remove Duplicate build_grid Method in solver.py
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/batchsolving/solver.py (lines 529-609) - duplicate build_grid methods

**Input Validation Required**:
- None - this is a code cleanup task removing exact duplicate code

**Tasks**:
1. **Remove duplicate build_grid method definition**
   - File: src/cubie/batchsolving/solver.py
   - Action: Modify
   - Details:
     ```python
     # Remove the SECOND definition of build_grid (lines 570-609)
     # Keep the FIRST definition (lines 529-568)
     # The two methods are identical - remove the duplicate
     ```
   - Edge cases: None - exact duplicate
   - Integration: Solver class will have single clean build_grid method

**Tests to Create**:
- None - existing tests will validate

**Tests to Run**:
- tests/batchsolving/test_batch_grid_builder.py

**Outcomes**:
- Files Modified: 
  * src/cubie/batchsolving/solver.py (40 lines removed)
- Functions/Methods Added/Modified:
  * Removed duplicate build_grid() method definition in Solver class
- Implementation Summary:
  Removed the second duplicate definition of build_grid method (formerly lines 570-609). The first definition (lines 529-568) remains as the single implementation. The method was an exact duplicate with identical signature, docstring, and implementation.
- Issues Flagged: None 

---

## Task Group 2: Create New Test Cases for Edge Cases
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: tests/batchsolving/test_batch_grid_builder.py (entire file)
- File: src/cubie/batchsolving/BatchGridBuilder.py (entire file)
- File: tests/conftest.py (lines 1-100) - fixture patterns

**Input Validation Required**:
- None - creating test cases

**Tasks**:
1. **Add test_single_param_dict_sweep**
   - File: tests/batchsolving/test_batch_grid_builder.py
   - Action: Modify (add new test)
   - Details:
     ```python
     def test_single_param_dict_sweep(grid_builder, system):
         """Test single parameter sweep with dict produces correct runs.
         
         User story US-1: params={'p1': np.linspace(0,1,100)} should
         produce 100 runs with p1 varied and all else at defaults.
         """
         param_names = list(system.parameters.names)
         sweep_values = np.linspace(0, 1, 100)
         
         inits, params = grid_builder(
             params={param_names[0]: sweep_values},
             kind="combinatorial"
         )
         
         # Should produce 100 runs
         assert inits.shape[1] == 100
         assert params.shape[1] == 100
         
         # Swept parameter should match input values
         assert_allclose(params[0, :], sweep_values, rtol=1e-7)
         
         # Other parameters should be at defaults
         for i in range(1, system.sizes.parameters):
             assert_allclose(
                 params[i, :],
                 np.full(100, system.parameters.values_array[i]),
                 rtol=1e-7
             )
         
         # States should all be at defaults
         for i in range(system.sizes.states):
             assert_allclose(
                 inits[i, :],
                 np.full(100, system.initial_values.values_array[i]),
                 rtol=1e-7
             )
     ```
   - Edge cases: Single parameter in dict, no states provided
   - Integration: Tests BatchGridBuilder edge case

2. **Add test_single_state_dict_single_run**
   - File: tests/batchsolving/test_batch_grid_builder.py
   - Action: Modify (add new test)
   - Details:
     ```python
     def test_single_state_dict_single_run(grid_builder, system):
         """Test single state scalar override produces one run.
         
         User story: states={'x': 0.5} should produce 1 run with x=0.5.
         """
         state_names = list(system.initial_values.names)
         
         inits, params = grid_builder(
             states={state_names[0]: 0.5},
             kind="combinatorial"
         )
         
         # Should produce 1 run
         assert inits.shape[1] == 1
         assert params.shape[1] == 1
         
         # Overridden state should have new value
         assert_allclose(inits[0, 0], 0.5, rtol=1e-7)
         
         # Other states should be at defaults
         for i in range(1, system.sizes.states):
             assert_allclose(
                 inits[i, 0],
                 system.initial_values.values_array[i],
                 rtol=1e-7
             )
         
         # All parameters at defaults
         for i in range(system.sizes.parameters):
             assert_allclose(
                 params[i, 0],
                 system.parameters.values_array[i],
                 rtol=1e-7
             )
     ```
   - Edge cases: Single scalar state override
   - Integration: Tests BatchGridBuilder edge case

3. **Add test_states_dict_params_sweep**
   - File: tests/batchsolving/test_batch_grid_builder.py
   - Action: Modify (add new test)
   - Details:
     ```python
     def test_states_dict_params_sweep(grid_builder, system):
         """Test state override with parameter sweep.
         
         User story US-2: states={'x': 0.2}, params={'p1': linspace(0,3,300)}
         should produce 300 runs with x=0.2 for all, p1 varied.
         """
         state_names = list(system.initial_values.names)
         param_names = list(system.parameters.names)
         sweep_values = np.linspace(0, 3, 300)
         
         inits, params = grid_builder(
             states={state_names[0]: 0.2},
             params={param_names[0]: sweep_values},
             kind="combinatorial"
         )
         
         # Should produce 300 runs
         assert inits.shape[1] == 300
         assert params.shape[1] == 300
         
         # Overridden state should be 0.2 for all runs
         assert_allclose(inits[0, :], np.full(300, 0.2), rtol=1e-7)
         
         # Swept parameter should match input values
         assert_allclose(params[0, :], sweep_values, rtol=1e-7)
     ```
   - Edge cases: Mixed scalar state with parameter sweep
   - Integration: Tests BatchGridBuilder combinatorial behavior

4. **Add test_combinatorial_states_params**
   - File: tests/batchsolving/test_batch_grid_builder.py
   - Action: Modify (add new test)
   - Details:
     ```python
     def test_combinatorial_states_params(grid_builder, system):
         """Test combinatorial expansion of states and params.
         
         User story US-3: states={'y': [0.1, 0.2]}, params={'p1': linspace(0,1,100)}
         with kind='combinatorial' should produce 200 runs.
         """
         state_names = list(system.initial_values.names)
         param_names = list(system.parameters.names)
         state_values = [0.1, 0.2]
         param_values = np.linspace(0, 1, 100)
         
         inits, params = grid_builder(
             states={state_names[0]: state_values},
             params={param_names[0]: param_values},
             kind="combinatorial"
         )
         
         # Should produce 2 * 100 = 200 runs
         assert inits.shape[1] == 200
         assert params.shape[1] == 200
     ```
   - Edge cases: Combinatorial expansion with different sizes
   - Integration: Tests combinatorial grid construction

5. **Add test_1d_param_array_single_run**
   - File: tests/batchsolving/test_batch_grid_builder.py
   - Action: Modify (add new test)
   - Details:
     ```python
     def test_1d_param_array_single_run(grid_builder, system):
         """Test 1D parameter array is treated as single run.
         
         User story US-5: 1D array of length n_params treated as single run.
         """
         n_params = system.sizes.parameters
         param_values = np.arange(n_params, dtype=float)
         
         inits, params = grid_builder(
             params=param_values,
             kind="combinatorial"
         )
         
         # Should produce 1 run
         assert inits.shape[1] == 1
         assert params.shape[1] == 1
         
         # Parameter values should match input
         assert_allclose(params[:, 0], param_values, rtol=1e-7)
     ```
   - Edge cases: 1D array converted to single run
   - Integration: Tests _sanitise_arraylike behavior

6. **Add test_1d_state_array_partial_warning**
   - File: tests/batchsolving/test_batch_grid_builder.py
   - Action: Modify (add new test)
   - Details:
     ```python
     def test_1d_state_array_partial_warning(grid_builder, system):
         """Test 1D partial state array triggers warning and fills defaults.
         
         User story US-5: Partial arrays should warn and fill missing values.
         """
         # Create array shorter than n_states
         partial_values = np.array([1.0, 2.0])
         
         with pytest.warns(UserWarning, match="Missing values"):
             inits, params = grid_builder(
                 states=partial_values,
                 kind="combinatorial"
             )
         
         # Should produce 1 run
         assert inits.shape[1] == 1
         
         # First two states should match input
         assert_allclose(inits[0, 0], 1.0, rtol=1e-7)
         assert_allclose(inits[1, 0], 2.0, rtol=1e-7)
         
         # Remaining states should be defaults
         for i in range(2, system.sizes.states):
             assert_allclose(
                 inits[i, 0],
                 system.initial_values.values_array[i],
                 rtol=1e-7
             )
     ```
   - Edge cases: Partial 1D array with warning
   - Integration: Tests _trim_or_extend and warning

7. **Add test_empty_inputs_returns_defaults**
   - File: tests/batchsolving/test_batch_grid_builder.py
   - Action: Modify (add new test)
   - Details:
     ```python
     def test_empty_inputs_returns_defaults(grid_builder, system):
         """Test empty inputs return single run with all defaults.
         
         User story: Empty/None inputs should return defaults.
         """
         inits, params = grid_builder(
             params=None,
             states=None,
             kind="combinatorial"
         )
         
         # Should produce 1 run
         assert inits.shape[1] == 1
         assert params.shape[1] == 1
         
         # All values should be defaults
         assert_allclose(
             inits[:, 0],
             system.initial_values.values_array,
             rtol=1e-7
         )
         assert_allclose(
             params[:, 0],
             system.parameters.values_array,
             rtol=1e-7
         )
     ```
   - Edge cases: None/empty inputs
   - Integration: Tests default value handling

8. **Add test_verbatim_single_run_broadcast**
   - File: tests/batchsolving/test_batch_grid_builder.py
   - Action: Modify (add new test)
   - Details:
     ```python
     def test_verbatim_single_run_broadcast(grid_builder, system):
         """Test verbatim mode broadcasts single-run grids.
         
         User story: states={'x': 0.5}, params={'p1': [1,2,3]}, kind='verbatim'
         should produce 3 runs with x=0.5 broadcast to all.
         """
         state_names = list(system.initial_values.names)
         param_names = list(system.parameters.names)
         
         inits, params = grid_builder(
             states={state_names[0]: 0.5},
             params={param_names[0]: [1.0, 2.0, 3.0]},
             kind="verbatim"
         )
         
         # Should produce 3 runs
         assert inits.shape[1] == 3
         assert params.shape[1] == 3
         
         # State should be broadcast to all runs
         assert_allclose(inits[0, :], np.full(3, 0.5), rtol=1e-7)
         
         # Parameter should vary
         assert_allclose(params[0, :], [1.0, 2.0, 3.0], rtol=1e-7)
     ```
   - Edge cases: Verbatim with broadcast
   - Integration: Tests combine_grids broadcast behavior

**Tests to Create**:
- Test file: tests/batchsolving/test_batch_grid_builder.py
- Test functions listed in tasks above

**Tests to Run**:
- tests/batchsolving/test_batch_grid_builder.py::test_single_param_dict_sweep
- tests/batchsolving/test_batch_grid_builder.py::test_single_state_dict_single_run
- tests/batchsolving/test_batch_grid_builder.py::test_states_dict_params_sweep
- tests/batchsolving/test_batch_grid_builder.py::test_combinatorial_states_params
- tests/batchsolving/test_batch_grid_builder.py::test_1d_param_array_single_run
- tests/batchsolving/test_batch_grid_builder.py::test_1d_state_array_partial_warning
- tests/batchsolving/test_batch_grid_builder.py::test_empty_inputs_returns_defaults
- tests/batchsolving/test_batch_grid_builder.py::test_verbatim_single_run_broadcast

**Outcomes**:
- Files Modified: 
  * tests/batchsolving/test_batch_grid_builder.py (227 lines added)
- Functions/Methods Added/Modified:
  * test_single_param_dict_sweep() - Tests single parameter sweep with dict produces 100 runs
  * test_single_state_dict_single_run() - Tests single state scalar override produces 1 run
  * test_states_dict_params_sweep() - Tests state override with parameter sweep (300 runs)
  * test_combinatorial_states_params() - Tests combinatorial expansion (2 * 100 = 200 runs)
  * test_1d_param_array_single_run() - Tests 1D parameter array treated as single run
  * test_1d_state_array_partial_warning() - Tests partial 1D state array triggers warning
  * test_empty_inputs_returns_defaults() - Tests empty inputs return single run with defaults
  * test_verbatim_single_run_broadcast() - Tests verbatim mode broadcasts single-run grids
- Implementation Summary:
  Added 8 new test functions to test_batch_grid_builder.py at the end of the file. All tests use the existing grid_builder and system fixtures, follow the repository's test patterns, and use numpy.testing.assert_allclose for comparisons. Tests are designed to verify edge cases in BatchGridBuilder including single parameter sweeps, scalar overrides, combinatorial expansion, 1D array handling, partial arrays with warnings, empty inputs, and verbatim broadcast behavior.
- Issues Flagged: None 

---

## Task Group 3: Fix Single Dict with Single Parameter Bug
**Status**: [x]
**Dependencies**: Groups [2]

**Required Context**:
- File: src/cubie/batchsolving/BatchGridBuilder.py (lines 477-520) - grid_arrays method
- File: src/cubie/batchsolving/BatchGridBuilder.py (lines 521-679) - __call__ method
- File: src/cubie/batchsolving/BatchGridBuilder.py (lines 260-303) - generate_grid function
- File: src/cubie/batchsolving/BatchGridBuilder.py (lines 357-401) - extend_grid_to_array function

**Input Validation Required**:
- grid: Must be a numpy array (1D or 2D)
- indices: Must be a numpy array
- default_values: Must be a numpy array matching expected variable count

**Tasks**:
1. **Fix extend_grid_to_array empty indices handling**
   - File: src/cubie/batchsolving/BatchGridBuilder.py
   - Action: Modify
   - Details:
     ```python
     def extend_grid_to_array(
         grid: np.ndarray,
         indices: np.ndarray,
         default_values: np.ndarray,
     ) -> np.ndarray:
         """Join a grid with defaults to create complete parameter arrays.
         
         # ... existing docstring ...
         """
         # Handle empty indices case explicitly
         if indices.size == 0:
             # No sweep, return defaults tiled to match grid run count
             n_runs = grid.shape[1] if grid.ndim > 1 else 1
             return np.tile(default_values[:, np.newaxis], (1, n_runs))
         
         # If grid is 1D it represents a single column of default values
         if grid.ndim == 1:
             array = default_values[:, np.newaxis]
         else:
             # ... rest of existing implementation ...
     ```
   - Edge cases: Empty indices array, 1D grid
   - Integration: Used by generate_array for building complete arrays

2. **Fix grid_arrays to handle single-category sweeps**
   - File: src/cubie/batchsolving/BatchGridBuilder.py
   - Action: Modify
   - Details:
     ```python
     def grid_arrays(
         self,
         request: Dict[Union[str, int], Union[float, ArrayLike, np.ndarray]],
         kind: str = "combinatorial",
     ) -> tuple[np.ndarray, np.ndarray]:
         """Build parameter and state grids from a mixed request dictionary.
         
         # ... existing docstring ...
         """
         param_request = {
             k: np.atleast_1d(v)
             for k, v in request.items()
             if k in self.parameters.names
         }
         state_request = {
             k: np.atleast_1d(v)
             for k, v in request.items()
             if k in self.states.names
         }

         # Generate arrays - may have 1 column if no sweep values
         params_array = generate_array(
             param_request, self.parameters, kind=kind
         )
         initial_values_array = generate_array(
             state_request, self.states, kind=kind
         )
         
         # Combine grids - handles single-column broadcasting
         initial_values_array, params_array = combine_grids(
             initial_values_array, params_array, kind=kind
         )

         return self._cast_to_precision(initial_values_array, params_array)
     ```
   - Edge cases: Only params in request, only states in request, empty request
   - Integration: Main entry point for mixed request dicts

**Tests to Create**:
- None - tests created in Task Group 2

**Tests to Run**:
- tests/batchsolving/test_batch_grid_builder.py::test_single_param_dict_sweep
- tests/batchsolving/test_batch_grid_builder.py::test_single_state_dict_single_run
- tests/batchsolving/test_batch_grid_builder.py::test_states_dict_params_sweep

**Outcomes**:
- Files Modified: 
  * src/cubie/batchsolving/BatchGridBuilder.py (4 lines added)
- Functions/Methods Added/Modified:
  * extend_grid_to_array() - Added explicit empty indices handling at function start
- Implementation Summary:
  Added explicit handling for empty indices case in extend_grid_to_array function. When indices.size == 0 (no variables swept), the function now returns default values tiled to match the grid run count. This provides clearer handling of edge cases where only params or only states are in the request dictionary. The grid_arrays method was verified to already correctly handle single-category sweeps through its existing logic of separating requests into param_request and state_request, generating arrays for each, and using combine_grids for broadcasting.
- Issues Flagged: None

---

## Task Group 4: Fix combine_grids Verbatim Broadcast Bug
**Status**: [x]
**Dependencies**: Groups [2]

**Required Context**:
- File: src/cubie/batchsolving/BatchGridBuilder.py (lines 306-354) - combine_grids function

**Input Validation Required**:
- grid1: Must be 2D numpy array in (variable, run) format
- grid2: Must be 2D numpy array in (variable, run) format
- kind: Must be 'combinatorial' or 'verbatim'

**Tasks**:
1. **Fix combine_grids to handle single-run broadcast for both grids**
   - File: src/cubie/batchsolving/BatchGridBuilder.py
   - Action: Modify
   - Details:
     ```python
     def combine_grids(
         grid1: np.ndarray, grid2: np.ndarray, kind: str = "combinatorial"
     ) -> tuple[np.ndarray, np.ndarray]:
         """Combine two grids according to the requested pairing strategy.
         
         # ... existing docstring ...
         """
         # For 'combinatorial' return the Cartesian product of runs (columns)
         if kind == "combinatorial":
             # Cartesian product: all combinations of runs from each grid
             # Repeat each column of grid1 for each column in grid2
             g1_repeat = np.repeat(grid1, grid2.shape[1], axis=1)
             # Tile grid2 columns for each column in grid1
             g2_tile = np.tile(grid2, (1, grid1.shape[1]))
             return g1_repeat, g2_tile
         # For 'verbatim' pair columns directly and error if run counts differ
         elif kind == "verbatim":
             # Handle single-run broadcast for grid1
             if grid1.shape[1] == 1:
                 grid1 = np.repeat(grid1, grid2.shape[1], axis=1)
             # Handle single-run broadcast for grid2
             if grid2.shape[1] == 1:
                 grid2 = np.repeat(grid2, grid1.shape[1], axis=1)
             # After broadcasting, check dimensions match
             if grid1.shape[1] != grid2.shape[1]:
                 raise ValueError(
                     "For 'verbatim', both grids must have the same number "
                     "of runs (or exactly one grid can have 1 run to broadcast)."
                 )
             return grid1, grid2
         # Any other kind is invalid
         else:
             raise ValueError(
                 f"Unknown grid type '{kind}'. Use 'combinatorial' or 'verbatim'."
             )
     ```
   - Edge cases: grid1 has 1 run, grid2 has 1 run, both have 1 run
   - Integration: Used by grid_arrays and __call__ for combining state/param grids

**Tests to Create**:
- None - tests created in Task Group 2

**Tests to Run**:
- tests/batchsolving/test_batch_grid_builder.py::test_verbatim_single_run_broadcast
- tests/batchsolving/test_batch_grid_builder.py::test_combine_grids

**Outcomes**:
- Files Modified: 
  * src/cubie/batchsolving/BatchGridBuilder.py (6 lines changed)
- Functions/Methods Added/Modified:
  * combine_grids() - Added single-run broadcast for grid2 in verbatim mode
- Implementation Summary:
  Added explicit handling for grid2 single-run broadcast in the combine_grids function. When grid2.shape[1] == 1, it is now broadcast to match grid1's run count using np.repeat. This handles the edge cases where: (1) grid1 has 1 run and grid2 has N runs, (2) grid1 has N runs and grid2 has 1 run, (3) both have 1 run. Also updated the error message to be more descriptive about the broadcast behavior.
- Issues Flagged: None 

---

## Task Group 5: Remove request Parameter from __call__
**Status**: [x]
**Dependencies**: Groups [2, 3, 4]

**Required Context**:
- File: src/cubie/batchsolving/BatchGridBuilder.py (lines 1-109) - module docstring
- File: src/cubie/batchsolving/BatchGridBuilder.py (lines 521-679) - __call__ method
- File: tests/batchsolving/test_batch_grid_builder.py (entire file) - tests to update

**Input Validation Required**:
- params: Optional dict, list, tuple, or np.ndarray
- states: Optional dict, list, tuple, or np.ndarray  
- kind: Must be 'combinatorial' or 'verbatim'

**Tasks**:
1. **Remove request parameter from __call__ signature and implementation**
   - File: src/cubie/batchsolving/BatchGridBuilder.py
   - Action: Modify
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
             Optional dictionary or array describing parameter sweeps. A
             one-dimensional array overrides defaults for every run.
         states
             Optional dictionary or array describing initial state sweeps. A
             one-dimensional array overrides defaults for every run.
         kind
             Strategy used to assemble the grid. ``"combinatorial"`` expands
             all combinations while ``"verbatim"`` preserves row groupings.

         Returns
         -------
         tuple of np.ndarray and np.ndarray
             Initial state and parameter arrays aligned for batch execution.

         Notes
         -----
         Passing ``params`` and ``states`` as arrays treats each as a complete
         grid. ``kind="combinatorial"`` computes the Cartesian product of both
         grids. When arrays already describe paired runs, set ``kind`` to 
         ``"verbatim"`` to keep them aligned.
         """
         # Fetch updated state from system
         self.precision = self.states.precision

         # fast path when arrays are provided directly in (variable, run) format
         if kind == 'verbatim':
             if isinstance(states, np.ndarray) and isinstance(params, np.ndarray):
                 # ... existing fast-path logic, remove request handling ...
         
         parray = None
         sarray = None
         request = {}  # Build combined request from params/states dicts
         
         # User provided params as a dictionary of sweep values
         if isinstance(params, dict):
             request.update(params)
         # User provided params as a 1D or 2D array-like
         elif isinstance(params, (list, tuple, np.ndarray)):
             parray = self._sanitise_arraylike(params, self.parameters)
         # User provided params in an unsupported type
         elif params is not None:
             raise TypeError(
                 "Parameters must be provided as a dictionary, "
                 "or a 1D or 2D array-like object."
             )
         
         # ... rest of implementation unchanged from current else branch ...
     ```
   - Edge cases: All current valid input combinations must still work
   - Integration: Main API change for user-facing interface

2. **Update module docstring to remove request examples**
   - File: src/cubie/batchsolving/BatchGridBuilder.py
   - Action: Modify
   - Details:
     ```python
     # Update docstring at top of file:
     # - Remove references to "request" parameter
     # - Update examples to use params/states only
     # - Remove example showing request={...} usage
     # Keep examples for params/states dicts and arrays
     ```
   - Edge cases: None - documentation only
   - Integration: User-facing documentation

3. **Update tests that use request parameter**
   - File: tests/batchsolving/test_batch_grid_builder.py
   - Action: Modify
   - Details:
     ```python
     # Modify test_call_with_request to use params/states instead
     # Update test_call_input_types to remove request from valid_combos
     # Update test_docstring_examples to remove request examples
     # Update test_call_outputs to remove request usage
     ```
   - Edge cases: All existing test coverage must be preserved
   - Integration: Test suite must pass

**Tests to Create**:
- None - updating existing tests

**Tests to Run**:
- tests/batchsolving/test_batch_grid_builder.py (entire file)

**Outcomes**:
- Files Modified: 
  * src/cubie/batchsolving/BatchGridBuilder.py (45 lines changed)
  * tests/batchsolving/test_batch_grid_builder.py (85 lines changed)
- Functions/Methods Added/Modified:
  * __call__() in BatchGridBuilder - Removed request parameter from signature and implementation
  * Module docstring - Updated to remove request references and examples
  * test_call_with_request() - Modified to use params/states instead of request
  * test_call_input_types() - Simplified to remove request from parameterization
  * test_call_outputs() - Removed request usage, now uses separate params/states dicts
  * test_docstring_examples() - Removed request examples (Examples 4-5)
  * test_grid_builder_precision_enforcement() - Updated to use params/states dicts
- Implementation Summary:
  Removed the `request` parameter from the `__call__` method signature and implementation. The method now only accepts `params`, `states`, and `kind` arguments. When dict inputs are provided, they are merged internally into a combined request dict for processing by grid_arrays(). The module docstring was updated to reflect the simplified API with 3 arguments instead of 4. All tests were updated to use the new API pattern with separate params and states dicts instead of the combined request dict.
- Issues Flagged: None

---

## Task Group 6: Final Verification and Documentation Cleanup
**Status**: [x]
**Dependencies**: Groups [1, 2, 3, 4, 5]

**Required Context**:
- File: src/cubie/batchsolving/BatchGridBuilder.py (entire file)
- File: src/cubie/batchsolving/solver.py (lines 529-568) - build_grid method
- File: tests/batchsolving/test_batch_grid_builder.py (entire file)

**Input Validation Required**:
- None - verification task

**Tasks**:
1. **Verify all acceptance criteria from user stories**
   - File: N/A (verification)
   - Action: Verify
   - Details:
     ```
     US-1: Single Parameter Sweep
     - Single parameter dict produces correct number of runs
     - Unspecified states use system default values
     - Unspecified parameters use system default values
     - Output arrays have shape (n_variables, n_runs)
     
     US-2: Mixed States and Single Parameter Sweep
     - State override applies to all runs
     - Parameter sweep produces expected run count
     - Other states and parameters use default values
     
     US-3: Combinatorial Grid Generation
     - Combinatorial produces correct product of runs
     - Each value combination is represented
     
     US-4: Simplified API without request Parameter
     - __call__() only accepts params and states (plus kind)
     - Clear error messages when inputs are malformed
     - Docstrings and examples updated
     
     US-5: 1D Input Handling
     - 1D arrays treated as single run (column vector)
     - Partial arrays warn about missing values
     ```
   - Edge cases: All user stories verified
   - Integration: Full system integration test

2. **Ensure grid_arrays method still works for legacy internal use**
   - File: src/cubie/batchsolving/BatchGridBuilder.py
   - Action: Verify
   - Details:
     ```
     # grid_arrays() should remain as internal method
     # Verify existing tests using grid_arrays() still pass
     ```
   - Edge cases: Internal API preserved
   - Integration: BatchGridBuilder internal method

**Tests to Create**:
- None

**Tests to Run**:
- tests/batchsolving/test_batch_grid_builder.py (entire file)
- Run pytest on batchsolving tests to verify no regressions

**Outcomes**:
- Files Modified: 
  * None (verification task only)
- Functions/Methods Added/Modified:
  * None
- Implementation Summary:
  All acceptance criteria verified against implementation:
  
  **US-1 (Single Parameter Sweep)**: Verified. `__call__` handles dict params 
  by updating request dict (line 577). `generate_array` uses `extend_grid_to_array`
  which fills unswept variables with defaults. Empty indices handling (lines 376-379)
  returns defaults tiled to match run count. Test: `test_single_param_dict_sweep`.
  
  **US-2 (Mixed States and Parameter Sweep)**: Verified. Both dicts merge into 
  `request`, `grid_arrays` generates separate arrays, and `combine_grids` handles
  Cartesian product correctly. Test: `test_states_dict_params_sweep`.
  
  **US-3 (Combinatorial Grid Generation)**: Verified. `combine_grids` (lines 
  318-325) computes Cartesian product with `np.repeat` and `np.tile`. 
  `unique_cartesian_product` handles within-category expansion. 
  Test: `test_combinatorial_states_params`.
  
  **US-4 (Simplified API)**: Verified. `__call__` signature (lines 517-522) has
  only `params`, `states`, and `kind`. Module docstring (lines 1-94) and method
  docstring (lines 523-548) document only three parameters. Error messages for
  invalid types provided (lines 582-585, 594-597). Tests: `test_call_with_request`,
  `test_call_input_types`.
  
  **US-5 (1D Input Handling)**: Verified. `_sanitise_arraylike` (lines 722-724)
  converts 1D to column vector. Warning for mismatched sizes (lines 727-733).
  `_trim_or_extend` handles extending with defaults. Tests: 
  `test_1d_param_array_single_run`, `test_1d_state_array_partial_warning`.
  
  **Legacy grid_arrays**: Verified. Method exists (lines 473-515) with 5 existing
  tests covering various scenarios. `Solver.build_grid` (lines 529-568 in solver.py)
  correctly uses new API pattern with `states=` and `params=` arguments.
  
- Issues Flagged: None 

---

# Summary

## Total Task Groups: 6

## Dependency Chain
```
Group 1 (Remove duplicate) ─────────────┐
                                        │
Group 2 (Create tests) ─────────────────┼──> Group 5 (Remove request) ──> Group 6 (Verify)
                                        │
Group 3 (Fix single dict bug) ──────────┤
                                        │
Group 4 (Fix combine_grids) ────────────┘
```

## Tests to Create: 8 new test functions
1. test_single_param_dict_sweep
2. test_single_state_dict_single_run
3. test_states_dict_params_sweep
4. test_combinatorial_states_params
5. test_1d_param_array_single_run
6. test_1d_state_array_partial_warning
7. test_empty_inputs_returns_defaults
8. test_verbatim_single_run_broadcast

## Tests to Run (total)
- tests/batchsolving/test_batch_grid_builder.py (all tests after each group)

## Estimated Complexity
- Group 1: Low (simple deletion)
- Group 2: Medium (8 new tests)
- Group 3: Medium (logic fixes)
- Group 4: Low (small fix)
- Group 5: High (API change + test updates)
- Group 6: Low (verification)

**Overall**: Medium complexity - primarily fixing edge cases and simplifying API
