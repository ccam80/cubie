# Technical Implementation Plan: save_variables and summarise_variables

## Component Overview

### New Parameters
Two new optional list-of-strings parameters to be added:
- `save_variables`: Optional[List[str]] - Variable names (states or observables) to save in time-domain output
- `summarise_variables`: Optional[List[str]] - Variable names (states or observables) to include in summary calculations

### Modified Components

#### 1. solve_ivp() Function
**Location**: `/home/runner/work/cubie/cubie/src/cubie/batchsolving/solver.py`

**Changes**:
- Add `save_variables: Optional[List[str]] = None` parameter to function signature
- Add `summarise_variables: Optional[List[str]] = None` parameter to function signature
- Forward these parameters through kwargs to `Solver.__init__()`

#### 2. Solver Class
**Location**: `/home/runner/work/cubie/cubie/src/cubie/batchsolving/solver.py`

**Changes to __init__**:
- Accept `save_variables` and `summarise_variables` in kwargs
- These will be merged into `output_settings` via `merge_kwargs_into_settings()` since they're added to `ALL_OUTPUT_FUNCTION_PARAMETERS`

**Changes to solve()**:
- Accept `save_variables` and `summarise_variables` in kwargs
- Forward to `self.update()` which calls `convert_output_labels()`

**Changes to convert_output_labels()**:
- Extend to handle `save_variables` and `summarise_variables` parameters
- Implement name resolution logic
- Merge with existing array-based parameters

#### 3. ALL_OUTPUT_FUNCTION_PARAMETERS Set
**Location**: `/home/runner/work/cubie/cubie/src/cubie/outputhandling/output_functions.py`

**Changes**:
- Add `"save_variables"` to the set
- Add `"summarise_variables"` to the set

## Detailed Behavior Specification

### Name Resolution Logic

The `convert_output_labels()` method will implement the following logic:

1. **Extract save_variables parameter**:
   - Check if `output_settings` contains `"save_variables"`
   - If present, extract the list of variable names

2. **Classify each variable in save_variables**:
   - For each variable name:
     a. Attempt to resolve via `system_interface.state_indices(name, silent=True)`
     b. If successful (no exception), add to temporary state_indices set
     c. Attempt to resolve via `system_interface.observable_indices(name, silent=True)`
     d. If successful (no exception), add to temporary observable_indices set
     e. If neither succeeds, raise ValueError with clear message

3. **Merge with existing saved indices**:
   - If `saved_state_indices` exists in output_settings, union with classified state indices
   - If `saved_observable_indices` exists in output_settings, union with classified observable indices
   - Store results back to output_settings

4. **Repeat for summarise_variables**:
   - Follow same classification logic for `summarise_variables`
   - Merge with `summarised_state_indices` and `summarised_observable_indices`

5. **Remove temporary keys**:
   - Remove `save_variables` and `summarise_variables` from output_settings after processing
   - These are user-facing parameters that should not be passed to OutputFunctions

### Fast-Path Optimization

Implement early exit when neither `save_variables` nor `summarise_variables` are present:
```python
# At start of convert_output_labels
if "save_variables" not in output_settings and "summarise_variables" not in output_settings:
    # Proceed with existing logic only
    pass
```

This ensures zero performance impact when using existing array-only parameters.

### Edge Cases

1. **Empty lists**: `save_variables=[]` should be treated as no-op, not error
2. **None values**: `save_variables=None` should be ignored (parameter not provided)
3. **Duplicate names**: Union operation naturally handles duplicates
4. **Mixed types**: If variable exists as both state and observable (unlikely but possible), add to both index arrays
5. **Invalid names**: Raise ValueError with list of unrecognized names and suggestions

## Integration Architecture

### Current Flow
```
solve_ivp() 
  -> Solver.__init__()
    -> merge_kwargs_into_settings() for output_settings
    -> convert_output_labels(output_settings)
      -> system_interface.state_indices() for saved_states
      -> system_interface.observable_indices() for saved_observables
      -> Replace label keys with index keys
  -> Solver.solve()
    -> self.update() if kwargs present
      -> convert_output_labels() again if output settings in updates
```

### Modified Flow
```
solve_ivp(save_variables=["x", "obs1"])
  -> Solver.__init__(save_variables=["x", "obs1"])
    -> merge_kwargs_into_settings() recognizes save_variables
    -> convert_output_labels(output_settings)
      -> Classify save_variables entries:
        "x" -> state_indices via system_interface
        "obs1" -> observable_indices via system_interface
      -> Union classified indices with existing saved_*_indices
      -> Remove save_variables from output_settings
      -> Proceed with existing label-to-index resolution
  -> Solver.solve()
    -> (same as before)
```

### SystemInterface Methods Used

**state_indices(keys_or_indices, silent=False)**:
- Accepts: str, int, List[Union[str, int]], or None
- Returns: np.ndarray of int16 indices
- Behavior: Calls `SystemValues.get_indices()`
- Error handling: Raises KeyError if name not found (unless silent=True)

**observable_indices(keys_or_indices, silent=False)**:
- Accepts: str, int, List[Union[str, int]], or None
- Returns: np.ndarray of int16 indices
- Behavior: Calls `SystemValues.get_indices()`
- Error handling: Raises KeyError if name not found (unless silent=True)

### Classification Algorithm

```python
def classify_variables(var_names, system_interface):
    """Classify variable names into states and observables.
    
    Returns:
        state_indices: np.ndarray of state indices
        observable_indices: np.ndarray of observable indices
    """
    state_list = []
    observable_list = []
    unrecognized = []
    
    for name in var_names:
        found_as_state = False
        found_as_observable = False
        
        # Try as state
        try:
            idx = system_interface.state_indices([name], silent=False)
            state_list.extend(idx.tolist())
            found_as_state = True
        except (KeyError, IndexError):
            pass
        
        # Try as observable
        try:
            idx = system_interface.observable_indices([name], silent=False)
            observable_list.extend(idx.tolist())
            found_as_observable = True
        except (KeyError, IndexError):
            pass
        
        if not found_as_state and not found_as_observable:
            unrecognized.append(name)
    
    if unrecognized:
        raise ValueError(
            f"Variables not found in states or observables: {unrecognized}. "
            f"Available states: {system_interface.states.names}. "
            f"Available observables: {system_interface.observables.names}."
        )
    
    return (
        np.array(state_list, dtype=np.int16) if state_list else np.array([], dtype=np.int16),
        np.array(observable_list, dtype=np.int16) if observable_list else np.array([], dtype=np.int16)
    )
```

## Dependencies and Imports

No new imports required. All necessary components are already imported:
- `SystemInterface` is already a member of `Solver`
- `numpy` is already imported
- `typing.List, Optional` are already imported

## Data Structures

### Input Format
```python
save_variables = ["x0", "x1", "observable1"]  # List of strings
summarise_variables = ["x0", "observable2"]   # List of strings
```

### Internal Format (after classification)
```python
# Classified into separate arrays
saved_state_indices = np.array([0, 1], dtype=np.int16)
saved_observable_indices = np.array([0], dtype=np.int16)
summarised_state_indices = np.array([0], dtype=np.int16)
summarised_observable_indices = np.array([1], dtype=np.int16)
```

### Output Format (to OutputFunctions)
The same as before - `OutputFunctions` receives arrays of indices, unchanged:
```python
output_config = OutputConfig(
    saved_state_indices=np.array([0, 1], dtype=np.int16),
    saved_observable_indices=np.array([0], dtype=np.int16),
    # ... other parameters
)
```

## Error Messages

### Variable Not Found
```
ValueError: Variables not found in states or observables: ['xyz', 'abc'].
Available states: ['x0', 'x1', 'x2'].
Available observables: ['observable1', 'observable2'].
```

### Duplicate Parameter Conflict
```
ValueError: Cannot specify both save_variables and saved_states/saved_state_indices 
in conflicting ways. Use either the unified save_variables parameter or the 
separate saved_states/saved_observables parameters, not both.
```
(Note: This error should NOT occur if using union semantics. Only raise if there's truly a conflict, which shouldn't happen with union approach.)

## Compatibility Notes

### Backward Compatibility
All existing parameters work exactly as before:
- `saved_states`, `saved_state_indices`
- `saved_observables`, `saved_observable_indices`
- `summarised_states`, `summarised_state_indices`
- `summarised_observables`, `summarised_observable_indices`

### Forward Compatibility
The new parameters can be used:
- Alone: `save_variables=["x", "obs1"]`
- With existing parameters: `saved_states=[0], save_variables=["obs1"]`
- Mixed with indices: `saved_state_indices=np.array([0]), save_variables=["obs1"]`

All combinations use union semantics - no conflicts.

## Testing Strategy

### Unit Tests Required
1. Test `save_variables` with pure state names
2. Test `save_variables` with pure observable names
3. Test `save_variables` with mixed state and observable names
4. Test `summarise_variables` with same variations
5. Test union with existing `saved_states` parameter
6. Test union with existing `saved_state_indices` parameter
7. Test empty list handling
8. Test None value handling
9. Test invalid variable names (should raise clear error)
10. Test performance: array-only path should not call name resolution

### Integration Tests Required
1. Test full solve with `save_variables` producing correct output
2. Test full solve with `summarise_variables` producing correct summaries
3. Test backward compatibility: existing tests should pass unchanged

### Test Location
`/home/runner/work/cubie/cubie/tests/batchsolving/test_solver.py`

## Performance Considerations

### Fast-Path Guarantee
When neither `save_variables` nor `summarise_variables` are present, the method should:
1. Check for presence of these keys (O(1) dict lookup)
2. Skip all name resolution logic if absent
3. Proceed with existing array-based logic only

This ensures zero overhead for existing workflows.

### Name Resolution Cost
When using string-based parameters:
- Classification happens once during solver initialization
- Cost is O(N*M) where N=number of variable names, M=number of states+observables
- In practice, M is small (<100 typically) and N is small (<10 typically)
- Total cost is negligible compared to kernel compilation and execution

### Memory Impact
- No additional memory allocated during kernel execution
- Temporary arrays during classification are small and short-lived
- Final indices arrays are same size as before

## Documentation Requirements

### Docstring Updates

**solve_ivp()**:
Add parameter documentation:
```
save_variables : list of str, optional
    Variable names (states or observables) to save in time-domain output.
    Alternative to specifying saved_states and saved_observables separately.
    Can be combined with existing parameters using union semantics.
    
summarise_variables : list of str, optional
    Variable names (states or observables) to include in summary calculations.
    Alternative to specifying summarised_states and summarised_observables.
    Can be combined with existing parameters using union semantics.
```

**Solver.__init__() and Solver.solve()**:
Same parameter documentation as above.

**Solver.convert_output_labels()**:
Update docstring to mention new parameters:
```
Notes
-----
Accepts both traditional parameters (saved_states, saved_observables, etc.)
and unified parameters (save_variables, summarise_variables). The unified
parameters are automatically classified into states and observables using
SystemInterface. Results are merged using set union.
```

## Implementation Order

1. Add parameter names to `ALL_OUTPUT_FUNCTION_PARAMETERS`
2. Modify `solve_ivp()` signature to accept new parameters
3. Modify `Solver.solve()` signature to accept new parameters (via kwargs)
4. Implement classification logic in `convert_output_labels()`
5. Add unit tests for classification logic
6. Add integration tests for full solve workflow
7. Update docstrings
8. Verify backward compatibility with existing tests

## Notes for Reviewer Agent

The implementation should:
- Maintain zero performance overhead for existing array-based workflows
- Use union semantics to avoid conflicts between old and new parameters
- Provide clear error messages for invalid variable names
- Leverage existing `SystemInterface` methods without modification
- Keep changes surgical and minimal
- Not modify `OutputFunctions` or any GPU kernel code
