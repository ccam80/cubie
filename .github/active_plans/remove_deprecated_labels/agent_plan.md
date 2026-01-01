# Remove Deprecated Label Parameters - Agent Implementation Plan

## Overview

This plan removes deprecated backward-compatibility code for label-based output parameters while preserving the unified `save_variables`/`summarise_variables` interface and read-only property access to variable labels.

## Component Descriptions

### 1. ALL_OUTPUT_FUNCTION_PARAMETERS Constant

**Location:** `src/cubie/outputhandling/output_functions.py`

**Current State:**
The constant includes four deprecated label-based parameters that are no longer needed:
- `"saved_states"`
- `"saved_observables"`
- `"summarised_states"`
- `"summarised_observables"`

**Expected Behavior After Changes:**
The constant should contain only:
- `"output_types"`
- `"save_variables"` (unified parameter)
- `"summarise_variables"` (unified parameter)
- `"saved_state_indices"` (index array)
- `"saved_observable_indices"` (index array)
- `"summarised_state_indices"` (index array)
- `"summarised_observable_indices"` (index array)
- `"dt_save"`
- `"precision"`

### 2. Solver.convert_output_labels() Method

**Location:** `src/cubie/batchsolving/solver.py` (lines ~267-383)

**Current State:**
The method contains three main sections:
1. `resolvers` dictionary - maps parameter names to resolver functions
2. `labels2index_keys` dictionary - maps deprecated label keys to index keys
3. Processing loop - renames deprecated keys to index keys
4. `save_variables`/`summarise_variables` processing

**Expected Behavior After Changes:**

The method should:
1. Remove deprecated entries from `resolvers` dictionary:
   - Remove `"saved_states"` entry
   - Remove `"saved_observables"` entry
   - Remove `"summarised_states"` entry
   - Remove `"summarised_observables"` entry
   - Keep `"saved_state_indices"`, `"saved_observable_indices"`, `"summarised_state_indices"`, `"summarised_observable_indices"` entries

2. Remove entire `labels2index_keys` dictionary (no longer needed)

3. Remove the key renaming loop (lines ~336-345, the `for inkey, outkey in labels2index_keys.items()` block)

4. Keep `save_variables`/`summarise_variables` processing unchanged

**Simplified Logic Flow:**
```
1. Process any remaining index parameters through resolvers (if they contain labels)
2. Process save_variables parameter (classify and merge)
3. Process summarise_variables parameter (classify and merge)
```

### 3. Solver Properties (NO CHANGES)

**Location:** `src/cubie/batchsolving/solver.py` (lines ~1032-1070)

**Current State:**
Four properties expose variable labels:
- `saved_states` - returns list of state labels from indices
- `saved_observables` - returns list of observable labels from indices
- `summarised_states` - returns list of state labels from indices
- `summarised_observables` - returns list of observable labels from indices

**Expected Behavior:**
**KEEP THESE PROPERTIES UNCHANGED** - they provide read-only access to labels and are not deprecated.

### 4. SolveSpec Attrs Class (NO CHANGES)

**Location:** `src/cubie/batchsolving/solveresult.py` (lines ~111-114)

**Current State:**
Contains four attrs fields for label lists:
- `saved_states: Optional[List[str]]`
- `saved_observables: Optional[List[str]]`
- `summarised_states: Optional[List[str]]`
- `summarised_observables: Optional[List[str]]`

**Expected Behavior:**
**KEEP THESE FIELDS UNCHANGED** - they store metadata about which variables were saved/summarised.

## Architectural Changes

### Before: convert_output_labels() Structure

```python
def convert_output_labels(self, output_settings):
    # Define resolvers for ALL parameters (including deprecated)
    resolvers = {
        "saved_states": ...,           # DEPRECATED - REMOVE
        "saved_observables": ...,      # DEPRECATED - REMOVE
        "summarised_states": ...,      # DEPRECATED - REMOVE
        "summarised_observables": ..., # DEPRECATED - REMOVE
        "saved_state_indices": ...,    # KEEP
        "saved_observable_indices": ..., # KEEP
        # ... etc
    }
    
    # Define mapping from deprecated to new keys
    labels2index_keys = {              # REMOVE ENTIRE DICT
        "saved_states": "saved_state_indices",
        "saved_observables": "saved_observable_indices",
        # ... etc
    }
    
    # Resolve labels to indices for ALL keys
    for key, resolver in resolvers.items():
        values = output_settings.get(key)
        if values is not None:
            output_settings[key] = resolver(values)
    
    # Rename deprecated keys to index keys
    for inkey, outkey in labels2index_keys.items():  # REMOVE THIS LOOP
        indices = output_settings.pop(inkey, None)
        if indices is not None:
            if output_settings.get(outkey, None) is not None:
                raise ValueError(...)
            output_settings[outkey] = indices
    
    # Process save_variables (KEEP THIS)
    if "save_variables" in output_settings:
        # ... classification and merging
    
    # Process summarise_variables (KEEP THIS)
    if "summarise_variables" in output_settings:
        # ... classification and merging
```

### After: convert_output_labels() Structure

```python
def convert_output_labels(self, output_settings):
    # Define resolvers ONLY for index parameters
    resolvers = {
        "saved_state_indices": self.system_interface.state_indices,
        "saved_observable_indices": self.system_interface.observable_indices,
        "summarised_state_indices": self.system_interface.state_indices,
        "summarised_observable_indices": self.system_interface.observable_indices,
    }
    
    # Resolve labels to indices for index parameters only
    for key, resolver in resolvers.items():
        values = output_settings.get(key)
        if values is not None:
            output_settings[key] = resolver(values)
    
    # Process save_variables (UNCHANGED)
    if "save_variables" in output_settings:
        # ... classification and merging
    
    # Process summarise_variables (UNCHANGED)
    if "summarise_variables" in output_settings:
        # ... classification and merging
```

## Integration Points

### 1. Solver Constructor

**Location:** `src/cubie/batchsolving/solver.py` (Solver.__init__)

**Current Flow:**
```python
def __init__(self, ...):
    # Constructor accepts **kwargs
    # Filters kwargs through ALL_OUTPUT_FUNCTION_PARAMETERS
    # Calls convert_output_labels(output_settings)
    # Passes to OutputFunctions/OutputConfig
```

**Expected Behavior After Changes:**
- Constructor continues to accept **kwargs
- Filtering removes deprecated params (not in ALL_OUTPUT_FUNCTION_PARAMETERS)
- `convert_output_labels()` no longer processes deprecated params
- Warning or error if deprecated params provided (automatically handled by filtering)

### 2. Solver.update() Method

**Location:** `src/cubie/batchsolving/solver.py` (Solver.update)

**Current Flow:**
```python
def update(self, updates_dict=None, **kwargs):
    # Merges updates
    # Calls convert_output_labels(updates_dict)
    # Updates kernel configuration
```

**Expected Behavior After Changes:**
- Same flow, but `convert_output_labels()` no longer handles deprecated params
- Unrecognized params (including deprecated ones) raise KeyError if not silent

### 3. OutputConfig.from_loop_settings()

**Location:** `src/cubie/outputhandling/output_config.py`

**Current State:**
Already expects index-based parameters only (no label parameters).

**Expected Behavior:**
No changes needed - already works correctly with index-based parameters.

### 4. Test Suite

**Location:** `tests/batchsolving/test_solver.py` and related test files

**Current State:**
Multiple tests use deprecated parameters:
- `test_backward_compatibility_existing_params` - uses `saved_states` parameter
- `test_save_variables_union_with_saved_states` - uses both old and new
- Other tests may use deprecated params

**Expected Behavior After Changes:**
- Tests using deprecated params must be updated to use:
  - `save_variables` / `summarise_variables` (preferred)
  - Or index-based params (`saved_state_indices`, etc.)
- Tests verifying properties still work (properties not deprecated)
- May need new test verifying deprecated params are rejected

## Edge Cases

### 1. Empty/None Parameters

**Scenario:** User provides `save_variables=None` or `save_variables=[]`

**Expected Behavior:** Should work correctly (existing behavior preserved)

### 2. Mixed Index and Unified Parameters

**Scenario:** User provides both `saved_state_indices=[0, 1]` and `save_variables=["x"]`

**Expected Behavior:** Indices merged using set union (existing behavior preserved)

### 3. Invalid Variable Names

**Scenario:** User provides `save_variables=["nonexistent"]`

**Expected Behavior:** Raises ValueError from `_classify_variables()` (existing behavior preserved)

### 4. Duplicate Indices

**Scenario:** After merging, indices contain duplicates

**Expected Behavior:** OutputConfig validation catches duplicates (existing behavior)

### 5. Property Access Before Solving

**Scenario:** User accesses `solver.saved_states` immediately after construction

**Expected Behavior:** Properties return correct labels (existing behavior)

## Data Structures

### Input Parameters (what users provide)

**Deprecated (removed):**
```python
{
    "saved_states": ["x", "y"],           # REMOVED
    "saved_observables": ["output"],      # REMOVED
    "summarised_states": ["x"],           # REMOVED
    "summarised_observables": [],         # REMOVED
}
```

**Current (kept):**
```python
{
    # Unified parameters (recommended)
    "save_variables": ["x", "y", "output"],
    "summarise_variables": ["x"],
    
    # Index parameters (advanced)
    "saved_state_indices": [0, 1],
    "saved_observable_indices": [0],
    "summarised_state_indices": [0],
    "summarised_observable_indices": [],
}
```

### Internal Representation (unchanged)

OutputConfig and OutputFunctions always work with indices:
```python
{
    "saved_state_indices": np.array([0, 1], dtype=np.int_),
    "saved_observable_indices": np.array([0], dtype=np.int_),
    "summarised_state_indices": np.array([0], dtype=np.int_),
    "summarised_observable_indices": np.array([], dtype=np.int_),
}
```

## Dependencies

### Modified Files
1. `src/cubie/outputhandling/output_functions.py` - update constant
2. `src/cubie/batchsolving/solver.py` - simplify convert_output_labels()
3. `tests/batchsolving/test_solver.py` - update tests
4. Other test files using deprecated parameters

### Unchanged Files
1. `src/cubie/outputhandling/output_config.py` - no changes needed
2. `src/cubie/batchsolving/solveresult.py` - properties/fields kept
3. Other files not directly using these parameters

## Testing Requirements

### Unit Tests to Update

1. **Tests using deprecated params as input:**
   - Convert to `save_variables`/`summarise_variables`
   - Or use index-based params

2. **Tests verifying deprecated param behavior:**
   - Remove or repurpose these tests
   - Add test verifying deprecated params raise errors

3. **Tests using properties:**
   - Keep these tests unchanged (properties not deprecated)

### Integration Tests

- Verify full solve workflow with new parameters
- Verify solver.update() works with new parameters
- Verify SolveResult contains correct label lists

### Validation Tests

- Test that providing deprecated params raises appropriate error
- Test that unified params work correctly
- Test that index params work correctly
- Test that properties return correct labels

## Documentation Updates

### Docstrings to Update

1. **`convert_output_labels()` docstring:**
   - Remove mentions of deprecated parameters
   - Update examples to use unified params

2. **`Solver.__init__()` docstring:**
   - Remove deprecated parameter descriptions
   - Keep property descriptions

3. **`ALL_OUTPUT_FUNCTION_PARAMETERS` docstring/comment:**
   - Update to reflect removed entries

### User-Facing Documentation

- Migration guide showing old vs. new usage
- Changelog entry documenting breaking change
- README updates if applicable

## Implementation Order

1. **Update ALL_OUTPUT_FUNCTION_PARAMETERS constant**
   - Remove four deprecated entries
   - Update any associated comments

2. **Simplify convert_output_labels() method**
   - Remove deprecated entries from resolvers
   - Remove labels2index_keys dictionary
   - Remove key renaming loop
   - Keep save_variables/summarise_variables logic

3. **Update docstrings**
   - Remove deprecated parameter documentation
   - Update examples

4. **Update tests**
   - Convert tests using deprecated params
   - Add test verifying deprecated params rejected
   - Verify property tests still pass

5. **Verify all tests pass**
   - Run full test suite
   - Check for any remaining references to deprecated params

## Success Criteria

- [ ] Four deprecated entries removed from ALL_OUTPUT_FUNCTION_PARAMETERS
- [ ] resolvers dictionary has 4 entries (one per index param type)
- [ ] labels2index_keys dictionary completely removed
- [ ] Key renaming loop removed from convert_output_labels()
- [ ] save_variables/summarise_variables processing unchanged
- [ ] Solver properties (saved_states, etc.) unchanged
- [ ] SolveSpec fields unchanged
- [ ] All tests updated to new interface
- [ ] All tests passing
- [ ] Net reduction in lines of code achieved
- [ ] Docstrings updated to remove deprecated param references
