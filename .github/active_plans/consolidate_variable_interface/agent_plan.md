# Variable Interface Consolidation - Agent Plan

## Overview

This document provides detailed technical specifications for consolidating variable list handling logic into `SystemInterface`. The goal is to eliminate duplicate logic across the user input → output pipeline and ensure consistent behavior for `None`, empty list, and provided values.

---

## Component Descriptions

### 1. SystemInterface (Target: Central Variable Resolution Hub)

**Location**: `src/cubie/batchsolving/SystemInterface.py`

**Current Role**: Wraps `SystemValues` instances for parameters, states, and observables. Provides `state_indices()`, `observable_indices()`, and label lookup methods.

**New Role**: Becomes the single source of truth for:
- Resolving variable labels to indices
- Merging label-based and index-based selections
- Handling None/empty semantics consistently
- Converting user-facing `save_variables`/`summarise_variables` to final index arrays

**New Methods to Add**:

#### `resolve_variable_labels(labels, silent=False)`
- **Purpose**: Convert a list of variable labels (which may be states OR observables) to separate state and observable index arrays
- **Behavior**:
  - If `labels` is `None`, return `(None, None)` to signal "use defaults"
  - If `labels` is empty list/array, return `(np.array([], dtype=np.int32), np.array([], dtype=np.int32))`
  - If `labels` contains valid names, resolve each to either state or observable indices
  - Raise `ValueError` if any label not found in states or observables (unless `silent=True`)
- **Returns**: `Tuple[Optional[np.ndarray], Optional[np.ndarray]]` - (state_indices, observable_indices)

#### `merge_variable_inputs(var_labels, state_indices, observable_indices)`
- **Purpose**: Merge label-based selections with direct index selections using set union
- **Behavior**:
  - Resolve `var_labels` using `resolve_variable_labels()` if not None
  - Handle the "all defaults" case: if all three inputs are None, signal that all variables should be used
  - Handle explicit empty: if any input is explicitly empty list/array, respect that
  - Compute union of resolved label indices with provided index arrays
- **Returns**: `Tuple[np.ndarray, np.ndarray]` - final (state_indices, observable_indices)

#### `convert_variable_labels(output_settings, max_states, max_observables)`
- **Purpose**: Full conversion of output_settings dict, replacing `save_variables`/`summarise_variables` with resolved index arrays
- **Behavior**:
  - Extract `save_variables`, `saved_state_indices`, `saved_observable_indices`
  - Extract `summarise_variables`, `summarised_state_indices`, `summarised_observable_indices`
  - For each pair (labels + indices), call `merge_variable_inputs()`
  - Apply "all" defaults when appropriate (both labels and indices are None)
  - Pop label keys from dict, update with final index arrays
  - Mutate `output_settings` in-place
- **Returns**: `None` (mutates in-place)

---

### 2. Solver (Simplified Delegation)

**Location**: `src/cubie/batchsolving/solver.py`

**Current Role**: Contains `_resolve_labels()`, `_merge_vars_and_indices()`, and `convert_output_labels()` methods that duplicate logic.

**New Role**: Thin wrapper that delegates variable conversion to `SystemInterface`.

**Methods to Modify**:

#### `convert_output_labels(output_settings)` → Becomes delegation only
- **Current**: 112 lines of logic
- **New**: Single call to `self.system_interface.convert_variable_labels(output_settings, self.system_sizes.states, self.system_sizes.observables)`

#### `_resolve_labels()` → Remove
- Move logic to `SystemInterface.resolve_variable_labels()`

#### `_merge_vars_and_indices()` → Remove
- Move logic to `SystemInterface.merge_variable_inputs()`

---

### 3. OutputConfig (Validation Only)

**Location**: `src/cubie/outputhandling/output_config.py`

**Current Role**: Receives index arrays and applies defaults in `_check_saved_indices()` and `_check_summarised_indices()`. Also validates bounds.

**New Role**: Pure validation - no default application. Indices should arrive already resolved.

**Methods to Modify**:

#### `_check_saved_indices()` → Simplify
- **Current**: If indices are empty, replaces with full range `np.arange(max_*, dtype=np.int_)`
- **New**: Only convert to numpy array if needed, no default expansion
- **Rationale**: Defaults are now handled upstream in SystemInterface

#### `_check_summarised_indices()` → Simplify
- **Current**: If indices are empty, copies from saved indices
- **New**: Only convert to numpy array if needed, no default expansion
- **Note**: The "summarised defaults to saved" behavior should move to SystemInterface

---

### 4. OutputFunctions

**Location**: `src/cubie/outputhandling/output_functions.py`

**Current Role**: Factory that builds CUDA output functions. Receives index arrays via constructor.

**New Role**: Unchanged - continues to receive final resolved index arrays.

**No changes required** - already receives processed indices from upstream.

---

## Behavioral Specifications

### None vs Empty vs Provided Semantics

The following truth table defines the canonical behavior:

| `var_labels` | `state_indices` | `obs_indices` | Result State Indices | Result Obs Indices |
|--------------|-----------------|---------------|---------------------|-------------------|
| `None` | `None` | `None` | All states | All observables |
| `None` | `[]` | `None` | Empty | All observables |
| `None` | `None` | `[]` | All states | Empty |
| `None` | `[]` | `[]` | Empty | Empty |
| `[]` | `None` | `None` | Empty | Empty |
| `["x"]` | `None` | `None` | Resolved "x" | Resolved "x" |
| `["x"]` | `[0]` | `None` | Union(resolved "x", [0]) | Resolved "x" |
| `None` | `[0, 1]` | `[2]` | `[0, 1]` | `[2]` |

### Key Implementation Details

1. **Distinguishing None from []**:
   - `None` means "not provided, use defaults"
   - `[]` or `np.array([])` means "explicitly no variables"
   - Use `is None` check, not truthiness

2. **Union Operation**:
   - Use `np.union1d()` to merge index arrays
   - Result should be sorted and unique (union1d guarantees this)
   - Cast to `np.int32` for consistency

3. **Label Resolution Order**:
   - First check states, then observables
   - A label can only match one category (states or observables, not both)
   - CuBIE systems don't allow duplicate names across states/observables

4. **Error Messages**:
   - Include available state and observable names when raising ValueError
   - Use existing pattern from `Solver._resolve_labels()` error message

---

## Integration Points

### Solver.__init__()
```
Current flow:
1. merge_kwargs_into_settings() extracts output_settings
2. self.convert_output_labels(output_settings) mutates settings
3. BatchSolverKernel receives mutated settings

New flow:
1. merge_kwargs_into_settings() extracts output_settings  
2. self.system_interface.convert_variable_labels(output_settings, ...) mutates settings
3. BatchSolverKernel receives mutated settings
```

### Solver.update()
```
Current flow:
1. Copy updates_dict
2. self.convert_output_labels(updates_dict)
3. Forward to kernel

New flow:
1. Copy updates_dict
2. self.system_interface.convert_variable_labels(updates_dict, ...)
3. Forward to kernel
```

### BatchSolverKernel → SingleIntegratorRun → OutputFunctions
No changes - index arrays flow through unchanged.

### OutputConfig.from_loop_settings()
```
Current flow:
1. Receives indices (may be None)
2. Converts None to empty array
3. _check_saved_indices() expands empty to all

New flow:
1. Receives indices (already resolved, never None for defaults)
2. Validates indices
3. No expansion - uses what was provided
```

---

## Edge Cases to Consider

1. **System with no observables**:
   - `max_observables = 0`
   - `observable_indices` should be empty array, not error
   - Resolved observable indices for any labels = empty

2. **Empty system**:
   - `max_states = 0` and `max_observables = 0`
   - Should work but result in empty arrays everywhere

3. **Mixed valid/invalid labels**:
   - Some labels valid, some not
   - Should raise error with all invalid labels listed, not just first

4. **Duplicate labels in input**:
   - `save_variables=["x", "x", "y"]`
   - Should deduplicate via union1d

5. **Label appears in both labels and indices**:
   - `save_variables=["x"]`, `saved_state_indices=[0]` where x is state 0
   - Union should deduplicate

---

## Dependencies and Imports

### SystemInterface New Imports
```python
from typing import Optional, Tuple
import numpy as np
```

### No New External Dependencies
All required functionality exists in numpy and standard library.

---

## Affected Files Summary

| File | Change Type | Scope |
|------|-------------|-------|
| `src/cubie/batchsolving/SystemInterface.py` | Add methods | +3 methods (~80-100 lines) |
| `src/cubie/batchsolving/solver.py` | Simplify | Remove ~60 lines, modify ~10 lines |
| `src/cubie/outputhandling/output_config.py` | Simplify | Modify ~30 lines |
| `tests/batchsolving/test_solver.py` | Verify/Update | May need test updates for new behavior |
| `tests/batchsolving/test_system_interface.py` | Add tests | New tests for SystemInterface methods |

---

## Testing Strategy

### Unit Tests for SystemInterface (New)

1. **resolve_variable_labels()**:
   - Test None returns (None, None)
   - Test empty list returns (empty, empty)
   - Test pure states list
   - Test pure observables list
   - Test mixed states/observables list
   - Test invalid label raises ValueError
   - Test silent mode returns empty arrays for invalid

2. **merge_variable_inputs()**:
   - Test all-None returns "all" signal
   - Test partial None with indices
   - Test labels with None indices
   - Test union of labels and indices
   - Test empty list explicit selection

3. **convert_variable_labels()**:
   - Test save_variables only
   - Test summarise_variables only
   - Test both
   - Test with pre-existing indices
   - Test dict mutation

### Integration Tests

1. **Solver with save_variables**:
   - Existing tests should pass
   - Add tests for empty list behavior

2. **solve_ivp with save_variables**:
   - Existing tests should pass

### Regression Tests

1. **Default behavior unchanged**:
   - No arguments → all variables saved
   
2. **Explicit empty behavior**:
   - Empty list → no variables saved (NEW - verify this is desired)
