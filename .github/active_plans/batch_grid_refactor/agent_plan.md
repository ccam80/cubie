# BatchGridBuilder Refactoring - Agent Plan

## Overview

This document provides the technical specification for refactoring the `BatchGridBuilder` module to eliminate the combined dictionary pattern. The goal is to keep `params` and `states` separate throughout processing, combining only at the final alignment step.

---

## Component Architecture

### Current Components (Before Refactoring)

The module currently contains:

1. **Module-level functions:**
   - `unique_cartesian_product(arrays)` - Pure utility for Cartesian product
   - `combinatorial_grid(request, values_instance)` - Dict → grid with expansion
   - `verbatim_grid(request, values_instance)` - Dict → grid without expansion
   - `generate_grid(request, values_instance, kind)` - Dispatcher for above two
   - `combine_grids(grid1, grid2, kind)` - Align two grids
   - `extend_grid_to_array(grid, indices, default_values)` - Fill with defaults
   - `generate_array(request, values_instance, kind)` - Dict → complete array

2. **BatchGridBuilder class:**
   - `__init__(interface)` - Stores parameters, states, precision
   - `from_system(system)` - Factory method
   - `grid_arrays(request, kind)` - **TO REMOVE** - Combined dict processing
   - `__call__(params, states, kind)` - Main entry point
   - `_trim_or_extend(arr, values_object)` - Array sizing helper
   - `_sanitise_arraylike(arr, values_object)` - Convert to 2D array
   - `_cast_to_precision(states, params)` - Precision casting
   - Static wrappers for module functions - **TO REMOVE**

### Target Components (After Refactoring)

1. **Module-level functions (retained):**
   - `unique_cartesian_product(arrays)` - Unchanged
   - `expand_dict_to_grid(request, values_instance, kind)` - Renamed/combined from `combinatorial_grid`, `verbatim_grid`, `generate_grid`
   - `combine_grids(grid1, grid2, kind)` - Unchanged
   - `extend_grid_to_array(grid, indices, default_values)` - Unchanged

2. **BatchGridBuilder class (refactored):**
   - `__init__(interface)` - Unchanged
   - `from_system(system)` - Unchanged
   - `__call__(params, states, kind)` - Refactored main entry point
   - `_process_input(input_data, values_object, kind)` - **NEW** - Unified processing
   - `_align_run_counts(states_array, params_array, kind)` - **NEW** - Grid alignment
   - `_trim_or_extend(arr, values_object)` - Unchanged
   - `_sanitise_arraylike(arr, values_object)` - Unchanged  
   - `_cast_to_precision(states, params)` - Unchanged

---

## Expected Behavior

### `__call__(params, states, kind)` - Refactored

The main entry point handles all input combinations with separate processing paths:

**Input Handling:**
- `params` and `states` are processed independently via `_process_input()`
- Neither is ever merged into a combined dictionary
- Each path produces a 2D array in (variable, run) format

**Processing Flow:**
1. Update precision from current system state
2. Process `params` → `params_array` via `_process_input()`
3. Process `states` → `states_array` via `_process_input()`
4. Align run counts via `_align_run_counts()`
5. Cast to precision via `_cast_to_precision()`

**Edge Cases:**
- Both None → Return single-run defaults
- One None → Use defaults for missing, process provided
- Both arrays → Skip dict expansion, go straight to alignment
- Both dicts → Expand each independently, then align
- Mixed (one dict, one array) → Process each appropriately, align

### `_process_input(input_data, values_object, kind)` - New

Unified processing for either params or states:

**Input Types:**
- `None` → Return single-column default array
- `dict` → Convert to grid via `expand_dict_to_grid()`, extend with defaults
- `list/tuple/array` → Sanitize via `_sanitise_arraylike()`

**Output:**
- Always returns 2D array in (variable, run) format
- All variables included (defaults filled for unspecified)
- Ready for alignment step

### `_align_run_counts(states_array, params_array, kind)` - New

Final alignment of independently processed arrays:

**Behavior:**
- Wraps `combine_grids()` with appropriate error handling
- For `kind="combinatorial"`: Cartesian product of runs
- For `kind="verbatim"`: Direct pairing (with single-run broadcast)

**Output:**
- Tuple of aligned (states_array, params_array)
- Both have identical run counts (shape[1])

### `expand_dict_to_grid(request, values_instance, kind)` - Renamed

Consolidates `combinatorial_grid`, `verbatim_grid`, `generate_grid`:

**Behavior:**
- Single function dispatching based on `kind`
- Returns `(indices, grid)` tuple
- Grid is partial (only requested variables)

**Note:** This is a simplification - the three existing functions can remain separate if preferred, with `generate_grid` as the dispatcher. The key change is that they only process a single category (params OR states) at a time, never a combined dict.

---

## Integration Points

### With SystemInterface

- `BatchGridBuilder.__init__()` receives `interface.parameters` and `interface.states`
- These are `SystemValues` instances providing:
  - `.names` - Variable names for key lookup
  - `.values_array` - Default values
  - `.n` - Variable count
  - `.precision` - NumPy dtype

### With Solver.solve()

```python
# solver.py lines 489-491
inits, params = self.grid_builder(
    states=initial_values, params=parameters, kind=grid_type
)
```

- No changes needed to the call site
- API remains: `__call__(params, states, kind)` → `(states_array, params_array)`

### With Solver.build_grid()

```python
# solver.py lines 566-568
return self.grid_builder(
    states=initial_values, params=parameters, kind=grid_type
)
```

- No changes needed to the call site

---

## Data Structures

### Input Types

```
params: Optional[Union[Dict[str, ArrayLike], np.ndarray, list, tuple]]
states: Optional[Union[Dict[str, ArrayLike], np.ndarray, list, tuple]]
kind: str  # "combinatorial" or "verbatim"
```

### Internal Arrays

All internal arrays use (variable, run) format:
- `shape[0]` = number of variables
- `shape[1]` = number of runs
- Contiguous in memory for CUDA coalescing

### Output

```
Tuple[np.ndarray, np.ndarray]  # (states_array, params_array)
```

Both arrays have:
- Matching `shape[1]` (run count)
- `dtype` matching system precision
- C-contiguous memory layout

---

## Dependencies and Imports

### Required (existing)

```python
from itertools import product
from typing import Dict, List, Optional, Union
from warnings import warn

import numpy as np
from numpy.typing import ArrayLike

from cubie.batchsolving.SystemInterface import SystemInterface
from cubie.odesystems.baseODE import BaseODE
from cubie.odesystems.SystemValues import SystemValues
```

### No new dependencies required

---

## Edge Cases

### Empty Dict Values

```python
params = {"p0": [1, 2], "p1": []}  # p1 is empty
```
- Empty values should be filtered out before grid generation
- Existing behavior in `combinatorial_grid`/`verbatim_grid` handles this

### Mismatched Verbatim Lengths

```python
# kind="verbatim" with different lengths
states = {"x0": [1, 2, 3]}     # 3 runs
params = {"p0": [1, 2]}        # 2 runs
```
- Should raise `ValueError` (existing behavior in `combine_grids`)
- Exception: single-run grids broadcast to match

### Partial Arrays

```python
# Array with fewer variables than system expects
states = np.array([[1, 2], [3, 4]])  # 2 vars, 2 runs
# But system has 3 states
```
- Trigger warning and fill missing with defaults
- Handled by `_sanitise_arraylike` → `_trim_or_extend`

### Scalar Dict Values

```python
params = {"p0": 0.5}  # Single scalar, not array
```
- Treated as single-run sweep
- Convert to `[0.5]` before processing

---

## Test Updates Required

### Tests to Modify

1. `test_grid_arrays` - Update to test new internal methods or remove
2. `test_call_with_request` - Ensure still works (API unchanged)
3. `test_combinatorial_and_verbatim_grid` - Keep, tests module functions
4. `test_generate_grid` - Keep, tests module functions
5. `test_generate_array` - Keep or remove based on function retention

### New Tests to Add

1. `test_process_input_dict` - Test `_process_input` with dict input
2. `test_process_input_array` - Test `_process_input` with array input
3. `test_process_input_none` - Test `_process_input` with None
4. `test_align_run_counts_combinatorial` - Test alignment with expansion
5. `test_align_run_counts_verbatim` - Test alignment with pairing

### Tests to Keep Unchanged

- `test_unique_cartesian_product`
- `test_combine_grids`
- `test_extend_grid_to_array`
- `test_call_input_types`
- `test_call_outputs`
- `test_docstring_examples`
- All user-facing API tests

---

## Removed Components

### `grid_arrays()` Method

This method currently:
1. Takes combined request dict
2. Separates into param_request and state_request
3. Generates arrays for each
4. Combines via `combine_grids`

This is exactly what `__call__()` should do directly with separate inputs. Remove entirely.

### Static Method Wrappers

The class has static methods that just call module functions:
- `BatchGridBuilder.unique_cartesian_product`
- `BatchGridBuilder.combinatorial_grid`
- `BatchGridBuilder.verbatim_grid`
- `BatchGridBuilder.generate_grid`
- `BatchGridBuilder.combine_grids`
- `BatchGridBuilder.extend_grid_to_array`
- `BatchGridBuilder.generate_array`

These exist for backward compatibility when the class shadows the module. Evaluate if tests import via class or module. If tests import module directly (`import cubie.batchsolving.BatchGridBuilder as batchgridmodule`), the static wrappers can be removed.

**Check current test imports before removing.**

---

## Implementation Strategy

### Phase 1: Add New Methods

1. Add `_process_input()` method
2. Add `_align_run_counts()` method
3. Keep existing code working

### Phase 2: Refactor `__call__()`

1. Rewrite `__call__()` to use new methods
2. Remove intermediate `request` dictionary
3. Remove call to `grid_arrays()`

### Phase 3: Remove Deprecated Code

1. Remove `grid_arrays()` method
2. Evaluate static wrapper removal
3. Update module docstring

### Phase 4: Update Tests

1. Update tests that relied on `grid_arrays()`
2. Add tests for new internal methods
3. Verify all existing API tests pass

---

## Validation Criteria

1. All existing tests pass (except those testing removed internals)
2. `__call__()` never combines params and states into single dict
3. Data flow is traceable: params in → params out, states in → states out
4. No new public API (only internal restructuring)
5. Code line count is reduced from current implementation
