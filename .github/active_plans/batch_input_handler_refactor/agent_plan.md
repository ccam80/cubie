# BatchInputHandler Refactor: Agent Plan

## Overview

This document provides detailed technical specifications for the detailed_implementer agent to create function-level implementation tasks, and for the reviewer agent to validate the final implementation.

---

## Component 1: BatchInputHandler Class

### Expected Behavior
The renamed `BatchInputHandler` class is the single source of truth for processing solver inputs. It takes raw user inputs (dicts, arrays, device arrays) and returns validated, precision-cast arrays ready for kernel execution.

### Architectural Changes
- Rename file from `BatchGridBuilder.py` to `BatchInputHandler.py`
- Rename class from `BatchGridBuilder` to `BatchInputHandler`
- Merge `_classify_inputs()` method from Solver
- Merge `_validate_arrays()` method from Solver
- Standardize all method argument order to `(states, params)`

### Current Methods to Update

#### `__call__` Method
**Current signature:**
```python
def __call__(
    self,
    params: Optional[Union[Dict, ArrayLike]] = None,
    states: Optional[Union[Dict, ArrayLike]] = None,
    kind: str = "combinatorial",
) -> tuple[np.ndarray, np.ndarray]:
```

**New signature:**
```python
def __call__(
    self,
    states: Optional[Union[Dict, ArrayLike]] = None,
    params: Optional[Union[Dict, ArrayLike]] = None,
    kind: str = "combinatorial",
) -> tuple[np.ndarray, np.ndarray]:
```

**Behavior:** Argument order changes but return value order stays the same (states, params).

#### `_are_right_sized_arrays` Method
**Current signature:**
```python
def _are_right_sized_arrays(
    self,
    inits: Optional[Union[ArrayLike, Dict]],
    params: Optional[Union[ArrayLike, Dict]],
) -> bool:
```

**New signature:**
```python
def _are_right_sized_arrays(
    self,
    states: Optional[Union[ArrayLike, Dict]],
    params: Optional[Union[ArrayLike, Dict]],
) -> bool:
```

**Behavior:** Rename `inits` to `states` for consistency. Logic unchanged.

#### `_try_fast_path_arrays` Method
**Current signature:**
```python
def _try_fast_path_arrays(
    self,
    states: Optional[Union[ArrayLike, Dict]],
    params: Optional[Union[ArrayLike, Dict]],
    kind: str,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
```

**Behavior:** Already uses correct order. No changes needed.

#### `_cast_to_precision` Method
**Current signature:**
```python
def _cast_to_precision(
    self, states: np.ndarray, params: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
```

**Behavior:** Already uses correct order. No changes needed.

#### `_align_run_counts` Method
**Current signature:**
```python
def _align_run_counts(
    self,
    states_array: np.ndarray,
    params_array: np.ndarray,
    kind: str,
) -> tuple[np.ndarray, np.ndarray]:
```

**Behavior:** Already uses correct order. No changes needed.

### New Methods to Add (from Solver)

#### `classify_inputs` Method
**Move from:** `Solver._classify_inputs`  
**New location:** `BatchInputHandler.classify_inputs`

```python
def classify_inputs(
    self,
    states: Union[np.ndarray, Dict[str, Union[float, np.ndarray]]],
    params: Union[np.ndarray, Dict[str, Union[float, np.ndarray]]],
) -> str:
```

**Behavior:** 
- Returns `'dict'` when either input is a dictionary
- Returns `'device'` when both have `__cuda_array_interface__`
- Returns `'array'` when both are correctly-shaped numpy arrays
- Falls back to `'dict'` for edge cases

**Integration:** This method uses `self.states.n` and `self.parameters.n` instead of `self.system_sizes.states/parameters`.

#### `validate_arrays` Method
**Move from:** `Solver._validate_arrays`  
**New location:** `BatchInputHandler.validate_arrays`

```python
def validate_arrays(
    self,
    states: np.ndarray,
    params: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
```

**Behavior:**
- Casts arrays to system precision if needed
- Returns validated arrays in (states, params) order

### Expected Interactions

1. **Solver.solve()** calls `self.input_handler.classify_inputs(states, params)`
2. Based on classification:
   - `'dict'` → calls `self.input_handler(states=..., params=..., kind=...)`
   - `'array'` → calls `self.input_handler.validate_arrays(states, params)`
   - `'device'` → passes through with minimal processing
3. Result always returns `(states_array, params_array)`

---

## Component 2: Solver Class Updates

### Expected Behavior
The Solver class simplifies by delegating all input handling to `BatchInputHandler`. It no longer contains `_classify_inputs` or `_validate_arrays`.

### Architectural Changes

#### Attribute Rename
- `self.grid_builder` → `self.input_handler`
- Type: `BatchInputHandler` (was `BatchGridBuilder`)

#### Method Removal
- Remove `_classify_inputs()` (moved to handler)
- Remove `_validate_arrays()` (moved to handler)

#### `solve()` Method Updates
**Current flow:**
```python
input_type = self._classify_inputs(initial_values, parameters)
if input_type == 'dict':
    inits, params = self.grid_builder(states=initial_values, params=parameters, ...)
elif input_type == 'array':
    inits, params = self._validate_arrays(initial_values, parameters)
else:
    inits, params = initial_values, parameters
```

**New flow:**
```python
input_type = self.input_handler.classify_inputs(
    states=initial_values, 
    params=parameters
)
if input_type == 'dict':
    inits, params = self.input_handler(
        states=initial_values, 
        params=parameters, 
        kind=grid_type
    )
elif input_type == 'array':
    inits, params = self.input_handler.validate_arrays(
        states=initial_values, 
        params=parameters
    )
else:
    inits, params = initial_values, parameters
```

**Key change:** All call sites use explicit keyword arguments.

#### `build_grid()` Method Updates
**Current:**
```python
return self.grid_builder(
    states=initial_values, params=parameters, kind=grid_type
)
```

**New:**
```python
return self.input_handler(
    states=initial_values, 
    params=parameters, 
    kind=grid_type
)
```

---

## Component 3: Module and Package Updates

### File Rename
- `src/cubie/batchsolving/BatchGridBuilder.py` → `src/cubie/batchsolving/BatchInputHandler.py`

### `__init__.py` Updates
**Current:**
```python
from cubie.batchsolving.BatchGridBuilder import BatchGridBuilder
...
"BatchGridBuilder",
```

**New:**
```python
from cubie.batchsolving.BatchInputHandler import BatchInputHandler
# Backward compatibility alias
BatchGridBuilder = BatchInputHandler
...
"BatchInputHandler",
"BatchGridBuilder",  # Deprecated alias
```

### Import Updates in Other Files
Files that import `BatchGridBuilder`:
- `src/cubie/batchsolving/solver.py`
- `tests/batchsolving/test_batch_grid_builder.py`
- `tests/batchsolving/conftest.py`
- `tests/batchsolving/test_solver.py`

---

## Component 4: Test Updates

### Test File Rename
- `tests/batchsolving/test_batch_grid_builder.py` → `tests/batchsolving/test_batch_input_handler.py`

### Test Import Updates
Update all imports from `BatchGridBuilder` to `BatchInputHandler`.

### Fixture Updates in conftest.py
```python
# Old
def batchconfig_instance(system) -> BatchGridBuilder:
    return BatchGridBuilder.from_system(system)

# New
def input_handler(system) -> BatchInputHandler:
    return BatchInputHandler.from_system(system)
```

### New Regression Test
Add test verifying positional argument behavior:

```python
def test_solve_ivp_positional_argument_order(system, solver_settings):
    """Verify positional args to solve_ivp route correctly.
    
    Regression test: states must go to states bucket,
    params must go to params bucket, even without keywords.
    """
    # Create distinct arrays
    n_states = system.sizes.states
    n_params = system.sizes.parameters
    
    # Use distinctive values to verify routing
    states = np.full((n_states, 2), 1.5)
    params = np.full((n_params, 2), 99.0)
    
    result = solve_ivp(
        system,
        states,      # positional: y0
        params,      # positional: parameters
        duration=0.01,
        dt=0.001,
    )
    
    # Verify states went to states bucket
    assert result.initial_values[0, 0] == 1.5
    # Verify params went to params bucket  
    assert result.parameters[0, 0] == 99.0
```

### Test Method Argument Order Updates
Update all test calls that use `(params=..., states=...)` to use `(states=..., params=...)` order.

---

## Component 5: Docstring Updates

### Module Docstring
Update the module docstring at the top of `BatchInputHandler.py` to:
- Reference the new class name
- Describe expanded responsibilities (validation, classification)
- Update example code to use new argument order

### Class Docstring
Update `BatchInputHandler` class docstring to describe:
- Input classification responsibility
- Array validation responsibility
- Grid construction responsibility

### Method Docstrings
Update all method docstrings that reference:
- `params, states` order → `states, params`
- `BatchGridBuilder` → `BatchInputHandler`
- `grid_builder` → `input_handler`

---

## Dependencies and Imports

### BatchInputHandler Imports
The class needs these imports (already present in BatchGridBuilder):
```python
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from numpy.typing import ArrayLike
from cubie.batchsolving.SystemInterface import SystemInterface
from cubie.odesystems.baseODE import BaseODE
from cubie.odesystems.SystemValues import SystemValues
```

### No New Dependencies
This refactor does not introduce new external dependencies.

---

## Edge Cases

### Empty SystemValues
When a system has no parameters (empty `SystemValues`):
- `classify_inputs` should handle `params=None` correctly
- `validate_arrays` should handle empty parameter arrays
- Current logic in `_are_right_sized_arrays` handles this - preserve it

### Mixed Input Types
When one input is dict and other is array:
- `classify_inputs` returns `'dict'`
- Full grid construction path is used
- Existing logic handles this - preserve it

### 1D Array Inputs
When user passes 1D arrays:
- `classify_inputs` returns `'dict'` (falls back)
- Grid builder handles conversion to 2D
- Existing logic handles this - preserve it

### Device Arrays with Wrong Shape
When device arrays have wrong shapes:
- Should NOT be classified as `'device'`
- Should fall back to dict path for proper error handling
- Current `_classify_inputs` already handles this

---

## Data Structures

### Input Types
```python
InputType = Union[
    None,                                           # Use defaults
    Dict[str, Union[float, np.ndarray]],           # Grid specification
    np.ndarray,                                     # Pre-built array
    DeviceNDArray,                                  # CUDA device array
]
```

### Return Type
```python
ProcessedInputs = Tuple[np.ndarray, np.ndarray]  # (states, params)
```

### Classification Result
```python
ClassificationType = Literal['dict', 'array', 'device']
```

---

## Validation Checklist for Reviewer

### Argument Order
- [ ] `BatchInputHandler.__call__` takes `(states, params)` order
- [ ] All internal methods use `(states, params)` order
- [ ] All call sites use explicit keyword arguments
- [ ] Return values are `(states, params)` order

### Functionality Preservation
- [ ] All existing tests pass (after updating imports/names)
- [ ] Fast paths work for device arrays
- [ ] Fast paths work for right-sized numpy arrays
- [ ] Dict inputs produce correct grid expansion
- [ ] Precision casting works correctly

### Code Quality
- [ ] No duplicate validation logic between Solver and handler
- [ ] Docstrings updated to reflect new names
- [ ] Module docstring reflects expanded responsibility
- [ ] Backward compatibility alias exists

### Regression Test
- [ ] Test verifies positional argument routing in solve_ivp
- [ ] Test uses distinctive values to verify correct buckets
- [ ] Test would fail if arguments were swapped
