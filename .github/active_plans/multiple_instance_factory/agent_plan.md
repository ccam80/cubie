# MultipleInstanceCUDAFactory Agent Plan

## Overview

This plan describes the implementation of `MultipleInstanceCUDAFactory`, a new base class in `CUDAFactory.py` that centralizes prefixed configuration key handling for CUDAFactory subclasses supporting multiple distinct instances.

---

## Component Descriptions

### 1. MultipleInstanceCUDAFactory (New Class)

**Location**: `src/cubie/CUDAFactory.py`

**Purpose**: Base class for CUDAFactory subclasses that need to differentiate configuration parameters by instance. Provides automatic mapping from prefixed external keys (e.g., `krylov_atol`) to unprefixed internal keys (e.g., `atol`).

**Expected Behavior**:
- Accepts `instance_label` parameter in `__init__`
- Stores `instance_label` as an instance attribute
- Overrides `update_compile_settings()` to intercept the updates dict
- Before calling super's `update_compile_settings()`:
  1. Makes a copy of the updates dict
  2. Scans for keys that start with `{instance_label}_`
  3. For each matching key, adds an entry with the unprefixed version
  4. Both prefixed and unprefixed forms can coexist; prefixed takes precedence
- Provides convenience method `get_prefixed_setting(key)` to access settings with automatic prefix handling

**Attributes**:
- `instance_label: str` - The prefix used to identify settings for this instance (e.g., "krylov", "newton")

**Methods**:
- `__init__(instance_label: str)` - Initialize with instance label, call `super().__init__()`
- `update_compile_settings(updates_dict, silent, **kwargs) -> Set[str]` - Override to transform prefixed keys
- `get_prefixed_setting(key: str) -> Any` - Convenience method to get a setting value, trying prefixed version first

### 2. MatrixFreeSolver (Modified Class)

**Location**: `src/cubie/integrators/matrix_free_solvers/base_solver.py`

**Current State**: 
- Inherits from `CUDAFactory`
- Has `settings_prefix` attribute set in `__init__`
- Has `_extract_prefixed_tolerance()` method for manual prefix handling

**Expected Behavior After Refactor**:
- Inherits from `MultipleInstanceCUDAFactory`
- Passes `instance_label` (formerly `settings_prefix`) to super's `__init__`
- The `_extract_prefixed_tolerance()` method can be simplified since prefix handling is now in base class
- The `_update_norm_and_config()` method remains but works with the transformed updates

**Key Changes**:
- Change inheritance: `class MatrixFreeSolver(MultipleInstanceCUDAFactory)`
- Update `__init__` to call `super().__init__(instance_label=settings_prefix)`
- Rename `settings_prefix` parameter to `instance_label` for consistency
- Simplify or keep `_extract_prefixed_tolerance()` for norm-specific tolerance extraction (this is a specialized case beyond generic prefix mapping)

### 3. LinearSolver (Modified Class)

**Location**: `src/cubie/integrators/matrix_free_solvers/linear_solver.py`

**Current State**:
- Inherits from `MatrixFreeSolver`
- `update()` method manually handles `krylov_atol` and `krylov_rtol` extraction
- Calls `self._extract_prefixed_tolerance()` to get norm updates

**Expected Behavior After Refactor**:
- Inherits from `MatrixFreeSolver` (unchanged)
- `update()` method leverages base class prefix handling
- The tolerance extraction for the norm factory is a specialized case that remains in `_extract_prefixed_tolerance()` since it routes to a separate object (norm factory)
- Compile settings updates benefit from automatic prefix stripping

**Note**: The norm tolerance handling is separate from compile settings because tolerances go to a nested `ScaledNorm` factory, not the compile settings. The `_extract_prefixed_tolerance()` method handles this specialized routing and should remain.

### 4. NewtonKrylov (Modified Class)

**Location**: `src/cubie/integrators/matrix_free_solvers/newton_krylov.py`

**Current State**:
- Inherits from `MatrixFreeSolver`
- `update()` method manually handles `newton_atol` and `newton_rtol` extraction
- Also forwards `krylov_*` parameters to nested `linear_solver`

**Expected Behavior After Refactor**:
- Inherits from `MatrixFreeSolver` (unchanged)
- `update()` method leverages base class prefix handling for compile settings
- Tolerance extraction and forwarding to nested solvers remains specialized logic
- The class attribute `settings_prefix = "newton_"` should be passed to `__init__` instead

---

## Architectural Changes

### Inheritance Hierarchy Change

**Before**:
```
CUDAFactory
    └── MatrixFreeSolver
            ├── LinearSolver
            └── NewtonKrylov
```

**After**:
```
CUDAFactory
    └── MultipleInstanceCUDAFactory
            └── MatrixFreeSolver
                    ├── LinearSolver
                    └── NewtonKrylov
```

### Update Flow Change

**Before** (in LinearSolver.update()):
1. Merge updates dict
2. Call `_extract_prefixed_tolerance()` to pop prefixed keys and get norm updates
3. Update norm factory
4. Call `update_compile_settings()` with remaining updates

**After** (in LinearSolver.update()):
1. Merge updates dict
2. Call `_extract_prefixed_tolerance()` for norm-specific routing (unchanged, specialized logic)
3. Update norm factory
4. Call `update_compile_settings()` - base class automatically maps prefixed keys
5. Prefixed compile settings like `krylov_max_iters` → `max_iters` handled automatically

---

## Integration Points

### 1. CUDAFactory.py Integration
- `MultipleInstanceCUDAFactory` is added directly below `CUDAFactory` class
- No changes to `CUDAFactory` itself
- No changes to `CUDAFactoryConfig`, `CUDADispatcherCache`, or `_CubieConfigBase`

### 2. MatrixFreeSolver Integration
- Import `MultipleInstanceCUDAFactory` from `cubie.CUDAFactory`
- Change base class from `CUDAFactory` to `MultipleInstanceCUDAFactory`
- Update constructor to pass `instance_label`

### 3. Solver Classes Integration
- `LinearSolver` and `NewtonKrylov` require minimal changes
- The `settings_prefix` attribute in `NewtonKrylov` class body should be removed (passed via `__init__` instead)

---

## Data Structures

### Updates Dict Transformation

**Input** (external update dict):
```python
{
    "krylov_atol": array([1e-8]),
    "krylov_max_iters": 150,
    "n": 10,
    "precision": np.float32,
}
```

**Transformed** (for compile settings):
```python
{
    "krylov_atol": array([1e-8]),  # kept for reference
    "atol": array([1e-8]),          # added unprefixed
    "krylov_max_iters": 150,        # kept for reference
    "max_iters": 150,               # added unprefixed
    "n": 10,                        # unchanged
    "precision": np.float32,        # unchanged
}
```

**Note**: The compile settings container only needs the unprefixed keys. The prefixed versions are kept in the dict for recognition tracking but ignored by the settings update.

---

## Dependencies and Imports

### CUDAFactory.py
No new imports required. Uses existing:
- `typing.Set, Any, Dict`
- `abc.ABC, abstractmethod`

### base_solver.py
Update import:
```python
from cubie.CUDAFactory import MultipleInstanceCUDAFactory, CUDAFactoryConfig
# Remove CUDAFactory from import if no longer used directly
```

---

## Edge Cases

### 1. Key Present in Both Forms
If updates contain both `krylov_atol` and `atol`, the prefixed version should take precedence after transformation (since it's added after scanning).

### 2. No Matching Prefixed Keys
If no keys match the instance label prefix, the updates dict passes through unchanged.

### 3. Instance Label with Trailing Underscore
The instance label should NOT include the trailing underscore. The transformation logic adds it:
```python
prefix = f"{self.instance_label}_"
```

### 4. Empty Instance Label
Not a valid use case. The base class should work with non-empty labels only. Validation is optional but recommended.

### 5. Nested Multi-Instance Factories
`NewtonKrylov` contains a nested `LinearSolver`. Each has its own instance label ("newton_" vs "krylov_"). The forwarding logic in `NewtonKrylov.update()` should continue to forward `krylov_*` keys to the linear solver.

---

## Expected Interactions

### Solver Initialization Flow
1. `LinearSolver.__init__()` called with precision, n, optional kwargs
2. Extracts tolerance kwargs (`krylov_atol`, `krylov_rtol`) for norm factory
3. Calls `super().__init__(instance_label="krylov", ...)`
4. `MatrixFreeSolver.__init__()` called
5. Calls `super().__init__(instance_label=instance_label)`
6. `MultipleInstanceCUDAFactory.__init__()` called
7. Stores `instance_label`, calls `CUDAFactory.__init__()`
8. Back in `LinearSolver.__init__()`: builds config, sets up compile settings

### Settings Update Flow
1. User calls `solver.update({"krylov_tolerance": 1e-8, "krylov_atol": 1e-7})`
2. `LinearSolver.update()` receives the dict
3. Calls `_extract_prefixed_tolerance()` → extracts `atol` for norm, modifies dict
4. Updates norm factory
5. Calls `update_compile_settings()` with remaining updates
6. `MultipleInstanceCUDAFactory.update_compile_settings()`:
   - Copies dict
   - Finds `krylov_tolerance` → adds `tolerance` key (if such field exists)
   - Calls `super().update_compile_settings()`
7. `CUDAFactory.update_compile_settings()`:
   - Forwards to `compile_settings.update()`
   - Invalidates cache if values changed

---

## Testing Strategy

### Test Cases for MultipleInstanceCUDAFactory

1. **test_multiple_instance_factory_prefix_mapping**
   - Create a `LinearSolver` instance
   - Call `update()` with prefixed keys
   - Verify the unprefixed values are applied to compile settings

2. **test_multiple_instance_factory_mixed_keys**
   - Update with both prefixed and unprefixed keys
   - Verify prefixed takes precedence when both forms are present

3. **test_multiple_instance_factory_no_prefix_match**
   - Update with keys that don't match the instance label
   - Verify they pass through unchanged and are recognized if valid

4. **test_multiple_instance_factory_instance_label_stored**
   - Verify `instance_label` attribute is correctly stored
   - Access via property or attribute

### Fixtures to Use
- Use the existing `precision` fixture
- Create a `LinearSolver` or simple test factory directly in the test
- Avoid mocks; use real objects with minimal configuration

---

## Files to Modify

1. **src/cubie/CUDAFactory.py**
   - Add `MultipleInstanceCUDAFactory` class after `CUDAFactory`

2. **src/cubie/integrators/matrix_free_solvers/base_solver.py**
   - Change `MatrixFreeSolver` to inherit from `MultipleInstanceCUDAFactory`
   - Update `__init__` to pass `instance_label`
   - Rename parameter from `settings_prefix` to `instance_label`

3. **src/cubie/integrators/matrix_free_solvers/linear_solver.py**
   - Update super().__init__ call to use `instance_label` parameter name
   - The `update()` method may be simplified but core logic remains

4. **src/cubie/integrators/matrix_free_solvers/newton_krylov.py**
   - Remove class attribute `settings_prefix = "newton_"`
   - Update super().__init__ call to use `instance_label` parameter name

5. **tests/test_CUDAFactory.py**
   - Add test functions for `MultipleInstanceCUDAFactory`
   - Use real `LinearSolver` instances

---

## Constraints

- Follow PEP8 with 79 character line limit
- Use numpydoc-style docstrings
- Type hints required in function signatures
- Do not use mocks in tests
- Maintain backwards compatibility for existing solver behavior
- Do not fix unrelated issues in the files being modified
