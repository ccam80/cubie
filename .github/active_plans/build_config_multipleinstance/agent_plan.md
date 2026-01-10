# Agent Plan: MultipleInstance build_config Integration

## Overview

This plan addresses integrating `MultipleInstanceCUDAFactoryConfig` with `build_config` and ensuring prefixed parameters propagate through nested solver hierarchies at both init time and update time.

## Component Descriptions

### 1. Enhanced build_config Function

**Location**: `src/cubie/_utils.py`

**Current Behavior**:
- Merges required and optional parameters
- Filters to valid fields using field names and aliases
- Creates config instance

**Target Behavior**:
- Accepts optional `instance_label` parameter
- When `instance_label` is provided:
  - Transforms prefixed keys (e.g., `krylov_atol` → `atol`) before field matching
  - Passes `instance_label` to the config constructor
- Maintains backward compatibility (no instance_label = current behavior)

**Expected Behavior**:
```python
# Without instance_label (current behavior)
config = build_config(SomeConfig, required={'precision': np.float32}, atol=1e-6)

# With instance_label (new behavior)
config = build_config(
    ScaledNormConfig, 
    required={'precision': np.float32, 'n': 3},
    instance_label="krylov",
    krylov_atol=1e-6,  # Transformed to atol before matching
    krylov_rtol=1e-4   # Transformed to rtol before matching
)
```

### 2. MultipleInstanceCUDAFactoryConfig Changes

**Location**: `src/cubie/CUDAFactory.py`

**Current State**:
- Has `init_from_prefixed()` classmethod for prefix-aware initialization
- Has `update()` method that handles prefix transformation
- Has `get_prefixed_attributes()` classmethod

**Target State**:
- `init_from_prefixed()` becomes deprecated (build_config handles this)
- `update()` remains unchanged (already works correctly)
- `get_prefixed_attributes()` becomes a key integration point for build_config
- Add `prefix` property that returns `f"{instance_label}_"` for consistency

### 3. ScaledNorm Factory

**Location**: `src/cubie/integrators/norms.py`

**Current State**:
- Constructor uses `build_config` but has comment about needing `init_from_prefixed`
- Uses `instance_type` parameter name (should be `instance_label` for consistency)

**Target State**:
- Constructor uses enhanced `build_config` with `instance_label`
- Renames `instance_type` to `instance_label` for consistency
- Passes all kwargs through to build_config, letting it handle prefix transformation

### 4. MatrixFreeSolver Base Class

**Location**: `src/cubie/integrators/matrix_free_solvers/base_solver.py`

**Current State**:
- Creates ScaledNorm in constructor with `instance_type=solver_type`
- `update()` method has `_extract_prefixed_tolerance()` helper

**Target State**:
- Passes all kwargs to ScaledNorm constructor
- ScaledNorm's build_config handles prefix extraction
- `update()` passes all updates to norm, which handles its own prefix transformation
- Remove `_extract_prefixed_tolerance()` if no longer needed

### 5. LinearSolver Factory

**Location**: `src/cubie/integrators/matrix_free_solvers/linear_solver.py`

**Current State**:
- Constructor has both `init_from_prefixed()` and `build_config` calls (duplication)
- Only `init_from_prefixed` result is used

**Target State**:
- Single `build_config` call with `instance_label="krylov"`
- Passes all kwargs to parent class (MatrixFreeSolver)
- Parent handles norm creation with forwarded kwargs

### 6. NewtonKrylov Factory

**Location**: `src/cubie/integrators/matrix_free_solvers/newton_krylov.py`

**Current State**:
- Constructor has both `init_from_prefixed()` and `build_config` calls (duplication)
- Only `init_from_prefixed` result is used
- Passes limited kwargs to LinearSolver

**Target State**:
- Single `build_config` call with `instance_label="newton"`
- Passes all kwargs to LinearSolver constructor (for krylov_ prefix params)
- Passes all kwargs to parent class (for newton_ prefix norm params)

## Architectural Changes Required

### A. build_config Enhancement

The function needs to:
1. Accept `instance_label` as optional parameter
2. Get prefixed attributes set from config class (if MultipleInstanceCUDAFactoryConfig subclass)
3. Transform keys: for each key matching `{instance_label}_{attr}`, add `attr` with same value
4. Continue with existing field filtering logic
5. Include `instance_label` in final kwargs if provided

### B. Prefix Resolution Logic

The prefix transformation logic (already in MultipleInstanceCUDAFactoryConfig.update()):
1. For each key in prefixed_attributes:
   - Check if unprefixed key exists → skip (let prefixed take precedence)
   - Check if prefixed key exists → copy value to unprefixed key
2. This same logic should be extracted and shared with build_config

### C. Nested Object Parameter Forwarding

Factories must forward all kwargs to nested objects:
- NewtonKrylov → LinearSolver: all kwargs
- LinearSolver → MatrixFreeSolver → ScaledNorm: all kwargs
- Each level extracts its own via build_config/update, ignores rest

## Integration Points

### 1. build_config ↔ MultipleInstanceCUDAFactoryConfig
- build_config checks if config_class has `get_prefixed_attributes()` method
- If yes, applies prefix transformation before field filtering

### 2. MatrixFreeSolver ↔ ScaledNorm
- MatrixFreeSolver constructor passes all kwargs to ScaledNorm
- ScaledNorm's build_config extracts tolerance params using prefix

### 3. LinearSolver ↔ MatrixFreeSolver
- LinearSolver passes all kwargs to super().__init__()
- MatrixFreeSolver forwards to ScaledNorm

### 4. NewtonKrylov ↔ LinearSolver
- NewtonKrylov passes all kwargs to LinearSolver constructor
- Also passes all kwargs to super().__init__() for newton_ prefix handling

## Data Structures

### Prefix Transformation Mapping
```python
# For instance_label="krylov", prefixed_attributes={"atol", "rtol"}
# Input: {"krylov_atol": 1e-8, "krylov_rtol": 1e-6, "precision": np.float32}
# Output: {"atol": 1e-8, "rtol": 1e-6, "precision": np.float32}
```

### Config Class Metadata
ScaledNormConfig and similar classes use `metadata={"prefixed": True}` on fields to indicate which are prefixed.

## Dependencies and Imports

### build_config needs:
- `from attrs import has, fields` (already imported in _utils.py)
- Access to MultipleInstanceCUDAFactoryConfig for isinstance check or duck typing

### Duck Typing Approach:
Rather than importing CUDAFactory classes, check for method existence:
```python
if hasattr(config_class, 'get_prefixed_attributes'):
    prefixed_attrs = config_class.get_prefixed_attributes()
    # Apply transformation
```

## Edge Cases

1. **Both prefixed and unprefixed provided**: Prefixed takes precedence
2. **instance_label is empty string**: Treated as no prefix (current behavior)
3. **Nested configs with same field names**: Each config extracts its own, no collision
4. **Config with no prefixed attributes**: Standard build_config behavior
5. **update() called with mixed prefixes**: Each nested object extracts its own

## Test Considerations

### Tests to Add/Update:

1. **test_build_config_with_instance_label**: Verify prefix transformation
2. **test_build_config_backward_compatible**: Existing behavior unchanged
3. **test_nested_prefix_propagation_init**: krylov_atol reaches ScaledNorm via init
4. **test_nested_prefix_propagation_update**: krylov_atol reaches ScaledNorm via update
5. **test_newton_krylov_forwards_krylov_params**: Verify full chain propagation
6. **test_no_manual_key_filtering**: Verify classes don't filter keys

### Existing Tests to Update:

- `tests/test_CUDAFactory.py`: Add tests for enhanced build_config
- `tests/integrators/matrix_free_solvers/test_linear_solver.py`: Verify prefix handling
- `tests/integrators/matrix_free_solvers/test_newton_krylov.py`: Verify nested propagation
- `tests/integrators/matrix_free_solvers/test_base_solver.py`: Update for changed signature
