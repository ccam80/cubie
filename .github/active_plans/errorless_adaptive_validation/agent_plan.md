# Agent Implementation Plan: Errorless Adaptive Validation

## Overview

This plan implements validation to prevent errorless algorithms from being paired with adaptive step controllers, which currently causes silent failures (zero step sizes). The implementation adds a validation check and makes controller defaults dynamic based on algorithm capabilities.

## Component 1: Enhanced Compatibility Validation

### File: `src/cubie/integrators/SingleIntegratorRunCore.py`

#### Behavior: Add validation call in `__init__`

**Location:** After controller instantiation (line ~128), before loop instantiation

**Purpose:** Invoke the existing but unused `check_compatibility()` method to validate algorithm-controller pairing

**Expected behavior:**
- After `self._step_controller = get_controller(...)` completes
- Before `self._loop = self.instantiate_loop(...)` executes
- Call `self.check_compatibility()` to validate the pairing
- If incompatible, raises ValueError before any CUDA compilation occurs

**Integration:**
- Integrates with existing initialization flow
- No changes to method signatures
- No changes to return values
- Validation is transparent to callers

#### Behavior: Improve `check_compatibility` error message

**Location:** `check_compatibility` method (lines 169-184)

**Purpose:** Provide actionable error messages that identify the specific algorithm and controller causing the incompatibility

**Expected behavior:**
- Access algorithm name from `self.compile_settings.algorithm` (already stored)
- Access controller name from `self.compile_settings.step_controller` (already stored)
- Include both names in error message
- Explain what makes them incompatible (no error estimate for adaptive control)
- Suggest solutions (use fixed controller or adaptive algorithm)

**Current implementation:**
```python
if (not self._algo_step.is_adaptive and
        self._step_controller.is_adaptive):
    raise ValueError(
        "Adaptive step controller cannot be used with fixed-step "
        "algorithm.",
    )
```

**Expected enhanced implementation:**
- Keep the same conditional logic
- Expand error message to include:
  - Algorithm name: `self.compile_settings.algorithm`
  - Controller type: `self.compile_settings.step_controller`
  - Explanation of why they're incompatible
  - Suggested fix

**Example enhanced message:**
```
Adaptive step controller 'pi' cannot be used with fixed-step algorithm 'erk'. 
The algorithm does not provide an error estimate required for adaptive stepping. 
Use step_controller='fixed' or choose an adaptive algorithm with error estimation.
```

**Data accessed:**
- `self._algo_step.is_adaptive` (bool)
- `self._step_controller.is_adaptive` (bool)
- `self.compile_settings.algorithm` (str)
- `self.compile_settings.step_controller` (str)

## Component 2: Dynamic Controller Defaults for Generic ERK

### File: `src/cubie/integrators/algorithms/generic_erk.py`

#### Current State Analysis

**Current implementation:**
```python
ERK_DEFAULTS = StepControlDefaults(
    step_controller={
        "step_controller": "pi",
        "dt_min": 1e-6,
        "dt_max": 1e-1,
        "kp": 0.6,
        "kd": 0.4,
        "deadband_min": 1.0,
        "deadband_max": 1.1,
        "min_gain": 0.5,
        "max_gain": 2.0,
    }
)
```

**Problem:** 
- Static defaults assume all ERK tableaus have error estimates
- Tableaus like `CLASSICAL_RK4_TABLEAU`, `HEUN_21_TABLEAU`, `RALSTON_33_TABLEAU` have no `b_hat` (no error estimate)
- These should default to `"fixed"` controller, not `"pi"`

#### Behavior: Make ERKStep choose defaults based on tableau

**Purpose:** Select appropriate default controller based on whether the tableau has an error estimate

**Expected behavior in ERKStep.__init__:**
1. Receive `tableau` parameter (defaults to `DEFAULT_ERK_TABLEAU`)
2. Check if `tableau.has_error_estimate` is True
3. If True: use adaptive defaults (current ERK_DEFAULTS)
4. If False: use fixed defaults (similar to EE_DEFAULTS)
5. Pass chosen defaults to `super().__init__(config, defaults)`

**Data structures needed:**

Define two constant default sets:
```python
ERK_ADAPTIVE_DEFAULTS = StepControlDefaults(
    step_controller={
        "step_controller": "pi",
        "dt_min": 1e-6,
        "dt_max": 1e-1,
        # ... existing adaptive parameters
    }
)

ERK_FIXED_DEFAULTS = StepControlDefaults(
    step_controller={
        "step_controller": "fixed",
        "dt": 1e-3,
    }
)
```

**Selection logic in `__init__`:**
```python
def __init__(self, ..., tableau: ERKTableau = DEFAULT_ERK_TABLEAU, ...):
    config = ERKStepConfig(...)
    
    # Select defaults based on tableau capability
    if tableau.has_error_estimate:
        defaults = ERK_ADAPTIVE_DEFAULTS
    else:
        defaults = ERK_FIXED_DEFAULTS
    
    super().__init__(config, defaults)
```

**Integration points:**
- `tableau.has_error_estimate` property already exists (base_algorithm_step.py:96-101)
- `StepControlDefaults` class already exists (base_algorithm_step.py:166-175)
- `super().__init__` already accepts defaults parameter
- No changes to `ERKStepConfig` needed
- No changes to `build_step` or device functions needed

**Backward compatibility:**
- `DEFAULT_ERK_TABLEAU` is `DORMAND_PRINCE_54_TABLEAU` which has error estimate
- Existing code using default tableau will get adaptive defaults (unchanged)
- Existing code explicitly specifying adaptive tableaus will get adaptive defaults (unchanged)
- Only code using errorless tableaus will get different defaults (and would have failed before)

## Component 3: Consider Other Algorithm Types

### Files to Review

#### `src/cubie/integrators/algorithms/generic_dirk.py`
- Check if DIRK tableaus can have error estimates
- If yes, apply same pattern as ERK

#### `src/cubie/integrators/algorithms/generic_firk.py`
- Check if FIRK tableaus can have error estimates  
- If yes, apply same pattern as ERK

#### `src/cubie/integrators/algorithms/generic_rosenbrock_w.py`
- Check if Rosenbrock tableaus can have error estimates
- If yes, apply same pattern as ERK

#### Fixed-step algorithms (no changes needed)
- `explicit_euler.py` - Already uses `EE_DEFAULTS` with `"fixed"`
- `backwards_euler.py` - Check current defaults
- `backwards_euler_predict_correct.py` - Check current defaults
- `crank_nicolson.py` - Check current defaults

**Expected behavior:**
- Review each algorithm's defaults
- Verify fixed-step algorithms use `"fixed"` controller defaults
- Verify or add dynamic defaults for tableau-based methods that support error estimates

## Component 4: Test Coverage

### New Test File: `tests/integrators/test_algorithm_controller_compatibility.py`

**Purpose:** Validate that incompatible algorithm-controller pairings are rejected

#### Test Cases

**Test 1: Errorless algorithm with adaptive controller raises ValueError**
- Setup: Create `SingleIntegratorRunCore` with explicit Euler algorithm and PI controller
- Expected: `ValueError` raised during initialization
- Validation: Error message contains algorithm and controller names

**Test 2: Errorless tableau with adaptive controller raises ValueError**
- Setup: Create `SingleIntegratorRunCore` with ERK using `CLASSICAL_RK4_TABLEAU` and PI controller
- Expected: `ValueError` raised during initialization
- Validation: Error message is actionable

**Test 3: Adaptive algorithm with adaptive controller succeeds**
- Setup: Create `SingleIntegratorRunCore` with ERK using `DORMAND_PRINCE_54_TABLEAU` and PI controller
- Expected: Successful initialization
- Validation: No error raised

**Test 4: Errorless algorithm with fixed controller succeeds**
- Setup: Create `SingleIntegratorRunCore` with explicit Euler and fixed controller
- Expected: Successful initialization
- Validation: No error raised

**Test 5: Default controller for errorless tableau is fixed**
- Setup: Create `ERKStep` with `CLASSICAL_RK4_TABLEAU`, no controller specified
- Expected: `controller_defaults.step_controller["step_controller"]` equals `"fixed"`

**Test 6: Default controller for adaptive tableau is adaptive**
- Setup: Create `ERKStep` with `DORMAND_PRINCE_54_TABLEAU`, no controller specified
- Expected: `controller_defaults.step_controller["step_controller"]` equals `"pi"`

**Test 7: Error message quality**
- Setup: Trigger incompatibility error
- Expected: Error message contains:
  - Algorithm name
  - Controller name
  - Explanation of incompatibility
  - Suggested fix

**Fixtures needed:**
- System fixture (from existing conftest.py)
- Precision fixture
- Various tableau fixtures
- Various controller setting dictionaries

**Test implementation pattern:**
```python
def test_errorless_with_adaptive_raises(system_fixture, precision):
    algorithm_settings = {
        "algorithm": "explicit_euler",
    }
    step_control_settings = {
        "step_controller": "pi",
        "dt_min": 1e-6,
        "dt_max": 1e-1,
    }
    
    with pytest.raises(ValueError) as exc_info:
        SingleIntegratorRunCore(
            system=system_fixture,
            algorithm_settings=algorithm_settings,
            step_control_settings=step_control_settings,
        )
    
    assert "explicit_euler" in str(exc_info.value).lower()
    assert "pi" in str(exc_info.value).lower()
```

## Edge Cases to Consider

### Edge Case 1: User Explicitly Overrides dt in Adaptive Controller
**Scenario:** User specifies `dt` parameter with adaptive controller
**Current behavior:** Controller likely ignores dt or uses it as initial
**Expected behavior:** Validation still applies; adaptive controller with errorless algorithm is invalid
**Handling:** No special handling; validation is independent of dt parameter

### Edge Case 2: Custom Tableau with Zero Error Weights
**Scenario:** User provides tableau with `b_hat` that produces all-zero error coefficients
**Current behavior:** `has_error_estimate` returns False (checks if any weight != 0.0)
**Expected behavior:** Treated as errorless, defaults to fixed controller
**Handling:** Already handled by `ButcherTableau.has_error_estimate` implementation

### Edge Case 3: Solver/solve_ivp with Invalid Configuration
**Scenario:** User calls `solve_ivp` with incompatible settings
**Current behavior:** Settings propagate to `SingleIntegratorRunCore`
**Expected behavior:** Same validation error raised during Solver initialization
**Handling:** No special handling; validation in `SingleIntegratorRunCore` catches all cases

### Edge Case 4: Algorithm Update After Initialization
**Scenario:** User updates algorithm settings after initialization (if supported)
**Current behavior:** CUDAFactory cache invalidation triggers rebuild
**Expected behavior:** Revalidation should occur
**Handling:** Out of scope; likely not supported by current architecture

## Dependencies and Constraints

### Existing Infrastructure Relied Upon

1. **`is_adaptive` Property**
   - `BaseAlgorithmStep.is_adaptive` (abstract)
   - `BaseStepController.is_adaptive` (concrete)
   - Both must return correct boolean values

2. **`has_error_estimate` Property**
   - `ButcherTableau.has_error_estimate`
   - Must correctly identify tableaus with/without error estimates

3. **`StepControlDefaults` Class**
   - Already supports dictionary-based defaults
   - `.copy()` method for creating independent instances

4. **`IntegratorRunSettings`**
   - Already stores algorithm and step_controller names
   - Accessible via `self.compile_settings`

### No Changes Required To

- CUDA kernel generation
- Loop compilation
- Device function signatures
- Memory allocation
- Output handling
- Public API of Solver or solve_ivp
- Any ODE system classes
- Any BaseODE subclasses

### Assumptions

1. All algorithm classes correctly implement `is_adaptive` property
2. All controller classes correctly implement `is_adaptive` property
3. Tableau `has_error_estimate` property is accurate
4. `IntegratorRunSettings` contains algorithm and controller names
5. Validation occurs before CUDA compilation starts

## Implementation Order

1. **First:** Enhance `check_compatibility` error message (low risk, immediate benefit)
2. **Second:** Add `check_compatibility()` call in `__init__` (enables validation)
3. **Third:** Implement dynamic defaults for ERK (prevents default incompatibilities)
4. **Fourth:** Review and update other tableau-based algorithms (DIRK, FIRK, Rosenbrock)
5. **Fifth:** Add comprehensive test coverage
6. **Sixth:** Update documentation if needed

## Success Criteria

### Functional
- ✅ Incompatible algorithm-controller pairs raise `ValueError` during initialization
- ✅ Error messages identify both algorithm and controller by name
- ✅ Error messages explain the incompatibility and suggest fixes
- ✅ Errorless tableaus default to fixed-step controllers
- ✅ Adaptive tableaus default to adaptive controllers
- ✅ Explicit controller specifications are validated
- ✅ All existing valid configurations continue to work

### Testing
- ✅ Test coverage includes all incompatibility scenarios
- ✅ Tests verify error message quality
- ✅ Tests verify default controller selection
- ✅ Tests use fixtures and follow repository patterns
- ✅ No mocks or patches used in tests

### Code Quality
- ✅ Follows PEP8 (79 char line length, 71 char comments)
- ✅ Type hints in function signatures
- ✅ Numpydoc-style docstrings
- ✅ No inline type annotations
- ✅ Descriptive variable and function names
- ✅ Comments explain complex logic to future developers
