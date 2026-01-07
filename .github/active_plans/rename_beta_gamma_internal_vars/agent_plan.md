# Agent Plan: Rename Beta/Gamma Internal Variables

## Overview
This plan details the renaming of internal code generation variables `beta` and `gamma` to `_cubie_codegen_beta` and `_cubie_codegen_gamma` to prevent naming conflicts with user-defined system variables.

## Component Analysis

### 1. Code Generation Templates (Primary Changes)

Three code generation modules create factory functions that use `beta` and `gamma`:

#### linear_operators.py
**Location:** `src/cubie/odesystems/symbolic/codegen/linear_operators.py`

**Functions to Update:**
- `generate_operator_apply_code()` - Generates linear operator without cached auxiliaries
- `generate_cached_operator_apply_code()` - Generates linear operator with cached auxiliaries  
- `generate_n_stage_linear_operator_code()` - Generates FIRK multi-stage operator

**Template Variables:**
- `OPERATOR_APPLY_TEMPLATE` - Contains factory function definition
- `CACHED_OPERATOR_APPLY_TEMPLATE` - Contains cached variant
- `N_STAGE_OPERATOR_TEMPLATE` - Contains multi-stage variant

**Renaming Required:**
1. Template string: Change parameter name in factory signature docstrings (documentation only)
2. Template string: Change local variable `beta = precision(beta)` → `_cubie_codegen_beta = precision(beta)`
3. Template string: Change local variable `gamma = precision(gamma)` → `_cubie_codegen_gamma = precision(gamma)`
4. SymPy symbols: `beta_sym = sp.Symbol("beta")` → `beta_sym = sp.Symbol("_cubie_codegen_beta")`
5. SymPy symbols: `gamma_sym = sp.Symbol("gamma")` → `gamma_sym = sp.Symbol("_cubie_codegen_gamma")`
6. Substitution dicts: Update keys from `"beta"` to `"_cubie_codegen_beta"`, `"gamma"` to `"_cubie_codegen_gamma"`

**Expected Behavior:**
Generated factory functions accept `beta` and `gamma` as parameters (unchanged interface), but internally assign them to `_cubie_codegen_beta` and `_cubie_codegen_gamma` before creating SymPy expressions. This ensures user symbols named `beta` or `gamma` don't collide during code generation.

#### preconditioners.py
**Location:** `src/cubie/odesystems/symbolic/codegen/preconditioners.py`

**Functions to Update:**
- `generate_neumann_preconditioner_code()` - Generates Neumann preconditioner
- `generate_neumann_preconditioner_cached_code()` - Generates cached variant
- `generate_n_stage_neumann_preconditioner_code()` - Generates multi-stage variant

**Template Variables:**
- `NEUMANN_TEMPLATE`
- `NEUMANN_CACHED_TEMPLATE`
- `N_STAGE_NEUMANN_TEMPLATE`

**Renaming Required:**
Same pattern as linear_operators.py:
1. Docstring parameter documentation (optional, for clarity)
2. Local variables `beta`, `gamma` → `_cubie_codegen_beta`, `_cubie_codegen_gamma`
3. Derived variables: `beta_inv = precision(1.0 / beta)` → `beta_inv = precision(1.0 / _cubie_codegen_beta)`
4. Derived variables: `h_eff_factor = precision(gamma * beta_inv)` → `h_eff_factor = precision(_cubie_codegen_gamma * beta_inv)`
5. SymPy symbols in code generation functions
6. Substitution dictionaries

**Special Note:** Preconditioners compute derived quantities (`beta_inv`, `h_eff_factor`) from `beta` and `gamma`. These computations must reference the renamed variables.

#### nonlinear_residuals.py
**Location:** `src/cubie/odesystems/symbolic/codegen/nonlinear_residuals.py`

**Functions to Update:**
- `generate_stage_residual_code()` - Generates single-stage residual
- `generate_n_stage_residual_code()` - Generates multi-stage residual for FIRK

**Template Variables:**
- `RESIDUAL_TEMPLATE`
- `N_STAGE_RESIDUAL_TEMPLATE`

**Renaming Required:**
Same pattern as above:
1. Docstrings (beta*M*u - gamma*h*f notation preserved, but variable names updated)
2. Local variables
3. SymPy symbols
4. Residual expression: `residual_expr = beta_sym * mv - gamma_sym * h_sym * dx_sym`

**Expected Behavior:**
Residual functions compute `beta * M * u - gamma * h * f(...)` where `beta` and `gamma` are solver coefficients. After renaming, generated code uses `_cubie_codegen_beta` and `_cubie_codegen_gamma` internally.

### 2. SymPy Symbol Handling

**Pattern in Current Code:**
```python
beta_sym = sp.Symbol("beta")
gamma_sym = sp.Symbol("gamma")
# ... later in substitution
substitution_map = {
    "beta": beta_sym,
    "gamma": gamma_sym,
    # ... other mappings
}
```

**Updated Pattern:**
```python
beta_sym = sp.Symbol("_cubie_codegen_beta")
gamma_sym = sp.Symbol("_cubie_codegen_gamma")
# ... later in substitution
substitution_map = {
    "_cubie_codegen_beta": beta_sym,
    "_cubie_codegen_gamma": gamma_sym,
    # ... other mappings
}
```

**Rationale:** SymPy will print these symbols as `_cubie_codegen_beta` in generated code, ensuring no collision with user-defined `beta` or `gamma` symbols.

### 3. Template String Updates

**Current Pattern:**
```python
TEMPLATE = (
    "def factory(constants, precision, beta=1.0, gamma=1.0, order=None):\n"
    "    beta = precision(beta)\n"
    "    gamma = precision(gamma)\n"
    # ...
)
```

**Updated Pattern:**
```python
TEMPLATE = (
    "def factory(constants, precision, beta=1.0, gamma=1.0, order=None):\n"
    "    _cubie_codegen_beta = precision(beta)\n"
    "    _cubie_codegen_gamma = precision(gamma)\n"
    # ...
)
```

**Rationale:** Factory function interface unchanged (`beta`, `gamma` parameters), but internal variables renamed to avoid namespace pollution in generated CUDA device functions.

### 4. Components NOT Requiring Changes

#### symbolicODE.py
**Why no changes:** 
- `get_solver_helper()` accepts `beta` and `gamma` as method parameters
- These are passed to factory functions (which handle the renaming)
- No code generation happens directly in this file
- Acts as interface layer between algorithms and codegen

#### baseODE.py
**Why no changes:**
- Abstract base class defines `get_solver_helper()` signature
- No code generation occurs at this level
- Interface remains stable

#### Algorithm Implementations
**Files:** `backwards_euler.py`, `crank_nicolson.py`, `generic_dirk.py`, etc.

**Why no changes:**
- Algorithms define coefficient values (e.g., `ALGO_CONSTANTS = {'beta': 1.0, 'gamma': 1.0}`)
- Pass these values to solver helper factories
- No knowledge of code generation internals
- Correctly use mathematical terminology for solver coefficients

#### ImplicitStepConfig
**File:** `ode_implicitstep.py`

**Why no changes:**
- Stores solver configuration including `beta` and `gamma` attributes
- Properties expose values as `self.precision(self._beta)`
- No code generation involvement
- Part of algorithm configuration layer, not code generation

#### Rosenbrock Tableaus
**File:** `generic_rosenbrockw_tableaus.py`

**Why no changes:**
- `RosenbrockTableau.gamma` is a different concept (diagonal shift coefficient)
- Not related to code generation variables
- Used during algorithm initialization, not during symbolic codegen
- Maintains consistency with Rosenbrock method literature

#### Test Instrumented Versions
**Location:** `tests/integrators/algorithms/instrumented/`

**Why no changes:**
- These are copies of algorithm step functions with added logging
- They don't participate in symbolic code generation
- Changes propagate from algorithm implementations only

### 5. Integration Points

#### Factory Invocation Chain
```
Algorithm (beta=1.0, gamma=1.0)
  ↓
SymbolicODE.get_solver_helper(beta=1.0, gamma=1.0)
  ↓
generate_*_code() → returns factory function source code
  ↓
exec() to create factory callable
  ↓
factory(constants, precision, beta=1.0, gamma=1.0, order=...)
  ↓
Internally: _cubie_codegen_beta = precision(beta)
            _cubie_codegen_gamma = precision(gamma)
  ↓
@cuda.jit device function with _cubie_codegen_* in closure
```

**Key Insight:** Renaming happens at the template level. The interface (factory function signature) remains unchanged, but the internal variable names in the generated device function use the prefixed names.

#### SymPy Expression Evaluation
When building residuals, operators, or preconditioners:
1. Create SymPy symbols with new names: `sp.Symbol("_cubie_codegen_beta")`
2. Build expressions using these symbols
3. SymPy's printer outputs the symbol names as-is
4. No collision with user symbols named `beta` or `gamma`

### 6. Edge Cases to Consider

#### User Defines `_cubie_codegen_beta`
**Likelihood:** Extremely low (very unconventional naming)  
**Behavior:** Would cause collision  
**Mitigation:** Not worth additional complexity; document as reserved prefix

#### Mixed Usage in Single Expression
**Scenario:** User has `beta` parameter, code uses `_cubie_codegen_beta`  
**Behavior:** No conflict - different symbols in SymPy namespace  
**Validation:** Existing tests with beta/gamma in system should work

#### Derived Quantities (beta_inv, h_eff_factor)
**Important:** These must reference renamed variables:
```python
# Before
beta_inv = precision(1.0 / beta)
h_eff_factor = precision(gamma * beta_inv)

# After
beta_inv = precision(1.0 / _cubie_codegen_beta)
h_eff_factor = precision(_cubie_codegen_gamma * beta_inv)
```

### 7. Testing Implications

#### Existing Tests
**Expected:** All tests should pass without modification  
**Reason:** Tests interact via public API (unchanged)  
**Exception:** Tests that inspect generated code strings may need updates

#### String Inspection Tests
**Files to check:**
- `tests/odesystems/symbolic/test_solver_helpers.py`
- `tests/odesystems/symbolic/test_*.py` files that validate generated code

**Potential Issue:** If tests assert on generated code containing `beta`/`gamma`  
**Fix:** Update assertions to expect `_cubie_codegen_beta`/`_cubie_codegen_gamma`

#### Functional Tests
**No changes expected:** Device functions behave identically, just with renamed internal variables

## Implementation Checklist

1. **linear_operators.py**
   - [ ] Update `OPERATOR_APPLY_TEMPLATE`
   - [ ] Update `CACHED_OPERATOR_APPLY_TEMPLATE`
   - [ ] Update `N_STAGE_OPERATOR_TEMPLATE` (if exists)
   - [ ] Update SymPy symbol creation in helper functions
   - [ ] Update substitution dictionaries

2. **preconditioners.py**
   - [ ] Update `NEUMANN_TEMPLATE`
   - [ ] Update `NEUMANN_CACHED_TEMPLATE`
   - [ ] Update `N_STAGE_NEUMANN_TEMPLATE`
   - [ ] Update SymPy symbol creation
   - [ ] Update derived quantity calculations (beta_inv, h_eff_factor)
   - [ ] Update substitution dictionaries

3. **nonlinear_residuals.py**
   - [ ] Update `RESIDUAL_TEMPLATE`
   - [ ] Update `N_STAGE_RESIDUAL_TEMPLATE`
   - [ ] Update SymPy symbol creation
   - [ ] Update residual expression construction
   - [ ] Update substitution dictionaries

4. **Testing**
   - [ ] Run full test suite
   - [ ] Check for string comparison test failures
   - [ ] Update any tests that inspect generated code

## Risk Assessment

**Low Risk:**
- Changes confined to code generation templates
- No algorithm logic changes
- No external API changes
- Existing functionality preserved

**Medium Risk:**
- Template strings are complex; typos could break code generation
- SymPy substitution must be updated consistently

**Mitigation:**
- Run tests after each file modification
- Verify generated code compiles without errors
- Check that existing systems with `beta`/`gamma` parameters work correctly

## Expected Outcomes

**After Implementation:**
1. Users can define ODE systems with `beta`, `gamma`, or `sigma` as state variables, parameters, or constants
2. Code generation produces conflict-free CUDA kernels
3. All existing tests pass
4. No breaking changes to user-facing API
5. Generated code is more clearly namespaced

**Generated Code Example:**
```python
# Factory function (interface unchanged)
def linear_operator(constants, precision, beta=1.0, gamma=1.0, order=None):
    # Internal renaming
    _cubie_codegen_beta = precision(beta)
    _cubie_codegen_gamma = precision(gamma)
    
    @cuda.jit(device=True, inline=True)
    def operator_apply(state, parameters, drivers, base_state, t, h, a_ij, v, out):
        # Uses _cubie_codegen_beta and _cubie_codegen_gamma
        # User's 'beta' parameter appears in 'parameters' array instead
        ...
```
