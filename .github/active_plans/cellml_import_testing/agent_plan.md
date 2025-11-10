# CellML Import Testing - Agent Implementation Plan

## Overview

This plan details the implementation tasks for testing and verifying CuBIE's CellML model import functionality. The work focuses on ensuring the existing `load_cellml_model` function works correctly with real CellML models and integrates properly with CuBIE's SymbolicODE system.

## Component Descriptions

### Test Fixtures

#### CellML Model Files
Location: `tests/fixtures/cellml/` (new directory)

Files to include:
1. **beeler_reuter_1977.cellml** - Primary test model
   - Downloaded from cellmlmanip test suite
   - Cardiac action potential model
   - 8 state variables
   - Representative complexity

2. **basic_ode.cellml** - Simple test model
   - Minimal ODE model for basic tests
   - 1-2 state variables
   - Fast test execution

3. **hodgkin_huxley_modified.cellml** - Alternative complex model
   - Neural action potential model
   - Different equation structure
   - Validates generality

#### Pytest Fixtures
Location: `tests/odesystems/symbolic/test_cellml.py`

Required fixtures:
- `cellml_model_paths` - Dictionary mapping model names to file paths
- `simple_cellml_model` - Returns path to basic_ode.cellml
- `complex_cellml_model` - Returns path to beeler_reuter_1977.cellml
- `skip_if_no_cellmlmanip` - Fixture using pytest.importorskip

### Test Suite Components

#### Test File Structure
File: `tests/odesystems/symbolic/test_cellml.py`

Test classes/functions:
1. **test_cellml_import_error** (existing) - Verify ImportError when cellmlmanip missing
2. **test_load_simple_model** - Load basic ODE model, verify structure
3. **test_load_complex_model** - Load Beeler-Reuter model, verify extraction
4. **test_states_are_symbols** - Verify returned states are sympy.Symbol
5. **test_equations_are_sympy_eq** - Verify returned equations are sympy.Eq
6. **test_derivatives_in_equations** - Verify equations contain derivatives
7. **test_integration_with_symbolic_ode** - Create SymbolicODE from loaded model
8. **test_solve_ivp_with_cellml** - End-to-end test with solve_ivp (marked slow)

#### Test Markers
- `@pytest.mark.cupy` - Already exists for CuPy-dependent tests
- Use for tests requiring cellmlmanip: `pytest.importorskip("cellmlmanip")`

### Source Code Components

#### load_cellml_model Function
File: `src/cubie/odesystems/symbolic/parsing/cellml.py`

Current behavior:
- Loads model via `cellmlmanip.load_model(path)`
- Extracts states from `model.get_state_variables()`
- Extracts derivatives from `model.get_derivatives()`
- Filters equations where `eq.lhs in derivatives`

Expected behavior verification needed:
1. States should be list of sympy symbols (not Dummy)
2. Equations should have derivatives as LHS
3. All state derivatives should be present
4. Return types match function signature

Potential corrections:
- May need to convert sympy.Dummy to sympy.Symbol
- May need to ensure free variable (time) is handled correctly
- May need to validate equation filtering logic

### Integration Points

#### With SymbolicODE
The loaded (states, equations) must be compatible with:
```python
ode_system = create_ODE_system(
    states=states,
    equations=equations,
    # ... other parameters
)
```

Requirements:
- States must be sympy.Symbol instances
- Equations must be sympy.Eq with derivatives
- Free variable should be extractable from derivatives

#### With solve_ivp
End-to-end validation:
```python
states, equations = load_cellml_model(path)
ode = create_ODE_system(states=states, equations=equations, ...)
result = solve_ivp(ode, initial_conditions, parameters, ...)
```

## Expected Behavior

### load_cellml_model Function

**Input**: 
- `path` (str): Filesystem path to CellML file

**Output**:
- `states` (list[sp.Symbol]): State variable symbols
- `equations` (list[sp.Eq]): Differential equations

**Behavior**:
1. Check if cellmlmanip is available (raise ImportError if not)
2. Load the CellML model from file
3. Extract state variables and convert to list
4. Extract derivatives and convert to list
5. Filter equations where LHS is a derivative
6. Return (states, equations) tuple

**Error handling**:
- ImportError if cellmlmanip not installed (already implemented)
- Should propagate cellmlmanip parsing errors
- Invalid file path should raise appropriate error

### Test Behavior

#### Basic Loading Test
```python
def test_load_simple_model(simple_cellml_model):
    pytest.importorskip("cellmlmanip")
    states, equations = load_cellml_model(simple_cellml_model)
    assert isinstance(states, list)
    assert isinstance(equations, list)
    assert len(states) > 0
    assert len(equations) > 0
```

#### Type Verification Test
```python
def test_states_are_symbols(complex_cellml_model):
    pytest.importorskip("cellmlmanip")
    states, equations = load_cellml_model(complex_cellml_model)
    for state in states:
        assert isinstance(state, sp.Symbol)
```

#### Equation Structure Test
```python
def test_equations_are_derivatives(complex_cellml_model):
    pytest.importorskip("cellmlmanip")
    states, equations = load_cellml_model(complex_cellml_model)
    for eq in equations:
        assert isinstance(eq, sp.Eq)
        assert isinstance(eq.lhs, sp.Derivative)
```

#### Integration Test
```python
@pytest.mark.slow
def test_integration_with_symbolic_ode(complex_cellml_model):
    pytest.importorskip("cellmlmanip")
    states, equations = load_cellml_model(complex_cellml_model)
    
    # This should not raise
    ode = create_ODE_system(
        states=states,
        equations=equations,
        precision=np.float64
    )
    
    assert ode is not None
    assert len(ode.states) == len(states)
```

## Dependencies and Imports

### Test File Imports
```python
import pytest
import sympy as sp
import numpy as np
from pathlib import Path

from cubie.odesystems.symbolic.parsing.cellml import load_cellml_model
from cubie import create_ODE_system, solve_ivp
```

### Optional Dependency Handling
```python
# In tests, use:
pytest.importorskip("cellmlmanip")

# Or for conditional behavior:
cellmlmanip = pytest.importorskip("cellmlmanip", reason="cellmlmanip required")
```

## Edge Cases to Consider

1. **cellmlmanip returns sympy.Dummy not Symbol**
   - Research shows cellmlmanip uses Dummy for variables
   - May need conversion to Symbol
   - Test should verify Symbol type

2. **Missing state derivatives**
   - Some models may have algebraic equations
   - Filter should only include ODEs
   - Verify all states have corresponding derivatives

3. **Free variable handling**
   - Time variable needs special handling
   - Derivative is with respect to free variable
   - May need to extract and validate free variable

4. **Model with no ODEs (algebraic only)**
   - Should handle gracefully
   - May return empty equations list
   - Or raise informative error

5. **Invalid CellML file**
   - cellmlmanip parsing errors should propagate
   - Test should verify error message is helpful

6. **Large models**
   - Performance with 50+ states
   - Memory usage
   - Mark as slow test

## Data Structures

### States List
```python
states = [
    sp.Symbol('V'),      # Membrane potential
    sp.Symbol('m'),      # Sodium activation
    sp.Symbol('h'),      # Sodium inactivation
    # ... more states
]
```

### Equations List
```python
equations = [
    sp.Eq(sp.Derivative(V, t), rhs_expression_1),
    sp.Eq(sp.Derivative(m, t), rhs_expression_2),
    sp.Eq(sp.Derivative(h, t), rhs_expression_3),
    # ... more equations
]
```

## Implementation Sequence

The detailed_implementer agent will break this down into specific tasks, but the general sequence is:

1. Create test fixture directory structure
2. Download/copy CellML model files
3. Create basic test fixtures (paths, skip conditions)
4. Implement basic loading test
5. Run test and identify any issues with load_cellml_model
6. Fix any issues in load_cellml_model
7. Implement type verification tests
8. Implement equation structure tests
9. Implement integration test with SymbolicODE
10. Implement end-to-end test with solve_ivp (optional, marked slow)
11. Add documentation/examples

## Validation Criteria

### For load_cellml_model function:
- Returns correct types (list[sp.Symbol], list[sp.Eq])
- All states have corresponding derivative equations
- Equations are properly formatted for SymbolicODE
- Handles missing dependency correctly

### For tests:
- All tests pass with cellmlmanip installed
- Tests are properly skipped without cellmlmanip
- Tests cover success cases and error cases
- Tests use real CellML model files
- Integration tests verify end-to-end workflow

### For overall implementation:
- No breaking changes to existing functionality
- New tests follow repository pytest patterns
- Code follows PEP8 and repository style
- Documentation is clear and helpful
