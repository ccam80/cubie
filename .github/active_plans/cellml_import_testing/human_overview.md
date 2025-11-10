# CellML Import Testing - Human Overview

## User Stories

### User Story 1: Load CellML Models
**As a** CuBIE user working with physiological models  
**I want to** import CellML model files directly into CuBIE  
**So that** I can simulate existing physiological models without manually translating them

**Acceptance Criteria:**
- The `load_cellml_model` function successfully loads a CellML file from disk
- The function returns a tuple of (states, equations) compatible with CuBIE's SymbolicODE
- States are returned as a list of sympy.Symbol objects
- Equations are returned as a list of sympy.Eq objects representing ODEs
- The function handles ImportError gracefully when cellmlmanip is not installed

### User Story 2: Verify CellML Integration
**As a** CuBIE developer  
**I want to** have comprehensive tests for CellML import functionality  
**So that** I can ensure the import works correctly and catch regressions

**Acceptance Criteria:**
- Tests verify that cellmlmanip correctly extracts state variables
- Tests verify that differential equations are extracted in the correct format
- Tests verify compatibility with CuBIE's SymbolicODE system
- Tests handle the optional dependency gracefully (with and without cellmlmanip)
- Tests use real CellML model files as fixtures

### User Story 3: Support Large Physiological Models
**As a** computational physiologist  
**I want to** import complex cardiac/neural models from the Physiome repository  
**So that** I can perform batch simulations of validated physiological models

**Acceptance Criteria:**
- The function successfully loads large models (e.g., Beeler-Reuter, Hodgkin-Huxley)
- Performance is acceptable for models with dozens of state variables
- All state variables and equations are correctly extracted
- The imported models can be used with CuBIE's solve_ivp function

## Executive Summary

This plan adds comprehensive testing and verification for CuBIE's CellML model import capability. The existing stub implementation in `src/cubie/odesystems/symbolic/parsing/cellml.py` provides basic functionality using the `cellmlmanip` library, but lacks testing with real models.

The implementation will:
1. Obtain test CellML model files (Beeler-Reuter cardiac model recommended)
2. Verify cellmlmanip integration and make any necessary corrections
3. Add comprehensive pytest fixtures and tests
4. Ensure compatibility with CuBIE's SymbolicODE workflow

## Key Technical Decisions

### CellML Model Selection
- **Primary test model**: Beeler-Reuter 1977 cardiac model (~45KB)
  - Well-established physiological model
  - Available in cellmlmanip test suite
  - Moderate complexity (8 state variables)
  - Representative of real-world use cases

- **Secondary models for edge cases**:
  - Simple ODE model (basic_ode.cellml) for quick tests
  - Hodgkin-Huxley modified for neural models
  
### Integration with cellmlmanip

Current implementation uses:
```python
model = cellmlmanip.load_model(path)
states = list(model.get_state_variables())
derivatives = list(model.get_derivatives())
equations = [eq for eq in model.equations if eq.lhs in derivatives]
```

**Research findings** from cellmlmanip repository:
- `load_model(path)` returns a parsed Model object
- `model.get_state_variables()` returns state variable symbols
- `model.get_derivatives()` returns derivative symbols  
- `model.equations` contains all equations as sympy.Eq objects
- The model object uses sympy.Dummy for variables (hash-based equality)

**Potential issues identified**:
1. The current implementation may not correctly filter ODE equations
2. cellmlmanip returns sympy.Dummy objects, need to verify compatibility with CuBIE
3. Need to handle the free variable (time) correctly

### Testing Strategy

**Fixture structure**:
```
@pytest.fixture
def cellml_model_file():
    # Returns path to test CellML file
    
@pytest.fixture
def loaded_cellml_model(cellml_model_file):
    # Calls load_cellml_model if cellmlmanip available
    # Uses pytest.importorskip for optional dependency
```

**Test categories**:
1. **Import tests**: Verify cellmlmanip optional dependency handling
2. **Extraction tests**: Verify states and equations are extracted correctly
3. **Format tests**: Verify sympy objects are compatible with CuBIE
4. **Integration tests**: Create SymbolicODE from loaded model and verify

### Data Flow

```
CellML File (.cellml)
    ↓
cellmlmanip.load_model()
    ↓
Model object (equations, variables, units)
    ↓
load_cellml_model() processing
    ↓
(states: list[sp.Symbol], equations: list[sp.Eq])
    ↓
create_ODE_system() / SymbolicODE
    ↓
CuBIE solve_ivp()
```

## Impact on Existing Architecture

**Minimal impact** - this is testing and verification work:
- No changes to CuBIE core architecture
- Possible minor fixes to `load_cellml_model` if issues discovered
- New test file: `tests/odesystems/symbolic/test_cellml.py`
- New test fixtures in repository (CellML files)

**Dependencies**:
- cellmlmanip remains optional (install with pip)
- Test CellML files will be added to repository (small file size acceptable)
- Tests will use `pytest.importorskip` for optional dependency

## Trade-offs and Alternatives

### CellML File Storage
**Decision**: Store test CellML files in `tests/fixtures/cellml/`
- **Pro**: Files available for testing without external downloads
- **Pro**: Reproducible tests
- **Con**: Adds ~100KB to repository size
- **Alternative**: Download files during test - rejected due to test isolation/reliability

### cellmlmanip API Usage
**Decision**: Use high-level `load_model()` API
- **Pro**: Simple, stable API
- **Pro**: Handles CellML parsing complexity
- **Con**: Less control over model details
- **Alternative**: Parse CellML XML directly - rejected as reinventing the wheel

### Test Coverage
**Decision**: Test with multiple model sizes
- **Pro**: Catches edge cases (simple vs complex models)
- **Pro**: Validates performance characteristics
- **Con**: Slower test suite
- **Mitigation**: Use pytest markers (e.g., `@pytest.mark.slow`)

## Expected Outcomes

1. **Verified functionality**: load_cellml_model confirmed working with real models
2. **Comprehensive tests**: Multiple test cases covering success and error paths
3. **Documentation**: Clear examples of how to use CellML import
4. **Foundation for future work**: Enables users to import Physiome models directly
