# Section 1 Research Summary

## Research Conducted

### 1. Python AST Module Capabilities

**Tested**: Basic function parsing with ast.parse() and NodeVisitor pattern

**Key Findings**:
- `ast.parse()` successfully converts function source to AST
- `ast.NodeVisitor` provides clean pattern for selective node traversal
- `ast.unparse()` available for debugging (Python 3.9+)
- Key node types for ODE parsing identified and tested

**Test Results**:
```python
# Successfully parsed and analyzed example function
def example_ode(t, y, constants):
    v = y[0]
    x = y["position"]
    k = constants.damping
    m = constants["mass"]
    dv = -k * v / m
    dx = v
    kinetic_energy = 0.5 * m * v**2
    return [dv, dx]

# Extracted patterns:
State Accesses:
  {'base': 'y', 'key': 0, 'type': 'subscript', 'key_type': 'int'}
  {'base': 'y', 'key': 'position', 'type': 'subscript', 'key_type': 'str'}

Constant Accesses:
  {'base': 'constants', 'key': 'damping', 'type': 'attribute'}
  {'base': 'constants', 'key': 'mass', 'type': 'subscript', 'key_type': 'str'}
```

### 2. Python inspect Module Capabilities

**Tested**: Function signature and source extraction

**Key Findings**:
- `inspect.signature()` provides structured parameter metadata
- `inspect.getsource()` works for file-defined functions
- `inspect.getsource()` raises OSError for REPL/Jupyter functions
- Parameter.kind distinguishes POSITIONAL_OR_KEYWORD, VAR_POSITIONAL, etc.

**Edge Cases Confirmed**:
- Lambda detection: `func.__name__ == '<lambda>'` works reliably
- REPL functions: OSError with message "could not get source code"
- Builtin functions: TypeError from getsource

### 3. CuBIE Parser Infrastructure

**Examined Files**:
- `src/cubie/odesystems/symbolic/parsing/parser.py` (1629 lines)
- `src/cubie/odesystems/symbolic/indexedbasemaps.py`
- `tests/odesystems/symbolic/test_parser.py`

**Key Findings**:
- `parse_input()` is main entry point at line 1382
- `_detect_input_type()` currently handles 'string' and 'sympy'
- `ParsedEquations` attrs class (line 302) structures output
- `KNOWN_FUNCTIONS` dict (line 242) maps function names to SymPy
- `TIME_SYMBOL = sp.Symbol("t", real=True)` (line 31)
- `IndexedBases` manages symbol categorization
- Existing test patterns in test_parser.py provide template

### 4. GitHub Code Search

**Query**: "ast.NodeVisitor parse function language:python"

**Findings**:
- 6496+ repositories use NodeVisitor pattern
- Confirms this is standard approach for AST analysis
- Examples found in pythoncodeanalysis, tailbiter, python_nle

**Pattern Validation**:
- NodeVisitor subclass with visit_* methods is idiomatic
- Common to track state during traversal
- Generic_visit() call preserves tree walking

### 5. scipy.integrate.solve_ivp Convention Research

**Standard Signature**: `fun(t, y, *args)`
- First parameter: time (scalar)
- Second parameter: state (array)
- Additional parameters: constants/parameters

**Rationale**: This convention is well-established in:
- scipy.integrate.solve_ivp
- MATLAB ode45
- Various ODE solver libraries

**Decision**: Adopt this convention for CuBIE function parser

## Design Decisions Based on Research

### Decision 1: Three-Module Architecture
**Based on**: CuBIE code organization patterns in parsing/ directory
**Rationale**: Separation of concerns matches existing structure

### Decision 2: ast.NodeVisitor Pattern  
**Based on**: GitHub search showing 6496+ examples, Python best practices
**Rationale**: Proven, idiomatic, extensible

### Decision 3: Comprehensive Error Messages
**Based on**: CuBIE existing error handling in parser.py
**Rationale**: Match quality of existing validation and error reporting

### Decision 4: Support Multiple Access Patterns
**Based on**: User convenience research (scipy, MATLAB patterns)
**Rationale**: Different users prefer different styles (index vs name)

## Implementation Risks Identified

### Risk 1: REPL/Jupyter Compatibility
**Issue**: Cannot get source for interactively defined functions
**Mitigation**: Clear error message directing to file-based definition or string input

### Risk 2: Complex Python Syntax
**Issue**: Python supports features not translatable to SymPy
**Mitigation**: Detect unsupported patterns early, provide helpful errors

### Risk 3: Symbol Name Inference
**Issue**: Inferring meaningful variable names from patterns
**Mitigation**: Covered in Section 2, allow user override

## Outstanding Questions (for Section 2)

1. How to handle observables in function-based input?
   - User explicitly lists observable names?
   - Detect from calculation but no return?

2. How to distinguish parameters from constants?
   - User must explicitly specify?
   - All additional args default to constants?

3. How to handle drivers in function-based input?
   - Passed as additional argument?
   - Special decorator or annotation?

These will be addressed in Section 2 (Variable Identification).

## References

- Python ast documentation: https://docs.python.org/3/library/ast.html
- Python inspect documentation: https://docs.python.org/3/library/inspect.html
- SymPy documentation: https://docs.sympy.org/
- scipy.integrate.solve_ivp: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
- MATLAB ode45: https://www.mathworks.com/help/matlab/ref/ode45.html
