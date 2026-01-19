# Agent Plan: Python Function Parser for CuBIE

## Component Overview

This plan details the implementation of a Python function parser that converts user-provided Python functions into SymPy symbolic expressions for CUDA code generation. The parser extracts variable information from function signatures and AST analysis, then integrates with existing symbolic ODE infrastructure.

## 1. Source Code Interactions

### 1.1 Function Inspection Module

**Purpose**: Extract function metadata and source code for analysis

**Components Needed**:
- Function signature extraction using `inspect.signature()`
- Source code retrieval using `inspect.getsource()`
- AST tree generation using `ast.parse()`
- Parameter extraction from signature object

**Expected Behavior**:
- Accept callable as input
- Validate callable is a standard function (not lambda, not method, not builtin)
- Extract ordered parameter names from signature
- Retrieve function source code as string
- Parse source into AST tree for traversal
- Handle cases where source is unavailable (raise informative error)

**Data Structures**:
- Store parameter names as ordered list
- Store AST as Python ast.Module object
- Store source string for error reporting

**Edge Cases**:
- Lambda functions: reject with clear error suggesting regular function
- Built-in functions: reject (no source available)
- Methods: accept but warn about `self` parameter
- Nested functions: accept if source available
- Functions from REPL: reject with error (no source)

### 1.2 AST Traversal and Analysis

**Purpose**: Walk AST to identify variable access patterns and operations

**Components Needed**:
- AST visitor class inheriting from `ast.NodeVisitor`
- Visit methods for: `Subscript`, `Attribute`, `Name`, `Return`, `Assign`
- Context tracking to distinguish LHS from RHS of assignments
- Return statement analysis for output identification

**Expected Behavior**:
- Traverse entire function body
- Collect all `y[index]` subscript accesses
- Collect all `constants.name` attribute accesses
- Collect all `constants["name"]` subscript accesses
- Track assignment targets vs values
- Identify return statement structure
- Build mapping of variable access patterns

**Data Structures**:
```python
{
    'state_accesses': [
        {'pattern': 'subscript_int', 'base': 'y', 'key': 0},
        {'pattern': 'subscript_str', 'base': 'y', 'key': 'velocity'},
        {'pattern': 'attribute', 'base': 'y', 'key': 'position'},
    ],
    'constant_accesses': [
        {'pattern': 'attribute', 'base': 'constants', 'key': 'damping'},
        {'pattern': 'subscript_str', 'base': 'constants', 'key': 'mass'},
    ],
    'assignments': {
        'dv': <ast.Expr>,
        'dx': <ast.Expr>,
    },
    'return_value': <ast.Return>
}
```

**Integration Points**:
- Must identify state argument name from signature (typically `y`)
- Must identify constants argument name from signature
- Must convert AST expressions to SymPy expressions

### 1.3 Expression Conversion

**Purpose**: Convert Python AST expressions to SymPy symbolic expressions

**Components Needed**:
- Expression converter that walks AST and builds SymPy
- Mapping from Python operators to SymPy operations
- Function call handling (sin, cos, exp, etc.)
- Symbol substitution for identified variables

**Expected Behavior**:
- Convert `ast.BinOp` to SymPy arithmetic operations
- Convert `ast.Call` to SymPy function applications
- Convert `ast.UnaryOp` to SymPy unary operations
- Substitute variable names with SymPy symbols from IndexedBases
- Handle function calls via KNOWN_FUNCTIONS mapping
- Support user-defined functions same as string parser

**Challenges**:
- Python syntax may differ from SymPy conventions
- Must track which names are states vs constants vs intermediates
- Array indexing needs special handling for symbolic conversion
- Conditional expressions need Piecewise conversion

## 2. Parsing and Identifying Variables

### 2.1 State Variable Identification

**Purpose**: Determine state variable names and ordering from access patterns

**Expected Behavior**:
- Scan function body for all accesses to state argument (e.g., `y`)
- Extract indices/keys from subscript operations
- Extract attribute names from attribute access
- Validate consistency of access patterns
- Build ordered list of state names

**Access Pattern Recognition**:

**Integer indexing**: `y[0]`, `y[1]`, `y[2]`
- Extract indices: 0, 1, 2
- Generate names: `y_0`, `y_1`, `y_2`
- Maintain index order

**String indexing**: `y["velocity"]`, `y["position"]`
- Extract keys: "velocity", "position"
- Use keys as state names directly
- Order by first appearance in function body

**Attribute access**: `y.velocity`, `y.position`
- Extract attribute names
- Use as state names directly
- Order by first appearance

**Mixed patterns**: Error or warning
- If both integer and named access detected, raise error
- Suggest using consistent access pattern

**User Override**:
- Allow user to explicitly specify `states` argument to override inference
- Use inference to validate against explicit specification
- Warn if inferred states don't match explicit list

### 2.2 Constant and Parameter Identification

**Purpose**: Distinguish between constants and parameters from additional function arguments

**Expected Behavior**:
- Arguments after state argument become constant sources
- Constants accessed via attribute or subscript on constant argument
- Direct argument names become constant symbols
- User can specify which are parameters vs constants

**Patterns**:

**Direct arguments**: `def f(t, y, k, m, damping)`
- Extract argument names: k, m, damping
- Each becomes a constant symbol by default
- User specifies parameters list to reclassify

**Dict/object arguments**: `def f(t, y, constants)`
- Scan for `constants.name` or `constants["name"]`
- Extract accessed names as constants
- All become constants unless user specifies parameters

**Combined**: `def f(t, y, k, m, constants)`
- Direct args: k, m
- Accessed from constants: whatever is referenced
- All available for parameter/constant classification

**User Specification**:
- Accept `parameters` argument as list of names
- Accept `constants` argument as list of names or dict of defaults
- Validate specified names exist in function
- Error if name not found in function

### 2.3 Observable Identification

**Purpose**: Identify which intermediate calculations should be saved as observables

**Expected Behavior**:
- User provides `observables` list of names
- Scan function body for assignment to observable names
- Extract RHS expressions for those assignments
- Create observable equations

**Pattern**:
```python
def f(t, y, constants):
    v = y[0]
    x = y[1]
    k = constants.damping
    
    dv = -k * v  # derivative
    dx = v       # derivative
    
    kinetic_energy = 0.5 * v**2  # observable
    
    return [dv, dx]
```

User specifies: `observables=["kinetic_energy"]`
- Parser finds `kinetic_energy = 0.5 * v**2`
- Creates observable equation in ParsedEquations
- Validates observable is assigned before return

### 2.4 Output Variable Identification

**Purpose**: Extract derivative variable names and ordering from return statement

**Expected Behavior**:
- Locate return statement in function body
- Analyze return value structure
- Extract variable names or expressions being returned
- Create mapping to state derivatives

**Return Patterns**:

**List of names**: `return [dv, dx, dy]`
- Extract names: dv, dx, dy
- Match to states by position or name matching
- If names follow `d{state}` pattern, auto-match

**Tuple of names**: `return (dv, dx)`
- Same as list

**Dict**: `return {"velocity": dv, "position": dx}`
- Keys should match state names
- Allows explicit mapping

**Array construction**: `return np.array([dv, dx])`
- Extract arguments to array constructor
- Use as list

**Single expression**: `return expr`
- Assume single-state system
- Create derivative for that state

**Validation**:
- Number of returned values should match number of states
- If user provided `states`, validate count matches
- If inferred states, use return count to validate

### 2.5 Symbol Management

**Purpose**: Build IndexedBases consistent with existing parser output

**Expected Behavior**:
- Create `IndexedBases` object with states, parameters, constants
- Generate SymPy symbols for all identified variables
- Create derivative symbols following `d{name}` convention
- Maintain mapping from function variable names to SymPy symbols
- Handle observables as in string parser

**Integration with Existing System**:
- Use `IndexedBases.from_user_inputs()` where possible
- Provide inferred names where user didn't specify
- Merge user-provided defaults with inferred structure
- Validate no conflicts between inference and user input

## 3. Integration with SymbolicODE

### 3.1 FunctionParser Class Design

**Purpose**: Encapsulate function parsing logic in cohesive class

**Class Structure**:
```python
class FunctionParser:
    """Parse Python function into symbolic ODE representation."""
    
    def __init__(
        self,
        func: Callable,
        states: Optional[...] = None,
        parameters: Optional[...] = None,
        constants: Optional[...] = None,
        observables: Optional[...] = None,
        drivers: Optional[...] = None,
        strict: bool = False,
    ):
        # Validate func is callable
        # Extract signature
        # Get source and parse AST
        # Store user specifications
    
    def parse(self) -> Tuple[ParsedEquations, IndexedBases, ...]:
        # Main parsing workflow
        # Returns same structure as string parser
```

**Methods**:
- `_extract_signature()`: Get function parameters
- `_parse_ast()`: Create AST tree
- `_identify_states()`: Find state variable accesses
- `_identify_constants()`: Find constant accesses
- `_identify_outputs()`: Analyze return statement
- `_build_indexed_bases()`: Create IndexedBases
- `_convert_to_sympy()`: Convert AST to SymPy equations
- `parse()`: Orchestrate full parsing

**Output Format**:
Must match `parse_input()` return type:
```python
Tuple[
    IndexedBases,
    Dict[str, object],  # all symbols
    Dict[str, Callable],  # user functions
    ParsedEquations,
    str,  # hash
]
```

### 3.2 Equation Construction

**Purpose**: Build ParsedEquations from identified variables and expressions

**Expected Behavior**:
- For each state, create equation `d{state} = expression`
- For each observable, create equation `observable = expression`
- For each auxiliary (intermediate variable), create equation
- Order equations to respect dependencies
- Partition into state_derivatives, observables, auxiliaries

**Process**:
1. Walk function body collecting assignments
2. Identify which assignments are states, observables, auxiliaries
3. Convert RHS expressions to SymPy
4. Substitute variable names with appropriate SymPy symbols
5. Build equation tuples `(lhs_symbol, rhs_expr)`
6. Create ParsedEquations with proper categorization

**Symbol Substitution**:
- State accesses `y[i]` → state symbols from IndexedBases
- Constant accesses → constant symbols from IndexedBases
- Intermediate names → auxiliary symbols (auto-created)
- Time variable `t` → TIME_SYMBOL

### 3.3 SymPy Expression Building

**Purpose**: Convert Python expressions to SymPy while maintaining semantics

**Expected Behavior**:
- Walk AST expression nodes
- Convert operators to SymPy equivalents
- Handle mathematical functions via KNOWN_FUNCTIONS
- Support user_functions same as string parser
- Substitute names with SymPy symbols

**Operator Mapping**:
- `ast.Add` → `sp.Add`
- `ast.Sub` → `sp.Add` with negation
- `ast.Mult` → `sp.Mul`
- `ast.Div` → `sp.Mul` with power
- `ast.Pow` → `sp.Pow`
- `ast.UnaryOp` → SymPy unary operations

**Function Calls**:
- Check KNOWN_FUNCTIONS mapping
- Check user_functions
- Create SymPy function application
- Handle derivatives for user functions as in string parser

**Challenges**:
- Python division vs SymPy division semantics
- Integer vs float literals
- Handling of comparisons in conditionals
- Array operations not directly translatable

### 3.4 Validation and Error Handling

**Purpose**: Provide clear errors when function structure is problematic

**Validation Checks**:
- Function has at least 2 parameters (t, y)
- State argument is accessed in function body
- Return statement exists
- Number of return values matches inferred/specified states
- All specified observables are assigned in function
- All specified parameters/constants are accessible
- No unsupported Python features used

**Error Messages**:
- "Function must have at least 2 parameters (time, state)"
- "State argument 'y' is never accessed in function body"
- "No return statement found in function"
- "Return statement has {n} values but {m} states were identified"
- "Observable '{name}' specified but never assigned in function"
- "Parameter '{name}' specified but not found in function arguments"

**Warnings**:
- Mixed state access patterns (int and string indexing)
- Unused function parameters
- State accessed but not used in derivatives
- Complex Python features that may not translate well

## 4. Integration with CuBIE and Cleanup/Modification

### 4.1 Parser Module Reorganization

**Purpose**: Organize parsing code for clarity and maintainability

**File Structure Changes**:

**Before**:
```
parsing/
  __init__.py
  parser.py  # all parsing logic
  auxiliary_caching.py
  cellml.py
  jvp_equations.py
```

**After**:
```
parsing/
  __init__.py
  parser.py  # parse_input() entry point and utilities
  string_parser.py  # StringParser class with string-specific logic
  function_parser.py  # FunctionParser class (new)
  auxiliary_caching.py
  cellml.py
  jvp_equations.py
```

**Code Movement**:
- Move string-specific parsing logic to `StringParser` class in `string_parser.py`
- Keep `parse_input()`, `_detect_input_type()`, and shared utilities in `parser.py`
- Create `FunctionParser` in new `function_parser.py`
- Update imports in `__init__.py`

### 4.2 parse_input() Modifications

**Purpose**: Route function input to FunctionParser

**Changes to _detect_input_type()**:
```python
def _detect_input_type(dxdt: Union[str, Iterable, Callable]) -> str:
    """Detect input format: 'string', 'sympy', or 'function'."""
    
    if callable(dxdt):
        return 'function'
    
    # ... existing string and sympy detection ...
```

**Changes to parse_input()**:
```python
def parse_input(...) -> Tuple[...]:
    # ... existing parameter handling ...
    
    input_type = _detect_input_type(dxdt)
    
    if input_type == "function":
        parser = FunctionParser(
            func=dxdt,
            states=states,
            parameters=parameters,
            constants=constants,
            observables=observables,
            drivers=drivers,
            strict=strict,
            # ... unit parameters ...
        )
        return parser.parse()
    
    elif input_type == "string":
        # ... existing string parsing ...
    
    elif input_type == "sympy":
        # ... existing sympy parsing ...
```

### 4.3 API Design

**Purpose**: Provide intuitive interface for function-based input

**User-Facing API**:

**Basic usage**:
```python
def my_ode(t, y):
    v = y[0]
    x = y[1]
    return [-0.1 * v, v]

system = create_ODE_system(
    dxdt=my_ode,
    states=["velocity", "position"],
    precision=np.float32,
)
```

**With constants**:
```python
def my_ode(t, y, constants):
    v = y["velocity"]
    x = y["position"]
    k = constants.damping
    return {"velocity": -k * v, "position": v}

system = create_ODE_system(
    dxdt=my_ode,
    states={"velocity": 1.0, "position": 0.0},
    constants={"damping": 0.1},
    precision=np.float32,
)
```

**With parameters and observables**:
```python
def my_ode(t, y, k, m):
    v = y[0]
    x = y[1]
    
    dv = -k * v / m
    dx = v
    
    kinetic_energy = 0.5 * m * v**2
    
    return [dv, dx]

system = create_ODE_system(
    dxdt=my_ode,
    states=["velocity", "position"],
    parameters={"k": 0.1, "m": 1.0},
    observables=["kinetic_energy"],
    precision=np.float32,
)
```

### 4.4 Testing Strategy

**Purpose**: Ensure function parser produces equivalent output to string parser

**Test Categories**:

**Equivalence tests**:
- Define same system as function and string
- Compare generated CUDA code
- Compare numerical results
- Test all access patterns (integer, string, attribute)

**Function-specific tests**:
- Lambda rejection
- Builtin rejection
- Missing return statement
- Mismatched return count
- Unused observables
- Invalid parameter references

**Integration tests**:
- Function system through full solve pipeline
- Parameter sweeps with function-defined system
- Observables saved correctly
- Driver integration

**Edge case tests**:
- Single-state system
- No constants
- No observables
- Complex expressions
- Nested function calls
- Conditional expressions

### 4.5 Documentation Updates

**Purpose**: Guide users in function-based ODE definition

**Documentation Needed**:
- Update `create_ODE_system()` docstring with function examples
- Add tutorial showing function vs string approach
- Document supported function signatures
- Document variable identification rules
- Show examples of each access pattern
- Explain parameter vs constant distinction
- Document limitations and unsupported features

**Example Content**:
```rst
Function-Based ODE Definition
==============================

CuBIE supports defining ODE systems using Python functions, providing
a familiar interface for users of scipy.integrate.solve_ivp and MATLAB's
ode45.

Basic Function Structure
------------------------

Your ODE function should have the signature::

    def f(t, y):
        # t is current time (scalar)
        # y is current state (access via indexing or attributes)
        ...
        return [dy0_dt, dy1_dt, ...]  # derivatives

State Access Patterns
---------------------

Integer indexing::
    v = y[0]  # First state
    x = y[1]  # Second state

String indexing::
    v = y["velocity"]
    x = y["position"]

Attribute access::
    v = y.velocity
    x = y.position

Constants and Parameters
------------------------

Additional function arguments provide constants::

    def f(t, y, k, m):
        # k and m are constants
        ...

Or as dict/object::

    def f(t, y, constants):
        k = constants.damping
        m = constants["mass"]
        ...

Specify which are parameters vs constants::

    system = create_ODE_system(
        dxdt=f,
        parameters=["k"],  # k varies in sweeps
        constants=["m"],   # m is fixed
    )
```

### 4.6 Backward Compatibility

**Purpose**: Ensure existing code continues to work

**Guarantees**:
- All existing string-based tests pass unchanged
- All existing SymPy-based tests pass unchanged
- API for `create_ODE_system()` adds optional callable support
- Default behavior unchanged when strings provided
- String parser renamed internally but not in public API

**Migration Path**:
- Users can adopt function interface incrementally
- Existing models don't need changes
- Can mix function and string systems in same codebase

## Component Dependencies

```
FunctionParser
  ├─ inspect.signature (stdlib)
  ├─ inspect.getsource (stdlib)
  ├─ ast.parse (stdlib)
  ├─ ast.NodeVisitor (stdlib)
  ├─ IndexedBases (existing)
  ├─ ParsedEquations (existing)
  ├─ KNOWN_FUNCTIONS (existing)
  └─ TIME_SYMBOL (existing)

parse_input()
  ├─ _detect_input_type (modified)
  ├─ StringParser (refactored from existing)
  ├─ FunctionParser (new)
  └─ SymPy normalization path (existing)

create_ODE_system()
  └─ parse_input() (accepts callable)

SymbolicODE
  └─ No changes required
```

## Expected Outputs

After implementation, users can:
1. Pass Python functions to `create_ODE_system()`
2. Use scipy/MATLAB-style function signatures
3. Access states via indexing or attributes
4. Organize constants naturally
5. Get clear errors for invalid function structure
6. Generate identical CUDA kernels as string parser

The implementation maintains clean separation between parsing strategies while reusing all existing symbolic infrastructure.
