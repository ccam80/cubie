# Section 1: Source Code Interactions - Detailed Agent Plan

## Component 1: Function Inspector Module

### File: `src/cubie/odesystems/symbolic/parsing/function_inspector.py`

#### Purpose
Extract function metadata (signature, source code, AST) with comprehensive validation and error handling.

#### Classes and Methods

##### Class: `FunctionInspector`

**Attributes:**
- `func: Callable` - The user-provided function
- `signature: inspect.Signature` - Extracted function signature
- `source: str` - Function source code
- `ast_tree: ast.Module` - Parsed AST tree
- `func_def: ast.FunctionDef` - Function definition node from AST
- `param_names: List[str]` - Ordered list of parameter names

**Methods:**

**`__init__(self, func: Callable)`**
- Accept callable as input
- Validate func is callable
- Immediately extract signature, source, and AST
- Raise clear errors for unsupported function types

Expected behavior:
- Call `_validate_function()` first
- Call `_extract_signature()`
- Call `_extract_source()`
- Call `_parse_ast()`
- Store all results as attributes

**`_validate_function(self) -> None`**
- Check if function is lambda: `func.__name__ == '<lambda>'`
- Check if function is builtin: try `inspect.getsource()`, catch TypeError
- Check if function is method: first param is 'self' → warn but allow
- Raise `TypeError` with specific message for unsupported types

Expected behavior:
- Lambda: `TypeError("Lambda functions are not supported. Please use 'def' syntax to define your ODE function.")`
- Builtin: `TypeError("Cannot parse builtin or C-extension functions. Please use a regular Python function.")`
- REPL function: Handled in `_extract_source()` via OSError

**`_extract_signature(self) -> inspect.Signature`**
- Use `inspect.signature(self.func)`
- Extract parameter names in order
- Store in `self.param_names` as list
- Return signature object

Expected behavior:
- Handle functions with defaults
- Preserve parameter order
- Capture all parameter metadata

**`_extract_source(self) -> str`**
- Use `inspect.getsource(self.func)`
- Catch `OSError` if source unavailable (REPL, dynamically generated)
- Strip leading/trailing whitespace
- Return source string

Expected behavior:
- Success: return clean source code
- REPL function: `OSError("Cannot retrieve source code for functions defined in interactive mode (REPL/Jupyter). Please define the function in a .py file or use string-based equation input.")`
- Preserve indentation of function body for AST parsing

**`_parse_ast(self) -> Tuple[ast.Module, ast.FunctionDef]`**
- Use `ast.parse(self.source)`
- Extract FunctionDef node (first body element)
- Validate it's actually a FunctionDef
- Return both module and function def

Expected behavior:
- Parse source into AST successfully
- Find function definition node
- Raise `ValueError` if AST structure unexpected

**`get_parameter_info(self) -> Dict[str, Dict[str, Any]]`**
- Return detailed parameter information
- For each parameter: name, kind, default, annotation

Expected behavior:
```python
{
    't': {'kind': 'POSITIONAL_OR_KEYWORD', 'default': None, 'annotation': None},
    'y': {'kind': 'POSITIONAL_OR_KEYWORD', 'default': None, 'annotation': None},
    'constants': {'kind': 'POSITIONAL_OR_KEYWORD', 'default': None, 'annotation': None}
}
```

**`validate_ode_signature(self) -> None`**
- Check function has at least 2 parameters
- First parameter conventionally 't' (time)
- Second parameter conventionally 'y' or 'state' (state vector)
- Warn if unconventional names used

Expected behavior:
- Less than 2 params: `ValueError("ODE function must have at least 2 parameters (time, state). Found {n} parameters.")`
- Unconventional names: `Warning("Expected first parameter to be 't' (time), found '{name}'. Expected second parameter to be 'y' or 'state', found '{name}'.")`

#### Module-Level Functions

**`is_lambda(func: Callable) -> bool`**
- Return `func.__name__ == '<lambda>'`

**`is_builtin(func: Callable) -> bool`**
- Try `inspect.getsource(func)`, catch TypeError
- Return True if TypeError, False otherwise

**`get_function_name(func: Callable) -> str`**
- Return `func.__name__`
- Use for error messages

#### Error Messages

Maintain consistency with CuBIE error message style:
- Clear indication of what went wrong
- Suggestion for how to fix
- Reference to alternative approaches when applicable

```python
# Examples
"Lambda functions are not supported. Please use 'def' syntax to define your ODE function."

"Cannot retrieve source code for functions defined in interactive mode (REPL/Jupyter). 
Please define the function in a .py file or use string-based equation input."

"ODE function must have at least 2 parameters (time, state). Found {n} parameters: {names}.
Expected signature: def f(t, y, ...)"

"Cannot parse builtin or C-extension functions. Please use a regular Python function 
defined in Python code, or use CuBIE's string-based equation syntax."
```

---

## Component 2: AST Visitor Module

### File: `src/cubie/odesystems/symbolic/parsing/ast_visitor.py`

#### Purpose
Traverse function AST to identify variable access patterns, assignments, and return values.

#### Data Structures

**AccessPattern TypedDict:**
```python
class AccessPattern(TypedDict):
    base: str           # 'y', 'constants', etc.
    key: Union[int, str]  # Index or attribute name
    pattern_type: str   # 'subscript_int', 'subscript_str', 'attribute'
    node: ast.AST       # Original AST node for error reporting
```

**VisitorResults TypedDict:**
```python
class VisitorResults(TypedDict):
    state_accesses: List[AccessPattern]
    constant_accesses: List[AccessPattern]
    assignments: Dict[str, ast.expr]
    return_node: Optional[ast.Return]
    function_calls: Set[str]
```

#### Classes and Methods

##### Class: `OdeAstVisitor(ast.NodeVisitor)`

**Attributes:**
- `state_param_name: str` - Name of state parameter (typically 'y')
- `constant_param_names: List[str]` - Names of constant parameters
- `state_accesses: List[AccessPattern]` - Collected state access patterns
- `constant_accesses: List[AccessPattern]` - Collected constant access patterns
- `assignments: Dict[str, ast.expr]` - Map of variable name to RHS expression
- `return_node: Optional[ast.Return]` - Return statement node
- `function_calls: Set[str]` - Set of function names called
- `_in_assignment_target: bool` - Track if visiting LHS of assignment

**Methods:**

**`__init__(self, state_param: str, constant_params: List[str])`**
- Store parameter names for identification
- Initialize empty collections

Expected behavior:
- Accept 'y' as state_param
- Accept list of constant parameter names
- Initialize all tracking attributes

**`visit_Subscript(self, node: ast.Subscript) -> None`**
- Check if subscript base is a Name node
- If base matches state_param_name or constant_param_names, record access
- Determine if subscript is integer or string constant
- Create AccessPattern and add to appropriate list
- Call `self.generic_visit(node)` to continue traversal

Expected behavior:
```python
# For: y[0]
AccessPattern(
    base='y',
    key=0,
    pattern_type='subscript_int',
    node=node
)

# For: constants["damping"]
AccessPattern(
    base='constants',
    key='damping',
    pattern_type='subscript_str',
    node=node
)
```

Validation:
- Ignore if not visiting a Name base (e.g., `arr[i][j]` should only record outer)
- Ignore if subscript is not a constant (e.g., `y[i]` with variable `i`)
- Record non-constant subscripts separately for warning

**`visit_Attribute(self, node: ast.Attribute) -> None`**
- Check if attribute access is on a Name node
- If base matches state_param_name or constant_param_names, record access
- Create AccessPattern with attribute name
- Call `self.generic_visit(node)`

Expected behavior:
```python
# For: constants.damping
AccessPattern(
    base='constants',
    key='damping',
    pattern_type='attribute',
    node=node
)
```

**`visit_Assign(self, node: ast.Assign) -> None`**
- Extract target (assumes single target)
- If target is Name node, store mapping: `name -> node.value`
- Set `_in_assignment_target = True` before visiting targets
- Set `_in_assignment_target = False` before visiting value
- Call `self.generic_visit(node)`

Expected behavior:
- Track all `variable = expression` assignments
- Map variable name to RHS expression AST node
- Used later for identifying intermediate variables and derivatives

**`visit_Return(self, node: ast.Return) -> None`**
- Store return node
- Warn if return_node already set (multiple returns)
- Call `self.generic_visit(node)`

Expected behavior:
- Capture return statement
- Only one return expected (warn if multiple)
- Return value can be: Name, List, Tuple, Dict, Call (e.g., np.array)

**`visit_Call(self, node: ast.Call) -> None`**
- Extract function name if func is Name or Attribute
- Add to function_calls set
- Call `self.generic_visit(node)`

Expected behavior:
- Track all function calls: `sin(x)`, `np.exp(y)`, etc.
- Used to validate against KNOWN_FUNCTIONS
- Handle both `func(...)` and `module.func(...)` patterns

**`get_results(self) -> VisitorResults`**
- Return collected data as VisitorResults dict

Expected behavior:
- Package all collected information
- Used by FunctionParser to build symbolic representation

**`validate_consistency(self) -> None`**
- Check state accesses for mixed patterns (int and str on same base)
- Check all constant parameters are actually used
- Check return statement exists

Expected behavior:
- Mixed state access: `ValueError("Inconsistent state access pattern: found both integer indexing (y[0]) and string indexing (y['name']). Use one pattern consistently.")`
- No return: `ValueError("Function must include a return statement to specify derivatives.")`
- Warn for unused parameters

#### Utility Functions

**`extract_subscript_key(slice_node: ast.expr) -> Union[int, str, None]`**
- If slice is Constant with int/str value, return value
- Otherwise return None (non-constant subscript)

Expected behavior:
- `ast.Constant(value=0)` → `0`
- `ast.Constant(value="velocity")` → `"velocity"`
- `ast.Name(id="i")` → `None`

**`get_access_location(node: ast.AST) -> str`**
- Return string describing source location for error messages
- Use node.lineno and node.col_offset if available

Expected behavior:
- Return: `"line {lineno}, column {col_offset}"`
- Used in error messages to help user locate issues

---

## Component 3: AST to SymPy Converter Module

### File: `src/cubie/odesystems/symbolic/parsing/ast_converter.py`

#### Purpose
Convert Python AST expression nodes to SymPy symbolic expressions.

#### Classes and Methods

##### Class: `AstToSympyConverter`

**Attributes:**
- `symbol_map: Dict[str, sp.Symbol]` - Maps variable names to SymPy symbols
- `known_functions: Dict[str, Callable]` - Maps function names to SymPy functions
- `time_symbol: sp.Symbol` - The time variable symbol

**Methods:**

**`__init__(self, symbol_map: Dict[str, sp.Symbol], known_functions: Dict[str, Callable], time_symbol: sp.Symbol)`**
- Store symbol mapping
- Store function mapping
- Store time symbol

Expected behavior:
- symbol_map from IndexedBases will provide: `{'v': Symbol('v'), 'x': Symbol('x'), 'k': Symbol('k')}`
- known_functions from KNOWN_FUNCTIONS dict
- time_symbol is TIME_SYMBOL from parser.py

**`convert(self, node: ast.expr) -> sp.Expr`**
- Main entry point for conversion
- Dispatch to specific convert methods based on node type
- Return SymPy expression

Expected behavior:
- `ast.BinOp` → call `_convert_binop()`
- `ast.UnaryOp` → call `_convert_unaryop()`
- `ast.Call` → call `_convert_call()`
- `ast.Name` → call `_convert_name()`
- `ast.Constant` → call `_convert_constant()`
- `ast.IfExp` → call `_convert_ifexp()`
- Other types → raise `NotImplementedError`

**`_convert_binop(self, node: ast.BinOp) -> sp.Expr`**
- Convert left and right operands recursively
- Map operator to SymPy equivalent
- Return constructed SymPy expression

Expected behavior:
```python
ast.Add → sp.Add(left, right)
ast.Sub → sp.Add(left, sp.Mul(-1, right))
ast.Mult → sp.Mul(left, right)
ast.Div → sp.Mul(left, sp.Pow(right, -1))
ast.Pow → sp.Pow(left, right)
ast.Mod → sp.Mod(left, right)
ast.FloorDiv → sp.floor(sp.Mul(left, sp.Pow(right, -1)))
```

**`_convert_unaryop(self, node: ast.UnaryOp) -> sp.Expr`**
- Convert operand recursively
- Map operator to SymPy equivalent
- Return constructed expression

Expected behavior:
```python
ast.UAdd → operand (no change)
ast.USub → sp.Mul(-1, operand)
ast.Not → sp.Not(operand)
```

**`_convert_call(self, node: ast.Call) -> sp.Expr`**
- Extract function name
- Look up in known_functions
- Convert arguments recursively
- Apply SymPy function to arguments
- Handle special cases (Piecewise, Min, Max, etc.)

Expected behavior:
- `sin(x)` → `sp.sin(symbol_map['x'])`
- `max(a, b)` → `sp.Max(symbol_map['a'], symbol_map['b'])`
- Unknown function → raise `ValueError("Unknown function '{name}'. Use user_functions parameter or string input.")`

**`_convert_name(self, node: ast.Name) -> sp.Symbol`**
- Look up variable name in symbol_map
- Return corresponding symbol
- Raise error if not found

Expected behavior:
- Found in map → return symbol
- Not found → raise `ValueError("Variable '{name}' not found in symbol mapping. Ensure all variables are declared.")`

**`_convert_constant(self, node: ast.Constant) -> sp.Expr`**
- Convert Python constant to SymPy
- Handle int, float, bool, None
- Preserve numeric precision

Expected behavior:
```python
ast.Constant(1) → sp.Integer(1)
ast.Constant(1.5) → sp.Float(1.5)
ast.Constant(True) → sp.true
ast.Constant(False) → sp.false
ast.Constant(None) → raise ValueError("None not supported in expressions")
```

**`_convert_ifexp(self, node: ast.IfExp) -> sp.Piecewise`**
- Convert to SymPy Piecewise
- Recursively convert test, body, orelse
- Create Piecewise with (body, test), (orelse, True)

Expected behavior:
```python
# a if x > 0 else b
# →
sp.Piecewise(
    (symbol_map['a'], sp.Gt(symbol_map['x'], 0)),
    (symbol_map['b'], True)
)
```

**`_convert_compare(self, node: ast.Compare) -> sp.Expr`**
- Convert comparison operations
- Handle chained comparisons
- Map operators to SymPy relational

Expected behavior:
```python
x > 0 → sp.Gt(symbol_map['x'], 0)
x < y → sp.Lt(symbol_map['x'], symbol_map['y'])
x == 0 → sp.Eq(symbol_map['x'], 0)
x >= y → sp.Ge(symbol_map['x'], symbol_map['y'])
```

**`_convert_boolop(self, node: ast.BoolOp) -> sp.Expr`**
- Convert boolean operations (and, or)
- Map to SymPy logical operations

Expected behavior:
```python
a and b → sp.And(symbol_map['a'], symbol_map['b'])
a or b → sp.Or(symbol_map['a'], symbol_map['b'])
```

**`_get_function_name(self, node: ast.expr) -> str`**
- Extract function name from Call node
- Handle both `func` and `module.func` patterns

Expected behavior:
- `ast.Name(id='sin')` → `'sin'`
- `ast.Attribute(value=Name(id='np'), attr='sin')` → `'sin'` (just attribute)
- `ast.Attribute(value=Name(id='sp'), attr='sin')` → `'sin'`

#### Operator Mapping Constants

**`BINOP_MAP: Dict[Type[ast.operator], Callable]`**
```python
BINOP_MAP = {
    ast.Add: lambda l, r: sp.Add(l, r),
    ast.Sub: lambda l, r: sp.Add(l, sp.Mul(-1, r)),
    ast.Mult: lambda l, r: sp.Mul(l, r),
    ast.Div: lambda l, r: sp.Mul(l, sp.Pow(r, -1)),
    ast.Pow: lambda l, r: sp.Pow(l, r),
    ast.Mod: lambda l, r: sp.Mod(l, r),
    ast.FloorDiv: lambda l, r: sp.floor(sp.Mul(l, sp.Pow(r, -1))),
}
```

**`UNARYOP_MAP: Dict[Type[ast.unaryop], Callable]`**
```python
UNARYOP_MAP = {
    ast.UAdd: lambda x: x,
    ast.USub: lambda x: sp.Mul(-1, x),
    ast.Not: lambda x: sp.Not(x),
}
```

**`CMPOP_MAP: Dict[Type[ast.cmpop], Type[sp.Relational]]`**
```python
CMPOP_MAP = {
    ast.Eq: sp.Eq,
    ast.NotEq: sp.Ne,
    ast.Lt: sp.Lt,
    ast.LtE: sp.Le,
    ast.Gt: sp.Gt,
    ast.GtE: sp.Ge,
}
```

---

## Integration Between Components

### Workflow

1. **User passes function to FunctionParser** (to be created in Section 3)
2. **FunctionParser creates FunctionInspector**
   - Validates function
   - Extracts signature, source, AST
3. **FunctionParser creates OdeAstVisitor**
   - Identifies state parameter from signature (position 1)
   - Identifies constant parameters (position 2+)
   - Visits AST to collect access patterns
4. **FunctionParser uses access patterns to build symbol_map**
   - State accesses → state symbols
   - Constant accesses → constant symbols
5. **FunctionParser creates AstToSympyConverter**
   - Provides symbol_map
   - Provides KNOWN_FUNCTIONS
6. **FunctionParser converts assignments to SymPy equations**
   - For each assignment, convert RHS using converter
   - Build equation tuples

### Data Flow Example

```python
# Input function
def my_ode(t, y, constants):
    v = y[0]
    x = y[1]
    k = constants.damping
    
    dv = -k * v
    dx = v
    
    return [dv, dx]

# After FunctionInspector
signature: Signature(parameters={'t': ..., 'y': ..., 'constants': ...})
param_names: ['t', 'y', 'constants']
ast_tree: <ast.Module with FunctionDef>

# After OdeAstVisitor
state_accesses: [
    {'base': 'y', 'key': 0, 'pattern_type': 'subscript_int'},
    {'base': 'y', 'key': 1, 'pattern_type': 'subscript_int'}
]
constant_accesses: [
    {'base': 'constants', 'key': 'damping', 'pattern_type': 'attribute'}
]
assignments: {
    'v': <ast.Subscript for y[0]>,
    'x': <ast.Subscript for y[1]>,
    'k': <ast.Attribute for constants.damping>,
    'dv': <ast.UnaryOp for -k * v>,
    'dx': <ast.Name for v>
}
return_node: <ast.Return with ast.List([dv, dx])>

# After building symbol_map (in Section 2)
symbol_map: {
    't': TIME_SYMBOL,
    'v': Symbol('y_0'),  # State 0
    'x': Symbol('y_1'),  # State 1
    'k': Symbol('damping'),  # Constant
    'dv': Symbol('dy_0'),  # Derivative 0
    'dx': Symbol('dy_1')   # Derivative 1
}

# After AstToSympyConverter
equations: [
    (Symbol('dy_0'), sp.Mul(-1, sp.Mul(Symbol('damping'), Symbol('y_0')))),
    (Symbol('dy_1'), Symbol('y_0'))
]
```

---

## Error Handling Strategy

### Error Categories

**1. Unsupported Function Type**
- Lambda, builtin, REPL-defined
- Raise `TypeError` immediately in `FunctionInspector.__init__`
- Provide clear alternative (use def syntax, define in file, use string input)

**2. Invalid Signature**
- Less than 2 parameters
- Raise `ValueError` in `validate_ode_signature()`
- Suggest expected signature

**3. Source Unavailable**
- REPL/Jupyter defined function
- Raise `OSError` in `_extract_source()`
- Suggest defining in file or using string input

**4. AST Parsing Error**
- Syntax error in source
- Catch `SyntaxError` from `ast.parse()`
- Re-raise with context about function name

**5. No Return Statement**
- Detected in `OdeAstVisitor.validate_consistency()`
- Raise `ValueError`
- Explain derivatives must be returned

**6. Mixed Access Patterns**
- Both int and str subscripts on same base
- Detected in `OdeAstVisitor.validate_consistency()`
- Raise `ValueError`
- Suggest using consistent pattern

**7. Unknown Function Call**
- Function not in KNOWN_FUNCTIONS or user_functions
- Detected in `AstToSympyConverter._convert_call()`
- Raise `ValueError`
- List available functions or suggest user_functions parameter

**8. Unknown Variable**
- Variable name not in symbol_map
- Detected in `AstToSympyConverter._convert_name()`
- Raise `ValueError`
- List available variables

### Error Message Template

All errors should follow this format:
```
{error_type}: {what_went_wrong}

Found: {specific_issue}
Expected: {what_should_be}

Suggestion: {how_to_fix}
Alternative: {other_approach}  # Optional
```

Example:
```
ValueError: ODE function must have at least 2 parameters

Found: 1 parameter ['x']
Expected: At least 2 parameters (time, state)

Suggestion: Add a time parameter as first argument, e.g.:
    def my_ode(t, x):
        return [-x]
```

---

## Testing Strategy for Section 1

### Test File Structure

Create `tests/odesystems/symbolic/test_function_inspector.py`:
- Test FunctionInspector class
- Test validation
- Test edge cases

Create `tests/odesystems/symbolic/test_ast_visitor.py`:
- Test OdeAstVisitor class
- Test pattern detection
- Test consistency validation

Create `tests/odesystems/symbolic/test_ast_converter.py`:
- Test AstToSympyConverter class
- Test operator conversion
- Test function call conversion

### Fixtures

```python
# Fixture for simple ODE function
@pytest.fixture
def simple_ode_func():
    def f(t, y):
        return [-0.1 * y[0]]
    return f

# Fixture for ODE with constants
@pytest.fixture
def ode_with_constants_func():
    def f(t, y, constants):
        k = constants.damping
        return [-k * y[0]]
    return f

# Fixture for ODE with mixed access
@pytest.fixture
def ode_mixed_access_func():
    def f(t, y):
        v = y[0]
        x = y["position"]  # Mixed int and str!
        return [-0.1 * v, v]
    return f

# Fixture for lambda (should reject)
@pytest.fixture
def lambda_func():
    return lambda t, y: [-0.1 * y[0]]
```

### Test Cases

#### FunctionInspector Tests

1. `test_extract_signature_basic` - Regular function with 2 params
2. `test_extract_signature_with_defaults` - Function with default values
3. `test_extract_source_success` - Source retrieval from file-defined function
4. `test_reject_lambda` - Lambda raises TypeError
5. `test_reject_builtin` - Builtin raises TypeError  
6. `test_parse_ast_success` - AST parsing successful
7. `test_validate_ode_signature_valid` - 2+ params passes
8. `test_validate_ode_signature_insufficient_params` - <2 params raises error
9. `test_get_parameter_info` - Returns correct parameter metadata

#### OdeAstVisitor Tests

1. `test_identify_int_subscripts` - Detects `y[0]`, `y[1]`
2. `test_identify_str_subscripts` - Detects `y["velocity"]`
3. `test_identify_attribute_access` - Detects `constants.damping`
4. `test_track_assignments` - Maps variable names to expressions
5. `test_capture_return` - Stores return node
6. `test_identify_function_calls` - Collects function names
7. `test_validate_consistency_mixed_access` - Raises error for mixed patterns
8. `test_validate_consistency_no_return` - Raises error for missing return
9. `test_get_results` - Returns complete VisitorResults

#### AstToSympyConverter Tests

1. `test_convert_constant_int` - Integer constant conversion
2. `test_convert_constant_float` - Float constant conversion
3. `test_convert_name` - Variable name lookup
4. `test_convert_binop_add` - Addition operator
5. `test_convert_binop_sub` - Subtraction operator
6. `test_convert_binop_mult` - Multiplication operator
7. `test_convert_binop_div` - Division operator
8. `test_convert_binop_pow` - Power operator
9. `test_convert_unaryop_neg` - Negation operator
10. `test_convert_call_sin` - sin() function
11. `test_convert_call_exp` - exp() function
12. `test_convert_call_unknown` - Unknown function raises error
13. `test_convert_ifexp` - Conditional to Piecewise
14. `test_convert_compare_gt` - Greater-than comparison
15. `test_convert_boolop_and` - Logical AND
16. `test_complex_expression` - Nested expression tree

### Parameterized Tests

Use pytest.mark.parametrize for:
- Different binary operators
- Different unary operators
- Different comparison operators
- Different mathematical functions
- Different access patterns

Example:
```python
@pytest.mark.parametrize("op_node,expected_type", [
    (ast.Add(), sp.Add),
    (ast.Sub(), sp.Add),  # Implemented as Add with negation
    (ast.Mult(), sp.Mul),
    (ast.Div(), sp.Mul),  # Implemented as Mul with inverse
    (ast.Pow(), sp.Pow),
])
def test_binop_conversion(op_node, expected_type):
    # Test operator conversion
    ...
```

---

## Dependencies

### Standard Library
- `inspect` - Function introspection
- `ast` - AST parsing and traversal
- `typing` - Type hints (TypedDict, List, Dict, etc.)

### CuBIE Modules
- `cubie.odesystems.symbolic.parsing.parser` - KNOWN_FUNCTIONS, TIME_SYMBOL
- `cubie.odesystems.symbolic.indexedbasemaps` - IndexedBases (used in Section 2)

### External
- `sympy` - Symbolic mathematics

---

## Code Style Notes

### Following CuBIE Conventions

1. **Type hints required** in function signatures
2. **Numpydoc-style docstrings** for all classes and methods
3. **No inline type annotations** in implementation
4. **Max line length 79** characters
5. **Comments max 71** characters
6. **Descriptive variable names**, not abbreviations
7. **Error messages** should be clear and actionable

### Import Style

For these modules (no CUDA device functions):
- Can use `import ast`, `import inspect` (standard library)
- Use `import sympy as sp` (complex module)
- Use `from typing import ...` for specific types

### Example Method Structure

```python
def extract_signature(self) -> inspect.Signature:
    """Extract function signature with parameter metadata.
    
    Returns
    -------
    inspect.Signature
        Signature object containing parameter information.
        
    Raises
    ------
    TypeError
        If function signature cannot be extracted.
        
    Examples
    --------
    >>> def f(t, y, k=0.1): return [-k * y[0]]
    >>> inspector = FunctionInspector(f)
    >>> sig = inspector.extract_signature()
    >>> list(sig.parameters.keys())
    ['t', 'y', 'k']
    """
    # Implementation here
    try:
        signature = inspect.signature(self.func)
    except (ValueError, TypeError) as e:
        raise TypeError(
            f"Cannot extract signature from function "
            f"'{get_function_name(self.func)}': {e}"
        )
    
    return signature
```

---

## Expected Outcomes

After implementing Section 1, the following will be available:

1. **FunctionInspector class** - Robust function metadata extraction
2. **OdeAstVisitor class** - Complete AST pattern detection  
3. **AstToSympyConverter class** - AST to SymPy conversion
4. **Comprehensive validation** - Clear errors for unsupported cases
5. **Test suite** - Full coverage of components

These components provide the foundation for Section 2 (Variable Identification) to build the complete function parser.
