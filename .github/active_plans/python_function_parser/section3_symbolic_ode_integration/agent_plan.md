# Section 3: Integration with SymbolicODE - Detailed Agent Plan

## Overview

This section details how FunctionParser integrates with the existing symbolic ODE infrastructure. The parser coordinates components from Sections 1 and 2, builds ParsedEquations and IndexedBases matching the expected format, and integrates with parse_input() for seamless routing.

---

## Component 1: Equation Constructor Module

### File: `src/cubie/odesystems/symbolic/parsing/equation_constructor.py`

#### Purpose
Convert AST expressions and variable classifications into SymPy equations for ParsedEquations.

#### Classes and Methods

##### Class: `EquationConstructor`

**Attributes:**
- `indexed_bases: IndexedBases` - Symbol collections from VariableClassifier
- `ast_visitor: AstVisitor` - AST analysis results from Section 1
- `converter: AstToSympyConverter` - AST-to-SymPy converter from Section 1
- `assignments: Dict[str, ast.Expr]` - Assignment statements from function body
- `return_values: List[ast.Expr]` - Return statement elements
- `observable_names: Set[str]` - User-specified observables
- `state_param_name: str` - Name of state parameter (e.g., 'y')

**Methods:**

**`__init__(self, indexed_bases: IndexedBases, ast_visitor: AstVisitor, observable_names: List[str], state_param_name: str)`**
- Store indexed bases and AST visitor results
- Initialize AST-to-SymPy converter with indexed_bases
- Extract assignments and return values from ast_visitor
- Store observable names as set for fast lookup

Expected behavior:
- Accept analyzed AST and symbol collections
- Prepare for equation construction
- Initialize converter with proper symbol substitution mapping

**`build_equations(self) -> ParsedEquations`**
- Main entry point for equation construction
- Coordinate derivative, observable, and auxiliary equation building
- Return ParsedEquations via from_equations() classmethod

Expected behavior:
1. Call `_build_derivative_equations()`
2. Call `_build_observable_equations()`
3. Call `_build_auxiliary_equations()`
4. Combine all equations in topological order
5. Return `ParsedEquations.from_equations(equations, self.indexed_bases)`

**`_build_derivative_equations(self) -> List[Tuple[sp.Symbol, sp.Expr]]`**
- Extract derivative equations from return statement
- Map return values to state derivative symbols

Expected behavior:
```python
# Function: return [dv, dx]
# States: velocity, position
# Output: [(Symbol('dvelocity'), <expr>), (Symbol('dposition'), <expr>)]
```

Implementation steps:
1. Get return values from ast_visitor (list of AST expressions)
2. Get state symbols from indexed_bases.dxdt.ref_map.keys()
3. Check counts match: `len(return_values) == len(state_symbols)`
4. For each (return_expr, state_deriv_symbol):
   - Convert return_expr to SymPy using converter
   - Create tuple (state_deriv_symbol, sympy_expr)
5. Return list of tuples

Ordering:
- Return values are ordered by function return statement
- State derivative symbols ordered by IndexedBases
- Match by position (index)

**`_build_observable_equations(self) -> List[Tuple[sp.Symbol, sp.Expr]]`**
- Extract observable equations from assignments
- Only include assignments to user-specified observable names

Expected behavior:
```python
# User specifies: observables=["kinetic_energy"]
# Function has: kinetic_energy = 0.5 * m * v**2
# Output: [(Symbol('kinetic_energy'), 0.5*m*v**2)]
```

Implementation steps:
1. Get observable symbols from indexed_bases.observables.ref_map
2. For each observable_name in self.observable_names:
   - Check if assignment exists in self.assignments
   - If not, raise ValueError with clear message
   - Convert RHS expression to SymPy
   - Get corresponding symbol from indexed_bases
   - Create tuple (observable_symbol, sympy_expr)
3. Return list of tuples

Validation:
- All user-specified observables must have assignments
- Error if observable not found: `ValueError(f"Observable '{name}' specified but no assignment found in function body")`

**`_build_auxiliary_equations(self) -> List[Tuple[sp.Symbol, sp.Expr]]`**
- Build equations for intermediate variables not in observables or derivatives
- These are helper assignments needed for evaluation but not saved

Expected behavior:
```python
# Function has intermediate: temp = k * x
# Not in observables, not a derivative
# Output: [(Symbol('aux_0'), k*x)]  # Auto-generated auxiliary symbol
```

Implementation steps:
1. Identify auxiliary assignments (not derivatives, not observables)
2. For each auxiliary assignment:
   - Generate auxiliary symbol (or use existing from LHS)
   - Convert RHS to SymPy
   - Create tuple
3. Return list of tuples

Auxiliary handling:
- Auxiliaries are intermediate calculations
- Not saved, just evaluated in dependency order
- Symbol names can be function-local (preserve if useful for debugging)

**`_convert_ast_to_sympy(self, ast_expr: ast.Expr, context: str) -> sp.Expr`**
- Convert AST expression to SymPy with context for error messages
- Wrap AstToSympyConverter with error handling

Expected behavior:
```python
# AST: BinOp(left=Name('v'), op=Mult(), right=Constant(2))
# Output: Symbol('velocity') * 2
```

Implementation:
- Call `self.converter.convert(ast_expr)`
- Catch conversion errors and re-raise with context
- Context examples: "derivative for state 'x'", "observable 'energy'"

**`_get_equation_order(self, equations: List[Tuple[sp.Symbol, sp.Expr]]) -> List[Tuple[sp.Symbol, sp.Expr]]`**
- Order equations to respect dependencies
- Not necessarily full topological sort (preserves function order where possible)

Expected behavior:
- Auxiliaries before they're used in derivatives/observables
- Otherwise preserve function body order
- Return ordered list

Implementation:
- Can use simple dependency tracking
- Or preserve function order (Python evaluates top-to-bottom)
- Ensure no forward references

**Validation:**

Check in build_equations():
- Number of return values matches number of states
- All observables have assignments
- No circular dependencies (would be caught by SymPy/codegen later)

Error messages:
```python
"Function returns {n} values but system has {m} states. "
"Each state requires one derivative expression in the return statement."

"Observable '{name}' specified but no assignment found in function body. "
"Add a line like: {name} = <expression>"

"Failed to convert expression for {context}: {original_error}"
```

---

## Component 2: FunctionParser Class

### File: `src/cubie/odesystems/symbolic/parsing/function_parser.py`

#### Purpose
Main orchestrator that coordinates all function parsing components and produces output compatible with parse_input().

#### Class: `FunctionParser`

**Attributes:**
- `func: Callable` - User-provided function
- `user_states: Optional[...]` - User-specified states
- `user_parameters: Optional[...]` - User-specified parameters
- `user_constants: Optional[...]` - User-specified constants
- `user_observables: Optional[List[str]]` - User-specified observables
- `user_drivers: Optional[...]` - User-specified drivers
- `user_functions: Optional[Dict[str, Callable]]` - User functions for expressions
- `strict: bool` - Strict validation mode
- `units: Dict[str, ...]` - Unit specifications for all symbol types
- `inspector: FunctionInspector` - Section 1 component
- `visitor: AstVisitor` - Section 1 component
- `classifier: VariableClassifier` - Section 2 component
- `constructor: EquationConstructor` - Section 3 component

**Methods:**

**`__init__(self, func: Callable, states=None, parameters=None, constants=None, observables=None, drivers=None, user_functions=None, strict=False, state_units=None, parameter_units=None, constant_units=None, observable_units=None, driver_units=None)`**
- Store all user inputs
- Validate func is callable
- Initialize to None all internal components (built in parse())

Expected behavior:
- Type check: func must be callable
- Store user specifications
- Don't perform parsing in __init__ (defer to parse())

**`parse(self) -> Tuple[IndexedBases, Dict[str, object], Dict[str, Callable], ParsedEquations, str]`**
- Main entry point orchestrating full parsing workflow
- Returns 5-tuple matching parse_input() output format

Expected behavior:
1. Inspect function (Section 1)
2. Analyze AST (Section 1)
3. Classify variables (Section 2)
4. Build IndexedBases (Section 2)
5. Construct equations (Section 3)
6. Build symbol dictionaries
7. Compute hash
8. Return 5-tuple

Implementation:
```python
def parse(self):
    # Section 1: Inspect function
    self.inspector = FunctionInspector(self.func)
    self.inspector.validate_ode_signature()
    
    # Section 1: Analyze AST
    self.visitor = AstVisitor(
        func_def=self.inspector.func_def,
        param_names=self.inspector.param_names
    )
    self.visitor.visit()
    
    # Section 2: Classify variables
    self.classifier = VariableClassifier(
        access_patterns=self.visitor.access_patterns,
        param_names=self.inspector.param_names,
        user_states=self.user_states,
        user_parameters=self.user_parameters,
        user_constants=self.user_constants,
        user_observables=self.user_observables,
        user_drivers=self.user_drivers,
        strict=self.strict
    )
    self.classifier.classify()
    
    # Section 2: Build IndexedBases
    indexed_bases = self.classifier.build_indexed_bases(
        state_units=self.state_units,
        parameter_units=self.parameter_units,
        constant_units=self.constant_units,
        observable_units=self.observable_units,
        driver_units=self.driver_units
    )
    
    # Section 3: Construct equations
    self.constructor = EquationConstructor(
        indexed_bases=indexed_bases,
        ast_visitor=self.visitor,
        observable_names=self.classifier.observable_names,
        state_param_name=self.inspector.param_names[1]  # Second param is state
    )
    parsed_equations = self.constructor.build_equations()
    
    # Build output components
    all_symbols = self._build_symbol_dict(indexed_bases)
    callables_dict = self.user_functions or {}
    fn_hash = self._compute_hash(parsed_equations, indexed_bases)
    
    return (indexed_bases, all_symbols, callables_dict, parsed_equations, fn_hash)
```

**`_build_symbol_dict(self, indexed_bases: IndexedBases) -> Dict[str, object]`**
- Build comprehensive symbol dictionary matching string parser output
- Include all symbols, time symbol, user functions

Expected behavior:
```python
{
    't': TIME_SYMBOL,
    'velocity': Symbol('velocity'),
    'position': Symbol('position'),
    'dvelocity': Symbol('dvelocity'),
    'dposition': Symbol('dposition'),
    'k': Symbol('k'),
    ...
    # User functions if provided
    'my_func': <callable>,
}
```

Implementation:
- Start with indexed_bases.all_symbols.copy()
- Add TIME_SYMBOL under 't'
- Add user_functions if provided
- Return dictionary

**`_compute_hash(self, equations: ParsedEquations, indexed_bases: IndexedBases) -> str`**
- Compute stable hash for system identification
- Use hash_system_definition() from sym_utils

Expected behavior:
- Call `hash_system_definition(equations, indexed_bases.constants.default_values, observable_labels=...)`
- Return hash string
- Hash must be stable for same system definition

**`validate_consistency(self) -> None`**
- Validate user specifications against inferred structure
- Called during parse() after classification

Expected behavior:
- If user provided states, check inferred states match
- If user provided parameters, check they exist in function
- Raise errors for inconsistencies

Validation checks:
- State count consistency
- Observable names exist in function
- Parameter names accessible in function
- No conflicts between parameter/constant categorization

**Error Handling:**

Wrap each phase in try-except to provide context:
```python
try:
    self.inspector = FunctionInspector(self.func)
except Exception as e:
    raise type(e)(f"Error inspecting function: {e}") from e
```

Maintain clear error messages pointing to function structure issues.

---

## Component 3: parse_input() Integration

### File: `src/cubie/odesystems/symbolic/parsing/parser.py` (modified)

#### Modifications to _detect_input_type()

**Current signature:**
```python
def _detect_input_type(dxdt: Union[str, Iterable]) -> str:
```

**Modified to:**
```python
def _detect_input_type(dxdt: Union[str, Iterable, Callable]) -> str:
```

**Add function detection:**
```python
def _detect_input_type(dxdt: Union[str, Iterable, Callable]) -> str:
    """Detect whether dxdt contains strings, SymPy expressions, or callable.
    
    Returns
    -------
    str
        One of 'string', 'sympy', or 'function' indicating input format.
    """
    if dxdt is None:
        raise TypeError("dxdt cannot be None")
    
    # Check if callable first (before iterable check)
    if callable(dxdt):
        return "function"
    
    # ... existing string/sympy detection unchanged ...
```

Place callable check before iterable check (callables may be iterable).

#### Modifications to parse_input()

**Current signature unchanged**, but add callable to type hint:
```python
def parse_input(
    dxdt: Union[str, Iterable[str], Callable],  # Add Callable
    states: Optional[Union[Dict[str, float], Iterable[str]]] = None,
    # ... rest unchanged ...
) -> Tuple[IndexedBases, Dict[str, object], Dict[str, Callable], ParsedEquations, str]:
```

**Add function routing after input_type detection:**
```python
    input_type = _detect_input_type(dxdt)
    
    if input_type == "function":
        from .function_parser import FunctionParser
        
        parser = FunctionParser(
            func=dxdt,
            states=states,
            parameters=parameters,
            constants=constants,
            observables=observables,
            drivers=drivers,
            user_functions=user_functions,
            strict=strict,
            state_units=state_units,
            parameter_units=parameter_units,
            constant_units=constant_units,
            observable_units=observable_units,
            driver_units=driver_units
        )
        
        (index_map, all_symbols, funcs, 
         equation_map, fn_hash) = parser.parse()
        
        # Note: user_function_derivatives not used for function input
        # (derivatives extracted from function definition itself)
        
    elif input_type == "string":
        # ... existing string parsing unchanged ...
        
    elif input_type == "sympy":
        # ... existing sympy parsing unchanged ...
```

**After routing section, common finalization:**
```python
    # Common for all input types (already exists, unchanged)
    for param in new_params:
        index_map.parameters.push(param)
        all_symbols[str(param)] = param
    
    if driver_dict is not None:
        index_map.drivers.set_passthrough_defaults(driver_dict)
    
    # Handle user functions (existing code)
    if user_functions:
        # ... existing user function handling ...
    
    parsed_equations = ParsedEquations.from_equations(equation_map, index_map)
    
    fn_hash = hash_system_definition(
        parsed_equations,
        index_map.constants.default_values,
        observable_labels=index_map.observables.ref_map.keys(),
    )
    
    return index_map, all_symbols, funcs, parsed_equations, fn_hash
```

Note: For function input path, ParsedEquations is already built by FunctionParser, but we maintain consistency by rebuilding from equation_map. Alternatively, accept it directly from parser.

#### Import Changes

At top of parser.py:
```python
from typing import (
    Any,
    Callable,  # Add if not present
    Dict,
    # ... rest ...
)
```

---

## Component 4: Validation Module

### File: `src/cubie/odesystems/symbolic/parsing/function_validator.py`

#### Purpose
Centralized validation logic for function structure and consistency checks.

#### Functions

**`validate_function_signature(func: Callable, inspector: FunctionInspector) -> None`**
- Validate function has acceptable signature for ODE
- Check parameter count and names

Expected behavior:
- At least 2 parameters required
- First parameter conventionally 't'
- Second parameter conventionally 'y' or 'state'
- Warnings for unconventional names

**`validate_return_statement(visitor: AstVisitor, indexed_bases: IndexedBases) -> None`**
- Validate return statement exists and has correct structure
- Check return value count matches state count

Expected behavior:
- Return statement must exist
- Return value count == number of states
- Return values must be expressions (not literals/strings)

**`validate_observables(observable_names: List[str], assignments: Dict[str, ast.Expr]) -> None`**
- Validate all user-specified observables have assignments
- Check observables are not also state derivatives

Expected behavior:
- Each observable name must appear as assignment LHS
- Clear error if observable not found
- Warning if observable name shadows state name

**`validate_variable_access(access_patterns: Dict[str, List[AccessPattern]]) -> None`**
- Validate state access patterns are consistent
- Check no mixed indexing types

Expected behavior:
- All state accesses use same pattern (int/str/attribute)
- Raise error for mixed patterns
- Guidance in error message

**`validate_user_specifications(classifier: VariableClassifier) -> None`**
- Validate user-provided states/parameters/constants match inferred
- Check for conflicts

Expected behavior:
- If user provided states, they should match inferred
- Parameters/constants should be accessible
- No name conflicts between categories

#### Error Message Templates

Create consistent, actionable error messages:

```python
SIGNATURE_ERROR = """
Function signature invalid for ODE system.
Expected: def f(t, y, ...) with at least 2 parameters
Found: {n} parameters: {names}

The first parameter should be time (conventionally 't'),
and the second should be the state vector (conventionally 'y').
"""

RETURN_MISSING = """
No return statement found in function {func_name}.
ODE functions must return derivative expressions for each state.

Example:
    def f(t, y):
        dv = -y[0]
        dx = y[1]
        return [dv, dx]
"""

RETURN_COUNT_MISMATCH = """
Return statement has {n_return} values but system has {n_states} states.
Each state requires exactly one derivative expression.

States identified: {state_names}
Return values: {return_count}
"""

OBSERVABLE_NOT_FOUND = """
Observable '{name}' specified but no assignment found in function body.
Observables must be assigned before the return statement.

Example:
    def f(t, y, constants):
        v = y[0]
        energy = 0.5 * constants.m * v**2  # Observable
        dv = ...
        return [dv]
"""

INCONSISTENT_ACCESS = """
Inconsistent state access pattern detected:
- Found integer indexing: y[{int_example}]
- Found string indexing: y['{str_example}']

Use one pattern consistently throughout the function.
"""
```

---

## Component 5: Testing Strategy

### File: `tests/odesystems/symbolic/parsing/test_function_parser.py`

#### Test Categories

**Equivalence Tests**
- Define same system as function and string
- Compare ParsedEquations output
- Compare IndexedBases structure
- Verify generated CUDA code identical
- Check numerical results match

Test systems:
- Simple linear ODE
- Nonlinear ODE with observables
- System with parameters and constants
- System with drivers

**Function Structure Tests**
- Valid signatures accepted
- Lambda rejection
- Builtin rejection
- Missing return error
- Return count mismatch error
- Observable not found error

**Access Pattern Tests**
- Integer indexing: `y[0]`, `y[1]`
- String indexing: `y["velocity"]`
- Attribute access: `y.velocity`
- Mixed patterns rejected
- Consistent patterns accepted

**Variable Classification Tests**
- States inferred correctly
- Parameters vs constants distinguished
- Observables identified
- Auxiliaries created
- Direct arguments as constants

**Integration Tests**
- Function through create_ODE_system()
- Function through full solve pipeline
- Parameter sweeps with function system
- Observables saved correctly
- Driver integration

**Edge Cases**
- Single-state system
- No constants/parameters
- No observables
- Complex nested expressions
- Conditional expressions (Piecewise)

#### Example Test

```python
def test_simple_function_equivalence():
    """Function-based system produces same output as string-based."""
    
    # Define as function
    def ode_func(t, y):
        v = y[0]
        x = y[1]
        dv = -0.1 * v
        dx = v
        return [dv, dx]
    
    system_func = create_ODE_system(
        dxdt=ode_func,
        states=["velocity", "position"]
    )
    
    # Define as string
    equations_str = [
        "dvelocity = -0.1 * velocity",
        "dposition = velocity"
    ]
    
    system_str = create_ODE_system(
        dxdt=equations_str,
        states=["velocity", "position"]
    )
    
    # Compare ParsedEquations
    assert len(system_func.equations) == len(system_str.equations)
    for eq_func, eq_str in zip(system_func.equations, system_str.equations):
        assert eq_func[0] == eq_str[0]  # LHS symbols match
        assert eq_func[1].equals(eq_str[1])  # RHS expressions equivalent
    
    # Compare IndexedBases structure
    assert system_func.indices.states.length == system_str.indices.states.length
    assert list(system_func.indices.states.symbol_map.keys()) == \
           list(system_str.indices.states.symbol_map.keys())
```

---

## Component 6: Documentation Updates

### Docstring Updates

**create_ODE_system() docstring:**
Add function example and clarify dxdt parameter:

```python
def create_ODE_system(
    dxdt: Union[str, Iterable[str], Callable],  # Updated type hint
    ...
) -> SymbolicODE:
    """Create a :class:`SymbolicODE` from symbolic definitions.

    Parameters
    ----------
    dxdt
        System equations defined as:
        
        - Single string with newline-delimited equations
        - Iterable of equation strings in ``lhs = rhs`` form
        - Iterable of SymPy expressions or equalities
        - **Python function** with signature ``(t, y, ...)`` returning derivatives
        
        For function input, the first parameter should be time (``t``),
        the second should be the state vector (``y``), and remaining
        parameters provide constants/parameters. The function must return
        a list/tuple of derivative expressions matching the number of states.
        
    states
        State labels either as an iterable or as a mapping to default initial
        values. For function input, states are inferred from ``y`` accesses
        (e.g., ``y[0]``, ``y["velocity"]``, or ``y.velocity``).
        
    observables
        Observable variable labels to expose from the generated system.
        For function input, these should be assigned in the function body
        before the return statement.
        
    parameters
        Parameter labels either as an iterable or as a mapping to default
        values. For function input, distinguish from constants by providing
        this list explicitly.
        
    constants
        Constant labels either as an iterable or as a mapping to default
        values. For function input, accessed from function parameters or
        via attribute/subscript on constant argument.
        
    # ... rest of parameters unchanged ...
    
    Returns
    -------
    SymbolicODE
        Fully constructed symbolic system ready for compilation.
    
    Examples
    --------
    **String-based definition:**
    
    >>> system = create_ODE_system(
    ...     dxdt=[
    ...         "dvelocity = -damping * velocity",
    ...         "dposition = velocity"
    ...     ],
    ...     states=["velocity", "position"],
    ...     constants={"damping": 0.1}
    ... )
    
    **Function-based definition:**
    
    >>> def my_ode(t, y, constants):
    ...     v = y["velocity"]
    ...     x = y["position"]
    ...     k = constants.damping
    ...     return {"velocity": -k * v, "position": v}
    >>> 
    >>> system = create_ODE_system(
    ...     dxdt=my_ode,
    ...     states={"velocity": 1.0, "position": 0.0},
    ...     constants={"damping": 0.1}
    ... )
    
    **With observables:**
    
    >>> def ode_with_obs(t, y, m):
    ...     v = y[0]
    ...     x = y[1]
    ...     
    ...     # Observable calculation
    ...     kinetic_energy = 0.5 * m * v**2
    ...     
    ...     # Derivatives
    ...     dv = -0.1 * v / m
    ...     dx = v
    ...     return [dv, dx]
    >>> 
    >>> system = create_ODE_system(
    ...     dxdt=ode_with_obs,
    ...     states=["velocity", "position"],
    ...     parameters={"m": 1.0},
    ...     observables=["kinetic_energy"]
    ... )
    """
```

### User Guide Section

Add new documentation page: `docs/user_guide/function_based_odes.rst`

Content outline:
1. Introduction to function-based ODE definition
2. Function signature requirements
3. State access patterns (int/str/attribute)
4. Constants and parameters
5. Observables
6. Return statement structure
7. Comparison with string-based approach
8. When to use functions vs strings
9. Common errors and solutions
10. Advanced examples

---

## Expected Outputs

After implementation, the following workflow succeeds:

```python
from cubie import create_ODE_system, solve_ivp
import numpy as np

# Define ODE as Python function
def damped_oscillator(t, y, k, m):
    """Damped harmonic oscillator."""
    v = y[0]  # velocity
    x = y[1]  # position
    
    # Observable
    kinetic_energy = 0.5 * m * v**2
    
    # Derivatives
    dv = -k * v / m
    dx = v
    
    return [dv, dx]

# Create system
system = create_ODE_system(
    dxdt=damped_oscillator,
    states=["velocity", "position"],
    parameters={"k": 0.1, "m": 1.0},
    observables=["kinetic_energy"],
    precision=np.float32
)

# Solve
result = solve_ivp(
    system,
    initial_values=[1.0, 0.0],
    t_span=(0, 10),
    parameters={"k": np.array([0.05, 0.1, 0.2])}
)

# Result contains trajectories and observables
print(result.states.shape)  # (3, n_steps, 2)
print(result.observables.shape)  # (3, n_steps, 1)
```

## Dependencies Between Components

```
EquationConstructor
  ├─ IndexedBases (from VariableClassifier, Section 2)
  ├─ AstVisitor (from Section 1)
  ├─ AstToSympyConverter (from Section 1)
  └─ ParsedEquations (existing)

FunctionParser
  ├─ FunctionInspector (Section 1)
  ├─ AstVisitor (Section 1)
  ├─ VariableClassifier (Section 2)
  ├─ EquationConstructor (Section 3)
  ├─ IndexedBases (existing)
  ├─ ParsedEquations (existing)
  └─ hash_system_definition (existing)

parse_input() modifications
  ├─ FunctionParser (Section 3)
  └─ _detect_input_type() modification

FunctionValidator
  └─ Standalone validation utilities

Tests
  ├─ create_ODE_system (existing)
  ├─ FunctionParser (Section 3)
  └─ All Section 1 & 2 components
```

## Implementation Order

1. **EquationConstructor** - Core equation building logic
2. **FunctionValidator** - Validation utilities
3. **FunctionParser** - Main orchestrator
4. **parse_input() modifications** - Integration point
5. **Tests** - Comprehensive test suite
6. **Documentation** - Docstrings and user guide

Each component should be implemented and tested before moving to the next.
