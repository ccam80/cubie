# Section 2: Parsing and Identifying Variables - Detailed Agent Plan

## Overview

This plan details the implementation of variable identification and classification components that bridge AST analysis (Section 1) with SymPy symbol creation for CuBIE's ODE system. The components take access patterns and generate IndexedBases with proper categorization.

---

## Component 1: Name Generator Module

### File: `src/cubie/odesystems/symbolic/parsing/name_generator.py`

#### Purpose
Convert access patterns from AST analysis into variable names following CuBIE conventions.

#### Classes and Methods

##### Class: `NameGenerator`

**Attributes:**
- `state_param_name: str` - Name of state parameter (e.g., 'y')
- `access_patterns: List[AccessPattern]` - Access patterns from AST visitor
- `_name_cache: Dict[Tuple, str]` - Memoization cache

**Methods:**

**`__init__(self, state_param_name: str)`**
- Store state parameter name
- Initialize empty cache

Expected behavior:
- Accept parameter name (typically 'y')
- Prepare for name generation

**`generate_state_names(self, state_accesses: List[AccessPattern]) -> List[str]`**
- Convert state access patterns to ordered state names
- Validate consistency of access patterns
- Return ordered list of unique state names

Expected behavior:
```python
# Input: [AccessPattern(base='y', key=0, pattern_type='subscript_int'), ...]
# Output: ['y_0', 'y_1', 'y_2']

# Input: [AccessPattern(base='y', key='velocity', pattern_type='subscript_str'), ...]
# Output: ['velocity', 'position']

# Input: [AccessPattern(base='y', key='velocity', pattern_type='attribute'), ...]
# Output: ['velocity', 'position']
```

Implementation steps:
1. Check if all patterns use same type (int, str, or attribute)
2. Raise error if mixed patterns detected
3. Extract keys/indices in order of first appearance
4. Generate names based on pattern type
5. Remove duplicates while preserving order
6. Return ordered list

**`generate_constant_names(self, constant_accesses: List[AccessPattern], direct_args: List[str]) -> List[str]`**
- Extract constant names from access patterns and direct arguments
- Return unique constant names

Expected behavior:
```python
# From: constants.damping, constants["mass"], direct arg 'k'
# Output: ['damping', 'mass', 'k']
```

Implementation steps:
1. Extract names from attribute accesses
2. Extract names from string subscript accesses
3. Add direct argument names
4. Remove duplicates preserving order
5. Return list

**`_generate_int_subscript_names(self, accesses: List[AccessPattern]) -> List[str]`**
- Generate names for integer subscript pattern
- Use `{base}_{index}` format

Expected behavior:
```python
# y[0], y[1], y[2] → ['y_0', 'y_1', 'y_2']
```

**`_generate_str_subscript_names(self, accesses: List[AccessPattern]) -> List[str]`**
- Extract string keys and use directly as names
- Preserve order of first appearance

Expected behavior:
```python
# y["velocity"], y["position"] → ['velocity', 'position']
```

**`_generate_attribute_names(self, accesses: List[AccessPattern]) -> List[str]`**
- Extract attribute names and use directly
- Preserve order of first appearance

Expected behavior:
```python
# y.velocity, y.position → ['velocity', 'position']
```

**`validate_consistency(self, accesses: List[AccessPattern]) -> None`**
- Check all accesses use same pattern type
- Raise ValueError if mixed patterns detected

Expected behavior:
- All subscript_int: pass
- All subscript_str: pass
- All attribute: pass
- Mixed int and str: `ValueError("Inconsistent state access: found both y[0] and y['name']. Use one pattern consistently.")`

#### Utility Functions

**`_deduplicate_preserving_order(names: List[str]) -> List[str]`**
- Remove duplicates while maintaining first appearance order

Expected behavior:
```python
['a', 'b', 'a', 'c'] → ['a', 'b', 'c']
```

**`_extract_pattern_types(accesses: List[AccessPattern]) -> Set[str]`**
- Return set of pattern_type values from access patterns

Expected behavior:
```python
# Input: accesses with pattern_type='subscript_int' and 'subscript_str'
# Output: {'subscript_int', 'subscript_str'}
```

#### Error Messages

```python
"Inconsistent state access pattern detected: found both integer indexing "
"(y[{index}]) and string indexing (y['{key}']). Use one pattern consistently "
"throughout the function."

"Inconsistent state access pattern detected: found both attribute access "
"(y.{attr}) and subscript access (y[...]). Use one pattern consistently."

"No state variables identified. The state parameter '{param}' is never "
"accessed in the function body. Expected at least one access like {param}[0] "
"or {param}['name']."
```

---

## Component 2: Variable Classifier Module

### File: `src/cubie/odesystems/symbolic/parsing/variable_classifier.py`

#### Purpose
Categorize identified variables into states, constants, parameters, observables based on user specifications.

#### Data Structures

**UserSpecifications TypedDict:**
```python
class UserSpecifications(TypedDict, total=False):
    states: Optional[Union[List[str], Dict[str, float]]]
    parameters: Optional[Union[List[str], Dict[str, float]]]
    constants: Optional[Union[List[str], Dict[str, float]]]
    observables: Optional[List[str]]
```

**ClassifiedVariables attrs class:**
```python
@attrs.define
class ClassifiedVariables:
    """Variables categorized by role in ODE system."""
    states: List[str]
    parameters: List[str]
    constants: List[str]
    observables: List[str]
    state_defaults: Dict[str, float]
    parameter_defaults: Dict[str, float]
    constant_defaults: Dict[str, float]
```

#### Classes and Methods

##### Class: `VariableClassifier`

**Attributes:**
- `inferred_states: List[str]` - State names from AST analysis
- `inferred_constants: List[str]` - Constant names from AST/args
- `user_specs: UserSpecifications` - User-provided overrides
- `assignments: Dict[str, ast.expr]` - Variable assignments from AST

**Methods:**

**`__init__(self, inferred_states: List[str], inferred_constants: List[str], user_specs: UserSpecifications, assignments: Dict[str, ast.expr])`**
- Store inferred variables and user specifications
- Initialize classification structures

Expected behavior:
- Accept inferred variable lists
- Accept user specifications (may be empty)
- Store for classification process

**`classify(self) -> ClassifiedVariables`**
- Main classification workflow
- Apply user overrides to inferred variables
- Validate completeness and consistency
- Return categorized variables

Expected behavior:
1. Start with inferred states as states
2. Start with inferred constants as constants
3. Apply user state override if provided
4. Apply parameter promotion (move from constants to parameters)
5. Extract observable assignments
6. Validate all user-specified names exist
7. Collect default values where provided
8. Return ClassifiedVariables object

**`_apply_state_override(self) -> List[str]`**
- If user provided states, use those instead of inferred
- Validate user states match function structure

Expected behavior:
- User provided states: use user list
- User provided dict: use keys as states
- No user override: use inferred states
- Warn if user states don't match inferred count

**`_promote_parameters(self, constants: List[str]) -> Tuple[List[str], List[str]]`**
- Split constants into parameters and remaining constants
- Based on user's parameters specification

Expected behavior:
```python
# constants = ['k', 'm', 'g']
# user_specs['parameters'] = ['k', 'm']
# Returns: parameters=['k', 'm'], constants=['g']
```

Validation:
- All parameter names must exist in constants
- Error if parameter name not found

**`_extract_observables(self) -> List[str]`**
- Validate user-specified observables are assigned in function
- Return validated observable list

Expected behavior:
- Check each observable name in assignments dict
- Raise error if observable not assigned
- Return observable list

**`_collect_defaults(self) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]`**
- Extract default values from user specifications
- Return state, parameter, constant default dicts

Expected behavior:
```python
# user_specs['states'] = {'velocity': 1.0, 'position': 0.0}
# user_specs['parameters'] = {'k': 0.1}
# user_specs['constants'] = {'g': 9.81}
# Returns: ({'velocity': 1.0, 'position': 0.0}, {'k': 0.1}, {'g': 9.81})
```

**`validate(self, classified: ClassifiedVariables) -> None`**
- Final validation of classified variables
- Check no category overlaps
- Check all user specs satisfied
- Check return count matches state count (if return analyzed)

Expected behavior:
- No variable in multiple categories
- All user-specified parameters found
- All user-specified observables found
- State count matches derivative count

#### Error Messages

```python
"Parameter '{name}' specified but not found in function. "
"Available constants: {available_names}. "
"Ensure the parameter is either a function argument or accessed from a "
"constant argument."

"Observable '{name}' specified but never assigned in function body. "
"Add an assignment like: {name} = expression"

"State name '{name}' appears in both user-specified states and as a "
"constant/parameter. Choose unique names or remove from one category."

"Function returns {n} values but {m} states were identified. "
"Ensure return statement has one value per state."
```

---

## Component 3: Symbol Manager Module

### File: `src/cubie/odesystems/symbolic/parsing/symbol_manager.py`

#### Purpose
Create IndexedBases and symbol mappings from classified variables for SymPy equation building.

#### Classes and Methods

##### Class: `SymbolManager`

**Attributes:**
- `classified_vars: ClassifiedVariables` - Categorized variables
- `time_symbol: sp.Symbol` - Time variable (from parser.TIME_SYMBOL)
- `indexed_bases: IndexedBases` - Generated symbol structure
- `symbol_map: Dict[str, sp.Symbol]` - Variable name → symbol mapping
- `derivative_map: Dict[str, sp.Symbol]` - State → derivative symbol mapping

**Methods:**

**`__init__(self, classified_vars: ClassifiedVariables, time_symbol: sp.Symbol = TIME_SYMBOL)`**
- Store classified variables
- Store time symbol
- Initialize empty structures

Expected behavior:
- Accept ClassifiedVariables from classifier
- Use default TIME_SYMBOL unless overridden
- Prepare for symbol creation

**`create_indexed_bases(self, drivers: Optional[List[str]] = None, unit_specs: Optional[Dict] = None) -> IndexedBases`**
- Create IndexedBases using `IndexedBases.from_user_inputs()`
- Use classified variables as input
- Handle default values

Expected behavior:
```python
# From ClassifiedVariables:
# states=['velocity', 'position']
# parameters=['k']
# constants=['m', 'g']
# observables=['kinetic_energy']

indexed_bases = IndexedBases.from_user_inputs(
    states=classified_vars.states,  # or state_defaults dict
    parameters=classified_vars.parameters,  # or param_defaults dict
    constants=classified_vars.constants,  # or const_defaults dict
    observables=classified_vars.observables,
    drivers=drivers or [],
)
```

Store result in `self.indexed_bases`

**`build_symbol_map(self, assignments: Dict[str, ast.expr]) -> Dict[str, sp.Symbol]`**
- Create mapping from function variable names to SymPy symbols
- Map state accesses, constant accesses, intermediates

Expected behavior:
```python
# Build mapping for all variables that appear in function
{
    't': TIME_SYMBOL,
    'v': indexed_bases.states.symbol_map['velocity'],
    'x': indexed_bases.states.symbol_map['position'],
    'k': indexed_bases.parameters.symbol_map['k'],
    'm': indexed_bases.constants.symbol_map['m'],
    'g': indexed_bases.constants.symbol_map['g'],
    'dv': indexed_bases.dxdt.symbol_map['dvelocity'],
    'dx': indexed_bases.dxdt.symbol_map['dposition'],
    'kinetic_energy': indexed_bases.observables.symbol_map['kinetic_energy'],
}
```

Include:
1. Time symbol
2. State symbols (from IndexedBases)
3. Parameter symbols
4. Constant symbols
5. Derivative symbols
6. Observable symbols
7. Auxiliary symbols (from assignments)

**`_map_state_accesses(self, access_patterns: List[AccessPattern]) -> Dict[str, sp.Symbol]`**
- Map access patterns to corresponding state symbols
- Return dict of access variable name → symbol

Expected behavior:
```python
# If function has: v = y[0], x = y[1]
# And states are ['velocity', 'position']
{
    'v': indexed_bases.states.symbol_map['velocity'],
    'x': indexed_bases.states.symbol_map['position']
}
```

**`_map_constant_accesses(self, access_patterns: List[AccessPattern]) -> Dict[str, sp.Symbol]`**
- Map constant accesses to parameter or constant symbols
- Check parameters first, then constants

Expected behavior:
```python
# If function has: k = constants.damping
# And 'damping' is in parameters
{
    'k': indexed_bases.parameters.symbol_map['damping']
}
```

**`_create_derivative_symbols(self) -> Dict[str, sp.Symbol]`**
- Create symbols for state derivatives
- Use `d{state}` convention

Expected behavior:
```python
# For states ['velocity', 'position']
{
    'dvelocity': Symbol('dvelocity', real=True),
    'dposition': Symbol('dposition', real=True)
}
```

Already created in IndexedBases.dxdt, just need to access

**`_map_assignment_targets(self, assignments: Dict[str, ast.expr]) -> Dict[str, sp.Symbol]`**
- For assignment targets not yet mapped, create auxiliary symbols
- Track anonymous intermediates

Expected behavior:
```python
# If function has: acc = dv / dt
# And 'acc' not in states/params/consts/obs
{
    'acc': Symbol('acc', real=True)  # Auxiliary
}
```

**`get_derivative_name(self, state_name: str) -> str`**
- Return derivative variable name for given state
- Use `d{state}` convention

Expected behavior:
```python
get_derivative_name('velocity') → 'dvelocity'
get_derivative_name('y_0') → 'dy_0'
```

**`get_symbol_for_name(self, name: str) -> Optional[sp.Symbol]`**
- Look up symbol for given variable name
- Search all categories

Expected behavior:
- Return symbol if found in any category
- Return None if not found
- Used during expression conversion

#### Integration Functions

**`create_symbols_from_function(classified_vars: ClassifiedVariables, assignments: Dict[str, ast.expr], drivers: Optional[List[str]] = None) -> Tuple[IndexedBases, Dict[str, sp.Symbol]]`**
- Convenience function combining IndexedBases creation and symbol mapping
- Returns both structures

Expected behavior:
```python
indexed_bases, symbol_map = create_symbols_from_function(
    classified_vars,
    assignments,
    drivers=['external_force']
)
# indexed_bases ready for ParsedEquations
# symbol_map ready for AST→SymPy conversion
```

---

## Component 4: Variable Identifier Orchestrator

### File: `src/cubie/odesystems/symbolic/parsing/variable_identifier.py`

#### Purpose
Orchestrate the full variable identification workflow, coordinating all components.

#### Classes and Methods

##### Class: `VariableIdentifier`

**Attributes:**
- `visitor_results: VisitorResults` - Output from Section 1 AST visitor
- `func_params: List[str]` - Function parameter names
- `user_specs: UserSpecifications` - User-provided specifications
- `name_generator: NameGenerator` - Name generation component
- `classifier: VariableClassifier` - Classification component
- `symbol_manager: SymbolManager` - Symbol creation component

**Methods:**

**`__init__(self, visitor_results: VisitorResults, func_params: List[str], user_specs: Optional[UserSpecifications] = None)`**
- Store inputs from Section 1
- Initialize user specs (empty dict if not provided)
- Create component instances

Expected behavior:
- Accept AST visitor results
- Accept function parameter list
- Accept optional user specifications
- Initialize components for workflow

**`identify(self) -> Tuple[IndexedBases, Dict[str, sp.Symbol], ClassifiedVariables]`**
- Main workflow orchestration
- Execute all identification steps
- Return complete symbol structures

Expected behavior:
1. Identify state parameter from func_params (position 1)
2. Generate state names from access patterns
3. Generate constant names from access patterns and args
4. Classify variables with user overrides
5. Create symbols via SymbolManager
6. Validate completeness
7. Return IndexedBases, symbol_map, and classified_vars

**`_identify_state_parameter(self) -> str`**
- Extract state parameter name from function params
- Typically second parameter (after time)

Expected behavior:
```python
# func_params = ['t', 'y', 'constants']
# Returns: 'y'
```

Validation:
- Must have at least 2 parameters
- Return parameter at index 1

**`_identify_constant_parameters(self) -> List[str]`**
- Extract constant parameter names
- Parameters after state parameter

Expected behavior:
```python
# func_params = ['t', 'y', 'constants', 'k', 'm']
# Returns: ['constants', 'k', 'm']
```

**`_extract_direct_constant_args(self) -> List[str]`**
- Get direct constant arguments (not from dict/object)
- Filter out state and time parameters

Expected behavior:
```python
# func_params = ['t', 'y', 'k', 'm']
# Returns: ['k', 'm']
```

**`_validate_return_statement(self, state_count: int) -> None`**
- Check return statement has correct number of values
- Matches state count

Expected behavior:
- Parse return node
- Count return values
- Raise error if count != state_count

**`validate_output(self, indexed_bases: IndexedBases, symbol_map: Dict[str, sp.Symbol]) -> None`**
- Final validation of generated structures
- Check completeness and consistency

Expected behavior:
- IndexedBases has all expected categories
- Symbol map covers all variables
- No missing symbols
- Lengths match expectations

#### Workflow

```python
# Example full workflow
visitor_results = ast_visitor.get_results()
user_specs = {
    'states': ['velocity', 'position'],
    'parameters': ['k'],
    'constants': {'g': 9.81},
    'observables': ['kinetic_energy']
}

identifier = VariableIdentifier(
    visitor_results,
    func_params=['t', 'y', 'k', 'g'],
    user_specs=user_specs
)

indexed_bases, symbol_map, classified_vars = identifier.identify()

# indexed_bases ready for equation building
# symbol_map ready for AST→SymPy conversion
# classified_vars available for reference
```

---

## Integration Between Components

### Component Dependencies

```
VariableIdentifier (orchestrator)
  ├─> NameGenerator
  │     ├─ generate_state_names()
  │     └─ generate_constant_names()
  │
  ├─> VariableClassifier
  │     ├─ classify()
  │     └─ validate()
  │
  └─> SymbolManager
        ├─ create_indexed_bases()
        └─ build_symbol_map()
```

### Data Flow

1. **VariableIdentifier receives**:
   - `VisitorResults` from Section 1 (access patterns, assignments, return)
   - Function parameters from FunctionInspector
   - User specifications (optional)

2. **NameGenerator produces**:
   - Ordered state names
   - Constant names

3. **VariableClassifier produces**:
   - ClassifiedVariables (states, parameters, constants, observables)
   - Default value dicts

4. **SymbolManager produces**:
   - IndexedBases (to Section 3 for ParsedEquations)
   - Symbol mapping (to Section 3 for expression conversion)

### Example Complete Flow

```python
# Input function
def my_ode(t, y, constants):
    v = y[0]
    x = y[1]
    k = constants.damping
    
    dv = -k * v
    dx = v
    
    kinetic = 0.5 * v**2
    
    return [dv, dx]

# After Section 1: AST Analysis
visitor_results = {
    'state_accesses': [
        {'base': 'y', 'key': 0, 'pattern_type': 'subscript_int'},
        {'base': 'y', 'key': 1, 'pattern_type': 'subscript_int'}
    ],
    'constant_accesses': [
        {'base': 'constants', 'key': 'damping', 'pattern_type': 'attribute'}
    ],
    'assignments': {
        'v': <AST for y[0]>,
        'x': <AST for y[1]>,
        'k': <AST for constants.damping>,
        'dv': <AST for -k * v>,
        'dx': <AST for v>,
        'kinetic': <AST for 0.5 * v**2>
    },
    'return_node': <AST Return([dv, dx])>
}

# NameGenerator
name_gen = NameGenerator('y')
state_names = name_gen.generate_state_names(visitor_results['state_accesses'])
# → ['y_0', 'y_1']

constant_names = name_gen.generate_constant_names(
    visitor_results['constant_accesses'],
    []  # No direct args
)
# → ['damping']

# VariableClassifier
user_specs = {
    'states': {'velocity': 1.0, 'position': 0.0},
    'parameters': ['damping'],
    'observables': ['kinetic']
}

classifier = VariableClassifier(
    state_names,
    constant_names,
    user_specs,
    visitor_results['assignments']
)

classified = classifier.classify()
# ClassifiedVariables(
#     states=['velocity', 'position'],
#     parameters=['damping'],
#     constants=[],
#     observables=['kinetic'],
#     state_defaults={'velocity': 1.0, 'position': 0.0},
#     parameter_defaults={},
#     constant_defaults={}
# )

# SymbolManager
symbol_mgr = SymbolManager(classified)
indexed_bases = symbol_mgr.create_indexed_bases()
symbol_map = symbol_mgr.build_symbol_map(visitor_results['assignments'])

# symbol_map = {
#     't': TIME_SYMBOL,
#     'v': Symbol('velocity'),
#     'x': Symbol('position'),
#     'k': Symbol('damping'),
#     'dv': Symbol('dvelocity'),
#     'dx': Symbol('dposition'),
#     'kinetic': Symbol('kinetic')
# }

# Output to Section 3
# - indexed_bases: for ParsedEquations creation
# - symbol_map: for AST→SymPy conversion
```

---

## Error Handling Strategy

### Error Categories

**1. Inconsistent Access Patterns**
- Detected in: `NameGenerator.validate_consistency()`
- Example: `y[0]` and `y["name"]` in same function
- Error: Clear description of conflict with line numbers

**2. Missing User-Specified Variables**
- Detected in: `VariableClassifier._promote_parameters()` and `_extract_observables()`
- Example: User specifies parameter 'k' but not in function
- Error: List available variables, suggest corrections

**3. Return Count Mismatch**
- Detected in: `VariableIdentifier._validate_return_statement()`
- Example: 2 states but return has 3 values
- Error: Expected vs actual count

**4. Name Collisions**
- Detected in: `VariableClassifier.validate()`
- Example: Same name in states and parameters
- Error: Identify collision, suggest resolution

**5. No States Found**
- Detected in: `NameGenerator.generate_state_names()`
- Example: State parameter never accessed
- Error: Suggest accessing state parameter

**6. Observable Not Assigned**
- Detected in: `VariableClassifier._extract_observables()`
- Example: User lists observable but no assignment
- Error: Show required assignment pattern

### Error Message Templates

All errors follow format:
```
{ErrorType}: {what_went_wrong}

Found: {specific_issue}
Expected: {correct_approach}

Suggestion: {how_to_fix}
```

Example:
```
ValueError: Inconsistent state access pattern

Found: Both integer indexing (y[0] at line 3) and string indexing 
(y["velocity"] at line 5)

Expected: Consistent access pattern throughout function

Suggestion: Choose one access style:
  - Integer: y[0], y[1], ...
  - String: y["velocity"], y["position"], ...
  - Attribute: y.velocity, y.position, ...
```

---

## Testing Strategy for Section 2

### Test File Structure

Create `tests/odesystems/symbolic/test_name_generator.py`:
- Test NameGenerator class
- Test each access pattern type
- Test consistency validation

Create `tests/odesystems/symbolic/test_variable_classifier.py`:
- Test VariableClassifier class
- Test parameter promotion
- Test observable validation

Create `tests/odesystems/symbolic/test_symbol_manager.py`:
- Test SymbolManager class
- Test IndexedBases creation
- Test symbol mapping

Create `tests/odesystems/symbolic/test_variable_identifier.py`:
- Test VariableIdentifier orchestration
- Integration tests

### Fixtures

```python
# Fixture for access patterns
@pytest.fixture
def int_subscript_patterns():
    return [
        AccessPattern(base='y', key=0, pattern_type='subscript_int', node=...),
        AccessPattern(base='y', key=1, pattern_type='subscript_int', node=...),
    ]

@pytest.fixture
def str_subscript_patterns():
    return [
        AccessPattern(base='y', key='velocity', pattern_type='subscript_str', node=...),
        AccessPattern(base='y', key='position', pattern_type='subscript_str', node=...),
    ]

@pytest.fixture
def mixed_patterns(int_subscript_patterns, str_subscript_patterns):
    return int_subscript_patterns + str_subscript_patterns

@pytest.fixture
def user_specs_basic():
    return {
        'states': ['velocity', 'position'],
        'parameters': ['k'],
        'constants': {'g': 9.81},
        'observables': ['kinetic']
    }
```

### Test Cases

#### NameGenerator Tests

1. `test_int_subscript_names` - y[0], y[1] → y_0, y_1
2. `test_str_subscript_names` - y["v"] → v
3. `test_attribute_names` - y.velocity → velocity
4. `test_mixed_access_error` - Mixed patterns raise error
5. `test_preserve_order` - Order matches first appearance
6. `test_deduplicate_accesses` - y[0] twice → single y_0
7. `test_constant_from_attribute` - constants.k → k
8. `test_constant_from_subscript` - constants["k"] → k
9. `test_direct_arg_constants` - Direct args become constants

#### VariableClassifier Tests

1. `test_classify_with_user_override` - User states override inferred
2. `test_promote_to_parameters` - Move from constants to parameters
3. `test_parameter_not_found_error` - Invalid param name errors
4. `test_observable_validation` - Observable must be assigned
5. `test_collect_defaults` - Extract default values
6. `test_state_count_validation` - Warn on count mismatch
7. `test_no_category_overlap` - Variable in single category only
8. `test_default_all_constants` - Args default to constants

#### SymbolManager Tests

1. `test_create_indexed_bases` - Proper structure
2. `test_build_symbol_map_states` - State symbols mapped
3. `test_build_symbol_map_constants` - Constant symbols mapped
4. `test_build_symbol_map_parameters` - Parameter symbols mapped
5. `test_derivative_symbols` - d{state} for each state
6. `test_observable_symbols` - Observable symbols created
7. `test_auxiliary_symbols` - Auxiliaries tracked
8. `test_time_symbol` - TIME_SYMBOL included

#### VariableIdentifier Integration Tests

1. `test_full_workflow_int_subscripts` - End-to-end with y[i]
2. `test_full_workflow_str_subscripts` - End-to-end with y["name"]
3. `test_full_workflow_attributes` - End-to-end with y.attr
4. `test_with_user_overrides` - User specs applied
5. `test_parameter_promotion` - Constants → parameters
6. `test_observable_extraction` - Observables identified
7. `test_direct_constant_args` - Direct args handled
8. `test_return_validation` - Return count checked

### Parameterized Tests

Use pytest.mark.parametrize for:
- Different access pattern types
- Different user specification combinations
- Different error scenarios

Example:
```python
@pytest.mark.parametrize("pattern_type,expected_names", [
    ('subscript_int', ['y_0', 'y_1', 'y_2']),
    ('subscript_str', ['velocity', 'position', 'acceleration']),
    ('attribute', ['v', 'x', 'a']),
])
def test_name_generation_by_pattern(pattern_type, expected_names):
    # Test name generation for each pattern type
    ...
```

---

## Dependencies

### From Section 1
- `AccessPattern` TypedDict
- `VisitorResults` TypedDict
- `OdeAstVisitor` for testing

### CuBIE Modules
- `cubie.odesystems.symbolic.indexedbasemaps.IndexedBases`
- `cubie.odesystems.symbolic.indexedbasemaps.IndexedBaseMap`
- `cubie.odesystems.symbolic.parsing.parser.TIME_SYMBOL`

### External
- `sympy` - Symbol creation
- `attrs` - Data classes

### Standard Library
- `typing` - Type hints
- `ast` - For assignments dict typing

---

## Code Style Notes

### Following CuBIE Conventions

1. **Type hints** in all function signatures
2. **Numpydoc docstrings** for classes and methods
3. **Max line length 79** characters
4. **Error messages** clear and actionable
5. **attrs classes** for data containers

### Example Method

```python
def generate_state_names(
    self, state_accesses: List[AccessPattern]
) -> List[str]:
    """Generate state variable names from access patterns.
    
    Parameters
    ----------
    state_accesses
        List of state access patterns from AST analysis.
        
    Returns
    -------
    list of str
        Ordered list of unique state variable names.
        
    Raises
    ------
    ValueError
        If access patterns are inconsistent (mixed types).
        
    Examples
    --------
    >>> patterns = [
    ...     AccessPattern(base='y', key=0, pattern_type='subscript_int'),
    ...     AccessPattern(base='y', key=1, pattern_type='subscript_int')
    ... ]
    >>> gen = NameGenerator('y')
    >>> gen.generate_state_names(patterns)
    ['y_0', 'y_1']
    """
    self.validate_consistency(state_accesses)
    
    pattern_types = _extract_pattern_types(state_accesses)
    pattern_type = list(pattern_types)[0]
    
    if pattern_type == 'subscript_int':
        names = self._generate_int_subscript_names(state_accesses)
    elif pattern_type == 'subscript_str':
        names = self._generate_str_subscript_names(state_accesses)
    elif pattern_type == 'attribute':
        names = self._generate_attribute_names(state_accesses)
    else:
        raise ValueError(f"Unknown pattern type: {pattern_type}")
    
    return _deduplicate_preserving_order(names)
```

---

## Expected Outcomes

After implementing Section 2, the following will be available:

1. **NameGenerator class** - Convert access patterns to variable names
2. **VariableClassifier class** - Categorize with user overrides
3. **SymbolManager class** - Create IndexedBases and symbol maps
4. **VariableIdentifier class** - Orchestrate full workflow
5. **Complete validation** - Consistency checks throughout
6. **Test suite** - Full coverage of all components

These components bridge Section 1 (AST analysis) with Section 3 (equation building), providing the variable identification and symbol creation infrastructure needed for the function parser.
