# SymPy-to-SymPy Input Pathway - Technical Specification

## Component Descriptions

### 1. Input Type Detection in `parse_input()`

The `parse_input()` function in `src/cubie/odesystems/symbolic/parsing/parser.py` must detect whether the `dxdt` parameter contains string equations or SymPy expressions.

**Type Detection Logic:**
- If `dxdt` is a string: Process as multi-line string equations (existing behavior)
- If `dxdt` is an iterable:
  - Inspect the first element
  - If element is `str`: String pathway
  - If element is `sp.Expr`, `sp.Equality`, or tuple of `(sp.Symbol, sp.Expr)`: SymPy pathway
  - Otherwise: Raise informative `TypeError`

**Detection Function:**
```python
def _detect_input_type(dxdt):
    """Detect whether dxdt contains strings or SymPy expressions.
    
    Returns 'string' or 'sympy'
    """
```

### 2. SymPy Expression Processing

New functionality to process SymPy expressions directly without string conversion.

**SymPy Expression Normalization:**
- Accept equations in multiple formats:
  - `sp.Equality` objects (e.g., `sp.Eq(dx, x + y)`)
  - Tuples of `(lhs, rhs)` where both are SymPy expressions
  - For convenience, accept `sp.Expr` and infer LHS from derivative patterns
- Convert all formats to standardized `(lhs_symbol, rhs_expr)` tuples

**Symbol Extraction:**
- Use `free_symbols` property to extract symbols from RHS expressions
- Match against known symbol categories (states, parameters, constants, drivers)
- Identify anonymous auxiliaries from symbols not in known categories

**Left-Hand Side Validation:**
- Validate LHS is either:
  - A derivative symbol (e.g., `sp.Symbol('dx')`)
  - A state symbol (when inferring derivative from assignment)
  - An observable symbol
  - An anonymous auxiliary (when not in declared symbols)
- Apply same validation rules as string-based `_lhs_pass()`

### 3. Unified Processing Pipeline

After type-specific parsing, both string and SymPy pathways converge to a unified processing flow.

**Common Intermediate Representation:**
- List of `(lhs_symbol, rhs_expr)` tuples where both are SymPy objects
- String pathway produces this via `parse_expr()` on RHS
- SymPy pathway produces this via normalization

**Shared Validation:**
- Symbol resolution (verify all RHS symbols are declared or anonymous)
- LHS categorization (states, observables, auxiliaries)
- User function integration (both pathways support same mechanism)

**ParsedEquations Construction:**
- Both pathways use `ParsedEquations.from_equations()` identically
- No changes needed to `ParsedEquations` class itself

### 4. CellML Adapter Simplification

The `load_cellml_model()` function in `src/cubie/odesystems/symbolic/parsing/cellml.py` must be updated to pass SymPy expressions directly.

**Current String Conversion (to be removed):**
- `_eq_to_equality_str()` function (converts SymPy to string with == notation)
- `_replace_eq_in_piecewise()` function (handles Eq in Piecewise)
- String building loop in `load_cellml_model()`

**New SymPy Direct Pathway:**
- Extract equations from `cellmlmanip.load_model()` as SymPy objects
- Substitute `Dummy` symbols with regular `Symbol` objects (existing code)
- Build list of `(lhs_symbol, rhs_expr)` tuples directly
- Pass tuples to `SymbolicODE.create()` without string conversion

**Timing Events:**
- Remove or repurpose `codegen_cellml_string_formatting` timing event
- Add `codegen_cellml_sympy_preparation` for new direct pathway

### 5. Symbol and Equation Extraction Using SymPy Built-ins

Replace manual symbol extraction with SymPy's built-in methods where applicable.

**In SymPy Processing Path:**
- Use `expr.free_symbols` to get all symbols in an expression
- Use `expr.lhs` and `expr.rhs` for `sp.Equality` objects
- Use `expr.atoms(sp.Symbol)` for comprehensive symbol collection when needed

**In String Processing Path (unchanged):**
- Continue using `parse_expr()` to convert strings to SymPy
- After conversion, use `free_symbols` for symbol extraction
- No changes to regex-based preprocessing (needed for string sanitization)

**Shared Symbol Resolution:**
- Both pathways resolve symbols against `IndexedBases` collections
- Use `free_symbols` intersection with known symbol sets
- Anonymous auxiliaries are `free_symbols` minus known symbols

## Expected Behavior

### Input Type Detection
- **String input**: Processed as before with no functional changes
- **SymPy input**: Processed directly, skipping string parsing
- **Mixed input**: Detected and rejected with clear error message
- **Empty input**: Handled gracefully (existing behavior)

### Symbol Categories
- **States**: Symbols appearing on LHS as derivatives (`d<name>`)
- **Parameters**: Symbols declared in parameters list and appearing in RHS
- **Constants**: Symbols declared in constants list and appearing in RHS
- **Drivers**: Symbols declared in drivers list and appearing in RHS
- **Observables**: Symbols declared in observables list and appearing on LHS
- **Anonymous Auxiliaries**: Symbols on LHS not in other categories

### Equation Processing
- **Derivative equations**: `dx = f(x, p, ...)` → state derivative
- **Observable assignments**: `obs = g(x, p, ...)` → observable equation
- **Auxiliary assignments**: `aux = h(x, p, ...)` → auxiliary equation
- **Algebraic constraints**: Supported in SymPy input via `sp.Eq` form

### User Function Compatibility
- User functions referenced in string input: Existing behavior (function name matching)
- User functions in SymPy input: Function symbols must be wrapped in `sp.Function`
- Device functions: Compatible with both input types via `is_devfunc()` detection

## Architectural Changes

### New Functions in `parser.py`

1. `_detect_input_type(dxdt)` → `str`
   - Inspects `dxdt` parameter to determine input type
   - Returns `'string'` or `'sympy'`

2. `_normalize_sympy_equations(equations, index_map)` → `list[(sp.Symbol, sp.Expr)]`
   - Converts various SymPy equation formats to standardized tuples
   - Handles `sp.Equality`, tuples, and bare expressions
   - Validates LHS symbols and categorizes them

3. `_extract_symbols_from_sympy(equations, index_map)` → `dict[str, sp.Symbol]`
   - Extracts all symbols from SymPy equations using `free_symbols`
   - Identifies anonymous auxiliaries
   - Validates symbol usage

### Modified Functions in `parser.py`

1. `parse_input()`:
   - Add type detection at the start
   - Branch to string or SymPy processing based on type
   - Converge to unified processing after type-specific parsing
   - Return same outputs as before (backward compatible)

2. `_lhs_pass()`:
   - Maintain current logic for string input
   - May be invoked from SymPy pathway for validation
   - No functional changes to existing behavior

3. `_rhs_pass()`:
   - Maintain current logic for string input
   - May be bypassed by SymPy pathway (already has SymPy expressions)
   - No functional changes to existing behavior

### Modified Functions in `cellml.py`

1. `load_cellml_model()`:
   - Remove string conversion code
   - Build list of SymPy equation tuples directly
   - Pass SymPy equations to `SymbolicODE.create()`
   - Update timing event tracking

2. Remove or archive:
   - `_eq_to_equality_str()` function
   - `_replace_eq_in_piecewise()` function
   - Related string formatting helpers

## Integration Points

### With IndexedBases
- `IndexedBases.from_user_inputs()`: No changes needed
- Symbol category lookup: Works identically for both input types
- Default value mapping: No changes needed

### With ParsedEquations
- `ParsedEquations.from_equations()`: Accepts SymPy tuples from both pathways
- Equation categorization: Uses `IndexedBases` symbol sets, no changes needed
- Iterator behavior: No changes needed

### With SymbolicODE
- `SymbolicODE.create()`: No changes to signature or behavior
- Passes `dxdt` parameter directly to `parse_input()`
- Receives `ParsedEquations` object identically from both pathways

### With Code Generation
- All codegen modules: Receive `ParsedEquations` as before
- No awareness of input type (already working with SymPy)
- No changes needed to any codegen functions

## Data Structures

### Input Format Types

**String Input (existing):**
```python
dxdt = "dx = -k * x\ndy = k * x"
# or
dxdt = ["dx = -k * x", "dy = k * x"]
```

**SymPy Input (new):**
```python
# Format 1: Equality objects
x, y, k = sp.symbols('x y k')
dx, dy = sp.symbols('dx dy')
dxdt = [
    sp.Eq(dx, -k * x),
    sp.Eq(dy, k * x)
]

# Format 2: Tuples
dxdt = [
    (dx, -k * x),
    (dy, k * x)
]

# Format 3: Mixed (tuples with Symbols and Equality for RHS constraints)
dxdt = [
    (dx, -k * x),
    (dy, k * x),
    sp.Eq(y + x, 1)  # Algebraic constraint
]
```

### Normalized Intermediate Form
Both pathways produce:
```python
equations = [
    (sp.Symbol('dx'), sp.Mul(sp.Symbol('k', real=True), sp.Symbol('x', real=True), evaluate=False)),
    (sp.Symbol('dy'), sp.Mul(sp.Symbol('k', real=True), sp.Symbol('x', real=True), evaluate=False))
]
```

## Edge Cases

### Empty Input
- **String**: Empty string or empty list → handled by existing validation
- **SymPy**: Empty list → same validation path

### Single Equation
- **String**: Single-line string or single-element list → works as before
- **SymPy**: Single `sp.Equality` or tuple → normalized to list of one

### Symbol Name Conflicts
- **String**: Invalid Python identifiers sanitized by preprocessing
- **SymPy**: User responsible for valid symbol names (SymPy enforces this)

### User Functions
- **String**: Function name strings matched against `user_functions` dict
- **SymPy**: Function symbols created via `sp.Function('name')` or `user_functions` callables

### Time Symbol
- **String**: `'t'` string automatically becomes `TIME_SYMBOL`
- **SymPy**: `sp.Symbol('t')` recognized and unified with `TIME_SYMBOL`

### Derivative vs State Name Collisions
- **String**: `dx` derivative implies `x` state (validated in `_lhs_pass`)
- **SymPy**: Same validation, `sp.Symbol('dx')` implies `sp.Symbol('x')` exists

### Indexed Variables
- **String**: `x[0]` normalized to `x0` before parsing
- **SymPy**: User must use `sp.Symbol('x0')` or `sp.IndexedBase('x')[0]`
  - If `sp.IndexedBase` encountered, convert to flattened symbols

### Piecewise Functions
- **String**: Converted via `_replace_if()` to `Piecewise` syntax
- **SymPy**: Already `sp.Piecewise` objects, used directly

### Units
- **String**: Units specified separately in parameters
- **SymPy**: Units metadata attached to symbols (if using `sp.physics.units`)
  - Extract units from symbol assumptions if present
  - Otherwise fall back to separate units dictionaries

## Dependencies

### Required SymPy Functionality
- `sp.Symbol`: Basic symbol creation
- `sp.Equality` / `sp.Eq`: Equation representation
- `sp.Expr`: Base class for expressions
- `free_symbols` property: Symbol extraction
- `lhs` and `rhs` properties: Equation decomposition
- `atoms()` method: Deep symbol collection

### No New External Dependencies
- All required SymPy features are already used elsewhere in CuBIE
- No new packages needed
- cellmlmanip remains optional dependency

### Internal CuBIE Dependencies
- `IndexedBases`: Symbol categorization and management
- `ParsedEquations`: Equation container and partitioning
- `hash_system_definition()`: System hashing (may need update for SymPy input)

## Testing Considerations

### Unit Tests for Type Detection
- Test detection with string input (various formats)
- Test detection with SymPy input (various formats)
- Test error handling for invalid/mixed input types

### Integration Tests for SymPy Pathway
- Test simple ODE system via SymPy input
- Test system with observables via SymPy input
- Test system with user functions via SymPy input
- Test algebraic constraints via `sp.Eq` input

### CellML Adapter Tests
- Verify existing CellML tests pass with new implementation
- Add timing comparison test (not for assertion, just measurement)
- Test symbol name sanitization edge cases

### Equivalence Tests
- Same system via string input and SymPy input produces identical results
- Hash values match for equivalent definitions
- Generated code is identical (byte-for-byte)

### Edge Case Tests
- Empty input handling
- Single equation handling
- Symbol name conflicts
- User function integration
- Piecewise expressions
- Indexed variables (if supported)

## Performance Considerations

### Expected Performance Improvements
- CellML adapter: Eliminate string serialization + parsing overhead
- Direct SymPy input: Skip `parse_expr()` call entirely
- Symbol extraction: `free_symbols` is O(n) tree traversal (same as current)

### Performance Neutral Aspects
- Type detection: O(1) inspection of first element
- Normalization: O(n) traversal to standardize format
- Validation: Same symbol resolution logic as before

### No Expected Performance Degradation
- String input pathway: Identical code path to before
- Code generation: Same inputs, same outputs
- Runtime execution: No changes to compiled kernels

## Migration Path

### For Existing Code (No Changes Required)
- All existing string-based input continues to work
- No deprecation warnings
- No API changes

### For CellML Users (Transparent)
- `load_cellml_model()` automatically uses new pathway
- No user-facing changes
- Expect faster loading (but not breaking if same speed)

### For Advanced Users (New Capability)
- Can now pass SymPy expressions directly
- Useful for programmatic ODE construction
- Integrates with other SymPy-based tools

## Future Extensibility

### Additional Input Formats
- Could support `sp.Matrix` for systems of equations
- Could support `sp.IndexedBase` for array-style variables
- Could support other symbolic math libraries (e.g., Mathematica imports)

### Enhanced Symbol Metadata
- Could extract units from SymPy symbol assumptions
- Could preserve symbolic constraints for validation
- Could leverage SymPy's assumption system for type inference

### Optimization Opportunities
- Could cache `free_symbols` results if expressions are reused
- Could parallelize symbol extraction for large systems
- Could optimize away intermediate expression copies
