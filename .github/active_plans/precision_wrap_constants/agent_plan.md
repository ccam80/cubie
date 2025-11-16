# Agent Implementation Plan: Precision Wrapping for Numeric Literals

## Architectural Overview

This plan implements automatic precision wrapping for all numeric literals in generated CUDA code. The solution modifies the `CUDAPrinter` class to intercept SymPy numeric types during code generation and wrap them with `precision()` calls.

## Component Descriptions

### CUDAPrinter Extensions

The `CUDAPrinter` class in `numba_cuda_printer.py` will gain three new print methods:

1. **_print_Float(expr: sp.Float) -> str**
   - Handles floating-point literals from SymPy
   - Returns: `f"precision({str(expr)})"`
   - Example: `sp.Float(0.5)` → `"precision(0.5)"`

2. **_print_Integer(expr: sp.Integer) -> str**
   - Handles integer literals from SymPy
   - Returns: `f"precision({str(expr)})"`
   - Example: `sp.Integer(2)` → `"precision(2)"`

3. **_print_Rational(expr: sp.Rational) -> str**
   - Handles rational number literals from SymPy
   - Returns: `f"precision({str(expr)})"`
   - Example: `sp.Rational(1, 2)` → `"precision(1/2)"`

### Expected Behavior

**Input:** SymPy expression containing numeric literals
```python
x = sp.Symbol('x')
expr = x + sp.Float(0.5) * sp.Integer(2)
```

**Current Output:**
```python
"x + 0.5*2"
```

**New Output:**
```python
"x + precision(0.5)*precision(2)"
```

### Integration Points

The new methods integrate with the existing SymPy printer infrastructure:

1. **Visitor Pattern**: SymPy's printer uses dynamic dispatch based on method names
   - `_print_Float` is automatically called for `sp.Float` instances
   - `_print_Integer` is automatically called for `sp.Integer` instances
   - `_print_Rational` is automatically called for `sp.Rational` instances

2. **Composition with Existing Methods**:
   - Works seamlessly with existing `_print_Symbol`, `_print_Function`, `_print_Piecewise`
   - No changes needed to `doprint()` or other infrastructure
   - Power replacement (`x**2` → `x*x`) happens after number printing

3. **Code Generation Pipeline**:
   ```
   ParsedEquations → print_cuda_multiple → CUDAPrinter.doprint → 
   _print_* methods → Generated Code String → Template Insertion
   ```

### Dependencies and Imports

No new imports required. The implementation uses:
- `sp.Float`, `sp.Integer`, `sp.Rational` - already imported via `sympy as sp`
- Built-in `str()` function
- f-string formatting

The `precision()` function is NOT imported in the printer module. It is:
- Passed as a parameter to generated factory functions
- Available in the scope where generated code executes
- Defined in generated code templates or passed from calling code

### Edge Cases to Consider

1. **Nested Expressions**
   - `sp.Float(0.5) + sp.Float(1.0)` → `"precision(0.5) + precision(1.0)"`
   - Recursive printing handles nesting automatically

2. **Rational Numbers**
   - `sp.Rational(1, 2)` prints as `"1/2"` in SymPy
   - Wrapped: `"precision(1/2)"` 
   - Python evaluates `1/2` to float at runtime, then precision() casts it

3. **Large Integers**
   - `sp.Integer(1000000)` → `"precision(1000000)"`
   - No precision loss, precision() handles the cast

4. **Negative Numbers**
   - `sp.Float(-0.5)` → `"precision(-0.5)"`
   - Unary minus is part of the number representation

5. **Scientific Notation**
   - `sp.Float(1.5e-10)` → `"precision(1.5e-10)"`
   - String representation preserves notation

6. **Piecewise Expressions**
   - Numeric literals in conditions and values both get wrapped
   - Example: `Piecewise((0.5, x > 0), (0, True))` →
     `"(precision(0.5) if x > precision(0) else (precision(0)))"`

7. **Constants Dictionary**
   - Already handled by `render_constant_assignments`
   - No conflict: constants use `precision(constants['name'])`, literals use `precision(value)`

### Expected Interactions

**With CellML Parser:**
- CellML parser creates `sp.Float()` for numeric values
- Quantity objects already converted to Float in cellml.py
- No changes needed to cellml.py

**With User-Supplied Equations:**
- Parser converts string equations to SymPy expressions
- String `"0.5"` becomes `sp.Float(0.5)` during parsing
- Wrapping happens transparently during code generation

**With Existing Code Generation:**
- dxdt.py, jacobian.py, etc. all use `print_cuda_multiple`
- They pass expressions to CUDAPrinter
- New behavior applies automatically

**With Template System:**
- Templates inject `precision` parameter into factory functions
- Generated code has `precision` in scope
- `precision(literal)` calls work at runtime

**With Different Precisions:**
- At compile time: all literals become `precision(value)` strings
- At runtime: `precision()` function casts to configured dtype
- Same generated code works for float16, float32, float64

### Data Structures

No new data structures required. The implementation operates purely on:
- Input: SymPy expression nodes (Float, Integer, Rational)
- Output: Python code strings with precision wrapping

### Architectural Changes

**Minimal Impact:**
- No changes to class hierarchy
- No new files or modules
- No changes to public API
- No changes to compilation pipeline

**Localized Change:**
- Only affects `CUDAPrinter` class
- Three new methods (~10 lines total)
- All downstream code benefits automatically

## Testing Strategy

### Unit Tests (test_cuda_printer.py)

Add test class `TestNumericPrecisionWrapping`:

1. **test_print_float_wrapped**
   - Input: `sp.Float(0.5)`
   - Expected: Contains `"precision(0.5)"`

2. **test_print_integer_wrapped**
   - Input: `sp.Integer(2)`
   - Expected: Contains `"precision(2)"`

3. **test_print_rational_wrapped**
   - Input: `sp.Rational(1, 2)`
   - Expected: Contains `"precision(1/2)"`

4. **test_expression_with_literals**
   - Input: `sp.Symbol('x') + sp.Float(0.5) * sp.Integer(2)`
   - Expected: All literals wrapped, expression structure preserved

5. **test_negative_numbers**
   - Input: `sp.Float(-0.5)`
   - Expected: `"precision(-0.5)"`

6. **test_scientific_notation**
   - Input: `sp.Float(1.5e-10)`
   - Expected: `"precision(1.5e-10)"`

7. **test_piecewise_with_literals**
   - Input: Piecewise with numeric literals in conditions and values
   - Expected: All literals wrapped

### Integration Tests (test_cellml.py)

Add test `test_cellml_numeric_literals_wrapped`:

1. Load a CellML model with known numeric literals
2. Inspect generated code string (before compilation)
3. Verify literals are wrapped with `precision()`
4. Verify model still compiles and runs

Add test `test_user_equation_literals_wrapped`:

1. Create SymbolicODE with equation containing magic numbers
2. Inspect generated code
3. Verify wrapping occurred
4. Verify solver runs correctly

### Regression Tests

Run full test suite to ensure:
- All existing CellML tests pass
- All symbolic ODE tests pass
- All solver tests pass
- No performance regression

## Implementation Sequence

This section is for the detailed_implementer agent. The detailed_implementer will create a function-by-function task list based on this plan.

1. Add the three print methods to CUDAPrinter class
2. Write unit tests for the new methods
3. Write integration tests for CellML and user equations
4. Run test suite to verify no regressions
5. Validate generated code compiles and executes correctly

## Validation Criteria

**Success Indicators:**
- All three print methods correctly wrap their respective types
- Unit tests pass
- CellML integration tests pass
- Existing test suite passes without regression
- Generated code contains `precision(literal)` for all numeric literals
- CUDA compilation succeeds with no type warnings
- Numerical results match expected values

**Failure Modes to Avoid:**
- Missing precision() wrapper on any numeric type
- Breaking existing constant wrapping mechanism
- Introducing syntax errors in generated code
- Performance degradation from excessive wrapping
