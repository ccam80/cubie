# Codegen Hash Unification - Agent Plan

## Overview

This document provides detailed technical specifications for unifying the codegen hashing mechanism in CuBIE's symbolic ODE system. The goal is to ensure that `hash_system_definition()` produces identical hashes for identical equation sets and constants, regardless of input format (strings vs SymPy objects) or processing path.

---

## Problem Statement

### Current Behavior

The `hash_system_definition()` function in `sym_utils.py` has multiple conditional branches handling different input types:

1. **Lines 167-185**: Handles `(list, tuple)` inputs with various element types
2. **Lines 186-188**: Handles nested list/tuple structures  
3. **Lines 189-190**: Handles string iterables
4. **Lines 191-193**: Handles other iterables (assumed to be `(symbol, expr)` pairs)
5. **Lines 194-195**: Handles plain strings

Each branch constructs the hash string differently. When SymPy processes equations, it may reorder them (e.g., during CSE or topological sorting), causing the hash string to differ from the original input order.

### Call Sites

1. **`parser.parse_input()` line 1378**: For string input path
2. **`parser.parse_input()` line 1420**: For SymPy input path  
3. **`symbolicODE.SymbolicODE.__init__()` lines 180-184**: When `fn_hash` is `None`
4. **`symbolicODE.SymbolicODE.build()` lines 371-373**: To check if system changed

---

## Component Specifications

### Component 1: Canonical Hash String Builder

**Location**: `src/cubie/odesystems/symbolic/sym_utils.py`

**Purpose**: Create a deterministic string representation of an equation set that is independent of input order.

**Behavior**:
- Accept an iterable of `(lhs, rhs)` equation tuples where LHS is a SymPy Symbol and RHS is a SymPy Expr
- Sort equations alphabetically by the string representation of the LHS symbol
- For each equation, generate a string in format `{lhs_name}={rhs_repr}` where `rhs_repr` is the SymPy string representation
- Concatenate all equation strings with a delimiter
- Normalize by removing whitespace

**Key Consideration**: SymPy's `str()` representation of expressions is consistent for identical expressions. The sorting step ensures order-independence.

### Component 2: Simplified `hash_system_definition()`

**Location**: `src/cubie/odesystems/symbolic/sym_utils.py`

**Current Signature**:
```python
def hash_system_definition(
    dxdt: Union[str, Iterable[str]],
    constants: Optional[Union[Dict[str, float], Iterable[str]]] = None,
) -> str
```

**New Signature**:
```python
def hash_system_definition(
    equations: Union[ParsedEquations, Iterable[Tuple[sp.Symbol, sp.Expr]]],
    constants: Optional[Dict[str, float]] = None,
) -> str
```

**Behavior**:
- Accept `ParsedEquations` object directly (use its `ordered` attribute)
- Accept `Iterable[Tuple[Symbol, Expr]]` for simpler testing
- Sort equations by LHS symbol name before building hash string
- Sort constants by key name before building hash string
- Build canonical string and return hash

**Removed**: All branches handling raw strings, lists of strings, or mixed formats. The function operates only on parsed SymPy structures.

### Component 3: Updated `parse_input()` Hash Computation

**Location**: `src/cubie/odesystems/symbolic/parsing/parser.py`

**Current Behavior**:
- Line 1378 (string path): `fn_hash = hash_system_definition(dxdt, constants)` - hashes raw input
- Line 1420 (sympy path): `fn_hash = hash_system_definition(substituted_eqs, constants)` - hashes substituted equations

**New Behavior**:
- Move hash computation to after `ParsedEquations` creation (around line 1486)
- Call `hash_system_definition(parsed_equations, index_map.constants.default_values)`
- Single hash computation point for both pathways

**Integration Points**:
- `ParsedEquations` is created on line 1486
- Hash should use `parsed_equations` and constants from `index_map.constants.default_values`

### Component 4: Updated `SymbolicODE.__init__()`

**Location**: `src/cubie/odesystems/symbolic/symbolicODE.py`

**Current Behavior** (lines 180-184):
```python
if fn_hash is None:
    dxdt_str = [f"{lhs}={str(rhs)}" for lhs, rhs in equations]
    constants = all_indexed_bases.constants.default_values
    fn_hash = hash_system_definition(dxdt_str, constants)
```

**New Behavior**:
- Keep the fallback for when `fn_hash` is None
- Update to use the new canonical `hash_system_definition()` signature
- Accept `equations` (which is `ParsedEquations`) directly

**Change**:
```python
if fn_hash is None:
    constants = all_indexed_bases.constants.default_values
    fn_hash = hash_system_definition(equations, constants)
```

### Component 5: Updated `SymbolicODE.build()`

**Location**: `src/cubie/odesystems/symbolic/symbolicODE.py`

**Current Behavior** (lines 371-376):
```python
new_hash = hash_system_definition(
    self.equations, self.indices.constants.default_values
)
if new_hash != self.fn_hash:
    self.gen_file = ODEFile(self.name, new_hash)
    self.fn_hash = new_hash
```

**New Behavior**:
- Keep the same logic but the function now works correctly with `ParsedEquations`
- No code change needed if `hash_system_definition()` properly handles `ParsedEquations`

---

## Expected Interactions

### Flow 1: String Input Path
```
User provides: ["dx = -k*x", "dy = k*x"]
↓
parse_input() parses strings
↓
ParsedEquations created with ordered equations
↓
hash_system_definition(parsed_equations, constants) called
↓
Equations sorted by LHS name: [(dx, -k*x), (dy, k*x)]
↓
Canonical string: "dx=-k*x|dy=k*x"
↓
Hash returned to caller
```

### Flow 2: SymPy Input Path
```
User provides: [sp.Eq(dy, k*x), sp.Eq(dx, -k*x)]  # Note: different order
↓
parse_input() normalizes SymPy equations
↓
ParsedEquations created with ordered equations
↓
hash_system_definition(parsed_equations, constants) called
↓
Equations sorted by LHS name: [(dx, -k*x), (dy, k*x)]  # Same order as Flow 1
↓
Canonical string: "dx=-k*x|dy=k*x"  # Same string as Flow 1
↓
Same hash returned
```

### Flow 3: Cache Check in build()
```
SymbolicODE.build() called
↓
hash_system_definition(self.equations, constants) called
↓
Same canonical sorting applied
↓
Hash compared with stored fn_hash
↓
If match: use cached code
If mismatch: regenerate (only if constants changed)
```

---

## Data Structures

### Input: `ParsedEquations`

```python
@attrs.define(frozen=True)
class ParsedEquations:
    ordered: Tuple[Tuple[sp.Symbol, sp.Expr], ...]
    state_derivatives: Tuple[Tuple[sp.Symbol, sp.Expr], ...]
    observables: Tuple[Tuple[sp.Symbol, sp.Expr], ...]
    auxiliaries: Tuple[Tuple[sp.Symbol, sp.Expr], ...]
    _state_symbols: frozenset[sp.Symbol]
    _observable_symbols: frozenset[sp.Symbol]
    _auxiliary_symbols: frozenset[sp.Symbol]
```

**Usage**: Iterate over `.ordered` to get all equations in their stored order.

### Input: Constants Dictionary

```python
constants: Dict[str, float]
# Example: {"k": 1.0, "c": 2.5}
```

**Usage**: Sort by key, then format as `key:value` pairs.

### Output: Hash String

A deterministic string representation of the combined equations + constants.

---

## Dependencies

### Required Imports in `sym_utils.py`
- `sympy as sp` (already present)
- `typing.Tuple, typing.Dict, typing.Iterable` (already present)
- `ParsedEquations` from `parsing.parser` (new import, may cause circular import - see Edge Cases)

### Circular Import Mitigation
`ParsedEquations` is defined in `parsing/parser.py` which imports from `sym_utils.py`. To avoid circular imports:
- Use `TYPE_CHECKING` guard for the import
- Accept `Iterable[Tuple[Symbol, Expr]]` as the type hint
- Perform `isinstance(equations, ParsedEquations)` check at runtime using duck typing

---

## Edge Cases

### 1. Empty Equation List
**Input**: `ParsedEquations` with no equations
**Behavior**: Return hash of just the constants portion

### 2. No Constants
**Input**: `None` or empty dict for constants
**Behavior**: Constants portion of hash string is empty

### 3. Observables vs State Derivatives
**Consideration**: The hash should include ALL equations in `ParsedEquations.ordered`, not just state derivatives. Observables affect codegen.

### 4. Auxiliary Variables (CSE Results)
**Consideration**: CSE-generated auxiliaries (e.g., `_cse0`, `_cse1`) must be included in the hash since they affect the generated code structure.

### 5. SymPy Expression Representation Stability
**Risk**: SymPy's `str()` of expressions might vary slightly between versions
**Mitigation**: Test with current SymPy version; accept minor variance risk. Alternative: use `srepr()` for more stable representation.

### 6. Floating Point Constant Values
**Risk**: Float precision in constant values (e.g., `0.1` vs `0.10000000000000001`)
**Mitigation**: Round or format to fixed precision when building constant string

---

## Testing Requirements

### Test 1: String vs SymPy Equivalence
```python
def test_hash_consistency_string_vs_sympy():
    """Verify identical hashes from string and SymPy input."""
    # String input
    string_result = parse_input(
        dxdt=["dx = -k*x", "dy = k*x"],
        states=['x', 'y'],
        parameters=['k']
    )
    
    # SymPy input (different order)
    x, y, k = sp.symbols('x y k')
    dx, dy = sp.symbols('dx dy')
    sympy_result = parse_input(
        dxdt=[sp.Eq(dy, k*x), sp.Eq(dx, -k*x)],  # Reversed order
        states=['x', 'y'],
        parameters=['k']
    )
    
    assert string_result[4] == sympy_result[4]  # fn_hash
```

### Test 2: Order Independence
```python
def test_hash_order_independence():
    """Verify hash is independent of equation input order."""
    hash1 = hash_system_definition([(dx, -k*x), (dy, k*x)], {})
    hash2 = hash_system_definition([(dy, k*x), (dx, -k*x)], {})
    
    assert hash1 == hash2
```

### Test 3: Constant Value Sensitivity
```python
def test_hash_constant_sensitivity():
    """Verify different constant values produce different hashes."""
    eqs = [(dx, -k*x)]
    hash1 = hash_system_definition(eqs, {"c": 1.0})
    hash2 = hash_system_definition(eqs, {"c": 2.0})
    
    assert hash1 != hash2
```

### Test 4: Cache Hit After Reload
```python
def test_cache_hit_after_system_recreation():
    """Verify recreating identical system hits cache."""
    # Create system, trigger codegen
    sys1 = SymbolicODE.create(dxdt=..., ...)
    _ = sys1.dxdt  # Trigger build
    
    # Recreate identical system
    sys2 = SymbolicODE.create(dxdt=..., ...)  # Same args
    
    # Should have same hash
    assert sys1.fn_hash == sys2.fn_hash
```

### Test 5: Existing Tests Still Pass
- `test_parser.py::TestHashSystemDefinition` tests must continue to pass
- Update test assertions if API changes require it

---

## Implementation Notes

### Backward Compatibility
The new `hash_system_definition()` signature is NOT backward compatible with callers passing raw strings. However:
- All calls are internal to CuBIE
- The `parse_input()` function is the primary public entry point
- Direct callers of `hash_system_definition()` are in `parser.py` and `symbolicODE.py`

### Migration Path
1. Update `hash_system_definition()` signature and implementation
2. Update callers in `parser.py` to call after `ParsedEquations` creation
3. Update callers in `symbolicODE.py` to pass equations object
4. Update tests to use new signature
5. Remove old test cases that tested raw string handling

### Code Location Summary
- `src/cubie/odesystems/symbolic/sym_utils.py`: Main function modification
- `src/cubie/odesystems/symbolic/parsing/parser.py`: Move hash call location
- `src/cubie/odesystems/symbolic/symbolicODE.py`: Update __init__ and build
- `tests/odesystems/symbolic/test_parser.py`: Update hash tests
- `tests/odesystems/symbolic/test_sym_utils.py`: Add new hash tests
