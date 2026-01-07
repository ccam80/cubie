# Agent Plan: Derivative Notation Fix

## Problem Statement

The parser in `src/cubie/odesystems/symbolic/parsing/parser.py` uses a naive
heuristic to identify derivative equations: any LHS symbol starting with "d"
is treated as a derivative. This causes false positives where auxiliary
variables like `delta_i` are misinterpreted as `d(elta_i)/dt`.

**Current problematic logic (lines 1055-1084 in `_lhs_pass`):**
```python
if lhs.startswith("d"):
    state_name = lhs[1:]
    # ... proceeds to treat this as a derivative even if state_name doesn't exist
```

## Solution Architecture

### Core Change: State-Aware Derivative Detection

The fundamental fix is to change the derivative detection from:
- **Before:** "Does LHS start with 'd'?" → treat as derivative
- **After:** "Does LHS start with 'd' AND is the remainder a known state?" → treat as derivative

This requires restructuring the LHS validation logic to:
1. First check if the symbol (without 'd' prefix) exists in declared states
2. Only then classify it as a derivative
3. If not a known state derivative, fall through to auxiliary handling

### Secondary Feature: Function Notation Support

Add support for explicit `d(x, t)` notation as an unambiguous alternative:
- Parse `d(x, t) = expr` as derivative of state `x` with respect to time `t`
- This provides an escape hatch when users want explicit clarity
- Uses regex pattern matching before the d-prefix heuristic

## Component Descriptions

### 1. Derivative Detection Regex Pattern

A new regex pattern to detect function-notation derivatives:

```
Pattern: d\s*\(\s*([A-Za-z_]\w*)\s*,\s*t\s*\)
```

This matches:
- `d(x, t)` - basic form
- `d( x , t )` - with whitespace
- `d(velocity, t)` - longer variable names

The captured group contains the state variable name.

### 2. Modified `_lhs_pass` Function

The function processes LHS symbols in this priority order:

1. **Function notation check:** Does LHS match `d(name, t)` pattern?
   - If yes: extract `name`, validate it's a state, proceed as derivative
   
2. **State-aware d-prefix check:** Does LHS start with 'd' AND is `lhs[1:]` a known state?
   - If yes: proceed as derivative
   - If no: fall through to auxiliary/observable handling
   
3. **Direct state assignment check:** Is LHS a state name without 'd' prefix?
   - If yes: raise error (states can't be directly assigned)
   
4. **Immutable input check:** Is LHS a parameter/constant/driver?
   - If yes: raise error (immutables can't be assigned)
   
5. **Default:** Treat as observable or anonymous auxiliary

### 3. Modified `_lhs_pass_sympy` Function

Parallel changes for SymPy input pathway:

1. **Derivative LHS check:** Is LHS a `sp.Derivative` object?
   - Already handled correctly - extract state from `Derivative(x, t)`
   
2. **Symbol LHS with d-prefix:** Does Symbol name start with 'd'?
   - Only treat as derivative if remainder is a known state
   - Otherwise, treat as auxiliary

### 4. Warning System for Ambiguous Cases

When a symbol like `delta` could be interpreted as either:
- Derivative of state `elta` (if `elta` exists)
- Auxiliary named `delta`

Issue a warning if the user's intent is unclear. However, the rule is clear:
- If `elta` is a declared state, `delta` is its derivative
- If `elta` is not declared, `delta` is an auxiliary

In non-strict mode, we may need to defer some decisions until RHS analysis
reveals what symbols are being used.

## Integration Points

### Parser Entry Point: `parse_input`

The main entry point remains unchanged in signature. Internal routing to
`_lhs_pass` vs `_lhs_pass_sympy` already exists based on input type detection.

### State Registration: `IndexedBases`

The `indexed_bases.state_names` set provides the list of declared states.
This is already available when `_lhs_pass` is called, making the state-aware
check straightforward.

### CellML Import: `load_cellml_model`

Uses the SymPy pathway with `Derivative(x, t)` notation from cellmlmanip.
No changes needed - already correctly handled.

### String Normalization: `_normalise_indexed_tokens`

Handles array-style access like `dx[0]`. Should work correctly with the new
logic since `x[0]` → `x0` normalization happens before LHS pass.

## Expected Behavior Changes

### Before (Buggy)

| Input | Current Behavior |
|-------|------------------|
| `dx = -k*x` (x is state) | ✅ Derivative of x |
| `delta_i = x + y` | ❌ Derivative of `elta_i` (phantom state created) |
| `done = x + y` | ❌ Derivative of `one` (if `one` is a state) |

### After (Fixed)

| Input | New Behavior |
|-------|--------------|
| `dx = -k*x` (x is state) | ✅ Derivative of x |
| `delta_i = x + y` (elta_i not state) | ✅ Auxiliary variable |
| `done = x + y` (one is state) | ✅ Derivative of `one` |
| `d(x, t) = -k*x` | ✅ Explicit derivative of x |

## Edge Cases

### Non-Strict Mode State Inference

In non-strict mode, states can be inferred from derivative usage. The new
logic must handle:

1. If `dx = ...` is seen and `x` is not yet declared:
   - Infer `x` as a new state (existing behavior)
   - Add `x` to `state_names` for subsequent checks

2. If `delta = ...` is seen and `elta` is not declared:
   - Do NOT infer `elta` as a state
   - Treat `delta` as an auxiliary
   - If `elta` appears on RHS later, it becomes a parameter (existing behavior)

### Observable-to-State Conversion

If `y` is declared as an observable but `dy = ...` appears:
- Current behavior: warn and convert `y` from observable to state
- New behavior: same, since `y` exists in observable_names which we can check

### Single-Letter States

States like `x`, `y`, `z` work correctly:
- `dx = ...` matches state `x`
- `dz = ...` matches state `z`
- `da = ...` where `a` is a parameter: treated as auxiliary (not error)

## Data Structures

### Existing Structures (No Changes)

- `IndexedBases`: Contains `state_names`, `observable_names`, etc.
- `ParsedEquations`: Container for categorized equations
- `anonymous_auxiliaries`: Dict of inferred auxiliary symbols

### New Regex Pattern (Addition)

```python
# Pattern for d(variable, t) function notation
_DERIVATIVE_FUNC_PATTERN = re.compile(
    r"^d\s*\(\s*([A-Za-z_]\w*)\s*,\s*t\s*\)$"
)
```

## Dependencies

### Python Standard Library
- `re`: Already imported for regex patterns

### SymPy
- `sp.Symbol`: Already used throughout
- `sp.Derivative`: Already handled in SymPy pathway

### Internal Dependencies
- `IndexedBases.state_names`: Required for state validation
- `EquationWarning`: Already exists for user warnings

## Test Strategy

### Unit Tests for `_lhs_pass`

1. **Basic derivative with declared state**
   - Input: `dx = ...` with state `x` declared
   - Expected: Treated as derivative

2. **Ambiguous prefix NOT a state**
   - Input: `delta_i = ...` with no state `elta_i`
   - Expected: Treated as auxiliary

3. **Ambiguous prefix IS a state**
   - Input: `delta = ...` with state `elta` declared
   - Expected: Treated as derivative

4. **Function notation with declared state**
   - Input: `d(x, t) = ...` with state `x` declared
   - Expected: Treated as derivative

5. **Function notation with undeclared state (strict)**
   - Input: `d(x, t) = ...` with no state `x`
   - Expected: Error in strict mode, infer state in non-strict

6. **Non-strict mode state inference**
   - Input: `dx = ...` with no state `x`
   - Expected: State `x` inferred, `dx` is its derivative

7. **Non-strict mode auxiliary inference**
   - Input: `delta = ...` with no state `elta`
   - Expected: `delta` treated as auxiliary, NOT inferring `elta` as state

### Integration Tests

1. **Full system with mixed notation**
   - States: `x`, `y`
   - Equations: `dx = -y`, `dy = x`, `delta = x + y`
   - Expected: 2 state derivatives, 1 auxiliary

2. **CellML-style SymPy input**
   - Input: `[Eq(Derivative(x, t), -k*x)]`
   - Expected: Works as before

## Backwards Compatibility

### Guaranteed Compatible

- All existing systems with properly declared states
- All CellML imports (use Derivative notation)
- All SymPy input using `dx` symbols or `Derivative`

### Behavior Change (Bug Fix)

- Systems that accidentally created phantom states from d-prefixed auxiliaries
  will now correctly treat those as auxiliaries
- This is the intended fix; any code relying on the buggy behavior was incorrect

## Implementation Order

1. Add `_DERIVATIVE_FUNC_PATTERN` regex constant
2. Modify `_lhs_pass` with state-aware logic
3. Modify `_lhs_pass_sympy` with parallel changes
4. Add/update tests for new behavior
5. Update docstrings with notation documentation
