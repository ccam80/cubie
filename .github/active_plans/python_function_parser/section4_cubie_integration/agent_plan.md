# Section 4: CuBIE Integration and Cleanup - Detailed Agent Plan

## Overview

This section details the reorganization of the parsing module, integration of FunctionParser with the existing infrastructure, backward compatibility preservation, API design, and documentation updates.

---

## Component 1: Module Reorganization

### Task 1.1: Extract String Parser Module

#### File: `src/cubie/odesystems/symbolic/parsing/string_parser.py`

**Purpose**: Isolate all string-specific parsing logic from `parser.py`

**Functions to Move from parser.py:**

1. **`_sanitise_input_math(expr_str: str) -> str`**
   - Move as-is from parser.py
   - No changes to implementation
   - Used only by string parsing pathway

2. **`_replace_if(expr_str: str) -> str`**
   - Move as-is from parser.py
   - Helper for `_sanitise_input_math`
   - String manipulation for ternary operators

3. **`_normalise_indexed_tokens(lines: Iterable[str]) -> list[str]`**
   - Move as-is from parser.py
   - Converts `x[0]` → `x0` in strings
   - String-specific preprocessing

4. **`_lhs_pass(lines: Sequence[str], indexed_bases: IndexedBases, strict: bool = True) -> Dict[str, sp.Symbol]`**
   - Move as-is from parser.py
   - Validates LHS of string equations
   - Infers state derivatives from d-prefix notation
   - Returns anonymous auxiliary symbols

5. **`_rhs_pass(lines: Sequence[str], all_symbols: Dict[str, object], user_funcs: Optional[Dict[str, Callable]], user_function_derivatives: Optional[Dict[str, Callable]], strict: bool, raw_lines: List[str]) -> Tuple[Dict[str, sp.Expr], Dict[str, Callable], List[sp.Symbol]]`**
   - Move as-is from parser.py
   - Parses RHS of string equations to SymPy
   - Handles function renaming for user functions
   - Returns validated equations, function mapping, and new parameters

6. **`_rename_user_calls(lines: Iterable[str], user_functions: Dict[str, Callable]) -> Tuple[List[str], Dict[str, str]]`**
   - Move as-is from parser.py
   - Renames user function calls in strings (avoids name collisions)
   - Returns modified lines and rename mapping

7. **`_build_sympy_user_functions(user_funcs: Dict[str, Callable], user_function_derivatives: Optional[Dict[str, Callable]]) -> Dict[str, sp.Function]`**
   - Move as-is from parser.py
   - Creates SymPy UndefinedFunction wrappers for user callables
   - Attaches derivative information if provided

**Module Structure:**
```python
"""String-based equation parsing for symbolic ODE systems."""

from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
)
import re
from warnings import warn

import sympy as sp
from sympy.parsing.sympy_parser import T, parse_expr
from sympy.core.function import AppliedUndef

from ..indexedbasemaps import IndexedBases
from .parser import TIME_SYMBOL, _INDEXED_NAME_PATTERN, EquationWarning, PARSE_TRANSORMS

# Constants from parser.py
_func_call_re = re.compile(r"\b([A-Za-z_]\w*)\s*\(")

# All functions listed above
```

**Dependencies:**
- Imports from `parser.py`: `TIME_SYMBOL`, `_INDEXED_NAME_PATTERN`, `EquationWarning`, `PARSE_TRANSORMS`
- Imports from `indexedbasemaps.py`: `IndexedBases`
- Standard library: `re`, `warnings`, `typing`
- SymPy: `sympy`, `parse_expr`, `AppliedUndef`

**Testing Impact:**
- All string parser tests remain in `test_parser.py`
- Update imports to `from cubie.odesystems.symbolic.parsing.string_parser import ...`
- Tests should pass unchanged after extraction

---

### Task 1.2: Extract Common Utilities Module

#### File: `src/cubie/odesystems/symbolic/parsing/common.py`

**Purpose**: Shared utilities used by both StringParser and FunctionParser

**Functions to Move from parser.py:**

1. **`_process_calls(equations: Iterable[str], user_functions: Optional[Dict[str, Callable]] = None) -> Dict[str, Callable]`**
   - Move as-is from parser.py
   - Identifies function calls in equations
   - Maps to SymPy functions or user functions
   - Used by both string and function parsing

2. **`_lhs_pass_sympy(equations: List[Tuple[sp.Symbol, sp.Expr]], indexed_bases: IndexedBases, strict: bool = True) -> Dict[str, sp.Symbol]`**
   - Move as-is from parser.py
   - SymPy-based LHS validation
   - Used after both string and function parsing convert to SymPy
   - Categorizes symbols as derivatives, observables, auxiliaries

3. **`_rhs_pass_sympy(equations: List[Tuple[sp.Symbol, sp.Expr]], all_symbols: Dict[str, object], indexed_bases: IndexedBases, user_funcs: Optional[Dict[str, Callable]], user_function_derivatives: Optional[Dict[str, Callable]], strict: bool) -> Tuple[List[Tuple[sp.Symbol, sp.Expr]], Dict[str, Callable], List[sp.Symbol]]`**
   - Move as-is from parser.py
   - SymPy-based RHS validation
   - Used by both SymPy input and function input paths
   - Validates symbols, processes user functions

4. **`_normalize_sympy_equations(equations: Iterable[Union[sp.Equality, Tuple[sp.Symbol, sp.Expr], sp.Expr]], index_map: IndexedBases) -> List[Tuple[sp.Symbol, sp.Expr]]`**
   - Move as-is from parser.py
   - Normalizes various SymPy formats to (lhs, rhs) tuples
   - Handles `sp.Equality`, derivatives, tuples
   - Used by SymPy input path and potentially function path

5. **`_process_user_functions_for_rhs(user_funcs: Optional[Dict[str, Callable]], user_function_derivatives: Optional[Dict[str, Callable]]) -> Dict[str, sp.Function]`**
   - Move as-is from parser.py (if exists as separate function)
   - Or extract from `_rhs_pass_sympy` if inline
   - Creates SymPy function wrappers for RHS validation

**Constants to Move:**
- `KNOWN_FUNCTIONS`: Mapping of string names to SymPy functions
- Any other shared constants

**Module Structure:**
```python
"""Common utilities for parsing symbolic ODE systems."""

from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

import sympy as sp
from sympy.core.function import AppliedUndef

from ..indexedbasemaps import IndexedBases
from .parser import TIME_SYMBOL

# KNOWN_FUNCTIONS mapping
KNOWN_FUNCTIONS = {
    "sin": sp.sin,
    "cos": sp.cos,
    # ... rest of mapping
}

# All functions listed above
```

**Dependencies:**
- Imports from `parser.py`: `TIME_SYMBOL`
- Imports from `indexedbasemaps.py`: `IndexedBases`
- SymPy: `sympy`, `AppliedUndef`

---

### Task 1.3: Refactor Main Parser Module

#### File: `src/cubie/odesystems/symbolic/parsing/parser.py`

**Purpose**: Main routing logic for `parse_input()`, SymPy equation handling, shared constants

**Content After Refactor:**

1. **Keep in parser.py:**
   - `TIME_SYMBOL` constant
   - `DRIVER_SETTING_KEYS` constant
   - `PARSE_TRANSORMS` constant
   - `_INDEXED_NAME_PATTERN` constant
   - `_DERIVATIVE_FUNC_PATTERN` constant
   - `EquationWarning` class
   - `ParsedEquations` attrs class
   - `_detect_input_type()` function (MODIFIED - see Task 2.1)
   - `_process_parameters()` function
   - `parse_input()` function (MODIFIED - see Task 2.2)

2. **Import from new modules:**
```python
from .string_parser import (
    _lhs_pass,
    _rhs_pass,
    _normalise_indexed_tokens,
    _sanitise_input_math,
    _rename_user_calls,
    _build_sympy_user_functions,
)
from .common import (
    _lhs_pass_sympy,
    _rhs_pass_sympy,
    _normalize_sympy_equations,
    _process_calls,
    KNOWN_FUNCTIONS,
)
from .function_parser import FunctionParser
```

3. **Modified Functions:**
   - `_detect_input_type()` - Add callable detection
   - `parse_input()` - Add function routing

**Structure After Refactor:**
- Lines 1-40: Imports and constants
- Lines 41-103: `_detect_input_type()` (modified)
- Lines 104-400: `_normalize_sympy_equations()` and related (REMOVED - moved to common.py)
- Lines 401-427: `ParsedEquations` class (kept)
- Lines 428-474: Input cleaning functions (REMOVED - moved to string_parser.py)
- Lines 475-713: Function handling (REMOVED - split between string_parser.py and common.py)
- Lines 714-770: `_process_parameters()` (kept)
- Lines 771-1073: LHS/RHS pass functions (REMOVED - moved to string_parser.py and common.py)
- Lines 1074-1380: Remaining string-specific helpers (REMOVED - moved to string_parser.py)
- Lines 1381-1629: `parse_input()` (kept, modified)

**Expected Line Count After Refactor:** ~400 lines (from 1629)

---

### Task 1.4: Update Module Exports

#### File: `src/cubie/odesystems/symbolic/parsing/__init__.py`

**Current Content:**
```python
"""Parsing utilities for symbolic ODE descriptions."""

from .auxiliary_caching import *  # noqa: F401,F403
from .cellml import *  # noqa: F401,F403
from .jvp_equations import *  # noqa: F401,F403
from .parser import *  # noqa: F401,F403

__all__ = ["load_cellml_model"]  # populated by star imports
```

**Modified Content:**
```python
"""Parsing utilities for symbolic ODE descriptions."""

# Main exports from parser.py
from .parser import (
    parse_input,
    ParsedEquations,
    EquationWarning,
    TIME_SYMBOL,
    DRIVER_SETTING_KEYS,
    PARSE_TRANSORMS,
    _detect_input_type,
    _process_parameters,
)

# String parser exports (for backward compatibility with tests)
from .string_parser import (
    _lhs_pass,
    _rhs_pass,
    _normalise_indexed_tokens,
    _sanitise_input_math,
    _replace_if,
    _rename_user_calls,
    _build_sympy_user_functions,
)

# Common utilities exports (for backward compatibility with tests)
from .common import (
    _lhs_pass_sympy,
    _rhs_pass_sympy,
    _normalize_sympy_equations,
    _process_calls,
    KNOWN_FUNCTIONS,
)

# Function parser (new)
from .function_parser import FunctionParser

# Other modules (unchanged)
from .auxiliary_caching import *  # noqa: F401,F403
from .cellml import *  # noqa: F401,F403
from .jvp_equations import *  # noqa: F401,F403

__all__ = [
    # Main API
    "parse_input",
    "ParsedEquations",
    "EquationWarning",
    # Parser classes
    "FunctionParser",
    # String parser
    "_lhs_pass",
    "_rhs_pass",
    "_normalise_indexed_tokens",
    "_sanitise_input_math",
    "_replace_if",
    "_rename_user_calls",
    "_build_sympy_user_functions",
    # Common utilities
    "_lhs_pass_sympy",
    "_rhs_pass_sympy",
    "_normalize_sympy_equations",
    "_process_calls",
    "KNOWN_FUNCTIONS",
    # Constants
    "TIME_SYMBOL",
    "DRIVER_SETTING_KEYS",
    "PARSE_TRANSORMS",
    # From other modules
    "load_cellml_model",  # from cellml
    "IndexedBases",  # Added for convenience
    "JVPEquations",  # from jvp_equations
]
```

**Rationale:**
- Explicit exports for clarity
- Backward compatibility with existing test imports
- FunctionParser available for advanced users
- Private functions (leading underscore) still exported for testing

---

## Component 2: Routing Logic Updates

### Task 2.1: Modify _detect_input_type()

#### File: `src/cubie/odesystems/symbolic/parsing/parser.py`

**Current Implementation:**
```python
def _detect_input_type(dxdt: Union[str, Iterable]) -> str:
    """Detect whether dxdt contains strings or SymPy expressions."""
    if dxdt is None:
        raise TypeError("dxdt cannot be None")
    
    if isinstance(dxdt, str):
        return "string"
    
    # ... iterable checking for strings vs SymPy
    
    raise TypeError(...)
```

**Modified Implementation:**
```python
def _detect_input_type(dxdt: Union[str, Iterable, Callable]) -> str:
    """Detect whether dxdt is a callable, string, or SymPy expression.
    
    Determines input format by inspecting the type of dxdt to categorize
    as callable (function), string-based, or SymPy-based input.
    
    Parameters
    ----------
    dxdt
        System equations as callable, string, or iterable.
    
    Returns
    -------
    str
        Either 'function', 'string', or 'sympy' indicating input format.
    
    Raises
    ------
    TypeError
        If input type cannot be determined or is invalid.
    ValueError
        If empty iterable is provided.
    """
    if dxdt is None:
        raise TypeError("dxdt cannot be None")
    
    # Check for callable first (new)
    if callable(dxdt):
        return "function"
    
    if isinstance(dxdt, str):
        return "string"
    
    try:
        items = list(dxdt)
    except TypeError:
        raise TypeError(
            f"dxdt must be string, callable, or iterable, "
            f"got {type(dxdt).__name__}"
        )
    
    if len(items) == 0:
        raise ValueError("dxdt iterable cannot be empty")
    
    first_elem = items[0]
    
    if isinstance(first_elem, str):
        return "string"
    elif isinstance(first_elem, (sp.Expr, sp.Equality)):
        return "sympy"
    elif isinstance(first_elem, tuple):
        if len(first_elem) == 2:
            lhs, rhs = first_elem
            if isinstance(lhs, (sp.Symbol, sp.Derivative)) and isinstance(
                rhs, sp.Expr
            ):
                return "sympy"
    
    raise TypeError(
        f"dxdt elements must be strings, callable, or SymPy expressions, "
        f"got {type(first_elem).__name__}. "
        f"Valid SymPy formats: sp.Equality, sp.Expr, or "
        f"tuple of (sp.Symbol|sp.Derivative, sp.Expr)"
    )
```

**Key Changes:**
1. Add `Callable` to type annotation
2. Check `callable(dxdt)` before string check
3. Return `"function"` for callables
4. Update error messages to mention callable option
5. Update docstring

**Testing:**
- Add test cases for callable detection
- Ensure string and SymPy detection unchanged
- Test error messages for invalid types

---

### Task 2.2: Modify parse_input() for Function Routing

#### File: `src/cubie/odesystems/symbolic/parsing/parser.py`

**Location in Function:** After `input_type = _detect_input_type(dxdt)` (line 1493)

**New Code Block to Insert:**
```python
    input_type = _detect_input_type(dxdt)
    
    if input_type == "function":
        # Route to FunctionParser
        from .function_parser import FunctionParser
        
        function_parser = FunctionParser(
            func=dxdt,
            indexed_bases=index_map,
            observables=observables if observables else [],
        )
        
        # Build equations via FunctionParser
        parsed_equations = function_parser.build_equations()
        
        # FunctionParser updates index_map in place with discovered symbols
        # Get all symbols including those added by parser
        all_symbols = index_map.all_symbols.copy()
        all_symbols.setdefault("t", TIME_SYMBOL)
        
        # FunctionParser doesn't use user_functions the same way
        # (functions are Python code, not symbolic)
        funcs = {}
        if user_functions:
            # Still process user_functions for consistency, though unlikely used
            funcs = _build_sympy_user_functions(
                user_functions, user_function_derivatives
            )
        
        # No new parameters inferred in function path (explicit signatures)
        new_params = []
        
        # Set equation_map for subsequent processing
        # ParsedEquations already built by FunctionParser
        # We need to extract equations in raw form for hash computation
        equation_map = parsed_equations.ordered
        
    elif input_type == "string":
        # Existing string handling code (unchanged)
        # ...
```

**Full Modified parse_input() Structure:**
```python
def parse_input(
    dxdt: Union[str, Iterable[str], Callable],  # Add Callable
    # ... rest of signature unchanged
) -> Tuple[IndexedBases, Dict[str, object], Dict[str, Callable], ParsedEquations, str]:
    """Process user equations and symbol metadata into structured components.
    
    Parameters
    ----------
    dxdt
        System equations as a callable function, newline-delimited string, 
        iterable of strings, or SymPy expressions.
    # ... rest of docstring, add callable documentation
    """
    # Parameter defaults (unchanged)
    if states is None:
        states = {}
        if strict:
            raise ValueError(...)
    # ... rest of defaults
    
    # Process parameters into IndexedBases (unchanged)
    index_map = _process_parameters(...)
    
    # Detect input type (modified to handle callable)
    input_type = _detect_input_type(dxdt)
    
    # NEW: Function pathway
    if input_type == "function":
        # Code block from above
        
    # EXISTING: String pathway (unchanged)
    elif input_type == "string":
        if isinstance(dxdt, str):
            lines = [...]
        # ... existing string processing
        
    # EXISTING: SymPy pathway (unchanged)
    elif input_type == "sympy":
        if isinstance(dxdt, (list, tuple)):
            equations = list(dxdt)
        # ... existing SymPy processing
        
    else:
        raise RuntimeError(
            f"Invalid input_type '{input_type}' from _detect_input_type"
        )
    
    # Post-processing (unchanged for all pathways)
    for param in new_params:
        index_map.parameters.push(param)
        all_symbols[str(param)] = param
    
    if driver_dict is not None:
        index_map.drivers.set_passthrough_defaults(driver_dict)
    
    # User function handling (unchanged)
    if user_functions:
        all_symbols.update(...)
        # ...
    
    # Build ParsedEquations (may already be built by FunctionParser)
    if input_type != "function":
        parsed_equations = ParsedEquations.from_equations(equation_map, index_map)
    # else: already built by FunctionParser
    
    # Compute hash (unchanged)
    fn_hash = hash_system_definition(
        parsed_equations,
        index_map.constants.default_values,
        observable_labels=index_map.observables.ref_map.keys(),
    )
    
    return index_map, all_symbols, funcs, parsed_equations, fn_hash
```

**Key Integration Points:**
1. FunctionParser receives `indexed_bases` with user-provided states/params/constants
2. FunctionParser returns `ParsedEquations` directly
3. FunctionParser updates `indexed_bases` with inferred auxiliaries/observables
4. Hash computation uses same logic regardless of input type
5. Return signature unchanged

**Error Handling:**
- FunctionParser raises errors during parsing (signature issues, AST problems)
- Errors propagate up to user with clear messages
- Stack traces point to function_parser.py for function-related errors

---

## Component 3: FunctionParser Integration

### Task 3.1: Finalize FunctionParser Interface

#### File: `src/cubie/odesystems/symbolic/parsing/function_parser.py`

**Class Interface Expected by parse_input():**

```python
class FunctionParser:
    """Parser for Python function-based ODE definitions.
    
    Extracts symbolic equations from Python function signatures and bodies,
    converting to SymPy expressions for CUDA code generation.
    """
    
    def __init__(
        self,
        func: Callable,
        indexed_bases: IndexedBases,
        observables: List[str],
    ):
        """Initialize parser with function and symbol collections.
        
        Parameters
        ----------
        func
            User-provided ODE function with signature (t, y, ...).
        indexed_bases
            Symbol collections populated from user inputs. Parser may
            add auxiliary symbols and inferred parameters.
        observables
            User-specified observable names to extract from function.
        """
        # Implementation from Sections 1-3
        
    def build_equations(self) -> ParsedEquations:
        """Parse function and build ParsedEquations.
        
        Returns
        -------
        ParsedEquations
            Structured equations ready for SymbolicODE compilation.
        
        Raises
        ------
        TypeError
            If function signature is invalid or unsupported.
        ValueError
            If function body contains unsupported constructs.
        """
        # Implementation from Section 3
```

**Coordination with parse_input():**
1. `indexed_bases` passed in pre-populated with user states/params/constants
2. `observables` list passed in from user
3. Parser modifies `indexed_bases` in place (adds auxiliaries)
4. Returns `ParsedEquations` constructed via `ParsedEquations.from_equations()`

**Expected Behavior:**
- FunctionParser uses `EquationConstructor` from Section 3
- `EquationConstructor.build_equations()` returns equations as list
- FunctionParser wraps in `ParsedEquations.from_equations(equations, indexed_bases)`
- No separate LHS/RHS pass needed (EquationConstructor handles)

---

### Task 3.2: Connect FunctionParser to Common Validation

**Integration Point:** FunctionParser should use `_lhs_pass_sympy` and `_rhs_pass_sympy` from `common.py` if possible

**Analysis:**
- FunctionParser currently builds equations internally via EquationConstructor
- EquationConstructor converts AST → SymPy equations directly
- Question: Should FunctionParser use common validation functions?

**Decision:**
- **Option A**: FunctionParser builds equations, then passes to `_lhs_pass_sympy` / `_rhs_pass_sympy`
  - Pros: Consistent validation with string parser
  - Cons: May duplicate work already done in EquationConstructor
  
- **Option B**: FunctionParser validates independently, ensure equivalent checks
  - Pros: Cleaner separation, function parser self-contained
  - Cons: Risk of divergent validation logic

**Chosen Approach: Option B with validation equivalence**
- FunctionParser validates within EquationConstructor
- Ensure validation checks match those in `_lhs_pass_sympy` / `_rhs_pass_sympy`:
  - State derivative symbols present for all states
  - Observable symbols assigned
  - No assignments to immutable inputs (params, constants, drivers)
  - Auxiliary symbols tracked correctly
- Add equivalence tests to verify same validation errors for same issues

**Rationale:**
- Function parser has richer information (AST, types) than string parser
- Better error messages possible with function context
- Avoids forcing SymPy equations through string-designed validation

---

## Component 4: Backward Compatibility Testing

### Task 4.1: Verify Existing Tests Pass

**Test Files to Check:**
1. `tests/odesystems/symbolic/test_parser.py`
   - All existing tests must pass unchanged
   - Update imports if needed: `from cubie.odesystems.symbolic.parsing import ...`
   - No test behavior changes

2. `tests/odesystems/symbolic/test_symbolicode.py`
   - Integration tests using `create_ODE_system()`
   - All existing tests must pass unchanged

3. `tests/odesystems/symbolic/test_indexedbasemaps.py`
   - Tests for IndexedBases
   - Should be unaffected

4. Any other tests importing from `parsing` module
   - Search: `grep -r "from.*parsing import" tests/`
   - Verify imports still work via `__init__.py` exports

**Success Criteria:**
- Zero test failures in existing parser tests
- Zero test failures in existing SymbolicODE tests
- All imports resolve correctly
- No deprecation warnings

### Task 4.2: Add Backward Compatibility Tests

#### File: `tests/odesystems/symbolic/test_backward_compatibility.py` (new)

**Purpose**: Explicitly test that old code patterns still work

**Test Cases:**

```python
def test_string_input_still_works():
    """Verify string input unchanged after refactor."""
    dxdt = ["dx = -k * x", "dy = k * x - d * y"]
    system = create_ODE_system(
        dxdt,
        states={"x": 1.0, "y": 0.0},
        parameters={"k": 0.1, "d": 0.05},
    )
    assert system is not None
    assert len(system.data.states) == 2


def test_sympy_input_still_works():
    """Verify SymPy input unchanged after refactor."""
    x, k, t = sp.symbols("x k t")
    eq = sp.Eq(sp.Derivative(x, t), -k * x)
    system = create_ODE_system(
        [eq],
        parameters={"k": 0.1},
    )
    assert system is not None


def test_parse_input_direct_call():
    """Verify parse_input can still be called directly."""
    from cubie.odesystems.symbolic.parsing import parse_input
    
    dxdt = "dx = -k * x"
    result = parse_input(
        dxdt,
        states={"x": 1.0},
        parameters={"k": 0.1},
    )
    index_map, symbols, funcs, equations, fn_hash = result
    assert len(equations.state_derivatives) == 1


def test_internal_function_imports():
    """Verify internal functions still importable (for tests)."""
    from cubie.odesystems.symbolic.parsing import (
        _lhs_pass,
        _rhs_pass,
        _normalise_indexed_tokens,
        _sanitise_input_math,
    )
    # If imports work, test passes
    assert callable(_lhs_pass)
    assert callable(_rhs_pass)
```

---

## Component 5: Equivalence Testing

### Task 5.1: Add String vs Function Equivalence Tests

#### File: `tests/odesystems/symbolic/test_equivalence.py` (new)

**Purpose**: Verify string and function input produce identical results

**Test Structure:**

```python
import pytest
import numpy as np
import sympy as sp
from cubie import create_ODE_system


class TestStringFunctionEquivalence:
    """Test that string and function inputs produce identical systems."""
    
    def test_simple_exponential_decay(self):
        """dx/dt = -k*x via string vs function."""
        # String version
        string_system = create_ODE_system(
            "dx = -k * x",
            states={"x": 1.0},
            parameters={"k": 0.1},
        )
        
        # Function version
        def ode_func(t, y, k):
            x = y[0]
            dx = -k * x
            return [dx]
        
        function_system = create_ODE_system(
            ode_func,
            states={"x": 1.0},
            parameters={"k": 0.1},
        )
        
        # Compare ODEData
        assert string_system.data.n_states == function_system.data.n_states
        assert string_system.data.n_parameters == function_system.data.n_parameters
        
        # Compare equations (as strings since symbols may differ)
        # This is tricky - need to compare SymPy expressions
        str_eqs = string_system.equations.state_derivatives
        func_eqs = function_system.equations.state_derivatives
        
        assert len(str_eqs) == len(func_eqs)
        
        # Compare RHS expressions (after symbol substitution)
        # Both should simplify to: -k*x
        str_rhs = str_eqs[0][1]
        func_rhs = func_eqs[0][1]
        
        # Normalize and compare
        assert sp.simplify(str_rhs - func_rhs) == 0
    
    
    def test_two_state_system(self):
        """Predator-prey via string vs function."""
        # String version
        string_system = create_ODE_system(
            ["dx = a*x - b*x*y", "dy = c*x*y - d*y"],
            states={"x": 1.0, "y": 0.5},
            parameters={"a": 1.0, "b": 0.1, "c": 0.075, "d": 0.5},
        )
        
        # Function version
        def predator_prey(t, y, a, b, c, d):
            x = y[0]
            y_pop = y[1]
            dx = a * x - b * x * y_pop
            dy = c * x * y_pop - d * y_pop
            return [dx, dy]
        
        function_system = create_ODE_system(
            predator_prey,
            states={"x": 1.0, "y": 0.5},
            parameters={"a": 1.0, "b": 0.1, "c": 0.075, "d": 0.5},
        )
        
        # Compare
        assert string_system.data.n_states == function_system.data.n_states
        assert string_system.data.n_parameters == function_system.data.n_parameters
    
    
    def test_with_observables(self):
        """System with observables via string vs function."""
        # String version
        string_system = create_ODE_system(
            ["dx = -k*x", "total = x + y"],
            states={"x": 1.0},
            observables=["total"],
            parameters={"k": 0.1, "y": 0.5},
        )
        
        # Function version
        def ode_with_obs(t, y, k, y_const):
            x = y[0]
            dx = -k * x
            total = x + y_const  # Observable
            return [dx]
        
        function_system = create_ODE_system(
            ode_with_obs,
            states={"x": 1.0},
            observables=["total"],
            parameters={"k": 0.1, "y": 0.5},
        )
        
        # Compare
        assert len(string_system.equations.observables) == 1
        assert len(function_system.equations.observables) == 1
```

**Note:** These tests verify structural equivalence. Full numerical equivalence tested in integration tests (Section 5.2).

---

### Task 5.2: Add Numerical Equivalence Tests

#### File: `tests/odesystems/symbolic/test_numerical_equivalence.py` (new)

**Purpose**: Verify string and function produce identical numerical results

**Test Structure:**

```python
import pytest
import numpy as np
from cubie import create_ODE_system, solve_ivp


class TestNumericalEquivalence:
    """Test that string and function inputs produce identical numerical results."""
    
    @pytest.mark.parametrize("algorithm", ["RK45", "ImplicitEuler"])
    def test_exponential_decay_numerical(self, algorithm):
        """Verify numerical solutions match."""
        # String version
        string_system = create_ODE_system(
            "dx = -k * x",
            states={"x": 1.0},
            parameters={"k": 0.1},
        )
        
        # Function version
        def ode_func(t, y, k):
            x = y[0]
            return [-k * x]
        
        function_system = create_ODE_system(
            ode_func,
            states={"x": 1.0},
            parameters={"k": 0.1},
        )
        
        # Solve both
        t_span = (0.0, 10.0)
        string_result = solve_ivp(
            string_system,
            t_span,
            algorithm=algorithm,
            dt_save=0.1,
        )
        
        function_result = solve_ivp(
            function_system,
            t_span,
            algorithm=algorithm,
            dt_save=0.1,
        )
        
        # Compare trajectories
        np.testing.assert_allclose(
            string_result.state,
            function_result.state,
            rtol=1e-10,
            atol=1e-10,
        )
```

**Success Criteria:**
- Numerical solutions match to machine precision
- All algorithms produce equivalent results
- Both explicit and implicit methods tested

---

## Component 6: API Documentation

### Task 6.1: Update create_ODE_system() Docstring

#### File: `src/cubie/odesystems/symbolic/symbolicODE.py`

**Current Docstring (line 73):**
```python
def create_ODE_system(
    dxdt: Union[str, Iterable[str]],
    # ...
) -> "SymbolicODE":
    """Create a :class:`SymbolicODE` from SymPy definitions.
    
    Parameters
    ----------
    dxdt
        System equations defined as either a single string or an iterable of
        equation strings in ``lhs = rhs`` form.
```

**Updated Docstring:**
```python
def create_ODE_system(
    dxdt: Union[str, Iterable[str], Callable],  # Add Callable
    # ...
) -> "SymbolicODE":
    """Create a :class:`SymbolicODE` from symbolic definitions.
    
    Accepts ODE systems defined as string equations, SymPy expressions,
    or Python functions with scipy.integrate.solve_ivp-compatible signatures.
    
    Parameters
    ----------
    dxdt
        System equations defined as:
        
        - **String or list of strings**: Equations in ``lhs = rhs`` form
          (e.g., ``"dx = -k * x"`` or ``["dx = a*x", "dy = -b*y"]``)
        - **Callable function**: Python function with signature ``(t, y, ...)``
          where ``t`` is time (scalar), ``y`` is state vector (accessed via
          indexing like ``y[0]`` or ``y["name"]``), and remaining arguments
          are parameters/constants. Function should return list/array of
          derivatives in same order as states.
        - **SymPy expressions**: List of SymPy Equality or (lhs, rhs) tuples
        
        Example function input::
        
            def my_ode(t, y, k, damping):
                x = y[0]  # or y["position"]
                v = y[1]  # or y["velocity"]
                dx = v
                dv = -k * x - damping * v
                return [dx, dv]
            
            system = create_ODE_system(
                my_ode,
                states={"position": 1.0, "velocity": 0.0},
                parameters={"k": 1.0, "damping": 0.1},
            )
```

**Additional Examples Section:**
Add comprehensive examples showing:
1. String input (existing example)
2. Function input (new example)
3. SymPy input (existing example)

---

### Task 6.2: Update parse_input() Docstring

#### File: `src/cubie/odesystems/symbolic/parsing/parser.py`

**Update Parameter Documentation:**
```python
def parse_input(
    dxdt: Union[str, Iterable[str], Callable],  # Add Callable
    # ...
) -> Tuple[...]:
    """Process user equations and symbol metadata into structured components.
    
    Parameters
    ----------
    dxdt
        System equations as:
        
        - Callable function with signature ``(t, y, ...)``
        - Newline-delimited string or iterable of strings
        - List of SymPy Equality objects or (lhs, rhs) tuples
        
        For callable input, the function should accept time as first argument,
        state vector as second argument, and parameters/constants as remaining
        arguments. State access patterns (indexing, attribute access) are
        analyzed to infer variable names. Return value should be list/tuple/array
        of derivatives in same order as states.
```

---

## Component 7: User Documentation

### Task 7.1: Update README.md

#### File: `readme.md`

**Current Relevant Section (lines 18-22):**
```markdown
- Set up and solve large parameter/initial condition sweeps of a system defined by a set of ODEs, entered either as:
  - A string or list of strings containing the equations of the system
  - A python function (not well tested yet)
  - A CellML model (tested on a subset of models in the CellML library so far)
```

**Updated Section:**
```markdown
- Set up and solve large parameter/initial condition sweeps of a system defined by a set of ODEs, entered as:
  - **String equations**: List of equations like `["dx = -k*x", "dy = k*x"]`
  - **Python function**: scipy-style function `def f(t, y, params): ...`
  - **SymPy expressions**: List of symbolic equations
  - **CellML model**: Import from CellML library (tested on subset of models)
```

**Add Usage Example Section:**

Insert after installation section (around line 92):

```markdown
## Quick Start

### String Input (Traditional)

```python
from cubie import create_ODE_system, solve_ivp

# Define system as strings
system = create_ODE_system(
    dxdt=["dx = -k * x", "dy = k * x - d * y"],
    states={"x": 1.0, "y": 0.0},
    parameters={"k": 0.1, "d": 0.05},
)

# Solve
result = solve_ivp(system, t_span=(0, 100), algorithm="RK45")
```

### Function Input (New)

```python
from cubie import create_ODE_system, solve_ivp

# Define system as Python function
def my_ode(t, y, k, d):
    """Standard scipy-style ODE function."""
    x = y[0]
    y_val = y[1]
    dx = -k * x
    dy = k * x - d * y_val
    return [dx, dy]

# Create system (signature analyzed automatically)
system = create_ODE_system(
    dxdt=my_ode,
    states={"x": 1.0, "y": 0.0},
    parameters={"k": 0.1, "d": 0.05},
)

# Solve (same API)
result = solve_ivp(system, t_span=(0, 100), algorithm="RK45")
```

**Benefits of function input:**
- IDE autocomplete and syntax checking
- Type hints and docstrings
- Easier debugging
- Unit testing of ODE function
- Familiar syntax for scipy/MATLAB users
```

---

### Task 7.2: Create Migration Guide

#### File: `docs/migration_guide.md` (new, or add to existing docs)

**Content:**

```markdown
# Migration Guide: String to Function Input

This guide shows how to convert existing string-based ODE definitions to Python functions.

## Why Use Functions?

- **IDE Support**: Autocomplete, syntax highlighting, type checking
- **Testing**: Unit test your ODE function independently
- **Debugging**: Use debugger, add print statements during development
- **Documentation**: Add docstrings and type hints
- **Familiarity**: Standard scipy.integrate.solve_ivp signature

## Basic Conversion

### String Version
```python
system = create_ODE_system(
    dxdt="dx = -k * x",
    states={"x": 1.0},
    parameters={"k": 0.1},
)
```

### Function Version
```python
def exponential_decay(t, y, k):
    x = y[0]
    dx = -k * x
    return [dx]

system = create_ODE_system(
    dxdt=exponential_decay,
    states={"x": 1.0},
    parameters={"k": 0.1},
)
```

## Multi-State System

### String Version
```python
system = create_ODE_system(
    dxdt=[
        "dx = a*x - b*x*y",
        "dy = c*x*y - d*y",
    ],
    states={"x": 1.0, "y": 0.5},
    parameters={"a": 1.0, "b": 0.1, "c": 0.075, "d": 0.5},
)
```

### Function Version
```python
def predator_prey(t, y, a, b, c, d):
    """Lotka-Volterra predator-prey model."""
    prey = y[0]      # x (prey population)
    predator = y[1]  # y (predator population)
    
    dprey = a * prey - b * prey * predator
    dpredator = c * prey * predator - d * predator
    
    return [dprey, dpredator]

system = create_ODE_system(
    dxdt=predator_prey,
    states={"x": 1.0, "y": 0.5},
    parameters={"a": 1.0, "b": 0.1, "c": 0.075, "d": 0.5},
)
```

## With Observables

### String Version
```python
system = create_ODE_system(
    dxdt=[
        "dx = -k * x",
        "dy = k * x - d * y",
        "total = x + y",  # Observable
    ],
    states={"x": 1.0, "y": 0.0},
    observables=["total"],
    parameters={"k": 0.1, "d": 0.05},
)
```

### Function Version
```python
def two_compartment(t, y, k, d):
    x = y[0]
    y_comp = y[1]
    
    dx = -k * x
    dy = k * x - d * y_comp
    
    total = x + y_comp  # Computed but not returned
    
    return [dx, dy]

system = create_ODE_system(
    dxdt=two_compartment,
    states={"x": 1.0, "y": 0.0},
    observables=["total"],  # Tell CuBIE to track 'total'
    parameters={"k": 0.1, "d": 0.05},
)
```

## Advanced: Named State Access

Instead of `y[0]`, `y[1]`, you can use attribute access if your function
expects a dict-like state object:

```python
def my_ode(t, y, params):
    # Access states by name
    position = y["position"]  # or y.position with proper object
    velocity = y["velocity"]  # or y.velocity
    
    # Access parameters
    spring_k = params["k"]
    damping = params["damping"]
    
    dposition = velocity
    dvelocity = -spring_k * position - damping * velocity
    
    return [dposition, dvelocity]

system = create_ODE_system(
    dxdt=my_ode,
    states={"position": 1.0, "velocity": 0.0},
    parameters={"k": 1.0, "damping": 0.1},
)
```

## When to Use Strings vs Functions

**Use Strings When:**
- Quick prototyping / exploratory analysis
- Simple systems (1-3 equations)
- Copy-pasting equations from papers
- Dynamically generating equations

**Use Functions When:**
- Complex systems (many states/parameters)
- Developing reusable ODE models
- Want IDE support and type checking
- Team collaboration (easier code review)
- Testing ODE logic independently

## Compatibility

Both string and function inputs produce identical CUDA kernels and numerical
results. You can mix approaches across different systems in the same project.
```

---

### Task 7.3: Add Error Message Guidance

Update error messages in FunctionParser to reference documentation:

```python
# In FunctionInspector._validate_function()
raise TypeError(
    "Lambda functions are not supported for ODE definitions. "
    "Please use 'def' syntax to define your function. "
    "See documentation: https://ccam80.github.io/cubie/function_input"
)

# In VariableClassifier
raise ValueError(
    f"Inconsistent state access patterns detected. "
    f"State '{name}' accessed via both {pattern1} and {pattern2}. "
    f"Please use consistent indexing throughout your function. "
    f"See documentation for details."
)
```

---

## Component 8: Testing Checklist

### Pre-Merge Validation

**Run all tests:**
```bash
# Full test suite
pytest

# Parser-specific tests
pytest tests/odesystems/symbolic/test_parser.py

# New equivalence tests
pytest tests/odesystems/symbolic/test_equivalence.py
pytest tests/odesystems/symbolic/test_numerical_equivalence.py

# Backward compatibility tests
pytest tests/odesystems/symbolic/test_backward_compatibility.py

# Integration tests
pytest tests/odesystems/symbolic/test_symbolicode.py
```

**Verify backward compatibility:**
- [ ] All existing parser tests pass unchanged
- [ ] All existing SymbolicODE tests pass unchanged
- [ ] All imports from `parsing` module work
- [ ] No new deprecation warnings

**Verify function input:**
- [ ] Simple exponential decay works
- [ ] Multi-state systems work
- [ ] Observables work
- [ ] Parameters and constants work
- [ ] Error messages are clear

**Verify equivalence:**
- [ ] String and function produce identical ParsedEquations structure
- [ ] String and function produce identical numerical results
- [ ] Hash computation identical for equivalent systems

**Documentation:**
- [ ] README.md updated with examples
- [ ] Docstrings updated for create_ODE_system()
- [ ] Docstrings updated for parse_input()
- [ ] Migration guide created
- [ ] Error messages reference documentation

---

## Summary

This plan details:
1. **Module reorganization**: Extract string_parser.py, common.py, keep parser.py as router
2. **Routing updates**: Modify `_detect_input_type()` and `parse_input()` to handle callables
3. **FunctionParser integration**: Ensure FunctionParser returns compatible ParsedEquations
4. **Backward compatibility**: Explicit exports, equivalence testing, no API changes
5. **Documentation**: README, docstrings, migration guide, error messages

**Key Principles:**
- Zero breaking changes to existing API
- Clean separation of string vs function parsing logic
- Comprehensive equivalence testing
- Clear documentation for users

**Integration Dependencies:**
- Sections 1-3 must be complete before this section
- FunctionParser must produce ParsedEquations compatible with existing infrastructure
- All validation must match existing string parser behavior

