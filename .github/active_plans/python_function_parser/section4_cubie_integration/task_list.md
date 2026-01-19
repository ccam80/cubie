# Implementation Task List
# Feature: Python Function Parser - Section 4: CuBIE Integration and Cleanup
# Plan Reference: .github/active_plans/python_function_parser/section4_cubie_integration/agent_plan.md

## Overview

Section 4 integrates the FunctionParser with CuBIE's existing infrastructure, reorganizes the parsing module, ensures backward compatibility, and documents the new capability.

**Critical Dependencies:**
- Sections 1-3 MUST be complete and passing tests
- FunctionParser class fully implemented (Section 3)
- All Section 1-3 components available in parsing module

---

## Task Group 1: Extract String-Specific Parsing Module
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: src/cubie/odesystems/symbolic/parsing/parser.py (lines 470-627: _sanitise_input_math, _replace_if, _normalise_indexed_tokens)
- File: src/cubie/odesystems/symbolic/parsing/parser.py (lines 806-1018: _lhs_pass)
- File: src/cubie/odesystems/symbolic/parsing/parser.py (lines 1152-1420: _rhs_pass, _rename_user_calls, _build_sympy_user_functions, _inline_nondevice_calls)
- File: src/cubie/odesystems/symbolic/indexedbasemaps.py (lines 1-50: IndexedBases import)
- File: .github/active_plans/python_function_parser/section4_cubie_integration/agent_plan.md (lines 13-96: Task 1.1 specification)

**Input Validation Required**:
- None (internal module reorganization)

**Tasks**:

1. **Create string_parser.py module**
   - File: src/cubie/odesystems/symbolic/parsing/string_parser.py
   - Action: Create
   - Details:
     ```python
     """String-based equation parsing for symbolic ODE systems."""
     
     import re
     from typing import (
         Callable,
         Dict,
         Iterable,
         List,
         Optional,
         Sequence,
         Tuple,
     )
     from warnings import warn
     
     import sympy as sp
     from sympy.parsing.sympy_parser import T, parse_expr
     from sympy.core.function import AppliedUndef
     
     from ..indexedbasemaps import IndexedBases
     from .parser import (
         TIME_SYMBOL,
         _INDEXED_NAME_PATTERN,
         EquationWarning,
         PARSE_TRANSORMS,
         KNOWN_FUNCTIONS,
     )
     
     # Pattern for function call detection
     _func_call_re = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
     
     # Pattern for derivative function notation
     _DERIVATIVE_FUNC_PATTERN = re.compile(
         r"^d\s*\(\s*([A-Za-z_]\w*)\s*,\s*t\s*\)$"
     )
     
     
     def _sanitise_input_math(expr_str: str) -> str:
         # Copy from parser.py lines 470-478
         # No changes to implementation
         pass
     
     
     def _replace_if(expr_str: str) -> str:
         # Copy from parser.py lines 481-497
         # No changes to implementation
         pass
     
     
     def _normalise_indexed_tokens(lines: Iterable[str]) -> list[str]:
         # Copy from parser.py lines 500-520
         # No changes to implementation
         pass
     
     
     def _rename_user_calls(
         lines: Iterable[str],
         user_functions: Optional[Dict[str, Callable]] = None,
     ) -> Tuple[List[str], Dict[str, str]]:
         # Copy from parser.py lines 630-651
         # No changes to implementation
         pass
     
     
     def _build_sympy_user_functions(
         user_functions: Optional[Dict[str, Callable]],
         rename: Dict[str, str],
         user_function_derivatives: Optional[Dict[str, Callable]] = None,
     ) -> Tuple[Dict[str, object], Dict[str, str], Dict[str, bool]]:
         # Copy from parser.py lines 654-722
         # No changes to implementation
         pass
     
     
     def _inline_nondevice_calls(
         expr: sp.Expr,
         user_functions: Dict[str, Callable],
         rename: Dict[str, str],
     ) -> sp.Expr:
         # Copy from parser.py lines 725-760
         # No changes to implementation
         pass
     
     
     def _process_calls(
         equations_input: Iterable[str],
         user_functions: Optional[Dict[str, Callable]] = None,
     ) -> Dict[str, Callable]:
         # Copy from parser.py lines 763-793
         # No changes to implementation
         pass
     
     
     def _lhs_pass(
         lines: Sequence[str],
         indexed_bases: IndexedBases,
         strict: bool = True,
     ) -> Dict[str, sp.Symbol]:
         # Copy from parser.py lines 806-1018
         # No changes to implementation
         pass
     
     
     def _rhs_pass(
         lines: Iterable[str],
         all_symbols: Dict[str, sp.Symbol],
         user_funcs: Optional[Dict[str, Callable]] = None,
         user_function_derivatives: Optional[Dict[str, Callable]] = None,
         strict: bool = True,
         raw_lines: Optional[Sequence[str]] = None,
     ) -> Tuple[
         List[Tuple[sp.Symbol, sp.Expr]], Dict[str, Callable], List[sp.Symbol]
     ]:
         # Copy from parser.py lines 1152-1276
         # No changes to implementation
         pass
     ```
   - Edge cases: None (exact code copy)
   - Integration: Imported by parser.py for string pathway

**Tests to Create**:
- Test file: tests/odesystems/symbolic/test_string_parser.py
- Test function: test_sanitise_input_math
- Description: Verify ternary conditionals converted to Piecewise
- Test function: test_normalise_indexed_tokens
- Description: Verify x[0] converted to x0
- Test function: test_lhs_pass_inference
- Description: Verify state derivative inference in non-strict mode
- Test function: test_rhs_pass_validation
- Description: Verify undefined symbol detection in strict mode

**Outcomes**:

---

## Task Group 2: Extract Common Utilities Module
**Status**: [ ]
**Dependencies**: Groups [1]

**Required Context**:
- File: src/cubie/odesystems/symbolic/parsing/parser.py (lines 47-98: _detect_input_type)
- File: src/cubie/odesystems/symbolic/parsing/parser.py (lines 100-233: _normalize_sympy_equations)
- File: src/cubie/odesystems/symbolic/parsing/parser.py (lines 235-299: KNOWN_FUNCTIONS)
- File: src/cubie/odesystems/symbolic/parsing/parser.py (lines 1020-1080: _lhs_pass_sympy)
- File: src/cubie/odesystems/symbolic/parsing/parser.py (lines 1083-1149: _process_user_functions_for_rhs, _rhs_pass_sympy)
- File: .github/active_plans/python_function_parser/section4_cubie_integration/agent_plan.md (lines 99-173: Task 1.2 specification)

**Input Validation Required**:
- None (internal module reorganization)

**Tasks**:

1. **Create common.py module**
   - File: src/cubie/odesystems/symbolic/parsing/common.py
   - Action: Create
   - Details:
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
     from warnings import warn
     
     import sympy as sp
     from sympy.core.function import AppliedUndef
     
     from ..indexedbasemaps import IndexedBases
     from .parser import TIME_SYMBOL, EquationWarning
     
     # Copy KNOWN_FUNCTIONS from parser.py lines 235-299
     KNOWN_FUNCTIONS = {
         "exp": sp.exp,
         "log": sp.log,
         "sqrt": sp.sqrt,
         # ... full mapping
     }
     
     
     def _normalize_sympy_equations(
         equations: Iterable[
             Union[sp.Equality, Tuple[sp.Symbol, sp.Expr], sp.Expr]
         ],
         index_map: IndexedBases,
     ) -> List[Tuple[sp.Symbol, sp.Expr]]:
         # Copy from parser.py lines 100-233
         # No changes to implementation
         pass
     
     
     def _lhs_pass_sympy(
         equations: List[Tuple[sp.Symbol, sp.Expr]],
         indexed_bases: IndexedBases,
         strict: bool = True,
     ) -> Dict[str, sp.Symbol]:
         # Copy from parser.py lines 1020-1080
         # No changes to implementation
         pass
     
     
     def _process_user_functions_for_rhs(
         user_funcs: Optional[Dict[str, Callable]],
         user_function_derivatives: Optional[Dict[str, Callable]],
     ) -> Dict[str, Callable]:
         # Copy from parser.py lines 1083-1105
         # No changes to implementation
         pass
     
     
     def _rhs_pass_sympy(
         equations: List[Tuple[sp.Symbol, sp.Expr]],
         all_symbols: Dict[str, sp.Symbol],
         indexed_bases: IndexedBases,
         user_funcs: Optional[Dict[str, Callable]] = None,
         user_function_derivatives: Optional[Dict[str, Callable]] = None,
         strict: bool = True,
     ) -> Tuple[
         List[Tuple[sp.Symbol, sp.Expr]], Dict[str, Callable], List[sp.Symbol]
     ]:
         # Copy from parser.py lines 1108-1149
         # No changes to implementation
         pass
     ```
   - Edge cases: None (exact code copy)
   - Integration: Imported by parser.py for all pathways, imported by string_parser.py

**Tests to Create**:
- Test file: tests/odesystems/symbolic/test_common_utilities.py
- Test function: test_normalize_sympy_equations_derivatives
- Description: Verify Derivative(x, t) converted to dx
- Test function: test_lhs_pass_sympy_state_assignment_error
- Description: Verify error when assigning to state directly
- Test function: test_rhs_pass_sympy_strict_validation
- Description: Verify undefined symbols raise ValueError in strict mode

**Outcomes**:

---

## Task Group 3: Refactor Main Parser Module
**Status**: [ ]
**Dependencies**: Groups [1, 2]

**Required Context**:
- File: src/cubie/odesystems/symbolic/parsing/parser.py (entire file)
- File: src/cubie/odesystems/symbolic/parsing/string_parser.py (from Task Group 1)
- File: src/cubie/odesystems/symbolic/parsing/common.py (from Task Group 2)
- File: .github/active_plans/python_function_parser/section4_cubie_integration/agent_plan.md (lines 176-234: Task 1.3 specification)

**Input Validation Required**:
- None (refactoring existing validated code)

**Tasks**:

1. **Update parser.py imports**
   - File: src/cubie/odesystems/symbolic/parsing/parser.py
   - Action: Modify
   - Details:
     ```python
     # Replace lines 1-15 with:
     """Parse symbolic ODE descriptions into structured SymPy objects."""
     
     import re
     from typing import (
         Any,
         Callable,
         Dict,
         Iterable,
         List,
         Optional,
         Sequence,
         Tuple,
         Union,
     )
     from warnings import warn
     
     import sympy as sp
     import attrs
     
     from ..indexedbasemaps import IndexedBases
     from ..sym_utils import hash_system_definition
     from cubie._utils import is_devfunc
     
     # Import from newly created modules
     from .string_parser import (
         _lhs_pass,
         _rhs_pass,
         _normalise_indexed_tokens,
         _sanitise_input_math,
         _rename_user_calls,
         _build_sympy_user_functions,
         _inline_nondevice_calls,
         _process_calls,
     )
     from .common import (
         _lhs_pass_sympy,
         _rhs_pass_sympy,
         _normalize_sympy_equations,
         _process_user_functions_for_rhs,
         KNOWN_FUNCTIONS,
     )
     ```
   - Edge cases: Ensure all imports resolve correctly
   - Integration: parser.py becomes routing logic only

2. **Remove extracted functions from parser.py**
   - File: src/cubie/odesystems/symbolic/parsing/parser.py
   - Action: Modify
   - Details:
     - Delete lines 235-299 (KNOWN_FUNCTIONS - now in common.py)
     - Delete lines 470-520 (_sanitise_input_math, _replace_if, _normalise_indexed_tokens - now in string_parser.py)
     - Delete lines 630-793 (_rename_user_calls, _build_sympy_user_functions, _inline_nondevice_calls, _process_calls - now in string_parser.py)
     - Delete lines 806-1018 (_lhs_pass - now in string_parser.py)
     - Delete lines 1020-1080 (_lhs_pass_sympy - now in common.py)
     - Delete lines 1083-1149 (_process_user_functions_for_rhs, _rhs_pass_sympy - now in common.py)
     - Delete lines 1152-1276 (_rhs_pass - now in string_parser.py)
   - Edge cases: Verify no orphaned references remain
   - Integration: Cleaned parser.py focuses on parse_input() and routing

3. **Keep essential parser.py content**
   - File: src/cubie/odesystems/symbolic/parsing/parser.py
   - Action: Verify (no changes needed)
   - Details:
     - Keep TIME_SYMBOL constant (line 22)
     - Keep DRIVER_SETTING_KEYS constant (line 23)
     - Keep PARSE_TRANSORMS constant (line 16-17)
     - Keep _INDEXED_NAME_PATTERN constant (line 19)
     - Keep _DERIVATIVE_FUNC_PATTERN constant (line 25-29)
     - Keep EquationWarning class (around line 465)
     - Keep ParsedEquations class (lines 301-426)
     - Keep _detect_input_type() function (lines 47-98)
     - Keep _process_parameters() function (lines 796-803)
     - Keep parse_input() function (lines 1279-end)
   - Edge cases: None
   - Integration: Core functionality preserved

**Tests to Create**:
- Test file: tests/odesystems/symbolic/test_parser_refactored.py
- Test function: test_imports_resolve_correctly
- Description: Verify all functions importable from parser module
- Test function: test_parse_input_string_pathway_unchanged
- Description: Verify existing string tests still pass

**Outcomes**:

---

## Task Group 4: Update Module Exports
**Status**: [ ]
**Dependencies**: Groups [1, 2, 3]

**Required Context**:
- File: src/cubie/odesystems/symbolic/parsing/__init__.py (entire file)
- File: src/cubie/odesystems/symbolic/parsing/string_parser.py (from Task Group 1)
- File: src/cubie/odesystems/symbolic/parsing/common.py (from Task Group 2)
- File: .github/active_plans/python_function_parser/section4_cubie_integration/agent_plan.md (lines 237-326: Task 1.4 specification)

**Input Validation Required**:
- None (module exports)

**Tasks**:

1. **Update __init__.py with explicit exports**
   - File: src/cubie/odesystems/symbolic/parsing/__init__.py
   - Action: Modify
   - Details:
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
         _inline_nondevice_calls,
         _process_calls,
     )
     
     # Common utilities exports (for backward compatibility with tests)
     from .common import (
         _lhs_pass_sympy,
         _rhs_pass_sympy,
         _normalize_sympy_equations,
         _process_user_functions_for_rhs,
         KNOWN_FUNCTIONS,
     )
     
     # Function parser (new) - will be added after Sections 1-3 complete
     # from .function_parser import FunctionParser
     
     # Other modules (unchanged)
     from .auxiliary_caching import *  # noqa: F401,F403
     from .cellml import *  # noqa: F401,F403
     from .jvp_equations import *  # noqa: F401,F403
     
     __all__ = [
         # Main API
         "parse_input",
         "ParsedEquations",
         "EquationWarning",
         # Constants
         "TIME_SYMBOL",
         "DRIVER_SETTING_KEYS",
         "PARSE_TRANSORMS",
         # String parser (private but exported for tests)
         "_lhs_pass",
         "_rhs_pass",
         "_normalise_indexed_tokens",
         "_sanitise_input_math",
         "_replace_if",
         "_rename_user_calls",
         "_build_sympy_user_functions",
         "_inline_nondevice_calls",
         "_process_calls",
         # Common utilities (private but exported for tests)
         "_lhs_pass_sympy",
         "_rhs_pass_sympy",
         "_normalize_sympy_equations",
         "_process_user_functions_for_rhs",
         "_detect_input_type",
         "_process_parameters",
         "KNOWN_FUNCTIONS",
         # From other modules
         "load_cellml_model",  # from cellml
     ]
     ```
   - Edge cases: Ensure all backward-compatible imports work
   - Integration: Tests import from parsing module unchanged

**Tests to Create**:
- Test file: tests/odesystems/symbolic/test_module_exports.py
- Test function: test_all_public_exports_available
- Description: Verify all __all__ items importable
- Test function: test_backward_compatible_imports
- Description: Verify existing test imports still work

**Outcomes**:

---

## Task Group 5: Modify _detect_input_type for Callables
**Status**: [ ]
**Dependencies**: Groups [1, 2, 3, 4]

**Required Context**:
- File: src/cubie/odesystems/symbolic/parsing/parser.py (lines 47-98: _detect_input_type)
- File: .github/active_plans/python_function_parser/section4_cubie_integration/agent_plan.md (lines 337-423: Task 2.1 specification)

**Input Validation Required**:
- dxdt parameter: Check not None, check callable(dxdt) for function type

**Tasks**:

1. **Add callable detection to _detect_input_type**
   - File: src/cubie/odesystems/symbolic/parsing/parser.py
   - Action: Modify
   - Details:
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
         
         # Check for callable FIRST (new priority)
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
   - Edge cases: callable objects that are also iterables (should return "function")
   - Integration: Routing logic in parse_input() depends on this

**Tests to Create**:
- Test file: tests/odesystems/symbolic/test_detect_input_type.py
- Test function: test_detect_function_input
- Description: Verify callable returns "function"
- Test function: test_detect_string_input
- Description: Verify string returns "string"
- Test function: test_detect_sympy_input
- Description: Verify SymPy returns "sympy"
- Test function: test_callable_priority_over_iterable
- Description: Verify callables that are also iterable return "function"

**Outcomes**:

---

## Task Group 6: Integrate FunctionParser into parse_input
**Status**: [ ]
**Dependencies**: Groups [1, 2, 3, 4, 5], Sections [1, 2, 3 complete]

**Required Context**:
- File: src/cubie/odesystems/symbolic/parsing/parser.py (lines 1279-end: parse_input function)
- File: src/cubie/odesystems/symbolic/parsing/function_parser.py (from Section 3: FunctionParser class)
- File: .github/active_plans/python_function_parser/section4_cubie_integration/agent_plan.md (lines 426-564: Task 2.2 specification)
- File: .github/active_plans/python_function_parser/section3_symbolic_ode_integration/agent_plan.md (lines 1-100: FunctionParser interface)

**Input Validation Required**:
- None (parse_input already validates parameters, FunctionParser handles callable validation)

**Tasks**:

1. **Add function pathway to parse_input**
   - File: src/cubie/odesystems/symbolic/parsing/parser.py
   - Action: Modify
   - Details:
     ```python
     def parse_input(
         dxdt: Union[str, Iterable[str], Callable],  # Add Callable to type hint
         # ... rest of signature unchanged
     ) -> Tuple[
         IndexedBases,
         Dict[str, object],
         Dict[str, Callable],
         ParsedEquations,
         str,
     ]:
         """Process user equations and symbol metadata into structured components.
         
         Parameters
         ----------
         dxdt
             System equations as callable function, newline-delimited string, 
             iterable of strings, or SymPy expressions. When callable, function 
             should have signature (t, y, ...) where t is time (scalar), y is 
             state vector, and remaining arguments are parameters/constants.
         # ... rest of docstring
         """
         # ... existing parameter defaults (lines 1293-1323)
         
         index_map = _process_parameters(...)  # Line 1325
         
         input_type = _detect_input_type(dxdt)  # Line 1327
         
         # NEW: Function pathway (insert after line 1327)
         if input_type == "function":
             from .function_parser import FunctionParser
             
             parser = FunctionParser(
                 func=dxdt,
                 indexed_bases=index_map,
                 observables=observables if observables else [],
             )
             
             parsed_equations = parser.build_equations()
             
             # FunctionParser updates index_map in place with discovered symbols
             all_symbols = index_map.all_symbols.copy()
             all_symbols.setdefault("t", TIME_SYMBOL)
             
             # Process user_functions for consistency (though unlikely used with function input)
             funcs = {}
             if user_functions:
                 funcs = _process_user_functions_for_rhs(
                     user_functions, user_function_derivatives
                 )
             
             # No new parameters inferred in function path (explicit signatures)
             new_params = []
             
             # equation_map already built by FunctionParser
             equation_map = parsed_equations.ordered
         
         elif input_type == "string":
             # Existing string handling (lines 1328-1381) - unchanged
             # ...
             pass
         
         elif input_type == "sympy":
             # Existing SymPy handling (lines 1383-1441) - unchanged
             # ...
             pass
         
         else:
             raise RuntimeError(
                 f"Invalid input_type '{input_type}' from _detect_input_type"
             )
         
         # Post-processing (lines 1443-1473) - unchanged
         # Build ParsedEquations if not already built
         if input_type != "function":
             parsed_equations = ParsedEquations.from_equations(equation_map, index_map)
         
         # ... rest of function unchanged
     ```
   - Edge cases: FunctionParser may raise errors; let them propagate with clear stack traces
   - Integration: Complete integration point for Sections 1-3

**Tests to Create**:
- Test file: tests/odesystems/symbolic/test_parse_input_function.py
- Test function: test_parse_input_simple_function
- Description: Verify simple ODE function parsed correctly
- Test function: test_parse_input_function_with_observables
- Description: Verify observables extracted from function
- Test function: test_parse_input_function_returns_correct_structure
- Description: Verify return tuple matches string/SymPy pathways

**Outcomes**:

---

## Task Group 7: Backward Compatibility Validation
**Status**: [ ]
**Dependencies**: Groups [1, 2, 3, 4, 5, 6]

**Required Context**:
- File: tests/odesystems/symbolic/test_parser.py (entire file - if exists)
- File: tests/odesystems/symbolic/test_symbolicode.py (entire file - if exists)
- File: .github/active_plans/python_function_parser/section4_cubie_integration/agent_plan.md (lines 683-768: Task 4.1-4.2 specification)

**Input Validation Required**:
- None (test validation)

**Tasks**:

1. **Create backward compatibility test file**
   - File: tests/odesystems/symbolic/test_backward_compatibility.py
   - Action: Create
   - Details:
     ```python
     """Test backward compatibility after parser refactoring."""
     
     import pytest
     import sympy as sp
     from cubie.odesystems.symbolic import create_ODE_system
     from cubie.odesystems.symbolic.parsing import (
         parse_input,
         _lhs_pass,
         _rhs_pass,
         _normalise_indexed_tokens,
         _sanitise_input_math,
     )
     
     
     def test_string_input_still_works():
         """Verify string input unchanged after refactor."""
         dxdt = ["dx = -k * x", "dy = k * x - d * y"]
         system = create_ODE_system(
             dxdt,
             states={"x": 1.0, "y": 0.0},
             parameters={"k": 0.1, "d": 0.05},
         )
         assert system is not None
         assert len(system.data.states.default_values) == 2
     
     
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
         # If imports work, test passes
         assert callable(_lhs_pass)
         assert callable(_rhs_pass)
         assert callable(_normalise_indexed_tokens)
         assert callable(_sanitise_input_math)
     ```
   - Edge cases: Ensure tests pass with refactored code
   - Integration: Validates backward compatibility guarantee

**Tests to Create**:
- Test file: tests/odesystems/symbolic/test_backward_compatibility.py (created in task)
- Test function: test_string_input_still_works (in created file)
- Description: String input creates valid system
- Test function: test_sympy_input_still_works (in created file)
- Description: SymPy input creates valid system
- Test function: test_internal_function_imports (in created file)
- Description: Internal functions importable for existing tests

**Outcomes**:

---

## Task Group 8: String vs Function Equivalence Tests
**Status**: [ ]
**Dependencies**: Groups [1, 2, 3, 4, 5, 6, 7], Sections [1, 2, 3 complete]

**Required Context**:
- File: src/cubie/odesystems/symbolic/symbolicODE.py (lines 1-100: create_ODE_system)
- File: .github/active_plans/python_function_parser/section4_cubie_integration/agent_plan.md (lines 771-886: Task 5.1 specification)

**Input Validation Required**:
- None (test validation)

**Tasks**:

1. **Create equivalence test file**
   - File: tests/odesystems/symbolic/test_equivalence.py
   - Action: Create
   - Details:
     ```python
     """Test that string and function inputs produce identical systems."""
     
     import pytest
     import numpy as np
     import sympy as sp
     from cubie import create_ODE_system
     
     
     class TestStringFunctionEquivalence:
         """Verify string and function produce identical ParsedEquations."""
         
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
             
             # Compare structure
             assert string_system.data.n_states == function_system.data.n_states
             assert string_system.data.n_parameters == function_system.data.n_parameters
             
             # Compare equations (simplified difference should be zero)
             str_eqs = string_system.equations.state_derivatives
             func_eqs = function_system.equations.state_derivatives
             assert len(str_eqs) == len(func_eqs)
             
             str_rhs = str_eqs[0][1]
             func_rhs = func_eqs[0][1]
             assert sp.simplify(str_rhs - func_rhs) == 0
         
         
         def test_two_state_system(self):
             """Predator-prey via string vs function."""
             string_system = create_ODE_system(
                 ["dx = a*x - b*x*y", "dy = c*x*y - d*y"],
                 states={"x": 1.0, "y": 0.5},
                 parameters={"a": 1.0, "b": 0.1, "c": 0.075, "d": 0.5},
             )
             
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
             
             assert string_system.data.n_states == function_system.data.n_states
             assert string_system.data.n_parameters == function_system.data.n_parameters
         
         
         def test_with_observables(self):
             """System with observables via string vs function."""
             string_system = create_ODE_system(
                 ["dx = -k*x", "total = x + y_const"],
                 states={"x": 1.0},
                 observables=["total"],
                 constants={"k": 0.1, "y_const": 0.5},
             )
             
             def ode_with_obs(t, y, k, y_const):
                 x = y[0]
                 dx = -k * x
                 total = x + y_const
                 return [dx]
             
             function_system = create_ODE_system(
                 ode_with_obs,
                 states={"x": 1.0},
                 observables=["total"],
                 constants={"k": 0.1, "y_const": 0.5},
             )
             
             assert len(string_system.equations.observables) == 1
             assert len(function_system.equations.observables) == 1
     ```
   - Edge cases: Symbol naming differences should not affect equivalence
   - Integration: Validates FunctionParser produces same structure as string parser

**Tests to Create**:
- Test file: tests/odesystems/symbolic/test_equivalence.py (created in task)
- Test function: test_simple_exponential_decay (in created file)
- Description: Single state system equivalence
- Test function: test_two_state_system (in created file)
- Description: Multi-state system equivalence
- Test function: test_with_observables (in created file)
- Description: Observable handling equivalence

**Outcomes**:

---

## Task Group 9: Update create_ODE_system Docstring
**Status**: [ ]
**Dependencies**: Groups [1, 2, 3, 4, 5, 6, 7, 8]

**Required Context**:
- File: src/cubie/odesystems/symbolic/symbolicODE.py (lines 1-150: create_ODE_system function)
- File: .github/active_plans/python_function_parser/section4_cubie_integration/agent_plan.md (lines 965-1023: Task 6.1 specification)

**Input Validation Required**:
- None (documentation update)

**Tasks**:

1. **Update create_ODE_system docstring**
   - File: src/cubie/odesystems/symbolic/symbolicODE.py
   - Action: Modify
   - Details:
     ```python
     def create_ODE_system(
         dxdt: Union[str, Iterable[str], Callable],  # Add Callable
         # ... rest of signature
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
         
         # ... rest of parameters documentation unchanged
         
         Returns
         -------
         SymbolicODE
             Compiled ODE system ready for batch integration.
         
         Notes
         -----
         Function input provides IDE support (autocomplete, syntax checking),
         easier debugging, and familiar syntax for scipy/MATLAB users. All input
         methods produce identical CUDA kernels and numerical results.
         
         Examples
         --------
         String input (traditional)::
         
             system = create_ODE_system(
                 dxdt=["dx = -k * x", "dy = k * x"],
                 states={"x": 1.0, "y": 0.0},
                 parameters={"k": 0.1},
             )
         
         Function input (new)::
         
             def exponential(t, y, k):
                 x = y[0]
                 return [-k * x]
             
             system = create_ODE_system(
                 dxdt=exponential,
                 states={"x": 1.0},
                 parameters={"k": 0.1},
             )
         """
         # Implementation unchanged
     ```
   - Edge cases: None (documentation only)
   - Integration: User-facing API documentation

**Tests to Create**:
- No tests needed (documentation only)

**Outcomes**:

---

## Task Group 10: Update README.md
**Status**: [ ]
**Dependencies**: Groups [1, 2, 3, 4, 5, 6, 7, 8, 9]

**Required Context**:
- File: readme.md (lines 1-100: current content)
- File: .github/active_plans/python_function_parser/section4_cubie_integration/agent_plan.md (lines 1067-1140: Task 7.1 specification)

**Input Validation Required**:
- None (documentation update)

**Tasks**:

1. **Update README.md feature list**
   - File: readme.md
   - Action: Modify
   - Details:
     ```markdown
     # Find and replace around lines 18-22:
     
     # OLD:
     - Set up and solve large parameter/initial condition sweeps of a system defined by a set of ODEs, entered either as:
       - A string or list of strings containing the equations of the system
       - A python function (not well tested yet)
       - A CellML model (tested on a subset of models in the CellML library so far)
     
     # NEW:
     - Set up and solve large parameter/initial condition sweeps of a system defined by a set of ODEs, entered as:
       - **String equations**: List of equations like `["dx = -k*x", "dy = k*x"]`
       - **Python function**: scipy-style function `def f(t, y, params): ...`
       - **SymPy expressions**: List of symbolic equations
       - **CellML model**: Import from CellML library (tested on subset of models)
     ```
   - Edge cases: Preserve other README content
   - Integration: User-facing documentation

2. **Add Quick Start section to README.md**
   - File: readme.md
   - Action: Modify
   - Details:
     ```markdown
     # Insert after installation section (around line 92):
     
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
   - Edge cases: None
   - Integration: User-facing quick start guide

**Tests to Create**:
- No tests needed (documentation only)

**Outcomes**:

---

## Summary

After completing all task groups:

1. **Total Task Groups**: 10
2. **Dependency Chain**:
   - Groups 1-4: Module reorganization (parallel after dependencies)
   - Group 5: Callable detection (depends on 1-4)
   - Group 6: FunctionParser integration (depends on 1-5 + Sections 1-3)
   - Groups 7-8: Testing (depends on 1-6)
   - Groups 9-10: Documentation (depends on 1-8)

3. **Tests to be Created**: 8 test files
   - test_string_parser.py
   - test_common_utilities.py
   - test_parser_refactored.py
   - test_module_exports.py
   - test_detect_input_type.py
   - test_parse_input_function.py
   - test_backward_compatibility.py
   - test_equivalence.py

4. **Estimated Complexity**: Medium-High
   - Large-scale refactoring of existing code
   - No breaking changes to public API
   - Comprehensive testing required
   - Documentation updates across multiple files

5. **Critical Success Factors**:
   - All existing tests must pass unchanged
   - String and function inputs produce identical results
   - Backward compatibility fully maintained
   - Clear documentation for users
