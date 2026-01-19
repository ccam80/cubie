# Implementation Task List
# Feature: Python Function Parser - Section 3: Integration with SymbolicODE
# Plan Reference: .github/active_plans/python_function_parser/section3_symbolic_ode_integration/agent_plan.md

## Overview

Section 3 integrates function parsing components from Sections 1 and 2 with CuBIE's SymbolicODE infrastructure. The EquationConstructor converts AST expressions to SymPy equations, FunctionParser orchestrates the full workflow, and parse_input() routes function-based input to the new parser.

**Dependencies:**
- Section 1: FunctionInspector, AstVisitor, AstToSympyConverter (must be complete)
- Section 2: VariableClassifier with build_indexed_bases() method (must be complete)

---

## Task Group 1: EquationConstructor - Core Equation Building
**Status**: [ ]
**Dependencies**: None (uses Section 1 & 2 components)

**Required Context**:
- File: src/cubie/odesystems/symbolic/parsing/parser.py (lines 301-426: ParsedEquations class)
- File: src/cubie/odesystems/symbolic/indexedbasemaps.py (lines 1-100: IndexedBases structure)
- File: .github/active_plans/python_function_parser/section1_source_code/agent_plan.md (entire file: AstVisitor, AstToSympyConverter interfaces)
- File: .github/active_plans/python_function_parser/section2_variable_parsing/agent_plan.md (entire file: VariableClassifier interface)

**Input Validation Required**:
- indexed_bases: Check type is IndexedBases (from cubie.odesystems.symbolic.indexedbasemaps)
- ast_visitor: Check has attributes access_patterns, assignments, return_node
- observable_names: Check type is list or set of strings
- state_param_name: Check type is str, non-empty

**Tasks**:

1. **Create EquationConstructor Module**
   - File: src/cubie/odesystems/symbolic/parsing/equation_constructor.py
   - Action: Create
   - Details:
     ```python
     from typing import Dict, List, Set, Tuple
     import ast
     import sympy as sp
     
     from ..indexedbasemaps import IndexedBases
     from .parser import ParsedEquations
     # Import AstVisitor, AstToSympyConverter from Section 1
     
     class EquationConstructor:
         """Convert AST expressions to SymPy equations for ParsedEquations.
         
         Parameters
         ----------
         indexed_bases : IndexedBases
             Symbol collections from VariableClassifier.
         ast_visitor : AstVisitor
             AST analysis results from Section 1.
         observable_names : List[str]
             User-specified observable variable names.
         state_param_name : str
             Name of state parameter (e.g., 'y').
         """
         
         def __init__(
             self,
             indexed_bases: IndexedBases,
             ast_visitor: AstVisitor,
             observable_names: List[str],
             state_param_name: str
         ):
             # Validate inputs (as specified in Input Validation Required)
             # Store indexed_bases, ast_visitor, observable_names
             # Create AstToSympyConverter with indexed_bases.all_symbols
             # Extract assignments and return_values from ast_visitor
             # Convert observable_names to set for fast lookup
     ```
   - Integration: Core component for Section 3, used by FunctionParser

2. **Implement build_equations() Method**
   - File: src/cubie/odesystems/symbolic/parsing/equation_constructor.py
   - Action: Modify
   - Details:
     ```python
     def build_equations(self) -> ParsedEquations:
         """Build ParsedEquations from AST analysis.
         
         Returns
         -------
         ParsedEquations
             Equations partitioned into state derivatives, observables,
             and auxiliaries.
             
         Raises
         ------
         ValueError
             If return count doesn't match state count.
         """
         # 1. Call _build_derivative_equations()
         # 2. Call _build_observable_equations()
         # 3. Call _build_auxiliary_equations()
         # 4. Combine all equations preserving topological order
         # 5. Return ParsedEquations.from_equations(equations, indexed_bases)
     ```
   - Edge cases: Empty observables list, no auxiliaries, single-state system
   - Integration: Entry point called by FunctionParser

3. **Implement _build_derivative_equations() Method**
   - File: src/cubie/odesystems/symbolic/parsing/equation_constructor.py
   - Action: Modify
   - Details:
     ```python
     def _build_derivative_equations(self) -> List[Tuple[sp.Symbol, sp.Expr]]:
         """Extract derivative equations from return statement.
         
         Returns
         -------
         list of tuple
             List of (derivative_symbol, expression) tuples.
             
         Raises
         ------
         ValueError
             If return value count doesn't match state count.
         """
         # 1. Get return_node from ast_visitor
         # 2. Extract return value elements (handle List, Tuple, Dict, Name)
         # 3. Get state derivative symbols from indexed_bases.dxdt.ref_map
         # 4. Check len(return_values) == len(state_symbols)
         # 5. For each (return_expr, deriv_symbol):
         #    - Convert AST to SymPy using _convert_ast_to_sympy()
         #    - Create tuple (deriv_symbol, sympy_expr)
         # 6. Return list of tuples
     ```
   - Edge cases: Return statement missing, return count mismatch, Dict return with name mapping
   - Integration: Generates equations for ParsedEquations.state_derivatives

4. **Implement _build_observable_equations() Method**
   - File: src/cubie/odesystems/symbolic/parsing/equation_constructor.py
   - Action: Modify
   - Details:
     ```python
     def _build_observable_equations(self) -> List[Tuple[sp.Symbol, sp.Expr]]:
         """Extract observable equations from assignments.
         
         Returns
         -------
         list of tuple
             List of (observable_symbol, expression) tuples.
             
         Raises
         ------
         ValueError
             If user-specified observable not assigned in function.
         """
         # 1. Get observable symbols from indexed_bases.observables.ref_map
         # 2. For each observable_name in self.observable_names:
         #    - Check if assignment exists in self.assignments
         #    - If not found: raise ValueError with clear message
         #    - Convert RHS AST to SymPy
         #    - Get corresponding symbol from indexed_bases
         #    - Create tuple (observable_symbol, sympy_expr)
         # 3. Return list of tuples
     ```
   - Edge cases: Observable not assigned, empty observable list
   - Integration: Generates equations for ParsedEquations.observables

5. **Implement _build_auxiliary_equations() Method**
   - File: src/cubie/odesystems/symbolic/parsing/equation_constructor.py
   - Action: Modify
   - Details:
     ```python
     def _build_auxiliary_equations(self) -> List[Tuple[sp.Symbol, sp.Expr]]:
         """Build equations for intermediate variables.
         
         Intermediate variables are assignments that are neither state
         derivatives nor observables.
         
         Returns
         -------
         list of tuple
             List of (auxiliary_symbol, expression) tuples.
         """
         # 1. Identify auxiliary assignments:
         #    - Not in derivative names (dv, dx, etc.)
         #    - Not in observable_names
         #    - Not state access assignments (v = y[0])
         # 2. For each auxiliary assignment:
         #    - Get or create auxiliary symbol
         #    - Convert RHS to SymPy
         #    - Create tuple
         # 3. Return list preserving function body order
     ```
   - Edge cases: No auxiliaries, auxiliaries used in observables
   - Integration: Generates equations for ParsedEquations.auxiliaries

6. **Implement _convert_ast_to_sympy() Helper Method**
   - File: src/cubie/odesystems/symbolic/parsing/equation_constructor.py
   - Action: Modify
   - Details:
     ```python
     def _convert_ast_to_sympy(
         self, ast_expr: ast.expr, context: str
     ) -> sp.Expr:
         """Convert AST expression to SymPy with error context.
         
         Parameters
         ----------
         ast_expr
             AST expression node to convert.
         context
             Description of where expression appears (for error messages).
             
         Returns
         -------
         sp.Expr
             SymPy symbolic expression.
             
         Raises
         ------
         ValueError
             If conversion fails (with context in message).
         """
         # Wrap self.converter.convert(ast_expr) with try-except
         # Add context to error message:
         # f"Failed to convert expression for {context}: {original_error}"
     ```
   - Edge cases: Unsupported AST node types, unknown functions
   - Integration: Used by all _build_*_equations() methods

7. **Implement _get_equation_order() Helper Method**
   - File: src/cubie/odesystems/symbolic/parsing/equation_constructor.py
   - Action: Modify
   - Details:
     ```python
     def _get_equation_order(
         self, equations: List[Tuple[sp.Symbol, sp.Expr]]
     ) -> List[Tuple[sp.Symbol, sp.Expr]]:
         """Order equations to respect dependencies.
         
         Preserves function body order where possible, but ensures
         auxiliaries appear before they're used in derivatives/observables.
         
         Parameters
         ----------
         equations
             Unordered equations.
             
         Returns
         -------
         list of tuple
             Equations in safe evaluation order.
         """
         # Simple approach: preserve function body order
         # (Python evaluates top-to-bottom, so order is already safe)
         # Future: implement dependency-based ordering if needed
         return equations
     ```
   - Edge cases: Circular dependencies (should not occur in valid functions)
   - Integration: Called in build_equations() before creating ParsedEquations

**Tests to Create**:
- Test file: tests/odesystems/symbolic/parsing/test_equation_constructor.py
- Test function: test_build_derivative_equations_list_return
  - Description: Verify derivatives extracted from list return statement
- Test function: test_build_derivative_equations_dict_return
  - Description: Verify derivatives extracted from dict return statement
- Test function: test_build_observable_equations_valid
  - Description: Verify observables extracted from assignments
- Test function: test_build_observable_not_found_error
  - Description: Verify error when observable not assigned
- Test function: test_build_auxiliary_equations
  - Description: Verify intermediates identified as auxiliaries
- Test function: test_return_count_mismatch_error
  - Description: Verify error when return count != state count
- Test function: test_build_equations_integration
  - Description: Full workflow from AST to ParsedEquations

**Outcomes**:
[Empty - to be filled by taskmaster agent]

---

## Task Group 2: FunctionParser - Main Orchestrator
**Status**: [ ]
**Dependencies**: Group 1 (EquationConstructor must exist)

**Required Context**:
- File: src/cubie/odesystems/symbolic/parsing/equation_constructor.py (entire file created in Group 1)
- File: src/cubie/odesystems/symbolic/parsing/parser.py (lines 1-150: parse_input signature, TIME_SYMBOL)
- File: src/cubie/odesystems/symbolic/sym_utils.py (hash_system_definition function)
- File: .github/active_plans/python_function_parser/section1_source_code/agent_plan.md (FunctionInspector, AstVisitor interfaces)
- File: .github/active_plans/python_function_parser/section2_variable_parsing/agent_plan.md (VariableClassifier interface)

**Input Validation Required**:
- func: Check is callable (not lambda, not builtin)
- states: Check is list/dict or None
- parameters: Check is list/dict or None
- constants: Check is list/dict or None
- observables: Check is list or None
- strict: Check is bool

**Tasks**:

1. **Create FunctionParser Module**
   - File: src/cubie/odesystems/symbolic/parsing/function_parser.py
   - Action: Create
   - Details:
     ```python
     from typing import Callable, Dict, List, Optional, Tuple, Union
     import sympy as sp
     
     from ..indexedbasemaps import IndexedBases
     from .parser import ParsedEquations, TIME_SYMBOL
     from .equation_constructor import EquationConstructor
     from ..sym_utils import hash_system_definition
     # Import FunctionInspector, AstVisitor from Section 1
     # Import VariableClassifier from Section 2
     
     class FunctionParser:
         """Parse Python function into symbolic ODE representation.
         
         Main orchestrator coordinating Sections 1, 2, and 3 components
         to produce output compatible with parse_input().
         
         Parameters
         ----------
         func : Callable
             User-provided ODE function.
         states : optional
             User-specified state names or defaults.
         parameters : optional
             User-specified parameter names or defaults.
         constants : optional
             User-specified constant names or defaults.
         observables : optional
             User-specified observable names.
         drivers : optional
             User-specified driver names.
         user_functions : optional
             User-defined functions for expressions.
         strict : bool
             Strict validation mode.
         state_units, parameter_units, constant_units, observable_units,
         driver_units : optional
             Unit specifications for symbol types.
         """
         
         def __init__(
             self,
             func: Callable,
             states=None,
             parameters=None,
             constants=None,
             observables=None,
             drivers=None,
             user_functions=None,
             strict=False,
             state_units=None,
             parameter_units=None,
             constant_units=None,
             observable_units=None,
             driver_units=None
         ):
             # Validate func is callable (as specified in Input Validation)
             # Store all user inputs
             # Initialize component attributes to None (built in parse())
     ```
   - Integration: Main entry point for function-based parsing

2. **Implement parse() Method**
   - File: src/cubie/odesystems/symbolic/parsing/function_parser.py
   - Action: Modify
   - Details:
     ```python
     def parse(self) -> Tuple[IndexedBases, Dict[str, object], 
                              Dict[str, Callable], ParsedEquations, str]:
         """Execute full parsing workflow.
         
         Returns same 5-tuple as parse_input() for string/SymPy input:
         - IndexedBases: Symbol collections
         - all_symbols: Dict mapping names to symbols/callables
         - callables_dict: User functions
         - ParsedEquations: Partitioned equations
         - fn_hash: System definition hash
         
         Returns
         -------
         tuple
             5-tuple matching parse_input() output format.
         """
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
             state_param_name=self.inspector.param_names[1]
         )
         parsed_equations = self.constructor.build_equations()
         
         # Build output components
         all_symbols = self._build_symbol_dict(indexed_bases)
         callables_dict = self.user_functions or {}
         fn_hash = self._compute_hash(parsed_equations, indexed_bases)
         
         return (indexed_bases, all_symbols, callables_dict, 
                 parsed_equations, fn_hash)
     ```
   - Edge cases: Missing return statement, no parameters, no observables
   - Integration: Coordinates all components, returns to parse_input()

3. **Implement _build_symbol_dict() Method**
   - File: src/cubie/odesystems/symbolic/parsing/function_parser.py
   - Action: Modify
   - Details:
     ```python
     def _build_symbol_dict(
         self, indexed_bases: IndexedBases
     ) -> Dict[str, object]:
         """Build comprehensive symbol dictionary.
         
         Matches string parser output format: all symbols, time symbol,
         and user functions.
         
         Parameters
         ----------
         indexed_bases
             Symbol collections.
             
         Returns
         -------
         dict
             Symbol name -> symbol/callable mapping.
         """
         # Start with indexed_bases.all_symbols.copy()
         # Add TIME_SYMBOL under 't' key
         # Add user_functions if provided
         # Return complete dictionary
     ```
   - Edge cases: No user functions, conflicting names
   - Integration: Provides symbols for downstream code generation

4. **Implement _compute_hash() Method**
   - File: src/cubie/odesystems/symbolic/parsing/function_parser.py
   - Action: Modify
   - Details:
     ```python
     def _compute_hash(
         self, equations: ParsedEquations, indexed_bases: IndexedBases
     ) -> str:
         """Compute stable hash for system identification.
         
         Uses hash_system_definition() from sym_utils.
         
         Parameters
         ----------
         equations
             Parsed equations.
         indexed_bases
             Symbol collections.
             
         Returns
         -------
         str
             Hash string for system.
         """
         # Call hash_system_definition(
         #     equations,
         #     indexed_bases.constants.default_values,
         #     observable_labels=list(indexed_bases.observables.ref_map.keys())
         # )
         # Return hash string
     ```
   - Edge cases: No constants, no observables
   - Integration: Provides cache key for compiled systems

5. **Implement validate_consistency() Method**
   - File: src/cubie/odesystems/symbolic/parsing/function_parser.py
   - Action: Modify
   - Details:
     ```python
     def validate_consistency(self) -> None:
         """Validate user specifications against inferred structure.
         
         Called during parse() after classification to check:
         - User states match inferred count
         - User observables exist in function
         - User parameters accessible in function
         - No conflicts between categories
         
         Raises
         ------
         ValueError
             If user specifications inconsistent with function.
         """
         # If user provided states, check inferred states match count
         # If user provided parameters, check they exist
         # If user provided observables, check assignments exist
         # Warn for mismatches, error for missing variables
     ```
   - Edge cases: User overrides all inferred values
   - Integration: Called by parse() after classification

6. **Add Error Handling to parse() Method**
   - File: src/cubie/odesystems/symbolic/parsing/function_parser.py
   - Action: Modify
   - Details:
     ```python
     # Wrap each phase in try-except to provide context:
     try:
         self.inspector = FunctionInspector(self.func)
     except Exception as e:
         raise type(e)(f"Error inspecting function: {e}") from e
     
     try:
         self.visitor = AstVisitor(...)
         self.visitor.visit()
     except Exception as e:
         raise type(e)(f"Error analyzing function AST: {e}") from e
     
     try:
         self.classifier = VariableClassifier(...)
         self.classifier.classify()
     except Exception as e:
         raise type(e)(f"Error classifying variables: {e}") from e
     
     try:
         self.constructor = EquationConstructor(...)
         parsed_equations = self.constructor.build_equations()
     except Exception as e:
         raise type(e)(f"Error building equations: {e}") from e
     ```
   - Edge cases: Any phase can fail with various exceptions
   - Integration: Provides clear error context for debugging

**Tests to Create**:
- Test file: tests/odesystems/symbolic/parsing/test_function_parser.py
- Test function: test_parse_simple_function
  - Description: Basic function with list return parsed correctly
- Test function: test_parse_with_constants
  - Description: Function with constant arguments parsed correctly
- Test function: test_parse_with_parameters
  - Description: Parameter promotion handled correctly
- Test function: test_parse_with_observables
  - Description: Observable assignments extracted correctly
- Test function: test_build_symbol_dict
  - Description: Symbol dictionary matches expected format
- Test function: test_compute_hash_stability
  - Description: Hash stable for same system definition
- Test function: test_validate_consistency_state_mismatch
  - Description: Warning/error when user states don't match inferred
- Test function: test_error_context_wrapping
  - Description: Exceptions wrapped with phase context

**Outcomes**:
[Empty - to be filled by taskmaster agent]

---

## Task Group 3: parse_input() Integration
**Status**: [ ]
**Dependencies**: Groups 1, 2 (FunctionParser must be complete)

**Required Context**:
- File: src/cubie/odesystems/symbolic/parsing/parser.py (lines 42-102: _detect_input_type function)
- File: src/cubie/odesystems/symbolic/parsing/parser.py (lines 700-900: parse_input function signature and body)
- File: src/cubie/odesystems/symbolic/parsing/function_parser.py (entire file from Group 2)

**Input Validation Required**:
- None (validation delegated to FunctionParser)

**Tasks**:

1. **Modify _detect_input_type() for Callable Support**
   - File: src/cubie/odesystems/symbolic/parsing/parser.py
   - Action: Modify
   - Details:
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
         
         # Check if callable BEFORE iterable check
         # (some callables may be iterable)
         if callable(dxdt):
             return "function"
         
         # ... existing string/sympy detection unchanged ...
     ```
   - Old code:
     ```python
     def _detect_input_type(dxdt: Union[str, Iterable]) -> str:
     ```
   - Edge cases: Callable that is also iterable, lambda functions
   - Integration: Routes to function parser when callable detected

2. **Update parse_input() Signature for Callable**
   - File: src/cubie/odesystems/symbolic/parsing/parser.py
   - Action: Modify
   - Details:
     ```python
     def parse_input(
         dxdt: Union[str, Iterable[str], Callable],  # Add Callable
         states: Optional[Union[Dict[str, float], Iterable[str]]] = None,
         # ... rest unchanged ...
     ) -> Tuple[IndexedBases, Dict[str, object], Dict[str, Callable], 
                ParsedEquations, str]:
     ```
   - Old code: Type hint missing `Callable` option
   - Edge cases: None
   - Integration: Documents callable support in signature

3. **Add Function Routing to parse_input()**
   - File: src/cubie/odesystems/symbolic/parsing/parser.py
   - Action: Modify
   - Details:
     ```python
     # After input_type detection, add function routing:
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
         
         return parser.parse()
     
     elif input_type == "string":
         # ... existing string parsing unchanged ...
         
     elif input_type == "sympy":
         # ... existing sympy parsing unchanged ...
     ```
   - Edge cases: FunctionParser raises exception
   - Integration: Routes callable to function parser, returns results directly

4. **Add Import Statement**
   - File: src/cubie/odesystems/symbolic/parsing/parser.py
   - Action: Modify
   - Details:
     ```python
     from typing import (
         Any,
         Callable,  # Add Callable if not present
         Dict,
         Iterable,
         List,
         Optional,
         Sequence,
         Tuple,
         Union,
     )
     ```
   - Edge cases: None
   - Integration: Enables Callable type hint

**Tests to Create**:
- Test file: tests/odesystems/symbolic/parsing/test_parser_integration.py
- Test function: test_detect_input_type_function
  - Description: Verify callable detected as "function"
- Test function: test_parse_input_routes_to_function_parser
  - Description: Verify callable routed to FunctionParser
- Test function: test_function_parser_output_format
  - Description: Verify FunctionParser returns correct 5-tuple
- Test function: test_backward_compatibility_string
  - Description: Verify string parsing still works
- Test function: test_backward_compatibility_sympy
  - Description: Verify SymPy parsing still works

**Outcomes**:
[Empty - to be filled by taskmaster agent]

---

## Task Group 4: FunctionValidator - Validation Utilities
**Status**: [ ]
**Dependencies**: Groups 1, 2, 3 (for testing context)

**Required Context**:
- File: src/cubie/odesystems/symbolic/parsing/function_parser.py (entire file)
- File: .github/active_plans/python_function_parser/section1_source_code/agent_plan.md (FunctionInspector interface)
- File: .github/active_plans/python_function_parser/section2_variable_parsing/agent_plan.md (VariableClassifier interface)

**Input Validation Required**:
- None (validation functions validate their own inputs)

**Tasks**:

1. **Create FunctionValidator Module with Error Templates**
   - File: src/cubie/odesystems/symbolic/parsing/function_validator.py
   - Action: Create
   - Details:
     ```python
     """Centralized validation utilities for function parsing."""
     from typing import Callable, Dict, List
     import ast
     from ..indexedbasemaps import IndexedBases
     # Import types from Sections 1 & 2
     
     # Error message templates
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
   - Integration: Provides consistent error messages across validators

2. **Implement validate_function_signature()**
   - File: src/cubie/odesystems/symbolic/parsing/function_validator.py
   - Action: Modify
   - Details:
     ```python
     def validate_function_signature(
         func: Callable, inspector: FunctionInspector
     ) -> None:
         """Validate function has acceptable ODE signature.
         
         Checks parameter count and names against ODE conventions.
         
         Parameters
         ----------
         func
             User-provided function.
         inspector
             FunctionInspector with extracted signature.
             
         Raises
         ------
         ValueError
             If signature invalid (< 2 params).
             
         Warnings
         ---------
         Warns if parameter names unconventional.
         """
         # Check at least 2 parameters
         # First parameter conventionally 't' (warn if not)
         # Second parameter conventionally 'y' or 'state' (warn if not)
     ```
   - Edge cases: Exactly 2 params, many params, unconventional names
   - Integration: Can be called by FunctionParser during validation

3. **Implement validate_return_statement()**
   - File: src/cubie/odesystems/symbolic/parsing/function_validator.py
   - Action: Modify
   - Details:
     ```python
     def validate_return_statement(
         visitor: AstVisitor, indexed_bases: IndexedBases
     ) -> None:
         """Validate return statement structure.
         
         Checks return statement exists and has correct value count.
         
         Parameters
         ----------
         visitor
             AST visitor with return_node.
         indexed_bases
             Symbol collections with state count.
             
         Raises
         ------
         ValueError
             If return missing or count mismatch.
         """
         # Check return_node exists
         # Count return values
         # Compare to len(indexed_bases.states.ref_map)
         # Raise with RETURN_MISSING or RETURN_COUNT_MISMATCH template
     ```
   - Edge cases: Missing return, single return value, dict return
   - Integration: Can be called by EquationConstructor

4. **Implement validate_observables()**
   - File: src/cubie/odesystems/symbolic/parsing/function_validator.py
   - Action: Modify
   - Details:
     ```python
     def validate_observables(
         observable_names: List[str], 
         assignments: Dict[str, ast.expr]
     ) -> None:
         """Validate all user-specified observables are assigned.
         
         Parameters
         ----------
         observable_names
             User-specified observable names.
         assignments
             Function body assignments.
             
         Raises
         ------
         ValueError
             If observable not assigned in function.
         """
         # For each name in observable_names:
         #   Check if name in assignments
         #   If not: raise ValueError with OBSERVABLE_NOT_FOUND template
     ```
   - Edge cases: Empty observable list, name typo
   - Integration: Can be called by EquationConstructor

5. **Implement validate_variable_access()**
   - File: src/cubie/odesystems/symbolic/parsing/function_validator.py
   - Action: Modify
   - Details:
     ```python
     def validate_variable_access(
         access_patterns: Dict[str, List[AccessPattern]]
     ) -> None:
         """Validate state access patterns are consistent.
         
         Checks no mixed indexing types (int vs str vs attribute).
         
         Parameters
         ----------
         access_patterns
             Access patterns from AST visitor.
             
         Raises
         ------
         ValueError
             If mixed access patterns detected.
         """
         # Extract pattern types
         # Check all same type
         # If mixed: raise ValueError with INCONSISTENT_ACCESS template
     ```
   - Edge cases: No accesses, single access type
   - Integration: Can be called by VariableClassifier

6. **Implement validate_user_specifications()**
   - File: src/cubie/odesystems/symbolic/parsing/function_validator.py
   - Action: Modify
   - Details:
     ```python
     def validate_user_specifications(
         classifier: VariableClassifier
     ) -> None:
         """Validate user-provided specs match inferred structure.
         
         Checks user states/parameters/constants are accessible
         and don't conflict.
         
         Parameters
         ----------
         classifier
             Variable classifier with user specs and inferred vars.
             
         Raises
         ------
         ValueError
             If user spec conflicts with function structure.
         """
         # If user provided states, check match inferred count
         # Check parameters accessible
         # Check no name conflicts between categories
     ```
   - Edge cases: User overrides everything, no user specs
   - Integration: Can be called by FunctionParser

**Tests to Create**:
- Test file: tests/odesystems/symbolic/parsing/test_function_validator.py
- Test function: test_validate_function_signature_valid
  - Description: Valid signature passes
- Test function: test_validate_function_signature_insufficient_params
  - Description: < 2 params raises error
- Test function: test_validate_return_statement_missing
  - Description: Missing return raises error
- Test function: test_validate_return_count_mismatch
  - Description: Wrong return count raises error
- Test function: test_validate_observables_not_found
  - Description: Unassigned observable raises error
- Test function: test_validate_variable_access_mixed
  - Description: Mixed access patterns raise error
- Test function: test_validate_user_specifications_conflict
  - Description: Conflicting user specs raise error

**Outcomes**:
[Empty - to be filled by taskmaster agent]

---

## Task Group 5: Integration Tests and Documentation
**Status**: [ ]
**Dependencies**: Groups 1, 2, 3, 4 (all components must be complete)

**Required Context**:
- File: src/cubie/odesystems/symbolic/symbolicODE.py (create_ODE_system function)
- File: src/cubie/odesystems/symbolic/parsing/parser.py (entire file)
- File: src/cubie/odesystems/symbolic/parsing/function_parser.py (entire file)
- File: tests/system_fixtures.py (existing test fixtures)

**Input Validation Required**:
- None (tests validate components)

**Tasks**:

1. **Create Equivalence Test Suite**
   - File: tests/odesystems/symbolic/parsing/test_function_equivalence.py
   - Action: Create
   - Details:
     ```python
     """Test function-based systems produce same output as string-based."""
     import pytest
     import numpy as np
     from cubie import create_ODE_system
     
     def test_simple_linear_equivalence():
         """Function and string parsers produce equivalent systems."""
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
         
         # Compare ParsedEquations structure
         assert len(system_func.equations) == len(system_str.equations)
         for eq_func, eq_str in zip(system_func.equations, 
                                     system_str.equations):
             assert eq_func[0] == eq_str[0]  # LHS symbols match
             assert eq_func[1].equals(eq_str[1])  # RHS equivalent
         
         # Compare IndexedBases
         assert (system_func.indices.states.length == 
                 system_str.indices.states.length)
     ```
   - Edge cases: Different access patterns, with constants, with observables
   - Integration: Validates function parser produces correct output

2. **Create Access Pattern Tests**
   - File: tests/odesystems/symbolic/parsing/test_function_access_patterns.py
   - Action: Create
   - Details:
     ```python
     """Test different state access patterns."""
     import pytest
     from cubie import create_ODE_system
     
     def test_integer_indexing():
         """Integer indexing y[0], y[1] parsed correctly."""
         def ode_func(t, y):
             return [-0.1 * y[0], y[0]]
         
         system = create_ODE_system(
             dxdt=ode_func,
             states=["velocity", "position"]
         )
         # Verify state symbols created
         # Verify equations correct
     
     def test_string_indexing():
         """String indexing y["name"] parsed correctly."""
         def ode_func(t, y):
             v = y["velocity"]
             return {"velocity": -0.1 * v, "position": v}
         
         system = create_ODE_system(
             dxdt=ode_func,
             states={"velocity": 1.0, "position": 0.0}
         )
         # Verify parsing succeeded
     
     def test_attribute_access():
         """Attribute access y.attr parsed correctly."""
         def ode_func(t, y):
             return [y.velocity * -0.1, y.velocity]
         
         system = create_ODE_system(
             dxdt=ode_func,
             states=["velocity", "position"]
         )
         # Verify parsing succeeded
     
     def test_mixed_access_error():
         """Mixed access patterns raise error."""
         def ode_func(t, y):
             v = y[0]
             x = y["position"]  # Mixed!
             return [-0.1 * v, v]
         
         with pytest.raises(ValueError, match="Inconsistent"):
             create_ODE_system(dxdt=ode_func, states=["v", "x"])
     ```
   - Edge cases: All three access types, mixed patterns
   - Integration: Validates Section 1 AST visitor and Section 2 name generator

3. **Create Observable Tests**
   - File: tests/odesystems/symbolic/parsing/test_function_observables.py
   - Action: Create
   - Details:
     ```python
     """Test observable extraction from functions."""
     import pytest
     from cubie import create_ODE_system
     
     def test_observable_extraction():
         """Observable assigned in function extracted correctly."""
         def ode_func(t, y, m):
             v = y[0]
             kinetic_energy = 0.5 * m * v**2
             return [-0.1 * v / m]
         
         system = create_ODE_system(
             dxdt=ode_func,
             states=["velocity"],
             parameters={"m": 1.0},
             observables=["kinetic_energy"]
         )
         
         # Verify observable in indexed_bases
         assert "kinetic_energy" in system.indices.observables.symbol_map
         # Verify observable equation created
         obs_eqs = [eq for eq in system.equations 
                    if eq[0] == system.indices.observables.symbol_map[
                        "kinetic_energy"]]
         assert len(obs_eqs) == 1
     
     def test_observable_not_found_error():
         """Observable not assigned raises error."""
         def ode_func(t, y):
             return [-0.1 * y[0]]
         
         with pytest.raises(ValueError, match="Observable.*not.*found"):
             create_ODE_system(
                 dxdt=ode_func,
                 states=["velocity"],
                 observables=["energy"]  # Not assigned!
             )
     ```
   - Edge cases: Multiple observables, no observables
   - Integration: Validates EquationConstructor observable extraction

4. **Update create_ODE_system() Docstring**
   - File: src/cubie/odesystems/symbolic/symbolicODE.py
   - Action: Modify
   - Details:
     ```python
     def create_ODE_system(
         dxdt: Union[str, Iterable[str], Callable],  # Add Callable
         # ... rest unchanged ...
     ) -> SymbolicODE:
         """Create a SymbolicODE from symbolic definitions.
         
         Parameters
         ----------
         dxdt
             System equations defined as:
             
             - Single string with newline-delimited equations
             - Iterable of equation strings in ``lhs = rhs`` form
             - Iterable of SymPy expressions or equalities
             - **Python function** with signature ``(t, y, ...)`` 
               returning derivatives
             
             For function input, the first parameter should be time (``t``),
             the second should be the state vector (``y``), and remaining
             parameters provide constants/parameters. The function must 
             return a list/tuple of derivative expressions matching the 
             number of states.
         
         # ... existing parameters ...
         
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
   - Old code: Missing function examples
   - Edge cases: None
   - Integration: Documents new functionality for users

5. **Create User Guide Section**
   - File: docs/user_guide/function_based_odes.rst
   - Action: Create
   - Details:
     ```rst
     Function-Based ODE Definition
     ==============================
     
     CuBIE supports defining ODE systems using Python functions,
     providing a familiar interface for users of scipy.integrate.solve_ivp
     and MATLAB's ode45.
     
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
     
     # ... more sections ...
     ```
   - Integration: Provides user-facing documentation

**Tests to Create**:
- Test file: tests/odesystems/symbolic/parsing/test_function_equivalence.py (created in task 1)
- Test file: tests/odesystems/symbolic/parsing/test_function_access_patterns.py (created in task 2)
- Test file: tests/odesystems/symbolic/parsing/test_function_observables.py (created in task 3)

**Outcomes**:
[Empty - to be filled by taskmaster agent]

---

## Summary

**Total Task Groups**: 5
**Dependency Chain**: 
- Group 1 (EquationConstructor) → Independent, uses Section 1 & 2
- Group 2 (FunctionParser) → Depends on Group 1
- Group 3 (parse_input Integration) → Depends on Groups 1, 2
- Group 4 (FunctionValidator) → Independent (can run parallel with 1-3, but tested after)
- Group 5 (Tests & Docs) → Depends on all groups

**Estimated Complexity**: Medium-High
- Section 3 ties together Sections 1 & 2
- EquationConstructor requires careful AST→SymPy conversion
- FunctionParser orchestrates multiple components
- Integration must maintain backward compatibility

**Tests to Create**: ~25 test functions across 7 test files
- Comprehensive coverage of equation building, parsing, validation
- Equivalence tests ensure function parser matches string parser
- Access pattern tests validate all three state access styles
- Observable tests verify extraction and validation

**Critical Success Criteria**:
1. ParsedEquations structure matches string parser output exactly
2. IndexedBases categorization identical to string parser
3. All existing string/SymPy tests pass unchanged
4. Function-based systems integrate seamlessly with batch solver
5. Error messages provide clear guidance for function structure issues
