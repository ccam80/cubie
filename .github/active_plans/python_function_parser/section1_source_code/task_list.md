# Implementation Task List
# Feature: Python Function Parser - Section 1 (Source Code Interactions)
# Plan Reference: .github/active_plans/python_function_parser/section1_source_code/agent_plan.md

## Task Group 1: Function Inspector Module
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: .github/active_plans/python_function_parser/section1_source_code/agent_plan.md (lines 1-136)
- File: .github/context/cubie_internal_structure.md (entire file)
- File: .github/copilot-instructions.md (entire file)

**Input Validation Required**:
- func parameter in `__init__`: Check `callable(func)` - if False, raise TypeError with message "func must be callable, got {type(func).__name__}"
- No additional validation needed in private methods (internal use only)

**Tasks**:
1. **Create FunctionInspector module**
   - File: src/cubie/odesystems/symbolic/parsing/function_inspector.py
   - Action: Create
   - Details:
     ```python
     """Extract function metadata for ODE parsing."""
     
     import ast
     import inspect
     from typing import Callable, Dict, List, Tuple, Any, Optional
     
     
     class FunctionInspector:
         """Extract function signature, source code, and AST.
         
         Parameters
         ----------
         func : Callable
             The user-provided function to inspect.
             
         Attributes
         ----------
         func : Callable
             The user-provided function.
         signature : inspect.Signature
             Extracted function signature.
         source : str
             Function source code.
         ast_tree : ast.Module
             Parsed AST tree.
         func_def : ast.FunctionDef
             Function definition node from AST.
         param_names : List[str]
             Ordered list of parameter names.
             
         Raises
         ------
         TypeError
             If func is not callable, is a lambda, or is a builtin.
         OSError
             If source code cannot be retrieved (REPL/Jupyter).
         """
         
         def __init__(self, func: Callable) -> None:
             """Initialize inspector and extract all metadata.
             
             Parameters
             ----------
             func : Callable
                 The function to inspect.
                 
             Raises
             ------
             TypeError
                 If func is not callable, lambda, or builtin.
             OSError
                 If source unavailable (REPL/Jupyter).
             """
             # Validation as specified in Input Validation Required
             if not callable(func):
                 raise TypeError(
                     f"func must be callable, got {type(func).__name__}"
                 )
             
             self.func = func
             self._validate_function()
             self.signature = self._extract_signature()
             self.source = self._extract_source()
             self.ast_tree, self.func_def = self._parse_ast()
             
         def _validate_function(self) -> None:
             """Validate function is supported type.
             
             Checks for lambda, builtin, and method functions.
             Raises appropriate errors with actionable messages.
             
             Raises
             ------
             TypeError
                 If lambda or builtin detected.
             """
             # Check lambda
             if self.func.__name__ == '<lambda>':
                 raise TypeError(
                     "Lambda functions are not supported. Please use 'def' "
                     "syntax to define your ODE function."
                 )
             
             # Check builtin - try to get source
             try:
                 inspect.getsource(self.func)
             except TypeError:
                 raise TypeError(
                     "Cannot parse builtin or C-extension functions. Please "
                     "use a regular Python function."
                 )
         
         def _extract_signature(self) -> inspect.Signature:
             """Extract function signature with parameter metadata.
             
             Returns
             -------
             inspect.Signature
                 Signature object containing parameter information.
             """
             sig = inspect.signature(self.func)
             self.param_names = list(sig.parameters.keys())
             return sig
         
         def _extract_source(self) -> str:
             """Retrieve function source code.
             
             Returns
             -------
             str
                 Clean function source code.
                 
             Raises
             ------
             OSError
                 If source unavailable (REPL/Jupyter).
             """
             try:
                 source = inspect.getsource(self.func)
             except OSError:
                 raise OSError(
                     "Cannot retrieve source code for functions defined in "
                     "interactive mode (REPL/Jupyter). Please define the "
                     "function in a .py file or use string-based equation input."
                 )
             return source.strip()
         
         def _parse_ast(self) -> Tuple[ast.Module, ast.FunctionDef]:
             """Parse source into AST tree.
             
             Returns
             -------
             tuple
                 (ast.Module, ast.FunctionDef) - module and function definition.
                 
             Raises
             ------
             ValueError
                 If AST structure unexpected.
             """
             tree = ast.parse(self.source)
             
             if not tree.body or not isinstance(tree.body[0], ast.FunctionDef):
                 raise ValueError(
                     f"Expected function definition in source, found "
                     f"{type(tree.body[0]).__name__ if tree.body else 'empty'}"
                 )
             
             func_def = tree.body[0]
             return tree, func_def
         
         def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
             """Return detailed parameter information.
             
             Returns
             -------
             dict
                 Parameter metadata: {name: {kind, default, annotation}}
             """
             param_info = {}
             for name, param in self.signature.parameters.items():
                 param_info[name] = {
                     'kind': str(param.kind),
                     'default': param.default if param.default is not inspect.Parameter.empty else None,
                     'annotation': param.annotation if param.annotation is not inspect.Parameter.empty else None,
                 }
             return param_info
         
         def validate_ode_signature(self) -> None:
             """Validate function signature matches ODE conventions.
             
             Checks for minimum 2 parameters (time, state).
             Warns if unconventional names used.
             
             Raises
             ------
             ValueError
                 If less than 2 parameters.
             """
             if len(self.param_names) < 2:
                 raise ValueError(
                     f"ODE function must have at least 2 parameters (time, state). "
                     f"Found {len(self.param_names)} parameter(s): {self.param_names}. "
                     f"Expected signature: def f(t, y, ...)"
                 )
             
             # Warn about unconventional names (non-blocking)
             first_param = self.param_names[0]
             second_param = self.param_names[1]
             
             if first_param != 't':
                 import warnings
                 warnings.warn(
                     f"Expected first parameter to be 't' (time), found '{first_param}'. "
                     f"This will be treated as the time variable.",
                     UserWarning
                 )
             
             if second_param not in ('y', 'state'):
                 import warnings
                 warnings.warn(
                     f"Expected second parameter to be 'y' or 'state', found '{second_param}'. "
                     f"This will be treated as the state vector.",
                     UserWarning
                 )
     
     
     def is_lambda(func: Callable) -> bool:
         """Check if function is a lambda.
         
         Parameters
         ----------
         func : Callable
             Function to check.
             
         Returns
         -------
         bool
             True if lambda, False otherwise.
         """
         return func.__name__ == '<lambda>'
     
     
     def is_builtin(func: Callable) -> bool:
         """Check if function is builtin or C-extension.
         
         Parameters
         ----------
         func : Callable
             Function to check.
             
         Returns
         -------
         bool
             True if builtin, False otherwise.
         """
         try:
             inspect.getsource(func)
             return False
         except (TypeError, OSError):
             return True
     
     
     def get_function_name(func: Callable) -> str:
         """Get function name for error messages.
         
         Parameters
         ----------
         func : Callable
             Function to get name from.
             
         Returns
         -------
         str
             Function name.
         """
         return func.__name__
     ```
   - Edge cases:
     - Lambda: Raise TypeError immediately in `_validate_function`
     - Builtin: Raise TypeError in `_validate_function`
     - REPL function: Raise OSError in `_extract_source`
     - Method (has self): Allow with warning in `validate_ode_signature`
     - Less than 2 params: Raise ValueError in `validate_ode_signature`
   - Integration:
     - Will be used by FunctionParser (created in Section 3)
     - Provides foundation for AST analysis

**Tests to Create**:
- Test file: tests/odesystems/symbolic/test_function_inspector.py
- Test function: test_extract_signature_basic
- Description: Regular function with 2 params extracts signature correctly
- Test function: test_extract_signature_with_defaults
- Description: Function with default values preserves defaults in signature
- Test function: test_extract_source_success
- Description: Source retrieval from file-defined function
- Test function: test_reject_lambda
- Description: Lambda raises TypeError with appropriate message
- Test function: test_reject_builtin
- Description: Builtin function raises TypeError
- Test function: test_parse_ast_success
- Description: AST parsing creates Module with FunctionDef
- Test function: test_validate_ode_signature_valid
- Description: Function with 2+ params passes validation
- Test function: test_validate_ode_signature_insufficient_params
- Description: Function with <2 params raises ValueError
- Test function: test_get_parameter_info
- Description: Returns correct parameter metadata dict
- Test function: test_is_lambda_utility
- Description: is_lambda() correctly identifies lambda functions
- Test function: test_is_builtin_utility
- Description: is_builtin() correctly identifies builtin functions

**Outcomes**:
[Empty - to be filled by taskmaster agent]

---

## Task Group 2: AST Visitor Module
**Status**: [ ]
**Dependencies**: Group 1

**Required Context**:
- File: .github/active_plans/python_function_parser/section1_source_code/agent_plan.md (lines 137-310)
- File: src/cubie/odesystems/symbolic/parsing/function_inspector.py (entire file)

**Input Validation Required**:
- state_param in `__init__`: Check `isinstance(state_param, str)` and `len(state_param) > 0`
- constant_params in `__init__`: Check `isinstance(constant_params, list)` and all items are strings
- No subscript validation beyond checking for constant indices (non-constant subscripts ignored, not errored)

**Tasks**:
1. **Create AST Visitor module**
   - File: src/cubie/odesystems/symbolic/parsing/ast_visitor.py
   - Action: Create
   - Details:
     ```python
     """AST traversal for identifying variable access patterns."""
     
     import ast
     from typing import Dict, List, Optional, Set, Union, TypedDict
     
     
     class AccessPattern(TypedDict):
         """Describes a variable access pattern in the AST.
         
         Attributes
         ----------
         base : str
             Variable name being accessed (e.g., 'y', 'constants').
         key : Union[int, str]
             Index or attribute name.
         pattern_type : str
             One of: 'subscript_int', 'subscript_str', 'attribute'.
         node : ast.AST
             Original AST node for error reporting.
         """
         base: str
         key: Union[int, str]
         pattern_type: str
         node: ast.AST
     
     
     class VisitorResults(TypedDict):
         """Results from AST traversal.
         
         Attributes
         ----------
         state_accesses : List[AccessPattern]
             State variable access patterns.
         constant_accesses : List[AccessPattern]
             Constant/parameter access patterns.
         assignments : Dict[str, ast.expr]
             Map of variable name to RHS expression.
         return_node : Optional[ast.Return]
             Return statement node.
         function_calls : Set[str]
             Set of function names called.
         """
         state_accesses: List[AccessPattern]
         constant_accesses: List[AccessPattern]
         assignments: Dict[str, ast.expr]
         return_node: Optional[ast.Return]
         function_calls: Set[str]
     
     
     class OdeAstVisitor(ast.NodeVisitor):
         """Traverse AST to identify variable access patterns.
         
         Parameters
         ----------
         state_param : str
             Name of state parameter (typically 'y').
         constant_params : List[str]
             Names of constant parameters.
             
         Attributes
         ----------
         state_accesses : List[AccessPattern]
             Collected state access patterns.
         constant_accesses : List[AccessPattern]
             Collected constant access patterns.
         assignments : Dict[str, ast.expr]
             Map of variable name to RHS expression.
         return_node : Optional[ast.Return]
             Return statement node.
         function_calls : Set[str]
             Set of function names called.
         """
         
         def __init__(self, state_param: str, constant_params: List[str]) -> None:
             """Initialize visitor with parameter names.
             
             Parameters
             ----------
             state_param : str
                 Name of state parameter.
             constant_params : List[str]
                 Names of constant parameters.
             """
             # Input validation as specified
             if not isinstance(state_param, str) or len(state_param) == 0:
                 raise ValueError("state_param must be non-empty string")
             if not isinstance(constant_params, list):
                 raise TypeError("constant_params must be a list")
             if not all(isinstance(p, str) for p in constant_params):
                 raise TypeError("all constant_params must be strings")
             
             self.state_param_name = state_param
             self.constant_param_names = constant_params
             self.state_accesses: List[AccessPattern] = []
             self.constant_accesses: List[AccessPattern] = []
             self.assignments: Dict[str, ast.expr] = {}
             self.return_node: Optional[ast.Return] = None
             self.function_calls: Set[str] = set()
             self._in_assignment_target = False
         
         def visit_Subscript(self, node: ast.Subscript) -> None:
             """Visit subscript node (e.g., y[0], constants["key"]).
             
             Parameters
             ----------
             node : ast.Subscript
                 Subscript node to process.
             """
             # Only track if base is a Name node
             if isinstance(node.value, ast.Name):
                 base_name = node.value.id
                 
                 # Extract subscript key if constant
                 key = extract_subscript_key(node.slice)
                 
                 if key is not None:
                     # Determine pattern type
                     if isinstance(key, int):
                         pattern_type = 'subscript_int'
                     elif isinstance(key, str):
                         pattern_type = 'subscript_str'
                     else:
                         # Skip non-int/str constants
                         self.generic_visit(node)
                         return
                     
                     # Record access if matches state or constant param
                     if base_name == self.state_param_name:
                         self.state_accesses.append({
                             'base': base_name,
                             'key': key,
                             'pattern_type': pattern_type,
                             'node': node,
                         })
                     elif base_name in self.constant_param_names:
                         self.constant_accesses.append({
                             'base': base_name,
                             'key': key,
                             'pattern_type': pattern_type,
                             'node': node,
                         })
             
             self.generic_visit(node)
         
         def visit_Attribute(self, node: ast.Attribute) -> None:
             """Visit attribute access (e.g., constants.damping).
             
             Parameters
             ----------
             node : ast.Attribute
                 Attribute node to process.
             """
             # Only track if base is a Name node
             if isinstance(node.value, ast.Name):
                 base_name = node.value.id
                 attr_name = node.attr
                 
                 # Record access if matches state or constant param
                 if base_name == self.state_param_name:
                     self.state_accesses.append({
                         'base': base_name,
                         'key': attr_name,
                         'pattern_type': 'attribute',
                         'node': node,
                     })
                 elif base_name in self.constant_param_names:
                     self.constant_accesses.append({
                         'base': base_name,
                         'key': attr_name,
                         'pattern_type': 'attribute',
                         'node': node,
                     })
             
             self.generic_visit(node)
         
         def visit_Assign(self, node: ast.Assign) -> None:
             """Visit assignment statement.
             
             Parameters
             ----------
             node : ast.Assign
                 Assignment node to process.
             """
             # Extract target (assumes single target)
             if len(node.targets) == 1:
                 target = node.targets[0]
                 if isinstance(target, ast.Name):
                     self.assignments[target.id] = node.value
             
             self.generic_visit(node)
         
         def visit_Return(self, node: ast.Return) -> None:
             """Visit return statement.
             
             Parameters
             ----------
             node : ast.Return
                 Return node to process.
             """
             if self.return_node is not None:
                 import warnings
                 warnings.warn(
                     "Multiple return statements found in function. "
                     "Only the last will be used.",
                     UserWarning
                 )
             self.return_node = node
             self.generic_visit(node)
         
         def visit_Call(self, node: ast.Call) -> None:
             """Visit function call.
             
             Parameters
             ----------
             node : ast.Call
                 Call node to process.
             """
             # Extract function name
             func_name = _get_function_name_from_call(node)
             if func_name:
                 self.function_calls.add(func_name)
             
             self.generic_visit(node)
         
         def get_results(self) -> VisitorResults:
             """Return collected data as VisitorResults.
             
             Returns
             -------
             VisitorResults
                 All collected information from traversal.
             """
             return {
                 'state_accesses': self.state_accesses,
                 'constant_accesses': self.constant_accesses,
                 'assignments': self.assignments,
                 'return_node': self.return_node,
                 'function_calls': self.function_calls,
             }
         
         def validate_consistency(self) -> None:
             """Validate access patterns are consistent.
             
             Checks for mixed int/str subscripts on same base.
             Checks for return statement.
             
             Raises
             ------
             ValueError
                 If inconsistent patterns or missing return.
             """
             # Check state access consistency
             if self.state_accesses:
                 has_int = any(
                     ap['pattern_type'] == 'subscript_int'
                     for ap in self.state_accesses
                 )
                 has_str = any(
                     ap['pattern_type'] == 'subscript_str'
                     for ap in self.state_accesses
                 )
                 
                 if has_int and has_str:
                     raise ValueError(
                         "Inconsistent state access pattern: found both integer "
                         "indexing (y[0]) and string indexing (y['name']). "
                         "Use one pattern consistently."
                     )
             
             # Check constant access consistency
             if self.constant_accesses:
                 const_bases = set(ap['base'] for ap in self.constant_accesses)
                 for base in const_bases:
                     base_accesses = [
                         ap for ap in self.constant_accesses
                         if ap['base'] == base
                     ]
                     has_int = any(
                         ap['pattern_type'] == 'subscript_int'
                         for ap in base_accesses
                     )
                     has_str = any(
                         ap['pattern_type'] == 'subscript_str'
                         for ap in base_accesses
                     )
                     
                     if has_int and has_str:
                         raise ValueError(
                             f"Inconsistent constant access pattern on '{base}': "
                             f"found both integer and string indexing. "
                             f"Use one pattern consistently."
                         )
             
             # Check return statement exists
             if self.return_node is None:
                 raise ValueError(
                     "Function must include a return statement to specify derivatives."
                 )
     
     
     def extract_subscript_key(slice_node: ast.expr) -> Union[int, str, None]:
         """Extract key from subscript slice if constant.
         
         Parameters
         ----------
         slice_node : ast.expr
             Slice node from subscript.
             
         Returns
         -------
         Union[int, str, None]
             Key if constant int/str, None otherwise.
         """
         if isinstance(slice_node, ast.Constant):
             if isinstance(slice_node.value, (int, str)):
                 return slice_node.value
         return None
     
     
     def get_access_location(node: ast.AST) -> str:
         """Get source location string for error messages.
         
         Parameters
         ----------
         node : ast.AST
             AST node.
             
         Returns
         -------
         str
             Location string like "line 5, column 12".
         """
         if hasattr(node, 'lineno') and hasattr(node, 'col_offset'):
             return f"line {node.lineno}, column {node.col_offset}"
         return "unknown location"
     
     
     def _get_function_name_from_call(node: ast.Call) -> Optional[str]:
         """Extract function name from Call node.
         
         Parameters
         ----------
         node : ast.Call
             Call node.
             
         Returns
         -------
         Optional[str]
             Function name if extractable, None otherwise.
         """
         if isinstance(node.func, ast.Name):
             return node.func.id
         elif isinstance(node.func, ast.Attribute):
             # For np.sin, just return 'sin'
             return node.func.attr
         return None
     ```
   - Edge cases:
     - Mixed int/str subscripts: Detected in `validate_consistency`, raise ValueError
     - No return statement: Detected in `validate_consistency`, raise ValueError
     - Multiple return statements: Warn in `visit_Return`, use last
     - Non-constant subscripts (y[i] with variable i): Ignored, not recorded
     - Nested subscripts (arr[i][j]): Only outer recorded
   - Integration:
     - Used by FunctionParser (Section 3) after FunctionInspector
     - Provides access patterns for symbol mapping (Section 2)

**Tests to Create**:
- Test file: tests/odesystems/symbolic/test_ast_visitor.py
- Test function: test_identify_int_subscripts
- Description: Detects y[0], y[1] patterns correctly
- Test function: test_identify_str_subscripts
- Description: Detects y["velocity"] patterns correctly
- Test function: test_identify_attribute_access
- Description: Detects constants.damping patterns correctly
- Test function: test_track_assignments
- Description: Maps variable names to RHS expressions
- Test function: test_capture_return
- Description: Stores return node correctly
- Test function: test_identify_function_calls
- Description: Collects function names from calls
- Test function: test_validate_consistency_mixed_access
- Description: Raises error for mixed int/str on same base
- Test function: test_validate_consistency_no_return
- Description: Raises error for missing return
- Test function: test_get_results
- Description: Returns complete VisitorResults dict
- Test function: test_extract_subscript_key_int
- Description: Extracts integer key from Constant node
- Test function: test_extract_subscript_key_str
- Description: Extracts string key from Constant node
- Test function: test_extract_subscript_key_none
- Description: Returns None for non-constant subscript

**Outcomes**:
[Empty - to be filled by taskmaster agent]

---

## Task Group 3: AST to SymPy Converter Module
**Status**: [ ]
**Dependencies**: Groups 1, 2

**Required Context**:
- File: .github/active_plans/python_function_parser/section1_source_code/agent_plan.md (lines 311-500)
- File: src/cubie/odesystems/symbolic/parsing/parser.py (lines 242-298)
- File: src/cubie/odesystems/symbolic/parsing/function_inspector.py (entire file)
- File: src/cubie/odesystems/symbolic/parsing/ast_visitor.py (entire file)

**Input Validation Required**:
- symbol_map in `__init__`: Check `isinstance(symbol_map, dict)`, all keys are strings, all values are `sp.Symbol` instances
- known_functions in `__init__`: Check `isinstance(known_functions, dict)`, all keys are strings, all values are callable
- time_symbol in `__init__`: Check `isinstance(time_symbol, sp.Symbol)`
- node in `convert`: No validation - will raise NotImplementedError for unsupported types

**Tasks**:
1. **Create AST to SymPy Converter module**
   - File: src/cubie/odesystems/symbolic/parsing/ast_converter.py
   - Action: Create
   - Details:
     ```python
     """Convert Python AST expressions to SymPy symbolic expressions."""
     
     import ast
     from typing import Callable, Dict, Type
     
     import sympy as sp
     
     
     # Operator mapping constants
     BINOP_MAP: Dict[Type[ast.operator], Callable] = {
         ast.Add: lambda l, r: sp.Add(l, r),
         ast.Sub: lambda l, r: sp.Add(l, sp.Mul(-1, r)),
         ast.Mult: lambda l, r: sp.Mul(l, r),
         ast.Div: lambda l, r: sp.Mul(l, sp.Pow(r, -1)),
         ast.Pow: lambda l, r: sp.Pow(l, r),
         ast.Mod: lambda l, r: sp.Mod(l, r),
         ast.FloorDiv: lambda l, r: sp.floor(sp.Mul(l, sp.Pow(r, -1))),
     }
     
     UNARYOP_MAP: Dict[Type[ast.unaryop], Callable] = {
         ast.UAdd: lambda x: x,
         ast.USub: lambda x: sp.Mul(-1, x),
         ast.Not: lambda x: sp.Not(x),
     }
     
     CMPOP_MAP: Dict[Type[ast.cmpop], Type[sp.Relational]] = {
         ast.Eq: sp.Eq,
         ast.NotEq: sp.Ne,
         ast.Lt: sp.Lt,
         ast.LtE: sp.Le,
         ast.Gt: sp.Gt,
         ast.GtE: sp.Ge,
     }
     
     
     class AstToSympyConverter:
         """Convert Python AST expression nodes to SymPy expressions.
         
         Parameters
         ----------
         symbol_map : Dict[str, sp.Symbol]
             Maps variable names to SymPy symbols.
         known_functions : Dict[str, Callable]
             Maps function names to SymPy functions.
         time_symbol : sp.Symbol
             The time variable symbol.
             
         Attributes
         ----------
         symbol_map : Dict[str, sp.Symbol]
             Variable name to symbol mapping.
         known_functions : Dict[str, Callable]
             Function name to SymPy function mapping.
         time_symbol : sp.Symbol
             Time variable symbol.
         """
         
         def __init__(
             self,
             symbol_map: Dict[str, sp.Symbol],
             known_functions: Dict[str, Callable],
             time_symbol: sp.Symbol,
         ) -> None:
             """Initialize converter with symbol and function mappings.
             
             Parameters
             ----------
             symbol_map : Dict[str, sp.Symbol]
                 Variable name to symbol mapping.
             known_functions : Dict[str, Callable]
                 Function name to SymPy function mapping.
             time_symbol : sp.Symbol
                 Time variable symbol.
             """
             # Input validation as specified
             if not isinstance(symbol_map, dict):
                 raise TypeError("symbol_map must be a dict")
             if not all(isinstance(k, str) for k in symbol_map.keys()):
                 raise TypeError("All symbol_map keys must be strings")
             if not all(isinstance(v, sp.Symbol) for v in symbol_map.values()):
                 raise TypeError("All symbol_map values must be sp.Symbol instances")
             
             if not isinstance(known_functions, dict):
                 raise TypeError("known_functions must be a dict")
             if not all(isinstance(k, str) for k in known_functions.keys()):
                 raise TypeError("All known_functions keys must be strings")
             if not all(callable(v) for v in known_functions.values()):
                 raise TypeError("All known_functions values must be callable")
             
             if not isinstance(time_symbol, sp.Symbol):
                 raise TypeError("time_symbol must be a sp.Symbol instance")
             
             self.symbol_map = symbol_map
             self.known_functions = known_functions
             self.time_symbol = time_symbol
         
         def convert(self, node: ast.expr) -> sp.Expr:
             """Main entry point for AST to SymPy conversion.
             
             Parameters
             ----------
             node : ast.expr
                 AST expression node to convert.
                 
             Returns
             -------
             sp.Expr
                 SymPy expression.
                 
             Raises
             ------
             NotImplementedError
                 If node type not supported.
             """
             if isinstance(node, ast.BinOp):
                 return self._convert_binop(node)
             elif isinstance(node, ast.UnaryOp):
                 return self._convert_unaryop(node)
             elif isinstance(node, ast.Call):
                 return self._convert_call(node)
             elif isinstance(node, ast.Name):
                 return self._convert_name(node)
             elif isinstance(node, ast.Constant):
                 return self._convert_constant(node)
             elif isinstance(node, ast.IfExp):
                 return self._convert_ifexp(node)
             elif isinstance(node, ast.Compare):
                 return self._convert_compare(node)
             elif isinstance(node, ast.BoolOp):
                 return self._convert_boolop(node)
             else:
                 raise NotImplementedError(
                     f"AST node type {type(node).__name__} not supported in "
                     f"expression conversion. Please use string-based equation input "
                     f"or simplify your function."
                 )
         
         def _convert_binop(self, node: ast.BinOp) -> sp.Expr:
             """Convert binary operation to SymPy.
             
             Parameters
             ----------
             node : ast.BinOp
                 Binary operation node.
                 
             Returns
             -------
             sp.Expr
                 SymPy expression.
             """
             left = self.convert(node.left)
             right = self.convert(node.right)
             op_type = type(node.op)
             
             if op_type not in BINOP_MAP:
                 raise NotImplementedError(
                     f"Binary operator {op_type.__name__} not supported"
                 )
             
             return BINOP_MAP[op_type](left, right)
         
         def _convert_unaryop(self, node: ast.UnaryOp) -> sp.Expr:
             """Convert unary operation to SymPy.
             
             Parameters
             ----------
             node : ast.UnaryOp
                 Unary operation node.
                 
             Returns
             -------
             sp.Expr
                 SymPy expression.
             """
             operand = self.convert(node.operand)
             op_type = type(node.op)
             
             if op_type not in UNARYOP_MAP:
                 raise NotImplementedError(
                     f"Unary operator {op_type.__name__} not supported"
                 )
             
             return UNARYOP_MAP[op_type](operand)
         
         def _convert_call(self, node: ast.Call) -> sp.Expr:
             """Convert function call to SymPy.
             
             Parameters
             ----------
             node : ast.Call
                 Call node.
                 
             Returns
             -------
             sp.Expr
                 SymPy expression.
                 
             Raises
             ------
             ValueError
                 If function not in known_functions.
             """
             func_name = self._get_function_name(node.func)
             
             if func_name not in self.known_functions:
                 raise ValueError(
                     f"Unknown function '{func_name}'. Use user_functions "
                     f"parameter or string input. Available functions: "
                     f"{sorted(self.known_functions.keys())}"
                 )
             
             # Convert arguments
             args = [self.convert(arg) for arg in node.args]
             
             # Apply SymPy function
             sympy_func = self.known_functions[func_name]
             return sympy_func(*args)
         
         def _convert_name(self, node: ast.Name) -> sp.Symbol:
             """Convert variable name to SymPy symbol.
             
             Parameters
             ----------
             node : ast.Name
                 Name node.
                 
             Returns
             -------
             sp.Symbol
                 SymPy symbol.
                 
             Raises
             ------
             ValueError
                 If name not in symbol_map.
             """
             name = node.id
             
             if name not in self.symbol_map:
                 raise ValueError(
                     f"Variable '{name}' not found in symbol mapping. "
                     f"Ensure all variables are declared. Available variables: "
                     f"{sorted(self.symbol_map.keys())}"
                 )
             
             return self.symbol_map[name]
         
         def _convert_constant(self, node: ast.Constant) -> sp.Expr:
             """Convert Python constant to SymPy.
             
             Parameters
             ----------
             node : ast.Constant
                 Constant node.
                 
             Returns
             -------
             sp.Expr
                 SymPy expression.
                 
             Raises
             ------
             ValueError
                 If None constant encountered.
             """
             value = node.value
             
             if value is None:
                 raise ValueError(
                     "None is not supported in expressions. "
                     "Use explicit numeric values."
                 )
             elif isinstance(value, bool):
                 return sp.true if value else sp.false
             elif isinstance(value, int):
                 return sp.Integer(value)
             elif isinstance(value, float):
                 return sp.Float(value)
             else:
                 raise ValueError(
                     f"Constant type {type(value).__name__} not supported"
                 )
         
         def _convert_ifexp(self, node: ast.IfExp) -> sp.Piecewise:
             """Convert conditional expression to Piecewise.
             
             Parameters
             ----------
             node : ast.IfExp
                 If expression node.
                 
             Returns
             -------
             sp.Piecewise
                 Piecewise expression.
             """
             test = self.convert(node.test)
             body = self.convert(node.body)
             orelse = self.convert(node.orelse)
             
             return sp.Piecewise((body, test), (orelse, True))
         
         def _convert_compare(self, node: ast.Compare) -> sp.Expr:
             """Convert comparison to SymPy relational.
             
             Parameters
             ----------
             node : ast.Compare
                 Compare node.
                 
             Returns
             -------
             sp.Expr
                 SymPy relational expression.
             """
             # Handle simple comparisons (left op comparator)
             # Chained comparisons (a < b < c) become And(a < b, b < c)
             if len(node.ops) == 1:
                 left = self.convert(node.left)
                 right = self.convert(node.comparators[0])
                 op_type = type(node.ops[0])
                 
                 if op_type not in CMPOP_MAP:
                     raise NotImplementedError(
                         f"Comparison operator {op_type.__name__} not supported"
                     )
                 
                 return CMPOP_MAP[op_type](left, right)
             else:
                 # Chained comparison: build And of individual comparisons
                 comparisons = []
                 left = self.convert(node.left)
                 
                 for op, comparator in zip(node.ops, node.comparators):
                     right = self.convert(comparator)
                     op_type = type(op)
                     
                     if op_type not in CMPOP_MAP:
                         raise NotImplementedError(
                             f"Comparison operator {op_type.__name__} not supported"
                         )
                     
                     comparisons.append(CMPOP_MAP[op_type](left, right))
                     left = right
                 
                 return sp.And(*comparisons)
         
         def _convert_boolop(self, node: ast.BoolOp) -> sp.Expr:
             """Convert boolean operation to SymPy logical.
             
             Parameters
             ----------
             node : ast.BoolOp
                 Boolean operation node.
                 
             Returns
             -------
             sp.Expr
                 SymPy logical expression.
             """
             values = [self.convert(val) for val in node.values]
             
             if isinstance(node.op, ast.And):
                 return sp.And(*values)
             elif isinstance(node.op, ast.Or):
                 return sp.Or(*values)
             else:
                 raise NotImplementedError(
                     f"Boolean operator {type(node.op).__name__} not supported"
                 )
         
         def _get_function_name(self, node: ast.expr) -> str:
             """Extract function name from Call func node.
             
             Parameters
             ----------
             node : ast.expr
                 Function node from Call.
                 
             Returns
             -------
             str
                 Function name.
             """
             if isinstance(node, ast.Name):
                 return node.id
             elif isinstance(node, ast.Attribute):
                 # For np.sin, just return 'sin'
                 return node.attr
             else:
                 raise ValueError(
                     f"Cannot extract function name from {type(node).__name__}"
                 )
     ```
   - Edge cases:
     - Unknown function: Raise ValueError in `_convert_call` with available functions list
     - Unknown variable: Raise ValueError in `_convert_name` with available variables list
     - None constant: Raise ValueError in `_convert_constant`
     - Unsupported node type: Raise NotImplementedError in `convert`
     - Chained comparisons (a < b < c): Build And in `_convert_compare`
   - Integration:
     - Used by FunctionParser (Section 3) to convert RHS expressions
     - Uses KNOWN_FUNCTIONS from parser.py
     - Uses symbol_map built from access patterns (Section 2)

**Tests to Create**:
- Test file: tests/odesystems/symbolic/test_ast_converter.py
- Test function: test_convert_constant_int
- Description: Integer constant converts to sp.Integer
- Test function: test_convert_constant_float
- Description: Float constant converts to sp.Float
- Test function: test_convert_constant_bool
- Description: Bool constants convert to sp.true/sp.false
- Test function: test_convert_name
- Description: Variable name lookup in symbol_map
- Test function: test_convert_binop_add
- Description: Addition operator converts to sp.Add
- Test function: test_convert_binop_sub
- Description: Subtraction converts to sp.Add with negation
- Test function: test_convert_binop_mult
- Description: Multiplication converts to sp.Mul
- Test function: test_convert_binop_div
- Description: Division converts to sp.Mul with inverse
- Test function: test_convert_binop_pow
- Description: Power converts to sp.Pow
- Test function: test_convert_unaryop_neg
- Description: Negation converts to sp.Mul(-1, x)
- Test function: test_convert_call_sin
- Description: sin(x) converts to sp.sin(symbol)
- Test function: test_convert_call_exp
- Description: exp(x) converts to sp.exp(symbol)
- Test function: test_convert_call_unknown
- Description: Unknown function raises ValueError
- Test function: test_convert_ifexp
- Description: Ternary converts to sp.Piecewise
- Test function: test_convert_compare_gt
- Description: Greater-than converts to sp.Gt
- Test function: test_convert_boolop_and
- Description: Logical AND converts to sp.And
- Test function: test_complex_expression
- Description: Nested expression tree converts correctly
- Test function: test_chained_comparison
- Description: a < b < c converts to sp.And(a < b, b < c)

**Outcomes**:
[Empty - to be filled by taskmaster agent]

---

## Summary

### Task Group Overview
- **Group 1**: Function Inspector (foundation)
- **Group 2**: AST Visitor (pattern detection)
- **Group 3**: AST to SymPy Converter (expression conversion)

### Dependency Chain
```
Group 1 (FunctionInspector)
    ↓
Group 2 (OdeAstVisitor) - depends on Group 1
    ↓
Group 3 (AstToSympyConverter) - depends on Groups 1, 2
```

### Test Files Created
1. `tests/odesystems/symbolic/test_function_inspector.py` - 10 tests
2. `tests/odesystems/symbolic/test_ast_visitor.py` - 12 tests
3. `tests/odesystems/symbolic/test_ast_converter.py` - 19 tests

**Total**: 41 tests covering all three modules

### Integration Points
- All modules integrate with Section 2 (Variable Identification)
- FunctionParser (Section 3) will orchestrate all three components
- Uses existing KNOWN_FUNCTIONS from parser.py
- Produces same output format as string parser (ParsedEquations, IndexedBases)

### Estimated Complexity
- **Group 1**: Low-Medium (standard library usage, clear patterns)
- **Group 2**: Medium (AST traversal requires careful pattern matching)
- **Group 3**: Medium-High (operator mapping, expression building, error handling)

**Overall**: Medium complexity - foundational infrastructure for function parsing
