"""Inspect Python callables to extract ODE structure via AST analysis.

Walks the abstract syntax tree of a user-provided function to identify
state accesses, constant/parameter accesses, local assignments, return
expressions, and function calls. Converts AST nodes to SymPy expressions.

Published Functions
-------------------
:func:`inspect_ode_function`
    Analyse a callable and return a :class:`FunctionInspection` dataclass.

Published Classes
-----------------
:class:`FunctionInspection`
    Result container holding parsed AST metadata.

:class:`AstToSympyConverter`
    Recursive converter from :mod:`ast` nodes to :mod:`sympy` expressions.
"""

import ast
import inspect
import textwrap
import warnings
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import sympy as sp

from .parser import KNOWN_FUNCTIONS, TIME_SYMBOL

# Map of module-qualified names to their bare equivalents
_MODULE_PREFIXES = {"math", "np", "numpy", "cmath"}


class FunctionInspection:
    """Result of inspecting an ODE function's AST.

    Parameters
    ----------
    param_names
        All parameter names from the function signature.
    state_param
        Name of the state vector parameter (second positional arg).
    constant_params
        Names of constant/parameter arguments (third+ positional args).
    state_accesses
        List of dicts with keys ``base``, ``key``, ``pattern_type``.
    constant_accesses
        List of dicts with keys ``base``, ``key``, ``pattern_type``.
    assignments
        Mapping of local variable names to their AST expression nodes.
    return_node
        The AST Return node found in the function body.
    function_calls
        Set of function names invoked in the body.
    func_def
        The parsed :class:`ast.FunctionDef` node.
    """

    def __init__(
        self,
        param_names: List[str],
        state_param: str,
        constant_params: List[str],
        state_accesses: List[Dict[str, Any]],
        constant_accesses: List[Dict[str, Any]],
        assignments: Dict[str, ast.expr],
        return_node: ast.Return,
        function_calls: Set[str],
        func_def: ast.FunctionDef,
    ) -> None:
        self.param_names = param_names
        self.state_param = state_param
        self.constant_params = constant_params
        self.state_accesses = state_accesses
        self.constant_accesses = constant_accesses
        self.assignments = assignments
        self.return_node = return_node
        self.function_calls = function_calls
        self.func_def = func_def


class _OdeAstVisitor(ast.NodeVisitor):
    """Walk an ODE function body collecting access patterns."""

    def __init__(
        self, state_param: str, constant_params: List[str]
    ) -> None:
        self.state_param = state_param
        self.constant_params = constant_params
        self.state_accesses: List[Dict[str, Any]] = []
        self.constant_accesses: List[Dict[str, Any]] = []
        self.assignments: Dict[str, ast.expr] = {}
        self.return_nodes: List[ast.Return] = []
        self.function_calls: Set[str] = set()

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.value, ast.Name):
            base = node.value.id
            slc = node.slice
            if isinstance(slc, ast.Constant):
                key = slc.value
                ptype = "int" if isinstance(key, int) else "string"
            elif isinstance(slc, ast.Index):
                # Python 3.8 compat
                inner = slc.value  # type: ignore[attr-defined]
                if isinstance(inner, ast.Constant):
                    key = inner.value
                    ptype = (
                        "int" if isinstance(key, int) else "string"
                    )
                else:
                    key = ast.dump(inner)
                    ptype = "expr"
            elif isinstance(slc, ast.Name):
                key = slc.id
                ptype = "name"
            else:
                key = ast.dump(slc)
                ptype = "expr"

            entry = {"base": base, "key": key, "pattern_type": ptype}
            if base == self.state_param:
                self.state_accesses.append(entry)
            elif base in self.constant_params:
                self.constant_accesses.append(entry)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.value, ast.Name):
            base = node.value.id
            entry = {
                "base": base,
                "key": node.attr,
                "pattern_type": "attribute",
            }
            if base == self.state_param:
                self.state_accesses.append(entry)
            elif base in self.constant_params:
                self.constant_accesses.append(entry)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.assignments[target.id] = node.value
            elif isinstance(target, ast.Tuple):
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        self.assignments[elt.id] = node.value
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        self.return_nodes.append(node)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_name(node)
        if name:
            self.function_calls.add(name)
        self.generic_visit(node)


def _call_name(node: ast.Call) -> Optional[str]:
    """Extract the callable name from a Call node."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        if isinstance(node.func.value, ast.Name):
            return f"{node.func.value.id}.{node.func.attr}"
    return None


def _resolve_func_name(name: str) -> Optional[str]:
    """Strip module prefix to get the bare function name."""
    if "." in name:
        parts = name.split(".")
        if parts[0] in _MODULE_PREFIXES:
            return parts[1]
    return name


class AstToSympyConverter:
    """Convert AST expression nodes to SymPy expressions.

    Parameters
    ----------
    symbol_map
        Mapping of variable names to SymPy symbols/expressions.
    """

    def __init__(self, symbol_map: Dict[str, sp.Basic]) -> None:
        self.symbol_map = symbol_map

    def convert(self, node: ast.expr) -> sp.Expr:
        """Recursively convert an AST node to a SymPy expression.

        Parameters
        ----------
        node
            AST expression node to convert.

        Returns
        -------
        sp.Expr
            Equivalent SymPy expression.

        Raises
        ------
        NotImplementedError
            If the AST node type is not supported.
        """
        if isinstance(node, ast.Constant):
            return self._convert_constant(node)
        elif isinstance(node, ast.Name):
            return self._convert_name(node)
        elif isinstance(node, ast.BinOp):
            return self._convert_binop(node)
        elif isinstance(node, ast.UnaryOp):
            return self._convert_unaryop(node)
        elif isinstance(node, ast.Call):
            return self._convert_call(node)
        elif isinstance(node, ast.Subscript):
            return self._convert_subscript(node)
        elif isinstance(node, ast.Attribute):
            return self._convert_attribute(node)
        elif isinstance(node, ast.Compare):
            return self._convert_compare(node)
        elif isinstance(node, ast.IfExp):
            return self._convert_ifexp(node)
        elif isinstance(node, ast.BoolOp):
            return self._convert_boolop(node)
        elif isinstance(node, ast.Tuple):
            # Should not appear at expression level; handled by
            # callers that unpack return tuples/lists
            raise NotImplementedError(
                "Tuple expressions should be unpacked by the caller"
            )
        elif isinstance(node, ast.List):
            raise NotImplementedError(
                "List expressions should be unpacked by the caller"
            )
        else:
            raise NotImplementedError(
                f"Unsupported AST node type: {type(node).__name__}. "
                f"Only arithmetic, function calls, comparisons, and "
                f"ternary if/else are supported."
            )

    def _convert_constant(self, node: ast.Constant) -> sp.Expr:
        val = node.value
        if isinstance(val, int):
            return sp.Integer(val)
        elif isinstance(val, float):
            return sp.Float(val)
        elif isinstance(val, bool):
            return sp.true if val else sp.false
        else:
            raise NotImplementedError(
                f"Unsupported constant type: {type(val).__name__}"
            )

    def _convert_name(self, node: ast.Name) -> sp.Expr:
        name = node.id
        if name in self.symbol_map:
            return self.symbol_map[name]
        # Create a real symbol for unknown names
        sym = sp.Symbol(name, real=True)
        self.symbol_map[name] = sym
        return sym

    def _convert_binop(self, node: ast.BinOp) -> sp.Expr:
        left = self.convert(node.left)
        right = self.convert(node.right)
        op = node.op
        if isinstance(op, ast.Add):
            return left + right
        elif isinstance(op, ast.Sub):
            return left - right
        elif isinstance(op, ast.Mult):
            return left * right
        elif isinstance(op, ast.Div):
            return left / right
        elif isinstance(op, ast.FloorDiv):
            return sp.floor(left / right)
        elif isinstance(op, ast.Pow):
            return left ** right
        elif isinstance(op, ast.Mod):
            return sp.Mod(left, right)
        else:
            raise NotImplementedError(
                f"Unsupported binary op: {type(op).__name__}"
            )

    def _convert_unaryop(self, node: ast.UnaryOp) -> sp.Expr:
        operand = self.convert(node.operand)
        if isinstance(node.op, ast.USub):
            return -operand
        elif isinstance(node.op, ast.UAdd):
            return operand
        elif isinstance(node.op, ast.Not):
            return sp.Not(operand)
        else:
            raise NotImplementedError(
                f"Unsupported unary op: {type(node.op).__name__}"
            )

    def _convert_call(self, node: ast.Call) -> sp.Expr:
        raw_name = _call_name(node)
        if raw_name is None:
            raise NotImplementedError(
                "Only named function calls are supported"
            )
        name = _resolve_func_name(raw_name)
        if name not in KNOWN_FUNCTIONS:
            raise NotImplementedError(
                f"Unknown function '{raw_name}'. Supported: "
                f"{sorted(KNOWN_FUNCTIONS.keys())}"
            )
        sp_func = KNOWN_FUNCTIONS[name]
        args = [self.convert(a) for a in node.args]
        return sp_func(*args)

    def _convert_subscript(self, node: ast.Subscript) -> sp.Expr:
        if isinstance(node.value, ast.Name):
            base_name = node.value.id
            slc = node.slice
            if isinstance(slc, ast.Constant):
                key = slc.value
            elif isinstance(slc, ast.Index):
                inner = slc.value  # type: ignore[attr-defined]
                if isinstance(inner, ast.Constant):
                    key = inner.value
                else:
                    raise NotImplementedError(
                        "Only constant subscripts are supported"
                    )
            else:
                raise NotImplementedError(
                    "Only constant subscripts are supported"
                )
            lookup = f"{base_name}[{key!r}]" if isinstance(
                key, str
            ) else f"{base_name}[{key}]"
            if lookup in self.symbol_map:
                return self.symbol_map[lookup]
            raise NotImplementedError(
                f"Subscript '{lookup}' not found in symbol map"
            )
        raise NotImplementedError("Complex subscript targets not supported")

    def _convert_attribute(self, node: ast.Attribute) -> sp.Expr:
        if isinstance(node.value, ast.Name):
            lookup = f"{node.value.id}.{node.attr}"
            if lookup in self.symbol_map:
                return self.symbol_map[lookup]
            raise NotImplementedError(
                f"Attribute '{lookup}' not found in symbol map"
            )
        raise NotImplementedError("Complex attribute targets not supported")

    def _convert_compare(self, node: ast.Compare) -> sp.Expr:
        left = self.convert(node.left)
        result = None
        for op, comparator_node in zip(node.ops, node.comparators):
            right = self.convert(comparator_node)
            rel = self._comparison_op(op, left, right)
            result = rel if result is None else sp.And(result, rel)
            left = right
        return result

    @staticmethod
    def _comparison_op(
        op: ast.cmpop, left: sp.Expr, right: sp.Expr
    ) -> sp.Expr:
        if isinstance(op, ast.Gt):
            return sp.Gt(left, right)
        elif isinstance(op, ast.GtE):
            return sp.Ge(left, right)
        elif isinstance(op, ast.Lt):
            return sp.Lt(left, right)
        elif isinstance(op, ast.LtE):
            return sp.Le(left, right)
        elif isinstance(op, ast.Eq):
            return sp.Eq(left, right)
        elif isinstance(op, ast.NotEq):
            return sp.Ne(left, right)
        else:
            raise NotImplementedError(
                f"Unsupported comparison: {type(op).__name__}"
            )

    def _convert_ifexp(self, node: ast.IfExp) -> sp.Expr:
        body = self.convert(node.body)
        test = self.convert(node.test)
        orelse = self.convert(node.orelse)
        return sp.Piecewise((body, test), (orelse, True))

    def _convert_boolop(self, node: ast.BoolOp) -> sp.Expr:
        values = [self.convert(v) for v in node.values]
        if isinstance(node.op, ast.And):
            return sp.And(*values)
        elif isinstance(node.op, ast.Or):
            return sp.Or(*values)
        else:
            raise NotImplementedError(
                f"Unsupported bool op: {type(node.op).__name__}"
            )


def inspect_ode_function(func: Callable) -> FunctionInspection:
    """Analyse a callable to extract ODE structure from its AST.

    Parameters
    ----------
    func
        A Python function defining an ODE right-hand side. Must accept
        at least two positional arguments (time, state).

    Returns
    -------
    FunctionInspection
        Parsed metadata about the function's structure.

    Raises
    ------
    TypeError
        If ``func`` is a lambda, builtin, or not callable.
    ValueError
        If ``func`` has fewer than 2 parameters or no return statement.
    """
    if not callable(func):
        raise TypeError(f"Expected callable, got {type(func).__name__}")

    # Reject lambdas
    if getattr(func, "__name__", "") == "<lambda>":
        raise TypeError(
            "Lambda functions are not supported. Use a named function "
            "with a def statement."
        )

    # Reject builtins without source
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        raise TypeError(
            "Cannot inspect source of builtin or C-extension functions. "
            "Use a pure Python function."
        )

    source = textwrap.dedent(source)
    tree = ast.parse(source)

    func_def = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_def = node
            break

    if func_def is None:
        raise ValueError("Could not find function definition in source")

    params = [arg.arg for arg in func_def.args.args]
    if len(params) < 2:
        raise ValueError(
            f"ODE function must accept at least 2 parameters "
            f"(time, state), got {len(params)}: {params}"
        )

    time_param = params[0]
    state_param = params[1]
    constant_params = params[2:]

    # Warn on unconventional names
    if time_param != "t":
        warnings.warn(
            f"First parameter '{time_param}' is conventionally named 't'",
            stacklevel=2,
        )
    if state_param not in ("y", "state", "x", "Y", "X"):
        warnings.warn(
            f"Second parameter '{state_param}' is conventionally named "
            f"'y' or 'state'",
            stacklevel=2,
        )

    visitor = _OdeAstVisitor(state_param, constant_params)
    visitor.visit(func_def)

    if not visitor.return_nodes:
        raise ValueError(
            "ODE function must contain a return statement"
        )
    if len(visitor.return_nodes) > 1:
        raise ValueError(
            "ODE function must contain exactly one return statement, "
            f"found {len(visitor.return_nodes)}"
        )

    # Validate consistent access patterns per base
    _validate_access_consistency(
        visitor.state_accesses, state_param
    )
    for cp in constant_params:
        cp_accesses = [
            a for a in visitor.constant_accesses if a["base"] == cp
        ]
        _validate_access_consistency(cp_accesses, cp)

    return FunctionInspection(
        param_names=params,
        state_param=state_param,
        constant_params=constant_params,
        state_accesses=visitor.state_accesses,
        constant_accesses=visitor.constant_accesses,
        assignments=visitor.assignments,
        return_node=visitor.return_nodes[0],
        function_calls=visitor.function_calls,
        func_def=func_def,
    )


def _validate_access_consistency(
    accesses: List[Dict[str, Any]], param_name: str
) -> None:
    """Reject mixed access patterns on the same base variable.

    Parameters
    ----------
    accesses
        Access records for a single base variable.
    param_name
        Name of the parameter for error messages.

    Raises
    ------
    ValueError
        If both integer and string subscript patterns are used on the
        same base.
    """
    patterns = {a["pattern_type"] for a in accesses}
    patterns.discard("expr")
    patterns.discard("name")
    if len(patterns) > 1:
        raise ValueError(
            f"Mixed access patterns on '{param_name}': {patterns}. "
            f"Use a single pattern (int subscript, string subscript, "
            f"or attribute access)."
        )
