"""Bridge function inspection output to the parser's equation format.

Converts the AST metadata from :func:`inspect_ode_function` into
``(equation_map, funcs, new_params)`` — the same triple that the string
and SymPy branches of :func:`parse_input` produce.

Published Functions
-------------------
:func:`parse_function_input`
    Convert a callable ODE definition into structured equation data.
"""

from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import ast
import sympy as sp

from ..indexedbasemaps import IndexedBases
from .function_inspector import (
    AstToSympyConverter,
    FunctionInspection,
    inspect_ode_function,
)
from .parser import KNOWN_FUNCTIONS, TIME_SYMBOL


def parse_function_input(
    func: Callable,
    index_map: IndexedBases,
    observables: Optional[List[str]] = None,
) -> Tuple[
    List[Tuple[sp.Symbol, sp.Expr]],
    Dict[str, Callable],
    List[sp.Symbol],
]:
    """Convert a callable ODE definition into structured equations.

    Parameters
    ----------
    func
        Python function defining the ODE right-hand side.
    index_map
        Pre-built indexed bases from user-supplied state/param metadata.
    observables
        Observable variable names to extract from local assignments.

    Returns
    -------
    tuple
        ``(equation_map, funcs, new_params)`` matching the output contract
        of the string and SymPy branches in :func:`parse_input`.
    """
    if observables is None:
        observables = []

    inspection = inspect_ode_function(func)
    symbol_map = _build_symbol_map(inspection, index_map)
    converter = AstToSympyConverter(symbol_map)

    # Convert return expressions to derivative equations
    ret_value = inspection.return_node.value
    ret_exprs = _unpack_return(ret_value, converter)

    state_names = index_map.state_names
    dxdt_names = list(index_map.dxdt_names)

    if len(ret_exprs) != len(state_names):
        raise ValueError(
            f"Return has {len(ret_exprs)} elements but system has "
            f"{len(state_names)} states: {state_names}"
        )

    # Build equation map: auxiliaries first, then observables, then dxdt
    equation_map: List[Tuple[sp.Symbol, sp.Expr]] = []

    # Auxiliary assignments (non-state, non-observable locals)
    observable_set = set(observables)
    dxdt_set = set(dxdt_names)
    state_set = set(state_names)
    aux_names = []
    for name, expr_node in inspection.assignments.items():
        if name in observable_set or name in dxdt_set or name in state_set:
            continue
        if name in inspection.constant_params:
            continue
        if name == inspection.state_param:
            continue
        # Skip aliases that are direct state/constant accesses
        if _is_access_alias(expr_node, inspection):
            continue
        aux_names.append(name)

    for name in aux_names:
        sym = symbol_map.get(name)
        if sym is None:
            sym = sp.Symbol(name, real=True)
            symbol_map[name] = sym
        rhs = converter.convert(inspection.assignments[name])
        equation_map.append((sym, rhs))

    # Observable assignments
    for obs_name in observables:
        if obs_name in inspection.assignments:
            obs_sym = index_map.observables.symbol_map.get(obs_name)
            if obs_sym is None:
                obs_sym = sp.Symbol(obs_name, real=True)
            rhs = converter.convert(inspection.assignments[obs_name])
            equation_map.append((obs_sym, rhs))

    # State derivative equations
    if isinstance(ret_value, ast.Dict):
        # Dict return: keys map to state names
        for key_node, expr in zip(ret_value.keys, ret_exprs):
            if isinstance(key_node, ast.Constant):
                sname = key_node.value
            else:
                raise ValueError(
                    "Dict return keys must be string literals"
                )
            if sname not in state_set:
                raise ValueError(
                    f"Dict return key '{sname}' is not a declared state"
                )
            dx_name = f"d{sname}"
            dx_sym = index_map.dxdt.symbol_map.get(dx_name)
            if dx_sym is None:
                dx_sym = sp.Symbol(dx_name, real=True)
            equation_map.append((dx_sym, expr))
    else:
        # List/tuple return: positional mapping
        for i, expr in enumerate(ret_exprs):
            dx_name = dxdt_names[i]
            dx_sym = index_map.dxdt.symbol_map.get(dx_name)
            if dx_sym is None:
                dx_sym = sp.Symbol(dx_name, real=True)
            equation_map.append((dx_sym, expr))

    funcs: Dict[str, Callable] = {}
    new_params: List[sp.Symbol] = []

    return equation_map, funcs, new_params


def _build_symbol_map(
    inspection: FunctionInspection,
    index_map: IndexedBases,
) -> Dict[str, sp.Basic]:
    """Build a mapping from function-local names to SymPy symbols.

    Parameters
    ----------
    inspection
        Result of AST analysis.
    index_map
        Pre-built indexed bases.

    Returns
    -------
    dict
        Variable name to SymPy symbol/expression mapping.
    """
    smap: Dict[str, sp.Basic] = {}

    # Time parameter
    smap[inspection.param_names[0]] = TIME_SYMBOL

    state_names = index_map.state_names
    state_symbols = list(index_map.states.symbol_map.values())

    # State accesses: build lookup keys for subscript/attribute patterns
    for acc in inspection.state_accesses:
        key = acc["key"]
        ptype = acc["pattern_type"]
        if ptype == "int":
            if 0 <= key < len(state_names):
                sym = state_symbols[key]
                lookup = f"{acc['base']}[{key}]"
                smap[lookup] = sym
        elif ptype == "string":
            if key in index_map.states.symbol_map:
                sym = index_map.states.symbol_map[key]
                lookup = f"{acc['base']}[{key!r}]"
                smap[lookup] = sym
        elif ptype == "attribute":
            if key in index_map.states.symbol_map:
                sym = index_map.states.symbol_map[key]
                lookup = f"{acc['base']}.{key}"
                smap[lookup] = sym

    # Constant/parameter accesses
    for acc in inspection.constant_accesses:
        key = acc["key"]
        ptype = acc["pattern_type"]
        base = acc["base"]
        # Search parameters, then constants for the symbol
        target_sym = None
        for ibm in (index_map.parameters, index_map.constants):
            if ptype == "int":
                names = list(ibm.symbol_map.keys())
                if 0 <= key < len(names):
                    target_sym = ibm.symbol_map[names[key]]
                    break
            elif ptype == "string":
                if key in ibm.symbol_map:
                    target_sym = ibm.symbol_map[key]
                    break
            elif ptype == "attribute":
                if key in ibm.symbol_map:
                    target_sym = ibm.symbol_map[key]
                    break

        if target_sym is not None:
            if ptype == "int":
                lookup = f"{base}[{key}]"
            elif ptype == "string":
                lookup = f"{base}[{key!r}]"
            else:
                lookup = f"{base}.{key}"
            smap[lookup] = target_sym

    # Assignments that alias state accesses: map local name to state symbol
    for name, expr_node in inspection.assignments.items():
        sym = _resolve_alias(expr_node, inspection, smap)
        if sym is not None:
            smap[name] = sym

    # dxdt symbols
    for dx_name, dx_sym in index_map.dxdt.symbol_map.items():
        smap[dx_name] = dx_sym

    # Observable symbols
    for obs_name, obs_sym in index_map.observables.symbol_map.items():
        smap[obs_name] = obs_sym

    return smap


def _resolve_alias(
    expr_node: ast.expr,
    inspection: FunctionInspection,
    smap: Dict[str, sp.Basic],
) -> Optional[sp.Basic]:
    """If expr_node is a direct access on state/constant, return symbol."""
    if isinstance(expr_node, ast.Subscript):
        if isinstance(expr_node.value, ast.Name):
            base = expr_node.value.id
            slc = expr_node.slice
            if isinstance(slc, ast.Constant):
                key = slc.value
                if isinstance(key, str):
                    lookup = f"{base}[{key!r}]"
                else:
                    lookup = f"{base}[{key}]"
                return smap.get(lookup)
    elif isinstance(expr_node, ast.Attribute):
        if isinstance(expr_node.value, ast.Name):
            lookup = f"{expr_node.value.id}.{expr_node.attr}"
            return smap.get(lookup)
    return None


def _is_access_alias(
    expr_node: ast.expr, inspection: FunctionInspection
) -> bool:
    """Check if an assignment is a direct state/constant access."""
    if isinstance(expr_node, ast.Subscript):
        if isinstance(expr_node.value, ast.Name):
            base = expr_node.value.id
            return (
                base == inspection.state_param
                or base in inspection.constant_params
            )
    elif isinstance(expr_node, ast.Attribute):
        if isinstance(expr_node.value, ast.Name):
            base = expr_node.value.id
            return (
                base == inspection.state_param
                or base in inspection.constant_params
            )
    return False


def _unpack_return(
    node: ast.expr, converter: AstToSympyConverter
) -> List[sp.Expr]:
    """Unpack a return value into a list of SymPy expressions.

    Parameters
    ----------
    node
        The return value AST node.
    converter
        AST-to-SymPy converter with populated symbol map.

    Returns
    -------
    list
        One SymPy expression per state derivative.
    """
    if isinstance(node, (ast.List, ast.Tuple)):
        return [converter.convert(elt) for elt in node.elts]
    elif isinstance(node, ast.Dict):
        return [converter.convert(v) for v in node.values]
    else:
        # Single expression — single-state system
        return [converter.convert(node)]
