import warnings
from collections import defaultdict, deque
from typing import Dict, Iterable, List, Optional, Tuple, Union

import sympy as sp


def topological_sort(
    assignments: Union[List[tuple], Dict[sp.Symbol, sp.Expr]],
) -> List[Tuple[sp.Symbol, sp.Expr]]:
    """
    Returns a topologically sorted list of assignments from an unsorted input.

    Uses `Kahn's algorithm <https://en.wikipedia.org/wiki/Topological_sorting>`

    Parameters
    ----------
        assignments: list of tuples or dict
            (lhs_symbol, rhs_expr) assignment tuples or dict of
            {lhs_symbol:rhs_expression}

    Returns
    -------
        list
            (lhs, rhs) assignment tuples in dependency order

    Raises
    ------
        ValueError: If there is a circular depenency in the assignments
    """
    # Build symbol to expression mapping
    if isinstance(assignments, list):
        sym_map = {sym: expr for sym, expr in assignments}
    else:
        sym_map = assignments.copy()

    deps = {}
    all_assignees = set(sym_map.keys())
    for sym, expr in sym_map.items():
        expr_deps = expr.free_symbols & all_assignees
        deps[sym] = expr_deps

    # Kahn's algorithm
    incoming_edges = {sym: len(dep_syms) for sym, dep_syms in deps.items()}

    graph = defaultdict(set)
    for sym, dep_syms in deps.items():
        for dep_sym in dep_syms:
            graph[dep_sym].add(sym)

    # Start with all symbols without dependencies
    queue = deque([sym for sym, degree in incoming_edges.items()
                   if degree == 0])
    result = []

    # Remove incoming edges for fully defined dependencies until none remain
    while queue:
        defined_symbol = queue.popleft()
        # Find the assignment tuple for this symbol
        assignment = sym_map[defined_symbol]
        result.append((defined_symbol, assignment))

        for dependent in graph[defined_symbol]:
            incoming_edges[dependent] -= 1
            if incoming_edges[dependent] == 0:
                queue.append(dependent)

    if len(result) != len(assignments):
        remaining = all_assignees - {sym for sym, _ in result}
        raise ValueError(f"Circular dependency detected. "
                         f"Remaining symbols: {remaining}")

    return result

def cse_and_stack(equations: Iterable[Tuple[sp.Symbol, sp.Expr]],
                  symbol: Optional[str] = None,
                  ) -> List[Tuple[sp.Symbol, sp.Expr]]:
    """Performs CSE and returns a list of provided and cse expressions.

    Parameters
    ----------
    equations: iterable of (sp.Symbol, sp.Expr)
        A list of (lhs, rhs) tuples.
    symbol: str, optional
        The desired prefix for newly created cse symbols.

    Returns
    -------
    list of tuples of (sp.Symbol, sp.Expr)
        CSE expressions and provided expressions in terms of CSEs, in the same
        format as provided expressions.
    """
    if symbol is None:
        symbol = "_cse"
    expr_labels = list(lhs for lhs, _ in equations)
    all_rhs = (rhs for _, rhs in equations)
    while any(str(label).startswith(symbol) for label in expr_labels):
        warnings.warn(f"CSE symbol {symbol} is already in use, it has been "
                      f"prepended with an underscore to _{symbol}")
        symbol = f"_{symbol}"

    cse_exprs, reduced_exprs = sp.cse(
        all_rhs, symbols=sp.numbered_symbols(symbol), order="none"
    )
    expressions = list(zip(expr_labels, reduced_exprs)) + list(cse_exprs)
    sorted_expressions = topological_sort(expressions)
    return sorted_expressions

def hash_system_definition(
    dxdt: Union[str, Iterable[str]],
    constants: Optional[Union[Dict[str, float], Iterable[str]]] = None
) -> str:
    """Generate a comprehensive hash of the system definition.

    Combines the dxdt equations and constant values into a single hash to
    properly detect when system definitions have changed and require rebuilding.

    Parameters
    ----------
    dxdt : str or iterable of str
        The string representation of the dxdt function.
    constants : dict, iterable of str, or None
        The constants definition. If dict, maps constant names to values.
        If iterable, assumes default values. If None, no constants.

    Returns
    -------
    str
        Hash string representing the complete system definition.

    Notes
    -----
    The hash includes:
    1. Normalized dxdt equations (whitespace removed)
    2. Sorted constant names and values (if provided)

    This ensures that changes to either equations or constant values will
    result in different hashes, triggering appropriate rebuilds.
    """
    # Process dxdt equations
    if isinstance(dxdt, (list, tuple)):
        if isinstance(dxdt[0], (list, tuple)):
            dxdt = [str(symbol) + str(expr) for symbol, expr in dxdt]
        dxdt_str = "".join(dxdt)
    else:
        dxdt_str = dxdt

    # Normalize dxdt by removing whitespace
    normalized_dxdt = "".join(dxdt_str.split())

    # Process constants
    constants_str = ""
    if constants is not None:
        constants_str = "|".join(f"{k}:{v}" for k, v in constants.items())

    # Combine components with separator
    combined = f"dxdt:{normalized_dxdt}|constants:{constants_str}"

    # Generate hash
    return str(hash(combined))
