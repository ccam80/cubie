import warnings
from collections import defaultdict, deque
from typing import Union, List, Dict, Iterable, Tuple, Optional
import sympy as sp


def topological_sort(
        assignments: Union[List[tuple], Dict[sp.Symbol, sp.Expr]],
        ) -> list[tuple[sp.Symbol, sp.Expr]]:
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

    return list(result)

def cse_and_stack(equations: Iterable[Tuple[sp.Symbol, sp.Expr]],
                  symbol: Optional[str] = None,
                  ) -> Tuple[Tuple[sp.Symbol, sp.Expr]]:
    """Performs CSE and returns a list of provided and cse expressions.

    Parameters
    ----------
    equations: iterable of (sp.Symbol, sp.Expr)
        A list of (lhs, rhs) tuples.
    symbol: str, optional
        The desired prefix for newly created cse symbols.

    Returns
    -------
    tuple of tuples of (sp.Symbol, sp.Expr)
        CSE expressions and provided expressions in terms of CSEs, in the same
        format as provided expressions.
    """
    if symbol is None:
        symbol = "_cse"
    expr_labels = (lhs for lhs, _ in equations)
    all_rhs = (rhs for _, rhs in equations)
    while any(str(label).startswith(symbol) for label in expr_labels):
        warnings.warn(f"CSE symbol {symbol} is already in use, it has been "
                      f"prepended with an underscore to _{symbol}")
        symbol = f"_{symbol}"

    cse_exprs, reduced_exprs = sp.cse(
        all_rhs, symbols=sp.numbered_symbols(symbol), order="none"
    )
    expressions = list(zip(expr_labels, reduced_exprs)) + cse_exprs
    expressions = topological_sort(expressions)
    return expressions
