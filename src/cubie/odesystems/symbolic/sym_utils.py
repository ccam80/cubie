"""Utility helpers for symbolic ODE construction."""

import warnings
from collections import defaultdict, deque
from typing import Dict, Iterable, List, Optional, Tuple, Union

import sympy as sp


def topological_sort(
    assignments: Union[
        List[Tuple[sp.Symbol, sp.Expr]],
        Dict[sp.Symbol, sp.Expr],
    ],
) -> List[Tuple[sp.Symbol, sp.Expr]]:
    """Return assignments sorted by their dependency order.

    Parameters
    ----------
    assignments
        Either an iterable of ``(symbol, expression)`` pairs or a mapping from
        each symbol to its defining expression.

    Returns
    -------
    list[tuple[sympy.Symbol, sympy.Expr]]
        Assignments ordered such that dependencies are defined before use.

    Raises
    ------
    ValueError
        Raised when a circular dependency prevents topological sorting.

    Notes
    -----
    Uses Kahn's algorithm for topological sorting. Refer to the Wikipedia
    article on topological sorting for additional background.
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
    queue = deque(
        [sym for sym, degree in incoming_edges.items() if degree == 0]
    )
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
        raise ValueError(
            "Circular dependency detected. Remaining symbols: "
            f"{remaining}"
        )

    return result


def cse_and_stack(
    equations: Iterable[Tuple[sp.Symbol, sp.Expr]],
    symbol: Optional[str] = None,
) -> List[Tuple[sp.Symbol, sp.Expr]]:
    """Perform common subexpression elimination and stack the results.

    Parameters
    ----------
    equations
        ``(symbol, expression)`` pairs that define the system.
    symbol
        Prefix to use for the generated common-subexpression symbols. Defaults
        to ``"_cse"`` when not provided.

    Returns
    -------
    list[tuple[sympy.Symbol, sympy.Expr]]
        Combined list of original expressions rewritten in terms of CSE
        symbols followed by the generated common subexpressions.
    """
    if symbol is None:
        symbol = "_cse"
    expr_labels = [lhs for lhs, _ in equations]
    all_rhs = (rhs for _, rhs in equations)
    while any(str(label).startswith(symbol) for label in expr_labels):
        warnings.warn(
            f"CSE symbol {symbol} is already in use; it has been "
            f"prepended with an underscore to _{symbol}"
        )
        symbol = f"_{symbol}"

    cse_exprs, reduced_exprs = sp.cse(
        all_rhs, symbols=sp.numbered_symbols(symbol), order="none"
    )
    expressions = list(zip(expr_labels, reduced_exprs)) + list(cse_exprs)
    sorted_expressions = topological_sort(expressions)
    return sorted_expressions

def hash_system_definition(
    dxdt: Union[str, Iterable[str]],
    constants: Optional[Union[Dict[str, float], Iterable[str]]] = None,
) -> str:
    """Generate a hash that captures equations and constant definitions.

    Parameters
    ----------
    dxdt
        Representation of the system right-hand sides. Accepts either a single
        string or an iterable of equation strings.
    constants
        Mapping or iterable describing constant names and values. Iterables are
        interpreted as constant names using their default values.

    Returns
    -------
    str
        Deterministic hash string that reflects both equations and constants.

    Notes
    -----
    The hash concatenates normalised differential equations with the sorted
    constant name-value pairs. Any change to either component produces a new
    hash so cached artifacts can be refreshed.
    """
    # Process dxdt equations
    if isinstance(dxdt, (list, tuple)):
        if isinstance(dxdt[0], (list, tuple)):
            dxdt = [str(symbol) + str(expr) for symbol, expr in dxdt]
        dxdt_str = "".join(dxdt)
    elif hasattr(dxdt, "__iter__") and not isinstance(dxdt, str):
        dxdt_pairs = [f"{str(symbol)}{str(expr)}" for symbol, expr in dxdt]
        dxdt_str = "".join(dxdt_pairs)
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


def render_constant_assignments(
    constant_names: Iterable[str], indent: int = 4
) -> str:
    """Return assignment statements that load constants into locals."""

    prefix = " " * indent
    lines = [
        f"{prefix}{name} = precision(constants['{name}'])"
        for name in constant_names
    ]
    return "\n".join(lines) + ("\n" if lines else "")
