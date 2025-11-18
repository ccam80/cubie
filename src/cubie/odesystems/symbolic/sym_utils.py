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
    if isinstance(dxdt, (list, tuple)) and len(dxdt) > 0:
        first_elem = dxdt[0]
        if isinstance(first_elem, sp.Equality) or \
           (isinstance(first_elem, tuple) and 
            len(first_elem) == 2 and 
            isinstance(first_elem[0], sp.Symbol)):
            hash_strings = []
            for eq in dxdt:
                if isinstance(eq, sp.Equality):
                    lhs_str = str(eq.lhs)
                    rhs_str = str(eq.rhs)
                elif isinstance(eq, tuple):
                    lhs_str = str(eq[0])
                    rhs_str = str(eq[1])
                else:
                    lhs_str = str(eq)
                    rhs_str = ""
                hash_strings.append(f"{lhs_str} = {rhs_str}")
            dxdt_str = "\n".join(hash_strings)
        elif isinstance(first_elem, (list, tuple)):
            dxdt = [str(symbol) + str(expr) for symbol, expr in dxdt]
            dxdt_str = "".join(dxdt)
        else:
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


def prune_unused_assignments(
    expressions: Iterable[Tuple[sp.Symbol, sp.Expr]],
    outputsym_str: str = "jvp",
    output_symbols: Optional[Iterable[sp.Symbol]] = None,
) -> List[Tuple[sp.Symbol, sp.Expr]]:
    """Remove assignments that do not contribute to the final JVP outputs.

    Parameters
    ----------
    expressions
        Topologically sorted assignments ``(lhs, rhs)``.
    outputsym_str
        Prefix identifying JVP output symbols. Ignored when
        ``output_symbols`` is provided.
    output_symbols
        Optional collection of output symbols to retain when pruning.
        When supplied, only assignments contributing to these symbols are
        kept.

    Returns
    -------
    List[Tuple[sp.Symbol, sp.Expr]]
        Pruned assignments that are required to compute the JVP outputs.

    Notes
    -----
    The function assumes that the list is topologically sorted and that output
    assignments have left-hand-side symbols whose names start with
    ``"jvp["``. It preserves the relative order of kept assignments.
    """
    exprs = list(expressions)
    if not exprs:
        return exprs

    lhs_symbols = [lhs for lhs, _ in exprs]
    all_lhs = set(lhs_symbols)

    # Detect outputs by name convention
    if output_symbols is not None:
        output_syms = set(output_symbols) & all_lhs
    else:
        output_syms = {
            lhs
            for lhs in lhs_symbols
            if str(lhs).startswith(f"{outputsym_str}[")
        }

    # If we can't detect outputs, do nothing
    if not output_syms:
        return exprs

    used: set[sp.Symbol] = set(output_syms)
    kept: list[Tuple[sp.Symbol, sp.Expr]] = []

    for lhs, rhs in reversed(exprs):
        if lhs in used:
            kept.append((lhs, rhs))
            # Only follow dependencies that are assigned to
            deps = rhs.free_symbols & all_lhs
            deps_syms = {s for s in deps if isinstance(s, sp.Symbol)}
            used.update(deps_syms)
    kept.reverse()
    return kept
