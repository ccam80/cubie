"""Utilities for symbolic Jacobian computation.

Adapted from :mod:`chaste_codegen._jacobian` under the MIT licence.
"""

from typing import Dict, Iterable, Tuple, Union

import sympy as sp
from sympy import IndexedBase

from cubie.odesystems.symbolic.sym_utils import (
    cse_and_stack,
    topological_sort,
)

_cache: dict = {}

def get_cache_counts() -> Dict[str, int]:
    """Return counts of cached items by kind (jac, jvp).

    Used in testing."""
    counts: Dict[str, int] = {"jac": 0, "jvp": 0}
    for value in _cache.values():
        # New scheme: value is a dict possibly containing both kinds
        if isinstance(value, dict):
            if "jac" in value:
                counts["jac"] += 1
            if "jvp" in value:
                counts["jvp"] += 1
        else:
            # Backward compatibility: count best-effort by type/shape
            try:
                if isinstance(value, sp.Matrix):
                    counts["jac"] += 1
                elif isinstance(value, list):
                    counts["jvp"] += 1
            except Exception:
                pass
    return counts


def get_cache_key(equations,
                  input_order: Dict[sp.Symbol, int],
                  output_order: Dict[sp.Symbol, int],
                  cse: bool):
    """Generate the cache key from equations, orders and cse flag.

    This single key is shared across all built artifacts (e.g. jac, jvp).
    """
    # Convert equations to a hashable form
    if isinstance(equations, dict):
        eq_tuple = tuple(equations.items())
    else:
        eq_tuple = tuple((tuple(eq_pair) for eq_pair in equations))

    input_tuple = tuple(input_order.items())
    output_tuple = tuple(output_order.items())

    return (eq_tuple, input_tuple, output_tuple, bool(cse))


def clear_cache():
    """Clear the unified symbolic cache (kept for API compatibility)."""
    _cache.clear()


def generate_jacobian(equations: Union[
                          Iterable[Tuple[sp.Symbol, sp.Expr]],
                          Dict[sp.Symbol, sp.Expr]],
                      input_order: Dict[sp.Symbol, int],
                      output_order: Dict[sp.Symbol, int],
                      use_cache: bool = True,
                      cache_cse: bool = True,
                      ):
    """Return the symbolic Jacobian matrix for the given equations.

    Parameters
    ----------
    equations : Union[List[Tuple[sp.Symbol, sp.Expr]], Dict[sp.Symbol, sp.Expr]]
        The full set of intermediate(auxiliary) and derivative equations.
    input_order : Dict[sp.Symbol, int]
        A dict mapping input symbols to their index in the input vector.
    output_order : List[sp.Symbol]
        A dict mapping output symbols to their index in the output vector.
    use_cache : bool, optional
        Whether to use caching for the Jacobian computation. Default is True.

    Returns
    -------
    sp.Matrix: The symbolic Jacobian matrix.
    """
    if isinstance(equations, dict):
        eq_list = list(equations.items())
    else:
        eq_list = list(equations)

    # Check cache first (Jacobian is independent of cse but we normalize to True for keying)
    cache_key = None
    if use_cache:
        cache_key = get_cache_key(eq_list, input_order, output_order, cse=cache_cse)
        cached_entry = _cache.get(cache_key)
        if isinstance(cached_entry, dict) and "jac" in cached_entry:
            return cached_entry["jac"]

    input_symbols = set(input_order.keys())
    sorted_inputs = sorted(input_symbols,
                           key=lambda symbol: input_order[symbol])
    output_symbols = set(output_order.keys())
    num_in = len(input_symbols)

    equations = topological_sort(eq_list)
    auxiliary_equations = [(lhs, eq) for lhs, eq in equations if lhs not in
                           output_symbols]
    aux_symbols = {lhs for lhs, _ in auxiliary_equations}
    output_equations = [(lhs, eq) for lhs, eq in equations if lhs in
                        output_symbols]

    auxiliary_gradients = {}
    partials_cache = {}

    # Chain rule auxiliary equations
    for sym, expr in auxiliary_equations:
        direct_grad = sp.Matrix(
                [[sp.diff(expr, in_sym)]
                 for in_sym in sorted_inputs]).T

        chain_grad = sp.zeros(1, num_in)
        for other_sym in expr.free_symbols & aux_symbols:
            if other_sym in auxiliary_gradients:
                key = (sym, other_sym)
                if key not in partials_cache:
                    partials_cache[key] = sp.diff(expr, other_sym)
                chain_grad += (partials_cache[key]
                               * auxiliary_gradients[other_sym])
            else:
                raise ValueError(f"Topological order violation: {sym} depends "
                                 f"on {other_sym} which is not yet processed.")
        auxiliary_gradients[sym] = direct_grad + chain_grad

    num_out = len(output_symbols)
    J = sp.zeros(num_out, num_in)

    for i, (out_sym, out_expr) in enumerate(output_equations):
        direct_row = sp.Matrix([[sp.diff(out_expr, in_sym)]
                                for in_sym in sorted_inputs]).T

        chain_row = sp.zeros(1, num_in)
        for aux_sym in out_expr.free_symbols & aux_symbols:
            partial = sp.diff(out_expr, aux_sym)
            chain_row += partial * auxiliary_gradients[aux_sym]
        J[output_order[out_sym],:] = chain_row + direct_row

    # Cache the result before returning
    if use_cache and cache_key is not None:
        entry = _cache.get(cache_key)
        if isinstance(entry, dict):
            entry["jac"] = J
        else:
            _cache[cache_key] = {"jac": J}

    return J


def _prune_unused_assignments(expressions: Iterable[Tuple[sp.Symbol,
sp.Expr]],
                              outputsym_str: str = "jvp"):
    """Remove assignments not required to compute final JVP outputs.

    The function assumes that the list is topologically sorted and that output
    assignments have LHS symbols whose names start with ``"jvp["``. It preserves
    the relative order of kept assignments.

    Parameters
    ----------
    expressions : Iterable[Tuple[sp.Symbol, sp.Expr]]
        A topologically sorted list of (lhs, rhs) assignments.

    Returns
    -------
        list of tuples of (sp.Symbol, sp.Expr)
        The pruned list of assignments.
    """
    exprs = list(expressions)
    if not exprs:
        return exprs

    lhs_symbols = [lhs for lhs, _ in exprs]
    all_lhs = set(lhs_symbols)

    # Detect outputs by name convention
    output_syms = {lhs for lhs in lhs_symbols
                   if str(lhs).startswith(f"{outputsym_str}[")}

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


def generate_analytical_jvp(equations: Union[
                              Iterable[Tuple[sp.Symbol, sp.Expr]],
                              Dict[sp.Symbol, sp.Expr]],
                         input_order: Dict[sp.Symbol, int],
                         output_order: Dict[sp.Symbol, int],
                         observables: Iterable[sp.Symbol] = None,
                         cse=True,
                              ):
    """Return symbolic expressions for the Jacobian-vector product."""

    if isinstance(equations, dict):
        eq_list = list(equations.items())
    else: # convert a generator if present
        eq_list = list(equations)

    # Cache key before any mutation of inputs
    cache_key = get_cache_key(eq_list, input_order, output_order, cse=cse)
    cached_entry = _cache.get(cache_key)
    if isinstance(cached_entry, dict) and "jvp" in cached_entry:
        return cached_entry["jvp"]

    n_inputs = len(input_order)
    n_outputs = len(output_order)

    # Swap out observables for auxiliary variables
    if observables is not None:
        obs_subs = dict(zip(observables,sp.numbered_symbols("aux_", start=1)))
    else:
        obs_subs = {}

    equations = [(lhs.subs(obs_subs), rhs.subs(obs_subs)) for lhs, rhs in
                  eq_list]
    jac = generate_jacobian(equations, input_order, output_order, use_cache=True, cache_cse=cse)

    prod_exprs = []
    j_symbols: Dict[Tuple[int, int], sp.Symbol] = {}

    # Flatten Jacobian, dropping zero-valued entries
    for i in range(n_outputs):
        for j in range(n_inputs):
            expr = jac[i, j]
            if expr == 0:
                continue
            sym = sp.Symbol(f"j_{i}{j}")
            prod_exprs.append((sym, expr))
            j_symbols[(i, j)] = sym

    # Sort outputs by their order for JVP
    sorted_outputs = sorted(
        output_order.keys(), key=lambda sym: output_order[sym]
    )
    v = IndexedBase("v", shape=(n_inputs,))
    for out_sym in sorted_outputs:
        sum_ = sp.S.Zero
        i = output_order[out_sym]
        for j in range(n_inputs):
            sym = j_symbols.get((i, j))
            if sym is not None:
                sum_ += sym * v[j]
        prod_exprs.append((sp.Symbol(f"jvp[{i}]"), sum_))

    # Remove output equations - they're not required
    exprs = [expr for expr in equations if expr[0] not in output_order]
    all_exprs = exprs + prod_exprs

    if cse:
        all_exprs = cse_and_stack(all_exprs)
    else:
        all_exprs = topological_sort(all_exprs)

    # Final sweep to drop any intermediates not contributing to the JVP
    all_exprs = _prune_unused_assignments(all_exprs)

    # Store in cache and return
    entry = _cache.get(cache_key)
    if isinstance(entry, dict):
        entry["jvp"] = all_exprs
    else:
        _cache[cache_key] = {"jvp": all_exprs}
    return all_exprs

