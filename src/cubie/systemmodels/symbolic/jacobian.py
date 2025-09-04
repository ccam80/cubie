"""Utilities for symbolic Jacobian computation.

Adapted from :mod:`chaste_codegen._jacobian` under the MIT licence.
"""

from typing import Dict, Iterable, Tuple, Union

import sympy as sp
from sympy import IndexedBase

from cubie.systemmodels.symbolic.numba_cuda_printer import print_cuda_multiple
from cubie.systemmodels.symbolic.parser import IndexedBases
from cubie.systemmodels.symbolic.sym_utils import (
    cse_and_stack,
    topological_sort,
)

JVP_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED JACOBIAN-VECTOR PRODUCT FACTORY\n"
    "def {func_name}(constants, precision):\n"
    '    """Auto-generated Jacobian factory."""\n'
    "    @cuda.jit((precision[:],\n"
    "               precision[:],\n"
    "               precision[:],\n"
    "               precision[:],\n"
    "               precision[:]),\n"
    "              device=True,\n"
    "             inline=True)\n"
    "    def jvp(state, parameters, drivers, v, jvp):\n"
    "    {body}\n"
    "    \n"
    "    return jvp\n"
)
VJP_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED VECTOR-JACOBIAN PRODUCT FACTORY\n"
    "def {func_name}(constants, precision):\n"
    '    """Auto-generated Jacobian factory."""\n'
    "    @cuda.jit((precision[:],\n"
    "               precision[:],\n"
    "               precision[:],\n"
    "               precision[:],\n"
    "               precision[:]),\n"
    "              device=True,\n"
    "             inline=True)\n"
    "    def vjp(state, parameters, drivers, v, vjp):\n"
    "    {body}\n"
    "    \n"
    "    return vjp\n"
)

I_MINUS_HJ_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED I_MINUS_HJ FACTORY\n"
    "def {func_name}(constants, precision, stages=1):\n"
    '    """Auto-generated I-hJ factory."""\n'
    "    @cuda.jit((precision[:],\n"
    "               precision[:],\n"
    "               precision[:],\n"
    "               precision,\n"
    "               precision[:],\n"
    "               precision[:]),\n"
    "              device=True,\n"
    "              inline=True)\n"
    "    def i_minus_hj(state, parameters, drivers, h, v, out):\n"
    "    {body}\n"
    "    \n"
    "    return out\n"
)

RES_PLUS_I_MINUS_HJ_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED RESIDUAL PLUS I_MINUS_HJ FACTORY\n"
    "def {func_name}(constants, precision, stages=1):\n"
    '    """Auto-generated residual plus I-hJ factory."""\n'
    "    @cuda.jit((precision[:],\n"
    "               precision[:],\n"
    "               precision[:],\n"
    "               precision,\n"
    "               precision[:],\n"
    "               precision[:]),\n"
    "              device=True,\n"
    "              inline=True)\n"
    "    def residual_plus_i_minus_hj(state, parameters, drivers, h, v, out):\n"
    "    {body}\n"
    "    \n"
    "    return out\n"
)

_cache: dict = {}

def get_cache_counts() -> Dict[str, int]:
    """Return counts of cached items by kind (jac, jvp, vjp).

    Used in testing."""
    counts: Dict[str, int] = {"jac": 0, "jvp": 0, "vjp": 0}
    for key in _cache.keys():
        kind = key[0] if isinstance(key, tuple) and key else None
        if kind in counts:
            counts[kind] += 1
    return counts

def _get_cache_key(equations, input_order, output_order):
    """Generate a base cache key from equations and orders."""
    # Convert equations to a hashable form
    if isinstance(equations, dict):
        eq_tuple = tuple(equations.items())
    else:
        eq_tuple = tuple((tuple(eq_pair) for eq_pair in equations))

    input_tuple = tuple(input_order.items())
    output_tuple = tuple(output_order.items())

    return (eq_tuple, input_tuple, output_tuple)


def _get_unified_cache_key(kind: str,
                           equations,
                           input_order,
                           output_order,
                           observables=None,
                           cse=True):
    """Generate a unified cache key for jac/jvp/vjp kinds."""
    base = _get_cache_key(equations, input_order, output_order)
    if kind == "jac":
        return ("jac", base)
    # jvp or vjp
    if observables is None:
        obs_tuple = None
    else:
        obs_tuple = tuple(observables)
    return (kind, base, obs_tuple, bool(cse))


def clear_cache():
    """Clear the unified symbolic cache (kept for API compatibility)."""
    _cache.clear()


def generate_jacobian(equations: Union[
                          Iterable[Tuple[sp.Symbol, sp.Expr]],
                          Dict[sp.Symbol, sp.Expr]],
                      input_order: Dict[sp.Symbol, int],
                      output_order: Dict[sp.Symbol, int],
                      use_cache: bool = True,
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

    # Check cache first
    cache_key = None
    if use_cache:
        cache_key = _get_unified_cache_key("jac", eq_list, input_order, output_order)
        if cache_key in _cache:
            return _cache[cache_key]

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
        _cache[cache_key] = J

    return J


def _prune_unused_assignments(expressions: Iterable[Tuple[sp.Symbol, sp.Expr]]):
    """Remove assignments that are not required to compute final jvp/vjp outputs.

    The function assumes that the list is topologically sorted and that output
    assignments have LHS symbols whose names start with either "jvp[" or "vjp[".
    It preserves the relative order of kept assignments.

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
    output_syms = {lhs for lhs in lhs_symbols if
                   (str(lhs).startswith("jvp[") or str(lhs).startswith("vjp["))}

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


def generate_jac_product(equations: Union[
                              Iterable[Tuple[sp.Symbol, sp.Expr]],
                              Dict[sp.Symbol, sp.Expr]],
                         input_order: Dict[sp.Symbol, int],
                         output_order: Dict[sp.Symbol, int],
                         observables: Iterable[sp.Symbol] = None,
                         direction='jvp',
                         cse=True,
                              ):
    """Returns symbolic expressions for vector-jacobian or jacobian-vector
    product, depending on the direction argument.."""
    # Materialize equations to avoid consuming generators and ensure stable keys
    if isinstance(equations, dict):
        eq_list = list(equations.items())
    else:
        eq_list = list(equations)

    # Caching key before any mutation of inputs
    cache_key = _get_unified_cache_key(direction,
                                       eq_list,
                                       input_order,
                                       output_order,
                                       observables,
                                       cse)
    if cache_key in _cache:
        return _cache[cache_key]

    n_inputs = len(input_order)
    n_outputs = len(output_order)

    # Swap out observables for auxiliary variables
    if observables is not None:
        obs_subs = dict(zip(observables,sp.numbered_symbols("aux_", start=1)))
    else:
        obs_subs = {}

    equations = [(lhs.subs(obs_subs), rhs.subs(obs_subs)) for lhs, rhs in
                  eq_list]
    jac = generate_jacobian(equations, input_order, output_order, use_cache=True)

    prod_exprs = []

    # Flatten Jacobian
    for i in range(n_outputs):
        for j in range(n_inputs):
            prod_exprs.append((sp.Symbol(f"j_{i}{j}"), jac[i, j]))

    # Sum over inputs for jvp, and outputs for vjp
    if direction == "jvp":
        # Sort outputs by their order for JVP
        sorted_outputs = sorted(
            output_order.keys(), key=lambda sym: output_order[sym]
        )
        v = IndexedBase("v", shape=(n_inputs,))
        for out_sym in sorted_outputs:
            sum_ = sp.S.Zero
            for j in range(n_inputs):
                sum_ += sp.Symbol(f"j_{output_order[out_sym]}{j}") * v[j]
            prod_exprs.append(
                (sp.Symbol(f"jvp[{output_order[out_sym]}]"), sum_)
            )
    else:
        # Sort inputs by their order for VJP
        sorted_inputs = sorted(
            input_order.keys(), key=lambda sym: input_order[sym]
        )
        v = IndexedBase("v", shape=(n_outputs,))
        for in_sym in sorted_inputs:
            sum_ = sp.S.Zero
            for j in range(n_outputs):
                sum_ += sp.Symbol(f"j_{j}{input_order[in_sym]}") * v[j]
            prod_exprs.append((sp.Symbol(f"vjp[{input_order[in_sym]}]"), sum_))

    # Remove output equations - they're not required
    exprs = [expr for expr in equations if expr[0] not in output_order]
    all_exprs = exprs + prod_exprs

    if cse:
        all_exprs = cse_and_stack(all_exprs)
    else:
        all_exprs = topological_sort(all_exprs)

    # Final sweep to drop any intermediates not contributing to jvp/vjp
    all_exprs = _prune_unused_assignments(all_exprs)

    # Store in cache and return
    _cache[cache_key] = all_exprs
    return all_exprs


def generate_analytical_jvp(equations: Union[
                                  Iterable[Tuple[sp.Symbol, sp.Expr]],
                                  Dict[sp.Symbol, sp.Expr]],
                              input_order: Dict[sp.Symbol, int],
                              output_order: Dict[sp.Symbol, int],
                              observables: Iterable[sp.Symbol] = None,
                              cse=True,
                              ):
    """Returns the symbolic expressions required to calculate
    the Jacobian-vector
    product."""
    return generate_jac_product(equations=equations,
                                input_order=input_order,
                                output_order=output_order,
                                observables=observables,
                                direction='jvp',
                                cse=cse)


def generate_analytical_vjp(equations: Union[
                                  Iterable[Tuple[sp.Symbol, sp.Expr]],
                                  Dict[sp.Symbol, sp.Expr]],
                              input_order: Dict[sp.Symbol, int],
                              output_order: Dict[sp.Symbol, int],
                              observables: Iterable[sp.Symbol] = None,
                              cse=True,
                              ):
    """Returns the symbolic expressions required to calculate
    the vector-Jacobian product."""
    return generate_jac_product(equations=equations,
                                input_order=input_order,
                                output_order=output_order,
                                observables=observables,
                                direction='vjp',
                                cse=cse)


def generate_jvp_code(equations: Iterable[Tuple[sp.Symbol, sp.Expr]],
                      index_map: IndexedBases,
                      func_name: str="jvp_factory",
                      cse=True):
    expressions = generate_analytical_jvp(
        equations,
        input_order=index_map.states.index_map,
        output_order=index_map.dxdt.index_map,
        observables=index_map.observable_symbols,
        cse=cse,
    )
    jvp_lines = print_cuda_multiple(expressions,
                                    symbol_map=index_map.all_arrayrefs)
    if not jvp_lines:
        jvp_lines = ["pass"]
    code = JVP_TEMPLATE.format(func_name=func_name,
                               body="    " + "\n        ".join(jvp_lines))
    return code


def generate_vjp_code(equations: Iterable[Tuple[sp.Symbol, sp.Expr]],
                      index_map: IndexedBases,
                      func_name: str="vjp_factory",
                      cse=True):
    expressions = generate_analytical_vjp(
        equations,
        input_order=index_map.states.index_map,
        output_order=index_map.dxdt.index_map,
        observables=index_map.observable_symbols,
        cse=cse,
    )
    vjp_lines = print_cuda_multiple(expressions,
                                    symbol_map=index_map.all_arrayrefs)
    if not vjp_lines:
        vjp_lines = ["pass"]
    code = VJP_TEMPLATE.format(func_name=func_name,
                               body="    " + "\n        ".join(vjp_lines))
    return code

def _split_jvp_expressions(exprs):
    aux = []
    jvp_terms = {}
    for lhs, rhs in exprs:
        lhs_str = str(lhs)
        if lhs_str.startswith("jvp["):
            index = int(lhs_str.split("[")[1].split("]")[0])
            jvp_terms[index] = rhs
        else:
            aux.append((lhs, rhs))
    return aux, jvp_terms


def _split_residual_expressions(exprs, index_map):
    aux = []
    res_terms = {}
    dxdt_syms = set(index_map.dxdt.ref_map.keys())
    for lhs, rhs in exprs:
        if lhs in dxdt_syms:
            index = index_map.dxdt.index_map[lhs]
            res_terms[index] = rhs
        else:
            aux.append((lhs, rhs))
    return aux, res_terms

def generate_i_minus_hj_code(equations: Iterable[Tuple[sp.Symbol, sp.Expr]],
                              index_map: IndexedBases,
                              func_name: str = "i_minus_hj_factory",
                              cse=True):
    jvp_exprs = generate_analytical_jvp(
        equations,
        input_order=index_map.states.index_map,
        output_order=index_map.dxdt.index_map,
        observables=index_map.observable_symbols,
        cse=cse,
    )

    aux, jvp_terms = _split_jvp_expressions(jvp_exprs)
    n_out = len(index_map.dxdt.ref_map)
    all_exprs = list(aux)
    for i in range(n_out):
        all_exprs.append(
            (
                sp.Symbol(f"out[{i}]"),
                sp.Symbol(f"v[{i}]") - sp.Symbol("h") * jvp_terms[i],
            )
        )

    if cse:
        all_exprs = cse_and_stack(all_exprs)
    else:
        all_exprs = topological_sort(all_exprs)
    lines = print_cuda_multiple(all_exprs, symbol_map=index_map.all_arrayrefs)
    if not lines:
        lines = ["pass"]
    code = I_MINUS_HJ_TEMPLATE.format(func_name=func_name,
                                      body="    " + "\n        ".join(lines))
    return code


def generate_residual_plus_i_minus_hj_code(
    equations: Iterable[Tuple[sp.Symbol, sp.Expr]],
    index_map: IndexedBases,
    func_name: str = "residual_plus_i_minus_hj_factory",
    cse=True,
):
    res_exprs = topological_sort(equations)
    res_aux, res_terms = _split_residual_expressions(res_exprs, index_map)
    jvp_exprs = generate_analytical_jvp(
        equations,
        input_order=index_map.states.index_map,
        output_order=index_map.dxdt.index_map,
        observables=index_map.observable_symbols,
        cse=cse,
    )
    jvp_aux, jvp_terms = _split_jvp_expressions(jvp_exprs)
    all_exprs = res_aux + jvp_aux
    n_out = len(index_map.dxdt.ref_map)
    for i in range(n_out):
        all_exprs.append(
            (
                sp.Symbol(f"out[{i}]"),
                res_terms[i] + sp.Symbol(f"v[{i}]") - sp.Symbol("h") * jvp_terms[i],
            )
        )
    if cse:
        all_exprs = cse_and_stack(all_exprs)
    else:
        all_exprs = topological_sort(all_exprs)
    lines = print_cuda_multiple(all_exprs, symbol_map=index_map.all_arrayrefs)
    if not lines:
        lines = ["pass"]
    code = RES_PLUS_I_MINUS_HJ_TEMPLATE.format(
        func_name=func_name, body="    " + "\n        ".join(lines)
    )
    return code

