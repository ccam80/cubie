"""Utilities for symbolic Jacobian computation.

Adapted from :mod:`chaste_codegen._jacobian` under the MIT licence.
"""
from typing import TYPE_CHECKING, Dict, Iterable, Tuple, Union

import sympy as sp
from sympy import IndexedBase

from cubie.systemmodels.symbolic.numba_cuda_printer import print_cuda_multiple
from cubie.systemmodels.symbolic.parser import IndexedBases
from cubie.systemmodels.symbolic.sym_utils import (
    cse_and_stack,
    topological_sort,
)

if TYPE_CHECKING:
    pass

# Simple cache for Jacobian matrices
_jacobian_cache = {}

def _get_cache_key(equations, input_order, output_order):
    """Generate a cache key for the Jacobian computation."""
    # Convert equations to a hashable form
    if isinstance(equations, dict):
        eq_tuple = tuple(equations.items())
    else:
        eq_tuple = tuple(equations)

    input_tuple = tuple(input_order.items())
    output_tuple = tuple(output_order.items())

    return (eq_tuple, input_tuple, output_tuple)

def clear_jacobian_cache():
    """Clear the Jacobian cache."""
    global _jacobian_cache
    _jacobian_cache.clear()

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
    "    def jvp(state, parameters, driver, v, jvp):\n"
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
    "    def vjp(state, parameters, driver, v, vjp):\n"
    "    {body}\n"
    "    \n"
    "    return vjp\n"
)

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
    # Check cache first
    cache_key = None
    if use_cache:
        cache_key = _get_cache_key(equations, input_order, output_order)
        if cache_key in _jacobian_cache:
            return _jacobian_cache[cache_key]

    input_symbols = set(input_order.keys())
    sorted_inputs = sorted(input_symbols,
                           key=lambda symbol: input_order[symbol])
    output_symbols = set(output_order.keys())
    num_in = len(input_symbols)

    equations = topological_sort(equations)
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
        _jacobian_cache[cache_key] = J

    return J

def generate_jac_product(equations: Union[
                              Iterable[Tuple[sp.Symbol, sp.Expr]],
                              Dict[sp.Symbol, sp.Expr]],
                         input_order: Dict[sp.Symbol, int],
                         output_order: Dict[sp.Symbol, int],
                         direction='jvp',
                         cse=True,
                         use_cache: bool = True
                              ):
    """Returns symbolic expressions for vector-jacobian or jacobian-vector
    product, depending on the direction argument.."""
    n_inputs = len(input_order)
    n_outputs = len(output_order)
    v = IndexedBase("v", shape=(n_outputs,))

    jac = generate_jacobian(equations, input_order, output_order, use_cache=use_cache)

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
    return all_exprs

def generate_analytical_jvp(equations: Union[
                                  Iterable[Tuple[sp.Symbol, sp.Expr]],
                                  Dict[sp.Symbol, sp.Expr]],
                              input_order: Dict[sp.Symbol, int],
                              output_order: Dict[sp.Symbol, int],
                              cse=True,
                              use_cache: bool = True
                              ):
    """Returns the symbolic expressions required to calculate
    the Jacobian-vector
    product."""
    return generate_jac_product(equations=equations,
                                input_order=input_order,
                                output_order=output_order,
                                direction='jvp',
                                cse=cse,
                                use_cache=use_cache)

def generate_analytical_vjp(equations: Union[
                                  Iterable[Tuple[sp.Symbol, sp.Expr]],
                                  Dict[sp.Symbol, sp.Expr]],
                              input_order: Dict[sp.Symbol, int],
                              output_order: Dict[sp.Symbol, int],
                              cse=True,
                              use_cache: bool = True
                              ):
    """Returns the symbolic expressions required to calculate
    the vector-Jacobian product."""
    return generate_jac_product(equations=equations,
                                input_order=input_order,
                                output_order=output_order,
                                direction='vjp',
                                cse=cse,
                                use_cache=use_cache)

def generate_jvp_code(equations: Iterable[Tuple[sp.Symbol, sp.Expr]],
                      index_map: IndexedBases,
                      func_name: str="jvp_factory",
                      cse=True):
    expressions = generate_analytical_jvp(
        equations,
        input_order=index_map.states.index_map,
        output_order=index_map.dxdt.index_map,
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
        cse=cse,
    )
    vjp_lines = print_cuda_multiple(expressions,
                                    symbol_map=index_map.all_arrayrefs)
    if not vjp_lines:
        vjp_lines = ["pass"]
    code = VJP_TEMPLATE.format(func_name=func_name,
                               body="    " + "\n        ".join(vjp_lines))
    return code
