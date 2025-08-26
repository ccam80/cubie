"""Utilities for symbolic Jacobian computation.

Adapted from :mod:`chaste_codegen._jacobian` under the MIT licence.
"""
from typing import TYPE_CHECKING, Union, Tuple, Dict, Iterable
from sympy import IndexedBase, Idx
import sympy as sp

from cubie.systemmodels.symbolic.numba_cuda_printer import print_cuda_multiple
from cubie.systemmodels.symbolic.parser import IndexedBases
from cubie.systemmodels.symbolic.sym_utils import topological_sort, \
    cse_and_stack

if TYPE_CHECKING:
    pass

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
    "    def jvp(state, parameters, driver, v, Jv):\n"
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
    "    def vjp(state, parameters, driver, v, Jv):\n"
    "    {body}\n"
    "    \n"
    "    return vjp\n"
)

def generate_jacobian(equations: Union[
                          Iterable[Tuple[sp.Symbol, sp.Expr]],
                          Dict[sp.Symbol, sp.Expr]],
                      input_order: Dict[sp.Symbol, int],
                      output_order: Dict[sp.Symbol, int],
                      ):
    """Return the symbolic Jacobian matrix for the given equations.

    Parameters
    ----------
    equations : Union[List[Tuple[sp.Symbol, sp.Expr]], Dict[sp.Symbol, sp.Expr]]
        The full set of intermediate(auxiliary) and derivative equations.
    input_symbols : List[sp.Symbol]
        The symbols to be differentiate the function wrt
    output_symbols : List[sp.Symbol]
        The symbols which represent the output of the function
    output_indices : Dict[sp.Symbol, int]
        A map of output symbol to index in the output vector, used to
        assemble the Jacobian matrix in the correct row order.
    """
    input_symbols = set(input_order.keys())
    sorted_inputs = sorted(input_order.keys(),
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
                                for in_sym in input_symbols]).T

        chain_row = sp.zeros(1, num_in)
        for aux_sym in out_expr.free_symbols & aux_symbols:
            partial = sp.diff(out_expr, aux_sym)
            chain_row += partial * auxiliary_gradients[aux_sym]
        J[i,output_order[out_sym]] = chain_row + direct_row

    return J

def generate_analytical_jvp(equations: Union[
                                  Iterable[Tuple[sp.Symbol, sp.Expr]],
                                  Dict[sp.Symbol, sp.Expr]],
                              input_order: Dict[sp.Symbol, int],
                              output_order: Dict[sp.Symbol, int],
                              cse=True
                              ):
    """Returns the symbolic expressions required to calculate
    the Jacobian-vector
    product."""
    jac = generate_jacobian(equations, input_order, output_order)
    n_inputs = len(input_order)
    n_outputs = len(output_order)
    v = IndexedBase("v", shape=(n_inputs,))
    i = Idx('i', n_outputs)
    j = Idx('j', n_inputs)
    jvp = jac[i, j] * v[j]
    jvp_exprs = list(zip(list(v), list(jvp)))
    all_exprs = equations + jvp_exprs

    if cse:
        all_exprs = cse_and_stack(all_exprs)

    all_exprs = topological_sort(all_exprs)
    return all_exprs

def generate_analytical_vjp(equations: Union[
                                  Iterable[Tuple[sp.Symbol, sp.Expr]],
                                  Dict[sp.Symbol, sp.Expr]],
                              input_order: Dict[sp.Symbol, int],
                              output_order: Dict[sp.Symbol, int],
                              cse=True
                              ):
    """Returns the symbolic expressions required to calculate
    the Jacobian-vector
    product."""
    jac = generate_jacobian(equations, input_order, output_order)
    n_inputs = len(input_order)
    n_outputs = len(output_order)
    v = IndexedBase("v", shape=(n_outputs,))
    i = Idx('i', n_outputs)
    j = Idx('j', n_inputs)
    jvp = v[i] * jac[i, j]
    jvp_exprs = list(zip(list(v), list(jvp)))
    all_exprs = equations + jvp_exprs

    if cse:
        all_exprs = cse_and_stack(all_exprs)
    else:
        all_exprs = topological_sort(all_exprs)
    return all_exprs

def generate_jvp_code(equations: Iterable[Tuple[sp.Symbol, sp.Expr]],
                      index_map: IndexedBases,
                      func_name: str="jvp_factory"):
    expressions = generate_analytical_jvp(
        equations,
        input_order=index_map.states.index_map,
        output_order=index_map.dxdt.index_map,
        cse=True,
    )
    jvp_lines = print_cuda_multiple(expressions,
                                    symbol_map=index_map.all_symbols)
    code = JVP_TEMPLATE.format(func_name=func_name,
                               body="\n    ".join(jvp_lines))
    return code

def generate_vjp_code(equations: Iterable[Tuple[sp.Symbol, sp.Expr]],
                      index_map: IndexedBases,
                      func_name: str="vjp_factory"):
    expressions = generate_analytical_jvp(
        equations,
        input_order=index_map.states.index_map,
        output_order=index_map.dxdt.index_map,
        cse=True,
    )
    vjp_lines = print_cuda_multiple(expressions,
                                    symbol_map=index_map.all_symbols)
    code = VJP_TEMPLATE.format(func_name=func_name,
                               body="\n    ".join(vjp_lines))
    return code

