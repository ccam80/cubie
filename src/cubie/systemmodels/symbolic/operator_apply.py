"""
Codegen for a matrix-free operator apply:
    out = beta * (M @ v) - gamma * h * (J @ v)

Supports two mass-matrix modes:
  1) Dense M argument (2D device array): operator_apply(..., M, v, out)
  2) Matrix-free mass apply device function captured in closure:
       mass_apply(state, parameters, drivers, v, out) writes out = M v

This mirrors the style of jvp/i_minus_hj/residual_plus_i_minus_hj.
"""

from typing import Iterable, Tuple, Dict, Literal
import sympy as sp

from cubie.systemmodels.symbolic.parser import IndexedBases
from cubie.systemmodels.symbolic.numba_cuda_printer import print_cuda_multiple
from cubie.systemmodels.symbolic.jacobian import generate_analytical_jvp

# Template for the dense-M variant (M passed as a 2D array)
OPERATOR_APPLY_DENSE_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED OPERATOR APPLY (DENSE M) FACTORY\n"
    "def {func_name}(constants, precision, beta=1.0, gamma=1.0):\n"
    '    """Auto-generated operator apply with dense mass matrix.\n'
    "    Computes out = beta * (M @ v) - gamma * h * (J @ v)\n"
    "    Returns device function:\n"
    "      operator_apply(state, parameters, drivers, h, M, v, out)\n"
    "    where M has type precision[:, :].\n"
    '    """\n'
    "    n_out = {n_out}\n"
    "    n_in = {n_in}\n"
    "    from numba import cuda\n"
    "    @cuda.jit((precision[:],\n"
    "               precision[:],\n"
    "               precision[:],\n"
    "               precision,\n"
    "               precision[:, :],\n"
    "               precision[:],\n"
    "               precision[:]),\n"
    "              device=True,\n"
    "              inline=True)\n"
    "    def operator_apply(state, parameters, drivers, h, M, v, out):\n"
    "        # 1) out <- M @ v\n"
    "        for i in range(n_out):\n"
    "            acc = precision(0.0)\n"
    "            for j in range(n_in):\n"
    "                acc += M[i, j] * v[j]\n"
    "            out[i] = acc\n"
    "        # 2) Inline Jv auxiliaries and updates: out[i] = beta*out[i] - gamma*h*(Jv)_i\n"
    "{body}\n"
    "    return operator_apply\n"
)

# Template for the matrix-free mass-apply variant
OPERATOR_APPLY_MASSAPPLY_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED OPERATOR APPLY (MASS APPLY) FACTORY\n"
    "def {func_name}(mass_apply, constants, precision, beta=1.0, gamma=1.0):\n"
    '    """Auto-generated operator apply using matrix-free mass apply.\n'
    "    Computes out = beta * (M @ v) - gamma * h * (J @ v)\n"
    "    Returns device function:\n"
    "      operator_apply(state, parameters, drivers, h, v, out)\n"
    "    where 'mass_apply' is a captured device function with signature:\n"
    "      mass_apply(state, parameters, drivers, v, out)  # writes out = M v\n"
    '    """\n'
    "    n_out = {n_out}\n"
    "    from numba import cuda\n"
    "    @cuda.jit((precision[:],\n"
    "               precision[:],\n"
    "               precision[:],\n"
    "               precision,\n"
    "               precision[:],\n"
    "               precision[:]),\n"
    "              device=True,\n"
    "              inline=True)\n"
    "    def operator_apply(state, parameters, drivers, h, v, out):\n"
    "        # 1) out <- M @ v via captured mass_apply\n"
    "        mass_apply(state, parameters, drivers, v, out)\n"
    "        # 2) Inline Jv auxiliaries and updates: out[i] = beta*out[i] - gamma*h*(Jv)_i\n"
    "{body}\n"
    "    return operator_apply\n"
)

def _split_jvp_expressions(exprs: Iterable[Tuple[sp.Symbol, sp.Expr]]):
    """Split topologically-sorted (lhs, rhs) into auxiliaries and jvp terms."""
    aux = []
    jvp_terms: Dict[int, sp.Expr] = {}
    for lhs, rhs in exprs:
        lhs_str = str(lhs)
        if lhs_str.startswith("jvp["):
            idx = int(lhs_str.split("[")[1].split("]")[0])
            jvp_terms[idx] = rhs
        else:
            aux.append((lhs, rhs))
    return aux, jvp_terms

def _build_body_for_update_from_jvp(
    jvp_exprs: Iterable[Tuple[sp.Symbol, sp.Expr]],
    index_map: IndexedBases,
) -> str:
    """
    Emit code lines that:
      - compute all auxiliary temps needed for Jv
      - then do, for each i: out[i] = beta*out[i] - gamma*h*(Jv)_i
    We purposely reference out[i] on the RHS to reuse Mv computed earlier.
    """
    aux, jvp_terms = _split_jvp_expressions(jvp_exprs)

    # 1) Print auxiliaries (no jvp[...] assignments)
    aux_lines = print_cuda_multiple(aux, symbol_map=index_map.all_arrayrefs)

    # 2) Final per-output updates using inline Jv expressions
    # We create per-i assignments to out[i] with Jv[i] sp.Expr on RHS.
    n_out = len(index_map.dxdt.ref_map)
    beta_sym = sp.Symbol("beta")
    gamma_sym = sp.Symbol("gamma")
    h_sym = sp.Symbol("h")
    out_updates = []
    for i in range(n_out):
        out_sym = sp.Symbol(f"out[{i}]")
        rhs = beta_sym * out_sym - gamma_sym * h_sym * jvp_terms[i]
        out_updates.append((out_sym, rhs))

    update_lines = print_cuda_multiple(out_updates, symbol_map=index_map.all_arrayrefs)

    # Indent for insertion into template
    body_lines = []
    for ln in aux_lines:
        body_lines.append("        " + ln)
    for ln in update_lines:
        body_lines.append("        " + ln)
    if not body_lines:
        body_lines.append("        pass")
    return "\n".join(body_lines)

def generate_operator_apply_code_from_jvp(
    jvp_exprs: Iterable[Tuple[sp.Symbol, sp.Expr]],
    index_map: IndexedBases,
    func_name: str = "operator_apply_factory",
    mass_mode: Literal["dense", "apply"] = "dense",
    cse: bool = True,
) -> str:
    """
    Emit code for the operator apply factory using precomputed JVP expressions.

    mass_mode:
      - 'dense': operator takes M (precision[:, :]) as an argument
      - 'apply': operator calls captured device mass_apply(state,params,drivers,v,out)
    """
    # Order/cse handling is already baked into jvp_exprs upstream; keep as-is here.
    n_out = len(index_map.dxdt.ref_map)
    n_in = len(index_map.states.index_map)

    body = _build_body_for_update_from_jvp(jvp_exprs, index_map)

    if mass_mode == "dense":
        return OPERATOR_APPLY_DENSE_TEMPLATE.format(
            func_name=func_name, n_out=n_out, n_in=n_in, body=body
        )
    else:
        return OPERATOR_APPLY_MASSAPPLY_TEMPLATE.format(
            func_name=func_name, n_out=n_out, body=body
        )

def generate_operator_apply_code(
    equations: Iterable[Tuple[sp.Symbol, sp.Expr]],
    index_map: IndexedBases,
    func_name: str = "operator_apply_factory",
    mass_mode: Literal["dense", "apply"] = "dense",
    cse: bool = True,
) -> str:
    """
    High-level entry: build JVP expressions, then emit operator apply code
    for a non-diagonal mass matrix.

    - mass_mode='dense'  → operator_apply(..., M, v, out)
    - mass_mode='apply'  → operator_apply(..., v, out) calling captured mass_apply
    """
    jvp_exprs = generate_analytical_jvp(
        equations,
        input_order=index_map.states.index_map,
        output_order=index_map.dxdt.index_map,
        observables=index_map.observable_symbols,
        cse=cse,
    )
    return generate_operator_apply_code_from_jvp(
        jvp_exprs=jvp_exprs,
        index_map=index_map,
        func_name=func_name,
        mass_mode=mass_mode,
        cse=cse,
    )

# --- Solver bridge (edits required if you choose 'dense' mass_mode) ---

# File: 'src/cubie/integrators/matrix_free_solvers/linear_solver.py'
# If you adopt the dense-M operator, update the operator_apply calls to pass M:
#
#   operator_apply(state, parameters, drivers, h, M, x, v_vec)         # initial apply
#   ...
#   operator_apply(state, parameters, drivers, h, M, z_vec, v_vec)     # in each iter
#
# For backwards-compatibility, you can provide a second factory:
#
# def linear_solver_with_mass_factory(operator_apply_M, preconditioner=None, ...):
#     @cuda.jit(device=True)
#     def linear_solver(state, parameters, drivers, h, M, rhs, x, residual, z_vec, v_vec):
#         # identical to current solver, but pass M through to operator_apply_M(...)
#         ...
#     return linear_solver
#
# If you choose the matrix-free 'mass_apply' path, no solver changes are needed:
# generate operator_apply(state, parameters, drivers, h, v, out) and use as before.