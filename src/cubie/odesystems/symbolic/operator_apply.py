"""Code generation for the linear operator ``β·M·v − γ·h·J·v``.

The mass matrix ``M`` is provided at code-generation time either as a NumPy
array or a SymPy matrix. Its entries are embedded directly into the generated
device routine to avoid extra passes or buffers.
"""

from typing import Iterable, Tuple, Dict
import sympy as sp

from cubie.odesystems.symbolic.parser import IndexedBases
from cubie.odesystems.symbolic.numba_cuda_printer import print_cuda_multiple
from cubie.odesystems.symbolic.jacobian import generate_analytical_jvp

OPERATOR_APPLY_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED LINEAR OPERATOR FACTORY\n"
    "def {func_name}(constants, precision, beta=1.0, gamma=1.0):\n"
    '    """Auto-generated linear operator.\n'
    "    Computes out = beta * (M @ v) - gamma * h * (J @ v)\n"
    "    Returns device function:\n"
    "      operator_apply(state, parameters, drivers, h, v, out)\n"
    '    """\n'
    "{const_lines}"
    "    @cuda.jit((precision[:],\n"
    "               precision[:],\n"
    "               precision[:],\n"
    "               precision,\n"
    "               precision[:],\n"
    "               precision[:]),\n"
    "              device=True,\n"
    "              inline=True)\n"
    "    def operator_apply(state, parameters, drivers, h, v, out):\n"
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

def _build_body_from_jvp(
    jvp_exprs: Iterable[Tuple[sp.Symbol, sp.Expr]],
    index_map: IndexedBases,
    M: sp.Matrix,
) -> str:
    """Return code body computing ``β·M·v − γ·h·Jv``."""
    aux, jvp_terms = _split_jvp_expressions(jvp_exprs)

    n_out = len(index_map.dxdt.ref_map)
    n_in = len(index_map.states.index_map)
    v = sp.IndexedBase("v")
    beta_sym = sp.Symbol("beta")
    gamma_sym = sp.Symbol("gamma")
    h_sym = sp.Symbol("h")

    mass_assigns = []
    out_updates = []
    for i in range(n_out):
        mv = sp.S.Zero
        for j in range(n_in):
            entry = M[i, j]
            if entry == 0:
                continue
            sym = sp.Symbol(f"m_{i}{j}")
            mass_assigns.append((sym, entry))
            mv += sym * v[j]
        rhs = beta_sym * mv - gamma_sym * h_sym * jvp_terms[i]
        out_updates.append((sp.Symbol(f"out[{i}]"), rhs))

    exprs = mass_assigns + aux + out_updates
    lines = print_cuda_multiple(exprs, symbol_map=index_map.all_arrayrefs)
    if not lines:
        return "        pass"
    return "\n".join("        " + ln for ln in lines)


def generate_operator_apply_code_from_jvp(
    jvp_exprs: Iterable[Tuple[sp.Symbol, sp.Expr]],
    index_map: IndexedBases,
    M: sp.Matrix,
    func_name: str = "operator_apply_factory",
    cse: bool = True,
) -> str:
    """Emit code for the operator apply factory using precomputed JVP expressions.

    The emitted factory expects ``constants`` as a mapping from names to values
    and embeds each constant as a standalone variable in the generated device
    function.
    """
    body = _build_body_from_jvp(jvp_exprs, index_map, M)
    const_lines = [
        f"    {name} = precision(constants['{name}'])"
        for name in index_map.constants.symbol_map
    ]
    const_block = "\n".join(const_lines) + ("\n" if const_lines else "")
    return OPERATOR_APPLY_TEMPLATE.format(
        func_name=func_name, body=body, const_lines=const_block
    )


def generate_operator_apply_code(
    equations: Iterable[Tuple[sp.Symbol, sp.Expr]],
    index_map: IndexedBases,
    M=None,
    func_name: str = "operator_apply_factory",
    cse: bool = True,
) -> str:
    """High-level entry: build JVP expressions, then emit operator apply code."""
    if M is None:
        n = len(index_map.states.index_map)
        M_mat = sp.eye(n)
    else:
        M_mat = sp.Matrix(M)
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
        M=M_mat,
        func_name=func_name,
        cse=cse,
    )


# ---------------------------------------------------------------------------
# Neumann preconditioner code generation
# ---------------------------------------------------------------------------

NEUMANN_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED NEUMANN PRECONDITIONER FACTORY\n"
    "def {func_name}(constants, precision, beta=1.0, gamma=1.0, order=1):\n"
    '    """Auto-generated Neumann preconditioner.\n'
    "    Approximates (beta*M - gamma*h*J)^[-1] via a truncated\n"
    "    Neumann series. Returns device function:\n"
    "      preconditioner(state, parameters, drivers, h, v, out, jvp)\n"
    "    where `jvp` is a caller-provided scratch buffer for J*v.\n"
    '    """\n'
    "    n = {n_out}\n"
    "    beta_inv = 1.0 / beta\n"
    "    h_eff_factor = gamma * beta_inv\n"
    "{const_lines}"
    "    @cuda.jit((precision[:],\n"
    "               precision[:],\n"
    "               precision[:],\n"
    "               precision,\n"
    "               precision[:],\n"
    "               precision[:],\n"
    "               precision[:]),\n"
    "              device=True,\n"
    "              inline=True)\n"
    "    def preconditioner(state, parameters, drivers, h, v, out, jvp):\n"
    "        # Horner form: S[m] = v + T S[m-1], T = (gamma/beta) * h * J\n"
    "        # Accumulator lives in `out`. Uses caller-provided `jvp` for "
    "JVP.\n"
    "        for i in range(n):\n"
    "            out[i] = v[i]\n"
    "        h_eff = h * h_eff_factor\n"
    "        for _ in range(order):\n"
    "{jv_body}\n"
    "            for i in range(n):\n"
    "                out[i] = v[i] + h_eff * jvp[i]\n"
    "        for i in range(n):\n"
    "            out[i] = beta_inv * out[i]\n"
    "    return preconditioner\n"
)


def generate_neumann_preconditioner_code(
    equations: Iterable[Tuple[sp.Symbol, sp.Expr]],
    index_map: IndexedBases,
    func_name: str = "neumann_preconditioner_factory",
    cse: bool = True,
) -> str:
    """High-level entry for Neumann preconditioner code generation.

    Parameters
    ----------
    equations : iterable of tuple
        Differential equations defining the system.
    index_map : IndexedBases
        Mapping of symbolic arrays to CUDA references.
    func_name : str, optional
        Name of the emitted factory, default
        ``"neumann_preconditioner_factory"``.
    cse : bool, optional
        Apply common-subexpression elimination, default ``True``.

    Returns
    -------
    str
        Source code for the factory function.
    """
    n_out = len(index_map.dxdt.ref_map)
    const_lines = [
        f"    {name} = precision(constants['{name}'])"
        for name in index_map.constants.symbol_map
    ]
    const_block = "\n".join(const_lines) + ("\n" if const_lines else "")
    jvp_exprs = generate_analytical_jvp(
        equations,
        input_order=index_map.states.index_map,
        output_order=index_map.dxdt.index_map,
        observables=index_map.observable_symbols,
        cse=cse,
    )
    # Emit using canonical names, then rewrite to drive JVP with `out` and
    # write into the caller-provided scratch buffer `jvp`.
    lines = print_cuda_multiple(jvp_exprs, symbol_map=index_map.all_arrayrefs)
    if not lines:
        lines = ["pass"]
    else:
        lines = [
            ln.replace("v[", "out[").replace("jvp[", "jvp[")
            for ln in lines
        ]
    jv_body = "\n".join("            " + ln for ln in lines)
    return NEUMANN_TEMPLATE.format(
            func_name=func_name, n_out=n_out, jv_body=jv_body,
            const_lines=const_block
    )


# ---------------------------------------------------------------------------
# Residual function code generation (Unified, compile-time mode)
# ---------------------------------------------------------------------------

RESIDUAL_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED RESIDUAL FACTORY\n"
    "def {func_name}(constants, precision, dxdt,  beta=1.0, gamma=1.0):\n"
    '    """Auto-generated residual. Mode fixed at codegen time.\n'
    "    - Stage mode: eval at base_state + a_ij*u; residual uses M@u\n"
    "    - End-state mode: eval at u; residual uses M@(u - base_state)\n"
    '    """\n'
    "{const_lines}"
    "    @cuda.jit((precision[:],\n"
    "               precision[:],\n"
    "               precision[:],\n"
    "               precision,\n"
    "               precision,\n"  
    "               precision[:],\n"
    "               precision[:],\n"
    "               precision[:]),\n"
    "              device=True,\n"
    "              inline=True)\n"
    "    def residual(u, parameters, drivers, h, a_ij, base_state, work, out):\n"
    "{eval_lines}\n"
    "\n"
    "        dxdt(work, parameters, drivers, work, out)\n"
    "\n"
    "{res_lines}\n"
    "    return residual\n"
)


def _build_residual_mode_lines(index_map: IndexedBases, M: sp.Matrix, is_stage: bool) -> Tuple[str, str]:
    """Return eval and residual lines for the chosen residual mode.

    - eval_lines: assignments to `work` for dxdt evaluation point
    - res_lines: final residual updates into `out`
    """
    n = len(index_map.states.index_map)

    beta_sym = sp.Symbol("beta")
    gamma_sym = sp.Symbol("gamma")
    h_sym = sp.Symbol("h")
    aij_sym = sp.Symbol("a_ij")
    u = sp.IndexedBase("u", shape=(n,))
    base = sp.IndexedBase("base_state", shape=(n,))
    work = sp.IndexedBase("work", shape=(n,))
    out = sp.IndexedBase("out", shape=(n,))

    symbol_map = dict(index_map.all_arrayrefs)
    symbol_map.update(
        {
            "beta": beta_sym,
            "gamma": gamma_sym,
            "h": h_sym,
            "a_ij": aij_sym,
            "u": u,
            "base_state": base,
            "work": work,
            "out": out,
        }
    )

    # Eval point
    eval_exprs = []
    for i in range(n):
        if is_stage:
            eval_exprs.append((work[i], base[i] + aij_sym * u[i]))
        else:
            eval_exprs.append((work[i], u[i]))
    eval_lines_list = print_cuda_multiple(eval_exprs, symbol_map=symbol_map) or ["pass"]
    eval_lines = "\n".join("        " + ln for ln in eval_lines_list)

    # Residual
    res_exprs = []
    for i in range(n):
        mv = sp.S.Zero
        for j in range(n):
            entry = M[i, j]
            if entry == 0:
                continue
            if is_stage:
                mv += entry * u[j]
            else:
                mv += entry * (u[j] - base[j])
        res_exprs.append((out[i], beta_sym * mv - gamma_sym * h_sym * out[i]))
    res_lines_list = print_cuda_multiple(res_exprs, symbol_map=symbol_map) or ["pass"]
    res_lines = "\n".join("        " + ln for ln in res_lines_list)

    return eval_lines, res_lines


def generate_residual_end_state_code(
    index_map: IndexedBases,
    M=None,
    func_name: str = "residual_end_state_factory",
) -> str:
    """Emit code for the end-state residual factory (compile-time mode)."""
    if M is None:
        n = len(index_map.states.index_map)
        M_mat = sp.eye(n)
    else:
        M_mat = sp.Matrix(M)
    eval_lines, res_lines = _build_residual_mode_lines(index_map, M_mat, is_stage=False)
    
    const_lines = [
        f"    {name} = precision(constants['{name}'])"
        for name in index_map.constants.symbol_map
    ]
    const_block = "\n".join(const_lines) + ("\n" if const_lines else "")

    return RESIDUAL_TEMPLATE.format(
        func_name=func_name,
        const_lines=const_block,
        eval_lines=eval_lines,
        res_lines=res_lines,
    )


def generate_stage_residual_code(
    index_map: IndexedBases,
    M=None,
    func_name: str = "stage_residual_factory",
) -> str:
    """Emit code for the stage residual factory (compile-time mode)."""
    if M is None:
        n = len(index_map.states.index_map)
        M_mat = sp.eye(n)
    else:
        M_mat = sp.Matrix(M)
    eval_lines, res_lines = _build_residual_mode_lines(index_map, M_mat, is_stage=True)
    const_lines = [
        f"    {name} = precision(constants['{name}'])"
        for name in index_map.constants.symbol_map
    ]
    const_block = "\n".join(const_lines) + ("\n" if const_lines else "")

    mode_lines = "    a_ij = precision(a_ij)\n"
    return RESIDUAL_TEMPLATE.format(
        func_name=func_name,
        extra_args="a_ij, ",
        const_lines=const_block,
        mode_lines=mode_lines,
        eval_lines=eval_lines,
        res_lines=res_lines,
    )
