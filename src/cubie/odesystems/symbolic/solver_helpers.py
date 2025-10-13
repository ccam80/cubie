"""Code generation helpers for implicit solver linear operators and residuals.

The mass matrix ``M`` is provided at code-generation time either as a NumPy
array or a SymPy matrix. Its entries are embedded directly into the generated
device routine to avoid extra passes or buffers.
"""

from typing import Dict, Iterable, List, Optional, Tuple, Union
import sympy as sp

from cubie.odesystems.symbolic import cse_and_stack, topological_sort
from cubie.odesystems.symbolic.parser import IndexedBases
from cubie.odesystems.symbolic.numba_cuda_printer import print_cuda_multiple
from cubie.odesystems.symbolic.jacobian import generate_analytical_jvp

CACHED_OPERATOR_APPLY_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED CACHED LINEAR OPERATOR FACTORY\n"
    "def {func_name}(constants, precision, beta=1.0, gamma=1.0, order=None):\n"
    '    """Auto-generated cached linear operator.\n'
    "    Computes out = beta * (M @ v) - gamma * h * (J @ v)\n"
    "    using cached auxiliary intermediates.\n"
    "    Returns device function:\n"
    "      operator_apply(state, parameters, drivers, cached_aux, h, v, out)\n"
    "    argument 'order' is ignored, included for compatibility with \n"
    "    preconditioner API. \n"
    '    """\n'
    "{const_lines}"
    "    @cuda.jit((precision[:],\n"
    "               precision[:],\n"
    "               precision[:],\n"
    "               precision[:],\n"
    "               precision,\n"
    "               precision[:],\n"
    "               precision[:]),\n"
    "              device=True,\n"
    "              inline=True)\n"
    "    def operator_apply(state, parameters, drivers, cached_aux, h, v, out):\n"
    "{body}\n"
    "    return operator_apply\n"
)


OPERATOR_APPLY_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED LINEAR OPERATOR FACTORY\n"
    "def {func_name}(constants, precision, beta=1.0, gamma=1.0, order=None):\n"
    '    """Auto-generated linear operator.\n'
    "    Computes out = beta * (M @ v) - gamma * h * (J @ v)\n"
    "    Returns device function:\n"
    "      operator_apply(state, parameters, drivers, h, v, out)\n"
    "    argument 'order' is ignored, included for compatibility with \n"
    "    preconditioner API. \n"
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


PREPARE_JAC_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED JACOBIAN PREPARATION FACTORY\n"
    "def {func_name}(constants, precision):\n"
    '    """Auto-generated Jacobian auxiliary preparation.\n'
    "    Populates cached_aux with intermediate Jacobian values.\n"
    '    """\n'
    "{const_lines}"
    "    @cuda.jit((precision[:],\n"
    "               precision[:],\n"
    "               precision[:],\n"
    "               precision[:]),\n"
    "              device=True,\n"
    "              inline=True)\n"
    "    def prepare_jac(state, parameters, drivers, cached_aux):\n"
    "{body}\n"
    "    return prepare_jac\n"
)


CACHED_JVP_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED CACHED JVP FACTORY\n"
    "def {func_name}(constants, precision):\n"
    '    """Auto-generated cached Jacobian-vector product.\n'
    "    Computes out = J @ v using cached auxiliaries.\n"
    '    """\n'
    "{const_lines}"
    "    @cuda.jit((precision[:],\n"
    "               precision[:],\n"
    "               precision[:],\n"
    "               precision[:],\n"
    "               precision[:],\n"
    "               precision[:]),\n"
    "              device=True,\n"
    "              inline=True)\n"
    "    def calculate_cached_jvp(\n"
    "        state, parameters, drivers, cached_aux, v, out\n"
    "    ):\n"
    "{body}\n"
    "    return calculate_cached_jvp\n"
)


def _split_jvp_expressions(
    exprs: Iterable[Tuple[sp.Symbol, sp.Expr]]
) -> Tuple[List[Tuple[sp.Symbol, sp.Expr]], Dict[int, sp.Expr]]:
    """Split topologically sorted expressions into auxiliaries and JVP terms.

    Parameters
    ----------
    exprs
        Expression pairs ordered for evaluation.

    Returns
    -------
    tuple of list and dict
        Auxiliary assignments followed by the indexed JVP expressions.
    """
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

def _build_operator_body(
    aux: List[Tuple[sp.Symbol, sp.Expr]],
    jvp_terms: Dict[int, sp.Expr],
    index_map: IndexedBases,
    M: sp.Matrix,
    cached_aux: bool = False,
) -> str:
    """Build the CUDA body computing ``β·M·v − γ·h·J·v``.

    Parameters
    ----------
    aux
        Auxiliary assignments that populate intermediate values.
    jvp_terms
        Mapping from output indices to the Jacobian-vector expressions.
    index_map
        Symbol indexing helpers produced by the parser.
    M
        Mass matrix to embed into the generated operator.
    cached_aux
        When ``True`` load auxiliary values from ``cached_aux`` instead of
        recomputing them.

    Returns
    -------
    str
        Indented CUDA code statements implementing the operator body.
    """
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

    if cached_aux:
        if aux:
            cached = sp.IndexedBase(
                "cached_aux", shape=(sp.Integer(len(aux)),)
            )
        else:
            cached = sp.IndexedBase("cached_aux")
        aux_assignments = [
            (lhs, cached[idx]) for idx, (lhs, _) in enumerate(aux)
        ]
    else:
        aux_assignments = aux

    exprs = mass_assigns + aux_assignments + out_updates
    lines = print_cuda_multiple(exprs, symbol_map=index_map.all_arrayrefs)
    if not lines:
        return "        pass"
    return "\n".join("        " + ln for ln in lines)


def _build_cached_neumann_body(
    jvp_exprs: Iterable[Tuple[sp.Symbol, sp.Expr]],
    index_map: IndexedBases,
) -> str:
    """Build the cached Neumann-series Jacobian-vector body."""

    aux, jvp_terms = _split_jvp_expressions(jvp_exprs)
    if aux:
        cached = sp.IndexedBase(
            "cached_aux", shape=(sp.Integer(len(aux)),)
        )
    else:
        cached = sp.IndexedBase("cached_aux")
    aux_assignments = [
        (lhs, cached[idx]) for idx, (lhs, _) in enumerate(aux)
    ]

    n_out = len(index_map.dxdt.ref_map)
    exprs: List[Tuple[sp.Symbol, sp.Expr]] = list(aux_assignments)
    for i in range(n_out):
        rhs = jvp_terms.get(i, sp.S.Zero)
        exprs.append((sp.Symbol(f"jvp[{i}]"), rhs))

    lines = print_cuda_multiple(exprs, symbol_map=index_map.all_arrayrefs)
    if not lines:
        return "            pass"
    replaced = [ln.replace("v[", "out[") for ln in lines]
    return "\n".join("            " + ln for ln in replaced)


def _build_cached_jvp_body(
    aux: List[Tuple[sp.Symbol, sp.Expr]],
    jvp_terms: Dict[int, sp.Expr],
    index_map: IndexedBases,
) -> str:
    """Build the CUDA body computing ``J·v`` with optional cached auxiliaries."""

    n_out = len(index_map.dxdt.ref_map)

    if aux:
        cached = sp.IndexedBase(
            "cached_aux", shape=(sp.Integer(len(aux)),)
        )
    else:
        cached = sp.IndexedBase("cached_aux")

    aux_assignments = [
        (lhs, cached[idx]) for idx, (lhs, _) in enumerate(aux)
    ]

    out_updates = []
    for i in range(n_out):
        rhs = jvp_terms.get(i, sp.S.Zero)
        out_updates.append((sp.Symbol(f"out[{i}]"), rhs))

    exprs = aux_assignments + out_updates
    lines = print_cuda_multiple(exprs, symbol_map=index_map.all_arrayrefs)
    if not lines:
        return "        pass"
    return "\n".join("        " + ln for ln in lines)


def _build_prepare_body(
    aux: List[Tuple[sp.Symbol, sp.Expr]], index_map: IndexedBases
) -> str:
    """Build the CUDA body populating the cached Jacobian auxiliaries."""

    if aux:
        cached = sp.IndexedBase(
            "cached_aux", shape=(sp.Integer(len(aux)),)
        )
    else:
        cached = sp.IndexedBase("cached_aux")
    exprs: List[Tuple[sp.Symbol, sp.Expr]] = []
    for idx, (lhs, rhs) in enumerate(aux):
        exprs.append((lhs, rhs))
        exprs.append((cached[idx], lhs))

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
    """Emit the operator apply factory from precomputed JVP expressions.

    Parameters
    ----------
    jvp_exprs
        Topologically ordered Jacobian-vector product expressions.
    index_map
        Symbol indexing helpers produced by the parser.
    M
        Mass matrix to embed into the generated operator.
    func_name
        Name assigned to the emitted factory.
    cse
        Unused placeholder kept for signature stability.

    Returns
    -------
    str
        Source code for the linear operator factory.

    Notes
    -----
    The emitted factory expects ``constants`` as a mapping from names to values
    and embeds each constant as a standalone variable in the generated device
    function.
    """
    aux, jvp_terms = _split_jvp_expressions(jvp_exprs)
    body = _build_operator_body(aux, jvp_terms, index_map, M, cached_aux=False)
    const_lines = [
        f"    {name} = precision(constants['{name}'])"
        for name in index_map.constants.symbol_map
    ]
    const_block = "\n".join(const_lines) + ("\n" if const_lines else "")
    return OPERATOR_APPLY_TEMPLATE.format(
        func_name=func_name, body=body, const_lines=const_block
    )


def generate_cached_operator_apply_code_from_jvp(
    jvp_exprs: Iterable[Tuple[sp.Symbol, sp.Expr]],
    index_map: IndexedBases,
    M: sp.Matrix,
    func_name: str = "linear_operator_cached",
) -> str:
    """Emit the cached linear operator factory from JVP expressions."""

    aux, jvp_terms = _split_jvp_expressions(jvp_exprs)
    body = _build_operator_body(aux, jvp_terms, index_map, M, cached_aux=True)
    const_lines = [
        f"    {name} = precision(constants['{name}'])"
        for name in index_map.constants.symbol_map
    ]
    const_block = "\n".join(const_lines) + ("\n" if const_lines else "")
    return CACHED_OPERATOR_APPLY_TEMPLATE.format(
        func_name=func_name,
        body=body,
        const_lines=const_block,
    )


def generate_prepare_jac_code_from_jvp(
    jvp_exprs: Iterable[Tuple[sp.Symbol, sp.Expr]],
    index_map: IndexedBases,
    func_name: str = "prepare_jac",
) -> Tuple[str, int]:
    """Emit the auxiliary preparation factory from JVP expressions."""

    aux, _ = _split_jvp_expressions(jvp_exprs)
    body = _build_prepare_body(aux, index_map)
    const_lines = [
        f"    {name} = precision(constants['{name}'])"
        for name in index_map.constants.symbol_map
    ]
    const_block = "\n".join(const_lines) + ("\n" if const_lines else "")
    code = PREPARE_JAC_TEMPLATE.format(
        func_name=func_name, body=body, const_lines=const_block
    )
    return code, len(aux)


def generate_cached_jvp_code_from_jvp(
    jvp_exprs: Iterable[Tuple[sp.Symbol, sp.Expr]],
    index_map: IndexedBases,
    func_name: str = "calculate_cached_jvp",
) -> str:
    """Emit the cached JVP factory from precomputed JVP expressions."""

    aux, jvp_terms = _split_jvp_expressions(jvp_exprs)
    body = _build_cached_jvp_body(aux, jvp_terms, index_map)
    const_lines = [
        f"    {name} = precision(constants['{name}'])"
        for name in index_map.constants.symbol_map
    ]
    const_block = "\n".join(const_lines) + ("\n" if const_lines else "")
    code = CACHED_JVP_TEMPLATE.format(
        func_name=func_name, body=body, const_lines=const_block
    )
    return code


def generate_operator_apply_code(
    equations: Iterable[Tuple[sp.Symbol, sp.Expr]],
    index_map: IndexedBases,
    M: Optional[Union[sp.Matrix, Iterable[Iterable[sp.Expr]]]] = None,
    func_name: str = "operator_apply_factory",
    cse: bool = True,
) -> str:
    """Generate the linear operator factory from system equations.

    Parameters
    ----------
    equations
        Differential equation tuples defining ``dx/dt``.
    index_map
        Symbol indexing helpers produced by the parser.
    M
        Mass matrix supplied as a SymPy matrix or nested iterable. Uses the
        identity matrix when omitted.
    func_name
        Name assigned to the emitted factory.
    cse
        Apply common subexpression elimination before emission.

    Returns
    -------
    str
        Source code for the linear operator factory.
    """
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


def generate_cached_operator_apply_code(
    equations: Iterable[Tuple[sp.Symbol, sp.Expr]],
    index_map: IndexedBases,
    M: Optional[Union[sp.Matrix, Iterable[Iterable[sp.Expr]]]] = None,
    func_name: str = "linear_operator_cached",
    cse: bool = True,
) -> str:
    """Generate the cached linear operator factory."""

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
    return generate_cached_operator_apply_code_from_jvp(
        jvp_exprs=jvp_exprs,
        index_map=index_map,
        M=M_mat,
        func_name=func_name,
    )


def generate_prepare_jac_code(
    equations: Iterable[Tuple[sp.Symbol, sp.Expr]],
    index_map: IndexedBases,
    func_name: str = "prepare_jac",
    cse: bool = True,
) -> Tuple[str, int]:
    """Generate the cached auxiliary preparation factory."""

    jvp_exprs = generate_analytical_jvp(
        equations,
        input_order=index_map.states.index_map,
        output_order=index_map.dxdt.index_map,
        observables=index_map.observable_symbols,
        cse=cse,
    )
    return generate_prepare_jac_code_from_jvp(
        jvp_exprs=jvp_exprs,
        index_map=index_map,
        func_name=func_name,
    )


def generate_cached_jvp_code(
    equations: Iterable[Tuple[sp.Symbol, sp.Expr]],
    index_map: IndexedBases,
    func_name: str = "calculate_cached_jvp",
    cse: bool = True,
) -> str:
    """Generate the cached Jacobian-vector product factory."""

    jvp_exprs = generate_analytical_jvp(
        equations,
        input_order=index_map.states.index_map,
        output_order=index_map.dxdt.index_map,
        observables=index_map.observable_symbols,
        cse=cse,
    )
    return generate_cached_jvp_code_from_jvp(
        jvp_exprs=jvp_exprs,
        index_map=index_map,
        func_name=func_name,
    )



# ---------------------------------------------------------------------------
# Neumann preconditioner code generation
# ---------------------------------------------------------------------------

NEUMANN_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED NEUMANN PRECONDITIONER FACTORY\n"
    "def {func_name}(constants, precision, beta=1.0, gamma=1.0, order=1):\n"
    '    """Auto-generated Neumann preconditioner.\n'
    "    Approximates (beta*I - gamma*h*J)^[-1] via a truncated\n"
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


NEUMANN_CACHED_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED CACHED NEUMANN PRECONDITIONER FACTORY\n"
    "def {func_name}(constants, precision, beta=1.0, gamma=1.0, order=1):\n"
    '    """Cached Neumann preconditioner using stored auxiliaries.\n'
    "    Approximates (beta*I - gamma*h*J)^[-1] via a truncated\n"
    "    Neumann series with cached auxiliaries. Returns device function:\n"
    "      preconditioner(\n"
    "          state, parameters, drivers, cached_aux, h, v, out, jvp\n"
    "      )\n"
    '    """\n'
    "    n = {n_out}\n"
    "    beta_inv = 1.0 / beta\n"
    "    h_eff_factor = gamma * beta_inv\n"
    "{const_lines}"
    "    @cuda.jit((precision[:],\n"
    "               precision[:],\n"
    "               precision[:],\n"
    "               precision[:],\n"
    "               precision,\n"
    "               precision[:],\n"
    "               precision[:],\n"
    "               precision[:]),\n"
    "              device=True,\n"
    "              inline=True)\n"
    "    def preconditioner(\n"
    "        state, parameters, drivers, cached_aux, h, v, out, jvp):\n"
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
    """Generate the Neumann preconditioner factory.

    Parameters
    ----------
    equations
        Differential equations defining the system.
    index_map
        Symbol indexing helpers produced by the parser.
    func_name
        Name assigned to the emitted factory.
    cse
        Apply common subexpression elimination before emission.

    Returns
    -------
    str
        Source code for the Neumann preconditioner factory.
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


def generate_neumann_preconditioner_cached_code(
    equations: Iterable[Tuple[sp.Symbol, sp.Expr]],
    index_map: IndexedBases,
    func_name: str = "neumann_preconditioner_cached",
    cse: bool = True,
) -> str:
    """Generate the cached Neumann preconditioner factory."""

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
    jv_body = _build_cached_neumann_body(jvp_exprs, index_map)
    return NEUMANN_CACHED_TEMPLATE.format(
        func_name=func_name,
        n_out=n_out,
        jv_body=jv_body,
        const_lines=const_block,
    )


# ---------------------------------------------------------------------------
# Residual function code generation (Unified, compile-time mode)
# ---------------------------------------------------------------------------

RESIDUAL_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED RESIDUAL FACTORY\n"
    "def {func_name}(constants, precision,  beta=1.0, gamma=1.0, order=None):\n"
    '    """Auto-generated residual function for Newton-Krylov ODE integration.\n'
    "    \n"
    "    Computes residual = beta * M @ v - gamma * h * (J @ eval_point)\n"
    "    where eval_point depends on the residual mode:\n"
    "    - Stage mode: eval_point = base_state + a_ij * u, residual uses M @ u\n"
    "    - End-state mode: eval_point = u, residual uses M @ (u - base_state)\n"
    "    \n"
    "    Uses dx_ numbered symbols for derivatives and aux_ symbols for observables,\n"
    "    following the same pattern as JVP generation.\n"
    "    \n"
    "    Order is ignored, included for compatibility with preconditioner API.\n"
    '    """\n'
    "{const_lines}"
    "    @cuda.jit((precision[:],\n"
    "               precision[:],\n"
    "               precision[:],\n"
    "               precision,\n"
    "               precision,\n"  
    "               precision[:],\n"
    "               precision[:]),\n"
    "              device=True,\n"
    "              inline=True)\n"
    "    def residual(u, parameters, drivers, h, a_ij, base_state, out):\n"
    "{res_lines}\n"
    "    return residual\n"
)


def _build_residual_lines(
    equations: Union[
        Iterable[Tuple[sp.Symbol, sp.Expr]], Dict[sp.Symbol, sp.Expr]
    ],
    index_map: IndexedBases,
    M: sp.Matrix,
    is_stage: bool,
    cse: bool = True,
) -> str:
    """Construct CUDA code lines for the requested residual mode.

    Parameters
    ----------
    equations
        Differential equation tuples or dictionary defining ``dx/dt``.
    index_map
        Symbol indexing helpers produced by the parser.
    M
        Mass matrix to embed into the generated residual.
    is_stage
        Flag selecting stage or end-state residual evaluation.
    cse
        Apply common subexpression elimination before emission.

    Returns
    -------
    str
        Indented CUDA code statements for the residual body.

    Notes
    -----
    Derivative symbols are rewritten to ``dx_`` indices and observables to
    ``aux_`` indices to mirror Jacobian-vector product emission.
    """
    if isinstance(equations, dict):
        eq_list = list(equations.items())
    else:
        eq_list = list(equations)

    n = len(index_map.states.index_map)

    beta_sym = sp.Symbol("beta")
    gamma_sym = sp.Symbol("gamma")
    h_sym = sp.Symbol("h")
    aij_sym = sp.Symbol("a_ij")
    u = sp.IndexedBase("u", shape=(n,))
    base = sp.IndexedBase("base_state", shape=(n,))
    out = sp.IndexedBase("out", shape=(n,))

    # Create symbol substitutions like in JVP generation
    # Convert dx variables to dx_ numbered symbols
    dx_subs = {}
    for i, (dx_sym, _) in enumerate(index_map.dxdt.index_map.items()):
        dx_subs[dx_sym] = sp.Symbol(f"dx_{i}")

    # Convert observable symbols to aux_ symbols
    obs_subs = {}
    if index_map.observable_symbols:
        obs_subs = dict(zip(index_map.observable_symbols,
                           sp.numbered_symbols("aux_", start=1)))

    # Apply substitutions to equations
    all_subs = {**dx_subs, **obs_subs}
    substituted_equations = [(lhs.subs(all_subs), rhs.subs(all_subs))
                            for lhs, rhs in eq_list]

    # Create evaluation point substitutions for state variables
    state_subs = {}
    state_symbols = list(index_map.states.index_map.keys())
    for i, state_sym in enumerate(state_symbols):
        if is_stage:
            # Stage mode: evaluation point is base + a_ij * u
            eval_point = base[i] + aij_sym * u[i]
        else:
            # End-state mode: evaluation point is u
            eval_point = u[i]
        state_subs[state_sym] = eval_point

    # Apply state substitutions to the RHS of equations
    eval_equations = []
    for lhs, rhs in substituted_equations:
        eval_rhs = rhs.subs(state_subs)
        eval_equations.append((lhs, eval_rhs))

    symbol_map = dict(index_map.all_arrayrefs)
    symbol_map.update({
        "beta": beta_sym,
        "gamma": gamma_sym,
        "h": h_sym,
        "a_ij": aij_sym,
        "u": u,
        "base_state": base,
        "out": out,
    })

    # Build complete expression list
    eval_exprs = eval_equations

    # Build residual expressions
    for i in range(n):
        mv = sp.S.Zero
        for j in range(n):
            entry = M[i, j]
            if entry == 0:
                continue
            if is_stage:
                # Stage mode: M @ u
                mv += entry * u[j]
            else:
                # End-state mode: M @ (u - base)
                mv += entry * (u[j] - base[j])
        
        # Get the dx symbol for this output
        dx_sym = sp.Symbol(f"dx_{i}")
        residual_expr = beta_sym * mv - gamma_sym * h_sym * dx_sym
        eval_exprs.append((out[i], residual_expr))

    if cse:
        eval_exprs = cse_and_stack(eval_exprs)
    else:
        eval_exprs = topological_sort(eval_exprs)

    lines = print_cuda_multiple(eval_exprs, symbol_map=symbol_map)
    if not lines:
        return "        pass"
    return "\n".join("        " + ln for ln in lines)

def generate_residual_code(
    equations: Iterable[Tuple[sp.Symbol, sp.Expr]],
    index_map: IndexedBases,
    M: Optional[Union[sp.Matrix, Iterable[Iterable[sp.Expr]]]] = None,
    is_stage: bool = True,
    func_name: str = "residual_factory",
    cse: bool = True,
) -> str:
    """Emit the residual factory for Newton--Krylov integration.

    Parameters
    ----------
    equations
        Differential equation tuples defining ``dx/dt``.
    index_map
        Symbol indexing helpers produced by the parser.
    M
        Mass matrix supplied as a SymPy matrix or nested iterable. Uses the
        identity matrix when omitted.
    is_stage
        Generate the stage residual when ``True``; otherwise emit the
        end-state residual.
    func_name
        Name assigned to the emitted factory.
    cse
        Apply common subexpression elimination before emission.

    Returns
    -------
    str
        Source code for the residual factory.
    """
    if M is None:
        n = len(index_map.states.index_map)
        M_mat = sp.eye(n)
    else:
        M_mat = sp.Matrix(M)

    res_lines = _build_residual_lines(
        equations=equations,
        index_map=index_map,
        M=M_mat,
        is_stage=is_stage,
        cse=cse,
    )
    const_lines = [
        f"    {name} = precision(constants['{name}'])"
        for name in index_map.constants.symbol_map
    ]
    const_block = "\n".join(const_lines) + ("\n" if const_lines else "")

    return RESIDUAL_TEMPLATE.format(
            func_name=func_name,
            const_lines=const_block,
            res_lines=res_lines,
    )

def generate_residual_end_state_code(
    equations: Iterable[Tuple[sp.Symbol, sp.Expr]],
    index_map: IndexedBases,
    M: Optional[Union[sp.Matrix, Iterable[Iterable[sp.Expr]]]] = None,
    func_name: str = "end_residual",
    cse: bool = True,
) -> str:
    """Generate the end-state residual factory.

    Parameters
    ----------
    equations
        Differential equation tuples defining ``dx/dt``.
    index_map
        Symbol indexing helpers produced by the parser.
    M
        Mass matrix supplied as a SymPy matrix or nested iterable. Uses the
        identity matrix when omitted.
    func_name
        Name assigned to the emitted factory.
    cse
        Apply common subexpression elimination before emission.

    Returns
    -------
    str
        Source code for the residual factory.
    """
    return generate_residual_code(
        equations=equations,
        index_map=index_map,
        M=M,
        is_stage=False,
        func_name=func_name,
        cse=cse,
    )


def generate_stage_residual_code(
    equations: Iterable[Tuple[sp.Symbol, sp.Expr]],
    index_map: IndexedBases,
    M: Optional[Union[sp.Matrix, Iterable[Iterable[sp.Expr]]]] = None,
    func_name: str = "stage_residual",
    cse: bool = True,
) -> str:
    """Generate the stage residual factory.

    Parameters
    ----------
    equations
        Differential equation tuples defining ``dx/dt``.
    index_map
        Symbol indexing helpers produced by the parser.
    M
        Mass matrix supplied as a SymPy matrix or nested iterable. Uses the
        identity matrix when omitted.
    func_name
        Name assigned to the emitted factory.
    cse
        Apply common subexpression elimination before emission.

    Returns
    -------
    str
        Source code for the residual factory.
    """
    return generate_residual_code(
            equations=equations,
            index_map=index_map,
            M=M,
            is_stage=True,
            func_name=func_name,
            cse=cse,
    )
