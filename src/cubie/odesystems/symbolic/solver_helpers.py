"""Code generation helpers for implicit solver linear operators and residuals.

The mass matrix ``M`` is provided at code-generation time either as a NumPy
array or a SymPy matrix. Its entries are embedded directly into the generated
device routine to avoid extra passes or buffers.
"""

from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import sympy as sp

from cubie.odesystems.symbolic.numba_cuda_printer import print_cuda_multiple
from cubie.odesystems.symbolic.jacobian import generate_analytical_jvp
from cubie.odesystems.symbolic.parser import IndexedBases, ParsedEquations
from cubie.odesystems.symbolic.sym_utils import (
    cse_and_stack,
    render_constant_assignments,
    topological_sort,
)

CACHED_OPERATOR_APPLY_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED CACHED LINEAR OPERATOR FACTORY\n"
    "def {func_name}(constants, precision, beta=1.0, gamma=1.0, order=None):\n"
    '    """Auto-generated cached linear operator.\n'
    "    Computes out = beta * (M @ v) - gamma * a_ij * h * (J @ v)\n"
    "    using cached auxiliary intermediates.\n"
    "    Returns device function:\n"
    "      operator_apply(\n"
    "          state, parameters, drivers, cached_aux, t, h, a_ij, v, out\n"
    "      )\n"
    "    argument 'order' is ignored, included for compatibility with\n"
    "    preconditioner API.\n"
    '    """\n'
    "{const_lines}"
    "    @cuda.jit((precision[:],\n"
    "               precision[:],\n"
    "               precision[:],\n"
    "               precision[:],\n"
    "               precision,\n"
    "               precision,\n"
    "               precision,\n"
    "               precision[:],\n"
    "               precision[:]),\n"
    "              device=True,\n"
    "              inline=True)\n"
    "    def operator_apply(\n"
    "        state, parameters, drivers, cached_aux, t, h, a_ij, v, out\n"
    "    ):\n"
    "{body}\n"
    "    return operator_apply\n"
)


OPERATOR_APPLY_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED LINEAR OPERATOR FACTORY\n"
    "def {func_name}(constants, precision, beta=1.0, gamma=1.0, order=None):\n"
    '    """Auto-generated linear operator.\n'
    "    Computes out = beta * (M @ v) - gamma * a_ij * h * (J @ v)\n"
    "    Returns device function:\n"
    "      operator_apply(state, parameters, drivers, t, h, a_ij, v, out)\n"
    "    argument 'order' is ignored, included for compatibility with\n"
    "    preconditioner API.\n"
    '    """\n'
    "{const_lines}"
    "    @cuda.jit((precision[:],\n"
    "               precision[:],\n"
    "               precision[:],\n"
    "               precision,\n"
    "               precision,\n"
    "               precision,\n"
    "               precision[:],\n"
    "               precision[:]),\n"
    "              device=True,\n"
    "              inline=True)\n"
    "    def operator_apply(state, parameters, drivers, t, h, a_ij, v, out):\n"
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
    "               precision,\n"
    "               precision[:]),\n"
    "              device=True,\n"
    "              inline=True)\n"
    "    def prepare_jac(state, parameters, drivers, t, cached_aux):\n"
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
    "               precision,\n"
    "               precision[:],\n"
    "               precision[:]),\n"
    "              device=True,\n"
    "              inline=True)\n"
    "    def calculate_cached_jvp(\n"
    "        state, parameters, drivers, cached_aux, t, v, out\n"
    "    ):\n"
    "{body}\n"
    "    return calculate_cached_jvp\n"
)


def _split_jvp_expressions(
    exprs: Iterable[Tuple[sp.Symbol, sp.Expr]]
) -> Tuple[
    List[Tuple[sp.Symbol, sp.Expr]],
    List[Tuple[sp.Symbol, sp.Expr]],
    Dict[int, sp.Expr],
]:
    """Split expressions into cached auxiliaries, runtime terms, and outputs.

    Parameters
    ----------
    exprs
        Expression pairs ordered for evaluation.

    Returns
    -------
    tuple of list, list, and dict
        Cached auxiliary assignments, runtime-only assignments, and the indexed
        JVP expressions.
    """
    ordered_exprs = list(exprs)
    assigned_symbols = {lhs for lhs, _ in ordered_exprs}

    dependencies: Dict[sp.Symbol, Set[sp.Symbol]] = {}
    downstream: Dict[sp.Symbol, Set[sp.Symbol]] = {}
    for lhs, rhs in ordered_exprs:
        deps = {
            sym
            for sym in rhs.free_symbols
            if sym in assigned_symbols and not str(sym).startswith("jvp[")
        }
        dependencies[lhs] = deps
        downstream[lhs] = set()

    for lhs, _ in ordered_exprs:
        lhs_str = str(lhs)
        if lhs_str.startswith("j_") or lhs_str.startswith("jvp["):
            downstream[lhs].add(lhs)

    for lhs, _ in reversed(ordered_exprs):
        targets = downstream[lhs]
        for dep in dependencies[lhs]:
            downstream[dep].update(targets)

    cached_aux: List[Tuple[sp.Symbol, sp.Expr]] = []
    runtime_aux: List[Tuple[sp.Symbol, sp.Expr]] = []
    jvp_terms: Dict[int, sp.Expr] = {}

    for lhs, rhs in ordered_exprs:
        lhs_str = str(lhs)
        if lhs_str.startswith("jvp["):
            idx = int(lhs_str.split("[")[1].split("]")[0])
            jvp_terms[idx] = rhs
            continue
        if lhs_str.startswith("j_"):
            runtime_aux.append((lhs, rhs))
            continue

        dependent_jacobians = {
            sym for sym in downstream[lhs] if str(sym).startswith("j_")
        }
        if not dependencies[lhs] and len(dependent_jacobians) > 1:
            cached_aux.append((lhs, rhs))
        else:
            runtime_aux.append((lhs, rhs))

    return cached_aux, runtime_aux, jvp_terms

def _build_operator_body(
    cached_assigns: List[Tuple[sp.Symbol, sp.Expr]],
    runtime_assigns: List[Tuple[sp.Symbol, sp.Expr]],
    jvp_terms: Dict[int, sp.Expr],
    index_map: IndexedBases,
    M: sp.Matrix,
    use_cached_aux: bool = False,
) -> str:
    """Build the CUDA body computing ``β·M·v − γ·h·J·v``.

    Parameters
    ----------
    cached_assigns
        Auxiliary assignments whose values are cached between kernel calls.
    runtime_assigns
        Assignments evaluated on demand without caching.
    jvp_terms
        Mapping from output indices to the Jacobian-vector expressions.
    index_map
        Symbol indexing helpers produced by the parser.
    M
        Mass matrix to embed into the generated operator.
    use_cached_aux
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
    a_ij_sym =  sp.Symbol("a_ij")
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
        rhs = beta_sym * mv - gamma_sym * a_ij_sym * h_sym * jvp_terms[i]
        out_updates.append((sp.Symbol(f"out[{i}]"), rhs))

    if use_cached_aux:
        if cached_assigns:
            cached = sp.IndexedBase(
                "cached_aux", shape=(sp.Integer(len(cached_assigns)),)
            )
        else:
            cached = sp.IndexedBase("cached_aux")
        aux_assignments = [
            (lhs, cached[idx]) for idx, (lhs, _) in enumerate(cached_assigns)
        ] + runtime_assigns
    else:
        aux_assignments = cached_assigns + runtime_assigns

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

    cached_aux, runtime_aux, jvp_terms = _split_jvp_expressions(jvp_exprs)
    if cached_aux:
        cached = sp.IndexedBase(
            "cached_aux", shape=(sp.Integer(len(cached_aux)),)
        )
    else:
        cached = sp.IndexedBase("cached_aux")

    aux_assignments = [
          (lhs, cached[idx]) for idx, (lhs, _) in enumerate(cached_aux)
    ] + runtime_aux

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
    cached_assigns: List[Tuple[sp.Symbol, sp.Expr]],
    runtime_assigns: List[Tuple[sp.Symbol, sp.Expr]],
    jvp_terms: Dict[int, sp.Expr],
    index_map: IndexedBases,
) -> str:
    """Build the CUDA body computing ``J·v`` with optional cached auxiliaries."""

    n_out = len(index_map.dxdt.ref_map)

    if cached_assigns:
        cached = sp.IndexedBase(
            "cached_aux", shape=(sp.Integer(len(cached_assigns)),)
        )
    else:
        cached = sp.IndexedBase("cached_aux")

    aux_assignments = [
        (lhs, cached[idx]) for idx, (lhs, _) in enumerate(cached_assigns)
    ] + runtime_assigns

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
    cached_assigns: List[Tuple[sp.Symbol, sp.Expr]], index_map: IndexedBases
) -> str:
    """Build the CUDA body populating the cached Jacobian auxiliaries."""

    if cached_assigns:
        cached = sp.IndexedBase(
            "cached_aux", shape=(sp.Integer(len(cached_assigns)),)
        )
    else:
        cached = sp.IndexedBase("cached_aux")
    exprs: List[Tuple[sp.Symbol, sp.Expr]] = []
    for idx, (lhs, rhs) in enumerate(cached_assigns):
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
    cached_aux, runtime_aux, jvp_terms = _split_jvp_expressions(jvp_exprs)
    body = _build_operator_body(
        cached_assigns=cached_aux,
        runtime_assigns=runtime_aux,
        jvp_terms=jvp_terms,
        index_map=index_map,
        M=M,
        use_cached_aux=False,
    )
    const_block = render_constant_assignments(index_map.constants.symbol_map)
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

    cached_aux, runtime_aux, jvp_terms = _split_jvp_expressions(jvp_exprs)
    body = _build_operator_body(cached_assigns=cached_aux,
                                runtime_assigns=runtime_aux,
                                jvp_terms=jvp_terms,
                                index_map=index_map,
                                M=M,
                                use_cached_aux=True)
    const_block = render_constant_assignments(index_map.constants.symbol_map)
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

    cached_aux, _, _ = _split_jvp_expressions(jvp_exprs)
    body = _build_prepare_body(cached_aux, index_map)
    const_block = render_constant_assignments(index_map.constants.symbol_map)
    code = PREPARE_JAC_TEMPLATE.format(
        func_name=func_name, body=body, const_lines=const_block
    )
    return code, len(cached_aux)


def generate_cached_jvp_code_from_jvp(
    jvp_exprs: Iterable[Tuple[sp.Symbol, sp.Expr]],
    index_map: IndexedBases,
    func_name: str = "calculate_cached_jvp",
) -> str:
    """Emit the cached JVP factory from precomputed JVP expressions."""

    cached_aux, runtime_aux, jvp_terms = _split_jvp_expressions(jvp_exprs)
    body = _build_cached_jvp_body(
        cached_assigns=cached_aux,
        runtime_assigns=runtime_aux,
        jvp_terms=jvp_terms,
        index_map=index_map,
    )
    const_block = render_constant_assignments(index_map.constants.symbol_map)
    code = CACHED_JVP_TEMPLATE.format(
        func_name=func_name, body=body, const_lines=const_block
    )
    return code


def generate_operator_apply_code(
    equations: ParsedEquations,
    index_map: IndexedBases,
    M: Optional[Union[sp.Matrix, Iterable[Iterable[sp.Expr]]]] = None,
    func_name: str = "operator_apply_factory",
    cse: bool = True,
    jvp_exprs: Optional[List[Tuple[sp.Symbol, sp.Expr]]] = None,
) -> str:
    """Generate the linear operator factory from system equations.

    Parameters
    ----------
    equations
        Parsed equations defining the system dynamics.
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
    if jvp_exprs is None:
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
    equations: ParsedEquations,
    index_map: IndexedBases,
    M: Optional[Union[sp.Matrix, Iterable[Iterable[sp.Expr]]]] = None,
    func_name: str = "linear_operator_cached",
    cse: bool = True,
    jvp_exprs: Optional[List[Tuple[sp.Symbol, sp.Expr]]] = None,
) -> str:
    """Generate the cached linear operator factory."""

    if M is None:
        n = len(index_map.states.index_map)
        M_mat = sp.eye(n)
    else:
        M_mat = sp.Matrix(M)
    if jvp_exprs is None:
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
    equations: ParsedEquations,
    index_map: IndexedBases,
    func_name: str = "prepare_jac",
    cse: bool = True,
    jvp_exprs: Optional[List[Tuple[sp.Symbol, sp.Expr]]] = None,
) -> Tuple[str, int]:
    """Generate the cached auxiliary preparation factory."""

    if jvp_exprs is None:
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
    equations: ParsedEquations,
    index_map: IndexedBases,
    func_name: str = "calculate_cached_jvp",
    cse: bool = True,
    jvp_exprs: Optional[List[Tuple[sp.Symbol, sp.Expr]]] = None,
) -> str:
    """Generate the cached Jacobian-vector product factory."""

    if jvp_exprs is None:
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
    "    Approximates (beta*I - gamma*a_ij*h*J)^[-1] via a truncated\n"
    "    Neumann series. Returns device function:\n"
    "      preconditioner(state, parameters, drivers, t, h, a_ij, v, out, jvp)\n"
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
    "               precision,\n"
    "               precision,\n"
    "               precision[:],\n"
    "               precision[:],\n"
    "               precision[:]),\n"
    "              device=True,\n"
    "              inline=True)\n"
    "    def preconditioner(\n"
    "        state, parameters, drivers, t, h, a_ij, v, out, jvp\n"
    "    ):\n"
    "        # Horner form: S[m] = v + T S[m-1], T = ((gamma*a_ij)/beta) * h * J\n"
    "        # Accumulator lives in `out`. Uses caller-provided `jvp` for JVP.\n"
    "        for i in range(n):\n"
    "            out[i] = v[i]\n"
    "        h_eff = h * h_eff_factor * a_ij\n"
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
    "    Approximates (beta*I - gamma*a_ij*h*J)^[-1] via a truncated\n"
    "    Neumann series with cached auxiliaries. Returns device function:\n"
    "      preconditioner(\n"
    "          state, parameters, drivers, cached_aux, t, h, a_ij, v, out, jvp\n"
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
    "               precision,\n"
    "               precision,\n"
    "               precision[:],\n"
    "               precision[:],\n"
    "               precision[:]),\n"
    "              device=True,\n"
    "              inline=True)\n"
    "    def preconditioner(\n"
    "        state, parameters, drivers, cached_aux, t, h, a_ij, v, out, jvp\n"
    "    ):\n"
    "        for i in range(n):\n"
    "            out[i] = v[i]\n"
    "        h_eff = h * h_eff_factor * a_ij\n"
    "        for _ in range(order):\n"
    "{jv_body}\n"
    "            for i in range(n):\n"
    "                out[i] = v[i] + h_eff * jvp[i]\n"
    "        for i in range(n):\n"
    "            out[i] = beta_inv * out[i]\n"
    "    return preconditioner\n"
)


def generate_neumann_preconditioner_code(
    equations: ParsedEquations,
    index_map: IndexedBases,
    func_name: str = "neumann_preconditioner_factory",
    cse: bool = True,
    jvp_exprs: Optional[List[Tuple[sp.Symbol, sp.Expr]]] = None,
) -> str:
    """Generate the Neumann preconditioner factory.

    Parameters
    ----------
    equations
        Parsed equations defining the system.
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
    const_block = render_constant_assignments(index_map.constants.symbol_map)
    if jvp_exprs is None:
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
    equations: ParsedEquations,
    index_map: IndexedBases,
    func_name: str = "neumann_preconditioner_cached",
    cse: bool = True,
    jvp_exprs: Optional[List[Tuple[sp.Symbol, sp.Expr]]] = None,
) -> str:
    """Generate the cached Neumann preconditioner factory."""

    n_out = len(index_map.dxdt.ref_map)
    const_block = render_constant_assignments(index_map.constants.symbol_map)
    if jvp_exprs is None:
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
    "    Computes the stage-increment residual\n"
    "    beta * M @ u - gamma * h * f(base_state + a_ij * u)\n"
    "    where ``u`` is the increment solved for by Newton's method.\n"
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
    "               precision,\n"
    "               precision[:],\n"
    "               precision[:]),\n"
    "              device=True,\n"
    "              inline=True)\n"
    "    def residual(u, parameters, drivers, t, h, a_ij, base_state, out):\n"
    "{res_lines}\n"
    "    return residual\n"
)


def _build_residual_lines(
    equations: ParsedEquations,
    index_map: IndexedBases,
    M: sp.Matrix,
    cse: bool = True,
) -> str:
    """Construct CUDA code lines for the stage-increment residual.

    Parameters
    ----------
    equations
        Parsed equations describing ``dx/dt`` assignments.
    index_map
        Symbol indexing helpers produced by the parser.
    M
        Mass matrix to embed into the generated residual.
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
    eq_list = equations.to_equation_list()

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
        eval_point = base[i] + aij_sym * u[i]
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
            mv += entry * u[j]
        
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
    equations: ParsedEquations,
    index_map: IndexedBases,
    M: Optional[Union[sp.Matrix, Iterable[Iterable[sp.Expr]]]] = None,
    func_name: str = "residual_factory",
    cse: bool = True,
) -> str:
    """Emit the stage-increment residual factory for Newton--Krylov integration.

    Parameters
    ----------
    equations
        Parsed equations defining the system dynamics.
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
    if M is None:
        n = len(index_map.states.index_map)
        M_mat = sp.eye(n)
    else:
        M_mat = sp.Matrix(M)

    res_lines = _build_residual_lines(
        equations=equations,
        index_map=index_map,
        M=M_mat,
        cse=cse,
    )
    const_block = render_constant_assignments(index_map.constants.symbol_map)

    return RESIDUAL_TEMPLATE.format(
            func_name=func_name,
            const_lines=const_block,
            res_lines=res_lines,
    )

def generate_stage_residual_code(
    equations: ParsedEquations,
    index_map: IndexedBases,
    M: Optional[Union[sp.Matrix, Iterable[Iterable[sp.Expr]]]] = None,
    func_name: str = "stage_residual",
    cse: bool = True,
) -> str:
    """Generate the stage residual factory.

    Parameters
    ----------
    equations
        Parsed equations defining ``dx/dt``.
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
        func_name=func_name,
        cse=cse,
    )
