"""Code generation helpers for implicit solver linear operators and residuals.

The mass matrix ``M`` is provided at code-generation time either as a NumPy
array or a SymPy matrix. Its entries are embedded directly into the generated
device routine to avoid extra passes or buffers.
"""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import sympy as sp

from cubie.odesystems.symbolic.numba_cuda_printer import print_cuda_multiple
from cubie.odesystems.symbolic.codegen.jacobian import generate_analytical_jvp
from cubie.odesystems.symbolic.jvp_equations import JVPEquations
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



def _partition_cached_assignments(
    equations: JVPEquations,
) -> Tuple[
    List[Tuple[sp.Symbol, sp.Expr]],
    List[Tuple[sp.Symbol, sp.Expr]],
    List[Tuple[sp.Symbol, sp.Expr]],
]:
    """Partition assignments into cached, runtime, and preparation subsets.

    Parameters
    ----------
    equations
        Structured representation of the Jacobian-vector product assignments.

    Returns
    -------
    tuple of list, list, list
        Cached auxiliary assignments, runtime assignments, and preparation
        assignments required to populate cached intermediates.
    """

    return equations.cached_partition()

def _build_operator_body(
    cached_assigns: List[Tuple[sp.Symbol, sp.Expr]],
    runtime_assigns: List[Tuple[sp.Symbol, sp.Expr]],
    jvp_terms: Dict[int, sp.Expr],
    index_map: IndexedBases,
    M: sp.Matrix,
    use_cached_aux: bool = False,
    prepare_assigns: Optional[List[Tuple[sp.Symbol, sp.Expr]]] = None,
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
    prepare_assigns
        Optional assignments required to populate cached auxiliaries. These are
        included when building the uncached operator so dependencies remain
        defined.

    Returns
    -------
    str
        Indented CUDA code statements implementing the operator body.

    Notes
    -----
    Constructs SymPy assignments for mass-matrix multiplications and auxiliary
    loads, renders them through the CUDA printer, and indents the result to fit
    within the generated device function.
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
        combined = list(prepare_assigns or []) + cached_assigns + runtime_assigns
        seen = set()
        aux_assignments = []
        for lhs, rhs in combined:
            if lhs in seen:
                continue
            seen.add(lhs)
            aux_assignments.append((lhs, rhs))

    exprs = mass_assigns + aux_assignments + out_updates
    lines = print_cuda_multiple(exprs, symbol_map=index_map.all_arrayrefs)
    if not lines:
        return "        pass"
    return "\n".join("        " + ln for ln in lines)


def _build_cached_neumann_body(
    equations: JVPEquations,
    index_map: IndexedBases,
) -> str:
    """Build the cached Neumann-series Jacobian-vector body.

    Parameters
    ----------
    equations
        Structured representation of the Jacobian-vector product assignments.
    index_map
        Symbol indexing helpers produced by the parser.

    Returns
    -------
    str
        Indented CUDA code statements implementing the cached JVP body.

    Notes
    -----
    Partitions auxiliary assignments using :class:`JVPEquations`, maps cached
    values to buffer loads, and reuses the CUDA printer to generate the
    Neumann-series update statements.
    """

    cached_aux, runtime_aux, _ = _partition_cached_assignments(equations)
    jvp_terms = equations.jvp_terms
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
    exprs = list(aux_assignments)
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
    """Build the CUDA body computing ``J·v`` with optional cached auxiliaries.

    Parameters
    ----------
    cached_assigns
        Auxiliary assignments stored in the cache.
    runtime_assigns
        Auxiliary assignments evaluated on demand.
    jvp_terms
        Mapping from output indices to Jacobian-vector expressions.
    index_map
        Symbol indexing helpers produced by the parser.

    Returns
    -------
    str
        Indented CUDA code statements implementing the cached JVP body.

    Notes
    -----
    Materializes cached intermediates from buffer slots, appends runtime
    assignments, and emits CUDA-formatted statements for each output update.
    """

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
    cached_assigns: List[Tuple[sp.Symbol, sp.Expr]],
    prepare_assigns: List[Tuple[sp.Symbol, sp.Expr]],
    index_map: IndexedBases,
) -> str:
    """Build the CUDA body populating the cached Jacobian auxiliaries.

    Parameters
    ----------
    cached_assigns
        Auxiliary assignments stored in the cache.
    prepare_assigns
        Assignments executed during cache population.
    index_map
        Symbol indexing helpers produced by the parser.

    Returns
    -------
    str
        Indented CUDA code statements storing computed auxiliaries into the
        cache buffer.

    Notes
    -----
    Walks the preparation order, renders assignments via the CUDA printer, and
    writes cached values into their corresponding buffer indices.
    """

    if cached_assigns:
        cached = sp.IndexedBase(
            "cached_aux", shape=(sp.Integer(len(cached_assigns)),)
        )
    else:
        cached = sp.IndexedBase("cached_aux")
    exprs = []
    cached_slots = {lhs: idx for idx, (lhs, _) in enumerate(cached_assigns)}
    for lhs, rhs in prepare_assigns:
        exprs.append((lhs, rhs))
        idx = cached_slots.get(lhs)
        if idx is not None:
            exprs.append((cached[idx], lhs))

    lines = print_cuda_multiple(exprs, symbol_map=index_map.all_arrayrefs)
    if not lines:
        return "        pass"
    return "\n".join("        " + ln for ln in lines)


def generate_operator_apply_code_from_jvp(
    equations: JVPEquations,
    index_map: IndexedBases,
    M: sp.Matrix,
    func_name: str = "operator_apply_factory",
    cse: bool = True,
) -> str:
    """Emit the operator apply factory from precomputed JVP expressions.

    Parameters
    ----------
    equations
        Structured Jacobian-vector product assignments.
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
    cached_aux, runtime_aux, prepare_assigns = _partition_cached_assignments(
        equations
    )
    body = _build_operator_body(
        cached_assigns=cached_aux,
        runtime_assigns=runtime_aux,
        jvp_terms=equations.jvp_terms,
        index_map=index_map,
        M=M,
        use_cached_aux=False,
        prepare_assigns=prepare_assigns,
    )
    const_block = render_constant_assignments(index_map.constants.symbol_map)
    return OPERATOR_APPLY_TEMPLATE.format(
        func_name=func_name, body=body, const_lines=const_block
    )


def generate_cached_operator_apply_code_from_jvp(
    equations: JVPEquations,
    index_map: IndexedBases,
    M: sp.Matrix,
    func_name: str = "linear_operator_cached",
) -> str:
    """Emit the cached linear operator factory from JVP expressions."""

    cached_aux, runtime_aux, _ = _partition_cached_assignments(equations)
    body = _build_operator_body(
        cached_assigns=cached_aux,
        runtime_assigns=runtime_aux,
        jvp_terms=equations.jvp_terms,
        index_map=index_map,
        M=M,
        use_cached_aux=True,
    )
    const_block = render_constant_assignments(index_map.constants.symbol_map)
    return CACHED_OPERATOR_APPLY_TEMPLATE.format(
        func_name=func_name,
        body=body,
        const_lines=const_block,
    )


def generate_prepare_jac_code_from_jvp(
    equations: JVPEquations,
    index_map: IndexedBases,
    func_name: str = "prepare_jac",
) -> Tuple[str, int]:
    """Emit the auxiliary preparation factory from JVP expressions."""

    cached_aux, _, prepare_assigns = _partition_cached_assignments(equations)
    body = _build_prepare_body(cached_aux, prepare_assigns, index_map)
    const_block = render_constant_assignments(index_map.constants.symbol_map)
    code = PREPARE_JAC_TEMPLATE.format(
        func_name=func_name, body=body, const_lines=const_block
    )
    return code, len(cached_aux)


def generate_cached_jvp_code_from_jvp(
    equations: JVPEquations,
    index_map: IndexedBases,
    func_name: str = "calculate_cached_jvp",
) -> str:
    """Emit the cached JVP factory from precomputed JVP expressions."""

    cached_aux, runtime_aux, _ = _partition_cached_assignments(equations)
    body = _build_cached_jvp_body(
        cached_assigns=cached_aux,
        runtime_assigns=runtime_aux,
        jvp_terms=equations.jvp_terms,
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
    jvp_equations: Optional[JVPEquations] = None,
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
    jvp_equations
        Optional precomputed :class:`JVPEquations` reused across helper
        generation.
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
    if jvp_equations is None:
        jvp_equations = generate_analytical_jvp(
            equations,
            input_order=index_map.states.index_map,
            output_order=index_map.dxdt.index_map,
            observables=index_map.observable_symbols,
            cse=cse,
        )
    return generate_operator_apply_code_from_jvp(
        equations=jvp_equations,
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
    jvp_equations: Optional[JVPEquations] = None,
) -> str:
    """Generate the cached linear operator factory."""

    if M is None:
        n = len(index_map.states.index_map)
        M_mat = sp.eye(n)
    else:
        M_mat = sp.Matrix(M)
    if jvp_equations is None:
        jvp_equations = generate_analytical_jvp(
            equations,
            input_order=index_map.states.index_map,
            output_order=index_map.dxdt.index_map,
            observables=index_map.observable_symbols,
            cse=cse,
        )
    return generate_cached_operator_apply_code_from_jvp(
        equations=jvp_equations,
        index_map=index_map,
        M=M_mat,
        func_name=func_name,
    )


def generate_prepare_jac_code(
    equations: ParsedEquations,
    index_map: IndexedBases,
    func_name: str = "prepare_jac",
    cse: bool = True,
    jvp_equations: Optional[JVPEquations] = None,
) -> Tuple[str, int]:
    """Generate the cached auxiliary preparation factory."""

    if jvp_equations is None:
        jvp_equations = generate_analytical_jvp(
            equations,
            input_order=index_map.states.index_map,
            output_order=index_map.dxdt.index_map,
            observables=index_map.observable_symbols,
            cse=cse,
        )
    return generate_prepare_jac_code_from_jvp(
        equations=jvp_equations,
        index_map=index_map,
        func_name=func_name,
    )


def generate_cached_jvp_code(
    equations: ParsedEquations,
    index_map: IndexedBases,
    func_name: str = "calculate_cached_jvp",
    cse: bool = True,
    jvp_equations: Optional[JVPEquations] = None,
) -> str:
    """Generate the cached Jacobian-vector product factory."""

    if jvp_equations is None:
        jvp_equations = generate_analytical_jvp(
            equations,
            input_order=index_map.states.index_map,
            output_order=index_map.dxdt.index_map,
            observables=index_map.observable_symbols,
            cse=cse,
        )
    return generate_cached_jvp_code_from_jvp(
        equations=jvp_equations,
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
    jvp_equations: Optional[JVPEquations] = None,
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

    jvp_equations
        Optional precomputed :class:`JVPEquations` reused across helper
        generation.

    Returns
    -------
    str
        Source code for the Neumann preconditioner factory.
    """
    n_out = len(index_map.dxdt.ref_map)
    const_block = render_constant_assignments(index_map.constants.symbol_map)
    if jvp_equations is None:
        jvp_equations = generate_analytical_jvp(
            equations,
            input_order=index_map.states.index_map,
            output_order=index_map.dxdt.index_map,
            observables=index_map.observable_symbols,
            cse=cse,
        )
    assignments = jvp_equations.ordered_assignments
    # Emit using canonical names, then rewrite to drive JVP with `out` and
    # write into the caller-provided scratch buffer `jvp`.
    lines = print_cuda_multiple(assignments, symbol_map=index_map.all_arrayrefs)
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
    jvp_equations: Optional[JVPEquations] = None,
) -> str:
    """Generate the cached Neumann preconditioner factory."""

    n_out = len(index_map.dxdt.ref_map)
    const_block = render_constant_assignments(index_map.constants.symbol_map)
    if jvp_equations is None:
        jvp_equations = generate_analytical_jvp(
            equations,
            input_order=index_map.states.index_map,
            output_order=index_map.dxdt.index_map,
            observables=index_map.observable_symbols,
            cse=cse,
        )
    jv_body = _build_cached_neumann_body(jvp_equations, index_map)
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
    "def {func_name}(constants, precision,  beta=1.0, gamma=1.0, "
    "order=None):\n"
    '    """Auto-generated residual function for Newton-Krylov ODE '
    'integration.\n'
    "    \n"
    "    Computes the stage-increment residual\n"
    "    beta * M @ u - gamma * h * f(base_state + a_ij * u)\n"
    "    where ``u`` is the increment solved for by Newton's method.\n"
    "    \n"
    "    Uses dx_ numbered symbols for derivatives and aux_ symbols for "
    "observables,\n"
    "    following the same pattern as JVP generation.\n"
    "    \n"
    "    Order is ignored, included for compatibility with preconditioner "
    "API.\n"
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


N_STAGE_RESIDUAL_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED N-STAGE RESIDUAL FACTORY\n"
    "def {func_name}(constants, precision, beta=1.0, gamma=1.0, order=None):\n"
    '    """Auto-generated FIRK residual for flattened stage increments.\n'
    "    Handles {stage_count} stages with ``s * n`` unknowns.\n"
    "    Order is ignored, included for compatibility with preconditioner API.\n"
    '    """\n'
    "{const_lines}"
    "{metadata_lines}"
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
    "{body}\n"
    "    return residual\n"
)


N_STAGE_OPERATOR_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED N-STAGE LINEAR OPERATOR FACTORY\n"
    "def {func_name}(constants, precision, beta=1.0, gamma=1.0, order=None):\n"
    '    """Auto-generated FIRK linear operator for flattened stages.\n'
    "    Handles {stage_count} stages with ``s * n`` unknowns.\n"
    "    Order is ignored, included for compatibility with preconditioner API.\n"
    '    """\n'
    "{const_lines}"
    "{metadata_lines}"
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


def _prepare_stage_data(
    stage_coefficients: Sequence[Sequence[Union[float, sp.Expr]]],
    stage_nodes: Sequence[Union[float, sp.Expr]],
) -> Tuple[sp.Matrix, Tuple[sp.Expr, ...], int]:
    """Normalise FIRK tableau metadata for code generation."""

    coeff_matrix = sp.Matrix(stage_coefficients).applyfunc(sp.S)
    node_exprs = tuple(sp.S(node) for node in stage_nodes)
    return coeff_matrix, node_exprs, coeff_matrix.rows


def _build_stage_metadata(
    stage_coefficients: sp.Matrix,
    stage_nodes: Tuple[sp.Expr, ...],
) -> Tuple[
    List[Tuple[sp.Symbol, sp.Expr]],
    List[List[sp.Symbol]],
    List[sp.Symbol],
]:
    """Create symbol assignments for FIRK coefficients and nodes."""

    stage_count = stage_coefficients.rows
    coeff_symbols: List[List[sp.Symbol]] = []
    node_symbols: List[sp.Symbol] = []
    metadata_exprs: List[Tuple[sp.Symbol, sp.Expr]] = []
    for stage_idx in range(stage_count):
        node_symbol = sp.Symbol(f"c_{stage_idx}")
        node_symbols.append(node_symbol)
        metadata_exprs.append((node_symbol, stage_nodes[stage_idx]))
        stage_row: List[sp.Symbol] = []
        for col_idx in range(stage_count):
            coeff_symbol = sp.Symbol(f"a_{stage_idx}_{col_idx}")
            stage_row.append(coeff_symbol)
            metadata_exprs.append(
                (coeff_symbol, stage_coefficients[stage_idx, col_idx])
            )
        coeff_symbols.append(stage_row)
    return metadata_exprs, coeff_symbols, node_symbols


def _build_n_stage_residual_lines(
    equations: ParsedEquations,
    index_map: IndexedBases,
    M: sp.Matrix,
    stage_coefficients: sp.Matrix,
    stage_nodes: Tuple[sp.Expr, ...],
    cse: bool = True,
) -> str:
    """Construct CUDA statements for the FIRK n-stage residual."""

    metadata_exprs, coeff_symbols, _ = _build_stage_metadata(
        stage_coefficients, stage_nodes
    )
    eq_list = equations.to_equation_list()
    state_symbols = list(index_map.states.index_map.keys())
    dx_symbols = list(index_map.dxdt.index_map.keys())
    observable_symbols = list(index_map.observable_symbols)
    state_count = len(state_symbols)
    stage_count = stage_coefficients.rows

    beta_sym = sp.Symbol("beta")
    gamma_sym = sp.Symbol("gamma")
    h_sym = sp.Symbol("h")
    total_states = sp.Integer(stage_count * state_count)
    u = sp.IndexedBase("u", shape=(total_states,))
    base_state = sp.IndexedBase("base_state", shape=(sp.Integer(state_count),))
    out = sp.IndexedBase("out", shape=(total_states,))

    eval_exprs: List[Tuple[sp.Symbol, sp.Expr]] = list(metadata_exprs)

    for stage_idx in range(stage_count):
        stage_dx_symbols = [
            sp.Symbol(f"dx_{stage_idx}_{idx}")
            for idx in range(len(dx_symbols))
        ]
        dx_subs = dict(zip(dx_symbols, stage_dx_symbols))

        if observable_symbols:
            stage_obs_symbols = [
                sp.Symbol(f"aux_{stage_idx}_{idx + 1}")
                for idx in range(len(observable_symbols))
            ]
            obs_subs = dict(zip(observable_symbols, stage_obs_symbols))
        else:
            obs_subs = {}
        substitution_map = {**dx_subs, **obs_subs}

        stage_state_subs = {}
        for state_idx, state_sym in enumerate(state_symbols):
            expr = base_state[state_idx]
            for contrib_idx in range(stage_count):
                coeff_value = stage_coefficients[stage_idx, contrib_idx]
                if coeff_value == 0:
                    continue
                coeff_sym = coeff_symbols[stage_idx][contrib_idx]
                expr += coeff_sym * u[
                    contrib_idx * state_count + state_idx
                ]
            stage_state_subs[state_sym] = expr

        substituted = [
            (
                lhs.subs(substitution_map),
                rhs.subs(substitution_map).subs(stage_state_subs),
            )
            for lhs, rhs in eq_list
        ]
        eval_exprs.extend(substituted)

        stage_offset = stage_idx * state_count
        for comp_idx in range(state_count):
            mv = sp.S.Zero
            for col_idx in range(state_count):
                entry = M[comp_idx, col_idx]
                if entry == 0:
                    continue
                mv += entry * u[stage_offset + col_idx]
            residual_expr = (
                beta_sym * mv
                - gamma_sym * h_sym * stage_dx_symbols[comp_idx]
            )
            eval_exprs.append((out[stage_offset + comp_idx], residual_expr))

    if cse:
        eval_exprs = cse_and_stack(eval_exprs)
    else:
        eval_exprs = topological_sort(eval_exprs)

    symbol_map = dict(index_map.all_arrayrefs)
    symbol_map.update(
        {
            "u": u,
            "base_state": base_state,
            "out": out,
            "beta": beta_sym,
            "gamma": gamma_sym,
            "h": h_sym,
        }
    )

    lines = print_cuda_multiple(eval_exprs, symbol_map=symbol_map)
    if not lines:
        return "        pass"
    return "\n".join("        " + ln for ln in lines)


def _build_n_stage_operator_lines(
    equations: ParsedEquations,
    index_map: IndexedBases,
    M: sp.Matrix,
    stage_coefficients: sp.Matrix,
    stage_nodes: Tuple[sp.Expr, ...],
    jvp_equations: JVPEquations,
    cse: bool = True,
) -> str:
    """Construct CUDA statements for the FIRK n-stage linear operator."""

    metadata_exprs, coeff_symbols, _ = _build_stage_metadata(
        stage_coefficients, stage_nodes
    )
    eq_list = equations.to_equation_list()
    state_symbols = list(index_map.states.index_map.keys())
    dx_symbols = list(index_map.dxdt.index_map.keys())
    observable_symbols = list(index_map.observable_symbols)
    state_count = len(state_symbols)
    stage_count = stage_coefficients.rows

    beta_sym = sp.Symbol("beta")
    gamma_sym = sp.Symbol("gamma")
    h_sym = sp.Symbol("h")
    total_states = sp.Integer(stage_count * state_count)
    state_vec = sp.IndexedBase("state", shape=(total_states,))
    base_state = sp.IndexedBase("base_state", shape=(sp.Integer(state_count),))
    direction_vec = sp.IndexedBase("v", shape=(total_states,))
    out = sp.IndexedBase("out", shape=(total_states,))

    jvp_terms = jvp_equations.jvp_terms
    aux_order = jvp_equations.non_jvp_order
    aux_exprs = jvp_equations.non_jvp_exprs
    eval_exprs: List[Tuple[sp.Symbol, sp.Expr]] = list(metadata_exprs)

    for stage_idx in range(stage_count):
        stage_dx_symbols = [
            sp.Symbol(f"dx_{stage_idx}_{idx}")
            for idx in range(len(dx_symbols))
        ]
        dx_subs = dict(zip(dx_symbols, stage_dx_symbols))

        if observable_symbols:
            stage_obs_symbols = [
                sp.Symbol(f"aux_{stage_idx}_{idx + 1}")
                for idx in range(len(observable_symbols))
            ]
            obs_subs = dict(zip(observable_symbols, stage_obs_symbols))
        else:
            obs_subs = {}
        substitution_map = {**dx_subs, **obs_subs}

        stage_state_subs = {}
        for state_idx, state_sym in enumerate(state_symbols):
            expr = base_state[state_idx]
            for contrib_idx in range(stage_count):
                coeff_value = stage_coefficients[stage_idx, contrib_idx]
                if coeff_value == 0:
                    continue
                coeff_sym = coeff_symbols[stage_idx][contrib_idx]
                expr += coeff_sym * state_vec[
                    contrib_idx * state_count + state_idx
                ]
            stage_state_subs[state_sym] = expr

        substituted = [
            (
                lhs.subs(substitution_map),
                rhs.subs(substitution_map).subs(stage_state_subs),
            )
            for lhs, rhs in eq_list
        ]
        eval_exprs.extend(substituted)

        direction_combos = []
        for comp_idx in range(state_count):
            combo = sp.S.Zero
            for contrib_idx in range(stage_count):
                coeff_value = stage_coefficients[stage_idx, contrib_idx]
                if coeff_value == 0:
                    continue
                coeff_sym = coeff_symbols[stage_idx][contrib_idx]
                combo += coeff_sym * direction_vec[
                    contrib_idx * state_count + comp_idx
                ]
            direction_combos.append(combo)
        v_indexed = sp.IndexedBase("v")
        v_subs = {
            v_indexed[idx]: direction_combos[idx] for idx in range(state_count)
        }

        stage_aux_assignments: List[Tuple[sp.Symbol, sp.Expr]] = []
        aux_subs: Dict[sp.Symbol, sp.Symbol] = {}
        for lhs in aux_order:
            stage_symbol = sp.Symbol(f"{str(lhs)}_{stage_idx}")
            rhs = aux_exprs[lhs]
            substituted_rhs = rhs.subs(substitution_map)
            substituted_rhs = substituted_rhs.subs(stage_state_subs)
            if aux_subs:
                substituted_rhs = substituted_rhs.subs(aux_subs)
            substituted_rhs = substituted_rhs.subs(v_subs)
            stage_aux_assignments.append((stage_symbol, substituted_rhs))
            aux_subs[lhs] = stage_symbol
        eval_exprs.extend(stage_aux_assignments)

        stage_jvp_symbols: Dict[int, sp.Symbol] = {}
        for idx, expr in jvp_terms.items():
            stage_symbol = sp.Symbol(f"jvp_{stage_idx}_{idx}")
            stage_jvp_symbols[idx] = stage_symbol
            substituted_expr = expr.subs(substitution_map)
            substituted_expr = substituted_expr.subs(stage_state_subs)
            if aux_subs:
                substituted_expr = substituted_expr.subs(aux_subs)
            eval_exprs.append(
                (stage_symbol, substituted_expr.subs(v_subs))
            )

        stage_offset = stage_idx * state_count
        for comp_idx in range(state_count):
            mv = sp.S.Zero
            for col_idx in range(state_count):
                entry = M[comp_idx, col_idx]
                if entry == 0:
                    continue
                mv += entry * direction_vec[stage_offset + col_idx]
            jvp_value = stage_jvp_symbols.get(comp_idx, sp.S.Zero)
            update_expr = beta_sym * mv - gamma_sym * h_sym * jvp_value
            eval_exprs.append((out[stage_offset + comp_idx], update_expr))

    if cse:
        eval_exprs = cse_and_stack(eval_exprs)
    else:
        eval_exprs = topological_sort(eval_exprs)

    symbol_map = dict(index_map.all_arrayrefs)
    symbol_map.update(
        {
            "state": state_vec,
            "base_state": base_state,
            "v": direction_vec,
            "out": out,
            "beta": beta_sym,
            "gamma": gamma_sym,
            "h": h_sym,
        }
    )

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


def generate_n_stage_residual_code(
    equations: ParsedEquations,
    index_map: IndexedBases,
    stage_coefficients: Sequence[Sequence[Union[float, sp.Expr]]],
    stage_nodes: Sequence[Union[float, sp.Expr]],
    M: Optional[Union[sp.Matrix, Iterable[Iterable[sp.Expr]]]] = None,
    func_name: str = "n_stage_residual",
    cse: bool = True,
) -> str:
    """Generate a flattened n-stage FIRK residual factory."""

    coeff_matrix, node_values, stage_count = _prepare_stage_data(
        stage_coefficients, stage_nodes
    )
    if M is None:
        state_dim = len(index_map.states.index_map)
        mass_matrix = sp.eye(state_dim)
    else:
        mass_matrix = sp.Matrix(M)
    body = _build_n_stage_residual_lines(
        equations=equations,
        index_map=index_map,
        M=mass_matrix,
        stage_coefficients=coeff_matrix,
        stage_nodes=node_values,
        cse=cse,
    )
    const_block = render_constant_assignments(index_map.constants.symbol_map)
    return N_STAGE_RESIDUAL_TEMPLATE.format(
        func_name=func_name,
        const_lines=const_block,
        metadata_lines="",
        body=body,
        stage_count=stage_count,
    )


def generate_n_stage_linear_operator_code(
    equations: ParsedEquations,
    index_map: IndexedBases,
    stage_coefficients: Sequence[Sequence[Union[float, sp.Expr]]],
    stage_nodes: Sequence[Union[float, sp.Expr]],
    M: Optional[Union[sp.Matrix, Iterable[Iterable[sp.Expr]]]] = None,
    func_name: str = "n_stage_linear_operator",
    cse: bool = True,
    jvp_equations: Optional[JVPEquations] = None,
) -> str:
    """Generate a flattened n-stage FIRK linear operator factory."""

    coeff_matrix, node_values, stage_count = _prepare_stage_data(
        stage_coefficients, stage_nodes
    )
    if M is None:
        state_dim = len(index_map.states.index_map)
        mass_matrix = sp.eye(state_dim)
    else:
        mass_matrix = sp.Matrix(M)
    if jvp_equations is None:
        jvp_equations = generate_analytical_jvp(
            equations,
            input_order=index_map.states.index_map,
            output_order=index_map.dxdt.index_map,
            observables=index_map.observable_symbols,
            cse=cse,
        )
    body = _build_n_stage_operator_lines(
        equations=equations,
        index_map=index_map,
        M=mass_matrix,
        stage_coefficients=coeff_matrix,
        stage_nodes=node_values,
        jvp_equations=jvp_equations,
        cse=cse,
    )
    const_block = render_constant_assignments(index_map.constants.symbol_map)
    return N_STAGE_OPERATOR_TEMPLATE.format(
        func_name=func_name,
        const_lines=const_block,
        metadata_lines="",
        body=body,
        stage_count=stage_count,
    )

