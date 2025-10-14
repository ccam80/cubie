"""Code generation helpers for implicit solver linear operators and residuals.

The mass matrix ``M`` is provided at code-generation time either as a NumPy
array or a SymPy matrix. Its entries are embedded directly into the generated
device routine to avoid extra passes or buffers.
"""

from collections import defaultdict
from typing import (
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

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
    "    def operator_apply(state, parameters, drivers, cached_aux, h, v, "
    "out):\n"
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



def simulate_removal(
    symbol: sp.Symbol,
    active_nodes: Set[sp.Symbol],
    current_ref_counts: Dict[sp.Symbol, int],
    dependencies: Dict[sp.Symbol, Set[sp.Symbol]],
    ops_cost: Dict[sp.Symbol, int],
) -> Tuple[int, Set[sp.Symbol]]:
    """Estimate saved operations when removing a symbol from active nodes.

    Parameters
    ----------
    symbol
        Candidate assignment symbol considered for caching.
    active_nodes
        Symbols that remain available for runtime execution.
    current_ref_counts
        Reference counts for each symbol across dependents and JVP usage.
    dependencies
        Mapping of symbols to the prerequisite assignments they rely on.
    ops_cost
        Operation counts accrued for computing each symbol.

    Returns
    -------
    tuple of int and set
        The estimated number of operations saved and the dependency closure
        removable alongside the symbol.

    Notes
    -----
    Performs a stack-based traversal that simulates removing the symbol, then
    cascades through dependents whose reference counts drop to zero while
    tallying the operations saved by omitting those assignments.
    """

    if symbol not in active_nodes:
        return 0, set()
    temp_counts = current_ref_counts.copy()
    to_remove = set()
    stack = [symbol]
    while stack:
        node = stack.pop()
        if node in to_remove:
            continue
        to_remove.add(node)
        for dep in dependencies.get(node, set()):
            if dep not in temp_counts:
                continue
            temp_counts[dep] -= 1
            if dep in active_nodes and temp_counts[dep] == 0:
                stack.append(dep)
    saved = sum(
        ops_cost.get(node, 0) for node in to_remove if node in active_nodes
    )
    return saved, to_remove


def _build_expression_costs(
    non_jvp_order: Sequence[sp.Symbol],
    non_jvp_exprs: Mapping[sp.Symbol, sp.Expr],
    assigned_symbols: Set[sp.Symbol],
    jvp_terms: Mapping[int, sp.Expr],
) -> Tuple[
    Dict[sp.Symbol, Set[sp.Symbol]],
    Dict[sp.Symbol, Set[sp.Symbol]],
    Dict[sp.Symbol, int],
    Dict[sp.Symbol, int],
]:
    """Build dependency graphs, operation costs, and JVP usage counts.

    Parameters
    ----------
    non_jvp_order
        Evaluation order for auxiliary assignments.
    non_jvp_exprs
        Mapping from auxiliary symbols to their SymPy expressions.
    assigned_symbols
        Symbols introduced by the expression sequence.
    jvp_terms
        Indexed Jacobian-vector expressions.

    Returns
    -------
    tuple of dict, dict, dict, and dict
        Dependency edges, dependent adjacency, operation counts, and JVP usage
        reference counts for each auxiliary symbol.

    Notes
    -----
    Counts operations using ``sympy.count_ops`` while tracking parent-child
    relationships and recording how many JVP expressions consume each auxiliary
    symbol.
    """

    dependencies = {}
    dependents = {sym: set() for sym in non_jvp_order}
    ops_cost = {}
    for lhs in non_jvp_order:
        rhs = non_jvp_exprs[lhs]
        ops_cost[lhs] = int(sp.count_ops(rhs, visual=False))
        deps = {
            sym
            for sym in rhs.free_symbols
            if sym in assigned_symbols and not str(sym).startswith("jvp[")
        }
        dependencies[lhs] = deps
        for dep in deps:
            if dep in dependents:
                dependents[dep].add(lhs)
    jvp_usage = defaultdict(int)
    for rhs in jvp_terms.values():
        for sym in rhs.free_symbols:
            if sym in dependents:
                jvp_usage[sym] += 1
    return dependencies, dependents, ops_cost, dict(jvp_usage)


def _select_cached_nodes(
    non_jvp_order: Sequence[sp.Symbol],
    dependencies: Dict[sp.Symbol, Set[sp.Symbol]],
    dependents: Dict[sp.Symbol, Set[sp.Symbol]],
    ops_cost: Dict[sp.Symbol, int],
    jvp_usage: Dict[sp.Symbol, int],
    max_cached_terms: int,
    min_ops_threshold: int,
) -> Tuple[List[sp.Symbol], Set[sp.Symbol]]:
    """Select auxiliary nodes to cache based on estimated savings.

    Parameters
    ----------
    non_jvp_order
        Evaluation order for candidate auxiliary assignments.
    dependencies
        Mapping from each symbol to the prerequisites required to compute it.
    dependents
        Mapping from each symbol to the downstream assignments that use it.
    ops_cost
        Operation counts required to compute each symbol.
    jvp_usage
        Usage counts contributed by Jacobian-vector expressions.
    max_cached_terms
        Maximum number of auxiliary expressions permitted in the cache.
    min_ops_threshold
        Minimum operations required for a symbol to qualify for caching.

    Returns
    -------
    tuple of list and set
        Symbols chosen for caching and the remaining symbols evaluated at
        runtime.

    Notes
    -----
    Iteratively evaluates each candidate with :func:`simulate_removal`, caching
    the symbol that yields the largest operation reduction until limits or
    savings thresholds halt the process.
    """

    active_nodes = set(non_jvp_order)
    current_ref_counts = {}
    for sym in non_jvp_order:
        current_ref_counts[sym] = len(dependents[sym]) + jvp_usage.get(sym, 0)
    cached_nodes = []
    candidate_order = list(non_jvp_order)
    while len(cached_nodes) < max_cached_terms:
        best_symbol = None
        best_saved = 0
        best_removal = set()
        for symbol in candidate_order:
            if symbol not in active_nodes:
                continue
            if ops_cost.get(symbol, 0) < min_ops_threshold:
                continue
            saved, removal = simulate_removal(
                symbol,
                active_nodes,
                current_ref_counts,
                dependencies,
                ops_cost,
            )
            if saved > best_saved:
                best_symbol = symbol
                best_saved = saved
                best_removal = removal
        if best_symbol is None or best_saved <= 0:
            break
        cached_nodes.append(best_symbol)
        for node in best_removal:
            if node in active_nodes:
                active_nodes.remove(node)
                current_ref_counts.pop(node, None)
        for node in best_removal:
            for dep in dependencies.get(node, set()):
                if dep in current_ref_counts:
                    current_ref_counts[dep] -= 1
    return cached_nodes, active_nodes


def _split_jvp_expressions(
    exprs: Iterable[Tuple[sp.Symbol, sp.Expr]],
    max_cached_terms: Optional[int] = None,
    min_ops_threshold: int = 10,
) -> Tuple[
    List[Tuple[sp.Symbol, sp.Expr]],
    List[Tuple[sp.Symbol, sp.Expr]],
    Dict[int, sp.Expr],
    List[Tuple[sp.Symbol, sp.Expr]],
]:
    """Split expressions into cached auxiliaries, runtime terms, and outputs.

    Parameters
    ----------
    exprs
        Expression pairs ordered for evaluation.
    max_cached_terms
        Maximum number of expressions to cache. Defaults to twice the number
        of JVP outputs.
    min_ops_threshold
        Minimum number of operations required for an expression to be
        considered for caching.

    Returns
    -------
    tuple of list, list, dict, and list
        Cached auxiliary assignments, runtime-only assignments, the indexed
        JVP expressions, and the assignments that must run when preparing the
        cache.

    Notes
    -----
    Separates Jacobian-vector outputs from auxiliary expressions, scores the
    auxiliaries with :func:`_build_expression_costs`, selects cache candidates
    via :func:`_select_cached_nodes`, and computes the closure required to
    populate cached intermediates.
    """

    ordered_exprs = list(exprs)
    if not ordered_exprs:
        return [], [], {}, []

    assigned_symbols = {lhs for lhs, _ in ordered_exprs}
    jvp_terms = {}
    non_jvp_order = []
    non_jvp_exprs = {}
    for lhs, rhs in ordered_exprs:
        lhs_str = str(lhs)
        if lhs_str.startswith("jvp["):
            idx = int(lhs_str.split("[")[1].split("]")[0])
            jvp_terms[idx] = rhs
        else:
            non_jvp_order.append(lhs)
            non_jvp_exprs[lhs] = rhs

    n_jvp = len(jvp_terms)
    if max_cached_terms is None:
        max_cached_terms = 2 * n_jvp

    (
        dependencies,
        dependents,
        ops_cost,
        jvp_usage,
    ) = _build_expression_costs(
        non_jvp_order, non_jvp_exprs, assigned_symbols, jvp_terms
    )

    cached_nodes, active_nodes = _select_cached_nodes(
        non_jvp_order,
        dependencies,
        dependents,
        ops_cost,
        jvp_usage,
        max_cached_terms,
        min_ops_threshold,
    )

    cached_set = set(cached_nodes)
    runtime_nodes = set(active_nodes)

    prepare_nodes = set()
    stack = list(cached_set)
    while stack:
        node = stack.pop()
        if node in prepare_nodes:
            continue
        prepare_nodes.add(node)
        for dep in dependencies.get(node, set()):
            if dep in non_jvp_exprs:
                stack.append(dep)

    cached_aux = []
    runtime_aux = []
    prepare_assigns = []

    for lhs, rhs in ordered_exprs:
        lhs_str = str(lhs)
        if lhs_str.startswith("jvp["):
            continue
        if lhs in prepare_nodes:
            prepare_assigns.append((lhs, rhs))
        if lhs in cached_set:
            cached_aux.append((lhs, rhs))
        elif lhs in runtime_nodes:
            runtime_aux.append((lhs, rhs))

    return cached_aux, runtime_aux, jvp_terms, prepare_assigns

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
    """Build the cached Neumann-series Jacobian-vector body.

    Parameters
    ----------
    jvp_exprs
        Topologically ordered Jacobian-vector product expressions.
    index_map
        Symbol indexing helpers produced by the parser.

    Returns
    -------
    str
        Indented CUDA code statements implementing the cached JVP body.

    Notes
    -----
    Splits auxiliaries and outputs with :func:`_split_jvp_expressions`, maps
    cached values to buffer loads, and reuses the CUDA printer to generate the
    Neumann-series update statements.
    """

    cached_aux, runtime_aux, jvp_terms, _ = _split_jvp_expressions(jvp_exprs)
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
    cached_aux, runtime_aux, jvp_terms, _ = _split_jvp_expressions(jvp_exprs)
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

    cached_aux, runtime_aux, jvp_terms, _ = _split_jvp_expressions(jvp_exprs)
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

    cached_aux, _, _, prepare_assigns = _split_jvp_expressions(jvp_exprs)
    body = _build_prepare_body(cached_aux, prepare_assigns, index_map)
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

    cached_aux, runtime_aux, jvp_terms, _ = _split_jvp_expressions(jvp_exprs)
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
    "def {func_name}(constants, precision,  beta=1.0, gamma=1.0, "
    "order=None):\n"
    '    """Auto-generated residual function for Newton-Krylov ODE '
    'integration.\n'
    "    \n"
    "    Computes residual = beta * M @ v - gamma * h * (J @ eval_point)\n"
    "    where eval_point depends on the residual mode:\n"
    "    - Stage mode: eval_point = base_state + a_ij * u, residual uses M @ "
    "u\n"
    "    - End-state mode: eval_point = u, residual uses M @ (u - "
    "base_state)\n"
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
    "               precision[:],\n"
    "               precision[:]),\n"
    "              device=True,\n"
    "              inline=True)\n"
    "    def residual(u, parameters, drivers, h, a_ij, base_state, out):\n"
    "{res_lines}\n"
    "    return residual\n"
)


def _build_residual_lines(
    equations: ParsedEquations,
    index_map: IndexedBases,
    M: sp.Matrix,
    is_stage: bool,
    cse: bool = True,
) -> str:
    """Construct CUDA code lines for the requested residual mode.

    Parameters
    ----------
    equations
        Parsed equations describing ``dx/dt`` assignments.
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
    equations: ParsedEquations,
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
        Parsed equations defining the system dynamics.
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
    const_block = render_constant_assignments(index_map.constants.symbol_map)

    return RESIDUAL_TEMPLATE.format(
            func_name=func_name,
            const_lines=const_block,
            res_lines=res_lines,
    )

def generate_residual_end_state_code(
    equations: ParsedEquations,
    index_map: IndexedBases,
    M: Optional[Union[sp.Matrix, Iterable[Iterable[sp.Expr]]]] = None,
    func_name: str = "end_residual",
    cse: bool = True,
) -> str:
    """Generate the end-state residual factory.

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
    return generate_residual_code(
        equations=equations,
        index_map=index_map,
        M=M,
        is_stage=False,
        func_name=func_name,
        cse=cse,
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
            is_stage=True,
            func_name=func_name,
            cse=cse,
    )
