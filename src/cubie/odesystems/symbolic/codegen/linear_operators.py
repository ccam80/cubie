"""Code generation helpers for linear operators and Jacobian products."""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import sympy as sp

from cubie.odesystems.symbolic.codegen.numba_cuda_printer import (
    print_cuda_multiple,
)
from cubie.odesystems.symbolic.codegen.jacobian import generate_analytical_jvp
from cubie.odesystems.symbolic.parsing.jvp_equations import JVPEquations
from cubie.odesystems.symbolic.parsing.parser import (
    IndexedBases,
    ParsedEquations,
    TIME_SYMBOL,
)
from cubie.odesystems.symbolic.sym_utils import (
    cse_and_stack,
    render_constant_assignments,
    topological_sort,
)

from ._stage_utils import build_stage_metadata, prepare_stage_data

CACHED_OPERATOR_APPLY_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED CACHED LINEAR OPERATOR FACTORY\n"
    "def {func_name}(constants, precision, beta=1.0, gamma=1.0, order=None):\n"
    '    """Auto-generated cached linear operator.\n'
    "    Computes out = beta * (M @ v) - gamma * a_ij * h * (J @ v)\n"
    "    using cached auxiliary intermediates.\n"
    "    Returns device function:\n"
    "      operator_apply(\n"
    "          state, parameters, drivers, cached_aux, base_state, t, h, a_ij, v, out\n"
    "      )\n"
    "    argument 'order' is ignored, included for compatibility with\n"
    "    preconditioner API.\n"
    '    """\n'
    "{const_lines}"
    "    @cuda.jit((precision[:],\n"
    "               precision[:],\n"
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
    "        state, parameters, drivers, cached_aux, base_state, t, h, a_ij, v, out\n"
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
    "      operator_apply(state, parameters, drivers, base_state, t, h, a_ij, v, out)\n"
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
    "    def operator_apply(state, parameters, drivers, base_state, t, h, a_ij, v, out):\n"
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
    """Partition assignments into cached, runtime, and preparation subsets."""

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
    """Build the CUDA body computing ``β·M·v − γ·h·J·v``."""

    n_out = len(index_map.dxdt.ref_map)
    n_in = len(index_map.states.index_map)
    v = sp.IndexedBase("v")
    beta_sym = sp.Symbol("beta")
    gamma_sym = sp.Symbol("gamma")
    a_ij_sym = sp.Symbol("a_ij")
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
    cached_assigns: List[Tuple[sp.Symbol, sp.Expr]],
    prepare_assigns: List[Tuple[sp.Symbol, sp.Expr]],
    index_map: IndexedBases,
) -> str:
    """Build the CUDA body populating the cached Jacobian auxiliaries."""

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
    """Emit the operator apply factory from precomputed JVP expressions."""

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
    """Generate the linear operator factory from system equations."""

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

    metadata_exprs, coeff_symbols, node_symbols = build_stage_metadata(
        stage_coefficients, stage_nodes
    )
    eq_list = equations.to_equation_list()
    state_symbols = list(index_map.states.index_map.keys())
    dx_symbols = list(index_map.dxdt.index_map.keys())
    observable_symbols = list(index_map.observable_symbols)
    driver_symbols = list(index_map.drivers.index_map.keys())
    state_count = len(state_symbols)
    stage_count = stage_coefficients.rows

    beta_sym = sp.Symbol("beta")
    gamma_sym = sp.Symbol("gamma")
    h_sym = sp.Symbol("h")
    time_arg = sp.Symbol("t")
    total_states = sp.Integer(stage_count * state_count)
    state_vec = sp.IndexedBase("state", shape=(total_states,))
    base_state = sp.IndexedBase("base_state", shape=(sp.Integer(state_count),))
    direction_vec = sp.IndexedBase("v", shape=(total_states,))
    out = sp.IndexedBase("out", shape=(total_states,))

    driver_count = len(driver_symbols)
    if driver_count:
        drivers = sp.IndexedBase(
            "drivers", shape=(sp.Integer(stage_count * driver_count),)
        )
    else:
        drivers = sp.IndexedBase("drivers")

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
        substitution_map[TIME_SYMBOL] = time_arg + h_sym * node_symbols[stage_idx]

        if driver_count:
            stage_driver_offset = stage_idx * driver_count
            for driver_idx, driver_sym in enumerate(driver_symbols):
                substitution_map[driver_sym] = drivers[
                    stage_driver_offset + driver_idx
                ]

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
            "t": time_arg,
        }
    )

    lines = print_cuda_multiple(eval_exprs, symbol_map=symbol_map)
    if not lines:
        return "        pass"
    return "\n".join("        " + ln for ln in lines)


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

    coeff_matrix, node_values, stage_count = prepare_stage_data(
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
    "               precision[:],\n"
    "               precision,\n"
    "               precision,\n"
    "               precision,\n"
    "               precision[:],\n"
    "               precision[:]),\n"
    "              device=True,\n"
    "              inline=True)\n"
    "    def operator_apply(state, parameters, drivers, base_state, t, h, a_ij, v, out):\n"
    "{body}\n"
    "    return operator_apply\n"
)


__all__ = [
    "generate_operator_apply_code",
    "generate_cached_operator_apply_code",
    "generate_prepare_jac_code",
    "generate_cached_jvp_code",
    "generate_operator_apply_code_from_jvp",
    "generate_cached_operator_apply_code_from_jvp",
    "generate_prepare_jac_code_from_jvp",
    "generate_cached_jvp_code_from_jvp",
    "generate_n_stage_linear_operator_code",
]
