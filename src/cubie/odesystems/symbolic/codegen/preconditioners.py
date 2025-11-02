"""Code generation helpers for implicit solver preconditioners."""

from typing import List, Optional, Tuple, Dict, Sequence, Union

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
from cubie.odesystems.symbolic.codegen._stage_utils import build_stage_metadata, \
    prepare_stage_data
from cubie.odesystems.symbolic.sym_utils import (
    render_constant_assignments,
    cse_and_stack,
    topological_sort,
)

NEUMANN_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED NEUMANN PRECONDITIONER FACTORY\n"
    "def {func_name}(constants, precision, beta=1.0, gamma=1.0, order=1):\n"
    '    """Auto-generated Neumann preconditioner.\n'
    "    Approximates (beta*I - gamma*a_ij*h*J)^[-1] via a truncated\n"
    "    Neumann series. Returns device function:\n"
    "      preconditioner(state, parameters, drivers, base_state, t, h, a_ij, v, out, jvp)\n"
    "    where `jvp` is a caller-provided scratch buffer for J*v.\n"
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
    "        state, parameters, drivers, base_state, t, h, a_ij, v, out, jvp\n"
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
    "          state, parameters, drivers, cached_aux, base_state, t, h, a_ij, v, out, jvp\n"
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
    "        state, parameters, drivers, cached_aux, base_state, t, h, a_ij, v, out, jvp\n"
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


N_STAGE_NEUMANN_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED N-STAGE NEUMANN PRECONDITIONER FACTORY\n"
    "def {func_name}(constants, precision, beta=1.0, gamma=1.0, order=1):\n"
    '    """Auto-generated FIRK Neumann preconditioner.\n'
    "    Handles {stage_count} stages with ``s * n`` unknowns.\n"
    "    Approximates the inverse of ``beta * I - gamma * h * (A ⊗ J)`` using\n"
    "    a truncated Neumann series applied to flattened stages.\n"
    "    Returns device function:\n"
    "      preconditioner(state, parameters, drivers, base_state, t, h, a_ij, v, out, jvp)\n"
    '    """\n'
    "{const_lines}"
    "{metadata_lines}"
    "    total_n = {total_states}\n"
    "    beta_inv = 1.0 / beta\n"
    "    h_eff_factor = gamma * beta_inv\n"
    "    stage_width = {state_count}\n"
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
    "    def preconditioner(state, parameters, drivers, base_state, t, h, a_ij, v, out, jvp):\n"
    "        for i in range(total_n):\n"
    "            out[i] = v[i]\n"
    "        h_eff = h * h_eff_factor\n"
    "        for _ in range(order):\n"
    "{jv_body}\n"
    "            for i in range(total_n):\n"
    "                out[i] = v[i] + h_eff * jvp[i]\n"
    "        for i in range(total_n):\n"
    "            out[i] = beta_inv * out[i]\n"
    "    return preconditioner\n"
)

def _build_cached_neumann_body(
    equations: JVPEquations,
    index_map: IndexedBases,
) -> str:
    """Build the cached Neumann-series Jacobian-vector body."""

    cached_aux, runtime_aux, _ = equations.cached_partition()
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

def _build_n_stage_neumann_lines(
    equations: ParsedEquations,
    index_map: IndexedBases,
    stage_coefficients: sp.Matrix,
    stage_nodes: Tuple[sp.Expr, ...],
    jvp_equations: JVPEquations,
    cse: bool = True,
) -> str:
    """Construct CUDA statements computing J·v for flattened FIRK stages."""

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

    total_states = sp.Integer(stage_count * state_count)
    state_vec = sp.IndexedBase("state", shape=(total_states,))
    base_state = sp.IndexedBase("base_state", shape=(sp.Integer(state_count),))
    direction_vec = sp.IndexedBase("out", shape=(total_states,))
    scratch = sp.IndexedBase("jvp", shape=(total_states,))
    time_arg = sp.Symbol("t")
    h_sym = sp.Symbol("h")

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
            substituted_expr = expr.subs(substitution_map)
            substituted_expr = substituted_expr.subs(stage_state_subs)
            if aux_subs:
                substituted_expr = substituted_expr.subs(aux_subs)
            substituted_expr = substituted_expr.subs(v_subs)
            eval_exprs.append((stage_symbol, substituted_expr))
            stage_jvp_symbols[idx] = stage_symbol

        stage_offset = stage_idx * state_count
        for comp_idx in range(state_count):
            jvp_value = stage_jvp_symbols.get(comp_idx, sp.S.Zero)
            eval_exprs.append(
                (scratch[stage_offset + comp_idx], jvp_value)
            )

    if cse:
        eval_exprs = cse_and_stack(eval_exprs)
    else:
        eval_exprs = topological_sort(eval_exprs)

    symbol_map = dict(index_map.all_arrayrefs)
    symbol_map.update(
        {
            "state": state_vec,
            "base_state": base_state,
            "out": direction_vec,
            "jvp": scratch,
            "t": time_arg,
        }
    )

    lines = print_cuda_multiple(eval_exprs, symbol_map=symbol_map)
    if not lines:
        return "            pass"
    return "\n".join("            " + ln for ln in lines)


def generate_n_stage_neumann_preconditioner_code(
    equations: ParsedEquations,
    index_map: IndexedBases,
    stage_coefficients: Sequence[Sequence[Union[float, sp.Expr]]],
    stage_nodes: Sequence[Union[float, sp.Expr]],
    func_name: str = "n_stage_neumann_preconditioner",
    cse: bool = True,
    jvp_equations: Optional[JVPEquations] = None,
) -> str:
    """Generate a flattened n-stage FIRK Neumann preconditioner factory."""

    coeff_matrix, node_values, stage_count = prepare_stage_data(
        stage_coefficients, stage_nodes
    )
    if jvp_equations is None:
        jvp_equations = generate_analytical_jvp(
            equations,
            input_order=index_map.states.index_map,
            output_order=index_map.dxdt.index_map,
            observables=index_map.observable_symbols,
            cse=cse,
        )
    body = _build_n_stage_neumann_lines(
        equations=equations,
        index_map=index_map,
        stage_coefficients=coeff_matrix,
        stage_nodes=node_values,
        jvp_equations=jvp_equations,
        cse=cse,
    )
    const_block = render_constant_assignments(index_map.constants.symbol_map)
    total_states = stage_count * len(index_map.states.index_map)
    state_count = len(index_map.states.index_map)
    return N_STAGE_NEUMANN_TEMPLATE.format(
        func_name=func_name,
        const_lines=const_block,
        metadata_lines="",
        jv_body=body,
        stage_count=stage_count,
        total_states=total_states,
        state_count=state_count,
    )

def generate_neumann_preconditioner_code(
    equations: ParsedEquations,
    index_map: IndexedBases,
    func_name: str = "neumann_preconditioner_factory",
    cse: bool = True,
    jvp_equations: Optional[JVPEquations] = None,
) -> str:
    """Generate the Neumann preconditioner factory."""

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
        func_name=func_name,
        n_out=n_out,
        jv_body=jv_body,
        const_lines=const_block,
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


__all__ = [
    "generate_neumann_preconditioner_code",
    "generate_neumann_preconditioner_cached_code",
    "generate_n_stage_neumann_preconditioner_code",
]
