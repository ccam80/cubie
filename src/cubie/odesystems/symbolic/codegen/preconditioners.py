"""Emit CUDA factory code for Neumann-series and Jacobi preconditioners.

Published Functions
-------------------
:func:`generate_neumann_preconditioner_code`
    Emit a factory computing a truncated Neumann series approximation
    to ``(I - gamma * h * J)^{-1} * v``.

:func:`generate_neumann_preconditioner_cached_code`
    Variant that reads precomputed auxiliary values from a cache buffer.

:func:`generate_n_stage_neumann_preconditioner_code`
    Emit a flattened multi-stage preconditioner for FIRK methods.

:func:`generate_n_stage_jacobi_preconditioner_code`
    Emit a diagonal Jacobi preconditioner for FIRK methods.

See Also
--------
:mod:`cubie.odesystems.symbolic.codegen.linear_operators`
    Companion linear operator code generators.
:mod:`cubie.odesystems.symbolic.codegen.jacobian`
    Produces the JVP expressions consumed by this module.
:mod:`cubie.odesystems.symbolic.codegen._stage_utils`
    Shared FIRK stage metadata helpers.
"""

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
from cubie.odesystems.symbolic.codegen._stage_utils import (
    build_stage_metadata,
    prepare_stage_data,
)
from cubie.odesystems.symbolic.sym_utils import (
    render_constant_assignments,
    cse_and_stack,
    topological_sort,
    prune_unused_assignments,
)
from cubie.time_logger import default_timelogger

# Register timing events for codegen functions
# Module-level registration required since codegen functions return code
# strings rather than cacheable objects that could auto-register
default_timelogger.register_event(
    "codegen_generate_neumann_preconditioner_code", "codegen",
    "Codegen time for generate_neumann_preconditioner_code")
default_timelogger.register_event(
    "codegen_generate_neumann_preconditioner_cached_code", "codegen",
    "Codegen time for generate_neumann_preconditioner_cached_code")
default_timelogger.register_event(
    "codegen_generate_n_stage_neumann_preconditioner_code", "codegen",
    "Codegen time for generate_n_stage_neumann_preconditioner_code")

NEUMANN_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED NEUMANN PRECONDITIONER FACTORY\n"
    "def {func_name}(constants, precision, beta=1.0, gamma=1.0, order=1):\n"
    '    """Auto-generated Neumann preconditioner.\n'
    "    Approximates (beta*I - gamma*a_ij*h*J)^[-1] via a truncated\n"
    "    Neumann series. Returns device function:\n"
    "      preconditioner(state, parameters, drivers, base_state, t, h, a_ij, v, out, jvp, scratch)\n"
    "    where `jvp` is a caller-provided scratch buffer for J*v.\n"
    '    """\n'
    "    n = int32({n_out})\n"
    "    # Use _cubie_codegen_ prefix to avoid conflicts with user-defined\n"
    "    # variables named beta or gamma (issue #373)\n"
    "    _cubie_codegen_gamma = precision(gamma)\n"
    "    _cubie_codegen_beta = precision(beta)\n"
    "    order = int32(order)\n"
    "    beta_inv = precision(1.0 / _cubie_codegen_beta)\n"
    "    h_eff_factor = precision(_cubie_codegen_gamma * beta_inv)\n"
    "{const_lines}"
    "    @cuda.jit(\n"
    "        # (precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision,\n"
    "        #  precision,\n"
    "        #  precision,\n"
    "        #  precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision[::1]),\n"
    "        device=True,\n"
    "        inline=True)\n"
    "    def preconditioner(\n"
    "        state, parameters, drivers, base_state, t, h, a_ij, v, out, jvp, scratch\n"
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
    "          state, parameters, drivers, cached_aux, base_state, t, h, a_ij, v, out, jvp, scratch\n"
    "      )\n"
    '    """\n'
    "    n = int32({n_out})\n"
    "    order = int32(order)\n"
    "    # Use _cubie_codegen_ prefix to avoid conflicts with user-defined\n"
    "    # variables named beta or gamma (issue #373)\n"
    "    _cubie_codegen_gamma = precision(gamma)\n"
    "    _cubie_codegen_beta = precision(beta)\n"
    "    beta_inv = precision(1.0 / _cubie_codegen_beta)\n"
    "    h_eff_factor = precision(_cubie_codegen_gamma * beta_inv)\n"
    "{const_lines}"
    "    @cuda.jit(\n"
    "        # (precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision,\n"
    "        #  precision,\n"
    "        #  precision,\n"
    "        #  precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision[::1]),\n"
    "        device=True,\n"
    "        inline=True)\n"
    "    def preconditioner(\n"
    "        state, parameters, drivers, cached_aux, base_state, t, h, a_ij, v, out, jvp, scratch\n"
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
    "      preconditioner(state, parameters, drivers, base_state, t, h, a_ij, v, out, jvp, scratch)\n"
    '    """\n'
    "{const_lines}"
    "{metadata_lines}"
    "    total_n = int32({total_states})\n"
    "    _cubie_codegen_gamma = precision(gamma)\n"
    "    _cubie_codegen_beta = precision(beta)\n"
    "    order = int32(order)\n"
    "    beta_inv = precision(1.0 / _cubie_codegen_beta)\n"
    "    h_eff_factor = precision(_cubie_codegen_gamma * beta_inv)\n"
    "    stage_width = int32({state_count})\n"
    "    @cuda.jit(\n"
    "        # (precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision,\n"
    "        #  precision,\n"
    "        #  precision,\n"
    "        #  precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision[::1]),\n"
    "        device=True,\n"
    "        inline=True)\n"
    "    def preconditioner(state, parameters, drivers, base_state, t, h, a_ij, v, out, jvp, scratch):\n"
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

def _build_neumann_body_with_state_subs(
    jvp_equations: JVPEquations,
    index_map: IndexedBases,
) -> str:
    """Build non-cached Neumann JVP body with inline state evaluation.
    
    For Newton-Krylov usage: state param is stage_increment,
    need to evaluate at base_state + a_ij * stage_increment
    """
    
    # Add state substitution for inline evaluation
    state_subs = {}
    state_symbols = list(index_map.states.index_map.keys())
    state_indexed = sp.IndexedBase("state")
    base_state_indexed = sp.IndexedBase("base_state")
    a_ij_sym = sp.Symbol("a_ij")
    
    for i, state_sym in enumerate(state_symbols):
        eval_point = base_state_indexed[i] + a_ij_sym * state_indexed[i]
        state_subs[state_sym] = eval_point
    
    # Apply substitution to all assignments
    assignments = jvp_equations.ordered_assignments
    substituted_assignments = [
        (lhs, rhs.subs(state_subs)) for lhs, rhs in assignments
    ]

    lines = print_cuda_multiple(substituted_assignments, symbol_map=index_map.all_arrayrefs)
    if not lines:
        lines = ["pass"]
    else:
        lines = [
            ln.replace("v[", "out[").replace("jvp[", "jvp[")
            for ln in lines
        ]
    substituted_assignments = prune_unused_assignments(
            substituted_assignments, outputsym_str='out'
    )
    return "\n".join("            " + ln for ln in lines)

def _build_cached_neumann_body(
    equations: JVPEquations,
    index_map: IndexedBases,
) -> str:
    """Build the cached Neumann-series Jacobian-vector body.
    
    For Rosenbrock usage: state param is actual state,
    evaluate at state directly (no substitution needed)
    """

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

    exprs = prune_unused_assignments(exprs, outputsym_str='v')
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
        v_indexed = sp.IndexedBase("v", shape=(state_count,))
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
    eval_exprs = prune_unused_assignments(eval_exprs, outputsym_str='jvp')

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
    default_timelogger.start_event("codegen_generate_n_stage_neumann_preconditioner_code")

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
    result = N_STAGE_NEUMANN_TEMPLATE.format(
        func_name=func_name,
        const_lines=const_block,
        metadata_lines="",
        jv_body=body,
        stage_count=stage_count,
        total_states=total_states,
        state_count=state_count,
    )
    default_timelogger.stop_event("codegen_generate_n_stage_neumann_preconditioner_code")
    return result

def generate_neumann_preconditioner_code(
    equations: ParsedEquations,
    index_map: IndexedBases,
    func_name: str = "neumann_preconditioner_factory",
    cse: bool = True,
    jvp_equations: Optional[JVPEquations] = None,
) -> str:
    """Generate the Neumann preconditioner factory.
    
    For Newton-Krylov usage: applies inline state evaluation.
    """
    default_timelogger.start_event("codegen_generate_neumann_preconditioner_code")

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
    jv_body = _build_neumann_body_with_state_subs(jvp_equations, index_map)
    result = NEUMANN_TEMPLATE.format(
        func_name=func_name,
        n_out=n_out,
        jv_body=jv_body,
        const_lines=const_block,
    )
    default_timelogger.stop_event("codegen_generate_neumann_preconditioner_code")
    return result

def generate_neumann_preconditioner_cached_code(
    equations: ParsedEquations,
    index_map: IndexedBases,
    func_name: str = "neumann_preconditioner_cached",
    cse: bool = True,
    jvp_equations: Optional[JVPEquations] = None,
) -> str:
    """Generate the cached Neumann preconditioner factory.
    
    For Rosenbrock usage: state param is actual state,
    no inline substitution needed.
    """
    default_timelogger.start_event("codegen_generate_neumann_preconditioner_cached_code")

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
    result = NEUMANN_CACHED_TEMPLATE.format(
        func_name=func_name,
        n_out=n_out,
        jv_body=jv_body,
        const_lines=const_block,
    )
    default_timelogger.stop_event("codegen_generate_neumann_preconditioner_cached_code")
    return result


N_STAGE_JACOBI_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED N-STAGE DIAGONAL JACOBI PRECONDITIONER FACTORY\n"
    "def {func_name}(constants, precision, beta=1.0, gamma=1.0, order=1):\n"
    '    """Auto-generated FIRK diagonal Jacobi preconditioner.\n'
    "    Handles {stage_count} stages with ``s * n`` unknowns.\n"
    "    Computes diagonal of ``beta * I - gamma * h * (A ⊗ J)`` and\n"
    "    applies pointwise inversion: ``out[k] = v[k] / d[k]``.\n"
    "    Returns device function:\n"
    "      preconditioner(state, parameters, drivers, base_state,"
    " t, h, a_ij, v, out, jvp, scratch)\n"
    '    """\n'
    "{const_lines}"
    "{metadata_lines}"
    "    total_n = int32({total_states})\n"
    "    _cubie_codegen_gamma = precision(gamma)\n"
    "    _cubie_codegen_beta = precision(beta)\n"
    "    stage_width = int32({state_count})\n"
    "    @cuda.jit(\n"
    "        device=True,\n"
    "        inline=True)\n"
    "    def preconditioner("
    "state, parameters, drivers, base_state, t, h, a_ij, v, out, jvp, scratch):\n"
    "        # Evaluate diagonal J_ii at each stage evaluation point,\n"
    "        # form d[s*n+i] = beta - gamma*h*a_ss*J_ii,\n"
    "        # apply out[k] = v[k] / d[k].\n"
    "{diag_body}\n"
    "    return preconditioner\n"
)


def _build_n_stage_jacobi_lines(
    equations: ParsedEquations,
    index_map: IndexedBases,
    stage_coefficients: sp.Matrix,
    stage_nodes: Tuple[sp.Expr, ...],
    cse: bool = True,
) -> str:
    """Build diagonal Jacobi preconditioner body for n-stage FIRK.

    Extracts J_ii = df_i/dy_i for each state, evaluates at each
    stage point, forms d = beta - gamma*h*a_ss*J_ii, and applies
    out[k] = v[k] / d[k].
    """
    from cubie.odesystems.symbolic.codegen.jacobian import (
        generate_jacobian,
    )

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

    # Get full Jacobian (cached from JVP generation)
    jac = generate_jacobian(
        equations,
        input_order=index_map.states.index_map,
        output_order=index_map.dxdt.index_map,
    )

    total_states = sp.Integer(stage_count * state_count)
    state_vec = sp.IndexedBase("state", shape=(total_states,))
    base_state = sp.IndexedBase(
        "base_state", shape=(sp.Integer(state_count),)
    )
    out_vec = sp.IndexedBase("out", shape=(total_states,))
    v_vec = sp.IndexedBase("v", shape=(total_states,))
    time_arg = sp.Symbol("t")
    h_sym = sp.Symbol("h")
    beta_sym = sp.Symbol("_cubie_codegen_beta")
    gamma_sym = sp.Symbol("_cubie_codegen_gamma")

    driver_count = len(driver_symbols)
    if driver_count:
        drivers = sp.IndexedBase(
            "drivers",
            shape=(sp.Integer(stage_count * driver_count),),
        )
    else:
        drivers = sp.IndexedBase("drivers")

    eval_exprs: List[Tuple[sp.Symbol, sp.Expr]] = list(metadata_exprs)

    for stage_idx in range(stage_count):
        # Build substitution: state symbols -> evaluation point
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
        substitution_map[TIME_SYMBOL] = (
            time_arg + h_sym * node_symbols[stage_idx]
        )

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
                coeff_value = stage_coefficients[
                    stage_idx, contrib_idx
                ]
                if coeff_value == 0:
                    continue
                coeff_sym = coeff_symbols[stage_idx][contrib_idx]
                expr += coeff_sym * state_vec[
                    contrib_idx * state_count + state_idx
                ]
            stage_state_subs[state_sym] = expr

        # Emit full equation list with stage subs so every
        # intermediate is defined as a local variable (matches
        # Neumann approach).  prune_unused_assignments removes
        # anything not transitively needed by out[].
        substituted_eqs = [
            (
                lhs.subs(substitution_map),
                rhs.subs(substitution_map).subs(
                    stage_state_subs
                ),
            )
            for lhs, rhs in eq_list
        ]
        eval_exprs.extend(substituted_eqs)

        # Substitute into auxiliary equations (needed by J_ii)
        aux_order_list = []
        aux_exprs_dict = {}
        for lhs, rhs in eq_list:
            if lhs not in set(dx_symbols):
                aux_order_list.append(lhs)
                aux_exprs_dict[lhs] = rhs

        stage_aux_assignments: List[Tuple[sp.Symbol, sp.Expr]] = []
        aux_subs: Dict[sp.Symbol, sp.Symbol] = {}
        for lhs in aux_order_list:
            stage_symbol = sp.Symbol(f"{str(lhs)}_{stage_idx}")
            rhs = aux_exprs_dict[lhs]
            substituted_rhs = rhs.subs(substitution_map)
            substituted_rhs = substituted_rhs.subs(stage_state_subs)
            if aux_subs:
                substituted_rhs = substituted_rhs.subs(aux_subs)
            stage_aux_assignments.append(
                (stage_symbol, substituted_rhs)
            )
            aux_subs[lhs] = stage_symbol
        eval_exprs.extend(stage_aux_assignments)

        # Extract diagonal: J_ii for each state
        # Build combined substitution map for a single pass
        combined_subs = {}
        combined_subs.update(substitution_map)
        combined_subs.update(stage_state_subs)
        combined_subs.update(aux_subs)
        diag_coeff = coeff_symbols[stage_idx][stage_idx]
        stage_offset = stage_idx * state_count
        for comp_idx in range(state_count):
            j_ii = jac[comp_idx, comp_idx]
            # Substitute all refs in one pass using xreplace
            substituted = j_ii.xreplace(combined_subs)

            diag_sym = sp.Symbol(
                f"diag_{stage_idx}_{comp_idx}"
            )
            diag_val = (
                beta_sym
                - gamma_sym * h_sym * diag_coeff * substituted
            )
            eval_exprs.append((diag_sym, diag_val))

            # out[k] = v[k] / d[k]
            eval_exprs.append((
                out_vec[stage_offset + comp_idx],
                v_vec[stage_offset + comp_idx] / diag_sym,
            ))

    if cse:
        eval_exprs = cse_and_stack(eval_exprs)
    else:
        eval_exprs = topological_sort(eval_exprs)

    symbol_map = dict(index_map.all_arrayrefs)
    symbol_map.update({
        "state": state_vec,
        "base_state": base_state,
        "out": out_vec,
        "v": v_vec,
        "t": time_arg,
    })
    eval_exprs = prune_unused_assignments(
        eval_exprs, outputsym_str='out'
    )

    lines = print_cuda_multiple(
        eval_exprs, symbol_map=symbol_map
    )
    if not lines:
        return "            pass"
    return "\n".join("        " + ln for ln in lines)


default_timelogger.register_event(
    "codegen_generate_n_stage_jacobi_preconditioner_code",
    "codegen",
    "Codegen time for generate_n_stage_jacobi_preconditioner_code",
)


def generate_n_stage_jacobi_preconditioner_code(
    equations: ParsedEquations,
    index_map: IndexedBases,
    stage_coefficients: Sequence[Sequence[Union[float, sp.Expr]]],
    stage_nodes: Sequence[Union[float, sp.Expr]],
    func_name: str = "n_stage_jacobi_preconditioner",
    cse: bool = True,
) -> str:
    """Generate a diagonal Jacobi preconditioner for n-stage FIRK.

    Computes ``diag(beta*I - gamma*h*(A_diag x J_diag))`` and
    applies pointwise inversion. Much cheaper than Neumann series
    and handles stiff diagonal entries correctly.

    Parameters
    ----------
    equations
        Parsed ODE equations.
    index_map
        Symbol-to-array mapping for states, parameters, etc.
    stage_coefficients
        Butcher tableau A matrix.
    stage_nodes
        Butcher tableau c vector.
    func_name
        Name for the generated factory function.
    cse
        Whether to apply common-subexpression elimination.

    Returns
    -------
    str
        Generated Python/CUDA factory function code.
    """
    default_timelogger.start_event(
        "codegen_generate_n_stage_jacobi_preconditioner_code"
    )

    coeff_matrix, node_values, stage_count = prepare_stage_data(
        stage_coefficients, stage_nodes
    )
    body = _build_n_stage_jacobi_lines(
        equations=equations,
        index_map=index_map,
        stage_coefficients=coeff_matrix,
        stage_nodes=node_values,
        cse=cse,
    )
    const_block = render_constant_assignments(
        index_map.constants.symbol_map
    )
    total_states = stage_count * len(index_map.states.index_map)
    state_count = len(index_map.states.index_map)

    metadata_lines = ""
    result = N_STAGE_JACOBI_TEMPLATE.format(
        func_name=func_name,
        const_lines=const_block,
        metadata_lines=metadata_lines,
        diag_body=body,
        stage_count=stage_count,
        total_states=total_states,
        state_count=state_count,
    )
    default_timelogger.stop_event(
        "codegen_generate_n_stage_jacobi_preconditioner_code"
    )
    return result


__all__ = [
    "generate_neumann_preconditioner_code",
    "generate_neumann_preconditioner_cached_code",
    "generate_n_stage_neumann_preconditioner_code",
    "generate_n_stage_jacobi_preconditioner_code",
]
