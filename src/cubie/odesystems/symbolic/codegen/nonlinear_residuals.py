"""Code generation helpers for nonlinear residual functions."""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import sympy as sp

from cubie.odesystems.symbolic.codegen.numba_cuda_printer import (
    print_cuda_multiple,
)
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

RESIDUAL_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED NONLINEAR RESIDUAL FACTORY\n"
    "def {func_name}(constants, precision, beta=1.0, gamma=1.0, order=None):\n"
    '    """Auto-generated nonlinear residual for implicit updates.\n'
    "    Computes beta * M * u - gamma * h * f(t, base_state + a_ij * u).\n"
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
    "               precision,\n"
    "               precision[:],\n"
    "               precision[:]),\n"
    "              device=True,\n"
    "              inline=True)\n"
    "    def residual(u, parameters, drivers, t, h, a_ij, base_state, out):\n"
    "{body}\n"
    "    return residual\n"
)


def _build_residual_lines(
    equations: ParsedEquations,
    index_map: IndexedBases,
    M: sp.Matrix,
    cse: bool = True,
) -> str:
    """Construct CUDA code lines for the stage-increment residual."""

    eq_list = equations.to_equation_list()

    n = len(index_map.states.index_map)

    beta_sym = sp.Symbol("beta")
    gamma_sym = sp.Symbol("gamma")
    h_sym = sp.Symbol("h")
    aij_sym = sp.Symbol("a_ij")
    u = sp.IndexedBase("u", shape=(n,))
    base = sp.IndexedBase("base_state", shape=(n,))
    out = sp.IndexedBase("out", shape=(n,))

    dx_subs = {}
    for i, (dx_sym, _) in enumerate(index_map.dxdt.index_map.items()):
        dx_subs[dx_sym] = sp.Symbol(f"dx_{i}")

    obs_subs = {}
    if index_map.observable_symbols:
        obs_subs = dict(
            zip(
                index_map.observable_symbols,
                sp.numbered_symbols("aux_", start=1),
            )
        )

    all_subs = {**dx_subs, **obs_subs}
    substituted_equations = [
        (lhs.subs(all_subs), rhs.subs(all_subs)) for lhs, rhs in eq_list
    ]

    state_subs = {}
    state_symbols = list(index_map.states.index_map.keys())
    for i, state_sym in enumerate(state_symbols):
        eval_point = base[i] + aij_sym * u[i]
        state_subs[state_sym] = eval_point

    eval_equations = []
    for lhs, rhs in substituted_equations:
        eval_rhs = rhs.subs(state_subs)
        eval_equations.append((lhs, eval_rhs))

    symbol_map = dict(index_map.all_arrayrefs)
    symbol_map.update(
        {
            "beta": beta_sym,
            "gamma": gamma_sym,
            "h": h_sym,
            "a_ij": aij_sym,
            "u": u,
            "base_state": base,
            "out": out,
        }
    )

    eval_exprs = eval_equations

    for i in range(n):
        mv = sp.S.Zero
        for j in range(n):
            entry = M[i, j]
            if entry == 0:
                continue
            mv += entry * u[j]

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


def _build_n_stage_residual_lines(
    equations: ParsedEquations,
    index_map: IndexedBases,
    M: sp.Matrix,
    stage_coefficients: sp.Matrix,
    stage_nodes: Tuple[sp.Expr, ...],
    cse: bool = True,
) -> str:
    """Construct CUDA statements for the FIRK n-stage residual."""

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
    u = sp.IndexedBase("u", shape=(total_states,))
    base_state = sp.IndexedBase("base_state", shape=(sp.Integer(state_count),))
    out = sp.IndexedBase("out", shape=(total_states,))

    driver_count = len(driver_symbols)
    if driver_count:
        drivers = sp.IndexedBase(
            "drivers", shape=(sp.Integer(stage_count * driver_count),)
        )
    else:
        drivers = sp.IndexedBase("drivers")

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
                expr += coeff_sym * u[contrib_idx * state_count + state_idx]
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

            dx_symbol = sp.Symbol(f"dx_{stage_idx}_{comp_idx}")
            update_expr = beta_sym * mv - gamma_sym * h_sym * dx_symbol
            eval_exprs.append((out[stage_offset + comp_idx], update_expr))

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
            "t": time_arg,
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
    """Emit the stage-increment residual factory for Newton--Krylov integration."""

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
    """Generate the stage residual factory."""

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

    coeff_matrix, node_values, stage_count = prepare_stage_data(
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


__all__ = [
    "generate_residual_code",
    "generate_stage_residual_code",
    "generate_n_stage_residual_code",
]
