"""Generate CUDA helpers evaluating time derivatives of symbolic systems."""

from typing import Dict, List, Optional, Tuple

import sympy as sp

from cubie.odesystems.symbolic.codegen import print_cuda_multiple
from cubie.odesystems.symbolic.codegen.jacobian import _prune_unused_assignments
from cubie.odesystems.symbolic.parsing import (
    IndexedBases,
    ParsedEquations,
    TIME_SYMBOL,
)
from cubie.odesystems.symbolic.sym_utils import (
    cse_and_stack,
    render_constant_assignments,
    topological_sort,
)


TIME_DERIVATIVE_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED TIME-DERIVATIVE FACTORY\n"
    "def {func_name}(constants, precision):\n"
    '    """Auto-generated time-derivative factory."""\n'
    "{const_lines}"
    "    @cuda.jit((precision[:],\n"
    "               precision[:],\n"
    "               precision[:],\n"
    "               precision[:],\n"
    "               precision[:],\n"
    "               precision[:],\n"
    "               precision),\n"
    "              device=True,\n"
    "              inline=True)\n"
    "    def time_derivative_rhs(\n"
    "        state, parameters, drivers, driver_dt, observables, out, t\n"
    "    ):\n"
    "{body}\n"
    "\n"
    "    return time_derivative_rhs\n"
)


def _build_time_derivative_assignments(
    equations: ParsedEquations,
    index_map: IndexedBases,
) -> Tuple[List[Tuple[sp.Symbol, sp.Expr]], Dict[sp.Symbol, sp.Expr]]:
    """Return assignments required for time-derivative evaluation."""

    sorted_equations = topological_sort(
        equations.non_observable_equations()
    )
    output_symbols = set(index_map.dxdt.ref_map.keys())
    driver_symbols = list(index_map.drivers.ref_map.keys())
    driver_dt: Optional[sp.IndexedBase] = None
    driver_indices = index_map.drivers.index_map
    if driver_symbols:
        driver_dt = sp.IndexedBase(
            "driver_dt", shape=(index_map.drivers.length,)
        )

    symbol_derivatives: Dict[sp.Symbol, sp.Expr] = {}
    derivative_symbols: Dict[sp.Symbol, sp.Symbol] = {}

    assignments: List[Tuple[sp.Symbol, sp.Expr]] = list(sorted_equations)
    derivative_assignments: List[Tuple[sp.Symbol, sp.Expr]] = []

    processed: set[sp.Symbol] = set()
    for lhs, rhs in sorted_equations:
        processed.add(lhs)
        direct_time = sp.diff(rhs, TIME_SYMBOL)

        driver_term = sp.S.Zero
        if driver_dt is not None:
            for driver in driver_symbols:
                if driver in rhs.free_symbols:
                    partial = sp.diff(rhs, driver)
                    driver_term += partial * driver_dt[driver_indices[driver]]

        chain_term = sp.S.Zero
        for dep in sorted(rhs.free_symbols & processed, key=str):
            derivative = symbol_derivatives.get(dep)
            if derivative is None:
                continue
            chain_term += sp.diff(rhs, dep) * derivative

        total = direct_time + driver_term + chain_term
        deriv_symbol = sp.Symbol(f"time_{lhs}")
        symbol_derivatives[lhs] = total
        derivative_symbols[lhs] = deriv_symbol
        derivative_assignments.append((deriv_symbol, total))

    assignments.extend(derivative_assignments)

    final_symbol_map: Dict[sp.Symbol, sp.Expr] = {}
    for out_sym in sorted(
        output_symbols, key=lambda sym: index_map.dxdt.index_map[sym]
    ):
        final_symbol = sp.Symbol(
            f"time_rhs[{index_map.dxdt.index_map[out_sym]}]"
        )
        final_symbol_map[final_symbol] = index_map.dxdt.ref_map[out_sym]
        assignments.append((final_symbol, derivative_symbols[out_sym]))

    return assignments, final_symbol_map


def generate_time_derivative_lines(
    equations: ParsedEquations,
    index_map: IndexedBases,
    cse: bool = True,
) -> List[str]:
    """Return CUDA assignments computing the explicit time derivative."""

    assignments, final_symbol_map = _build_time_derivative_assignments(
        equations, index_map
    )

    if cse:
        processed = cse_and_stack(assignments)
    else:
        processed = topological_sort(assignments)

    processed = _prune_unused_assignments(
        processed,
        outputsym_str="time_rhs",
        output_symbols=final_symbol_map.keys(),
    )

    symbol_map = dict(index_map.all_arrayrefs)
    symbol_map.update(final_symbol_map)

    lines = print_cuda_multiple(processed, symbol_map=symbol_map)
    if not lines:
        lines = ["pass"]
    return lines


def generate_time_derivative_fac_code(
    equations: ParsedEquations,
    index_map: IndexedBases,
    func_name: str = "time_derivative_rhs_factory",
    cse: bool = True,
) -> str:
    """Return source for the time-derivative CUDA factory."""

    body_lines = generate_time_derivative_lines(
        equations, index_map=index_map, cse=cse
    )
    body = "\n".join(f"        {line}" for line in body_lines)
    const_block = render_constant_assignments(
        index_map.constants.symbol_map
    )
    return TIME_DERIVATIVE_TEMPLATE.format(
        func_name=func_name,
        const_lines=const_block,
        body=body,
    )
