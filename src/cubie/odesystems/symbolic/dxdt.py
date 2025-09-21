from typing import Iterable, Optional, Tuple

import sympy as sp

from cubie.odesystems.symbolic.jacobian import _prune_unused_assignments
from cubie.odesystems.symbolic.numba_cuda_printer import print_cuda_multiple
from cubie.odesystems.symbolic.parser import IndexedBases
from cubie.odesystems.symbolic.sym_utils import (
    cse_and_stack,
    topological_sort,
)

DXDT_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED DXDT FACTORY\n"
    "def {func_name}(constants, precision):\n"
    '    """Auto-generated dxdt factory."""\n'
    "{const_lines}"
    "    @cuda.jit((precision[:],\n"
    "               precision[:],\n"
    "               precision[:],\n"
    "               precision[:],\n"
    "               precision[:]),\n"
    "              device=True,\n"
    "              inline=True)\n"
    "    def dxdt(state, parameters, drivers, observables, out):\n"
    "    {body}\n"
    "    \n"
    "    return dxdt\n"
)

OBSERVABLES_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED OBSERVABLES FACTORY\n"
    "def {func_name}(constants, precision):\n"
    '    """Auto-generated observables factory."""\n'
    "{const_lines}"
    "    @cuda.jit((precision[:],\n"
    "               precision[:],\n"
    "               precision[:],\n"
    "               precision[:]),\n"
    "              device=True,\n"
    "              inline=True)\n"
    "    def get_observables(state, parameters, drivers, observables):\n"
    "    {body}\n"
    "    \n"
    "    return get_observables\n"
)


def generate_dxdt_lines(
    equations: Iterable[Tuple[sp.Symbol, sp.Expr]],
    index_map: Optional[IndexedBases] = None,
    cse: bool = True,
):
    """Return the dxdt lines for the given equations."""

    if cse:
        equations = cse_and_stack(equations)
    else:
        equations = topological_sort(equations)
    dxdt_lines = print_cuda_multiple(
        equations, symbol_map=index_map.all_arrayrefs
    )
    if not dxdt_lines:
        dxdt_lines = ["pass"]
    return dxdt_lines


def generate_observables_lines(
    equations: Iterable[Tuple[sp.Symbol, sp.Expr]],
    index_map: IndexedBases,
    cse: bool = True,
):
    """Return lines that evaluate observables for the given equations."""
    if cse:
        equations = cse_and_stack(equations.copy())
    else:
        equations = topological_sort(equations.copy())
    out_subs = dict(zip(index_map.dxdt.ref_map.keys(),sp.numbered_symbols(
            "dxout_", start=1)))
    equations = [(lhs.subs(out_subs), rhs.subs(out_subs)) for lhs, rhs in
                 equations]

    arrayrefs = index_map.all_arrayrefs
    equations = [(lhs.subs(arrayrefs), rhs.subs(arrayrefs)) for lhs, rhs in
                 equations]
    equations = _prune_unused_assignments(equations, "observables")
    obs_lines = print_cuda_multiple(
        equations, symbol_map=index_map.all_arrayrefs
    )
    if not obs_lines:
        obs_lines = ["pass"]
    return obs_lines

def generate_dxdt_fac_code(
    equations: Iterable[Tuple[sp.Symbol, sp.Expr]],
    index_map: Optional[IndexedBases] = None,
    func_name: str = "dxdt_factory",
    cse: bool = True,
) -> str:
    """Return source for a ``dx/dt`` factory.

    The emitted factory has signature ``func(constants, precision)`` where
    ``constants`` is a mapping from constant names to numeric values. Each
    constant is embedded as a separate variable in the generated device
    function.
    """
    dxdt_lines = generate_dxdt_lines(equations, index_map=index_map, cse=cse)
    const_lines = [
        f"    {name} = precision(constants['{name}'])"
        for name in index_map.constants.symbol_map
    ]
    const_block = "\n".join(const_lines) + ("\n" if const_lines else "")

    code = DXDT_TEMPLATE.format(
        func_name=func_name,
        const_lines=const_block,
        body="    " + "\n        ".join(dxdt_lines),
    )
    return code


def generate_observables_fac_code(
    equations: Iterable[Tuple[sp.Symbol, sp.Expr]],
    index_map: IndexedBases,
    func_name: str = "observables",
    cse: bool = True,
) -> str:
    """Return source for an observables-only factory."""

    obs_lines = generate_observables_lines(
        equations, index_map=index_map, cse=cse
    )
    const_lines = [
        f"    {name} = precision(constants['{name}'])"
        for name in index_map.constants.symbol_map
    ]
    const_block = "\n".join(const_lines) + ("\n" if const_lines else "")

    code = OBSERVABLES_TEMPLATE.format(
        func_name=func_name,
        const_lines=const_block,
        body="    " + "\n        ".join(obs_lines),
    )
    return code


