from typing import Iterable, Optional, Tuple

import sympy as sp

from cubie.systemmodels.symbolic.numba_cuda_printer import print_cuda_multiple
from cubie.systemmodels.symbolic.parser import IndexedBases
from cubie.systemmodels.symbolic.sym_utils import (
    cse_and_stack,
    topological_sort,
)

DXDT_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED DXDT FACTORY\n"
    "def {func_name}(constants, precision):\n"
    '    """Auto-generated dxdt factory."""\n'
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

RESIDUAL_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED RESIDUAL FACTORY\n"
    "def {func_name}(constants, precision):\n"
    '    """Auto-generated residual factory."""\n'
    "    @cuda.jit((precision[:],\n"
    "               precision[:],\n"
    "               precision[:],\n"
    "               precision[:]),\n"
    "              device=True,\n"
    "              inline=True)\n"
    "    def residual(state, parameters, drivers, out):\n"
    "    {body}\n"
    "    \n"
    "    return out\n"
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

def generate_dxdt_fac_code(equations: Iterable[Tuple[sp.Symbol, sp.Expr]],
                           index_map: Optional[IndexedBases] = None,
                           func_name="dxdt_factory",
                           cse=True):
    """Return the dxdt factory function for the given equations."""
    dxdt_lines = generate_dxdt_lines(equations, index_map=index_map, cse=cse)

    code = DXDT_TEMPLATE.format(
        func_name=func_name,
        body="    " + "\n        ".join(dxdt_lines)
    )
    return code




def generate_residual_code(
    equations: Iterable[Tuple[sp.Symbol, sp.Expr]],
    index_map: IndexedBases,
    func_name: str = "residual_factory",
    cse: bool = True,
):
    """Return a residual factory for the provided equations.

    Implementation note: reuse generate_dxdt_lines to avoid duplicating symbolic-to-CUDA
    printing logic. We post-process the resulting lines to:
      - write into `out[...]` instead of `dxdt[...]`
      - replace any references to observables (e.g., "observables[i]") with
        placeholder symbol names so that the generated code does not reference the
        observables array directly.
    """
    # Generate dxdt code lines first
    dxdt_lines = generate_dxdt_lines(equations, index_map=index_map, cse=cse)

    # Replace LHS target from dxdt[...] to out[...]
    res_lines = [line.replace("dxdt[", "out[") for line in dxdt_lines]

    # Replace any observable array references with placeholder symbol names
    # e.g., replace "observables[3]" with the corresponding symbol name
    try:
        observable_ref_map = index_map.observables.ref_map
    except AttributeError:
        observable_ref_map = {}

    if observable_ref_map:
        for sym, ref in observable_ref_map.items():
            placeholder = str(sym)
            res_lines = [ln.replace(ref, placeholder) for ln in res_lines]

    if not res_lines:
        res_lines = ["pass"]

    code = RESIDUAL_TEMPLATE.format(
        func_name=func_name, body="    " + "\n        ".join(res_lines)
    )
    return code