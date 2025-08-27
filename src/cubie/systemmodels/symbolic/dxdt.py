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
    "    def dxdt(state, parameters, driver, observables, dxdt):\n"
    "    {body}\n"
    "    \n"
    "    return dxdt\n"
)

def generate_dxdt_fac_code(equations: Iterable[Tuple[sp.Symbol, sp.Expr]],
                           index_map: Optional[IndexedBases] = None,
                           func_name="dxdt_factory",
                           cse=True):
    """Return the dxdt factory function for the given equations."""
    if cse:
        equations = cse_and_stack(equations)
    else:
        equations = topological_sort(equations)

    dxdt_lines = print_cuda_multiple(equations,
                                     symbol_map=index_map.all_arrayrefs)
    if not dxdt_lines:
        dxdt_lines = ["pass"]
    code = DXDT_TEMPLATE.format(func_name=func_name,
                                body="    " + "\n        ".join(dxdt_lines))
    return code