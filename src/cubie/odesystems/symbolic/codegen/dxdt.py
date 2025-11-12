"""Utilities that emit CUDA ``dx/dt`` factories from SymPy expressions."""

from typing import Optional

import sympy as sp

from cubie.odesystems.symbolic.codegen import print_cuda_multiple
from cubie.odesystems.symbolic.codegen.jacobian import _prune_unused_assignments
from cubie.odesystems.symbolic.parsing import IndexedBases, ParsedEquations
from cubie.odesystems.symbolic.sym_utils import (
    cse_and_stack,
    render_constant_assignments,
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
    "               precision[:],\n"
    "               precision),\n"
    "              device=True,\n"
    "              inline=True)\n"
    "    def dxdt(state, parameters, drivers, observables, out, t):\n"
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
    "               precision[:],\n"
    "               precision),\n"
    "              device=True,\n"
    "              inline=True)\n"
    "    def get_observables(state, parameters, drivers, observables, t):\n"
    "    {body}\n"
    "    \n"
    "    return get_observables\n"
)


def generate_dxdt_lines(
    equations: ParsedEquations,
    index_map: Optional[IndexedBases] = None,
    cse: bool = True,
) -> list[str]:
    """Generate CUDA assignment statements for ``dx/dt`` updates.

    Parameters
    ----------
    equations
        Parsed equations describing ``dx/dt`` assignments.
    index_map
        Indexed bases that supply CUDA array references for each symbol.
    cse
        Whether to apply common subexpression elimination before emission.

    Returns
    -------
    list of str
        CUDA source lines that evaluate the ``dx/dt`` equations.

    Notes
    -----
    ``index_map`` must expose ``all_arrayrefs`` containing each symbol in
    ``equations``.
    """

    working_equations = equations.non_observable_equations()

    if cse:
        processed = cse_and_stack(working_equations)
    else:
        processed = topological_sort(working_equations)

    symbol_map = None
    if index_map is not None:
        observable_symbols = set(index_map.observables.ref_map.keys())
        processed = [
            (lhs, rhs)
            for lhs, rhs in processed
            if lhs not in observable_symbols
        ]
        processed = _prune_unused_assignments(
            processed,
            output_symbols=index_map.dxdt.ref_map.keys(),
        )
        symbol_map = index_map.all_arrayrefs

    dxdt_lines = print_cuda_multiple(processed, symbol_map=symbol_map)
    if not dxdt_lines:
        dxdt_lines = ["pass"]
    return dxdt_lines


def generate_observables_lines(
    equations: ParsedEquations,
    index_map: IndexedBases,
    cse: bool = True,
) -> list[str]:
    """Generate CUDA source for observable calculations.

    Parameters
    ----------
    equations
        Parsed equations describing observable assignments.
    index_map
        Indexed bases used to substitute CUDA array references.
    cse
        Whether to apply common subexpression elimination before emission.

    Returns
    -------
    list of str
        CUDA source lines that compute the observables.

    Notes
    -----
    ``equations`` should support ``copy`` to avoid mutating the caller's
    expression list when applying substitutions.
    """
    working_equations = list(equations.observable_system)

    if cse:
        processed = cse_and_stack(working_equations)
    else:
        processed = topological_sort(working_equations)
    out_subs = dict(
        zip(
            index_map.dxdt.ref_map.keys(),
            sp.numbered_symbols("dxout_", start=1),
        )
    )
    substituted = [
        (lhs.subs(out_subs), rhs.subs(out_subs))
        for lhs, rhs in processed
    ]

    arrayrefs = index_map.all_arrayrefs
    substituted = [
        (lhs.subs(arrayrefs), rhs.subs(arrayrefs))
        for lhs, rhs in substituted
    ]
    substituted = _prune_unused_assignments(substituted, "observables")
    obs_lines = print_cuda_multiple(
        substituted, symbol_map=index_map.all_arrayrefs
    )
    if not obs_lines:
        obs_lines = ["pass"]
    return obs_lines

def generate_dxdt_fac_code(
    equations: ParsedEquations,
    index_map: Optional[IndexedBases] = None,
    func_name: str = "dxdt_factory",
    cse: bool = True,
) -> str:
    """Emit Python source for a ``dx/dt`` CUDA factory.

    Parameters
    ----------
    equations
        Parsed equations describing ``dx/dt`` assignments.
    index_map
        Indexed bases that provide both symbol references and constants.
    func_name
        Name of the generated factory function.
    cse
        Whether to apply common subexpression elimination before emission.

    Returns
    -------
    str
        Python source code implementing the requested factory.

    Notes
    -----
    The generated factory expects ``func(constants, precision)`` and returns a
    CUDA device function compiled with :func:`numba.cuda.jit`.
    """
    dxdt_lines = generate_dxdt_lines(
        equations, index_map=index_map, cse=cse
    )
    const_block = render_constant_assignments(
        index_map.constants.symbol_map
    )

    code = DXDT_TEMPLATE.format(
        func_name=func_name,
        const_lines=const_block,
        body="    " + "\n        ".join(dxdt_lines),
    )
    return code


def generate_observables_fac_code(
    equations: ParsedEquations,
    index_map: IndexedBases,
    func_name: str = "observables",
    cse: bool = True,
) -> str:
    """Emit Python source for an observables CUDA factory.

    Parameters
    ----------
    equations
        Parsed equations describing observable assignments.
    index_map
        Indexed bases that provide symbol and constant lookups.
    func_name
        Name of the generated factory function.
    cse
        Whether to apply common subexpression elimination before emission.

    Returns
    -------
    str
        Python source code implementing the requested factory.
    """

    obs_lines = generate_observables_lines(
        equations, index_map=index_map, cse=cse
    )
    const_block = render_constant_assignments(
        index_map.constants.symbol_map
    )

    code = OBSERVABLES_TEMPLATE.format(
        func_name=func_name,
        const_lines=const_block,
        body="    " + "\n        ".join(obs_lines),
    )
    return code


