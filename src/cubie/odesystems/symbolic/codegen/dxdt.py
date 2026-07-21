"""Emit CUDA ``dxdt`` and observables factory code from parsed equations.

Published Functions
-------------------
:func:`generate_dxdt_fac_code`
    Return a string containing the ``dxdt_factory`` function definition
    ready for disk caching and import.

:func:`generate_observables_fac_code`
    Return a string containing the ``observables_factory`` function
    definition.

See Also
--------
:class:`~cubie.odesystems.symbolic.symbolicODE.SymbolicODE`
    Calls these generators inside :meth:`SymbolicODE.build`.
:mod:`cubie.odesystems.symbolic.engine`
    Expression engine used for manipulation and printing.
:class:`~cubie.odesystems.symbolic.odefile.ODEFile`
    Disk cache that stores and imports the generated code.
"""

from typing import Optional

from cubie.odesystems.symbolic.engine import expr as ir
from cubie.odesystems.symbolic.engine.adapter import system_ir
from cubie.odesystems.symbolic.engine.assignments import (
    cse_and_stack,
    prune_unused,
    topological_sort,
)
from cubie.odesystems.symbolic.engine.printer import (
    print_cuda_multiple,
)
from cubie.odesystems.symbolic.parsing import IndexedBases, ParsedEquations
from cubie.odesystems.symbolic.sym_utils import (
    render_constant_assignments,
)
from cubie.time_logger import default_timelogger

# Register timing events for codegen functions
# Module-level registration required since codegen functions return code
# strings rather than cacheable objects that could auto-register
default_timelogger.register_event("codegen_generate_dxdt_fac_code", "codegen",
                                   "Codegen time for generate_dxdt_fac_code")
default_timelogger.register_event("codegen_generate_observables_fac_code",
                                   "codegen",
                                   "Codegen time for generate_observables_fac_code")

DXDT_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED DXDT FACTORY\n"
    "def {func_name}(constants, precision, lineinfo=None):\n"
    '    """Auto-generated dxdt factory."""\n'
    "{const_lines}"
    "    \n"
    "    @cuda.jit(\n"
    "        # (precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision),\n"
    "        device=True,\n"
    "        inline=True,\n"
    "        **get_jit_kwargs(lineinfo))\n"
    "    def dxdt(state, parameters, drivers, observables, out, t):\n"
    "    {body}\n"
    "    \n"
    "    return dxdt\n"
)

OBSERVABLES_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED OBSERVABLES FACTORY\n"
    "def {func_name}(constants, precision, lineinfo=None):\n"
    '    """Auto-generated observables factory."""\n'
    "{const_lines}"
    "    @cuda.jit(\n"
    "        # (precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision),\n"
    "        device=True,\n"
    "        inline=True,\n"
    "        **get_jit_kwargs(lineinfo))\n"
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
    """
    sysir = system_ir(equations, index_map)
    working_equations = sysir.non_observable_equations()

    if cse:
        processed = cse_and_stack(working_equations)
    else:
        processed = topological_sort(working_equations)

    observable_symbols = sysir.observable_set
    processed = [
        (lhs, rhs)
        for lhs, rhs in processed
        if lhs not in observable_symbols
    ]
    processed = prune_unused(
        processed, output_symbols=sysir.dxdt_symbols
    )

    dxdt_lines = print_cuda_multiple(
        processed,
        symbol_map=sysir.arrayrefs,
        constant_names=sysir.constant_names,
        function_aliases=sysir.function_aliases,
    )
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
    """
    # Early return if no observables
    if not index_map.observables.ref_map:
        return ["pass"]

    sysir = system_ir(equations, index_map)
    working_equations = list(sysir.equations)

    if cse:
        processed = cse_and_stack(working_equations)
    else:
        processed = topological_sort(working_equations)

    # dx/dt outputs are not written by the observables kernel; route
    # them to throwaway locals instead of the out array.
    out_subs = {
        dx_sym: ir.sym(f"_cubie_codegen_dxout_{position + 1}")
        for position, dx_sym in enumerate(sysir.dxdt_symbols)
    }
    memo: dict = {}
    substituted = [
        (
            ir.xreplace(lhs, out_subs, memo),
            ir.xreplace(rhs, out_subs, memo),
        )
        for lhs, rhs in processed
    ]

    observable_targets = [
        sysir.arrayrefs[obs.name] for obs in sysir.observable_symbols
    ]
    substituted = prune_unused(
        substituted,
        output_symbols=list(sysir.observable_symbols)
        + observable_targets,
    )
    obs_lines = print_cuda_multiple(
        substituted,
        symbol_map=sysir.arrayrefs,
        constant_names=sysir.constant_names,
        function_aliases=sysir.function_aliases,
    )
    assert obs_lines, "internal error: codegen produced an empty body"
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
    default_timelogger.start_event("codegen_generate_dxdt_fac_code")
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
    default_timelogger.stop_event("codegen_generate_dxdt_fac_code")
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
    default_timelogger.start_event("codegen_generate_observables_fac_code")

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
    default_timelogger.stop_event("codegen_generate_observables_fac_code")
    return code
