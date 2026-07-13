"""Structural simplification and tearing for symbolic ODE/DAE systems.

Ports ModelingToolkit.jl's ``mtkcompile`` pipeline (alias elimination,
integer-linear singularity removal, Pantelides index reduction, dummy
derivative state selection, and Carpanzano/Modia tearing) to cubie's
SymPy-based symbolic system representation.

The algorithm core follows the factored SciML layout:

- :mod:`cubie.odesystems.symbolic.structural.bipartite` and
  :mod:`~cubie.odesystems.symbolic.structural.digraph` port
  BipartiteGraphs.jl,
- :mod:`~cubie.odesystems.symbolic.structural.clil` ports
  StateSelection.jl's exact integer linear algebra,
- the pass modules port StateSelection.jl's decision procedures,
- :mod:`~cubie.odesystems.symbolic.structural.system_structure` and
  :mod:`~cubie.odesystems.symbolic.structural.reassemble` port
  ModelingToolkitTearing's symbolic bridge.

The user-facing entry point is
:func:`~cubie.odesystems.symbolic.structural.simplify.structural_simplify`.
"""

from cubie.odesystems.symbolic.structural.simplify import (
    structural_simplify,
    SimplifiedSystem,
)
from cubie.odesystems.symbolic.structural.system_structure import (
    StructuralState,
)
from cubie.odesystems.symbolic.structural.errors import (
    ExtraEquationsSystemError,
    ExtraVariablesSystemError,
    InvalidSystemError,
)

__all__ = [
    "structural_simplify",
    "SimplifiedSystem",
    "StructuralState",
    "ExtraEquationsSystemError",
    "ExtraVariablesSystemError",
    "InvalidSystemError",
]
