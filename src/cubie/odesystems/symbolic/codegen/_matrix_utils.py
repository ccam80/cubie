"""Mass-matrix conversion shared by the codegen builders.

Published Functions
-------------------
:func:`mass_matrix_ir`
    Normalise a mass matrix (``None``, SymPy matrix, NumPy array, or
    nested sequences) into row-major IR entries, defaulting to the
    identity.
"""

from typing import List, Optional

from cubie.odesystems.symbolic.engine import expr as ir
from cubie.odesystems.symbolic.engine.from_sympy import from_sympy

__all__ = ["mass_matrix_ir"]


def _entry_to_ir(entry) -> ir.Expr:
    """Convert one matrix entry to an IR expression."""
    if isinstance(entry, ir.Expr):
        return entry
    if isinstance(entry, (int, float)):
        return ir.num(entry)
    # NumPy scalars expose item(); SymPy scalars convert directly.
    item = getattr(entry, "item", None)
    if item is not None:
        return ir.num(item())
    return from_sympy(entry)


def mass_matrix_ir(M, n: int) -> List[List[ir.Expr]]:
    """Return the mass matrix as row-major IR entries.

    Parameters
    ----------
    M
        Mass matrix as ``None`` (identity), a SymPy matrix, a NumPy
        array, or nested sequences.
    n
        State dimension used for the identity default.

    Returns
    -------
    list of list
        Row-major IR entries.
    """
    if M is None:
        return [
            [ir.ONE if i == j else ir.ZERO for j in range(n)]
            for i in range(n)
        ]
    tolist = getattr(M, "tolist", None)
    rows = tolist() if tolist is not None else [list(row) for row in M]
    return [[_entry_to_ir(entry) for entry in row] for row in rows]
