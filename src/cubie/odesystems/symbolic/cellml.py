"""Minimal CellML parsing helpers using ``cellmlmanip``.

This wrapper is heavily inspired by :mod:`chaste_codegen.model_with_conversions`
from the chaste-codegen project (MIT licence).  Only a tiny subset required for
basic model loading is implemented here.
"""

from __future__ import annotations



try:  # pragma: no cover - optional dependency
    import cellmlmanip  # type: ignore
except Exception:  # pragma: no cover
    cellmlmanip = None  # type: ignore


def load_cellml_model(path: str):
    """Load a CellML model and return states and derivative equations.

    Parameters
    ----------
    path: str
        Path to the CellML file.

    Returns
    -------
    Tuple[List[sympy.Symbol], List[sympy.Eq]]
    """
    if cellmlmanip is None:  # pragma: no cover
        raise ImportError("cellmlmanip is required for CellML parsing")
    model = cellmlmanip.load_model(path)
    states = list(model.get_state_variables())
    derivatives = list(model.get_derivatives())
    equations = [eq for eq in model.equations if eq.lhs in derivatives]
    return states, equations