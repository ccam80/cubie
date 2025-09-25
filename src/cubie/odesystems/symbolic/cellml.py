"""Minimal CellML parsing helpers using ``cellmlmanip``.

This wrapper is heavily inspired by
:mod:`chaste_codegen.model_with_conversions` from the chaste-codegen project
(MIT licence). Only a tiny subset required for basic model loading is
implemented here.
"""

from __future__ import annotations

try:  # pragma: no cover - optional dependency
    import cellmlmanip  # type: ignore
except Exception:  # pragma: no cover
    cellmlmanip = None  # type: ignore

import sympy as sp


def load_cellml_model(path: str) -> tuple[list[sp.Symbol], list[sp.Eq]]:
    """Load a CellML model and extract states and derivatives.

    Parameters
    ----------
    path
        Filesystem path to the CellML source file.

    Returns
    -------
    tuple[list[sympy.Symbol], list[sympy.Eq]]
        States and differential equations defined by the model.
    """
    if cellmlmanip is None:  # pragma: no cover
        raise ImportError("cellmlmanip is required for CellML parsing")
    model = cellmlmanip.load_model(path)
    states = list(model.get_state_variables())
    derivatives = list(model.get_derivatives())
    equations = [eq for eq in model.equations if eq.lhs in derivatives]
    return states, equations
