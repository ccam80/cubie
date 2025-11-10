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
    raw_states = list(model.get_state_variables())
    raw_derivatives = list(model.get_derivatives())
    
    # Convert Dummy symbols to regular Symbols
    # cellmlmanip returns Dummy symbols but we need regular Symbols
    dummy_to_symbol = {}
    for dummy_state in raw_states:
        if isinstance(dummy_state, sp.Dummy):
            symbol = sp.Symbol(dummy_state.name)
            dummy_to_symbol[dummy_state] = symbol
    
    states = [dummy_to_symbol.get(s, s) for s in raw_states]
    
    # Filter equations and substitute Dummy with Symbol
    equations = []
    for eq in model.equations:
        if eq.lhs in raw_derivatives:
            # Substitute all Dummy symbols with regular Symbols
            eq_substituted = eq.subs(dummy_to_symbol)
            equations.append(eq_substituted)
    
    return states, equations
