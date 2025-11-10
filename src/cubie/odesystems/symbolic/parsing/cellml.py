"""Minimal CellML parsing helpers using ``cellmlmanip``.

This wrapper is heavily inspired by
:mod:`chaste_codegen.model_with_conversions` from the chaste-codegen project
(MIT licence). Only a tiny subset required for basic model loading is
implemented here.
"""

try:  # pragma: no cover - optional dependency
    import cellmlmanip  # type: ignore
except Exception:  # pragma: no cover
    cellmlmanip = None  # type: ignore

import sympy as sp
from pathlib import Path


def load_cellml_model(path: str) -> tuple[list[sp.Symbol], list[sp.Eq]]:
    """Load a CellML model and extract states and derivatives.

    Parameters
    ----------
    path : str
        Filesystem path to the CellML source file. Must have .cellml
        extension and be a valid CellML 1.0 or 1.1 model file.

    Returns
    -------
    states : list[sympy.Symbol]
        List of sympy.Symbol objects representing state variables.
    equations : list[sympy.Eq]
        List of sympy.Eq objects with derivatives on LHS and RHS
        expressions containing state variables.

    Raises
    ------
    ImportError
        If cellmlmanip is not installed.
    TypeError
        If path is not a string.
    FileNotFoundError
        If the specified CellML file does not exist.
    ValueError
        If the file does not have .cellml extension.
    """
    if cellmlmanip is None:  # pragma: no cover
        raise ImportError("cellmlmanip is required for CellML parsing")
    
    # Validate input type
    if not isinstance(path, str):
        raise TypeError(
            f"path must be a string, got {type(path).__name__}"
        )
    
    # Validate file existence
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"CellML file not found: {path}")
    
    # Validate file extension
    if not path.endswith('.cellml'):
        raise ValueError(
            f"File must have .cellml extension, got: {path}"
        )
    
    model = cellmlmanip.load_model(path)
    raw_states = list(model.get_state_variables())
    raw_derivatives = list(model.get_derivatives())
    
    # Convert Dummy symbols to regular Symbols
    # cellmlmanip returns Dummy symbols but we need regular Symbols
    states = []
    dummy_to_symbol = {}
    for raw_state in raw_states:
        if isinstance(raw_state, sp.Dummy):
            symbol = sp.Symbol(raw_state.name)
            dummy_to_symbol[raw_state] = symbol
            states.append(symbol)
        else:
            states.append(raw_state)
    
    # Filter derivative equations and substitute symbols
    equations = [
        eq.subs(dummy_to_symbol)
        for eq in model.equations
        if eq.lhs in raw_derivatives
    ]
    
    return states, equations
