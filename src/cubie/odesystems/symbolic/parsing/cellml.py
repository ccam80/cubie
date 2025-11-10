"""Minimal CellML parsing helpers using ``cellmlmanip``.

This module provides functionality to import CellML models into CuBIE's
symbolic ODE framework. It wraps the cellmlmanip library to extract
state variables and differential equations in a format compatible with
SymbolicODE.

The implementation is inspired by
:mod:`chaste_codegen.model_with_conversions` from the chaste-codegen
project (MIT licence). Only a minimal subset required for basic model
loading is implemented here.

Examples
--------
Basic CellML model loading workflow:

>>> from cubie.odesystems.symbolic.parsing.cellml import (
...     load_cellml_model
... )
>>> import sympy as sp
>>> 
>>> # Load a CellML model file
>>> states, equations = load_cellml_model("cardiac_model.cellml")
>>> 
>>> # Inspect the extracted data
>>> print(f"Found {len(states)} state variables")
>>> print(f"State names: {[s.name for s in states]}")
>>> 
>>> # Verify equation format
>>> for eq in equations:
...     assert isinstance(eq.lhs, sp.Derivative)
...     assert isinstance(eq.rhs, sp.Expr)

Notes
-----
The cellmlmanip dependency is optional. Install with:

    pip install cellmlmanip

CellML models can be obtained from the Physiome Model Repository:
https://models.physiomeproject.org/

See Also
--------
load_cellml_model : Main function for loading CellML files
"""

try:  # pragma: no cover - optional dependency
    import cellmlmanip  # type: ignore
except Exception:  # pragma: no cover
    cellmlmanip = None  # type: ignore

import sympy as sp
from pathlib import Path


def load_cellml_model(path: str) -> tuple[list[sp.Symbol], list[sp.Eq]]:
    """Load a CellML model and extract states and derivatives.

    This function uses the cellmlmanip library to parse CellML files
    and extract the state variables and differential equations in a
    format compatible with CuBIE's SymbolicODE system.

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
        If cellmlmanip is not installed. Install with:
        pip install cellmlmanip
    TypeError
        If path is not a string.
    FileNotFoundError
        If the specified CellML file does not exist.
    ValueError
        If the file does not have .cellml extension.

    Examples
    --------
    Load a CellML model and verify structure:

    >>> states, equations = load_cellml_model("model.cellml")
    >>> len(states)  # Number of state variables
    8
    >>> isinstance(states[0], sp.Symbol)
    True

    Notes
    -----
    - Only differential equations are extracted (algebraic equations
      filtered)
    - State variables are converted from sympy.Dummy to sympy.Symbol
    - Supports CellML 1.0 and 1.1 formats
    - CellML models from Physiome repository are compatible
    - The cellmlmanip library handles the complex CellML XML parsing
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
