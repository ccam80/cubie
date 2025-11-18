"""Minimal CellML parsing helpers using ``cellmlmanip``.

This module provides functionality to import CellML models into CuBIE's
symbolic ODE framework. It wraps the cellmlmanip library to load
CellML files and convert them directly into SymbolicODE objects.

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
>>> 
>>> # Load a CellML model file - returns initialized SymbolicODE
>>> ode_system = load_cellml_model("cardiac_model.cellml")
>>> 
>>> # The model is ready to use with solve_ivp
>>> print(f"Model has {ode_system.num_states} states")
>>> print(f"Model has {len(ode_system.indices.observables)} observables")

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
    from cellmlmanip.model import Quantity  # type: ignore
except Exception:  # pragma: no cover
    cellmlmanip = None  # type: ignore
    Quantity = None  # type: ignore

import sympy as sp
from pathlib import Path
import numpy as np
from typing import Optional, List
import re

from cubie._utils import PrecisionDType
from cubie.time_logger import _default_timelogger

# Register timing events for cellml import functions
# Module-level registration required for proper event tracking
_default_timelogger.register_event(
    "codegen_cellml_load_model", "codegen",
    "Codegen time for cellmlmanip.load_model()"
)
_default_timelogger.register_event(
    "codegen_cellml_symbol_conversion", "codegen",
    "Codegen time for converting Dummy symbols to Symbols"
)
_default_timelogger.register_event(
    "codegen_cellml_equation_processing", "codegen",
    "Codegen time for processing differential and algebraic equations"
)
_default_timelogger.register_event(
    "codegen_cellml_sympy_preparation", "codegen",
    "Codegen time for preparing SymPy equations for parser"
)


def _sanitize_symbol_name(name: str) -> str:
    """Sanitize CellML symbol names for Python identifiers.
    
    CellML uses $ for namespacing and allows names starting with _
    followed by numbers. We need to convert these to valid Python
    identifiers.
    """
    # Replace $ with _
    name = name.replace('$', '_')
    
    # Replace . with _
    name = name.replace('.', '_')
    
    # If name starts with _, check if next char is a digit
    # If so, prepend with 'var_' to make it valid
    if name.startswith('_') and len(name) > 1 and name[1].isdigit():
        name = 'var' + name
    
    # Ensure name doesn't start with a digit
    if name and name[0].isdigit():
        name = 'var_' + name
    
    # Replace any remaining invalid characters with _
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    
    return name


def load_cellml_model(
    path: str,
    precision: PrecisionDType = np.float32,
    name: Optional[str] = None,
    parameters: Optional[List[str]] = None,
    observables: Optional[List[str]] = None,
):
    """Load a CellML model and return an initialized SymbolicODE system.

    This function uses the cellmlmanip library to parse CellML files
    and converts them into a ready-to-use SymbolicODE system with all
    differential equations and algebraic constraints properly configured.

    Parameters
    ----------
    path : str
        Filesystem path to the CellML source file. Must have .cellml
        extension and be a valid CellML 1.0 or 1.1 model file.
    precision : numpy dtype, optional
        Target floating-point precision for compiled kernels.
        Default is np.float32.
    name : str, optional
        Identifier for the generated system. If None, uses the
        filename without extension.
    parameters : list of str, optional
        List of symbol names to assign as parameters. Otherwise,
        these symbols become constants or anonymous auxiliaries.
    observables : list of str, optional
        List of symbol names to assign as observables. Otherwise,
        these symbols become anonymous auxiliaries.

    Returns
    -------
    SymbolicODE
        Fully initialized ODE system ready for use with solve_ivp.
        State variables are configured with initial values from the
        CellML model, and algebraic equations are set up according
        to the parameters and observables specifications.

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
    Load a CellML model and run a simulation:

    >>> from cubie import load_cellml_model, solve_ivp
    >>> import numpy as np
    >>> 
    >>> # Load the model
    >>> ode_system = load_cellml_model("beeler_reuter_model_1977.cellml")
    >>> 
    >>> # Set up simulation
    >>> t_span = (0.0, 100.0)
    >>> initial_states = np.ones(ode_system.num_states, dtype=np.float32)
    >>> 
    >>> # Run simulation
    >>> result = solve_ivp(ode_system, t_span, initial_states)

    Notes
    -----
    - Differential equations become state equations in the ODE system
    - Algebraic equations become observables or anonymous auxiliaries
    - State variables are converted from sympy.Dummy to sympy.Symbol
    - Initial values from CellML are preserved in the ODE system
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
    
    # Use filename as default name if not provided
    if name is None:
        name = path_obj.stem
    
    _default_timelogger.start_event("codegen_cellml_load_model")
    model = cellmlmanip.load_model(path)
    raw_states = list(model.get_state_variables())
    raw_derivatives = list(model.get_derivatives())
    _default_timelogger.stop_event("codegen_cellml_load_model")
    
    # Extract initial values and units from CellML model
    initial_values = {}
    state_units = {}
    
    _default_timelogger.start_event("codegen_cellml_symbol_conversion")
    # Convert Dummy symbols to regular Symbols with sanitized names
    # cellmlmanip returns Dummy symbols but we need regular Symbols
    states = []
    dummy_to_symbol = {}
    for raw_state in raw_states:
        clean_name = _sanitize_symbol_name(raw_state.name)
        symbol = sp.Symbol(clean_name)
        dummy_to_symbol[raw_state] = symbol
        states.append(symbol)
        
        # Get initial value if available
        if hasattr(raw_state, 'initial_value') and raw_state.initial_value is not None:
            initial_values[clean_name] = float(raw_state.initial_value)
        
        # Get units if available
        if hasattr(raw_state, 'units'):
            state_units[clean_name] = str(raw_state.units)
        else:
            state_units[clean_name] = "dimensionless"
    
    # Collect units for all other symbols
    all_symbol_units = {}
    
    # Also convert any other Dummy symbols in the model equations
    # Special handling for numeric quantities (e.g., _0.5, _1.0, _3)
    for eq in model.equations:
        for atom in eq.atoms(sp.Dummy):
            if atom not in dummy_to_symbol:
                clean_name = _sanitize_symbol_name(atom.name)
                
                # Check if this is a numeric quantity (name starts with _)
                if atom.name.startswith('_'):
                    try:
                        # Try to parse as a float
                        value = float(atom.name[1:])
                        # Use Integer for whole numbers, Float for decimals
                        if value == int(value):
                            dummy_to_symbol[atom] = sp.Integer(int(value))
                        else:
                            dummy_to_symbol[atom] = sp.Float(value)
                        continue
                    except (ValueError, IndexError):
                        # Not a numeric value, treat as regular symbol
                        pass
                
                # Regular symbol conversion
                dummy_to_symbol[atom] = sp.Symbol(clean_name)
                
                # Extract units for this symbol
                if hasattr(atom, 'units'):
                    all_symbol_units[clean_name] = str(atom.units)
                else:
                    all_symbol_units[clean_name] = "dimensionless"
    _default_timelogger.stop_event("codegen_cellml_symbol_conversion")
    
    _default_timelogger.start_event("codegen_cellml_equation_processing")
    # Filter differential equations and algebraic equations separately
    differential_equations = []
    algebraic_equations = []
    
    for eq in model.equations:
        eq_substituted = eq.subs(dummy_to_symbol)
        if eq.lhs in raw_derivatives:
            differential_equations.append(eq_substituted)
        else:
            algebraic_equations.append(eq_substituted)
    _default_timelogger.stop_event("codegen_cellml_equation_processing")
    
    _default_timelogger.start_event("codegen_cellml_sympy_preparation")
    
    dxdt_equations = []
    for eq in differential_equations:
        state_var = eq.lhs.args[0]
        lhs_sym = sp.Symbol(f"d{state_var.name}", real=True)
        dxdt_equations.append((lhs_sym, eq.rhs))
    
    constants_dict = {}
    parameters_dict = {}
    algebraic_equation_tuples = []
    observable_units = {}
    
    if parameters is None:
        parameters_set = set()
    elif isinstance(parameters, dict):
        parameters_set = set(parameters.keys())
    else:
        parameters_set = set(parameters)
    
    for eq in algebraic_equations:
        if isinstance(eq.rhs, sp.Number):
            var_name = str(eq.lhs)
            var_value = float(eq.rhs)
            
            if var_name in parameters_set:
                parameters_dict[var_name] = var_value
            else:
                constants_dict[var_name] = var_value
        else:
            algebraic_equation_tuples.append((eq.lhs, eq.rhs))
            
            lhs_name = str(eq.lhs)
            if lhs_name in all_symbol_units:
                observable_units[lhs_name] = all_symbol_units[lhs_name]
    
    all_equations = dxdt_equations + algebraic_equation_tuples
    
    parameter_units = {}
    if parameters:
        for param in parameters:
            if param in all_symbol_units:
                parameter_units[param] = all_symbol_units[param]
    
    if observables:
        for obs in observables:
            if obs not in observable_units and obs in all_symbol_units:
                observable_units[obs] = all_symbol_units[obs]
    
    if parameters is not None and isinstance(parameters, dict):
        parameters_dict = {**parameters_dict, **parameters}
    
    _default_timelogger.stop_event("codegen_cellml_sympy_preparation")
    
    from cubie.odesystems.symbolic.symbolicODE import SymbolicODE
    
    return SymbolicODE.create(
        dxdt=all_equations,
        states=initial_values if initial_values else None,
        parameters=parameters_dict if parameters_dict else None,
        constants=constants_dict if constants_dict else None,
        observables=observables,
        name=name,
        precision=precision,
        strict=False,
        state_units=state_units if state_units else None,
        parameter_units=parameter_units if parameter_units else None,
        observable_units=observable_units if observable_units else None,
    )
