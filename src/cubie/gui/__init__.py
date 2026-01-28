"""Qt-based GUI utilities for CuBIE ODE systems.

This module provides graphical editors for managing constants, parameters,
and initial state values in SymbolicODE systems. The GUIs are designed to
be loosely coupled and simply use the public API of SymbolicODE.

Requires one of: PyQt6, PyQt5, PySide6, or PySide2.

Example
-------
>>> from cubie.odesystems.symbolic import load_cellml_model
>>> from cubie.gui import ConstantsEditor, StatesEditor
>>>
>>> ode = load_cellml_model("model.cellml")
>>> editor = ConstantsEditor(ode)
>>> editor.show()
"""

from cubie.gui.constants_editor import ConstantsEditor
from cubie.gui.states_editor import StatesEditor

__all__ = ["ConstantsEditor", "StatesEditor"]
