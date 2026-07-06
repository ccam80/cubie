"""Qt-based GUI utilities for CuBIE ODE systems.

Graphical editors for managing constants, parameters, and initial
state values in :class:`~cubie.odesystems.symbolic.SymbolicODE`
systems.  The editors use the public API of ``SymbolicODE`` and are
loosely coupled from the rest of the library.

Requires one of: PyQt6, PyQt5, PySide6, or PySide2 (via ``qtpy``).

Published Classes
-----------------
:class:`ConstantsEditor`
    Dialog for viewing and editing constants and parameters.

    >>> from cubie.gui import ConstantsEditor
    >>> editor = ConstantsEditor(ode)
    >>> editor.exec()

:class:`StatesEditor`
    Dialog for viewing and editing initial state values.

    >>> from cubie.gui import StatesEditor
    >>> editor = StatesEditor(ode)
    >>> editor.exec()

See Also
--------
:mod:`cubie.gui.constants_editor`
    Constants/parameters editor and pre-parse editor.
:mod:`cubie.gui.states_editor`
    Initial-states editor.
:class:`~cubie.odesystems.symbolic.SymbolicODE`
    ODE system class consumed by the editors.
"""

from cubie.gui.constants_editor import ConstantsEditor
from cubie.gui.states_editor import StatesEditor

__all__ = ["ConstantsEditor", "StatesEditor"]
