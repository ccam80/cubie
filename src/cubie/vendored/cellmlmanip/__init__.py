"""Main module for loading, parsing and manipulating CellML models.

Vendored into CuBIE from cellmlmanip 0.3.6
(https://github.com/ModellingWebLab/cellmlmanip, BSD 3-Clause; see the
adjacent LICENSE file). Local modifications: absolute intra-package
imports (``cellmlmanip.x``) rewritten to relative (``.x``), and a
Pint >= 0.20 compatibility fallback in ``units.py``.
"""
from ._config import __version__, __version_int__, version  # noqa
from .main import load_model  # noqa
